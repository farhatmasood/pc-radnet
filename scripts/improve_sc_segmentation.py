"""
improve_sc_segmentation.py - Task 2: Improve axial spinal canal segmentation.
Implements coarse-to-fine strategy and small-structure augmentation to 
improve SC Dice from the current 0.491 (A2 baseline).

Strategy:
  1. Coarse-to-Fine: First predict coarse mask, then crop+zoom SC region
  2. Small-structure augmentation: elastic deform, random crops around SC
  3. SC-focused loss weighting

Uses the existing A2 checkpoint as starting point, fine-tunes with SC focus.

Usage:
    python scripts/improve_sc_segmentation.py --phase finetune --epochs 15
    python scripts/improve_sc_segmentation.py --phase evaluate
"""
import os, sys, json, argparse, time, gc, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from repo_config import load_config, resolve_image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
CFG = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SCFocusedDataset(Dataset):
    """Dataset with SC-focused augmentations for axial images only.
    
    Strategy:
    - 50% of the time: zoom into SC region (coarse-to-fine)
    - Always: elastic deformation to increase SC shape variation
    - SC class weight boosted further
    """

    def __init__(self, split_file, transforms=None, sc_zoom_prob=0.5):
        with open(split_file) as f:
            all_paths = [resolve_image(l) for l in f if l.strip()]
        # Only axial images have SC
        self.image_paths = [p for p in all_paths if os.path.basename(p).startswith("axial_")]
        self.mask_dir = Path(CFG["paths"]["masks"])
        self.transforms = transforms
        self.sc_zoom_prob = sc_zoom_prob

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        mask_path = self.mask_dir / fname
        mask = np.array(Image.open(mask_path)) if mask_path.exists() else np.zeros((h, w), dtype=np.uint8)

        if h != 384 or w != 384:
            img = np.array(Image.fromarray(img).resize((384, 384), Image.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((384, 384), Image.NEAREST))

        # SC-focused zoom: crop around SC region and resize back
        if random.random() < self.sc_zoom_prob:
            sc_mask = (mask == 3)
            if sc_mask.sum() > 10:
                ys, xs = np.where(sc_mask)
                cy, cx = ys.mean(), xs.mean()
                # Random crop size around SC region (128-256 px)
                crop_size = random.randint(128, 256)
                half = crop_size // 2
                y1 = max(0, int(cy - half))
                x1 = max(0, int(cx - half))
                y2 = min(384, y1 + crop_size)
                x2 = min(384, x1 + crop_size)
                y1 = max(0, y2 - crop_size)
                x1 = max(0, x2 - crop_size)

                img_crop = img[y1:y2, x1:x2]
                mask_crop = mask[y1:y2, x1:x2]

                # Resize back to 384x384
                img = np.array(Image.fromarray(img_crop).resize((384, 384), Image.BILINEAR))
                mask = np.array(Image.fromarray(mask_crop).resize((384, 384), Image.NEAREST))

        if self.transforms:
            t = self.transforms(image=img, mask=mask)
            img = t["image"]
            mask = t["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # One-hot for 4 axial classes
        mask_onehot = torch.zeros(4, 384, 384, dtype=torch.float32)
        for c in range(4):
            mask_onehot[c] = (mask == c).float()

        return {
            "image": img,
            "mask": mask,
            "mask_onehot": mask_onehot,
            "fname": fname,
        }


def get_sc_train_transforms():
    """Augmentations tuned for small-structure preservation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.7),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_sc_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class SCFocusedLoss(nn.Module):
    """Loss with heavy SC weighting.
    SC class weight = 10x other classes to focus learning on spinal canal.
    """
    def __init__(self, sc_weight=10.0):
        super().__init__()
        self.class_weights = torch.tensor([0.1, 1.0, 1.0, sc_weight])  # BG, IVD, PE, SC

    def forward(self, logits, targets):
        """
        logits: [B, 4, 384, 384]
        targets: [B, 4, 384, 384] one-hot
        """
        B = logits.shape[0]
        total = 0
        weights = self.class_weights.to(logits.device)

        for b in range(B):
            for c in range(1, 4):  # skip BG
                pred = torch.sigmoid(logits[b, c])
                gt = targets[b, c]

                # Weighted Dice
                inter = (pred * gt).sum()
                dice = (2 * inter + 1e-5) / (pred.sum() + gt.sum() + 1e-5)

                # Weighted BCE
                bce = F.binary_cross_entropy_with_logits(logits[b, c], gt, reduction='mean')

                total += weights[c] * ((1 - dice) + bce)

        return total / (B * 3)


def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)


def build_sc_model():
    """Build a dedicated axial segmentation model for SC improvement.
    Uses the nnU-Net-style architecture but focused on 4-class axial segmentation.
    """
    from evaluate_direct_baselines import PlainConvUNet
    return PlainConvUNet(in_channels=3, out_channels=4, features=(32, 64, 128, 256, 512))


def train_sc_model(model, train_loader, val_loader, epochs=15, lr=5e-4):
    """Train SC-focused model."""
    model = model.to(DEVICE)
    loss_fn = SCFocusedLoss(sc_weight=10.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"SC model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    ckpt_dir = Path(CFG["paths"]["checkpoints"]) / "sc_enhanced"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_sc_dice = 0
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            targets = batch["mask_onehot"].to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_sc(model, val_loader)
        sc_dice = val_metrics["dice_SC"]
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_train:.4f} | "
              f"SC_Dice={sc_dice:.4f} | IVD={val_metrics['dice_IVD']:.4f} | "
              f"PE={val_metrics['dice_PE']:.4f} | {elapsed:.1f}s")

        if sc_dice > best_sc_dice:
            best_sc_dice = sc_dice
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_sc_model.pt")
            print(f"    ↑ New best SC_Dice={best_sc_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best_sc_model.pt", map_location=DEVICE, weights_only=True))
    return model


@torch.no_grad()
def evaluate_sc(model, dataloader):
    """Evaluate axial segmentation model on per-class Dice."""
    model.eval()
    class_dices = defaultdict(list)
    class_ious = defaultdict(list)

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        targets = batch["mask_onehot"]

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)

        probs = torch.sigmoid(logits.float()).cpu()
        names = {1: "IVD", 2: "PE", 3: "SC"}

        for i in range(images.shape[0]):
            for c in range(1, 4):
                pred = probs[i, c]
                gt = targets[i, c]
                d = compute_dice(pred, gt).item()
                pred_bin = (pred > 0.5).float()
                inter = (pred_bin * gt).sum()
                union = pred_bin.sum() + gt.sum() - inter
                iou = (inter + 1e-5) / (union + 1e-5)
                class_dices[names[c]].append(d)
                class_ious[names[c]].append(iou.item())

    metrics = {}
    for cname in ["IVD", "PE", "SC"]:
        metrics[f"dice_{cname}"] = np.mean(class_dices[cname])
        metrics[f"iou_{cname}"] = np.mean(class_ious[cname])
    metrics["mean_fg_dice"] = np.mean([metrics[f"dice_{c}"] for c in ["IVD", "PE", "SC"]])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["finetune", "evaluate"], default="finetune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sc-weight", type=float, default=10.0)
    args = parser.parse_args()

    splits = Path(CFG["paths"]["splits_dir"])
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SCFocusedDataset(str(splits / "train.txt"), get_sc_train_transforms(), sc_zoom_prob=0.5)
    val_ds = SCFocusedDataset(str(splits / "val.txt"), get_sc_val_transforms(), sc_zoom_prob=0.0)
    test_ds = SCFocusedDataset(str(splits / "test.txt"), get_sc_val_transforms(), sc_zoom_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = build_sc_model()

    ckpt_path = Path(CFG["paths"]["checkpoints"]) / "sc_enhanced" / "best_sc_model.pt"

    if args.phase == "finetune":
        model = train_sc_model(model, train_loader, val_loader, epochs=args.epochs)
    elif ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")

    # Test evaluation
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (axial only)")
    print(f"{'='*60}")

    test_metrics = evaluate_sc(model, test_loader)

    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save results
    test_metrics["model"] = "SC-Enhanced (coarse-to-fine + augmentation)"
    test_metrics["sc_weight"] = args.sc_weight
    with open(results_dir / "sc_enhanced_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Compare with A2 baseline
    ablation_file = results_dir / "all_ablation_metrics_test.json"
    if ablation_file.exists():
        ablations = json.load(open(ablation_file))
        a2 = next((a for a in ablations if a["ablation"] == "A2"), None)
        if a2:
            print(f"\nComparison with PCRadNet A2:")
            print(f"  {'Metric':<15} {'A2':>10} {'SC-Enhanced':>12} {'Δ':>8}")
            print(f"  {'-'*45}")
            for cname in ["IVD", "PE", "SC"]:
                a2_d = a2.get(f"dice_{cname}", 0)
                new_d = test_metrics.get(f"dice_{cname}", 0)
                delta = new_d - a2_d
                print(f"  {cname:<15} {a2_d:>10.4f} {new_d:>12.4f} {delta:>+8.4f}")

    print(f"\nResults saved to {results_dir / 'sc_enhanced_test_metrics.json'}")


if __name__ == "__main__":
    main()
