"""
evaluate_direct_baselines.py - Train + evaluate nnU-Net-style, SwinUNETR, SAM ViT-B
on LSMA-PQR using the same train/val/test splits as PCRadNet.

Fair comparison:
  - Same 384×384 input, same train/val/test splits, same per-class Dice/IoU metrics
  - All models: 5-class output (axial cls 4 ignored in axial images)
  - Mixed precision bf16, early stopping on val mDice

Usage:
    python scripts/evaluate_direct_baselines.py --model swinunetr --epochs 30
    python scripts/evaluate_direct_baselines.py --model nnunet --epochs 30
    python scripts/evaluate_direct_baselines.py --model sam_ft --epochs 20
    python scripts/evaluate_direct_baselines.py --evaluate-only
"""
import os, sys, json, argparse, time, gc
import random
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

# ── Seed ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping: unified 5-class output
# axial: 0=BG, 1=IVD, 2=PE, 3=SC, (4=empty)
# sagittal: 0=BG, 1=VB, 2=IVD_sag, 3=Sacrum, 4=SC_sag
NUM_CLASSES = 5  # including BG
CLASS_NAMES = {
    "axial":    {1: "IVD", 2: "PE", 3: "SC"},
    "sagittal": {1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"},
}
ALL_FG_NAMES = ["IVD", "PE", "SC", "VB", "IVD_sag", "Sacrum", "SC_sag"]


# ═════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════

class SpineSegDataset(Dataset):
    """Lightweight segmentation-only dataset for baseline models."""

    def __init__(self, split_file, transforms=None):
        with open(split_file) as f:
            self.image_paths = [resolve_image(l) for l in f if l.strip()]
        self.mask_dir = Path(CFG["paths"]["masks"])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        stem = fname.replace(".png", "")
        parts = stem.split("_")
        view = parts[0]
        is_axial = view == "axial"

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        mask_path = self.mask_dir / fname
        mask = np.array(Image.open(mask_path)) if mask_path.exists() else np.zeros((h, w), dtype=np.uint8)

        if h != 384 or w != 384:
            img = np.array(Image.fromarray(img).resize((384, 384), Image.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((384, 384), Image.NEAREST))

        if self.transforms:
            t = self.transforms(image=img, mask=mask)
            img = t["image"]
            mask = t["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # One-hot with 5 classes (pad axial class 4 with zeros)
        mask_onehot = torch.zeros(NUM_CLASSES, 384, 384, dtype=torch.float32)
        valid_cls = 4 if is_axial else 5
        for c in range(valid_cls):
            mask_onehot[c] = (mask == c).float()
        # For axial, class 4 stays zero

        # Class mask: indicates which foreground classes are valid
        cls_mask = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        cls_mask[:valid_cls] = 1.0

        return {
            "image": img,
            "mask": mask,
            "mask_onehot": mask_onehot,
            "cls_mask": cls_mask,
            "is_axial": is_axial,
            "fname": fname,
        }


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def collate_fn(batch):
    result = {}
    for k in batch[0]:
        if k == "fname":
            result[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        else:
            result[k] = torch.tensor([b[k] for b in batch])
    return result


# ═════════════════════════════════════════════════════════════════════════
# DICE / IOU METRICS
# ═════════════════════════════════════════════════════════════════════════

def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)


def compute_iou(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


class DiceBCELoss(nn.Module):
    """Combined Dice + weighted BCE loss for multi-class segmentation."""
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, cls_mask):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W] one-hot
        cls_mask: [B, C] valid class indicators
        """
        B, C, H, W = logits.shape
        total = 0
        n = 0
        for b in range(B):
            for c in range(1, C):  # skip BG
                if cls_mask[b, c] < 0.5:
                    continue
                pred = torch.sigmoid(logits[b, c])
                gt = targets[b, c]
                # Dice
                inter = (pred * gt).sum()
                dice = (2 * inter + 1e-5) / (pred.sum() + gt.sum() + 1e-5)
                # BCE
                bce = F.binary_cross_entropy_with_logits(logits[b, c], gt, reduction='mean')
                total += (1 - dice) + bce
                n += 1
        return total / max(n, 1)


# ═════════════════════════════════════════════════════════════════════════
# MODEL 1: SwinUNETR (MONAI)
# ═════════════════════════════════════════════════════════════════════════

def build_swinunetr():
    """Build SwinUNETR for 2D segmentation using MONAI spatial_dims=2."""
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        img_size=(384, 384),
        in_channels=3,
        out_channels=NUM_CLASSES,
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        use_checkpoint=True,
        spatial_dims=2,
    )
    return model


# ═════════════════════════════════════════════════════════════════════════
# MODEL 2: nnU-Net-style Plain ConvUNet
# ═════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PlainConvUNet(nn.Module):
    """nnU-Net-style encoder-decoder with instance norm and LeakyReLU.
    Follows the nnU-Net 2D architecture: strided conv downsampling, transposed conv upsampling.
    """
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(prev_ch, f))
            self.pools.append(nn.Conv2d(f, f, 2, stride=2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        prev_ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            prev_ch = f

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)


# ═════════════════════════════════════════════════════════════════════════
# MODEL 3: SAM ViT-B Fine-tuned with Segmentation Head
# ═════════════════════════════════════════════════════════════════════════

class SAMFineTuned(nn.Module):
    """SAM ViT-B image encoder with frozen weights + trainable segmentation decoder."""

    def __init__(self, sam_checkpoint, out_channels=NUM_CLASSES):
        super().__init__()
        from segment_anything import sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.image_encoder = sam.image_encoder

        # Freeze SAM encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # SAM encoder output: [B, 256, 24, 24] for 384x384 input (patch_size=16, 384/16=24)
        # Decoder to upsample to full resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=4),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, x):
        # SAM expects 1024x1024 input - resize
        x_resized = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        with torch.no_grad():
            features = self.image_encoder(x_resized)  # [B, 256, 64, 64]
        logits = self.decoder(features)  # upsample to ~384
        # Resize to exact target
        logits = F.interpolate(logits, size=(384, 384), mode='bilinear', align_corners=False)
        return logits


# ═════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════

def train_model(model, model_name, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train a segmentation model with early stopping."""
    model = model.to(DEVICE)
    loss_fn = DiceBCELoss()

    # Only optimize trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"\n{'='*60}")
    print(f"Training {model_name}: {n_params:,} trainable parameters")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    ckpt_dir = Path(CFG["paths"]["checkpoints"]) / "baselines"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0
    patience = 10
    patience_counter = 0
    grad_accum = 4  # accumulate over 4 steps

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(DEVICE)
            targets = batch["mask_onehot"].to(DEVICE)
            cls_mask = batch["cls_mask"].to(DEVICE)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = loss_fn(logits, targets, cls_mask) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            n_batches += 1

        # Flush remaining gradients
        if n_batches % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_model(model, val_loader)
        val_dice = val_metrics["mean_fg_dice"]

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | train_loss={avg_train:.4f} | "
              f"val_mDice={val_dice:.4f} | val_mIoU={val_metrics['mean_fg_iou']:.4f} | {elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / f"best_{model_name}.pt")
            print(f"    ↑ New best mDice={best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best
    model.load_state_dict(torch.load(ckpt_dir / f"best_{model_name}.pt", map_location=DEVICE, weights_only=True))
    print(f"Best val mDice for {model_name}: {best_dice:.4f}")
    return model


# ═════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, dataloader, save_csv=None):
    """Evaluate segmentation model. Returns per-class Dice/IoU."""
    model.eval()
    records = []

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        cls_mask = batch["cls_mask"]
        fnames = batch["fname"]
        is_axial = batch["is_axial"]

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)  # [B, 5, 384, 384]

        probs = torch.sigmoid(logits.float())
        targets = batch["mask_onehot"]

        for i in range(images.shape[0]):
            ax = bool(is_axial[i])
            plane = "axial" if ax else "sagittal"
            n_cls = 4 if ax else 5

            for c in range(1, n_cls):
                pred = probs[i, c].cpu()
                gt = targets[i, c]
                d = compute_dice(pred, gt).item()
                iou = compute_iou(pred, gt).item()
                cname = CLASS_NAMES[plane].get(c, f"cls{c}")
                records.append({
                    "fname": fnames[i], "plane": plane,
                    "class_idx": c, "class_name": cname,
                    "dice": d, "iou": iou,
                })

    df = pd.DataFrame(records)
    if save_csv and len(df) > 0:
        df.to_csv(save_csv, index=False)

    # Aggregate
    metrics = {}
    if len(df) == 0:
        return {"mean_fg_dice": 0, "mean_fg_iou": 0}

    metrics["mean_fg_dice"] = df["dice"].mean()
    metrics["mean_fg_iou"] = df["iou"].mean()

    for plane in ["axial", "sagittal"]:
        sub = df[df["plane"] == plane]
        if len(sub) > 0:
            metrics[f"{plane}_dice"] = sub["dice"].mean()
            metrics[f"{plane}_iou"] = sub["iou"].mean()

    for cname in ALL_FG_NAMES:
        sub = df[df["class_name"] == cname]
        if len(sub) > 0:
            metrics[f"dice_{cname}"] = sub["dice"].mean()
            metrics[f"iou_{cname}"] = sub["iou"].mean()

    # Prompted dice (mean over all prompted class predictions per image)
    per_img = df.groupby("fname").agg({"dice": "mean", "iou": "mean"}).reset_index()
    metrics["prompted_dice_mean"] = per_img["dice"].mean()
    metrics["prompted_iou_mean"] = per_img["iou"].mean()

    return metrics


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["swinunetr", "nnunet", "sam_ft", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    splits = Path(CFG["paths"]["splits_dir"])
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SpineSegDataset(str(splits / "train.txt"), get_train_transforms())
    val_ds = SpineSegDataset(str(splits / "val.txt"), get_val_transforms())
    test_ds = SpineSegDataset(str(splits / f"{args.split}.txt"), get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=collate_fn)

    models_to_run = [args.model] if args.model != "all" else ["swinunetr", "nnunet", "sam_ft"]
    all_results = []

    for model_name in models_to_run:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*60}")

        if model_name == "swinunetr":
            model = build_swinunetr()
            lr = 1e-4
        elif model_name == "nnunet":
            model = PlainConvUNet()
            lr = 1e-3
        elif model_name == "sam_ft":
            sam_ckpt = Path(CFG["paths"]["pretrained"]["sam_vit_b"])
            if not sam_ckpt.exists():
                print(f"SAM checkpoint not found: {sam_ckpt}")
                continue
            model = SAMFineTuned(str(sam_ckpt))
            lr = 5e-4
        else:
            continue

        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {n_total:,}, Trainable: {n_train:,}")

        ckpt_path = Path(CFG["paths"]["checkpoints"]) / "baselines" / f"best_{model_name}.pt"

        if not args.evaluate_only:
            model = train_model(model, model_name, train_loader, val_loader,
                                epochs=args.epochs, lr=lr)
        else:
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
                print(f"  Loaded checkpoint: {ckpt_path}")
            else:
                print(f"  No checkpoint found: {ckpt_path}, evaluating random init")

        # Test evaluation
        csv_path = results_dir / f"baseline_{model_name}_{args.split}_perclass.csv"
        test_metrics = evaluate_model(model, test_loader, save_csv=str(csv_path))
        test_metrics["model"] = model_name
        test_metrics["trainable_params"] = n_train

        vram = torch.cuda.max_memory_allocated() / 1e9
        test_metrics["vram_gb"] = round(vram, 3)

        all_results.append(test_metrics)

        print(f"\n  {model_name} Test Results:")
        print(f"    mDice:     {test_metrics['mean_fg_dice']:.4f}")
        print(f"    mIoU:      {test_metrics['mean_fg_iou']:.4f}")
        for k, v in test_metrics.items():
            if k.startswith("dice_"):
                print(f"    {k}: {v:.4f}")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Save combined results - merge with existing results from previous runs
    out_path = results_dir / f"baseline_direct_{args.split}.json"
    existing_results = []
    if out_path.exists():
        with open(out_path) as f:
            existing_results = json.load(f)
    # Merge: keep existing models not in current run, add current run results
    current_models = {r["model"] for r in all_results}
    merged = [r for r in existing_results if r["model"] not in current_models] + all_results
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    all_results = merged  # for comparison table
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"BASELINE COMPARISON ({args.split} set)")
    print(f"{'='*80}")
    header = f"{'Model':<20} {'mDice':>8} {'mIoU':>8} {'IVD':>8} {'PE':>8} {'SC':>8} {'VB':>8} {'IVD_s':>8} {'Sacrum':>8} {'SC_s':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['model']:<20} "
              f"{r['mean_fg_dice']:>8.4f} "
              f"{r['mean_fg_iou']:>8.4f} "
              f"{r.get('dice_IVD', 0):>8.4f} "
              f"{r.get('dice_PE', 0):>8.4f} "
              f"{r.get('dice_SC', 0):>8.4f} "
              f"{r.get('dice_VB', 0):>8.4f} "
              f"{r.get('dice_IVD_sag', 0):>8.4f} "
              f"{r.get('dice_Sacrum', 0):>8.4f} "
              f"{r.get('dice_SC_sag', 0):>8.4f}")

    # Add PCRadNet A2 for comparison
    ablation_file = results_dir / "all_ablation_metrics_test.json"
    if ablation_file.exists():
        ablations = json.load(open(ablation_file))
        a2 = next((a for a in ablations if a["ablation"] == "A2"), None)
        if a2:
            print(f"{'PCRadNet-A2':<20} "
                  f"{a2['mean_fg_dice']:>8.4f} "
                  f"{(a2['mean_fg_dice'] + a2.get('prompted_iou_mean', 0))/2:>8.4f} "
                  f"{a2.get('dice_IVD', 0):>8.4f} "
                  f"{a2.get('dice_PE', 0):>8.4f} "
                  f"{a2.get('dice_SC', 0):>8.4f} "
                  f"{a2.get('dice_VB', 0):>8.4f} "
                  f"{a2.get('dice_IVD_sag', 0):>8.4f} "
                  f"{a2.get('dice_Sacrum', 0):>8.4f} "
                  f"{a2.get('dice_SC_sag', 0):>8.4f}")


if __name__ == "__main__":
    main()
