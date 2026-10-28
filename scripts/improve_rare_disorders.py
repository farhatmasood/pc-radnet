"""
improve_rare_disorders.py - Task 3: Improve rare disorder detection.
Implements oversampling, focal loss, and class-balanced sampling.

Current rare disorder F1 scores (A2):
  Stenosis: 0.06, Desiccated: 0.02, LigFlavum: 0.09, Spondylolisthesis: 0.12

Strategy:
  1. Focal Loss (γ=2) instead of BCE for disorder head
  2. Oversampling: repeat minority class patients 3-5×
  3. Class-balanced sampling with sqrt-inverse-frequency weighting

Usage:
    python scripts/improve_rare_disorders.py --epochs 20
    python scripts/improve_rare_disorders.py --evaluate
"""
import os, sys, json, argparse, time, gc, random, copy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

DISORDER_NAMES = [
    "Herniation", "Protrusion", "Bulge", "ThecalComp", "NerveComp",
    "Stenosis", "Spondylolisthesis", "Desiccated", "LigFlavum"
]
DISORDER_COLS = [
    "Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
    "Thecal_Sac_Compression", "Nerve_Root_Compression",
    "Spinal_Canal_Stenosis", "Spondylolisthesis",
    "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"
]


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights

    def forward(self, logits, targets):
        """
        logits: [B, 9]
        targets: [B, 9]
        """
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = focal_weight * bce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            loss = loss * alpha_t.unsqueeze(0)

        return loss.mean()


class DisorderDataset(Dataset):
    """Patient-level disorder classification dataset with oversampling."""

    def __init__(self, split_file, transforms=None, oversample_rare=True):
        with open(split_file) as f:
            all_paths = [resolve_image(l) for l in f if l.strip()]

        self.transforms = transforms

        # Group by patient
        root = Path(CFG["paths"]["dataset_root"])
        df_notes = pd.read_csv(root / "structured_notes.csv")

        self.disorders_map = {}
        for _, row in df_notes.iterrows():
            pid = int(row["Patient_ID"])
            self.disorders_map[pid] = [int(row.get(c, 0)) for c in DISORDER_COLS]

        # Build patient-level dataset (one entry per unique patient with image paths)
        patient_images = defaultdict(list)
        for p in all_paths:
            fname = os.path.basename(p)
            parts = fname.replace(".png", "").split("_")
            pid = int(parts[2])
            if pid in self.disorders_map:
                patient_images[pid].append(p)

        self.patient_ids = sorted(patient_images.keys())
        self.patient_images = patient_images
        self.mask_dir = Path(CFG["paths"]["masks"])

        # Compute class frequencies
        labels = np.array([self.disorders_map[pid] for pid in self.patient_ids])
        self.class_counts = labels.sum(axis=0)
        self.class_freq = self.class_counts / len(self.patient_ids)
        print(f"  Disorder counts: {dict(zip(DISORDER_NAMES, self.class_counts.astype(int).tolist()))}")

        # Oversample rare classes
        if oversample_rare:
            self._oversample()

    def _oversample(self):
        """Repeat patients with rare disorders to balance dataset."""
        labels = np.array([self.disorders_map[pid] for pid in self.patient_ids])
        max_count = labels.sum(axis=0).max()

        oversampled = list(self.patient_ids)
        for d_idx in range(9):
            positive_pids = [pid for pid in self.patient_ids
                             if self.disorders_map[pid][d_idx] == 1]
            if len(positive_pids) == 0:
                continue
            # Repeat until we reach at least 1/3 of max_count
            target = max(int(max_count * 0.33), len(positive_pids))
            while len([p for p in oversampled if self.disorders_map[p][d_idx] == 1]) < target:
                oversampled.append(random.choice(positive_pids))

        random.shuffle(oversampled)
        self.patient_ids_expanded = oversampled
        print(f"  Oversampled: {len(self.patient_ids)} → {len(self.patient_ids_expanded)} entries")

    def __len__(self):
        return len(getattr(self, 'patient_ids_expanded', self.patient_ids))

    def __getitem__(self, idx):
        pids = getattr(self, 'patient_ids_expanded', self.patient_ids)
        pid = pids[idx]

        # Pick a random image for this patient
        img_path = random.choice(self.patient_images[pid])
        fname = os.path.basename(img_path)

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        if h != 384 or w != 384:
            img = np.array(Image.fromarray(img).resize((384, 384), Image.BILINEAR))

        if self.transforms:
            t = self.transforms(image=img)
            img = t["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        disorder = torch.tensor(self.disorders_map[pid], dtype=torch.float32)

        return {
            "image": img,
            "disorder": disorder,
            "patient_id": pid,
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


class DisorderClassifier(nn.Module):
    """Lightweight disorder classifier using DINOv2 features."""

    def __init__(self, backbone="resnet50", num_disorders=9):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_disorders),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def train_disorder_model(model, train_loader, val_loader, epochs=20, lr=1e-4):
    """Train disorder classifier with focal loss."""
    model = model.to(DEVICE)

    # Compute class weights (inverse sqrt frequency)
    all_labels = []
    for batch in train_loader:
        all_labels.append(batch["disorder"])
    labels_tensor = torch.cat(all_labels, dim=0)
    class_pos = labels_tensor.sum(dim=0)
    class_neg = len(labels_tensor) - class_pos
    alpha = torch.sqrt(class_neg / (class_pos + 1))
    alpha = alpha / alpha.mean()
    print(f"  Focal loss alpha weights: {alpha.numpy().round(2).tolist()}")

    focal_loss = FocalLoss(gamma=2.0, alpha=alpha)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"  Model: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    ckpt_dir = Path(CFG["paths"]["checkpoints"]) / "disorder_enhanced"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            targets = batch["disorder"].to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = focal_loss(logits, targets)

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
        val_metrics = evaluate_disorders(model, val_loader)
        val_f1 = val_metrics["macro_f1"]
        elapsed = time.time() - t0

        rare_f1s = [val_metrics.get(f"f1_{d}", 0) for d in ["Stenosis", "Spondylolisthesis", "Desiccated", "LigFlavum"]]
        rare_mean = np.mean(rare_f1s)

        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_train:.4f} | "
              f"macro_F1={val_f1:.4f} | rare_F1={rare_mean:.4f} | {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_disorder_model.pt")
            print(f"    ↑ New best macro_F1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best_disorder_model.pt", map_location=DEVICE, weights_only=True))
    return model


@torch.no_grad()
def evaluate_disorders(model, dataloader, threshold=0.5):
    """Evaluate disorder predictions, returning per-class F1."""
    model.eval()
    all_preds = []
    all_gt = []

    # Aggregate per patient (max pool across images)
    patient_preds = defaultdict(list)
    patient_gt = {}

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        pids = batch["patient_id"].numpy()
        gt = batch["disorder"].numpy()

        for i in range(len(pids)):
            pid = int(pids[i])
            patient_preds[pid].append(probs[i])
            if pid not in patient_gt:
                patient_gt[pid] = gt[i]

    # Average predictions per patient
    for pid in sorted(patient_preds.keys()):
        all_preds.append(np.mean(patient_preds[pid], axis=0))
        all_gt.append(patient_gt[pid])

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)
    pred_bin = (all_preds >= threshold).astype(int)

    metrics = {}
    f1s = []
    for i, dname in enumerate(DISORDER_NAMES):
        tp = (pred_bin[:, i] * all_gt[:, i]).sum()
        fp = (pred_bin[:, i] * (1 - all_gt[:, i])).sum()
        fn = ((1 - pred_bin[:, i]) * all_gt[:, i]).sum()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        metrics[f"precision_{dname}"] = float(prec)
        metrics[f"recall_{dname}"] = float(rec)
        metrics[f"f1_{dname}"] = float(f1)
        f1s.append(f1)

    metrics["macro_f1"] = float(np.mean(f1s))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    splits = Path(CFG["paths"]["splits_dir"])
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("RARE DISORDER DETECTION IMPROVEMENT")
    print("="*60)

    train_ds = DisorderDataset(str(splits / "train.txt"), get_train_transforms(), oversample_rare=True)
    val_ds = DisorderDataset(str(splits / "val.txt"), get_val_transforms(), oversample_rare=False)
    test_ds = DisorderDataset(str(splits / "test.txt"), get_val_transforms(), oversample_rare=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = DisorderClassifier(backbone=args.backbone)

    ckpt_path = Path(CFG["paths"]["checkpoints"]) / "disorder_enhanced" / "best_disorder_model.pt"

    if not args.evaluate:
        model = train_disorder_model(model, train_loader, val_loader, epochs=args.epochs)
    elif ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded: {ckpt_path}")
    else:
        print(f"No checkpoint found: {ckpt_path}")

    # Test evaluation
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")

    test_metrics = evaluate_disorders(model.to(DEVICE), test_loader, threshold=args.threshold)

    # Print comparison with A2
    ablation_file = results_dir / "all_ablation_metrics_test.json"
    a2_disorder_f1s = {}
    if ablation_file.exists():
        ablations = json.load(open(ablation_file))
        a2 = next((a for a in ablations if a["ablation"] == "A2"), None)
        if a2:
            for dname in DISORDER_NAMES:
                prec = a2.get(f"disorder_{dname}_prec", 0)
                rec = a2.get(f"disorder_{dname}_rec", 0)
                a2_disorder_f1s[dname] = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\n{'Disorder':<25} {'A2 F1':>8} {'Enhanced F1':>12} {'Δ':>8}")
    print("-" * 53)
    for dname in DISORDER_NAMES:
        a2_f1 = a2_disorder_f1s.get(dname, 0)
        new_f1 = test_metrics.get(f"f1_{dname}", 0)
        delta = new_f1 - a2_f1
        marker = " *" if dname in ["Stenosis", "Spondylolisthesis", "Desiccated", "LigFlavum"] else ""
        print(f"  {dname:<23} {a2_f1:>8.4f} {new_f1:>12.4f} {delta:>+8.4f}{marker}")

    a2_macro = np.mean(list(a2_disorder_f1s.values())) if a2_disorder_f1s else 0
    print(f"  {'Macro F1':<23} {a2_macro:>8.4f} {test_metrics['macro_f1']:>12.4f} "
          f"{test_metrics['macro_f1'] - a2_macro:>+8.4f}")

    # Save results
    test_metrics["model"] = "Enhanced (focal + oversample + ResNet50)"
    test_metrics["backbone"] = args.backbone
    test_metrics["threshold"] = args.threshold
    with open(results_dir / "disorder_enhanced_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to {results_dir / 'disorder_enhanced_test_metrics.json'}")


if __name__ == "__main__":
    main()
