"""
sc_finetune_a3.py - SC-focused fine-tuning of PCRadNet A3 checkpoint.

Fine-tunes the A3 model (best_by_mdice_A3.pt) on axial images only,
with coarse-to-fine SC zoom augmentation and boosted SC loss weight.

Usage:
    python scripts/sc_finetune_a3.py --epochs 12
    python scripts/sc_finetune_a3.py --eval-only
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

# ── paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CFG = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

from train import (
    PCRadNet, collate_fn, compute_dice, compute_iou,
    PROMPT_BANK, parse_yolo_polygons, polygons_to_mask,
)

RESULTS_DIR  = Path(CFG["paths"]["results"])
CKPT_DIR     = Path(CFG["paths"]["checkpoints"])
MASK_DIR     = Path(CFG["paths"]["masks"])
IMG_DIR      = Path(CFG["paths"]["images"])

# ═══════════════════════════════════════════════════════════════════════
# Dataset: axial-only with SC-focused zoom augmentation
# ═══════════════════════════════════════════════════════════════════════
class AxialSCDataset(Dataset):
    """Axial-only SpineDataset with optional coarse-to-fine SC zoom."""

    def __init__(self, split_file, transforms=None, sc_zoom_prob=0.5):
        with open(split_file) as f:
            all_paths = [resolve_image(l) for l in f if l.strip()]
        # keep axial only
        self.image_paths = [p for p in all_paths
                            if os.path.basename(p).lower().startswith("axial")]
        self.transforms = transforms
        self.sc_zoom_prob = sc_zoom_prob

        # Load Pfirrmann grades for metadata
        df_p = pd.read_csv(
            Path(CFG["paths"]["dataset_root"]) / "PfirrmannGrade.csv"
        )
        self.grades = {}
        for _, row in df_p.iterrows():
            pid = int(row["Patient_ID"])
            self.grades[pid] = {
                "D3": int(row["D3"]), "D4": int(row["D4"]), "D5": int(row["D5"])
            }

    def _parse_fname(self, fname):
        stem = fname.replace(".png", "")
        parts = stem.split("_")
        return {
            "view": parts[0],
            "sequence": parts[1],
            "patient_id": int(parts[2]),
            "level": parts[3],
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        info  = self._parse_fname(fname)
        pid   = info["patient_id"]
        level = info["level"]

        img  = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        mask_path = MASK_DIR / fname
        if mask_path.exists():
            mask = np.array(Image.open(mask_path))
        else:
            lbl_path = Path(CFG["paths"]["labels"]) / fname.replace(".png", ".txt")
            polys = parse_yolo_polygons(str(lbl_path), h, w)
            mask  = polygons_to_mask(polys, h, w, 4)

        if h != 384 or w != 384:
            img  = np.array(Image.fromarray(img ).resize((384, 384), Image.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((384, 384), Image.NEAREST))

        # ── coarse-to-fine SC zoom ──────────────────────────────────
        if self.sc_zoom_prob > 0 and random.random() < self.sc_zoom_prob:
            sc_pixels = (mask == 3)
            if sc_pixels.sum() > 10:
                ys, xs = np.where(sc_pixels)
                cy, cx = int(ys.mean()), int(xs.mean())
                crop = random.randint(128, 256)
                half = crop // 2
                y1 = max(0,   cy - half);  y2 = min(384, y1 + crop);  y1 = max(0, y2 - crop)
                x1 = max(0,   cx - half);  x2 = min(384, x1 + crop);  x1 = max(0, x2 - crop)
                img  = np.array(Image.fromarray(img [y1:y2, x1:x2]).resize((384, 384), Image.BILINEAR))
                mask = np.array(Image.fromarray(mask[y1:y2, x1:x2]).resize((384, 384), Image.NEAREST))

        if self.transforms:
            t    = self.transforms(image=img, mask=mask)
            img  = t["image"]
            mask = t["mask"].long()
        else:
            img  = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # one-hot [4, 384, 384]
        mask_oh = torch.zeros(4, 384, 384, dtype=torch.float32)
        for c in range(4):
            mask_oh[c] = (mask == c).float()

        # prompted mask: pick a random foreground class present
        present = [c for c in range(1, 4) if mask_oh[c].sum() > 5]
        tgt_cls = random.choice(present) if present else 1
        prompt_map = {1: "intervertebral disc", 2: "posterior element", 3: "spinal canal"}
        prompt_text = random.choice(PROMPT_BANK["axial"][tgt_cls])
        prompted_mask = mask_oh[tgt_cls]

        grade = self.grades.get(pid, {}).get(level, -1)

        return {
            "image":         img,
            "mask":          mask,
            "mask_onehot":   mask_oh,
            "prompted_mask": prompted_mask,
            "prompt_text":   prompt_text,
            "target_cls":    tgt_cls,
            "plane_id":      torch.tensor(0, dtype=torch.long),
            "num_classes":   4,
            "grade":         grade,
            "height":        -1.0,
            "disorder":      torch.zeros(9, dtype=torch.float32),
            "patient_id":    pid,
            "level":         level,
            "fname":         fname,
        }


def get_sc_train_tfm():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                           rotate_limit=20, p=0.7),
        A.ElasticTransform(alpha=30, sigma=5, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.25),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_sc_val_tfm():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════
# Loss: SC-weighted (SC class = 10× other foreground classes)
# ═══════════════════════════════════════════════════════════════════════
SC_WEIGHT  = 2.5
CLS_WEIGHTS = torch.tensor([0.1, 1.0, 1.0, SC_WEIGHT]).to(DEVICE)  # BG,IVD,PE,SC


def sc_focused_loss(seg_logits_list, prompted_logits, targets_oh, prompted_gt):
    """
    Segmentation loss with SC-boosted class weighting.
    seg_logits_list:  list of B tensors each [1, 4, 384, 384]  (axial only)
    prompted_logits:  [B, 1, 384, 384]
    targets_oh:       [B, 4, 384, 384]
    prompted_gt:      [B, 384, 384]
    """
    # Stack list → [B, 4, 384, 384]
    seg_logits = torch.cat(seg_logits_list, dim=0)   # each [1,4,H,W] → [B,4,H,W]
    device = seg_logits.device
    B = seg_logits.shape[0]
    total = torch.tensor(0.0, device=device, requires_grad=True)

    # ── per-class segmentation loss ─────────────────────────────────
    for c in range(1, 4):                           # skip BG
        pred = torch.sigmoid(seg_logits[:, c])      # [B, H, W]
        gt   = targets_oh[:, c].to(device)
        w    = CLS_WEIGHTS[c]

        inter = (pred * gt).sum(dim=(1,2))
        dice_loss = 1 - (2*inter + 1e-5) / (pred.sum(dim=(1,2)) + gt.sum(dim=(1,2)) + 1e-5)

        bce_loss = F.binary_cross_entropy_with_logits(
            seg_logits[:, c], gt, reduction='mean'
        )
        total = total + w * (dice_loss.mean() + bce_loss)

    # ── prompted mask loss ───────────────────────────────────────────
    p_pred = torch.sigmoid(prompted_logits[:, 0])
    p_gt   = prompted_gt.float()
    inter  = (p_pred * p_gt).sum(dim=(1,2))
    p_dice = 1 - (2*inter + 1e-5) / (p_pred.sum(dim=(1,2)) + p_gt.sum(dim=(1,2)) + 1e-5)
    p_bce  = F.binary_cross_entropy_with_logits(
        prompted_logits[:, 0], p_gt, reduction='mean'
    )
    total = total + (p_dice.mean() + p_bce)

    return total / 4.0    # normalise


# ═══════════════════════════════════════════════════════════════════════
# Evaluation: per-class Dice on axial test set with per-disc-level split
# ═══════════════════════════════════════════════════════════════════════
AXIAL_NAMES = {1: "IVD", 2: "PE", 3: "SC"}
LEVEL_MAP   = {"D3": "L3/L4", "D4": "L4/L5", "D5": "L5/S1"}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    rows = []
    for batch in loader:
        images   = batch["image"].to(DEVICE)
        plane_ids = batch["plane_id"].to(DEVICE)
        mask_oh  = batch["mask_onehot"].to(DEVICE)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(images, batch["prompt_text"], plane_ids)

        # stack list of [1,C,H,W] → [B,C,H,W]
        seg_logits_cat = torch.cat(out["seg_logits_list"], dim=0).float().cpu()
        seg_probs = torch.sigmoid(seg_logits_cat)
        prom_probs = torch.sigmoid(out["prompted_logits"].float()).cpu()
        mask_oh_cpu = batch["mask_onehot"].cpu()

        for i in range(images.shape[0]):
            row = {
                "fname":  batch["fname"][i],
                "level":  batch["level"][i],
                "grade":  batch["grade"][i].item() if isinstance(batch["grade"][i], torch.Tensor)
                          else batch["grade"][i],
            }
            row["prompted_dice"] = compute_dice(
                prom_probs[i, 0], batch["prompted_mask"][i].cpu()
            ).item()
            for c, name in AXIAL_NAMES.items():
                if c < seg_probs.shape[1]:
                    row[f"dice_{name}"] = compute_dice(
                        seg_probs[i, c], mask_oh_cpu[i, c]
                    ).item()
            rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=12)
    parser.add_argument("--lr",        type=float, default=2e-5)
    parser.add_argument("--batch-size",type=int,   default=4)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    splits    = Path(CFG["paths"]["splits_dir"])
    out_ckpt  = CKPT_DIR / "sc_enhanced_a3.pt"
    out_json  = RESULTS_DIR / "sc_enhanced_a3_test_metrics.json"
    out_csv   = RESULTS_DIR / "sc_enhanced_a3_per_level.csv"

    # ── Build model from A3 checkpoint ──────────────────────────────
    print("Loading PCRadNet A3 …")
    model = PCRadNet(CFG, use_calibrator=True, lora_rank=8).to(DEVICE)

    a3_ckpt = CKPT_DIR / "best_by_mdice_A3.pt"
    state = torch.load(a3_ckpt, map_location=DEVICE, weights_only=False)
    # train.py saves under "model_state_dict" key
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"  Loaded: {a3_ckpt}")

    # Freeze encoder fully; only fine-tune decoder + calibrator
    for name, p in model.named_parameters():
        if "encoder" in name and "lora" not in name.lower():
            p.requires_grad = False

    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_n:,}")

    # ── Datasets ────────────────────────────────────────────────────
    train_ds = AxialSCDataset(str(splits/"train.txt"), get_sc_train_tfm(), sc_zoom_prob=0.5)
    val_ds   = AxialSCDataset(str(splits/"val.txt"),   get_sc_val_tfm(),   sc_zoom_prob=0.0)
    test_ds  = AxialSCDataset(str(splits/"test.txt"),  get_sc_val_tfm(),   sc_zoom_prob=0.0)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,} axial images")

    # ── Fine-tuning ──────────────────────────────────────────────────
    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scaler = GradScaler()

        best_sc = 0.0
        patience_cnt = 0
        PATIENCE = 5

        print(f"\nFine-tuning for {args.epochs} epochs  (lr={args.lr}, bs={args.batch_size}) …")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            n_steps = 0

            for batch in train_ld:
                images    = batch["image"].to(DEVICE)
                plane_ids = batch["plane_id"].to(DEVICE)
                mask_oh   = batch["mask_onehot"].to(DEVICE)
                prom_gt   = batch["prompted_mask"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(images, batch["prompt_text"], plane_ids)
                    # seg_logits_list: list of B tensors [1,4,H,W]
                    prom_logits = out["prompted_logits"]   # [B,1,384,384]
                    loss = sc_focused_loss(out["seg_logits_list"], prom_logits,
                                          mask_oh[:, :4], prom_gt)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_steps += 1

            scheduler.step()

            # Validate
            val_df = evaluate(model, val_ld)
            val_sc = val_df["dice_SC"].mean() if "dice_SC" in val_df else 0.0
            val_ivd = val_df["dice_IVD"].mean() if "dice_IVD" in val_df else 0.0
            val_pe  = val_df["dice_PE"].mean()  if "dice_PE" in val_df else 0.0
            elapsed = time.time() - t0

            print(f"  Ep {epoch:3d}/{args.epochs} | "
                  f"loss={total_loss/n_steps:.4f} | "
                  f"SC={val_sc:.4f}  IVD={val_ivd:.4f}  PE={val_pe:.4f} | "
                  f"{elapsed:.0f}s")

            if val_sc > best_sc:
                best_sc = val_sc
                patience_cnt = 0
                torch.save(model.state_dict(), out_ckpt)
                print(f"    ↑ Best SC_Dice={best_sc:.4f}  saved → {out_ckpt.name}")
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # Load best
        model.load_state_dict(
            torch.load(out_ckpt, map_location=DEVICE, weights_only=False)
        )
        print(f"\nLoaded best checkpoint (SC_Dice={best_sc:.4f})")

    else:
        # eval-only: load previously saved fine-tuned checkpoint
        if out_ckpt.exists():
            model.load_state_dict(
                torch.load(out_ckpt, map_location=DEVICE, weights_only=False)
            )
            print(f"Loaded fine-tuned ckpt: {out_ckpt}")
        else:
            print("No fine-tuned checkpoint found; evaluating raw A3 model.")

    # ── Test evaluation ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (axial, n=468)")
    print(f"{'='*60}")

    test_df = evaluate(model, test_ld)

    # Overall class means
    metrics = {}
    for cls in ["IVD", "PE", "SC"]:
        col = f"dice_{cls}"
        if col in test_df.columns:
            metrics[f"dice_{cls}"] = float(test_df[col].mean())
    metrics["mean_fg_dice"] = float(
        np.mean([metrics.get(f"dice_{c}", 0) for c in ["IVD","PE","SC"]])
    )
    metrics["prompted_dice"] = float(test_df["prompted_dice"].mean())
    metrics["base_model"] = "A3"
    metrics["sc_weight"]  = SC_WEIGHT
    metrics["model"]      = "SC-Enhanced A3 (fine-tuned PCRadNet A3)"

    print("\n  Class-level means:")
    for k, v in sorted(metrics.items()):
        if k.startswith("dice") or k == "mean_fg_dice" or k == "prompted_dice":
            print(f"    {k:<20}: {v:.4f}")

    # Per-disc-level breakdown
    test_df["level_label"] = test_df["level"].map(LEVEL_MAP)
    level_stats = []
    for lv_code, lv_label in LEVEL_MAP.items():
        sub = test_df[test_df["level"] == lv_code]
        if len(sub) == 0:
            continue
        row = {"level_code": lv_code, "level_label": lv_label, "n": len(sub)}
        for cls in ["IVD", "PE", "SC"]:
            col = f"dice_{cls}"
            if col in sub.columns:
                row[f"dice_{cls}_mean"] = float(sub[col].mean())
                row[f"dice_{cls}_std"]  = float(sub[col].std())
        row["prompted_dice_mean"] = float(sub["prompted_dice"].mean())
        row["prompted_dice_std"]  = float(sub["prompted_dice"].std())
        level_stats.append(row)
        print(f"\n  {lv_label} (n={len(sub)}):")
        for cls in ["IVD","PE","SC"]:
            print(f"    dice_{cls}: {row.get(f'dice_{cls}_mean',0):.4f} "
                  f"± {row.get(f'dice_{cls}_std',0):.4f}")

    # Compare with A3 baseline
    print("\n  Delta vs A3 baseline (IVD=0.948, PE=0.853, SC=0.527):")
    a3_base = {"dice_IVD": 0.948, "dice_PE": 0.853, "dice_SC": 0.527}
    for cls in ["IVD","PE","SC"]:
        k = f"dice_{cls}"
        delta = metrics.get(k, 0) - a3_base[k]
        print(f"    {cls}: {metrics.get(k,0):.4f}  (Δ={delta:+.4f})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(level_stats).to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_json.name}  |  {out_csv.name}")


if __name__ == "__main__":
    main()
