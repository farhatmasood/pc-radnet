"""
baseline_zeroshot.py — SAM ViT-B zero-shot baseline evaluation
Evaluates SAM ViT-B with GT bounding-box prompts on the val/test set.

Usage:
    python scripts/baseline_zeroshot.py --split val
    python scripts/baseline_zeroshot.py --split test
"""
import os, sys, json, argparse, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

with open(SCRIPT_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anatomy prompts for zero-shot detection
PROMPTS_AXIAL = [
    "intervertebral disc",
    "posterior element",
    "spinal canal",
]
PROMPTS_SAG = [
    "vertebral body",
    "intervertebral disc",
    "sacrum",
    "spinal canal",
]


def parse_filename(fname):
    stem = fname.replace(".png", "")
    parts = stem.split("_")
    return {
        "view": parts[0],
        "sequence": parts[1],
        "patient_id": int(parts[2]),
        "level": parts[3],
    }


def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)


def compute_iou(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════
# SAM ViT-B BASELINE (box prompt from GT bounding boxes)
# ═══════════════════════════════════════════════════════════════════

def evaluate_sam_vitb(split_file, max_samples=None):
    """Evaluate SAM ViT-B with GT bounding box prompts."""
    from segment_anything import sam_model_registry, SamPredictor

    sam_ckpt = Path(CFG["paths"]["dataset_root"]) / "RL" / "checkpoints" / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        print(f"SAM checkpoint not found: {sam_ckpt}")
        return pd.DataFrame()

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam = sam.to(dtype=torch.float32, device=DEVICE)
    predictor = SamPredictor(sam)
    print("SAM ViT-B loaded.")

    with open(split_file) as f:
        image_paths = [l.strip() for l in f if l.strip()]
    if max_samples:
        image_paths = image_paths[:max_samples]

    mask_dir = Path(CFG["paths"]["masks"])
    results = []

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        info = parse_filename(fname)
        is_axial = info["view"] == "axial"

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        mask_path = mask_dir / fname
        if not mask_path.exists():
            continue
        gt_mask = np.array(Image.open(mask_path))

        predictor.set_image(img)
        num_cls = 4 if is_axial else 5

        for cls_idx in range(1, num_cls):
            gt_cls = (gt_mask == cls_idx).astype(np.uint8)
            if gt_cls.sum() == 0:
                results.append({
                    "fname": fname, "patient_id": info["patient_id"],
                    "level": info["level"],
                    "plane": "axial" if is_axial else "sagittal",
                    "target_cls": cls_idx, "dice": 0, "iou": 0,
                })
                continue

            # Get bounding box from GT mask
            ys, xs = np.where(gt_cls > 0)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

            masks, scores, _ = predictor.predict(
                box=bbox[None, :],
                multimask_output=True,
            )
            # Take best mask
            best_idx = scores.argmax()
            pred = torch.from_numpy(masks[best_idx]).float()
            gt_t = torch.from_numpy(gt_cls).float()

            dice = compute_dice(pred, gt_t).item()
            iou = compute_iou(pred, gt_t).item()

            results.append({
                "fname": fname, "patient_id": info["patient_id"],
                "level": info["level"],
                "plane": "axial" if is_axial else "sagittal",
                "target_cls": cls_idx, "dice": dice, "iou": iou,
            })

        if (idx + 1) % 50 == 0:
            print(f"  SAM: {idx+1}/{len(image_paths)} images")

    del sam, predictor
    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# AGGREGATE & COMPARE
# ═══════════════════════════════════════════════════════════════════

def aggregate_baseline_metrics(df, model_name):
    """Compute aggregate metrics for a baseline."""
    if len(df) == 0:
        return {}

    metrics = {
        "model": model_name,
        "n_samples": len(df),
        "mean_dice": df["dice"].mean(),
        "std_dice": df["dice"].std(),
        "mean_iou": df["iou"].mean(),
        "std_iou": df["iou"].std(),
    }

    # Per-plane
    for plane in ["axial", "sagittal"]:
        sub = df[df["plane"] == plane]
        if len(sub) > 0:
            metrics[f"{plane}_dice"] = sub["dice"].mean()
            metrics[f"{plane}_iou"] = sub["iou"].mean()

    return metrics


def generate_comparison_table(all_metrics, save_dir):
    """Generate comparison table for baselines vs PCRadNet."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(save_dir / "baseline_comparison.csv", index=False)

    # LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of PC-RadNet against zero-shot baselines on the validation set.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Params & mDice$\uparrow$ & mIoU$\uparrow$ & Axial Dice & Sag Dice \\",
        r"\midrule",
    ]

    for m in all_metrics:
        line = (f"{m['model']} & — & {m['mean_dice']:.4f} & {m['mean_iou']:.4f} "
                f"& {m.get('axial_dice', 0):.4f} & {m.get('sagittal_dice', 0):.4f} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(save_dir / "table_baselines.tex", "w") as f:
        f.write("\n".join(lines))

    print(f"\nBaseline comparison saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    split_file = Path(CFG["paths"]["scripts"]) / "data" / f"{args.split}.txt"
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    print("\n" + "="*60)
    print("Evaluating SAM ViT-B (box prompt)")
    print("="*60)
    t0 = time.time()
    df_sam = evaluate_sam_vitb(str(split_file), args.max_samples)
    if len(df_sam) > 0:
        df_sam.to_csv(results_dir / f"baseline_sam_{args.split}.csv", index=False)
        metrics = aggregate_baseline_metrics(df_sam, "SAM ViT-B (box prompt)")
        metrics["time_sec"] = time.time() - t0
        all_metrics.append(metrics)
        print(f"  mDice={metrics['mean_dice']:.4f}, mIoU={metrics['mean_iou']:.4f}")

    if all_metrics:
        generate_comparison_table(all_metrics, results_dir)
        with open(results_dir / f"baseline_metrics_{args.split}.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
