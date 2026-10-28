"""
threshold_sensitivity.py - Task 6: Threshold sensitivity analysis for disorder detection.
Sweeps decision threshold from 0.1 to 0.9 for all 9 disorder classes,
using saved per-image predictions from PCRadNet A2 and A3.

Usage:
    python scripts/threshold_sensitivity.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from repo_config import load_config

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

THRESHOLDS = np.arange(0.1, 1.0, 0.05).tolist()


def load_predictions_and_gt(ablation="A2"):
    """Load patient-level disorder predictions and ground truth.
    Re-runs PCRadNet on test set to get raw sigmoid outputs.
    """
    # Load ground truth
    root = Path(CFG["paths"]["dataset_root"])
    df_notes = pd.read_csv(root / "structured_notes.csv")

    gt_map = {}
    for _, row in df_notes.iterrows():
        pid = int(row["Patient_ID"])
        gt_map[pid] = [int(row.get(c, 0)) for c in DISORDER_COLS]

    # Load model and get raw predictions
    sys.path.insert(0, str(SCRIPT_DIR))

    from train import (SpineDataset, PCRadNet, get_val_transforms,
                            collate_fn as peft_collate)

    ckpt_path = Path(CFG["paths"]["checkpoints"]) / f"best_by_mdice_{ablation}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None, None, None

    # Build model
    model = PCRadNet(CFG).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load test data
    split_file = Path(CFG["paths"]["splits_dir"]) / "test.txt"
    test_ds = SpineDataset(str(split_file), CFG, get_val_transforms(), use_metadata=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=peft_collate
    )

    # Collect per-patient disorder sigmoid outputs
    patient_preds = defaultdict(list)
    patient_gt = {}

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            plane_ids = batch["plane_id"].to(DEVICE)

            outputs = model(images, batch["prompt_text"], plane_ids)
            disorder_logits = outputs.get("disorder_logits")
            if disorder_logits is None:
                continue

            probs = torch.sigmoid(disorder_logits).cpu().numpy()
            pids = batch["patient_id"].numpy()
            gt_disorders = batch["disorder"].numpy()

            for i in range(len(pids)):
                pid = int(pids[i])
                patient_preds[pid].append(probs[i])
                if pid not in patient_gt:
                    patient_gt[pid] = gt_disorders[i]

    # Average predictions per patient (multiple images per patient)
    pids_sorted = sorted(patient_preds.keys())
    all_preds = np.array([np.mean(patient_preds[pid], axis=0) for pid in pids_sorted])
    all_gt = np.array([patient_gt[pid] for pid in pids_sorted])

    del model
    torch.cuda.empty_cache()

    return all_preds, all_gt, pids_sorted


def compute_f1_at_threshold(preds, gt, threshold):
    """Compute F1, precision, recall at given threshold."""
    pred_bin = (preds >= threshold).astype(int)
    tp = (pred_bin * gt).sum()
    fp = (pred_bin * (1 - gt)).sum()
    fn = ((1 - pred_bin) * gt).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(f1), float(precision), float(recall)


def run_threshold_analysis(all_preds, all_gt):
    """Run threshold sweep for all disorders."""
    results = []

    for d_idx, dname in enumerate(DISORDER_NAMES):
        preds_d = all_preds[:, d_idx]
        gt_d = all_gt[:, d_idx]
        n_pos = int(gt_d.sum())
        n_neg = int(len(gt_d) - n_pos)

        best_f1 = 0
        best_thresh = 0.5

        for thresh in THRESHOLDS:
            f1, prec, rec = compute_f1_at_threshold(preds_d, gt_d, thresh)
            results.append({
                "disorder": dname,
                "threshold": round(thresh, 2),
                "f1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "n_positive": n_pos,
                "n_negative": n_neg,
            })
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"  {dname:25s} | n_pos={n_pos:3d} | best_thresh={best_thresh:.2f} | best_F1={best_f1:.4f}")

    return pd.DataFrame(results)


def find_optimal_thresholds(df):
    """Find optimal threshold per disorder that maximizes F1."""
    optimal = {}
    for dname in DISORDER_NAMES:
        sub = df[df["disorder"] == dname]
        best_row = sub.loc[sub["f1"].idxmax()]
        optimal[dname] = {
            "optimal_threshold": float(best_row["threshold"]),
            "best_f1": float(best_row["f1"]),
            "precision_at_best": float(best_row["precision"]),
            "recall_at_best": float(best_row["recall"]),
            "n_positive": int(best_row["n_positive"]),
            "default_f1": float(sub[sub["threshold"] == 0.5]["f1"].values[0]) if 0.5 in sub["threshold"].values else 0,
        }
    return optimal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="A2", choices=["A0", "A1", "A2", "A3"])
    args = parser.parse_args()

    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Threshold sensitivity analysis for ablation {args.ablation}")
    print("=" * 70)

    all_preds, all_gt, pids = load_predictions_and_gt(args.ablation)
    if all_preds is None:
        return

    print(f"Loaded {len(pids)} patients, {all_preds.shape[1]} disorders")
    print(f"Sweeping thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f}")
    print()

    # Run analysis
    df = run_threshold_analysis(all_preds, all_gt)
    df.to_csv(results_dir / f"threshold_sensitivity_{args.ablation}.csv", index=False)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(df)
    with open(results_dir / f"optimal_thresholds_{args.ablation}.json", "w") as f:
        json.dump(optimal, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLDS vs DEFAULT (0.5)")
    print(f"{'='*70}")
    print(f"{'Disorder':<25} {'Default F1':>10} {'Optimal θ':>10} {'Best F1':>10} {'ΔF1':>8}")
    print("-" * 63)
    total_default = 0
    total_optimal = 0
    for dname, info in optimal.items():
        delta = info["best_f1"] - info["default_f1"]
        print(f"{dname:<25} {info['default_f1']:>10.4f} {info['optimal_threshold']:>10.2f} "
              f"{info['best_f1']:>10.4f} {delta:>+8.4f}")
        total_default += info["default_f1"]
        total_optimal += info["best_f1"]
    print("-" * 63)
    n = len(optimal)
    print(f"{'Mean':<25} {total_default/n:>10.4f} {'':>10} {total_optimal/n:>10.4f} "
          f"{(total_optimal-total_default)/n:>+8.4f}")

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
