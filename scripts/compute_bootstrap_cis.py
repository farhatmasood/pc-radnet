"""Compute 95% bootstrap CIs for all ablation variants and SAM baseline.
Run AFTER evaluate.py --all-ablations --split test regenerates per-image CSVs.
"""
import numpy as np
import pandas as pd
import pathlib
import json

np.random.seed(42)
N_BOOT = 10_000
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

variants = {
    "A0": RESULTS_DIR / "test_A0_per_image.csv",
    "A1": RESULTS_DIR / "test_A1_per_image.csv",
    "A2": RESULTS_DIR / "test_A2_per_image.csv",
    "A3": RESULTS_DIR / "test_A3_per_image.csv",
    "SAM": RESULTS_DIR / "baseline_sam_test.csv",
}

all_cis = {}
for name, path in variants.items():
    if not path.exists():
        print(f"[SKIP] {name}: {path} not found")
        continue
    df = pd.read_csv(path)
    # Determine dice column name
    dice_col = "prompted_dice" if "prompted_dice" in df.columns else "dice"
    iou_col = "prompted_iou" if "prompted_iou" in df.columns else "iou"
    
    dice = df[dice_col].values
    iou = df[iou_col].values if iou_col in df.columns else None
    n = len(dice)
    
    boot_dice = np.array([np.mean(np.random.choice(dice, size=n, replace=True)) for _ in range(N_BOOT)])
    ci_dice = np.percentile(boot_dice, [2.5, 97.5])
    
    result = {
        "n": n,
        "mean_dice": float(np.mean(dice)),
        "ci_dice_lo": float(ci_dice[0]),
        "ci_dice_hi": float(ci_dice[1]),
    }
    
    if iou is not None:
        boot_iou = np.array([np.mean(np.random.choice(iou, size=n, replace=True)) for _ in range(N_BOOT)])
        ci_iou = np.percentile(boot_iou, [2.5, 97.5])
        result.update({
            "mean_iou": float(np.mean(iou)),
            "ci_iou_lo": float(ci_iou[0]),
            "ci_iou_hi": float(ci_iou[1]),
        })
    
    all_cis[name] = result
    print(f"{name}: Dice={result['mean_dice']:.3f} [{ci_dice[0]:.3f}, {ci_dice[1]:.3f}]"
          + (f", IoU={result.get('mean_iou',0):.3f} [{ci_iou[0]:.3f}, {ci_iou[1]:.3f}]" if iou is not None else "")
          + f" (n={n})")

# Save results
out_path = RESULTS_DIR / "bootstrap_cis_all.json"
with open(out_path, "w") as f:
    json.dump(all_cis, f, indent=2)
print(f"\nSaved to {out_path}")
