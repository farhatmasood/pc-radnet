"""
original_clinical_outputs.py - clinical statistics of the original A0-A3 evaluation.

The grading, height and disorder values reported for A0-A3 come from the per-image files
written by scripts/evaluate.py (results/{split}_{variant}_per_image.csv). This script
converts them to the column layout of evaluate_unified.py
(results/analysis/unified/{variant}orig_{split}_clinical.csv) and computes the same
patient-level statistics as patient_statistics.py.

Outputs: results/analysis/unified/*orig_*_clinical.csv, results/analysis/disorders_*orig.csv,
         results/analysis/stats_clinical_orig.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patient_statistics as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ["fname", "patient_id", "level", "plane", "gt_grade", "pred_grade", "gt_height", "pred_height"]


def convert(variant, split):
    d = pd.read_csv(ROOT / "results" / f"{split}_{variant}_per_image.csv")
    out = d[BASE].copy()
    for k in range(1, 5):  # cumulative probabilities were not stored by the original evaluation
        out[f"grade_p{k}"] = np.nan
    for c in d.columns:
        if c.startswith("disorder_"):
            out["dis_" + c[len("disorder_"):]] = d[c]
    sc.U.mkdir(parents=True, exist_ok=True)
    out.to_csv(sc.U / f"{variant}orig_{split}_clinical.csv", index=False)


def main():
    res = {}
    for v in ["A0", "A1", "A2", "A3"]:
        for split in ["test", "val"]:
            convert(v, split)
        name = f"{v}orig"
        g = sc.grading(name)
        g.pop("nonmonotone_rate", None)  # undefined without cumulative probabilities
        res[name] = {"grading": g, "height": sc.height(name), "disorders": sc.disorders(name)}
        print(name, round(g["image_level"]["exact"], 4), round(res[name]["height"]["model"]["mae"], 3),
              round(res[name]["disorders"]["macro_pat_AUC"], 3), flush=True)
    with open(sc.OUT / "stats_clinical_orig.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
