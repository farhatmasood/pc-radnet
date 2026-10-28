"""
rsna_summary.py - RSNA 2024 agreement of the unadapted model.

Input:  results/external_v2/rsna_results.csv (scripts/evaluate_external_v2.py, 100 studies)
Output: results/analysis/rsna_recomputed.json

  all5 / L3-S1     sagittal rows: predicted Pfirrmann grade mapped to three-tier canal stenosis
                   severity (0 Normal/Mild, 1 Moderate, 2 Severe) against the RSNA label; exact and
                   within-one agreement, linear-weighted kappa, and constant predictors per severity
  ax_all / ax_L3-S1  axial rows: stenosis probability of the disorder head against any
                   Moderate/Severe label; ROC-AUC, F1 at 0.5 and an always-positive F1
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LOWER = ["l3_l4", "l4_l5", "l5_s1"]


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def severity(sub):
    sub = sub.dropna(subset=["pred_severity"])
    y, p = sub.gt_severity.astype(int).values, sub.pred_severity.astype(int).values
    r = {"n": int(len(sub)), "studies": int(sub.study_id.nunique()), "exact": float((y == p).mean()),
         "w1": float((abs(y - p) <= 1).mean()), "kappa_lin": float(cohen_kappa_score(y, p, weights="linear")),
         "gt_dist": np.bincount(y, minlength=3).tolist(), "pred_dist": np.bincount(p, minlength=3).tolist()}
    for k in range(3):
        r[f"const{k}_exact"] = float((y == k).mean())
        r[f"const{k}_w1"] = float((abs(y - k) <= 1).mean())
    return r


def stenosis_head(sub):
    sub = sub.dropna(subset=["stenosis_prob"])
    y, s = sub.gt_positive.astype(int).values, sub.stenosis_prob.values
    return {"n": int(len(sub)), "pos": int(y.sum()), "auc": float(roc_auc_score(y, s)),
            "f1@0.5": f1(y, (s >= 0.5).astype(int)), "always_pos_f1": f1(y, np.ones_like(y)),
            "studies": int(sub.study_id.nunique())}


def main():
    d = pd.read_csv(ROOT / "results" / "external_v2" / "rsna_results.csv")
    sag, ax = d[d.plane == "sagittal"], d[d.plane == "axial"]
    res = {"all5": severity(sag), "L3-S1": severity(sag[sag.level.isin(LOWER)]),
           "ax_all": stenosis_head(ax), "ax_L3-S1": stenosis_head(ax[ax.level.isin(LOWER)])}
    with open(ROOT / "results" / "analysis" / "rsna_recomputed.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["L3-S1"]), json.dumps(res["ax_all"]))


if __name__ == "__main__":
    main()
