"""
ablation_table.py - one row per model for the ablation table, from
results/analysis/unified/<name>_{test,val}_*.csv.

Columns: prompted Dice / IoU (original prompts, patient-cluster 95% CI), all-class prompted
Dice, multi-class output selection Dice, grade exact accuracy and quadratic kappa (468 axial
images), height MAE (234 T2w axial images), patient-level disorder macro-AUC, and the paired
patient-level difference in prompted Dice against a reference model.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patient_statistics as sc  # noqa: E402

U = sc.U


def row(name, ref=None):
    op = pd.read_csv(U / f"{name}_test_origprompt.csv")
    ac = pd.read_csv(U / f"{name}_test_allclass.csv")
    c = pd.read_csv(U / f"{name}_test_clinical.csv")
    r = {"name": name}
    r["dice"] = sc.fast_cluster_mean_ci(op, "prompted_dice")
    r["iou"] = sc.fast_cluster_mean_ci(op, "prompted_iou")[0]
    r["dice_allclass"] = sc.fast_cluster_mean_ci(ac, "prompted_dice")[0]
    r["dice_mcsel"] = sc.fast_cluster_mean_ci(ac, "mc_dice")[0]
    g = c[c.gt_grade > 0]
    r["grade_exact"] = float((g.gt_grade == g.pred_grade).mean())
    r["kappa_quad"] = float(cohen_kappa_score(g.gt_grade, g.pred_grade, labels=[1, 2, 3, 4, 5], weights="quadratic"))
    h = c[(c.gt_height > 0) & c.fname.str.contains("axial_T2_")]
    r["height_mae"] = float((h.pred_height - h.gt_height).abs().mean())
    pt = sc.patient_scores(c)
    aucs = []
    for d in sc.DIS:
        y = pt[f"dis_{d}_gt"].values.astype(int)
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, pt[f"dis_{d}_prob"].values))
    r["dis_macro_auc"] = float(np.mean(aucs))
    if ref:
        b = pd.read_csv(U / f"{ref}_test_origprompt.csv")
        r["vs_ref"] = sc.paired_patient_test(op, b, "prompted_dice", key=("fname",), n_perm=20000)
    return r


if __name__ == "__main__":
    ref = sys.argv[1]
    names = sys.argv[2:]
    out = [row(n, ref if n != ref else None) for n in names]
    for r in out:
        print(json.dumps(r))
    path = sc.OUT / "ablation_rows.json"
    old = json.load(open(path)) if path.exists() else {}
    old.update({r["name"]: r for r in out})
    json.dump(old, open(path, "w"), indent=1)
