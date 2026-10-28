"""Patient-level cluster-bootstrap CIs (10,000 replicates, pair-weighted mean) for the zero-shot
baselines run on the full test split, plus LSMA-PQR sagittal prompted Dice of A2 (per class and
per-slice mean over VB, IVD_sag, SC_sag, the definition used for the SPIDER rows).
Output: results/analysis/baselines_full/summary.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "baselines_full"
rng = np.random.default_rng(42)


def cluster_ci(df, col, reps=10000):
    g = df.groupby("patient_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return [float(df[col].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


res = {}
for name in ["medsam2", "sam_auto"]:
    f = OUT / f"{name}_test_full.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f)
    t = json.load(open(OUT / f"{name}_timing.json"))
    res[name] = {"n_pairs": len(d), "n_images": int(d.fname.nunique()), "n_patients": int(d.patient_id.nunique()),
                 "dice": cluster_ci(d, "dice"), "iou": float(d.iou.mean()), "s_per_image": t["avg_time_s"],
                 "per_class": d.groupby("target_cls").dice.mean().round(4).to_dict()}

a = pd.read_csv(ROOT / "results" / "analysis" / "unified" / "A2_test_allclass.csv")
sag = a[a.plane == "sagittal"].pivot(index="fname", columns="cls_name", values="prompted_dice")
res["lsma_sag_prompted"] = {
    "n": len(sag),
    "per_class_mean_sd": {k: [float(sag[k].mean()), float(sag[k].std())] for k in sag.columns},
    "per_slice_mean_3cls": float(sag[["VB", "IVD_sag", "SC_sag"]].mean(1).mean()),
    "per_slice_mean_4cls": float(sag.mean(1).mean()),
}
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
