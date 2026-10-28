"""
patient_statistics.py - patient-level statistics (segmentation, grading,
disc height, disorders), computed from results/analysis/unified/*.csv.

All confidence intervals resample patients (cluster bootstrap, 10,000 replicates), so
the 2-8 correlated images or disc levels contributed by one patient move together.
Paired model comparisons use (i) a cluster bootstrap of the mean difference and (ii) an
exact-style sign-flip permutation test on patient-level mean differences (100,000 draws).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "results" / "analysis" / "unified"
OUT = ROOT / "results" / "analysis"
B = 10000
RNG = np.random.default_rng(2026)
DIS = ["Herniation", "Protrusion", "Bulge", "ThecalComp", "NerveComp", "Stenosis",
       "Spondylolisthesis", "Desiccated", "LigFlavum"]


def cluster_boot(df, stat, B=B, seed=0):
    """stat(df_subset) -> float; resample patients with replacement."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id")}
    pids = np.array(list(groups))
    vals = []
    for _ in range(B):
        s = rng.choice(pids, len(pids), replace=True)
        vals.append(stat(pd.concat([groups[p] for p in s], ignore_index=True)))
    vals = np.array(vals, float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def fast_cluster_mean_ci(df, col, B=B, seed=0):
    """Cluster bootstrap CI of an image-weighted mean using patient sums."""
    g = df.groupby("patient_id")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(df[col].mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_patient_test(a, b, col, key=("fname", "cls"), B=B, n_perm=100000, seed=1):
    m = a.merge(b, on=list(key), suffixes=("_a", "_b"))
    m["patient_id"] = m["patient_id_a"]
    m["d"] = m[f"{col}_a"] - m[f"{col}_b"]
    pdiff = m.groupby("patient_id")["d"].mean().values
    obs = m["d"].mean()
    mean, lo, hi = fast_cluster_mean_ci(m, "d", B=B, seed=seed)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(pdiff)))
    perm = (signs * pdiff).mean(1)
    p = (np.sum(np.abs(perm) >= abs(pdiff.mean())) + 1) / (n_perm + 1)
    from scipy.stats import wilcoxon
    w = wilcoxon(pdiff)
    return {"mean_diff": float(obs), "ci": [lo, hi], "n_pairs": int(len(m)), "n_patients": int(len(pdiff)),
            "perm_p": float(p), "wilcoxon_patient_p": float(w.pvalue),
            "patients_a_better": int((pdiff > 0).sum())}


# ---------------------------------------------------------------- segmentation
def seg_summary(name, split="test"):
    ac = pd.read_csv(U / f"{name}_{split}_allclass.csv")
    op = pd.read_csv(U / f"{name}_{split}_origprompt.csv")
    out = {}
    for col in ["prompted_dice", "prompted_iou", "mc_dice", "mc_iou"]:
        out[f"allclass_{col}"] = fast_cluster_mean_ci(ac, col)
    for col in ["prompted_dice", "prompted_iou"]:
        out[f"orig_{col}"] = fast_cluster_mean_ci(op, col)
    out["per_class_prompted"] = ac.groupby("cls_name")["prompted_dice"].mean().round(4).to_dict()
    out["per_class_mc"] = ac.groupby("cls_name")["mc_dice"].mean().round(4).to_dict()
    mc_cols = [c for c in op.columns if c.startswith("mc_dice_")]
    out["orig_per_class_mc"] = {c[8:]: float(op[c].dropna().mean()) for c in mc_cols}
    fg = []
    for _, r in op.iterrows():
        cols = ["mc_dice_IVD", "mc_dice_PE", "mc_dice_SC"] if r.plane == "axial" else \
            ["mc_dice_VB", "mc_dice_IVD_sag", "mc_dice_Sacrum", "mc_dice_SC_sag"]
        fg.append(np.mean([r[c] for c in cols]))
    op = op.assign(mean_fg=fg)
    out["orig_mean_fg_dice"] = fast_cluster_mean_ci(op, "mean_fg")
    return out


# ---------------------------------------------------------------- grading
def kappa(y, p, w):
    return cohen_kappa_score(y, p, labels=[1, 2, 3, 4, 5], weights=w)


def grade_metrics(df):
    y, p = df.gt_grade.values, df.pred_grade.values
    return {"exact": float((y == p).mean()), "within1": float((np.abs(y - p) <= 1).mean()),
            "mae": float(np.abs(y - p).mean()), "kappa": float(kappa(y, p, None)),
            "kappa_lin": float(kappa(y, p, "linear")), "kappa_quad": float(kappa(y, p, "quadratic"))}


def grading(name, split="test"):
    c = pd.read_csv(U / f"{name}_{split}_clinical.csv")
    g = c[c.gt_grade > 0].copy()
    tr = pd.read_csv(U / f"{name}_val_clinical.csv")  # not used for constants; constants are fixed grades
    res = {"n_images": int(len(g)), "n_patients": int(g.patient_id.nunique())}
    res["image_level"] = grade_metrics(g)
    res["image_level_ci"] = {k: cluster_boot(g, lambda d, k=k: grade_metrics(d)[k], B=2000, seed=3)
                             for k in ["exact", "within1", "mae", "kappa_lin", "kappa_quad"]}
    # disc level: T2 axial image of each disc (one observation per disc)
    t2 = g[g.fname.str.contains("_T2_")]
    res["disc_level_T2"] = grade_metrics(t2)
    res["disc_level_T2_ci"] = {k: cluster_boot(t2, lambda d, k=k: grade_metrics(d)[k], B=2000, seed=4)
                               for k in ["exact", "within1", "mae", "kappa_lin", "kappa_quad"]}
    # constant predictors
    const = {}
    for k in [2, 3]:
        d = g.assign(pred_grade=k)
        const[f"always_{k}"] = grade_metrics(d)
    res["constant"] = const
    # T1 vs T2 agreement on the same disc
    g["disc"] = g.patient_id.astype(str) + "_" + g.level
    piv = g.pivot_table(index="disc", columns=g.fname.str.contains("_T2_").map({True: "T2", False: "T1"}),
                        values="pred_grade")
    res["t1_t2_same_grade"] = float((piv["T1"] == piv["T2"]).mean())
    # confusion matrix
    cm = pd.crosstab(g.gt_grade, g.pred_grade).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    res["confusion"] = cm.values.tolist()
    # cumulative-probability monotonicity
    P = g[[f"grade_p{k}" for k in range(1, 5)]].values
    res["nonmonotone_rate"] = float((np.diff(P, axis=1) > 1e-6).any(1).mean())
    return res


# ---------------------------------------------------------------- height
def ccc(x, y):
    mx, my = x.mean(), y.mean()
    return 2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (mx - my) ** 2)


def icc_a1(x, y):
    X = np.stack([x, y], 1)
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def height_metrics(d, pred="pred_height", ref="gt_height"):
    e = d[pred] - d[ref]
    return {"mae": float(e.abs().mean()), "bias": float(e.mean()), "loa_lo": float(e.mean() - 1.96 * e.std()),
            "loa_hi": float(e.mean() + 1.96 * e.std()), "r": float(np.corrcoef(d[pred], d[ref])[0, 1]),
            "ccc": float(ccc(d[pred].values, d[ref].values)), "icc_a1": float(icc_a1(d[pred].values, d[ref].values))}


def height(name, split="test"):
    c = pd.read_csv(U / f"{name}_{split}_clinical.csv")
    h = c[(c.gt_height > 0) & c.fname.str.contains("axial_T2_")].copy()
    res = {"n": int(len(h)), "model": height_metrics(h)}
    res["model_ci"] = {k: cluster_boot(h, lambda d, k=k: height_metrics(d)[k], B=2000, seed=5)
                       for k in ["mae", "r", "ccc", "icc_a1"]}
    # constant baseline: mean training-set height
    import height_baseline
    mu = height_baseline.train_mean()
    hc = h.assign(const=mu)
    e = hc.const - hc.gt_height
    res["constant_train_mean"] = {"value": mu, "mae": float(e.abs().mean())}
    res["all_axial"] = height_metrics(c[(c.gt_height > 0) & c.plane.eq("axial")])
    res["n_all_axial"] = int(((c.gt_height > 0) & c.plane.eq("axial")).sum())
    return res


# ---------------------------------------------------------------- disorders
def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def patient_scores(c, q=0.75):
    agg = {f"dis_{d}_prob": (lambda s: np.quantile(s, q)) for d in DIS}
    agg.update({f"dis_{d}_gt": "max" for d in DIS})
    return c.groupby("patient_id").agg(agg).reset_index()


def best_threshold(y, s, grid=np.round(np.arange(0.05, 0.96, 0.01), 2)):
    best, bt = -1, 0.5
    for t in grid:
        v = f1(y, (s >= t).astype(int))[2]
        if v > best + 1e-12:
            best, bt = v, t
    return float(bt)


def disorders(name):
    val = pd.read_csv(U / f"{name}_val_clinical.csv")
    te = pd.read_csv(U / f"{name}_test_clinical.csv")
    pv, pt = patient_scores(val), patient_scores(te)
    rows = []
    for d in DIS:
        y_img, s_img = te[f"dis_{d}_gt"].values.astype(int), te[f"dis_{d}_prob"].values
        thr_img = best_threshold(val[f"dis_{d}_gt"].values.astype(int), val[f"dis_{d}_prob"].values)
        thr_pat = best_threshold(pv[f"dis_{d}_gt"].values.astype(int), pv[f"dis_{d}_prob"].values)
        p05 = f1(y_img, (s_img >= 0.5).astype(int))
        pth = f1(y_img, (s_img >= thr_img).astype(int))
        y_p, s_p = pt[f"dis_{d}_gt"].values.astype(int), pt[f"dis_{d}_prob"].values
        pp = f1(y_p, (s_p >= thr_pat).astype(int))
        ap = f1(y_p, np.ones_like(y_p))
        try:
            auc = roc_auc_score(y_p, s_p)
        except ValueError:
            auc = np.nan
        rows.append({"disorder": d, "img_support": int(y_img.sum()), "img_P": p05[0], "img_R": p05[1],
                     "img_F1": p05[2], "thr_img_val": thr_img, "img_F1_thr": pth[2], "delta_thr": pth[2] - p05[2],
                     "pat_support": int(y_p.sum()), "pat_n": int(len(y_p)), "thr_pat_val": thr_pat,
                     "pat_P": pp[0], "pat_R": pp[1], "pat_F1": pp[2], "pat_AUC": auc,
                     "always_pos_F1": ap[2]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"disorders_{name}.csv", index=False)
    return {"table": df.round(4).to_dict(orient="records"),
            "macro_img_F1": float(df.img_F1.mean()), "macro_img_F1_thr": float(df.img_F1_thr.mean()),
            "macro_delta": float(df.delta_thr.mean()), "macro_pat_F1": float(df.pat_F1.mean()),
            "macro_always_pos_F1": float(df.always_pos_F1.mean()), "macro_pat_AUC": float(df.pat_AUC.mean())}


def main(names):
    res = {}
    for n in names:
        r = {"seg": seg_summary(n)}
        try:
            r["grading"] = grading(n)
            r["height"] = height(n)
            r["disorders"] = disorders(n)
        except Exception as e:  # segmentation-only checkpoints still get seg stats
            r["clinical_error"] = repr(e)
        res[n] = r
        print(n, json.dumps({k: r["seg"][k] for k in ["allclass_prompted_dice", "allclass_mc_dice",
                                                      "orig_prompted_dice"]}), flush=True)
    with open(OUT / "stats_core.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A0", "A1", "A2", "A3"])
