"""
sagittal_height.py - disc localisation, level assignment and disc-height measurement on
mid-sagittal images from PC-RadNet's own sagittal segmentation (no ground truth used
at inference).

Steps per mid-sagittal image
  1. IVD components = connected components of the prompted "intervertebral disc" mask
     (sagittal plane), area >= 40 px; sacrum = largest component of the "sacrum" mask.
  2. Level assignment: the IVD component nearest to the sacrum is L5-S1; the remaining
     components are ordered cranially by centroid row, giving L4-L5 and L3-L4.
  3. Height: principal axes of each component; the disc height is the mask extent along
     the minor axis, taken as the median over chords in the central 20 % of the major axis.
  4. Pixels are converted to mm with the study's in-plane spacing on the 384 x 384 grid.

Validation (evaluation only): level assignment is checked against the stored landmark
midpoints; the same measurement on the ground-truth masks isolates the algorithmic error.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repo_config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = Path(load_config()["paths"]["dataset_root"])
OUT = ROOT / "results" / "analysis"
LEVELS = ["D5", "D4", "D3"]  # caudal to cranial: L5-S1, L4-L5, L3-L4


def load_gt():
    df = pd.read_csv(DS / "IVDHeights.csv")
    gt = {}
    for _, r in df.iterrows():
        pid = int(r.Patient_ID)
        gt[pid] = {}
        for L in LEVELS:
            a, b = r[f"{L}_Coord"].split(";")
            x1, y1 = map(float, a.split(","))
            x2, y2 = map(float, b.split(","))
            d = np.hypot(x2 - x1, y2 - y1)
            gt[pid][L] = {"h": float(r[f"{L}_Ht"]), "mid": ((x1 + x2) / 2, (y1 + y2) / 2),
                          "spacing": float(r[f"{L}_Ht"]) / d if d > 0 else np.nan}
    return gt


def components(mask, min_area=40):
    lab, n = ndimage.label(mask)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) >= min_area:
            comps.append(np.stack([xs, ys], 1).astype(float))
    return comps


def thickness(pts):
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    major, minor = vt[0], vt[1]
    a = (pts - c) @ major
    b = (pts - c) @ minor
    span = a.max() - a.min()
    centre = np.abs(a - (a.max() + a.min()) / 2) <= 0.1 * span
    bins = np.round(a[centre]).astype(int)
    chords = []
    for k in np.unique(bins):
        bb = b[centre][bins == k]
        chords.append(bb.max() - bb.min() + 1.0)
    return float(np.median(chords)) if chords else np.nan, c


def assign_levels(ivd_mask, sac_mask):
    comps = components(ivd_mask)
    if not comps:
        return {}
    sac = components(sac_mask, min_area=100)
    cents = [p.mean(0) for p in comps]
    if sac:
        s = max(sac, key=len)
        dist = [np.min(np.hypot(*(s - c).T)) for c in cents]
        i5 = int(np.argmin(dist))
    else:
        i5 = int(np.argmax([c[1] for c in cents]))
    above = [i for i in range(len(comps)) if cents[i][1] < cents[i5][1] and i != i5]
    above.sort(key=lambda i: -cents[i][1])
    order = [i5] + above
    return {LEVELS[k]: comps[order[k]] for k in range(min(3, len(order)))}


def contains(pts, xy, tol=3.0):
    return bool(np.min(np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])) <= tol)


def measure(ivd_mask, sac_mask, gt_p, spacing):
    res = {}
    lv = assign_levels(ivd_mask, sac_mask)
    for L in LEVELS:
        if L not in lv:
            res[L] = {"pred_mm": np.nan, "level_ok": False}
            continue
        t, c = thickness(lv[L])
        res[L] = {"pred_mm": t * spacing, "level_ok": contains(lv[L], gt_p[L]["mid"])}
    return res


def main(model="A2", split="test", seq="T2", thr=128):
    gt = load_gt()
    npz = np.load(OUT / "unified" / f"{model}_{split}_sagmasks.npz")
    keys = sorted({k.split("|")[0] for k in npz.files})
    rows = []
    for fname in keys:
        if f"_{seq}_" not in fname:
            continue
        pid = int(fname.split("_")[2])
        if pid not in gt:
            continue
        g = gt[pid]
        spacing = np.nanmedian([g[L]["spacing"] for L in LEVELS])
        ivd_p = npz[f"{fname}|IVD_sag"] >= thr if f"{fname}|IVD_sag" in npz.files else np.zeros((384, 384), bool)
        sac_p = npz[f"{fname}|Sacrum"] >= thr if f"{fname}|Sacrum" in npz.files else np.zeros((384, 384), bool)
        gm = np.array(Image.open(DS / "masks_indexed" / fname))
        pred = measure(ivd_p, sac_p, g, spacing)
        orac = measure(gm == 2, gm == 3, g, spacing)
        for L in LEVELS:
            rows.append({"fname": fname, "patient_id": pid, "level": L, "spacing": spacing,
                         "gt_mm": g[L]["h"], "pred_mm": pred[L]["pred_mm"], "level_ok": pred[L]["level_ok"],
                         "oracle_mm": orac[L]["pred_mm"], "oracle_level_ok": orac[L]["level_ok"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sag_height_{model}_{split}_{seq}.csv", index=False)
    ok = df.dropna(subset=["pred_mm"])
    summ = {"model": model, "split": split, "seq": seq, "n_discs": len(df), "n_measured": len(ok),
            "level_assignment_acc": float(df.level_ok.mean()),
            "mae_all": float((ok.pred_mm - ok.gt_mm).abs().mean()),
            "mae_level_ok": float((ok[ok.level_ok].pred_mm - ok[ok.level_ok].gt_mm).abs().mean()),
            "oracle_mae": float((df.oracle_mm - df.gt_mm).abs().mean()),
            "oracle_level_acc": float(df.oracle_level_ok.mean())}
    print(json.dumps(summ, indent=1))
    return df


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:]))
