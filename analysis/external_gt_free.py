"""
external_gt_free.py - external evaluation without ground-truth information at inference.

Differences from scripts/evaluate_external_v2.py
  * no ROI crop: the whole slice is padded to square and resized to 384 x 384;
  * SPIDER mid-sagittal slice = geometric centre of the volume (n_slices // 2), not the
    centre of the slices that contain annotated foreground;
  * ground truth is used only to score the predictions;
  * SPIDER disorders are scored per patient against an always-positive reference, and
    the prompted head as well as the multi-class sagittal channel are reported.
The earlier GT-cropped setting is re-run with the same code (--oracle-crop) so the two
settings can be compared directly.

Usage:
    python external_gt_free.py --dataset spider|spineweb|spineparsenet [--oracle-crop]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_external_v2 as ev  # noqa: E402

ev.CKPT_DIR = ROOT / "checkpoints"
OUT = ROOT / "results" / "analysis" / "external_rev"
OUT.mkdir(parents=True, exist_ok=True)
ev.RESULTS_DIR = OUT
ev.VIZ_DIR = OUT / "viz"
ev.VIZ_DIR.mkdir(exist_ok=True)

_orig_pre = ev.preprocess_slice


def set_crop(oracle):
    def pre(img_2d, mask_2d=None, target_size=384, do_roi_crop=True):
        return _orig_pre(img_2d, mask_2d, target_size=target_size, do_roi_crop=oracle)
    ev.preprocess_slice = pre


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


@torch.no_grad()
def spider(model, oracle):
    import SimpleITK as sitk
    from sklearn.metrics import roc_auc_score
    d = ev.DATASET_PATHS["spider"]
    grad = pd.read_csv(d / "radiological_gradings.csv")
    bulg = grad.groupby("Patient")["Disc bulging"].max().to_dict()
    files = sorted(f for f in os.listdir(d / "images") if "_t2." in f and "SPACE" not in f)
    rows, drows = [], []
    for fname in files:
        pid = int(fname.split("_")[0])
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "images" / fname)))
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "masks" / fname)))
        if oracle:
            valid = [s for s in range(img.shape[2]) if (ev.spider_label_to_ours(msk[:, :, s]) > 0).mean() > 0.01]
            if not valid:
                continue
            s = valid[len(valid) // 2]
        else:
            s = img.shape[2] // 2
        ours = ev.spider_label_to_ours(msk[:, :, s])
        x, m384 = ev.preprocess_slice(img[:, :, s], ours, target_size=384)
        seg, prm, _, dis = ev.run_per_class_inference(model, x, plane_id=1, class_list=[1, 2, 3, 4])
        row = {"patient": pid, "slice": int(s), "n_slices": int(img.shape[2])}
        for c, nm in [(1, "VB"), (2, "IVD_sag"), (4, "SC_sag")]:
            gt = torch.from_numpy((m384 == c).astype(np.float32))
            if gt.sum() < 10:
                continue
            row[f"prompted_dice_{nm}"] = ev.compute_dice(prm[c], gt)
            row[f"mc_dice_{nm}"] = ev.compute_dice(seg[c], gt)
        gt_vb = torch.from_numpy((m384 == 1).astype(np.float32))
        if gt_vb.sum() >= 10:
            union = torch.maximum(prm[1], prm[3])
            row["prompted_dice_VBuSacrum_vs_VB"] = ev.compute_dice(union, gt_vb)
        rows.append(row)
        if pid in bulg:
            drows.append({"patient": pid, "gt_bulging": int(bulg[pid]), "prob_bulging": float(dis[2])})
    seg = pd.DataFrame(rows)
    dd = pd.DataFrame(drows).drop_duplicates("patient")
    y = dd.gt_bulging.values
    res = {"setting": "oracle_crop" if oracle else "gt_free", "n_slices": int(len(seg))}
    for c in [c for c in seg.columns if "dice" in c]:
        res[c] = [float(seg[c].mean()), float(seg[c].std()), int(seg[c].notna().sum())]
    pc = [c for c in seg.columns if c.startswith("prompted_dice_") and "VBu" not in c]
    mc = [c for c in seg.columns if c.startswith("mc_dice_")]
    res["prompted_mean_fg"] = float(seg[pc].mean(1).mean())
    res["mc_mean_fg"] = float(seg[mc].mean(1).mean())
    res["bulging"] = {"n_patients": int(len(dd)), "n_pos": int(y.sum()),
                      "model_P_R_F1@0.5": f1(y, (dd.prob_bulging.values > 0.5).astype(int)),
                      "always_positive_P_R_F1": f1(y, np.ones_like(y)),
                      "auc": float(roc_auc_score(y, dd.prob_bulging.values)) if 0 < y.sum() < len(y) else None}
    tag = res["setting"]
    seg.to_csv(OUT / f"spider_seg_{tag}.csv", index=False)
    dd.to_csv(OUT / f"spider_bulging_{tag}.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["spider", "spineweb", "spineparsenet"], required=True)
    ap.add_argument("--oracle-crop", action="store_true")
    a = ap.parse_args()
    set_crop(a.oracle_crop)
    model = ev.load_model("best_by_mdice_A2.pt")
    tag = "oracle_crop" if a.oracle_crop else "gt_free"
    if a.dataset == "spider":
        res = spider(model, a.oracle_crop)
    elif a.dataset == "spineweb":
        ev.RESULTS_DIR = OUT / f"spineweb_{tag}"; ev.RESULTS_DIR.mkdir(exist_ok=True)
        res = ev.evaluate_spineweb(model, save_viz=False)
    else:
        ev.RESULTS_DIR = OUT / f"spineparsenet_{tag}"; ev.RESULTS_DIR.mkdir(exist_ok=True)
        res = ev.evaluate_spine_new(model, max_cases=50, save_viz=False)
    res["setting"] = tag
    print(json.dumps(res, indent=1, default=str))
    with open(OUT / f"{a.dataset}_{tag}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
