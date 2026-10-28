"""
same_split_summary.py - segmentation comparison table: networks trained on the LSMA-PQR split,
pre-trained models without fine-tuning, and PC-RadNet (A2) on the same image and class pairs.

Inputs:
  results/baseline_{nnunet,swinunetr,sam_ft}_test_perclass.csv  scripts/evaluate_direct_baselines.py
      (2D U-Net, 2D SwinUNETR, SAM ViT-B frozen encoder + trained decoder; 2,028 pairs)
  results/baseline_sam_test.csv                                  scripts/baseline_zeroshot.py --split test
      (SAM ViT-B with ground-truth boxes, released weights; 2,028 pairs)
  results/expanded_baselines/baseline_{medsam2,sam_auto}_test.csv scripts/evaluate_expanded_baselines.py
      (earlier 7-patient run; the full-test-split values are in results/analysis/baselines_full/)
  results/analysis/unified/A2_test_allclass.csv                  analysis/evaluate_unified.py
      (A2 prompted head and multi-class channel; 2,026 pairs)
Output: results/analysis/baselines_same_split.json
  all_pairs = [mean, 95% CI low, 95% CI high], patient-level cluster bootstrap (10,000 replicates);
  per_class = mean Dice per class; unet_minus_pc_prompted = mean paired difference U-Net - A2.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patient_statistics as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TRAINED = {"unet": "nnunet", "swinunetr": "swinunetr", "sam_ft": "sam_ft"}


def ci(d, col):
    return [round(x, 4) for x in sc.fast_cluster_mean_ci(d, col, seed=0)]


def with_pid(d):
    if "patient_id" not in d:
        d["patient_id"] = d.fname.str.split("_").str[2].astype(int)
    return d


def main():
    res = {}
    for name, stem in TRAINED.items():
        d = with_pid(pd.read_csv(RES / f"baseline_{stem}_test_perclass.csv"))
        res[name] = {"all_pairs": ci(d, "dice"), "n": int(len(d)),
                     "per_class": d.groupby("class_name").dice.mean().round(3).to_dict()}
    d = with_pid(pd.read_csv(RES / "baseline_sam_test.csv"))
    res["sam_box"] = {"all_pairs": ci(d, "dice"), "n": int(len(d))}
    for name in ["medsam2", "sam_auto"]:
        d = with_pid(pd.read_csv(RES / "expanded_baselines" / f"baseline_{name}_test.csv"))
        res[name] = {"all_pairs": ci(d, "dice"), "n": int(len(d)), "patients": int(d.patient_id.nunique())}
    a2 = pd.read_csv(sc.U / "A2_test_allclass.csv")
    res["pcradnet_allclass_prompted"] = ci(a2, "prompted_dice")
    res["pcradnet_allclass_mc"] = ci(a2, "mc_dice")
    unet = with_pid(pd.read_csv(RES / "baseline_nnunet_test_perclass.csv"))
    m = unet.merge(a2, left_on=["fname", "class_name"], right_on=["fname", "cls_name"])
    res["paired_n"] = int(len(m))
    res["unet_minus_pc_prompted"] = round(float((m.dice - m.prompted_dice).mean()), 4)
    with open(sc.OUT / "baselines_same_split.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v["all_pairs"] if isinstance(v, dict) else v for k, v in res.items()}))


if __name__ == "__main__":
    main()
