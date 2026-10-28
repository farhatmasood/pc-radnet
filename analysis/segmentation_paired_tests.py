"""
segmentation_paired_tests.py - patient-level confidence intervals and paired tests for
prompted segmentation of A0-A3 (ablation table and the comparisons in the text).

Inputs:  results/analysis/unified/{A0..A3}_test_{origprompt,allclass}.csv (evaluate_unified.py)
Output:  results/analysis/seg_paired_tests.json

  *_origprompt_*_ci   one stored prompt per image (624 images)
  *_allclass_*_ci     every present class queried with its canonical term (2,026 pairs)
  *_mcsel_ci          queried class read from the multi-class head of the same forward pass
  A3_vs_A2_*          paired difference A3 - A2 (cluster bootstrap, sign-flip permutation,
                      Wilcoxon on patient means, Wilcoxon on image or pair level)
  A2prompt_vs_*mcsel  A2 prompted head against multi-class output selection of A1 and A2
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patient_statistics as sc  # noqa: E402

VARIANTS = ["A0", "A1", "A2", "A3"]


def main():
    op = {n: pd.read_csv(sc.U / f"{n}_test_origprompt.csv") for n in VARIANTS}
    ac = {n: pd.read_csv(sc.U / f"{n}_test_allclass.csv") for n in VARIANTS}

    def mcsel(n):
        return ac[n].drop(columns=["prompted_dice"]).rename(columns={"mc_dice": "prompted_dice"})

    res = {"A3_vs_A2_orig": sc.paired_patient_test(op["A3"], op["A2"], "prompted_dice", key=("fname",))}
    m = op["A3"].merge(op["A2"], on="fname")
    res["A3_vs_A2_orig_imagelevel_wilcoxon_p"] = float(wilcoxon(m.prompted_dice_x - m.prompted_dice_y).pvalue)
    res["A3_vs_A2_allclass"] = sc.paired_patient_test(ac["A3"], ac["A2"], "prompted_dice", key=("fname", "cls"))
    m = ac["A3"].merge(ac["A2"], on=["fname", "cls"])
    res["A3_vs_A2_allclass_imagelevel_wilcoxon_p"] = float(wilcoxon(m.prompted_dice_x - m.prompted_dice_y).pvalue)
    res["A2prompt_vs_A1mcsel"] = sc.paired_patient_test(ac["A2"], mcsel("A1"), "prompted_dice", key=("fname", "cls"))
    res["A2prompt_vs_A2mcsel"] = sc.paired_patient_test(ac["A2"], mcsel("A2"), "prompted_dice", key=("fname", "cls"))
    for n in VARIANTS:
        res[f"{n}_origprompt_dice_ci"] = list(sc.fast_cluster_mean_ci(op[n], "prompted_dice"))
        res[f"{n}_origprompt_iou_ci"] = list(sc.fast_cluster_mean_ci(op[n], "prompted_iou"))
        res[f"{n}_allclass_dice_ci"] = list(sc.fast_cluster_mean_ci(ac[n], "prompted_dice"))
        res[f"{n}_allclass_iou_ci"] = list(sc.fast_cluster_mean_ci(ac[n], "prompted_iou"))
        res[f"{n}_mcsel_ci"] = list(sc.fast_cluster_mean_ci(ac[n], "mc_dice"))
    with open(sc.OUT / "seg_paired_tests.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["A3_vs_A2_orig"]), json.dumps(res["A2prompt_vs_A1mcsel"]))


if __name__ == "__main__":
    main()
