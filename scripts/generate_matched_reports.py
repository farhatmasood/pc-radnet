"""
generate_matched_reports.py - Generate RATF reports for the 20 MedGemma-matched
patient IDs using existing test-set inference predictions (test_A2_per_image.csv).

Outputs:
  results/ratf_reports_matched/report_XXXX.txt   (20 plain-text RATF reports)
  results/ratf_reports_matched/summary.json        (per-patient grade/height summary)

Usage:
    python scripts/generate_matched_reports.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from repo_config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import yaml
CFG = load_config()

# -- paths ----------------------------------------------------------------
RESULTS_DIR = Path(CFG["paths"]["results"])
# Ensemble: average A1 + A3 predictions (best mDice ablations)
PREDICTIONS_CSV_A1 = RESULTS_DIR / "test_A1_per_image.csv"
PREDICTIONS_CSV_A3 = RESULTS_DIR / "test_A3_per_image.csv"
OUT_DIR = RESULTS_DIR / "ratf_reports_matched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-disorder patient-level decision thresholds
# Calibrated to balance sensitivity/specificity per pathology
DISORDER_THRESHOLDS = {
    "Herniation": 0.40,        # lower - model tends to under-predict
    "Protrusion": 0.35,        # lower - moderate under-prediction
    "Bulge": 0.70,             # higher - model over-predicts bulge
    "ThecalComp": 0.55,        # slightly above default
    "NerveComp": 0.65,         # higher - model over-predicts nerve comp
    "Stenosis": 0.20,          # lower - rare, model misses
    "Spondylolisthesis": 0.40, # lower - rare, model misses
    "Desiccated": 0.35,        # lower - model under-predicts
    "LigFlavum": 0.25,         # lower - rare, model misses
}

TARGET_IDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113,
              118, 124, 125, 140, 150, 152, 165, 167, 177, 181]

# -- report constants (mirrors generate_report.py) -----------------------
LEVEL_MAP   = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
GRADE_DESC  = {1: "Normal", 2: "Near-normal", 3: "Moderate degeneration",
               4: "Advanced degeneration", 5: "Severe collapse"}
GRADE_CLIN  = {1: "No clinical significance",
               2: "Minimal degeneration; conservative management",
               3: "Moderate degeneration; consider follow-up",
               4: "Significant degeneration; clinical review recommended",
               5: "Severe collapse; specialist referral advised"}
DISORDER_NAMES = [
    "disc herniation", "disc protrusion", "disc bulge",
    "thecal sac compression", "nerve root compression",
    "spinal canal stenosis", "spondylolisthesis",
    "desiccated disc", "ligamentum flavum hypertrophy",
]
DISORDER_COLS = [
    "disorder_Herniation_prob", "disorder_Protrusion_prob", "disorder_Bulge_prob",
    "disorder_ThecalComp_prob", "disorder_NerveComp_prob", "disorder_Stenosis_prob",
    "disorder_Spondylolisthesis_prob", "disorder_Desiccated_prob", "disorder_LigFlavum_prob",
]


def classify_height(h):
    if h >= 12:
        return "normal"
    elif h >= 9:
        return "mildly reduced"
    elif h >= 6:
        return "moderately reduced"
    else:
        return "severely reduced"


def generate_patient_report(pid, level_data, active_disorders=None):
    """Generate structured RATF report.

    Parameters
    ----------
    active_disorders : set or None
        Indices of disorders that passed the patient-level gate.
        Only these disorders will appear in per-level findings.
    """
    lines = []
    lines.append("LUMBAR SPINE MRI REPORT")
    lines.append(f"Patient ID: {pid:04d}")

    lines.append("")
    lines.append("TECHNIQUE:")
    lines.append("  MRI of the lumbar spine was performed in axial and sagittal planes")
    lines.append("  using T1 and T2 weighted sequences.")
    lines.append("")
    lines.append("FINDINGS:")

    for level in ["D3", "D4", "D5"]:
        if level not in level_data:
            continue
        ld = level_data[level]
        lname = LEVEL_MAP[level]
        lines.append(f"\n  {lname} Level:")

        grade = ld.get("grade", -1)
        if grade > 0:
            lines.append(f"    Pfirrmann Grade: {grade}/5 — {GRADE_DESC.get(grade, 'unknown')}")
            lines.append(f"    Clinical: {GRADE_CLIN.get(grade, 'unknown')}")
        else:
            lines.append("    Pfirrmann Grade: N/A")

        ht = ld.get("height", -1)
        if ht > 0:
            lines.append(f"    Disc Height: {ht:.1f} px ({classify_height(ht)})")
        else:
            lines.append("    Disc Height: N/A")

        disorders = ld.get("disorders", [])
        present = []
        for i, d in enumerate(disorders):
            if active_disorders is not None:
                if i in active_disorders and d > 0.05:
                    present.append(DISORDER_NAMES[i])
            else:
                if d > 0.5:
                    present.append(DISORDER_NAMES[i])
        if present:
            lines.append(f"    Findings: {', '.join(present)}")
        else:
            lines.append("    Findings: No significant abnormality detected.")

        dice = ld.get("seg_dice", -1)
        if dice > 0:
            qual = ("excellent" if dice > 0.9 else
                    "good" if dice > 0.8 else
                    "moderate" if dice > 0.7 else "limited")
            lines.append(f"    Segmentation confidence: {qual} (Dice = {dice:.3f})")

    lines.append("")
    lines.append("IMPRESSION:")
    imp_items = []
    for level in ["D3", "D4", "D5"]:
        if level not in level_data:
            continue
        ld = level_data[level]
        lname = LEVEL_MAP[level]
        grade = ld.get("grade", -1)
        if grade >= 4:
            imp_items.append(f"Advanced disc degeneration at {lname}")
        elif grade == 3:
            imp_items.append(f"Moderate disc degeneration at {lname}")
        disorders = ld.get("disorders", [])
        for i, d in enumerate(disorders):
            if active_disorders is not None:
                if i in active_disorders and d > 0.05:
                    imp_items.append(f"{DISORDER_NAMES[i].capitalize()} at {lname}")
            else:
                if d > 0.5:
                    imp_items.append(f"{DISORDER_NAMES[i].capitalize()} at {lname}")

    if imp_items:
        for k, item in enumerate(imp_items, 1):
            lines.append(f"  {k}. {item}")
    else:
        lines.append("  No significant abnormality detected in the lumbar spine.")

    lines.append("")
    return "\n".join(lines)


def aggregate_patient_data(df_pid):
    """Aggregate per-image rows → per-level dict for one patient."""
    level_data = {}
    for level in ["D3", "D4", "D5"]:
        rows = df_pid[(df_pid["level"] == level) & (df_pid["plane"] == "axial")]
        if rows.empty:
            continue

        # Grade: mode (most common predicted grade)
        valid_grades = rows[rows["pred_grade"] > 0]["pred_grade"]
        grade = int(valid_grades.mode().iloc[0]) if not valid_grades.empty else -1

        # Height: mean predicted height
        valid_ht = rows[rows["pred_height"] > 0]["pred_height"]
        height = float(valid_ht.mean()) if not valid_ht.empty else -1.0

        # Disorders: max probability across images
        disorders = []
        for col in DISORDER_COLS:
            if col in rows.columns:
                disorders.append(float(rows[col].max()))
            else:
                disorders.append(0.0)

        # Segmentation confidence: mean prompted_dice
        seg_dice = float(rows["prompted_dice"].mean()) if "prompted_dice" in rows.columns else -1.0

        level_data[level] = {
            "grade": grade,
            "height": height,
            "disorders": disorders,
            "seg_dice": seg_dice,
        }
    return level_data


def aggregate_patient_data_ensemble(df_pid_a1, df_pid_a3):
    """Aggregate per-image rows → per-level dict using A1+A3 ensemble."""
    level_data = {}
    for level in ["D3", "D4", "D5"]:
        rows_a1 = df_pid_a1[(df_pid_a1["level"] == level) & (df_pid_a1["plane"] == "axial")]
        rows_a3 = df_pid_a3[(df_pid_a3["level"] == level) & (df_pid_a3["plane"] == "axial")]
        if rows_a1.empty:
            continue

        # Grade: use A1 (primary checkpoint)
        valid_grades = rows_a1[rows_a1["pred_grade"] > 0]["pred_grade"]
        grade = int(valid_grades.mode().iloc[0]) if not valid_grades.empty else -1

        # Height: use A1
        valid_ht = rows_a1[rows_a1["pred_height"] > 0]["pred_height"]
        height = float(valid_ht.mean()) if not valid_ht.empty else -1.0

        # Disorders: average max prob from A1 and A3
        disorders = []
        for col in DISORDER_COLS:
            max_a1 = float(rows_a1[col].max()) if col in rows_a1.columns else 0.0
            max_a3 = float(rows_a3[col].max()) if not rows_a3.empty and col in rows_a3.columns else 0.0
            disorders.append((max_a1 + max_a3) / 2.0)

        # Segmentation confidence: use A1
        seg_dice = float(rows_a1["prompted_dice"].mean()) if "prompted_dice" in rows_a1.columns else -1.0

        level_data[level] = {
            "grade": grade,
            "height": height,
            "disorders": disorders,
            "seg_dice": seg_dice,
        }
    return level_data


def main():
    # Load ensemble predictions (A1 + A3)
    df_a1 = pd.read_csv(PREDICTIONS_CSV_A1)
    df_a3 = pd.read_csv(PREDICTIONS_CSV_A3)

    df_a1_targets = df_a1[df_a1["patient_id"].isin(TARGET_IDS)].copy()
    df_a3_targets = df_a3[df_a3["patient_id"].isin(TARGET_IDS)].copy()

    # Compute patient-level ensemble probabilities for gating
    DISORDER_SHORT = list(DISORDER_THRESHOLDS.keys())
    patient_active = {}  # pid -> set of active disorder indices
    for pid in sorted(TARGET_IDS):
        sub_a1 = df_a1_targets[(df_a1_targets["patient_id"] == pid) &
                               (df_a1_targets["plane"] == "axial")]
        sub_a3 = df_a3_targets[(df_a3_targets["patient_id"] == pid) &
                               (df_a3_targets["plane"] == "axial")]
        if sub_a1.empty:
            sub_a1 = df_a1_targets[df_a1_targets["patient_id"] == pid]
        if sub_a3.empty:
            sub_a3 = df_a3_targets[df_a3_targets["patient_id"] == pid]

        active = set()
        for i, (col, short) in enumerate(zip(DISORDER_COLS, DISORDER_SHORT)):
            max_a1 = float(sub_a1[col].max()) if col in sub_a1.columns else 0.0
            max_a3 = float(sub_a3[col].max()) if col in sub_a3.columns else 0.0
            ens_prob = (max_a1 + max_a3) / 2.0
            if ens_prob >= DISORDER_THRESHOLDS[short]:
                active.add(i)
        patient_active[pid] = active

    summary = {}
    reports_generated = 0

    for pid in sorted(TARGET_IDS):
        # Use A1 predictions for per-level details (grade, height, seg)
        df_pid = df_a1_targets[df_a1_targets["patient_id"] == pid]
        if df_pid.empty:
            print(f"  WARNING: no rows found for patient {pid:04d}")
            continue

        # For disorder probabilities, average A1 and A3 per image
        df_pid_a3 = df_a3_targets[df_a3_targets["patient_id"] == pid]
        level_data = aggregate_patient_data_ensemble(df_pid, df_pid_a3)
        active = patient_active[pid]

        report_text = generate_patient_report(pid, level_data,
                                              active_disorders=active)

        out_path = OUT_DIR / f"report_{pid:04d}.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)

        summary[f"{pid:04d}"] = {
            k: {sk: sv for sk, sv in v.items() if sk != "disorders"}
            for k, v in level_data.items()
        }
        for lvl, ld in level_data.items():
            present = [DISORDER_NAMES[i]
                       for i in range(len(ld["disorders"]))
                       if i in active and ld["disorders"][i] > 0.30]
            summary[f"{pid:04d}"][lvl]["disorders_present"] = present

        reports_generated += 1
        print(f"  Patient {pid:04d}: active={[DISORDER_NAMES[i] for i in sorted(active)]}")

    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. {reports_generated}/{len(TARGET_IDS)} reports written to {OUT_DIR}")
    print(f"Summary saved to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
