"""
compare_medgemma_reports.py - MedGemma-4B-IT vs RATF Report Generation Comparison
Generates radiology reports using MedGemma (multimodal VLM) and compares
against the template-based RATF pipeline on factual consistency metrics.

Usage:
    python scripts/compare_medgemma_reports.py --mode generate
    python scripts/compare_medgemma_reports.py --mode evaluate
    python scripts/compare_medgemma_reports.py --mode all
"""
import os, sys, json, argparse, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import yaml
import torch
from repo_config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()

# Paths
MEDGEMMA_MODEL_PATH = r"D:\2.3 Code_s\MedGemma\models\models--google--medgemma-4b-it\snapshots\efe6cc02361759b6bd501c654ddb7c9d25ec509d"
RESULTS_DIR = Path(CFG["paths"]["results"])
DATASET_ROOT = Path(CFG["paths"]["dataset_root"])
IMAGES_DIR = Path(CFG["paths"]["images"])

DISORDER_NAMES = [
    "disc herniation", "disc protrusion", "disc bulge",
    "thecal sac compression", "nerve root compression",
    "spinal canal stenosis", "spondylolisthesis",
    "desiccated disc", "ligamentum flavum hypertrophy",
]

DISORDER_COLS = [
    "Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
    "Thecal_Sac_Compression", "Nerve_Root_Compression",
    "Spinal_Canal_Stenosis", "Spondylolisthesis",
    "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy",
]

LEVEL_MAP = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}

GRADE_DESC = {
    1: "Grade I — homogeneous, bright white signal, normal disc height",
    2: "Grade II — inhomogeneous with or without horizontal bands, bright white signal",
    3: "Grade III — inhomogeneous, intermediate gray signal, normal to slightly decreased disc height",
    4: "Grade IV — inhomogeneous, dark gray to black signal, slightly decreased disc height",
    5: "Grade V — inhomogeneous, black with collapsed disc space",
}


# ═══════════════════════════════════════════════════════════════════
# MedGemma Report Generator
# ═══════════════════════════════════════════════════════════════════

def load_medgemma():
    """Load MedGemma-4B-IT model and processor."""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading MedGemma from {MEDGEMMA_MODEL_PATH}")
    print("  Loading processor...")
    processor = AutoProcessor.from_pretrained(MEDGEMMA_MODEL_PATH)

    print("  Loading model (bfloat16, auto device map)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  MedGemma loaded. Peak VRAM: {vram:.2f} GB")
    return model, processor


def build_medgemma_prompt(patient_id, level_data):
    """Build a clinical prompt for MedGemma from structured predictions."""
    prompt_parts = [
        "You are a board-certified radiologist. Based on the provided lumbar spine MRI image(s) "
        "and the following automated measurements, generate a structured radiology report.\n\n"
        "AUTOMATED MEASUREMENTS:\n"
    ]

    for level in ["D3", "D4", "D5"]:
        if level not in level_data:
            continue
        ld = level_data[level]
        level_name = LEVEL_MAP[level]
        grade = ld.get("grade", -1)
        height = ld.get("height", -1)

        prompt_parts.append(f"\n{level_name}:")
        if grade > 0:
            prompt_parts.append(f"  Pfirrmann Grade: {grade}/5 ({GRADE_DESC.get(grade, '')})")
        if height > 0:
            prompt_parts.append(f"  Disc Height: {height:.1f} pixels")

        disorders = ld.get("disorders", [])
        detected = [DISORDER_NAMES[i] for i, d in enumerate(disorders) if d > 0.5]
        if detected:
            prompt_parts.append(f"  Detected findings: {', '.join(detected)}")
        else:
            prompt_parts.append(f"  No significant findings detected.")

    prompt_parts.append(
        "\n\nPlease generate a complete structured radiology report with:\n"
        "1. TECHNIQUE section\n"
        "2. FINDINGS section (per disc level)\n"
        "3. IMPRESSION section (summary of key findings)\n"
        "Report should follow standard ACR reporting guidelines."
    )

    return "\n".join(prompt_parts)


def generate_medgemma_report(model, processor, patient_id, level_data, image_paths=None):
    """Generate a report for one patient using MedGemma."""
    from PIL import Image

    prompt = build_medgemma_prompt(patient_id, level_data)

    # Build message with image if available
    content = []
    images_loaded = []

    if image_paths:
        # Use up to 3 sagittal images (one per level) for visual context
        for img_path in image_paths[:3]:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    content.append({"type": "image", "image": img})
                    images_loaded.append(img)
                except Exception:
                    pass

    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        report = processor.decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        report = f"[MedGemma generation failed: {e}]"

    # Clean up images
    for img in images_loaded:
        img.close()

    return report


def generate_medgemma_text_only(model, processor, patient_id, level_data):
    """Text-only MedGemma generation (no images, pure NLG from measurements)."""
    prompt = build_medgemma_prompt(patient_id, level_data)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        report = processor.decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        report = f"[MedGemma generation failed: {e}]"

    return report


# ═══════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════

def load_patient_predictions(predictions_csv):
    """Load per-image predictions and aggregate per patient."""
    df = pd.read_csv(predictions_csv)
    patients = defaultdict(lambda: {"levels": {}, "images": []})

    for _, row in df.iterrows():
        pid = row["patient_id"]
        level = row["level"]
        plane = row.get("plane", "axial")
        fname = row["fname"]

        patients[pid]["images"].append(str(IMAGES_DIR / fname))

        if level not in ["D3", "D4", "D5"]:
            continue

        if level not in patients[pid]["levels"]:
            patients[pid]["levels"][level] = {
                "grade": -1, "height": -1, "disorders": [0] * 9,
            }

        ld = patients[pid]["levels"][level]

        if "pred_grade" in row and not pd.isna(row.get("gt_grade", -1)) and row.get("gt_grade", -1) > 0:
            ld["grade"] = int(row["pred_grade"])

        if "pred_height" in row and not pd.isna(row.get("gt_height", -1)) and row.get("gt_height", -1) > 0:
            ld["height"] = float(row["pred_height"])

        for i, dname in enumerate([
            "Herniation", "Protrusion", "Bulge", "ThecalComp",
            "NerveComp", "Stenosis", "Spondylolisthesis",
            "Desiccated", "LigFlavum"
        ]):
            prob_col = f"disorder_{dname}_prob"
            if prob_col in row and not pd.isna(row[prob_col]):
                ld["disorders"][i] = max(ld["disorders"][i], row[prob_col])

    return patients


# ═══════════════════════════════════════════════════════════════════
# Report Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

def evaluate_report_factual_consistency(report_text, gt_disorders, gt_grade_dict=None):
    """Evaluate factual consistency of a report against ground truth."""
    text_lower = report_text.lower()

    tp, fp, fn = 0, 0, 0
    per_disorder = {}

    for i, (dcol, dname) in enumerate(zip(DISORDER_COLS, DISORDER_NAMES)):
        gt_present = int(gt_disorders.get(dcol, 0)) == 1

        # Check if disorder is mentioned in report
        # Use multiple synonyms for robust matching
        synonyms = {
            "disc herniation": ["herniation", "herniated"],
            "disc protrusion": ["protrusion", "protruding"],
            "disc bulge": ["bulge", "bulging"],
            "thecal sac compression": ["thecal sac compression", "thecal sac"],
            "nerve root compression": ["nerve root compression", "nerve root", "neural compression"],
            "spinal canal stenosis": ["stenosis", "canal narrowing", "spinal canal stenosis"],
            "spondylolisthesis": ["spondylolisthesis", "listhesis"],
            "desiccated disc": ["desiccated", "desiccation", "disc desiccation"],
            "ligamentum flavum hypertrophy": ["ligamentum flavum", "flavum hypertrophy"],
        }

        gen_mentioned = any(
            syn.lower() in text_lower
            for syn in synonyms.get(dname, [dname])
        )

        if gt_present and gen_mentioned:
            tp += 1
            per_disorder[dname] = "TP"
        elif not gt_present and gen_mentioned:
            fp += 1
            per_disorder[dname] = "FP"
        elif gt_present and not gen_mentioned:
            fn += 1
            per_disorder[dname] = "FN"
        else:
            per_disorder[dname] = "TN"

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "per_disorder": per_disorder,
    }


def compute_report_metrics(reports_dict, gt_notes_df, method_name="RATF"):
    """Compute aggregate report quality metrics."""
    results = []

    for pid, report_text in reports_dict.items():
        patient_gt = gt_notes_df[gt_notes_df["Patient_ID"] == pid]
        if len(patient_gt) == 0:
            continue

        gt_row = patient_gt.iloc[0]
        gt_disorders = {col: gt_row.get(col, 0) for col in DISORDER_COLS}
        metrics = evaluate_report_factual_consistency(report_text, gt_disorders)
        metrics["patient_id"] = pid
        metrics["method"] = method_name
        results.append(metrics)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_generation(use_images=False, max_patients=0):
    """Generate reports using MedGemma for test patients."""
    predictions_csv = RESULTS_DIR / "test_A2_per_image.csv"
    if not predictions_csv.exists():
        print(f"Predictions CSV not found: {predictions_csv}")
        return {}

    patients = load_patient_predictions(str(predictions_csv))
    print(f"Loaded predictions for {len(patients)} test patients")

    # Load MedGemma
    model, processor = load_medgemma()

    output_dir = RESULTS_DIR / "medgemma_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    times = []

    patient_ids = sorted(patients.keys())
    # Skip already-generated patients
    already_done = {int(p.stem.split('_')[-1]) for p in output_dir.glob('medgemma_report_*.txt')
                    if p.stat().st_size > 200}  # skip failed stubs
    patient_ids = [p for p in patient_ids if p not in already_done]
    if max_patients > 0:
        patient_ids = patient_ids[:max_patients]
    print(f"  Generating for {len(patient_ids)} patients ({len(already_done)} already done)")

    for pid in patient_ids:
        level_data = patients[pid]["levels"]
        if not level_data:
            continue

        t0 = time.time()
        if use_images:
            # Gather sagittal images for visual context
            sag_images = [p for p in patients[pid]["images"] if "sag_" in p]
            report = generate_medgemma_report(model, processor, pid, level_data, sag_images)
        else:
            report = generate_medgemma_text_only(model, processor, pid, level_data)

        elapsed = time.time() - t0
        times.append(elapsed)
        reports[pid] = report

        # Save
        with open(output_dir / f"medgemma_report_{pid:04d}.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  Patient {pid:04d}: {elapsed:.1f}s ({len(report)} chars)")

    print(f"\nGenerated {len(reports)} MedGemma reports")
    print(f"Mean generation time: {np.mean(times):.2f}s per patient")
    print(f"Total time: {sum(times):.1f}s")

    # Save timing info
    timing = {
        "n_patients": len(reports),
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "total_time_s": float(sum(times)),
        "model": "MedGemma-4B-IT",
        "use_images": use_images,
    }
    with open(output_dir / "medgemma_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # Cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return reports


def run_evaluation():
    """Evaluate and compare RATF vs MedGemma reports."""
    gt_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")

    # Load RATF reports
    ratf_dir = RESULTS_DIR / "reports"
    ratf_reports = {}
    if ratf_dir.exists():
        for f in sorted(ratf_dir.glob("report_*.txt")):
            pid = int(f.stem.split("_")[1])
            ratf_reports[pid] = f.read_text(encoding="utf-8", errors="replace")

    # Load MedGemma reports
    mg_dir = RESULTS_DIR / "medgemma_reports"
    mg_reports = {}
    if mg_dir.exists():
        for f in sorted(mg_dir.glob("medgemma_report_*.txt")):
            pid = int(f.stem.split("_")[-1])
            mg_reports[pid] = f.read_text(encoding="utf-8", errors="replace")

    # Evaluate both
    results_all = []

    if ratf_reports:
        ratf_metrics = compute_report_metrics(ratf_reports, gt_notes, "RATF")
        results_all.append(ratf_metrics)
        print(f"\n{'='*60}")
        print(f"RATF Template Reports ({len(ratf_metrics)} patients)")
        print(f"{'='*60}")
        print(f"  Precision: {ratf_metrics['precision'].mean():.3f} ± {ratf_metrics['precision'].std():.3f}")
        print(f"  Recall:    {ratf_metrics['recall'].mean():.3f} ± {ratf_metrics['recall'].std():.3f}")
        print(f"  F1:        {ratf_metrics['f1'].mean():.3f} ± {ratf_metrics['f1'].std():.3f}")

    if mg_reports:
        mg_metrics = compute_report_metrics(mg_reports, gt_notes, "MedGemma")
        results_all.append(mg_metrics)
        print(f"\n{'='*60}")
        print(f"MedGemma Reports ({len(mg_metrics)} patients)")
        print(f"{'='*60}")
        print(f"  Precision: {mg_metrics['precision'].mean():.3f} ± {mg_metrics['precision'].std():.3f}")
        print(f"  Recall:    {mg_metrics['recall'].mean():.3f} ± {mg_metrics['recall'].std():.3f}")
        print(f"  F1:        {mg_metrics['f1'].mean():.3f} ± {mg_metrics['f1'].std():.3f}")

    if len(results_all) >= 2:
        combined = pd.concat(results_all, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "report_comparison_metrics.csv", index=False)

        # Summary comparison table
        summary = combined.groupby("method").agg(
            n=("patient_id", "count"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        ).round(3)

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string())

        summary.to_csv(RESULTS_DIR / "report_comparison_summary.csv")

        # Additional metrics: word count, report length
        method_stats = {}
        for method, reports in [("RATF", ratf_reports), ("MedGemma", mg_reports)]:
            lengths = [len(r) for r in reports.values()]
            word_counts = [len(r.split()) for r in reports.values()]
            method_stats[method] = {
                "mean_chars": float(np.mean(lengths)),
                "mean_words": float(np.mean(word_counts)),
                "std_words": float(np.std(word_counts)),
            }

        print("\nReport Statistics:")
        for method, stats in method_stats.items():
            print(f"  {method}: {stats['mean_words']:.0f} ± {stats['std_words']:.0f} words, "
                  f"{stats['mean_chars']:.0f} chars")

        with open(RESULTS_DIR / "report_comparison_stats.json", "w") as f:
            json.dump(method_stats, f, indent=2)

    elif len(results_all) == 1:
        results_all[0].to_csv(RESULTS_DIR / "report_comparison_metrics.csv", index=False)
        print("Only one method available for comparison. Generate MedGemma reports first.")
    else:
        print("No reports found. Generate reports first with --mode generate.")


def main():
    parser = argparse.ArgumentParser(description="MedGemma vs RATF Report Comparison")
    parser.add_argument("--mode", choices=["generate", "evaluate", "all"],
                        default="all", help="generate, evaluate, or all")
    parser.add_argument("--use-images", action="store_true",
                        help="Pass MRI images to MedGemma (multimodal mode)")
    parser.add_argument("--max-patients", type=int, default=0,
                        help="Max patients for MedGemma generation (0=all)")
    args = parser.parse_args()

    if args.mode in ["generate", "all"]:
        run_generation(use_images=args.use_images, max_patients=args.max_patients)

    if args.mode in ["evaluate", "all"]:
        run_evaluation()


if __name__ == "__main__":
    main()
