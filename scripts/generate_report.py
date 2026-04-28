"""
report_generator.py — Module 4: Findings-Grounded Report Generation (FGRG)
Section 6 (FAISS) + Section 10 (report generation).

Two modes:
  (A) Template-based: deterministic, auditable, no extra VRAM
  (B) LLM-augmented (optional): Phi-3-mini LoRA or FAISS RAG

Usage:
    python scripts/report_generator.py --mode template --predictions results/test_per_image.csv
    python scripts/report_generator.py --mode rag --build-index
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
with open(SCRIPT_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE-BASED REPORT GENERATOR (Mode A)
# ═══════════════════════════════════════════════════════════════════

LEVEL_MAP = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
GRADE_DESC = {
    1: "homogeneous, bright white",
    2: "inhomogeneous with or without horizontal bands, bright white",
    3: "inhomogeneous, intermediate gray",
    4: "inhomogeneous, dark gray to black",
    5: "inhomogeneous, black with no disc space distinction",
}
GRADE_CLINICAL = {
    1: "Normal disc signal",
    2: "Mild disc degeneration",
    3: "Moderate disc degeneration",
    4: "Advanced disc degeneration",
    5: "Severe disc degeneration with disc space collapse",
}
HEIGHT_THRESHOLDS = {"normal": 8.0, "mild_loss": 5.0, "severe_loss": 3.0}  # pixels


def classify_height(ht_px):
    """Classify IVD height into clinical categories."""
    if ht_px < 0:
        return "not measured"
    if ht_px >= HEIGHT_THRESHOLDS["normal"]:
        return "normal disc height"
    elif ht_px >= HEIGHT_THRESHOLDS["mild_loss"]:
        return "mild disc height loss"
    elif ht_px >= HEIGHT_THRESHOLDS["severe_loss"]:
        return "moderate disc height loss"
    else:
        return "severe disc height loss"


DISORDER_NAMES = [
    "disc herniation", "disc protrusion", "disc bulge",
    "thecal sac compression", "nerve root compression",
    "spinal canal stenosis", "spondylolisthesis",
    "desiccated disc", "ligamentum flavum hypertrophy",
]


def generate_patient_report(patient_data):
    """
    Generate a structured radiological report for one patient.

    patient_data: dict with keys per level (D3/D4/D5):
        {
            "D3": {"grade": 2, "height": 9.5, "disorders": [0,0,1,...],
                   "seg_dice": 0.92, "seg_quality": "good"},
            ...
        }
    """
    sections = []

    # Header
    pid = patient_data.get("patient_id", "?")
    sections.append(f"LUMBAR SPINE MRI REPORT")
    sections.append(f"Patient ID: {pid}")
    sections.append(f"Date: {patient_data.get('date', 'N/A')}")
    sections.append("")

    # TECHNIQUE
    sections.append("TECHNIQUE:")
    sections.append("  MRI of the lumbar spine was performed in axial and sagittal planes")
    sections.append("  using T1 and T2 weighted sequences.")
    sections.append("")

    # FINDINGS per level
    sections.append("FINDINGS:")
    for level in ["D3", "D4", "D5"]:
        if level not in patient_data:
            continue
        ld = patient_data[level]
        level_name = LEVEL_MAP[level]
        sections.append(f"\n  {level_name} Level:")

        # Grade
        grade = ld.get("grade", -1)
        if grade > 0:
            sections.append(f"    Pfirrmann Grade: {grade}/5 — {GRADE_DESC.get(grade, 'unknown')}")
            sections.append(f"    Clinical: {GRADE_CLINICAL.get(grade, 'unknown')}")

        # Height
        ht = ld.get("height", -1)
        if ht > 0:
            ht_class = classify_height(ht)
            sections.append(f"    Disc Height: {ht:.1f} px ({ht_class})")

        # Disorders
        disorders = ld.get("disorders", [])
        if any(d > 0.5 for d in disorders):
            present = [DISORDER_NAMES[i] for i, d in enumerate(disorders) if d > 0.5]
            sections.append(f"    Findings: {', '.join(present)}")
        else:
            sections.append(f"    Findings: No significant abnormality detected.")

        # Segmentation quality indicator
        seg_dice = ld.get("seg_dice", -1)
        if seg_dice > 0:
            qual = "excellent" if seg_dice > 0.9 else "good" if seg_dice > 0.8 else "moderate" if seg_dice > 0.7 else "limited"
            sections.append(f"    Segmentation confidence: {qual} (Dice = {seg_dice:.3f})")

    sections.append("")

    # IMPRESSION
    sections.append("IMPRESSION:")
    all_findings = []
    for level in ["D3", "D4", "D5"]:
        if level not in patient_data:
            continue
        ld = patient_data[level]
        level_name = LEVEL_MAP[level]
        grade = ld.get("grade", -1)

        if grade >= 4:
            all_findings.append(f"Advanced disc degeneration at {level_name}")
        elif grade == 3:
            all_findings.append(f"Moderate disc degeneration at {level_name}")

        disorders = ld.get("disorders", [])
        present = [DISORDER_NAMES[i] for i, d in enumerate(disorders) if d > 0.5]
        for p in present:
            all_findings.append(f"{p.capitalize()} at {level_name}")

    if all_findings:
        for i, f_text in enumerate(all_findings, 1):
            sections.append(f"  {i}. {f_text}")
    else:
        sections.append("  No significant abnormality detected in the lumbar spine.")

    sections.append("")

    return "\n".join(sections)


def reports_from_predictions(predictions_csv, output_dir):
    """Generate reports for all patients from per-image predictions."""
    df = pd.read_csv(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by patient
    patients = defaultdict(dict)
    for _, row in df.iterrows():
        pid = row["patient_id"]
        level = row["level"]

        if level not in ["D3", "D4", "D5"]:
            continue

        if level not in patients[pid]:
            patients[pid][level] = {
                "grade": -1, "height": -1, "disorders": [0]*9, "seg_dices": [],
            }

        ld = patients[pid][level]

        # Use grade if available
        if "pred_grade" in row and row.get("gt_grade", -1) != -1:
            ld["grade"] = int(row["pred_grade"])

        # Height
        if "pred_height" in row and row.get("gt_height", -1) != -1:
            ld["height"] = float(row["pred_height"])

        # Disorders — take max probability across images for same patient
        for i, dname in enumerate([
            "Herniation", "Protrusion", "Bulge", "ThecalComp",
            "NerveComp", "Stenosis", "Spondylolisthesis",
            "Desiccated", "LigFlavum"
        ]):
            prob_col = f"disorder_{dname}_prob"
            if prob_col in row and not pd.isna(row[prob_col]):
                ld["disorders"][i] = max(ld["disorders"][i], row[prob_col])

        # Seg quality
        if "prompted_dice" in row:
            ld["seg_dices"].append(row["prompted_dice"])

    # Generate reports
    reports = {}
    for pid, level_data in sorted(patients.items()):
        patient_data = {"patient_id": pid}
        for level, ld in level_data.items():
            ld["seg_dice"] = np.mean(ld["seg_dices"]) if ld["seg_dices"] else -1
            del ld["seg_dices"]
            patient_data[level] = ld

        report = generate_patient_report(patient_data)
        reports[pid] = report

        # Save individual report
        with open(output_dir / f"report_{pid:04d}.txt", "w") as f:
            f.write(report)

    print(f"Generated {len(reports)} patient reports in {output_dir}")
    return reports


# ═══════════════════════════════════════════════════════════════════
# FAISS RAG INDEX (Mode B — Section 6)
# ═══════════════════════════════════════════════════════════════════

def build_faiss_index():
    """Build FAISS index from structured_reports.csv for RAG retrieval."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install faiss-cpu and sentence-transformers:")
        print("  pip install faiss-cpu sentence-transformers")
        return

    root = Path(CFG["paths"]["dataset_root"])
    df = pd.read_csv(root / "structured_reports.csv")

    # Build report texts
    texts = []
    metadata = []
    for _, row in df.iterrows():
        # Combine available columns into a report snippet
        cols = [c for c in df.columns if c != "Patient_ID"]
        text_parts = []
        for c in cols:
            if pd.notna(row[c]) and str(row[c]).strip():
                text_parts.append(f"{c}: {row[c]}")
        text = " | ".join(text_parts)
        texts.append(text)
        metadata.append({"patient_id": int(row.get("Patient_ID", 0)), "row_idx": _})

    print(f"Encoding {len(texts)} report snippets...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save
    faiss_path = Path(CFG["paths"]["faiss_index"])
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))

    meta_path = Path(CFG["paths"]["faiss_meta"])
    with open(meta_path, "w") as f:
        json.dump({"texts": texts, "metadata": metadata}, f)

    print(f"FAISS index saved: {faiss_path} ({len(texts)} vectors, dim={dim})")
    return index


def rag_retrieve(query_text, top_k=3):
    """Retrieve similar reports from FAISS index."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []

    faiss_path = Path(CFG["paths"]["faiss_index"])
    meta_path = Path(CFG["paths"]["faiss_meta"])
    if not faiss_path.exists():
        print("FAISS index not found. Run --build-index first.")
        return []

    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = encoder.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        results.append({
            "rank": i + 1,
            "score": float(score),
            "text": meta["texts"][idx],
            "patient_id": meta["metadata"][idx]["patient_id"],
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# REPORT QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════

def evaluate_reports(generated_dir, ground_truth_csv):
    """Evaluate generated reports against structured ground truth."""
    root = Path(CFG["paths"]["dataset_root"])
    gt_notes = pd.read_csv(root / "structured_notes.csv")
    gt_reports = pd.read_csv(root / "structured_reports.csv")

    gen_dir = Path(generated_dir)
    results = []

    disorder_cols = ["Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
                     "Thecal_Sac_Compression", "Nerve_Root_Compression",
                     "Spinal_Canal_Stenosis", "Spondylolisthesis",
                     "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"]

    for report_file in sorted(gen_dir.glob("report_*.txt")):
        pid = int(report_file.stem.split("_")[1])
        with open(report_file) as f:
            gen_text = f.read().lower()

        # Check factual consistency: disorders mentioned vs ground truth
        patient_notes = gt_notes[gt_notes["Patient_ID"] == pid]
        if len(patient_notes) == 0:
            continue

        gt_row = patient_notes.iloc[0]
        tp, fp, fn = 0, 0, 0
        for dcol, dname in zip(disorder_cols, DISORDER_NAMES):
            gt_present = int(gt_row.get(dcol, 0)) == 1
            gen_mentioned = dname.lower() in gen_text

            if gt_present and gen_mentioned:
                tp += 1
            elif not gt_present and gen_mentioned:
                fp += 1
            elif gt_present and not gen_mentioned:
                fn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results.append({
            "patient_id": pid,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"\nReport Factual Consistency ({len(df)} patients):")
        print(f"  Precision: {df['precision'].mean():.3f} +/- {df['precision'].std():.3f}")
        print(f"  Recall:    {df['recall'].mean():.3f} +/- {df['recall'].std():.3f}")
        print(f"  F1:        {df['f1'].mean():.3f} +/- {df['f1'].std():.3f}")

    return df


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "rag"], default="template")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to per-image predictions CSV")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--evaluate-reports", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(CFG["paths"]["results"])

    if args.build_index:
        build_faiss_index()
        return

    if args.evaluate_reports:
        gen_dir = args.output_dir or str(results_dir / "generated_reports")
        evaluate_reports(gen_dir, None)
        return

    if args.predictions:
        output_dir = args.output_dir or str(results_dir / "generated_reports")
        reports_from_predictions(args.predictions, output_dir)
    else:
        # Demo: generate a sample report
        sample = {
            "patient_id": 1,
            "D3": {"grade": 2, "height": 9.5, "disorders": [0,0,0,0,0,0,0,0,0], "seg_dice": 0.95},
            "D4": {"grade": 3, "height": 7.2, "disorders": [0,1,1,0,0,0,0,0,0], "seg_dice": 0.88},
            "D5": {"grade": 4, "height": 4.1, "disorders": [1,0,0,1,1,1,0,1,0], "seg_dice": 0.82},
        }
        report = generate_patient_report(sample)
        print(report)


if __name__ == "__main__":
    main()
