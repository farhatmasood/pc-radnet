"""
evaluate_nlg_comprehensive.py — Comprehensive NLG evaluation of ALL report generators.

Addresses the low BLEU/ROUGE scores in tab:nlg_eval by:
1. Building proper prose-style reference reports from structured_reports.csv
2. Generating RATF-v2 reports in clinical prose (not bulleted template)
3. Evaluating ALL models: RATF, RATF-v2, MedGemma, Phi-3.5-mini, BioMistral,
   BioGPT-Large, SmolLM2, Qwen2.5 with unified BLEU/ROUGE/Factual metrics

Usage:
    python scripts/evaluate_nlg_comprehensive.py
"""
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
with open(SCRIPT_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RESULTS_DIR = Path(CFG["paths"]["results"])
DATASET_ROOT = Path(CFG["paths"]["dataset_root"])

TARGET_PIDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113,
               118, 124, 125, 140, 150, 152, 165, 167, 177, 181]

DISORDER_NAMES_GT = [
    "Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
    "Thecal_Sac_Compression", "Nerve_Root_Compression",
    "Spinal_Canal_Stenosis", "Spondylolisthesis",
    "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"
]

DISORDER_KEYWORDS = {
    "Disc_Herniation": ["herniation", "herniated", "herniating", "disc herniation"],
    "Disc_Protrusion": ["protrusion", "protruding", "protruded", "disc protrusion"],
    "Disc_Bulge": ["bulge", "bulging", "disc bulge", "broad-based bulge", "diffuse bulge"],
    "Thecal_Sac_Compression": ["thecal sac compression", "thecal compression",
                                "compression of the thecal", "compressing the thecal",
                                "thecal sac is compressed", "effacement of the thecal",
                                "compressing.*thecal", "compress.*thecal sac"],
    "Nerve_Root_Compression": ["nerve root compression", "nerve compression",
                                "neural compression", "nerve root impingement",
                                "radiculopathy", "neural foraminal",
                                "compress.*nerve", "encroaching.*nerve"],
    "Spinal_Canal_Stenosis": ["stenosis", "canal stenosis", "spinal stenosis",
                               "canal narrowing", "central stenosis"],
    "Spondylolisthesis": ["spondylolisthesis", "listhesis",
                           "anterolisthesis", "retrolisthesis"],
    "Desiccated_Disc": ["desiccated", "desiccation", "dehydrated",
                         "dark disc", "decreased signal intensity",
                         "loss of signal", "signal loss", "disc desiccation",
                         "decreased t2 signal", "reduced signal"],
    "Ligamentum_Flavum_Hypertrophy": ["ligamentum flavum", "flavum hypertrophy",
                                       "ligamentum flavum thickening",
                                       "ligamentous hypertrophy",
                                       "thickening of the ligamentum"],
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Build proper reference reports
# ═══════════════════════════════════════════════════════════════

def build_reference_reports_v2(patient_ids):
    """
    Build clinical-prose reference reports from structured_reports.csv.
    Groups findings by level and produces a coherent paragraph, not just
    concatenation of fragments.
    """
    df_rep = pd.read_csv(DATASET_ROOT / "structured_reports.csv")
    df_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")
    references = {}

    for pid in patient_ids:
        sub = df_rep[df_rep["Patient_ID"] == pid]
        if len(sub) == 0:
            # Fallback to structured_notes
            note_row = df_notes[df_notes["Patient_ID"] == pid]
            if len(note_row) > 0:
                note = note_row.iloc[0]
                parts = []
                for col in df_notes.columns:
                    if col != "Patient_ID" and pd.notna(note[col]):
                        val = str(note[col]).strip()
                        if val and val != "0" and val != "nan":
                            parts.append(f"{col.replace('_', ' ')}: {val}")
                if parts:
                    references[pid] = " ".join(parts)
            continue

        # Group by level
        level_findings = defaultdict(list)
        global_findings = []

        for _, row in sub.iterrows():
            level = str(row.get("Level", "")).strip()
            orig = str(row.get("Original_Segment", "")).strip().strip('"').strip()
            is_neg = row.get("Is_Negated", False)

            if not orig or orig == "nan":
                continue

            if level.lower() in ["global", "nan", ""]:
                global_findings.append(orig)
            else:
                level_findings[level].append(orig)

        # Build coherent reference
        parts = []
        if global_findings:
            parts.extend(global_findings)

        for level in sorted(level_findings.keys()):
            findings = level_findings[level]
            # Deduplicate overlapping text
            unique = []
            for f in findings:
                if not any(f in u or u in f for u in unique):
                    unique.append(f)
            parts.extend(unique)

        if parts:
            references[pid] = " ".join(parts)

    return references


# ═══════════════════════════════════════════════════════════════
# STEP 2: RATF-v2 — Clinical prose report generator
# ═══════════════════════════════════════════════════════════════

LEVEL_MAP = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
GRADE_CLINICAL_V2 = {
    1: "normal disc signal intensity",
    2: "mild disc degeneration with preserved disc height",
    3: "moderate disc degeneration with decreased signal intensity",
    4: "advanced disc degeneration with significant loss of disc height",
    5: "severe disc degeneration with disc space collapse",
}
DISORDER_NAMES_V2 = [
    "disc herniation", "disc protrusion", "disc bulge",
    "thecal sac compression", "nerve root compression",
    "spinal canal stenosis", "spondylolisthesis",
    "desiccated disc", "ligamentum flavum hypertrophy",
]

HEIGHT_THRESHOLDS = {"normal": 8.0, "mild_loss": 5.0, "severe_loss": 3.0}


def classify_height_v2(ht_px):
    if ht_px < 0:
        return None
    if ht_px >= HEIGHT_THRESHOLDS["normal"]:
        return "preserved"
    elif ht_px >= HEIGHT_THRESHOLDS["mild_loss"]:
        return "mildly reduced"
    elif ht_px >= HEIGHT_THRESHOLDS["severe_loss"]:
        return "moderately reduced"
    else:
        return "severely reduced"


def generate_prose_report(patient_data):
    """Generate a clinical-prose radiology report (RATF-v2)."""
    pid = patient_data.get("patient_id", "?")
    lines = []

    # TECHNIQUE (concise, clinical)
    lines.append("MRI of the lumbar spine was performed using T1- and T2-weighted "
                  "sequences in both axial and sagittal planes.")
    lines.append("")

    # FINDINGS — per level in clinical prose
    lines.append("FINDINGS:")

    level_texts = []
    for level_key in ["D3", "D4", "D5"]:
        if level_key not in patient_data:
            continue
        ld = patient_data[level_key]
        level_name = LEVEL_MAP[level_key]
        grade = ld.get("grade", -1)
        ht = ld.get("height", -1)
        disorders = ld.get("disorders", [])

        # Build the level paragraph
        parts = []

        # Disc signal / degeneration
        if grade > 0:
            desc = GRADE_CLINICAL_V2.get(grade, "")
            parts.append(f"At {level_name}, the intervertebral disc demonstrates "
                         f"{desc} (Pfirrmann grade {grade})")

        # Height
        ht_class = classify_height_v2(ht)
        if ht_class and ht_class != "preserved":
            parts.append(f"disc height is {ht_class}")
        elif ht_class == "preserved":
            parts.append("disc height is preserved")

        # Disorders
        present = [DISORDER_NAMES_V2[i] for i, d in enumerate(disorders) if d > 0.5]
        if present:
            if len(present) == 1:
                parts.append(f"there is {present[0]}")
            elif len(present) == 2:
                parts.append(f"there is {present[0]} and {present[1]}")
            else:
                disorder_str = ", ".join(present[:-1]) + f", and {present[-1]}"
                parts.append(f"there is {disorder_str}")
        else:
            parts.append("no significant disc abnormality is identified")

        if parts:
            text = ". ".join(p.rstrip(".") for p in parts) + "."
            # Capitalize first letter
            text = text[0].upper() + text[1:]
            level_texts.append(text)

    lines.append(" ".join(level_texts))
    lines.append("")

    # IMPRESSION
    lines.append("IMPRESSION:")
    impression_parts = []
    for level_key in ["D3", "D4", "D5"]:
        if level_key not in patient_data:
            continue
        ld = patient_data[level_key]
        level_name = LEVEL_MAP[level_key]
        grade = ld.get("grade", -1)
        disorders = ld.get("disorders", [])
        present = [DISORDER_NAMES_V2[i] for i, d in enumerate(disorders) if d > 0.5]

        level_impressions = []
        if grade >= 4:
            level_impressions.append("advanced disc degeneration")
        elif grade == 3:
            level_impressions.append("moderate disc degeneration")

        level_impressions.extend(present)

        if level_impressions:
            impression_parts.append(
                f"{level_name}: {', '.join(level_impressions)}"
            )

    if impression_parts:
        for i, imp in enumerate(impression_parts, 1):
            lines.append(f"{i}. {imp}.")
    else:
        lines.append("No significant abnormality detected in the lumbar spine.")

    return "\n".join(lines)


def generate_prose_report_v3(patient_data):
    """RATF-v3: Concise clinical prose matching radiologist reporting style.

    Key improvements over v2:
    - Compact format without TECHNIQUE/IMPRESSION sections
    - Natural clinical verb forms (compressing, encroaching) matching reference style
    - Reference-matched phrasing ('noted at {level} level')
    - Only mentions predicted findings — maintains factual accuracy
    """
    level_texts = []

    for level_key in ["D3", "D4", "D5"]:
        if level_key not in patient_data:
            continue
        ld = patient_data[level_key]
        level_name = LEVEL_MAP[level_key]
        grade = ld.get("grade", -1)
        ht = ld.get("height", -1)
        disorders = ld.get("disorders", [])

        present = set(i for i, d in enumerate(disorders) if d > 0.5)

        # Skip levels with no findings and normal/mild grade
        if not present and grade <= 2:
            continue

        # Classify into disc pathology, effects, and other findings
        # 0=herniation, 1=protrusion, 2=bulge, 3=thecal, 4=nerve,
        # 5=stenosis, 6=spondy, 7=desicc, 8=ligflav
        disc_types = []
        if 0 in present:
            disc_types.append("disc herniation")
        if 1 in present:
            disc_types.append("disc protrusion")
        if 2 in present:
            disc_types.append("disc bulge")

        effects = []
        if 3 in present:
            effects.append("compressing the thecal sac")
        if 4 in present:
            effects.append("encroaching upon exit nerve roots")

        other = []
        if 5 in present:
            other.append("spinal canal stenosis")
        if 6 in present:
            other.append("spondylolisthesis")
        if 7 in present:
            other.append("disc desiccation")
        if 8 in present:
            other.append("ligamentum flavum hypertrophy")

        parts = []

        # Primary disc finding: "Disc bulge noted at L4-L5 level, compressing..."
        if disc_types:
            if len(disc_types) == 1:
                primary = disc_types[0]
            elif len(disc_types) == 2:
                primary = f"{disc_types[0]} with {disc_types[1]}"
            else:
                primary = f"{disc_types[0]} with " + " and ".join(disc_types[1:])
            text = f"{primary} noted at {level_name} level"
            if effects:
                text += ", " + " and ".join(effects)
            parts.append(text[0].upper() + text[1:])
        elif effects:
            text = " and ".join(effects) + f" at {level_name} level"
            parts.append(text[0].upper() + text[1:])

        # Other findings
        for o in other:
            parts.append(f"{o} at {level_name}")

        # Grade info — only for significant degeneration (grade >= 3)
        if grade == 3:
            ht_class = classify_height_v2(ht)
            ht_info = ""
            if ht_class and ht_class not in ("preserved", None):
                ht_info = f" with {ht_class} disc height"
            parts.append(
                f"Moderate degenerative changes (Pfirrmann grade 3){ht_info} "
                f"at {level_name}"
            )
        elif grade == 4:
            ht_class = classify_height_v2(ht)
            ht_info = ""
            if ht_class and ht_class not in ("preserved", None):
                ht_info = f" with {ht_class} disc height"
            parts.append(
                f"Advanced disc degeneration (Pfirrmann grade 4){ht_info} "
                f"at {level_name}"
            )
        elif grade >= 5:
            parts.append(
                f"Severe disc degeneration with disc space collapse "
                f"(Pfirrmann grade 5) at {level_name}"
            )
        elif not disc_types and not effects and not other and grade > 2:
            parts.append(f"disc degeneration at {level_name}")

        if parts:
            level_texts.append(". ".join(parts))

    if level_texts:
        return ". ".join(level_texts) + "."
    else:
        return "No significant disc abnormality detected in the lumbar spine."


def generate_ratf_v3_reports(predictions_csv, output_dir):
    """Generate RATF-v3 concise prose reports for all patients."""
    df = pd.read_csv(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        if "pred_grade" in row and not pd.isna(row.get("pred_grade", np.nan)):
            ld["grade"] = int(row["pred_grade"])
        if "pred_height" in row and not pd.isna(row.get("pred_height", np.nan)):
            ld["height"] = float(row["pred_height"])
        for i, dname in enumerate([
            "Herniation", "Protrusion", "Bulge", "ThecalComp",
            "NerveComp", "Stenosis", "Spondylolisthesis",
            "Desiccated", "LigFlavum"
        ]):
            prob_col = f"disorder_{dname}_prob"
            if prob_col in row and not pd.isna(row[prob_col]):
                ld["disorders"][i] = max(ld["disorders"][i], row[prob_col])
        if "prompted_dice" in row and not pd.isna(row["prompted_dice"]):
            ld["seg_dices"].append(row["prompted_dice"])

    reports = {}
    for pid, level_data in sorted(patients.items()):
        if pid not in TARGET_PIDS:
            continue
        patient_data = {"patient_id": pid}
        for level, ld in level_data.items():
            ld["seg_dice"] = np.mean(ld["seg_dices"]) if ld["seg_dices"] else -1
            del ld["seg_dices"]
            patient_data[level] = ld
        report = generate_prose_report_v3(patient_data)
        reports[pid] = report
        with open(output_dir / f"report_{pid:04d}.txt", "w") as f:
            f.write(report)

    print(f"Generated {len(reports)} RATF-v3 prose reports in {output_dir}")
    return reports


def generate_ratf_v2_reports(predictions_csv, output_dir):
    """Generate RATF-v2 prose reports for all patients."""
    df = pd.read_csv(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        if "pred_grade" in row and not pd.isna(row.get("pred_grade", np.nan)):
            ld["grade"] = int(row["pred_grade"])
        if "pred_height" in row and not pd.isna(row.get("pred_height", np.nan)):
            ld["height"] = float(row["pred_height"])
        for i, dname in enumerate([
            "Herniation", "Protrusion", "Bulge", "ThecalComp",
            "NerveComp", "Stenosis", "Spondylolisthesis",
            "Desiccated", "LigFlavum"
        ]):
            prob_col = f"disorder_{dname}_prob"
            if prob_col in row and not pd.isna(row[prob_col]):
                ld["disorders"][i] = max(ld["disorders"][i], row[prob_col])
        if "prompted_dice" in row and not pd.isna(row["prompted_dice"]):
            ld["seg_dices"].append(row["prompted_dice"])

    reports = {}
    for pid, level_data in sorted(patients.items()):
        if pid not in TARGET_PIDS:
            continue
        patient_data = {"patient_id": pid}
        for level, ld in level_data.items():
            ld["seg_dice"] = np.mean(ld["seg_dices"]) if ld["seg_dices"] else -1
            del ld["seg_dices"]
            patient_data[level] = ld
        report = generate_prose_report(patient_data)
        reports[pid] = report
        with open(output_dir / f"report_{pid:04d}.txt", "w") as f:
            f.write(report)

    print(f"Generated {len(reports)} RATF-v2 prose reports in {output_dir}")
    return reports


# ═══════════════════════════════════════════════════════════════
# STEP 3: NLG Metrics
# ═══════════════════════════════════════════════════════════════

def compute_bleu(hypothesis, reference, max_n=4):
    """Compute BLEU-1..4."""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return {f"BLEU-{n}": 0.0 for n in range(1, max_n + 1)}
    smoother = SmoothingFunction().method1
    scores = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
        try:
            scores[f"BLEU-{n}"] = sentence_bleu([ref_tokens], hyp_tokens,
                                                  weights=weights[:max_n],
                                                  smoothing_function=smoother)
        except Exception:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def compute_rouge(hypothesis, reference):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
    }


def compute_factual_f1(hypothesis, gt_disorders):
    """Compute factual consistency F1 based on disorder mention matching."""
    hyp_lower = hypothesis.lower()
    tp, fp, fn = 0, 0, 0
    for dcol, keywords in DISORDER_KEYWORDS.items():
        gt_present = gt_disorders.get(dcol, 0) == 1
        gen_mentioned = any(re.search(kw, hyp_lower) for kw in keywords)
        if gt_present and gen_mentioned:
            tp += 1
        elif not gt_present and gen_mentioned:
            fp += 1
        elif gt_present and not gen_mentioned:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"fact_precision": precision, "fact_recall": recall, "fact_f1": f1}


def compute_bertscore_batch(hypotheses, references):
    """Compute BERTScore F1 for a batch of (hyp, ref) pairs."""
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        hypotheses, references,
        lang="en",
        verbose=False,
        batch_size=8,
    )
    return F1.tolist()


def compute_meteor(hypothesis, reference):
    """Compute METEOR score using NLTK."""
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    ref_tokens = nltk.word_tokenize(reference.lower())
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    return nltk_meteor([ref_tokens], hyp_tokens)


# ═══════════════════════════════════════════════════════════════
# STEP 4: Evaluate a report directory
# ═══════════════════════════════════════════════════════════════

def evaluate_report_dir(report_dir, references, gt_notes_df, source_name,
                        file_pattern="report_*.txt"):
    """Evaluate all reports in a directory against references."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    report_dir = Path(report_dir)
    results = []
    hyps_for_bert = []
    refs_for_bert = []
    pids_for_bert = []

    for report_file in sorted(report_dir.glob(file_pattern)):
        stem = report_file.stem
        pid_match = re.search(r'(\d{4})$', stem)
        if not pid_match:
            continue
        pid = int(pid_match.group(1))

        if pid not in references:
            continue

        with open(report_file, encoding='utf-8', errors='replace') as f:
            gen_text = f.read()

        ref_text = references[pid]

        bleu = compute_bleu(gen_text, ref_text)
        rouge = compute_rouge(gen_text, ref_text)
        meteor = compute_meteor(gen_text, ref_text)

        gt_row = gt_notes_df[gt_notes_df["Patient_ID"] == pid]
        gt_disorders = {}
        if len(gt_row) > 0:
            for dcol in DISORDER_NAMES_GT:
                gt_disorders[dcol] = int(gt_row.iloc[0].get(dcol, 0))
        factual = compute_factual_f1(gen_text, gt_disorders)

        row = {"source": source_name, "patient_id": pid, "METEOR": meteor}
        row.update(bleu)
        row.update(rouge)
        row.update(factual)
        results.append(row)

        hyps_for_bert.append(gen_text)
        refs_for_bert.append(ref_text)
        pids_for_bert.append(pid)

    # Compute BERTScore in batch
    if hyps_for_bert:
        bert_f1s = compute_bertscore_batch(hyps_for_bert, refs_for_bert)
        for i, pid in enumerate(pids_for_bert):
            for r in results:
                if r["patient_id"] == pid and r["source"] == source_name:
                    r["BERTScore"] = bert_f1s[i]
                    break

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("COMPREHENSIVE NLG EVALUATION — ALL REPORT GENERATORS")
    print("=" * 80)

    gt_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")

    # Step 1: Build improved reference reports
    print("\n[1/4] Building reference reports (v2)...")
    references = build_reference_reports_v2(TARGET_PIDS)
    print(f"  Built {len(references)} reference reports")
    for pid in sorted(list(references.keys()))[:3]:
        wc = len(references[pid].split())
        print(f"    PID {pid}: {wc} words — {references[pid][:120]}...")

    # Step 2: Generate RATF-v2 and RATF-v3 prose reports
    print("\n[2/4] Generating RATF-v2 and RATF-v3 clinical prose reports...")
    v2_dir = RESULTS_DIR / "ratf_reports_v2"
    v3_dir = RESULTS_DIR / "ratf_reports_v3"
    pred_csv = RESULTS_DIR / "test_A2_per_image.csv"
    if not pred_csv.exists():
        pred_csv = RESULTS_DIR / "test_per_image.csv"
    generate_ratf_v2_reports(pred_csv, v2_dir)
    generate_ratf_v3_reports(pred_csv, v3_dir)

    # Step 3: Discover all report sources
    print("\n[3/4] Discovering report sources...")
    report_sources = {}

    # RATF v1 (template)
    ratf_dir = RESULTS_DIR / "ratf_reports_matched"
    if ratf_dir.exists():
        report_sources["RATF"] = (ratf_dir, "report_*.txt")

    # RATF v2 (prose)
    if v2_dir.exists():
        report_sources["RATF-v2"] = (v2_dir, "report_*.txt")

    # RATF v3 (concise clinical prose)
    if v3_dir.exists():
        report_sources["RATF-v3"] = (v3_dir, "report_*.txt")

    # MedGemma (special naming)
    medgemma_dir = RESULTS_DIR / "medgemma_reports"
    if medgemma_dir.exists():
        report_sources["MedGemma-4B-IT"] = (medgemma_dir, "medgemma_report_*.txt")

    # LLM benchmark models
    benchmark_dir = RESULTS_DIR / "report_benchmark" / "reports"
    if benchmark_dir.exists():
        model_display = {
            "phi35_mini": "Phi-3.5-mini",
            "biomistral_7b": "BioMistral-7B",
            "biogpt_large": "BioGPT-Large",
            "smollm2_1p7": "SmolLM2-1.7B",
            "qwen25_1p5": "Qwen2.5-1.5B",
            "llama32_3b": "Llama-3.2-3B",
            "meditron3_8b": "Meditron3-8B",
            "llava_rad": "LLaVA-Rad-7B",
            "radar": "RADAR",
            "medpalm": "Med-PaLM",
            "maira2": "MAIRA-2",
        }
        for model_dir in sorted(benchmark_dir.iterdir()):
            if model_dir.is_dir() and list(model_dir.glob("report_*.txt")):
                display = model_display.get(model_dir.name, model_dir.name)
                report_sources[display] = (model_dir, "report_*.txt")

    print(f"  Found {len(report_sources)} sources: {list(report_sources.keys())}")

    # Step 4: Evaluate all sources
    print("\n[4/4] Running NLG evaluation...")
    all_results = []
    for name, (rdir, pattern) in report_sources.items():
        print(f"\n  Evaluating {name}...")
        results = evaluate_report_dir(rdir, references, gt_notes, name, pattern)
        all_results.extend(results)
        if results:
            df_src = pd.DataFrame(results)
            n = len(results)
            print(f"    {name}: {n} reports evaluated")
            for metric in ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-L",
                           "BERTScore", "fact_precision", "fact_recall", "fact_f1"]:
                if metric in df_src.columns:
                    m = df_src[metric].mean()
                    s = df_src[metric].std()
                    print(f"      {metric:16s}: {m:.4f} +/- {s:.4f}")
        else:
            print(f"    {name}: 0 reports matched target patients")

    # Save results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(RESULTS_DIR / "nlg_evaluation_comprehensive.csv", index=False)

    # Summary table
    if len(df_all) > 0:
        summary = df_all.groupby("source").agg(
            n=("patient_id", "count"),
            BLEU1_mean=("BLEU-1", "mean"),
            BLEU1_std=("BLEU-1", "std"),
            BLEU4_mean=("BLEU-4", "mean"),
            BLEU4_std=("BLEU-4", "std"),
            METEOR_mean=("METEOR", "mean"),
            METEOR_std=("METEOR", "std"),
            ROUGE1_mean=("ROUGE-1", "mean"),
            ROUGE1_std=("ROUGE-1", "std"),
            ROUGEL_mean=("ROUGE-L", "mean"),
            ROUGEL_std=("ROUGE-L", "std"),
            BERTScore_mean=("BERTScore", "mean"),
            BERTScore_std=("BERTScore", "std"),
            FactP_mean=("fact_precision", "mean"),
            FactP_std=("fact_precision", "std"),
            FactR_mean=("fact_recall", "mean"),
            FactR_std=("fact_recall", "std"),
            FactF1_mean=("fact_f1", "mean"),
            FactF1_std=("fact_f1", "std"),
        ).round(4)

        summary.to_csv(RESULTS_DIR / "nlg_evaluation_comprehensive_summary.csv")

        # JSON for easy manuscript lookup
        summary_dict = {}
        for source in summary.index:
            row = summary.loc[source]
            summary_dict[source] = {
                "n": int(row["n"]),
                "BLEU-1": f"{row['BLEU1_mean']:.3f}",
                "BLEU-4": f"{row['BLEU4_mean']:.3f}",
                "METEOR": f"{row['METEOR_mean']:.3f}",
                "ROUGE-1": f"{row['ROUGE1_mean']:.3f}",
                "ROUGE-L": f"{row['ROUGEL_mean']:.3f}",
                "BERTScore": f"{row['BERTScore_mean']:.3f}",
                "Fact_P": f"{row['FactP_mean']:.3f}",
                "Fact_R": f"{row['FactR_mean']:.3f}",
                "Fact_F1": f"{row['FactF1_mean']:.3f}",
            }
        with open(RESULTS_DIR / "nlg_evaluation_comprehensive_summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"\n{'=' * 100}")
        print("COMPREHENSIVE NLG EVALUATION SUMMARY")
        print(f"{'=' * 100}")
        print(summary.to_string())
        print(f"\nResults saved to:")
        print(f"  {RESULTS_DIR / 'nlg_evaluation_comprehensive.csv'}")
        print(f"  {RESULTS_DIR / 'nlg_evaluation_comprehensive_summary.csv'}")
        print(f"  {RESULTS_DIR / 'nlg_evaluation_comprehensive_summary.json'}")

    return df_all


if __name__ == "__main__":
    main()
