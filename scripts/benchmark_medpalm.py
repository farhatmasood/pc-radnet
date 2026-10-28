"""
benchmark_medpalm.py

Evaluate Med-PaLM (open-source reimplementation, kyegomez/Med-Palm) for
lumbar spine MRI report generation using the same matched-patient protocol
as all other LLM baselines.

Med-PaLM uses a ViT encoder (from x-transformers) + PaLM-style decoder
with cross-attention, wrapped in AutoregressiveWrapper for generation.

As no pretrained checkpoint is distributed for this open-source
reimplementation, we use the same **seeded-completion** strategy applied
to other non-instruction-tuned baselines (BioGPT-Large, BioMistral-7B):
the TECHNIQUE and FINDINGS sections are pre-filled from the predicted
measurements, and the model generates the IMPRESSION continuation.
The final report = seed_prefix + model-generated continuation.

Outputs:
  results/report_benchmark/reports/medpalm/report_XXXX.txt
  results/report_benchmark/metrics_per_patient_medpalm.csv
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from repo_config import load_config

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()

RESULTS_DIR = Path(CFG["paths"]["results"])
DATASET_ROOT = Path(CFG["paths"]["dataset_root"])
PREDICTIONS_CSV = RESULTS_DIR / "test_A2_per_image.csv"
OUT_ROOT = RESULTS_DIR / "report_benchmark"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_IDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113,
              118, 124, 125, 140, 150, 152, 165, 167, 177, 181]

LEVEL_MAP = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
DISORDER_NAMES = [
    "disc herniation", "disc protrusion", "disc bulge",
    "thecal sac compression", "nerve root compression",
    "spinal canal stenosis", "spondylolisthesis",
    "desiccated disc", "ligamentum flavum hypertrophy",
]
DISORDER_COLS_GT = [
    "Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
    "Thecal_Sac_Compression", "Nerve_Root_Compression",
    "Spinal_Canal_Stenosis", "Spondylolisthesis",
    "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy",
]
DISORDER_PROB_COLS_PRED = [
    "disorder_Herniation_prob", "disorder_Protrusion_prob", "disorder_Bulge_prob",
    "disorder_ThecalComp_prob", "disorder_NerveComp_prob", "disorder_Stenosis_prob",
    "disorder_Spondylolisthesis_prob", "disorder_Desiccated_prob", "disorder_LigFlavum_prob",
]


# ── Data loading (same as other benchmarks) ─────────────────────────
def aggregate_patient_data(df_pid: pd.DataFrame) -> Dict[str, dict]:
    level_data: Dict[str, dict] = {}
    for level in ["D3", "D4", "D5"]:
        rows = df_pid[(df_pid["level"] == level) & (df_pid["plane"] == "axial")]
        if rows.empty:
            continue
        valid_grades = rows[rows["pred_grade"] > 0]["pred_grade"]
        grade = int(valid_grades.mode().iloc[0]) if not valid_grades.empty else -1
        valid_ht = rows[rows["pred_height"] > 0]["pred_height"]
        height = float(valid_ht.mean()) if not valid_ht.empty else -1.0
        disorders = []
        for col in DISORDER_PROB_COLS_PRED:
            disorders.append(float(rows[col].max()) if col in rows.columns else 0.0)
        level_data[level] = {"grade": grade, "height": height, "disorders": disorders}
    return level_data


def load_patient_measurements() -> Dict[int, Dict[str, dict]]:
    df = pd.read_csv(PREDICTIONS_CSV)
    out: Dict[int, Dict[str, dict]] = {}
    for pid in TARGET_IDS:
        df_pid = df[df["patient_id"] == pid]
        if not df_pid.empty:
            out[pid] = aggregate_patient_data(df_pid)
    return out


def _format_level_block(level_data: Dict[str, dict]) -> str:
    lines = []
    for level in ["D3", "D4", "D5"]:
        if level not in level_data:
            continue
        ld = level_data[level]
        lname = LEVEL_MAP[level]
        grade = ld.get("grade", -1)
        height = ld.get("height", -1.0)
        disorders = ld.get("disorders", [])
        present = [DISORDER_NAMES[i] for i, prob in enumerate(disorders) if prob > 0.5]
        lines.append(f"- {lname}:")
        lines.append(f"  - Pfirrmann grade: {grade if grade > 0 else 'N/A'}")
        lines.append("  - Disc height (pixels): {:.2f}".format(height) if height > 0 else "  - Disc height (pixels): N/A")
        lines.append("  - Disorders: " + (", ".join(present) if present else "none"))
    return "\n".join(lines)


def build_prompt_seeded(patient_id: int, level_data: Dict[str, dict]) -> tuple:
    """Build seeded-completion prompt (same strategy as BioGPT/BioMistral).

    Returns (full_prompt, seed_prefix).
    seed_prefix contains pre-filled TECHNIQUE + FINDINGS with per-level
    disorder/grade findings, plus the IMPRESSION header.
    The model only needs to generate the impression continuation.
    """
    finding_seeds = []
    for level in ["D3", "D4", "D5"]:
        if level not in level_data:
            continue
        ld = level_data[level]
        lname = LEVEL_MAP[level]
        grade = ld.get("grade", -1)
        height = ld.get("height", -1.0)
        disorders = ld.get("disorders", [])
        present = [DISORDER_NAMES[i] for i, prob in enumerate(disorders) if prob > 0.5]
        grade_str = f"Pfirrmann grade {grade}" if grade > 0 else "normal disc signal"
        height_str = f"disc height {height:.1f} pixels" if height > 0 else ""
        disorder_str = ", ".join(present) if present else "no significant abnormality"
        parts = [grade_str]
        if height_str:
            parts.append(height_str)
        parts.append(disorder_str)
        finding_seeds.append(f"- {lname}: {'; '.join(parts)}.")

    levels_block = _format_level_block(level_data)

    context = f"""Below are structured lumbar spine MRI measurements. Write a radiological report.

Patient ID: {patient_id:04d}
Structured measurements:
{levels_block}

Report:
"""

    seed_prefix = (
        "TECHNIQUE: Lumbar spine MRI was reviewed using the provided structured measurements.\n\n"
        "FINDINGS:\n"
        + "\n".join(finding_seeds)
        + "\n\nIMPRESSION:\n"
    )
    return context + seed_prefix, seed_prefix


# ── Evaluation (same synonyms as other benchmarks) ──────────────────
def evaluate_report_factual_consistency(report_text: str, gt_disorders: Dict[str, int]) -> Dict:
    text_lower = report_text.lower()
    synonyms = {
        "disc herniation": ["herniation", "herniated", "herniating", "disc herniation"],
        "disc protrusion": ["protrusion", "protruding", "protruded", "disc protrusion"],
        "disc bulge": ["bulge", "bulging", "disc bulge", "broad-based bulge", "diffuse bulge"],
        "thecal sac compression": ["thecal sac compression", "thecal sac", "thecal compression",
                                    "compression of the thecal", "effacement of the thecal"],
        "nerve root compression": ["nerve root compression", "nerve root", "neural compression",
                                    "nerve compression", "radiculopathy", "neural foraminal"],
        "spinal canal stenosis": ["stenosis", "canal narrowing", "spinal canal stenosis",
                                   "spinal stenosis", "central stenosis"],
        "spondylolisthesis": ["spondylolisthesis", "listhesis", "anterolisthesis", "retrolisthesis"],
        "desiccated disc": ["desiccated", "desiccation", "disc desiccation", "dehydrated",
                            "loss of signal", "dark disc", "decreased signal intensity",
                            "decreased t2 signal", "reduced signal"],
        "ligamentum flavum hypertrophy": ["ligamentum flavum", "flavum hypertrophy",
                                          "ligamentous hypertrophy", "thickening of the ligamentum"],
    }

    tp, fp, fn = 0, 0, 0
    for dcol, dname in zip(DISORDER_COLS_GT, DISORDER_NAMES):
        gt_present = int(gt_disorders.get(dcol, 0)) == 1
        gen_mentioned = any(s in text_lower for s in synonyms.get(dname, [dname]))
        if gt_present and gen_mentioned:
            tp += 1
        elif (not gt_present) and gen_mentioned:
            fp += 1
        elif gt_present and (not gen_mentioned):
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ── Med-PaLM model loader & generator ──────────────────────────────
def load_medpalm():
    """Load Med-PaLM model (open-source reimplementation).

    This uses the medpalm PyPI package (kyegomez/Med-Palm) which provides
    the Med-PaLM architecture: ViT encoder + PaLM decoder with
    cross-attention, wrapped in AutoregressiveWrapper for generation.

    Note: No pretrained checkpoint is available for this open-source
    implementation - the model initializes with the architecture skeleton
    and generates using its PaLM decoder backbone.
    """
    from medpalm import MedPalm

    # Use a tokenizer for encoding/decoding text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Med-PaLM architecture: ViT encoder + PaLM decoder
    # num_tokens must cover ALL token IDs (including added special tokens)
    VOCAB_SIZE = len(tokenizer)  # includes added tokens beyond vocab_size
    model = MedPalm(
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=VOCAB_SIZE,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        attn_flash=False,
    )

    # Use CPU - model is only 97M params and CUDA has indexing issues
    # with the medpalm transformer's ALiBi implementation
    device = torch.device("cpu")
    model = model.float().to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Med-PaLM model loaded: {total_params/1e6:.1f}M parameters on {device}")

    return model, tokenizer, device


def generate_report(model, tokenizer, device, prompt: str,
                    seed_prefix: str = "",
                    max_new_tokens: int = 150) -> str:
    """Generate a report using Med-PaLM's autoregressive decoder.

    Uses seeded-completion: the prompt already contains the full context +
    seed_prefix (TECHNIQUE/FINDINGS). The model generates a continuation
    (the IMPRESSION). The final report = seed_prefix + continuation.
    """
    # Tokenize prompt (context + seed_prefix already concatenated)
    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                  truncation=True, max_length=768)
    prompt_len = input_ids.shape[1]
    input_ids = input_ids.squeeze(0).to(device)  # [seq_len]

    # Create a dummy image (Med-PaLM expects image input for cross-attention)
    dummy_img = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device)

    # Get image context embeddings
    with torch.no_grad():
        encoded_img = model.encoder(dummy_img, return_embeddings=True)

    # Generate using the decoder (AutoregressiveWrapper)
    with torch.no_grad():
        output_ids = model.decoder.generate(
            start_tokens=input_ids,
            seq_len=max_new_tokens,
            temperature=0.8,
            context=encoded_img,
        )

    # Decode only the NEW tokens (the impression continuation)
    new_token_ids = output_ids[prompt_len:]
    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    # Final report = seed_prefix (TECHNIQUE+FINDINGS) + model continuation
    if seed_prefix:
        return seed_prefix + continuation
    return continuation


# ── Main benchmark ──────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Med-PaLM Report Generation Benchmark")
    print("=" * 70)

    # Load measurements
    patient_data = load_patient_measurements()
    print(f"Loaded measurements for {len(patient_data)} patients")

    # Load ground truth
    gt_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")

    # Load model
    model, tokenizer, device = load_medpalm()

    # Output dir
    report_dir = OUT_ROOT / "reports" / "medpalm"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate reports
    results = []
    for pid in TARGET_IDS:
        if pid not in patient_data:
            print(f"  SKIP patient {pid}: no measurement data")
            continue

        prompt, seed_prefix = build_prompt_seeded(pid, patient_data[pid])
        t0 = time.time()
        try:
            report = generate_report(model, tokenizer, device, prompt,
                                     seed_prefix=seed_prefix)
        except Exception as e:
            report = f"ERROR generating report: {e}"
            print(f"  ERROR patient {pid}: {e}")
        elapsed = time.time() - t0

        # Save report
        report_path = report_dir / f"report_{pid:04d}.txt"
        report_path.write_text(report, encoding="utf-8")

        # Evaluate
        gt_row = gt_notes[gt_notes["Patient_ID"] == pid]
        if not gt_row.empty:
            gt_disorders = gt_row.iloc[0].to_dict()
            metrics = evaluate_report_factual_consistency(report, gt_disorders)
        else:
            metrics = {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0}

        word_count = len(report.split())
        results.append({
            "patient_id": pid,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "words": word_count,
            "time_s": elapsed,
        })
        print(f"  Patient {pid:04d}: {word_count} words, {elapsed:.1f}s")

    # Summary
    df = pd.DataFrame(results)
    df.to_csv(OUT_ROOT / "metrics_per_patient_medpalm.csv", index=False)

    print("\n" + "=" * 70)
    print("Med-PaLM RESULTS SUMMARY")
    print("=" * 70)
    print(f"  n = {len(df)}")
    print(f"  Precision: {df['precision'].mean():.3f} +/- {df['precision'].std():.3f}")
    print(f"  Recall:    {df['recall'].mean():.3f} +/- {df['recall'].std():.3f}")
    print(f"  F1:        {df['f1'].mean():.3f} +/- {df['f1'].std():.3f}")
    print(f"  Words:     {df['words'].mean():.0f} +/- {df['words'].std():.0f}")
    print(f"  Time/pt:   {df['time_s'].mean():.2f}s")

    # Save summary JSON
    summary = {
        "model": "Med-PaLM",
        "model_id": "kyegomez/Med-Palm (open-source reimplementation)",
        "architecture": "ViT encoder + PaLM decoder (x-transformers)",
        "n": len(df),
        "precision_mean": float(df["precision"].mean()),
        "recall_mean": float(df["recall"].mean()),
        "f1_mean": float(df["f1"].mean()),
        "words_mean": float(df["words"].mean()),
        "mean_time_s": float(df["time_s"].mean()),
    }
    with open(OUT_ROOT / "summary_medpalm.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
