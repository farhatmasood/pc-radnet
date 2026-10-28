"""
benchmark_meditron3.py

Evaluate Meditron3-8B (OpenMeditron, 2024) for lumbar spine MRI report generation
using the same matched-patient protocol as benchmark_report_llms_v2.py.

Meditron3-8B is a medical-domain continued-pretraining of Llama-3.1-8B on
clinical guidelines, peer-reviewed publications, and synthetic medical data.
It is a foundation model (NOT instruction-tuned) but inherits the Llama-3.1
chat template.

Loaded with 4-bit NF4 quantization to fit within 8GB VRAM.

Outputs:
  results/report_benchmark/reports/meditron3_8b/report_XXXX.txt
  results/report_benchmark/metrics_per_patient_meditron3_8b.csv
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ── Data loading (same as benchmark_report_llms_v2.py) ──────────────
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
        lines.append(f"  - Disc height (pixels): {height:.2f}" if height > 0 else "  - Disc height (pixels): N/A")
        lines.append("  - Disorders: " + (", ".join(present) if present else "none"))
    return "\n".join(lines)


def build_prompt(patient_id: int, level_data: Dict[str, dict]) -> str:
    levels_block = _format_level_block(level_data)
    return f"""You are a board-certified neuroradiologist.
Generate a lumbar spine MRI report from the structured model measurements below.
Use exactly these sections: TECHNIQUE, FINDINGS, IMPRESSION.
Do not invent measurements or findings not present in the inputs.

Patient ID: {patient_id:04d}
Structured measurements:
{levels_block}

Output constraints:
- Mention only abnormalities supported by input disorders list.
- Keep report concise and clinically structured.
- 120 to 260 words."""


# ── Evaluation (same synonyms as benchmark_report_llms_v2.py) ──────
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


# ── Model loader & generator ────────────────────────────────────────
def load_model():
    """Load Meditron3-8B with 4-bit quantization for 8GB VRAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "OpenMeditron/Meditron3-8B"
    print(f"  Loading Meditron3-8B ({model_id}) with 4-bit quantization ...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    token = os.environ.get("HF_TOKEN") or True  # True uses cached token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


def generate_report(model, tokenizer, prompt: str, max_new_tokens: int = 350) -> str:
    """Generate report using Meditron3-8B with Llama-3.1 chat template."""
    messages = [
        {"role": "system", "content": "You are a board-certified neuroradiologist."},
        {"role": "user", "content": prompt},
    ]

    # Use the tokenizer's built-in chat template (Llama-3.1 format)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Meditron3-8B Report Generation Benchmark")
    print("=" * 70)

    gt_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")
    patient_inputs = load_patient_measurements()
    print(f"Loaded measurements for {len(patient_inputs)} patients")

    model_dir = OUT_ROOT / "reports" / "meditron3_8b"
    model_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    reports: Dict[int, str] = {}
    elapsed: List[float] = []

    for pid in TARGET_IDS:
        if pid not in patient_inputs:
            print(f"  Skipping patient {pid}: no measurements")
            continue

        prompt = build_prompt(pid, patient_inputs[pid])
        t0 = time.time()
        try:
            report_text = generate_report(model, tokenizer, prompt)
        except Exception as e:
            report_text = f"[generation failed: {type(e).__name__}: {e}]"
            print(f"  ERROR patient {pid}: {e}")
        dt = time.time() - t0
        elapsed.append(dt)

        (model_dir / f"report_{pid:04d}.txt").write_text(report_text, encoding="utf-8")
        reports[pid] = report_text
        print(f"  Patient {pid:04d}: {len(report_text.split())} words, {dt:.1f}s")

    # ── Evaluate ────────────────────────────────────────────────────
    rows = []
    for pid in TARGET_IDS:
        if pid not in reports:
            continue
        gt = gt_notes[gt_notes["Patient_ID"] == pid]
        if gt.empty:
            continue
        gt_row = gt.iloc[0]
        gt_disorders = {c: int(gt_row.get(c, 0)) for c in DISORDER_COLS_GT}
        m = evaluate_report_factual_consistency(reports[pid], gt_disorders)
        m["patient_id"] = pid
        m["method"] = "Meditron3-8B"
        m["word_count"] = len(reports[pid].split())
        rows.append(m)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_ROOT / "metrics_per_patient_meditron3_8b.csv", index=False)

    print("\n" + "=" * 70)
    print("Meditron3-8B RESULTS SUMMARY")
    print("=" * 70)
    if not df_metrics.empty:
        print(f"  n = {len(df_metrics)}")
        print(f"  Precision: {df_metrics['precision'].mean():.3f} +/- {df_metrics['precision'].std():.3f}")
        print(f"  Recall:    {df_metrics['recall'].mean():.3f} +/- {df_metrics['recall'].std():.3f}")
        print(f"  F1:        {df_metrics['f1'].mean():.3f} +/- {df_metrics['f1'].std():.3f}")
        print(f"  Words:     {df_metrics['word_count'].mean():.0f} +/- {df_metrics['word_count'].std():.0f}")
        if elapsed:
            print(f"  Time/pt:   {np.mean(elapsed):.2f}s")
    else:
        print("  NO VALID REPORTS")

    summary = {
        "model": "Meditron3-8B",
        "model_id": "OpenMeditron/Meditron3-8B",
        "base_model": "meta-llama/Meta-Llama-3.1-8B",
        "n": int(len(df_metrics)),
        "precision_mean": float(df_metrics["precision"].mean()) if not df_metrics.empty else None,
        "recall_mean": float(df_metrics["recall"].mean()) if not df_metrics.empty else None,
        "f1_mean": float(df_metrics["f1"].mean()) if not df_metrics.empty else None,
        "words_mean": float(df_metrics["word_count"].mean()) if not df_metrics.empty else None,
        "mean_time_s": float(np.mean(elapsed)) if elapsed else None,
    }
    with open(OUT_ROOT / "summary_meditron3_8b.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_ROOT}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
