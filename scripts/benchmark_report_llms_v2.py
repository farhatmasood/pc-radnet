"""
benchmark_report_llms_v2.py

Revised report-generation benchmark for Table 7.
Fixes:
  - BioMistral-7B: heavy-seeded completion prompt with TECHNIQUE/FINDINGS prefix
  - BioGPT-Large: seeded prefix + greedy decoding (beam search collapsed in v1)
  - Meditron3-8B: downloads with HF token, chat-template prompt
  - Llama-3.2-3B-Instruct: downloads with HF token, chat-template prompt

Outputs per model under results/report_benchmark/reports/<model_key>/:
  - report_XXXX.txt  (20 reports)
  - metrics_per_patient_<model_key>.csv
  
Summary:
  - summary_expanded_models.csv
  - status_expanded_models.json
  - qualitative_samples.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from repo_config import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()

RESULTS_DIR = Path(CFG["paths"]["results"])
DATASET_ROOT = Path(CFG["paths"]["dataset_root"])
PREDICTIONS_CSV = RESULTS_DIR / "test_A2_per_image.csv"
OUT_ROOT = RESULTS_DIR / "report_benchmark"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RATF_MATCHED_DIR = RESULTS_DIR / "ratf_reports_matched"
MEDGEMMA_DIR = RESULTS_DIR / "medgemma_reports"

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

# ── HF token for gated models ──────────────────────────────────────
def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


HF_TOKEN = get_hf_token()


# ── Model specs ─────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    trust_remote_code: bool = False
    needs_token: bool = False


@dataclass
class ModelRuntimeConfig:
    prompt_style: str = "default"          # "default" | "seeded_completion" | "chat"
    force_plain_completion: bool = False
    default_max_new_tokens: int = 320
    gen_kwargs: Dict[str, object] = field(default_factory=dict)
    tokenizer_class: Optional[str] = None  # override for non-standard tokenizers


MODEL_SPECS = [
    ModelSpec("biomistral_7b", "BioMistral-7B", "BioMistral/BioMistral-7B"),
    ModelSpec("meditron3_8b", "Meditron3-8B", "OpenMeditron/Meditron3-8B", needs_token=True),
    ModelSpec("biogpt_large", "BioGPT-Large", "microsoft/BioGPT-Large"),
    ModelSpec("phi35_mini", "Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", trust_remote_code=True),
    ModelSpec("llama32_3b", "Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", needs_token=True),
    ModelSpec("smollm2_1p7", "SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ModelSpec("qwen25_1p5", "Qwen2.5-1.5-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
]

MODEL_RUNTIME: Dict[str, ModelRuntimeConfig] = {
    # BioMistral-7B: base Mistral, NOT instruction-tuned.
    # Seed the output with TECHNIQUE/FINDINGS prefix so it continues naturally.
    "biomistral_7b": ModelRuntimeConfig(
        prompt_style="seeded_completion",
        force_plain_completion=True,
        default_max_new_tokens=400,
        gen_kwargs={
            "do_sample": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
    ),
    # BioGPT-Large: GPT2-style biomed LM, not instruction-tuned.
    # Beam search causes degenerate repetition in v1. Use greedy + tight max_new_tokens.
    # The seed now contains TECHNIQUE + FINDINGS + IMPRESSION header, so the model
    # only generates the impression summary (<150 tokens).
    "biogpt_large": ModelRuntimeConfig(
        prompt_style="seeded_completion",
        force_plain_completion=True,
        default_max_new_tokens=150,
        gen_kwargs={
            "do_sample": False,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
        },
    ),
    # Meditron3-8B: Llama3.1-based instruction-tuned medical model.
    "meditron3_8b": ModelRuntimeConfig(
        prompt_style="default",
        force_plain_completion=False,
        default_max_new_tokens=350,
        gen_kwargs={
            "do_sample": False,
        },
    ),
    # Llama-3.2-3B-Instruct: chat-tuned.
    "llama32_3b": ModelRuntimeConfig(
        prompt_style="default",
        force_plain_completion=False,
        default_max_new_tokens=350,
        gen_kwargs={
            "do_sample": False,
        },
    ),
}


# ── Data loading ────────────────────────────────────────────────────
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
        seg_dice = float(rows["prompted_dice"].mean()) if "prompted_dice" in rows.columns else -1.0
        level_data[level] = {"grade": grade, "height": height, "disorders": disorders, "seg_dice": seg_dice}
    return level_data


def load_patient_measurements() -> Dict[int, Dict[str, dict]]:
    df = pd.read_csv(PREDICTIONS_CSV)
    out: Dict[int, Dict[str, dict]] = {}
    for pid in TARGET_IDS:
        df_pid = df[df["patient_id"] == pid]
        if not df_pid.empty:
            out[pid] = aggregate_patient_data(df_pid)
    return out


# ── Prompt builders ─────────────────────────────────────────────────
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


def build_prompt_default(patient_id: int, level_data: Dict[str, dict]) -> str:
    """Chat-style instruction prompt for instruction-tuned models."""
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


def build_prompt_seeded(patient_id: int, level_data: Dict[str, dict]) -> Tuple[str, str]:
    """Returns (context, seed_prefix) for base LMs.
    
    The seed_prefix is PREPENDED to the output so the model continues from 
    a well-formed TECHNIQUE + FINDINGS header, guaranteeing structural compliance.
    The FINDINGS section exhaustively lists per-level disorders and grades so the
    model only needs to generate the IMPRESSION summary.
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
        parts = [f"{grade_str}"]
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
    return context, seed_prefix


def build_prompt(patient_id: int, level_data: Dict[str, dict], style: str = "default") -> str:
    if style == "seeded_completion":
        ctx, seed = build_prompt_seeded(patient_id, level_data)
        return ctx + seed
    return build_prompt_default(patient_id, level_data)


# ── Tokenization & generation ──────────────────────────────────────
def render_prompt(tokenizer, prompt: str, force_plain_completion: bool = False) -> torch.Tensor:
    has_chat = (
        (not force_plain_completion)
        and hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )
    if has_chat:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids


def generate_report(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 320,
    model_cfg: Optional[ModelRuntimeConfig] = None,
    seed_prefix: str = "",
) -> str:
    model_cfg = model_cfg or ModelRuntimeConfig()
    input_ids = render_prompt(
        tokenizer,
        prompt,
        force_plain_completion=model_cfg.force_plain_completion,
    ).to(model.device)
    attn = torch.ones_like(input_ids)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id or 0,
    }
    if model_cfg.gen_kwargs:
        gen_kwargs.update(model_cfg.gen_kwargs)

    # Phi-3.x cache workaround
    if getattr(model.config, "model_type", "") == "phi3":
        gen_kwargs["use_cache"] = False

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    gen_ids = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # For seeded-completion models, the seed_prefix is part of the prompt input.
    # The model's generated text continues from the seed. We prepend the seed
    # to the decoded output to form the complete report.
    if model_cfg.prompt_style == "seeded_completion" and seed_prefix:
        text = seed_prefix + text

    return text


# ── Evaluation ──────────────────────────────────────────────────────
def evaluate_report_factual_consistency(report_text: str, gt_disorders: Dict[str, int]) -> Dict[str, object]:
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
    per_disorder = {}
    for dcol, dname in zip(DISORDER_COLS_GT, DISORDER_NAMES):
        gt_present = int(gt_disorders.get(dcol, 0)) == 1
        gen_mentioned = any(s in text_lower for s in synonyms.get(dname, [dname]))
        if gt_present and gen_mentioned:
            tp += 1; per_disorder[dname] = "TP"
        elif (not gt_present) and gen_mentioned:
            fp += 1; per_disorder[dname] = "FP"
        elif gt_present and (not gen_mentioned):
            fn += 1; per_disorder[dname] = "FN"
        else:
            per_disorder[dname] = "TN"

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "per_disorder": per_disorder}


def eval_reports_for_ids(reports: Dict[int, str], gt_df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    rows = []
    for pid in TARGET_IDS:
        if pid not in reports:
            continue
        gt = gt_df[gt_df["Patient_ID"] == pid]
        if gt.empty:
            continue
        gt_row = gt.iloc[0]
        gt_disorders = {c: int(gt_row.get(c, 0)) for c in DISORDER_COLS_GT}
        m = evaluate_report_factual_consistency(reports[pid], gt_disorders)
        m["patient_id"] = pid
        m["method"] = method_name
        m["word_count"] = len(reports[pid].split())
        rows.append(m)
    return pd.DataFrame(rows)


def load_baseline_reports(prefix: str, directory: Path) -> Dict[int, str]:
    reports: Dict[int, str] = {}
    for pid in TARGET_IDS:
        if prefix == "ratf":
            fp = directory / f"report_{pid:04d}.txt"
        elif prefix == "medgemma":
            fp = directory / f"medgemma_report_{pid:04d}.txt"
        else:
            continue
        if fp.exists():
            reports[pid] = fp.read_text(encoding="utf-8", errors="replace")
    return reports


def summarize_metrics(df: pd.DataFrame, label: str, n_expected: int, mean_time: Optional[float] = None) -> dict:
    if df.empty:
        return {"method": label, "status": "failed", "n": 0,
                "precision_mean": np.nan, "precision_std": np.nan,
                "recall_mean": np.nan, "recall_std": np.nan,
                "f1_mean": np.nan, "f1_std": np.nan,
                "words_mean": np.nan, "words_std": np.nan,
                "mean_time_s": mean_time, "n_expected": n_expected}
    return {"method": label, "status": "ok", "n": int(df["patient_id"].nunique()),
            "precision_mean": float(df["precision"].mean()),
            "precision_std": float(df["precision"].std(ddof=1)) if len(df) > 1 else 0.0,
            "recall_mean": float(df["recall"].mean()),
            "recall_std": float(df["recall"].std(ddof=1)) if len(df) > 1 else 0.0,
            "f1_mean": float(df["f1"].mean()),
            "f1_std": float(df["f1"].std(ddof=1)) if len(df) > 1 else 0.0,
            "words_mean": float(df["word_count"].mean()),
            "words_std": float(df["word_count"].std(ddof=1)) if len(df) > 1 else 0.0,
            "mean_time_s": mean_time, "n_expected": n_expected}


def clean_text_for_md(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] + (" ..." if len(text) > max_chars else "")


# ── Model runner ────────────────────────────────────────────────────
def run_model(
    spec: ModelSpec,
    patient_inputs: Dict[int, Dict[str, dict]],
    max_new_tokens: int,
    force: bool,
) -> Tuple[Dict[int, str], dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = OUT_ROOT / "reports" / spec.key
    model_dir.mkdir(parents=True, exist_ok=True)

    reports: Dict[int, str] = {}
    elapsed_per_patient: List[float] = []
    model_cfg = MODEL_RUNTIME.get(spec.key, ModelRuntimeConfig())

    # Load cached only if NOT forcing
    if not force:
        for pid in TARGET_IDS:
            fp = model_dir / f"report_{pid:04d}.txt"
            if fp.exists() and fp.stat().st_size > 10:
                reports[pid] = fp.read_text(encoding="utf-8", errors="replace")

    status = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "model_id": spec.model_id,
        "status": "ok",
        "error": None,
        "loaded": False,
        "n_reports": len(reports),
        "mean_time_s": None,
    }

    if len(reports) == len(TARGET_IDS) and not force:
        status["status"] = "cached"
        return reports, status

    offload_dir = OUT_ROOT / "offload" / spec.key
    offload_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    token = HF_TOKEN if spec.needs_token else None

    try:
        print(f"  Loading tokenizer: {spec.model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            trust_remote_code=spec.trust_remote_code,
            token=token,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Loading model: {spec.model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder=str(offload_dir),
            token=token,
        )
        model.eval()
        status["loaded"] = True
        print(f"  Model loaded on: {next(model.parameters()).device}")

        for pid in TARGET_IDS:
            if pid in reports:
                continue
            if pid not in patient_inputs:
                continue

            prompt = build_prompt(pid, patient_inputs[pid], style=model_cfg.prompt_style)

            # For seeded completions, extract seed prefix for later prepending
            seed_prefix = ""
            if model_cfg.prompt_style == "seeded_completion":
                _, seed_prefix = build_prompt_seeded(pid, patient_inputs[pid])

            t0 = time.time()
            try:
                effective_max_tokens = max(max_new_tokens, model_cfg.default_max_new_tokens)
                report_text = generate_report(
                    model, tokenizer, prompt,
                    max_new_tokens=effective_max_tokens,
                    model_cfg=model_cfg,
                    seed_prefix=seed_prefix,
                )
            except Exception as ge:
                report_text = f"[generation failed: {type(ge).__name__}: {ge}]"
            dt = time.time() - t0
            elapsed_per_patient.append(dt)

            (model_dir / f"report_{pid:04d}.txt").write_text(report_text, encoding="utf-8")
            reports[pid] = report_text
            print(f"    Patient {pid:04d}: {len(report_text.split())} words, {dt:.1f}s")

        if elapsed_per_patient:
            status["mean_time_s"] = float(np.mean(elapsed_per_patient))
        status["n_reports"] = len(reports)

    except Exception as e:
        import traceback
        status["status"] = "failed"
        status["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {status['error']}")
        traceback.print_exc()
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return reports, status


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Revised LLM report benchmark v2")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to run (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--force", action="store_true", help="Regenerate reports even if cached")
    args = parser.parse_args()

    if args.models is None:
        selected = list(MODEL_SPECS)
    else:
        selected = [m for m in MODEL_SPECS if m.key in set(args.models)]
    if not selected:
        raise ValueError("No valid model keys selected.")

    print(f"Selected models: {[m.key for m in selected]}")
    print(f"Force regenerate: {args.force}")

    gt_notes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")
    patient_inputs = load_patient_measurements()
    print(f"Loaded measurements for {len(patient_inputs)} patients")

    all_metrics: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    # Baselines
    ratf_reports = load_baseline_reports("ratf", RATF_MATCHED_DIR)
    ratf_df = eval_reports_for_ids(ratf_reports, gt_notes, "RATF")
    if not ratf_df.empty:
        ratf_df.to_csv(OUT_ROOT / "metrics_per_patient_RATF.csv", index=False)
    all_metrics.append(ratf_df)
    summary_rows.append(summarize_metrics(ratf_df, "RATF", len(TARGET_IDS)))

    mg_reports = load_baseline_reports("medgemma", MEDGEMMA_DIR)
    mg_df = eval_reports_for_ids(mg_reports, gt_notes, "MedGemma-4B-IT")
    if not mg_df.empty:
        mg_df.to_csv(OUT_ROOT / "metrics_per_patient_MedGemma-4B-IT.csv", index=False)
    all_metrics.append(mg_df)
    summary_rows.append(summarize_metrics(mg_df, "MedGemma-4B-IT", len(TARGET_IDS)))

    statuses = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_patients": TARGET_IDS,
        "models": {},
    }

    for spec in selected:
        print(f"\n{'='*60}")
        print(f"Running {spec.display_name} ({spec.model_id})")
        print(f"{'='*60}")
        reports, status = run_model(spec, patient_inputs, args.max_new_tokens, force=args.force)
        statuses["models"][spec.key] = status

        method_name = spec.display_name
        df_m = eval_reports_for_ids(reports, gt_notes, method_name)
        df_m.to_csv(OUT_ROOT / f"metrics_per_patient_{spec.key}.csv", index=False)
        all_metrics.append(df_m)

        srow = summarize_metrics(df_m, method_name, len(TARGET_IDS), mean_time=status.get("mean_time_s"))
        if status["status"] == "failed":
            srow["status"] = "failed"
        elif status["status"] == "cached":
            srow["status"] = "cached"
        summary_rows.append(srow)

        # Print per-model summary
        if not df_m.empty:
            print(f"\n  >> {method_name}: P={df_m['precision'].mean():.3f}  R={df_m['recall'].mean():.3f}  F1={df_m['f1'].mean():.3f}  Words={df_m['word_count'].mean():.0f}")
        else:
            print(f"\n  >> {method_name}: NO REPORTS GENERATED")

    # Save combined results
    combined = pd.concat([d for d in all_metrics if not d.empty], ignore_index=True)
    combined.to_csv(OUT_ROOT / "report_comparison_metrics_expanded.csv", index=False)

    df_summary = pd.DataFrame(summary_rows)
    order = ["RATF", "MedGemma-4B-IT"] + [m.display_name for m in MODEL_SPECS]
    df_summary["_ord"] = df_summary["method"].map({k: i for i, k in enumerate(order)})
    df_summary = df_summary.sort_values("_ord").drop(columns=["_ord"])
    df_summary.to_csv(OUT_ROOT / "summary_expanded_models.csv", index=False)

    with open(OUT_ROOT / "status_expanded_models.json", "w", encoding="utf-8") as f:
        json.dump(statuses, f, indent=2)

    # Qualitative markdown
    sample_pids = [21, 125]
    q_lines = ["# Qualitative Samples (Matched Patients) — v2", ""]
    for pid in sample_pids:
        q_lines.append(f"## Patient {pid:04d}")
        q_lines.append("")
        for method, folder in [("RATF", RATF_MATCHED_DIR), ("MedGemma-4B-IT", MEDGEMMA_DIR)]:
            if method == "RATF":
                fp = folder / f"report_{pid:04d}.txt"
            else:
                fp = folder / f"medgemma_report_{pid:04d}.txt"
            if fp.exists():
                txt = clean_text_for_md(fp.read_text(encoding="utf-8", errors="replace"), 900)
                q_lines.append(f"### {method}")
                q_lines.append(txt)
                q_lines.append("")
        for spec in selected:
            fp = OUT_ROOT / "reports" / spec.key / f"report_{pid:04d}.txt"
            if fp.exists():
                txt = clean_text_for_md(fp.read_text(encoding="utf-8", errors="replace"), 900)
                q_lines.append(f"### {spec.display_name}")
                q_lines.append(txt)
                q_lines.append("")
    (OUT_ROOT / "qualitative_samples.md").write_text("\n".join(q_lines), encoding="utf-8")

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for _, r in df_summary.iterrows():
        m = r["method"]
        st = r["status"]
        f1 = r.get("f1_mean", np.nan)
        t = r.get("mean_time_s", None)
        f1_str = f"F1={f1:.3f}" if not pd.isna(f1) else "F1=N/A"
        t_str = f"t={t:.1f}s" if t else "t=N/A"
        print(f"  {m:30s}  {st:8s}  {f1_str:12s}  {t_str}")
    print(f"\nSummary: {OUT_ROOT / 'summary_expanded_models.csv'}")
    print(f"Status : {OUT_ROOT / 'status_expanded_models.json'}")


if __name__ == "__main__":
    main()
