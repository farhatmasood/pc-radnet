"""BERTScore F1 for the three RATF stages on the 20-patient benchmark, computed exactly as in
scripts/evaluate_nlg.py (full report text vs. build_reference_reports_v2, bert_score
defaults for lang="en": RoBERTa-large, layer 17, no idf, no rescaling).
Before scoring, two stored rows are reproduced as a check: RATF-v3 (prose template, 0.870) and RATF (0.788).
Output: results/analysis/ratf/bertscore_bench20.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_nlg as nlg  # noqa: E402

BENCH_IDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113, 118, 124, 125, 140, 150, 152, 165, 167, 177, 181]
refs = nlg.build_reference_reports_v2(BENCH_IDS)
SOURCES = {
    "check_RATF-v3 (stored 0.870)": ROOT / "results" / "ratf_reports_v3",
    "check_RATF (stored 0.788)": ROOT / "results" / "ratf_reports_matched",
    "RATF-T": ROOT / "results" / "analysis" / "ratf" / "bench20_T",
    "RATF-R": ROOT / "results" / "analysis" / "ratf" / "bench20_R",
    "RATF-RC": ROOT / "results" / "analysis" / "ratf" / "bench20_RC",
}
out = {}
for name, d in SOURCES.items():
    pids = [p for p in BENCH_IDS if (d / f"report_{p:04d}.txt").exists() and p in refs]
    hyps = [(d / f"report_{p:04d}.txt").read_text(encoding="utf-8", errors="replace") for p in pids]
    f1 = nlg.compute_bertscore_batch(hyps, [refs[p] for p in pids])
    out[name] = {"n": len(pids), "BERTScore_mean": float(np.mean(f1)), "BERTScore_sd": float(np.std(f1, ddof=1))}
    print(name, out[name], flush=True)
json.dump(out, open(ROOT / "results" / "analysis" / "ratf" / "bertscore_bench20.json", "w"), indent=1)
