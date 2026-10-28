"""Per-source summary for the report tables (n = 20 matched patients)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ratf_analysis as rr

gts = rr.gt_sets()
inputs = rr.level_inputs(rr.BENCH_IDS)
src = rr.stored_reports()
for k in ["T", "R", "RC"]:
    d = rr.OUT / f"bench20_{k}"
    src[f"RATF-{k}"] = {int(f.stem.split("_")[-1]): f.read_text(encoding="utf-8") for f in d.glob("report_*.txt")}
rows = []
for nm, rp in src.items():
    rp = {p: t for p, t in rp.items() if p in rr.BENCH_IDS}
    s = rr.score_source(rp, gts, inputs)
    rows.append({"source": nm, "n": len(s), "P": s.P.mean(), "R": s.R.mean(), "F1": s.F1.mean(),
                 "F1_sd": s.F1.std(), "kept_all_input_findings": s.input_findings_kept.mean(),
                 "unsupported_technique_reports": (s.unsupported_technique > 0).mean(),
                 "reports_with_levels_outside": (s.levels_outside_L3S1 > 0).mean(),
                 "words": s.words.mean(), "words_sd": s.words.std()})
    s.to_csv(rr.OUT / f"perpatient_{nm.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=False)
tab = pd.DataFrame(rows).sort_values("F1")
tab.to_csv(rr.OUT / "bench20_table.csv", index=False)
print(tab.round(3).to_string())
# paired bootstrap over patients: RATF-T minus Phi-3.5-mini
a = pd.read_csv(rr.OUT / "perpatient_RATF-T.csv").set_index("patient_id").F1
b = pd.read_csv(rr.OUT / "perpatient_Phi-3.5-mini.csv").set_index("patient_id").F1
d = (a - b).dropna().values
rng = np.random.default_rng(0)
bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
print("RATF-T minus Phi F1:", d.mean(), np.percentile(bs, [2.5, 97.5]))
