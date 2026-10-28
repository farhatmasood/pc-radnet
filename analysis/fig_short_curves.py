"""Validation prompted Dice per epoch for the 15-epoch runs and the full-length A2-SDA run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "analysis" / "train_logs"
RUNS = [("A2_s42_short", "A2, seed 42", "#1f77b4", "-"), ("A2_s7_short", "A2, seed 7", "#1f77b4", "--"),
        ("anatOnly_s42_short", "A2-anat", "#d62728", "-"), ("simpleEmb_s42_short", "A2-simple", "#2ca02c", "-"),
        ("segOnly_s42_short", "A2-seg", "#9467bd", "-"), ("noSDA_s42", "A2$-$SDA (full length)", "#7f7f7f", ":")]
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.2, 3.4))
for name, lab, col, ls in RUNS:
    f = LOG / f"{name}.jsonl"
    if not f.exists():
        continue
    v = [json.loads(l)["val_dice"] for l in open(f) if '"epoch"' in l]
    ax.plot(range(1, len(v) + 1), v, ls, color=col, label=lab, lw=1.4)
ax.axvline(5.5, color="k", lw=0.6, ls=":")
ax.axvline(15.5, color="k", lw=0.6, ls="--")
ax.text(15.7, 0.25, "end of 15-epoch run", fontsize=7, rotation=90, va="bottom")
ax.set_xlabel("Epoch (Phase 1: 1-5, Phase 2: 6 onwards)")
ax.set_ylabel("Validation prompted Dice")
ax.set_xlim(1, 40)
ax.set_ylim(0.2, 0.9)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
out = ROOT / "results" / "analysis" / "figures" / "figure_short_curves.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
