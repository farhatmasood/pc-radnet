"""Figure 11: disc-height agreement of the A2 height head, T2w axial test images (n = 234)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
c = pd.read_csv(ROOT / "results" / "analysis" / "unified" / "A2_test_clinical.csv")
h = c[(c.gt_height > 0) & c.fname.str.contains("axial_T2_")]
x, y = h.gt_height.values, h.pred_height.values
st = json.load(open(ROOT / "results" / "analysis" / "stats_core.json"))["A2"]["height"]
m = st["model"]
d = y - x
mean = (x + y) / 2
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(mean, d, s=9, alpha=0.55, color="#4878a8")
ax[0].axhline(m["bias"], color="k", lw=1.2, label=f"Bias = {m['bias']:+.2f} mm")
ax[0].axhline(m["loa_hi"], color="#c0392b", ls="--", lw=1, label=f"95% LoA = [{m['loa_lo']:.2f}, {m['loa_hi']:+.2f}] mm")
ax[0].axhline(m["loa_lo"], color="#c0392b", ls="--", lw=1)
ax[0].set_xlabel("Mean of reference and predicted height (mm)")
ax[0].set_ylabel("Predicted - reference (mm)")
ax[0].set_title("(A) Bland-Altman agreement")
ax[0].legend(fontsize=7.5, loc="lower left")
ax[0].text(0.98, 0.97, f"n = {len(h)} discs (78 patients)\nMAE = {m['mae']:.2f} mm", transform=ax[0].transAxes,
           ha="right", va="top", fontsize=7.5)
lim = [0, max(x.max(), y.max()) + 1]
ax[1].scatter(x, y, s=9, alpha=0.55, color="#3c8d5a")
ax[1].plot(lim, lim, "k--", lw=1, label="Identity")
k, b = np.polyfit(x, y, 1)
ax[1].plot(lim, [k * lim[0] + b, k * lim[1] + b], color="#e67e22", lw=1.4, label="Least-squares fit")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("Reference height, mid-sagittal T2w (mm)")
ax[1].set_ylabel("Predicted height, axial T2w (mm)")
ax[1].set_title(f"(B) r = {m['r']:.2f}, CCC = {m['ccc']:.2f}, ICC(A,1) = {m['icc_a1']:.2f}")
ax[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
out = ROOT / "results" / "analysis" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "figure11_rev.pdf", bbox_inches="tight")
print("ok", m)
