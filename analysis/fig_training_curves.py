"""Figure 4 (training curves) rebuilt from the logged runs of the evaluated A0-A3 checkpoints.
A0, A1, A3: per-epoch lines of wandb/run-*/files/output.log. A2: offline W&B history (5 March run,
best validation Dice 0.856 at phase-2 epoch 49, the checkpoint used for every reported A2 result).
Overall epoch = phase-1 epoch (1-5) or 5 + phase-2 epoch.
Output: results/analysis/figures/figure4_rev.pdf and results/analysis/training_curves.json"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "wandb"
PAT = re.compile(r"\[Epoch (\d+)/(\d+)\].*?train_dice=([\d.]+) \|.*?val_dice=([\d.]+) val_iou=([\d.]+).*?gpu_peak=([\d.]+)GB")
RUNS = {
    "A0": [("log", "run-20260306_154434-aiotua7o"), ("log", "run-20260306_155340-vpdiumux")],
    "A1": [("log", "run-20260306_190535-19ewmube"), ("log", "run-20260306_200244-unb2a3xv")],
    "A2": [("wb", "offline-run-20260305_085156-t8zeu06i/run-t8zeu06i.wandb"),
           ("wb", "offline-run-20260305_092850-jq4nmyxr/run-jq4nmyxr.wandb")],
    "A3": [("log", "run-20260306_214348-4r2jc3m2"), ("log", "run-20260306_221021-7jkceu79")],
}
COLORS = {"A0": "#7f7f7f", "A1": "#2f6fad", "A2": "#2ca02c", "A3": "#d62728"}


def from_log(run):
    rows = {}
    for line in open(W / run / "files" / "output.log", encoding="utf-8", errors="ignore"):
        m = PAT.search(line)
        if m:
            rows[int(m[1])] = dict(tr=float(m[3]), vd=float(m[4]), vi=float(m[5]), gpu=float(m[6]))
    return [dict(ep=k, **v) for k, v in sorted(rows.items())]


def from_wandb(path):
    ds = datastore.DataStore()
    ds.open_for_scan(str(W / path))
    rows = {}
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") == "history":
            h = {(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in r.history.item}
            if "val/dice" in h:
                rows[int(h["epoch"])] = dict(tr=h["train/dice"], vd=h["val/dice"], vi=h["val/iou"],
                                             gpu=h["gpu_mem_peak_gb"])
    return [dict(ep=k, **v) for k, v in sorted(rows.items())]


curves = {}
for abl, parts in RUNS.items():
    seq = []
    for ph, (kind, src) in enumerate(parts):
        rows = from_log(src) if kind == "log" else from_wandb(src)
        for r in rows:
            r["ep"] += 5 * ph
        seq += rows
    curves[abl] = seq
json.dump(curves, open(ROOT / "results" / "analysis" / "training_curves.json", "w"), indent=0)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
g = a2.twinx()
for abl, seq in curves.items():
    ep = [r["ep"] for r in seq]
    c = COLORS[abl]
    a1.plot(ep, [r["tr"] for r in seq], "--", color=c, lw=1.2, alpha=0.6)
    a1.plot(ep, [r["vd"] for r in seq], "-", color=c, lw=1.8, label=abl)
    a2.plot(ep, [r["vi"] for r in seq], "-", color=c, lw=1.8)
    g.scatter(ep, [r["gpu"] for r in seq], s=6, color="0.6", alpha=0.5)
    b = max(seq, key=lambda r: r["vd"])
    a1.scatter(b["ep"], b["vd"], marker="*", s=170, color=c, edgecolor="k", zorder=5)
    a2.scatter(b["ep"], b["vi"], marker="*", s=170, color=c, edgecolor="k", zorder=5)
    if ep[-1] < 55:  # early stopping
        a1.scatter(ep[-1], seq[-1]["vd"], marker="D", s=45, facecolor="white", edgecolor=c, lw=1.5, zorder=5)
        a2.scatter(ep[-1], seq[-1]["vi"], marker="D", s=45, facecolor="white", edgecolor=c, lw=1.5, zorder=5)
for ax, t, yl in [(a1, "(A) Prompted Dice: train (--), validation (—)", "Dice"),
                  (a2, "(B) Validation prompted IoU", "IoU")]:
    ax.axvline(5, ls=":", color="0.5", lw=1)
    ax.text(5.6, 0.90, "warmup\nend", fontsize=8, color="0.4")
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(yl)
    ax.set_title(t, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
g.set_ylim(0, 10)
g.axhline(8, ls="--", color="#d62728", lw=1)
g.text(41, 8.15, "8 GB budget", color="#d62728", fontsize=8)
g.set_ylabel("Peak GPU memory (GB)", color="0.45")
g.spines["right"].set_visible(True)
h, l = a1.get_legend_handles_labels()
h += [plt.Line2D([], [], marker="*", ls="", ms=11, color="0.3"),
      plt.Line2D([], [], marker="D", ls="", ms=6, mfc="white", color="0.3")]
l += ["best validation", "early-stopped"]
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(ROOT / "results" / "analysis" / "figures" / "figure4_rev.pdf")
fig.savefig(ROOT / "results" / "analysis" / "figures" / "figure4_rev.png", dpi=110)
for abl, seq in curves.items():
    b = max(seq, key=lambda r: r["vd"])
    print(abl, "epochs", len(seq), "last", seq[-1]["ep"], "best", b, "vd range after warmup",
          round(min(r["vd"] for r in seq if r["ep"] > 5), 3), round(max(r["vd"] for r in seq), 3),
          "vi range", round(min(r["vi"] for r in seq if r["ep"] > 5), 3), round(max(r["vi"] for r in seq), 3))
