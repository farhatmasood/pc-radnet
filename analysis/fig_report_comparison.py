"""Figure 12: same-patient report comparison (test patient 0021).

PC-RadNet panel = slot-checked RATF output (results/analysis/ratf/bench20_RC) plus the
reference-free segmentation confidence (mean probability inside the predicted disc mask,
axial T2w). Other panels reproduce the stored outputs of the benchmarked models; long
outputs are truncated to fit and marked with [...].
"""
import re
import textwrap
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PID = 21
LEV = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}


def ratf_text():
    t = (ROOT / "results" / "analysis" / "ratf" / "bench20_RC" / f"report_{PID:04d}.txt").read_text()
    ac = pd.read_csv(ROOT / "results" / "analysis" / "unified" / "A2_test_allclass.csv")
    s = ac[(ac.patient_id == PID) & (ac.cls_name == "IVD") & ac.fname.str.contains("_T2_")]
    conf = "; ".join(f"{LEV[r.level]} {r.prompted_conf:.2f}" for r in s.itertuples())
    t = t.replace("FINDINGS:", "LUMBAR SPINE MRI REPORT (draft for radiologist review)\nPatient ID: 0021\n\nFINDINGS:")
    t = t.replace("Estimated disc height", "\nEstimated disc height")
    t = t.replace("IMPRESSION:", f"\nDisc segmentation confidence (mean mask probability, no reference mask used): {conf}.\n\nIMPRESSION:")
    t = re.sub(r"(?<=\.) (?=L\d-[LS]\d:)", "\n", t)
    t = re.sub(r"(?<=\.) (?=At L)", "\n", t)
    return t


def read(p):
    return Path(p).read_text(encoding="utf-8", errors="ignore").replace("**", "").replace("#", "").strip()


def wrap(text, width, max_lines):
    lines = []
    for para in text.splitlines():
        lines.extend(textwrap.wrap(para, width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["[...]"]
    return "\n".join(lines)


def main():
    R = ROOT / "results"
    panels = [("PC-RadNet (RATF, revised)", ratf_text(), "#f9d9c9"),
              ("MedGemma-4B-IT", read(R / "medgemma_reports" / f"medgemma_report_{PID:04d}.txt"), "#fbefc4"),
              ("BioGPT-Large", read(R / "report_benchmark/reports/biogpt_large" / f"report_{PID:04d}.txt"), "#dcefd6"),
              ("RADAR", read(R / "report_benchmark/reports/radar" / f"report_{PID:04d}.txt"), "#d6e4f2"),
              ("MAIRA-2", read(R / "report_benchmark/reports/maira2" / f"report_{PID:04d}.txt"), "#f6d6ec")]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16, 7.3))
    boxes = [(0.005, 0.02, 0.25, 0.93), (0.26, 0.02, 0.25, 0.93), (0.515, 0.02, 0.25, 0.93),
             (0.77, 0.36, 0.225, 0.59), (0.77, 0.02, 0.225, 0.31)]
    widths = [58, 58, 58, 52, 52]
    max_lines = [52, 52, 52, 30, 14]
    for (title, txt, col), (x, y, w, h), wd, ml in zip(panels, boxes, widths, max_lines):
        ax = fig.add_axes([x, y, w, h])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.005", fc=col, ec="#888888",
                                    transform=ax.transAxes))
        ax.text(0.5, 1.035, title, ha="center", va="bottom", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.015, 0.985, wrap(txt, wd, ml), ha="left", va="top", fontsize=8.2, transform=ax.transAxes,
                family="DejaVu Sans", linespacing=1.25)
    out = ROOT / "results" / "analysis" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure12_rev.pdf", bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
