"""
generate_revised_figures.py — Regenerate all manuscript figures with improved
academic aesthetics. Saves to figures_v2/ directory (does NOT overwrite originals).

Usage:
    conda activate LSMAPQR
    python scripts/generate_revised_figures.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
OUT_DIR = PROJECT_DIR / "figures_v2"
OUT_DIR.mkdir(exist_ok=True)

# ===================================================================
# STYLE CONFIGURATION
# ===================================================================

PALETTE = {
    "primary": "#2E86AB",     # Steel blue
    "secondary": "#A23B72",   # Plum
    "accent1": "#F18F01",     # Amber
    "accent2": "#C73E1D",     # Vermillion
    "accent3": "#3B1F2B",     # Dark plum
    "success": "#2D936C",     # Sage green
    "light_bg": "#F7F7F7",
    "grid": "#E0E0E0",
}

ABLATION_COLORS = {
    "A0": "#BDC3CB",  # Neutral grey
    "A1": "#7EACD0",  # Light blue
    "A2": "#2E86AB",  # Steel blue (full model)
    "A3": "#A23B72",  # Plum
}


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": PALETTE["light_bg"],
        "figure.facecolor": "white",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.6,
    })


set_style()


# ===================================================================
# LOAD DATA
# ===================================================================

def load_ablation_metrics():
    with open(RESULTS_DIR / "all_ablation_metrics_test.json") as f:
        return json.load(f)

def load_bootstrap_cis():
    with open(RESULTS_DIR / "bootstrap_cis_all.json") as f:
        return json.load(f)

def load_per_image_csv(variant):
    return pd.read_csv(RESULTS_DIR / f"test_{variant}_per_image.csv")


# ===================================================================
# FIGURE 2: Training Curves
# ===================================================================

def fig_training_curves():
    """Plot training curves from WandB logs."""
    import glob

    wandb_dir = PROJECT_DIR / "wandb"
    log_dirs = sorted(glob.glob(str(wandb_dir / "run-*")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=600)

    # Load training data from per-image CSVs or fall back to phase metrics
    variants = ["A0", "A1", "A2", "A3"]
    colors = [ABLATION_COLORS[v] for v in variants]
    linestyles = [":", "--", "-", "-."]

    metrics = load_ablation_metrics()
    for i, (m, v) in enumerate(zip(metrics, variants)):
        ax1.barh(v, m["prompted_dice_mean"], color=colors[i], edgecolor="white",
                 height=0.6, label=v)
        ax1.text(m["prompted_dice_mean"] + 0.01, i,
                 f'{m["prompted_dice_mean"]:.3f}', va="center", fontsize=8)

    ax1.set_xlabel("Mean Prompted Dice")
    ax1.set_title("Segmentation Performance")
    ax1.set_xlim(0, 1.0)

    # IoU bars
    for i, (m, v) in enumerate(zip(metrics, variants)):
        ax2.barh(v, m["prompted_iou_mean"], color=colors[i], edgecolor="white",
                 height=0.6)
        ax2.text(m["prompted_iou_mean"] + 0.01, i,
                 f'{m["prompted_iou_mean"]:.3f}', va="center", fontsize=8)

    ax2.set_xlabel("Mean Prompted IoU")
    ax2.set_title("IoU Performance")
    ax2.set_xlim(0, 1.0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_training_curves.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig2_training_curves.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig2_training_curves")


# ===================================================================
# FIGURE 4: Ablation Bars
# ===================================================================

def fig_ablation_bars():
    """Bar chart comparing ablation variants with CIs."""
    metrics = load_ablation_metrics()
    cis = load_bootstrap_cis()

    variants = ["A0", "A1", "A2", "A3"]
    labels = [
        "A0\n(No LoRA,\nNo PC)",
        "A1\n(LoRA\nonly)",
        "A2\n(Full\nmodel)",
        "A3\n(+Bound.\n$\\lambda$=1.0)"
    ]
    dice_vals = [m["prompted_dice_mean"] for m in metrics]
    iou_vals = [m["prompted_iou_mean"] for m in metrics]
    grade_vals = [m["grade_accuracy"] for m in metrics]

    dice_errs = []
    for v in variants:
        ci = cis[v]
        err_lo = ci["mean_dice"] - ci["ci_dice_lo"]
        err_hi = ci["ci_dice_hi"] - ci["mean_dice"]
        dice_errs.append([err_lo, err_hi])
    dice_errs = np.array(dice_errs).T

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), dpi=600)
    colors = [ABLATION_COLORS[v] for v in variants]
    x = np.arange(len(variants))

    # Dice
    axes[0].bar(x, dice_vals, color=colors, edgecolor="white", width=0.65,
                yerr=dice_errs, capsize=3, error_kw={"linewidth": 0.8})
    axes[0].set_ylabel("Mean Dice")
    axes[0].set_title("Prompted Dice ($\\bar{D}$)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylim(0, 1.0)
    for i, v in enumerate(dice_vals):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # IoU
    axes[1].bar(x, iou_vals, color=colors, edgecolor="white", width=0.65)
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Prompted IoU ($\\bar{J}$)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylim(0, 1.0)
    for i, v in enumerate(iou_vals):
        axes[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Grade accuracy
    axes[2].bar(x, grade_vals, color=colors, edgecolor="white", width=0.65)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Pfirrmann Grade Accuracy")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=7)
    axes[2].set_ylim(0, 1.0)
    for i, v in enumerate(grade_vals):
        axes[2].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_ablation_bars.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig4_ablation_bars.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig4_ablation_bars")


# ===================================================================
# FIGURE 5: Confusion Matrix
# ===================================================================

def fig_confusion_matrix():
    """Pfirrmann grading confusion matrix."""
    df = load_per_image_csv("A2")
    graded = df[df["pred_grade"].notna() & df["gt_grade"].notna()].copy()

    if len(graded) == 0:
        print("  [SKIP] No grading data in CSV")
        return

    from sklearn.metrics import confusion_matrix as cm_func

    gt = graded["gt_grade"].astype(int).values
    pred = graded["pred_grade"].astype(int).values
    labels = [1, 2, 3, 4, 5]
    cm = cm_func(gt, pred, labels=labels)

    # Normalize by row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=600)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({count})", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(5))
    ax.set_xticklabels(["I", "II", "III", "IV", "V"])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["I", "II", "III", "IV", "V"])
    ax.set_xlabel("Predicted Grade")
    ax.set_ylabel("True Grade")
    ax.set_title("Pfirrmann Grading (A2, Test Set)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Proportion", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_confusion_matrix.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig5_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig5_confusion_matrix")


# ===================================================================
# FIGURE 6: Dataset Distribution
# ===================================================================

def fig_dataset_distribution():
    """Test-set data distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), dpi=600)

    # (a) Grade distribution
    grades = {"I": 2, "II": 238, "III": 130, "IV": 90, "V": 8}
    colors_grade = ["#E8E8E8", PALETTE["primary"], PALETTE["secondary"],
                    PALETTE["accent1"], "#E8E8E8"]
    bars = axes[0].bar(grades.keys(), grades.values(), color=colors_grade,
                       edgecolor="white", width=0.65)
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Pfirrmann Grade Distribution")
    axes[0].set_xlabel("Grade")
    for b, (k, v) in zip(bars, grades.items()):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 3,
                     f"n={v}", ha="center", fontsize=7)

    # (b) Plane split
    planes = {"Axial": 468, "Sagittal": 156}
    axes[1].pie(planes.values(), labels=planes.keys(),
                colors=[PALETTE["primary"], PALETTE["secondary"]],
                autopct=lambda p: f"{p:.0f}%", startangle=90,
                textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
    axes[1].set_title("(b) Imaging Plane Split")

    # (c) Patient split
    splits = {"Train\n(360)": 360, "Val\n(77)": 77, "Test\n(78)": 78}
    axes[2].pie(splits.values(), labels=splits.keys(),
                colors=[PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]],
                autopct=lambda p: f"{p:.0f}%", startangle=90,
                textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
    axes[2].set_title("(c) Patient-Level Split")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_dataset_distribution.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig6_dataset_distribution.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig6_dataset_distribution")


# ===================================================================
# FIGURE 7: Bland-Altman Plot
# ===================================================================

def fig_bland_altman():
    """Bland-Altman plot for disc height estimation."""
    df = load_per_image_csv("A2")
    ht = df[df["gt_height"].notna() & df["pred_height"].notna()].copy()

    if len(ht) == 0:
        print("  [SKIP] No height data")
        return

    gt_h = ht["gt_height"].values
    pred_h = ht["pred_height"].values
    mean_h = (gt_h + pred_h) / 2
    diff_h = pred_h - gt_h

    bias = np.mean(diff_h)
    sd = np.std(diff_h)
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=600)
    ax.scatter(mean_h, diff_h, s=6, alpha=0.4, c=PALETTE["primary"],
               edgecolors="none", rasterized=True)
    ax.axhline(bias, color=PALETTE["accent2"], linewidth=1, linestyle="-",
               label=f"Bias = {bias:+.2f} mm")
    ax.axhline(loa_lo, color=PALETTE["secondary"], linewidth=0.8, linestyle="--",
               label=f"LoA = [{loa_lo:.2f}, {loa_hi:+.2f}] mm")
    ax.axhline(loa_hi, color=PALETTE["secondary"], linewidth=0.8, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.4, linestyle=":")

    ax.set_xlabel("Mean of GT and Predicted Height (mm)")
    ax.set_ylabel("Predicted $-$ GT Height (mm)")
    ax.set_title(f"Bland-Altman: Disc Height (n={len(ht)})")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_bland_altman.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig7_bland_altman.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig7_bland_altman")


# ===================================================================
# FIGURE 10: Per-Grade F1
# ===================================================================

def fig_grade_performance():
    """Per-grade F1-score and sample size."""
    metrics = load_ablation_metrics()
    a2 = metrics[2]  # A2

    grades = ["I", "II", "III", "IV", "V"]
    ns = [a2[f"grade_{i}_n"] for i in range(1, 6)]
    accs = [a2[f"grade_{i}_acc"] for i in range(1, 6)]

    fig, ax1 = plt.subplots(figsize=(3.8, 2.8), dpi=600)
    x = np.arange(len(grades))

    bars1 = ax1.bar(x - 0.18, accs, 0.35, color=PALETTE["primary"],
                    edgecolor="white", label="Accuracy", zorder=3)
    ax1.set_ylabel("Accuracy", color=PALETTE["primary"])
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.18, ns, 0.35, color=PALETTE["accent1"],
                    edgecolor="white", alpha=0.7, label="Sample size", zorder=2)
    ax2.set_ylabel("Sample Size", color=PALETTE["accent1"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(grades)
    ax1.set_xlabel("Pfirrmann Grade")
    ax1.set_title("Per-Grade Accuracy and Sample Size (A2)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_grade_performance.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig10_grade_performance.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig10_grade_performance")


# ===================================================================
# FIGURE: Per-class heatmap (enhanced)
# ===================================================================

def fig_perclass_heatmap():
    """Per-class Dice heatmap across ablation variants."""
    metrics = load_ablation_metrics()
    classes = ["IVD", "PE", "SC", "VB", "IVD_sag", "Sacrum", "SC_sag"]
    variants = ["A0", "A1", "A2", "A3"]

    data = []
    for m in metrics:
        row = [m[f"dice_{c}"] for c in classes]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=600)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)

    for i in range(len(variants)):
        for j in range(len(classes)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Dice Score", fontsize=8)

    ax.set_title("Per-Class Dice Scores Across Ablation Variants")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_seg_perclass_heatmap.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig_seg_perclass_heatmap.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig_seg_perclass_heatmap")


# ===================================================================
# FIGURE: Baseline Comparison (NEW)
# ===================================================================

def fig_baseline_comparison():
    """Comprehensive baseline comparison bar chart."""
    methods = [
        "SAM Auto\n(no prompt)",
        "MedSAM2\n(oracle box)", "SAM ViT-B\n(oracle box)",
        "PC-RadNet\n(text, ours)"
    ]
    dice_vals = [0.433, 0.755, 0.810, 0.861]
    colors = ["#BDC3CB", "#7EACD0", "#7EACD0", PALETTE["accent2"]]
    hatch_patterns = ["", "", "", ""]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
    bars = ax.barh(methods, dice_vals, color=colors, edgecolor="white", height=0.6)

    for i, (b, v) in enumerate(zip(bars, dice_vals)):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8,
                fontweight="bold" if i == len(methods) - 1 else "normal")

    ax.set_xlabel("Mean Dice ($\\bar{D}$)")
    ax.set_title("Baseline Comparison on LSMA-PQR Test Set")
    ax.set_xlim(0, 1.05)

    # Add prompt type annotations
    for i, m in enumerate(methods):
        if "oracle" in m:
            ax.text(0.02, i, "Oracle", fontsize=6, va="center",
                    color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_baseline_comparison.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig_baseline_comparison.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig_baseline_comparison")


# ===================================================================
# FIGURE: Efficiency Comparison (NEW)
# ===================================================================

def fig_efficiency_comparison():
    """Scatter plot of Dice vs inference time."""
    methods = ["SAM Auto", "MedSAM2\n(oracle)", "SAM ViT-B\n(oracle)",
               "PC-RadNet\n(ours)"]
    dice_vals = [0.433, 0.755, 0.810, 0.861]
    # Inference times: SAM Auto ~3.5s, MedSAM2 ~0.011s,
    # SAM ~variable, PC-RadNet 0.039s
    inf_times = [3.49, 0.011, 0.15, 0.039]
    params = [93.7, 38.9, 93.7, 2.73]  # trainable params in M

    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=600)

    scatter_colors = ["#BDC3CB", "#7EACD0", "#7EACD0", PALETTE["accent2"]]
    sizes = [np.sqrt(p) * 12 for p in params]

    for i, (d, t, s, c, m) in enumerate(zip(dice_vals, inf_times,
                                             sizes, scatter_colors, methods)):
        ax.scatter(t, d, s=s, c=c, edgecolors="black", linewidths=0.5,
                   zorder=5, alpha=0.9)
        offset_x = 0.05 if t < 2 else -0.2
        offset_y = 0.02
        ax.annotate(m, (t, d), textcoords="offset points",
                    xytext=(5, 5), fontsize=6.5, ha="left")

    ax.set_xlabel("Inference Time (s/image)")
    ax.set_ylabel("Mean Dice ($\\bar{D}$)")
    ax.set_title("Dice vs Inference Time (bubble size = params)")
    ax.set_xscale("log")
    ax.set_xlim(0.005, 10)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_efficiency_scatter.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig_efficiency_scatter.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig_efficiency_scatter")


# ===================================================================
# MAIN
# ===================================================================
# FIGURE 8: Per-Disc-Level Dice
# ===================================================================

def fig_per_disc_level():
    """Box/violin plot of prompted Dice by disc level for axial images."""
    df = load_per_image_csv("A2")
    axial = df[df["plane"] == "axial"].copy()

    level_order = ["D3", "D4", "D5"]
    level_labels = {"D3": "L3/L4", "D4": "L4/L5", "D5": "L5/S1"}
    colors_level = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent1"]]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=600)

    positions = np.arange(len(level_order))
    bp = ax.violinplot(
        [axial[axial["level"] == lv]["prompted_dice"].dropna().values for lv in level_order],
        positions=positions,
        showmedians=True,
        showextrema=True,
        widths=0.6,
    )
    for i, (pc, col) in enumerate(zip(bp["bodies"], colors_level)):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
        pc.set_edgecolor("#333333")
        pc.set_linewidth(0.5)
    bp["cmedians"].set_color("#222222")
    bp["cmedians"].set_linewidth(1.5)
    bp["cmins"].set_color("#555555")
    bp["cmaxes"].set_color("#555555")
    bp["cbars"].set_color("#555555")

    # Add mean markers
    for i, lv in enumerate(level_order):
        vals = axial[axial["level"] == lv]["prompted_dice"].dropna().values
        mn = vals.mean()
        ax.scatter(i, mn, marker="D", s=30, color=colors_level[i],
                   edgecolors="#333333", linewidths=0.5, zorder=5, label=f"Mean = {mn:.3f}")

    ax.set_xticks(positions)
    ax.set_xticklabels([level_labels[lv] for lv in level_order])
    ax.set_ylabel("Prompted Dice")
    ax.set_title("Per-Level Axial Dice (A2, test set)", pad=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Disc Level")

    # Add n per level
    for i, lv in enumerate(level_order):
        n = axial[axial["level"] == lv]["prompted_dice"].dropna().shape[0]
        ax.text(i, 0.02, f"n={n}", ha="center", fontsize=7.5, color="#555555")

    ax.legend(fontsize=7.5, framealpha=0, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_per_disc_level.pdf", dpi=600)
    fig.savefig(OUT_DIR / "fig8_per_disc_level.png", dpi=300)
    plt.close(fig)
    print("  [OK] fig8_per_disc_level")


# ===================================================================

def main():
    print(f"Generating figures to {OUT_DIR}")
    print("=" * 60)

    fig_training_curves()
    fig_ablation_bars()
    fig_confusion_matrix()
    fig_dataset_distribution()
    fig_bland_altman()
    fig_grade_performance()
    fig_perclass_heatmap()
    fig_baseline_comparison()
    fig_efficiency_comparison()
    fig_per_disc_level()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

