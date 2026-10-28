"""
visualize.py - Generate all figures for the manuscript.
Sections 10-11: qualitative results, training curves, ablation plots.

Usage:
    python scripts/visualize.py --all
    python scripts/visualize.py --training-curves
    python scripts/visualize.py --qualitative --checkpoint best_by_mdice_A2.pt
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib
from repo_config import load_config
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CFG = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = Path(CFG["paths"]["figures"])
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for segmentation overlays
AXIAL_COLORS = [(0, 0, 0, 0), (0.2, 0.6, 1.0, 0.5), (1.0, 0.4, 0.2, 0.5), (0.2, 1.0, 0.4, 0.5)]
SAG_COLORS = [(0, 0, 0, 0), (1.0, 0.8, 0.2, 0.5), (0.2, 0.6, 1.0, 0.5),
              (0.8, 0.2, 0.8, 0.5), (0.2, 1.0, 0.4, 0.5)]


def set_style():
    """Set consistent matplotlib style for publication."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Architecture Diagram (placeholder - real diagram in LaTeX/TikZ)
# ═══════════════════════════════════════════════════════════════════

def fig_architecture():
    """Generate simplified architecture block diagram."""
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    modules = [
        ("DINOv2\n(Frozen)", 0.05, 0.3, 0.15, 0.4, "#4A90D9"),
        ("Spine\nAdapter", 0.22, 0.3, 0.12, 0.4, "#67B7DC"),
        ("Prompt\nCalibrator", 0.36, 0.3, 0.14, 0.4, "#E74C3C"),
        ("Seg\nDecoder", 0.52, 0.55, 0.12, 0.35, "#2ECC71"),
        ("Grade\nHead", 0.52, 0.15, 0.12, 0.35, "#F39C12"),
        ("Height\nHead", 0.66, 0.55, 0.12, 0.35, "#9B59B6"),
        ("Disorder\nHead", 0.66, 0.15, 0.12, 0.35, "#1ABC9C"),
        ("Report\nGenerator", 0.82, 0.3, 0.14, 0.4, "#34495E"),
    ]

    for label, x, y, w, h, color in modules:
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor="black", facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    # Arrows
    arrow_kw = dict(arrowstyle="->", color="black", lw=1.5)
    from matplotlib.patches import FancyArrowPatch
    arrows = [
        ((0.20, 0.5), (0.22, 0.5)),
        ((0.34, 0.5), (0.36, 0.5)),
        ((0.50, 0.6), (0.52, 0.6)),
        ((0.50, 0.4), (0.52, 0.4)),
        ((0.64, 0.7), (0.66, 0.7)),
        ((0.64, 0.3), (0.66, 0.3)),
        ((0.78, 0.5), (0.82, 0.5)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    # Input/Output labels
    ax.text(0.02, 0.5, "MRI\nImage", ha="center", va="center", fontsize=9)
    ax.text(0.98, 0.5, "Report", ha="center", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PCRadNet Architecture", fontsize=14, fontweight="bold", pad=10)

    plt.savefig(FIG_DIR / "fig1_architecture.pdf")
    plt.savefig(FIG_DIR / "fig1_architecture.png")
    plt.close()
    print("Saved fig1_architecture.pdf/png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Training Curves
# ═══════════════════════════════════════════════════════════════════

def fig_training_curves():
    """Plot training curves from WandB logs or checkpoint files."""
    set_style()
    ckpt_dir = Path(CFG["paths"]["checkpoints"])

    # Collect metrics from phase checkpoints
    ablations = ["A0", "A1", "A2", "A3"]
    colors = {"A0": "#E74C3C", "A1": "#F39C12", "A2": "#2ECC71", "A3": "#4A90D9"}
    labels = {
        "A0": "A0: No LoRA, No Calibrator",
        "A1": "A1: LoRA only",
        "A2": "A2: LoRA + Calibrator",
        "A3": "A3: Full (+ boundary loss)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for abl in ablations:
        epochs, train_dice, val_dice, val_iou, val_loss = [], [], [], [], []
        for ep in range(1, 100):
            # Check both phases
            for phase in [1, 2]:
                ckpt_path = ckpt_dir / f"phase{phase}_{abl}_epoch_{ep:02d}.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    epochs.append(ckpt["epoch"] + (0 if phase == 1 else 5))
                    train_dice.append(ckpt.get("train_dice", 0))
                    val_dice.append(ckpt.get("val_dice", 0))
                    val_iou.append(ckpt.get("val_iou", 0))

        if not epochs:
            continue

        c = colors.get(abl, "gray")
        lbl = labels.get(abl, abl)

        axes[0, 0].plot(epochs, train_dice, color=c, label=lbl, alpha=0.8)
        axes[0, 1].plot(epochs, val_dice, color=c, label=lbl, alpha=0.8)
        axes[1, 0].plot(epochs, val_iou, color=c, label=lbl, alpha=0.8)

    axes[0, 0].set_title("Train Dice")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Dice")
    axes[0, 0].legend()

    axes[0, 1].set_title("Val Dice")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()

    axes[1, 0].set_title("Val IoU")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].legend()

    # VRAM usage (from last checkpoint)
    axes[1, 1].set_title("GPU Memory Usage")
    for abl in ablations:
        for phase in [2, 1]:
            best_path = ckpt_dir / f"best_by_mdice_{abl}.pt"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                vram = ckpt.get("vram_peak_gb", 0)
                params = ckpt.get("trainable_params", 0)
                axes[1, 1].bar(labels.get(abl, abl)[:6], vram,
                               color=colors.get(abl, "gray"), alpha=0.8)
                break
    axes[1, 1].set_ylabel("Peak VRAM (GB)")
    axes[1, 1].axhline(y=6.87, color="red", linestyle="--", label="Budget")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_training_curves.pdf")
    plt.savefig(FIG_DIR / "fig2_training_curves.png")
    plt.close()
    print("Saved fig2_training_curves.pdf/png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Qualitative Segmentation Results
# ═══════════════════════════════════════════════════════════════════

def fig_qualitative(checkpoint_path=None, n_samples=6):
    """Generate qualitative segmentation results grid."""
    set_style()
    from torch.amp import autocast

    if checkpoint_path is None:
        ckpt_dir = Path(CFG["paths"]["checkpoints"])
        checkpoint_path = ckpt_dir / "best_by_mdice_A2.pt"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}. Skipping qualitative figure.")
        return

    from train import PCRadNet, SpineDataset, get_val_transforms, collate_fn, compute_dice, compute_iou

    model = PCRadNet(CFG, use_calibrator=True, lora_rank=CFG["model"]["lora_rank"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    split_file = Path(CFG["paths"]["splits_dir"]) / "test.txt"
    ds = SpineDataset(str(split_file), CFG, transforms=get_val_transforms())

    # Select diverse samples (mix axial/sagittal, different levels)
    indices = list(range(min(n_samples * 5, len(ds))))
    np.random.shuffle(indices)
    selected = indices[:n_samples]

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    from torch.utils.data import Subset

    for row_idx, ds_idx in enumerate(selected):
        sample = ds[ds_idx]
        img = sample["image"].unsqueeze(0).to(DEVICE)
        plane_id = torch.tensor([sample["plane_id"]], device=DEVICE)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(img, [sample["prompt_text"]], plane_id)

        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_vis = (sample["image"] * std + mean).clamp(0, 1).permute(1, 2, 0).float().numpy()

        gt_mask = sample["mask"].float().numpy()
        pred_prompted = torch.sigmoid(out["prompted_logits"][0, 0]).float().cpu().numpy()
        gt_prompted = sample["prompted_mask"].float().numpy()

        # Column 0: Input image
        axes[row_idx, 0].imshow(img_vis)
        axes[row_idx, 0].set_title(f"{sample['fname']}\nPrompt: '{sample['prompt_text']}'", fontsize=8)
        axes[row_idx, 0].axis("off")

        # Column 1: Ground truth mask
        cmap = ListedColormap(AXIAL_COLORS if sample["plane_id"] == 0 else SAG_COLORS)
        axes[row_idx, 1].imshow(img_vis)
        axes[row_idx, 1].imshow(gt_mask, cmap=cmap, alpha=0.5,
                                vmin=0, vmax=3 if sample["plane_id"] == 0 else 4)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=9)
        axes[row_idx, 1].axis("off")

        # Column 2: Predicted prompted mask
        axes[row_idx, 2].imshow(img_vis)
        axes[row_idx, 2].imshow(pred_prompted, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        dice_val = compute_dice(
            torch.from_numpy(pred_prompted),
            torch.from_numpy(gt_prompted)
        ).item()
        axes[row_idx, 2].set_title(f"Predicted (Dice={dice_val:.3f})", fontsize=9)
        axes[row_idx, 2].axis("off")

        # Column 3: GT prompted mask
        axes[row_idx, 3].imshow(img_vis)
        axes[row_idx, 3].imshow(gt_prompted, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[row_idx, 3].set_title("GT Prompted Mask", fontsize=9)
        axes[row_idx, 3].axis("off")

    plt.suptitle("PCRadNet Qualitative Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_qualitative.pdf")
    plt.savefig(FIG_DIR / "fig3_qualitative.png")
    plt.close()
    print("Saved fig3_qualitative.pdf/png")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Ablation Bar Chart
# ═══════════════════════════════════════════════════════════════════

def fig_ablation_bars():
    """Bar chart comparing ablation variants."""
    set_style()
    results_dir = Path(CFG["paths"]["results"])

    # Try loading ablation metrics
    metrics_file = results_dir / "all_ablation_metrics_test.json"
    if not metrics_file.exists():
        metrics_file = results_dir / "all_ablation_metrics_val.json"
    if not metrics_file.exists():
        print("No ablation metrics found. Skipping ablation bar chart.")
        return

    with open(metrics_file) as f:
        all_metrics = json.load(f)

    ablations = [m["ablation"] for m in all_metrics]
    dices = [m.get("prompted_dice_mean", 0) for m in all_metrics]
    ious = [m.get("prompted_iou_mean", 0) for m in all_metrics]
    grade_accs = [m.get("grade_accuracy", 0) for m in all_metrics]

    x = np.arange(len(ablations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, dices, width, label="mDice", color="#2ECC71", alpha=0.8)
    ax.bar(x, ious, width, label="mIoU", color="#4A90D9", alpha=0.8)
    ax.bar(x + width, grade_accs, width, label="Grade Acc", color="#F39C12", alpha=0.8)

    ax.set_xlabel("Ablation Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x)
    ax.set_xticklabels(ablations)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_ablation_bars.pdf")
    plt.savefig(FIG_DIR / "fig4_ablation_bars.png")
    plt.close()
    print("Saved fig4_ablation_bars.pdf/png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Confusion Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════

def fig_confusion_matrix():
    """Heatmap of Pfirrmann grading confusion matrix."""
    set_style()
    results_dir = Path(CFG["paths"]["results"])
    cm_path = results_dir / "confusion_matrix_grading.csv"
    if not cm_path.exists():
        print("No confusion matrix found. Skipping.")
        return

    cm = np.loadtxt(cm_path, delimiter=",", skiprows=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel="True Grade", xlabel="Predicted Grade",
           title="Pfirrmann Grading Confusion Matrix")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_confusion_matrix.pdf")
    plt.savefig(FIG_DIR / "fig5_confusion_matrix.png")
    plt.close()
    print("Saved fig5_confusion_matrix.pdf/png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--architecture", action="store_true")
    parser.add_argument("--training-curves", action="store_true")
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--ablation-bars", action="store_true")
    parser.add_argument("--confusion-matrix", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.all or args.architecture:
        fig_architecture()
    if args.all or args.training_curves:
        fig_training_curves()
    if args.all or args.qualitative:
        fig_qualitative(args.checkpoint)
    if args.all or args.ablation_bars:
        fig_ablation_bars()
    if args.all or args.confusion_matrix:
        fig_confusion_matrix()


if __name__ == "__main__":
    main()


def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)
