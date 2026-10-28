"""
evaluate.py - Full test-set evaluation for PCRadNet
Sections 10-11: per-class metrics, confusion matrices, statistical tests, LaTeX tables.

Usage:
    python scripts/evaluate.py --checkpoint best_by_mdice_A2.pt
    python scripts/evaluate.py --checkpoint best_by_mdice_A2.pt --split test
"""
import os, sys, json, argparse, gc, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

import yaml
from PIL import Image as PILImage

# ── Ensure scripts/ is importable ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import (
    PCRadNet, SpineDataset, get_val_transforms, collate_fn,
    compute_dice, compute_iou, PROMPT_BANK, SEED, DEVICE, CFG,
)

# Seed
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

AXIAL_NAMES = {0: "BG", 1: "IVD", 2: "PE", 3: "SC"}
SAG_NAMES = {0: "BG", 1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"}


# ═══════════════════════════════════════════════════════════════════
# PER-IMAGE EVALUATION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model, dataloader, device, save_dir, split="test"):
    """Comprehensive per-image evaluation with all metrics."""
    model.eval()
    results = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        plane_ids = batch["plane_id"].to(device)
        fnames = batch["fname"]

        for k in ["mask", "mask_onehot", "prompted_mask", "grade", "height", "disorder"]:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images, batch["prompt_text"], plane_ids)

        B = images.shape[0]
        for i in range(B):
            row = {
                "fname": fnames[i],
                "patient_id": batch["patient_id"][i].item(),
                "level": batch["level"][i],
                "plane": "axial" if plane_ids[i] == 0 else "sagittal",
                "prompt": batch["prompt_text"][i],
                "target_cls": batch["target_cls"][i].item(),
            }

            # Prompted mask metrics
            pred_prompted = torch.sigmoid(outputs["prompted_logits"][i, 0])
            gt_prompted = batch["prompted_mask"][i]
            row["prompted_dice"] = compute_dice(pred_prompted, gt_prompted).item()
            row["prompted_iou"] = compute_iou(pred_prompted, gt_prompted).item()

            # Per-class segmentation metrics
            seg_logits = outputs["seg_logits_list"][i]  # [1, C, H, W]
            num_cls = seg_logits.shape[1]
            class_names = AXIAL_NAMES if plane_ids[i] == 0 else SAG_NAMES

            for c in range(num_cls):
                pred_c = torch.sigmoid(seg_logits[0, c])
                gt_c = batch["mask_onehot"][i, c]
                d = compute_dice(pred_c, gt_c).item()
                iou = compute_iou(pred_c, gt_c).item()
                name = class_names.get(c, f"cls_{c}")
                row[f"dice_{name}"] = d
                row[f"iou_{name}"] = iou

            # Mean foreground dice (skip background)
            fg_dices = [row.get(f"dice_{class_names[c]}", 0) for c in range(1, num_cls)]
            row["mean_fg_dice"] = np.mean(fg_dices) if fg_dices else 0

            # Grading
            if batch["grade"][i].item() > 0:
                probs = torch.sigmoid(outputs["grade_logits"][i])
                pred_grade = int((probs > 0.5).sum().item()) + 1
                gt_grade = batch["grade"][i].item()
                row["gt_grade"] = gt_grade
                row["pred_grade"] = pred_grade
                row["grade_correct"] = int(pred_grade == gt_grade)
                row["grade_error"] = abs(pred_grade - gt_grade)
            else:
                row["gt_grade"] = -1
                row["pred_grade"] = -1

            # Height
            if batch["height"][i].item() > 0:
                pred_ht = outputs["height_pred"][i].item()
                gt_ht = batch["height"][i].item()
                row["gt_height"] = gt_ht
                row["pred_height"] = pred_ht
                row["height_ae"] = abs(pred_ht - gt_ht)
            else:
                row["gt_height"] = -1
                row["pred_height"] = -1

            # Disorders
            disorder_probs = torch.sigmoid(outputs["disorder_logits"][i])
            for j, dname in enumerate([
                "Herniation", "Protrusion", "Bulge", "ThecalComp",
                "NerveComp", "Stenosis", "Spondylolisthesis",
                "Desiccated", "LigFlavum"
            ]):
                row[f"disorder_{dname}_prob"] = disorder_probs[j].item()
                row[f"disorder_{dname}_gt"] = batch["disorder"][i, j].item()

            results.append(row)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {(batch_idx+1)*B}/{len(dataloader.dataset)} images")

    df = pd.DataFrame(results)
    df.to_csv(save_dir / f"{split}_per_image.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════
# AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_aggregate_metrics(df, split="test"):
    """Compute aggregate metrics from per-image results."""
    metrics = {"split": split, "n_images": len(df)}

    # Segmentation
    metrics["prompted_dice_mean"] = df["prompted_dice"].mean()
    metrics["prompted_dice_std"] = df["prompted_dice"].std()
    metrics["prompted_iou_mean"] = df["prompted_iou"].mean()
    metrics["prompted_iou_std"] = df["prompted_iou"].std()
    metrics["mean_fg_dice"] = df["mean_fg_dice"].mean()

    # Per-plane metrics
    for plane in ["axial", "sagittal"]:
        sub = df[df["plane"] == plane]
        if len(sub) > 0:
            metrics[f"{plane}_prompted_dice"] = sub["prompted_dice"].mean()
            metrics[f"{plane}_mean_fg_dice"] = sub["mean_fg_dice"].mean()

    # Per-class dice (aggregate)
    class_cols = [c for c in df.columns if c.startswith("dice_") and c != "dice_BG"]
    for col in class_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            metrics[col] = valid.mean()

    # Grading
    graded = df[df["gt_grade"] > 0]
    if len(graded) > 0:
        metrics["grade_accuracy"] = graded["grade_correct"].mean()
        metrics["grade_mae"] = graded["grade_error"].mean()
        metrics["grade_within1"] = (graded["grade_error"] <= 1).mean()
        metrics["n_graded"] = len(graded)

        # Per-grade accuracy
        for g in range(1, 6):
            g_sub = graded[graded["gt_grade"] == g]
            if len(g_sub) > 0:
                metrics[f"grade_{g}_acc"] = g_sub["grade_correct"].mean()
                metrics[f"grade_{g}_n"] = len(g_sub)

    # Height
    ht_valid = df[df["gt_height"] > 0]
    if len(ht_valid) > 0:
        metrics["height_mae"] = ht_valid["height_ae"].mean()
        metrics["height_pearson_r"] = ht_valid[["gt_height", "pred_height"]].corr().iloc[0, 1]
        metrics["n_height"] = len(ht_valid)

    # Disorders
    disorder_names = ["Herniation", "Protrusion", "Bulge", "ThecalComp",
                      "NerveComp", "Stenosis", "Spondylolisthesis",
                      "Desiccated", "LigFlavum"]
    for dname in disorder_names:
        gt_col = f"disorder_{dname}_gt"
        prob_col = f"disorder_{dname}_prob"
        if gt_col in df.columns:
            gt = df[gt_col].values
            prob = df[prob_col].values
            pred = (prob > 0.5).astype(int)
            metrics[f"disorder_{dname}_acc"] = (pred == gt).mean()
            if gt.sum() > 0:
                tp = ((pred == 1) & (gt == 1)).sum()
                fp = ((pred == 1) & (gt == 0)).sum()
                fn = ((pred == 0) & (gt == 1)).sum()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                metrics[f"disorder_{dname}_prec"] = precision
                metrics[f"disorder_{dname}_rec"] = recall

    return metrics


# ═══════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════

def build_confusion_matrices(df, save_dir):
    """Build and save confusion matrices for grading."""
    graded = df[df["gt_grade"] > 0]
    if len(graded) == 0:
        return

    from sklearn.metrics import confusion_matrix, classification_report
    y_true = graded["gt_grade"].values
    y_pred = graded["pred_grade"].values

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5],
                                    target_names=["Grade 1", "Grade 2", "Grade 3",
                                                    "Grade 4", "Grade 5"],
                                    output_dict=True)

    np.savetxt(save_dir / "confusion_matrix_grading.csv", cm, delimiter=",",
               header="G1,G2,G3,G4,G5", fmt="%d")

    with open(save_dir / "classification_report_grading.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nPfirrmann Grading Confusion Matrix:")
    print(f"{'':>8} {'G1':>5} {'G2':>5} {'G3':>5} {'G4':>5} {'G5':>5}")
    for i in range(5):
        row_str = f"True G{i+1}: " + " ".join(f"{cm[i,j]:5d}" for j in range(5))
        print(row_str)


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════

def statistical_tests(results_dict, save_dir):
    """
    Paired Wilcoxon signed-rank tests between ablations.
    results_dict: {ablation_name: per_image_df}
    """
    from scipy.stats import wilcoxon

    ablation_names = sorted(results_dict.keys())
    if len(ablation_names) < 2:
        print("Need >=2 ablations for statistical tests.")
        return

    tests = []
    reference = ablation_names[-1]  # Full model (A2 or A3)
    ref_df = results_dict[reference]

    for abl in ablation_names:
        if abl == reference:
            continue
        abl_df = results_dict[abl]

        # Align by fname
        merged = ref_df[["fname", "prompted_dice"]].merge(
            abl_df[["fname", "prompted_dice"]], on="fname", suffixes=("_ref", "_abl")
        )
        if len(merged) < 10:
            continue

        stat, p = wilcoxon(merged["prompted_dice_ref"], merged["prompted_dice_abl"])
        diff = (merged["prompted_dice_ref"] - merged["prompted_dice_abl"]).mean()
        tests.append({
            "reference": reference,
            "compared": abl,
            "metric": "prompted_dice",
            "ref_mean": merged["prompted_dice_ref"].mean(),
            "abl_mean": merged["prompted_dice_abl"].mean(),
            "mean_diff": diff,
            "wilcoxon_stat": stat,
            "p_value": p,
            "significant_005": p < 0.05,
        })

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(save_dir / "statistical_tests.csv", index=False)
    print("\nStatistical Tests:")
    print(tests_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# LaTeX TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_latex_tables(metrics_list, save_dir):
    """Generate LaTeX tables for the manuscript."""

    # Table 1: Ablation Study
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study results on the test set. mDice and mIoU are computed over foreground classes.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Variant & Params$\downarrow$ & mDice$\uparrow$ & mIoU$\uparrow$ & Grade Acc.$\uparrow$ & Ht.~MAE$\downarrow$ & VRAM (GB)$\downarrow$ \\",
        r"\midrule",
    ]

    for m in metrics_list:
        name = m.get("ablation", "—")
        params = m.get("trainable_params", 0)
        dice = m.get("prompted_dice_mean", 0)
        iou = m.get("prompted_iou_mean", 0)
        gacc = m.get("grade_accuracy", 0)
        hmae = m.get("height_mae", 0)
        vram = m.get("vram_gb", "—")

        # Bold best
        line = (f"{name} & {params/1e6:.2f}M & {dice:.4f} & {iou:.4f} "
                f"& {gacc:.3f} & {hmae:.2f} & {vram} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(save_dir / "table_ablation.tex", "w") as f:
        f.write("\n".join(lines))

    # Table 2: Per-class segmentation
    lines2 = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-class Dice scores on the test set.}",
        r"\label{tab:perclass}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& IVD & PE & SC & VB & IVD$_\text{sag}$ & Sacrum & SC$_\text{sag}$ & mDice \\",
        r"\midrule",
    ]

    for m in metrics_list:
        name = m.get("ablation", "—")
        vals = []
        for cls in ["IVD", "PE", "SC", "VB", "IVD_sag", "Sacrum", "SC_sag"]:
            v = m.get(f"dice_{cls}", 0)
            vals.append(f"{v:.4f}" if v > 0 else "—")
        mdice = m.get("mean_fg_dice", 0)
        line = f"{name} & " + " & ".join(vals) + f" & {mdice:.4f} \\\\"
        lines2.append(line)

    lines2.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(save_dir / "table_perclass.tex", "w") as f:
        f.write("\n".join(lines2))

    print(f"LaTeX tables saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default="best_by_mdice_A2.pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--ablation", type=str, default="A2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--all-ablations", action="store_true",
                        help="Evaluate ALL ablation checkpoints for statistical comparison")
    args = parser.parse_args()

    ckpt_dir = Path(CFG["paths"]["checkpoints"])
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(CFG["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloader
    split_file = Path(CFG["paths"]["splits_dir"]) / f"{args.split}.txt"
    ds = SpineDataset(split_file, CFG, transforms=get_val_transforms())
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=CFG["hardware"]["num_workers"],
        pin_memory=CFG["hardware"]["pin_memory"],
        collate_fn=collate_fn,
    )
    print(f"Evaluating on {args.split}: {len(ds)} images")

    if args.all_ablations:
        # Evaluate all ablation checkpoints
        all_metrics = []
        all_per_image = {}
        for abl in ["A0", "A1", "A2", "A3"]:
            ckpt_path = ckpt_dir / f"best_by_mdice_{abl}.pt"
            if not ckpt_path.exists():
                print(f"Skipping {abl}: checkpoint not found")
                continue

            use_cal = abl in ["A2", "A3"]
            lora_r = CFG["model"]["lora_rank"] if abl != "A0" else 0
            model = PCRadNet(CFG, use_calibrator=use_cal, lora_rank=lora_r)

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(DEVICE)

            df = evaluate_full(model, loader, DEVICE,
                               results_dir, split=f"{args.split}_{abl}")

            metrics = compute_aggregate_metrics(df, split=f"{args.split}_{abl}")
            metrics["ablation"] = abl
            metrics["trainable_params"] = ckpt.get("trainable_params", 0)
            metrics["vram_gb"] = ckpt.get("vram_peak_gb", "—")
            all_metrics.append(metrics)
            all_per_image[abl] = df

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Statistical tests
        if len(all_per_image) >= 2:
            statistical_tests(all_per_image, results_dir)

        # LaTeX tables
        generate_latex_tables(all_metrics, results_dir)

        # Save all metrics
        with open(results_dir / f"all_ablation_metrics_{args.split}.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

    else:
        # Single checkpoint
        ckpt_path = ckpt_dir / args.checkpoint
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        use_cal = args.ablation in ["A2", "A3"]
        lora_r = CFG["model"]["lora_rank"] if args.ablation != "A0" else 0
        model = PCRadNet(CFG, use_calibrator=use_cal, lora_rank=lora_r)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
              f"val_dice={ckpt.get('val_dice', '?')})")

        df = evaluate_full(model, loader, DEVICE, results_dir, split=args.split)
        metrics = compute_aggregate_metrics(df, split=args.split)
        build_confusion_matrices(df, results_dir)

        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS — {args.split} set ({args.ablation})")
        print(f"{'='*60}")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")

        with open(results_dir / f"{args.split}_{args.ablation}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
