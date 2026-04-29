"""
evaluate_external_v2.py — Corrected cross-dataset evaluation of PC-RadNet

Key fixes over v1:
  1. Aspect-ratio-preserving resize (pad to square, then resize to 384×384)
  2. Only evaluate slices with meaningful foreground (>1% pixels)
  3. Per-class prompting: each class evaluated with its own text prompt
  4. ROI cropping: bounding-box crop around foreground before resize
  5. Multiple random prompts averaged for robust grading/disorder estimation
  6. Qualitative visualisation output

Usage:
    python scripts/evaluate_external_v2.py --dataset spider
    python scripts/evaluate_external_v2.py --dataset spine_new
    python scripts/evaluate_external_v2.py --dataset rsna
    python scripts/evaluate_external_v2.py --dataset xray
    python scripts/evaluate_external_v2.py --dataset all
"""
import os, sys, json, argparse, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image as PILImage
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── Ensure scripts/ importable ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import PC-RadNet, PROMPT_BANK, SEED, DEVICE, CFG

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
CKPT_DIR = Path(CFG["paths"]["checkpoints"])
RESULTS_DIR = Path(CFG["paths"]["results"]) / "external_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = RESULTS_DIR / "visualisations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

_ext = CFG["paths"].get("external_datasets", {})
DATASET_PATHS = {
    k: Path(v) for k, v in _ext.items() if v
}

SAG_NAMES = {0: "BG", 1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"}
AXIAL_NAMES = {0: "BG", 1: "IVD", 2: "PE", 3: "SC"}

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING (corrected)
# ═══════════════════════════════════════════════════════════════

def roi_crop(img_2d, mask_2d, margin=20):
    """Crop to bounding box of foreground with margin. Returns cropped img, mask."""
    fg = mask_2d > 0
    if fg.sum() == 0:
        return img_2d, mask_2d
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add margin
    rmin = max(0, rmin - margin)
    rmax = min(img_2d.shape[0] - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img_2d.shape[1] - 1, cmax + margin)
    return img_2d[rmin:rmax+1, cmin:cmax+1], mask_2d[rmin:rmax+1, cmin:cmax+1]


def pad_to_square(img):
    """Pad shorter dimension with zeros to make square."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    if img.ndim == 2:
        padded = np.zeros((size, size), dtype=img.dtype)
    else:
        padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    if img.ndim == 2:
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    else:
        padded[y_offset:y_offset+h, x_offset:x_offset+w, :] = img
    return padded


def preprocess_slice(img_2d, mask_2d=None, target_size=384, do_roi_crop=True):
    """
    Preprocess a 2D slice for model input.
    1. Optional ROI crop to foreground bounding box (+ margin)
    2. Pad to square (preserve aspect ratio)
    3. Resize to target_size × target_size
    4. Normalise to [0,1] then ImageNet mean/std
    Returns: img_tensor [3, H, W], mask_resized [H, W] or None
    """
    if do_roi_crop and mask_2d is not None and mask_2d.max() > 0:
        img_crop, mask_crop = roi_crop(img_2d, mask_2d, margin=30)
    else:
        img_crop = img_2d
        mask_crop = mask_2d

    # Convert to RGB
    if img_crop.ndim == 2:
        img_rgb = np.stack([img_crop] * 3, axis=-1)
    else:
        img_rgb = img_crop

    # Normalise intensity to [0, 1]
    img_f = img_rgb.astype(np.float64)
    p1, p99 = np.percentile(img_f, [1, 99])
    if p99 > p1:
        img_f = np.clip((img_f - p1) / (p99 - p1), 0, 1)
    else:
        img_f = np.zeros_like(img_f)

    # Pad to square
    img_sq = pad_to_square((img_f * 255).astype(np.uint8))

    # Resize
    pil = PILImage.fromarray(img_sq)
    pil = pil.resize((target_size, target_size), PILImage.BILINEAR)
    img_arr = np.array(pil).astype(np.float32) / 255.0

    # ImageNet normalise
    img_arr = (img_arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

    # Resize mask
    mask_resized = None
    if mask_crop is not None:
        mask_sq = pad_to_square(mask_crop)
        pil_m = PILImage.fromarray(mask_sq.astype(np.uint8))
        pil_m = pil_m.resize((target_size, target_size), PILImage.NEAREST)
        mask_resized = np.array(pil_m)

    return tensor, mask_resized


def compute_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    if union == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return (2 * inter / (union + 1e-8)).item()


def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    if union == 0:
        return 1.0 if target.sum() == 0 else 0.0
    return (inter / (union + 1e-8)).item()


def load_model(ckpt_name="best_by_mdice_A2.pt"):
    ckpt_path = CKPT_DIR / ckpt_name
    model = PCRadNet(CFG, use_calibrator=True, lora_rank=CFG["model"]["lora_rank"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    print(f"Loaded checkpoint: {ckpt_name} (val_dice={ckpt.get('val_dice', '?'):.4f})")
    return model


@torch.no_grad()
def run_inference(model, img_tensor, prompt_text, plane_id):
    """Run inference. Returns dict with all predictions."""
    img = img_tensor.unsqueeze(0).to(DEVICE)
    pid = torch.tensor([plane_id], device=DEVICE)

    with autocast("cuda", dtype=torch.bfloat16):
        out = model(img, [prompt_text], pid)

    seg_logits = out["seg_logits_list"][0].float()  # [1, C, H, W]
    prompted_logits = out["prompted_logits"][0, 0].float()  # [H, W]
    grade_logits = out["grade_logits"][0].float()  # [K-1]
    height_pred = out["height_pred"][0].float().item()
    disorder_logits = out["disorder_logits"][0].float()  # [9]

    probs = torch.sigmoid(grade_logits.float())
    pred_grade = int((probs > 0.5).sum().item()) + 1
    disorder_probs = torch.sigmoid(disorder_logits.float()).cpu().numpy()

    return {
        "seg_logits": seg_logits,
        "prompted_logits": prompted_logits,
        "pred_grade": pred_grade,
        "height_mm": height_pred,
        "disorder_probs": disorder_probs,
    }


@torch.no_grad()
def run_per_class_inference(model, img_tensor, plane_id, class_list):
    """Run inference with each class prompt, return per-class seg predictions.
    
    For each class in class_list, use that class's prompt text,
    run inference, and collect the corresponding channel from seg_logits.
    Final segmentation = max-confidence channel across prompts.
    """
    all_seg_probs = {}
    all_prompted_probs = {}
    grade_preds = []
    disorder_accum = []

    view_key = "sagittal" if plane_id == 1 else "axial"
    class_names = SAG_NAMES if plane_id == 1 else AXIAL_NAMES

    for cls_id in class_list:
        if cls_id == 0:
            continue  # skip BG
        prompts = PROMPT_BANK[view_key].get(cls_id, ["structure"])
        prompt_text = prompts[0]  # use first (canonical) prompt

        out = run_inference(model, img_tensor, prompt_text, plane_id)

        # Store the specific class channel
        seg_probs = torch.sigmoid(out["seg_logits"][0])  # [C, H, W]
        all_seg_probs[cls_id] = seg_probs[cls_id].cpu()

        # Also store the prompted head output
        all_prompted_probs[cls_id] = torch.sigmoid(out["prompted_logits"]).cpu()

        grade_preds.append(out["pred_grade"])
        disorder_accum.append(out["disorder_probs"])

    # Aggregate grade: majority vote
    avg_grade = int(np.round(np.mean(grade_preds))) if grade_preds else 3
    avg_disorder = np.mean(disorder_accum, axis=0) if disorder_accum else np.zeros(9)

    return all_seg_probs, all_prompted_probs, avg_grade, avg_disorder


# ═══════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════

def save_visualisation(img_raw, gt_mask, pred_mask, class_names, fname, save_dir):
    """Save side-by-side visualisation: input | GT | prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input
    if img_raw.ndim == 2:
        axes[0].imshow(img_raw, cmap="gray")
    else:
        axes[0].imshow(img_raw)
    axes[0].set_title("Input", fontsize=12)
    axes[0].axis("off")

    n_classes = max(gt_mask.max(), pred_mask.max()) + 1
    colors = ["black", "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"]
    cmap = ListedColormap(colors[:n_classes])

    # GT
    axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    im = axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation="nearest")
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # Legend
    patches = []
    for c, name in class_names.items():
        if c < len(colors):
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=colors[c], label=name))
    fig.legend(handles=patches, loc="lower center", ncol=n_classes, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# SPIDER LABEL MAPPING
# ═══════════════════════════════════════════════════════════════

def spider_label_to_ours(mask_2d):
    """Map SPIDER labels to our sagittal scheme.
    SPIDER: 0=BG, 1-8=VB, 100=SC, 201-208=IVD
    Ours:   0=BG, 1=VB,   2=IVD_sag, 3=Sacrum, 4=SC_sag
    """
    out = np.zeros_like(mask_2d, dtype=np.uint8)
    out[(mask_2d >= 1) & (mask_2d <= 8)] = 1       # VB
    out[(mask_2d >= 201) & (mask_2d <= 208)] = 2    # IVD
    out[mask_2d == 100] = 4                         # SC
    return out


# ═══════════════════════════════════════════════════════════════
# 1. SPIDER EVALUATION (CORRECTED)
# ═══════════════════════════════════════════════════════════════

def evaluate_spider(model, max_patients=None, save_viz=True, n_viz=20, mid_sagittal_only=False):
    """Evaluate on SPIDER T2 sagittal MRI with corrected preprocessing.
    If mid_sagittal_only=True, only the single central valid sagittal slice
    per volume is evaluated (matches training data distribution)."""
    import SimpleITK as sitk

    spider_dir = DATASET_PATHS["spider"]
    img_dir = spider_dir / "images"
    mask_dir = spider_dir / "masks"
    gradings_csv = spider_dir / "radiological_gradings.csv"
    df_grade = pd.read_csv(gradings_csv)

    # Build patient grade/disorder maps
    patient_grades = defaultdict(dict)
    patient_disorders = defaultdict(dict)
    for _, row in df_grade.iterrows():
        pid = int(row["Patient"])
        ivd_label = int(row["IVD label"])
        patient_grades[pid][ivd_label] = int(row["Pfirrman grade"])
        patient_disorders[pid][ivd_label] = {
            "herniation": int(row["Disc herniation"]),
            "spondylolisthesis": int(row["Spondylolisthesis"]),
            "bulging": int(row["Disc bulging"]),
            "narrowing": int(row["Disc narrowing"]),
        }

    all_files = sorted(os.listdir(img_dir))
    t2_files = [f for f in all_files if "_t2." in f and "SPACE" not in f]
    if max_patients:
        t2_files = t2_files[:max_patients]

    print(f"\n{'='*60}")
    print(f"SPIDER Evaluation (v2): {len(t2_files)} T2 volumes")
    print(f"{'='*60}")

    seg_results = []
    grade_results = []
    disorder_results = []
    viz_count = 0

    for vol_idx, fname in enumerate(t2_files):
        pid = int(fname.split("_")[0])
        img_sitk = sitk.ReadImage(str(img_dir / fname))
        mask_sitk = sitk.ReadImage(str(mask_dir / fname))
        img_arr = sitk.GetArrayFromImage(img_sitk)   # [Z, Y, X]
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        # Sagittal slices are along axis 2 (thickest spacing = X dimension)
        n_sag = img_arr.shape[2]

        # Identify slices with meaningful foreground (>1%)
        valid_slices = []
        for s in range(n_sag):
            our_mask = spider_label_to_ours(mask_arr[:, :, s])
            fg_frac = (our_mask > 0).mean()
            if fg_frac > 0.01:
                valid_slices.append(s)

        if not valid_slices:
            continue

        # Restrict to mid-sagittal slice if requested
        if mid_sagittal_only:
            mid_idx_now = valid_slices[len(valid_slices) // 2]
            eval_slices = [mid_idx_now]
        else:
            eval_slices = valid_slices

        # Evaluate each valid slice with per-class prompting
        for s in eval_slices:
            img_slice = img_arr[:, :, s]
            mask_slice = mask_arr[:, :, s]
            our_mask = spider_label_to_ours(mask_slice)

            # Classes present in this slice
            classes_present = list(set(np.unique(our_mask)) - {0})
            if not classes_present:
                continue

            # Preprocess with ROI crop + aspect-ratio-preserving resize
            img_tensor, mask_384 = preprocess_slice(
                img_slice, our_mask, target_size=384, do_roi_crop=True
            )

            # Per-class inference
            seg_probs, prompted_probs, avg_grade, avg_disorder = \
                run_per_class_inference(model, img_tensor, plane_id=1, class_list=classes_present)

            row = {"patient": pid, "slice": s, "fname": fname}

            for cls_id in classes_present:
                cname = SAG_NAMES[cls_id]
                gt_c = torch.from_numpy((mask_384 == cls_id).astype(np.float32))

                if gt_c.sum() < 10:  # skip trivially small regions
                    continue

                if cls_id in seg_probs:
                    d = compute_dice(seg_probs[cls_id], gt_c)
                    iou = compute_iou(seg_probs[cls_id], gt_c)
                    row[f"dice_{cname}"] = d
                    row[f"iou_{cname}"] = iou

                # Also evaluate prompted head
                if cls_id in prompted_probs:
                    d_p = compute_dice(prompted_probs[cls_id], gt_c)
                    row[f"prompted_dice_{cname}"] = d_p

            seg_results.append(row)

            # Save visualisation for a few slices
            if save_viz and viz_count < n_viz and len(classes_present) >= 2:
                # Build predicted mask from per-class probs
                pred_map = np.zeros((384, 384), dtype=np.uint8)
                for cls_id in classes_present:
                    if cls_id in seg_probs:
                        pred_binary = (seg_probs[cls_id].numpy() > 0.5).astype(np.uint8)
                        pred_map[pred_binary == 1] = cls_id

                # Denormalise image for visualisation
                img_viz = img_tensor.permute(1, 2, 0).numpy()
                img_viz = img_viz * IMAGENET_STD + IMAGENET_MEAN
                img_viz = np.clip(img_viz, 0, 1)

                save_visualisation(
                    img_viz, mask_384, pred_map, SAG_NAMES,
                    f"spider_p{pid}_s{s}.png", VIZ_DIR
                )
                viz_count += 1

        # Per-patient grading (mid-sagittal)
        mid_idx = valid_slices[len(valid_slices) // 2]
        img_mid = img_arr[:, :, mid_idx]
        mask_mid = mask_arr[:, :, mid_idx]
        our_mask_mid = spider_label_to_ours(mask_mid)
        classes_mid = list(set(np.unique(our_mask_mid)) - {0})

        img_t_mid, _ = preprocess_slice(img_mid, our_mask_mid, do_roi_crop=True)
        _, _, avg_grade, avg_disorder = run_per_class_inference(
            model, img_t_mid, plane_id=1, class_list=classes_mid if classes_mid else [1, 2]
        )

        # Compare with GT grades
        if pid in patient_grades:
            for ivd_lbl, gt_g in patient_grades[pid].items():
                grade_results.append({
                    "patient": pid, "ivd_label": ivd_lbl,
                    "pred_grade": avg_grade, "gt_grade": gt_g,
                    "correct": int(avg_grade == gt_g),
                    "error": abs(avg_grade - gt_g),
                    "within1": int(abs(avg_grade - gt_g) <= 1),
                })

        # Disorder detection
        if pid in patient_disorders:
            has_herniation = any(d["herniation"] for d in patient_disorders[pid].values())
            has_spondylo = any(d["spondylolisthesis"] for d in patient_disorders[pid].values())
            has_bulging = any(d["bulging"] for d in patient_disorders[pid].values())
            has_narrowing = any(d["narrowing"] for d in patient_disorders[pid].values())

            disorder_results.append({
                "patient": pid,
                "gt_herniation": int(has_herniation),
                "pred_herniation": int(avg_disorder[0] > 0.5),
                "prob_herniation": float(avg_disorder[0]),
                "gt_spondylolisthesis": int(has_spondylo),
                "pred_spondylolisthesis": int(avg_disorder[6] > 0.5),
                "prob_spondylolisthesis": float(avg_disorder[6]),
                "gt_bulging": int(has_bulging),
                "pred_bulging": int(avg_disorder[2] > 0.5),
                "prob_bulging": float(avg_disorder[2]),
                "gt_narrowing": int(has_narrowing),
                "pred_narrowing": int(avg_disorder[7] > 0.5),  # Desiccated ≈ narrowing
                "prob_narrowing": float(avg_disorder[7]),
            })

        if (vol_idx + 1) % 10 == 0:
            print(f"  Processed {vol_idx+1}/{len(t2_files)} volumes")

    # Aggregate
    seg_df = pd.DataFrame(seg_results)
    grade_df = pd.DataFrame(grade_results)
    disorder_df = pd.DataFrame(disorder_results)

    metrics = {"dataset": "SPIDER", "n_volumes": len(t2_files)}

    # Segmentation metrics
    for cname in ["VB", "IVD_sag", "SC_sag"]:
        col = f"dice_{cname}"
        if col in seg_df.columns:
            valid = seg_df[col].dropna()
            if len(valid) > 0:
                metrics[f"dice_{cname}"] = float(valid.mean())
                metrics[f"dice_{cname}_std"] = float(valid.std())
                metrics[f"iou_{cname}"] = float(seg_df[f"iou_{cname}"].dropna().mean())
                metrics[f"n_slices_{cname}"] = len(valid)

        # Prompted dice
        pcol = f"prompted_dice_{cname}"
        if pcol in seg_df.columns:
            valid_p = seg_df[pcol].dropna()
            if len(valid_p) > 0:
                metrics[f"prompted_dice_{cname}"] = float(valid_p.mean())

    # Mean FG Dice
    fg_dices = []
    for cname in ["VB", "IVD_sag", "SC_sag"]:
        if f"dice_{cname}" in metrics:
            fg_dices.append(metrics[f"dice_{cname}"])
    if fg_dices:
        metrics["mean_fg_dice"] = float(np.mean(fg_dices))

    # Grading
    if len(grade_df) > 0:
        metrics["grade_accuracy"] = float(grade_df["correct"].mean())
        metrics["grade_mae"] = float(grade_df["error"].mean())
        metrics["grade_within1"] = float(grade_df["within1"].mean())
        metrics["n_graded"] = len(grade_df)

    # Disorders
    if len(disorder_df) > 0:
        for dname in ["herniation", "spondylolisthesis", "bulging"]:
            gt = disorder_df[f"gt_{dname}"].values
            pred = disorder_df[f"pred_{dname}"].values
            tp = int(((pred == 1) & (gt == 1)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt == 1)).sum())
            tn = int(((pred == 0) & (gt == 0)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            acc = (tp + tn) / max(tp + fp + fn + tn, 1)
            metrics[f"{dname}_precision"] = float(prec)
            metrics[f"{dname}_recall"] = float(rec)
            metrics[f"{dname}_f1"] = float(f1)
            metrics[f"{dname}_accuracy"] = float(acc)
            metrics[f"{dname}_support"] = int(gt.sum())

    suffix = "_midsag" if mid_sagittal_only else ""
    seg_df.to_csv(RESULTS_DIR / f"spider_seg_per_slice{suffix}.csv", index=False)
    grade_df.to_csv(RESULTS_DIR / f"spider_grading{suffix}.csv", index=False)
    disorder_df.to_csv(RESULTS_DIR / f"spider_disorders{suffix}.csv", index=False)

    print(f"\nSPIDER Results (v2 — corrected):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 2. RSNA 2024 (CORRECTED)
# ═══════════════════════════════════════════════════════════════

def evaluate_rsna(model, max_studies=100, save_viz=True, n_viz=10):
    """Evaluate on RSNA 2024 — stenosis detection & grading."""
    import pydicom

    rsna_dir = DATASET_PATHS["rsna"]
    train_csv = pd.read_csv(rsna_dir / "train.csv")
    series_csv = pd.read_csv(rsna_dir / "train_series_descriptions.csv")

    severity_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

    train_dir = rsna_dir / "train_images"
    study_ids = sorted([int(d) for d in os.listdir(train_dir)])
    if max_studies:
        study_ids = study_ids[:max_studies]

    print(f"\n{'='*60}")
    print(f"RSNA 2024 Evaluation (v2): {len(study_ids)} studies")
    print(f"{'='*60}")

    results = []
    viz_count = 0

    for idx, study_id in enumerate(study_ids):
        study_dir = train_dir / str(study_id)
        if not study_dir.exists():
            continue

        study_series = series_csv[series_csv["study_id"] == study_id]
        study_row = train_csv[train_csv["study_id"] == study_id]
        if len(study_row) == 0:
            continue
        study_row = study_row.iloc[0]

        # Process Sagittal T2
        sag_t2 = study_series[study_series["series_description"] == "Sagittal T2/STIR"]
        for _, ser_row in sag_t2.iterrows():
            series_dir = study_dir / str(ser_row["series_id"])
            if not series_dir.exists():
                continue

            dcm_files = sorted(os.listdir(series_dir), key=lambda x: int(x.replace(".dcm", "")))
            if not dcm_files:
                continue

            # Load mid-sagittal slice
            mid_idx = len(dcm_files) // 2
            try:
                ds = pydicom.dcmread(str(series_dir / dcm_files[mid_idx]))
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            img_tensor, _ = preprocess_slice(pixel_array, mask_2d=None, do_roi_crop=False)

            # Run with multiple prompts and average
            grades = []
            disorders = []
            for cls_id in [1, 2, 4]:  # VB, IVD, SC
                prompts = PROMPT_BANK["sagittal"][cls_id]
                out = run_inference(model, img_tensor, prompts[0], plane_id=1)
                grades.append(out["pred_grade"])
                disorders.append(out["disorder_probs"])

            avg_grade = int(np.round(np.mean(grades)))
            avg_disorder = np.mean(disorders, axis=0)

            # Map Pfirrmann → stenosis severity
            if avg_grade <= 2:
                pred_severity = 0
            elif avg_grade == 3:
                pred_severity = 1
            else:
                pred_severity = 2

            for level in levels:
                gt_val = study_row.get(f"spinal_canal_stenosis_{level}", "Normal/Mild")
                gt_sev = severity_map.get(gt_val, 0)
                results.append({
                    "study_id": study_id, "level": level, "plane": "sagittal",
                    "gt_stenosis": gt_val, "gt_severity": gt_sev,
                    "pred_pfirrmann": avg_grade, "pred_severity": pred_severity,
                    "correct": int(pred_severity == gt_sev),
                    "error": abs(pred_severity - gt_sev),
                    "within1": int(abs(pred_severity - gt_sev) <= 1),
                    "stenosis_prob": float(avg_disorder[5]),
                    "gt_positive": int(gt_sev >= 1),
                    "pred_positive": int(avg_disorder[5] > 0.5),
                })

        # Process Axial T2
        axial_t2 = study_series[study_series["series_description"] == "Axial T2"]
        for _, ser_row in axial_t2.iterrows():
            series_dir = study_dir / str(ser_row["series_id"])
            if not series_dir.exists():
                continue
            dcm_files = sorted(os.listdir(series_dir), key=lambda x: int(x.replace(".dcm", "")))
            if not dcm_files:
                continue

            mid_idx = len(dcm_files) // 2
            try:
                ds = pydicom.dcmread(str(series_dir / dcm_files[mid_idx]))
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            img_tensor, _ = preprocess_slice(pixel_array, mask_2d=None, do_roi_crop=False)
            out = run_inference(model, img_tensor, "spinal canal", plane_id=0)

            stenosis_prob = float(out["disorder_probs"][5])
            for level in levels:
                gt_val = study_row.get(f"spinal_canal_stenosis_{level}", "Normal/Mild")
                gt_sev = severity_map.get(gt_val, 0)
                results.append({
                    "study_id": study_id, "level": level, "plane": "axial",
                    "gt_stenosis": gt_val, "gt_severity": gt_sev,
                    "stenosis_prob": stenosis_prob,
                    "gt_positive": int(gt_sev >= 1),
                    "pred_positive": int(stenosis_prob > 0.5),
                })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(study_ids)} studies")

    df = pd.DataFrame(results)
    metrics = {"dataset": "RSNA_2024", "n_studies": len(study_ids)}

    # Sagittal severity mapping
    sag = df[df["plane"] == "sagittal"]
    if len(sag) > 0 and "correct" in sag.columns:
        metrics["stenosis_mapping_accuracy"] = float(sag["correct"].mean())
        metrics["stenosis_within1"] = float(sag["within1"].mean())
        metrics["n_sag_levels"] = len(sag)

    # Axial stenosis detection
    axial = df[df["plane"] == "axial"]
    if len(axial) > 0 and "gt_positive" in axial.columns:
        gt = axial["gt_positive"].values
        pred = axial["pred_positive"].values
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics["axial_stenosis_precision"] = float(prec)
        metrics["axial_stenosis_recall"] = float(rec)
        metrics["axial_stenosis_f1"] = float(f1)

    df.to_csv(RESULTS_DIR / "rsna_results.csv", index=False)

    print(f"\nRSNA 2024 Results (v2):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 3. SPINE DATASET NEW (CORRECTED)
# ═══════════════════════════════════════════════════════════════

def evaluate_spine_new(model, max_cases=50, save_viz=True, n_viz=10):
    """Evaluate VB segmentation on Spine DataSet (NEW) — sagittal MRI."""
    import nibabel as nib

    ds_dir = DATASET_PATHS["spine_new"]
    img_dir = ds_dir / "train_images"
    mask_dir = ds_dir / "train_ground_truth"

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii")])
    if max_cases:
        img_files = img_files[:max_cases]

    print(f"\n{'='*60}")
    print(f"Spine DataSet (NEW) v2: {len(img_files)} volumes")
    print(f"{'='*60}")

    results = []
    viz_count = 0

    for vol_idx, img_fname in enumerate(img_files):
        case_num = img_fname.replace("Case", "").replace(".nii", "")
        mask_fname = f"mask_case{case_num}.nii"
        mask_path = mask_dir / mask_fname
        if not mask_path.exists():
            continue

        img_vol = nib.load(str(img_dir / img_fname)).get_fdata()
        mask_vol = nib.load(str(mask_path)).get_fdata().astype(int)

        # Sagittal slices along axis 2 (thickest spacing)
        n_sag = img_vol.shape[2]

        for s in range(n_sag):
            img_slice = img_vol[:, :, s]
            mask_slice = mask_vol[:, :, s]

            # Map: all vertebrae labels (1-18) → VB class 1
            vb_mask = (mask_slice >= 1).astype(np.uint8)
            if vb_mask.sum() < 100:  # skip near-empty slices
                continue

            # Preprocess with ROI crop
            img_tensor, mask_384 = preprocess_slice(
                img_slice, vb_mask, target_size=384, do_roi_crop=True
            )

            # Run with VB prompt
            out = run_inference(model, img_tensor, "vertebral body", plane_id=1)
            seg_probs = torch.sigmoid(out["seg_logits"][0])  # [5, 384, 384]
            pred_vb = seg_probs[1].cpu()  # class 1 = VB
            gt_vb = torch.from_numpy((mask_384 == 1).astype(np.float32))

            d = compute_dice(pred_vb, gt_vb)
            iou = compute_iou(pred_vb, gt_vb)

            # Also try prompted head
            prompted_pred = torch.sigmoid(out["prompted_logits"]).cpu()
            d_prompted = compute_dice(prompted_pred, gt_vb)

            results.append({
                "case": img_fname, "slice": s,
                "dice_VB": d, "iou_VB": iou,
                "prompted_dice_VB": d_prompted,
            })

            # Visualisation
            if save_viz and viz_count < n_viz and d > 0.3:
                pred_map = (pred_vb.numpy() > 0.5).astype(np.uint8)
                img_viz = img_tensor.permute(1, 2, 0).numpy()
                img_viz = img_viz * IMAGENET_STD + IMAGENET_MEAN
                img_viz = np.clip(img_viz, 0, 1)
                save_visualisation(
                    img_viz, mask_384, pred_map,
                    {0: "BG", 1: "VB"}, f"spinenew_{img_fname}_s{s}.png", VIZ_DIR
                )
                viz_count += 1

        if (vol_idx + 1) % 10 == 0:
            print(f"  Processed {vol_idx+1}/{len(img_files)} volumes")

    df = pd.DataFrame(results)
    metrics = {
        "dataset": "Spine_DataSet_NEW", "n_volumes": len(img_files),
        "n_slices": len(df),
        "dice_VB": float(df["dice_VB"].mean()),
        "dice_VB_std": float(df["dice_VB"].std()),
        "iou_VB": float(df["iou_VB"].mean()),
        "prompted_dice_VB": float(df["prompted_dice_VB"].mean()),
    }

    df.to_csv(RESULTS_DIR / "spine_new_results.csv", index=False)

    print(f"\nSpine DataSet (NEW) Results (v2):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 4. CROSS-MODALITY: X-RAY
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# SPINEWEB LABEL MAPPING
# ═══════════════════════════════════════════════════════════════

def spineweb_label_to_ours(mask_2d):
    """Map SpineWeb labels to our sagittal scheme.

    SpineWeb uses label 255 as the filled vertebral body voxels in each
    per-vertebra .mha file, and the combined AKA*_mask.mha aggregates
    all vertebral body pixels under the same value 255.
    Labels 10-244 are contour/boundary annotation points (not dense body mask).

    Ours:  0=BG, 1=VB (all vertebral bodies incl. sacrum)
    No IVD or SC ground-truth labels exist in SpineWeb.
    """
    out = np.zeros_like(mask_2d, dtype=np.uint8)
    out[mask_2d == 255] = 1   # Dense vertebral body fill
    return out


# ═══════════════════════════════════════════════════════════════
# 4. SPINEWEB EVALUATION
# Criminisi et al., SpineWeb challenge: 17 sagittal MRI volumes
# (T1/T2 FSE, TIRM) with dense vertebral body segmentation labels.
# Only the sagittal head (plane_id=1) is used; the class 1 (VB)
# prediction is compared against the binarised vertebra mask.
# Sacrum (S1 label 91) is evaluated against class 3 (Sacrum).
# ═══════════════════════════════════════════════════════════════

def evaluate_spineweb(model, save_viz=True, n_viz=20):
    """Evaluate on SpineWeb sagittal MRI volumes with VB ground-truth masks.

    Plane selection: sagittal head (plane_id=1) used for all Sag* sequences.
    Coronal volumes (SeriesDescription starts with 'COR') are skipped to keep
    the plane-head assignment consistent.
    """
    import SimpleITK as sitk

    sw_dir = DATASET_PATHS["spineweb"]
    dcm_dir = sw_dir / "SpineDatasets"
    seg_dir = sw_dir / "SpineSegmented"

    # Collect valid paired cases with sagittal orientation
    cases = []
    for dcm_file in sorted(dcm_dir.glob("*.dcm")):
        case_name = dcm_file.stem
        # Find mask: match case name case-insensitively
        mask_file = next(seg_dir.glob(f"{case_name}/*_mask.mha"), None)
        if mask_file is None:
            continue
        # Check series description to skip coronal volumes
        try:
            import pydicom
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            desc = getattr(ds, "SeriesDescription", "").upper()
        except Exception:
            desc = ""
        if "COR" in desc:
            continue  # Skip coronal — not matched by sagittal head
        cases.append((dcm_file, mask_file, desc))

    print(f"\n{'='*60}")
    print(f"SpineWeb Evaluation: {len(cases)} sagittal MRI volumes")
    print(f"Head used: SAGITTAL (plane_id=1)  —  classes: VB, Sacrum")
    print(f"{'='*60}")

    seg_results = []
    viz_count = 0

    for vol_idx, (dcm_file, mask_file, desc) in enumerate(cases):
        # Load 3D volume via SimpleITK (handles multi-frame DICOM)
        try:
            img_sitk = sitk.ReadImage(str(dcm_file))
            mask_sitk = sitk.ReadImage(str(mask_file))
        except Exception as e:
            print(f"  SKIP {dcm_file.name}: {e}")
            continue

        img_arr  = sitk.GetArrayFromImage(img_sitk)   # [N, H, W]
        mask_arr = sitk.GetArrayFromImage(mask_sitk)  # [N, H, W]

        if img_arr.shape != mask_arr.shape:
            # Resize mask to match image spatially
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(img_sitk)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_sitk = resample.Execute(mask_sitk)
            mask_arr = sitk.GetArrayFromImage(mask_sitk)

        n_slices = img_arr.shape[0]

        for s in range(n_slices):
            img_slice  = img_arr[s]
            mask_slice = mask_arr[s]
            our_mask   = spineweb_label_to_ours(mask_slice)

            # Only evaluate slices with meaningful VB foreground (label 255 → class 1)
            fg_frac = (our_mask == 1).mean()
            if fg_frac < 0.005:
                continue

            classes_present = [1]  # Only VB has GT labels in SpineWeb

            img_tensor, mask_384 = preprocess_slice(
                img_slice, our_mask, target_size=384, do_roi_crop=True
            )

            # Sagittal head inference (plane_id=1)
            seg_probs, prompted_probs, _, _ = run_per_class_inference(
                model, img_tensor, plane_id=1, class_list=classes_present
            )

            row = {"case": dcm_file.stem, "slice": s, "series_desc": desc}

            for cls_id in classes_present:
                cname = SAG_NAMES[cls_id]
                gt_c = torch.from_numpy((mask_384 == cls_id).astype(np.float32))
                if gt_c.sum() < 10:
                    continue
                if cls_id in seg_probs:
                    row[f"dice_{cname}"] = compute_dice(seg_probs[cls_id], gt_c)
                    row[f"iou_{cname}"]  = compute_iou(seg_probs[cls_id], gt_c)
                if cls_id in prompted_probs:
                    row[f"prompted_dice_{cname}"] = compute_dice(prompted_probs[cls_id], gt_c)

            seg_results.append(row)

            # Qualitative visualisation
            if save_viz and viz_count < n_viz and 1 in classes_present:
                pred_map = np.zeros((384, 384), dtype=np.uint8)
                for cls_id in classes_present:
                    if cls_id in seg_probs:
                        pred_binary = (seg_probs[cls_id].numpy() > 0.5).astype(np.uint8)
                        pred_map[pred_binary == 1] = cls_id
                img_viz = img_tensor.permute(1, 2, 0).numpy()
                img_viz = img_viz * IMAGENET_STD + IMAGENET_MEAN
                img_viz = np.clip(img_viz, 0, 1)
                viz_cls = {0: "BG", 1: "VB"}
                save_visualisation(
                    img_viz, mask_384, pred_map, viz_cls,
                    f"spineweb_{dcm_file.stem}_s{s:02d}.png", VIZ_DIR
                )
                viz_count += 1

        if (vol_idx + 1) % 5 == 0 or (vol_idx + 1) == len(cases):
            print(f"  Processed {vol_idx+1}/{len(cases)} volumes")

    seg_df = pd.DataFrame(seg_results)
    metrics = {"dataset": "SpineWeb", "modality": "MRI_sagittal", "n_volumes": len(cases)}

    for cname in ["VB"]:
        col = f"dice_{cname}"
        if col in seg_df.columns:
            valid = seg_df[col].dropna()
            if len(valid) > 0:
                metrics[f"dice_{cname}"]     = float(valid.mean())
                metrics[f"dice_{cname}_std"] = float(valid.std())
                metrics[f"iou_{cname}"]      = float(seg_df[f"iou_{cname}"].dropna().mean())
                metrics[f"n_slices_{cname}"] = len(valid)
        pcol = f"prompted_dice_{cname}"
        if pcol in seg_df.columns:
            valid_p = seg_df[pcol].dropna()
            if len(valid_p) > 0:
                metrics[f"prompted_dice_{cname}"] = float(valid_p.mean())

    # Mean VB Dice (primary metric — only class with GT labels)
    if "dice_VB" in metrics:
        metrics["mean_fg_dice"] = metrics["dice_VB"]

    seg_df.to_csv(RESULTS_DIR / "spineweb_results.csv", index=False)

    print(f"\nSpineWeb Results:")
    print(f"  n_volumes  : {metrics['n_volumes']}")
    print(f"  n_slices   : {metrics.get('n_slices_VB', 0)}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")

    # Save JSON
    with open(RESULTS_DIR / "spineweb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def evaluate_xray(model, max_images=50):
    """Cross-modality test on X-ray (expects low performance)."""
    xray_dir = DATASET_PATHS["xray"]

    img_files = []
    for root, dirs, files in os.walk(xray_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "viz" not in root:
                img_files.append(os.path.join(root, f))
        if len(img_files) > max_images * 2:
            break
    img_files = img_files[:max_images]

    print(f"\n{'='*60}")
    print(f"Scoliosis X-ray (v2): {len(img_files)} images")
    print(f"{'='*60}")

    results = []
    for img_path in img_files:
        try:
            pil = PILImage.open(img_path).convert("RGB")
            img_arr = np.array(pil)
        except Exception:
            continue

        img_tensor, _ = preprocess_slice(img_arr, mask_2d=None, do_roi_crop=False)

        out = run_inference(model, img_tensor, "vertebral body", plane_id=1)
        seg_probs = torch.sigmoid(out["seg_logits"][0])

        for c, cname in SAG_NAMES.items():
            if c == 0:
                continue
            area = (seg_probs[c] > 0.5).float().mean().item()
            results.append({
                "image": os.path.basename(img_path), "class": cname,
                "pred_area_fraction": area,
            })

    df = pd.DataFrame(results)
    metrics = {"dataset": "Scoliosis_Xray", "modality": "X-ray", "n_images": len(img_files)}
    for cname in ["VB", "IVD_sag", "SC_sag"]:
        sub = df[df["class"] == cname]
        if len(sub) > 0:
            metrics[f"{cname}_mean_pred_area"] = float(sub["pred_area_fraction"].mean())

    df.to_csv(RESULTS_DIR / "xray_results.csv", index=False)

    print(f"\nX-ray Results (v2):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["spider", "rsna", "spine_new", "xray", "spineweb", "all"])
    parser.add_argument("--checkpoint", type=str, default="best_by_mdice_A2.pt")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--mid-sagittal-only", action="store_true",
                        help="SPIDER only: evaluate the single central valid sagittal slice per volume")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    all_metrics = {}
    save_viz = not args.no_viz

    if args.dataset in ("spider", "all"):
        all_metrics["spider"] = evaluate_spider(
            model, max_patients=args.max_patients, save_viz=save_viz,
            mid_sagittal_only=getattr(args, 'mid_sagittal_only', False)
        )

    if args.dataset in ("rsna", "all"):
        all_metrics["rsna"] = evaluate_rsna(model, max_studies=args.max_patients or 100, save_viz=save_viz)

    if args.dataset in ("spine_new", "all"):
        all_metrics["spine_new"] = evaluate_spine_new(model, max_cases=args.max_patients or 50, save_viz=save_viz)

    if args.dataset in ("xray", "all"):
        all_metrics["xray"] = evaluate_xray(model, max_images=args.max_patients or 50)

    if args.dataset in ("spineweb", "all"):
        all_metrics["spineweb"] = evaluate_spineweb(model, save_viz=save_viz)

    with open(RESULTS_DIR / "external_evaluation_metrics_v2.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*80}")
    print(f"CROSS-DATASET EVALUATION SUMMARY (v2 — corrected)")
    print(f"{'='*80}")
    print(f"{'Dataset':<25} {'Mod.':<7} {'n':<8} {'Seg Dice':<12} {'Grade':<10} {'W-1':<10} {'Dis-F1':<10}")
    print(f"{'-'*80}")
    print(f"{'LSMA-PQR (ours)':<25} {'MRI':<7} {'624':<8} {'0.859':<12} {'61.8%':<10} {'92.3%':<10} {'0.634':<10}")

    if "spider" in all_metrics:
        m = all_metrics["spider"]
        seg = f"{m.get('mean_fg_dice', 0):.3f}"
        grd = f"{m.get('grade_accuracy', 0)*100:.1f}%"
        w1 = f"{m.get('grade_within1', 0)*100:.1f}%"
        bf1 = f"{m.get('bulging_f1', 0):.3f}"
        n = sum(m.get(f"n_slices_{c}", 0) for c in ["VB", "IVD_sag", "SC_sag"])
        print(f"{'SPIDER':<25} {'MRI':<7} {str(n):<8} {seg:<12} {grd:<10} {w1:<10} {bf1:<10}")

    if "rsna" in all_metrics:
        m = all_metrics["rsna"]
        acc = f"{m.get('stenosis_mapping_accuracy', 0)*100:.1f}%"
        w1 = f"{m.get('stenosis_within1', 0)*100:.1f}%"
        print(f"{'RSNA 2024':<25} {'MRI':<7} {str(m.get('n_sag_levels', 0)):<8} {'---':<12} {acc:<10} {w1:<10} {'---':<10}")

    if "spine_new" in all_metrics:
        m = all_metrics["spine_new"]
        vb = f"{m.get('dice_VB', 0):.3f}"
        print(f"{'Spine DataSet (NEW)':<25} {'MRI':<7} {str(m.get('n_slices', 0)):<8} {vb:<12} {'---':<10} {'---':<10} {'---':<10}")

    if "xray" in all_metrics:
        m = all_metrics["xray"]
        print(f"{'Scoliosis X-ray':<25} {'X-ray':<7} {str(m.get('n_images', 0)):<8} {'---':<12} {'---':<10} {'---':<10} {'---':<10}")

    if "spineweb" in all_metrics:
        m = all_metrics["spineweb"]
        seg = f"{m.get('dice_VB', 0):.3f}"
        n = m.get('n_slices_VB', 0)
        print(f"{'SpineWeb':<25} {'MRI':<7} {str(n):<8} {seg:<12} {'---':<10} {'---':<10} {'---':<10}")

    print(f"{'='*80}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
