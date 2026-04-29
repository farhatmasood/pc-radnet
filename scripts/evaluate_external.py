"""
evaluate_external.py — Cross-dataset & cross-modality evaluation of PC-RadNet

Evaluates the trained PC-RadNet model on external datasets:
  1. SPIDER (MRI, sagittal): Segmentation (VB, IVD, SC) + Pfirrmann grading + disorder
  2. RSNA 2024 (MRI, axial+sagittal): Stenosis detection
  3. Spine DataSet NEW (MRI): VB segmentation
  4. Scoliosis X-ray: Cross-modality robustness (expected low performance)

Usage:
    python scripts/evaluate_external.py --dataset spider
    python scripts/evaluate_external.py --dataset rsna
    python scripts/evaluate_external.py --dataset spine_new
    python scripts/evaluate_external.py --dataset xray
    python scripts/evaluate_external.py --dataset all
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
RESULTS_DIR = Path(CFG["paths"]["results"]) / "external"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

_ext = CFG["paths"].get("external_datasets", {})
DATASET_PATHS = {
    k: Path(v) for k, v in _ext.items() if v
}

# SPIDER label mapping → our sagittal classes
# Our sagittal: {0: BG, 1: VB, 2: IVD_sag, 3: Sacrum, 4: SC_sag}
# SPIDER: 0=BG, 1-8=VB instances, 100=SC, 201-208=IVD instances

SAG_NAMES = {0: "BG", 1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"}
AXIAL_NAMES = {0: "BG", 1: "IVD", 2: "PE", 3: "SC"}


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


def preprocess_slice(img_2d, target_size=384):
    """Convert a 2D grayscale/RGB array to model input tensor."""
    if img_2d.ndim == 2:
        img_rgb = np.stack([img_2d] * 3, axis=-1)
    elif img_2d.shape[-1] == 1:
        img_rgb = np.concatenate([img_2d] * 3, axis=-1)
    else:
        img_rgb = img_2d

    # Normalize to 0-1
    img_f = img_rgb.astype(np.float32)
    vmin, vmax = img_f.min(), img_f.max()
    if vmax > vmin:
        img_f = (img_f - vmin) / (vmax - vmin)
    else:
        img_f = np.zeros_like(img_f)

    # Resize
    pil = PILImage.fromarray((img_f * 255).astype(np.uint8))
    pil = pil.resize((target_size, target_size), PILImage.BILINEAR)
    img_arr = np.array(pil).astype(np.float32) / 255.0

    # ImageNet normalize
    img_arr = (img_arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
    return tensor


def resize_mask(mask_2d, target_size=384):
    """Resize a 2D integer mask via nearest-neighbour."""
    pil = PILImage.fromarray(mask_2d.astype(np.uint8))
    pil = pil.resize((target_size, target_size), PILImage.NEAREST)
    return np.array(pil)


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
    """Run single-image inference. Returns dict of predictions."""
    img = img_tensor.unsqueeze(0).to(DEVICE)
    pid = torch.tensor([plane_id], device=DEVICE)

    with autocast("cuda", dtype=torch.bfloat16):
        out = model(img, [prompt_text], pid)

    seg_logits = out["seg_logits_list"][0]  # [1, C, H, W]
    prompted_logits = out["prompted_logits"][0, 0]  # [H, W]
    grade_logits = out["grade_logits"][0]  # [K-1]
    height_pred = out["height_pred"][0].item()
    disorder_logits = out["disorder_logits"][0]  # [9]

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


# ═══════════════════════════════════════════════════════════════
# 1. SPIDER EVALUATION
# ═══════════════════════════════════════════════════════════════

def spider_label_to_ours(mask_2d):
    """Map SPIDER labels to our sagittal label scheme.
    SPIDER: 0=BG, 1-8=VB, 100=SC, 201-208=IVD
    Ours:   0=BG, 1=VB,   2=IVD_sag, 3=Sacrum, 4=SC_sag
    """
    out = np.zeros_like(mask_2d, dtype=np.uint8)
    # VB: labels 1-8
    out[(mask_2d >= 1) & (mask_2d <= 8)] = 1
    # IVD: labels 201-208
    out[(mask_2d >= 201) & (mask_2d <= 208)] = 2
    # SC: label 100
    out[mask_2d == 100] = 4
    # No Sacrum equivalent in SPIDER (label 3 skipped)
    return out


def evaluate_spider(model, max_patients=None):
    """Evaluate on SPIDER T2 sagittal MRI."""
    import SimpleITK as sitk

    spider_dir = DATASET_PATHS["spider"]
    img_dir = spider_dir / "images"
    mask_dir = spider_dir / "masks"
    gradings_csv = spider_dir / "radiological_gradings.csv"

    df_grade = pd.read_csv(gradings_csv)

    # Build patient → disc grades mapping
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

    # Find T2 images (primary for Pfirrmann assessment)
    all_files = sorted(os.listdir(img_dir))
    t2_files = [f for f in all_files if "_t2." in f and "SPACE" not in f]

    if max_patients:
        t2_files = t2_files[:max_patients]

    print(f"\n{'='*60}")
    print(f"SPIDER Evaluation: {len(t2_files)} T2 volumes")
    print(f"{'='*60}")

    all_results = []
    seg_results = []
    grade_results = []
    disorder_results = []

    for vol_idx, fname in enumerate(t2_files):
        pid = int(fname.split("_")[0])
        img_sitk = sitk.ReadImage(str(img_dir / fname))
        mask_sitk = sitk.ReadImage(str(mask_dir / fname))

        img_arr = sitk.GetArrayFromImage(img_sitk)   # [Z, Y, X]
        mask_arr = sitk.GetArrayFromImage(mask_sitk)  # [Z, Y, X]
        n_slices = img_arr.shape[2]  # X dimension = sagittal slices

        # Evaluate segmentation on ALL slices with labels
        for s in range(n_slices):
            img_slice = img_arr[:, :, s]  # [Z, Y] = one sagittal slice
            mask_slice = mask_arr[:, :, s]
            our_mask = spider_label_to_ours(mask_slice)

            # Skip empty slices
            if our_mask.max() == 0:
                continue

            our_mask_384 = resize_mask(our_mask, 384)
            img_tensor = preprocess_slice(img_slice)

            # Run sagittal prompts to get per-class segmentation
            prompt = random.choice(PROMPT_BANK["sagittal"][1])
            out = run_inference(model, img_tensor, prompt, plane_id=1)

            seg_logits = out["seg_logits"]  # [1, 5, H, W]
            row = {"patient": pid, "slice": s, "fname": fname}

            # Per-class Dice (skip BG=0 and Sacrum=3)
            for c, cname in SAG_NAMES.items():
                if c == 0 or c == 3:  # Skip BG and Sacrum (not in SPIDER)
                    continue
                gt_c = torch.from_numpy((our_mask_384 == c).astype(np.float32)).to(DEVICE)
                pred_c = torch.sigmoid(seg_logits[0, c]).to(DEVICE)

                if gt_c.sum() == 0 and pred_c.max() < 0.5:
                    continue  # Skip if no GT and no prediction

                d = compute_dice(pred_c, gt_c)
                iou = compute_iou(pred_c, gt_c)
                row[f"dice_{cname}"] = d
                row[f"iou_{cname}"] = iou

            seg_results.append(row)

        # Grading: use mid-sagittal slice for per-disc grading
        mid_slice_idx = n_slices // 2
        img_mid = img_arr[:, :, mid_slice_idx]
        mask_mid = mask_arr[:, :, mid_slice_idx]
        img_tensor_mid = preprocess_slice(img_mid)

        prompt_disc = random.choice(PROMPT_BANK["sagittal"][2])
        out_mid = run_inference(model, img_tensor_mid, prompt_disc, plane_id=1)
        pred_grade = out_mid["pred_grade"]
        disorder_probs = out_mid["disorder_probs"]

        # Compare with GT grades: use the median grade of visible discs
        if pid in patient_grades:
            gt_grades_list = list(patient_grades[pid].values())
            gt_median = int(np.median(gt_grades_list))

            grade_results.append({
                "patient": pid,
                "pred_grade": pred_grade,
                "gt_median_grade": gt_median,
                "correct": int(pred_grade == gt_median),
                "error": abs(pred_grade - gt_median),
                "within1": int(abs(pred_grade - gt_median) <= 1),
                "n_discs": len(gt_grades_list),
            })

            # Per-disc grade comparison (assign model prediction to each disc)
            for ivd_lbl, gt_g in patient_grades[pid].items():
                grade_results.append({
                    "patient": pid,
                    "ivd_label": ivd_lbl,
                    "pred_grade": pred_grade,
                    "gt_grade": gt_g,
                    "correct": int(pred_grade == gt_g),
                    "error": abs(pred_grade - gt_g),
                    "within1": int(abs(pred_grade - gt_g) <= 1),
                })

        # Disorder detection on mid-sagittal
        # Map our disorder indices: 0=Herniation, 5=Stenosis, 6=Spondylolisthesis
        # SPIDER disorders: herniation, spondylolisthesis, bulging, narrowing
        if pid in patient_disorders:
            # Aggregate patient-level disorders
            has_herniation = any(d["herniation"] for d in patient_disorders[pid].values())
            has_spondylo = any(d["spondylolisthesis"] for d in patient_disorders[pid].values())
            has_bulging = any(d["bulging"] for d in patient_disorders[pid].values())
            has_narrowing = any(d["narrowing"] for d in patient_disorders[pid].values())

            disorder_results.append({
                "patient": pid,
                # Our disorder head indices:
                # 0=Herniation, 1=Protrusion, 2=Bulge, 3=ThecalComp,
                # 4=NerveComp, 5=Stenosis, 6=Spondylolisthesis,
                # 7=Desiccated, 8=LigFlavum
                "gt_herniation": int(has_herniation),
                "pred_herniation": int(disorder_probs[0] > 0.5),
                "pred_herniation_prob": float(disorder_probs[0]),
                "gt_spondylolisthesis": int(has_spondylo),
                "pred_spondylolisthesis": int(disorder_probs[6] > 0.5),
                "pred_spondylolisthesis_prob": float(disorder_probs[6]),
                "gt_bulging": int(has_bulging),
                "pred_bulging": int(disorder_probs[2] > 0.5),
                "pred_bulging_prob": float(disorder_probs[2]),
            })

        if (vol_idx + 1) % 10 == 0:
            print(f"  Processed {vol_idx+1}/{len(t2_files)} volumes")

    # Aggregate results
    seg_df = pd.DataFrame(seg_results)
    grade_df = pd.DataFrame(grade_results)
    disorder_df = pd.DataFrame(disorder_results)

    metrics = {"dataset": "SPIDER", "n_volumes": len(t2_files)}

    # Segmentation metrics
    for cname in ["VB", "IVD_sag", "SC_sag"]:
        col = f"dice_{cname}"
        if col in seg_df.columns:
            valid = seg_df[col].dropna()
            metrics[f"dice_{cname}"] = valid.mean()
            metrics[f"dice_{cname}_std"] = valid.std()
            metrics[f"iou_{cname}"] = seg_df[f"iou_{cname}"].dropna().mean()
            metrics[f"n_slices_{cname}"] = len(valid)

    fg_dices = []
    for cname in ["VB", "IVD_sag", "SC_sag"]:
        if f"dice_{cname}" in metrics:
            fg_dices.append(metrics[f"dice_{cname}"])
    if fg_dices:
        metrics["mean_fg_dice"] = np.mean(fg_dices)

    # Grading metrics
    if len(grade_df) > 0:
        per_disc = grade_df[grade_df["gt_grade"].notna()] if "gt_grade" in grade_df.columns else grade_df
        if "gt_grade" in per_disc.columns:
            metrics["grade_accuracy"] = per_disc["correct"].mean()
            metrics["grade_mae"] = per_disc["error"].mean()
            metrics["grade_within1"] = per_disc["within1"].mean()
            metrics["n_graded"] = len(per_disc)

    # Disorder metrics
    if len(disorder_df) > 0:
        for dname in ["herniation", "spondylolisthesis", "bulging"]:
            gt_col = f"gt_{dname}"
            pred_col = f"pred_{dname}"
            if gt_col in disorder_df.columns:
                gt = disorder_df[gt_col].values
                pred = disorder_df[pred_col].values
                tp = ((pred == 1) & (gt == 1)).sum()
                fp = ((pred == 1) & (gt == 0)).sum()
                fn = ((pred == 0) & (gt == 1)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                metrics[f"{dname}_precision"] = prec
                metrics[f"{dname}_recall"] = rec
                metrics[f"{dname}_f1"] = f1
                metrics[f"{dname}_support"] = int(gt.sum())

    # Save results
    seg_df.to_csv(RESULTS_DIR / "spider_seg_per_slice.csv", index=False)
    grade_df.to_csv(RESULTS_DIR / "spider_grading.csv", index=False)
    disorder_df.to_csv(RESULTS_DIR / "spider_disorders.csv", index=False)

    print(f"\nSPIDER Results:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 2. RSNA 2024 EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_rsna(model, max_studies=50):
    """Evaluate on RSNA 2024 Lumbar Spine MRI — stenosis detection."""
    import pydicom

    rsna_dir = DATASET_PATHS["rsna"]
    train_csv = pd.read_csv(rsna_dir / "train.csv")
    series_csv = pd.read_csv(rsna_dir / "train_series_descriptions.csv")
    coord_csv = pd.read_csv(rsna_dir / "train_label_coordinates.csv")

    # Map RSNA severity → ordinal
    severity_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

    train_dir = rsna_dir / "train_images"
    study_ids = sorted([int(d) for d in os.listdir(train_dir)])

    if max_studies:
        study_ids = study_ids[:max_studies]

    print(f"\n{'='*60}")
    print(f"RSNA 2024 Evaluation: {len(study_ids)} studies")
    print(f"{'='*60}")

    results = []

    for idx, study_id in enumerate(study_ids):
        study_dir = train_dir / str(study_id)
        if not study_dir.exists():
            continue

        # Get series info
        study_series = series_csv[series_csv["study_id"] == study_id]

        # Get GT labels
        study_row = train_csv[train_csv["study_id"] == study_id]
        if len(study_row) == 0:
            continue
        study_row = study_row.iloc[0]

        # Find Sagittal T2 series (best for stenosis)
        sag_t2 = study_series[study_series["series_description"] == "Sagittal T2/STIR"]
        axial_t2 = study_series[study_series["series_description"] == "Axial T2"]

        # Process sagittal T2
        for _, ser_row in sag_t2.iterrows():
            series_dir = study_dir / str(ser_row["series_id"])
            if not series_dir.exists():
                continue

            dcm_files = sorted(os.listdir(series_dir), key=lambda x: int(x.replace(".dcm", "")))
            n_slices = len(dcm_files)
            if n_slices == 0:
                continue

            # Load mid-sagittal slice
            mid_idx = n_slices // 2
            dcm_path = series_dir / dcm_files[mid_idx]
            try:
                ds = pydicom.dcmread(str(dcm_path))
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            img_tensor = preprocess_slice(pixel_array)
            prompt = "spinal canal"
            out = run_inference(model, img_tensor, prompt, plane_id=1)

            # Map our Pfirrmann prediction → approximate stenosis severity
            pred_grade = out["pred_grade"]
            # Heuristic: Pfirrmann 1-2 ≈ Normal/Mild, 3 ≈ Moderate, 4-5 ≈ Severe
            if pred_grade <= 2:
                pred_severity = 0
            elif pred_grade == 3:
                pred_severity = 1
            else:
                pred_severity = 2

            # Compare with GT stenosis for each level
            for level in levels:
                gt_val = study_row.get(f"spinal_canal_stenosis_{level}", "Normal/Mild")
                gt_sev = severity_map.get(gt_val, 0)
                results.append({
                    "study_id": study_id,
                    "level": level,
                    "gt_stenosis": gt_val,
                    "gt_severity": gt_sev,
                    "pred_pfirrmann": pred_grade,
                    "pred_severity_mapped": pred_severity,
                    "correct": int(pred_severity == gt_sev),
                    "error": abs(pred_severity - gt_sev),
                    "within1": int(abs(pred_severity - gt_sev) <= 1),
                })

        # Process axial T2 for segmentation quality
        for _, ser_row in axial_t2.iterrows():
            series_dir = study_dir / str(ser_row["series_id"])
            if not series_dir.exists():
                continue

            dcm_files = sorted(os.listdir(series_dir), key=lambda x: int(x.replace(".dcm", "")))
            if len(dcm_files) == 0:
                continue

            mid_idx = len(dcm_files) // 2
            dcm_path = series_dir / dcm_files[mid_idx]
            try:
                ds = pydicom.dcmread(str(dcm_path))
                pixel_array = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            img_tensor = preprocess_slice(pixel_array)
            prompt = "spinal canal"
            out = run_inference(model, img_tensor, prompt, plane_id=0)

            # Store disorder predictions for stenosis
            disorder_probs = out["disorder_probs"]
            stenosis_prob = float(disorder_probs[5])  # Index 5 = Stenosis

            for level in levels:
                gt_val = study_row.get(f"spinal_canal_stenosis_{level}", "Normal/Mild")
                gt_sev = severity_map.get(gt_val, 0)
                gt_positive = int(gt_sev >= 1)  # Moderate or Severe = positive

                results.append({
                    "study_id": study_id,
                    "level": level,
                    "plane": "axial",
                    "gt_stenosis": gt_val,
                    "gt_positive": gt_positive,
                    "pred_stenosis_prob": stenosis_prob,
                    "pred_positive": int(stenosis_prob > 0.5),
                })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(study_ids)} studies")

    df = pd.DataFrame(results)

    metrics = {"dataset": "RSNA_2024", "n_studies": len(study_ids)}

    # Sagittal grading mapping results
    if "pred_severity_mapped" in df.columns:
        sag_df = df.dropna(subset=["pred_severity_mapped"])
        if "correct" in sag_df.columns and len(sag_df) > 0:
            metrics["stenosis_mapping_accuracy"] = sag_df["correct"].mean()
            metrics["stenosis_within1"] = sag_df["within1"].mean()
            metrics["n_sag_levels"] = len(sag_df)

    # Axial stenosis detection results
    axial_df = df.dropna(subset=["pred_positive"]) if "pred_positive" in df.columns else pd.DataFrame()
    if "gt_positive" in axial_df.columns and len(axial_df) > 0:
        gt = axial_df["gt_positive"].values
        pred = axial_df["pred_positive"].values
        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics["axial_stenosis_precision"] = float(prec)
        metrics["axial_stenosis_recall"] = float(rec)
        metrics["axial_stenosis_f1"] = float(f1)
        metrics["n_axial_levels"] = len(axial_df)

    df.to_csv(RESULTS_DIR / "rsna_results.csv", index=False)

    print(f"\nRSNA 2024 Results:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 3. SPINE DATASET NEW — VB SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def evaluate_spine_new(model, max_cases=50):
    """Evaluate VB segmentation on Spine DataSet (NEW)."""
    import nibabel as nib

    ds_dir = DATASET_PATHS["spine_new"]
    img_dir = ds_dir / "train_images"
    mask_dir = ds_dir / "train_ground_truth"

    # Collect paired files
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii")])
    if max_cases:
        img_files = img_files[:max_cases]

    print(f"\n{'='*60}")
    print(f"Spine DataSet (NEW) Evaluation: {len(img_files)} volumes")
    print(f"{'='*60}")

    results = []

    for vol_idx, img_fname in enumerate(img_files):
        case_num = img_fname.replace("Case", "").replace(".nii", "")
        mask_fname = f"mask_case{case_num}.nii"
        mask_path = mask_dir / mask_fname

        if not mask_path.exists():
            continue

        img_vol = nib.load(str(img_dir / img_fname)).get_fdata()
        mask_vol = nib.load(str(mask_path)).get_fdata().astype(int)

        # Shape: (X, Y, Z) — Z slices
        n_slices = img_vol.shape[2]

        for s in range(n_slices):
            img_slice = img_vol[:, :, s]
            mask_slice = mask_vol[:, :, s]

            # Map: individual vertebrae (1-18) → VB (class 1 in our scheme)
            vb_mask = (mask_slice >= 1).astype(np.uint8)

            if vb_mask.sum() == 0:
                continue

            vb_mask_384 = resize_mask(vb_mask, 384)
            img_tensor = preprocess_slice(img_slice)

            prompt = "vertebral body"
            out = run_inference(model, img_tensor, prompt, plane_id=1)

            seg_logits = out["seg_logits"]
            gt_vb = torch.from_numpy(vb_mask_384.astype(np.float32)).to(DEVICE)
            pred_vb = torch.sigmoid(seg_logits[0, 1]).to(DEVICE)  # class 1 = VB

            d = compute_dice(pred_vb, gt_vb)
            iou = compute_iou(pred_vb, gt_vb)

            results.append({
                "case": img_fname, "slice": s,
                "dice_VB": d, "iou_VB": iou,
            })

        if (vol_idx + 1) % 10 == 0:
            print(f"  Processed {vol_idx+1}/{len(img_files)} volumes")

    df = pd.DataFrame(results)

    metrics = {
        "dataset": "Spine_DataSet_NEW",
        "n_volumes": len(img_files),
        "n_slices": len(df),
        "dice_VB": df["dice_VB"].mean(),
        "dice_VB_std": df["dice_VB"].std(),
        "iou_VB": df["iou_VB"].mean(),
    }

    df.to_csv(RESULTS_DIR / "spine_new_results.csv", index=False)

    print(f"\nSpine DataSet (NEW) Results:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# 4. CROSS-MODALITY: SCOLIOSIS X-RAY
# ═══════════════════════════════════════════════════════════════

def evaluate_xray(model, max_images=50):
    """Cross-modality test on X-ray images (expected low performance)."""
    xray_dir = DATASET_PATHS["xray"]

    # Find image directories
    data_dir = xray_dir / "data"
    if not data_dir.exists():
        # Try alternative paths
        for subdir in ["train", "test", "images"]:
            alt = xray_dir / subdir
            if alt.exists():
                data_dir = alt
                break

    # Collect JPG/PNG images
    img_files = []
    for root, dirs, files in os.walk(xray_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "viz" not in root:
                img_files.append(os.path.join(root, f))
        if len(img_files) > max_images * 2:
            break

    img_files = img_files[:max_images]

    print(f"\n{'='*60}")
    print(f"Scoliosis X-ray Cross-Modality: {len(img_files)} images")
    print(f"{'='*60}")

    results = []

    for img_idx, img_path in enumerate(img_files):
        try:
            pil = PILImage.open(img_path).convert("RGB")
            img_arr = np.array(pil)
        except Exception:
            continue

        img_tensor = preprocess_slice(img_arr)

        # Try sagittal prompts (spine is visible in AP X-ray)
        prompt = "vertebral body"
        out = run_inference(model, img_tensor, prompt, plane_id=1)

        # Get predicted mask area (as fraction of image)
        seg_logits = out["seg_logits"]
        for c, cname in SAG_NAMES.items():
            if c == 0:
                continue
            pred = torch.sigmoid(seg_logits[0, c])
            area = (pred > 0.5).float().mean().item()
            results.append({
                "image": os.path.basename(img_path),
                "class": cname,
                "pred_area_fraction": area,
                "pred_grade": out["pred_grade"],
                "pred_height": out["height_mm"],
            })

    df = pd.DataFrame(results)

    # Summarise the predicted mask areas (expect very small or zero)
    metrics = {
        "dataset": "Scoliosis_Xray",
        "modality": "X-ray (AP)",
        "n_images": len(img_files),
    }

    for cname in ["VB", "IVD_sag", "SC_sag"]:
        sub = df[df["class"] == cname]
        if len(sub) > 0:
            metrics[f"{cname}_mean_pred_area"] = sub["pred_area_fraction"].mean()

    df.to_csv(RESULTS_DIR / "xray_results.csv", index=False)

    print(f"\nScoliosis X-ray Results (Cross-Modality):")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="External dataset evaluation")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["spider", "rsna", "spine_new", "xray", "all"])
    parser.add_argument("--checkpoint", type=str, default="best_by_mdice_A2.pt")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Max patients/studies per dataset (for quick testing)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    all_metrics = {}

    if args.dataset in ("spider", "all"):
        all_metrics["spider"] = evaluate_spider(model, max_patients=args.max_patients)

    if args.dataset in ("rsna", "all"):
        all_metrics["rsna"] = evaluate_rsna(model, max_studies=args.max_patients or 50)

    if args.dataset in ("spine_new", "all"):
        all_metrics["spine_new"] = evaluate_spine_new(model, max_cases=args.max_patients or 50)

    if args.dataset in ("xray", "all"):
        all_metrics["xray"] = evaluate_xray(model, max_images=args.max_patients or 50)

    # Save combined metrics
    with open(RESULTS_DIR / "external_evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<25} {'Modality':<10} {'Seg Dice':<12} {'Grade Acc':<12} {'Note'}")
    print(f"{'-'*70}")

    # In-distribution reference
    print(f"{'LSMA-PQR (ours)':<25} {'MRI':<10} {'0.859':<12} {'61.8%':<12} {'In-distribution'}")

    if "spider" in all_metrics:
        m = all_metrics["spider"]
        seg = f"{m.get('mean_fg_dice', 0):.3f}"
        grd = f"{m.get('grade_accuracy', 0)*100:.1f}%"
        print(f"{'SPIDER':<25} {'MRI':<10} {seg:<12} {grd:<12} {'Cross-dataset'}")

    if "rsna" in all_metrics:
        m = all_metrics["rsna"]
        sten = f"{m.get('stenosis_mapping_accuracy', 0)*100:.1f}%"
        print(f"{'RSNA 2024':<25} {'MRI':<10} {'N/A':<12} {sten:<12} {'Stenosis map'}")

    if "spine_new" in all_metrics:
        m = all_metrics["spine_new"]
        vb = f"{m.get('dice_VB', 0):.3f} (VB)"
        print(f"{'Spine DataSet (NEW)':<25} {'MRI':<10} {vb:<12} {'N/A':<12} {'VB only'}")

    if "xray" in all_metrics:
        m = all_metrics["xray"]
        area = m.get("VB_mean_pred_area", 0)
        print(f"{'Scoliosis X-ray':<25} {'X-ray':<10} {'N/A':<12} {'N/A':<12} {f'Area={area:.3f}'}")

    print(f"{'='*70}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
