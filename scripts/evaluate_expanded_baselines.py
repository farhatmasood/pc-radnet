"""
evaluate_expanded_baselines.py - Expanded baseline evaluation for PCRadNet
Evaluates SAM2 (automatic) and MedSAM2 on the LSMA-PQR test set for Table 4 expansion.

Usage:
    conda activate pc_radnet
    python scripts/evaluate_expanded_baselines.py --model sam2 --split test --max-samples 50
    python scripts/evaluate_expanded_baselines.py --model medsam2 --split test --max-samples 50
    python scripts/evaluate_expanded_baselines.py --all --split test --max-samples 50
"""
import os, sys, json, argparse, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from repo_config import load_config, resolve_image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CFG = load_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEDSAM2_ROOT = Path(CFG["paths"]["pretrained"]["medsam2_root"])
MEDSAM2_CKPT = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"


def parse_filename(fname):
    stem = fname.replace(".png", "")
    parts = stem.split("_")
    return {
        "view": parts[0],
        "sequence": parts[1],
        "patient_id": int(parts[2]),
        "level": parts[3],
    }


def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)


def compute_iou(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


# ===================================================================
# SAM2 BASELINE (automatic mask generation)
# ===================================================================

def evaluate_sam2(split_file, max_samples=None):
    """Evaluate SAM2 with automatic mask generation on LSMA-PQR.
    SAM2 uses the automatic mask generator (no prompts) and we match
    generated masks to GT classes using IoU-based assignment.
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Find SAM2 checkpoint
    sam2_checkpoints = [
        Path(r"D:\2.2 Dataset_s\LSMA-PQR\RL\MedSAM2\checkpoints\MedSAM2_latest.pt"),
    ]

    # Try to build SAM2 with a hiera config
    sam2_cfg_dir = Path(r"D:\2.2 Dataset_s\LSMA-PQR\RL\MedSAM2\sam2\configs")
    cfg_files = list(sam2_cfg_dir.glob("**/*.yaml")) if sam2_cfg_dir.exists() else []
    print(f"Found {len(cfg_files)} SAM2 config files")

    # Use SAM2 with automatic mask generation via bounding box prompts
    # from GT annotations (oracle) since SAM2 doesn't do text prompting
    from segment_anything import sam_model_registry, SamPredictor

    # Fallback: use SAM ViT-B with automatic mask generator
    sam_ckpt = Path(CFG["paths"]["pretrained"]["sam_vit_b"])
    if not sam_ckpt.exists():
        print(f"SAM checkpoint not found: {sam_ckpt}")
        return pd.DataFrame()

    # Load SAM for automatic mask generation
    from segment_anything import SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam = sam.to(dtype=torch.float32, device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    print("SAM ViT-B Auto Mask Generator loaded.")

    with open(split_file) as f:
        image_paths = [resolve_image(l) for l in f if l.strip()]
    if max_samples:
        image_paths = image_paths[:max_samples]

    mask_dir = Path(CFG["paths"]["masks"])
    results = []
    inference_times = []

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        info = parse_filename(fname)
        is_axial = info["view"] == "axial"

        img = np.array(Image.open(img_path).convert("RGB"))

        # Load ground truth mask
        mask_path = mask_dir / fname
        if not mask_path.exists():
            continue
        gt_mask = np.array(Image.open(mask_path))

        # Generate automatic masks
        t0 = time.time()
        masks = mask_generator.generate(img)
        t1 = time.time()
        inference_times.append(t1 - t0)

        num_cls = 4 if is_axial else 5

        for cls_idx in range(1, num_cls):
            gt_cls = (gt_mask == cls_idx).astype(np.float32)
            if gt_cls.sum() == 0:
                results.append({
                    "fname": fname, "patient_id": info["patient_id"],
                    "level": info["level"],
                    "plane": "axial" if is_axial else "sagittal",
                    "target_cls": cls_idx, "dice": 0.0, "iou": 0.0,
                })
                continue

            # Match generated masks to GT class by IoU
            best_dice = 0.0
            best_iou = 0.0
            gt_t = torch.from_numpy(gt_cls).float()

            for m in masks:
                pred_mask = torch.from_numpy(m["segmentation"].astype(np.float32))
                d = compute_dice(pred_mask, gt_t).item()
                iou_val = compute_iou(pred_mask, gt_t).item()
                if d > best_dice:
                    best_dice = d
                    best_iou = iou_val

            results.append({
                "fname": fname, "patient_id": info["patient_id"],
                "level": info["level"],
                "plane": "axial" if is_axial else "sagittal",
                "target_cls": cls_idx, "dice": best_dice, "iou": best_iou,
            })

        if (idx + 1) % 20 == 0:
            print(f"  SAM Auto: {idx+1}/{len(image_paths)} images, "
                  f"mean_time={np.mean(inference_times):.3f}s")

    del sam, mask_generator
    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(results), np.mean(inference_times) if inference_times else 0


# ===================================================================
# MEDSAM2 BASELINE (box-prompted medical SAM)
# ===================================================================

def evaluate_medsam2(split_file, max_samples=None):
    """Evaluate MedSAM2 with GT bounding box prompts on LSMA-PQR.
    This represents a strong medical-domain baseline with oracle prompts.
    """
    sys.path.insert(0, str(MEDSAM2_ROOT))

    # Set Hydra config search path before importing
    os.environ["SAM2_BUILD_ALLOW_ERRORS"] = "0"
    sys.path.insert(0, str(MEDSAM2_ROOT))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    sam2_cfg_dir = str(MEDSAM2_ROOT / "sam2" / "configs")
    model_cfg = "sam2.1_hiera_t512.yaml"

    # Initialize Hydra with MedSAM2 config directory
    with initialize_config_dir(config_dir=sam2_cfg_dir, version_base="1.2"):
        sam2_model = build_sam2(model_cfg, str(MEDSAM2_CKPT), device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    print("MedSAM2 loaded.")

    with open(split_file) as f:
        image_paths = [resolve_image(l) for l in f if l.strip()]
    if max_samples:
        image_paths = image_paths[:max_samples]

    mask_dir = Path(CFG["paths"]["masks"])
    results = []
    inference_times = []

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        info = parse_filename(fname)
        is_axial = info["view"] == "axial"

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        mask_path = mask_dir / fname
        if not mask_path.exists():
            continue
        gt_mask = np.array(Image.open(mask_path))

        # Set image
        with torch.inference_mode():
            predictor.set_image(img)

        num_cls = 4 if is_axial else 5

        for cls_idx in range(1, num_cls):
            gt_cls = (gt_mask == cls_idx).astype(np.uint8)
            if gt_cls.sum() == 0:
                results.append({
                    "fname": fname, "patient_id": info["patient_id"],
                    "level": info["level"],
                    "plane": "axial" if is_axial else "sagittal",
                    "target_cls": cls_idx, "dice": 0.0, "iou": 0.0,
                })
                continue

            # Get bounding box from GT mask
            ys, xs = np.where(gt_cls > 0)
            bbox = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])

            t0 = time.time()
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    box=bbox,
                    multimask_output=True,
                )
            t1 = time.time()
            inference_times.append(t1 - t0)

            # Take best mask
            best_idx = scores.argmax()
            pred = torch.from_numpy(masks[best_idx].astype(np.float32))
            gt_t = torch.from_numpy(gt_cls.astype(np.float32))

            dice = compute_dice(pred, gt_t).item()
            iou_val = compute_iou(pred, gt_t).item()

            results.append({
                "fname": fname, "patient_id": info["patient_id"],
                "level": info["level"],
                "plane": "axial" if is_axial else "sagittal",
                "target_cls": cls_idx, "dice": dice, "iou": iou_val,
            })

        if (idx + 1) % 20 == 0:
            print(f"  MedSAM2: {idx+1}/{len(image_paths)} images, "
                  f"mean_time={np.mean(inference_times):.3f}s")

    del sam2_model, predictor
    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(results), np.mean(inference_times) if inference_times else 0


# ===================================================================
# AGGREGATE METRICS
# ===================================================================

def aggregate_metrics(df, model_name, inference_time_per_img=0):
    if len(df) == 0:
        return {}

    metrics = {
        "model": model_name,
        "n_samples": len(df),
        "mean_dice": float(df["dice"].mean()),
        "std_dice": float(df["dice"].std()),
        "mean_iou": float(df["iou"].mean()),
        "std_iou": float(df["iou"].std()),
        "inference_time_per_img_sec": float(inference_time_per_img),
    }

    for plane in ["axial", "sagittal"]:
        sub = df[df["plane"] == plane]
        if len(sub) > 0:
            metrics[f"{plane}_dice"] = float(sub["dice"].mean())
            metrics[f"{plane}_iou"] = float(sub["iou"].mean())

    # Per-class dice
    for cls in sorted(df["target_cls"].unique()):
        sub = df[df["target_cls"] == cls]
        metrics[f"class_{cls}_dice"] = float(sub["dice"].mean())

    return metrics


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sam2", "medsam2"], default="medsam2")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--all", action="store_true", help="Run all baselines")
    args = parser.parse_args()

    split_file = Path(CFG["paths"]["splits_dir"]) / f"{args.split}.txt"
    results_dir = Path(CFG["paths"]["results"]) / "expanded_baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    if args.all or args.model == "sam2":
        print("\n" + "="*60)
        print("Evaluating SAM ViT-B Automatic Mask Generation")
        print("="*60)
        t0 = time.time()
        df_sam2, avg_time = evaluate_sam2(str(split_file), args.max_samples)
        total_time = time.time() - t0
        if len(df_sam2) > 0:
            df_sam2.to_csv(results_dir / f"baseline_sam_auto_{args.split}.csv", index=False)
            metrics = aggregate_metrics(df_sam2, "SAM Auto", avg_time)
            metrics["total_time_sec"] = total_time
            all_metrics.append(metrics)
            print(f"  mDice={metrics['mean_dice']:.4f}, mIoU={metrics['mean_iou']:.4f}")
            print(f"  Avg inference time: {avg_time:.3f}s/image")

    if args.all or args.model == "medsam2":
        print("\n" + "="*60)
        print("Evaluating MedSAM2 (oracle box prompts)")
        print("="*60)
        t0 = time.time()
        try:
            df_medsam2, avg_time = evaluate_medsam2(str(split_file), args.max_samples)
            total_time = time.time() - t0
            if len(df_medsam2) > 0:
                df_medsam2.to_csv(results_dir / f"baseline_medsam2_{args.split}.csv",
                                  index=False)
                metrics = aggregate_metrics(df_medsam2, "MedSAM2 (oracle box)", avg_time)
                metrics["total_time_sec"] = total_time
                all_metrics.append(metrics)
                print(f"  mDice={metrics['mean_dice']:.4f}, mIoU={metrics['mean_iou']:.4f}")
                print(f"  Avg inference time: {avg_time:.3f}s/image")
        except Exception as e:
            print(f"  MedSAM2 evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        with open(results_dir / f"expanded_baselines_{args.split}.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
