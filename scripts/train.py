"""
train_peft.py — PCRadNet: Prompt-Calibrated Multi-Task Spine Segmentation
Sections 4-5, 8-10 of the project plan.

Usage:
    python scripts/train_peft.py --validate-model
    python scripts/train_peft.py --phase=1
    python scripts/train_peft.py --phase=2 --ablation=A2
"""
import os, sys, json, argparse, time, hashlib, gc, datetime, copy, shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── Seed everything ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Config ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
with open(SCRIPT_DIR / "config.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(CFG["hardware"]["gpu_fraction"])

# ═════════════════════════════════════════════════════════════════════════
# SECTION A — DATASET
# ═════════════════════════════════════════════════════════════════════════
AXIAL_CLASSES = CFG["classes"]["axial"]
SAG_CLASSES   = CFG["classes"]["sagittal"]

PROMPT_BANK = {
    "axial": {
        1: ["intervertebral disc", "IVD", "disc"],
        2: ["posterior element", "posterior arch", "facet joint"],
        3: ["spinal canal", "neural canal", "canal"],
    },
    "sagittal": {
        1: ["vertebral body", "vertebra"],
        2: ["intervertebral disc", "disc"],
        3: ["sacrum", "sacral bone"],
        4: ["spinal canal", "canal"],
    }
}


def parse_yolo_polygons(label_path, img_h, img_w):
    """Parse YOLO polygon format → list of (class_id, polygon_points)."""
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            pts = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i+1] * img_h
                pts.append([int(x), int(y)])
            polygons.append((cls_id, np.array(pts, dtype=np.int32)))
    return polygons


def polygons_to_mask(polygons, h, w, num_classes):
    """Convert polygon list to indexed mask [H, W].
    YOLO labels use 0-based class IDs; masks use 1-based (0 = background).
    """
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    for cls_id, pts in polygons:
        if cls_id < num_classes and len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(cls_id + 1))  # +1: reserve 0 for BG
    return mask


class SpineDataset(Dataset):
    """Multi-task spine MRI dataset with prompt conditioning."""

    def __init__(self, split_file, cfg, transforms=None, use_metadata=True):
        self.cfg = cfg
        self.transforms = transforms
        self.img_dir = Path(cfg["paths"]["images"])
        self.lbl_dir = Path(cfg["paths"]["labels"])
        self.mask_dir = Path(cfg["paths"]["masks"])

        # Load file list
        with open(split_file) as f:
            self.image_paths = [l.strip() for l in f if l.strip()]

        # Load metadata
        self.pfirrmann = {}
        self.ivd_heights = {}
        self.disorders = {}
        if use_metadata:
            self._load_metadata()

    def _load_metadata(self):
        root = Path(self.cfg["paths"]["dataset_root"])
        # Pfirrmann grades
        df_p = pd.read_csv(root / "PfirrmannGrade.csv")
        for _, row in df_p.iterrows():
            pid = int(row["Patient_ID"])
            self.pfirrmann[pid] = {
                "D3": int(row["D3"]), "D4": int(row["D4"]), "D5": int(row["D5"])
            }
        # IVD Heights
        df_h = pd.read_csv(root / "IVDHeights.csv")
        for _, row in df_h.iterrows():
            pid = int(row["Patient_ID"])
            self.ivd_heights[pid] = {
                "D3": float(row["D3_Ht"]), "D4": float(row["D4_Ht"]), "D5": float(row["D5_Ht"])
            }
        # Structured notes (disorders)
        df_n = pd.read_csv(root / "structured_notes.csv")
        disorder_cols = ["Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
                         "Thecal_Sac_Compression", "Nerve_Root_Compression",
                         "Spinal_Canal_Stenosis", "Spondylolisthesis",
                         "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"]
        for _, row in df_n.iterrows():
            pid = int(row["Patient_ID"])
            self.disorders[pid] = [int(row[c]) for c in disorder_cols]

    def __len__(self):
        return len(self.image_paths)

    def _parse_filename(self, fname):
        """Parse: {view}_{seq}_{pid}_{level}.png"""
        stem = fname.replace(".png", "")
        parts = stem.split("_")
        return {
            "view": parts[0],      # axial / sag
            "sequence": parts[1],   # T1 / T2
            "patient_id": int(parts[2]),
            "level": parts[3],      # D3/D4/D5 or S*
        }

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        info = self._parse_filename(fname)

        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        # Load indexed mask
        mask_path = self.mask_dir / fname
        if mask_path.exists():
            mask = np.array(Image.open(mask_path))
        else:
            # Fallback: build from YOLO labels
            lbl_path = self.lbl_dir / fname.replace(".png", ".txt")
            n_cls = 4 if info["view"] == "axial" else 5
            polygons = parse_yolo_polygons(str(lbl_path), h, w)
            mask = polygons_to_mask(polygons, h, w, n_cls)

        # Resize to 384x384 if needed
        if h != 384 or w != 384:
            from PIL import Image as PILImage
            img = np.array(PILImage.fromarray(img).resize((384, 384), PILImage.BILINEAR))
            mask = np.array(PILImage.fromarray(mask).resize((384, 384), PILImage.NEAREST))

        # Apply augmentations
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # Determine plane info
        is_axial = info["view"] == "axial"
        num_classes = 4 if is_axial else 5
        plane_id = 0 if is_axial else 1

        # Create one-hot mask [C, H, W]
        mask_onehot = torch.zeros(num_classes, 384, 384, dtype=torch.float32)
        for c in range(num_classes):
            mask_onehot[c] = (mask == c).float()

        # Metadata
        pid = info["patient_id"]
        level = info["level"]

        # Pfirrmann grade (only for D3/D4/D5 levels)
        grade = -1
        if level in ["D3", "D4", "D5"] and pid in self.pfirrmann:
            grade = self.pfirrmann[pid][level]

        # IVD height
        height = -1.0
        if level in ["D3", "D4", "D5"] and pid in self.ivd_heights:
            height = self.ivd_heights[pid][level]

        # Disorders (patient-level)
        disorder = torch.zeros(9, dtype=torch.float32)
        if pid in self.disorders:
            disorder = torch.tensor(self.disorders[pid], dtype=torch.float32)

        # Random text prompt for the dominant foreground class
        fg_classes = [c for c in range(1, num_classes) if mask_onehot[c].sum() > 100]
        if fg_classes:
            target_cls = random.choice(fg_classes)
        else:
            target_cls = random.randint(1, num_classes - 1)

        view_key = "axial" if is_axial else "sagittal"
        prompts = PROMPT_BANK[view_key].get(target_cls, ["structure"])
        prompt_text = random.choice(prompts)

        # Binary mask for prompted class
        prompted_mask = mask_onehot[target_cls]

        return {
            "image": img,                  # [3, 384, 384]
            "mask": mask,                  # [H, W] indexed
            "mask_onehot": mask_onehot,    # [C, H, W]
            "prompted_mask": prompted_mask, # [H, W]
            "prompt_text": prompt_text,
            "target_cls": target_cls,
            "plane_id": plane_id,
            "num_classes": num_classes,
            "grade": grade,                # 1-5 or -1
            "height": height,             # pixels or -1
            "disorder": disorder,          # [9]
            "patient_id": pid,
            "level": level,
            "fname": fname,
        }


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def collate_fn(batch):
    """Custom collate for variable-class masks."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if k in ["prompt_text", "fname", "level"]:
            result[k] = [b[k] for b in batch]
        elif k in ["mask_onehot"]:
            # Pad to max classes in batch
            max_c = max(b[k].shape[0] for b in batch)
            padded = []
            for b in batch:
                c = b[k].shape[0]
                if c < max_c:
                    pad = torch.zeros(max_c - c, 384, 384)
                    padded.append(torch.cat([b[k], pad], dim=0))
                else:
                    padded.append(b[k])
            result[k] = torch.stack(padded)
        elif isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        else:
            result[k] = torch.tensor([b[k] for b in batch])
    return result


# ═════════════════════════════════════════════════════════════════════════
# SECTION B — MODEL: PCRadNet
# ═════════════════════════════════════════════════════════════════════════

class SpineDomainAdapter(nn.Module):
    """Lightweight bottleneck adapter for frozen ViT features.
    h' = h + W_up(GELU(W_down(LN(h))))
    """
    def __init__(self, d_model, bottleneck_dim=32):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(self.ln(x))))


class PromptCalibrator(nn.Module):
    """Novel: Maps text prompt + view/level tokens to prompt embedding adjustment.
    Given anatomy text prompt, produces spatial bias for segmentation decoder.
    """
    def __init__(self, text_dim=768, hidden_dim=256, num_views=2, num_levels=6):
        super().__init__()
        # Simple text encoder (learned embeddings for known anatomy terms)
        self.anatomy_vocab = nn.Embedding(20, text_dim)  # 20 anatomy terms
        self.view_embed = nn.Embedding(num_views, hidden_dim)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Calibration: text_feat → spatial attention bias
        self.calibrator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Anatomy term → index mapping (built at init)
        self.term_to_idx = {}
        all_terms = []
        for view_prompts in PROMPT_BANK.values():
            for cls_prompts in view_prompts.values():
                all_terms.extend(cls_prompts)
        for i, t in enumerate(sorted(set(all_terms))):
            self.term_to_idx[t.lower()] = min(i, 19)

    def forward(self, prompt_texts, plane_ids):
        """
        prompt_texts: list of strings
        plane_ids: [B] tensor
        Returns: calibration_feat [B, hidden_dim]
        """
        B = len(prompt_texts)
        device = plane_ids.device

        # Encode text prompts
        text_indices = []
        for t in prompt_texts:
            idx = self.term_to_idx.get(t.lower(), 0)
            text_indices.append(idx)
        text_idx = torch.tensor(text_indices, device=device)
        text_feat = self.text_proj(self.anatomy_vocab(text_idx))  # [B, hidden_dim]

        # View embedding
        view_feat = self.view_embed(plane_ids)  # [B, hidden_dim]

        # Level embedding (use 0 as default)
        level_feat = self.level_embed(torch.zeros(B, dtype=torch.long, device=device))

        # Calibrate
        combined = torch.cat([text_feat, view_feat, level_feat], dim=-1)
        calib = self.calibrator(combined)  # [B, hidden_dim]
        return calib


class SegmentationDecoder(nn.Module):
    """ConvTranspose decoder from ViT features to segmentation masks."""
    def __init__(self, in_dim=768, hidden_dim=256, max_classes=5):
        super().__init__()
        # Project ViT tokens to spatial features
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )

        # Prompt-conditioned spatial attention
        self.prompt_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Decoder blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        # Class heads (separate for axial/sagittal, + prompted binary head)
        self.axial_head = nn.Conv2d(16, 4, kernel_size=1)
        self.sag_head = nn.Conv2d(16, 5, kernel_size=1)
        self.prompted_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, vit_features, calib_feat, plane_ids):
        """
        vit_features: [B, N, D] ViT patch tokens
        calib_feat: [B, hidden_dim] from PromptCalibrator
        plane_ids: [B] (0=axial, 1=sagittal)
        Returns: seg_logits [B, C, H, W], prompted_logits [B, 1, H, W]
        """
        B, N, D = vit_features.shape
        h = w = int(N ** 0.5)  # Should be 27 for 384/14

        # Project
        x = self.proj(vit_features)  # [B, N, hidden_dim]

        # Apply prompt calibration as spatial gate
        gate = self.prompt_gate(calib_feat).unsqueeze(1)  # [B, 1, hidden_dim]
        x = x * (1.0 + gate)  # Multiplicative modulation

        # Reshape to spatial
        x = x.permute(0, 2, 1).reshape(B, -1, h, w)  # [B, hidden_dim, h, w]

        # Decode
        x = self.decoder(x)  # [B, 16, h*16, w*16]

        # Crop to 384x384 if needed
        if x.shape[-1] != 384:
            x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False)

        # Prompted head (always)
        prompted_logits = self.prompted_head(x)  # [B, 1, 384, 384]

        # Multi-class head per plane
        seg_logits_list = []
        for i in range(B):
            if plane_ids[i] == 0:
                seg_logits_list.append(self.axial_head(x[i:i+1]))
            else:
                seg_logits_list.append(self.sag_head(x[i:i+1]))

        return seg_logits_list, prompted_logits


class GradeHead(nn.Module):
    """Ordinal Pfirrmann grading head (1-5)."""
    def __init__(self, in_dim=768):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # 4 cumulative thresholds for ordinal regression
        )

    def forward(self, vit_features):
        """vit_features: [B, N, D] → grade_logits [B, 4]"""
        x = vit_features.mean(dim=1)  # [B, D]
        return self.head(x)


class HeightHead(nn.Module):
    """IVD height regression head."""
    def __init__(self, in_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, vit_features):
        """vit_features: [B, N, D] → height [B, 1]"""
        x = vit_features.mean(dim=1)
        return self.head(x).squeeze(-1)


class DisorderHead(nn.Module):
    """Multi-label disorder classification."""
    def __init__(self, in_dim=768, num_disorders=9):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_disorders),
        )

    def forward(self, vit_features):
        """vit_features: [B, N, D] → disorder_logits [B, 9]"""
        x = vit_features.mean(dim=1)
        return self.head(x)


class PCRadNet(nn.Module):
    """
    PCRadNet: Prompt-Calibrated PEFT for Multi-Task Spine Segmentation.

    Novel contributions:
    [C1] Prompt Calibrator — text anatomy prompt → spatial attention modulation
    [C2] Multi-task: segmentation + grading + height + disorder in one forward pass
    [C3] Modality-agnostic via SpineDomainAdapter (swap per modality)
    [C5] Ordinal Pfirrmann consistency loss
    """

    def __init__(self, cfg, use_calibrator=True, lora_rank=8):
        super().__init__()
        self.cfg = cfg
        self.use_calibrator = use_calibrator
        self.lora_rank = lora_rank

        # ── Module 1: Vision Encoder (DINOv2 — FROZEN + LoRA + Adapter) ──
        import timm
        self.encoder = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=384,
            num_classes=0,
        )
        self.encoder_dim = 768

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Apply LoRA
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.05,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()

        # Enable gradient checkpointing
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(True)
        elif hasattr(self.encoder, 'base_model'):
            if hasattr(self.encoder.base_model, 'set_grad_checkpointing'):
                self.encoder.base_model.set_grad_checkpointing(True)

        # SpineDomainAdapter (inserted into encoder)
        adapter_rank = cfg["model"].get("adapter_rank", 32)
        self.domain_adapter = SpineDomainAdapter(self.encoder_dim, adapter_rank)

        # ── Module 2: Prompt Calibrator ──
        self.calibrator = PromptCalibrator(text_dim=self.encoder_dim, hidden_dim=256)
        if not use_calibrator:
            for p in self.calibrator.parameters():
                p.requires_grad = False

        # ── Module 3: Multi-Task Decoder ──
        self.seg_decoder = SegmentationDecoder(
            in_dim=self.encoder_dim, hidden_dim=256, max_classes=5
        )
        self.grade_head = GradeHead(self.encoder_dim)
        self.height_head = HeightHead(self.encoder_dim)
        self.disorder_head = DisorderHead(self.encoder_dim, 9)

    def forward(self, images, prompt_texts, plane_ids, **kwargs):
        """
        images: [B, 3, 384, 384]
        prompt_texts: list[str]
        plane_ids: [B] (0=axial, 1=sag)
        """
        # Encode image → ViT features
        features = self.encoder.forward_features(images)  # [B, N+1, D] or [B, N, D]

        # Remove CLS token if present
        if features.shape[1] == 730:  # 729 patches + 1 CLS for 384/14
            features = features[:, 1:, :]  # [B, 729, 768]
        elif features.shape[1] > 729:
            features = features[:, 1:730, :]

        # Apply SpineDomainAdapter
        features = self.domain_adapter(features)

        # Prompt Calibrator
        calib_feat = self.calibrator(prompt_texts, plane_ids)

        # Segmentation
        seg_logits_list, prompted_logits = self.seg_decoder(features, calib_feat, plane_ids)

        # Multi-task heads
        grade_logits = self.grade_head(features)
        height_pred = self.height_head(features)
        disorder_logits = self.disorder_head(features)

        return {
            "seg_logits_list": seg_logits_list,
            "prompted_logits": prompted_logits,
            "grade_logits": grade_logits,
            "height_pred": height_pred,
            "disorder_logits": disorder_logits,
            "features": features,
        }

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        return {"trainable": trainable, "frozen": frozen, "total": total}


# ═════════════════════════════════════════════════════════════════════════
# SECTION C — LOSSES
# ═════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Sobel edge-based boundary loss."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        pred_soft = torch.sigmoid(pred)
        if pred_soft.dim() == 3:
            pred_soft = pred_soft.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1).float()

        pred_edge_x = F.conv2d(pred_soft, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_soft, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)

        gt_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        gt_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-8)

        return F.mse_loss(pred_edge, gt_edge)


class OrdinalLoss(nn.Module):
    """CORN-style ordinal regression loss for Pfirrmann grading."""
    def __init__(self, num_ranks=5):
        super().__init__()
        self.num_ranks = num_ranks

    def forward(self, logits, targets):
        """
        logits: [B, num_ranks-1] cumulative thresholds
        targets: [B] integer grades 1-5
        """
        valid = targets > 0  # -1 means no label
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid]
        targets = targets[valid] - 1  # 0-indexed: 0-4

        # Binary targets for cumulative model
        binary_targets = torch.zeros_like(logits)
        for i in range(logits.shape[1]):
            binary_targets[:, i] = (targets > i).float()

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)

        # Ordinal consistency penalty: ensure monotonic predictions
        probs = torch.sigmoid(logits)
        consistency = F.relu(probs[:, 1:] - probs[:, :-1]).mean()

        return loss + 0.1 * consistency


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for PCRadNet."""
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.cfg = cfg
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.ordinal_loss = OrdinalLoss()

        # Class weights for BCE
        self.class_weights = class_weights
        self.task_weights = cfg["loss"]["task_weights"]

    def forward(self, outputs, batch):
        losses = {}
        device = batch["image"].device

        # ── Segmentation loss (per-sample, handles variable classes) ──
        seg_loss = torch.tensor(0.0, device=device)
        total_seg = 0
        for i, seg_logits in enumerate(outputs["seg_logits_list"]):
            gt = batch["mask_onehot"][i:i+1, :seg_logits.shape[1]]
            # Determine plane for class-weight lookup
            plane_key = "axial" if batch["plane_id"][i] == 0 else "sagittal"
            # Dice + weighted BCE for each foreground class
            for c in range(1, seg_logits.shape[1]):  # skip background
                pred_c = seg_logits[0, c:c+1]
                gt_c = gt[0, c:c+1]
                seg_loss = seg_loss + self.dice_loss(pred_c.unsqueeze(0), gt_c.unsqueeze(0))
                # Apply class weight as pos_weight to counteract class imbalance
                pw = 1.0
                if self.class_weights and plane_key in self.class_weights:
                    pw = float(self.class_weights[plane_key].get(str(c), 1.0))
                pos_w = pred_c.new_tensor([pw])  # same device & dtype as logits
                seg_loss = seg_loss + F.binary_cross_entropy_with_logits(
                    pred_c, gt_c, pos_weight=pos_w
                )
                total_seg += 1

        if total_seg > 0:
            seg_loss = seg_loss / total_seg
        losses["seg"] = seg_loss

        # ── Prompted mask loss ──
        prompted_loss = self.dice_loss(
            outputs["prompted_logits"].squeeze(1),
            batch["prompted_mask"]
        ) + F.binary_cross_entropy_with_logits(
            outputs["prompted_logits"].squeeze(1),
            batch["prompted_mask"]
        )
        losses["prompted"] = prompted_loss

        # ── Boundary loss ──
        boundary_loss = self.boundary_loss(
            outputs["prompted_logits"].squeeze(1),
            batch["prompted_mask"]
        )
        losses["boundary"] = boundary_loss

        # ── Grading loss ──
        grade_loss = self.ordinal_loss(outputs["grade_logits"], batch["grade"])
        losses["grade"] = grade_loss

        # ── Height loss ──
        valid_ht = batch["height"] > 0
        if valid_ht.sum() > 0:
            ht_loss = F.smooth_l1_loss(
                outputs["height_pred"][valid_ht],
                batch["height"][valid_ht].float()
            )
        else:
            ht_loss = torch.tensor(0.0, device=device)
        losses["height"] = ht_loss

        # ── Disorder loss ──
        disorder_loss = F.binary_cross_entropy_with_logits(
            outputs["disorder_logits"],
            batch["disorder"]
        )
        losses["disorder"] = disorder_loss

        # ── Total weighted loss ──
        tw = self.task_weights
        total = (tw["seg"] * (seg_loss + prompted_loss) +
                 self.cfg["loss"]["boundary_lambda"] * boundary_loss +
                 tw["grade"] * grade_loss +
                 tw["height"] * ht_loss +
                 tw["disorder"] * disorder_loss)
        losses["total"] = total

        return losses


# ═════════════════════════════════════════════════════════════════════════
# SECTION D — METRICS
# ═════════════════════════════════════════════════════════════════════════

def compute_dice(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    return (2 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)


def compute_iou(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model on dataloader. Returns metrics dict."""
    model.eval()
    metrics = defaultdict(list)
    total_loss = 0
    n = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        plane_ids = batch["plane_id"].to(device)

        for k in ["mask", "mask_onehot", "prompted_mask", "grade", "height", "disorder"]:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        outputs = model(images, batch["prompt_text"], plane_ids)
        losses = loss_fn(outputs, batch)
        total_loss += losses["total"].item() * images.shape[0]
        n += images.shape[0]

        # Per-sample prompted mask metrics
        for i in range(images.shape[0]):
            pred = torch.sigmoid(outputs["prompted_logits"][i, 0])
            gt = batch["prompted_mask"][i]
            d = compute_dice(pred, gt).item()
            iou = compute_iou(pred, gt).item()
            metrics["dice"].append(d)
            metrics["iou"].append(iou)

        # Grading accuracy
        valid_g = batch["grade"] > 0
        if valid_g.sum() > 0:
            probs = torch.sigmoid(outputs["grade_logits"][valid_g])
            pred_grades = (probs > 0.5).sum(dim=1) + 1
            gt_grades = batch["grade"][valid_g]
            correct = (pred_grades == gt_grades).float()
            within1 = ((pred_grades - gt_grades).abs() <= 1).float()
            metrics["grade_acc"].extend(correct.tolist())
            metrics["grade_within1"].extend(within1.tolist())

        # Height MAE
        valid_h = batch["height"] > 0
        if valid_h.sum() > 0:
            mae = (outputs["height_pred"][valid_h] - batch["height"][valid_h].float()).abs()
            metrics["height_mae"].extend(mae.tolist())

    result = {
        "loss": total_loss / max(n, 1),
        "dice": np.mean(metrics["dice"]) if metrics["dice"] else 0,
        "iou": np.mean(metrics["iou"]) if metrics["iou"] else 0,
        "grade_acc": np.mean(metrics["grade_acc"]) if metrics["grade_acc"] else 0,
        "grade_within1": np.mean(metrics["grade_within1"]) if metrics["grade_within1"] else 0,
        "height_mae": np.mean(metrics["height_mae"]) if metrics["height_mae"] else 0,
        "n_samples": n,
    }
    return result


# ═════════════════════════════════════════════════════════════════════════
# SECTION E — TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════

def train_phase(model, train_loader, val_loader, cfg, phase=1, ablation="A2"):
    """Train PC-RadNet for a given phase."""
    device = DEVICE
    model = model.to(device)

    epochs = cfg["training"][f"phase{phase}_epochs"]
    lr = cfg["training"]["lr"] if phase == 1 else cfg["training"]["lr_phase2"]
    grad_accum = cfg["training"]["grad_accum"]
    grad_clip = cfg["training"]["grad_clip"]

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    # Loss
    class_weights_path = Path(cfg["paths"]["results"]) / "class_weights.json"
    class_weights = None
    if class_weights_path.exists():
        with open(class_weights_path) as f:
            class_weights = json.load(f)
    loss_fn = MultiTaskLoss(cfg, class_weights).to(device)

    # WandB
    try:
        import wandb
        run_name = f"phase{phase}_{ablation}_r{cfg['model']['lora_rank']}_seed{SEED}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=cfg["wandb"]["project"],
            name=run_name,
            config={
                "phase": phase,
                "ablation": ablation,
                "lr": lr,
                "epochs": epochs,
                "batch_size": cfg["training"]["batch_size"],
                "lora_rank": cfg["model"]["lora_rank"],
                "seed": SEED,
            },
        )
        use_wandb = True
    except Exception as e:
        print(f"WandB init failed: {e}")
        use_wandb = False

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0
    best_iou = 0
    patience = cfg["training"]["early_stop_patience"]
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_dice = []
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            plane_ids = batch["plane_id"].to(device)

            for k in ["mask", "mask_onehot", "prompted_mask", "grade", "height", "disorder"]:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images, batch["prompt_text"], plane_ids)
                losses = loss_fn(outputs, batch)
                loss = losses["total"] / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Track metrics
            for k, v in losses.items():
                epoch_losses[k] += v.item()

            # Dice for prompted mask
            with torch.no_grad():
                pred = torch.sigmoid(outputs["prompted_logits"].squeeze(1))
                for i in range(images.shape[0]):
                    d = compute_dice(pred[i], batch["prompted_mask"][i]).item()
                    epoch_dice.append(d)

            n_batches += 1

            # Print progress
            if (batch_idx + 1) % 20 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                print(f"  [Ep {epoch}/{epochs} | Batch {batch_idx+1:04d}/{len(train_loader)}] "
                      f"loss={losses['total'].item():.4f} dice={np.mean(epoch_dice[-20:]):.3f} "
                      f"gpu={gpu_mem:.2f}GB lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # ── Epoch summary ──
        epoch_time = time.time() - t0
        train_dice = np.mean(epoch_dice) if epoch_dice else 0
        avg_loss = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # ── Validation ──
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9

        print(f"[Epoch {epoch}/{epochs}] "
              f"train_loss={avg_loss.get('total', 0):.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
              f"val_iou={val_metrics['iou']:.4f} | "
              f"grade_acc={val_metrics['grade_acc']:.3f} ht_mae={val_metrics['height_mae']:.2f} | "
              f"gpu_peak={gpu_mem:.2f}GB time={epoch_time:.0f}s")

        # WandB log
        if use_wandb:
            log_d = {
                "epoch": epoch,
                "train/loss_total": avg_loss.get("total", 0),
                "train/loss_seg": avg_loss.get("seg", 0),
                "train/loss_grade": avg_loss.get("grade", 0),
                "train/dice": train_dice,
                "val/loss": val_metrics["loss"],
                "val/dice": val_metrics["dice"],
                "val/iou": val_metrics["iou"],
                "val/grade_acc": val_metrics["grade_acc"],
                "val/grade_within1": val_metrics["grade_within1"],
                "val/height_mae": val_metrics["height_mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "gpu_mem_peak_gb": gpu_mem,
                "epoch_time_sec": epoch_time,
            }
            wandb.log(log_d)

        # ── Checkpointing ──
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "train_dice": train_dice,
            "grade_acc": val_metrics["grade_acc"],
            "height_mae": val_metrics["height_mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "timestamp": str(datetime.datetime.now()),
            "vram_peak_gb": gpu_mem,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "seed": SEED,
            "phase": phase,
            "ablation": ablation,
        }

        # Save every epoch
        torch.save(ckpt_data, ckpt_dir / f"phase{phase}_{ablation}_epoch_{epoch:02d}.pt")

        # Best by dice
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(ckpt_data, ckpt_dir / f"best_by_mdice_{ablation}.pt")
            patience_counter = 0
            print(f"  * New best dice: {best_dice:.4f}")
        else:
            patience_counter += 1

        # Best by IoU
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(ckpt_data, ckpt_dir / f"best_by_miou_{ablation}.pt")

        # Early stopping
        if patience_counter >= patience and phase == 2:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

        # Quality Gate 3: IoU > 0 by epoch 2
        if epoch == 2 and phase == 1 and val_metrics["iou"] < 0.01:
            print("[WARN] QUALITY GATE 3 FAIL: val_iou ~ 0 at epoch 2")
            print("Executing failure analysis...")
            failure_analysis(model, val_loader, loss_fn, device)

    # Save last
    torch.save(ckpt_data, ckpt_dir / f"last_epoch_{ablation}.pt")

    if use_wandb:
        wandb.finish()

    print(f"\nPhase {phase} complete. Best dice={best_dice:.4f}, Best IoU={best_iou:.4f}")
    return best_dice, best_iou


# ═════════════════════════════════════════════════════════════════════════
# SECTION F — FAILURE ANALYSIS (Section 9)
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def failure_analysis(model, val_loader, loss_fn, device):
    """Diagnose training failures."""
    model.eval()
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS — Section 9")
    print("=" * 60)

    # D3: Gradient flow check
    model.train()
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    plane_ids = batch["plane_id"].to(device)
    for k in ["mask", "mask_onehot", "prompted_mask", "grade", "height", "disorder"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    outputs = model(images, batch["prompt_text"], plane_ids)
    losses = loss_fn(outputs, batch)
    losses["total"].backward()

    print("\nD3. GRADIENT FLOW:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            gn = p.grad.norm().item() if p.grad is not None else 0.0
            if gn == 0:
                print(f"  [WARN] ZERO GRAD: {n}")
            elif gn < 1e-6:
                print(f"  ⚠ LOW GRAD: {n} | grad_norm={gn:.8f}")

    # D4: Prediction collapse check
    model.eval()
    print("\nD4. PREDICTION COLLAPSE CHECK:")
    for batch in [next(iter(val_loader))]:
        images = batch["image"].to(device)
        plane_ids = batch["plane_id"].to(device)
        outputs = model(images, batch["prompt_text"], plane_ids)
        pred = torch.sigmoid(outputs["prompted_logits"])
        print(f"  Prompted mask: unique={pred.unique()[:10]}, sum={pred.sum().item():.1f}")
        for i, seg in enumerate(outputs["seg_logits_list"]):
            sm = torch.sigmoid(seg)
            print(f"  Seg[{i}]: unique={sm.unique()[:5]}, sum={sm.sum().item():.1f}")

    print("\n" + "=" * 60)


# ═════════════════════════════════════════════════════════════════════════
# SECTION G — MAIN
# ═════════════════════════════════════════════════════════════════════════

def validate_model_only():
    """Print model architecture, parameter counts, VRAM estimate."""
    model = PCRadNet(CFG, use_calibrator=True, lora_rank=CFG["model"]["lora_rank"])
    counts = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"PC-RadNet Model Summary")
    print(f"{'='*60}")
    print(f"Trainable params : {counts['trainable']:>12,}")
    print(f"Frozen params    : {counts['frozen']:>12,}")
    print(f"Total params     : {counts['total']:>12,}")
    print(f"Trainable ratio  : {100*counts['trainable']/counts['total']:.2f}%")

    # VRAM estimate with dummy forward pass
    model = model.to(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    dummy_img = torch.randn(2, 3, 384, 384, device=DEVICE)
    dummy_plane = torch.tensor([0, 1], device=DEVICE)
    with autocast("cuda", dtype=torch.bfloat16):
        out = model(dummy_img, ["disc", "vertebral body"], dummy_plane)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM (BS=2, bf16): {vram:.2f} GB")
    print(f"Within budget     : {'YES' if vram < 6.4 else 'NO'} (limit=6.4GB)")

    # Output shapes
    print(f"\nOutput shapes:")
    print(f"  seg_logits_list: {len(out['seg_logits_list'])} items")
    for i, s in enumerate(out["seg_logits_list"]):
        print(f"    [{i}] {s.shape}")
    print(f"  prompted_logits: {out['prompted_logits'].shape}")
    print(f"  grade_logits: {out['grade_logits'].shape}")
    print(f"  height_pred: {out['height_pred'].shape}")
    print(f"  disorder_logits: {out['disorder_logits'].shape}")

    del model, dummy_img
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="PC-RadNet Training")
    parser.add_argument("--validate-model", action="store_true")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ablation", type=str, default="A2",
                        choices=["A0", "A1", "A2", "A3"])
    args = parser.parse_args()

    if args.validate_model:
        validate_model_only()
        return

    # ── Build dataloaders ────────────────────────────────────────────
    split_dir = Path(CFG["paths"]["scripts"]) / "data"
    train_ds = SpineDataset(split_dir / "train.txt", CFG, transforms=get_train_transforms())
    val_ds = SpineDataset(split_dir / "val.txt", CFG, transforms=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=CFG["training"]["batch_size"],
        shuffle=True, num_workers=CFG["hardware"]["num_workers"],
        pin_memory=CFG["hardware"]["pin_memory"],
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["training"]["batch_size"],
        shuffle=False, num_workers=CFG["hardware"]["num_workers"],
        pin_memory=CFG["hardware"]["pin_memory"],
        collate_fn=collate_fn,
    )

    print(f"Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} images, {len(val_loader)} batches")

    # ── Configure ablation ───────────────────────────────────────────
    use_calibrator = args.ablation in ["A2", "A3"]
    lora_rank = CFG["model"]["lora_rank"] if args.ablation != "A0" else 0

    print(f"\nAblation: {args.ablation}")
    print(f"  Use calibrator: {use_calibrator}")
    print(f"  LoRA rank: {lora_rank}")

    # ── Build model ──────────────────────────────────────────────────
    model = PCRadNet(CFG, use_calibrator=use_calibrator, lora_rank=lora_rank)
    counts = model.count_parameters()
    print(f"  Trainable: {counts['trainable']:,}")

    # ── Load Phase-1 checkpoint for Phase-2 ──────────────────────────
    if args.phase == 2:
        ckpt_path = Path(CFG["paths"]["checkpoints"]) / f"best_by_mdice_{args.ablation}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  [OK] Loaded Phase-1 checkpoint: {ckpt_path.name} "
                  f"(epoch {ckpt.get('epoch', '?')}, dice={ckpt.get('val_dice', '?'):.4f})")
            del ckpt
        else:
            print(f"  [WARN] No Phase-1 checkpoint found at {ckpt_path}; training Phase-2 from scratch.")

    # ── Train ────────────────────────────────────────────────────────
    best_dice, best_iou = train_phase(
        model, train_loader, val_loader, CFG,
        phase=args.phase, ablation=args.ablation,
    )

    # Save final metrics
    results_path = Path(CFG["paths"]["results"]) / f"phase{args.phase}_{args.ablation}_metrics.json"
    with open(results_path, "w") as f:
        json.dump({
            "phase": args.phase,
            "ablation": args.ablation,
            "best_dice": best_dice,
            "best_iou": best_iou,
            "trainable_params": counts["trainable"],
            "total_params": counts["total"],
        }, f, indent=2)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
