"""
model_variants.py - model variants used for the controlled ablations.

All variants reuse the original PCRadNet modules from scripts/train.py
and change exactly one factor:
    A2         original configuration (anatomy + plane + constant level index)
    noSDA      Spine Domain Adapter replaced by identity
    anatOnly   calibrator receives the anatomy term only (plane and level inputs zeroed)
    simpleEmb  calibrator replaced by one learned 256-d vector per vocabulary term
    levelTrue  calibrator receives the true disc-level index (axial D3/D4/D5 -> 0/1/2, sagittal -> 3)
    segOnly    clinical heads receive no loss (segmentation-only training)
    clinOnly   segmentation losses switched off (clinical heads only)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ORIG_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(ORIG_SCRIPTS))

import train as tp  # noqa: E402

VARIANTS = ["A2", "noSDA", "anatOnly", "simpleEmb", "levelTrue", "segOnly", "clinOnly"]
LEVEL_INDEX = {"D3": 0, "D4": 1, "D5": 2}


def level_ids_from_names(levels, device):
    return torch.tensor([LEVEL_INDEX.get(l, 3) for l in levels], dtype=torch.long, device=device)


class SimpleConditioner(nn.Module):
    """One learned calibration vector per vocabulary term (no MLP, no plane/level input)."""

    def __init__(self, ref_calibrator, hidden_dim=256):
        super().__init__()
        self.term_to_idx = dict(ref_calibrator.term_to_idx)
        self.table = nn.Embedding(20, hidden_dim)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, prompt_texts, plane_ids, level_ids=None):
        idx = torch.tensor([self.term_to_idx.get(t.lower(), 0) for t in prompt_texts],
                           device=plane_ids.device)
        return self.table(idx)


class RevCalibrator(nn.Module):
    """Wraps the original PromptCalibrator and controls which conditioning signals it sees."""

    def __init__(self, base, mode):
        super().__init__()
        self.base = base
        self.mode = mode
        self.term_to_idx = base.term_to_idx

    def forward(self, prompt_texts, plane_ids, level_ids=None):
        b = self.base
        B = len(prompt_texts)
        device = plane_ids.device
        idx = torch.tensor([b.term_to_idx.get(t.lower(), 0) for t in prompt_texts], device=device)
        text_feat = b.text_proj(b.anatomy_vocab(idx))
        if self.mode == "anatOnly":
            view_feat = torch.zeros_like(text_feat)
            level_feat = torch.zeros_like(text_feat)
        else:
            view_feat = b.view_embed(plane_ids)
            if self.mode == "levelTrue" and level_ids is not None:
                level_feat = b.level_embed(level_ids)
            else:
                level_feat = b.level_embed(torch.zeros(B, dtype=torch.long, device=device))
        return b.calibrator(torch.cat([text_feat, view_feat, level_feat], dim=-1))


class RevModel(tp.PCRadNet):
    """PCRadNet with an optional level input and swappable conditioning modules."""

    def __init__(self, cfg, variant="A2", lora_rank=8):
        super().__init__(cfg, use_calibrator=True, lora_rank=lora_rank)
        self.variant = variant
        if variant == "noSDA":
            self.domain_adapter = nn.Identity()
        if variant == "simpleEmb":
            self.calibrator = SimpleConditioner(self.calibrator)
        else:
            mode = variant if variant in ("anatOnly", "levelTrue") else "full"
            self.calibrator = RevCalibrator(self.calibrator, mode)
            if variant == "anatOnly":
                # unused inputs are frozen so the trainable count reflects what is learned
                for p in list(self.calibrator.base.view_embed.parameters()) + \
                         list(self.calibrator.base.level_embed.parameters()):
                    p.requires_grad = False

    def encode(self, images):
        features = self.encoder.forward_features(images)
        if features.shape[1] == 730:
            features = features[:, 1:, :]
        elif features.shape[1] > 729:
            features = features[:, 1:730, :]
        return self.domain_adapter(features)

    def forward(self, images, prompt_texts, plane_ids, level_ids=None, **kwargs):
        features = self.encode(images)
        calib_feat = self.calibrator(prompt_texts, plane_ids, level_ids)
        seg_logits_list, prompted_logits = self.seg_decoder(features, calib_feat, plane_ids)
        return {
            "seg_logits_list": seg_logits_list,
            "prompted_logits": prompted_logits,
            "grade_logits": self.grade_head(features),
            "height_pred": self.height_head(features),
            "disorder_logits": self.disorder_head(features),
            "calib_feat": calib_feat,
        }


def load_original_into_rev(model, state_dict):
    """Map an original PCRadNet state dict (A0-A3) onto a RevModel('A2')."""
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("calibrator."):
            mapped["calibrator.base." + k[len("calibrator."):]] = v
        else:
            mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return missing, unexpected
