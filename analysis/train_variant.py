"""
train_variant.py - trains one model variant with the original two-phase protocol.

Protocol (identical to scripts/train.py):
    Phase 1: 5 epochs, lr 1e-4; Phase 2: up to 50 epochs, lr 5e-5, early stopping
    (patience 10) on validation prompted Dice; AdamW (wd 0.01), batch 2 x 8 accumulation,
    bf16 autocast, CosineAnnealingWarmRestarts(T0=10, Tmult=2), grad clip 1.0.
Only the validation split is evaluated during training. The test split is never read.

Usage:
    python train_variant.py --variant noSDA --seed 42
"""
import os
import sys
import json
import time
import copy
import random
import argparse
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_variants import tp, RevModel, VARIANTS, level_ids_from_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints" / "variants"
LOG_ROOT = ROOT / "results" / "analysis" / "train_logs"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cfg(variant):
    cfg = copy.deepcopy(tp.CFG)
    tw = cfg["loss"]["task_weights"]
    if variant == "segOnly":
        tw["grade"] = tw["height"] = tw["disorder"] = 0.0
    if variant == "clinOnly":
        tw["seg"] = 0.0
        cfg["loss"]["boundary_lambda"] = 0.0
    return cfg


def to_device(batch, device):
    for k in ["mask", "mask_onehot", "prompted_mask", "grade", "height", "disorder"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    m = defaultdict(list)
    tot, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        plane_ids = batch["plane_id"].to(device)
        level_ids = level_ids_from_names(batch["level"], device)
        batch = to_device(batch, device)
        out = model(images, batch["prompt_text"], plane_ids, level_ids)
        losses = loss_fn(out, batch)
        tot += losses["total"].item() * images.shape[0]
        n += images.shape[0]
        pr = torch.sigmoid(out["prompted_logits"].float())
        for i in range(images.shape[0]):
            m["dice"].append(tp.compute_dice(pr[i, 0], batch["prompted_mask"][i]).item())
            m["iou"].append(tp.compute_iou(pr[i, 0], batch["prompted_mask"][i]).item())
        vg = batch["grade"] > 0
        if vg.sum() > 0:
            pg = (torch.sigmoid(out["grade_logits"][vg].float()) > 0.5).sum(1) + 1
            m["grade_acc"].extend((pg == batch["grade"][vg]).float().tolist())
        vh = batch["height"] > 0
        if vh.sum() > 0:
            m["height_mae"].extend((out["height_pred"][vh].float() - batch["height"][vh].float()).abs().tolist())
    return {"loss": tot / max(n, 1),
            **{k: float(np.mean(v)) for k, v in m.items()}}


def run_phase(model, phase, train_loader, val_loader, cfg, tag, monitor, log):
    device = tp.DEVICE
    epochs = cfg["training"][f"phase{phase}_epochs"]
    lr = cfg["training"]["lr"] if phase == 1 else cfg["training"]["lr_phase2"]
    accum = cfg["training"]["grad_accum"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    scaler = GradScaler()
    with open(ROOT / "results" / "class_weights.json") as f:
        cw = json.load(f)
    loss_fn = tp.MultiTaskLoss(cfg, cw).to(device)

    best, bad = -1e9, 0
    best_path = CKPT_ROOT / f"{tag}_best.pt"
    phase1_path = CKPT_ROOT / f"{tag}_phase1_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        run_loss, nb = 0.0, 0
        for bi, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            plane_ids = batch["plane_id"].to(device)
            level_ids = level_ids_from_names(batch["level"], device)
            batch = to_device(batch, device)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(images, batch["prompt_text"], plane_ids, level_ids)
                losses = loss_fn(out, batch)
                loss = losses["total"] / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            run_loss += losses["total"].item()
            nb += 1
        sched.step()
        val = validate(model, val_loader, loss_fn, device)
        score = -val["loss"] if monitor == "loss" else val["dice"]
        rec = {"phase": phase, "epoch": epoch, "train_loss": run_loss / max(nb, 1),
               "time_s": time.time() - t0, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
               **{f"val_{k}": v for k, v in val.items()}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "phase": phase,
                        "val": val, "variant": model.variant},
                       phase1_path if phase == 1 else best_path)
        else:
            bad += 1
        if phase == 2 and bad >= cfg["training"]["early_stop_patience"]:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    return phase1_path if phase == 1 else best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="time N training iterations and exit")
    ap.add_argument("--short", action="store_true",
                    help="reduced budget: phase 1 = 5 epochs, phase 2 = 10 epochs; otherwise identical")
    args = ap.parse_args()

    seed_all(args.seed)
    tag = f"{args.variant}_s{args.seed}" + ("_short" if args.short else "")
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args.variant)
    if args.short:
        cfg["training"]["phase2_epochs"] = 10
        cfg["hardware"]["num_workers"] = 6  # data loading only; computation unchanged
    monitor = "loss" if args.variant == "clinOnly" else "dice"

    split_dir = tp.Path(CFG["paths"]["splits_dir"])
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_ds = tp.SpineDataset(split_dir / "train.txt", cfg, transforms=tp.get_train_transforms())
    val_ds = tp.SpineDataset(split_dir / "val.txt", cfg, transforms=tp.get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                              num_workers=cfg["hardware"]["num_workers"], pin_memory=True,
                              collate_fn=tp.collate_fn, drop_last=True, generator=g,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                            num_workers=cfg["hardware"]["num_workers"], pin_memory=True,
                            collate_fn=tp.collate_fn, persistent_workers=True)

    model = RevModel(cfg, variant=args.variant, lora_rank=cfg["model"]["lora_rank"]).to(tp.DEVICE)
    if args.no_grad_ckpt:
        base = model.encoder.base_model.model if hasattr(model.encoder, "base_model") else model.encoder
        if hasattr(base, "set_grad_checkpointing"):
            base.set_grad_checkpointing(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    meta = {"variant": args.variant, "seed": args.seed, "trainable": n_train, "total": n_total,
            "task_weights": cfg["loss"]["task_weights"], "boundary": cfg["loss"]["boundary_lambda"],
            "monitor": monitor}
    print(json.dumps(meta), flush=True)

    if args.bench:
        cw = json.load(open(ROOT / "results" / "class_weights.json"))
        loss_fn = tp.MultiTaskLoss(cfg, cw).to(tp.DEVICE)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        scaler = GradScaler()
        model.train()
        t0 = None
        for bi, batch in enumerate(train_loader):
            if bi == 5:
                torch.cuda.synchronize(); t0 = time.time()
            images = batch["image"].to(tp.DEVICE)
            plane_ids = batch["plane_id"].to(tp.DEVICE)
            level_ids = level_ids_from_names(batch["level"], tp.DEVICE)
            batch = to_device(batch, tp.DEVICE)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(images, batch["prompt_text"], plane_ids, level_ids)
                loss = loss_fn(out, batch)["total"] / 8
            scaler.scale(loss).backward()
            if (bi + 1) % 8 == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if bi == 5 + args.bench:
                break
        torch.cuda.synchronize()
        print("s/it", (time.time() - t0) / args.bench, "peak GB", torch.cuda.max_memory_allocated() / 1e9)
        return

    with open(LOG_ROOT / f"{tag}.jsonl", "w") as log:
        log.write(json.dumps({"meta": meta}) + "\n")
        p1 = run_phase(model, 1, train_loader, val_loader, cfg, tag, monitor, log)
        state = torch.load(p1, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(state)
        run_phase(model, 2, train_loader, val_loader, cfg, tag, monitor, log)
    os.remove(p1)
    print("done", tag, flush=True)


if __name__ == "__main__":
    main()
