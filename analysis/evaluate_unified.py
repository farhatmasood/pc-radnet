"""
evaluate_unified.py - one evaluation pipeline for every checkpoint used in the extended analysis.

For each image of the requested split it records
  * prompted-head Dice/IoU for every foreground class present (GT area > 100 px),
    each queried with its canonical vocabulary term ("all-class" protocol);
  * the multi-class plane-head channel of the same class in the same forward pass
    ("multi-class output selection" baseline);
  * the prompted Dice under the original one-random-prompt-per-image protocol, using
    the exact prompts stored in results/<split>_A2_per_image.csv;
  * clinical outputs (grade logits, height, disorder probabilities), which do not
    depend on the prompt because the clinical heads read pre-gate features.

Usage:
    python evaluate_unified.py --name A2 --ckpt ../checkpoints/best_by_mdice_A2.pt --kind orig
    python evaluate_unified.py --name noSDA_s42 --ckpt ../checkpoints/variants/noSDA_s42_best.pt --kind rev --variant noSDA
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_variants import tp, RevModel, load_original_into_rev, level_ids_from_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "unified"
AX = {1: "IVD", 2: "PE", 3: "SC"}
SG = {1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"}
DIS = ["Herniation", "Protrusion", "Bulge", "ThecalComp", "NerveComp", "Stenosis",
       "Spondylolisthesis", "Desiccated", "LigFlavum"]


def dice_iou(prob, gt):
    pb = (prob > 0.5).float()
    inter = (pb * gt).sum()
    d = (2 * inter + 1e-5) / (pb.sum() + gt.sum() + 1e-5)
    j = (inter + 1e-5) / (pb.sum() + gt.sum() - inter + 1e-5)
    return d.item(), j.item()


def build(args):
    if args.kind == "orig":
        lora = 0 if args.name.startswith("A0") else 8
        model = RevModel(tp.CFG, variant="A2", lora_rank=lora)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        missing, unexpected = load_original_into_rev(model, sd)
        assert not unexpected, unexpected
        assert not [m for m in missing if "level" not in m], missing
    else:
        model = RevModel(tp.CFG, variant=args.variant, lora_rank=8)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(sd)
    return model.to(tp.DEVICE).eval()


@torch.no_grad()
def run(model, split, save_sag):
    split_file = Path(tp.CFG["paths"]["splits_dir"]) / f"{split}.txt"
    ds = tp.SpineDataset(split_file, tp.CFG, transforms=tp.get_val_transforms())
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=tp.collate_fn)
    orig = pd.read_csv(ROOT / "results" / f"{split}_A2_per_image.csv").set_index("fname")
    rows_p, rows_o, rows_c, sag = [], [], [], {}
    t_forward = []
    for batch in dl:
        fname = batch["fname"][0]
        img = batch["image"].to(tp.DEVICE)
        plane = batch["plane_id"].to(tp.DEVICE)
        lev = level_ids_from_names(batch["level"], tp.DEVICE)
        onehot = batch["mask_onehot"][0].to(tp.DEVICE)
        is_ax = int(plane.item()) == 0
        names = AX if is_ax else SG
        view = "axial" if is_ax else "sagittal"
        base = {"fname": fname, "patient_id": int(batch["patient_id"][0]), "level": batch["level"][0],
                "plane": view}

        # original protocol: the stored random prompt for this image
        op = orig.loc[fname, "prompt"]
        ot = int(orig.loc[fname, "target_cls"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(img, [op], plane, lev)
        torch.cuda.synchronize()
        t_forward.append(time.perf_counter() - t0)
        pr = torch.sigmoid(out["prompted_logits"][0, 0].float())
        d, j = dice_iou(pr, onehot[ot])
        seg = torch.sigmoid(out["seg_logits_list"][0][0].float())
        ro = {**base, "prompt": op, "target_cls": ot, "prompted_dice": d, "prompted_iou": j}
        for c, nm in names.items():
            ro[f"mc_dice_{nm}"], ro[f"mc_iou_{nm}"] = dice_iou(seg[c], onehot[c])
        rows_o.append(ro)

        # clinical outputs (prompt independent)
        gl = out["grade_logits"][0].float()
        gp = torch.sigmoid(gl)
        rc = {**base, "gt_grade": int(batch["grade"][0]), "pred_grade": int((gp > 0.5).sum()) + 1,
              "gt_height": float(batch["height"][0]), "pred_height": float(out["height_pred"][0].float())}
        for k in range(4):
            rc[f"grade_p{k + 1}"] = float(gp[k])
        dp = torch.sigmoid(out["disorder_logits"][0].float())
        for k, nm in enumerate(DIS):
            rc[f"dis_{nm}_prob"] = float(dp[k])
            rc[f"dis_{nm}_gt"] = float(batch["disorder"][0, k])
        rows_c.append(rc)

        # all-class protocol + multi-class output selection
        keep = {}
        for c, nm in names.items():
            area = float(onehot[c].sum())
            if area <= 100:
                continue
            term = tp.PROMPT_BANK[view][c][0]
            with autocast("cuda", dtype=torch.bfloat16):
                o2 = model(img, [term], plane, lev)
            p2 = torch.sigmoid(o2["prompted_logits"][0, 0].float())
            s2 = torch.sigmoid(o2["seg_logits_list"][0][0, c].float())
            pd_, pj = dice_iou(p2, onehot[c])
            md, mj = dice_iou(s2, onehot[c])
            mask = p2 > 0.5
            conf = float(p2[mask].mean()) if mask.any() else 0.0
            rows_p.append({**base, "cls": c, "cls_name": nm, "term": term, "gt_area": area,
                           "pred_area": float(mask.sum()), "prompted_dice": pd_, "prompted_iou": pj,
                           "mc_dice": md, "mc_iou": mj, "prompted_conf": conf})
            if save_sag and not is_ax:
                keep[nm] = (p2.cpu().numpy() * 255).astype(np.uint8)
        if save_sag and not is_ax:
            keep["mc_argmax"] = torch.argmax(torch.sigmoid(out["seg_logits_list"][0][0].float()), 0) \
                .cpu().numpy().astype(np.uint8)
            sag[fname] = keep
    return pd.DataFrame(rows_p), pd.DataFrame(rows_o), pd.DataFrame(rows_c), sag, t_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["orig", "rev"], required=True)
    ap.add_argument("--variant", default="A2")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--save-sag", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = build(args)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for split in args.splits.split(","):
        dp, do, dc, sag, tf = run(model, split, args.save_sag)
        dp.to_csv(OUT / f"{args.name}_{split}_allclass.csv", index=False)
        do.to_csv(OUT / f"{args.name}_{split}_origprompt.csv", index=False)
        dc.to_csv(OUT / f"{args.name}_{split}_clinical.csv", index=False)
        if sag:
            np.savez_compressed(OUT / f"{args.name}_{split}_sagmasks.npz",
                                **{f"{k}|{n}": v for k, d in sag.items() for n, v in d.items()})
        summ = {"name": args.name, "split": split, "n_images": len(do),
                "orig_prompted_dice": do.prompted_dice.mean(), "orig_prompted_iou": do.prompted_iou.mean(),
                "allclass_prompted_dice": dp.prompted_dice.mean(), "allclass_mc_dice": dp.mc_dice.mean(),
                "forward_ms_median": 1000 * float(np.median(tf)), "trainable_params_eval": n_tr}
        print(json.dumps(summ), flush=True)
        with open(OUT / f"{args.name}_{split}_summary.json", "w") as f:
            json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
