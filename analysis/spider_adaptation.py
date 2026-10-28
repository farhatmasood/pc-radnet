"""
spider_adaptation.py - controlled training-data-composition experiment on SPIDER.

Split: official SPIDER subsets (T2, non-SPACE volumes). "training" patients (172) are used
for adaptation (15 % held out for early stopping); the official "validation" subset (38
patients) is the test set. Inputs are prepared without ground truth (whole slice, padded to
square, resized to 384; the five slices around the geometric centre are used for training,
the central slice for testing).

Label harmonisation: SPIDER annotates vertebral bodies, discs and the canal but not the
sacrum, so the sacrum channel is not supervised on SPIDER images.

Regimes (identical optimiser and schedule, segmentation losses only):
  direct   - LSMA-PQR model (A2), no SPIDER data
  spider   - trained on SPIDER only, from the DINOv2/LoRA initialisation
  finetune - A2 fine-tuned on SPIDER
  joint    - A2 fine-tuned on SPIDER + LSMA-PQR sagittal training images
Each regime is scored on the SPIDER test patients and on the LSMA-PQR sagittal test images.
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_variants import tp, RevModel, load_original_into_rev  # noqa: E402

ROOT = HERE.parent
SP = Path(tp.CFG["paths"]["external_datasets"]["spider"])
OUT = ROOT / "results" / "analysis" / "spider_adapt"
CACHE = OUT / "spider_slices.npz"
SAG = {1: "VB", 2: "IVD_sag", 3: "Sacrum", 4: "SC_sag"}
CW = json.load(open(ROOT / "results" / "class_weights.json"))["sagittal"]


def to_ours(m):
    o = np.zeros_like(m, dtype=np.uint8)
    o[(m >= 1) & (m <= 8)] = 1
    o[(m >= 201) & (m <= 208)] = 2
    o[m == 100] = 4
    return o


def square_resize(a, size, nearest):
    from PIL import Image
    h, w = a.shape
    s = max(h, w)
    p = np.zeros((s, s), a.dtype)
    p[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    im = Image.fromarray(p)
    return np.array(im.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR))


def prepare():
    import SimpleITK as sitk
    ov = pd.read_csv(SP / "overview.csv")
    ov = ov[ov.new_file_name.str.endswith("_t2")]
    imgs, msks, pids, subs, offs = [], [], [], [], []
    for _, r in ov.iterrows():
        f = r.new_file_name + ".mha"
        if not (SP / "images" / f).exists():
            continue
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(SP / "images" / f))).astype(np.float64)
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(SP / "masks" / f)))
        c = img.shape[2] // 2
        for o in range(-2, 3):
            s = c + o
            if not 0 <= s < img.shape[2]:
                continue
            sl = img[:, :, s]
            p1, p99 = np.percentile(sl, [1, 99])
            sl = np.clip((sl - p1) / max(p99 - p1, 1e-6), 0, 1)
            imgs.append(square_resize((sl * 255).astype(np.uint8), 384, False))
            msks.append(square_resize(to_ours(msk[:, :, s]), 384, True))
            pids.append(int(r.new_file_name.split("_")[0]))
            subs.append(r.subset)
            offs.append(o)
    np.savez_compressed(CACHE, img=np.stack(imgs), msk=np.stack(msks), pid=np.array(pids),
                        sub=np.array(subs), off=np.array(offs))


class SpiderSet(Dataset):
    def __init__(self, img, msk, pid, train):
        self.img, self.msk, self.pid, self.train = img, msk, pid, train
        self.tf = tp.get_train_transforms() if train else tp.get_val_transforms()

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        rgb = np.repeat(self.img[i][..., None], 3, 2)
        t = self.tf(image=rgb, mask=self.msk[i])
        m = t["mask"].long()
        oh = torch.stack([(m == c).float() for c in range(5)])
        present = [c for c in (1, 2, 4) if oh[c].sum() > 100] or [1]
        c = random.choice(present) if self.train else present[0]
        return {"image": t["image"], "mask": m, "mask_onehot": oh, "prompted_mask": oh[c],
                "prompt_text": random.choice(tp.PROMPT_BANK["sagittal"][c]) if self.train
                else tp.PROMPT_BANK["sagittal"][c][0],
                "target_cls": c, "plane_id": 1, "num_classes": 5, "grade": -1, "height": -1.0,
                "disorder": torch.zeros(9), "patient_id": int(self.pid[i]), "level": "S",
                "fname": f"spider_{self.pid[i]}_{i}", "is_spider": 1}


class LsmaSag(Dataset):
    def __init__(self, split, train):
        files = [l.strip() for l in open(Path(tp.CFG["paths"]["splits_dir"]) / f"{split}.txt") if "sag_" in l]
        tmp = ROOT / "results" / "analysis" / "spider_adapt" / f"lsma_sag_{split}.txt"
        tmp.write_text("\n".join(files))
        self.ds = tp.SpineDataset(tmp, tp.CFG, transforms=tp.get_train_transforms() if train
                                  else tp.get_val_transforms())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        d["is_spider"] = 0
        return d


def seg_loss(out, batch):
    dl = tp.DiceLoss()
    total, n = 0.0, 0
    for i, lg in enumerate(out["seg_logits_list"]):
        gt = batch["mask_onehot"][i]
        for c in range(1, 5):
            if c == 3 and int(batch["is_spider"][i]) == 1:
                continue
            p = lg[0, c:c + 1]
            g = gt[c:c + 1]
            total = total + dl(p.unsqueeze(0), g.unsqueeze(0)) + \
                F.binary_cross_entropy_with_logits(p, g, pos_weight=p.new_tensor([CW[str(c)]]))
            n += 1
    pr = out["prompted_logits"].squeeze(1)
    pm = batch["prompted_mask"]
    prompted = dl(pr, pm) + F.binary_cross_entropy_with_logits(pr, pm)
    bound = tp.BoundaryLoss().to(pr.device)(pr, pm)
    return total / max(n, 1) + prompted + 0.2 * bound


@torch.no_grad()
def score(model, loader, classes):
    model.eval()
    rows = []
    for b in loader:
        x = b["image"].to(tp.DEVICE)
        pl = torch.ones(1, dtype=torch.long, device=tp.DEVICE)
        oh = b["mask_onehot"][0].to(tp.DEVICE)
        for c in classes:
            if oh[c].sum() <= 100:
                continue
            with autocast("cuda", dtype=torch.bfloat16):
                o = model(x, [tp.PROMPT_BANK["sagittal"][c][0]], pl)
            pr = (torch.sigmoid(o["prompted_logits"][0, 0].float()) > 0.5).float()
            d = (2 * (pr * oh[c]).sum() + 1e-5) / (pr.sum() + oh[c].sum() + 1e-5)
            rows.append({"patient_id": int(b["patient_id"][0]), "cls": SAG[c], "dice": float(d)})
    df = pd.DataFrame(rows)
    return df


def make_model(init):
    m = RevModel(tp.CFG, variant="A2", lora_rank=8)
    if init == "a2":
        sd = torch.load(ROOT / "checkpoints" / "best_by_mdice_A2.pt", map_location="cpu",
                        weights_only=False)["model_state_dict"]
        load_original_into_rev(m, sd)
    base = m.encoder.base_model.model if hasattr(m.encoder, "base_model") else m.encoder
    base.set_grad_checkpointing(False)
    return m.to(tp.DEVICE)


def train(model, train_ds, val_loader, epochs, lr, tag, log):
    dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=tp.collate_fn,
                    drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sc = GradScaler()
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        for bi, b in enumerate(dl):
            for k in ["mask_onehot", "prompted_mask"]:
                b[k] = b[k].to(tp.DEVICE)
            with autocast("cuda", dtype=torch.bfloat16):
                o = model(b["image"].to(tp.DEVICE), b["prompt_text"], b["plane_id"].to(tp.DEVICE))
                loss = seg_loss(o, b) / 8
            sc.scale(loss).backward()
            if (bi + 1) % 8 == 0:
                sc.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                sc.step(opt); sc.update(); opt.zero_grad()
        sch.step()
        v = score(model, val_loader, [1, 2, 4]).dice.mean()
        log.write(json.dumps({"tag": tag, "epoch": ep, "val_dice": float(v)}) + "\n"); log.flush()
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--regimes", default="direct,spider,finetune,joint")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if not CACHE.exists():
        prepare()
    z = np.load(CACHE)
    img, msk, pid, sub, off = z["img"], z["msk"], z["pid"], z["sub"], z["off"]
    trp = sorted(set(pid[sub == "training"]))
    rng = np.random.default_rng(0)
    valp = set(rng.choice(trp, int(round(0.15 * len(trp))), replace=False))
    tr = (sub == "training") & ~np.isin(pid, list(valp))
    va = np.isin(pid, list(valp)) & (off == 0)
    te = (sub == "validation") & (off == 0)
    sp_tr = SpiderSet(img[tr], msk[tr], pid[tr], True)
    sp_va = DataLoader(SpiderSet(img[va], msk[va], pid[va], False), batch_size=1, collate_fn=tp.collate_fn)
    sp_te = DataLoader(SpiderSet(img[te], msk[te], pid[te], False), batch_size=1, collate_fn=tp.collate_fn)
    ls_tr = LsmaSag("train", True)
    ls_te = DataLoader(LsmaSag("test", False), batch_size=1, collate_fn=tp.collate_fn, num_workers=2)
    info = {"spider_train_slices": int(tr.sum()), "spider_train_patients": int(len(set(pid[tr]))),
            "spider_val_patients": len(valp), "spider_test_patients": int(te.sum()),
            "lsma_sag_train": len(ls_tr)}
    print(json.dumps(info), flush=True)
    res = {"info": info}
    with open(OUT / "train_log.jsonl", "a") as log:
        for r in a.regimes.split(","):
            if r == "direct":
                m = make_model("a2")
            elif r == "spider":
                m = train(make_model("dino"), sp_tr, sp_va, a.epochs, 1e-4, r, log)
            elif r == "finetune":
                m = train(make_model("a2"), sp_tr, sp_va, a.epochs, 5e-5, r, log)
            else:
                m = train(make_model("a2"), ConcatDataset([sp_tr, ls_tr]), sp_va, a.epochs, 5e-5, r, log)
            s_te = score(m, sp_te, [1, 2, 4])
            l_te = score(m, ls_te, [1, 2, 3, 4])
            s_te.to_csv(OUT / f"{r}_spider_test.csv", index=False)
            l_te.to_csv(OUT / f"{r}_lsma_sag_test.csv", index=False)
            res[r] = {"spider_test": {**s_te.groupby("cls").dice.mean().round(4).to_dict(),
                                      "mean": float(s_te.dice.mean())},
                      "lsma_sag_test": {**l_te.groupby("cls").dice.mean().round(4).to_dict(),
                                        "mean": float(l_te.dice.mean())}}
            print(r, json.dumps(res[r]), flush=True)
            del m
            torch.cuda.empty_cache()
            with open(OUT / "spider_adapt_results.json", "w") as f:
                json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
