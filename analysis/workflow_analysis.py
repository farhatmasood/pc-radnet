"""
workflow_analysis.py - examination-level workflow analysis.

(1) Review flags from information available at inference only:
      F1  T1w/T2w disagreement: the two axial slices of the same disc give different grades;
      F2  low ordinal margin on the T2w slice: some cumulative probability within 0.15 of 0.5.
    Grading accuracy is compared between flagged and unflagged discs (T2w prediction).
(2) Timing on the GPU for one image (one structure) and for one complete examination
    (8 images, every structure of each plane, clinical heads and the RATF report), for
    a naive pipeline (one full forward per prompt) and for a shared-encoder pipeline
    (encoder once per image, decoder once per prompt).
(3) Per-image time of the same-split segmentation networks.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_variants import tp, RevModel, load_original_into_rev  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results" / "analysis"


def flags():
    c = pd.read_csv(OUT / "unified" / "A2_test_clinical.csv")
    g = c[c.gt_grade > 0].copy()
    g["seq"] = np.where(g.fname.str.contains("_T2_"), "T2", "T1")
    piv = g.pivot_table(index=["patient_id", "level"], columns="seq", values="pred_grade").reset_index()
    t2 = g[g.seq == "T2"].merge(piv, on=["patient_id", "level"])
    P = t2[[f"grade_p{k}" for k in range(1, 5)]].values
    t2["F1"] = t2["T1"] != t2["T2"]
    t2["F2"] = (np.abs(P - 0.5) < 0.15).any(1)
    t2["flag"] = t2.F1 | t2.F2
    def summ(d):
        if len(d) == 0:
            return {}
        return {"n": int(len(d)), "exact": float((d.pred_grade == d.gt_grade).mean()),
                "mae": float((d.pred_grade - d.gt_grade).abs().mean()),
                "kappa_quad": float(cohen_kappa_score(d.gt_grade, d.pred_grade, labels=[1, 2, 3, 4, 5],
                                                      weights="quadratic"))}
    res = {"all": summ(t2)}
    for f in ["F1", "F2", "flag"]:
        res[f"{f}_rate"] = float(t2[f].mean())
        res[f"{f}_flagged"] = summ(t2[t2[f]])
        res[f"{f}_unflagged"] = summ(t2[~t2[f]])
    res["patients_with_any_flag"] = float(t2.groupby("patient_id").flag.any().mean())
    return res


@torch.no_grad()
def timing(n_patients=10):
    m = RevModel(tp.CFG, variant="A2", lora_rank=8)
    sd = torch.load(ROOT / "checkpoints" / "best_by_mdice_A2.pt", map_location="cpu", weights_only=False)
    load_original_into_rev(m, sd["model_state_dict"])
    m = m.to(tp.DEVICE).eval()
    from ratf_analysis import Retriever, template_sentence, findings, levels_in
    retr = Retriever()
    ds = tp.SpineDataset(Path(tp.CFG["paths"]["splits_dir"]) / "test.txt", tp.CFG, transforms=tp.get_val_transforms())
    by_pid = {}
    for i, p in enumerate(ds.image_paths):
        by_pid.setdefault(int(Path(p).stem.split("_")[2]), []).append(i)
    pids = sorted(by_pid)[:n_patients]
    for _ in range(5):  # warm-up
        x = ds[0]["image"][None].to(tp.DEVICE)
        with autocast("cuda", dtype=torch.bfloat16):
            m(x, ["intervertebral disc"], torch.zeros(1, dtype=torch.long, device=tp.DEVICE))
    single, naive, shared = [], [], []
    for pid in pids:
        items = [ds[i] for i in by_pid[pid]]
        # one image, one structure
        x = items[0]["image"][None].to(tp.DEVICE)
        pl = torch.zeros(1, dtype=torch.long, device=tp.DEVICE)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with autocast("cuda", dtype=torch.bfloat16):
            m(x, ["intervertebral disc"], pl)
        torch.cuda.synchronize(); single.append(time.perf_counter() - t0)
        for mode in ["naive", "shared"]:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lv = {}
            for it in items:
                x = it["image"][None].to(tp.DEVICE)
                plane = int(it["plane_id"])
                pt = torch.tensor([plane], device=tp.DEVICE)
                terms = [tp.PROMPT_BANK["axial" if plane == 0 else "sagittal"][c][0]
                         for c in (range(1, 4) if plane == 0 else range(1, 5))]
                with autocast("cuda", dtype=torch.bfloat16):
                    if mode == "naive":
                        for t in terms:
                            o = m(x, [t], pt)
                    else:
                        f = m.encode(x)
                        for t in terms:
                            m.seg_decoder(f, m.calibrator([t], pt), pt)
                        o = {"grade_logits": m.grade_head(f), "height_pred": m.height_head(f),
                             "disorder_logits": m.disorder_head(f)}
                if plane == 0:
                    g = int((torch.sigmoid(o["grade_logits"][0].float()) > 0.5).sum()) + 1
                    lv.setdefault(it["level"], {"grade": g, "height": float(o["height_pred"][0]), "find": []})
            torch.cuda.synchronize()
            for L, ld in lv.items():  # report stage (template + slot-checked retrieval)
                s = template_sentence(L, ld)
                hit, sim = retr.top(s)
                _ = sim >= 0.75 and findings(hit) == set(ld["find"]) and levels_in(hit)
            (naive if mode == "naive" else shared).append(time.perf_counter() - t0)
    return {"n_patients": len(pids), "single_image_one_structure_ms": 1000 * float(np.median(single)),
            "exam_naive_s": float(np.median(naive)), "exam_shared_encoder_s": float(np.median(shared)),
            "forward_passes_naive": 26, "encoder_passes_shared": 8}


@torch.no_grad()
def baseline_timing():
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_direct_baselines as edb
    out = {}
    x = torch.randn(1, 3, 384, 384, device=tp.DEVICE)
    builders = {"unet": lambda: edb.PlainConvUNet(), "swinunetr": edb.build_swinunetr,
                "sam_ft": lambda: edb.SAMFineTuned(str(Path(tp.CFG["paths"]["pretrained"]["sam_vit_b"])))
                if hasattr(edb, "SAMFineTuned") else None}
    for name, b in builders.items():
        try:
            net = b()
            if net is None:
                continue
            net = net.to(tp.DEVICE).eval()
            ts = []
            for i in range(30):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with autocast("cuda", dtype=torch.bfloat16):
                    net(x)
                torch.cuda.synchronize()
                if i >= 5:
                    ts.append(time.perf_counter() - t0)
            out[name] = {"ms_per_image": 1000 * float(np.median(ts)),
                         "params_total": int(sum(p.numel() for p in net.parameters()))}
            del net
            torch.cuda.empty_cache()
        except Exception as e:
            out[name] = {"error": repr(e)}
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if what in ("flags", "all"):
        res["flags"] = flags()
    if what in ("timing", "all"):
        res["timing"] = timing()
        res["baseline_timing"] = baseline_timing()
    print(json.dumps(res, indent=1))
    with open(OUT / f"workflow_{what}.json", "w") as f:
        json.dump(res, f, indent=1)
