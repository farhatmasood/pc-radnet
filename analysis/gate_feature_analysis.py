"""
gate_feature_analysis.py - interpretability analysis on the reported model (A2).

(a) Gate analysis in the 256-d calibration space: sigma(W_g c) for the seven
    anatomy x plane conditions (canonical terms), top-32 channels by across-condition
    variance, and the cosine similarity between gate vectors (Figure 15).
(b) Feature-space plots (Figure 16): one pooled 256-d decoder-input vector per
    (test image, present class), before and after gating. Because the gated vector is a
    function of the supplied anatomy term, its separation by class is expected by
    construction and is reported as such.
(c) A label-independent check: token features pooled inside each ground-truth structure
    (GT used for analysis only), compared between the frozen DINOv2 encoder and the
    adapted A2 encoder. Silhouette scores are computed in the original feature space
    with cosine distance.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_variants import tp, RevModel, load_original_into_rev  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "interpretability"
FIG = ROOT / "results" / "analysis" / "figures"
CONDS = [("Axial IVD", 0, "intervertebral disc"), ("Axial PE", 0, "posterior element"),
         ("Axial SC", 0, "spinal canal"), ("Sag VB", 1, "vertebral body"),
         ("Sag IVD", 1, "intervertebral disc"), ("Sag Sacrum", 1, "sacrum"), ("Sag SC", 1, "spinal canal")]
CLS = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 1): 3, (1, 2): 4, (1, 3): 5, (1, 4): 6}


def load_a2():
    m = RevModel(tp.CFG, variant="A2", lora_rank=8)
    sd = torch.load(ROOT / "checkpoints" / "best_by_mdice_A2.pt", map_location="cpu", weights_only=False)
    load_original_into_rev(m, sd["model_state_dict"])
    return m.to(tp.DEVICE).eval()


@torch.no_grad()
def gate_analysis(m):
    G = []
    for _, plane, term in CONDS:
        c = m.calibrator([term], torch.tensor([plane], device=tp.DEVICE))
        G.append(torch.sigmoid(m.seg_decoder.prompt_gate[0](c)).float().cpu().numpy()[0])
    G = np.stack(G)
    top = np.argsort(-G.var(0))[:32]
    Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    cos = Gn @ Gn.T
    active = (G[:, top] > 0.5).sum(1)
    return {"labels": [c[0] for c in CONDS], "top_channels": top.tolist(), "gate_top": G[:, top].tolist(),
            "cosine": cos.tolist(), "active_top32_per_condition": active.tolist(),
            "active_all256_per_condition": (G > 0.5).sum(1).tolist(), "gate_range": [float(G.min()), float(G.max())]}


@torch.no_grad()
def features(m):
    ds = tp.SpineDataset(Path(tp.CFG["paths"]["splits_dir"]) / "test.txt", tp.CFG, transforms=tp.get_val_transforms())
    dl = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=tp.collate_fn)
    import timm
    raw = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=384, num_classes=0)
    raw = raw.to(tp.DEVICE).eval()
    ung, gat, lab, reg_raw, reg_a2, reg_lab = [], [], [], [], [], []
    for b in dl:
        x = b["image"].to(tp.DEVICE)
        plane = int(b["plane_id"][0])
        pt = torch.tensor([plane], device=tp.DEVICE)
        oh = b["mask_onehot"][0]
        f = m.encode(x)                                   # [1,729,768] adapted tokens
        u = m.seg_decoder.proj(f)                         # [1,729,256] ungated
        fr = raw.forward_features(x)[:, 1:730, :]         # frozen DINOv2 tokens
        for c in range(1, oh.shape[0]):
            if oh[c].sum() <= 100 or (plane, c) not in CLS:
                continue
            term = tp.PROMPT_BANK["axial" if plane == 0 else "sagittal"][c][0]
            cal = m.calibrator([term], pt)
            g = torch.sigmoid(m.seg_decoder.prompt_gate[0](cal))
            ung.append(u.mean(1)[0].float().cpu().numpy())
            gat.append((u * (1 + g.unsqueeze(1))).mean(1)[0].float().cpu().numpy())
            lab.append(CLS[(plane, c)])
            frac = F.adaptive_avg_pool2d(oh[c][None, None], (27, 27)).flatten() >= 0.5
            if frac.sum() >= 1:
                idx = frac.nonzero().flatten().to(tp.DEVICE)
                reg_raw.append(fr[0, idx].mean(0).float().cpu().numpy())
                reg_a2.append(f[0, idx].mean(0).float().cpu().numpy())
                reg_lab.append(CLS[(plane, c)])
    return (np.array(ung), np.array(gat), np.array(lab), np.array(reg_raw), np.array(reg_a2), np.array(reg_lab))


def plots(gate, ung, gat, lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import umap
    import umap.umap_ as uu
    _ca = uu.check_array

    def _check_array(*a, ensure_all_finite=True, **k):  # sklearn<1.6 names it force_all_finite
        return _ca(*a, force_all_finite=ensure_all_finite, **k)
    uu.check_array = _check_array
    FIG.mkdir(parents=True, exist_ok=True)
    names = [c[0] for c in CONDS]
    colors = plt.cm.tab10(np.arange(7))
    # Figure 15
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.3, 1]})
    im = ax[0].imshow(np.array(gate["gate_top"]), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[0].set_yticks(range(7)); ax[0].set_yticklabels(names)
    ax[0].set_xticks(range(32)); ax[0].set_xticklabels(gate["top_channels"], rotation=90, fontsize=7)
    ax[0].set_xlabel("Calibration-space channel (top 32 of 256 by across-condition variance)")
    ax[0].set_title("(A) Gate values $\\sigma(W_g c)$; channel scaling = 1 + gate", fontsize=10)
    plt.colorbar(im, ax=ax[0], fraction=0.02)
    cs = np.array(gate["cosine"])
    im2 = ax[1].imshow(cs, cmap="magma", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax[1].text(j, i, f"{cs[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                       color="white" if cs[i, j] < 0.6 else "black")
    ax[1].set_xticks(range(7)); ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax[1].set_yticks(range(7)); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_title("(B) Cosine similarity of gate vectors", fontsize=10)
    plt.colorbar(im2, ax=ax[1], fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "figure15_gate_cosine_A2.pdf", bbox_inches="tight")
    plt.close(fig)
    # Figure 16
    embs = {}
    for key, X in [("ungated", ung), ("gated", gat)]:
        embs[("t-SNE", key)] = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(X)
        embs[("UMAP", key)] = umap.UMAP(random_state=0).fit_transform(X)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    panel = {("t-SNE", "ungated"): (0, 0, "A"), ("t-SNE", "gated"): (0, 1, "B"),
             ("UMAP", "ungated"): (1, 0, "C"), ("UMAP", "gated"): (1, 1, "D")}
    for k, (r, c, p) in panel.items():
        E = embs[k]
        for i in range(7):
            s = lab == i
            ax[r, c].scatter(E[s, 0], E[s, 1], s=5, color=colors[i], label=names[i])
        ax[r, c].set_title(f"({p}) {k[0]}, {'before' if k[1] == 'ungated' else 'after'} prompt gating", fontsize=10)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, 0].legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "figure16_tsne_A2.pdf", bbox_inches="tight")
    plt.close(fig)
    return {f"{k[0]}_{k[1]}_silhouette_2d": float(silhouette_score(v, lab)) for k, v in embs.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_a2()
    gate = gate_analysis(m)
    ung, gat, lab, rr, ra, rl = features(m)
    res = {"gate": gate, "n_points": int(len(lab)), "n_region_points": int(len(rl)),
           "silhouette_256d_ungated": float(silhouette_score(ung, lab, metric="cosine")),
           "silhouette_256d_gated": float(silhouette_score(gat, lab, metric="cosine")),
           "region_silhouette_frozen_dinov2": float(silhouette_score(rr, rl, metric="cosine")),
           "region_silhouette_A2_adapted": float(silhouette_score(ra, rl, metric="cosine"))}
    res.update(plots(gate, ung, gat, lab))
    print(json.dumps({k: v for k, v in res.items() if k != "gate"}, indent=1))
    print(json.dumps({k: gate[k] for k in ["active_top32_per_condition", "active_all256_per_condition", "gate_range"]}))
    with open(OUT / "gate_feature_A2.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
