"""
ratf_analysis.py - report-generation analysis .

1. Builds RATF reports from PC-RadNet (A2) predictions in three stages:
     T   template only (deterministic sentence per level),
     R   template + retrieval: a level sentence is replaced by its nearest training-set
         radiologist sentence when cosine similarity >= tau (as described in the paper),
     RC  slot-checked retrieval: the replacement is accepted only if the retrieved sentence
         names the same level and the same finding set and contains no negated finding;
         the Pfirrmann clause is kept from the template.
   The retrieval index holds sentences from training-split patients only.
2. Scores every report source (RATF stages, stored LLM outputs) with one negation-aware
   finding extractor, plus slot-fidelity and contradiction checks.
Inputs follow the matched benchmark: per level, grade = mode of the axial predictions,
height = mean prediction (mm), finding present if the level-wise max probability > 0.5.
"""
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repo_config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "results" / "analysis" / "ratf"
DS = Path(load_config()["paths"]["dataset_root"])
LEVEL = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
GRADE_WORD = {1: "no", 2: "mild", 3: "moderate", 4: "advanced", 5: "severe"}
DIS = ["disc herniation", "disc protrusion", "disc bulge", "thecal sac compression",
       "nerve root compression", "spinal canal stenosis", "spondylolisthesis",
       "desiccated disc", "ligamentum flavum hypertrophy"]
GT_COLS = ["Disc_Herniation", "Disc_Protrusion", "Disc_Bulge", "Thecal_Sac_Compression",
           "Nerve_Root_Compression", "Spinal_Canal_Stenosis", "Spondylolisthesis",
           "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"]
PROB = ["Herniation", "Protrusion", "Bulge", "ThecalComp", "NerveComp", "Stenosis",
        "Spondylolisthesis", "Desiccated", "LigFlavum"]
BENCH_IDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113, 118, 124, 125, 140, 150, 152, 165, 167, 177, 181]

KW = {
    "disc herniation": [r"herniat", r"extru"],
    "disc protrusion": [r"protru"],
    "disc bulge": [r"bulg"],
    "thecal sac compression": [r"thecal sac compression", r"compress\w*[^.;]*thecal", r"thecal[^.;]*compress",
                               r"effac\w*[^.;]*thecal", r"indent\w*[^.;]*thecal", r"abut\w*[^.;]*thecal"],
    "nerve root compression": [r"nerve root", r"compress\w*[^.;]*nerve", r"encroach\w*[^.;]*(nerve|neural)",
                               r"neural compression", r"radiculopathy"],
    "spinal canal stenosis": [r"stenosis", r"canal narrowing"],
    "spondylolisthesis": [r"listhesis"],
    "desiccated disc": [r"desiccat", r"dehydrat", r"dark disc"],
    "ligamentum flavum hypertrophy": [r"ligamentum flavum", r"flavum hypertrophy", r"ligamentous hypertrophy"],
}
NEG = re.compile(r"\b(no|not|without|negative for|absence of|free of|neither|nor)\b")
LEV_RE = re.compile(r"\b([LD]\d{1,2})\s*[-/]\s*([LSD]\d{1,2})\b", re.I)


def clauses(text):
    return [c for c in re.split(r"[.;:\n]|\s-\s|,\s(?=and\b)", text.lower()) if c.strip()]


def findings(text):
    """Set of findings asserted (not negated) in the text."""
    out = set()
    for cl in clauses(text):
        for name, pats in KW.items():
            for p in pats:
                m = re.search(p, cl)
                if m and not NEG.search(cl[:m.start()]):
                    out.add(name)
                    break
    return out


def levels_in(text):
    return {f"{a.upper()}-{b.upper()}" for a, b in LEV_RE.findall(text)}


def prf(pred, gt):
    tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def gt_sets():
    g = pd.read_csv(DS / "structured_notes.csv").set_index("Patient_ID")
    return {int(i): {d for d, c in zip(DIS, GT_COLS) if int(r[c]) == 1} for i, r in g.iterrows()}


def level_inputs(pids):
    df = pd.read_csv(RES / "test_A2_per_image.csv")
    data = {}
    for pid in pids:
        sub = df[(df.patient_id == pid) & (df.plane == "axial")]
        lv = {}
        for L in ["D3", "D4", "D5"]:
            rows = sub[sub.level == L]
            if rows.empty:
                continue
            g = rows[rows.pred_grade > 0].pred_grade
            lv[L] = {"grade": int(g.mode().iloc[0]) if len(g) else -1,
                     "height": float(rows.pred_height.mean()),
                     "find": [DIS[i] for i, c in enumerate(PROB) if rows[f"disorder_{c}_prob"].max() > 0.5]}
        data[pid] = lv
    return data


def find_phrase(fs):
    if not fs:
        return "no focal disc abnormality is predicted"
    if len(fs) == 1:
        return f"{fs[0]} is noted"
    return ", ".join(fs[:-1]) + f" and {fs[-1]} are noted"


def template_sentence(L, ld):
    g = ld["grade"]
    return (f"At {LEVEL[L]}, {find_phrase(ld['find'])}, with {GRADE_WORD.get(g, 'indeterminate')} "
            f"disc degeneration (Pfirrmann grade {g}).")


class Retriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        tr = {int(Path(l.strip()).stem.split("_")[2]) for l in open(ROOT / "data" / "train.txt") if l.strip()}
        rep = pd.read_csv(DS / "structured_reports.csv")
        rep = rep[rep.Patient_ID.isin(tr)]
        sents = []
        for s in rep.Original_Segment.dropna().astype(str):
            s = s.strip().strip('"').strip()
            if len(s.split()) >= 4:
                sents.append(s)
        self.sents = sorted(set(sents))
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        e = self.enc.encode(self.sents, batch_size=64, normalize_embeddings=True)
        self.emb = np.asarray(e, dtype=np.float32)
        self.n_train_patients = len(tr)

    def top(self, q):
        v = self.enc.encode([q], normalize_embeddings=True)[0]
        s = self.emb @ v
        i = int(np.argmax(s))
        return self.sents[i], float(s[i])


def build_reports(data, retr, tau):
    out = {"T": {}, "R": {}, "RC": {}}
    audit = []
    for pid, lv in data.items():
        parts = {k: [] for k in out}
        imp = []
        for L, ld in lv.items():
            t = template_sentence(L, ld)
            hit, sim = retr.top(t)
            tf = set(ld["find"])
            hf = findings(hit)
            hl = levels_in(hit)
            same_level = hl == {LEVEL[L]}
            same_find = hf == tf
            parts["T"].append(t)
            parts["R"].append(hit if sim >= tau else t)
            accept = sim >= tau and same_level and same_find
            parts["RC"].append((hit.rstrip(". ") + f"; Pfirrmann grade {ld['grade']}.") if accept else t)
            audit.append({"patient_id": pid, "level": LEVEL[L], "sim": sim, "replaced_R": sim >= tau,
                          "accepted_RC": accept, "level_match": same_level, "finding_match": same_find,
                          "retrieved": hit, "template": t})
            imp.append(f"{LEVEL[L]}: Pfirrmann grade {ld['grade']}; " +
                       (", ".join(ld["find"]) if ld["find"] else "no focal disc abnormality predicted") + ".")
        heights = "; ".join(f"{LEVEL[L]} {ld['height']:.1f} mm" for L, ld in lv.items())
        for k in out:
            out[k][pid] = ("FINDINGS: " + " ".join(parts[k]) +
                           f"\nEstimated disc height (axial regression, coarse): {heights}." +
                           "\nIMPRESSION: " + " ".join(imp))
    return out, pd.DataFrame(audit)


def findings_section(text):
    m = re.search(r"findings:(.*?)(impression:|$)", text, re.I | re.S)
    return m.group(1) if m else text


def score_source(reports, gts, inputs=None):
    rows = []
    for pid, txt in reports.items():
        pred = findings(txt)
        p, r, f, tp, fp, fn = prf(pred, gts[pid])
        lv = levels_in(txt)
        unsupported = int(bool(re.search(r"tesla|\b1\.5\s*t\b|\b3\s*t\b|contrast|gadolin", txt.lower())))
        outside = len({x for x in lv if x not in {"L3-L4", "L4-L5", "L5-S1"}})
        row = {"patient_id": pid, "P": p, "R": r, "F1": f, "tp": tp, "fp": fp, "fn": fn,
               "words": len(txt.split()), "unsupported_technique": unsupported, "levels_outside_L3S1": outside}
        if inputs is not None:
            fin = findings(findings_section(txt))
            want = set().union(*[set(ld["find"]) for ld in inputs[pid].values()]) if inputs[pid] else set()
            row["input_findings_kept"] = float(want <= fin) if want else 1.0
            row["extra_findings_vs_input"] = len(fin - want)
        rows.append(row)
    return pd.DataFrame(rows)


def lexical(reports, pids):
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_nlg as nlg
    refs = nlg.build_reference_reports_v2(pids)
    vals = []
    for pid in pids:
        if pid not in refs or pid not in reports:
            continue
        h, ref = reports[pid], refs[pid]
        b = nlg.compute_bleu(h, ref)
        r = nlg.compute_rouge(h, ref)
        m = nlg.compute_meteor(h, ref)
        vals.append({"BLEU1": b.get("BLEU-1", np.nan), "ROUGEL": r["ROUGE-L"], "METEOR": m if np.isscalar(m) else m.get("METEOR", np.nan)})
    return pd.DataFrame(vals).mean().to_dict()


def stored_reports():
    src = {}
    base = RES / "report_benchmark" / "reports"
    names = {"medpalm": "Med-PaLM (re-implementation)", "meditron3_8b": "Meditron3-8B", "llama32_3b": "Llama-3.2-3B",
             "biomistral_7b": "BioMistral-7B", "biogpt_large": "BioGPT-Large", "llava_rad": "LLaVA-Rad-7B",
             "qwen25_1p5": "Qwen2.5-1.5B", "phi35_mini": "Phi-3.5-mini", "smollm2_1p7": "SmolLM2-1.7B",
             "maira2": "MAIRA-2", "radar": "RADAR"}
    for d, nm in names.items():
        src[nm] = {int(f.stem.split("_")[-1]): f.read_text(encoding="utf-8", errors="ignore")
                   for f in (base / d).glob("report_*.txt")}
    src["MedGemma-4B-IT"] = {int(f.stem.split("_")[-1]): f.read_text(encoding="utf-8", errors="ignore")
                             for f in (RES / "medgemma_reports").glob("medgemma_report_*.txt")}
    src["Prose template (earlier RATF-v3)"] = {int(f.stem.split("_")[-1]): f.read_text(encoding="utf-8")
                                              for f in (RES / "ratf_reports_v3").glob("report_*.txt")}
    return src


def main(tau=0.75):
    OUT.mkdir(parents=True, exist_ok=True)
    gts = gt_sets()
    test_pids = sorted({int(Path(l.strip()).stem.split("_")[2]) for l in open(ROOT / "data" / "test.txt") if l.strip()})
    retr = Retriever()
    summary = {"index_sentences": len(retr.sents), "index_patients": retr.n_train_patients, "tau": tau}

    for tag, pids in [("bench20", BENCH_IDS), ("test78", test_pids)]:
        data = level_inputs(pids)
        reps, audit = build_reports(data, retr, tau)
        audit.to_csv(OUT / f"audit_{tag}.csv", index=False)
        for k, rp in reps.items():
            (OUT / f"{tag}_{k}").mkdir(exist_ok=True)
            for pid, txt in rp.items():
                (OUT / f"{tag}_{k}" / f"report_{pid:04d}.txt").write_text(txt, encoding="utf-8")
        res = {}
        for k, rp in reps.items():
            s = score_source(rp, gts, data)
            s.to_csv(OUT / f"scores_{tag}_{k}.csv", index=False)
            res[k] = {c: float(s[c].mean()) for c in s.columns if c != "patient_id"}
        a = audit
        res["retrieval_audit"] = {
            "n_level_sentences": int(len(a)), "replaced_R": float(a.replaced_R.mean()),
            "replaced_level_mismatch": float((a.replaced_R & ~a.level_match).sum() / max(a.replaced_R.sum(), 1)),
            "replaced_finding_mismatch": float((a.replaced_R & ~a.finding_match).sum() / max(a.replaced_R.sum(), 1)),
            "accepted_RC": float(a.accepted_RC.mean()), "median_sim": float(a.sim.median())}
        sweep = []
        for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            rep_t = a.sim >= t
            sweep.append({"tau": t, "replaced": float(rep_t.mean()),
                          "faithful_among_replaced": float((rep_t & a.level_match & a.finding_match).sum() / max(rep_t.sum(), 1))})
        res["tau_sweep"] = sweep
        if tag == "bench20":
            res["lexical"] = {k: lexical(rp, pids) for k, rp in reps.items()}
        summary[tag] = res
        print(tag, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()
                                                                     if isinstance(vv, float)}) for k, v in res.items()
                                if k not in ("tau_sweep", "lexical")}, indent=1), flush=True)

    # stored LLM outputs and earlier RATF text, rescored with the same extractor
    bench_inputs = level_inputs(BENCH_IDS)
    rows = []
    for nm, rp in stored_reports().items():
        rp = {p: t for p, t in rp.items() if p in BENCH_IDS}
        s = score_source(rp, gts, bench_inputs)
        rows.append({"source": nm, "n": len(s), **{c: float(s[c].mean()) for c in s.columns if c != "patient_id"}})
    for k in ["T", "R", "RC"]:
        s = pd.read_csv(OUT / f"scores_bench20_{k}.csv")
        rows.append({"source": f"RATF-{k}", "n": len(s), **{c: float(s[c].mean()) for c in s.columns if c != "patient_id"}})
    tab = pd.DataFrame(rows).sort_values("F1")
    tab.to_csv(OUT / "bench20_unified_scores.csv", index=False)
    print(tab.round(3).to_string())
    # paired bootstrap: RATF-T vs best LLM
    with open(OUT / "ratf_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
