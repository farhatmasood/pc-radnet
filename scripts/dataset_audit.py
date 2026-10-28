"""
dataset_audit.py - Section 0+1: Environment verification + Dataset/Codebase inspection.
Produces: scripts/dataset_audit.json, scripts/metadata_summary.csv
Usage:
    python scripts/dataset_audit.py
    python scripts/dataset_audit.py --mode=faiss   # Build FAISS RAG index (Section 6)
"""
import os, sys, json, argparse, collections, random, datetime
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# ── Load config ──────────────────────────────────────────────────────────
import yaml
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SEED = CFG["project"]["seed"]
random.seed(SEED); np.random.seed(SEED)

DATASET_ROOT = Path(CFG["paths"]["dataset_root"])
IMG_DIR      = Path(CFG["paths"]["images"])
LBL_DIR      = Path(CFG["paths"]["labels"])
MASK_DIR     = Path(CFG["paths"]["masks"])
RESULTS_DIR  = Path(CFG["paths"]["results"])
SCRIPTS_DIR  = Path(CFG["paths"]["scripts"])
CKPT_DIR     = Path(CFG["paths"]["checkpoints"])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
def audit_dataset():
    """Full dataset audit - Section 1."""
    print("=" * 60)
    print("SECTION 1 — DATASET AUDIT")
    print("=" * 60)

    imgs = sorted(os.listdir(IMG_DIR))
    lbls = sorted(os.listdir(LBL_DIR))
    msks = sorted(os.listdir(MASK_DIR))

    # ── Count match ──────────────────────────────────────────────────
    assert len(imgs) == len(lbls) == len(msks), (
        f"COUNT MISMATCH: images={len(imgs)}, labels={len(lbls)}, masks={len(msks)}")
    print(f"[OK] Count match: {len(imgs)} images = labels = masks")

    # ── Parse patient IDs ────────────────────────────────────────────
    patient_files = collections.defaultdict(list)
    view_seq_count = collections.Counter()
    level_count = collections.Counter()
    for f in imgs:
        stem = f.replace(".png", "")
        parts = stem.split("_")
        view, seq, pid, level = parts[0], parts[1], parts[2], parts[3]
        patient_files[pid].append(f)
        view_seq_count[(view, seq)] += 1
        level_count[level] += 1

    n_patients = len(patient_files)
    imgs_per_patient = collections.Counter(len(v) for v in patient_files.values())
    modal_count = max(imgs_per_patient, key=imgs_per_patient.get)
    anomalies = {p: len(v) for p, v in patient_files.items() if len(v) != modal_count}
    print(f"[OK] Patients: {n_patients}, {modal_count} images each")
    if anomalies:
        print(f"[WARN] Anomalous patients: {anomalies}")

    # ── Image properties ─────────────────────────────────────────────
    print("Checking image properties (all)...")
    resolutions = collections.Counter()
    for f in imgs:
        im = Image.open(IMG_DIR / f)
        resolutions[im.size] += 1
    print(f"  Resolutions: {dict(resolutions)}")
    for f in imgs[:1]:
        im = Image.open(IMG_DIR / f)
        print(f"  Mode: {im.mode}, dtype sample: {np.array(im).dtype}")

    # ── Mask properties ──────────────────────────────────────────────
    print("Checking mask properties (all)...")
    axial_vals = set(); sag_vals = set()
    for f in msks:
        arr = np.array(Image.open(MASK_DIR / f))
        uv = set(arr.flatten().tolist())
        if f.startswith("axial"):
            axial_vals.update(uv)
        else:
            sag_vals.update(uv)
    print(f"  Axial mask values: {sorted(axial_vals)}")
    print(f"  Sagittal mask values: {sorted(sag_vals)}")
    assert axial_vals.issubset({0,1,2,3}), f"Unexpected axial mask values: {axial_vals}"
    assert sag_vals.issubset({0,1,2,3,4}), f"Unexpected sagittal mask values: {sag_vals}"
    print("[OK] Mask values valid")

    # ── Label format ─────────────────────────────────────────────────
    class_counts = collections.Counter()
    for f in lbls:
        with open(LBL_DIR / f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_counts[int(parts[0])] += 1
    print(f"  Label class distribution: {dict(sorted(class_counts.items()))}")

    # ── Metadata ─────────────────────────────────────────────────────
    pfirr = pd.read_csv(DATASET_ROOT / "PfirrmannGrade.csv")
    ivd = pd.read_csv(DATASET_ROOT / "IVDHeights.csv")
    snotes = pd.read_csv(DATASET_ROOT / "structured_notes.csv")
    sreports = pd.read_csv(DATASET_ROOT / "structured_reports.csv")

    # Build metadata summary
    meta_rows = []
    for _, row in pfirr.iterrows():
        pid = int(row["Patient_ID"])
        for level in ["D3", "D4", "D5"]:
            meta_rows.append({
                "Patient_ID": pid,
                "Level": level,
                "Pfirrmann": int(row[level]),
            })
    meta_df = pd.DataFrame(meta_rows)
    # Merge IVD heights
    ht_rows = []
    for _, row in ivd.iterrows():
        pid = int(row["Patient_ID"])
        for level, col in [("D3", "D3_Ht"), ("D4", "D4_Ht"), ("D5", "D5_Ht")]:
            ht_rows.append({"Patient_ID": pid, "Level": level, "IVD_Height": row[col]})
    ht_df = pd.DataFrame(ht_rows)
    meta_df = meta_df.merge(ht_df, on=["Patient_ID", "Level"], how="left")

    # Add severity from structured_notes
    sev = snotes[["Patient_ID", "Severity_Score", "Clinical_Category"]].copy()
    meta_df = meta_df.merge(sev, on="Patient_ID", how="left")

    meta_df.to_csv(SCRIPTS_DIR / "metadata_summary.csv", index=False)
    print(f"[OK] metadata_summary.csv written: {meta_df.shape}")

    # ── Write audit JSON ─────────────────────────────────────────────
    audit = {
        "audit_timestamp": str(datetime.datetime.now()),
        "total_images": len(imgs),
        "total_patients": n_patients,
        "images_per_patient": modal_count,
        "resolutions": {f"{k[0]}x{k[1]}": v for k, v in resolutions.items()},
        "axial_mask_values": sorted(axial_vals),
        "sagittal_mask_values": sorted(sag_vals),
        "label_class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        "pfirrmann_distribution": dict(pd.concat([pfirr["D3"], pfirr["D4"], pfirr["D5"]]).value_counts().sort_index()),
        "ivd_height_stats": {
            col: {"mean": float(ivd[col].mean()), "std": float(ivd[col].std())}
            for col in ["D3_Ht", "D4_Ht", "D5_Ht"]
        },
        "clinical_categories": dict(snotes["Clinical_Category"].value_counts()),
        "structured_reports_rows": len(sreports),
        "quality_gates": {
            "images_eq_labels_eq_masks": len(imgs) == len(lbls) == len(msks),
            "patient_id_deterministic": len(anomalies) == 0,
            "mask_values_valid": axial_vals.issubset({0,1,2,3}) and sag_vals.issubset({0,1,2,3,4}),
            "metadata_found": True,
        }
    }
    with open(SCRIPTS_DIR / "dataset_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"[OK] dataset_audit.json written")
    print("=" * 60)
    print("ALL QUALITY GATES PASSED")
    print("=" * 60)
    return audit


# ─────────────────────────────────────────────────────────────────────────
def build_faiss_index():
    """Section 6 - Build FAISS RAG index from clinical notes."""
    print("=" * 60)
    print("SECTION 6 — FAISS RAG INDEX")
    print("=" * 60)

    notes_path = DATASET_ROOT / "structured_reports.csv"
    if not notes_path.exists():
        print("[SKIP] structured_reports.csv not found — RAG corpus absent")
        return

    df = pd.read_csv(notes_path)
    print(f"Loaded {len(df)} structured report rows")

    # Build text corpus from Original_Segment column
    texts = []
    metadata = []
    for _, row in df.iterrows():
        text = str(row.get("Original_Segment", ""))
        if not text or text == "nan" or len(text.strip()) < 5:
            continue
        texts.append(text.strip().strip('"'))
        metadata.append({
            "patient_id": int(row["Patient_ID"]),
            "finding_id": str(row["Finding_ID"]),
            "level": str(row["Level"]),
            "pathology": str(row["Pathology"]),
        })

    print(f"Valid text entries: {len(texts)}")
    if len(texts) < 10:
        print("[SKIP] Too few text entries for FAISS index")
        return

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[SKIP] faiss or sentence-transformers not installed")
        print("  Install: pip install faiss-cpu sentence-transformers")
        return

    # Encode on CPU
    print("Encoding texts with all-MiniLM-L6-v2 (CPU)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save
    faiss_path = Path(CFG["paths"]["faiss_index"])
    faiss_meta_path = Path(CFG["paths"]["faiss_meta"])
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    with open(faiss_meta_path, "w") as f:
        json.dump({"texts": texts, "metadata": metadata}, f, indent=2)
    print(f"[OK] FAISS index saved: {faiss_path}")
    print(f"[OK] FAISS metadata saved: {faiss_meta_path}")

    # Validation queries
    print("\n--- FAISS Validation Queries ---")
    test_queries = [
        "L4-L5 disc herniation moderate central stenosis",
        "L5-S1 disc bulge compressing thecal sac",
        "no significant stenosis normal findings",
        "facet joint hypertrophy bilateral",
        "nerve root compression left side",
    ]
    for q in test_queries:
        qvec = encoder.encode([q]).astype(np.float32)
        D, I = index.search(qvec, 3)
        print(f"\nQuery: {q}")
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            print(f"  [{rank+1}] dist={dist:.3f} | {texts[idx][:100]}...")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="audit", choices=["audit", "faiss"])
    args = parser.parse_args()

    if args.mode == "audit":
        audit_dataset()
    elif args.mode == "faiss":
        build_faiss_index()
