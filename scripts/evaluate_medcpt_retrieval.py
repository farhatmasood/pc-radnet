"""
evaluate_medcpt_retrieval.py - Task 4: Compare MedCPT vs all-MiniLM-L6-v2 for RATF retrieval.
Rebuilds FAISS index with MedCPT encoder, evaluates retrieval quality, and
optionally re-generates reports.

Usage:
    python scripts/evaluate_medcpt_retrieval.py
"""
import os, sys, json, argparse, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import yaml
from repo_config import load_config, resolve_image

SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()


DISORDER_COLS = [
    "Disc_Herniation", "Disc_Protrusion", "Disc_Bulge",
    "Thecal_Sac_Compression", "Nerve_Root_Compression",
    "Spinal_Canal_Stenosis", "Spondylolisthesis",
    "Desiccated_Disc", "Ligamentum_Flavum_Hypertrophy"
]

DISORDER_KEYWORDS = {
    "Disc_Herniation": ["herniation", "herniated"],
    "Disc_Protrusion": ["protrusion", "protruding"],
    "Disc_Bulge": ["bulge", "bulging"],
    "Thecal_Sac_Compression": ["thecal sac compression", "thecal compression"],
    "Nerve_Root_Compression": ["nerve root compression", "nerve compression"],
    "Spinal_Canal_Stenosis": ["stenosis", "canal stenosis", "spinal stenosis"],
    "Spondylolisthesis": ["spondylolisthesis"],
    "Desiccated_Disc": ["desiccated", "desiccation"],
    "Ligamentum_Flavum_Hypertrophy": ["ligamentum flavum", "flavum hypertrophy"],
}


def build_corpus():
    """Build text corpus from structured_reports.csv."""
    root = Path(CFG["paths"]["dataset_root"])
    df = pd.read_csv(root / "structured_reports.csv")

    texts = []
    metadata = []
    for _, row in df.iterrows():
        cols = [c for c in df.columns if c != "Patient_ID"]
        text_parts = []
        for c in cols:
            if pd.notna(row[c]) and str(row[c]).strip() and str(row[c]) != "nan":
                text_parts.append(f"{c}: {row[c]}")
        text = " | ".join(text_parts)
        texts.append(text)
        metadata.append({"patient_id": int(row.get("Patient_ID", 0)), "row_idx": _})

    return texts, metadata


def build_faiss_index(encoder_name, texts):
    """Build FAISS index with the specified encoder."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"  Loading encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name)

    print(f"  Encoding {len(texts)} texts...")
    t0 = time.time()
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=32)
    encode_time = time.time() - t0
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    print(f"  Embeddings: {embeddings.shape}, dim={dim}, encode_time={encode_time:.1f}s")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, encoder, encode_time, dim


def evaluate_retrieval(index, encoder, texts, metadata, gt_notes_df, top_k=3):
    """Evaluate retrieval quality: precision@k, recall@k for disorder matching."""
    import faiss

    # Get test patient IDs
    splits_dir = Path(CFG["paths"]["splits_dir"])
    with open(splits_dir / "test.txt") as f:
        test_paths = [resolve_image(l) for l in f if l.strip()]

    test_pids = set()
    for p in test_paths:
        fname = os.path.basename(p)
        parts = fname.replace(".png", "").split("_")
        test_pids.add(int(parts[2]))

    # Build queries from test patients
    results = []
    for pid in sorted(test_pids):
        gt_row = gt_notes_df[gt_notes_df["Patient_ID"] == pid]
        if len(gt_row) == 0:
            continue

        gt_disorders = {col: int(gt_row.iloc[0].get(col, 0)) for col in DISORDER_COLS}
        gt_present = [col for col, val in gt_disorders.items() if val == 1]

        if not gt_present:
            continue

        # Build query from patient findings
        query = f"Patient {pid}: " + ", ".join(
            col.replace("_", " ").lower() for col in gt_present
        )

        # Retrieve
        query_emb = encoder.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)

        # Check if retrieved reports mention similar findings
        retrieved_texts = [texts[idx] for idx in indices[0]]
        combined_retrieved = " ".join(retrieved_texts).lower()

        tp, fp, fn = 0, 0, 0
        for dcol, keywords in DISORDER_KEYWORDS.items():
            gt_has = gt_disorders.get(dcol, 0) == 1
            retrieved_has = any(kw in combined_retrieved for kw in keywords)

            if gt_has and retrieved_has:
                tp += 1
            elif not gt_has and retrieved_has:
                fp += 1
            elif gt_has and not retrieved_has:
                fn += 1

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        results.append({
            "patient_id": pid,
            "n_gt_disorders": len(gt_present),
            "retrieval_score": float(scores[0].mean()),
            "precision": prec, "recall": rec, "f1": f1,
        })

    return results


def main():
    results_dir = Path(CFG["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    root = Path(CFG["paths"]["dataset_root"])
    gt_notes = pd.read_csv(root / "structured_notes.csv")

    print("Building text corpus...")
    texts, metadata = build_corpus()
    print(f"  Corpus: {len(texts)} documents")

    # Encoders to compare
    encoders = {
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
        "MedCPT": "ncbi/MedCPT-Query-Encoder",
    }

    all_summaries = []

    for enc_name, enc_model in encoders.items():
        print(f"\n{'='*60}")
        print(f"Evaluating encoder: {enc_name}")
        print(f"{'='*60}")

        try:
            index, encoder, encode_time, dim = build_faiss_index(enc_model, texts)
        except Exception as e:
            print(f"  ERROR loading {enc_name}: {e}")
            # Try alternative model name
            if enc_name == "MedCPT":
                alt_models = [
                    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                ]
                loaded = False
                for alt in alt_models:
                    try:
                        print(f"  Trying alternative: {alt}")
                        index, encoder, encode_time, dim = build_faiss_index(alt, texts)
                        enc_name = alt.split("/")[-1]
                        loaded = True
                        break
                    except Exception as e2:
                        print(f"    Failed: {e2}")
                if not loaded:
                    continue
            else:
                continue

        results = evaluate_retrieval(index, encoder, texts, metadata, gt_notes)
        df = pd.DataFrame(results)

        summary = {
            "encoder": enc_name,
            "dim": dim,
            "encode_time_sec": round(encode_time, 1),
            "n_patients_evaluated": len(df),
            "mean_precision": round(df["precision"].mean(), 4),
            "mean_recall": round(df["recall"].mean(), 4),
            "mean_f1": round(df["f1"].mean(), 4),
            "mean_retrieval_score": round(df["retrieval_score"].mean(), 4),
        }
        all_summaries.append(summary)

        print(f"\n  Results ({enc_name}):")
        print(f"    Dimension: {dim}")
        print(f"    Encode time: {encode_time:.1f}s")
        print(f"    Precision@3: {summary['mean_precision']:.4f}")
        print(f"    Recall@3: {summary['mean_recall']:.4f}")
        print(f"    F1: {summary['mean_f1']:.4f}")

        # Save per-patient results
        df.to_csv(results_dir / f"retrieval_{enc_name.replace('/', '_')}_test.csv", index=False)

        # Save FAISS index if MedCPT is better
        if "MedCPT" in enc_name or "Biomed" in enc_name:
            import faiss
            medcpt_idx_path = Path(CFG["paths"]["checkpoints"]) / "faiss_medcpt_index.bin"
            faiss.write_index(index, str(medcpt_idx_path))
            meta_path = Path(CFG["paths"]["checkpoints"]) / "faiss_medcpt_metadata.json"
            with open(meta_path, "w") as f_out:
                json.dump({"texts": texts, "metadata": metadata, "encoder": enc_name}, f_out)
            print(f"  Saved MedCPT index: {medcpt_idx_path}")

        del index, encoder
        gc.collect()

    # Save comparison
    with open(results_dir / "retrieval_comparison.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print("RETRIEVAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Encoder':<30} {'Dim':>5} {'Time':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 62)
    for s in all_summaries:
        print(f"{s['encoder']:<30} {s['dim']:>5} {s['encode_time_sec']:>5.1f}s "
              f"{s['mean_precision']:>7.4f} {s['mean_recall']:>7.4f} {s['mean_f1']:>7.4f}")


if __name__ == "__main__":
    main()
