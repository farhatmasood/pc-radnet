"""Run the zero-shot SAM ViT-B (automatic masks) and MedSAM-2 (GT box) baselines on the full
LSMA-PQR test split (624 images, 78 patients), reusing scripts/evaluate_expanded_baselines.py.
Outputs: results/analysis/baselines_full/{medsam2,sam_auto}_test_full.csv and *_timing.json"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_expanded_baselines as eb  # noqa: E402

split = ROOT / "data" / "test.txt"
out = ROOT / "results" / "analysis" / "baselines_full"
out.mkdir(parents=True, exist_ok=True)
print("masks:", eb.CFG["paths"]["masks"], Path(eb.CFG["paths"]["masks"]).exists())
print("split:", split, split.exists())

# one model per process: MedSAM-2 puts its own sam2 package on sys.path
runs = {"medsam2": eb.evaluate_medsam2, "sam_auto": eb.evaluate_sam2}
for name in (sys.argv[1:] or list(runs)):
    fn = runs[name]
    t0 = time.time()
    df, t_img = fn(str(split))
    df.to_csv(out / f"{name}_test_full.csv", index=False)
    info = {"avg_time_s": float(t_img), "total_s": time.time() - t0, "n_rows": len(df),
            "n_images": int(df.fname.nunique()), "n_patients": int(df.patient_id.nunique()),
            "mean_dice": float(df.dice.mean()), "mean_iou": float(df.iou.mean())}
    json.dump(info, open(out / f"{name}_timing.json", "w"), indent=1)
    print(name, info, flush=True)
