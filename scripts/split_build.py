"""
split_build.py - Section 2: Patient-level stratified splits.
Produces:
  scripts/data/train.txt, val.txt, test.txt  - image paths per line
  results/split_report.csv                    - patient → split mapping
  results/class_weights.json                  - inverse frequency weights
"""
import os, json, random, collections
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import yaml
from repo_config import load_config
SCRIPT_DIR = Path(__file__).resolve().parent
CFG = load_config()

SEED = CFG["project"]["seed"]
random.seed(SEED); np.random.seed(SEED)

IMG_DIR   = Path(CFG["paths"]["images"])
MASK_DIR  = Path(CFG["paths"]["masks"])
LBL_DIR   = Path(CFG["paths"]["labels"])
DATA_DIR  = Path(CFG["paths"]["splits_dir"])
RESULTS   = Path(CFG["paths"]["results"])
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = Path(CFG["paths"]["dataset_root"])

# ── Parse patient IDs and files ──────────────────────────────────────────
imgs = sorted(os.listdir(IMG_DIR))
patient_files = collections.defaultdict(list)
for f in imgs:
    stem = f.replace(".png", "")
    parts = stem.split("_")
    pid = parts[2]  # e.g., "0001"
    patient_files[pid].append(f)

patient_ids = sorted(patient_files.keys())
print(f"Total patients: {len(patient_ids)}")

# ── Load Pfirrmann for stratification ────────────────────────────────────
pfirr = pd.read_csv(DATASET_ROOT / "PfirrmannGrade.csv")
# Create a composite stratification label: max Pfirrmann across D3/D4/D5
pfirr["max_grade"] = pfirr[["D3", "D4", "D5"]].max(axis=1)
pid_to_grade = dict(zip(pfirr["Patient_ID"].astype(str).str.zfill(4), pfirr["max_grade"]))

# Build arrays for stratification
pids_arr = np.array(patient_ids)
strat_labels = np.array([pid_to_grade.get(p, 2) for p in patient_ids])

# Bin rare grades (1 and 5) into adjacent bins to ensure min 2 per class
# Grade 1 → 2, Grade 5 → 4 for stratification only (original grades unchanged)
strat_binned = strat_labels.copy()
strat_binned[strat_binned == 1] = 2
strat_binned[strat_binned == 5] = 4

# ── Stratified split: Train 70% / Val 15% / Test 15% ────────────────────
# First split: 70% train, 30% temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(sss1.split(pids_arr, strat_binned))

# Second split: temp → 50% val, 50% test (= 15% each of total)
temp_pids = pids_arr[temp_idx]
temp_labels = strat_binned[temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx_rel, test_idx_rel = next(sss2.split(temp_pids, temp_labels))
val_idx = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]

train_pids = set(pids_arr[train_idx])
val_pids = set(pids_arr[val_idx])
test_pids = set(pids_arr[test_idx])

print(f"Train: {len(train_pids)} patients")
print(f"Val:   {len(val_pids)} patients")
print(f"Test:  {len(test_pids)} patients")

# Verify no leakage
assert len(train_pids & val_pids) == 0, "LEAKAGE: train ∩ val"
assert len(train_pids & test_pids) == 0, "LEAKAGE: train ∩ test"
assert len(val_pids & test_pids) == 0, "LEAKAGE: val ∩ test"
assert len(train_pids | val_pids | test_pids) == len(patient_ids), "MISSING patients"
print("[OK] No leakage. All patients assigned.")

# ── Verify Pfirrmann distribution per split ──────────────────────────────
def grade_dist(pid_set):
    grades = [pid_to_grade.get(p, 2) for p in pid_set]
    return dict(collections.Counter(grades))

print(f"Train grade dist: {grade_dist(train_pids)}")
print(f"Val grade dist:   {grade_dist(val_pids)}")
print(f"Test grade dist:  {grade_dist(test_pids)}")

# ── Write split manifests ────────────────────────────────────────────────
def write_split(split_name, pid_set):
    fpath = DATA_DIR / f"{split_name}.txt"
    lines = []
    for pid in sorted(pid_set):
        for f in sorted(patient_files[pid]):
            lines.append(str(IMG_DIR / f))
    with open(fpath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[OK] {split_name}.txt: {len(lines)} images")
    return lines

train_files = write_split("train", train_pids)
val_files = write_split("val", val_pids)
test_files = write_split("test", test_pids)

# ── Write split report CSV ──────────────────────────────────────────────
rows = []
for pid in sorted(patient_ids):
    if pid in train_pids:
        split = "train"
    elif pid in val_pids:
        split = "val"
    else:
        split = "test"
    rows.append({
        "Patient_ID": pid,
        "Split": split,
        "Num_Images": len(patient_files[pid]),
        "Max_Pfirrmann": pid_to_grade.get(pid, -1),
        "Files": ";".join(sorted(patient_files[pid])),
    })
split_df = pd.DataFrame(rows)
split_df.to_csv(RESULTS / "split_report.csv", index=False)
print(f"[OK] split_report.csv: {split_df.shape}")

# ── Compute class frequency weights ─────────────────────────────────────
print("Computing class weights from train masks...")
# Count pixels per class across train split (per plane)
axial_class_px = collections.Counter()
sag_class_px = collections.Counter()
n_axial = 0; n_sag = 0

for img_path in train_files:
    fname = os.path.basename(img_path)
    mask_path = MASK_DIR / fname
    arr = np.array(Image.open(mask_path))
    if fname.startswith("axial"):
        n_axial += 1
        for c in range(4):
            axial_class_px[c] += int((arr == c).sum())
    else:
        n_sag += 1
        for c in range(5):
            sag_class_px[c] += int((arr == c).sum())

# Inverse frequency: w_c = N_total / (N_classes * N_c)
def compute_weights(px_counts, n_classes):
    total = sum(px_counts.values())
    weights = {}
    for c in range(n_classes):
        nc = px_counts.get(c, 1)
        weights[c] = total / (n_classes * max(nc, 1))
    return weights

axial_weights = compute_weights(axial_class_px, 4)
sag_weights = compute_weights(sag_class_px, 5)

class_weights = {
    "axial": {str(k): round(v, 4) for k, v in axial_weights.items()},
    "sagittal": {str(k): round(v, 4) for k, v in sag_weights.items()},
    "axial_pixel_counts": {str(k): v for k, v in axial_class_px.items()},
    "sagittal_pixel_counts": {str(k): v for k, v in sag_class_px.items()},
}

with open(RESULTS / "class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)
print(f"[OK] class_weights.json written")
print(f"  Axial weights: {axial_weights}")
print(f"  Sagittal weights: {sag_weights}")

# ── Verify assertions ────────────────────────────────────────────────────
total_train = len(train_files)
total_val = len(val_files)
total_test = len(test_files)
total = total_train + total_val + total_test
print(f"\nFinal split: train={total_train} ({100*total_train/total:.1f}%) "
      f"val={total_val} ({100*total_val/total:.1f}%) "
      f"test={total_test} ({100*total_test/total:.1f}%)")

# Verify all files have matching label and mask
missing = []
for split_files in [train_files, val_files, test_files]:
    for img_path in split_files:
        fname = os.path.basename(img_path).replace(".png", ".txt")
        if not (LBL_DIR / fname).exists():
            missing.append(f"label: {fname}")
        mask_fname = os.path.basename(img_path)
        if not (MASK_DIR / mask_fname).exists():
            missing.append(f"mask: {mask_fname}")

if missing:
    print(f"[WARN] Missing files: {missing[:10]}")
else:
    print("[OK] All images have matching labels and masks")

print("\n" + "=" * 60)
print("SECTION 2 COMPLETE — SPLITS BUILT")
print("=" * 60)
