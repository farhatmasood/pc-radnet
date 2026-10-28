# Dataset: LSMA-PQR

LSMA-PQR contains 515 consecutive symptomatic patients with 4,120 multi-planar lumbar spine MRI images (384 × 384).

**Images per patient (eight):**
- axial T1w and T2w slices at L3-L4, L4-L5 and L5-S1 (six images);
- one mid-sagittal T1w and one mid-sagittal T2w slice.

**Annotations:**
- **Segmentation masks for seven structures:** axial IVD, posterior elements and spinal canal; sagittal vertebral body, IVD, sacrum and spinal canal. Masks are provided in YOLO-polygon and indexed formats.
- **Pfirrmann grades (I to V):** for 1,545 discs, graded on mid-sagittal T2w images.
- **Disc heights in millimetres:** for the same discs, measured on mid-sagittal T2w images.
- **Reports and labels:** 515 structured reports with binary labels for nine findings.

**Access:** [Mendeley Data](https://data.mendeley.com/datasets/p3r4xd2488/2). The imaging data are not part of this repository.

## Setup

After download, place the release at `data/LSMA-PQR/`, or set `paths.dataset_root` and the paths below it in `configs/config.yaml` to the local copy. The scripts expect these folders and files:

```
LSMA-PQR/
├── images_png/            axial_T1_0001_D3.png, ..., sag_T2_0001_S8.png
├── labels_yolo/           YOLO-polygon labels with the same stem
├── masks_indexed/         per-plane indexed masks (axial 0-3, sagittal 0-4)
├── PfirrmannGrade.csv
├── IVDHeights.csv
├── structured_reports.csv
├── structured_notes.csv
├── notes.csv
└── Report/
```

## Splits

`train.txt`, `val.txt` and `test.txt` list one image file name per line:

| Split | Patients | Images | Share |
|---|---|---|---|
| Train | 360 | 2,880 | 70% |
| Validation | 77 | 616 | 15% |
| Test | 78 | 624 | 15% |

- **Construction:** the splits are patient-level and stratified by each patient's maximum Pfirrmann grade. `scripts/split_build.py` builds them.
- **Loading:** `scripts/repo_config.py` joins each name with `paths.images`.
- **Use of the test split:** never used for model selection, early stopping or threshold tuning.
