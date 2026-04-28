# Dataset: LSMA-PQR

The **LSMA-PQR** dataset comprises 515 consecutive symptomatic patients (4,120 multi-planar lumbar spine MRI images at 384×384 resolution). Each patient contributes eight acquisitions: three axial disc-level slices (L3-L4, L4-L5, L5-S1) in T1- and T2-weighted sequences and two mid-sagittal slices (T1w, T2w). Annotations include pixel-level segmentation masks for seven anatomical structures, Pfirrmann degeneration grades (I–V) for 1,545 disc-level assessments, intervertebral disc height measurements, and binary disorder labels for nine pathological findings across 515 structured radiological reports.

**Access:** The dataset is publicly available on Mendeley Data at [https://data.mendeley.com/datasets/p3r4xd2488/1](https://data.mendeley.com/datasets/p3r4xd2488/1).

**Splits:** Patient-level stratified splits (70/15/15%) are provided in `train.txt`, `val.txt`, and `test.txt`. Stratification is based on the maximum Pfirrmann grade per patient to ensure proportional grade representation across splits.

> **Note:** Raw imaging data is not included in this repository. Download the dataset from the link above and update `configs/config.yaml` with local paths before training.
