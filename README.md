# PC-RadNet

> A prompt-calibrated, parameter-efficient foundation model for comprehensive multi-task segmentation, clinical assessment and structured report generation in lumbar spine MRI.

---

## Overview

Lumbar spine MRI analysis requires concurrent anatomical segmentation, disc degeneration grading, height quantification, multi-label pathology detection and structured report generation — tasks traditionally addressed by separate resource-intensive models that impose spatial annotation requirements at inference. **PC-RadNet** unifies these capabilities within a single lightweight architecture, eliminating the need for bounding-box or point annotations at test time.

Built upon a frozen DINOv2 backbone with only **2.73 M trainable parameters** (3.08% of total), PC-RadNet introduces three novel components: a **Spine Domain Adapter** for radiological feature adaptation, a **Prompt Calibrator** for multi-conditional feature gating via clinical text prompts, and a multi-task head ensemble for simultaneous prediction of segmentation masks, Pfirrmann grades, disc heights and spinal disorders. A **retrieval-augmented template filling (RATF)** module generates structured radiology reports from PC-RadNet predictions compliant with RSNA guidelines. The entire multi-task pipeline executes in **39 ms per image** on a single consumer GPU (0.63 GB VRAM).

---

## Architecture

![PC-RadNet Framework](assets/architecture.pdf)

PC-RadNet produces five clinical outputs in a single forward pass:

- **Semantic segmentation** — 7 anatomical classes across axial and sagittal planes (IVD, PE, SC, VB, IVD_sag, Sacrum, SC_sag)
- **Pfirrmann grading** — Ordinal disc degeneration assessment (Grades I–V) via CORN regression
- **Disc height estimation** — Continuous regression in millimetres
- **Disorder detection** — Multi-label classification for 9 spinal pathologies
- **Structured reporting** — RSNA-compliant radiology reports via RATF

---

## Results

### Segmentation (Ablation Study, Test Set, n=624)

| Variant | Params | Dice (D̄) | IoU (J̄) | Grade Acc. | VRAM |
|---------|--------|-----------|---------|------------|------|
| A0 | 1.00 M | 0.378 | 0.261 | 0.528 | 0.54 GB |
| A1 | 2.19 M | 0.375 | 0.260 | 0.652 | 0.62 GB |
| **A2** | **2.73 M** | **0.861** | **0.770** | **0.618** | **0.63 GB** |
| A3 | 2.73 M | 0.871 | 0.785 | 0.583 | 0.63 GB |

### Report Generation (Top 5 of 13 Methods, n=20 Matched Patients)

| Method | BLEU-1 | ROUGE-L | BERTScore | Factual F1 |
|--------|--------|---------|-----------|------------|
| **RATF (ours)** | **0.221** | **0.298** | **0.870** | **0.798** |
| Med-PaLM | 0.103 | 0.150 | 0.828 | 0.711 |
| Meditron3-8B | 0.060 | 0.106 | 0.813 | 0.699 |
| Llama-3.2-3B | 0.052 | 0.099 | 0.813 | 0.680 |
| BioMistral-7B | 0.097 | 0.134 | 0.827 | 0.655 |

Full results including per-class segmentation, cross-dataset validation on SPIDER, RSNA 2024, SpineWeb and SpineParseNet are reported in the paper.

---

## Repository Structure

```
PC-RadNet/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── configs/
│   └── config.yaml
├── data/
│   ├── README.md
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── src/
│   └── pc_radnet/
│       └── __init__.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_report.py
│   ├── visualize.py
│   ├── generate_figures.py
│   ├── baseline_zeroshot.py
│   ├── evaluate_external.py
│   ├── evaluate_external_v2.py
│   ├── benchmark_report_llms.py
│   ├── evaluate_nlg.py
│   ├── improve_sc_segmentation.py
│   ├── disorder_analysis.py
│   ├── optimize_thresholds.py
│   ├── compute_bootstrap_cis.py
│   └── split_build.py
├── results/
│   ├── quantitative/
│   ├── qualitative/
│   └── figures/
├── manuscript/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
└── assets/
    └── architecture.pdf
```

---

## Installation

```bash
# Clone
git clone https://github.com/<username>/PC-RadNet.git
cd PC-RadNet

# Create conda environment
conda env create -f environment.yml
conda activate pc_radnet
```

---

## Dataset

PC-RadNet is trained and evaluated on the **LSMA-PQR** dataset (515 patients, 4,120 images, 7 anatomical classes, Pfirrmann grades, disc heights, 9 disorder labels).

The dataset is publicly available on [Mendeley Data](https://data.mendeley.com/datasets/p3r4xd2488/1). After downloading, update paths in `configs/config.yaml`.

---

## Training

```bash
# Phase 1 — Warmup (5 epochs, frozen backbone, high LR)
python scripts/train.py --phase=1

# Phase 2 — Fine-tune (up to 50 epochs, cosine annealing, early stopping)
python scripts/train.py --phase=2 --ablation=A2
```

Training follows a two-phase protocol: Phase 1 uses a learning rate of 1×10⁻⁴ with AdamW and batch size 2 (effective batch size 16 via gradient accumulation). Phase 2 reduces the learning rate to 5×10⁻⁵ with cosine annealing and warm restarts (T₀=10, T_mult=2), with early stopping patience of 10 epochs.

---

## Evaluation

```bash
# Evaluate a single ablation variant
python scripts/evaluate.py --checkpoint best_by_mdice_A2.pt --split test

# Evaluate all ablation variants with statistical tests
python scripts/evaluate.py --all-ablations --split test
```

---

## Report Generation

```bash
# Template-based report generation from predictions
python scripts/generate_report.py --mode template \
    --predictions results/quantitative/test_per_image.csv \
    --output-dir results/reports/

# Build FAISS RAG index for retrieval-augmented reports
python scripts/generate_report.py --mode rag --build-index

# Evaluate report quality
python scripts/generate_report.py --evaluate-reports \
    --output-dir results/reports/
```

---

## Citation

```bibtex
@article{masood2026pcradnet,
  title   = {PC-RadNet: A Prompt-Calibrated, Parameter-Efficient Foundation Model
             for Comprehensive Multi-Task Segmentation, Clinical Assessment and
             Report Generation in Spine MRI},
  author  = {Masood, Rao Farhat and Taj, Imtiaz Ahmad},
  journal = {Medical Image Analysis},
  year    = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
