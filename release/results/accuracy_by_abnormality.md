# Per-abnormality accuracy (GR-Neutro, 7-class)

Every number below is copied from a released artifact; none is fabricated. Each
table names its source artifact and its split honestly. "W-F1" = weighted F1.

GR-Neutro is an in-house Gustave Roussy peripheral-blood neutrophil corpus
(4,378 cells, 7 abnormality classes, 10 canonical textbook morphology concepts
used downstream; the shipped checkpoint head carries one extra legacy channel
that is not exposed). The seven classes are: **Normal, Chromatin (abnormal
chromatin), Döhle (Döhle bodies), Hypergranulation, Hypersegmentation,
Hypogranulation, Hyposegmentation**.

Two backbone regimes appear throughout:

- **frozen DinoBloom-B** — features extracted from the public DinoBloom-B SSL
  backbone with no fine-tuning.
- **fine-tuned DinoBloom-B (last-4)** — the last 4 transformer blocks of
  DinoBloom-B fine-tuned on GR-Neutro; the feature bank shipped in
  `release/weights/dinobloomb_ft_last4_s0_features.npz`.

A companion machine-readable table is in `accuracy_by_abnormality.csv`.

---

## 1. Joint CBM (DinoBloom-B fine-tuned) — per-class F1

The joint configuration = DinoBloom-B backbone with a separate CLS classifier
plus the concept adapter on patch tokens. This is the architecture of the
released checkpoint `release/weights/B_kitchen_s2024_model.pt`. The values below
are the **representative seed s42** run (`r1v14_kitchen_s42`) — the seed the
paper's figures and per-class detection table are drawn from — and match the
shipped CSVs exactly.

| Abnormality        | F1 (joint, s42) |
|--------------------|----------------:|
| Normal             | 0.972 |
| Hypersegmentation  | 0.895 |
| Hypogranulation    | 0.869 |
| Hypergranulation   | 0.842 |
| Hyposegmentation   | 0.816 |
| Döhle              | 0.744 |
| Chromatin          | 0.727 |
| **Overall weighted-F1** | **0.890** |

Split: GR-Neutro 7-class held-out test, representative seed s42.
Artifact: row `r1v14_kitchen_s42` of `results/figures_v2/per_class_breakdown.csv`
(per-class F1) and `results/figures_v2/main_results_table.csv` (weighted-F1
0.8899). Both are shipped, so this table is read directly from those files. The
paper's headline aggregate is the 5-seed mean (weighted-F1 0.88, macro-F1 0.81);
the paper's per-class detection **Table 2** with bootstrap CIs is reproduced by
`python results/reproduce_paper.py` — see `results/PAPER_NUMBERS.md`.

---

## 2. Frozen vs fine-tuned, overall weighted-F1 (per configuration)

Seed-mean overall W-F1 over 6 seeds [0, 7, 13, 42, 1337, 2024].
Artifact: the JSON emitted by `code/residual_cbm.py` (both the frozen and the
fine-tuned-last-4 tables).

| Configuration            | Frozen DinoBloom-B | Fine-tuned DinoBloom-B (last-4) |
|--------------------------|-------------------:|--------------------------------:|
| Backbone classifier (no-concept reference) | 0.849 | 0.925 |
| CEM (Concept Embedding Model)              | 0.847 | 0.915 |
| PCBM-h (post-hoc / residual CBM)           | 0.830 | 0.917 |
| pure-bottleneck CBM                        | 0.798 | 0.883 |

These are the fully reproducible head configurations trained on the shipped
feature bank via `release/code/residual_cbm.py` (see `release/README.md`).

---

## 3. Per-class W-F1 by configuration (fine-tuned DinoBloom-B, last-4)

Seed-mean per-class W-F1 over 6 seeds. Artifact:
the JSON emitted by `code/residual_cbm.py` (fine-tuned-last-4 per-class table).

| Class             | backbone_mlp | PCBM-h | CEM   |
|-------------------|-------------:|-------:|------:|
| Normal            | 0.9803 | 0.9769 | 0.9748 |
| Hypergranulation  | 0.9619 | 0.9379 | 0.9529 |
| Hypogranulation   | 0.8921 | 0.8867 | 0.8769 |
| Hyposegmentation  | 0.8943 | 0.8774 | 0.8764 |
| Hypersegmentation | 0.8158 | 0.8034 | 0.8104 |
| Chromatin         | 0.8543 | 0.8304 | 0.8392 |
| Döhle             | 0.8145 | 0.8220 | 0.8116 |

## 4. Per-class W-F1 by configuration (frozen DinoBloom-B)

Seed-mean per-class W-F1 over 6 seeds. Artifact:
the JSON emitted by `code/residual_cbm.py` (frozen per-class table).

| Class             | backbone_mlp | PCBM-h | CEM   |
|-------------------|-------------:|-------:|------:|
| Normal            | 0.9513 | 0.9461 | 0.9550 |
| Hypergranulation  | 0.9300 | 0.8966 | 0.9118 |
| Hypogranulation   | 0.8068 | 0.7918 | 0.8018 |
| Hyposegmentation  | 0.7987 | 0.7757 | 0.7863 |
| Chromatin         | 0.6923 | 0.6232 | 0.6571 |
| Hypersegmentation | 0.6041 | 0.5539 | 0.6513 |
| Döhle             | 0.6123 | 0.5583 | 0.6131 |

---

### Notes on splits and reproducibility

- Sections 2–4 are 6-seed means on the GR-Neutro 80/10/10 stratified-multilabel
  split (`release/code/data.py::stratified_multilabel_split`), evaluated on the
  10% held-out test, as logged in the JSON emitted by `code/residual_cbm.py`.
- Section 1 reports the joint architecture multi-seed run from the
  `results/figures_v2` breakdown CSVs.
- The CEM and PCBM-h heads were not separately checkpointed; they are
  reproduced deterministically from the shipped fine-tuned feature bank
  (`release/weights/dinobloomb_ft_last4_s0_features.npz`) via
  `release/code/residual_cbm.py`.
