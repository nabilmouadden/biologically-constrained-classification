# GR-Neutro concept-bottleneck reproducibility release

Runnable code, best-model weights, and per-abnormality accuracy for the
biologically-constrained haematology classification work: a no-concept backbone
classifier (frozen + fine-tuned) and four concept architectures — **CEM**
(Concept Embedding Model), **PCBM-h** (post-hoc / residual CBM), **pure-bottleneck
CBM**, and **joint CBM** — over a **DinoBloom-B** backbone on **GR-Neutro**
(in-house Gustave Roussy peripheral-blood neutrophil corpus; 4,378 cells, 7
abnormality classes, 10 textbook morphology concepts).

```
release/
├── code/                      # self-contained runnable code
│   ├── models.py              # DinoBloomBackbone, ConceptAdapter, ConstraintModule, JointModel
│   ├── residual_cbm.py        # PCBM-h + CEM + pure-bottleneck heads on cached features
│   ├── train.py               # end-to-end training (backbone + concept arch)
│   ├── data.py                # GR-Neutro loading + stratified-multilabel split + transforms
│   ├── morphometry_concepts_v2.py  # deterministic label-free morphometry (10 concepts)
│   ├── infer.py               # single-image inference -> class + concepts
│   ├── concept_config_gr_neutro.json  # concepts, class->concept matrix, constraint matrix
│   └── concepts_10.json       # the canonical 10 downstream concepts
├── data/                      # de-identified derived artifacts (no raw images)
│   ├── cell_manifest.json     # per-cell filename + label vector + class names
│   ├── morphometry_concepts.csv  # deterministic measured concept values
│   └── sample/sample_cell.png # one synthetic cell image for the infer.py demo
├── weights/                   # best-model weights (Git LFS) + their README
│   ├── B_kitchen_s2024_model.pt          (328 MB) joint CBM
│   ├── cbm_joint_s0_model.pt             (328 MB) pure-bottleneck CBM
│   ├── cbm_lcon4p0_s0_model.pt           (328 MB) pure-bottleneck CBM (λ=4.0)
│   ├── dinobloomb_ft_last4_s0_features.npz (14 MB) fine-tuned feature bank
│   │                                      (embeds labels + class_names + concepts)
│   └── README.md
├── requirements.txt           # pinned CPU dependencies (verified end-to-end)
└── results/
    ├── accuracy_by_abnormality.md   # per-abnormality accuracy tables
    ├── accuracy_by_abnormality.csv
    ├── PAPER_NUMBERS.md             # each paper number -> artifact -> command
    ├── reproduce_paper.py           # prints every paper number from shipped JSONs
    ├── acceptance_analyses/         # cached test-set analyses (paper Table 2, calibration, utility)
    │   ├── per_class_cis.json
    │   ├── per_class_calibration.json
    │   ├── clinical_utility.json
    │   └── compute_acceptance.py    # reproducer for the three JSONs above
    └── figures_v2/                  # joint-CBM per-class + summary CSVs (provenance)
```

---

## 0. Dependencies

- **Python** 3.10+
- **PyTorch** (CUDA build recommended for training; CPU works for single-image
  inference) + **torchvision**
- **timm** (loads the DinoBloom-B backbone from HF-hub)
- **numpy**, **scikit-learn**, **pandas**, **scipy**, **scikit-image**, **Pillow**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` carries the exact CPU pins the release was verified on
end-to-end (Python 3.12). It sets `--extra-index-url https://download.pytorch.org/whl/cpu`
so this single command resolves the **CPU** builds of torch/torchvision (no CUDA
or `nvidia-*` wheels are pulled). For a CUDA build instead, drop the `+cpu`
suffixes and install the matching CUDA torch/torchvision wheels.

### The DinoBloom-B backbone

The frozen DinoBloom-B backbone is the **public** checkpoint, fetched
automatically by `timm` the first time a model is built:

```
hf-hub:1aurent/vit_base_patch14_224.dinobloom
```

The **first** `infer.py` / `train.py` run fetches this backbone from HF-hub
(network required); it is then cached under `$HF_HOME` (defaults to
`~/.cache/huggingface`) and subsequent runs are offline. The released `*.pt`
checkpoints already contain the **fine-tuned last-N** backbone blocks plus all
heads; only the frozen lower backbone is fetched from HF-hub.

---

## Getting the weights after cloning

The four files under `release/weights/` are stored with **Git LFS**:

```bash
git clone git@github.com:nabilmouadden/biologically-constrained-classification.git
cd biologically-constrained-classification
git checkout release/models-code
git lfs install
git lfs pull            # materialises the 3x 328 MB *.pt + 14 MB *.npz
```

If `git lfs pull` is skipped, the `.pt`/`.npz` files will be small text
pointers, not real weights.

---

## 1. Extract backbone features (frozen + fine-tuned)

**Frozen** features: run the DinoBloom-B backbone in eval mode over every cell
and dump CLS tokens to an npz (`features` (N,768) float32, `paths` (N,) str).
The released fine-tuned bank already follows this schema.

**Fine-tuned (last-4)** features: the shipped
`weights/dinobloomb_ft_last4_s0_features.npz` is the fine-tuned CLS feature bank
(schema: `features` (N,768) float32, `paths` (N,) bare `<Class>/<file>.jpg`, plus
embedded `labels`, `class_names`, `concepts`). To regenerate it from your own
GR-Neutro copy, fine-tune the last-4 blocks of DinoBloom-B with `train.py`
(`--backbone dinobloom_b --unfreeze_last_n 4`) and extract CLS tokens over the
full corpus in eval mode.

---

## 2. Train each configuration

### 2a. Concept heads on cached features (CEM, PCBM-h, pure-bottleneck)

`residual_cbm.py` trains **CEM**, **PCBM-h**, and the **pure-bottleneck** head
on a feature bank (frozen or fine-tuned), multi-seed, with paired-bootstrap CIs
and the random-orthogonal-residual control + faithfulness curves:

The labels and measured concepts come from the two shipped de-identified files
under `data/`, which are the argparse defaults — so the headline command is just:

```bash
# fine-tuned features (shipped) + shipped data/ defaults -> runs end-to-end:
python code/residual_cbm.py --out runs/residual_cbm_ft.json

# equivalent, with paths spelled out:
python code/residual_cbm.py \
    --features  weights/dinobloomb_ft_last4_s0_features.npz \
    --manifest  data/cell_manifest.json \
    --morpho    data/morphometry_concepts.csv \
    --out       runs/residual_cbm_ft.json

# no-manifest fallback: derive labels from the feature bank's embedded
# 'labels'/'class_names' arrays (only needs the npz + morphometry CSV):
python code/residual_cbm.py --labels_from_npz \
    --morpho data/morphometry_concepts.csv --out runs/residual_cbm_ft.json

# frozen features: extract your own dinobloom_features.npz first, then:
python code/residual_cbm.py \
    --features runs/dinobloom_features.npz --labels_from_npz \
    --morpho data/morphometry_concepts.csv --out runs/residual_cbm_frozen.json
```

> The released CEM and PCBM-h heads were **not separately checkpointed**; they
> are reproduced deterministically from the shipped fine-tuned feature bank via
> this script. The frozen backbone they sit on is the public DinoBloom-B
> download.

### 2b. End-to-end backbone + concept architecture (joint CBM / pure CBM)

`train.py` trains the backbone together with the concept adapter and constraint
module. Pick the architecture with `--mode`:

```bash
# Joint CBM (separate CLS classifier + concept adapter) -> B_kitchen-style:
python code/train.py --tag joint_run --mode joint \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --data_csv /path/annotations.csv --data_root /path/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 2024

# Pure-bottleneck CBM (class flows only through concepts) -> cbm_*-style:
python code/train.py --tag cbm_run --mode cbm \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --lambda_concept_loss 2.0 \
    --data_csv /path/annotations.csv --data_root /path/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 0
```

Each run writes `outputs/<tag>/model.pt` (same format as the released
checkpoints) plus `predictions.pt` and `summary.json`.

A **no-concept backbone classifier** is the `--baseline` flag (adapter +
constraint frozen, class head only) or the `backbone_mlp` reference reported by
`residual_cbm.py`.

---

## 3. Inference on a new single-cell image

`infer.py` loads any released `model.pt`, predicts the abnormality class, prints
the learned concept activations, AND independently runs the deterministic
morphometry to print the 10 measured textbook-concept values:

```bash
cd code
# a synthetic sample image ships with the release for a first smoke run:
python infer.py \
    --weights ../weights/cbm_joint_s0_model.pt \
    --image   ../data/sample/sample_cell.png
# or point --image at your own single-cell PNG/JPG.
```

The first run fetches the DinoBloom-B backbone from HF-hub (§0). Output layout
(values are computed live from the image; the block below is the shipped-sample
run):

```
[infer] device=cpu  weights=../weights/cbm_joint_s0_model.pt
[infer] architecture mode=cbm  (10 concepts exposed)
=== Predicted abnormality class ===
  Hyposegmentation  (p=0.412)
  full class distribution:
    Hyposegmentation   0.412
    Chromatin          0.273
    ...
=== Model-predicted concept activations (learned concept head) ===
    nuclear_lobulation_degree          0.232
    ...   (10 concepts)
=== Measured morphometry concepts (deterministic, label-free; 10 canonical) ===
    nuclear_lobulation_degree          0.000
    ...   (10 concepts)
```

The architecture (joint vs pure-bottleneck CBM) is read from the checkpoint, so
the same command works for all three `*.pt` files, and exactly **10** concepts
are exposed (the checkpoint head's trailing legacy channel is not surfaced).

---

## 4. Per-abnormality accuracy

See `results/accuracy_by_abnormality.md` (+ `.csv`). Headline:

- **Detection (paper Table 2, representative seed s42, n=438):** per-class F1
  Normal 0.99, Hypersegmentation 0.91, Hypogranulation 0.90, Hypergranulation
  0.89, Hyposegmentation 0.80, Döhle 0.73, Chromatin 0.68, each with a 1000×
  bootstrap 95% CI. Paper aggregate over 5 seeds: **macro-F1 0.81, weighted-F1
  0.88** (single-seed s42 is 0.843 / 0.900).
- **Joint CBM per-class F1 (seed s42):** Normal 0.972, Hypersegmentation 0.895,
  Hypogranulation 0.869, Hypergranulation 0.842, Hyposegmentation 0.816, Döhle
  0.744, Chromatin 0.727; overall weighted-F1 0.890 (`r1v14_kitchen_s42`).
- **Interpretability cost — frozen → fine-tuned overall W-F1:** backbone 0.849 →
  0.925, CEM 0.847 → 0.915, PCBM-h 0.830 → 0.917, pure-bottleneck 0.798 → 0.883
  (6-seed mean, 95% CI [0.852, 0.883]).

Every number is labelled with its split and its source artifact, and no value is
fabricated. What reproduces from the release alone, and what needs your own
GR-Neutro image copy:

- **Reproducible from shipped artifacts (CPU):** the concept-head table (CEM,
  PCBM-h, pure-bottleneck, backbone-MLP; frozen and fine-tuned) via
  `code/residual_cbm.py` on the shipped feature bank + `data/`; single-image
  inference via `code/infer.py`; the paper's Table 2 / calibration / clinical-
  utility numbers via `python results/reproduce_paper.py` on the shipped cached
  JSONs; the `results/figures_v2/*.csv` provenance for the joint-CBM per-class F1.
- **Needs your own GR-Neutro images:** the end-to-end backbone fine-tune and the
  joint / pure CBM checkpoints from scratch via `code/train.py` (the corpus is not
  redistributed — see the data note below). The released `*.pt` weights are the
  frozen result of those runs.

### Reproduce the paper numbers

Every headline number maps to a shipped artifact and a command; the full table
is in **`results/PAPER_NUMBERS.md`**. One-shot check:

```bash
python results/reproduce_paper.py     # reads results/acceptance_analyses/*.json
```

| Paper number | Value | Artifact | Command |
|---|---|---|---|
| Per-class detection + bootstrap CIs (Table 2) | Normal 0.99 … Chromatin 0.68 | `results/acceptance_analyses/per_class_cis.json` | `python results/reproduce_paper.py` |
| Macro-F1 / weighted-F1 (5-seed aggregate) | 0.81 / 0.88 | 5-seed mean; single-seed s42 (0.843/0.900) from `per_class_cis.json` | `python results/reproduce_paper.py` |
| Calibration pooled ECE / Brier | 0.054 / 0.029 | `results/acceptance_analyses/per_class_calibration.json` | `python results/reproduce_paper.py` |
| Internal abnormal-vs-normal AUROC | 0.997 | `results/acceptance_analyses/clinical_utility.json` | `python results/reproduce_paper.py` |
| Triage acc no-defer / 10% / 20% | 0.836 / 0.886 / 0.946 | `results/acceptance_analyses/clinical_utility.json` | `python results/reproduce_paper.py` |
| Interpretability-head W-F1 (fine-tuned, 6-seed mean) | 0.925 / 0.915 / 0.917 / 0.867 | `weights/dinobloomb_ft_last4_s0_features.npz` + `data/` | `python code/residual_cbm.py --out runs/residual_cbm_ft.json` |
| External transfer AUROC (Barrera–Merino 2024) | 0.851 ± 0.049 | paper table `m1_external_block.tex`; external cohort not redistributed | — |

The three cached JSONs are the outputs of
`results/acceptance_analyses/compute_acceptance.py`, which recomputes them from
the frozen representative-seed (s42) test predictions.

### Concept counts (nested sets)

The concept head is trained on a set of textbook morphology concepts. Three
counts appear across the paper and this release and are **nested**, not
conflicting:

- **4 (paper, reported):** the reliable core retained after two-rater
  inter-rater-agreement (κ) filtering — the paper's quantitative concept analysis
  is restricted to this subset.
- **10 (release, canonical):** the fuller downstream concept vocabulary the
  checkpoints and `code/` operate on (`concepts_10.json`,
  `concept_config_gr_neutro.json`, `data/morphometry_concepts.csv`, `infer.py`).
- **The shipped `*.pt` head** was trained one channel wider than these 10; that
  trailing channel is stored as a neutral `legacy_channel_10` placeholder and is
  **intentionally not exposed** by any released file — `infer.py` slices the
  concept output to the canonical 10 at read-out, and the shipped morphometry CSV
  drops it.

So the paper's reported concepts (4) ⊂ the release-canonical concepts (10), a
curated subset — not a different set.

---

## Data note

GR-Neutro is an **in-house Gustave Roussy** peripheral-blood neutrophil corpus
and is **not** redistributed here. The code expects an `annotations.csv`
(header `filename,path,<class1>,...,<class7>`) and an image root. Public
datasets used elsewhere in the paper (AML Matek 2019, MLL-23, Bodzas 2023,
Acevedo 2019) are cited in the paper; DinoBloom-B's SSL pretraining did **not**
include GR-Neutro.
