# GR-Neutro concept-bottleneck reproducibility release

Runnable code, released model weights, and **per-abnormality accuracy for every
weight** for the biologically-constrained haematology classification work: a
no-concept backbone classifier (frozen + fine-tuned) and the concept
architectures — **Joint CBM**, **Pure-bottleneck CBM**, **Sequential CBM**,
**Independent CBM**, plus **CEM** (Concept Embedding Model) and **PCBM-h**
(post-hoc / residual CBM) — over a **DinoBloom-B** backbone on **GR-Neutro**
(in-house Gustave Roussy peripheral-blood neutrophil corpus; 4,378 cells, 7
abnormality classes, 11 textbook morphology concepts, 10 used downstream).

**Weights are hosted on the Hugging Face Hub** (too large for GitHub):
👉 **https://huggingface.co/nabimu9/gr-neutro-cbm-weights**

```
<repo root>/
├── code/                      # self-contained runnable code
│   ├── models.py              # DinoBloomBackbone, ConceptAdapter, ConstraintModule, JointModel
│   ├── residual_cbm.py        # PCBM-h + CEM + pure-bottleneck heads on cached features
│   ├── train.py               # end-to-end training (backbone + concept arch)
│   ├── data.py                # GR-Neutro loading + stratified-multilabel split + transforms
│   ├── cache_ft_features.py   # fine-tune DinoBloom-B (last-N) and dump CLS feature bank
│   ├── morphometry_concepts_v2.py  # deterministic label-free morphometry (10 concepts)
│   ├── infer.py               # single-image inference -> class + concepts
│   ├── concept_config_gr_neutro.json  # concepts, class->concept matrix, constraint matrix
│   └── concepts_10.json       # the canonical 10 downstream concepts
├── weights/
│   ├── README.md              # what each released weight is + how to load it
│   └── download_weights.sh    # one-command pull of all weights from the HF Hub
└── results/
    ├── accuracy_by_abnormality.md
    └── accuracy_by_abnormality.csv
```

---

## Released weights and their accuracy

All weights live in the HF repo above. **Every released weight's accuracy is
reported below** (and in `results/accuracy_by_abnormality.{md,csv}`). Numbers come
straight from the run artifacts; nothing is fabricated. The two tables use
**different evaluation splits** (each end-to-end model's own held-out test split
vs. the residual-CBM stratified split), so compare *within* a table.

### End-to-end checkpoints — DinoBloom-B last-6 fine-tuned, seed 42

| Weight (`*.pt`) | Architecture | Test W-F1 | Macro-F1 | Subset acc. | Multi-seed mean W-F1 |
|---|---|--:|--:|--:|--:|
| `joint_cbm_dinobloomB_ft_s42.pt` | **Joint CBM** (separate classifier ∥ concept adapter + constraint) | **0.912** | 0.861 | 0.865 | 0.887 ± 0.012 |
| `pure_bottleneck_cbm_dinobloomB_ft_s42.pt` | **Pure-bottleneck CBM** (transparent; class only through concepts, λ=2) | **0.909** | 0.862 | 0.847 | ≈ 0.88 |
| `backbone_baseline_dinobloomB_ft_s42.pt` | **No-concept backbone baseline** (fine-tuned classifier) | **0.907** | 0.865 | 0.868 | 0.884 |
| `cbm_sequential_dinobloomB_ft_s42.pt` | **Sequential CBM** | **0.873** | 0.804 | 0.756 | — |
| `cbm_independent_dinobloomB_ft_s42.pt` | **Independent CBM** | **0.772** | 0.748 | 0.585 | — |

**Per-abnormality F1 (seed 42):**

| Class (n) | Joint CBM | Pure-bottleneck | Backbone baseline | Sequential | Independent |
|---|--:|--:|--:|--:|--:|
| Normal (199) | 0.987 | 0.980 | 0.982 | 0.962 | 0.791 |
| Hypogranulation (108) | 0.903 | 0.910 | 0.897 | 0.888 | 0.824 |
| Hyposegmentation (67) | 0.855 | 0.826 | 0.821 | 0.846 | 0.786 |
| Chromatin (35) | 0.694 | 0.750 | 0.719 | 0.429 | 0.437 |
| Hypersegmentation (19) | 0.895 | 0.919 | 0.919 | 0.919 | 0.872 |
| Döhle (19) | 0.848 | 0.757 | 0.848 | 0.743 | 0.684 |
| Hypergranulation (16) | 0.842 | 0.889 | 0.865 | 0.842 | 0.842 |

### Feature-bank heads — frozen DinoBloom-B, 6-seed mean W-F1 [95% CI]

Reproduced from the released feature banks (`*_features.npz`) via `residual_cbm.py`.

| Method | W-F1 | Macro-F1 | Interpretable? |
|---|--:|--:|---|
| `backbone_mlp` (reference head) | 0.8495 [0.844, 0.855] | 0.771 | no |
| `cem` (Concept Embedding Model) | 0.8471 [0.838, 0.856] | 0.768 | yes |
| `pcbmh` (residual CBM, r=10) | 0.8297 [0.817, 0.842] | 0.735 | yes |
| `pcbmh_highrank` (r=64) | 0.8318 [0.828, 0.836] | 0.740 | yes |
| `pure_bottleneck` (fully transparent) | 0.7984 [0.788, 0.808] | 0.698 | yes |

Two findings carried by these heads: **concepts are load-bearing for accuracy**
(PCBM-h beats a matched-rank random-orthogonal residual by +0.0385 W-F1 [+0.033,
+0.044]) but **not intervention-faithful** (replacing predicted concepts with the
true measured values barely moves the prediction: +0.002 / −0.001 W-F1 over a full
0→1 intervention for PCBM-h / CEM).

### Feature banks

| File (`*.npz`) | What | Shape |
|---|---|---|
| `dinobloom_b_frozen_features.npz` | Frozen DinoBloom-B CLS features, all 4,378 cells | `features` (4378, 768) |
| `dinobloom_b_ft_last4_features.npz` | Fine-tuned (last-4) DinoBloom-B CLS bank, seed 0 | `features` (4378, 768) |

---

## 0. Dependencies

- **Python** 3.10+
- **PyTorch** (CUDA for training; CPU works for single-image inference) + **torchvision**
- **timm** (loads the DinoBloom-B backbone from HF-hub)
- **numpy**, **scikit-learn**, **pandas**, **scipy**, **scikit-image**, **Pillow**
- **huggingface_hub** (to pull the weights)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or:
pip install torch torchvision timm numpy scikit-learn pandas scipy scikit-image pillow huggingface_hub
```

### The DinoBloom-B backbone

The frozen DinoBloom-B backbone is the **public** checkpoint, fetched
automatically by `timm` the first time a model is built:
`hf-hub:1aurent/vit_base_patch14_224.dinobloom`. The released `*.pt` checkpoints
already contain the fine-tuned last-N backbone blocks plus all heads.

---

## Getting the weights

All weights are on the HF Hub — no Git LFS needed.

```bash
# one command, pulls everything into ./weights/
bash weights/download_weights.sh

# or with the HF CLI directly:
hf download nabimu9/gr-neutro-cbm-weights --local-dir weights

# or in Python:
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('nabimu9/gr-neutro-cbm-weights', local_dir='weights')"
```

See `weights/README.md` for what each file is and how to load it.

---

## 1. Extract backbone features (frozen + fine-tuned)

The released banks already follow the schema (`features` (N,768) float32,
`paths` (N,) str). To regenerate the fine-tuned bank:

```bash
python code/cache_ft_features.py \
    --variant dinobloom_b --unfreeze_last_n 4 --epochs 30 --seed 0 \
    --annotations /path/to/gr_neutro/annotations.csv \
    --data_root   /path/to/gr_neutro \
    --out         runs/dinobloom_b_ft_last4_features.npz
```

---

## 2. Train each configuration

### 2a. Concept heads on cached features (CEM, PCBM-h, pure-bottleneck)

```bash
python code/residual_cbm.py \
    --features  weights/dinobloom_b_ft_last4_features.npz \
    --manifest  /path/to/cell_manifest_full_extended.json \
    --morpho    /path/to/morphometry_concepts.csv \
    --out       runs/residual_cbm_ft

# frozen bank:
python code/residual_cbm.py \
    --features  weights/dinobloom_b_frozen_features.npz \
    --manifest  ... --morpho ... --out runs/residual_cbm_frozen
```

CEM / PCBM-h heads are reproduced deterministically from a feature bank by this
script (the frozen backbone they sit on is the public DinoBloom-B download).

### 2b. End-to-end backbone + concept architecture

`train.py` trains the backbone with the concept adapter + constraint module.
Pick the architecture with `--mode`:

```bash
# Joint CBM (separate CLS classifier + concept adapter):
python code/train.py --tag joint_run --mode joint \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --data_csv /path/annotations.csv --data_root /path/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 42

# Pure-bottleneck CBM (class flows only through concepts):
python code/train.py --tag cbm_run --mode cbm --lambda_concept_loss 2.0 \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --data_csv /path/annotations.csv --data_root /path/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 42

# No-concept backbone baseline: add --baseline (classifier-driven head).
```

Sequential and Independent CBM are produced by the matching concept-training
schedule on top of `--mode cbm` (concepts-first-then-frozen vs. concept and class
heads trained independently). Each run writes `outputs/<tag>/model.pt` (same format
as the released checkpoints) + `predictions.pt` + `summary.json`.

---

## 3. Inference on a new single-cell image

`infer.py` loads any released `*.pt`, predicts the abnormality class, prints the
learned concept activations, AND independently runs the deterministic morphometry
to print the 10 measured textbook-concept values:

```bash
bash weights/download_weights.sh           # if not done yet
cd code
python infer.py \
    --weights ../weights/joint_cbm_dinobloomB_ft_s42.pt \
    --image   /path/to/one_cell.png
```

The architecture (joint / pure-bottleneck / sequential / independent CBM) is read
from each checkpoint's stored `args`, so the same command works for all of them.

---

## 4. Per-abnormality accuracy

Full tables in `results/accuracy_by_abnormality.md` (+ `.csv`) and in the
"Released weights and their accuracy" section above. Headline: the **Joint CBM**
and **Pure-bottleneck CBM** reach backbone-baseline parity (0.91 W-F1) while
exposing 10 auditable concepts; accuracy degrades down the
joint → sequential → independent training schedule (0.91 → 0.87 → 0.77).

Every number is sourced from a released artifact and labelled with its split.

---

## Data note

GR-Neutro is an **in-house Gustave Roussy** peripheral-blood neutrophil corpus and
is **not** redistributed here. The code expects an `annotations.csv` (header
`filename,path,<class1>,...,<class7>`) and an image root. Public datasets used
elsewhere in the paper (AML Matek 2019, MLL-23, Bodzas 2023, Acevedo 2019) are
cited in the paper; DinoBloom-B's SSL pretraining did **not** include GR-Neutro.
