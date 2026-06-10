# GR-Neutro concept-bottleneck reproducibility release

Runnable code, released model weights, and per-abnormality accuracy for a
DinoBloom-B backbone classifier (frozen + fine-tuned) and the concept
architectures — Joint CBM, Pure-bottleneck CBM, Sequential CBM, Independent CBM,
CEM (Concept Embedding Model), and PCBM-h (post-hoc / residual CBM) — on GR-Neutro
(in-house Gustave Roussy peripheral-blood neutrophil corpus; 4,378 cells, 7
abnormality classes, 10 downstream textbook morphology concepts).

Weights are hosted on the Hugging Face Hub: **https://huggingface.co/nabimu9/gr-neutro-cbm-weights**

```
<repo root>/
├── code/
│   ├── models.py                    # DinoBloomBackbone, ConceptAdapter, ConstraintModule, JointModel
│   ├── residual_cbm.py              # PCBM-h + CEM + pure-bottleneck heads on cached features
│   ├── train.py                     # end-to-end training (backbone + concept arch)
│   ├── data.py                      # GR-Neutro loading + stratified-multilabel split + transforms
│   ├── finetune_7class.py           # single-label 7-class fine-tune of DinoBloom-B
│   ├── cache_ft_features.py         # fine-tune DinoBloom-B (last-N) and dump CLS feature bank
│   ├── make_manifest.py             # annotations.csv -> cell_manifest_full_extended.json
│   ├── morphometry_concepts_v2.py   # deterministic label-free morphometry (10 concepts)
│   ├── infer.py                     # single-image inference -> class + concepts
│   ├── concept_config_gr_neutro.json
│   └── concepts_10.json
├── examples/                        # synthetic fixtures for an offline wiring smoke test
├── weights/                         # README + download script (weights pulled from HF)
├── results/                         # per-abnormality accuracy tables
├── PREPROCESSING.md                 # preprocessing run order
└── README.md
```

---

## Released weights and their accuracy

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

### Head checkpoints (`heads/` on HF) — seed 42 W-F1, frozen / fine-tuned

Standalone concept-head checkpoints hosted under `heads/` on the HF repo. Files are
named `heads/<method>_<frozen|ft_last4>_s42_head.pt`.

| Method | Frozen W-F1 | Fine-tuned (last-4) W-F1 |
|---|--:|--:|
| CEM | 0.844 | 0.936 |
| `backbone_mlp` | 0.832 | 0.935 |
| PCBM-h | 0.823 | 0.928 |
| `pure_bottleneck` | 0.781 | 0.890 |

### Feature banks

| File (`*.npz`) | What | Shape |
|---|---|---|
| `dinobloom_b_frozen_features.npz` | Frozen DinoBloom-B CLS features, all 4,378 cells | `features` (4378, 768) |
| `dinobloom_b_ft_last4_features.npz` | Fine-tuned (last-4) DinoBloom-B CLS bank, seed 0 | `features` (4378, 768) |

---

## Dependencies

- Python 3.10+
- PyTorch + torchvision (CUDA for training; CPU works for single-image inference)
- timm, numpy, scikit-learn, pandas, scipy, scikit-image, Pillow, huggingface_hub, matplotlib

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only inference (no CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The frozen DinoBloom-B backbone is the public checkpoint
`hf-hub:1aurent/vit_base_patch14_224.dinobloom`, fetched automatically by `timm`
on first build. To warm the HF cache on a networked node before an offline run:

```bash
python -c "import timm; timm.create_model('hf-hub:1aurent/vit_base_patch14_224.dinobloom', pretrained=True, img_size=224)"
```

---

## Getting the weights

```bash
bash weights/download_weights.sh                      # pulls everything into ./weights/
# or:
hf download nabimu9/gr-neutro-cbm-weights --local-dir weights
# or in Python:
python -c "from huggingface_hub import snapshot_download; snapshot_download('nabimu9/gr-neutro-cbm-weights', local_dir='weights')"
```

See `weights/README.md` for what each file is and how to load it.

---

## Preprocessing

See `PREPROCESSING.md` for the full run order (morphometry → manifest → feature bank →
`residual_cbm.py`). The synthetic fixtures in `examples/` provide an offline wiring
smoke test.

---

## Training

Feature-bank concept heads (CEM, PCBM-h, pure-bottleneck):

```bash
python code/residual_cbm.py \
    --features weights/dinobloom_b_ft_last4_features.npz \
    --manifest ./runs/cell_manifest_full_extended.json \
    --morpho   ./runs/morphometry/morphometry_concepts.csv \
    --out      runs/residual_cbm/results.json
```

End-to-end backbone + concept architecture:

```bash
# Joint CBM (separate CLS classifier + concept adapter):
python code/train.py --tag joint_run --mode joint \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --data_csv ./data/gr_neutro/annotations.csv --data_root ./data/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 42

# Pure-bottleneck CBM (class flows only through concepts):
python code/train.py --tag cbm_run --mode cbm --lambda_concept_loss 2.0 \
    --backbone dinobloom_b --unfreeze_last_n 6 \
    --data_csv ./data/gr_neutro/annotations.csv --data_root ./data/gr_neutro \
    --config code/concept_config_gr_neutro.json --seed 42

# No-concept backbone baseline: add --baseline.
```

Each run writes `outputs/<tag>/model.pt` (same format as the released checkpoints) +
`predictions.pt` + `summary.json`.

---

## Inference

```bash
bash weights/download_weights.sh           # if not done yet
cd code
python infer.py \
    --weights ../weights/joint_cbm_dinobloomB_ft_s42.pt \
    --image   /path/to/one_cell.png
```

`infer.py` loads any released `*.pt`, predicts the abnormality class, prints the
learned concept activations, and independently runs the deterministic morphometry to
print the 10 measured textbook-concept values. The architecture is read from each
checkpoint's stored `args`, so the same command works for all of them.

---

## Data note

GR-Neutro is an in-house Gustave Roussy peripheral-blood neutrophil corpus and is not
redistributed here. The code expects an `annotations.csv` (header
`filename,path,Normal,Chromatin,Dohle,Hypergranulation,Hypersegmentation,Hypogranulation,Hyposegmentation`,
one-hot) and an image root. DinoBloom-B's SSL pretraining did not include GR-Neutro.
