# Released weights (Hugging Face Hub)

All weights are hosted at **https://huggingface.co/nabimu9/gr-neutro-cbm-weights**
(too large for GitHub). Pull them into this directory with:

```bash
bash download_weights.sh          # from the weights/ dir
# or:
hf download nabimu9/gr-neutro-cbm-weights --local-dir .
```

## End-to-end checkpoints (full models, DinoBloom-B last-6 fine-tuned, seed 42)

| File | Architecture | Test W-F1 | How to load |
|------|--------------|----------:|-------------|
| `joint_cbm_dinobloomB_ft_s42.pt` | **Joint CBM** — separate CLS classifier + concept adapter + constraint (mode=joint). | 0.912 | `infer.py --weights joint_cbm_dinobloomB_ft_s42.pt` |
| `pure_bottleneck_cbm_dinobloomB_ft_s42.pt` | **Pure-bottleneck CBM** — class predicted ONLY through the concept layer (mode=cbm), λ_concept=2. | 0.909 | `infer.py --weights pure_bottleneck_cbm_dinobloomB_ft_s42.pt` |
| `backbone_baseline_dinobloomB_ft_s42.pt` | **No-concept backbone baseline** — fine-tuned DinoBloom-B classifier (baseline head). | 0.907 | `infer.py --weights backbone_baseline_dinobloomB_ft_s42.pt` |
| `cbm_sequential_dinobloomB_ft_s42.pt` | **Sequential CBM** — concepts trained first then frozen → class. | 0.873 | `infer.py --weights cbm_sequential_dinobloomB_ft_s42.pt` |
| `cbm_independent_dinobloomB_ft_s42.pt` | **Independent CBM** — concept and class heads trained independently. | 0.772 | `infer.py --weights cbm_independent_dinobloomB_ft_s42.pt` |

## Feature banks (for CEM / PCBM-h / frozen baseline via `residual_cbm.py`)

| File | What | Size |
|------|------|-----:|
| `dinobloom_b_frozen_features.npz` | Frozen DinoBloom-B CLS features (4378, 768) | 15 MB |
| `dinobloom_b_ft_last4_features.npz` | Fine-tuned (last-4) CLS bank, seed 0 (4378, 768) | 14 MB |

## What each `.pt` contains

`torch.save` dict with:

- `state_dict` — backbone + concept adapter + constraint module + class head
  (`classifier` for joint, `class_from_concepts` for the CBM modes).
- `args` — full training argparse namespace (`backbone`, `mode`, `unfreeze_last_n`,
  `concept_dim`, `num_heads`, λ's, seed, tag, …).
- `class_names` — the 7 GR-Neutro classes, in order.
- `concepts` — concept names the adapter predicts (these checkpoints predate the
  vacuolization drop, so they carry **11** names; the loader builds the adapter at
  that width automatically and the 10 downstream concepts are in `concepts_10.json`).
- `prior_C` — symmetric concept-constraint matrix used by the constraint module.
- `normal_idx` — index of the Normal class (0).

`infer.py` reads `args["mode"]` and rebuilds the correct architecture
automatically — you do not pass the mode by hand.

### Loading in code

```python
import torch
from models import DinoBloomBackbone, JointModel

ck = torch.load("joint_cbm_dinobloomB_ft_s42.pt", map_location="cpu", weights_only=False)
backbone = DinoBloomBackbone(variant=ck["args"]["backbone"],
                             unfreeze_last_n=ck["args"]["unfreeze_last_n"])
model = JointModel(backbone, num_concepts=len(ck["concepts"]),
                   num_classes=len(ck["class_names"]), prior_C=ck["prior_C"],
                   mode=ck["args"].get("mode") or "joint")
model.load_state_dict(ck["state_dict"], strict=False)
model.eval()
```

Full per-abnormality accuracy for every weight is in `../results/accuracy_by_abnormality.md`.
