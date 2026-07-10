# Released weights

All `.pt` and `.npz` files in this directory are stored via **Git LFS**. After
cloning, run `git lfs pull` to materialise them (see `release/README.md`).

| File | Size | Type | Architecture | How to load |
|------|-----:|------|--------------|-------------|
| `B_kitchen_s2024_model.pt`        | 328 MB | full model | **Joint CBM** — DinoBloom-B backbone (last-6 fine-tuned) + separate CLS classifier + concept adapter (mode=joint). seed 2024. | `infer.py --weights B_kitchen_s2024_model.pt` |
| `cbm_joint_s0_model.pt`           | 328 MB | full model | **Pure-bottleneck CBM** — class predicted ONLY through the concept layer (mode=cbm), λ_concept=2.0. seed 0. | `infer.py --weights cbm_joint_s0_model.pt` |
| `cbm_lcon4p0_s0_model.pt`         | 328 MB | full model | **Pure-bottleneck CBM**, stronger concept loss λ_concept=4.0 (mode=cbm). seed 0. | `infer.py --weights cbm_lcon4p0_s0_model.pt` |
| `dinobloomb_ft_last4_s0_features.npz` | 14 MB | feature bank | CLS features (N=4378, 768-d) from the **fine-tuned DinoBloom-B (last-4)** backbone, seed 0. Keys: `features` (N,768) float32, `paths` (N,) bare `<Class>/<file>.jpg`, `labels` (N,) int64, `class_names` (7,), `concepts` (10,). | `np.load(...)['features']`, feed to `residual_cbm.py` |

> Naming note: despite the `cbm_joint_` prefix, `cbm_joint_s0_model.pt` is a
> **pure-bottleneck CBM** (`mode=cbm`). The genuine *joint* architecture (a
> separate CLS classifier alongside the concept adapter) is
> `B_kitchen_s2024_model.pt`. `infer.py` reads the architecture from each
> checkpoint's stored `args["mode"]` and rebuilds the correct model
> automatically, so you do not need to pass the mode by hand.

## What each checkpoint contains

`*.pt` files are `torch.save` dicts with keys:

- `state_dict` — model weights (backbone + concept adapter + constraint module
  + the class head: a `classifier` for joint, a `class_from_concepts` for CBM).
- `args` — the full training argparse namespace (`backbone`, `mode`,
  `unfreeze_last_n`, `concept_dim`, `num_heads`, λ's, seed, tag, …).
- `class_names` — the 7 GR-Neutro classes, in order.
- `concepts` — the concept names the adapter predicts. These checkpoints predate
  the concept-vocabulary reduction to the current **10 canonical** concepts, so
  the stored head is one channel wider; the trailing entry is a neutral
  `legacy_channel_10` placeholder. The loader builds the adapter at the stored
  width automatically, but `infer.py` **slices the concept output back to the
  canonical 10** at read-out: the trailing legacy channel is intentionally not
  exposed by any released file and is not part of the reported concept set.
- `prior_C` — the symmetric concept-constraint matrix used by the constraint
  module.
- `normal_idx` — index of the Normal class (0).

## Accuracy of each checkpoint's configuration

(See `release/results/accuracy_by_abnormality.md` for full per-class tables and
artifact citations.)

| Checkpoint | Configuration | Overall weighted-F1 |
|------------|---------------|--------------------:|
| `B_kitchen_s2024_model.pt` | Joint CBM (DinoBloom-B fine-tuned) | 0.890 (representative seed s42, `r1v14_kitchen_s42`); 5-seed aggregate 0.88 |
| `cbm_joint_s0_model.pt`    | Pure-bottleneck CBM (fine-tuned) | ≈ 0.883 (pure-bottleneck, fine-tuned, 6-seed mean) |
| `cbm_lcon4p0_s0_model.pt`  | Pure-bottleneck CBM, λ_concept=4.0 (fine-tuned) | ≈ 0.883 (pure-bottleneck family) |

The pure-bottleneck overall figure is the 6-seed `pure_bottleneck` (fine-tuned
last-4) W-F1 from `outputs/residual_cbm/results.md`; the two `cbm_*` checkpoints
are single-seed (seed 0) members of that family and are flagged single-seed.

## Loading a full model in code

```python
import torch
from models import DinoBloomBackbone, JointModel

ck = torch.load("cbm_joint_s0_model.pt", map_location="cpu", weights_only=False)
backbone = DinoBloomBackbone(variant=ck["args"]["backbone"],
                             unfreeze_last_n=ck["args"]["unfreeze_last_n"])
model = JointModel(backbone, num_concepts=len(ck["concepts"]),
                   num_classes=len(ck["class_names"]), prior_C=ck["prior_C"],
                   mode=ck["args"].get("mode") or "joint")
model.load_state_dict(ck["state_dict"], strict=False)
model.eval()
```

`infer.py` does exactly this and adds image preprocessing + morphometry.
