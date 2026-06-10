# Preprocessing

Run order to go from raw `annotations.csv` + images to inputs `residual_cbm.py` consumes.

| Step | Script | Inputs | Output |
|---|---|---|---|
| 1 | `code/morphometry_concepts_v2.py` | data dir holding `annotations.csv` (+ image `path`s) | `morphometry_concepts.csv` (10 concepts + `seg_ok`) |
| 2 | `code/make_manifest.py` | `annotations.csv` | `cell_manifest_full_extended.json` |
| 3 | `code/cache_ft_features.py` | `annotations.csv`, image root | feature bank `*.npz` (or download the released bank) |
| 4 | `code/residual_cbm.py` | feature bank + manifest + morphometry CSV | `runs/residual_cbm/results.json` |

## 1. Deterministic morphometry concepts

`--data-dir` must contain `annotations.csv` (whose `path` column points at each cell image).

```bash
python code/morphometry_concepts_v2.py \
    --data-dir ./data/gr_neutro \
    --out-dir  ./runs/morphometry
# -> ./runs/morphometry/morphometry_concepts.csv
```

## 2. Cell manifest

```bash
python code/make_manifest.py \
    --annotations ./data/gr_neutro/annotations.csv \
    --out         ./runs/cell_manifest_full_extended.json
```

## 3. Feature bank

Regenerate the fine-tuned bank, or download the released bank (see `weights/README.md`).

```bash
python code/cache_ft_features.py \
    --variant dinobloom_b --unfreeze_last_n 4 --epochs 30 --seed 0 \
    --annotations ./data/gr_neutro/annotations.csv \
    --data_root   ./data/gr_neutro \
    --out         ./runs/dinobloom_b_ft_last4_features.npz
```

## 4. Residual / concept heads

```bash
python code/residual_cbm.py \
    --features ./weights/dinobloom_b_frozen_features.npz \
    --manifest ./runs/cell_manifest_full_extended.json \
    --morpho   ./runs/morphometry/morphometry_concepts.csv \
    --out      ./runs/residual_cbm/results.json
```

## Wiring smoke test

The synthetic fixtures in `examples/` let you check the manifest/morphometry wiring
offline (no real data, no GPU):

```bash
python code/make_manifest.py \
    --annotations examples/sample_annotations.csv \
    --out /tmp/_manifest.json
```
