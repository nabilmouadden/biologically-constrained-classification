# ch3_gr_neutro — concept-explanation adapter on GR-Neutro (multi-label)

The morphological-concept adapter from the ch3 / AML Matek work, ported to the
**GR-Neutro** multi-label setting (7 abnormality classes, 10 morphological
concepts derived from `Morphological_Concepts.docx`).

If you want to add a concept-explanation module to a GR-Neutro-like project,
**start from this code, not from `thesis/ch3/`** — the AML Matek version assumes
single-label (CrossEntropy) classification with cached frozen features. This
version was rebuilt for:

- **Multi-label** classification (`BCEWithLogitsLoss` everywhere, not `CrossEntropy`)
- **Live forward** through a partially fine-tuned DinoBloom backbone (no cached features)
- **Hard-binarized concept targets** at training time (soft 0.5 baselines give zero gradient)
- **Soft cooccur loss** `-<C, mean p p^T>_F` (the docx has no hard `-1` mutex pairs)
- **Multi-label concept-target aggregation** via max-deviation-from-Normal

## Files

| file | purpose |
|------|---------|
| `models.py` | `DinoBloomBackbone`, `ConceptAdapter`, `ConstraintModule`, `JointModel`, `build_prior_C`, `aggregate_concept_target` |
| `data.py` | `read_annotations`, `stratified_multilabel_split`, `GRNeutroDataset`, train/eval transforms |
| `train.py` | full joint training loop (focal + EMA + posw + hard concept targets, all optional) |
| `evaluate.py` | conformal coverage, completeness probe, threshold tuning, cooccurrence vs prior |
| `make_figures_v2.py` | biologist-readable figures (attention overlay, per-class/concept bars, confusion + P/R/F1, examples) |
| `tta_eval.py` | 6× test-time augmentation |
| `ensemble_eval.py` | average sigmoid probs across multiple runs |
| `bootstrap_ci.py` | bootstrap 95% CI on headline metrics |
| `min_class_optimize.py` | per-class threshold tuning that maximizes the *minimum* class F1 |
| `summarize_runs.py` | aggregate `outputs/*/summary.json` + `eval.json` into one leaderboard CSV |
| `concept_config_gr_neutro.json` | 10-concept vocabulary + class→concept matrix + constraint matrix C |
| `EXPERIMENT_JOURNAL.md` | chronological audit (every run, every diagnosis) |
| `RESEARCH_MEMO.md` | final memo: results, what worked, what didn't, honest weaknesses |
| `run_train.sh`, `run_lambda_sweep.sh`, `run_seeds.sh`, `iterate.sh` | SLURM submission scripts |

## Quick start (Ruche)

```bash
# 1. Build the extended annotations.csv from the per-class folders
python -c "from data import read_annotations; print(read_annotations('annotations.csv', 'gr_neutro_extended/'))"

# 2. Train the headline config (focal + EMA + hard targets + λ=0.01)
sbatch --export=ALL,TAG=mytrain run_train.sh \
  --tag mytrain --epochs 60 --seed 42 --lambda_constraint 0.01 \
  --lambda_concept_loss 2.0 --concept_target_mode hard \
  --focal_gamma 2.0 --focal_alpha 0.25 --ema_decay 0.999 \
  --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 --lr_constraint 1e-3 \
  --unfreeze_last_n 6 --batch_size 32 --backbone dinobloom_s
# add --backbone dinobloom_b for the bigger backbone

# 3. Evaluate (conformal + probe + threshold tuning)
python evaluate.py --run_dir outputs/mytrain

# 4. Figures for biologist presentation
python make_figures_v2.py \
  --primary_tag mytrain \
  --leaderboard_tags mytrain,...
```

## Key result on the extended GR-Neutro dataset (4378 cells, 7 classes)

Best single model `B_kitchen_s42` (DinoBloom-B + kitchen-sink config, seed=42):
- Classification weighted F1 = **0.910**
- Mean concept F1 = **0.896**
- Coverage @α=0.05 = **0.976**
- 3 of 7 classes ≥0.90 (Normal, Hypersegmentation, Hypogranulation)
- 6 of 10 concepts ≥0.90

The remaining gap (Chromatin, Dohle, Hyposegmentation) is biologically meaningful
class confusion, not a model issue — see RESEARCH_MEMO.md §0 and the confusion
matrix in `figures_v2/`.

## Important divergences from `thesis/ch3/` (AML Matek)

1. **`bce_class = BCEWithLogitsLoss(pos_weight=...)`** instead of CrossEntropy. Every
   cell can have multiple positive abnormalities.
2. **`concept_target_mode = 'hard'`**: targets binarized at 0.5 before BCE. Soft-BCE
   with target=0.5 has zero gradient at p=0.5 → adapter never moves. **This single
   change roughly doubled mean concept F1** (see EXPERIMENT_JOURNAL.md §"Wave 1").
3. **`ConstraintModule.soft_cooccur_loss`**: docx has no `-1` mutex pairs; hard
   `violation_loss` from ch3 collapses to zero. Replaced with
   `-<C, mean p p^T>_F` (off-diagonal) which gives meaningful gradient on every
   C[i,j] entry, weighted by |C[i,j]|.
4. **`λ_constraint = 0.01`**, not 0.1. The ch3 default over-regularizes concepts.
   Pareto sweep in `figures_v2/per_concept_f1.pdf` shows the optimum.
5. **`aggregate_concept_target(class_concept, multilabel, normal_idx)`**: for a
   multi-label cell, the concept target picks the active class whose value is
   furthest from the Normal baseline per concept. Mean aggregation dilutes the
   defining abnormality signal.
6. **Partial fine-tuning** of `unfreeze_last_n` (default 6) DinoBloom blocks, *not*
   cached features. The GR-Neutro live training pipeline expects this.
