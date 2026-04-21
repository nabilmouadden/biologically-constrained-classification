# gr_neutro — starter kit

This directory seeds a follow-up on GR-Neutro that applies the concept-bottleneck
methodology from `/aml_matek/` to the 7-class multi-label neutrophil-abnormality task
already handled by the MIDL 2025 codebase in `/src/`.

Recommended reading before editing anything in this folder:

1. `docs/onboarding.md` — overview of the three layers (MIDL 2025,
   `aml_matek/`, `gr_neutro/`) and what each one contributes.
2. `docs/concept_design.md` — methodology for deriving concept vocabularies
   from clinical literature and building constraint matrices.
3. `docs/gr_neutro_notes.md` — pipeline configuration and measurement
   protocol specific to GR-Neutro.

## Contents

| File | Purpose |
|------|---------|
| `concept_config.json` | Two proposed concept vocabularies. **Option A** (classes-as-concepts): the 7 class labels are themselves the concepts; the violation penalty runs directly on the sigmoid outputs of the 7-way classifier. **Option B** (low-level decomposition): a draft vocabulary that breaks the 7 classes into shared morphological features; partially spec'd, marked TODO. |

## Proposed workflow

### Step 1 — run the default pipeline
`configs/gr_neutro.yaml` is already wired to train with all five loss
components — BCE, constraint-matching, mutex violation penalty, uncertainty,
and entropy regularization. A first run establishes the control line:

```bash
python examples/train.py --config configs/gr_neutro.yaml --output_dir ./checkpoints
```

### Step 2 — sweep `lambda_viol` and measure
Fix the other hyperparameters and vary `training.loss.lambda_viol ∈
{0, 0.01, 0.05, 0.1, 0.3, 1.0}`. Report, per run:

- weighted F1 and per-class F1,
- **mutex violation rate** per pair (count of samples where both
  `σ(concept_a)` and `σ(concept_b)` exceed 0.5 on a mutex pair),
- conformal coverage at α = 0.05.

The Pareto curve of mutex-violation-rate vs. weighted-F1 is the load-bearing
figure for a write-up.

### Step 3 — write up
A defensible headline:
*"On GR-Neutro — a multi-label neutrophil abnormality task where mutex
constraints are hard biological contradictions — the direct co-activation
penalty drops mutex-violation rate substantially without loss of weighted F1,
producing biologically coherent multi-label predictions."*

## Why GR-Neutro is the appropriate test for this methodology

AML Matek turned out to be a poor sandbox for constraint-based methods
because:

- the task is **single-label 15-way softmax** — the classifier emits one
  class at a time, and mutex violations at the class-output level are
  satisfied by construction;
- the classes are **visually well separated** (myeloblast vs lymphocyte vs
  eosinophil) — even the unconstrained baseline violates mutex pairs less
  than 0.5% of the time, leaving almost no headroom for a regularizer.

GR-Neutro inverts both properties:

- **multi-label with 7 independent sigmoids** — hyper and hypogranulation
  can both fire on the same cell because they do not compete for a softmax
  slot;
- **tight visual boundaries** — a moderately abnormal neutrophil sits at the
  crossover between too-many and too-few granules, and a non-constrained
  model has incentive to hedge by firing both labels;
- **hard biological contradictions** — hyper/hypo grade the same physical
  system in opposite directions. A human grader would never assign both;
  an unconstrained model plausibly does.

This is the regime where a direct violation penalty is most likely to produce
a large, clean effect on per-pair violation rate without meaningful accuracy
cost.
