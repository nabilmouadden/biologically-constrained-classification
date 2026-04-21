# gr_neutro — starter kit

This directory seeds a follow-up on GR-Neutro that applies the concept-bottleneck
methodology from `/aml_matek/` to the 7-class multi-label neutrophil-abnormality task
already handled by the MIDL 2025 codebase in `/src/`.

Recommended reading before editing anything in this folder:

1. `docs/onboarding.md` — overview of the three layers (MIDL 2025,
   `aml_matek/`, `gr_neutro/`) and what each one contributes.
2. `docs/concept_design.md` — methodology for deriving concept vocabularies
   from clinical literature and building constraint matrices.
3. `docs/gr_neutro_notes.md` — the specific code snippet to port from
   `aml_matek/` and a protocol for what to measure on GR-Neutro.

## Contents

| File | Purpose |
|------|---------|
| `concept_config.json` | Two proposed concept vocabularies. **Option A** (classes-as-concepts): the 7 class labels are themselves the concepts; the violation penalty runs directly on the sigmoid outputs of the 7-way classifier. **Option B** (low-level decomposition): a draft vocabulary that breaks the 7 classes into shared morphological features; partially spec'd, marked TODO. |

## Proposed workflow

### Step 1 — reproduce the MIDL 2025 baseline
The codebase in `/src/` already trains on GR-Neutro. Reproducing its numbers
establishes the control line.

### Step 2 — add the violation-loss term
The MIDL 2025 `ConstraintLoss._compute_constraint_loss` only computes
`‖RRᵀ − C‖² + α‖R‖₁`, which does not backpropagate into the classifier (see
`docs/onboarding.md` for the empirical confirmation). The minimal addition is
a second loss term that does:

```python
def violation_loss(self, logits, exclusive_pairs):
    """Penalize σ(logit_a) · σ(logit_b) over mutually-exclusive (a, b) pairs."""
    p = torch.sigmoid(logits)                    # (B, K)
    ei, ej = exclusive_pairs[:, 0], exclusive_pairs[:, 1]
    return (p[:, ei] * p[:, ej]).sum(dim=-1).mean()
```

This should be added as an explicit loss component with its own weight. On
GR-Neutro the mutex pairs are Normal↔any-abnormality, hyper↔hypogranulation,
and hyper↔hyposegmentation. The reference implementation (computing `exclusive_pairs`
from a prior constraint matrix C and applying the penalty) is in
`aml_matek/models.py::ConstraintModule`.

### Step 3 — sweep and measure
At minimum, compare:

- the MIDL 2025 baseline (no violation term),
- the MIDL 2025 pipeline with the violation term at several λ values
  (e.g. `{0.01, 0.05, 0.1, 0.3, 1.0}`).

Report weighted F1, per-class F1, **mutex violation rate** (e.g. count of
predictions with both `hyper` and `hypo` above 0.5 on the same cell), and
conformal coverage at α=0.05.

### Step 4 — write up
A defensible headline for a short paper or thesis chapter:
*"On GR-Neutro — a multi-label neutrophil abnormality task where mutex
constraints are hard biological contradictions — adding a direct co-activation
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
