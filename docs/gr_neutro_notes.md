# GR-Neutro notes

## The dataset

GR-Neutro is a **multi-label** neutrophil-abnormality classification task.
Each cell image may have zero, one, or multiple of the following 7 labels:

| idx | label | morphology |
|-----|-------|------------|
| 0 | Normal | within-reference-range neutrophil (3–5 lobes, normal granulation, no inclusions) |
| 1 | Chromatin | abnormal chromatin pattern (coarsely clumped, pyknotic) |
| 2 | Dohle | Döhle body — a basophilic cytoplasmic inclusion, sky-blue under Giemsa |
| 3 | Hypergranulation | too many or too dark ("toxic") granules |
| 4 | Hypersegmentation | ≥5 lobes (megaloblastic change) |
| 5 | Hypogranulation | too few granules, washed-out cytoplasm |
| 6 | Hyposegmentation | ≤2 lobes, unsegmented, Pelger-Huët-like |

## Why GR-Neutro is an appropriate sandbox for constraint methods

AML Matek failed to show constraint value because:

- 15-way **softmax** — exactly one class fires per cell, so mutex pairs at
  the class output level are satisfied by construction,
- classes are **visually distinct** — myeloblast ≠ lymphocyte ≠ eosinophil,
  and the baseline classifier rarely violates any biological constraint in
  the first place.

GR-Neutro inverts both properties:

1. **Multi-label sigmoid outputs.** `σ(hypergranulation)` and
   `σ(hypogranulation)` are independent. Nothing in the loss function
   prevents them from both firing at 0.6 on the same cell — an assignment a
   hematologist would flag as nonsense.

2. **Tight visual boundaries.** The same granule system is graded in both
   directions (hyper vs hypo). The same segmentation axis is read in both
   directions (hyper vs hypo). Intermediate cells are visually ambiguous,
   and a non-constrained model has incentive to hedge by firing both.

Mutex constraints on GR-Neutro are **hard biological contradictions**, not
statistical tendencies. That is the regime where a direct co-activation
penalty is most likely to produce a large, clean effect.

## What to port from `aml_matek/`

A single loss term. In `src/models/losses.py`, `ConstraintLoss` currently
sums:

- `bce_loss` (over class outputs),
- `constraint_loss` = `‖RRᵀ − C‖² + α‖R‖₁`,
- `uncertainty_loss` and `entropy_loss` (unchanged).

The new fifth component is computed from the class logits and the hard-mutex
pairs of C:

```python
def _compute_violation_loss(self, logits, prior_C):
    """Direct co-activation penalty over mutually-exclusive class pairs."""
    # Find mutex pairs from the prior matrix (where C == -1)
    # Upper triangle only to avoid double-counting
    mutex_mask = (prior_C < -0.5).triu(diagonal=1)
    ai, aj = mutex_mask.nonzero(as_tuple=True)
    if ai.numel() == 0:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)                       # (B, K)
    return (p[:, ai] * p[:, aj]).sum(dim=-1).mean()
```

Added to the total:

```python
violation_loss = self._compute_violation_loss(logits, C)
total_loss = bce_loss + \
             self.lambda_con * constraint_loss + \
             self.lambda_viol * violation_loss + \
             self.lambda_unc * uncertainty_loss + \
             self.lambda_entropy * entropy_loss
```

A reasonable starting point is `lambda_viol = lambda_con = 0.1`, followed by
a sweep over `lambda_viol ∈ {0, 0.01, 0.05, 0.1, 0.3, 1.0}` analogous to the
λ sweep in `aml_matek/`. The resulting Pareto curve (violation-rate vs
weighted-F1) is one of the most load-bearing figures for the follow-up paper.

## What to measure

A single table of the following form is the intended headline slide:

| | baseline (MIDL code, no viol term) | + viol term, λ=0.1 | + viol term, λ=1.0 |
|---|---|---|---|
| Weighted F1 | — | — | — |
| Macro F1 | — | — | — |
| **Mutex violation rate (hyper ∧ hypo)** | — | — | — |
| **Mutex violation rate (hyperseg ∧ hyposeg)** | — | — | — |
| **Mutex violation rate (Normal ∧ any abnormality)** | — | — | — |
| Conformal coverage @ α=0.05 | — | — | — |
| Per-class F1 for each of 7 classes | — | — | — |

Expected outcome: weighted F1 stays within ≈1 pp of the baseline; the three
mutex rates each drop by at least a factor of 5. If that is not what the
numbers show, the result should be diagnosed before a draft is written.

## Practical considerations

- **Class balance.** GR-Neutro is imbalanced (Normal is the majority; Dohle
  is rare). Per-class `pos_weight = #neg_k / #pos_k` in BCE is a single-line
  change that tends to matter; it was what recovered `band_nucleus` in the
  AML Matek experiments.
- **Uncertainty thresholding.** The MIDL 2025 pipeline has MC-dropout
  uncertainty and an adaptive threshold. The violation-loss term does not
  replace these — they measure different things (per-pair coherence vs
  per-prediction confidence). Keeping them both is recommended.
- **Patient-level split.** If patient identifiers are available, splits
  should be by patient rather than by cell; cell-level splits leak subtle
  staining and imaging signatures.
- **Conformal coverage on multi-label sigmoids.** The implementation in
  `aml_matek/` computes per-concept conformal quantiles and ports directly
  because each GR-Neutro label is an independent binary decision
  (`aml_matek/evaluate.py::conformal_per_concept`).

## Follow-up directions

1. **Decomposed concepts** (Option B in `gr_neutro/concept_config.json`) —
   break the 7 classes into ≈13 low-level features. This only pays off if
   the low-level features genuinely transfer across classes.
2. **External validation.** Asking a hematologist to grade a sample of 50
   random test cells and comparing to model predictions is the experiment
   that addresses the reviewer question "do the concepts mean what you say
   they mean?"
3. **Per-patient calibration.** If enough patients are available, running
   conformal calibration per patient rather than globally can absorb drift
   across scanners and staining protocols.
