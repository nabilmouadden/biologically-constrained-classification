# Designing a concept vocabulary and a constraint matrix that works

This document is the methodology guide. It is organized as a checklist that
can be followed top-to-bottom when adapting the framework to a new dataset.
The running examples are AML Matek (complete — `aml_matek/`) and GR-Neutro
(follow-up seed — `gr_neutro/`).

## 0. What counts as a concept

A **concept** is a binary or soft-valued morphological feature that a trained
hematologist would call out when describing a single cell under the
microscope. Examples: *multilobed nucleus*, *fine dispersed chromatin*,
*azurophilic granules*, *high nuclear-to-cytoplasmic ratio*.

A usable concept satisfies three properties:

1. **Clinical provenance** — it comes from a standard textbook or grading
   protocol, not from intuition. AML Matek concepts trace to Hoffbrand,
   *Essential Haematology* ch. 1, 8–10 and Briggs, *Haematology in Practice*
   ch. 2–4. GR-Neutro concepts should trace to standard neutrophil-
   abnormality grading (Naegli / Briggs ch. 3).

2. **Localizable** — a concept should plausibly correspond to something
   visible in a patch of the image. A concept that cannot be localized
   will not produce meaningful cross-attention visualizations, and the
   interpretability claim weakens.

3. **Either clearly labelable from class labels, or clearly not** — a
   concept whose target per class has to be guessed is a concept the model
   cannot reliably learn and an outside reader cannot reliably validate.

## 1. Seeding the vocabulary from biology

The starting point is a textbook chapter. Writing one descriptive sentence
per cell type (e.g. "Myeloblast: round nucleus, fine dispersed chromatin,
2–5 visible nucleoli, high N:C ratio, basophilic cytoplasm, usually
agranular") yields a candidate list where every bolded feature is a
potential concept. A sweep across all classes typically produces 15–25
candidates.

Working backwards from classifier outputs to "what features would help the
model classify X" should be avoided. That direction produces concepts that
are essentially class indicators in disguise.

## 2. Pruning to a tight set

A candidate concept is kept only when:

- it appears in at least 3 classes (otherwise it is a class indicator
  re-labeled);
- it has an agreed clinical definition (not "looks unusual");
- its target value per class has an obvious answer for ≥80% of classes.

AML Matek ended with 18 concepts from a candidate list of about 30. Concepts
that were dropped were mostly class indicators — e.g. *Auer rods* (only
myeloblasts carry them, so the concept carries no information beyond the
class label).

## 3. Building the class-to-concept soft matrix

Each `(class, concept)` pair is assigned a value in `[0, 1]`:

- `1.0` if the concept is definitionally present in every cell of that class,
- `0.0` if definitionally absent,
- a **soft value** if the class is heterogeneous (e.g. transitional
  maturation stages); the value is the estimated fraction of cells in the
  class that exhibit the concept.

Soft values are the appropriate encoding for classes that span a biological
continuum. Myeloblasts in AML Matek are assigned
`agranular_cytoplasm=0.7, azurophilic_granules=0.3` because the class
contains both Type I (no granules) and Type II (a few granules) blasts; the
70/30 split is drawn from the clinical literature.

Soft values shape training *targets*. They do not override mutex constraints
at prediction time: a single cell cannot have both fine and condensed
chromatin, even if class X has target
`fine_chromatin=0.3, condensed_chromatin=0.7`. The mutex constraint operates
on individual predictions; the soft target operates on the class distribution.

## 4. Building the constraint matrix

Two kinds of entries:

- **Mutual exclusions** (`value = −1`). These encode physical impossibilities
  or definitional oppositions. A nucleus cannot be both round and multilobed.
  Fine and condensed chromatin are endpoints of a single axis; a cell is one
  or the other. This is the entry type that contributes most to evaluation,
  since the violation-rate metric is computed over these pairs.

- **Positive co-occurrences** (`value ∈ {0.3, 1.0}`). `1.0` is reserved for
  physical necessities — e.g. fine chromatin ⇒ visible nucleoli, because
  nucleoli are only visible when chromatin is dispersed; condensed chromatin
  obscures them. `0.3` encodes statistical tendencies — e.g. condensed
  chromatin tends to co-occur with multilobed nucleus in granulopoiesis but
  not in lymphocytes.

Over-populating this matrix is harmful. Every constraint is a hypothesis the
model has to satisfy; an incorrect constraint will fight the data. Absent
evidence, `0` is the correct entry.

## 5. Choosing between classes-as-concepts and decomposed concepts

**Single-label tasks with broad class separation** (AML Matek): class labels
already do most of the work. A decomposed-concept CBM (the route taken in
`aml_matek/`) provides interpretability but does not noticeably change the
classifier.
Constraint matrices on *class* outputs are nearly vacuous because the
softmax already satisfies mutex pairs.

**Multi-label tasks** (GR-Neutro): classes can themselves serve as concepts.
Each label is an independent morphological claim, and the constraint matrix
on class outputs carries real signal because independent sigmoids can fire
simultaneously. This is the low-effort, high-signal path and is the
recommended starting point for GR-Neutro.

**Multi-label tasks where classes share underlying features**: full
decomposition into low-level concepts can pay off, because each concept
contributes to multiple classes and statistical strength is shared. The
gain materializes only when the features genuinely transfer.

## 6. The constraint loss — what actually works

The constraint loss in the MIDL 2025 implementation is

```
L_con = ‖R Rᵀ − C‖_F² + α ‖R‖_1
```

This updates `R` only; it does not flow gradient into the adapter or the
classifier. The AML Matek experiments confirmed this empirically: before a
second loss term was added, constrained and unconstrained runs produced
bit-identical concept predictions.

The minimum correction is a **direct co-activation penalty**:

```
L_viol = (1/B) Σ_{batch} Σ_{(a, b) ∈ mutex} σ(ẑ_a) · σ(ẑ_b)
```

This term does propagate gradient through `σ'` and into the classifier /
adapter. Using both terms is recommended: the R-matching term regularizes
the relationship matrix independently of the predictor; the violation term
shapes predictions. Reference implementation:
`aml_matek/models.py::ConstraintModule.violation_loss`.

## 7. Validation, not just measurement

A low overall violation rate is not sufficient evidence that a constraint is
working. Four diagnostics are recommended before declaring success:

1. **Per-pair breakdown.** An aggregate violation rate of 0.001 can conceal
   `fine_chromatin × condensed_chromatin` at 0.009 if many other pairs are
   already near zero. The per-pair bar chart is more informative than the
   mean.
2. **Class-conditional failure.** In the AML Matek experiments, global
   concept F1 was 0.85 while `band_nucleus` F1 was 0 — the 65 positive
   training samples were swamped by 5,400 samples for the visually similar
   `multilobed_nucleus`. Per-concept F1 conditional on the rare-positive
   slice catches this; `aml_matek/diag_band.py` is the template.
3. **Attention overlays.** A concept can score high F1 while attending to
   the wrong part of the cell (shortcut learning). Across a sample of cells,
   `nuclear_*` concept attention should overlap the nucleus,
   `granules_*` the cytoplasm, and so on. Misaligned attention undermines
   the interpretability claim.
4. **Rare concepts.** Concepts supported by fewer than ~100 training samples
   tend to collapse under shared BCE. Per-class positive-class weighting
   (`pos_weight = #neg / #pos`), accessible in `aml_matek/train.py` via
   `--concept_pos_weight`, typically recovers them at a small cost to the
   common ones.

## 8. What a clean concept-validation report contains

- Concept list with clinical citations (textbook chapter and page).
- Class-to-concept soft matrix with a one-sentence justification per row.
- Constraint matrix with a one-sentence reason per non-zero entry.
- Per-concept F1 bar chart with supports labeled.
- Per-mutex-pair violation rate, constrained vs unconstrained.
- Attention maps for 2 cells per class, overlaid for each concept.
- Learned co-occurrence matrix of predicted concepts, side-by-side with
  the prior matrix.

The corresponding templates live in `aml_matek/make_figures.py`, and the
generated outputs are in `/figures/`.
