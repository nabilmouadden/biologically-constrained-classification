# Synthetic example fixtures

These files (`sample_annotations.csv`, `cell_manifest_full_extended.json`,
`morphometry_concepts.csv`) are **synthetic** fixtures with fake filenames and
made-up concept values. They contain **no real patient data**. Their only purpose
is to smoke-test the pipeline wiring offline — that `make_manifest.py` produces a
valid manifest and that `residual_cbm.py`'s `load_aligned` / `load_morpho` parse
the manifest and morphometry schemas correctly. They are **not** a dataset and
must not be used to compute or report any result.
