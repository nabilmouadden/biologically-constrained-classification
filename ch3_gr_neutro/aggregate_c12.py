"""Aggregate C12 (patient x cohort joint auxiliary) multi-seed results.

Reads outputs/multi_cohort_patient/c12_patient_cohort_s{seed}/summary.json
and writes a single results.json with multi-seed mean +/- std rows for the
LaTeX block in papers/ai_conference/outputs/c12_patient_cohort/c12_block.tex.

Also computes the concept-rank decompression signal:
  - n_concepts_above_0p05 (rank threshold)
  - gap_sigma_k_kplus1 for k = 6, 7, 8 (where the C1 wall sat at k=7)
  - normalised singular-value spectrum.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/multi_cohort_patient")
    ap.add_argument("--tag", default="c12_patient_cohort")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 1337])
    ap.add_argument("--out_json", default="outputs/multi_cohort_patient/results.json")
    args = ap.parse_args()

    root = Path(args.root)
    rows = {}
    missing = []
    for seed in args.seeds:
        path = root / f"{args.tag}_s{seed}" / "summary.json"
        if not path.exists():
            missing.append(str(path))
            continue
        rows[seed] = json.loads(path.read_text())

    if not rows:
        print(f"[warn] no seed summaries found under {root} for tag={args.tag}",
              file=sys.stderr)
        print(f"[warn] missing: {missing}", file=sys.stderr)
        sys.exit(1)

    def agg(key, transform=lambda x: x):
        vals = [transform(rows[s].get(key)) for s in rows
                if rows[s].get(key) is not None]
        if not vals: return None
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                    n=len(vals), values=[float(v) for v in vals])

    out = dict(
        tag=args.tag, seeds=list(rows.keys()),
        per_seed={int(s): rows[s] for s in rows},
        gr_test_weighted_f1=agg("gr_test_weighted_f1"),
        gr_test_macro_f1=agg("gr_test_macro_f1"),
        mll_normal_recall=agg("mll_normal_recall"),
        n_concepts_above_0p05=agg("n_concepts_above_0p05"),
        gap_sigma7_sigma8=agg("gap_sigma7_sigma8"),
        gap_sigma8_sigma9=agg("gap_sigma8_sigma9"),
        biomedclip_mean_abs_rho=agg("biomedclip_mean_abs_rho"),
    )

    # Average normalised SV spectrum across seeds (length-11 vector).
    svs = [rows[s].get("concept_singular_values_normalized") for s in rows]
    svs = [v for v in svs if v]
    if svs:
        # pad to common length
        K = min(len(v) for v in svs)
        arr = np.array([v[:K] for v in svs])
        out["concept_singular_values_normalized_mean"] = arr.mean(axis=0).tolist()
        out["concept_singular_values_normalized_std"] = arr.std(axis=0).tolist()

    op = Path(args.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"[write] {op}")
    print(f"[seeds] {list(rows.keys())} ({len(rows)} of {len(args.seeds)} found)")
    if out["gr_test_weighted_f1"]:
        print(f"[summary] GR W-F1 = {out['gr_test_weighted_f1']['mean']:.4f} "
              f"+/- {out['gr_test_weighted_f1']['std']:.4f} (n={out['gr_test_weighted_f1']['n']})")
    if out["mll_normal_recall"]:
        print(f"[summary] MLL Normal recall = {out['mll_normal_recall']['mean']:.4f} "
              f"+/- {out['mll_normal_recall']['std']:.4f}")
    if out["n_concepts_above_0p05"]:
        print(f"[summary] # concept SV >= 0.05 = {out['n_concepts_above_0p05']['mean']:.2f} "
              f"+/- {out['n_concepts_above_0p05']['std']:.2f}")


if __name__ == "__main__":
    main()
