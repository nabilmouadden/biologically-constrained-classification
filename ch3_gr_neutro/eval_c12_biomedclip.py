"""Compute per-concept Spearman correlations between C12 concept probabilities
on GR-Neutro and BiomedCLIP grounding scores. Matches concept names columnwise
between the two cohorts.

Outputs JSON with per-seed and aggregate stats; the LaTeX block reads from this.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomedclip_npz",
        default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/biomedclip_scores.npz")
    ap.add_argument("--c12_predictions",
        default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/multi_cohort_patient/predictions_s{seed}.pt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 1337])
    ap.add_argument("--out_json",
        default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/multi_cohort_patient/biomedclip_rho.json")
    args = ap.parse_args()

    bd = np.load(args.biomedclip_npz, allow_pickle=True)
    bm_paths = [str(p) for p in bd["paths"].tolist()]
    bm_concepts = [str(c) for c in bd["concepts"].tolist()]
    bm_scores = bd["biomed_scores"]
    bm_basename = [Path(p).name for p in bm_paths]
    bm_index = {bn: i for i, bn in enumerate(bm_basename)}
    print(f"[bm] {len(bm_paths)} cells, concepts={bm_concepts}")
    print(f"[bm] shape={bm_scores.shape}")

    per_seed = {}
    for seed in args.seeds:
        path = Path(args.c12_predictions.format(seed=seed))
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        gr_paths = d["gr_paths"]
        concepts = d["concepts"]
        cp = d["full_concept_probs"][0]  # GR-Neutro

        # Align: keep cells in both BM and GR; columns by concept name.
        rows = []
        for i, p in enumerate(gr_paths):
            bn = Path(p).name
            if bn in bm_index:
                rows.append((i, bm_index[bn]))
        if not rows:
            print(f"[seed {seed}] zero alignment between BM and GR — skip")
            continue
        rows = np.array(rows)
        cp_align = cp[rows[:, 0]]            # (n, K)
        bm_align = bm_scores[rows[:, 1]]    # (n, M)

        # Column intersection.
        common = [c for c in concepts if c in bm_concepts]
        cp_cols = [concepts.index(c) for c in common]
        bm_cols = [bm_concepts.index(c) for c in common]

        rhos = []
        for k, c in zip(range(len(common)), common):
            r, _ = spearmanr(cp_align[:, cp_cols[k]], bm_align[:, bm_cols[k]])
            if not np.isnan(r):
                rhos.append(float(r))
        per_seed[seed] = dict(
            n_aligned=int(rows.shape[0]),
            common_concepts=common,
            rho_per_concept=rhos,
            mean_abs_rho=float(np.mean(np.abs(rhos))) if rhos else None,
        )
        print(f"[seed {seed}] n_aligned={rows.shape[0]} mean|rho|={per_seed[seed]['mean_abs_rho']:.4f}")
        for c, r in zip(common, rhos):
            print(f"   {c}: rho={r:.3f}")

    # Aggregate
    rhos_per_concept = {}
    for seed in per_seed:
        for c, r in zip(per_seed[seed]["common_concepts"], per_seed[seed]["rho_per_concept"]):
            rhos_per_concept.setdefault(c, []).append(r)
    agg_rho = {c: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
               for c, v in rhos_per_concept.items()}
    mean_abs_rhos = [per_seed[s]["mean_abs_rho"] for s in per_seed
                     if per_seed[s].get("mean_abs_rho") is not None]
    out = dict(
        per_seed=per_seed,
        per_concept_aggregate=agg_rho,
        mean_abs_rho_aggregate=dict(
            mean=float(np.mean(mean_abs_rhos)) if mean_abs_rhos else None,
            std=float(np.std(mean_abs_rhos)) if mean_abs_rhos else None,
            n=len(mean_abs_rhos),
            values=mean_abs_rhos,
        ),
        c1_smoke_mean_abs_rho=0.106,  # from multi_cohort_block.tex
    )
    op = Path(args.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\n[write] {op}")
    if mean_abs_rhos:
        print(f"[summary] mean|rho| across seeds = "
              f"{np.mean(mean_abs_rhos):.4f} +/- {np.std(mean_abs_rhos):.4f}")
        print(f"[c1 baseline] 0.106")


if __name__ == "__main__":
    main()
