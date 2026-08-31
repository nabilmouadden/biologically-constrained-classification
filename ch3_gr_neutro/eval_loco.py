"""Aggregate LOCO multi-cohort results across 4 configs x 3 seeds.

Reads outputs/c1_loco/<tag>_held<u>_s<seed>/summary.json and writes a single
results.json with per-config mean/sd metrics. Also computes the per-config
held-out cohort Spearman rho against BiomedCLIP scores when GR-Neutro is the
held-out cohort (it is the only cohort with BiomedCLIP per-concept scores).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CFG_TO_HELDOUT = {"A": 0, "B": 1, "C": 2, "D": 3}
HELDOUT_NAME = {0: "GR-Neutro", 1: "AML-Matek", 2: "MLL-23", 3: "Bodzas"}

BIOMEDCLIP_SCORES = ("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/"
                      "outputs/vlm_grounding/scores/biomedclip_scores.npz")


def compute_biomedclip_rho(pred_pt: Path, gr_feat_npz: str):
    """For Config A (held_out=GR-Neutro), compute per-concept Spearman rho
    between held-out CBM concept probs and BiomedCLIP v1 scores on GR-Neutro.
    """
    try:
        import torch
        from scipy.stats import spearmanr
        d = torch.load(pred_pt, map_location="cpu", weights_only=False)
        cp = d["held_cp"]  # (N_gr, K)
        # Get GR-Neutro file basenames in original load order.
        feat = np.load(gr_feat_npz, allow_pickle=True)
        paths = feat["paths"]
        basenames = [Path(p).name for p in paths]
        bc = np.load(BIOMEDCLIP_SCORES, allow_pickle=True)
        bc_scores = bc["v1_scores"]  # (4378, K)
        bc_filenames = bc["filenames"]
        bc_idx = {fn: i for i, fn in enumerate(bc_filenames)}
        # Join.
        idx_pairs = [(i, bc_idx.get(bn, -1)) for i, bn in enumerate(basenames)]
        keep_i = np.array([i for i, j in idx_pairs if j >= 0], dtype=int)
        bc_rows = np.array([j for i, j in idx_pairs if j >= 0], dtype=int)
        if len(keep_i) == 0:
            return {"rho_per_concept": None, "mean_abs_rho": None}
        cp_j = cp[keep_i]
        bc_j = bc_scores[bc_rows]
        K = cp.shape[1]
        rhos = []
        for k in range(K):
            r, _ = spearmanr(cp_j[:, k], bc_j[:, k])
            rhos.append(float(abs(r)) if not np.isnan(r) else 0.0)
        return {"rho_per_concept": rhos,
                "mean_abs_rho": float(np.mean(rhos)),
                "n_joined": int(len(keep_i))}
    except Exception as e:
        print(f"[rho] err: {e}", file=sys.stderr)
        return {"rho_per_concept": None, "mean_abs_rho": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/c1_loco")
    ap.add_argument("--seeds", default="0,42,1337")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / "results.json"
    seeds = [int(s) for s in args.seeds.split(",")]

    per_config = {}
    for cfg_name, held_u in CFG_TO_HELDOUT.items():
        per_seed = {}
        for s in seeds:
            tag = f"cfgC1L{cfg_name}"  # canonical tag prefix
            d = root / f"{tag}_held{held_u}_s{s}" / "summary.json"
            if not d.exists():
                # Tolerate alternate tag names
                cand = list(root.glob(f"*_held{held_u}_s{s}/summary.json"))
                if cand: d = cand[0]
            if not d.exists():
                continue
            per_seed[s] = json.loads(d.read_text())
        per_config[cfg_name] = dict(
            held_out_cohort=HELDOUT_NAME[held_u],
            n_seeds=len(per_seed),
            seeds=per_seed,
        )

        # Compute mean+sd across seeds for key metrics.
        def collect(key_chain):
            vals = []
            for s, summ in per_seed.items():
                v = summ
                ok = True
                for k in key_chain:
                    if isinstance(v, dict) and k in v:
                        v = v[k]
                    else:
                        ok = False; break
                if ok and v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            return vals

        nr = collect(["held_normal_recall"])
        wf = collect(["pooled_wf1"])
        mf = collect(["pooled_mf1"])
        na = collect(["n_concepts_above_0p05"])
        gp = collect(["gap_sigma7_sigma8"])

        agg = dict()
        for name, vals in [
            ("held_normal_recall", nr),
            ("pooled_indist_wf1", wf),
            ("pooled_indist_mf1", mf),
            ("n_concepts_above_0p05", na),
            ("gap_sigma7_sigma8", gp),
        ]:
            if vals:
                agg[name] = dict(mean=float(np.mean(vals)),
                                 sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                 n=len(vals))
            else:
                agg[name] = None
        per_config[cfg_name]["aggregate"] = agg

        # If GR-Neutro is held-out (Config A), compute BiomedCLIP rho per concept.
        if held_u == 0:
            rho_per_seed = {}
            gr_npz = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/dinobloom_features.npz"
            for s in seeds:
                pred = root / f"predictions_cfgC1L{cfg_name}_held0_s{s}.pt"
                if not pred.exists():
                    cand = list(root.glob(f"predictions_*held0_s{s}.pt"))
                    if cand: pred = cand[0]
                if pred.exists():
                    rho_per_seed[s] = compute_biomedclip_rho(pred, gr_npz)
            mean_abs = [v["mean_abs_rho"] for v in rho_per_seed.values()
                        if v and v.get("mean_abs_rho") is not None]
            if mean_abs:
                agg["biomedclip_mean_abs_rho"] = dict(
                    mean=float(np.mean(mean_abs)),
                    sd=float(np.std(mean_abs, ddof=1)) if len(mean_abs) > 1 else 0.0,
                    n=len(mean_abs))
            per_config[cfg_name]["biomedclip_rho_per_seed"] = rho_per_seed

    # Published comparators on held-out MLL-23 for Config C.
    published = dict(
        mll23_pure_cbm_normal_recall=0.473,
        mll23_cda_normal_recall=0.550,
        comment="Published pure-CBM and CDA Normal-recall on MLL-23 (zero-shot transfer from GR-Neutro).",
    )

    out = dict(
        seeds=seeds,
        configs=per_config,
        published_comparators=published,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[save] {out_path}")
    # Print a compact summary.
    for cfg_name, blob in per_config.items():
        agg = blob.get("aggregate", {})
        nr = agg.get("held_normal_recall") or {}
        wf = agg.get("pooled_indist_wf1") or {}
        print(f"[{cfg_name}] held={blob['held_out_cohort']:<10s} "
              f"n_seeds={blob['n_seeds']}  "
              f"Normal-recall={nr.get('mean', float('nan')):.3f} ± {nr.get('sd', float('nan')):.3f}  "
              f"in-dist WF1={wf.get('mean', float('nan')):.3f} ± {wf.get('sd', float('nan')):.3f}")


if __name__ == "__main__":
    main()
