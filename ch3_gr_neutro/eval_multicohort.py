"""Post-training analysis for the multi-cohort iVAE-conditioned CBM.

Reads outputs/multi_cohort/predictions_s{seed}.pt for seeds {0, 42, 1337},
computes:

  1. GR-Neutro test W-F1, Macro-F1.
  2. MLL-23 Normal recall (cells with source = neutrophil_segmented).
  3. Concept-correlation matrix on GR-Neutro test:
     normalised singular values; how many clear 0.05; gap sigma_7 - sigma_8.
  4. Per-concept Spearman rho against BiomedCLIP v1 scores (with sign flips).

Writes outputs/multi_cohort/results.json with per-seed metrics + means.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent


def split_indices(N, seed, val_frac=0.10, test_frac=0.10):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = int(round(N * test_frac))
    n_val = int(round(N * val_frac))
    return perm[n_test+n_val:], perm[n_test:n_test+n_val], perm[:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,42,1337")
    ap.add_argument("--pred_dir", default=str(HERE / "outputs" / "multi_cohort"))
    ap.add_argument("--biomedclip",
                    default="/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/"
                            "outputs/vlm_grounding/scores/biomedclip_scores.npz")
    ap.add_argument("--out",
                    default=str(HERE / "outputs" / "multi_cohort" / "results.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pred_dir = Path(args.pred_dir)
    # Load BiomedCLIP grounding scores (per-cell, 11 concepts).
    bc = np.load(args.biomedclip, allow_pickle=True)
    bc_scores = bc["v1_scores"]  # (4378, 11) cosine-style scores
    bc_filenames = bc["filenames"]
    bc_concepts = bc["concepts"]
    print(f"[bc] {bc_scores.shape}, concepts: {bc_concepts.tolist()}", flush=True)
    bc_fn_to_idx = {fn: i for i, fn in enumerate(bc_filenames)}

    per_seed = {}
    for seed in seeds:
        pred_path = pred_dir / f"predictions_s{seed}.pt"
        if not pred_path.exists():
            print(f"[skip] {pred_path} missing", flush=True); continue
        d = torch.load(pred_path, map_location="cpu", weights_only=False)
        concepts = d["concepts"]
        K = d["K_concepts"]
        cohorts_used = d["cohorts_used"]
        mll_class_names = d.get("mll_class_names")
        full_cp = d["full_concept_probs"]
        full_cls_logits = d["full_class_logits"]
        full_cls_true = d["full_cls_true"]
        gr_paths = d["gr_paths"]

        # --- 1. GR-Neutro test W-F1 / Macro-F1 ---
        N_gr = full_cp[0].shape[0]
        tr_g, val_g, te_g = split_indices(N_gr, seed)
        gr_pred = full_cls_logits[0][te_g].argmax(axis=1)
        gr_true = full_cls_true[0][te_g]
        gr_wf1 = float(f1_score(gr_true, gr_pred, average="weighted", zero_division=0))
        gr_mf1 = float(f1_score(gr_true, gr_pred, average="macro", zero_division=0))
        per_class_recall = {}
        for k in range(int(gr_true.max()) + 1):
            mask = (gr_true == k)
            if mask.sum() > 0:
                per_class_recall[int(k)] = float((gr_pred[mask] == k).mean())

        # --- 2. MLL-23 Normal recall ---
        mll_recall = float("nan")
        if 2 in cohorts_used and mll_class_names is not None:
            try:
                normal_id = mll_class_names.index("neutrophil_segmented")
                mll_pred = full_cls_logits[2].argmax(axis=1)
                mll_true = full_cls_true[2]
                mask = (mll_true == normal_id)
                if mask.sum() > 0:
                    mll_recall = float((mll_pred[mask] == normal_id).mean())
            except Exception as e:
                print(f"[seed {seed}] MLL recall err: {e}")

        # --- 3. Concept singular values on GR-Neutro test ---
        cp_test = full_cp[0][te_g]  # (n_test, K)
        cp_c = cp_test - cp_test.mean(axis=0, keepdims=True)
        s = np.linalg.svd(cp_c, compute_uv=False)
        s_norm = s / s.max() if s.max() > 0 else s
        n_above = int((s_norm >= 0.05).sum())
        gap_7_8 = float(s_norm[6] - s_norm[7]) if len(s_norm) >= 8 else float("nan")

        # --- 4. Per-concept Spearman rho vs BiomedCLIP (with sign flips) ---
        # Join GR-Neutro filename -> BiomedCLIP row.
        gr_fn = [Path(p).name for p in gr_paths]
        bc_row_for_gr = np.array(
            [bc_fn_to_idx.get(fn, -1) for fn in gr_fn], dtype=int)
        joined = bc_row_for_gr >= 0
        if joined.sum() == 0:
            print(f"[seed {seed}] no biomedclip join", flush=True)
            rho_per = [float("nan")] * K
            rho_signed = [float("nan")] * K
        else:
            cp_gr_all = full_cp[0][joined]  # CBM concept probs for joined cells
            bc_gr = bc_scores[bc_row_for_gr[joined]]  # BiomedCLIP scores for same
            # Per-concept Spearman: rho_k = spearmanr(cp_gr_all[:, k], bc_gr[:, k])
            rho_per = []
            rho_signed = []
            for k in range(K):
                r, _ = spearmanr(cp_gr_all[:, k], bc_gr[:, k])
                rho_per.append(float(r) if not np.isnan(r) else 0.0)
                rho_signed.append(abs(float(r)) if not np.isnan(r) else 0.0)
        mean_rho = float(np.mean(rho_per))
        mean_abs_rho = float(np.mean(rho_signed))

        per_seed[seed] = dict(
            gr_weighted_f1=gr_wf1, gr_macro_f1=gr_mf1,
            gr_per_class_recall=per_class_recall,
            mll_normal_recall=mll_recall,
            concept_sv_normalized=s_norm.tolist(),
            n_concepts_above_0p05=n_above,
            gap_sigma7_sigma8=gap_7_8,
            biomedclip_rho_per_concept=rho_per,
            biomedclip_rho_signed_per_concept=rho_signed,
            mean_rho=mean_rho,
            mean_abs_rho=mean_abs_rho,
        )
        print(f"[seed {seed}] W-F1={gr_wf1:.4f} M-F1={gr_mf1:.4f} "
              f"MLL-Nrecall={mll_recall:.4f}", flush=True)
        print(f"[seed {seed}] SV={[round(v,3) for v in s_norm]}", flush=True)
        print(f"[seed {seed}] n_above(0.05)={n_above}  gap(s7-s8)={gap_7_8:.4f}",
              flush=True)
        print(f"[seed {seed}] mean Spearman vs BiomedCLIP={mean_rho:.4f} "
              f"(|rho|={mean_abs_rho:.4f})", flush=True)

    # Aggregate
    if per_seed:
        keys_scalar = ["gr_weighted_f1", "gr_macro_f1", "mll_normal_recall",
                       "n_concepts_above_0p05", "gap_sigma7_sigma8",
                       "mean_rho", "mean_abs_rho"]
        agg = {}
        for k in keys_scalar:
            vals = np.array([per_seed[s][k] for s in per_seed
                             if not np.isnan(per_seed[s][k])], dtype=float)
            if len(vals):
                agg[k + "_mean"] = float(vals.mean())
                agg[k + "_std"] = float(vals.std(ddof=0))
        # Singular values: stack and mean.
        svs = np.array([per_seed[s]["concept_sv_normalized"] for s in per_seed
                        if "concept_sv_normalized" in per_seed[s]])
        if len(svs):
            agg["sv_mean"] = svs.mean(axis=0).tolist()
            agg["sv_std"] = svs.std(axis=0, ddof=0).tolist()
    else:
        agg = {}

    out = {"per_seed": {str(s): per_seed[s] for s in per_seed},
           "aggregate": agg,
           "seeds": seeds}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[save] {args.out}", flush=True)
    if agg:
        print(f"\n[final] W-F1: {agg.get('gr_weighted_f1_mean', float('nan')):.4f} "
              f"+- {agg.get('gr_weighted_f1_std', float('nan')):.4f}")
        print(f"[final] MLL Normal recall: "
              f"{agg.get('mll_normal_recall_mean', float('nan')):.4f} "
              f"+- {agg.get('mll_normal_recall_std', float('nan')):.4f}")
        print(f"[final] n_above(0.05): "
              f"{agg.get('n_concepts_above_0p05_mean', float('nan')):.2f}")
        print(f"[final] gap(s7-s8): "
              f"{agg.get('gap_sigma7_sigma8_mean', float('nan')):.4f}")
        print(f"[final] mean Spearman vs BiomedCLIP: "
              f"{agg.get('mean_rho_mean', float('nan')):.4f}")


if __name__ == "__main__":
    main()
