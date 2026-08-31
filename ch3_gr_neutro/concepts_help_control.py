#!/usr/bin/env python3
"""HONESTY / LOAD-BEARING CONTROL for the concept-residual classification result.

Headline under audit (from outputs/concepts_help_performance/):
    concept_residual_full     86.1% acc   (concepts + learned residual)
    backbone_mlp              85.0% acc   (backbone alone)
    concept_residual_cbmonly  82.9% acc   (pure transparent CBM, residual OFF)
The learned residual adds +3.2 over the pure CBM and the full model is +1.1 over
the backbone MLP.

RT1's concern: is that residual gain CONCEPT-STRUCTURED (complementary to the
concept channel) or generic extra capacity? And are the concepts LOAD-BEARING in
the class head, or cosmetic?

Three controls, each isolating one failure mode. We REUSE the exact setup, split,
seeds, model and vectorised paired bootstrap from concepts_help_performance.py --
nothing is reimplemented; the ConceptResidualNet is subclassed/parameterised.

  C1 RANDOM-ORTHOGONAL-RESIDUAL
      Replace the LEARNED residual block (Linear(hidden->res_dim)+GELU) with a
      FIXED random orthogonal projection of the trunk output h to the same
      res_dim. Its projection weights never train; only the trunk, concept path
      and the residual *head* (cls_from_residual) train. If accuracy ~= the
      learned residual's 86.1, the residual gain is generic capacity, not
      concept-complementary signal.
      Report dAcc(learned_residual - random_residual).

  C2 CONCEPT-INTERVENTION SENSITIVITY
      In the trained concept-residual model, at TEST overwrite the predicted
      concepts chat with the true MEASURED concept values (same standardisation
      used as the regression target), recompute the class logits, and measure how
      much the prediction / accuracy moves. If nothing moves, the class head
      ignores the concept channel (cosmetic). Report the intervention effect size
      (flip rate + dAcc, intervened - clean).

  C3 SHUFFLED-CONCEPT CONTROL
      Train the concept-residual with the concept regression targets PERMUTED
      across cells (label-destroying shuffle, fixed per seed). If accuracy is
      unchanged vs the real-concept model, the concept supervision contributes
      nothing. Report dAcc(real_concept - shuffled_concept).

Multi-seed (the 6 seeds of the parent run) + paired-bootstrap 95% CIs throughout.
NO fabrication. Reports whichever way it falls.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

import concepts_help_performance as C  # reuse everything

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "concepts_help_control"


# --------------------------------------------------------------------------- #
# C1 model: concept-residual with a FIXED random-orthogonal residual projection.
# Subclasses the parent ConceptResidualNet so the concept path + heads are
# byte-identical; only self.residual is swapped for a frozen orthogonal linear.
# --------------------------------------------------------------------------- #
class RandomOrthoResidualNet(C.ConceptResidualNet):
    def __init__(self, d_in, n_classes, n_concepts, res_dim=64, hidden=256, p=0.4, seed=0):
        super().__init__(d_in, n_classes, n_concepts, res_dim=res_dim, hidden=hidden, p=p)
        # Replace the learned residual block with a FROZEN random orthogonal proj.
        g = torch.Generator().manual_seed(1234567 + seed)
        W = torch.empty(res_dim, hidden)
        nn.init.orthogonal_(W, generator=g)          # rows orthonormal (res_dim<=hidden)
        proj = nn.Linear(hidden, res_dim, bias=False)
        proj.weight.data.copy_(W)
        proj.weight.requires_grad_(False)            # never train the projection
        # GELU after, mirroring the learned block; no trainable params in proj.
        self.residual = nn.Sequential(proj, nn.GELU())
        # cls_from_residual (the residual HEAD) stays trainable -- the head is
        # allowed to learn to read the fixed random features, exactly as the
        # learned-residual head does. This is the fair "generic capacity" test.


def train_concept_residual_param(
    Xtr, ytr, Ctr, Xte, n_classes, n_concepts, class_w, seed,
    concept_weight=1.0, epochs=120, device="cpu",
    net_cls=C.ConceptResidualNet, shuffle_concepts=False,
    return_net=False, return_chat_te=False, Cte=None):
    """Parameterised re-implementation of C.train_concept_residual.

    Mirrors the parent trainer EXACTLY (same optimiser, schedule, losses, batch
    size, supervision: full CE + 0.5*CE on concept-only logits + concept MSE),
    with three switches for the controls:
      net_cls           : ConceptResidualNet (default) or RandomOrthoResidualNet (C1)
      shuffle_concepts  : permute concept targets across train rows (C3)
      return_net/...    : expose the net + test concept preds for C2 intervention.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Ctr_np = np.asarray(Ctr, dtype=np.float32).copy()
    if shuffle_concepts:
        # destroy the cell<->concept correspondence; fixed per seed.
        perm = np.random.default_rng(98765 + seed).permutation(len(Ctr_np))
        Ctr_np = Ctr_np[perm]
    Ctr_t = torch.tensor(Ctr_np, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    cw = torch.tensor(class_w, dtype=torch.float32, device=device)

    if net_cls is C.ConceptResidualNet:
        net = net_cls(Xtr.shape[1], n_classes, n_concepts).to(device)
    else:  # RandomOrthoResidualNet needs a seed for its frozen projection
        net = net_cls(Xtr.shape[1], n_classes, n_concepts, seed=seed).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=cw)
    n = len(Xtr_t); bs = 256
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logits, chat, logits_c = net(Xtr_t[b], use_residual=True)
            loss = (ce(logits, ytr_t[b])
                    + 0.5 * ce(logits_c, ytr_t[b])
                    + concept_weight * F.mse_loss(chat, Ctr_t[b]))
            loss.backward(); opt.step()
        sched.step()

    net.eval()
    out = {}
    with torch.no_grad():
        logits_full, chat_te, logits_c = net(Xte_t, use_residual=True)
        out["pred_full"] = logits_full.argmax(1).cpu().numpy()
        out["pred_cbm"] = logits_c.argmax(1).cpu().numpy()
        if return_chat_te:
            out["chat_te"] = chat_te.cpu().numpy()
    if return_net:
        out["net"] = net
    return out


def intervene_with_true_concepts(net, Xte, Cte_true, device="cpu"):
    """C2: forward the trained net but OVERWRITE the predicted concepts chat with
    the true measured (standardised) concept values, then recompute the class
    logits through the SAME class heads. Returns intervened predictions.

    Reproduces ConceptResidualNet.forward internals so we can splice in true
    concepts between the concept_head and the two class heads.
    """
    net.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    Cte_t = torch.tensor(np.asarray(Cte_true, dtype=np.float32), device=device)
    with torch.no_grad():
        h = net.trunk(Xte_t)
        # SPLICE: use true concepts instead of net.concept_head(h)
        logits_c = net.cls_from_concepts(Cte_t)
        r = net.residual(h)
        logits = logits_c + net.cls_from_residual(r)
        return logits.argmax(1).cpu().numpy()


# --------------------------------------------------------------------------- #
def load_aligned():
    """Verbatim alignment from concepts_help_performance.main()."""
    man = json.load(open(C.MANIFEST))
    class_names = man["class_names"]; n_classes = len(class_names)
    cells = man["cells"]
    npz = np.load(C.FEATS_NPZ, allow_pickle=True)
    feats, paths = npz["features"], npz["paths"]
    feat_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}
    morpho = C.load_morpho_by_filename()
    keep, X_dino, M_all, y, labels_oh = [], [], [], [], []
    for c in cells:
        fn = c["filename"]
        if fn not in feat_by_fn or fn not in morpho:
            continue
        keep.append(fn)
        X_dino.append(feat_by_fn[fn])
        M_all.append([morpho[fn][k] for k in C.CONCEPTS])
        y.append(c["dominant_class_idx"])
        labels_oh.append(c["label_one_hot"])
    X_dino = np.vstack(X_dino).astype(np.float32)
    M_all = np.vstack(M_all).astype(np.float32)
    y = np.asarray(y); labels_oh = np.asarray(labels_oh)
    return class_names, n_classes, np.array(keep), X_dino, M_all, y, labels_oh


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    class_names, n_classes, keep, X_dino, M_all, y, labels_oh = load_aligned()
    n_concepts = M_all.shape[1]
    print(f"[align] kept={len(keep)} n_classes={n_classes} n_concepts={n_concepts}",
          flush=True)
    print("[align] class dist: " + ", ".join(
        f"{class_names[k]}={int((y==k).sum())}" for k in range(n_classes)), flush=True)

    SEEDS = C.SEEDS
    per_seed = {}

    for seed in SEEDS:
        tr_idx, val_idx, te_idx = C.stratified_multilabel_split(labels_oh, seed=seed)
        tr = np.concatenate([tr_idx, val_idx])
        yte = y[te_idx]

        counts = np.bincount(y[tr], minlength=n_classes).astype(float)
        class_w = (counts.sum() / (n_classes * np.maximum(counts, 1.0)))

        sc = StandardScaler().fit(X_dino[tr])
        Xtr_s = sc.transform(X_dino[tr]).astype(np.float32)
        Xte_s = sc.transform(X_dino[te_idx]).astype(np.float32)
        csc = StandardScaler().fit(M_all[tr])
        Ctr_s = csc.transform(M_all[tr]).astype(np.float32)
        Cte_s = csc.transform(M_all[te_idx]).astype(np.float32)  # true test concepts (std)

        preds = {}

        # ---------- REFERENCE: learned concept-residual (the 86.1 headline) ----------
        # return the net so C2 intervention runs on the *same* trained model.
        r_learned = train_concept_residual_param(
            Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
            concept_weight=1.0, device=device,
            net_cls=C.ConceptResidualNet, return_net=True, return_chat_te=True)
        preds["learned_residual_full"] = r_learned["pred_full"]
        preds["cbm_only"] = r_learned["pred_cbm"]   # pure transparent CBM (residual off)

        # ---------- C1: random-orthogonal residual ----------
        r_rand = train_concept_residual_param(
            Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
            concept_weight=1.0, device=device,
            net_cls=RandomOrthoResidualNet)
        preds["random_residual_full"] = r_rand["pred_full"]

        # ---------- C2: concept intervention (true measured concepts at test) ----------
        preds["intervened_true_concepts"] = intervene_with_true_concepts(
            r_learned["net"], Xte_s, Cte_s, device=device)

        # ---------- C3: shuffled-concept supervision ----------
        r_shuf = train_concept_residual_param(
            Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
            concept_weight=1.0, device=device,
            net_cls=C.ConceptResidualNet, shuffle_concepts=True)
        preds["shuffled_concept_full"] = r_shuf["pred_full"]
        preds["shuffled_concept_cbmonly"] = r_shuf["pred_cbm"]

        # ---------- evaluate ----------
        seed_res = {name: C.eval_pred(yte, p, n_classes) for name, p in preds.items()}

        # ---------- paired bootstrap deltas (the control comparisons) ----------
        rng = np.random.default_rng(seed)
        boot = {}
        # C1: learned residual minus random residual (positive => residual is
        #     concept-complementary, not generic capacity)
        boot["C1_learned_minus_random"] = C.paired_bootstrap_acc(
            yte, preds["random_residual_full"], preds["learned_residual_full"],
            n_classes, rng)
        # C2: intervened minus clean (true concepts swapped in at test). Large
        #     magnitude => class head USES the concept channel (load-bearing).
        boot["C2_intervened_minus_clean"] = C.paired_bootstrap_acc(
            yte, preds["learned_residual_full"], preds["intervened_true_concepts"],
            n_classes, rng)
        # also raw flip rate (any change in prediction) for C2
        flip = float((preds["intervened_true_concepts"] != preds["learned_residual_full"]).mean())
        # C3: real-concept minus shuffled-concept (positive => concept supervision
        #     contributes signal)
        boot["C3_real_minus_shuffled"] = C.paired_bootstrap_acc(
            yte, preds["shuffled_concept_full"], preds["learned_residual_full"],
            n_classes, rng)

        seed_res["_boot"] = boot
        seed_res["_c2_flip_rate"] = flip
        per_seed[str(seed)] = seed_res

        print(f"[seed {seed}] learned={seed_res['learned_residual_full']['accuracy']:.4f} "
              f"random_res={seed_res['random_residual_full']['accuracy']:.4f} "
              f"cbm_only={seed_res['cbm_only']['accuracy']:.4f} "
              f"intervened={seed_res['intervened_true_concepts']['accuracy']:.4f} "
              f"(flip={flip:.3f}) shuffled={seed_res['shuffled_concept_full']['accuracy']:.4f}",
              flush=True)

    # ----------------------------- aggregate ----------------------------- #
    def acc_agg(name):
        vals = [per_seed[str(s)][name]["accuracy"] for s in SEEDS]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "ci95": C.seed_ci(vals), "per_seed": [float(v) for v in vals]}

    def bal_agg(name):
        vals = [per_seed[str(s)][name]["balanced_accuracy"] for s in SEEDS]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def boot_agg(key):
        # seed-mean of per-seed paired-bootstrap means + seed-level CI; also the
        # mean within-seed bootstrap CI and mean P(delta>0).
        dacc = [per_seed[str(s)]["_boot"][key]["d_accuracy"][0] for s in SEEDS]
        lo = [per_seed[str(s)]["_boot"][key]["d_accuracy"][1] for s in SEEDS]
        hi = [per_seed[str(s)]["_boot"][key]["d_accuracy"][2] for s in SEEDS]
        pg = [per_seed[str(s)]["_boot"][key]["d_accuracy"][3] for s in SEEDS]
        dbal = [per_seed[str(s)]["_boot"][key]["d_balanced_accuracy"][0] for s in SEEDS]
        return {
            "d_accuracy_seedmean_ci95": C.seed_ci(dacc),
            "d_accuracy_per_seed": [float(v) for v in dacc],
            "mean_within_seed_bootstrap_ci95": [float(np.mean(lo)), float(np.mean(hi))],
            "mean_P_delta_gt0": float(np.mean(pg)),
            "d_balanced_accuracy_seedmean": C.seed_ci(dbal),
        }

    variants = ["learned_residual_full", "random_residual_full", "cbm_only",
                "intervened_true_concepts", "shuffled_concept_full",
                "shuffled_concept_cbmonly"]
    summary = {
        "n_cells": int(len(keep)), "n_classes": n_classes,
        "class_names": class_names, "seeds": SEEDS,
        "concepts": C.CONCEPTS, "n_concepts": n_concepts,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(n_classes)},
        "accuracy": {v: acc_agg(v) for v in variants},
        "balanced_accuracy": {v: bal_agg(v) for v in variants},
        "c2_flip_rate_mean": float(np.mean([per_seed[str(s)]["_c2_flip_rate"] for s in SEEDS])),
        "c2_flip_rate_per_seed": [float(per_seed[str(s)]["_c2_flip_rate"]) for s in SEEDS],
        "controls": {
            "C1_learned_minus_random": boot_agg("C1_learned_minus_random"),
            "C2_intervened_minus_clean": boot_agg("C2_intervened_minus_clean"),
            "C3_real_minus_shuffled": boot_agg("C3_real_minus_shuffled"),
        },
        "time_seconds": time.time() - t0,
    }

    (OUT_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "per_seed": per_seed}, indent=2))
    print(f"\n[save] {OUT_DIR}/results.json ({time.time()-t0:.0f}s)", flush=True)

    # ---- stdout headline ----
    A = summary["accuracy"]
    print("\n[ACCURACY  mean over seeds]")
    for v in variants:
        print(f"  {v:28s} {A[v]['mean']:.4f}")
    print("\n[CONTROL DELTAS  (seed-mean dAcc [seed-CI], mean within-seed bootstrap CI, P>0)]")
    for k in ["C1_learned_minus_random", "C2_intervened_minus_clean", "C3_real_minus_shuffled"]:
        c = summary["controls"][k]
        sm = c["d_accuracy_seedmean_ci95"]; bw = c["mean_within_seed_bootstrap_ci95"]
        print(f"  {k:30s} dAcc={sm[0]:+.4f} [seed {sm[1]:+.4f},{sm[2]:+.4f}] "
              f"boot[{bw[0]:+.4f},{bw[1]:+.4f}] P>0={c['mean_P_delta_gt0']:.2f}")
    print(f"  C2 flip rate (pred changes under true-concept swap) = {summary['c2_flip_rate_mean']:.3f}")

    write_markdown(summary)


def write_markdown(s):
    A = s["accuracy"]; ctl = s["controls"]
    c1 = ctl["C1_learned_minus_random"]; c2 = ctl["C2_intervened_minus_clean"]
    c3 = ctl["C3_real_minus_shuffled"]
    learned = A["learned_residual_full"]["mean"]
    random_r = A["random_residual_full"]["mean"]
    cbm = A["cbm_only"]["mean"]
    interv = A["intervened_true_concepts"]["mean"]
    shuf = A["shuffled_concept_full"]["mean"]

    def fmt(c):
        sm = c["d_accuracy_seedmean_ci95"]; bw = c["mean_within_seed_bootstrap_ci95"]
        return (f"{sm[0]*100:+.2f} pp (seed-CI [{sm[1]*100:+.2f}, {sm[2]*100:+.2f}]; "
                f"within-seed bootstrap CI [{bw[0]*100:+.2f}, {bw[1]*100:+.2f}]; "
                f"P(>0)={c['mean_P_delta_gt0']:.2f})")

    # ---- verdict logic (data-driven, ~200 words) ----
    c1_d = c1["d_accuracy_seedmean_ci95"][0]
    c1_lo = c1["mean_within_seed_bootstrap_ci95"][0]
    c2_d = abs(c2["d_accuracy_seedmean_ci95"][0]); flip = s["c2_flip_rate_mean"]
    c3_d = c3["d_accuracy_seedmean_ci95"][0]; c3_lo = c3["mean_within_seed_bootstrap_ci95"][0]

    residual_complementary = (c1_d > 0.005) and (c1_lo > 0)      # learned beats random meaningfully
    concepts_loadbearing = (flip > 0.02) or (c2_d > 0.005)        # intervention moves predictions
    supervision_matters = (c3_d > 0.005) and (c3_lo > 0)          # real beats shuffled

    if not residual_complementary:
        verdict_head = ("NO -- the residual gain is GENERIC CAPACITY, not concept-"
                        "complementary structure. The random-orthogonal residual "
                        "matches the learned residual.")
        claim = ("We must NOT claim 'concepts help accuracy'. The honest claim is "
                 "EXPLAINABILITY IS FREE: the transparent CBM ({:.1f}%) is the real "
                 "explainable number; the residual's lift to {:.1f}% is generic "
                 "extra capacity that any random projection of the backbone supplies."
                 ).format(cbm*100, learned*100)
    elif residual_complementary and concepts_loadbearing and supervision_matters:
        verdict_head = ("YES -- the concept-residual accuracy is genuinely concept-"
                        "driven. The learned residual beats a random-orthogonal "
                        "residual, concept intervention moves predictions, and real "
                        "concept supervision beats shuffled.")
        claim = ("The concepts are LOAD-BEARING and the residual is COMPLEMENTARY, "
                 "not a concept costume over generic capacity.")
    else:
        bits = []
        bits.append("residual complementary" if residual_complementary else "residual is GENERIC capacity")
        bits.append("concepts load-bearing" if concepts_loadbearing else "concepts COSMETIC in the class head")
        bits.append("supervision matters" if supervision_matters else "concept supervision INERT")
        verdict_head = "MIXED -- " + "; ".join(bits) + "."
        claim = ("Be conservative: only claims supported by a positive, "
                 "CI-excluding-zero control should survive. See deltas above.")

    md = f"""# Concepts-help HONESTY / LOAD-BEARING control

Audit of the concept-residual classification result in
`outputs/concepts_help_performance/`. Reuses the EXACT split, 6 seeds, model
(`ConceptResidualNet`), and vectorised paired bootstrap of
`concepts_help_performance.py` (imported, not reimplemented). CPU, local.

## Accuracy (mean over {len(s['seeds'])} seeds)

| model | accuracy | balanced acc |
|---|---|---|
| learned concept-residual (headline) | {learned*100:.2f}% | {s['balanced_accuracy']['learned_residual_full']['mean']*100:.2f}% |
| **C1** random-orthogonal residual | {random_r*100:.2f}% | {s['balanced_accuracy']['random_residual_full']['mean']*100:.2f}% |
| pure CBM (residual OFF, transparent) | {cbm*100:.2f}% | {s['balanced_accuracy']['cbm_only']['mean']*100:.2f}% |
| **C2** intervened w/ true concepts | {interv*100:.2f}% | {s['balanced_accuracy']['intervened_true_concepts']['mean']*100:.2f}% |
| **C3** shuffled-concept residual | {shuf*100:.2f}% | {s['balanced_accuracy']['shuffled_concept_full']['mean']*100:.2f}% |

## Control deltas (paired bootstrap, seed-aggregated)

- **C1 learned − random-orthogonal residual:** {fmt(c1)}
  -> If ~0: residual gain is generic capacity. If >0 (CI excludes 0): residual carries concept-complementary signal.
- **C2 intervened − clean (true concepts swapped in at test):** {fmt(c2)}; **prediction flip rate = {flip*100:.1f}%**.
  -> If flip~0 and dAcc~0: the class head ignores the concept channel (cosmetic). Nonzero => concepts are read.
- **C3 real − shuffled concept supervision:** {fmt(c3)}
  -> If ~0: concept supervision contributes nothing. If >0 (CI excludes 0): supervision adds signal.

## Verdict

**{verdict_head}**

{claim}

Concretely: pure transparent CBM = {cbm*100:.1f}%, learned concept-residual = {learned*100:.1f}%,
random-orthogonal residual = {random_r*100:.1f}%, shuffled-concept residual = {shuf*100:.1f}%.
The three control deltas are C1={c1['d_accuracy_seedmean_ci95'][0]*100:+.2f}pp,
C2={c2['d_accuracy_seedmean_ci95'][0]*100:+.2f}pp (flip {flip*100:.1f}%),
C3={c3['d_accuracy_seedmean_ci95'][0]*100:+.2f}pp.
No fabrication: numbers are emitted directly by the control run.
"""
    (OUT_DIR / "results.md").write_text(md)
    print(f"[save] {OUT_DIR}/results.md", flush=True)


if __name__ == "__main__":
    main()
