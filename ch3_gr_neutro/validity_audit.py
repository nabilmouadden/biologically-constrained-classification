#!/usr/bin/env python3
"""Construct-validity / prevalence-specificity audit for morphometry concept channels.

Motivation: the cytoplasmic_vacuolization_degree channel was DETERMINISTIC and passed
the concept->class signed-AUC>=0.60 reliability gate (twosided AUC 0.634), yet it measured
an artefact for a feature that is biologically ABSENT from the GR-Neutro neutrophil cohort.
Determinism (reproducibility) is NOT construct validity. This audit screens every remaining
concept channel for vacuolization-style invalidity using three label-free checks:

  1. PREVALENCE / base-rate   : fraction of cells scored "positive" (>0.1, >0.5) and channel mean.
                                A channel firing on a large majority of cells for a feature that
                                should be rare/specific is a red flag (vacuolization fired ~85%).
  2. FIRES-WHERE-ABSENT       : positive-rate and mean WITHIN the Normal class (and within classes
     (specificity)             where the named feature is not biologically expected). A faithful
                                abnormality channel should be near baseline on Normal cells.
  3. CONFOUND / nuisance      : how much of the channel's class-separation is explained by generic
                                nuisance summaries (cell/segmentation size + granule count + N:C),
                                via a nuisance-only OLS vs a nuisance+class ANOVA-style decomposition.

Verdict per channel: VALID vs SUSPECT (vacuolization-style: high prevalence incl. Normal, or
nuisance-driven, or AUC near chance / reliability-fail).

No image loading, no model: operates purely on the already-computed per-cell concept CSV plus the
raw nuisance summaries it carries. Tiny CPU job.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# the 10 concept channels under audit (vacuolization is the known-bad reference, audited too for contrast)
CONCEPTS = [
    "nuclear_lobulation_degree",
    "nuclear_contour_irregularity",
    "nucleus_to_cytoplasm_ratio",
    "chromatin_condensation_level",
    "chromatin_clumping_pattern",
    "cytoplasmic_granule_density",
    "granule_coarseness",
    "cytoplasmic_texture_uniformity",
    "cytoplasm_basophilia_level",
    "cytoplasmic_inclusion_visibility",
]
VACUOLIZATION = "cytoplasmic_vacuolization_degree"  # reference invalid channel

CLASSES = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
           "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]

# generic nuisance summaries available in the CSV (size / segmentation / stain-agnostic counts).
# These are NOT, in general, the named biological construct of a channel; if a channel's class-signal
# is largely reconstructable from these, its separation is nuisance-driven. HOWEVER for a few channels
# one of these raw summaries IS the channel's own construct (e.g. lobe_count == lobulation), so it
# must be EXCLUDED from that channel's nuisance set -- regressing a channel on its own construct would
# falsely brand the construct itself as a "nuisance". CONSTRUCT_RAW maps each channel to the raw
# column(s) that ARE its construct and must be dropped from the nuisance regression for that channel.
NUISANCE = ["lobe_count", "nc_ratio_raw", "granule_count"]
CONSTRUCT_RAW = {
    "nuclear_lobulation_degree":   ["lobe_count"],     # lobulation IS lobe count
    "nucleus_to_cytoplasm_ratio":  ["nc_ratio_raw"],   # N:C IS the raw N:C ratio
    "cytoplasmic_granule_density": ["granule_count"],  # density IS granule count / area
}

# named target class(es) the channel is supposed to be ELEVATED in (its construct's home).
# A faithful abnormality channel must score meaningfully higher on its target than on Normal.
TARGET_CLASS = {
    "nuclear_lobulation_degree":      "Hypersegmentation",  # high-lobe target (also separates Hyposeg low)
    "nuclear_contour_irregularity":   "Hypersegmentation",
    "nucleus_to_cytoplasm_ratio":     "Hypergranulation",
    "chromatin_condensation_level":   "Chromatin",
    "chromatin_clumping_pattern":     "Chromatin",
    "cytoplasmic_granule_density":    "Hypergranulation",
    "granule_coarseness":             "Hypergranulation",
    "cytoplasmic_texture_uniformity": "Hypergranulation",   # uniformity LOW (non-uniform) in hypergran.
    "cytoplasm_basophilia_level":     "Dohle",
    "cytoplasmic_inclusion_visibility":"Dohle",
    "cytoplasmic_vacuolization_degree":"Hypergranulation",
}
# channels whose construct is LOW in the target (signed "low" direction); for these the target mean
# should be BELOW Normal, not above.
TARGET_LOW = {"cytoplasmic_texture_uniformity"}


def auc_mw(pos, neg):
    """Mann-Whitney AUC of `pos` scoring higher than `neg` (two-sided reported as max(a,1-a))."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average-rank tie correction
    s = np.sort(allv)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    a = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(a), float(max(a, 1 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df["seg_ok"] == 1].reset_index(drop=True)
    n = len(df)

    # recover the single class label from the one-hot columns
    onehot = df[CLASSES].values
    cls = np.array(CLASSES)[onehot.argmax(1)]
    df["__class"] = cls

    with open(args.validation) as f:
        val = json.load(f)
    per_val = val.get("per_concept", {})

    results = {}
    for ch in CONCEPTS + [VACUOLIZATION]:
        v = df[ch].astype(float).values
        # ---- (1) prevalence / base-rate ----
        prev_01 = float((v > 0.1).mean())
        prev_05 = float((v > 0.5).mean())
        mean_all = float(v.mean())
        std_all = float(v.std())

        # ---- (2) fires-where-absent / specificity ----
        # A faithful abnormality channel must (a) NOT fire on most Normal cells, and
        # (b) be elevated on its named target class relative to Normal (in the construct's signed
        # direction). We read specificity as target-vs-Normal contrast, not a blunt absent/rest ratio.
        normal_mask = df["__class"].values == "Normal"
        mean_normal = float(v[normal_mask].mean())
        prev05_normal = float((v[normal_mask] > 0.5).mean())
        tgt = TARGET_CLASS.get(ch, "Normal")
        tgt_mask = df["__class"].values == tgt
        mean_target = float(v[tgt_mask].mean())
        # signed target-vs-Normal gap (positive => fires more on target than Normal as it should).
        # For LOW-direction channels the construct is low in the target, so flip the sign.
        gap = mean_target - mean_normal
        if ch in TARGET_LOW:
            gap = -gap
        target_minus_normal = float(gap)

        # per-class means for the report
        per_class_mean = {c: float(v[df["__class"].values == c].mean()) for c in CLASSES}
        per_class_prev05 = {c: float((v[df["__class"].values == c] > 0.5).mean()) for c in CLASSES}

        # ---- best class-separation AUC (from validation table; recomputed for cross-check) ----
        vv = per_val.get(ch, {})
        best_auc_twosided = vv.get("best_auc_twosided")
        reliable_gate = vv.get("reliable")
        # recompute best two-sided AUC against each class (sanity / source of truth)
        recomputed = {}
        for c in CLASSES:
            m = df["__class"].values == c
            _, a2 = auc_mw(v[m], v[~m])
            recomputed[c] = round(a2, 4)
        best_class = max(recomputed, key=recomputed.get)
        best_auc_recomputed = recomputed[best_class]

        # ---- (3) confound check: does the channel's class-separation survive removing GENERIC nuisance?
        # Build a per-channel nuisance set: the generic size/segmentation summaries MINUS any column
        # that IS this channel's own construct (regressing a channel on its own construct would falsely
        # brand the construct as a nuisance). Residualise the channel on these nuisances, then check
        # whether the best-class separation (two-sided AUC) survives. A channel whose AUC collapses to
        # chance once generic size/count nuisance is removed is nuisance-driven, not construct-driven.
        nuis_cols = [c for c in NUISANCE if c not in CONSTRUCT_RAW.get(ch, [])]
        Xn = df[nuis_cols].astype(float).values
        keepc = Xn.std(0) > 1e-9
        Xn = Xn[:, keepc]
        nuis_used = [c for c, k in zip(nuis_cols, keepc) if k]
        if Xn.shape[1] > 0:
            Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
            reg = LinearRegression().fit(Xn, v)
            v_hat = reg.predict(Xn)
            nuis_r2 = float(1 - np.var(v - v_hat) / (np.var(v) + 1e-12))
            resid = v - v_hat
        else:
            nuis_r2 = 0.0
            resid = v.copy()

        def class_eta2(values):
            grand = values.mean()
            ss_tot = ((values - grand) ** 2).sum()
            ss_between = 0.0
            for c in CLASSES:
                m = df["__class"].values == c
                if m.sum() == 0:
                    continue
                ss_between += m.sum() * (values[m].mean() - grand) ** 2
            return float(ss_between / (ss_tot + 1e-12))

        eta2_raw = class_eta2(v)
        eta2_resid = class_eta2(resid)
        class_signal_retained = float(eta2_resid / (eta2_raw + 1e-12))
        nuisance_driven_frac = float(1 - class_signal_retained)
        # best-class two-sided AUC after nuisance residualisation -- the decisive confound read
        bm = df["__class"].values == best_class
        _, best_auc_resid = auc_mw(resid[bm], resid[~bm])

        # ---- verdict ----
        flags = []
        # vacuolization-style prevalence: fires on a large majority incl. Normal
        if prev_05 > 0.5 and prev05_normal > 0.4:
            flags.append(f"high prevalence ({prev_05:.0%} of all, {prev05_normal:.0%} of Normal cells fire >0.5)")
        # near-saturated channel (almost no variance / unique values) -> measuring a constant
        if std_all < 0.02:
            flags.append(f"near-constant channel (std={std_all:.3f}); measures no real variation")
        # fails the reliability gate outright
        if reliable_gate is False or (best_auc_twosided is not None and best_auc_twosided < 0.60):
            flags.append(f"AUC near chance / reliability-fail (best two-sided AUC {best_auc_twosided})")
        # reversed polarity: signed AUC < 0.5 on its named target -> not tracking the named construct
        rev = False
        for t in vv.get("targets", []):
            if t.get("signal_reversed"):
                rev = True
                flags.append(f"REVERSED polarity on named target '{t['class']}' "
                             f"(signed AUC {t.get('morphometry_auc')}); not tracking the named construct")
        # not elevated on its named target relative to Normal -> not tracking the named construct.
        # (Only meaningful for a channel that otherwise passes the gate; reversed channels already flagged.)
        if (not rev) and best_auc_twosided is not None and best_auc_twosided >= 0.60 \
                and target_minus_normal < 0.02:
            flags.append(f"not target-specific: not elevated on named target '{tgt}' vs Normal "
                         f"(target-Normal gap {target_minus_normal:+.3f}); separation comes from other "
                         f"classes, not the construct's home")
        # nuisance-driven separation: the best-class AUC collapses below the gate once GENERIC
        # (non-construct) size/count nuisance is removed.
        if best_auc_twosided is not None and best_auc_twosided >= 0.60 and best_auc_resid < 0.60:
            flags.append(f"nuisance-driven: best-class two-sided AUC drops {best_auc_recomputed:.3f}"
                         f"->{best_auc_resid:.3f} (below gate) after removing generic nuisance "
                         f"{nuis_used}; separation is size/count-driven, not the named construct")

        verdict = "SUSPECT" if flags else "VALID"

        results[ch] = {
            "prevalence": {"pos_rate_gt0.1": round(prev_01, 4),
                            "pos_rate_gt0.5": round(prev_05, 4),
                            "mean": round(mean_all, 4), "std": round(std_all, 4)},
            "specificity": {"mean_on_Normal": round(mean_normal, 4),
                             "pos_rate_gt0.5_on_Normal": round(prev05_normal, 4),
                             "target_class": tgt,
                             "mean_on_target": round(mean_target, 4),
                             "target_minus_normal_signed": round(target_minus_normal, 4)},
            "per_class_mean": {k: round(x, 3) for k, x in per_class_mean.items()},
            "per_class_pos_rate_gt0.5": {k: round(x, 3) for k, x in per_class_prev05.items()},
            "separation": {"best_class": best_class,
                            "best_auc_twosided_recomputed": round(best_auc_recomputed, 4),
                            "best_auc_twosided_validation": best_auc_twosided,
                            "reliable_gate_passed": reliable_gate,
                            "reversed_polarity": rev},
            "confound": {"nuisance_used": nuis_used,
                          "nuisance_R2_on_channel": round(nuis_r2, 4),
                          "class_eta2_raw": round(eta2_raw, 4),
                          "class_eta2_after_nuisance": round(eta2_resid, 4),
                          "class_signal_retained": round(class_signal_retained, 4),
                          "nuisance_driven_frac": round(nuisance_driven_frac, 4),
                          "best_class_auc_after_nuisance": round(best_auc_resid, 4)},
            "flags": flags,
            "verdict": verdict,
        }

    n_valid = sum(1 for c in CONCEPTS if results[c]["verdict"] == "VALID")
    n_suspect = sum(1 for c in CONCEPTS if results[c]["verdict"] == "SUSPECT")
    summary = {
        "n_cells": int(n),
        "n_concepts_audited": len(CONCEPTS),
        "n_valid": n_valid,
        "n_suspect": n_suspect,
        "suspect_concepts": [c for c in CONCEPTS if results[c]["verdict"] == "SUSPECT"],
        "valid_concepts": [c for c in CONCEPTS if results[c]["verdict"] == "VALID"],
        "vacuolization_reference": {
            "verdict": results[VACUOLIZATION]["verdict"],
            "flags": results[VACUOLIZATION]["flags"],
            "prevalence_gt0.5": results[VACUOLIZATION]["prevalence"]["pos_rate_gt0.5"],
            "mean": results[VACUOLIZATION]["prevalence"]["mean"],
            "pos_rate_gt0.5_on_Normal": results[VACUOLIZATION]["specificity"]["pos_rate_gt0.5_on_Normal"],
        },
        "nuisance_proxies": NUISANCE,
        "construct_raw_excluded_per_channel": CONSTRUCT_RAW,
    }

    out = {"summary": summary, "per_concept": results}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # markdown
    lines = []
    lines.append("# Construct-validity / prevalence-specificity audit\n")
    lines.append(f"Cells: {n}. Concepts audited: {len(CONCEPTS)}. "
                 f"VALID: {n_valid}. SUSPECT: {n_suspect}.\n")
    lines.append(f"Generic nuisance proxies: {', '.join(NUISANCE)} "
                 f"(each channel's own construct-defining raw column is excluded from its nuisance "
                 f"regression: {CONSTRUCT_RAW})\n")
    lines.append("\n## Reference: cytoplasmic_vacuolization_degree (the known-invalid channel)\n")
    rv = results[VACUOLIZATION]
    lines.append(f"- prevalence >0.5 = {rv['prevalence']['pos_rate_gt0.5']:.0%}, "
                 f"mean = {rv['prevalence']['mean']}, fires>0.5 on Normal = "
                 f"{rv['specificity']['pos_rate_gt0.5_on_Normal']:.0%}")
    lines.append(f"- best two-sided AUC (validation) = {rv['separation']['best_auc_twosided_validation']}, "
                 f"reliability gate passed = {rv['separation']['reliable_gate_passed']}")
    lines.append(f"- VERDICT: **{rv['verdict']}** -- flags: {('; '.join(rv['flags'])) or 'none'}\n")

    lines.append("\n## Per-concept audit\n")
    lines.append("| concept | prev>0.5 | mean | fires>0.5 on Normal | best AUC (2s) | gate | target-Normal gap | AUC after nuisance | reversed | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ch in CONCEPTS:
        r = results[ch]
        lines.append("| {} | {:.0%} | {:.2f} | {:.0%} | {} | {} | {:+.3f} | {:.3f} | {} | **{}** |".format(
            ch, r["prevalence"]["pos_rate_gt0.5"], r["prevalence"]["mean"],
            r["specificity"]["pos_rate_gt0.5_on_Normal"],
            r["separation"]["best_auc_twosided_validation"],
            "pass" if r["separation"]["reliable_gate_passed"] else "FAIL",
            r["specificity"]["target_minus_normal_signed"],
            r["confound"]["best_class_auc_after_nuisance"],
            "YES" if r["separation"]["reversed_polarity"] else "no",
            r["verdict"]))
    lines.append("")
    for ch in CONCEPTS:
        r = results[ch]
        lines.append(f"\n### {ch} -- {r['verdict']}")
        if r["flags"]:
            for fl in r["flags"]:
                lines.append(f"- FLAG: {fl}")
        else:
            lines.append("- no flags (specific prevalence, near-baseline on absent classes, not nuisance-driven)")
        lines.append(f"- per-class mean: {r['per_class_mean']}")
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print("AUDIT DONE")


if __name__ == "__main__":
    main()
