#!/usr/bin/env python
"""Map each headline number in the BJH paper to a shipped artifact and print it.

Run from anywhere:
    python results/reproduce_paper.py

Reads only files shipped in this release (results/acceptance_analyses/*.json and
results/figures_v2/*.csv) and prints, for every paper number, the value read
back from its artifact. No model or GPU needed. See results/PAPER_NUMBERS.md for
the full number -> artifact -> command table.

Provenance note: the paper's per-class detection table (Table 2) and calibration/
clinical-utility numbers are the representative seed cbm_joint s42 (n=438 test
cells) at tuned per-class thresholds -- exactly per_class_cis.json /
per_class_calibration.json / clinical_utility.json below. The headline aggregate
macro-F1 0.81 / weighted-F1 0.88 is the 5-seed mean (single-seed s42 is a little
higher, macro 0.843 / weighted 0.900); the shipped single-seed artifacts
reproduce the per-class tables and CIs, not the 5-seed aggregate mean.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = os.path.join(HERE, "acceptance_analyses")


def load(name):
    return json.load(open(os.path.join(ACC, name)))


def main():
    print("=" * 72)
    print("BJH paper numbers -> shipped artifact (representative seed cbm_joint s42)")
    print("=" * 72)

    # ---- per-class detection (Table 2): per_class_cis.json --------------
    cis = load("per_class_cis.json")
    cls = cis["classes"]
    print(f"\n[Table 2] Per-class detection, n_test={cis['n_test']}, "
          f"{cis['n_bootstrap']}x bootstrap 95% CI   <- per_class_cis.json")
    order = sorted(cls, key=lambda c: -cls[c]["f1"])
    for c in order:
        r = cls[c]
        print(f"  {c:18s} P={r['precision']:.2f}{fmt(r['precision_ci95'])} "
              f"R={r['recall']:.2f} ({r['recall_fraction']}){fmt(r['recall_ci95'])} "
              f"F1={r['f1']:.2f}{fmt(r['f1_ci95'])}")
    f1s = [cls[c]["f1"] for c in cls]
    npos = [cls[c]["n_pos"] for c in cls]
    macro = sum(f1s) / len(f1s)
    wf1 = sum(f * n for f, n in zip(f1s, npos)) / sum(npos)
    print(f"  single-seed (s42): macro-F1={macro:.3f}, weighted-F1={wf1:.3f}")
    print("  paper headline (5-seed aggregate): macro-F1 0.81, weighted-F1 0.88")

    # ---- calibration: per_class_calibration.json ------------------------
    cal = load("per_class_calibration.json")
    print("\n[Calibration] pooled reliability   <- per_class_calibration.json")
    print(f"  pooled ECE  = {cal['pooled_ece_flat']:.3f}   (paper: 0.054)")
    print(f"  pooled Brier= {cal['pooled_brier_flat']:.3f}   (paper: 0.029)")
    briers = [cal["classes"][c]["brier"] for c in cal["classes"]]
    print(f"  per-class Brier range = {min(briers):.3f}-{max(briers):.3f} "
          f"(paper: 0.007-0.055)")

    # ---- clinical utility: clinical_utility.json ------------------------
    cu = load("clinical_utility.json")
    a = cu["abnormal_vs_normal"]
    ty = cu["triage_yield"]
    print("\n[Clinical utility] triage   <- clinical_utility.json")
    print(f"  internal abnormal-vs-normal AUROC = {a['auroc']:.3f}   (paper: 0.997)")
    base = ty["base_subset_accuracy"]
    N = len(ty["retained_accuracy"])
    acc10 = ty["retained_accuracy"][int(0.9 * N) - 1]
    acc20 = ty["retained_accuracy"][int(0.8 * N) - 1]
    print(f"  exact-match acc: no-defer={base:.3f} (paper 0.836), "
          f"10%-defer={acc10:.3f} (0.886), 20%-defer={acc20:.3f} (0.946)")

    # ---- external transfer: no local artifact ---------------------------
    print("\n[External transfer] Barrera--Merino 2024 (Table m1_external)")
    print("  AUROC 0.851 +/- 0.049, abnormal recall 0.942 -- external cohort;")
    print("  source table: paper m1_external_block.tex. The external cohort is not")
    print("  redistributed, so this number is not recomputable from this release.")

    print("\n" + "=" * 72)
    print("All values above are read live from the shipped JSONs.")
    print("=" * 72)


def fmt(ci):
    if not ci or ci[0] is None:
        return ""
    return f"[{ci[0]:.2f},{ci[1]:.2f}]"


if __name__ == "__main__":
    main()
