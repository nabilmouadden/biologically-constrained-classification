"""Aggregate all run summary.json + eval.json into a single CSV table.

Usage:  python summarize_runs.py [--out_root outputs] [--csv master_results.csv]

Run after each batch of jobs to track the leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    ap.add_argument("--csv", default=str(HERE / "master_results.csv"))
    args = ap.parse_args()
    rows = []
    for d in sorted(Path(args.out_root).iterdir()):
        s_path = d / "summary.json"
        if not s_path.exists():
            continue
        s = json.loads(s_path.read_text())
        e_path = d / "eval.json"
        e = json.loads(e_path.read_text()) if e_path.exists() else {}
        cls = s.get("test_classification", {})
        cons = s.get("test_concepts", {})
        a = s.get("args", {})
        row = dict(
            tag=s.get("tag", d.name),
            seed=a.get("seed"),
            baseline=a.get("baseline", False),
            lambda_constraint=a.get("lambda_constraint"),
            unfreeze_last_n=a.get("unfreeze_last_n"),
            epochs=a.get("epochs"),
            weighted_f1=cls.get("weighted_f1"),
            macro_f1=cls.get("macro_f1"),
            micro_f1=cls.get("micro_f1"),
            balanced_accuracy=cls.get("balanced_accuracy"),
            mean_ap=cls.get("mean_ap"),
            mean_concept_f1=cons.get("mean_concept_f1"),
            macro_concept_f1=cons.get("macro_concept_f1"),
            coverage_at_0p05=e.get("conformal", {}).get("marginal_positive_coverage"),
            mean_set_size=e.get("conformal", {}).get("mean_set_size"),
            probe_subset_acc=e.get("completeness_probe", {}).get("subset_accuracy"),
            probe_weighted_f1=e.get("completeness_probe", {}).get("weighted_f1"),
            probe_fit_split=e.get("completeness_probe", {}).get("fit_split"),
            cooc_corr=e.get("cooccurrence", {}).get("off_diag_pearson_corr"),
            violation_rate=e.get("violation", {}).get("violation_rate"),
            tuned_weighted_f1=e.get("classification_tuned", {}).get("weighted_f1"),
            tuned_macro_f1=e.get("classification_tuned", {}).get("macro_f1"),
            tuned_concept_f1=e.get("concept_tuned", {}).get("mean_concept_f1"),
        )
        rows.append(row)
    if not rows:
        print(f"[summarize] no summary.json files found under {args.out_root}")
        return
    fields = list(rows[0].keys())
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        # Sort by weighted_f1 desc, NaN last
        rows.sort(key=lambda r: r.get("weighted_f1") or -1, reverse=True)
        w.writerows(rows)
    print(f"[summarize] {args.csv} ({len(rows)} runs)")
    # Pretty print top 10
    head = ["tag", "weighted_f1", "tuned_weighted_f1", "macro_f1",
             "mean_concept_f1", "tuned_concept_f1",
             "coverage_at_0p05", "probe_subset_acc", "probe_weighted_f1",
             "violation_rate", "lambda_constraint", "seed"]
    print("\n" + " | ".join(f"{h:>20s}" for h in head))
    for r in rows[:10]:
        print(" | ".join(
            (f"{r.get(h,'')!s:>20s}" if not isinstance(r.get(h), float)
             else f"{r.get(h):>20.4f}") for h in head))


if __name__ == "__main__":
    main()
