"""Ensemble evaluation: average sigmoid probabilities across multiple runs.

Combines outputs/<tag1>/predictions.pt + outputs/<tag2>/predictions.pt + ... into
a synthetic predictions_ensemble.pt. All runs must share the same split (same seed).

Usage:  python ensemble_eval.py --tags tag1,tag2,tag3 --out_tag ensemble_top3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="Comma-separated list of run tags.")
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    out_root = Path(args.out_root)
    blobs = [torch.load(out_root / t / "predictions.pt", map_location="cpu", weights_only=False)
             for t in tags]

    # Verify shared split
    ref_test_idx = blobs[0]["test_idx"].tolist() if hasattr(blobs[0]["test_idx"], "tolist") else list(blobs[0]["test_idx"])
    for b, t in zip(blobs[1:], tags[1:]):
        idx = b["test_idx"].tolist() if hasattr(b["test_idx"], "tolist") else list(b["test_idx"])
        assert idx == ref_test_idx, f"Run {t} has a different test split"

    out_dir = out_root / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Average sigmoids on val/test, then convert back to logits
    def avg_split(split):
        cls_probs = torch.stack([b[split]["class_logits"].sigmoid() for b in blobs]).mean(0)
        con_probs = torch.stack([b[split]["concept_logits"].sigmoid() for b in blobs]).mean(0)
        eps = 1e-7
        cls_logits = torch.log(cls_probs.clamp(eps, 1-eps) / (1 - cls_probs).clamp(eps, 1-eps))
        con_logits = torch.log(con_probs.clamp(eps, 1-eps) / (1 - con_probs).clamp(eps, 1-eps))
        out = dict(class_logits=cls_logits, concept_logits=con_logits,
                    labels=blobs[0][split]["labels"],
                    concept_targets=blobs[0][split]["concept_targets"])
        if "attn" in blobs[0][split]:
            out["attn"] = blobs[0][split]["attn"]
        return out

    pkg = {
        "val": avg_split("val"),
        "test": avg_split("test"),
        "test_idx": blobs[0]["test_idx"],
        "val_idx": blobs[0]["val_idx"],
        "train_idx": blobs[0]["train_idx"],
        "concepts": blobs[0]["concepts"],
        "class_names": blobs[0]["class_names"],
        "is_baseline": False,
        "_ensemble_of": tags,
    }
    if "train" in blobs[0]:
        # Average train preds too (for completeness probe). All blobs must have train.
        if all("train" in b for b in blobs):
            cls_p = torch.stack([b["train"]["class_logits"].sigmoid() for b in blobs]).mean(0)
            con_p = torch.stack([b["train"]["concept_logits"].sigmoid() for b in blobs]).mean(0)
            eps = 1e-7
            pkg["train"] = dict(
                class_logits=torch.log(cls_p.clamp(eps, 1-eps) / (1-cls_p).clamp(eps, 1-eps)),
                concept_logits=torch.log(con_p.clamp(eps, 1-eps) / (1-con_p).clamp(eps, 1-eps)),
                labels=blobs[0]["train"]["labels"],
                concept_targets=blobs[0]["train"]["concept_targets"],
            )

    torch.save(pkg, out_dir / "predictions.pt")
    # Also a minimal summary.json compatible with summarize_runs.py
    summary = dict(
        tag=args.out_tag,
        args=dict(seed=blobs[0].get("seed", -1), baseline=False, lambda_constraint=None),
        test_classification={"weighted_f1": float("nan"), "macro_f1": float("nan"),
                              "micro_f1": float("nan"), "balanced_accuracy": float("nan"),
                              "mean_ap": float("nan")},
        test_concepts={"mean_concept_f1": float("nan"), "macro_concept_f1": float("nan")},
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ensemble] {out_dir}/predictions.pt averaged over {len(tags)} runs:")
    for t in tags:
        print(f"  - {t}")
    print(f"[ensemble] now run: python evaluate.py --run_dir {out_dir}")


if __name__ == "__main__":
    main()
