#!/usr/bin/env python3
"""
score_iou.py — ONE-command true IoU once human masks come back.

Computes per-class IoU/Dice of OUR pipeline vs HUMAN gold-standard masks for the
manual_gt_kit. Our masks are read from ours/<CLASS>/<stem>__ours.npz (saved when
the kit was built); human masks from cells/<CLASS>/<stem>__gt_{nucleus,
cytoplasm,wbc}.png (white=region). Supports two annotation options:
  A) __gt_nucleus.png + __gt_cytoplasm.png  -> wbc_gt = nucleus ∪ cytoplasm
  B) __gt_nucleus.png + __gt_wbc.png        -> cyto_gt = wbc − nucleus

Cells missing human masks are skipped and reported as pending. This is the
DEFINITIVE ground-truth number (human GT), distinct from the Cellpose reference.

Run on SLURM (cpu_short) per the project compute rule, even though it is light:
    sbatch seg_iou_score_kit.sbatch     # or include in the eval job
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_COLS = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
              "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]


def iou_dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    s = a.sum() + b.sum()
    iou = float(inter / union) if union > 0 else np.nan
    dice = float(2 * inter / s) if s > 0 else np.nan
    return iou, dice


def load_mask(p: Path):
    if not p.exists():
        return None
    m = np.asarray(Image.open(p).convert("L"))
    return m > 127


def summarize(xs):
    v = np.array([x for x in xs if x is not None and not np.isnan(x)], float)
    if len(v) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "median": round(float(np.median(v)), 4), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-dir", required=True)
    args = ap.parse_args()
    kit = Path(args.kit_dir)
    man = pd.read_csv(kit / "manifest.csv")

    rows, pending = [], []
    for _, r in man.iterrows():
        cls, stem = r["class"], r["stem"]
        ours = np.load(kit / r["ours_npz"])
        our_nuc = ours["nuc"].astype(bool)
        our_wbc = ours["wbc"].astype(bool)
        our_cyto = ours["cyto"].astype(bool)

        gt_nuc = load_mask(kit / "cells" / cls / f"{stem}__gt_nucleus.png")
        gt_cyto = load_mask(kit / "cells" / cls / f"{stem}__gt_cytoplasm.png")
        gt_wbc = load_mask(kit / "cells" / cls / f"{stem}__gt_wbc.png")

        if gt_nuc is None and gt_wbc is None and gt_cyto is None:
            pending.append({"class": cls, "stem": stem})
            continue
        # derive whichever is missing
        if gt_wbc is None and gt_nuc is not None and gt_cyto is not None:
            gt_wbc = np.logical_or(gt_nuc, gt_cyto)
        if gt_cyto is None and gt_wbc is not None and gt_nuc is not None:
            gt_cyto = np.logical_and(gt_wbc, ~gt_nuc)

        rec = {"class": cls, "stem": stem, "our_seg_ok": int(r["our_seg_ok"])}
        if gt_nuc is not None:
            rec["nuc_iou"], rec["nuc_dice"] = iou_dice(our_nuc, gt_nuc)
        if gt_cyto is not None:
            rec["cyto_iou"], rec["cyto_dice"] = iou_dice(our_cyto, gt_cyto)
        if gt_wbc is not None:
            rec["wbc_iou"], rec["wbc_dice"] = iou_dice(our_wbc, gt_wbc)
        rows.append(rec)

    if not rows:
        print(f"No human masks found yet. {len(pending)} cells pending. "
              f"Annotate per INSTRUCTIONS.md then re-run.", flush=True)
        out = {"status": "pending", "n_pending": len(pending),
               "n_scored": 0, "pending": pending}
        with open(kit / "manual_iou_results.json", "w") as f:
            json.dump(out, f, indent=2)
        return

    df = pd.DataFrame(rows)
    df.to_csv(kit / "manual_per_cell_iou.csv", index=False)

    per_class = {}
    for cls in CLASS_COLS:
        sub = df[df["class"] == cls]
        if not len(sub):
            continue
        d = {"n_scored": int(len(sub))}
        for k in ["nuc", "cyto", "wbc"]:
            col = f"{k}_iou"
            if col in sub:
                d[f"{k}_iou"] = summarize(sub[col].tolist())
                d[f"{k}_dice"] = summarize(sub[f"{k}_dice"].tolist())
        per_class[cls] = d

    overall = {}
    for k in ["nuc", "cyto", "wbc"]:
        col = f"{k}_iou"
        if col in df:
            overall[f"{k}_iou"] = summarize(df[col].tolist())
            overall[f"{k}_dice"] = summarize(df[f"{k}_dice"].tolist())

    out = {"status": "scored",
           "note": ("HUMAN ground-truth IoU (definitive). Distinct from the "
                    "Cellpose reference in iou_results.json."),
           "n_scored": len(df), "n_pending": len(pending),
           "overall": overall, "per_class": per_class, "pending": pending}
    with open(kit / "manual_iou_results.json", "w") as f:
        json.dump(out, f, indent=2)

    md = ["# Manual gold-standard IoU (ours vs HUMAN masks)\n",
          f"- Scored {len(df)} cells; {len(pending)} pending annotation.",
          f"- **Overall** "
          + ", ".join(f"{k} IoU {overall.get(k+'_iou',{}).get('mean')}"
                      for k in ["nuc", "cyto", "wbc"] if k+"_iou" in overall)
          + "\n",
          "| class | n | nucleus IoU | cytoplasm IoU | WBC IoU |",
          "|---|---|---|---|---|"]
    for cls, d in per_class.items():
        def g(k):
            s = d.get(f"{k}_iou")
            return f"{s['mean']:.3f}±{s['std']:.3f}" if s and s["mean"] is not None else "—"
        md.append(f"| {cls} | {d['n_scored']} | {g('nuc')} | "
                  f"{g('cyto')} | {g('wbc')} |")
    with open(kit / "manual_iou_results.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("Scored. overall:", json.dumps(overall), flush=True)


if __name__ == "__main__":
    main()
