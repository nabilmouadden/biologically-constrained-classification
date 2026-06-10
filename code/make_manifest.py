#!/usr/bin/env python3
"""Build cell_manifest_full_extended.json from a GR-Neutro annotations.csv.

annotations.csv header (one-hot, exactly one class column = 1 per row):
    filename,path,Normal,Chromatin,Dohle,Hypergranulation,Hypersegmentation,Hypogranulation,Hyposegmentation

Output schema (cell_manifest_full_extended.json):
    {
      "scope": str,
      "n_total": int,
      "class_names": [7 class names, in the canonical order above],
      "cells": [
        {
          "idx": int,                     # 0-based row index
          "filename": str,                # csv "filename"
          "path": str,                    # csv "path"
          "label_one_hot": [7 ints],      # the 7 class columns, in canonical order
          "dominant_class_idx": int,      # argmax over the 7 one-hot columns
          "dominant_class_name": str      # class_names[dominant_class_idx]
        },
        ...
      ]
    }

Run:
    python code/make_manifest.py \
        --annotations data/gr_neutro/annotations.csv \
        --out cell_manifest_full_extended.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CLASS_NAMES = [
    "Normal", "Chromatin", "Dohle", "Hypergranulation",
    "Hypersegmentation", "Hypogranulation", "Hyposegmentation",
]


def build_manifest(annotations_csv, data_root=None, scope="gr_neutro"):
    cells = []
    with open(annotations_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CLASS_NAMES if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"annotations.csv missing class columns {missing}; "
                f"header was {reader.fieldnames}")
        for idx, row in enumerate(reader):
            one_hot = [int(round(float(row[c]))) for c in CLASS_NAMES]
            if sum(one_hot) != 1:
                raise ValueError(
                    f"row {idx} ({row.get('filename')}) is not one-hot: {one_hot}")
            dom = one_hot.index(1)
            cells.append({
                "idx": idx,
                "filename": row["filename"],
                "path": row["path"],
                "label_one_hot": one_hot,
                "dominant_class_idx": dom,
                "dominant_class_name": CLASS_NAMES[dom],
            })
    return {
        "scope": scope,
        "n_total": len(cells),
        "class_names": list(CLASS_NAMES),
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True,
                    help="path to GR-Neutro annotations.csv")
    ap.add_argument("--data_root", default=None,
                    help="optional image root (recorded as scope; not required)")
    ap.add_argument("--out", required=True,
                    help="output cell_manifest_full_extended.json path")
    ap.add_argument("--scope", default="gr_neutro",
                    help="manifest scope tag")
    args = ap.parse_args()

    man = build_manifest(args.annotations, args.data_root, scope=args.scope)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2))
    print(f"[make_manifest] wrote {man['n_total']} cells -> {out}")


if __name__ == "__main__":
    main()
