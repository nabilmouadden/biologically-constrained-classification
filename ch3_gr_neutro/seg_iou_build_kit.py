#!/usr/bin/env python3
"""
Build the MANUAL gold-standard annotation kit for true segmentation IoU.

For ~5-10 cells/class (~50 cells) we write, under manual_gt_kit/:
  cells/<CLASS>/<stem>.png                     original cell image
  cells/<CLASS>/<stem>__blank_nucleus.png      blank (black) canvas to paint
  cells/<CLASS>/<stem>__blank_cytoplasm.png    blank (black) canvas to paint
  cells/<CLASS>/<stem>__guide.png              faint image to trace over
  ours/<CLASS>/<stem>__ours.npz                OUR nucleus+wbc masks (for scorer)
  manifest.csv                                 cell list + class
  INSTRUCTIONS.md                              how a human annotates
  labelme_seed/<stem>.json                     labelme-ready skeleton (optional)

A human draws nucleus + cytoplasm masks (white-on-black PNG, or labelme/QuPath
polygons exported to PNG). Then `score_iou.py` computes per-class IoU of OUR
pipeline vs the human masks in ONE command. This is the definitive
ground-truth vehicle; Cellpose (kit A) is only an independent reference.

SLURM ONLY (it runs our segmentation).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from morphometry_concepts import segment_cell

CLASS_COLS = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
              "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]

INSTRUCTIONS = """# Manual gold-standard segmentation kit (GR-Neutro)

This kit produces the **definitive human ground-truth** segmentation masks so we
can compute the true IoU/Dice of the morphometry pipeline. (Cellpose, used
elsewhere, is only an independent *reference* — it is NOT human ground truth.)

## What you annotate
For every cell in `cells/<CLASS>/`, two masks of the **central white blood
cell** (the target neutrophil), ignoring other cells / red cells:

1. **Nucleus** — the dark purple/blue lobulated nuclear material only.
2. **Cytoplasm** — the pale cell body *around* the nucleus, out to the cell
   membrane. (Whole WBC = nucleus ∪ cytoplasm; you may instead paint the whole
   WBC and we derive cytoplasm = WBC − nucleus — see option B.)

## File naming (so the scorer finds your masks automatically)
Save each mask as an 8-bit PNG, **same width/height as the cell image**,
white (255) = inside the region, black (0) = outside. Save NEXT TO the cell:

    cells/<CLASS>/<stem>__gt_nucleus.png
    cells/<CLASS>/<stem>__gt_cytoplasm.png      (option A)
  OR
    cells/<CLASS>/<stem>__gt_wbc.png            (option B: whole-cell instead)

`__blank_nucleus.png` / `__blank_cytoplasm.png` are empty canvases you can
paint on (e.g. in GIMP/Photoshop: open the blank, paint the region white,
export with the `__gt_` name). `__guide.png` is a faint copy of the cell to
trace over. Use whichever tool you like (GIMP, QuPath, labelme); only the final
white-on-black PNG with the `__gt_` name matters.

### labelme users
`labelme_seed/<stem>.json` references the image. Draw polygons named exactly
`nucleus` and `cytoplasm` (or `wbc`), then run:
    labelme_json_to_png.py (provided)  OR export masks manually.

## When done
From `outputs/seg_iou/` run ONE command:

    python score_iou.py --kit-dir manual_gt_kit

It loads our saved masks (`ours/.../__ours.npz`) + your `__gt_*` PNGs and
prints/writes per-class nucleus & cytoplasm & WBC IoU/Dice
(`manual_gt_kit/manual_iou_results.{json,md}`). Cells without human masks are
skipped and listed as pending.
"""


def make_blank(shape):
    return Image.fromarray(np.zeros(shape[:2], dtype=np.uint8))


def make_guide(rgb_u8):
    # faint, brightened image to trace over
    g = (rgb_u8.astype(np.float32) * 0.55 + 255 * 0.45).clip(0, 255)
    return Image.fromarray(g.astype(np.uint8))


def labelme_seed(stem, fname, h, w):
    return {
        "version": "5.0.1", "flags": {},
        "shapes": [
            {"label": "nucleus", "points": [], "group_id": None,
             "shape_type": "polygon", "flags": {}},
            {"label": "cytoplasm", "points": [], "group_id": None,
             "shape_type": "polygon", "flags": {}},
        ],
        "imagePath": fname, "imageData": None,
        "imageHeight": int(h), "imageWidth": int(w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="manual_gt_kit dir")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "labelme_seed").mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    ann = pd.read_csv(data_dir / "annotations.csv").reset_index(drop=True)
    rng = np.random.RandomState(args.seed)

    manifest = []
    for cls in CLASS_COLS:
        pos = ann[ann[cls] == 1]
        # prefer single-label cells for clean ground truth, top up if needed
        n_lab = ann[CLASS_COLS].sum(axis=1)
        single = pos[n_lab.loc[pos.index] == 1]
        pool = single if len(single) >= args.per_class else pos
        take = rng.choice(pool.index, min(args.per_class, len(pool)),
                          replace=False)
        cdir = out / "cells" / cls
        odir = out / "ours" / cls
        cdir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        for i in take:
            r = ann.loc[i]
            path = r["path"]; stem = Path(path).stem
            try:
                pil = Image.open(path).convert("RGB")
                rgb_u8 = np.asarray(pil)
                rgb = rgb_u8.astype(np.float64) / 255.0
            except Exception as e:
                print("skip load", path, e, flush=True)
                continue
            h, w = rgb_u8.shape[:2]
            # save originals + blanks + guide
            pil.save(cdir / f"{stem}.png")
            make_blank(rgb_u8.shape).save(cdir / f"{stem}__blank_nucleus.png")
            make_blank(rgb_u8.shape).save(cdir / f"{stem}__blank_cytoplasm.png")
            make_guide(rgb_u8).save(cdir / f"{stem}__guide.png")
            # our masks for the scorer
            seg = segment_cell(rgb)
            if seg.get("ok", False):
                np.savez_compressed(
                    odir / f"{stem}__ours.npz",
                    nuc=seg["nuc"].astype(np.uint8),
                    wbc=seg["wbc"].astype(np.uint8),
                    cyto=seg["cyto"].astype(np.uint8))
                our_ok = 1
            else:
                np.savez_compressed(
                    odir / f"{stem}__ours.npz",
                    nuc=np.zeros((h, w), np.uint8),
                    wbc=np.zeros((h, w), np.uint8),
                    cyto=np.zeros((h, w), np.uint8))
                our_ok = 0
            with open(out / "labelme_seed" / f"{stem}.json", "w") as f:
                json.dump(labelme_seed(stem, f"../cells/{cls}/{stem}.png",
                                       h, w), f, indent=2)
            manifest.append({"class": cls, "stem": stem,
                             "cell_png": str((cdir / f"{stem}.png")
                                             .relative_to(out)),
                             "ours_npz": str((odir / f"{stem}__ours.npz")
                                             .relative_to(out)),
                             "our_seg_ok": our_ok,
                             "gt_nucleus_png": f"cells/{cls}/{stem}__gt_nucleus.png",
                             "gt_cytoplasm_png": f"cells/{cls}/{stem}__gt_cytoplasm.png",
                             "gt_wbc_png": f"cells/{cls}/{stem}__gt_wbc.png"})

    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    with open(out / "INSTRUCTIONS.md", "w") as f:
        f.write(INSTRUCTIONS)
    print(f"Kit built: {len(manifest)} cells across {len(CLASS_COLS)} classes "
          f"in {out}", flush=True)


if __name__ == "__main__":
    main()
