#!/usr/bin/env python3
"""RICH interpretable morphometric feature bank for GR-Neutro.

Goal: close the accuracy gap between the 11-concept interpretable classifier
(W-F1 0.66) and the black-box DinoBloom-B backbone (W-F1 0.83) WITHOUT adding a
black box. Every feature here is a NAMED measurement a cytologist understands —
no learned embeddings.

Reuses the segmentation (nucleus / cytoplasm / WBC masks) from
`morphometry_concepts.segment_cell` and `count_lobes`, then computes ~70 named
morphometric features per cell across four families:

  * Nuclear shape / size  : area, perimeter, eccentricity, solidity, convexity,
    circularity / form-factor, major/minor axis, axis ratio, extent, lobe count,
    lobe-size mean/std/min/max, inter-lobe bridge thinness (min EDT on skeleton),
    #connected components, #components after bridge-erosion.
  * Chromatin / nuclear texture : GLCM (contrast, dissimilarity, homogeneity,
    energy, correlation, ASM) at d in {1,3} averaged over 4 angles; LBP histogram
    (10 uniform bins); intensity mean/std/skew/kurtosis; hetero/euchromatin
    fraction (dark/light fraction relative to nuclear median).
  * Cytoplasm : area, N:C ratio, hematoxylin/eosin deconvolved channel mean/std,
    basophilia, granule count + size mean/std/p90 + density + spatial dispersion,
    cytoplasm GLCM (contrast/homogeneity/energy/correlation) + LBP (10 bins),
    vacuole count/area-fraction, Dohle-like inclusion count/area-fraction.
  * Whole cell : total WBC area, cell circularity, cell eccentricity, cell
    solidity.

Output: a per-cell feature matrix (CSV) for all cells, one row per cell, with the
7 one-hot class columns + dominant_class_idx carried through, plus seg_ok flag.

Run on SLURM only (never the Ruche login node).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scipy import ndimage as ndi
from scipy.stats import skew, kurtosis
from skimage.color import rgb2hed
from skimage.feature import (blob_log, graycomatrix, graycoprops,
                             local_binary_pattern)
from skimage.measure import label, regionprops
from skimage.morphology import (remove_small_holes, remove_small_objects,
                                binary_erosion, disk, skeletonize)
from scipy.ndimage import distance_transform_edt

# segmentation + lobe counting reused verbatim from the 11-concept module
from morphometry_concepts import segment_cell, count_lobes, _safe01

warnings.filterwarnings("ignore")

CLASS_COLS = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
              "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]

LBP_P = 8
LBP_R = 1.0
LBP_BINS = LBP_P + 2  # uniform method -> P+2 bins


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _glcm_feats(patch_u8, mask, prefix, levels=32):
    """GLCM texture on a uint8 patch, only meaningful inside `mask`. Returns
    dict of named features averaged over angles, for distances 1 and 3."""
    out = {}
    pq = patch_u8.copy()
    pq[~mask] = 0
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for d in (1, 3):
        try:
            glcm = graycomatrix(pq, distances=[d], angles=angles,
                                levels=levels, symmetric=True, normed=True)
            for prop in ("contrast", "dissimilarity", "homogeneity",
                         "energy", "correlation", "ASM"):
                out[f"{prefix}_glcm_{prop}_d{d}"] = float(
                    graycoprops(glcm, prop).mean())
        except Exception:
            for prop in ("contrast", "dissimilarity", "homogeneity",
                         "energy", "correlation", "ASM"):
                out[f"{prefix}_glcm_{prop}_d{d}"] = 0.0
    return out


def _lbp_hist(gray_patch, mask, prefix):
    """Uniform LBP histogram (P+2 bins) over masked pixels."""
    out = {}
    try:
        lbp = local_binary_pattern(gray_patch, LBP_P, LBP_R, method="uniform")
        vals = lbp[mask]
        if vals.size > 0:
            hist, _ = np.histogram(vals, bins=np.arange(LBP_BINS + 1),
                                   density=True)
        else:
            hist = np.zeros(LBP_BINS)
    except Exception:
        hist = np.zeros(LBP_BINS)
    for b in range(LBP_BINS):
        out[f"{prefix}_lbp_b{b}"] = float(hist[b]) if b < len(hist) else 0.0
    return out


def _to_u8(arr, mask):
    """Robustly scale arr -> uint8 [0,31] using 2/98 pct of masked values."""
    if mask.sum() == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[mask], [2, 98])
    q = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return (q * 31).astype(np.uint8)


# --------------------------------------------------------------------------- #
# rich feature computation
# --------------------------------------------------------------------------- #
def compute_rich_features(rgb, seg):
    """Return an ordered dict of ~70 NAMED morphometric features."""
    nuc = seg["nuc"]; cyto = seg["cyto"]; wbc = seg["wbc"]
    h_chan = seg["h_chan"]; lab = seg["lab"]; hed = seg["hed"]; sat = seg["sat"]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    H, W = gray.shape
    f = {}

    # ===================================================================== #
    # 1. NUCLEAR SHAPE / SIZE
    # ===================================================================== #
    nuc_f = remove_small_holes(nuc, area_threshold=64)
    nuc_lbl = label(nuc_f)
    rps = regionprops(nuc_lbl)
    if rps:
        big = max(rps, key=lambda r: r.area)
        tot_area = float(sum(r.area for r in rps))
        tot_perim = float(sum(r.perimeter for r in rps))
        f["nuc_area"] = tot_area
        f["nuc_perimeter"] = tot_perim
        f["nuc_form_factor"] = (tot_perim ** 2) / (4 * np.pi * tot_area + 1e-9)
        f["nuc_circularity"] = (4 * np.pi * tot_area) / (tot_perim ** 2 + 1e-9)
        f["nuc_eccentricity"] = float(big.eccentricity)
        f["nuc_solidity"] = float(big.solidity)
        f["nuc_extent"] = float(big.extent)
        f["nuc_major_axis"] = float(big.major_axis_length)
        f["nuc_minor_axis"] = float(big.minor_axis_length)
        f["nuc_axis_ratio"] = float(big.major_axis_length /
                                    (big.minor_axis_length + 1e-9))
        # convexity = convex-hull perimeter / actual perimeter (<=1, low=spiky)
        try:
            hull_perim = float(regionprops(big.convex_image.astype(int))[0].perimeter)
            f["nuc_convexity"] = hull_perim / (big.perimeter + 1e-9)
        except Exception:
            f["nuc_convexity"] = 1.0
        f["nuc_equiv_diameter"] = float(big.equivalent_diameter)
    else:
        for k in ("nuc_area", "nuc_perimeter", "nuc_form_factor", "nuc_circularity",
                  "nuc_eccentricity", "nuc_solidity", "nuc_extent", "nuc_major_axis",
                  "nuc_minor_axis", "nuc_axis_ratio", "nuc_convexity",
                  "nuc_equiv_diameter"):
            f[k] = 0.0

    # lobes
    n_lobes, cc = count_lobes(nuc)
    f["lobe_count"] = float(n_lobes)
    f["nuc_components"] = float(cc)
    # bridge-erosion: erode then recount connected components (lobes split at
    # thin chromatin bridges). #components after erosion ~ lobe count.
    er = binary_erosion(nuc_f, disk(2))
    er = remove_small_objects(er, min_size=30)
    f["nuc_components_after_erosion"] = float(label(er).max())
    # lobe-size distribution from watershed basins
    dist = distance_transform_edt(nuc_f)
    lobe_sizes = []
    bridge_thin = 0.0
    if dist.max() >= 1:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from skimage.filters import gaussian as _g
        dist_s = _g(dist, sigma=1.0)
        md = max(3, int(dist_s.max() * 0.55))
        coords = peak_local_max(dist_s, min_distance=md,
                                threshold_abs=max(1.5, dist_s.max() * 0.30),
                                labels=nuc_f)
        if len(coords):
            markers = np.zeros(nuc_f.shape, dtype=int)
            for i, (r, c) in enumerate(coords, 1):
                markers[r, c] = i
            ws = watershed(-dist_s, markers, mask=nuc_f)
            areas = np.bincount(ws.ravel())[1:]
            lobe_sizes = [a for a in areas if a >= 40]
        # inter-lobe bridge thinness: minimum EDT along the nuclear skeleton
        # (thin connecting bridges = small EDT). Lower = thinner bridges.
        try:
            sk = skeletonize(nuc_f)
            if sk.sum() > 0:
                bridge_thin = float(dist[sk].min())
        except Exception:
            bridge_thin = 0.0
    if lobe_sizes:
        ls = np.array(lobe_sizes, dtype=float)
        f["lobe_size_mean"] = float(ls.mean())
        f["lobe_size_std"] = float(ls.std())
        f["lobe_size_min"] = float(ls.min())
        f["lobe_size_max"] = float(ls.max())
        f["lobe_size_cv"] = float(ls.std() / (ls.mean() + 1e-9))
    else:
        f["lobe_size_mean"] = f["nuc_area"]
        f["lobe_size_std"] = 0.0
        f["lobe_size_min"] = f["nuc_area"]
        f["lobe_size_max"] = f["nuc_area"]
        f["lobe_size_cv"] = 0.0
    f["interlobe_bridge_thinness"] = bridge_thin

    # ===================================================================== #
    # 2. CHROMATIN / NUCLEAR TEXTURE
    # ===================================================================== #
    if nuc.sum() > 30:
        nvals = h_chan[nuc]
        f["chromatin_mean_H"] = float(np.mean(nvals))
        f["chromatin_std_H"] = float(np.std(nvals))
        f["chromatin_skew_H"] = float(skew(nvals)) if nvals.size > 2 else 0.0
        f["chromatin_kurt_H"] = float(kurtosis(nvals)) if nvals.size > 2 else 0.0
        med = np.median(nvals)
        f["chromatin_hetero_frac"] = float((nvals > med * 1.15).mean())  # dark/dense
        f["chromatin_euchro_frac"] = float((nvals < med * 0.85).mean())  # light/open
        gvals = gray[nuc]
        f["nuc_gray_mean"] = float(np.mean(gvals))
        f["nuc_gray_std"] = float(np.std(gvals))
        # GLCM + LBP on H channel inside nuclear bbox
        ys, xs = np.where(nuc)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = h_chan[y0:y1, x0:x1]
        m = nuc[y0:y1, x0:x1]
        f.update(_glcm_feats(_to_u8(patch, m), m, "chromatin"))
        f.update(_lbp_hist(gray[y0:y1, x0:x1], m, "chromatin"))
    else:
        for k in ("chromatin_mean_H", "chromatin_std_H", "chromatin_skew_H",
                  "chromatin_kurt_H", "chromatin_hetero_frac",
                  "chromatin_euchro_frac", "nuc_gray_mean", "nuc_gray_std"):
            f[k] = 0.0
        f.update(_glcm_feats(np.zeros((4, 4), np.uint8),
                             np.ones((4, 4), bool), "chromatin"))
        f.update(_lbp_hist(np.zeros((4, 4)), np.ones((4, 4), bool), "chromatin"))

    # ===================================================================== #
    # 3. CYTOPLASM
    # ===================================================================== #
    nc = seg["nuc_area"] / (seg["cyto_area"] + 1e-9)
    f["nc_ratio"] = float(nc)
    f["cyto_area"] = float(seg["cyto_area"])
    if cyto.sum() > 30:
        e_chan = hed[:, :, 1]
        Lc = lab[:, :, 0] / 100.0
        a_star = lab[:, :, 1]
        b_star = lab[:, :, 2]
        # deconvolved stain channel stats within cytoplasm
        f["cyto_H_mean"] = float(np.mean(h_chan[cyto]))
        f["cyto_H_std"] = float(np.std(h_chan[cyto]))
        f["cyto_E_mean"] = float(np.mean(e_chan[cyto]))
        f["cyto_E_std"] = float(np.std(e_chan[cyto]))
        f["cyto_L_mean"] = float(np.mean(Lc[cyto]))
        f["cyto_a_mean"] = float(np.mean(a_star[cyto]))
        f["cyto_b_mean"] = float(np.mean(b_star[cyto]))
        f["cyto_sat_mean"] = float(np.mean(sat[cyto]))
        # basophilia (blue/purple cast): high H + low (negative) b*
        f["cyto_basophilia"] = float(np.clip(
            0.5 * _norm(f["cyto_H_mean"], 0.02, 0.18) +
            0.5 * _norm(-f["cyto_b_mean"], -25.0, 5.0), 0, 1))
        f["cyto_gray_var"] = float(np.var(gray[cyto]))

        # ---- granules: blob detection on bright/eosin signal ----
        ys, xs = np.where(cyto)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cmask = cyto[y0:y1, x0:x1]
        gsig = 1.0 - gray[y0:y1, x0:x1]
        esig = e_chan[y0:y1, x0:x1]
        sig = 0.5 * _safe01(gsig) + 0.5 * _safe01(esig)
        sig_m = np.where(cmask, sig, 0)
        try:
            blobs = blob_log(sig_m, min_sigma=1, max_sigma=4, num_sigma=5,
                             threshold=0.08)
        except Exception:
            blobs = np.empty((0, 3))
        kept = []
        for b in blobs:
            r, c = int(b[0]), int(b[1])
            if 0 <= r < cmask.shape[0] and 0 <= c < cmask.shape[1] and cmask[r, c]:
                kept.append(b)
        kept = np.array(kept) if kept else np.empty((0, 3))
        n_gran = len(kept)
        f["granule_count"] = float(n_gran)
        f["granule_density_per_kpx"] = float(n_gran / (seg["cyto_area"] / 1000.0 + 1e-9))
        if n_gran > 0:
            sig_sizes = kept[:, 2]
            f["granule_size_mean"] = float(sig_sizes.mean())
            f["granule_size_std"] = float(sig_sizes.std())
            f["granule_size_p90"] = float(np.percentile(sig_sizes, 90))
            # spatial dispersion: std of granule positions / cyto equiv radius
            pts = kept[:, :2]
            cen = pts.mean(axis=0)
            disp = np.sqrt(((pts - cen) ** 2).sum(axis=1)).mean()
            cyto_radius = np.sqrt(seg["cyto_area"] / np.pi) + 1e-9
            f["granule_dispersion"] = float(disp / cyto_radius)
        else:
            f["granule_size_mean"] = 0.0
            f["granule_size_std"] = 0.0
            f["granule_size_p90"] = 0.0
            f["granule_dispersion"] = 0.0

        # ---- cytoplasm texture GLCM + LBP ----
        cy_patch_e = e_chan[y0:y1, x0:x1]
        f.update(_glcm_feats(_to_u8(cy_patch_e, cmask), cmask, "cyto"))
        f.update(_lbp_hist(gray[y0:y1, x0:x1], cmask, "cyto"))

        # ---- Dohle-like inclusions: locally bluer + paler than bulk cyto ----
        b_cy = b_star[cyto]; L_cy = Lc[cyto]
        b_med, b_sd = float(np.median(b_cy)), float(np.std(b_cy) + 1e-6)
        L_med, L_sd = float(np.median(L_cy)), float(np.std(L_cy) + 1e-6)
        incl = (b_star < b_med - 1.2 * b_sd) & (Lc > L_med) & (sat < 0.5) & cyto
        incl = remove_small_objects(incl, min_size=15)
        f["inclusion_area_frac"] = float(incl.sum() / (seg["cyto_area"] + 1e-9))
        f["inclusion_count"] = float(label(incl).max())

        # ---- vacuoles: bright low-chroma holes above cyto L median ----
        vac = (Lc > L_med + 1.6 * L_sd) & (sat < 0.20) & cyto
        vac = remove_small_objects(vac, min_size=10)
        f["vacuole_area_frac"] = float(vac.sum() / (seg["cyto_area"] + 1e-9))
        f["vacuole_count"] = float(label(vac).max())
    else:
        for k in ("cyto_H_mean", "cyto_H_std", "cyto_E_mean", "cyto_E_std",
                  "cyto_L_mean", "cyto_a_mean", "cyto_b_mean", "cyto_sat_mean",
                  "cyto_basophilia", "cyto_gray_var", "granule_count",
                  "granule_density_per_kpx", "granule_size_mean", "granule_size_std",
                  "granule_size_p90", "granule_dispersion", "inclusion_area_frac",
                  "inclusion_count", "vacuole_area_frac", "vacuole_count"):
            f[k] = 0.0
        f.update(_glcm_feats(np.zeros((4, 4), np.uint8),
                             np.ones((4, 4), bool), "cyto"))
        f.update(_lbp_hist(np.zeros((4, 4)), np.ones((4, 4), bool), "cyto"))

    # ===================================================================== #
    # 4. WHOLE CELL
    # ===================================================================== #
    wbc_lbl = label(remove_small_holes(wbc, area_threshold=256))
    wrps = regionprops(wbc_lbl)
    if wrps:
        wbig = max(wrps, key=lambda r: r.area)
        f["cell_area"] = float(wbig.area)
        f["cell_circularity"] = (4 * np.pi * wbig.area) / (wbig.perimeter ** 2 + 1e-9)
        f["cell_eccentricity"] = float(wbig.eccentricity)
        f["cell_solidity"] = float(wbig.solidity)
    else:
        f["cell_area"] = float(seg["wbc_area"])
        f["cell_circularity"] = 0.0
        f["cell_eccentricity"] = 0.0
        f["cell_solidity"] = 0.0

    return f


def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir containing annotations.csv (paths resolved from CSV)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all cells")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(data_dir / "annotations.csv")
    if args.limit:
        ann = ann.groupby(
            ann[CLASS_COLS].idxmax(axis=1), group_keys=False
        ).apply(lambda g: g.head(max(1, args.limit // 7)))
    print(f"Processing {len(ann)} cells", flush=True)

    rows = []
    feat_names = None
    n_fail = 0
    fail_reasons = {}
    for i, r in enumerate(ann.itertuples(index=False)):
        path = getattr(r, "path")
        fname = getattr(r, "filename")
        labels = {c: int(getattr(r, c)) for c in CLASS_COLS}
        dom = int(np.argmax([labels[c] for c in CLASS_COLS]))
        try:
            rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
        except Exception:
            n_fail += 1
            fail_reasons["load_error"] = fail_reasons.get("load_error", 0) + 1
            continue

        seg = segment_cell(rgb)
        base = {"filename": fname, "path": path, **labels,
                "dominant_class_idx": dom}
        if not seg.get("ok", False):
            n_fail += 1
            reason = seg.get("reason", "unknown")
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            base["seg_ok"] = 0
            base["seg_reason"] = reason
            if feat_names:
                for k in feat_names:
                    base[k] = np.nan
            rows.append(base)
            continue

        feats = compute_rich_features(rgb, seg)
        if feat_names is None:
            feat_names = list(feats.keys())
            print(f"[features] {len(feat_names)} named features:\n  "
                  + ", ".join(feat_names), flush=True)
        base["seg_ok"] = 1
        base["seg_reason"] = ""
        base.update(feats)
        rows.append(base)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ann)} done, {n_fail} seg-fail", flush=True)

    df = pd.DataFrame(rows)
    # ensure all feature columns exist even for seg-fail rows
    if feat_names:
        for k in feat_names:
            if k not in df.columns:
                df[k] = np.nan
    df.to_csv(out_dir / "features.csv", index=False)
    meta = {
        "n_cells": int(len(df)),
        "n_seg_ok": int((df["seg_ok"] == 1).sum()),
        "seg_fail": int(n_fail),
        "seg_fail_rate": round(n_fail / max(1, len(df)), 4),
        "seg_fail_reasons": fail_reasons,
        "n_features": len(feat_names) if feat_names else 0,
        "feature_names": feat_names or [],
        "class_cols": CLASS_COLS,
    }
    with open(out_dir / "features_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {len(df)} rows, {meta['n_features']} features. "
          f"seg-fail={n_fail} ({100*n_fail/max(1,len(df)):.1f}%) "
          f"reasons={fail_reasons}", flush=True)


if __name__ == "__main__":
    main()
