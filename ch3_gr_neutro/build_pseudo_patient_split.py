"""Cluster GR-Neutro cells into pseudo-patients via Ward-linkage hierarchical
clustering on DinoBloom features, then emit an 80/10/10 train/val/test split
keyed by pseudo-patient so no cluster appears in two splits.

Addresses R4: "no patient-level split; ~5-10 pp inflation likely". This is
a proxy for the patient-level split that the Acevedo source release does not
support (no patient IDs).

Inputs (Ruche paths):
    --features  /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/dinobloom_features.npz
                NPZ with keys: 'paths' (basenames or paths), 'features' (N, D)
    --annotations  /gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv
                CSV header: filename,path,<class1>,<class2>,...

Outputs:
    --out_csv     Pseudo-patient split CSV: filename,split,pseudo_patient
    --out_meta    JSON sidecar with cluster size stats and per-split counts
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


def load_features(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    if "features" not in keys:
        raise KeyError(f"expected 'features' in {npz_path}, got {keys}")
    feats = np.asarray(d["features"]).astype(np.float32)
    # Path/identifier key: 'paths', 'filenames', or 'names'
    for k in ("paths", "filenames", "names", "ids"):
        if k in keys:
            paths = [str(x) for x in d[k]]
            break
    else:
        raise KeyError(f"no id key (paths/filenames/names/ids) in {npz_path}, got {keys}")
    return feats, paths


def load_annotation_basenames(csv_path: str):
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    basenames = [row[0] for row in rows]
    class_names = header[2:]
    labels = np.array([[int(x) for x in row[2:]] for row in rows], dtype=int)
    return basenames, class_names, labels


def align_features_to_annotations(feat_paths, feat_matrix, annotation_basenames):
    """Map feature rows (keyed by file basename) to annotation row order. Cells
    with no annotation row are dropped; cells in annotations but missing from
    features raise."""
    feat_basename_to_idx = {}
    for i, p in enumerate(feat_paths):
        bn = Path(p).name
        feat_basename_to_idx[bn] = i
    aligned_feat_idx = []
    missing = []
    for bn in annotation_basenames:
        i = feat_basename_to_idx.get(bn)
        if i is None:
            missing.append(bn)
        else:
            aligned_feat_idx.append(i)
    if missing:
        raise RuntimeError(
            f"{len(missing)} annotation basenames have no feature row; "
            f"example: {missing[:3]}"
        )
    return feat_matrix[aligned_feat_idx]


def ward_cluster(feats: np.ndarray, n_clusters: int = 50) -> np.ndarray:
    """Ward-linkage hierarchical clustering. Returns cluster id per row."""
    # Standardise per-feature before Ward (Ward assumes Euclidean and is sensitive
    # to scale; DinoBloom outputs are already roughly unit-normed but we
    # z-score to be safe).
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats_z = (feats - mean) / std

    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(feats_z)
    return labels


def pseudo_patient_split(cluster_ids: np.ndarray, seed: int = 42,
                          train_frac: float = 0.80, val_frac: float = 0.10):
    """Assign each cluster to train/val/test. NO cell from a cluster appears
    in two splits. Frac is over clusters (so cell fractions only approximate
    train_frac if cluster sizes are heterogeneous)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster_ids)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # remainder -> test
    train_clusters = set(uniq[:n_train].tolist())
    val_clusters = set(uniq[n_train:n_train + n_val].tolist())
    test_clusters = set(uniq[n_train + n_val:].tolist())
    splits = np.empty(len(cluster_ids), dtype=object)
    for i, c in enumerate(cluster_ids):
        if c in train_clusters:
            splits[i] = "train"
        elif c in val_clusters:
            splits[i] = "val"
        else:
            splits[i] = "test"
    return splits, {"train": sorted(train_clusters), "val": sorted(val_clusters),
                     "test": sorted(test_clusters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--n_clusters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    print(f"[load] features={args.features}")
    feats, feat_paths = load_features(args.features)
    print(f"[load] N_feat={len(feat_paths)} D={feats.shape[1]}")

    print(f"[load] annotations={args.annotations}")
    annot_basenames, class_names, labels = load_annotation_basenames(args.annotations)
    print(f"[load] N_ann={len(annot_basenames)} classes={class_names}")

    aligned = align_features_to_annotations(feat_paths, feats, annot_basenames)
    print(f"[align] aligned shape={aligned.shape}")

    print(f"[cluster] Ward linkage, K={args.n_clusters}")
    cluster_ids = ward_cluster(aligned, n_clusters=args.n_clusters)
    sizes = Counter(cluster_ids.tolist())
    size_vals = np.array(list(sizes.values()))
    print(f"[cluster] mean={size_vals.mean():.1f} median={np.median(size_vals):.0f} "
          f"min={size_vals.min()} max={size_vals.max()} std={size_vals.std():.1f}")

    splits, cluster_assignment = pseudo_patient_split(cluster_ids, seed=args.seed)
    split_counts = Counter(splits.tolist())
    print(f"[split] {dict(split_counts)} (total {len(splits)})")

    # Per-class counts per split
    per_split_per_class = defaultdict(lambda: np.zeros(len(class_names), dtype=int))
    for i, s in enumerate(splits):
        per_split_per_class[s] += labels[i]
    for s in ("train", "val", "test"):
        print(f"[per-class {s}] " + ", ".join(
            f"{c}={int(per_split_per_class[s][k])}" for k, c in enumerate(class_names)))

    # Write split CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "pseudo_patient"])
        for bn, s, c in zip(annot_basenames, splits, cluster_ids):
            w.writerow([bn, s, int(c)])
    print(f"[write] {out_csv}")

    meta = {
        "n_cells": int(len(annot_basenames)),
        "n_clusters": int(args.n_clusters),
        "seed": int(args.seed),
        "cluster_size_mean": float(size_vals.mean()),
        "cluster_size_median": float(np.median(size_vals)),
        "cluster_size_min": int(size_vals.min()),
        "cluster_size_max": int(size_vals.max()),
        "cluster_size_std": float(size_vals.std()),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "per_split_per_class": {s: per_split_per_class[s].tolist()
                                  for s in ("train", "val", "test")},
        "class_names": class_names,
        "n_clusters_per_split": {k: len(v) for k, v in cluster_assignment.items()},
        "features_npz": args.features,
        "annotations_csv": args.annotations,
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[write] {args.out_meta}")


if __name__ == "__main__":
    main()
