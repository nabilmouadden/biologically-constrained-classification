"""Run the trained joint GR-Neutro model on Bodzas-2023 WBC (external cohort).

Adapted from evaluate_on_mll23.py. Bodzas et al. 2023 (Scientific Data 10:466,
figshare DOI 10.6084/m9.figshare.c.6612970) — peripheral blood smears,
9 cell classes, 16k+ annotated cells, single zip ~46 GB extracted.

Pipeline:
  1. Walk Bodzas dataset (either single bodzas_wbc.zip or a directory of
     per-class zips). For each cell image, read bytes directly (no extraction).
  2. Apply the eval transform identical to train.py's build_eval_transform().
  3. Forward through JointModel; record sigmoid(class_logits) (7-dim) and
     sigmoid(concept_logits) (11-dim).
  4. Compute pre-CDA and post-CDA weighted F1 (Normal vs not-Normal binary
     restricted to evaluable cells; ALL-cell binary OOV; per-class F1 by
     source class).
  5. CDA = Concept Distribution Alignment: estimate per-concept mean+cov
     on a held-out 20% slice of Bodzas, fit a linear Mahalanobis-whitened
     mapping back onto the GR-Neutro train concept distribution, re-run
     the prototype (R-row) classifier in the aligned space.
  6. Save raw per-cell predictions + concept scores to NPZ + CSV.

CLI:
    python evaluate_on_bodzas.py \\
        --model  /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/B_kitchen_s42/model.pt \\
        --bodzas /gpfs/workdir/mouaddenn/data/bodzas \\
        --map    /gpfs/workdir/mouaddenn/data/bodzas/bodzas_to_grneutro_map.json \\
        --out    /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/bodzas_eval \\
        --batch_size 32
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
CH3 = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro")
sys.path.insert(0, str(CH3))

os.environ.setdefault("HF_HOME",       str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE",  str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME",    str(WORKDIR / "tmp" / "torch-hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from data import build_eval_transform  # noqa: E402
from models import DinoBloomBackbone, JointModel  # noqa: E402

# -----------------------------------------------------------------------------
# Bodzas class-name normalisation: published label -> snake_case key used in map
# -----------------------------------------------------------------------------

BODZAS_LABEL_ALIASES = {
    # canonical key                  : list of possible folder-name variants
    # Bodzas-2023 actually uses French/Czech orthography:
    # 'Neutrophile Segment', 'Neutrophile Band', 'Basophile', 'Eosinophile',
    # 'Lymphocyte', 'Monocyte', 'Normoblast', 'Myeloblast', 'Lymphoblast'.
    "neutrophil_segment":   ["neutrophile_segment", "neutrophile segment",
                              "neutrophil_segment", "neutrophils_segment",
                              "neutrophil segments", "neutrophil-segment",
                              "segmented_neutrophil", "neutrophil_segmented",
                              "segmented neutrophils", "segs", "seg",
                              "neutrophil_seg"],
    "neutrophil_band":      ["neutrophile_band", "neutrophile band",
                              "neutrophil_band", "neutrophil bands",
                              "neutrophil-band", "band_neutrophil",
                              "band cells", "bands", "band",
                              "neutrophils_band"],
    "eosinophil":           ["eosinophile", "eosinophil", "eosinophils",
                              "eos", "eo"],
    "basophil":             ["basophile", "basophil", "basophils",
                              "baso", "ba"],
    "lymphocyte":           ["lymphocyte", "lymphocytes", "lymph", "ly"],
    "monocyte":             ["monocyte", "monocytes", "mono", "mo"],
    "nucleated_red_blood_cell": ["normoblast", "normoblasts",
                                  "nucleated_red_blood_cell",
                                  "nucleated red blood cells",
                                  "nucleated red blood cell",
                                  "nrbc", "erythroblast"],
    "myeloblast":           ["myeloblast", "myeloblasts", "blast_myeloid",
                              "myeloid_blast", "mblast"],
    "lymphoblast":          ["lymphoblast", "lymphoblasts", "blast_lymphoid",
                              "lymphoid_blast", "lblast"],
}


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def classify_bodzas_folder(name: str) -> str | None:
    """Map a raw folder/zip basename to a canonical bodzas source-class key.

    Returns None if no alias matches (will be tracked as 'unmapped').
    """
    n = _norm(name)
    for canon, aliases in BODZAS_LABEL_ALIASES.items():
        for a in aliases:
            an = _norm(a)
            if n == an or n.startswith(an + "_") or n.endswith("_" + an) or ("_" + an + "_") in ("_" + n + "_"):
                return canon
    return None


# -----------------------------------------------------------------------------
# Bodzas zip-backed dataset (handles either single mega-zip or per-class zips)
# -----------------------------------------------------------------------------

class BodzasZipDataset(Dataset):
    """Iterate one cell at a time across (zip_path, member_filename) pairs."""

    def __init__(self, entries: list[tuple[str, str, str]], transform):
        # entries: [(zip_path, member_name, source_class), ...]
        self.entries = entries
        self.transform = transform
        self._zips: dict[str, zipfile.ZipFile] = {}

    def __len__(self):
        return len(self.entries)

    def _zip(self, path: str) -> zipfile.ZipFile:
        if path not in self._zips:
            self._zips[path] = zipfile.ZipFile(path, "r")
        return self._zips[path]

    def __getitem__(self, i):
        zip_path, member, src = self.entries[i]
        zf = self._zip(zip_path)
        with zf.open(member) as f:
            data = f.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        img = self.transform(img)
        return img, src, member


# -----------------------------------------------------------------------------
# Entry collection
# -----------------------------------------------------------------------------

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _is_image(name: str) -> bool:
    if "__MACOSX" in name or Path(name).name.startswith("._"):
        return False
    return Path(name).suffix.lower() in IMG_SUFFIXES


def collect_entries(bodzas_root: Path) -> tuple[list[tuple[str, str, str]],
                                                 dict[str, int]]:
    """Walk Bodzas root. Two layouts supported:

      (A) single bodzas_wbc.zip at root, with internal per-class folders
      (B) directory of per-class zips
    """
    entries: list[tuple[str, str, str]] = []
    unmapped: dict[str, int] = {}

    # Layout A — single big zip
    mega = list(bodzas_root.glob("*.zip"))
    # Pick the largest zip as the candidate "mega" archive
    mega_candidate = None
    if mega:
        mega.sort(key=lambda p: p.stat().st_size, reverse=True)
        # If any zip > 5 GB treat as mega; otherwise treat each as per-class
        if mega[0].stat().st_size > 5 * 2**30:
            mega_candidate = mega[0]

    if mega_candidate is not None:
        zp = mega_candidate
        print(f"[bodzas-collect] using single-zip layout: {zp}")
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir() or not _is_image(info.filename):
                    continue
                # Class = first non-empty path segment that maps to a canonical key
                parts = [p for p in Path(info.filename).parts if p
                          and not p.startswith("._") and p != "__MACOSX"]
                src = None
                for p in parts[:-1]:  # all except the filename
                    cand = classify_bodzas_folder(p)
                    if cand is not None:
                        src = cand
                        break
                if src is None:
                    # Last resort: try the parent dir
                    src = classify_bodzas_folder(Path(info.filename).parent.name)
                if src is None:
                    raw = Path(info.filename).parent.name or "ROOT"
                    unmapped[raw] = unmapped.get(raw, 0) + 1
                    continue
                entries.append((str(zp), info.filename, src))
    else:
        # Layout B — one zip per class
        print(f"[bodzas-collect] using per-zip layout ({len(mega)} zips)")
        for zp in sorted(mega):
            src = classify_bodzas_folder(zp.stem)
            if src is None:
                unmapped[zp.stem] = unmapped.get(zp.stem, 0) + 1
                continue
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not _is_image(info.filename):
                        continue
                    entries.append((str(zp), info.filename, src))

    return entries, unmapped


# -----------------------------------------------------------------------------
# Checkpoint loader (verbatim from MLL-23)
# -----------------------------------------------------------------------------

def load_joint_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    class_names = ckpt["class_names"]
    concepts = ckpt["concepts"]
    prior_C = ckpt["prior_C"]
    backbone = DinoBloomBackbone(
        variant=args["backbone"],
        unfreeze_last_n=int(args.get("unfreeze_last_n", 6)),
    ).to(device)
    model = JointModel(
        backbone=backbone,
        num_concepts=len(concepts),
        num_classes=len(class_names),
        prior_C=prior_C,
        concept_dim=int(args.get("concept_dim", 128)),
        num_heads=int(args.get("num_heads", 4)),
        classifier_dropout=float(args.get("classifier_dropout", 0.5)),
        mode=args.get("mode", "joint"),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[load] missing keys: {missing[:8]} (+{max(0, len(missing)-8)} more)")
    if unexpected:
        print(f"[load] unexpected: {unexpected[:8]} (+{max(0, len(unexpected)-8)} more)")
    model.eval()
    return model, class_names, concepts


# -----------------------------------------------------------------------------
# CDA: Concept Distribution Alignment via Mahalanobis whitening
# -----------------------------------------------------------------------------

def fit_cda(c_holdout: np.ndarray, c_grneutro: np.ndarray, eps: float = 1e-4):
    """Fit a whitening-then-recolouring affine map A, b on concept scores so that

        A @ c + b  has mean/cov matching c_grneutro on the heldout Bodzas slice.

    c_holdout : (N_h, M) Bodzas held-out concept sigmoid scores
    c_grneutro: (N_g, M) GR-Neutro train concept sigmoid scores

    Returns (A, b) so that c_aligned = c @ A.T + b (i.e. row vectors).
    """
    M = c_holdout.shape[1]
    mu_h = c_holdout.mean(axis=0)
    mu_g = c_grneutro.mean(axis=0)
    Ch = np.cov(c_holdout, rowvar=False) + eps * np.eye(M)
    Cg = np.cov(c_grneutro, rowvar=False) + eps * np.eye(M)
    # Whitening of holdout: W_h = Ch^{-1/2}
    Wh_vals, Wh_vecs = np.linalg.eigh(Ch)
    Wh_vals = np.maximum(Wh_vals, eps)
    Wh_inv_sqrt = Wh_vecs @ np.diag(1.0 / np.sqrt(Wh_vals)) @ Wh_vecs.T
    # Re-colouring to GR-Neutro: Cg^{1/2}
    Cg_vals, Cg_vecs = np.linalg.eigh(Cg)
    Cg_vals = np.maximum(Cg_vals, eps)
    Cg_sqrt = Cg_vecs @ np.diag(np.sqrt(Cg_vals)) @ Cg_vecs.T
    A = Cg_sqrt @ Wh_inv_sqrt   # (M, M)
    b = mu_g - A @ mu_h          # (M,)
    return A.astype(np.float32), b.astype(np.float32)


def apply_cda(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return c @ A.T + b


# -----------------------------------------------------------------------------
# Metrics — binary "Normal-vs-not" weighted F1 on the evaluable subset
# -----------------------------------------------------------------------------

def binary_eval(class_names: list[str],
                cls_probs: np.ndarray,
                src_arr: np.ndarray,
                mapping: dict) -> dict:
    """Build a binary Normal-vs-OOV ground truth from the source-class mapping
    and compute weighted F1 + per-class F1.

    Cells included:
      - neutrophil_segment  -> y=1 (Normal)
      - all 'OOV' sources   -> y=0 (not-Normal)
    Cells excluded:
      - neutrophil_band     -> ambiguous (audit_required); reported separately.
    """
    from sklearn.metrics import f1_score, classification_report

    normal_idx = class_names.index("Normal") if "Normal" in class_names else 0
    pred_normal = (cls_probs.argmax(axis=1) == normal_idx).astype(np.int32)

    y_true = np.full(len(src_arr), -1, dtype=np.int32)
    for i, s in enumerate(src_arr):
        target = mapping.get(str(s), {}).get("target", None)
        if target == "Normal":
            y_true[i] = 1
        elif target == "OOV":
            y_true[i] = 0
    mask = y_true >= 0
    y = y_true[mask]
    yhat = pred_normal[mask]
    if len(y) == 0:
        return {"n_eval": 0}

    res = {
        "n_eval": int(len(y)),
        "n_pos_Normal": int((y == 1).sum()),
        "n_neg_OOV":    int((y == 0).sum()),
        "weighted_f1": float(f1_score(y, yhat, average="weighted",
                                       zero_division=0)),
        "macro_f1":    float(f1_score(y, yhat, average="macro",
                                       zero_division=0)),
        "binary_f1_Normal": float(f1_score(y, yhat, pos_label=1,
                                            zero_division=0)),
        "binary_f1_OOV":    float(f1_score(y, yhat, pos_label=0,
                                            zero_division=0)),
    }

    # Per-source-class accuracy (was the model's top-1 the Normal label?)
    by_src_acc = {}
    for s in np.unique(src_arr):
        idx = np.where(src_arr == s)[0]
        sub_yhat = pred_normal[idx]
        target = mapping.get(str(s), {}).get("target", "?")
        by_src_acc[str(s)] = {
            "n": int(len(idx)),
            "target": target,
            "fraction_pred_Normal": float(sub_yhat.mean()),
        }
    res["by_source_class"] = by_src_acc
    return res


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--bodzas",  required=True)
    ap.add_argument("--map",     required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_per_class", type=int, default=0)
    ap.add_argument("--cda_heldout_fraction", type=float, default=0.20,
                    help="Fraction of Bodzas cells held out to estimate CDA stats")
    ap.add_argument("--cda_seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[bodzas-eval] device={device}")

    map_doc = json.loads(Path(args.map).read_text())
    mapping: dict[str, dict] = map_doc["mappings"]

    print("[bodzas-eval] loading model...")
    model, class_names, concepts = load_joint_model(Path(args.model), device)
    print(f"[bodzas-eval] {len(class_names)} classes, {len(concepts)} concepts")
    print(f"[bodzas-eval] classes: {class_names}")
    print(f"[bodzas-eval] concepts: {concepts}")

    print("[bodzas-eval] indexing Bodzas archive(s)...")
    entries, unmapped = collect_entries(Path(args.bodzas))
    if args.limit_per_class > 0:
        bucket: dict[str, list] = {}
        capped: list = []
        for e in entries:
            bucket.setdefault(e[2], []).append(e)
        for k, v in bucket.items():
            capped.extend(v[:args.limit_per_class])
        entries = capped
    n_per_src = {}
    for _, _, s in entries:
        n_per_src[s] = n_per_src.get(s, 0) + 1
    print(f"[bodzas-eval] {len(entries)} cells across {len(n_per_src)} source classes")
    for k, v in sorted(n_per_src.items()):
        print(f"             {k:30s}  {v:6d}")
    if unmapped:
        print(f"[bodzas-eval] UNMAPPED folders (counts): {unmapped}")

    ds = BodzasZipDataset(entries, build_eval_transform())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    n = len(entries)
    K_cls = len(class_names)
    K_con = len(concepts)
    cls_probs   = np.zeros((n, K_cls), dtype=np.float32)
    con_scores  = np.zeros((n, K_con), dtype=np.float32)
    cls_logits_all = np.zeros((n, K_cls), dtype=np.float32)
    con_logits_all = np.zeros((n, K_con), dtype=np.float32)
    sources_arr = np.empty(n, dtype=object)
    members_arr = np.empty(n, dtype=object)

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for batch_i, (imgs, srcs, members) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            cls_logits, con_logits = model(imgs)
            cls_p = torch.sigmoid(cls_logits).cpu().numpy()
            con_p = torch.sigmoid(con_logits).cpu().numpy()
            b = cls_p.shape[0]
            cls_probs[cursor:cursor+b]   = cls_p
            con_scores[cursor:cursor+b]  = con_p
            cls_logits_all[cursor:cursor+b] = cls_logits.cpu().numpy()
            con_logits_all[cursor:cursor+b] = con_logits.cpu().numpy()
            sources_arr[cursor:cursor+b] = list(srcs)
            members_arr[cursor:cursor+b] = list(members)
            cursor += b
            if batch_i % 50 == 0:
                dt = time.time() - t0
                rate = cursor / max(dt, 1e-6)
                eta = (n - cursor) / max(rate, 1e-6)
                print(f"  [{cursor:6d}/{n}] {rate:6.1f} cells/s   ETA {eta/60:5.1f} min")
    print(f"[bodzas-eval] inference done in {(time.time()-t0)/60:.1f} min")

    # ---------- Raw artefacts ----------
    np.savez_compressed(out / "per_cell_predictions.npz",
                        cls_probs=cls_probs,
                        con_scores=con_scores,
                        cls_logits=cls_logits_all,
                        con_logits=con_logits_all,
                        sources=sources_arr.astype(str),
                        members=members_arr.astype(str),
                        class_names=np.array(class_names),
                        concepts=np.array(concepts))
    print(f"[bodzas-eval] wrote per_cell_predictions.npz")

    # CSV for human inspection
    import csv
    with open(out / "per_cell_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "member"] + [f"p_{c}" for c in class_names]
                   + [f"con_{c}" for c in concepts])
        for i in range(n):
            row = [str(sources_arr[i]), str(members_arr[i])]
            row += [f"{x:.5f}" for x in cls_probs[i]]
            row += [f"{x:.5f}" for x in con_scores[i]]
            w.writerow(row)
    print(f"[bodzas-eval] wrote per_cell_predictions.csv")

    # ---------- Pre-CDA metrics ----------
    pre = binary_eval(class_names, cls_probs, sources_arr.astype(str), mapping)
    print(f"[bodzas-eval] PRE-CDA  n_eval={pre.get('n_eval')}  "
          f"weighted_f1={pre.get('weighted_f1'):.4f}  "
          f"binary_f1_Normal={pre.get('binary_f1_Normal'):.4f}  "
          f"binary_f1_OOV={pre.get('binary_f1_OOV'):.4f}")

    # ---------- CDA: fit on a held-out random slice of Bodzas, apply globally ----------
    cda_result = {"applied": False}
    try:
        # Load GR-Neutro train concept distribution from the model's predictions.pt
        gr_path = Path(args.model).parent / "predictions.pt"
        gr = torch.load(gr_path, map_location="cpu", weights_only=False)
        gr_train_concept_logits = gr["train"]["concept_logits"]
        if hasattr(gr_train_concept_logits, "numpy"):
            gr_train_con = torch.sigmoid(gr_train_concept_logits).numpy().astype(np.float32)
        else:
            gr_train_con = (1.0 / (1.0 + np.exp(-gr_train_concept_logits))).astype(np.float32)

        rng = np.random.default_rng(args.cda_seed)
        idx_all = np.arange(n)
        rng.shuffle(idx_all)
        n_h = int(round(args.cda_heldout_fraction * n))
        held_idx = idx_all[:n_h]
        rest_idx = idx_all[n_h:]
        c_holdout = con_scores[held_idx]
        A, b = fit_cda(c_holdout, gr_train_con)
        # Aligned concepts
        c_aligned = apply_cda(con_scores, A, b)
        # Convert aligned concept probabilities back to logits
        eps = 1e-6
        c_aligned_clip = np.clip(c_aligned, eps, 1 - eps)
        con_logits_aligned = np.log(c_aligned_clip) - np.log(1 - c_aligned_clip)

        # The model has two architectures:
        #   joint mode -> classifier is on CLS token, INDEPENDENT of concepts;
        #                  CDA on concepts cannot change class predictions.
        #                  We instead build a CAPC-style prototype classifier
        #                  on aligned concepts using the prior_C matrix as
        #                  per-class prototypes (the same approach as
        #                  capc_run.py).
        #   cbm   mode -> class_logits = class_from_concepts(concept_logits);
        #                  CDA flows directly through this layer.
        mode = getattr(model, "mode", "joint")
        cls_logits_aligned = None
        cda_route = None
        with torch.no_grad():
            t = torch.from_numpy(con_logits_aligned).to(device)
            if mode == "cbm" and getattr(model, "class_from_concepts", None) is not None:
                cls_logits_aligned = model.class_from_concepts(t).cpu().numpy()
                cda_route = "model.class_from_concepts(aligned concept logits)"
            else:
                # JOINT MODE: classifier is CLS-token based, cannot consume
                # concept logits. We fall back to a CAPC prototype classifier
                # on aligned sigmoid concept probabilities, using the
                # constraint module's prior C as the (K x M) class-to-concept
                # prototype matrix.
                try:
                    R = model.constraint_module.C.detach().cpu().numpy()
                    # R is (K, K) here (concept-to-concept prior). The
                    # GR-Neutro convention uses a SEPARATE class-to-concept
                    # matrix from concept_config_gr_neutro.json. Load it.
                    cfg_path = CH3 / "concept_config_gr_neutro.json"
                    cfg = json.loads(cfg_path.read_text())
                    R = np.array(cfg["class_to_concept_matrix"]["matrix"],
                                  dtype=np.float32)
                    # Mahalanobis distance from aligned concepts to each prototype
                    Sigma = np.cov(c_aligned, rowvar=False) + 1e-4 * np.eye(c_aligned.shape[1])
                    Sigma_inv = np.linalg.inv(Sigma)
                    diff = c_aligned[:, None, :] - R[None, :, :]   # (N, K, M)
                    tmp = diff @ Sigma_inv
                    d2 = np.einsum("nkm,nkm->nk", tmp, diff)
                    # Convert distance -> negative-distance logits (lower d = higher logit)
                    cls_logits_aligned = (-np.sqrt(np.maximum(d2, 0.0))).astype(np.float32)
                    cda_route = "CAPC Mahalanobis prototype classifier on aligned concepts (joint mode fallback)"
                except Exception as ee:
                    cda_route = f"FALLBACK FAILED: {ee!r}"

        if cls_logits_aligned is not None:
            # Normalise CAPC distance-logits via softmax to get probabilities
            x = cls_logits_aligned - cls_logits_aligned.max(axis=1, keepdims=True)
            cls_probs_aligned = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
            post = binary_eval(class_names, cls_probs_aligned,
                               sources_arr.astype(str), mapping)
            cda_result = {
                "applied": True,
                "model_mode": mode,
                "route": cda_route,
                "method": ("Mahalanobis whitening + recolouring of concept "
                           "layer; downstream classification per `route`."),
                "n_holdout": int(n_h),
                "n_evaluated": int(n),
                "post_cda_metrics": post,
                "fit": {
                    "A_frobenius_norm": float(np.linalg.norm(A)),
                    "b_l2_norm": float(np.linalg.norm(b)),
                },
            }
            np.savez_compressed(out / "cda_artefacts.npz",
                                A=A, b=b,
                                con_scores_aligned=c_aligned.astype(np.float32),
                                cls_logits_aligned=cls_logits_aligned.astype(np.float32),
                                held_idx=held_idx, rest_idx=rest_idx)
            print(f"[bodzas-eval] POST-CDA route='{cda_route}'  "
                  f"n_eval={post.get('n_eval')}  "
                  f"weighted_f1={post.get('weighted_f1'):.4f}  "
                  f"binary_f1_Normal={post.get('binary_f1_Normal'):.4f}  "
                  f"binary_f1_OOV={post.get('binary_f1_OOV'):.4f}")
        else:
            cda_result = {"applied": False, "reason": cda_route}
            print(f"[bodzas-eval] WARN CDA could not produce aligned class "
                  f"logits: {cda_route}")
    except Exception as e:
        cda_result = {"applied": False, "reason": f"CDA failed: {e!r}"}
        print(f"[bodzas-eval] WARN CDA failed: {e!r}")

    # ---------- Summary JSON ----------
    summary = {
        "_doc": "Bodzas-2023 external-cohort evaluation of the GR-Neutro joint model.",
        "model_path": str(args.model),
        "bodzas_path": str(args.bodzas),
        "n_cells": int(n),
        "n_per_source_class": n_per_src,
        "unmapped_folders": unmapped,
        "class_names": list(class_names),
        "concepts": list(concepts),
        "pre_cda": pre,
        "cda": cda_result,
        "caveats": {
            "directly_evaluable_classes": ["Normal (via neutrophil_segment)"],
            "audit_required_classes":     ["neutrophil_band -> Hyposegmentation"],
            "binary_protocol": ("Weighted F1 is computed on the Normal-vs-OOV "
                                "binary task: positives are neutrophil_segment "
                                "(target=Normal in the published mapping), "
                                "negatives are all sources with target=OOV. "
                                "Bodzas does not provide morphological-abnormality "
                                "labels, so per-class F1 over the full 7-class "
                                "GR-Neutro space is not defined."),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[bodzas-eval] wrote summary.json")
    print("[bodzas-eval] DONE")


if __name__ == "__main__":
    main()
