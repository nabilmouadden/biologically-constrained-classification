"""C4 — Joint training + auxiliary VLM-disagreement head.

Adapts ch3_gr_neutro/train.py to add a second concept-side head that predicts
the per-cell VLM disagreement vector delta(x_c) = |s_BiomedCLIP(x_c) - s_OpenCLIP(x_c)|.

Theory: Khemakhem et al. 2020 (iVAE) shows that an auxiliary variable u that is
asymptotically class-conditional-independent of y, combined with sufficient
variability in the conditional p(z|u), restores identifiability of the latent
factors up to permutation + scaling. The class label y, by Lemma 1 of this
paper, does NOT satisfy iVAE's conditions (it is the very thing the population
minimiser collapses concepts onto). The VLM-disagreement vector delta is, by
construction of the disjoint VLM training corpora, statistically weakly coupled
to y — we therefore use it as an auxiliary signal.

Loss = BCE_cls + lambda_concept * BCE_concept(textbook target t_c)
       + lambda_aux * MSE_aux(delta_c)
       + lambda_constraint * (R-match + cooccur)

The auxiliary head is a small MLP on the same concept embedding the BCE head
reads from. Both heads share parameters until the final layer; the shared
representation is what gets identifiability benefit (in the iVAE sense).

WARNING: per the empirical pre-check (prepare_disagreement.py), delta is NOT
strictly class-independent on GR-Neutro (mean eta-squared ~ 0.048, all 11
concepts reject Kruskal-Wallis equality of distributions). The trained model
should still benefit from the auxiliary signal (additional information not in
the class label), but the iVAE identifiability claim is *approximate*, not
exact. We honestly report both readings.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent  # vlm_aux_work
CH3 = HERE.parent  # ch3_gr_neutro
sys.path.insert(0, str(CH3))

# Make HF/torch caches point to project tmp (no internet on compute nodes).
os.environ.setdefault("HF_HOME", str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(WORKDIR / "tmp" / "torch-hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from data import (read_annotations, stratified_multilabel_split,
                  GRNeutroDataset, build_train_transform, build_eval_transform,
                  make_balanced_sampler)
from models import (DinoBloomBackbone, JointModel, build_prior_C,
                    build_class_concept_targets, aggregate_concept_target,
                    ConceptAdapter)


# ============================================================
#  Per-cell VLM disagreement registry: filename -> delta vector
# ============================================================

def load_disagreement_registry(npy_dir: Path):
    """Returns {basename: delta_vec (11,)} float32 and concept names."""
    delta = np.load(npy_dir / "delta_primary.npy")     # (N, 11)
    fnames_paths = np.load(npy_dir / "filenames.npy")  # (N,) absolute paths
    concepts = list(np.load(npy_dir / "concepts.npy"))
    # Convert absolute paths to basenames (annotations.csv uses basenames).
    basenames = [Path(p).name for p in fnames_paths]
    reg = {b: delta[i].astype(np.float32) for i, b in enumerate(basenames)}
    return reg, concepts


def lookup_deltas(filenames, reg, concept_dim: int):
    """Return (B, K) float32 and (B,) bool mask of which cells have delta."""
    out = np.zeros((len(filenames), concept_dim), dtype=np.float32)
    mask = np.zeros(len(filenames), dtype=bool)
    for i, fn in enumerate(filenames):
        v = reg.get(fn)
        if v is not None:
            out[i] = v
            mask[i] = True
    return out, mask


# ============================================================
#  Joint model with auxiliary head (NEW)
# ============================================================

class JointModelAux(nn.Module):
    """Joint or CBM model + an extra MLP head predicting per-concept VLM disagreement.

    The aux head reads the post-attention concept-vector representation (the
    output of ConceptAdapter's attention layer, BEFORE the final scalar head).
    This is the per-concept embedding the iVAE-style identifiability argument
    applies to.
    """

    def __init__(self, backbone, num_concepts, num_classes, prior_C,
                 concept_dim=128, num_heads=4, classifier_dropout=0.5,
                 mode="joint"):
        super().__init__()
        # Reuse the standard JointModel but tap into ConceptAdapter to expose
        # the post-attention concept tokens before the scalar head.
        assert mode in ("joint", "cbm")
        self.base = JointModel(
            backbone=backbone, num_concepts=num_concepts, num_classes=num_classes,
            prior_C=prior_C, concept_dim=concept_dim, num_heads=num_heads,
            classifier_dropout=classifier_dropout, mode=mode,
        )
        # New aux head: scalar regressor per concept on the concept-attended embedding.
        # Architecture mirrors ConceptAdapter.head but returns a separate scalar.
        self.aux_head = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, 1),
        )
        # We need to hook into ConceptAdapter.forward to grab the attended tokens.
        # Simplest: monkey-patch forward to also return the pre-head concept embedding.
        ca = self.base.concept_adapter
        self._attached_aux(ca)

    def _attached_aux(self, ca: ConceptAdapter):
        ca._stashed_post_attn = None
        orig_forward = ca.forward

        def new_forward(patch_tokens):
            B = patch_tokens.shape[0]
            kv = ca.proj(patch_tokens)
            q = ca.queries.unsqueeze(0).expand(B, -1, -1)
            attn_out, attn_w = ca.mha(q, kv, kv, need_weights=True,
                                       average_attn_weights=True)
            ca.last_attn = attn_w.detach()
            ca._stashed_post_attn = attn_out   # (B, K, concept_dim)
            logits = ca.head(attn_out).squeeze(-1)
            return logits
        ca.forward = new_forward

    def forward(self, x):
        cls_logits, con_logits = self.base(x)
        # Grab the stashed post-attention concept tokens.
        post = self.base.concept_adapter._stashed_post_attn  # (B, K, D)
        aux = self.aux_head(post).squeeze(-1)                # (B, K) -- predicted delta
        return cls_logits, con_logits, aux


# ============================================================
#  Helpers borrowed from train.py
# ============================================================

def per_class_pos_weight(y, cap=50.0):
    pos = y.sum(axis=0); neg = y.shape[0] - pos
    pw = np.divide(neg, np.maximum(pos, 1.0))
    return torch.tensor(np.minimum(pw, cap), dtype=torch.float32)


def per_concept_pos_weight(soft_targets, cap=50.0):
    hard = (soft_targets >= 0.5).float()
    pos = hard.sum(0); neg = hard.shape[0] - pos
    return (neg / pos.clamp(min=1.0)).clamp(max=cap)


def evaluate_split(probs, y, threshold=0.5, class_names=None):
    from sklearn.metrics import (f1_score, average_precision_score,
                                  balanced_accuracy_score, precision_recall_fscore_support)
    pred = (probs >= threshold).astype(int)
    K = y.shape[1]
    prec, rec, f1_per, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0)
    aps = []
    for k in range(K):
        if 0 < y[:, k].sum() < len(y):
            aps.append(average_precision_score(y[:, k], probs[:, k]))
        else:
            aps.append(float("nan"))
    out = dict(
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        mean_ap=float(np.nanmean(aps)),
        per_class_aps=[float(a) for a in aps],
    )
    if class_names is not None:
        out["per_class"] = {
            name: dict(precision=float(prec[k]), recall=float(rec[k]),
                       f1=float(f1_per[k]), support=int(support[k]))
            for k, name in enumerate(class_names)
        }
    return out


def evaluate_concepts(probs, soft_y, threshold=0.5):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    y = (soft_y >= 0.5).astype(int)
    pred = (probs >= threshold).astype(int)
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0)
    return dict(
        per_concept_f1=[float(f) for f in f1_per],
        mean_concept_f1=float(np.mean(f1_per)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_csv", default=str(WORKDIR / "data/gr_neutro_extended/annotations.csv"))
    ap.add_argument("--data_root", default=str(WORKDIR / "data/gr_neutro_extended"))
    ap.add_argument("--config", default=str(CH3 / "concept_config_gr_neutro.json"))
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    ap.add_argument("--delta_dir", default=str(HERE),
                    help="Directory with delta_primary.npy, filenames.npy, concepts.npy")
    ap.add_argument("--backbone", default="dinobloom_s")
    ap.add_argument("--unfreeze_last_n", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_classifier", type=float, default=1e-4)
    ap.add_argument("--lr_adapter", type=float, default=1e-3)
    ap.add_argument("--lr_constraint", type=float, default=1e-3)
    ap.add_argument("--lr_aux", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--classifier_dropout", type=float, default=0.5)
    ap.add_argument("--concept_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--lambda_constraint", type=float, default=0.1)
    ap.add_argument("--lambda_concept_loss", type=float, default=1.0)
    ap.add_argument("--lambda_aux", type=float, default=0.5,
                    help="Multiplier on MSE on VLM disagreement prediction.")
    ap.add_argument("--concept_target_mode", default="hard", choices=["soft", "hard"])
    ap.add_argument("--mode", default="joint", choices=["joint", "cbm"])
    ap.add_argument("--strong_aug", action="store_true", default=True)
    ap.add_argument("--no_strong_aug", dest="strong_aug", action="store_false")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--posw_cap_class", type=float, default=50.0)
    ap.add_argument("--posw_cap_concept", type=float, default=50.0)
    ap.add_argument("--posw_class", action="store_true", default=True)
    ap.add_argument("--no_posw_class", dest="posw_class", action="store_false")
    ap.add_argument("--posw_concept", action="store_true", default=True)
    ap.add_argument("--no_posw_concept", dest="posw_concept", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [tag] {args.tag}  [seed] {args.seed}  [lam_aux] {args.lambda_aux}")

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    class_names, rows = read_annotations(args.data_csv, args.data_root)
    K_classes = len(class_names)
    labels_np = np.array([r[2] for r in rows], dtype=int)
    train_idx, val_idx, test_idx = stratified_multilabel_split(
        labels_np, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    # IMPORTANT: we need filenames at training time to look up delta.
    train_ds = GRNeutroDataset(train_rows, build_train_transform(strong=args.strong_aug),
                                 return_filename=True)
    val_ds = GRNeutroDataset(val_rows, build_eval_transform(), return_filename=True)
    test_ds = GRNeutroDataset(test_rows, build_eval_transform(), return_filename=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---------- concept config + delta registry ----------
    cfg = json.loads(Path(args.config).read_text())
    concepts = cfg["concepts"]
    K_concepts = len(concepts)
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"])
    normal_idx = int(cfg.get("normal_class_index", 0))

    delta_reg, delta_concepts = load_disagreement_registry(Path(args.delta_dir))
    assert delta_concepts == concepts, (
        f"delta concepts {delta_concepts} != config concepts {concepts}")
    print(f"[delta_reg] {len(delta_reg)} cells indexed")

    # ---------- model ----------
    backbone = DinoBloomBackbone(variant=args.backbone, unfreeze_last_n=args.unfreeze_last_n).to(device)
    model = JointModelAux(
        backbone=backbone, num_concepts=K_concepts, num_classes=K_classes,
        prior_C=prior_C, concept_dim=args.concept_dim, num_heads=args.num_heads,
        classifier_dropout=args.classifier_dropout, mode=args.mode,
    ).to(device)

    # ---------- losses & weights ----------
    pw_class = (per_class_pos_weight(labels_np[train_idx], cap=args.posw_cap_class).to(device)
                if args.posw_class else None)
    multilabel_t = torch.tensor(labels_np, dtype=torch.float32)
    concept_targets_all = aggregate_concept_target(class_concept, multilabel_t,
                                                     normal_idx=normal_idx)
    pw_concept = (per_concept_pos_weight(concept_targets_all[train_idx],
                                           cap=args.posw_cap_concept).to(device)
                  if args.posw_concept else None)
    bce_class = nn.BCEWithLogitsLoss(pos_weight=pw_class)
    bce_concept = nn.BCEWithLogitsLoss(pos_weight=pw_concept)

    # ---------- optimizer ----------
    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    cls_params = ([p for p in model.base.classifier.parameters() if p.requires_grad]
                  if model.base.classifier is not None else
                  [p for p in model.base.class_from_concepts.parameters() if p.requires_grad])
    ad_params = [p for p in model.base.concept_adapter.parameters() if p.requires_grad]
    cs_params = [p for p in model.base.constraint_module.parameters() if p.requires_grad]
    aux_params = [p for p in model.aux_head.parameters() if p.requires_grad]
    param_groups = []
    if bb_params:  param_groups.append({"params": bb_params, "lr": args.lr_backbone})
    if cls_params: param_groups.append({"params": cls_params, "lr": args.lr_classifier})
    if ad_params:  param_groups.append({"params": ad_params, "lr": args.lr_adapter})
    if cs_params:  param_groups.append({"params": cs_params, "lr": args.lr_constraint})
    if aux_params: param_groups.append({"params": aux_params, "lr": args.lr_aux})
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "train_total", "train_bce_cls", "train_bce_con", "train_mse_aux",
            "train_cst", "train_class_f1", "train_concept_f1",
            "val_class_f1", "val_concept_f1", "val_mse_aux", "lr",
        ])

    def epoch_pass(loader, training: bool):
        model.train(training)
        n = 0
        tot = tot_cls = tot_con = tot_aux = tot_cst = 0.0
        all_clp, all_cnp, all_y, all_ct = [], [], [], []
        all_aux_pred, all_aux_true, all_aux_mask = [], [], []
        for x, y, fn in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Build delta target and mask from filenames.
            delta_np, mask_np = lookup_deltas(list(fn), delta_reg, K_concepts)
            delta_t = torch.from_numpy(delta_np).to(device)
            mask_t = torch.from_numpy(mask_np).to(device)
            ct_soft = aggregate_concept_target(class_concept.to(device), y,
                                                 normal_idx=normal_idx)
            ct_loss = (ct_soft >= 0.5).float() if args.concept_target_mode == "hard" else ct_soft
            with torch.set_grad_enabled(training):
                cls_logits, con_logits, aux_pred = model(x)
                L_cls = bce_class(cls_logits, y)
                L_con = bce_concept(con_logits, ct_loss)
                L_cst = model.base.constraint_module.constraint_loss(con_logits)
                if mask_t.any():
                    aux_err = (aux_pred[mask_t] - delta_t[mask_t]).pow(2)
                    L_aux = aux_err.mean()
                else:
                    L_aux = aux_pred.new_zeros(())
                L = (L_cls
                     + args.lambda_concept_loss * L_con
                     + args.lambda_aux * L_aux
                     + args.lambda_constraint * L_cst)
                if training:
                    optim.zero_grad(set_to_none=True)
                    L.backward()
                    optim.step()
            B = x.size(0); n += B
            tot += L.item() * B
            tot_cls += L_cls.item() * B
            tot_con += L_con.item() * B
            tot_aux += float(L_aux) * B
            tot_cst += float(L_cst) * B
            all_clp.append(cls_logits.detach().sigmoid().cpu().numpy())
            all_cnp.append(con_logits.detach().sigmoid().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_ct.append(ct_soft.detach().cpu().numpy())
            all_aux_pred.append(aux_pred.detach().cpu().numpy())
            all_aux_true.append(delta_np)
            all_aux_mask.append(mask_np)
        cls_probs = np.concatenate(all_clp)
        con_probs = np.concatenate(all_cnp)
        y_np = np.concatenate(all_y); ct_np = np.concatenate(all_ct)
        aux_pred_np = np.concatenate(all_aux_pred)
        aux_true_np = np.concatenate(all_aux_true)
        aux_mask_np = np.concatenate(all_aux_mask)
        cls_m = evaluate_split(cls_probs, y_np.astype(int))
        con_m = evaluate_concepts(con_probs, ct_np)
        mse_aux = float(((aux_pred_np[aux_mask_np] - aux_true_np[aux_mask_np])**2).mean()) \
                  if aux_mask_np.any() else float("nan")
        return dict(loss=tot/n, cls=tot_cls/n, con=tot_con/n, aux=tot_aux/n, cst=tot_cst/n,
                    cls_f1=cls_m["macro_f1"], con_f1=con_m["mean_concept_f1"],
                    mse_aux=mse_aux,
                    cls_probs=cls_probs, con_probs=con_probs, y=y_np, ct=ct_np,
                    aux_pred=aux_pred_np, aux_true=aux_true_np, aux_mask=aux_mask_np,
                    wf1=cls_m["weighted_f1"])

    t0 = time.time()
    best_score = -1.0
    best_state = None
    for ep in range(args.epochs):
        tr = epoch_pass(train_loader, training=True)
        val = epoch_pass(val_loader, training=False)
        sched.step()
        lr_now = optim.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                ep, f"{tr['loss']:.5f}", f"{tr['cls']:.5f}", f"{tr['con']:.5f}",
                f"{tr['aux']:.5f}", f"{tr['cst']:.4f}",
                f"{tr['cls_f1']:.4f}", f"{tr['con_f1']:.4f}",
                f"{val['cls_f1']:.4f}", f"{val['con_f1']:.4f}",
                f"{val['mse_aux']:.5f}", f"{lr_now:.6f}",
            ])
        score = val["cls_f1"] + val["con_f1"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] L={tr['loss']:.3f} (cls={tr['cls']:.3f} "
                  f"con={tr['con']:.3f} aux={tr['aux']:.4f} cst={tr['cst']:.2f}) | "
                  f"tr cls_f1={tr['cls_f1']:.3f} con_f1={tr['con_f1']:.3f} | "
                  f"val cls_f1={val['cls_f1']:.3f} con_f1={val['con_f1']:.3f} "
                  f"aux_mse={val['mse_aux']:.4f} | ({time.time()-t0:.0f}s)")

    print(f"[done] training: {time.time()-t0:.0f}s; best val score={best_score:.3f}")

    # ---------- reload best, dump test ----------
    model.load_state_dict(best_state)
    model.eval()

    def dump_preds(loader):
        all_cls, all_con, all_aux, all_y, all_ct, all_fn = [], [], [], [], [], []
        for x, y, fn in loader:
            x = x.to(device); y = y.to(device)
            ct_soft = aggregate_concept_target(class_concept.to(device), y, normal_idx=normal_idx)
            with torch.no_grad():
                cls_logits, con_logits, aux_pred = model(x)
            all_cls.append(cls_logits.cpu()); all_con.append(con_logits.cpu())
            all_aux.append(aux_pred.cpu())
            all_y.append(y.cpu()); all_ct.append(ct_soft.cpu())
            all_fn.extend(list(fn))
        return dict(class_logits=torch.cat(all_cls), concept_logits=torch.cat(all_con),
                    aux_pred=torch.cat(all_aux),
                    labels=torch.cat(all_y), concept_targets=torch.cat(all_ct),
                    filenames=all_fn)

    test_pred = dump_preds(test_loader)
    val_pred = dump_preds(val_loader)
    train_eval_ds = GRNeutroDataset(train_rows, build_eval_transform(), return_filename=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=True)
    train_pred = dump_preds(train_eval_loader)

    cls_m = evaluate_split(test_pred["class_logits"].sigmoid().numpy(),
                            test_pred["labels"].numpy().astype(int),
                            threshold=0.5, class_names=class_names)
    con_m = evaluate_concepts(test_pred["concept_logits"].sigmoid().numpy(),
                                test_pred["concept_targets"].numpy())
    print(f"\n[{args.tag}] TEST  wf1={cls_m['weighted_f1']:.4f}  "
          f"macro_f1={cls_m['macro_f1']:.4f}  "
          f"mean_concept_f1={con_m['mean_concept_f1']:.4f}")

    torch.save({
        "state_dict": best_state, "args": vars(args),
        "class_names": class_names, "concepts": concepts,
        "prior_C": prior_C, "normal_idx": normal_idx,
    }, out_dir / "model.pt")
    torch.save({
        "test": test_pred, "val": val_pred, "train": train_pred,
        "test_idx": test_idx, "val_idx": val_idx, "train_idx": train_idx,
        "concepts": concepts, "class_names": class_names,
    }, out_dir / "predictions.pt")
    (out_dir / "split.json").write_text(json.dumps({
        "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(), "seed": args.seed,
    }))
    summary = dict(
        tag=args.tag, args=vars(args),
        test_classification=cls_m, test_concepts=con_m,
        time_seconds=time.time() - t0,
        best_val_selection_score=best_score,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {out_dir}/model.pt  predictions.pt  summary.json")


if __name__ == "__main__":
    main()
