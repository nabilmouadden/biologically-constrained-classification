"""GR-Neutro joint training: backbone fine-tune + classifier + concept adapter.

Loss = BCE_class(pos_weight) + BCE_concept(pos_weight)
       + lambda_cst * (R_match + soft_cooccur).

Multi-label everywhere (BCEWithLogitsLoss + sigmoid). Pre-cached features are
NOT used — backbone is partial-fine-tuned through `unfreeze_last_n` blocks.

Design choices documented inline.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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
                     build_class_concept_targets, aggregate_concept_target)


class FocalBCE(nn.Module):
    """Multi-label focal BCE. Pairs with pos_weight."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - pt) ** self.gamma
        return (focal * bce).sum(dim=1).mean()


class EMA:
    """Polyak averaging of trainable params."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


def per_class_pos_weight(y: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """neg/pos per class; clamped at `cap` so sparse classes don't blow up loss scale."""
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = np.divide(neg, np.maximum(pos, 1.0))
    pw = np.minimum(pw, cap)
    return torch.tensor(pw, dtype=torch.float32)


def per_concept_pos_weight(soft_targets: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Treat soft targets >=0.5 as positives for the pos_weight computation."""
    hard = (soft_targets >= 0.5).float()
    pos = hard.sum(0)
    neg = hard.shape[0] - pos
    pw = neg / pos.clamp(min=1.0)
    return pw.clamp(max=cap)


def evaluate_split(probs: np.ndarray, y: np.ndarray, threshold: float = 0.5,
                   class_names: list[str] | None = None) -> dict:
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
    bal = []
    for k in range(K):
        if 0 < y[:, k].sum() < len(y):
            bal.append(balanced_accuracy_score(y[:, k], pred[:, k]))
    out = dict(
        macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        micro_f1=float(f1_score(y, pred, average="micro", zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)),
        mean_ap=float(np.nanmean(aps)),
        subset_accuracy=float((pred == y).all(axis=1).mean()),
        mean_per_class_accuracy=float((pred == y).mean(axis=0).mean()),
        balanced_accuracy=float(np.nanmean(bal)) if bal else float("nan"),
        per_class_aps=[float(a) for a in aps],
        threshold=threshold,
    )
    if class_names is not None:
        out["per_class"] = {
            name: dict(precision=float(prec[k]), recall=float(rec[k]),
                       f1=float(f1_per[k]), support=int(support[k]),
                       accuracy=float((pred[:, k] == y[:, k]).mean()))
            for k, name in enumerate(class_names)
        }
    return out


def evaluate_concepts(probs: np.ndarray, soft_y: np.ndarray, threshold: float = 0.5):
    """Concept F1 against thresholded soft targets (>=0.5 = positive)."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    y = (soft_y >= 0.5).astype(int)
    pred = (probs >= threshold).astype(int)
    prec, rec, f1_per, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0)
    return dict(
        per_concept_f1=[float(f) for f in f1_per],
        per_concept_precision=[float(p) for p in prec],
        per_concept_recall=[float(r) for r in rec],
        per_concept_support=[int(s) for s in support],
        mean_concept_f1=float(np.mean(f1_per)),
        macro_concept_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
        threshold=threshold,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_csv", default=str(WORKDIR / "data/gr_neutro_extended/annotations.csv"))
    ap.add_argument("--data_root", default=str(WORKDIR / "data/gr_neutro_extended"))
    ap.add_argument("--config", default=str(HERE / "concept_config_gr_neutro.json"))
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    ap.add_argument("--backbone", default="dinobloom_s")
    ap.add_argument("--unfreeze_last_n", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_classifier", type=float, default=1e-4)
    ap.add_argument("--lr_adapter", type=float, default=1e-3)
    ap.add_argument("--lr_constraint", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--classifier_dropout", type=float, default=0.5)
    ap.add_argument("--concept_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--lambda_constraint", type=float, default=0.1,
                    help="Concept-side constraint regularization weight (R-match + cooccur).")
    ap.add_argument("--lambda_concept_loss", type=float, default=1.0,
                    help="Multiplier on the concept BCE term in the total loss.")
    ap.add_argument("--concept_target_mode", default="hard",
                    choices=["soft", "hard"],
                    help="`soft`: BCE against continuous targets in [0,1] (matches "
                         "ch3 AML Matek). `hard`: binarize targets at 0.5 before BCE — "
                         "stronger gradient when target = 0.5 because soft-BCE has zero "
                         "gradient at p=t=0.5. Default `hard` for GR-Neutro.")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="If >0, replace classifier BCE with FocalBCE(gamma).")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="If >0, maintain Polyak-averaged weights and evaluate with them.")
    ap.add_argument("--posw_cap_class", type=float, default=50.0)
    ap.add_argument("--posw_cap_concept", type=float, default=50.0)
    ap.add_argument("--posw_class", action="store_true", default=True)
    ap.add_argument("--no_posw_class", dest="posw_class", action="store_false")
    ap.add_argument("--posw_concept", action="store_true", default=True)
    ap.add_argument("--no_posw_concept", dest="posw_concept", action="store_false")
    ap.add_argument("--balanced_sampling", action="store_true", default=False)
    ap.add_argument("--strong_aug", action="store_true", default=True)
    ap.add_argument("--no_strong_aug", dest="strong_aug", action="store_false")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--baseline", action="store_true",
                    help="Train classifier-only (no concept adapter, no constraint). "
                         "Reproduces the MIDL-style baseline on this code path.")
    ap.add_argument("--no_concept_adapter", action="store_true",
                    help="Same as --baseline but keeps the constraint module placeholder. "
                         "Use --baseline for canonical baseline runs.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [tag] {args.tag}  [seed] {args.seed}")

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    class_names, rows = read_annotations(args.data_csv, args.data_root)
    K_classes = len(class_names)
    labels_np = np.array([r[2] for r in rows], dtype=int)
    train_idx, val_idx, test_idx = stratified_multilabel_split(
        labels_np, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"[per-class train pos] " + ", ".join(
        f"{c}={int(labels_np[train_idx, k].sum())}" for k, c in enumerate(class_names)))

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    train_ds = GRNeutroDataset(train_rows, build_train_transform(strong=args.strong_aug))
    val_ds   = GRNeutroDataset(val_rows,   build_eval_transform())
    test_ds  = GRNeutroDataset(test_rows,  build_eval_transform())

    sampler = make_balanced_sampler(labels_np[train_idx]) if args.balanced_sampling else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ---------- concept config ----------
    cfg = json.loads(Path(args.config).read_text())
    concepts = cfg["concepts"]
    K_concepts = len(concepts)
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"])
    normal_idx = int(cfg.get("normal_class_index", 0))
    print(f"[cfg] {K_concepts} concepts; {K_classes} classes; normal_idx={normal_idx}")
    print(f"[cfg] C nonzero off-diag: {int((prior_C.abs() > 0).sum().item() - K_concepts)}; "
          f"C entries < 0: {int((prior_C < 0).sum().item())}; "
          f"C entries > 0 (off-diag): {int(((prior_C > 0) & (prior_C < 1.0)).sum().item())}")

    # ---------- model ----------
    backbone = DinoBloomBackbone(variant=args.backbone, unfreeze_last_n=args.unfreeze_last_n).to(device)
    model = JointModel(
        backbone=backbone, num_concepts=K_concepts, num_classes=K_classes,
        prior_C=prior_C, concept_dim=args.concept_dim, num_heads=args.num_heads,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    # If baseline, freeze the adapter+constraint. Adapter still receives a forward pass
    # (cheap; ~0.5M params) but contributes no gradient or loss term — keeps the
    # checkpoint shape stable across runs.
    is_baseline = args.baseline or args.no_concept_adapter
    if is_baseline:
        for p in model.concept_adapter.parameters():
            p.requires_grad_(False)
        for p in model.constraint_module.parameters():
            p.requires_grad_(False)
        print("[baseline] concept adapter + constraint module frozen, contribute no loss")

    # ---------- weights for class-level BCE ----------
    pw_class = (per_class_pos_weight(labels_np[train_idx], cap=args.posw_cap_class).to(device)
                if args.posw_class else None)
    if pw_class is not None:
        print(f"[posw class] {pw_class.cpu().round(decimals=2).tolist()}")

    # Compute concept targets for the whole dataset (cheap: tensor ops),
    # then derive concept pos_weight from the train fraction.
    multilabel_t = torch.tensor(labels_np, dtype=torch.float32)
    concept_targets_all = aggregate_concept_target(class_concept, multilabel_t,
                                                     normal_idx=normal_idx)
    pw_concept = (per_concept_pos_weight(concept_targets_all[train_idx],
                                           cap=args.posw_cap_concept).to(device)
                  if args.posw_concept else None)
    if pw_concept is not None:
        print(f"[posw concept] {pw_concept.cpu().round(decimals=2).tolist()}")

    if args.focal_gamma > 0:
        bce_class = FocalBCE(gamma=args.focal_gamma, alpha=args.focal_alpha,
                              pos_weight=pw_class).to(device)
        print(f"[focal] gamma={args.focal_gamma} alpha={args.focal_alpha} on classifier")
    else:
        bce_class = nn.BCEWithLogitsLoss(pos_weight=pw_class)
    bce_concept = nn.BCEWithLogitsLoss(pos_weight=pw_concept)

    # ---------- optimizer ----------
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    adapter_params = [p for p in model.concept_adapter.parameters() if p.requires_grad]
    constraint_params = [p for p in model.constraint_module.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:    param_groups.append({"params": backbone_params,    "lr": args.lr_backbone})
    if classifier_params:  param_groups.append({"params": classifier_params,  "lr": args.lr_classifier})
    if adapter_params:     param_groups.append({"params": adapter_params,     "lr": args.lr_adapter})
    if constraint_params:  param_groups.append({"params": constraint_params,  "lr": args.lr_constraint})
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[ema] decay={args.ema_decay}")

    # ---------- training ----------
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "train_total", "train_bce_cls", "train_bce_con", "train_cst",
            "train_class_f1", "train_concept_f1",
            "val_class_f1", "val_concept_f1", "lr",
        ])

    def epoch_pass(loader, training: bool, use_ema: bool = False):
        model.train(training)
        ema_backup = None
        if use_ema and ema is not None:
            ema_backup = ema.apply_to(model)
        n = 0
        tot = tot_cls = tot_con = tot_cst = 0.0
        all_clp, all_cnp, all_y, all_ct = [], [], [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ct_soft = aggregate_concept_target(class_concept.to(device), y,
                                                 normal_idx=normal_idx)
            ct_loss = (ct_soft >= 0.5).float() if args.concept_target_mode == "hard" else ct_soft
            with torch.set_grad_enabled(training):
                cls_logits, con_logits = model(x)
                L_cls = bce_class(cls_logits, y)
                L_con = bce_concept(con_logits, ct_loss) if not is_baseline else con_logits.new_zeros(())
                L_cst = (model.constraint_module.constraint_loss(con_logits)
                         if not is_baseline else con_logits.new_zeros(()))
                L = L_cls + args.lambda_concept_loss * L_con + args.lambda_constraint * L_cst
                if training:
                    optim.zero_grad(set_to_none=True)
                    L.backward()
                    optim.step()
                    if ema is not None:
                        ema.update(model)
            B = x.size(0)
            n += B
            tot += L.item() * B
            tot_cls += L_cls.item() * B
            tot_con += float(L_con) * B
            tot_cst += float(L_cst) * B
            all_clp.append(cls_logits.detach().sigmoid().cpu().numpy())
            all_cnp.append(con_logits.detach().sigmoid().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_ct.append(ct_soft.detach().cpu().numpy())
        if use_ema and ema is not None:
            ema.restore(model, ema_backup)
        cls_probs = np.concatenate(all_clp)
        con_probs = np.concatenate(all_cnp)
        y_np = np.concatenate(all_y)
        ct_np = np.concatenate(all_ct)
        cls_m = evaluate_split(cls_probs, y_np.astype(int))
        con_m = evaluate_concepts(con_probs, ct_np)
        return dict(loss=tot/n, cls=tot_cls/n, con=tot_con/n, cst=tot_cst/n,
                    cls_f1=cls_m["macro_f1"], con_f1=con_m["mean_concept_f1"],
                    cls_probs=cls_probs, con_probs=con_probs,
                    y=y_np, ct=ct_np)

    t0 = time.time()
    best_score = -1.0
    best_state = None
    for ep in range(args.epochs):
        tr = epoch_pass(train_loader, training=True)
        val = epoch_pass(val_loader, training=False, use_ema=(ema is not None))
        sched.step()
        lr_now = optim.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                ep, f"{tr['loss']:.5f}", f"{tr['cls']:.5f}", f"{tr['con']:.5f}",
                f"{tr['cst']:.4f}",
                f"{tr['cls_f1']:.4f}", f"{tr['con_f1']:.4f}",
                f"{val['cls_f1']:.4f}", f"{val['con_f1']:.4f}", f"{lr_now:.6f}",
            ])
        # Selection score: classification F1 + concept F1 (concept term is 0 for baseline,
        # which reduces selection to classification F1 — exactly what we want there).
        score = val["cls_f1"] + (0.0 if is_baseline else val["con_f1"])
        if score > best_score:
            best_score = score
            # Snapshot EMA-applied weights when EMA is on, so the saved checkpoint
            # is the one we just evaluated.
            if ema is not None:
                bb = ema.apply_to(model)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model, bb)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] L={tr['loss']:.3f} (cls={tr['cls']:.3f} "
                  f"con={tr['con']:.3f} cst={tr['cst']:.2f}) | "
                  f"tr cls_f1={tr['cls_f1']:.3f} con_f1={tr['con_f1']:.3f} | "
                  f"val cls_f1={val['cls_f1']:.3f} con_f1={val['con_f1']:.3f} | "
                  f"({time.time() - t0:.0f}s)")

    print(f"[done] training: {time.time()-t0:.0f}s; best val selection score={best_score:.3f}")

    # Reload best checkpoint and dump test/val/cal predictions.
    model.load_state_dict(best_state)
    model.eval()

    def dump_preds(loader):
        all_cls, all_con, all_y, all_ct = [], [], [], []
        all_attn = []
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            ct_soft = aggregate_concept_target(class_concept.to(device), y, normal_idx=normal_idx)
            with torch.no_grad():
                cls_logits, con_logits = model(x)
            all_cls.append(cls_logits.cpu()); all_con.append(con_logits.cpu())
            all_y.append(y.cpu()); all_ct.append(ct_soft.cpu())
            if not is_baseline:
                all_attn.append(model.concept_adapter.last_attn.cpu())
        out = dict(class_logits=torch.cat(all_cls), concept_logits=torch.cat(all_con),
                   labels=torch.cat(all_y), concept_targets=torch.cat(all_ct))
        if all_attn:
            out["attn"] = torch.cat(all_attn)
        return out

    test_pred = dump_preds(test_loader)
    val_pred = dump_preds(val_loader)
    # Dump train preds on best checkpoint (no augmentation): the completeness
    # probe needs train concept logits to fit the linear classifier.
    train_eval_ds = GRNeutroDataset(train_rows, build_eval_transform())
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=True)
    train_pred = dump_preds(train_eval_loader)

    # Final metrics dump for downstream evaluate.py / make_figures.py.
    cls_m = evaluate_split(test_pred["class_logits"].sigmoid().numpy(),
                             test_pred["labels"].numpy().astype(int),
                             threshold=0.5, class_names=class_names)
    con_m = evaluate_concepts(test_pred["concept_logits"].sigmoid().numpy(),
                                test_pred["concept_targets"].numpy())
    print(f"\n[{args.tag}] TEST  weighted_f1={cls_m['weighted_f1']:.4f}  "
          f"macro_f1={cls_m['macro_f1']:.4f}  "
          f"mean_concept_f1={con_m['mean_concept_f1']:.4f}")
    print(f"[{args.tag}] per-class F1: " + ", ".join(
        f"{n}={cls_m['per_class'][n]['f1']:.3f}" for n in class_names))
    print(f"[{args.tag}] per-concept F1: " + ", ".join(
        f"{c}={con_m['per_concept_f1'][k]:.3f}" for k, c in enumerate(concepts)))

    torch.save({
        "state_dict": best_state,
        "args": vars(args),
        "class_names": class_names,
        "concepts": concepts,
        "prior_C": prior_C,
        "normal_idx": normal_idx,
    }, out_dir / "model.pt")
    torch.save({
        "test": test_pred, "val": val_pred, "train": train_pred,
        "test_idx": test_idx, "val_idx": val_idx, "train_idx": train_idx,
        "concepts": concepts, "class_names": class_names,
        "is_baseline": is_baseline,
    }, out_dir / "predictions.pt")
    (out_dir / "split.json").write_text(json.dumps({
        "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(), "seed": args.seed,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
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
