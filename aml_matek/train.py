"""Train the concept adapter + classifier on cached backbone features.

Ablations (plan §6):
  --joint        vs  --frozen        : is the classifier co-trained with the adapter?
  --constrained  vs  --unconstrained : is the constraint regularizer active? (lambda=0.1 vs 0)

The backbone is never loaded; features are pre-cached.
"""

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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from models import JointModel, build_prior_C, build_class_concept_targets

WORKDIR = Path("/gpfs/workdir/mouaddenn")
THESIS = WORKDIR / "thesis" / "aml_matek"
DATA = WORKDIR / "data"


def stratified_3way_split(labels: np.ndarray, seed: int = 0):
    """Return (train_idx, cal_idx, test_idx) with 60/20/20 split, stratified by class.

    Uses two successive stratified splits: first split off 40% as (cal+test),
    then split that 50/50 into cal/test. Classes with very few samples
    (e.g. LYA=11) should still get >=2 in each of cal/test.
    """
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx, test_size=0.40, stratify=labels, random_state=seed,
    )
    cal_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=labels[rest_idx], random_state=seed,
    )
    return train_idx, cal_idx, test_idx


def run_baseline(args, out_dir):
    """Linear probe on CLS: classifier only, no adapter, no constraint.

    Used as the parity reference for the 'matching accuracy with explanations' claim.
    Dumps a predictions.pt compatible with evaluate.py / make_figures.py, marked
    is_baseline=True so downstream scripts skip concept-dependent metrics.
    """
    feat_path = DATA / args.dataset / "features" / f"{args.backbone}.pt"
    print(f"[baseline:load] {feat_path}")
    blob = torch.load(feat_path, map_location="cpu", weights_only=False)
    features = blob["features"]                       # keep fp16 in RAM
    labels_np = blob["labels"].numpy()
    embed_dim = blob["embed_dim"]
    num_classes = blob["num_classes"]

    train_idx, cal_idx, test_idx = stratified_3way_split(labels_np, seed=args.seed)
    print(f"[baseline:split] train={len(train_idx)} cal={len(cal_idx)} test={len(test_idx)}")

    def make_loader(idx, shuffle):
        sub_f = features[idx]
        sub_y = torch.from_numpy(labels_np[idx]).long()
        return DataLoader(TensorDataset(sub_f, sub_y),
                          batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    cal_loader = make_loader(cal_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_classifier, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_cls", "train_class_acc", "val_class_acc", "lr"])

    def epoch_pass(loader, training):
        model.train() if training else model.eval()
        tot_loss = 0.0
        n = 0
        correct = 0
        with torch.set_grad_enabled(training):
            for f, y in loader:
                f = f.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True)
                cls = f[:, 0, :]
                logits = model(cls)
                L = ce_loss(logits, y)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    L.backward()
                    optimizer.step()
                B = f.size(0)
                n += B
                tot_loss += L.item() * B
                correct += (logits.argmax(1) == y).sum().item()
        return dict(loss=tot_loss / n, acc=correct / n)

    best_val = -1
    for ep in range(args.epochs):
        tr = epoch_pass(train_loader, training=True)
        val = epoch_pass(cal_loader, training=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr['loss']:.5f}",
                                    f"{tr['acc']:.4f}", f"{val['acc']:.4f}", f"{lr_now:.6f}"])
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[baseline ep {ep:3d}] loss={tr['loss']:.4f} "
                  f"tr_acc={tr['acc']:.3f} val_acc={val['acc']:.3f}")
        if val["acc"] > best_val:
            best_val = val["acc"]
            torch.save({"state_dict": model.state_dict(), "epoch": ep,
                        "args": vars(args), "val_acc": val["acc"]},
                       out_dir / "model.pt")

    # Reload best checkpoint, dump predictions.pt
    bl = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(bl["state_dict"])
    model.eval()

    def dump(loader):
        logits, labels = [], []
        with torch.no_grad():
            for f, y in loader:
                f = f.to(device).float()
                logits.append(model(f[:, 0, :]).cpu())
                labels.append(y)
        return dict(class_logits=torch.cat(logits), labels=torch.cat(labels))

    test_pred = dump(test_loader)
    cal_pred = dump(cal_loader)

    torch.save({
        "test": test_pred,
        "cal": cal_pred,
        "test_idx": test_idx,
        "cal_idx": cal_idx,
        "is_baseline": True,
        "num_classes": num_classes,
    }, out_dir / "predictions.pt")
    print(f"[baseline dump] predictions.pt written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["dinobloom_s", "dinov2_vitb14", "resnet50"])
    ap.add_argument("--dataset", required=True, choices=["aml_matek"])
    ap.add_argument("--baseline", action="store_true",
                    help="Linear-probe baseline: classifier only, no adapter/constraint.")
    head_group = ap.add_mutually_exclusive_group(required=False)
    head_group.add_argument("--joint", action="store_true")
    head_group.add_argument("--frozen", action="store_true")
    cst_group = ap.add_mutually_exclusive_group(required=False)
    cst_group.add_argument("--constrained", action="store_true")
    cst_group.add_argument("--unconstrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr_adapter", type=float, default=1e-3)
    ap.add_argument("--lr_classifier", type=float, default=1e-4)
    ap.add_argument("--lambda_constraint", type=float, default=0.1)
    ap.add_argument("--concept_pos_weight", action="store_true",
                    help="Use per-concept positive-class weighting in BCE, "
                         "computed from training-set class-derived concept targets. "
                         "Fixes rare-concept failure (e.g. band_nucleus when NGB is 0.6%% of data).")
    ap.add_argument("--tag", default=None,
                    help="Override the auto-generated output directory name. "
                         "Value becomes outputs/<tag>/. Subdirectories allowed (e.g. lambda_sweep/lam0p05).")
    ap.add_argument("--seed", type=int, default=42)
    # num_workers=0 because features are already in RAM tensors (no I/O). Workers
    # with num_workers>=1 fork copies of the 7-15 GB feature tensor per process,
    # which is what caused the DINOv2 OOM on the first run.
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Validate config. Baseline mode: ignore head/constraint flags.
    if not args.baseline:
        if not (args.joint ^ args.frozen):
            raise SystemExit("Either --joint or --frozen is required (not both).")
        if not (args.constrained ^ args.unconstrained):
            raise SystemExit("Either --constrained or --unconstrained is required (not both).")

    # Resolve tags for the output directory.
    if args.tag:
        out_dir = THESIS / "outputs" / args.tag
    elif args.baseline:
        out_dir = THESIS / "outputs" / f"{args.backbone}_{args.dataset}_baseline"
    else:
        head_tag = "joint" if args.joint else "frozen"
        cst_tag = "const" if args.constrained else "unconst"
        suffix = "_posw" if args.concept_pos_weight else ""
        out_dir = THESIS / "outputs" / f"{args.backbone}_{args.dataset}_{head_tag}_{cst_tag}{suffix}"
    lam = args.lambda_constraint if (not args.baseline and args.constrained) else 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline:
        run_baseline(args, out_dir)
        return

    # ---- load cached features ----
    feat_path = DATA / args.dataset / "features" / f"{args.backbone}.pt"
    print(f"[load] {feat_path}")
    blob = torch.load(feat_path, map_location="cpu", weights_only=False)
    # Keep features in float16 in RAM — ~2x smaller. Cast to float32 per batch on GPU.
    features = blob["features"]                      # (N, 1+P, d) float16
    labels_np = blob["labels"].numpy()
    embed_dim = blob["embed_dim"]
    num_patches = blob["num_patches"]
    num_classes = blob["num_classes"]
    print(f"[load] features={tuple(features.shape)} ; embed_dim={embed_dim} P={num_patches} C={num_classes}")

    # ---- load concept config ----
    cfg = json.loads((THESIS / "concept_config.json").read_text())
    concepts = cfg["concepts"]
    num_concepts = len(concepts)
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])  # (C, K)
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"])                    # (K, K)
    # Mutually exclusive pairs as (i, j) indices — feed directly into the
    # violation penalty so constrained runs differ from unconstrained.
    concept_idx = {c: i for i, c in enumerate(concepts)}
    exclusive_pairs = [
        (concept_idx[p["concepts"][0]], concept_idx[p["concepts"][1]])
        for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]
    ]
    print(f"[cfg] {num_concepts} concepts, {len(exclusive_pairs)} mutually-exclusive pairs")

    # Per-image soft concept target = row of class_concept indexed by label.
    concept_targets = class_concept[torch.from_numpy(labels_np)]   # (N, K)

    # ---- 60/20/20 stratified split ----
    train_idx, cal_idx, test_idx = stratified_3way_split(labels_np, seed=args.seed)
    print(f"[split] train={len(train_idx)} cal={len(cal_idx)} test={len(test_idx)}")

    def make_loader(idx, shuffle):
        sub_f = features[idx]
        sub_y = torch.from_numpy(labels_np[idx]).long()
        sub_c = concept_targets[idx]
        ds = TensorDataset(sub_f, sub_y, sub_c)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    cal_loader = make_loader(cal_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    # ---- build model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointModel(
        feature_dim=embed_dim,
        num_concepts=num_concepts,
        num_classes=num_classes,
        prior_C=prior_C,
        exclusive_pairs=exclusive_pairs,
    ).to(device)

    # --frozen: freeze the classifier; only adapter + constraint module train.
    # With a random frozen classifier, class accuracy is chance; this is the intended
    # ablation (does concept learning need classification signal?).
    # Downstream class performance is then measured via the linear probe in evaluate.py.
    if args.frozen:
        for p in model.classifier.parameters():
            p.requires_grad_(False)

    # Separate LR for classifier vs adapter/constraint (per plan §6).
    adapter_params = list(model.concept_adapter.parameters()) + list(model.constraint_module.parameters())
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = [{"params": adapter_params, "lr": args.lr_adapter}]
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": args.lr_classifier})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce_loss = nn.CrossEntropyLoss()
    if args.concept_pos_weight:
        # Compute per-concept pos_weight = #negatives / #positives from the
        # TRAINING soft targets. Rare concepts (e.g. band_nucleus, supported only
        # by NGB which is 0.6% of Matek) get large weights so their signal doesn't
        # drown out in shared BCE.
        train_soft = concept_targets[train_idx]                  # (N_tr, K)
        pos_count = train_soft.sum(dim=0)                        # (K,)
        neg_count = train_soft.shape[0] - pos_count
        # Clamp to avoid divide-by-zero for concepts with zero train support.
        pos_weight = neg_count / (pos_count.clamp(min=1.0))      # (K,)
        # Cap to keep optimization stable.
        pos_weight = pos_weight.clamp(max=50.0)
        print(f"[weight] concept pos_weight: min={pos_weight.min():.2f} "
              f"max={pos_weight.max():.2f} mean={pos_weight.mean():.2f}")
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        bce_loss = nn.BCEWithLogitsLoss()

    # ---- training log ----
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_total", "train_cls", "train_concept", "train_constraint",
            "train_class_acc", "train_concept_f1_at_0p5",
            "val_class_acc", "val_concept_f1_at_0p5", "lr",
        ])

    def epoch_pass(loader, training):
        if training:
            model.train()
        else:
            model.eval()
        tot_total = tot_cls = tot_con = tot_cst = 0.0
        n_samples = 0
        correct_cls = 0
        # for concept F1@0.5
        c_tp = torch.zeros(num_concepts)
        c_fp = torch.zeros(num_concepts)
        c_fn = torch.zeros(num_concepts)
        with torch.set_grad_enabled(training):
            for f, y, c_soft in loader:
                f = f.to(device, non_blocking=True).float()   # cast fp16->fp32 on GPU
                y = y.to(device, non_blocking=True)
                c_soft = c_soft.to(device, non_blocking=True)
                class_logits, concept_logits = model(f)
                L_cls = ce_loss(class_logits, y)
                L_con = bce_loss(concept_logits, c_soft)
                L_cst = model.constraint_module.constraint_loss(concept_logits)
                L = L_cls + L_con + lam * L_cst
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    L.backward()
                    optimizer.step()
                B = f.size(0)
                n_samples += B
                tot_total += L.item() * B
                tot_cls += L_cls.item() * B
                tot_con += L_con.item() * B
                tot_cst += L_cst.item() * B
                correct_cls += (class_logits.argmax(1) == y).sum().item()
                # concept F1@0.5 against HARD-THRESHOLDED soft target
                # (target >= 0.5 counts as positive).
                p_hat = (concept_logits.sigmoid() >= 0.5).float().cpu()
                g = (c_soft >= 0.5).float().cpu()
                c_tp += (p_hat * g).sum(0)
                c_fp += (p_hat * (1 - g)).sum(0)
                c_fn += ((1 - p_hat) * g).sum(0)

        precision = c_tp / (c_tp + c_fp + 1e-9)
        recall = c_tp / (c_tp + c_fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        stats = dict(
            total=tot_total / n_samples,
            cls=tot_cls / n_samples,
            concept=tot_con / n_samples,
            constraint=tot_cst / n_samples,
            class_acc=correct_cls / n_samples,
            concept_f1=f1.mean().item(),
        )
        return stats

    t0 = time.time()
    best_val = -1
    for ep in range(args.epochs):
        tr = epoch_pass(train_loader, training=True)
        val = epoch_pass(cal_loader, training=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, f"{tr['total']:.5f}", f"{tr['cls']:.5f}", f"{tr['concept']:.5f}",
                f"{tr['constraint']:.5f}",
                f"{tr['class_acc']:.4f}", f"{tr['concept_f1']:.4f}",
                f"{val['class_acc']:.4f}", f"{val['concept_f1']:.4f}", f"{lr_now:.6f}",
            ])
        print(f"[ep {ep:3d}] total={tr['total']:.4f} cls={tr['cls']:.4f} "
              f"con={tr['concept']:.4f} cst={tr['constraint']:.2f} | "
              f"tr_acc={tr['class_acc']:.3f} val_acc={val['class_acc']:.3f} "
              f"tr_f1={tr['concept_f1']:.3f} val_f1={val['concept_f1']:.3f}")
        # Track best by (val_class_acc + val_concept_f1). For frozen runs, class_acc is chance,
        # so this reduces to concept_f1 alone, which is the right metric for that ablation.
        score = val["class_acc"] + val["concept_f1"]
        if score > best_val:
            best_val = score
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep,
                "args": vars(args),
                "val_stats": val,
            }, out_dir / "model.pt")

    print(f"[done] {time.time()-t0:.1f}s; best val score={best_val:.3f}")

    # ---- dump test-set predictions for evaluate.py ----
    # Load best checkpoint
    blob = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    def dump_preds(loader, split_name):
        all_class, all_concept, all_labels, all_ctgt = [], [], [], []
        all_attn = []
        with torch.no_grad():
            for f, y, c_soft in loader:
                f = f.to(device).float()
                cl, co = model(f)
                all_class.append(cl.cpu())
                all_concept.append(co.cpu())
                all_labels.append(y)
                all_ctgt.append(c_soft)
                all_attn.append(model.concept_adapter.last_attn.cpu())
        return dict(
            class_logits=torch.cat(all_class),
            concept_logits=torch.cat(all_concept),
            labels=torch.cat(all_labels),
            concept_targets=torch.cat(all_ctgt),
            attn=torch.cat(all_attn),
        )

    test_pred = dump_preds(test_loader, "test")
    cal_pred = dump_preds(cal_loader, "cal")

    torch.save({
        "test": test_pred,
        "cal": cal_pred,
        "test_idx": test_idx,
        "cal_idx": cal_idx,
        "train_idx": train_idx,
        "concepts": concepts,
        "num_classes": num_classes,
        "grid_hw": blob["args"] and None,  # grid comes from feature cache; attached below
    }, out_dir / "predictions.pt")
    # Also save grid_hw and a subset of attention maps on a representative set (2 per class).
    # This file is the one evaluate.py uses for visualizations.
    feat_blob = torch.load(feat_path, map_location="cpu", weights_only=False)
    grid_hw = feat_blob["grid_hw"]
    filenames_test = [feat_blob["filenames"][i] for i in test_idx]

    # Pick 2 images per class from test for attention dumps (up to 2*num_classes = 30 images).
    rng = np.random.default_rng(args.seed)
    picked = []
    y_test = labels_np[test_idx]
    for cls in range(num_classes):
        where = np.where(y_test == cls)[0]
        if len(where) == 0:
            continue
        # choose up to 2
        chosen = rng.choice(where, size=min(2, len(where)), replace=False)
        picked.extend(chosen.tolist())
    picked = sorted(set(picked))
    sub_idx = torch.tensor(picked, dtype=torch.long)
    torch.save({
        "attn": test_pred["attn"][sub_idx],              # (n, K, P)
        "filenames": [filenames_test[i] for i in picked],
        "labels": test_pred["labels"][sub_idx],
        "concept_preds": test_pred["concept_logits"][sub_idx].sigmoid(),
        "grid_hw": grid_hw,
        "concepts": concepts,
    }, out_dir / "attention_weights.pt")

    print(f"[dump] predictions.pt and attention_weights.pt written to {out_dir}")


if __name__ == "__main__":
    main()
