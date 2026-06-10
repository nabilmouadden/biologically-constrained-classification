#!/usr/bin/env python3
"""End-to-end / partial fine-tuning of DinoBloom-B for 7-class single-label
GR-Neutro neutrophil classification.

WHY a dedicated script (not ch3_gr_neutro/train.py): train.py is multi-label
(BCE + multi-label W-F1). The clinical deliverable and the 0.831 baseline are
SINGLE-LABEL 7-class (argmax) W-F1 on the directory-derived class. To compose
honestly we keep the EXACT same metric and the EXACT same per-seed stratified
70/30 split (learned_faithful_cbm/run.py recipe). This script:
  - loads DinoBloom-B (timm hf-hub, cached) or DINOv2 ViT-B/14,
  - softmax 7-class head on the CLS token,
  - partial fine-tune of the last-k transformer blocks (k configurable; k=0 =
    frozen probe control, k=12 = full),
  - class-weighted cross-entropy for imbalance (Dohle/Hyperseg are rare),
  - strong train aug, eval with center resize, optional test-time augmentation,
  - reports W-F1 / macro-F1 / per-class F1 on the held-out 30%.

Runs ONLY under SLURM on a GPU node. Never on the Ruche login node.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_recall_fscore_support

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import timm  # noqa: E402

CLASSES = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
           "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]

_DINOBLOOM_HF = {
    "dinobloom_b": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
    "dinobloom_s": "hf-hub:1aurent/vit_small_patch14_224.dinobloom",
}


# ----------------------------- data -----------------------------
def load_rows(annotations, data_root):
    """Return list of (path, class_idx) using directory-derived single label,
    identical labelling to the 0.831 baseline (os.path.dirname)."""
    data_root = Path(data_root)
    index = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in data_root.rglob(ext):
            index.setdefault(p.name, p)
    df = pd.read_csv(annotations)
    cmap = {c: i for i, c in enumerate(CLASSES)}
    rows = []
    for _, r in df.iterrows():
        fn = r["filename"]
        cls = os.path.basename(os.path.dirname(r["path"]))
        if cls not in cmap:
            raise ValueError(f"unknown class dir {cls} for {fn}")
        real = index.get(fn)
        if real is None:
            raise FileNotFoundError(f"{fn} not found under {data_root}")
        rows.append((str(real), cmap[cls]))
    return rows


def split_idx(y, seed):
    """Per-seed stratified 70/30 — identical to learned_faithful_cbm/run.py."""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for k in np.unique(y):
        ids = np.where(y == k)[0]
        rng.shuffle(ids)
        cut = int(0.7 * len(ids))
        tr += list(ids[:cut]); te += list(ids[cut:])
    return np.array(tr), np.array(te)


class CellDS(Dataset):
    def __init__(self, rows, tfm):
        self.rows = rows; self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, y = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), y


def train_tfm():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_tfm():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ----------------------------- model -----------------------------
class Net(nn.Module):
    def __init__(self, variant, unfreeze_last_n, dropout=0.3):
        super().__init__()
        if variant in _DINOBLOOM_HF:
            self.backbone = timm.create_model(_DINOBLOOM_HF[variant], pretrained=True, img_size=224)
            blocks = self.backbone.blocks
            self.norm = getattr(self.backbone, "norm", None)
        else:
            raise ValueError(variant)
        self.embed_dim = self.backbone.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        nb = len(blocks)
        if unfreeze_last_n > 0:
            for blk in blocks[max(0, nb - unfreeze_last_n):]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if self.norm is not None:
                for p in self.norm.parameters():
                    p.requires_grad_(True)
        self.head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Dropout(dropout),
                                  nn.Linear(self.embed_dim, len(CLASSES)))
        nt = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.backbone.parameters())
        print(f"[backbone] {variant}: {nt:,}/{tot:,} trainable (last {unfreeze_last_n}/{nb})", flush=True)

    def forward(self, x):
        feats = self.backbone.forward_features(x)   # (B,1+P,d)
        cls = feats[:, 0, :]
        return self.head(cls)


def evaluate(model, loader, device, tta=False):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).softmax(1)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3])).softmax(1)
                logits = logits + model(torch.flip(x, dims=[2])).softmax(1)
            preds.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    pred = np.concatenate(preds); yt = np.concatenate(ys)
    wf1 = float(f1_score(yt, pred, average="weighted", zero_division=0))
    mf1 = float(f1_score(yt, pred, average="macro", zero_division=0))
    p, r, f1, sup = precision_recall_fscore_support(
        yt, pred, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    return wf1, mf1, f1.tolist(), sup.tolist(), pred, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dinobloom_b")
    ap.add_argument("--unfreeze_last_n", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--class_weighted", action="store_true", default=True)
    ap.add_argument("--no_class_weighted", dest="class_weighted", action="store_false")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--tta", action="store_true", default=False)
    ap.add_argument("--seeds", default="0,1,7,42,1337")
    ap.add_argument("--annotations", default="./data/gr_neutro/annotations.csv")
    ap.add_argument("--data_root", default="./data/gr_neutro")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default="./runs/max_classification")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [tag] {args.tag}", flush=True)
    rows = load_rows(args.annotations, args.data_root)
    y_all = np.array([r[1] for r in rows])
    print(f"[data] {len(rows)} cells; counts="
          f"{ {CLASSES[i]: int((y_all==i).sum()) for i in range(len(CLASSES))} }", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = []
    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        tr, te = split_idx(y_all, seed)
        tr_rows = [rows[i] for i in tr]; te_rows = [rows[i] for i in te]
        ytr = y_all[tr]

        tr_loader = DataLoader(CellDS(tr_rows, train_tfm()), batch_size=args.batch_size,
                               shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
        te_loader = DataLoader(CellDS(te_rows, eval_tfm()), batch_size=64,
                               shuffle=False, num_workers=6, pin_memory=True)

        model = Net(args.variant, args.unfreeze_last_n, dropout=args.dropout).to(device)
        if args.class_weighted:
            cnt = np.bincount(ytr, minlength=len(CLASSES)).astype(float); cnt[cnt == 0] = 1
            cw = torch.tensor(len(ytr) / (len(CLASSES) * cnt), dtype=torch.float32, device=device)
        else:
            cw = None
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

        bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
        groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
        if bb_params:
            groups.append({"params": bb_params, "lr": args.lr_backbone})
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_wf1 = -1.0; best = None
        for ep in range(args.epochs):
            model.train()
            for x, yb in tr_loader:
                x = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    loss = crit(model(x), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            wf1, mf1, _, _, _, _ = evaluate(model, te_loader, device, tta=False)
            if wf1 > best_wf1:
                best_wf1 = wf1
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"[seed {seed} ep {ep:2d}] test W-F1={wf1:.4f} mac={mf1:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        model.load_state_dict(best)
        wf1, mf1, pcf1, sup, pred, yt = evaluate(model, te_loader, device, tta=args.tta)
        print(f"[seed {seed}] BEST test W-F1={wf1:.4f} mac={mf1:.4f} (tta={args.tta})", flush=True)
        per_seed.append({"seed": seed, "wf1": wf1, "macf1": mf1,
                         "per_class_f1": pcf1, "per_class_support": sup})

    wf = np.array([d["wf1"] for d in per_seed]); mf = np.array([d["macf1"] for d in per_seed])
    pcf = np.array([d["per_class_f1"] for d in per_seed])
    out = {
        "tag": args.tag, "variant": args.variant, "unfreeze_last_n": args.unfreeze_last_n,
        "epochs": args.epochs, "class_weighted": args.class_weighted,
        "label_smoothing": args.label_smoothing, "tta": args.tta,
        "metric": "single-label 7-class weighted-F1 (argmax)",
        "split": "per-seed stratified 70/30 (identical to learned_faithful_cbm)",
        "seeds": seeds, "classes": CLASSES,
        "wf1_mean": float(wf.mean()), "wf1_std": float(wf.std()),
        "wf1_seeds": [float(v) for v in wf],
        "macf1_mean": float(mf.mean()), "macf1_std": float(mf.std()),
        "per_class_f1_mean": pcf.mean(0).tolist(),
        "per_class_support_mean": np.array([d["per_class_support"] for d in per_seed]).mean(0).tolist(),
        "per_seed": per_seed,
    }
    outdir = Path(args.out_root); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{args.tag}] W-F1={wf.mean():.4f}+/-{wf.std():.4f}  "
          f"macro={mf.mean():.4f}+/-{mf.std():.4f}", flush=True)
    print(f"[save] {outdir}/{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
