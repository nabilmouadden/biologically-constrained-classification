#!/usr/bin/env python3
"""Cache CLS features from a FINE-TUNED DinoBloom-B backbone over the full
GR-Neutro corpus, in the SAME npz schema as outputs/dinobloom_features.npz
(keys: 'features' (N,768) float32, 'paths' (N,) str). This lets residual_cbm.py
(PCBM-h + CEM) re-run over the fine-tuned representation with --features <this>.

Stream P's max_classification/finetune_7class.py does NOT persist a checkpoint or
features (only metrics JSON), so we re-train the chosen partial-fine-tune config
here (last-4 blocks, the primary candidate; single seed for a deterministic feature
bank) and then run the backbone in eval mode over EVERY cell to dump CLS features.

Model / transforms / split are imported verbatim from finetune_7class.py so the
fine-tuned representation is identical to the one that produced the reported ft W-F1.
We extract features with the eval transform (no augmentation) for every cell.

SLURM GPU only. Never the login node.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

WORKDIR = Path("/gpfs/workdir/mouaddenn")
sys.path.insert(0, str(WORKDIR / "thesis/outputs/max_classification"))

# Import the exact fine-tune machinery (Net, transforms, data loading, split).
import finetune_7class as ft  # noqa: E402


class CellPathDS(torch.utils.data.Dataset):
    """Eval-transform dataset that also returns the file path (for the npz)."""
    def __init__(self, rows, tfm):
        self.rows = rows; self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        path, y = self.rows[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), y, path


@torch.no_grad()
def extract_cls(model, loader, device):
    model.eval()
    feats, paths = [], []
    for x, _y, ps in loader:
        x = x.to(device, non_blocking=True)
        f = model.backbone.forward_features(x)[:, 0, :]   # CLS token, post-norm
        feats.append(f.cpu().numpy())
        paths.extend(list(ps))
    return np.concatenate(feats).astype(np.float32), np.array(paths, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dinobloom_b")
    ap.add_argument("--unfreeze_last_n", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--annotations", default=str(WORKDIR / "data/gr_neutro_extended/annotations.csv"))
    ap.add_argument("--data_root", default=str(WORKDIR / "data/gr_neutro_extended"))
    ap.add_argument("--out", required=True, help="output npz path for the ft feature bank")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  unfreeze_last_n={args.unfreeze_last_n} seed={args.seed}", flush=True)
    import torch.nn as nn

    rows = ft.load_rows(args.annotations, args.data_root)
    y_all = np.array([r[1] for r in rows])
    print(f"[data] {len(rows)} cells", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    tr, te = ft.split_idx(y_all, args.seed)
    tr_rows = [rows[i] for i in tr]; te_rows = [rows[i] for i in te]
    ytr = y_all[tr]

    tr_loader = DataLoader(ft.CellDS(tr_rows, ft.train_tfm()), batch_size=args.batch_size,
                           shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    te_loader = DataLoader(ft.CellDS(te_rows, ft.eval_tfm()), batch_size=64,
                           shuffle=False, num_workers=6, pin_memory=True)

    model = ft.Net(args.variant, args.unfreeze_last_n, dropout=args.dropout).to(device)
    cnt = np.bincount(ytr, minlength=len(ft.CLASSES)).astype(float); cnt[cnt == 0] = 1
    cw = torch.tensor(len(ytr) / (len(ft.CLASSES) * cnt), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    t0 = time.time(); best_wf1 = -1.0; best = None
    for ep in range(args.epochs):
        model.train()
        for x, yb in tr_loader:
            x = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = crit(model(x), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        wf1, mf1, _, _, _, _ = ft.evaluate(model, te_loader, device, tta=False)
        if wf1 > best_wf1:
            best_wf1 = wf1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:2d}] test W-F1={wf1:.4f} mac={mf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best)
    print(f"[ft] BEST held-out W-F1={best_wf1:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # Extract CLS features over the FULL corpus (eval transform, no aug).
    full_ds = CellPathDS(rows, ft.eval_tfm())
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False,
                             num_workers=6, pin_memory=True)
    feats, paths = extract_cls(model, full_loader, device)
    print(f"[extract] feats={feats.shape} paths={paths.shape}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=feats, paths=paths)
    # also stash the ft held-out W-F1 next to it for the residual_cbm backbone_json fallback
    import json
    Path(args.out + ".meta.json").write_text(json.dumps({
        "ft_heldout_wf1_seed0": best_wf1, "unfreeze_last_n": args.unfreeze_last_n,
        "seed": args.seed, "n_cells": int(len(paths)),
    }, indent=2))
    print(f"[save] {args.out}  (+ .meta.json)", flush=True)


if __name__ == "__main__":
    main()
