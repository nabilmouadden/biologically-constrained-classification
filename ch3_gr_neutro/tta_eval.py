"""TTA evaluation: average sigmoid probabilities across N flip/rot augmentations.

Reads outputs/<tag>/model.pt + the original test data, forwards each test image
through K transforms, averages the sigmoid outputs, and saves
outputs/<tag>/predictions_tta.pt. Then re-runs evaluate.py on the TTA output
to get the boosted metrics.

Usage:  python tta_eval.py --tag <tag>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HOME", str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from data import read_annotations, stratified_multilabel_split, GRNeutroDataset
from models import (DinoBloomBackbone, JointModel, build_prior_C,
                     build_class_concept_targets, aggregate_concept_target)


TTA_TFMS = [
    # 6 deterministic transforms: identity, hflip, vflip, rot90, rot180, rot270.
    lambda img: img,
    lambda img: img.transpose(method=2),  # hflip (FLIP_LEFT_RIGHT = 0)
    lambda img: img.transpose(method=1),  # vflip (FLIP_TOP_BOTTOM = 1)
    lambda img: img.rotate(90),
    lambda img: img.rotate(180),
    lambda img: img.rotate(270),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    ap.add_argument("--data_csv", default=str(WORKDIR / "data/gr_neutro_extended/annotations.csv"))
    ap.add_argument("--data_root", default=str(WORKDIR / "data/gr_neutro_extended"))
    ap.add_argument("--config", default=str(HERE / "concept_config_gr_neutro.json"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.tag
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    class_names, rows = read_annotations(args.data_csv, args.data_root)
    labels_np = np.array([r[2] for r in rows], dtype=int)
    train_idx, val_idx, test_idx = stratified_multilabel_split(
        labels_np, test_size=train_args.get("test_size", 0.10),
        val_size=train_args.get("val_size", 0.10),
        seed=train_args["seed"])

    cfg = json.loads(Path(args.config).read_text())
    concepts = cfg["concepts"]
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBloomBackbone(variant=train_args["backbone"],
                                   unfreeze_last_n=train_args["unfreeze_last_n"]).to(device)
    model = JointModel(backbone, num_concepts=len(concepts), num_classes=len(class_names),
                        prior_C=prior_C,
                        concept_dim=train_args.get("concept_dim", 128),
                        num_heads=train_args.get("num_heads", 4),
                        classifier_dropout=train_args.get("classifier_dropout", 0.5)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    is_baseline = train_args.get("baseline", False) or train_args.get("no_concept_adapter", False)

    eval_norm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def run_tta(idxs):
        sub_rows = [rows[i] for i in idxs]
        all_cls_probs = None
        all_con_probs = None
        for tta in TTA_TFMS:
            tfm = transforms.Compose([transforms.Lambda(tta), eval_norm])
            ds = GRNeutroDataset(sub_rows, tfm)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
            cls_p, con_p = [], []
            for x, y in dl:
                x = x.to(device)
                with torch.no_grad():
                    cl, co = model(x)
                cls_p.append(cl.sigmoid().cpu())
                con_p.append(co.sigmoid().cpu())
            cls_p = torch.cat(cls_p).numpy()
            con_p = torch.cat(con_p).numpy()
            if all_cls_probs is None:
                all_cls_probs = cls_p
                all_con_probs = con_p
            else:
                all_cls_probs = all_cls_probs + cls_p
                all_con_probs = all_con_probs + con_p
        all_cls_probs /= len(TTA_TFMS)
        all_con_probs /= len(TTA_TFMS)
        # Convert back to logits for compatibility with evaluate.py
        eps = 1e-7
        cls_logits = np.log(np.clip(all_cls_probs, eps, 1 - eps) /
                             np.clip(1 - all_cls_probs, eps, 1 - eps))
        con_logits = np.log(np.clip(all_con_probs, eps, 1 - eps) /
                             np.clip(1 - all_con_probs, eps, 1 - eps))
        return cls_logits, con_logits

    val_cl, val_co = run_tta(val_idx)
    test_cl, test_co = run_tta(test_idx)

    # Build predictions_tta.pt in the same format as predictions.pt for evaluate.py reuse.
    pkg = {
        "val": dict(class_logits=torch.tensor(val_cl), concept_logits=torch.tensor(val_co),
                    labels=torch.tensor(labels_np[val_idx], dtype=torch.float32),
                    concept_targets=aggregate_concept_target(class_concept,
                                                              torch.tensor(labels_np[val_idx], dtype=torch.float32))),
        "test": dict(class_logits=torch.tensor(test_cl), concept_logits=torch.tensor(test_co),
                     labels=torch.tensor(labels_np[test_idx], dtype=torch.float32),
                     concept_targets=aggregate_concept_target(class_concept,
                                                               torch.tensor(labels_np[test_idx], dtype=torch.float32))),
        "test_idx": test_idx, "val_idx": val_idx, "train_idx": train_idx,
        "concepts": concepts, "class_names": class_names,
        "is_baseline": is_baseline,
    }
    # Pass through train preds from the original predictions.pt (no TTA for probe fit)
    src = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    if "train" in src:
        pkg["train"] = src["train"]
    torch.save(pkg, run_dir / "predictions_tta.pt")

    print(f"[tta] saved {run_dir}/predictions_tta.pt with {len(TTA_TFMS)}-fold TTA")
    print(f"[tta] now run: python evaluate.py --run_dir {run_dir} (after renaming or modifying eval to use predictions_tta.pt)")


if __name__ == "__main__":
    main()
