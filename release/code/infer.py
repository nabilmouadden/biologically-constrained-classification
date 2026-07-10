#!/usr/bin/env python3
"""Single-image inference for the GR-Neutro concept-bottleneck release.

Given ONE peripheral-blood single-cell image and ONE released checkpoint
(``release/weights/*.pt``), this script:

  1. Loads the checkpoint, reads its self-describing metadata (the concept list,
     the class names, the prior concept-constraint matrix ``prior_C``, the
     backbone variant and the ``mode``), and rebuilds the exact ``JointModel``
     that produced it. Loading is metadata-driven, so the same entrypoint works
     for the joint architecture (``B_kitchen_*`` -> separate CLS classifier +
     concept adapter) and the pure-bottleneck CBM (``cbm_*`` -> class predicted
     only through the concept layer).
  2. Runs the backbone forward on the image and prints
       - the predicted abnormality class (argmax over the 7 GR-Neutro classes),
         with the full per-class probability vector, and
       - the model's predicted concept activations (the concept-adapter head).
  3. INDEPENDENTLY runs the deterministic morphometry pipeline
     (``morphometry_concepts_v2.compute_concepts``) on the same image and prints
     the 10 measured textbook-morphology concept values in [0,1]. These are the
     label-free *measured* concepts; they are NOT learned and do not depend on
     the checkpoint.

Nothing is fabricated: every number printed is computed live from the image and
the released weights.

Example
-------
    python infer.py \
        --weights ../weights/cbm_joint_s0_model.pt \
        --image /path/to/one_cell.png

Notes
-----
* The DinoBloom-B backbone weights are pulled by ``timm`` from the public
  HF-hub repo ``hf-hub:1aurent/vit_base_patch14_224.dinobloom`` the first time a
  checkpoint is loaded (needs network on first run, then cached). The released
  ``*.pt`` checkpoints already contain the fine-tuned last-N backbone blocks +
  the heads, so the only thing fetched from HF-hub is the frozen lower backbone.
* The checkpoints were trained with 11 concept-adapter outputs (the historical
  concept set, one channel wider than the current vocabulary). The loader reads
  the concept list from the checkpoint, so the model rebuilds at the correct
  width automatically. The deterministic-morphometry block reports the canonical
  10 concepts listed in ``concepts_10.json``; see ``release/README.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent

from models import DinoBloomBackbone, JointModel  # noqa: E402
from morphometry_concepts_v2 import segment_cell, compute_concepts  # noqa: E402

# Canonical 10 measured concepts. The deterministic morphometry computes one
# extra legacy channel; we surface only these 10 as the released set.
CANON_10 = json.loads((HERE / "concepts_10.json").read_text())["concepts"]


def build_eval_transform():
    """ImageNet-normalised 224x224 eval transform (matches data.build_eval_transform)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(weights_path: str, device: torch.device):
    """Rebuild the JointModel described by a released checkpoint and load weights."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    concepts = ckpt["concepts"]
    class_names = ckpt["class_names"]
    prior_C = ckpt["prior_C"]
    args = ckpt.get("args", {}) or {}
    backbone_variant = args.get("backbone", "dinobloom_b")
    unfreeze_n = int(args.get("unfreeze_last_n", 6))
    concept_dim = int(args.get("concept_dim", 128))
    num_heads = int(args.get("num_heads", 4))
    classifier_dropout = float(args.get("classifier_dropout", 0.5))
    # Older checkpoints predate the --mode flag; absence == the default "joint".
    mode = args.get("mode") or "joint"

    backbone = DinoBloomBackbone(variant=backbone_variant, unfreeze_last_n=unfreeze_n)
    model = JointModel(
        backbone=backbone,
        num_concepts=len(concepts),
        num_classes=len(class_names),
        prior_C=prior_C,
        concept_dim=concept_dim,
        num_heads=num_heads,
        classifier_dropout=classifier_dropout,
        mode=mode,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys when loading: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading: {unexpected}")
    model.to(device).eval()
    return model, concepts, class_names, mode


@torch.no_grad()
def run_model(model, image_path: str, device: torch.device):
    tfm = build_eval_transform()
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    class_logits, concept_logits = model(x)
    class_prob = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
    concept_act = torch.sigmoid(concept_logits)[0].cpu().numpy()
    return class_prob, concept_act


def run_morphometry(image_path: str):
    """Deterministic textbook-morphology concepts for one image. Returns dict or None."""
    rgb = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float64) / 255.0
    seg = segment_cell(rgb)
    if not seg.get("ok", False):
        return None, seg.get("reason", "segmentation_failed")
    concepts, _raw = compute_concepts(rgb, seg)
    return concepts, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="path to a released model.pt")
    ap.add_argument("--image", required=True, help="path to one single-cell image")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[infer] device={device}  weights={args.weights}")
    model, concepts, class_names, mode = load_model(args.weights, device)

    # The released checkpoints carry a wider concept head than the released
    # vocabulary; only the canonical 10 concepts are exposed. The trailing
    # legacy channel(s) are intentionally not surfaced (see CANON_10 /
    # concepts_10.json). Slice both the concept list and activations to the
    # canonical width at read-out.
    concepts = [c for c in concepts if c in CANON_10]
    print(f"[infer] architecture mode={mode}  ({len(concepts)} concepts exposed)")

    class_prob, concept_act = run_model(model, args.image, device)
    concept_act = concept_act[:len(concepts)]
    pred_idx = int(np.argmax(class_prob))
    print("\n=== Predicted abnormality class ===")
    print(f"  {class_names[pred_idx]}  (p={class_prob[pred_idx]:.3f})")
    print("  full class distribution:")
    for name, p in sorted(zip(class_names, class_prob), key=lambda t: -t[1]):
        print(f"    {name:<18s} {p:.3f}")

    print("\n=== Model-predicted concept activations (learned concept head) ===")
    for c, a in zip(concepts, concept_act):
        print(f"    {c:<34s} {a:.3f}")

    print("\n=== Measured morphometry concepts (deterministic, label-free; 10 canonical) ===")
    morpho, err = run_morphometry(args.image)
    if morpho is None:
        print(f"    [segmentation failed: {err}] -- no measured concepts for this image")
    else:
        for c in CANON_10:
            v = morpho.get(c)
            print(f"    {c:<34s} {v:.3f}" if v is not None else f"    {c:<34s} n/a")


if __name__ == "__main__":
    main()
