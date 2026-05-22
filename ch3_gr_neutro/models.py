"""GR-Neutro joint model: DinoBloom-S backbone (partial fine-tune)
   + multi-label classifier on CLS  + concept adapter on patch tokens
   + constraint module (R-matching + soft cooccur).

Differs from thesis/ch3/models.py in three ways:
  1. Backbone is loaded inside the model (live forward, no cached features).
  2. Classifier is multi-label (sigmoid logits, BCE), not multi-class softmax.
  3. ConstraintModule uses a SOFT cooccur loss -<C, mean p p^T>_F instead of
     hard mutex penalties, because the GR-Neutro concept matrix has no -1 entries.

Shape contract: forward returns (class_logits, concept_logits, attention_weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


_DINOBLOOM_HF = {
    "dinobloom_s": "hf-hub:1aurent/vit_small_patch14_224.dinobloom",
    "dinobloom_b": "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
    "dinobloom_l": "hf-hub:1aurent/vit_large_patch14_224.dinobloom",
}


class DinoBloomBackbone(nn.Module):
    """timm DinoBloom variant; returns full token sequence (B, 1+P, d).

    `unfreeze_last_n` last transformer blocks are trainable; everything earlier
    is frozen. Final norm is also trainable when any block is. Mirrors the
    setup in thesis/gr_neutro/train_live.py.
    """

    def __init__(self, variant: str = "dinobloom_s", unfreeze_last_n: int = 6):
        super().__init__()
        if variant not in _DINOBLOOM_HF:
            raise ValueError(f"Unknown DinoBloom variant: {variant}")
        self.model = timm.create_model(_DINOBLOOM_HF[variant], pretrained=True, img_size=224)
        self.embed_dim = self.model.embed_dim
        for p in self.model.parameters():
            p.requires_grad_(False)
        n_blocks = len(self.model.blocks)
        if unfreeze_last_n > 0:
            for blk in self.model.blocks[max(0, n_blocks - unfreeze_last_n):]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.model, "norm"):
                for p in self.model.norm.parameters():
                    p.requires_grad_(True)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        self._unfreeze_n = unfreeze_last_n
        self._n_blocks = n_blocks
        print(f"[backbone] {variant}: {n_train:,}/{n_total:,} trainable "
              f"(last {unfreeze_last_n}/{n_blocks} blocks)")

    def forward(self, x):
        return self.model.forward_features(x)   # (B, 1+P, d)


class ConceptAdapter(nn.Module):
    """K learnable concept queries cross-attend to patch tokens.

    Identical structure to thesis/ch3/models.py::ConceptAdapter — preserved on
    purpose so the architecture matches the AML Matek experiments.
    """

    def __init__(self, feature_dim: int, num_concepts: int,
                 concept_dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert concept_dim % num_heads == 0
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.proj = nn.Linear(feature_dim, concept_dim)
        self.queries = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.02)
        self.mha = nn.MultiheadAttention(concept_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, 1),
        )
        self.last_attn = None

    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        kv = self.proj(patch_tokens)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_w = self.mha(q, kv, kv, need_weights=True, average_attn_weights=True)
        self.last_attn = attn_w.detach()
        logits = self.head(attn_out).squeeze(-1)
        return logits


class ConstraintModule(nn.Module):
    """R-matching loss + soft cooccur penalty over predicted concept probs.

    GR-Neutro's prior C has no -1 entries, so we replace ch3's hard-mutex
    `violation_loss` with a SOFT cooccur term that uses C[i,j] directly.
    """

    def __init__(self, num_concepts: int, prior_C: torch.Tensor):
        super().__init__()
        assert prior_C.shape == (num_concepts, num_concepts)
        self.R = nn.Parameter(torch.randn(num_concepts, num_concepts) * 0.01)
        self.register_buffer("C", prior_C.clone())
        self.register_buffer("eye", torch.eye(num_concepts))

    def R_matching_loss(self):
        return torch.linalg.norm(self.R @ self.R.T - self.C, ord="fro") ** 2

    def soft_cooccur_loss(self, concept_logits):
        """L_cooc = -<C_off, mean_b p_b p_b^T>_F.

        For C[i,j] > 0: encourages high p_i*p_j (cooccurrence).
        For C[i,j] < 0: penalizes high p_i*p_j (anti-cooccurrence).
        Off-diagonal only so we don't reward p_i^2.
        """
        p = concept_logits.sigmoid()
        Cm = self.C * (1.0 - self.eye)
        pp_mean = (p.unsqueeze(2) * p.unsqueeze(1)).mean(dim=0)
        return -(Cm * pp_mean).sum()

    def constraint_loss(self, concept_logits):
        return self.R_matching_loss() + self.soft_cooccur_loss(concept_logits)

    def violation_rate(self, concept_logits, threshold: float = 0.5,
                       tol: float = -0.5) -> tuple[float, dict]:
        """Empirical mutex violation rate (no gradient).

        Although the prior C has no entries below `tol` by default, this is
        useful for ablation experiments that override C with hard mutex pairs.
        Reports overall rate plus per-pair counts.
        """
        with torch.no_grad():
            mutex_mask = (self.C < tol).triu(diagonal=1)
            ai, aj = mutex_mask.nonzero(as_tuple=True)
            if ai.numel() == 0:
                return 0.0, {}
            pred = (concept_logits.sigmoid() >= threshold).float()
            both = (pred[:, ai] * pred[:, aj])
            per_pair = both.sum(dim=0).cpu().numpy()
            rate = float(both.any(dim=1).float().mean().item())
            return rate, {"pair_indices": list(zip(ai.tolist(), aj.tolist())),
                          "violations_per_pair": per_pair.tolist()}


class JointModel(nn.Module):
    """Backbone + multi-label classifier on CLS + concept adapter on patches.

    Forward: returns (class_logits, concept_logits) — both pre-sigmoid for
    BCEWithLogitsLoss. Attention weights live in `concept_adapter.last_attn`.
    """

    def __init__(self, backbone: DinoBloomBackbone, num_concepts: int,
                 num_classes: int, prior_C: torch.Tensor,
                 concept_dim: int = 128, num_heads: int = 4,
                 classifier_dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        d = backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(classifier_dropout),
            nn.Linear(d, num_classes),
        )
        self.concept_adapter = ConceptAdapter(d, num_concepts, concept_dim, num_heads)
        self.constraint_module = ConstraintModule(num_concepts, prior_C)

    def forward(self, x):
        feats = self.backbone(x)              # (B, 1+P, d)
        cls = feats[:, 0, :]
        patches = feats[:, 1:, :]
        class_logits = self.classifier(cls)
        concept_logits = self.concept_adapter(patches)
        return class_logits, concept_logits


def build_prior_C(concepts: list[str], spec: dict) -> torch.Tensor:
    """Concept-config JSON -> symmetric (K, K) tensor with diagonal=1.

    Reads `mutually_exclusive_pairs`, `negative_associated_pairs`, and
    `cooccurring_pairs` (extending ch3's 2-list schema with one for soft
    negatives). All three contribute to C; ordering doesn't matter.
    """
    idx = {c: i for i, c in enumerate(concepts)}
    K = len(concepts)
    C = torch.zeros(K, K, dtype=torch.float32)
    for key in ("mutually_exclusive_pairs", "negative_associated_pairs", "cooccurring_pairs"):
        for pair in spec.get(key, []):
            a, b = pair["concepts"]
            v = float(pair["value"])
            C[idx[a], idx[b]] = v
            C[idx[b], idx[a]] = v
    for i in range(K):
        C[i, i] = 1.0
    return C


def build_class_concept_targets(matrix: list[list[float]]) -> torch.Tensor:
    """(num_classes, num_concepts) float32 tensor of soft labels."""
    return torch.tensor(matrix, dtype=torch.float32)


@torch.no_grad()
def aggregate_concept_target(class_concept: torch.Tensor,
                             multilabel: torch.Tensor,
                             normal_idx: int = 0) -> torch.Tensor:
    """Per-image concept target via max-deviation-from-Normal aggregation.

    For each cell and each concept k, pick the active class i (label==1) whose
    |class_concept[i,k] - class_concept[normal_idx,k]| is largest, and emit
    that class's signed value. If no class is active (shouldn't happen in
    GR-Neutro, every cell has >=1 label), fall back to the Normal row.

    class_concept: (C, K)  multilabel: (B, C) {0,1}  -> (B, K)
    """
    baseline = class_concept[normal_idx]                 # (K,)
    dev = class_concept - baseline.unsqueeze(0)          # (C, K) signed
    abs_dev = dev.abs()                                  # (C, K)
    B = multilabel.shape[0]
    abs_exp = abs_dev.unsqueeze(0).expand(B, -1, -1)     # (B, C, K)
    dev_exp = dev.unsqueeze(0).expand(B, -1, -1)         # (B, C, K)
    mask = multilabel.unsqueeze(-1).bool()               # (B, C, 1)
    masked_abs = torch.where(mask, abs_exp, torch.full_like(abs_exp, -1.0))
    has_active = multilabel.sum(dim=1) > 0               # (B,)
    best_idx = masked_abs.argmax(dim=1)                  # (B, K)
    best_dev = torch.gather(dev_exp, 1, best_idx.unsqueeze(1)).squeeze(1)
    target = baseline.unsqueeze(0) + best_dev
    # Where no class is active (shouldn't happen), fall back to baseline.
    if (~has_active).any():
        target[~has_active] = baseline
    return target.clamp(0.0, 1.0)
