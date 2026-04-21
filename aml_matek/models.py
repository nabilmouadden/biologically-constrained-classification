"""Concept adapter, constraint module, and joint model.

Operates on PRE-CACHED features of shape (B, 1+P, d).
Backbone is never touched — features are inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptAdapter(nn.Module):
    """K learnable concept queries cross-attend to patch tokens.

    Input:  patch tokens (B, P, d)
    Output: concept logits (B, K) — apply sigmoid externally (we return logits for BCEWithLogitsLoss).
    Also stores the most-recent attention weights in self.last_attn (B, K, P)
    for visualization.
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
        # patch_tokens: (B, P, d)
        B = patch_tokens.shape[0]
        kv = self.proj(patch_tokens)                               # (B, P, concept_dim)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # (B, K, concept_dim)
        # attn_output: (B, K, concept_dim), attn_weights: (B, K, P)
        attn_out, attn_w = self.mha(q, kv, kv, need_weights=True, average_attn_weights=True)
        self.last_attn = attn_w.detach()
        logits = self.head(attn_out).squeeze(-1)                    # (B, K)
        return logits


class ConstraintModule(nn.Module):
    """Biological-constraint regularization with TWO components.

    1. R-matching loss: ||R R^T - C||_F^2 where R is a learned (K, K) matrix
       and C is the prior constraint matrix. This is gradient-isolated from the
       concept adapter (R is a separate parameter).

    2. Violation loss: a direct penalty on co-activation of mutually exclusive
       concept pairs, computed from the CURRENT concept logits. This term
       backpropagates INTO the concept adapter and is the mechanism that actually
       reduces prediction violations.

    The plan's original spec (v1) had only component 1 and explicitly forbade
    applying R to the concept logits. That made the "constrained" vs
    "unconstrained" ablation degenerate (adapter gradient paths identical).
    Component 2 was added to make the ablation meaningful — it does not apply
    R to the logits; it directly penalizes biologically impossible co-predictions.
    """

    def __init__(self, num_concepts: int, prior_C: torch.Tensor,
                 exclusive_pairs: list[tuple[int, int]] | None = None):
        super().__init__()
        assert prior_C.shape == (num_concepts, num_concepts)
        self.R = nn.Parameter(torch.randn(num_concepts, num_concepts) * 0.01)
        self.register_buffer("C", prior_C.clone())
        if exclusive_pairs:
            ei = torch.tensor([p[0] for p in exclusive_pairs], dtype=torch.long)
            ej = torch.tensor([p[1] for p in exclusive_pairs], dtype=torch.long)
        else:
            ei = torch.zeros(0, dtype=torch.long)
            ej = torch.zeros(0, dtype=torch.long)
        self.register_buffer("excl_i", ei)
        self.register_buffer("excl_j", ej)

    def R_matching_loss(self):
        return torch.linalg.norm(self.R @ self.R.T - self.C, ord="fro") ** 2

    def violation_loss(self, concept_logits):
        if self.excl_i.numel() == 0:
            return concept_logits.new_zeros(())
        p = concept_logits.sigmoid()                           # (B, K)
        pair = p[:, self.excl_i] * p[:, self.excl_j]           # (B, n_pairs)
        # Sum over pairs (total penalty grows with vocabulary of constraints),
        # mean over the batch. Gradient flows into the adapter through sigmoid(concept_logits).
        return pair.sum(dim=-1).mean()

    def constraint_loss(self, concept_logits):
        """Combined regularizer used during training.

        With lambda=0 (unconstrained ablation), both terms are zeroed out in the
        total loss, so the adapter is untouched by constraints.
        """
        return self.R_matching_loss() + self.violation_loss(concept_logits)


class JointModel(nn.Module):
    """Combines concept adapter + classifier over cached backbone features.

    Forward expects features of shape (B, 1+P, d):
      index 0 = CLS (ViT) or GAP vector (ResNet, synthesized by cache_features.py).
      indices 1..P = patch / spatial tokens.
    """

    def __init__(self, feature_dim: int, num_concepts: int, num_classes: int,
                 prior_C: torch.Tensor,
                 exclusive_pairs: list[tuple[int, int]] | None = None,
                 concept_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.concept_adapter = ConceptAdapter(feature_dim, num_concepts, concept_dim, num_heads)
        self.constraint_module = ConstraintModule(num_concepts, prior_C, exclusive_pairs)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        cls = features[:, 0, :]          # (B, d)
        patches = features[:, 1:, :]     # (B, P, d)
        class_logits = self.classifier(cls)
        concept_logits = self.concept_adapter(patches)
        return class_logits, concept_logits


def build_prior_C(concepts: list[str], spec: dict) -> torch.Tensor:
    """Convert the JSON mutually_exclusive / cooccurring pair list into a symmetric (K,K) tensor.

    spec = concept_config["concept_constraint_matrix"]
    """
    idx = {c: i for i, c in enumerate(concepts)}
    K = len(concepts)
    C = torch.zeros(K, K, dtype=torch.float32)
    for pair in spec.get("mutually_exclusive_pairs", []):
        a, b = pair["concepts"]
        v = float(pair["value"])
        C[idx[a], idx[b]] = v
        C[idx[b], idx[a]] = v
    for pair in spec.get("cooccurring_pairs", []):
        a, b = pair["concepts"]
        v = float(pair["value"])
        C[idx[a], idx[b]] = v
        C[idx[b], idx[a]] = v
    # Diagonal: each concept has correlation 1 with itself.
    for i in range(K):
        C[i, i] = 1.0
    return C


def build_class_concept_targets(matrix: list[list[float]]) -> torch.Tensor:
    """Return (num_classes, num_concepts) float32 soft-label tensor."""
    return torch.tensor(matrix, dtype=torch.float32)
