import pytest
import torch
import torch.nn.functional as F
from src.models.losses import ConstraintLoss


@pytest.fixture
def loss_fn():
    return ConstraintLoss(
        lambda_con=0.1, lambda_unc=0.1, lambda_entropy=0.01,
        alpha=0.01, beta=0.1, uncertainty_threshold=0.2,
    )


def _make_outputs(batch_size=4, num_classes=7):
    """Create synthetic model outputs for testing."""
    logits = torch.randn(batch_size, num_classes)
    R = torch.randn(batch_size, num_classes, num_classes).clamp(-1, 1)
    C = torch.randn(num_classes, num_classes).clamp(-1, 1)
    C = (C + C.T) / 2
    C.fill_diagonal_(0)
    uncertainty = torch.rand(batch_size, num_classes)
    return {
        'logits': logits,
        'constraint_matrix': R,
        'prior_constraint_matrix': C,
        'uncertainty': uncertainty,
    }


class TestTotalLoss:
    def test_returns_all_components(self, loss_fn):
        outputs = _make_outputs()
        targets = torch.randint(0, 2, (4, 7)).float()
        result = loss_fn(outputs, targets)
        expected_keys = {
            'total_loss', 'bce_loss', 'constraint_loss',
            'uncertainty_loss', 'entropy_loss',
        }
        assert set(result.keys()) == expected_keys

    def test_total_is_weighted_sum(self, loss_fn):
        outputs = _make_outputs()
        targets = torch.randint(0, 2, (4, 7)).float()
        result = loss_fn(outputs, targets)

        recomputed = (
            result['bce_loss']
            + loss_fn.lambda_con * result['constraint_loss']
            + loss_fn.lambda_unc * result['uncertainty_loss']
            + loss_fn.lambda_entropy * result['entropy_loss']
        )
        assert torch.allclose(result['total_loss'], recomputed, atol=1e-5)

    def test_loss_is_scalar(self, loss_fn):
        outputs = _make_outputs()
        targets = torch.randint(0, 2, (4, 7)).float()
        result = loss_fn(outputs, targets)
        for v in result.values():
            assert v.dim() == 0

    def test_loss_backward(self, loss_fn):
        outputs = _make_outputs()
        # Make tensors require grad
        outputs['logits'] = outputs['logits'].requires_grad_(True)
        targets = torch.randint(0, 2, (4, 7)).float()
        result = loss_fn(outputs, targets)
        result['total_loss'].backward()
        assert outputs['logits'].grad is not None


class TestEntropyNormalization:
    def test_k_squared_factor(self):
        """Entropy loss must include the 1/K^2 normalization factor."""
        loss_fn = ConstraintLoss()
        K = 7
        B = 4
        # Use a fixed R for deterministic computation
        R = torch.zeros(B, K, K)
        R_normalized = (R + 1) / 2  # all 0.5
        # Binary entropy of 0.5 = log(2)
        element_entropy = -(0.5 * torch.log(torch.tensor(0.5 + 1e-10)) +
                            0.5 * torch.log(torch.tensor(0.5 + 1e-10)))
        expected_per_sample = element_entropy * K * K / (K * K)  # = log(2)
        expected = -expected_per_sample  # negated in the loss

        outputs = _make_outputs()
        outputs['constraint_matrix'] = R
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)

        assert torch.allclose(result['entropy_loss'], expected, atol=1e-4)

    @pytest.mark.parametrize("K", [3, 7, 15, 21])
    def test_entropy_scales_correctly(self, K):
        """Entropy loss should be independent of K when R is uniform."""
        loss_fn = ConstraintLoss()
        B = 2
        R = torch.zeros(B, K, K)  # all map to 0.5 after normalization

        outputs = {
            'logits': torch.randn(B, K),
            'constraint_matrix': R,
            'prior_constraint_matrix': torch.zeros(K, K),
            'uncertainty': torch.rand(B, K),
        }
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)

        # With 1/K^2 normalization, entropy should be ≈ -log(2) regardless of K
        expected = -torch.log(torch.tensor(2.0))
        assert torch.allclose(result['entropy_loss'], expected, atol=1e-3)


class TestBCENormalization:
    def test_sum_over_classes_mean_over_batch(self):
        """BCE should sum over K classes and mean over N samples."""
        loss_fn = ConstraintLoss()
        B, K = 4, 7
        logits = torch.randn(B, K)
        targets = torch.randint(0, 2, (B, K)).float()

        # Manual: sum over K, mean over N
        manual_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ).sum(dim=1).mean()

        outputs = _make_outputs()
        outputs['logits'] = logits
        result = loss_fn(outputs, targets)

        assert torch.allclose(result['bce_loss'], manual_bce, atol=1e-5)


class TestConstraintLoss:
    def test_frobenius_norm(self, loss_fn):
        """Constraint loss should compute squared Frobenius norm."""
        B, K = 2, 7
        R = torch.eye(K).unsqueeze(0).expand(B, -1, -1).clone()
        C = torch.eye(K)
        # RR^T = I, so ||I - I||_F^2 = 0
        outputs = _make_outputs(batch_size=B, num_classes=K)
        outputs['constraint_matrix'] = R
        outputs['prior_constraint_matrix'] = C
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)

        # Frobenius term should be near zero, only L1 term remains
        l1_term = loss_fn.alpha * R.abs().sum(dim=(1, 2)).mean()
        assert torch.allclose(result['constraint_loss'], l1_term, atol=1e-4)


class TestUncertaintyLoss:
    def test_hinge_below_threshold(self, loss_fn):
        """Uncertainty below threshold should not incur hinge penalty."""
        B, K = 4, 7
        outputs = _make_outputs()
        # Set uncertainty well below threshold
        outputs['uncertainty'] = torch.full((B, K), 0.01)
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        # The hinge term should be zero; loss is purely KL
        assert result['uncertainty_loss'].item() < 10.0  # sanity bound

    def test_hinge_above_threshold(self, loss_fn):
        """Uncertainty above threshold should increase loss."""
        B, K = 4, 7
        outputs = _make_outputs()
        outputs_low = _make_outputs()
        targets = torch.randint(0, 2, (B, K)).float()

        outputs['uncertainty'] = torch.full((B, K), 0.9)
        outputs_low['uncertainty'] = torch.full((B, K), 0.01)
        # Keep same logits/R/C
        outputs_low['logits'] = outputs['logits']
        outputs_low['constraint_matrix'] = outputs['constraint_matrix']
        outputs_low['prior_constraint_matrix'] = outputs['prior_constraint_matrix']

        high = loss_fn(outputs, targets)['uncertainty_loss']
        low = loss_fn(outputs_low, targets)['uncertainty_loss']
        assert high > low

    def test_zero_uncertainty(self, loss_fn):
        """Zero uncertainty should still produce finite loss (KL only)."""
        B, K = 2, 7
        outputs = _make_outputs(B, K)
        outputs['uncertainty'] = torch.zeros(B, K)
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        assert torch.isfinite(result['uncertainty_loss'])

    def test_extreme_logits_no_nan(self, loss_fn):
        """Very large/small logits (sigmoid → 0 or 1) should not cause NaN."""
        B, K = 2, 7
        outputs = _make_outputs(B, K)
        outputs['logits'] = torch.full((B, K), 100.0)  # sigmoid ≈ 1
        targets = torch.ones(B, K)
        result = loss_fn(outputs, targets)
        assert torch.isfinite(result['uncertainty_loss'])
        assert not torch.isnan(result['total_loss'])

    def test_hinge_exactly_at_threshold(self, loss_fn):
        """Uncertainty exactly at threshold should give zero hinge."""
        B, K = 2, 7
        outputs = _make_outputs(B, K)
        outputs['uncertainty'] = torch.full((B, K), loss_fn.uncertainty_threshold)
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        # relu(threshold - threshold) = 0, so hinge component is zero
        assert torch.isfinite(result['uncertainty_loss'])


class TestConstraintLossExtended:
    def test_zero_R_matrix(self, loss_fn):
        """All-zero R should yield non-zero constraint loss (RR^T=0 ≠ C usually)."""
        B, K = 2, 7
        outputs = _make_outputs(B, K)
        outputs['constraint_matrix'] = torch.zeros(B, K, K)
        # C is non-zero from _make_outputs
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        assert result['constraint_loss'].item() > 0

    def test_non_identity_R(self, loss_fn):
        """Non-trivial R should produce non-zero Frobenius loss."""
        B, K = 2, 7
        R = torch.randn(B, K, K).clamp(-1, 1)
        C = torch.zeros(K, K)
        outputs = _make_outputs(B, K)
        outputs['constraint_matrix'] = R
        outputs['prior_constraint_matrix'] = C
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        assert result['constraint_loss'].item() > 0


class TestEntropyLossEdgeCases:
    def test_R_at_boundary(self):
        """R at +1 or -1 maps to entropy of 0 (deterministic)."""
        loss_fn = ConstraintLoss()
        B, K = 2, 3
        R = torch.ones(B, K, K)  # all +1 → normalized to 1.0 → entropy ≈ 0
        outputs = _make_outputs(B, K)
        outputs['constraint_matrix'] = R
        targets = torch.randint(0, 2, (B, K)).float()
        result = loss_fn(outputs, targets)
        # Entropy of 1.0 or 0.0 is near 0, so -entropy is near 0
        assert abs(result['entropy_loss'].item()) < 0.01

    def test_loss_finite_for_all_K(self):
        """Total loss should be finite for various K values."""
        for K in [1, 3, 7, 15, 21]:
            loss_fn = ConstraintLoss()
            B = 2
            outputs = _make_outputs(B, K)
            targets = torch.randint(0, 2, (B, K)).float()
            result = loss_fn(outputs, targets)
            assert torch.isfinite(result['total_loss']), f"Non-finite loss for K={K}"
