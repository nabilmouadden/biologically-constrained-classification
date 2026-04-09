import pytest
import torch
from src.models.adaptive_threshold import AdaptiveThreshold


@pytest.fixture
def threshold():
    return AdaptiveThreshold(num_classes=7, base_threshold=0.5)


class TestAdaptiveThreshold:
    def test_output_shape(self, threshold):
        uncertainty = torch.rand(4, 7)
        probs = torch.rand(4, 7)
        out = threshold(uncertainty, probs)
        assert out.shape == (4, 7)

    def test_clamped_to_01(self, threshold):
        uncertainty = torch.rand(4, 7) * 10  # large values
        probs = torch.rand(4, 7) * 10
        out = threshold(uncertainty, probs)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_zero_uncertainty_zero_prob(self, threshold):
        """With default params (beta=0, delta=0), threshold equals alpha * base."""
        uncertainty = torch.zeros(2, 7)
        probs = torch.zeros(2, 7)
        out = threshold(uncertainty, probs)
        expected = threshold.threshold_alpha.data * 0.5
        assert torch.allclose(out, expected.unsqueeze(0).expand(2, -1).clamp(0, 1))

    def test_learnable_params_update(self, threshold):
        """Parameters should receive gradients during training."""
        uncertainty = torch.rand(4, 7)
        probs = torch.rand(4, 7)
        out = threshold(uncertainty, probs)
        loss = out.sum()
        loss.backward()
        assert threshold.threshold_alpha.grad is not None
        assert threshold.threshold_beta.grad is not None
        assert threshold.threshold_delta.grad is not None

    def test_batch_size_one(self, threshold):
        uncertainty = torch.rand(1, 7)
        probs = torch.rand(1, 7)
        out = threshold(uncertainty, probs)
        assert out.shape == (1, 7)

    def test_negative_alpha(self):
        """Negative alpha should still be clamped to [0, 1]."""
        at = AdaptiveThreshold(num_classes=3, base_threshold=0.5)
        at.threshold_alpha.data.fill_(-2.0)
        uncertainty = torch.zeros(2, 3)
        probs = torch.zeros(2, 3)
        out = at(uncertainty, probs)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_nonzero_beta_delta(self):
        """Verify uncertainty and probability components contribute."""
        at = AdaptiveThreshold(num_classes=3, base_threshold=0.5)
        at.threshold_alpha.data.fill_(0.0)
        at.threshold_beta.data.fill_(1.0)
        at.threshold_delta.data.fill_(0.0)
        uncertainty = torch.full((2, 3), 0.3)
        probs = torch.zeros(2, 3)
        out = at(uncertainty, probs)
        # threshold = 0*0.5 + 1*0.3 + 0*0 = 0.3
        assert torch.allclose(out, torch.full((2, 3), 0.3))

    def test_all_components_combined(self):
        """Verify formula: t = alpha*base + beta*U + delta*p, clamped."""
        at = AdaptiveThreshold(num_classes=2, base_threshold=0.4)
        at.threshold_alpha.data = torch.tensor([1.0, 0.5])
        at.threshold_beta.data = torch.tensor([0.2, 0.3])
        at.threshold_delta.data = torch.tensor([0.1, 0.1])
        uncertainty = torch.tensor([[0.5, 0.5]])
        probs = torch.tensor([[0.8, 0.8]])
        out = at(uncertainty, probs)
        expected = torch.tensor([[
            1.0 * 0.4 + 0.2 * 0.5 + 0.1 * 0.8,  # = 0.58
            0.5 * 0.4 + 0.3 * 0.5 + 0.1 * 0.8,  # = 0.43
        ]]).clamp(0, 1)
        assert torch.allclose(out, expected, atol=1e-5)
