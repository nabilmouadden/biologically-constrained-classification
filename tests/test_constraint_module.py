import pytest
import torch
from src.models.constraint_module import ConstraintModule


@pytest.fixture
def module():
    return ConstraintModule(num_classes=7, feature_dim=64)


@pytest.fixture
def module_with_prior():
    prior = torch.randn(7, 7)
    prior = (prior + prior.T) / 2  # symmetric
    prior.fill_diagonal_(0)
    prior = prior.clamp(-1, 1)
    return ConstraintModule(num_classes=7, feature_dim=64, prior_constraint_matrix=prior)


class TestConstraintModuleForward:
    def test_output_keys(self, module):
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        expected_keys = {
            'logits', 'probs', 'predictions', 'uncertainty',
            'constraint_matrix', 'thresholds', 'prior_constraint_matrix',
        }
        assert set(out.keys()) == expected_keys

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shapes(self, module, batch_size):
        x = torch.randn(batch_size, 64)
        out = module(x, training=True, mc_samples=3)

        K = 7
        assert out['logits'].shape == (batch_size, K)
        assert out['probs'].shape == (batch_size, K)
        assert out['predictions'].shape == (batch_size, K)
        assert out['uncertainty'].shape == (batch_size, K)
        assert out['constraint_matrix'].shape == (batch_size, K, K)
        assert out['thresholds'].shape == (batch_size, K)

    def test_3d_input(self, module):
        """Features with patch/token dimension [B, N, D]."""
        x = torch.randn(4, 16, 64)
        out = module(x, training=True, mc_samples=3)
        assert out['logits'].shape == (4, 7)

    def test_constraint_matrix_range(self, module):
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        R = out['constraint_matrix']
        assert R.min() >= -1.0
        assert R.max() <= 1.0

    def test_probs_range(self, module):
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        assert (out['probs'] >= 0).all()
        assert (out['probs'] <= 1).all()

    def test_predictions_are_binary(self, module):
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        unique = out['predictions'].unique()
        assert all(v in [0.0, 1.0] for v in unique)

    def test_uncertainty_non_negative(self, module):
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        assert (out['uncertainty'] >= 0).all()


class TestMCDropout:
    def test_different_samples_differ(self, module):
        """MC dropout should produce variability."""
        torch.manual_seed(0)
        x = torch.randn(4, 64)
        out1 = module(x, training=True, mc_samples=50)
        out2 = module(x, training=True, mc_samples=50)
        # With different random draws the logits should not be identical
        # (there's an astronomically small chance they would be)
        assert not torch.allclose(out1['logits'], out2['logits'])

    def test_inference_more_samples(self, module):
        """Inference mode should use the full mc_samples count."""
        x = torch.randn(2, 64)
        # Just ensure no error with large mc_samples
        out = module(x, training=False, mc_samples=100)
        assert out['logits'].shape == (2, 7)


class TestWithPrior:
    def test_prior_stored(self, module_with_prior):
        assert module_with_prior.prior_constraints.shape == (7, 7)

    def test_prior_returned(self, module_with_prior):
        x = torch.randn(2, 64)
        out = module_with_prior(x, training=True, mc_samples=3)
        assert torch.equal(out['prior_constraint_matrix'], module_with_prior.prior_constraints)


class TestNoPrior:
    def test_default_prior_is_zeros(self, module):
        """When no prior is given, prior_constraints should be all zeros."""
        assert torch.allclose(module.prior_constraints, torch.zeros(7, 7))


class TestConstraintMatrixGeneration:
    def test_scaling_centering(self, module):
        """Constraint matrix should be centered and scaled to [-1, 1]."""
        x = torch.randn(4, 64)
        out = module(x, training=True, mc_samples=3)
        R = out['constraint_matrix']
        # max abs value should be close to 1 (scaling maps max to 1)
        for i in range(R.shape[0]):
            max_abs = R[i].abs().max().item()
            assert max_abs <= 1.0
            assert max_abs > 0.5  # should be scaled up close to 1

    def test_different_inputs_different_R(self, module):
        """Different inputs should produce different constraint matrices."""
        x1 = torch.randn(2, 64)
        x2 = torch.randn(2, 64) + 5.0
        out1 = module(x1, training=True, mc_samples=3)
        out2 = module(x2, training=True, mc_samples=3)
        assert not torch.allclose(out1['constraint_matrix'], out2['constraint_matrix'])

    def test_uniform_features_no_nan(self, module):
        """Constant features should not cause NaN (tests the 1e-8 guard)."""
        x = torch.ones(2, 64)
        out = module(x, training=True, mc_samples=3)
        assert not torch.isnan(out['constraint_matrix']).any()


class TestMCForwardEdgeCases:
    def test_mc_samples_one(self, module):
        """mc_samples=1 should still produce valid output."""
        x = torch.randn(2, 64)
        out = module(x, training=True, mc_samples=1)
        assert out['logits'].shape == (2, 7)
        # With 1 sample, uncertainty should be zero (no variance)
        assert torch.allclose(out['uncertainty'], torch.zeros(2, 7))

    def test_training_caps_mc_samples(self, module):
        """In training, mc_samples is capped at 5."""
        x = torch.randn(2, 64)
        # This should run with min(5, 100) = 5 samples internally
        out = module(x, training=True, mc_samples=100)
        assert out['logits'].shape == (2, 7)

    def test_mc_uncertainty_increases_with_dropout(self):
        """Higher dropout rate should generally yield higher uncertainty."""
        mod_low = ConstraintModule(num_classes=7, feature_dim=64, dropout_rate=0.1)
        mod_high = ConstraintModule(num_classes=7, feature_dim=64, dropout_rate=0.9)
        # Copy weights so the only difference is dropout rate
        mod_high.load_state_dict(mod_low.state_dict(), strict=False)

        torch.manual_seed(42)
        x = torch.randn(8, 64)
        out_low = mod_low(x, training=False, mc_samples=100)
        torch.manual_seed(42)
        out_high = mod_high(x, training=False, mc_samples=100)
        # On average, higher dropout should yield higher uncertainty
        assert out_high['uncertainty'].mean() > out_low['uncertainty'].mean()
