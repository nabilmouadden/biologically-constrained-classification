import pytest
import torch
import torch.nn as nn
from src.models import create_model, ConstraintLoss
from src.models.constraint_priors import get_constraint_matrix


class _MockBackbone(nn.Module):
    """Simple backbone that returns fixed-dim features."""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, feature_dim)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


@pytest.fixture
def model_and_loss():
    backbone = _MockBackbone(feature_dim=64)
    prior = get_constraint_matrix('gr_neutro')
    model, loss_fn = create_model(
        backbone=backbone,
        num_classes=7,
        feature_dim=64,
        prior_constraint_matrix=prior,
    )
    return model, loss_fn


class TestCreateModel:
    def test_returns_model_and_loss(self):
        backbone = _MockBackbone(64)
        model, loss_fn = create_model(backbone=backbone, num_classes=7, feature_dim=64)
        assert isinstance(loss_fn, ConstraintLoss)
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'constraint_module')

    def test_loss_config_passthrough(self):
        backbone = _MockBackbone(64)
        _, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            loss_config={'lambda_con': 0.5, 'lambda_unc': 0.2}
        )
        assert loss_fn.lambda_con == 0.5
        assert loss_fn.lambda_unc == 0.2

    def test_default_loss_config(self):
        backbone = _MockBackbone(64)
        _, loss_fn = create_model(backbone=backbone, num_classes=7, feature_dim=64)
        assert loss_fn.lambda_con == 0.1  # default


class TestFullForwardPass:
    def test_training_forward(self, model_and_loss):
        model, loss_fn = model_and_loss
        model.train()
        x = torch.randn(4, 3, 16, 16)
        out = model(x, training=True, mc_samples=3)
        assert out['logits'].shape == (4, 7)

    def test_inference_forward(self, model_and_loss):
        model, _ = model_and_loss
        model.eval()
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            out = model(x, training=False, mc_samples=10)
        assert out['predictions'].shape == (2, 7)

    def test_training_step_with_backward(self, model_and_loss):
        model, loss_fn = model_and_loss
        model.train()
        x = torch.randn(4, 3, 16, 16)
        targets = torch.randint(0, 2, (4, 7)).float()

        out = model(x, training=True, mc_samples=3)
        loss_dict = loss_fn(out, targets)

        loss_dict['total_loss'].backward()

        # Check that core constraint module params received gradients
        # (adaptive_threshold params don't get gradients from the loss since
        # thresholds are only used for predictions, not in the loss computation)
        for name, param in model.constraint_module.named_parameters():
            if param.requires_grad and 'adaptive_threshold' not in name:
                assert param.grad is not None, f"No gradient for {name}"

    def test_inference_vs_training_mode(self, model_and_loss):
        model, _ = model_and_loss
        x = torch.randn(2, 3, 16, 16)

        model.train()
        out_train = model(x, training=True, mc_samples=3)

        model.eval()
        with torch.no_grad():
            out_eval = model(x, training=False, mc_samples=3)

        # Both should produce valid output shapes
        assert out_train['logits'].shape == out_eval['logits'].shape


class TestCreateModelEdgeCases:
    def test_no_prior_matrix(self):
        """Model works without prior constraint matrix."""
        backbone = _MockBackbone(64)
        model, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            prior_constraint_matrix=None,
        )
        x = torch.randn(2, 3, 16, 16)
        out = model(x, training=True, mc_samples=3)
        assert out['logits'].shape == (2, 7)
        # Prior should be all zeros
        assert torch.allclose(out['prior_constraint_matrix'], torch.zeros(7, 7))

    def test_loss_config_none_uses_defaults(self):
        """Explicitly passing loss_config=None should use defaults."""
        backbone = _MockBackbone(64)
        _, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            loss_config=None,
        )
        assert loss_fn.lambda_con == 0.1
        assert loss_fn.lambda_unc == 0.1
        assert loss_fn.lambda_entropy == 0.01

    def test_different_num_classes(self):
        """Model should work with different class counts."""
        for K in [3, 7, 15, 21]:
            backbone = _MockBackbone(64)
            prior = torch.zeros(K, K)
            model, loss_fn = create_model(
                backbone=backbone, num_classes=K, feature_dim=64,
                prior_constraint_matrix=prior,
            )
            x = torch.randn(2, 3, 16, 16)
            out = model(x, training=True, mc_samples=3)
            assert out['logits'].shape == (2, K)
            assert out['constraint_matrix'].shape == (2, K, K)

    def test_end_to_end_train_val(self):
        """Full train step followed by eval step."""
        backbone = _MockBackbone(64)
        prior = get_constraint_matrix('gr_neutro')
        model, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            prior_constraint_matrix=prior,
            loss_config={'lambda_con': 0.1, 'lambda_unc': 0.1, 'lambda_entropy': 0.01},
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train step
        model.train()
        x = torch.randn(4, 3, 16, 16)
        targets = torch.randint(0, 2, (4, 7)).float()
        out = model(x, training=True, mc_samples=3)
        loss = loss_fn(out, targets)['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval step
        model.eval()
        with torch.no_grad():
            out_eval = model(x, training=False, mc_samples=10)
        assert out_eval['predictions'].shape == (4, 7)
        assert (out_eval['predictions'] >= 0).all()
        assert (out_eval['predictions'] <= 1).all()
