"""
tests.test_model
===================

Model-level tests: forward-pass output shapes for PS-GNN and PS-ViT,
GAT attention-weight normalization, and gradient flow through
:class:`~ps_gnn.models.losses.PhysicsInformedPSLoss`.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")  # ps_gnn.models.ps_gnn imports this at module level

from ps_gnn.models.losses import PhysicsInformedPSLoss, PhysicsInformedPSLossConfig
from ps_gnn.models.ps_gnn import PSGNN, PSGNNConfig
from ps_gnn.models.ps_vit import PSViT, PSViTConfig


class TestPSGNN:
    """Tests for :class:`ps_gnn.models.ps_gnn.PSGNN`."""

    def test_forward_output_shapes(self):
        model = PSGNN()
        n_nodes, n_edges = 50, 200
        x = torch.randn(n_nodes, 19)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        logits, attention = model(x, edge_index, return_attention=True)

        assert logits.shape == (n_nodes, 2)
        assert attention is not None
        assert attention.shape[1] == 1

    def test_forward_without_attention(self):
        model = PSGNN()
        n_nodes, n_edges = 30, 100
        x = torch.randn(n_nodes, 19)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        logits, attention = model(x, edge_index, return_attention=False)
        assert logits.shape == (n_nodes, 2)
        assert attention is None

    def test_attention_weights_sum_to_one_per_target_node(self):
        """GAT softmax attention must sum to ~1.0 over each target node's incoming edges."""
        model = PSGNN()
        n_nodes, n_edges = 60, 240
        x = torch.randn(n_nodes, 19)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        model.eval()
        with torch.no_grad():
            h = model.encoder(x.to(model.device))
            h = torch.relu(model.bn1(model.conv1(h, edge_index.to(model.device))))
            _, (attn_edge_index, attn_weights) = model.conv2(
                h, edge_index.to(model.device), return_attention_weights=True
            )
            attn_mean = attn_weights.mean(dim=-1)
            target = attn_edge_index[1]
            sums = torch.zeros(n_nodes, device=attn_mean.device).scatter_add_(0, target, attn_mean)

        np.testing.assert_allclose(sums.numpy(), np.ones(n_nodes), atol=1e-4)

    def test_encode_matches_forward_classifier_composition(self):
        """PSGNN.encode() followed by the classifier head must equal forward()'s logits."""
        model = PSGNN()
        model.eval()
        n_nodes, n_edges = 40, 150
        x = torch.randn(n_nodes, 19)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        with torch.no_grad():
            logits_direct, _ = model(x, edge_index, return_attention=False)
            hidden, _ = model.encode(x, edge_index, return_attention=False)
            logits_via_encode = model.classifier(hidden)

        torch.testing.assert_close(logits_direct, logits_via_encode)

    def test_gradient_flow(self):
        model = PSGNN()
        n_nodes, n_edges = 30, 100
        x = torch.randn(n_nodes, 19, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        logits, _ = model(x, edge_index, return_attention=False)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    def test_custom_config(self):
        config = PSGNNConfig(
            in_channels=19, gcn_hidden=64, gat_hidden=64, gat_heads=4, sage_hidden=32
        )
        model = PSGNN(config)
        x = torch.randn(10, 19)
        edge_index = torch.randint(0, 10, (2, 30))
        logits, _ = model(x, edge_index, return_attention=False)
        assert logits.shape == (10, 2)


class TestPSViT:
    """Tests for :class:`ps_gnn.models.ps_vit.PSViT`."""

    def test_forward_output_shape(self):
        model = PSViT()
        x = torch.randn(2, 10, 2, 32, 32)  # (B, T, C, H, W)
        logits = model(x)
        assert logits.shape == (2, 2)

    def test_forward_handles_non_patch_multiple_input(self):
        """Input dimensions not divisible by patch_size must be zero-padded, not error."""
        model = PSViT()
        x = torch.randn(1, 5, 2, 30, 35)  # not divisible by patch_size=8
        logits = model(x)
        assert logits.shape == (1, 2)

    def test_gradient_flow(self):
        model = PSViT()
        x = torch.randn(1, 6, 2, 24, 24, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_custom_config(self):
        config = PSViTConfig(embed_dim=32, lstm_hidden=64, transformer_dim=64, transformer_layers=2)
        model = PSViT(config)
        x = torch.randn(1, 4, 2, 16, 16)
        logits = model(x)
        assert logits.shape == (1, 2)


class TestPhysicsInformedPSLoss:
    """Tests for :class:`ps_gnn.models.losses.PhysicsInformedPSLoss`."""

    def _make_batch(self, n_nodes=40, n_edges=150, t_steps=10):
        model = PSGNN()
        x = torch.randn(n_nodes, 19, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        phase = torch.randn(t_steps, n_nodes) * 0.3
        y = torch.randint(0, 2, (n_nodes,))
        return model, x, edge_index, phase, y

    def test_total_equals_weighted_sum_of_components(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)
        hidden, _ = model.encode(x, edge_index, return_attention=False)

        loss_fn = PhysicsInformedPSLoss()
        out = loss_fn(logits, y, phase, edge_index, hidden=hidden)

        expected = 0.7 * out.cross_entropy + 0.2 * out.phase_stability + 0.1 * out.spatial_coherence
        torch.testing.assert_close(out.total, expected)

    def test_gradients_flow_to_input_features(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)
        hidden, _ = model.encode(x, edge_index, return_attention=False)

        loss_fn = PhysicsInformedPSLoss()
        out = loss_fn(logits, y, phase, edge_index, hidden=hidden)
        out.total.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0

    def test_gradients_flow_to_model_parameters(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)
        hidden, _ = model.encode(x, edge_index, return_attention=False)

        loss_fn = PhysicsInformedPSLoss()
        out = loss_fn(logits, y, phase, edge_index, hidden=hidden)
        out.total.backward()

        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert sum(grad_norms) > 0

    def test_missing_hidden_raises_when_spatial_loss_enabled(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)

        loss_fn = PhysicsInformedPSLoss()
        with pytest.raises(ValueError):
            loss_fn(logits, y, phase, edge_index, hidden=None)

    def test_ablation_toggles_disable_components(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)

        config = PhysicsInformedPSLossConfig(use_phase_loss=False, use_spatial_loss=False)
        loss_fn = PhysicsInformedPSLoss(config)
        out = loss_fn(logits, y, phase, edge_index, hidden=None)

        torch.testing.assert_close(out.total, 0.7 * out.cross_entropy)

    def test_all_unlabeled_batch_does_not_produce_nan(self):
        model, x, edge_index, phase, y = self._make_batch()
        logits, _ = model(x, edge_index, return_attention=False)
        hidden, _ = model.encode(x, edge_index, return_attention=False)

        y_unlabeled = torch.full_like(y, -1)
        loss_fn = PhysicsInformedPSLoss()
        out = loss_fn(logits, y_unlabeled, phase, edge_index, hidden=hidden)

        assert torch.isfinite(out.total)
        assert out.cross_entropy.item() == 0.0

    def test_empty_edge_index_gives_zero_spatial_coherence(self):
        model, x, _, phase, y = self._make_batch()
        empty_edges = torch.zeros((2, 0), dtype=torch.long)
        logits, _ = model(x, empty_edges, return_attention=False)
        hidden, _ = model.encode(x, empty_edges, return_attention=False)

        loss_fn = PhysicsInformedPSLoss()
        out = loss_fn(logits, y, phase, empty_edges, hidden=hidden)
        assert out.spatial_coherence.item() == 0.0

    def test_circular_variance_handles_wrap_boundary(self):
        """Phase samples straddling the +/-pi wrap boundary must be treated
        as physically close (low circular variance), not as wildly unstable
        (the way a linear standard deviation would incorrectly report)."""
        loss_fn = PhysicsInformedPSLoss()

        # A single node, two acquisitions, with phase sitting right at the
        # wrap boundary: 3.1 and -3.1 rad are only ~0.08 rad apart on the
        # unit circle, even though they differ by ~6.2 rad on the real line.
        phase = torch.tensor([[3.1], [-3.1]], requires_grad=True)  # shape (T=2, N=1)

        # Force the softmax probability of the PS class to ~1.0 for this
        # node, so the probability-weighted mean inside _phase_stability
        # collapses to (approximately) this single node's own circular
        # variance value -- letting us assert on it directly.
        logits = torch.tensor([[-10.0, 10.0]])  # (N=1, n_classes=2), heavily favors class 1

        circular_loss = loss_fn._phase_stability(logits, phase)

        # The old linear std dev (population, ddof=0) for these two
        # symmetric samples is exactly 3.1 -- huge, and physically wrong.
        linear_std_equivalent = phase.detach().std(dim=0, unbiased=False).mean()
        assert linear_std_equivalent.item() == pytest.approx(3.1, abs=1e-4)

        # The new circular variance correctly reports near-zero dispersion.
        assert circular_loss.item() < 0.1
        assert circular_loss.item() < linear_std_equivalent.item()

        # Gradients must still flow through the circular variance calculation.
        circular_loss.backward()
        assert phase.grad is not None
        assert torch.isfinite(phase.grad).all()
        assert phase.grad.abs().sum() > 0