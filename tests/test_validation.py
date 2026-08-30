"""
tests.test_validation
========================

Validation tests for :mod:`ps_gnn.validation.sbas_check`. Mocks a
"PS-GNN selection" as points drawn from genuinely lower-noise phase
series, and asserts the SBAS inversion correctly reports higher temporal
coherence for those points than for a randomly-selected point set drawn
from the full (noisier, unfiltered) candidate pool — mirroring the real
comparison the module performs against an ADI baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from ps_gnn.validation.sbas_check import (
    build_interferogram_network,
    run_sbas_validation,
    wls_sbas_inversion,
)


@pytest.fixture
def synthetic_network_phase():
    """A synthetic (T, N) phase stack with two point qualities:

    - Points ``[0:30]``: genuinely low-noise ("PS-GNN selected").
    - Points ``[30:100]``: the full noisier candidate pool ("random pool").

    Returns
    -------
    dict
        ``phase`` (T, N), ``good_mask`` (N,) bool for the low-noise
        points, ``pool_mask`` (N,) bool for the full candidate pool
        (including the good points, matching how a real ADI baseline
        selection is typically more inclusive/noisier on average).
    """
    rng = np.random.default_rng(7)
    t_steps, n_good, n_pool_extra = 14, 30, 70
    n_points = n_good + n_pool_extra

    true_velocity = rng.uniform(-0.12, 0.12, size=n_points)
    true_cumulative = np.outer(np.arange(t_steps), true_velocity)
    clean_phase = ((true_cumulative + np.pi) % (2 * np.pi)) - np.pi

    noise = np.zeros((t_steps, n_points))
    noise[:, :n_good] = rng.normal(0, 0.08, size=(t_steps, n_good))  # low noise
    noise[:, n_good:] = rng.normal(0, 1.3, size=(t_steps, n_pool_extra))  # high noise

    phase = clean_phase + noise

    good_mask = np.zeros(n_points, dtype=bool)
    good_mask[:n_good] = True

    pool_mask = np.ones(n_points, dtype=bool)  # the full pool, everyone

    return {"phase": phase, "good_mask": good_mask, "pool_mask": pool_mask}


class TestWLSSBASInversion:
    """Correctness tests for the from-scratch WLS SBAS solver."""

    def test_recovers_noise_free_ground_truth(self):
        rng = np.random.default_rng(0)
        t_steps, n_points = 10, 5
        # Keep displacement well within (-pi, pi) to avoid a genuine
        # phase-unwrapping ambiguity, which is a physical limitation of
        # single-reference wrapped phase, not a solver defect.
        velocity = rng.uniform(-0.15, 0.15, size=n_points)
        cumulative = np.outer(np.arange(t_steps), velocity)
        phase = ((cumulative + np.pi) % (2 * np.pi)) - np.pi

        inverted, residual_var, coherence = wls_sbas_inversion(phase, np.ones(n_points))

        np.testing.assert_allclose(inverted[-1], cumulative[-1], atol=1e-6)
        np.testing.assert_allclose(residual_var, 0.0, atol=1e-8)
        np.testing.assert_allclose(coherence, 1.0, atol=1e-6)

    def test_noisy_points_show_worse_inversion_quality(self):
        rng = np.random.default_rng(1)
        t_steps, n_points = 12, 4
        velocity = rng.uniform(-0.1, 0.1, size=n_points)
        cumulative = np.outer(np.arange(t_steps), velocity)
        clean_phase = ((cumulative + np.pi) % (2 * np.pi)) - np.pi

        phase = clean_phase.copy()
        phase[:, 2:] += rng.normal(0, 1.5, size=(t_steps, 2))  # points 2,3 are noisy

        _, residual_var, coherence = wls_sbas_inversion(phase, np.ones(n_points))

        assert residual_var[:2].mean() < residual_var[2:].mean()
        assert coherence[:2].mean() > coherence[2:].mean()

    def test_network_has_expected_number_of_pairs(self):
        network = build_interferogram_network(n_epochs=8)
        assert len(network.pairs) == 8 * 7 // 2  # all-pairs network
        assert network.design_matrix.shape == (28, 7)


class TestSBASValidationHypothesis:
    """Tests for :func:`run_sbas_validation`'s core hypothesis check."""

    def test_ps_gnn_selected_points_beat_random_pool(self, synthetic_network_phase):
        """The genuinely cleaner ('PS-GNN') point set should show higher
        temporal coherence than a random sample from the full noisy pool,
        by more than the project's 15% hypothesis threshold."""
        phase = synthetic_network_phase["phase"]
        good_mask = synthetic_network_phase["good_mask"]

        rng = np.random.default_rng(9)
        n_points = phase.shape[1]
        random_mask = np.zeros(n_points, dtype=bool)
        random_idx = rng.choice(n_points, size=30, replace=False)
        random_mask[random_idx] = True

        result = run_sbas_validation(phase, good_mask, random_mask, min_improvement_pct=15.0)

        assert result.ps_gnn_mean_coherence > result.adi_baseline_mean_coherence
        assert result.ps_gnn_mean_residual_variance < result.adi_baseline_mean_residual_variance
        assert result.hypothesis_passed

    def test_hypothesis_fails_when_comparison_is_reversed(self, synthetic_network_phase):
        """Sanity check: the hypothesis check is not a tautology -- it must
        correctly report failure when the 'PS-GNN' set is actually worse."""
        phase = synthetic_network_phase["phase"]
        good_mask = synthetic_network_phase["good_mask"]
        pool_mask = synthetic_network_phase["pool_mask"]
        noisy_only_mask = pool_mask & ~good_mask

        result = run_sbas_validation(phase, noisy_only_mask, good_mask, min_improvement_pct=15.0)
        assert not result.hypothesis_passed

    def test_strict_mode_raises_on_failed_hypothesis(self, synthetic_network_phase):
        phase = synthetic_network_phase["phase"]
        good_mask = synthetic_network_phase["good_mask"]
        pool_mask = synthetic_network_phase["pool_mask"]
        noisy_only_mask = pool_mask & ~good_mask

        with pytest.raises(AssertionError):
            run_sbas_validation(
                phase, noisy_only_mask, good_mask, min_improvement_pct=15.0, strict=True
            )

    def test_raises_on_empty_mask(self, synthetic_network_phase):
        phase = synthetic_network_phase["phase"]
        n_points = phase.shape[1]
        empty_mask = np.zeros(n_points, dtype=bool)
        full_mask = np.ones(n_points, dtype=bool)

        with pytest.raises(ValueError):
            run_sbas_validation(phase, empty_mask, full_mask)
