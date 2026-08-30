"""
ps_gnn.validation.sbas_check
===============================

The final scientific sanity check: does PS-GNN's selection actually
produce *better geophysical time series* than the classical ADI
threshold, once both point sets are run through a real SBAS inversion?
Classification metrics (F1, AUC) validate agreement with labels, but the
entire point of PSI is the quality of the resulting deformation
time-series — this module closes that loop.

A basic weighted least-squares (WLS) SBAS inversion is implemented from
scratch (:func:`wls_sbas_inversion`) using a small-baseline interferogram
network with physically-motivated, temporal-baseline-dependent weights
(longer time gaps between acquisitions decorrelate more, per the
standard PSI/SBAS decorrelation model), so that WLS genuinely
outperforms unweighted least squares here rather than being a
mathematically vacuous formality.

:func:`run_sbas_validation` then runs this inversion independently for
the PS-GNN-selected point set and an ADI-threshold baseline point set
over the *same* phase data, and compares temporal coherence, residual
phase variance, and spatial coverage — including a formal check of the
project's core hypothesis: PS-GNN should yield >15% higher temporal
coherence than the ADI baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SBASNetwork:
    """A small-baseline interferogram network and its design matrix.

    Attributes
    ----------
    pairs : list[tuple[int, int]]
        ``(i, j)`` acquisition-index pairs (``i < j``) forming each
        interferogram.
    design_matrix : np.ndarray, shape (n_ifg, n_epochs - 1)
        SBAS design matrix ``A`` such that, for cumulative per-epoch
        phase ``m`` (relative to epoch 0, ``n_epochs - 1`` unknown
        increments), the interferometric phase observation for pair
        ``(i, j)`` is ``d_ij = sum_{k=i}^{j-1} m_k = (A @ m)_row``.
    n_epochs : int
        Total number of SAR acquisitions.
    """

    pairs: list[tuple[int, int]]
    design_matrix: np.ndarray
    n_epochs: int


def build_interferogram_network(n_epochs: int, max_pairs: int | None = None) -> SBASNetwork:
    """Build an all-pairs small-baseline interferogram network.

    Every acquisition pair ``(i, j)`` with ``i < j`` is treated as a
    (small-baseline) interferogram — the simplest network topology that
    still gives meaningful redundancy for WLS (a purely sequential
    ``(i, i+1)`` network would be exactly determined, giving OLS and WLS
    identical solutions).

    Parameters
    ----------
    n_epochs : int
        Number of SAR acquisitions.
    max_pairs : int, optional
        If given, randomly subsample the full ``n_epochs choose 2`` pair
        set down to this many pairs (for tractability on long time
        series), always keeping the sequential ``(i, i+1)`` pairs so the
        network stays fully connected.

    Returns
    -------
    SBASNetwork

    Raises
    ------
    ValueError
        If ``n_epochs < 2``.
    """
    if n_epochs < 2:
        raise ValueError(
            f"Need at least 2 epochs to build an interferogram network, got {n_epochs}"
        )

    all_pairs = [(i, j) for i in range(n_epochs) for j in range(i + 1, n_epochs)]

    if max_pairs is not None and max_pairs < len(all_pairs):
        sequential = [(i, i + 1) for i in range(n_epochs - 1)]
        rest = [p for p in all_pairs if p not in set(sequential)]
        rng = np.random.default_rng(42)
        n_extra = max(max_pairs - len(sequential), 0)
        extra = (
            [rest[k] for k in rng.choice(len(rest), size=min(n_extra, len(rest)), replace=False)]
            if rest and n_extra > 0
            else []
        )
        pairs = sequential + extra
    else:
        pairs = all_pairs

    n_unknowns = n_epochs - 1
    design_matrix = np.zeros((len(pairs), n_unknowns), dtype=np.float64)
    for row, (i, j) in enumerate(pairs):
        design_matrix[row, i:j] = 1.0

    return SBASNetwork(pairs=pairs, design_matrix=design_matrix, n_epochs=n_epochs)


def _interferogram_weights(
    n_pairs_baselines: np.ndarray, point_coherence: np.ndarray, decorrelation_rate: float
) -> np.ndarray:
    """Per-point, per-interferogram weights combining coherence and baseline decay.

    ``weight[n, k] = point_coherence[n] * exp(-decorrelation_rate * baseline[k])``,
    reflecting that (a) a more temporally-stable point should be trusted
    more overall, and (b) any single interferogram is trusted less the
    longer its temporal baseline (more time for scatterer/atmospheric
    decorrelation), which is what gives WLS genuine leverage over OLS
    here.

    Parameters
    ----------
    n_pairs_baselines : np.ndarray, shape (n_ifg,)
        Temporal baseline (``j - i``, in epochs) of every interferogram.
    point_coherence : np.ndarray, shape (N,)
        Overall per-point temporal coherence, in ``[0, 1]``.
    decorrelation_rate : float
        Decay rate (per epoch) of the baseline-dependent weight term.

    Returns
    -------
    np.ndarray, shape (N, n_ifg)
    """
    baseline_decay = np.exp(-decorrelation_rate * n_pairs_baselines)  # (n_ifg,)
    return point_coherence[:, None] * baseline_decay[None, :]


def wls_sbas_inversion(
    phase: np.ndarray,
    point_coherence: np.ndarray,
    network: SBASNetwork | None = None,
    decorrelation_rate: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares SBAS time-series inversion.

    For every point, solves ``m_n = argmin_m || W_n^(1/2) (A m - d_n) ||^2``
    where ``d_n`` are the (wrapped-then-treated-as-relative) interferometric
    phase observations for every pair in ``network``, and ``W_n`` are the
    per-interferogram weights from :func:`_interferogram_weights`.

    Parameters
    ----------
    phase : np.ndarray, shape (T, N)
        Per-point phase time series (radians), referenced to a common
        master (i.e. NOT already-differenced interferograms — this
        function forms interferogram observations internally as
        ``phase[j] - phase[i]``, wrapped to ``(-pi, pi]``).
    point_coherence : np.ndarray, shape (N,)
        Overall per-point temporal coherence (e.g. from
        :func:`ps_gnn.data.preprocessing._temporal_coherence`), used as
        the point-level weighting term.
    network : SBASNetwork, optional
        Interferogram network to invert. Defaults to the full all-pairs
        network for ``T = phase.shape[0]`` epochs (see
        :func:`build_interferogram_network`).
    decorrelation_rate : float, default 0.05
        Baseline-dependent weight decay rate (per epoch).

    Returns
    -------
    inverted_series : np.ndarray, shape (T, N)
        Cumulative inverted phase per point per epoch, with
        ``inverted_series[0] == 0`` (reference epoch).
    residual_variance : np.ndarray, shape (N,)
        Variance of the (weighted) misfit ``A @ m_n - d_n`` per point —
        lower is better (more internally-consistent network).
    temporal_coherence_inverted : np.ndarray, shape (N,)
        Phasor-magnitude coherence of the inversion residuals,
        ``|mean(exp(1j * residual))|`` — a direct measure of how
        consistent the inverted time series is with the observed
        interferometric phase (1.0 = perfectly consistent).

    Raises
    ------
    ValueError
        If ``phase`` and ``point_coherence`` have inconsistent point
        counts.
    """
    t_steps, n_points = phase.shape
    if point_coherence.shape[0] != n_points:
        raise ValueError(
            f"point_coherence has {point_coherence.shape[0]} points but phase has {n_points}"
        )

    network = network or build_interferogram_network(t_steps)
    pairs_arr = np.array(network.pairs)
    baselines = (pairs_arr[:, 1] - pairs_arr[:, 0]).astype(np.float64)

    # Interferometric observations: phase difference, wrapped to (-pi, pi].
    d = phase[pairs_arr[:, 1], :] - phase[pairs_arr[:, 0], :]  # (n_ifg, N)
    d = (d + np.pi) % (2 * np.pi) - np.pi

    weights = _interferogram_weights(baselines, point_coherence, decorrelation_rate)  # (N, n_ifg)

    A = network.design_matrix  # (n_ifg, T-1)
    n_unknowns = A.shape[1]

    inverted_increments = np.zeros((n_unknowns, n_points), dtype=np.float64)
    residual_variance = np.zeros(n_points, dtype=np.float64)
    temporal_coherence_inverted = np.zeros(n_points, dtype=np.float64)

    for n in range(n_points):
        w = weights[n]  # (n_ifg,)
        sqrt_w = np.sqrt(np.clip(w, a_min=1e-8, a_max=None))
        A_w = A * sqrt_w[:, None]
        d_w = d[:, n] * sqrt_w

        # Weighted least squares via the normal equations' pseudo-inverse
        # (small, dense system: n_unknowns = T - 1, tractable for typical
        # InSAR stack lengths).
        m_n, *_ = np.linalg.lstsq(A_w, d_w, rcond=None)
        inverted_increments[:, n] = m_n

        residual = A @ m_n - d[:, n]
        residual_variance[n] = float(np.var(residual))
        temporal_coherence_inverted[n] = float(np.abs(np.mean(np.exp(1j * residual))))

    cumulative = np.vstack([np.zeros((1, n_points)), np.cumsum(inverted_increments, axis=0)])
    return cumulative, residual_variance, temporal_coherence_inverted


@dataclass
class SBASComparisonResult:
    """Comparison of PS-GNN vs. ADI-baseline point sets through SBAS inversion.

    Attributes
    ----------
    ps_gnn_mean_coherence, adi_baseline_mean_coherence : float
        Mean inverted-series temporal coherence for each point set.
    ps_gnn_mean_residual_variance, adi_baseline_mean_residual_variance : float
        Mean inversion residual phase variance for each point set.
    ps_gnn_coverage, adi_baseline_coverage : float
        Fraction of the total candidate points selected by each method
        (spatial coverage proxy).
    coherence_improvement_pct : float
        Relative improvement of PS-GNN's mean coherence over the ADI
        baseline, as a percentage (can be negative).
    hypothesis_passed : bool
        Whether ``coherence_improvement_pct`` exceeds
        ``min_improvement_pct`` (default hypothesis: >15%).
    """

    ps_gnn_mean_coherence: float
    adi_baseline_mean_coherence: float
    ps_gnn_mean_residual_variance: float
    adi_baseline_mean_residual_variance: float
    ps_gnn_coverage: float
    adi_baseline_coverage: float
    coherence_improvement_pct: float
    hypothesis_passed: bool


def run_sbas_validation(
    phase: np.ndarray,
    ps_gnn_mask: np.ndarray,
    adi_baseline_mask: np.ndarray,
    point_coherence: np.ndarray | None = None,
    decorrelation_rate: float = 0.05,
    min_improvement_pct: float = 15.0,
    strict: bool = False,
) -> SBASComparisonResult:
    """Compare PS-GNN and ADI-baseline point selections via SBAS inversion.

    Parameters
    ----------
    phase : np.ndarray, shape (T, N)
        Phase time series for every candidate point in the scene.
    ps_gnn_mask, adi_baseline_mask : np.ndarray, shape (N,), dtype bool
        Boolean selection masks for the two methods being compared, over
        the same ``N`` candidate points.
    point_coherence : np.ndarray, shape (N,), optional
        Per-point temporal coherence for weighting (see
        :func:`wls_sbas_inversion`). If omitted, computed directly from
        ``phase`` via the phasor-magnitude definition.
    decorrelation_rate : float, default 0.05
        Passed through to :func:`wls_sbas_inversion`.
    min_improvement_pct : float, default 15.0
        The project's core hypothesis threshold: PS-GNN should yield at
        least this much relative improvement in mean temporal coherence
        over the ADI baseline.
    strict : bool, default False
        If True, raise ``AssertionError`` when the hypothesis check
        fails, instead of only reporting ``hypothesis_passed=False``.

    Returns
    -------
    SBASComparisonResult

    Raises
    ------
    ValueError
        If neither mask selects any points.
    AssertionError
        If ``strict=True`` and the hypothesis check fails.
    """
    if point_coherence is None:
        point_coherence = np.abs(np.mean(np.exp(1j * phase), axis=0))

    if ps_gnn_mask.sum() == 0 or adi_baseline_mask.sum() == 0:
        raise ValueError(
            "Both ps_gnn_mask and adi_baseline_mask must select at least one point "
            f"(got {int(ps_gnn_mask.sum())} and {int(adi_baseline_mask.sum())})."
        )

    network = build_interferogram_network(phase.shape[0])

    _, ps_gnn_resid_var, ps_gnn_coh = wls_sbas_inversion(
        phase[:, ps_gnn_mask], point_coherence[ps_gnn_mask], network, decorrelation_rate
    )
    _, adi_resid_var, adi_coh = wls_sbas_inversion(
        phase[:, adi_baseline_mask], point_coherence[adi_baseline_mask], network, decorrelation_rate
    )

    ps_gnn_mean_coh = float(np.mean(ps_gnn_coh))
    adi_mean_coh = float(np.mean(adi_coh))
    improvement_pct = (
        100.0 * (ps_gnn_mean_coh - adi_mean_coh) / adi_mean_coh
        if adi_mean_coh > 0
        else float("inf")
    )
    hypothesis_passed = improvement_pct > min_improvement_pct

    result = SBASComparisonResult(
        ps_gnn_mean_coherence=ps_gnn_mean_coh,
        adi_baseline_mean_coherence=adi_mean_coh,
        ps_gnn_mean_residual_variance=float(np.mean(ps_gnn_resid_var)),
        adi_baseline_mean_residual_variance=float(np.mean(adi_resid_var)),
        ps_gnn_coverage=float(ps_gnn_mask.mean()),
        adi_baseline_coverage=float(adi_baseline_mask.mean()),
        coherence_improvement_pct=improvement_pct,
        hypothesis_passed=hypothesis_passed,
    )

    logger.info(
        "SBAS validation: PS-GNN coherence=%.4f, ADI baseline=%.4f (%.1f%% relative change, "
        "hypothesis >%.0f%% %s).",
        ps_gnn_mean_coh,
        adi_mean_coh,
        improvement_pct,
        min_improvement_pct,
        "PASSED" if hypothesis_passed else "NOT met",
    )

    if strict:
        assert hypothesis_passed, (
            f"SBAS validation hypothesis failed: PS-GNN mean temporal coherence "
            f"({ps_gnn_mean_coh:.4f}) is only {improvement_pct:.1f}% higher than the ADI "
            f"baseline ({adi_mean_coh:.4f}); expected > {min_improvement_pct:.0f}%."
        )

    return result
