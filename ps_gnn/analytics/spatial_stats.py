"""
ps_gnn.analytics.spatial_stats
=================================

Spatial-statistics diagnostics for validating PS-GNN's predictions beyond
plain classification metrics:

- :func:`morans_i_residuals` — Moran's I on prediction residuals, to
  check that model errors are *not* spatially clustered (spatially
  clustered residuals indicate the model is systematically missing a
  spatial pattern, e.g. failing near a particular structure type or
  terrain feature, rather than making independent random mistakes).
- :func:`ripleys_k_clustering` — Ripley's K-function comparing the
  predicted PS point pattern against Complete Spatial Randomness (CSR,
  a homogeneous Poisson process), to check whether detected PS points
  cluster more than chance would predict (expected, since PS pixels
  concentrate on structures) or are suspiciously regular/dispersed
  (which could indicate an artifact of the graph's spatial-proximity
  edge threshold).
- :func:`calibration_curve` — bins Monte Carlo Dropout uncertainty (or
  predicted probability) against empirical precision, producing a
  reliability diagram: a well-calibrated model's high-confidence bins
  should have high empirical precision, and vice versa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass
class MoranIResult:
    """Result of a Moran's I spatial autocorrelation test.

    Attributes
    ----------
    I : float
        Moran's I statistic. Near ``-1/(n-1)`` (~0 for large n) indicates
        no spatial autocorrelation; positive values indicate clustering
        (similar residuals near each other); negative values indicate
        dispersion (dissimilar residuals near each other).
    expected_I : float
        Expected value of I under the null hypothesis of no spatial
        autocorrelation, ``-1 / (n - 1)``.
    p_value : float or None
        Two-sided p-value from a permutation test (if ``esda`` is
        installed and used), or ``None`` if only the manual fallback
        computation (I statistic only, no significance test) was used.
    n_permutations : int or None
        Number of permutations used for the p-value, if computed.
    backend : str
        Which implementation computed the result: ``"esda"`` or
        ``"manual"``.
    """

    I: float
    expected_I: float
    p_value: float | None
    n_permutations: int | None
    backend: str


def morans_i_residuals(
    residuals: np.ndarray,
    coords: np.ndarray,
    k_neighbors: int = 8,
    n_permutations: int = 999,
    random_state: int = 42,
) -> MoranIResult:
    """Compute Moran's I spatial autocorrelation on prediction residuals.

    Uses a K-nearest-neighbor spatial weights matrix (row-standardized).
    Prefers ``esda`` (the standard PySAL spatial-statistics package) for
    a proper permutation-based significance test; falls back to a manual
    NumPy implementation (I statistic only, no p-value) if ``esda`` is
    not installed, so this function always returns a usable result.

    Parameters
    ----------
    residuals : np.ndarray, shape (N,)
        Per-point residuals (e.g. ``predicted_probability - true_label``,
        or ``0``/``1`` correctness).
    coords : np.ndarray, shape (N, 2)
        Point ground coordinates.
    k_neighbors : int, default 8
        Number of nearest neighbors for the spatial weights matrix.
    n_permutations : int, default 999
        Number of permutations for the significance test (``esda`` path
        only).
    random_state : int, default 42

    Returns
    -------
    MoranIResult

    Raises
    ------
    ValueError
        If ``residuals`` and ``coords`` have inconsistent lengths, or if
        there are fewer than ``k_neighbors + 1`` points.
    """
    n = residuals.shape[0]
    if coords.shape[0] != n:
        raise ValueError(f"residuals has {n} points but coords has {coords.shape[0]}")
    if n <= k_neighbors:
        raise ValueError(f"Need more than k_neighbors={k_neighbors} points, got {n}")

    expected_I = -1.0 / (n - 1)

    try:
        from esda.moran import Moran
        from libpysal.weights import KNN

        np.random.seed(random_state)  # esda's Moran uses the global numpy RNG for permutations
        w = KNN.from_array(coords, k=k_neighbors)
        w.transform = "r"
        moran = Moran(residuals, w, permutations=n_permutations)
        return MoranIResult(
            I=float(moran.I),
            expected_I=float(moran.EI),
            p_value=float(moran.p_sim),
            n_permutations=n_permutations,
            backend="esda",
        )
    except ImportError:
        logger.warning(
            "esda/libpysal not installed; falling back to a manual Moran's I "
            "computation (I statistic only, no significance test)."
        )
        return _manual_morans_i(residuals, coords, k_neighbors, expected_I)


def _manual_morans_i(
    residuals: np.ndarray, coords: np.ndarray, k_neighbors: int, expected_I: float
) -> MoranIResult:
    """Manual, dependency-free Moran's I computation via a KNN weights matrix.

    Parameters
    ----------
    residuals : np.ndarray, shape (N,)
    coords : np.ndarray, shape (N, 2)
    k_neighbors : int
    expected_I : float

    Returns
    -------
    MoranIResult
        With ``p_value=None`` and ``backend="manual"``.
    """
    n = residuals.shape[0]
    tree = cKDTree(coords)
    _, neighbor_idx = tree.query(coords, k=k_neighbors + 1)  # includes self at [:, 0]
    neighbor_idx = neighbor_idx[:, 1:]  # drop self

    z = residuals - residuals.mean()
    denom = np.sum(z**2)
    if denom < 1e-12:
        return MoranIResult(
            I=0.0, expected_I=expected_I, p_value=None, n_permutations=None, backend="manual"
        )

    weight = 1.0 / k_neighbors  # row-standardized KNN weights
    numerator = 0.0
    for i in range(n):
        numerator += z[i] * weight * np.sum(z[neighbor_idx[i]])

    S0 = n * k_neighbors * weight  # sum of all weights (row-standardized -> = n)
    moran_i = (n / S0) * (numerator / denom)

    return MoranIResult(
        I=float(moran_i), expected_I=expected_I, p_value=None, n_permutations=None, backend="manual"
    )


@dataclass
class RipleyKResult:
    """Result of a Ripley's K-function clustering test.

    Attributes
    ----------
    support : np.ndarray, shape (n_distances,)
        Distance values at which K was evaluated.
    observed_k : np.ndarray, shape (n_distances,)
        Observed Ripley's K at each distance.
    csr_low, csr_high : np.ndarray, shape (n_distances,)
        Lower/upper simulation envelope under Complete Spatial Randomness
        (from Monte Carlo Poisson simulations), typically the 2.5th/97.5th
        percentiles.
    p_value : np.ndarray, shape (n_distances,)
        Per-distance simulated p-value (fraction of CSR simulations at
        least as extreme as the observation).
    is_clustered : np.ndarray, shape (n_distances,), dtype bool
        True at distances where ``observed_k > csr_high`` (significant
        clustering relative to CSR).
    is_dispersed : np.ndarray, shape (n_distances,), dtype bool
        True at distances where ``observed_k < csr_low`` (significant
        dispersion / regularity relative to CSR).
    """

    support: np.ndarray
    observed_k: np.ndarray
    csr_low: np.ndarray
    csr_high: np.ndarray
    p_value: np.ndarray
    is_clustered: np.ndarray
    is_dispersed: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        """Return the per-distance results as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "distance": self.support,
                "observed_k": self.observed_k,
                "csr_low": self.csr_low,
                "csr_high": self.csr_high,
                "p_value": self.p_value,
                "is_clustered": self.is_clustered,
                "is_dispersed": self.is_dispersed,
            }
        )


def ripleys_k_clustering(
    points_xy: np.ndarray,
    n_distance_bins: int = 10,
    n_simulations: int = 99,
    random_state: int = 42,
) -> RipleyKResult:
    """Compare a PS point pattern against Complete Spatial Randomness via Ripley's K.

    Parameters
    ----------
    points_xy : np.ndarray, shape (N, 2)
        Ground coordinates of detected PS points.
    n_distance_bins : int, default 10
        Number of distances at which to evaluate K, spanning from a small
        positive value up to roughly a quarter of the point pattern's
        bounding-box diagonal (a common rule-of-thumb upper bound to
        avoid severe edge effects).
    n_simulations : int, default 99
        Number of CSR (homogeneous Poisson) Monte Carlo simulations for
        the significance envelope.
    random_state : int, default 42

    Returns
    -------
    RipleyKResult

    Raises
    ------
    ImportError
        If ``pointpats`` is not installed.
    ValueError
        If fewer than 3 points are provided (K is not meaningful).
    """
    try:
        from pointpats.distance_statistics import k_test
    except ImportError as exc:
        raise ImportError(
            "ripleys_k_clustering requires the `pointpats` package: `pip install pointpats`."
        ) from exc

    if points_xy.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for Ripley's K, got {points_xy.shape[0]}")

    diag = np.hypot(*(points_xy.max(axis=0) - points_xy.min(axis=0)))
    max_distance = max(diag / 4, 1e-6)

    result = k_test(
        points_xy,
        support=(0.0, max_distance, n_distance_bins),
        n_simulations=n_simulations,
        keep_simulations=True,
    )

    simulations = np.asarray(result.simulations)  # (n_simulations, n_distance_bins)
    csr_low = np.percentile(simulations, 2.5, axis=0)
    csr_high = np.percentile(simulations, 97.5, axis=0)

    observed = np.asarray(result.statistic)
    p_value = np.mean(simulations >= observed[None, :], axis=0)
    p_value = np.minimum(p_value, 1 - p_value) * 2  # two-sided

    return RipleyKResult(
        support=np.asarray(result.support),
        observed_k=observed,
        csr_low=csr_low,
        csr_high=csr_high,
        p_value=p_value,
        is_clustered=observed > csr_high,
        is_dispersed=observed < csr_low,
    )


def calibration_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a reliability diagram: binned confidence vs. empirical precision.

    Parameters
    ----------
    confidence : np.ndarray, shape (N,)
        A per-prediction confidence score in ``[0, 1]`` — either the
        predicted PS-class probability, or a derived confidence measure
        such as ``1 - normalized_mc_dropout_variance``.
    correct : np.ndarray, shape (N,), dtype bool or {0, 1}
        Whether each prediction was correct (matches ground truth).
    n_bins : int, default 10
        Number of equal-width confidence bins.

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_low``, ``bin_high``, ``mean_confidence``,
        ``empirical_precision``, ``count``, ``calibration_gap``
        (``mean_confidence - empirical_precision``; positive means
        overconfident, negative means underconfident). Bins with zero
        samples are omitted.

    Raises
    ------
    ValueError
        If ``confidence`` and ``correct`` have inconsistent lengths.
    """
    if confidence.shape[0] != correct.shape[0]:
        raise ValueError(
            f"confidence has {confidence.shape[0]} entries but correct has {correct.shape[0]}"
        )

    correct = np.asarray(correct).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_conf = float(confidence[mask].mean())
        empirical_precision = float(correct[mask].mean())
        rows.append(
            {
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "mean_confidence": mean_conf,
                "empirical_precision": empirical_precision,
                "count": count,
                "calibration_gap": mean_conf - empirical_precision,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    """Compute the Expected Calibration Error (ECE) from a calibration curve.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Output of :func:`calibration_curve`.

    Returns
    -------
    float
        Sample-weighted mean absolute calibration gap across bins.
    """
    if calibration_df.empty:
        return float("nan")
    total = calibration_df["count"].sum()
    weighted_gap = (calibration_df["calibration_gap"].abs() * calibration_df["count"]).sum()
    return float(weighted_gap / total)
