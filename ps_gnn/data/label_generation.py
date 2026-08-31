"""
ps_gnn.data.label_generation
==============================

Ground-truth generation for training the PS-GNN classifier.

Supervised PS detection needs labels, but "true" PS status is rarely known
with certainty in real data. This module implements two complementary
labeling strategies:

1. :func:`inject_synthetic_ps` — embed synthetic scatterers with *known*
   amplitude/phase stability into real (or real-looking) SAR stacks. This
   gives perfect, noise-free ground truth for controlled experiments and
   sanity-checking model behavior.

2. :func:`generate_stamps_pseudo_labels` — mine high-confidence positive
   and negative examples from an existing classical StaMPS PS selection,
   using strict thresholds (temporal coherence, number of coherent
   acquisitions) so that only the most trustworthy pixels are used as
   pseudo-labels for weak/semi-supervised training on real data.

Both functions return a boolean/binary label array aligned with the pixel
grid of the input stack (or a sparse point set of ``(row, col)`` indices
plus labels), ready to be consumed by
:mod:`ps_gnn.data.preprocessing`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticPSResult:
    """Output of :func:`inject_synthetic_ps`.

    Attributes
    ----------
    amplitude : np.ndarray, shape (T, H, W)
        Amplitude stack with synthetic PS signals injected.
    phase : np.ndarray, shape (T, H, W)
        Phase stack with synthetic PS signals injected.
    labels : np.ndarray, shape (H, W), dtype=uint8
        Binary label grid: 1 where a synthetic PS was injected, 0 elsewhere.
    """

    amplitude: np.ndarray
    phase: np.ndarray
    labels: np.ndarray


@dataclass
class StampsPseudoLabelConfig:
    """Configuration for StaMPS-derived pseudo-labeling.

    Attributes
    ----------
    coherence_threshold : float
        Minimum temporal coherence for a StaMPS PS candidate to be
        considered a trustworthy positive.
    min_acquisitions : int
        Minimum number of acquisitions in which the candidate must be
        coherently tracked by StaMPS.
    top_fraction : float
        Fraction (0, 1] of the most confident candidates (after applying
        the two thresholds above) to keep as positive pseudo-labels.
    negative_sample_ratio : float
        Ratio of negative to positive pseudo-labels to draw from pixels
        StaMPS rejected (or that fall below threshold), to keep the
        training set balanced-ish without discarding all negative
        information.
    random_state : int
        Seed for reproducible negative sampling.
    """

    coherence_threshold: float = 0.9
    min_acquisitions: int = 20
    top_fraction: float = 0.10
    negative_sample_ratio: float = 1.0
    random_state: int = 42


def inject_synthetic_ps(
    amplitude: np.ndarray,
    phase: np.ndarray,
    n_ps: int = 25,
    min_separation_px: int = 2,
    random_state: int = 42,
    amplitude_percentile: float = 95.0,
    amplitude_boost: float = 1.5,
) -> SyntheticPSResult:
    """Embed perfectly stable synthetic PS pixels into a SAR stack.

    Each synthetic PS pixel is given a **constant, high amplitude** and
    **zero phase variance** across the whole time series — the idealized
    signature of a mechanically stable, dominant scatterer (a corner
    reflector, a building edge, exposed rock), and by construction:

    - Amplitude Dispersion Index (ADI) ``= std(amplitude) / mean(amplitude)
      == 0`` exactly (a constant series has zero standard deviation).
    - Temporal coherence ``= |mean(exp(1j * phase))| == 1`` exactly (a
      constant phase has a unit-magnitude mean phasor).

    This gives perfect, unambiguous ground truth for controlled
    experiments and sanity-checking that a model can at least recover the
    easiest possible case, before being evaluated on real, imperfect PS
    candidates (see :func:`generate_stamps_pseudo_labels` for those).

    Parameters
    ----------
    amplitude : np.ndarray, shape (T, H, W)
        Real (or realistic) amplitude stack to inject synthetic PS into.
    phase : np.ndarray, shape (T, H, W)
        Corresponding wrapped phase stack.
    n_ps : int, default 25
        Number of synthetic PS pixels to inject.
    min_separation_px : int, default 2
        Minimum Chebyshev distance enforced between injected pixels, so
        that synthetic PS points don't cluster into a single blob.
    random_state : int, default 42
        Seed for reproducibility.
    amplitude_percentile : float, default 95.0
        Percentile of the *original* amplitude distribution used as the
        base "high amplitude" value for injected pixels (computed once,
        globally, before any injection).
    amplitude_boost : float, default 1.5
        Multiplier applied to that percentile, so injected PS pixels are
        clearly brighter than the ambient scene rather than merely
        matching its upper tail.

    Returns
    -------
    SyntheticPSResult
        The modified amplitude/phase stacks and the binary label grid
        (``1`` at injected PS pixels, ``0`` elsewhere).

    Raises
    ------
    ValueError
        If ``amplitude`` and ``phase`` shapes disagree, or if ``n_ps``
        cannot be placed given ``min_separation_px`` and the grid size.
    """
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"amplitude and phase must share shape, got {amplitude.shape} vs {phase.shape}"
        )
    if amplitude.ndim != 3:
        raise ValueError(f"Expected (T, H, W) stacks, got ndim={amplitude.ndim}")

    _t_steps, height, width = amplitude.shape
    rng = np.random.default_rng(random_state)

    amp_out = amplitude.copy().astype(np.float32)
    phase_out = phase.copy().astype(np.float32)
    labels = np.zeros((height, width), dtype=np.uint8)

    # A single, deterministic "high amplitude" value for every injected
    # pixel, derived once from the scene's own amplitude distribution so
    # the synthetic PS points sit clearly above ambient brightness
    # regardless of the input stack's absolute units/scale.
    high_amplitude = float(np.percentile(amplitude, amplitude_percentile)) * amplitude_boost

    injected: list[tuple[int, int]] = []
    max_attempts = n_ps * 50
    attempts = 0

    while len(injected) < n_ps and attempts < max_attempts:
        attempts += 1
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width))

        if any(max(abs(r - pr), abs(c - pc)) < min_separation_px for pr, pc in injected):
            continue

        # Constant amplitude -> ADI == 0 exactly; constant (zero) phase ->
        # temporal coherence == 1 exactly. A perfectly stable scatterer.
        amp_out[:, r, c] = high_amplitude
        phase_out[:, r, c] = 0.0
        labels[r, c] = 1
        injected.append((r, c))

    if len(injected) < n_ps:
        raise ValueError(
            f"Could only place {len(injected)}/{n_ps} synthetic PS points with "
            f"min_separation_px={min_separation_px} on a {height}x{width} grid. "
            "Reduce n_ps or min_separation_px."
        )

    logger.info("Injected %d synthetic PS pixels (%d attempts).", len(injected), attempts)

    return SyntheticPSResult(amplitude=amp_out, phase=phase_out, labels=labels)


def _load_stamps_candidates(stamps_output_path: str | Path) -> dict[str, np.ndarray]:
    """Load StaMPS PS candidate arrays from a ``.mat`` (or ``.npz``) file.

    StaMPS (``ps2.mat`` / ``ps_plot`` outputs) is a MATLAB-format file. We
    support both real ``.mat`` files (via ``scipy.io``) and a lightweight
    ``.npz`` convention (useful for tests / synthetic fixtures) with the
    same field names, so this function works without requiring a real
    StaMPS run to be available.

    Expected fields (StaMPS convention, simplified):

    - ``lonlat`` or ``rowcol`` : (N, 2) candidate pixel coordinates.
    - ``coh_ps`` : (N,) temporal coherence per candidate.
    - ``n_ifg`` : (N,) number of interferograms/acquisitions the candidate
      was coherently tracked in.

    Parameters
    ----------
    stamps_output_path : str or pathlib.Path
        Path to the StaMPS output file (``.mat`` or ``.npz``).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``rowcol``, ``coh_ps``, ``n_ifg``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If required fields are missing from the file.
    """
    path = Path(stamps_output_path)
    if not path.exists():
        raise FileNotFoundError(f"StaMPS output not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
    else:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - scipy is a hard dep in pyproject
            raise ImportError("scipy is required to read StaMPS .mat files") from exc
        data = loadmat(path)

    required = {"rowcol", "coh_ps", "n_ifg"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"StaMPS output {path} is missing required fields: {sorted(missing)}. "
            f"Available fields: {sorted(k for k in data if not k.startswith('__'))}"
        )

    return {
        "rowcol": np.asarray(data["rowcol"]).reshape(-1, 2).astype(np.int64),
        "coh_ps": np.asarray(data["coh_ps"]).reshape(-1).astype(np.float64),
        "n_ifg": np.asarray(data["n_ifg"]).reshape(-1).astype(np.int64),
    }


def generate_stamps_pseudo_labels(
    stamps_output_path: str | Path,
    grid_shape: tuple[int, int] | None = None,
    coherence_threshold: float = 0.9,
    min_acquisitions: int = 20,
    top_fraction: float = 0.10,
    negative_sample_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate high-confidence binary pseudo-labels from a StaMPS output.

    Candidates are first filtered to those StaMPS tracked coherently
    (``coh_ps > coherence_threshold`` and ``n_ifg > min_acquisitions``);
    among the survivors, only the ``top_fraction`` most confident (highest
    coherence) are kept as positive pseudo-labels. A matched sample of
    low-confidence / rejected candidates is drawn as negatives, controlled
    by ``negative_sample_ratio``, so the resulting label set is usable for
    (weakly) supervised training without being trivially imbalanced.

    Parameters
    ----------
    stamps_output_path : str or pathlib.Path
        Path to a StaMPS ``.mat`` output (or a compatible ``.npz``) with
        fields ``rowcol``, ``coh_ps``, ``n_ifg``.
    grid_shape : tuple[int, int], optional
        ``(height, width)`` of the target pixel grid, used only for
        bounds-checking of pixel coordinates. If omitted, no bounds
        checking is performed.
    coherence_threshold : float, default 0.9
        Minimum temporal coherence for a candidate to be eligible as a
        positive label.
    min_acquisitions : int, default 20
        Minimum number of acquisitions the candidate must be coherently
        present in to be eligible as a positive label.
    top_fraction : float, default 0.10
        Fraction of the eligible high-confidence candidates to keep as
        positives (the most confident ``top_fraction`` by coherence).
    negative_sample_ratio : float, default 1.0
        Number of negatives sampled per positive, drawn from candidates
        that failed the coherence/acquisition thresholds.
    random_state : int, default 42
        Seed for reproducible negative sampling.

    Returns
    -------
    pixel_coords : np.ndarray, shape (n_labeled, 2)
        Row/column coordinates of all labeled pixels (positives then
        negatives).
    labels : np.ndarray, shape (n_labeled,), dtype=uint8
        Binary labels aligned with ``pixel_coords`` (1 = PS, 0 = non-PS).

    Raises
    ------
    ValueError
        If no candidates satisfy the confidence thresholds, or if
        ``top_fraction`` is not in ``(0, 1]``.
    """
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    candidates = _load_stamps_candidates(stamps_output_path)
    rowcol, coh, n_ifg = candidates["rowcol"], candidates["coh_ps"], candidates["n_ifg"]

    if grid_shape is not None:
        height, width = grid_shape
        in_bounds = (
            (rowcol[:, 0] >= 0)
            & (rowcol[:, 0] < height)
            & (rowcol[:, 1] >= 0)
            & (rowcol[:, 1] < width)
        )
        rowcol, coh, n_ifg = rowcol[in_bounds], coh[in_bounds], n_ifg[in_bounds]

    high_confidence_mask = (coh > coherence_threshold) & (n_ifg > min_acquisitions)
    n_eligible = int(high_confidence_mask.sum())
    if n_eligible == 0:
        raise ValueError(
            "No StaMPS candidates satisfy coherence_threshold="
            f"{coherence_threshold} and min_acquisitions={min_acquisitions}. "
            "Loosen the thresholds or check the input file."
        )

    eligible_idx = np.flatnonzero(high_confidence_mask)
    eligible_coh = coh[eligible_idx]

    n_positive = max(1, int(np.ceil(top_fraction * n_eligible)))
    # Highest-coherence candidates first.
    order = np.argsort(-eligible_coh)
    positive_idx = eligible_idx[order[:n_positive]]

    rng = np.random.default_rng(random_state)
    rejected_idx = np.flatnonzero(~high_confidence_mask)
    n_negative = min(len(rejected_idx), round(negative_sample_ratio * n_positive))
    negative_idx = (
        rng.choice(rejected_idx, size=n_negative, replace=False)
        if n_negative > 0
        else np.array([], dtype=np.int64)
    )

    selected_idx = np.concatenate([positive_idx, negative_idx])
    pixel_coords = rowcol[selected_idx]
    labels = np.concatenate(
        [np.ones(len(positive_idx), dtype=np.uint8), np.zeros(len(negative_idx), dtype=np.uint8)]
    )

    logger.info(
        "StaMPS pseudo-labeling: %d eligible candidates -> %d positives (top %.1f%%), "
        "%d negatives sampled.",
        n_eligible,
        n_positive,
        top_fraction * 100,
        n_negative,
    )

    return pixel_coords, labels
