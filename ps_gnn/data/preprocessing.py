"""
ps_gnn.data.preprocessing
============================

Converts raw Sentinel-1 SLC-derived amplitude/phase stacks (plus a DEM,
incidence angle, land cover, and binary labels) into
:class:`torch_geometric.data.Data` graph objects ready for training or
inference with :mod:`ps_gnn.models.ps_gnn`.

Pipeline overview
------------------
1. **Node feature engineering** (:func:`compute_node_features`): a
   19-dimensional feature vector per candidate pixel, combining amplitude
   statistics, temporal coherence, local spatial texture, incidence angle,
   and one-hot land cover.
2. **Graph construction** (:func:`build_graph`): edges are added between
   pixels that are both spatially close (< ``max_distance_m``) and
   phase-correlated (> ``min_phase_correlation``), with normalized
   distance and phase correlation as edge features.
3. **Assembly** (:func:`build_pyg_dataset`): wraps features, edges, and
   labels into a single :class:`torch_geometric.data.Data` object and
   (optionally) persists it to disk.

``torch`` / ``torch_geometric`` are imported lazily inside the functions
that need them, so the feature-engineering and graph-construction logic
(which is pure NumPy/SciPy) remains importable and testable even in
environments without a GPU-oriented deep learning stack installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, skew
from tqdm import tqdm

if TYPE_CHECKING:
    from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------

N_WORLDCOVER_CLASSES = 10
# ESA WorldCover v200 class codes, remapped to a dense [0, 9] index.
WORLDCOVER_CODE_TO_INDEX: dict[int, int] = {
    10: 0,  # Tree cover
    20: 1,  # Shrubland
    30: 2,  # Grassland
    40: 3,  # Cropland
    50: 4,  # Built-up
    60: 5,  # Bare / sparse vegetation
    70: 6,  # Snow and ice
    80: 7,  # Permanent water bodies
    90: 8,  # Herbaceous wetland
    95: 9,  # Mangroves / Moss and lichen (merged into last slot)
}

FEATURE_NAMES: list[str] = [
    "mean_amp",
    "std_amp",
    "adi",
    "skewness",
    "kurtosis",
    "temporal_coherence",
    "local_var",
    "edge_strength",
    "incidence_angle",
    *[f"lc_{code}" for code in WORLDCOVER_CODE_TO_INDEX],
]
N_NODE_FEATURES = len(FEATURE_NAMES)
assert N_NODE_FEATURES == 19, f"Expected 19 node features, got {N_NODE_FEATURES}"


@dataclass
class GraphConstructionConfig:
    """Configuration for spatial-proximity + phase-correlation graph edges.

    Attributes
    ----------
    max_distance_m : float
        Maximum ground distance (meters) between two pixels for an edge to
        be considered.
    min_phase_correlation : float
        Minimum Pearson correlation of the (unwrapped-cosine) phase time
        series between two pixels for an edge to be considered.
    pixel_spacing_m : float
        Ground sampling distance of the input rasters, used to convert
        pixel offsets to meters when a geotransform is unavailable.
    max_neighbors : int
        Cap on the number of spatial neighbor candidates considered per
        node (via KD-tree query), for tractability on large scenes.
    """

    max_distance_m: float = 100.0
    min_phase_correlation: float = 0.7
    pixel_spacing_m: float = 5.0
    max_neighbors: int = 8


def _temporal_coherence(phase: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute per-pixel temporal coherence from a wrapped phase stack.

    Defined as the magnitude of the mean unit phasor:
    ``|mean(exp(1j * phase))|`` over time, ranging from 0 (incoherent,
    random phase) to 1 (perfectly stable phase).

    Parameters
    ----------
    phase : np.ndarray, shape (T, H, W)
        Wrapped phase stack, radians.
    axis : int, default 0
        Temporal axis.

    Returns
    -------
    np.ndarray, shape (H, W)
        Per-pixel temporal coherence in ``[0, 1]``.
    """
    phasor = np.exp(1j * phase)
    return np.abs(np.mean(phasor, axis=axis))


def _one_hot_worldcover(worldcover: np.ndarray) -> np.ndarray:
    """One-hot encode an ESA WorldCover class raster into 10 channels.

    Parameters
    ----------
    worldcover : np.ndarray, shape (H, W)
        Integer WorldCover class codes (e.g. 10, 20, ..., 95). Unknown
        codes are mapped to channel 0 with a warning.

    Returns
    -------
    np.ndarray, shape (10, H, W)
        One-hot encoded land cover, dtype float32.
    """
    height, width = worldcover.shape
    one_hot = np.zeros((N_WORLDCOVER_CLASSES, height, width), dtype=np.float32)

    unknown_codes = set(np.unique(worldcover)) - set(WORLDCOVER_CODE_TO_INDEX)
    if unknown_codes:
        logger.warning(
            "Unrecognized WorldCover codes %s encountered; mapping to class 0.",
            sorted(unknown_codes),
        )

    for code, idx in WORLDCOVER_CODE_TO_INDEX.items():
        one_hot[idx][worldcover == code] = 1.0
    if unknown_codes:
        mask = np.isin(worldcover, list(unknown_codes))
        one_hot[0][mask] = 1.0

    return one_hot


def compute_node_features(
    amplitude: np.ndarray,
    phase: np.ndarray,
    incidence_angle: np.ndarray | float,
    worldcover: np.ndarray,
    local_window: int = 5,
) -> np.ndarray:
    """Compute the 19-dimensional per-pixel node feature vector.

    Feature layout (see :data:`FEATURE_NAMES`):

    ============  ================================================
    Index         Feature
    ============  ================================================
    0             Amplitude mean
    1             Amplitude std
    2             Amplitude Dispersion Index (ADI = std / mean)
    3             Amplitude skewness (temporal)
    4             Amplitude kurtosis (temporal)
    5             Temporal coherence (from phase)
    6             Local variance of mean amplitude (5x5 window)
    7             Sobel edge strength (on mean amplitude)
    8             Incidence angle (degrees)
    9-18          One-hot ESA WorldCover (10 classes)
    ============  ================================================

    Parameters
    ----------
    amplitude : np.ndarray, shape (T, H, W)
        Multi-temporal SAR amplitude stack.
    phase : np.ndarray, shape (T, H, W)
        Multi-temporal wrapped phase stack (radians), same shape as
        ``amplitude``.
    incidence_angle : np.ndarray or float
        Either a scalar (constant incidence angle for the whole scene) or
        an ``(H, W)`` array of per-pixel incidence angles, degrees.
    worldcover : np.ndarray, shape (H, W)
        ESA WorldCover integer class codes.
    local_window : int, default 5
        Window size (pixels) for the local spatial-variance feature.

    Returns
    -------
    np.ndarray, shape (19, H, W), dtype float32
        Stacked node features, channel-first, ready to be reshaped to
        ``(H * W, 19)`` for graph nodes.

    Raises
    ------
    ValueError
        If input shapes are inconsistent.
    """
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"amplitude and phase must share shape, got {amplitude.shape} vs {phase.shape}"
        )
    if amplitude.ndim != 3:
        raise ValueError(f"Expected (T, H, W) amplitude stack, got ndim={amplitude.ndim}")

    _, height, width = amplitude.shape
    if worldcover.shape != (height, width):
        raise ValueError(
            f"worldcover shape {worldcover.shape} does not match raster grid ({height}, {width})"
        )

    amp_mean = np.mean(amplitude, axis=0)
    amp_std = np.std(amplitude, axis=0)
    adi = amp_std / (amp_mean + 1e-8)
    amp_skew = skew(amplitude, axis=0, bias=False, nan_policy="omit")
    amp_kurt = kurtosis(amplitude, axis=0, bias=False, nan_policy="omit")
    amp_skew = np.nan_to_num(amp_skew, nan=0.0)
    amp_kurt = np.nan_to_num(amp_kurt, nan=0.0)

    coherence = _temporal_coherence(phase, axis=0)

    local_variance = ndimage.generic_filter(amp_mean, np.var, size=local_window, mode="reflect")

    sobel_x = ndimage.sobel(amp_mean, axis=1, mode="reflect")
    sobel_y = ndimage.sobel(amp_mean, axis=0, mode="reflect")
    sobel_edge = np.hypot(sobel_x, sobel_y)

    if np.isscalar(incidence_angle):
        incidence_grid = np.full((height, width), float(incidence_angle), dtype=np.float32)
    else:
        incidence_grid = np.asarray(incidence_angle, dtype=np.float32)
        if incidence_grid.shape != (height, width):
            raise ValueError(
                f"incidence_angle shape {incidence_grid.shape} does not match "
                f"raster grid ({height}, {width})"
            )

    worldcover_onehot = _one_hot_worldcover(worldcover)

    features = np.stack(
        [
            amp_mean,
            amp_std,
            adi,
            amp_skew,
            amp_kurt,
            coherence,
            local_variance,
            sobel_edge,
            incidence_grid,
        ],
        axis=0,
    ).astype(np.float32)
    features = np.concatenate([features, worldcover_onehot], axis=0)

    assert (
        features.shape[0] == N_NODE_FEATURES
    ), f"Expected {N_NODE_FEATURES} feature channels, built {features.shape[0]}"
    return features


def _phase_correlation(phase_ts_a: np.ndarray, phase_ts_b: np.ndarray) -> float:
    """Pearson correlation between two phase time series (via cos/sin).

    Wrapped phase is not directly correlatable with Pearson correlation,
    so we correlate the real (cosine) component of the unit phasors, which
    is well-defined on the circle and standard practice for assessing
    phase similarity between candidate PS neighbors.

    Parameters
    ----------
    phase_ts_a, phase_ts_b : np.ndarray, shape (T,)
        Wrapped phase time series for two pixels.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``.
    """
    a = np.cos(phase_ts_a)
    b = np.cos(phase_ts_b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def build_graph(
    node_coords: np.ndarray,
    phase_flat: np.ndarray,
    config: GraphConstructionConfig,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build graph edges from spatial proximity AND phase correlation.

    An edge ``(i, j)`` is added if and only if:

    1. The Euclidean ground distance between nodes ``i`` and ``j`` is below
       ``config.max_distance_m``, **and**
    2. The phase-correlation between their time series exceeds
       ``config.min_phase_correlation``.

    Spatial neighbor *candidates* are found efficiently with a KD-tree
    (a K-nearest-neighbors query bounded by ``config.max_neighbors``); the
    phase-correlation criterion is then applied as a filter on top, since
    it requires the (comparatively expensive) full time-series comparison.

    Parameters
    ----------
    node_coords : np.ndarray, shape (N, 2)
        Ground coordinates (e.g. meters, in a projected CRS) of each
        candidate node/pixel, ``(x, y)`` or ``(row * spacing, col *
        spacing)``.
    phase_flat : np.ndarray, shape (T, N)
        Phase time series for every node, aligned with ``node_coords``
        (i.e. the raster's ``(T, H, W)`` phase stack reshaped to
        ``(T, H * W)``).
    config : GraphConstructionConfig
        Edge construction thresholds (``max_distance_m``,
        ``min_phase_correlation``, ``max_neighbors``).
    show_progress : bool, default False
        Whether to display a tqdm progress bar over nodes.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E), dtype int64
        Source/target node indices for every (directed) edge. Edges are
        added symmetrically (undirected graph, stored as two directed
        edges per undirected pair) for use with standard PyG message
        passing.
    edge_attr : np.ndarray, shape (E, 2), dtype float32
        Per-edge features: ``[normalized_distance, phase_correlation]``.
        Distance is normalized by ``config.max_distance_m`` to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``node_coords`` and ``phase_flat`` have inconsistent node
        counts.
    """
    n_nodes = node_coords.shape[0]
    if phase_flat.shape[1] != n_nodes:
        raise ValueError(f"phase_flat has {phase_flat.shape[1]} nodes but node_coords has {n_nodes}")

    tree = cKDTree(node_coords)
    sources: list[int] = []
    targets: list[int] = []
    distances: list[float] = []
    correlations: list[float] = []

    iterator = range(n_nodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Building graph edges", unit="node")

    for i in iterator:
        dists, neighbor_idx = tree.query(
            node_coords[i],
            k=min(config.max_neighbors + 1, n_nodes),  # +1 to account for self
            distance_upper_bound=config.max_distance_m,
        )
        dists = np.atleast_1d(dists)
        neighbor_idx = np.atleast_1d(neighbor_idx)

        for dist, j in zip(dists, neighbor_idx):
            if j == i or j >= n_nodes or not np.isfinite(dist):
                continue
            if j <= i:
                # Avoid duplicating undirected pairs; symmetric edges are
                # added explicitly below once the pair passes both filters.
                continue

            corr = _phase_correlation(phase_flat[:, i], phase_flat[:, j])
            if corr <= config.min_phase_correlation:
                continue

            norm_dist = float(dist / config.max_distance_m)
            # Add both directions so message passing is symmetric.
            sources.extend([i, j])
            targets.extend([j, i])
            distances.extend([norm_dist, norm_dist])
            correlations.extend([corr, corr])

    if not sources:
        logger.warning(
            "build_graph produced zero edges; consider relaxing max_distance_m "
            "(%.1f m) or min_phase_correlation (%.2f).",
            config.max_distance_m,
            config.min_phase_correlation,
        )
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 2), dtype=np.float32)
    else:
        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = np.array(list(zip(distances, correlations)), dtype=np.float32)

    logger.info(
        "Built graph with %d nodes and %d directed edges (%.2f avg degree).",
        n_nodes,
        edge_index.shape[1],
        edge_index.shape[1] / max(n_nodes, 1),
    )
    return edge_index, edge_attr


def build_pyg_dataset(
    amp_path: str | Path,
    phase_path: str | Path,
    dem_path: str | Path,
    labels: np.ndarray,
    graph_config: GraphConstructionConfig,
    worldcover: np.ndarray | None = None,
    incidence_angle: float | np.ndarray = 38.0,
    memmap_threshold_gb: float = 10.0,
    output_path: str | Path | None = None,
) -> Data:
    """End-to-end: rasters + labels -> a single PyG ``Data`` graph object.

    Loads amplitude/phase/DEM rasters (using ``rasterio``, and
    ``np.memmap`` for stacks larger than ``memmap_threshold_gb`` to avoid
    exhausting memory on huge scenes), computes the 19-D node feature
    vector for every pixel, constructs edges by spatial proximity + phase
    correlation, and assembles everything into a
    :class:`torch_geometric.data.Data` object with ``x``, ``edge_index``,
    ``edge_attr``, ``y``, ``pos``, and ``phase`` attributes.

    Parameters
    ----------
    amp_path, phase_path : str or pathlib.Path
        Paths to the multi-temporal amplitude / phase rasters (GeoTIFF or
        ``.npy``), shape ``(T, H, W)``.
    dem_path : str or pathlib.Path
        Path to the co-registered DEM raster, shape ``(H, W)``. Used here
        only to derive the pixel grid / geotransform for ground distances;
        elevation values themselves are consumed by later 3D visualization
        stages, not by the node features.
    labels : np.ndarray, shape (H, W)
        Binary PS labels aligned with the raster grid (``0``/``1``), e.g.
        from :mod:`ps_gnn.data.label_generation`. Unlabeled pixels should
        be encoded as ``-1`` and are excluded from downstream losses (but
        still contribute to the graph structure as context nodes with
        ``y = -1``).
    graph_config : GraphConstructionConfig
        Edge construction thresholds.
    worldcover : np.ndarray, shape (H, W), optional
        ESA WorldCover class codes. If omitted, defaults to an
        all-"bare/sparse vegetation" grid (a neutral placeholder).
    incidence_angle : float or np.ndarray, default 38.0
        Scalar or per-pixel incidence angle in degrees.
    memmap_threshold_gb : float, default 10.0
        Rasters larger than this (in GB) are loaded via ``np.memmap``
        instead of being read fully into memory.
    output_path : str or pathlib.Path, optional
        If given, the resulting ``Data`` object is serialized to this path
        via ``torch.save``.

    Returns
    -------
    torch_geometric.data.Data
        Graph with ``x`` (N, 19), ``edge_index`` (2, E), ``edge_attr``
        (E, 2), ``y`` (N,), ``pos`` (N, 2) node pixel coordinates, and
        ``phase`` (T, N) — the flattened phase time series per node,
        attached for use by the physics-informed loss
        (:class:`ps_gnn.models.losses.PhysicsInformedPSLoss`) and SBAS
        validation (:mod:`ps_gnn.validation.sbas_check`), neither of
        which is part of the standard PyG ``Data`` schema.

    Raises
    ------
    ImportError
        If ``torch`` / ``torch_geometric`` are not installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "build_pyg_dataset requires torch and torch_geometric. "
            "Install with `pip install torch torch-geometric`."
        ) from exc

    amplitude = _load_raster_stack(amp_path, memmap_threshold_gb)
    phase = _load_raster_stack(phase_path, memmap_threshold_gb)
    dem = _load_raster_stack(dem_path, memmap_threshold_gb)
    if dem.ndim == 3:
        dem = dem[0]

    height, width = dem.shape
    if worldcover is None:
        worldcover = np.full((height, width), 60, dtype=np.int32)  # bare/sparse

    logger.info("Computing node features for a %dx%d grid...", height, width)
    features = compute_node_features(amplitude, phase, incidence_angle, worldcover)
    x = features.reshape(N_NODE_FEATURES, -1).T  # (N, 19)

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    node_coords = np.stack(
        [cols.ravel() * graph_config.pixel_spacing_m, rows.ravel() * graph_config.pixel_spacing_m],
        axis=1,
    ).astype(np.float32)

    phase_flat = phase.reshape(phase.shape[0], -1)  # (T, N)
    edge_index, edge_attr = build_graph(node_coords, phase_flat, graph_config)

    y = labels.reshape(-1).astype(np.int64)

    data = Data(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        pos=torch.from_numpy(node_coords),
    )
    data.phase = torch.from_numpy(phase_flat.astype(np.float32))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        logger.info("Saved graph Data object to %s", output_path)

    return data


def _load_raster_stack(path: str | Path, memmap_threshold_gb: float) -> np.ndarray:
    """Load a raster stack (GeoTIFF or ``.npy``), memmapping if large.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.npy`` array or a GeoTIFF (read via ``rasterio``).
    memmap_threshold_gb : float
        If the file on disk exceeds this size (GB), open it memory-mapped
        rather than loading it fully into RAM.

    Returns
    -------
    np.ndarray
        The loaded (or memmapped) array, shape ``(T, H, W)`` or ``(H, W)``.
    """
    path = Path(path)
    size_gb = path.stat().st_size / (1024**3)
    mmap_mode = "r" if size_gb > memmap_threshold_gb else None

    if path.suffix == ".npy":
        return np.load(path, mmap_mode=mmap_mode)

    import rasterio

    with rasterio.open(path) as src:
        if size_gb > memmap_threshold_gb:
            logger.info(
                "%s is %.1f GB (> %.1f GB threshold); consider pre-converting "
                "to a memmap-friendly format for repeated access.",
                path.name,
                size_gb,
                memmap_threshold_gb,
            )
        return src.read()
