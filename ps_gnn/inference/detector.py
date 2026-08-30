"""
ps_gnn.inference.detector
============================

Production inference pipeline for PS-GNN. Handles the three concerns a
research forward-pass script doesn't need to worry about but a real
deployment does:

1. **Tiling** — full Sentinel-1 scenes are far too large to fit as a
   single graph in GPU memory. :func:`generate_tiles` splits the scene
   into overlapping windows; each tile is processed independently and the
   results are stitched back together using only each tile's non-overlap
   "core" region, avoiding boundary artifacts from double-processed
   overlap strips.
2. **Uncertainty** — :func:`mc_dropout_predict` runs multiple stochastic
   forward passes with dropout kept active at inference time (Monte Carlo
   Dropout), giving both a mean PS probability and a variance that
   downstream users can use to flag low-confidence detections.
3. **Post-processing** — :func:`postprocess_predictions` applies a
   confidence threshold and groups adjacent high-confidence PS pixels
   into connected clusters (candidate structures), which is more
   actionable for a human reviewer than an unstructured point cloud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from tqdm import tqdm

from ps_gnn.data.preprocessing import (
    GraphConstructionConfig,
    build_graph,
    compute_node_features,
)
from ps_gnn.models.ps_gnn import PSGNN

logger = logging.getLogger(__name__)


@dataclass
class TileSpec:
    """Bounds of one processing tile and its non-overlap "core" region.

    Attributes
    ----------
    row0, row1, col0, col1 : int
        Half-open pixel bounds of the full tile (including overlap
        context) within the scene.
    core_row0, core_row1, core_col0, core_col1 : int
        Half-open pixel bounds of the sub-region whose predictions are
        actually kept when stitching tiles back together. Interior tile
        edges are trimmed by half the overlap (context is used for graph
        connectivity near the edge but not double-counted in the output);
        edges touching the scene boundary are not trimmed.
    """

    row0: int
    row1: int
    col0: int
    col1: int
    core_row0: int
    core_row1: int
    core_col0: int
    core_col1: int


@dataclass
class TilingConfig:
    """Sliding-window tiling parameters.

    Attributes
    ----------
    tile_size : int
        Side length (pixels) of each square tile, before scene-boundary
        clipping.
    overlap : int
        Overlap (pixels) between adjacent tiles, used as graph-building
        context near tile edges and trimmed from the stitched output.
    """

    tile_size: int = 1024
    overlap: int = 128


def generate_tiles(height: int, width: int, config: TilingConfig | None = None) -> list[TileSpec]:
    """Compute a sliding-window tiling that exactly covers an (H, W) grid.

    Tiles are stepped by ``tile_size - overlap`` along each axis; the last
    tile in each row/column is shifted inward (not simply clipped) so
    every tile has the full ``tile_size`` extent whenever the scene is at
    least that large, while core regions still partition the scene
    exactly (no gaps, no double-counted pixels).

    Parameters
    ----------
    height, width : int
        Scene dimensions in pixels.
    config : TilingConfig, optional
        Tiling parameters. Defaults to ``TilingConfig()`` (1024px tiles,
        128px overlap).

    Returns
    -------
    list[TileSpec]
        Tiles covering the scene; the union of every tile's core region
        exactly equals the full ``[0, height) x [0, width)`` grid with no
        overlap.

    Raises
    ------
    ValueError
        If ``overlap >= tile_size``.
    """
    config = config or TilingConfig()
    if config.overlap >= config.tile_size:
        raise ValueError(
            f"overlap ({config.overlap}) must be smaller than tile_size ({config.tile_size})"
        )

    def _axis_tiles(extent: int) -> list[tuple[int, int, int, int]]:
        """Return (start, end, core_start, core_end) along one axis."""
        tile_size = min(config.tile_size, extent)
        stride = max(tile_size - config.overlap, 1)

        starts = list(range(0, max(extent - tile_size, 0) + 1, stride))
        if not starts or starts[-1] + tile_size < extent:
            starts.append(max(extent - tile_size, 0))
        starts = sorted(set(starts))

        half_overlap = config.overlap // 2
        segments = []
        for i, start in enumerate(starts):
            end = start + tile_size
            core_start = start if i == 0 else start + half_overlap
            core_end = end if i == len(starts) - 1 else end - half_overlap
            segments.append((start, end, core_start, core_end))

        # Guarantee exact, non-overlapping, gap-free core coverage even
        # under edge-case rounding by snapping consecutive core boundaries
        # to meet exactly at their midpoint.
        for i in range(len(segments) - 1):
            s0, e0, cs0, ce0 = segments[i]
            s1, e1, cs1, ce1 = segments[i + 1]
            boundary = (ce0 + cs1) // 2
            segments[i] = (s0, e0, cs0, boundary)
            segments[i + 1] = (s1, e1, boundary, ce1)

        return segments

    row_segments = _axis_tiles(height)
    col_segments = _axis_tiles(width)

    tiles = []
    for row0, row1, core_row0, core_row1 in row_segments:
        for col0, col1, core_col0, core_col1 in col_segments:
            tiles.append(
                TileSpec(row0, row1, col0, col1, core_row0, core_row1, core_col0, core_col1)
            )
    return tiles


def _enable_mc_dropout(model: nn.Module) -> None:
    """Set every ``nn.Dropout`` submodule to training mode, in place.

    Used to keep dropout stochastic during otherwise-``eval()`` inference
    (Monte Carlo Dropout), while leaving BatchNorm and other layers in
    their normal evaluation behavior (running statistics, no update).

    Parameters
    ----------
    model : torch.nn.Module
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: PSGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_passes: int = 10,
    ps_class_index: int = 1,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate PS-probability mean and variance via Monte Carlo Dropout.

    Parameters
    ----------
    model : PSGNN
        A trained model. Its BatchNorm layers are left in evaluation mode
        (using running statistics); only Dropout layers are kept active.
    x : Tensor, shape (N, 19)
    edge_index : Tensor, shape (2, E)
    n_passes : int, default 10
        Number of stochastic forward passes.
    ps_class_index : int, default 1
        Index of the "PS" class in the model's output logits.
    show_progress : bool, default False
        Show a tqdm progress bar over passes.

    Returns
    -------
    mean_probability : np.ndarray, shape (N,)
        Mean predicted PS-class probability across passes.
    variance : np.ndarray, shape (N,)
        Variance of the PS-class probability across passes (epistemic
        uncertainty proxy).
    """
    model.eval()
    _enable_mc_dropout(model)

    all_probs = []
    iterator = range(n_passes)
    if show_progress:
        iterator = tqdm(iterator, desc="MC Dropout passes")

    with torch.no_grad():
        for _ in iterator:
            # NOTE: intentionally does not call model.predict_proba(), which
            # internally calls model.eval() and would silently switch every
            # Dropout submodule back to inference mode (0 dropout applied) on
            # every pass, collapsing all passes to an identical, deterministic
            # forward pass and reporting zero epistemic uncertainty.
            logits, _ = model(x, edge_index, return_attention=False)
            probs = torch.softmax(logits, dim=-1)[:, ps_class_index]
            all_probs.append(probs.cpu().numpy())

    stacked = np.stack(all_probs, axis=0)  # (n_passes, N)
    return stacked.mean(axis=0), stacked.var(axis=0)


def postprocess_predictions(
    mean_probability: np.ndarray,
    grid_shape: tuple[int, int],
    valid_row_col: np.ndarray,
    confidence_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold PS probabilities and cluster adjacent detections.

    Parameters
    ----------
    mean_probability : np.ndarray, shape (N,)
        Mean PS-class probability per node (see
        :func:`mc_dropout_predict`).
    grid_shape : tuple[int, int]
        ``(height, width)`` of the full scene.
    valid_row_col : np.ndarray, shape (N, 2)
        Row/column pixel coordinates of every node in ``mean_probability``.
    confidence_threshold : float, default 0.85
        Nodes with ``mean_probability < confidence_threshold`` are
        excluded from the final PS mask.

    Returns
    -------
    ps_mask : np.ndarray, shape (H, W), dtype bool
        True where a pixel is a high-confidence PS detection.
    cluster_labels : np.ndarray, shape (H, W), dtype int32
        Connected-component cluster ID for every PS pixel (8-connectivity),
        ``0`` for non-PS / unlabeled pixels.
    """
    height, width = grid_shape
    prob_grid = np.zeros((height, width), dtype=np.float32)
    prob_grid[valid_row_col[:, 0], valid_row_col[:, 1]] = mean_probability

    ps_mask = prob_grid >= confidence_threshold

    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    cluster_labels, n_clusters = ndimage.label(ps_mask, structure=structure)
    logger.info(
        "Post-processing: %d / %d pixels above %.2f confidence, forming %d clusters.",
        int(ps_mask.sum()),
        ps_mask.size,
        confidence_threshold,
        n_clusters,
    )
    return ps_mask, cluster_labels.astype(np.int32)


class PSDetector:
    """End-to-end tiled inference: rasters -> PS detections with uncertainty.

    Parameters
    ----------
    model : PSGNN
        A trained PS-GNN model.
    tiling_config : TilingConfig, optional
        Sliding-window tiling parameters.
    graph_config : GraphConstructionConfig, optional
        Graph edge-construction thresholds, applied independently within
        each tile.
    mc_dropout_passes : int, default 10
        Number of Monte Carlo Dropout forward passes per tile.
    confidence_threshold : float, default 0.85
        Minimum mean PS-probability to keep a detection.
    """

    def __init__(
        self,
        model: PSGNN,
        tiling_config: TilingConfig | None = None,
        graph_config: GraphConstructionConfig | None = None,
        mc_dropout_passes: int = 10,
        confidence_threshold: float = 0.85,
    ) -> None:
        self.model = model
        self.tiling_config = tiling_config or TilingConfig()
        self.graph_config = graph_config or GraphConstructionConfig()
        self.mc_dropout_passes = mc_dropout_passes
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        incidence_angle: float | np.ndarray,
        worldcover: np.ndarray | None = None,
        generate_report: bool = False,
        report_kwargs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Run tiled inference with uncertainty over a full scene.

        Parameters
        ----------
        amplitude, phase : np.ndarray, shape (T, H, W)
            Multi-temporal SAR stacks for the full scene.
        incidence_angle : float or np.ndarray
            Scalar or ``(H, W)`` incidence angle, degrees.
        worldcover : np.ndarray, shape (H, W), optional
            ESA WorldCover class codes. Defaults to a neutral placeholder
            if omitted (see :func:`ps_gnn.data.preprocessing.compute_node_features`).
        generate_report : bool, default False
            If True, attempt to trigger spatial-stats/mapping/report
            generation on the results after inference (see
            :mod:`ps_gnn.analytics.report_generator`). Wrapped in a
            try/except so that a failure in the (optional, UI-heavy)
            reporting stack — e.g. in a headless environment without a
            display or with reporting dependencies missing — never causes
            core inference to fail; a warning is logged instead.
        report_kwargs : dict, optional
            Extra keyword arguments forwarded to the report generator.

        Returns
        -------
        dict[str, np.ndarray]
            ``ps_mask`` (H, W bool), ``probability`` (H, W float32, mean
            MC-Dropout probability), ``uncertainty`` (H, W float32,
            MC-Dropout variance), ``cluster_labels`` (H, W int32).
        """
        _, height, width = amplitude.shape
        if worldcover is None:
            worldcover = np.full((height, width), 60, dtype=np.int32)

        tiles = generate_tiles(height, width, self.tiling_config)
        logger.info(
            "Running tiled inference over %d tiles (%dx%d scene).", len(tiles), height, width
        )

        prob_grid = np.zeros((height, width), dtype=np.float32)
        var_grid = np.zeros((height, width), dtype=np.float32)
        covered_mask = np.zeros((height, width), dtype=bool)

        for tile in tqdm(tiles, desc="Tiled inference"):
            amp_tile = amplitude[:, tile.row0 : tile.row1, tile.col0 : tile.col1]
            phase_tile = phase[:, tile.row0 : tile.row1, tile.col0 : tile.col1]
            wc_tile = worldcover[tile.row0 : tile.row1, tile.col0 : tile.col1]
            if np.isscalar(incidence_angle):
                inc_tile = incidence_angle
            else:
                inc_tile = incidence_angle[tile.row0 : tile.row1, tile.col0 : tile.col1]

            tile_h, tile_w = amp_tile.shape[1], amp_tile.shape[2]
            features = compute_node_features(amp_tile, phase_tile, inc_tile, wc_tile)
            x = torch.from_numpy(features.reshape(features.shape[0], -1).T.astype(np.float32))

            rows, cols = np.meshgrid(np.arange(tile_h), np.arange(tile_w), indexing="ij")
            node_coords = np.stack(
                [
                    cols.ravel() * self.graph_config.pixel_spacing_m,
                    rows.ravel() * self.graph_config.pixel_spacing_m,
                ],
                axis=1,
            ).astype(np.float32)
            phase_flat = phase_tile.reshape(phase_tile.shape[0], -1)

            edge_index_np, _ = build_graph(
                node_coords, phase_flat, self.graph_config, show_progress=False
            )
            edge_index = torch.from_numpy(edge_index_np)

            mean_prob, var_prob = mc_dropout_predict(
                self.model, x, edge_index, n_passes=self.mc_dropout_passes
            )
            mean_prob_grid = mean_prob.reshape(tile_h, tile_w)
            var_prob_grid = var_prob.reshape(tile_h, tile_w)

            cr0, cr1 = tile.core_row0 - tile.row0, tile.core_row1 - tile.row0
            cc0, cc1 = tile.core_col0 - tile.col0, tile.core_col1 - tile.col0

            prob_grid[tile.core_row0 : tile.core_row1, tile.core_col0 : tile.core_col1] = (
                mean_prob_grid[cr0:cr1, cc0:cc1]
            )
            var_grid[tile.core_row0 : tile.core_row1, tile.core_col0 : tile.core_col1] = (
                var_prob_grid[cr0:cr1, cc0:cc1]
            )
            covered_mask[tile.core_row0 : tile.core_row1, tile.core_col0 : tile.core_col1] = True

        if not covered_mask.all():
            logger.warning(
                "%d pixels were not covered by any tile's core region; check tiling config.",
                int((~covered_mask).sum()),
            )

        rows_all, cols_all = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        all_row_col = np.stack([rows_all.ravel(), cols_all.ravel()], axis=1)
        ps_mask, cluster_labels = postprocess_predictions(
            prob_grid.ravel(), (height, width), all_row_col, self.confidence_threshold
        )

        results = {
            "ps_mask": ps_mask,
            "probability": prob_grid,
            "uncertainty": var_grid,
            "cluster_labels": cluster_labels,
        }

        if generate_report:
            try:
                from ps_gnn.analytics.report_generator import generate_html_report

                generate_html_report(results, **(report_kwargs or {}))
            except Exception:
                logger.exception(
                    "Report generation failed; continuing without a report "
                    "(core inference results are unaffected)."
                )

        return results
