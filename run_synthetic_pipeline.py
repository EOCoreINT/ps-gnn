#!/usr/bin/env python3
"""
run_synthetic_pipeline_hardened.py
====================================
A *scientifically hardened* end-to-end synthetic pipeline for ps-gnn.

This is a deliberate rewrite of `run_synthetic_pipeline.py` to fix three
issues discovered by inspecting that script's own log output:

1. **Near-edgeless graph.** `inject_synthetic_ps`'s "perfectly stable"
   points have *exactly* zero phase variance. `_phase_correlation`
   returns 0.0 (its degenerate-signal fallback) whenever either input
   series has zero variance -- so two exactly-noise-free PS points can
   *never* form an edge with each other, no matter how close together
   they are. The original run's "332 directed edges / 0.02 avg degree"
   were essentially all chance correlations among random background
   noise, not real structure. This script instead injects PS points with
   small, controlled, non-zero amplitude/phase noise (still well within
   the classical ADI<0.25 PS threshold), so genuine edges can form.

2. **Trivially separable classification task.** Exact-zero-ADI points
   are perfectly separable on a single feature; F1=1.0 by epoch 7 proves
   nothing about the graph or the physics-informed loss. This script
   additionally injects "false neighbor" pixels -- moderately noisy,
   vegetation-like scatterers placed immediately adjacent to each true
   PS point, with phase deliberately blended to partially resemble the
   PS point's own phase -- so a naive proximity/correlation-based graph
   *will* connect them to the true PS. This directly tests the paper's
   central claim: does the trained GAT layer learn to assign *lower*
   attention weight to these false-neighbor edges than to edges between
   genuinely similar, stable points?

3. **Baseline mislabeling.** The original script passed a uniformly
   *random* pixel sample as `adi_baseline_mask` to `run_sbas_validation`,
   which tests "clean points beat random noise" (true by construction),
   not "PS-GNN beats classical ADI thresholding" (the actual project
   hypothesis). This script instead builds a genuine top-K-lowest-ADI
   baseline, matched in size to the GNN's detection count.

No changes are made to the `ps_gnn` package itself -- everything below
uses only its existing public API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from ps_gnn.analytics.report_generator import generate_html_report
from ps_gnn.data.label_generation import inject_synthetic_ps
from ps_gnn.data.preprocessing import GraphConstructionConfig, build_graph, compute_node_features
from ps_gnn.inference.detector import PSDetector, TilingConfig
from ps_gnn.models.ps_gnn import PSGNN
from ps_gnn.training.trainer import Trainer, TrainerConfig
from ps_gnn.validation.sbas_check import run_sbas_validation
from ps_gnn.visualization.charts import plot_training_curves
from ps_gnn.visualization.maps import create_3d_terrain_plot, create_folium_map
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hardened_pipeline")

# Node role codes, used only within this script for diagnostics (not part
# of the ps_gnn package's schema).
ROLE_BACKGROUND = 0
ROLE_TRUE_PS = 1
ROLE_FALSE_NEIGHBOR = 2


def inject_realistic_ps_with_false_neighbors(
    base_amplitude: np.ndarray,
    base_phase: np.ndarray,
    n_ps: int,
    min_separation_px: int,
    random_state: int,
    amplitude_percentile: float,
    amplitude_boost: float,
    amp_noise_std_frac: float = 0.03,
    phase_noise_std: float = 0.10,
    n_false_neighbors_per_ps: int = 2,
    false_neighbor_max_offset_px: int = 2,
    false_neighbor_adi: float = 0.5,
    false_neighbor_extra_phase_std: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject realistic (non-zero-variance) PS points plus adjacent 'false neighbors'.

    Starts from :func:`ps_gnn.data.label_generation.inject_synthetic_ps`
    (which places `n_ps` perfectly-stable points), then:

    1. Adds small, controlled Gaussian noise to those points' amplitude
       and phase, giving them realistic (low but non-zero) ADI and phase
       variance -- enough that :func:`ps_gnn.data.preprocessing.build_graph`
       can actually compute a meaningful phase correlation between them,
       rather than hitting the zero-variance degenerate case.
    2. For each true PS point, places `n_false_neighbors_per_ps` pixels
       within `false_neighbor_max_offset_px` pixels of it whose phase is
       the true PS's own phase series *plus* a small amount of additional
       noise (rather than a weighted blend -- see note below), but whose
       *amplitude* is genuinely unstable (moderate ADI). This models a
       specific, deceptive failure mode: a scatterer that is phase-coherent
       with (and therefore graph-connected to) a true PS -- e.g. via
       geometric coupling or layover -- but is not itself a stable
       reflector. A model relying on phase-correlation-driven proximity
       alone would be fooled by this edge; one that also attends to the
       full node feature vector (ADI, local texture, ...) should not be.

    Why additive noise, not a weighted blend
    -------------------------------------------
    An earlier version of this function built false-neighbor phase as
    ``blend * true_ps_phase + (1 - blend) * independent_noise``. This
    turned out to reliably produce *near-zero* correlation with the true
    PS even at ``blend=0.9``: Pearson correlation normalizes by each
    signal's own standard deviation, and a genuinely stable PS point has
    (by definition) a tiny phase standard deviation -- so any
    independent-noise component with variance comparable to or larger
    than that tiny value dominates the correlation estimate regardless of
    its blend weight. Adding a small amount of *absolute* extra noise on
    top of the true PS's own phase series (rather than mixing in an
    independent, full-range random signal) keeps the combined signal's
    variance in the same small regime as the true PS's own noise, which
    is what actually preserves a strong, edge-forming correlation.

    Parameters
    ----------
    base_amplitude, base_phase : np.ndarray, shape (T, H, W)
        Background SAR stack to inject into.
    n_ps : int
        Number of true PS points to place.
    min_separation_px : int
        Minimum spacing between true PS points.
    random_state : int
        Seed.
    amplitude_percentile, amplitude_boost : float
        Passed through to `inject_synthetic_ps` for the base PS amplitude.
    amp_noise_std_frac : float, default 0.03
        Relative amplitude noise (fraction of the base PS amplitude)
        added to true PS points, giving ADI ~= this value.
    phase_noise_std : float, default 0.10
        Phase noise (radians) added to true PS points.
    n_false_neighbors_per_ps : int, default 2
        Number of false-neighbor pixels planted next to each true PS.
    false_neighbor_max_offset_px : int, default 2
        Max pixel offset (Chebyshev) for false-neighbor placement.
    false_neighbor_adi : float, default 0.5
        Target ADI for false-neighbor pixels (moderate amplitude
        instability -- well above the classical ADI<0.25 PS threshold).
    false_neighbor_extra_phase_std : float, default 0.15
        Extra phase noise (radians), added on top of the true PS's own
        phase series, for each false neighbor. Calibrated (see module
        tests) to keep phase correlation with the true PS comfortably
        above typical graph-construction thresholds (~30% of pairs clear
        0.4 correlation, vs. ~1.5% for unrelated background pairs at the
        same threshold) while remaining a distinct, noisier signal.

    Returns
    -------
    amplitude, phase : np.ndarray, shape (T, H, W)
        Modified stacks.
    node_roles : np.ndarray, shape (H, W), dtype uint8
        ``ROLE_BACKGROUND`` (0), ``ROLE_TRUE_PS`` (1), or
        ``ROLE_FALSE_NEIGHBOR`` (2) for every pixel.
    """
    t_steps, height, width = base_amplitude.shape
    rng = np.random.default_rng(random_state)

    base_result = inject_synthetic_ps(
        base_amplitude,
        base_phase,
        n_ps=n_ps,
        min_separation_px=min_separation_px,
        random_state=random_state,
        amplitude_percentile=amplitude_percentile,
        amplitude_boost=amplitude_boost,
    )
    amplitude = base_result.amplitude.copy()
    phase = base_result.phase.copy()
    node_roles = np.zeros((height, width), dtype=np.uint8)

    ps_rows, ps_cols = np.nonzero(base_result.labels)
    occupied = {(int(r), int(c)) for r, c in zip(ps_rows, ps_cols)}

    for r, c in zip(ps_rows, ps_cols):
        r, c = int(r), int(c)
        node_roles[r, c] = ROLE_TRUE_PS

        # Add realistic (non-zero) noise so this point isn't a mathematically
        # degenerate zero-variance signal -- this is what lets it actually
        # form graph edges.
        base_high_amplitude = float(amplitude[0, r, c])  # constant across T pre-noise
        amp_noise = rng.normal(0.0, amp_noise_std_frac * base_high_amplitude, size=t_steps)
        amplitude[:, r, c] = np.clip(base_high_amplitude + amp_noise, a_min=1e-3, a_max=None)

        phase_noise = rng.normal(0.0, phase_noise_std, size=t_steps)
        phase[:, r, c] = ((phase_noise + np.pi) % (2 * np.pi)) - np.pi

        true_ps_phase_series = phase[:, r, c].copy()

        placed = 0
        attempts = 0
        while placed < n_false_neighbors_per_ps and attempts < 20:
            attempts += 1
            dr = int(rng.integers(-false_neighbor_max_offset_px, false_neighbor_max_offset_px + 1))
            dc = int(rng.integers(-false_neighbor_max_offset_px, false_neighbor_max_offset_px + 1))
            nr, nc = r + dr, c + dc
            if dr == 0 and dc == 0:
                continue
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in occupied:
                continue

            # Vegetation-like scatterer: moderate amplitude, high fractional
            # noise (target ADI ~= false_neighbor_adi).
            base_amp_here = float(np.mean(base_amplitude[:, nr, nc]))
            fn_amp = base_amp_here * (1.0 + rng.normal(0.0, false_neighbor_adi, size=t_steps))
            amplitude[:, nr, nc] = np.clip(fn_amp, a_min=1e-3, a_max=None)

            # Phase: the true PS's own (already-noisy) phase series, plus a
            # small amount of *additional* noise -- see the "Why additive
            # noise" note in this function's docstring for why a weighted
            # blend with independent noise does not work here.
            extra_noise = rng.normal(0.0, false_neighbor_extra_phase_std, size=t_steps)
            fn_phase = true_ps_phase_series + extra_noise
            phase[:, nr, nc] = ((fn_phase + np.pi) % (2 * np.pi)) - np.pi

            node_roles[nr, nc] = ROLE_FALSE_NEIGHBOR
            occupied.add((nr, nc))
            placed += 1

    return amplitude, phase, node_roles


def compute_edge_role_statistics(edge_index: np.ndarray, node_roles: np.ndarray) -> dict[str, int]:
    """Categorize graph edges by the roles of their endpoints, for diagnostics.

    Parameters
    ----------
    edge_index : np.ndarray, shape (2, E)
    node_roles : np.ndarray, shape (H, W), dtype uint8
        Flattened internally to align with node indices.

    Returns
    -------
    dict[str, int]
        Counts of undirected edge pairs by category:
        ``ps_ps``, ``ps_false_neighbor``, ``ps_background``,
        ``false_neighbor_false_neighbor``, ``false_neighbor_background``,
        ``background_background``.
    """
    roles_flat = node_roles.ravel()
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst  # count each undirected pair once
    src, dst = src[keep], dst[keep]
    src_roles, dst_roles = roles_flat[src], roles_flat[dst]

    pair_roles = np.stack(
        [np.minimum(src_roles, dst_roles), np.maximum(src_roles, dst_roles)], axis=1
    )

    def _count(role_a: int, role_b: int) -> int:
        return int(np.sum((pair_roles[:, 0] == role_a) & (pair_roles[:, 1] == role_b)))

    return {
        "ps_ps": _count(ROLE_TRUE_PS, ROLE_TRUE_PS),
        "ps_false_neighbor": _count(ROLE_TRUE_PS, ROLE_FALSE_NEIGHBOR),
        "ps_background": _count(ROLE_TRUE_PS, ROLE_BACKGROUND),
        "false_neighbor_false_neighbor": _count(ROLE_FALSE_NEIGHBOR, ROLE_FALSE_NEIGHBOR),
        "false_neighbor_background": _count(ROLE_FALSE_NEIGHBOR, ROLE_BACKGROUND),
        "background_background": _count(ROLE_BACKGROUND, ROLE_BACKGROUND),
    }


def compute_attention_by_edge_role(
    model: PSGNN, x: torch.Tensor, edge_index: torch.Tensor, node_roles: np.ndarray
) -> dict[str, float]:
    """Run one forward pass and average GAT attention weight by edge role category.

    This is the key mechanistic check for the project's central hypothesis:
    a model that has actually learned to identify stable scatterers should
    assign *lower* mean attention to ``ps_false_neighbor`` edges (a true PS
    connected to a deliberately-planted unstable, superficially-similar
    neighbor) than to ``ps_ps`` edges (two genuinely stable points) or
    edges among consistently noisy background.

    Parameters
    ----------
    model : PSGNN
    x : Tensor, shape (N, 19)
    edge_index : Tensor, shape (2, E)
    node_roles : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    dict[str, float]
        Mean attention weight per edge-role category present in the graph
        (categories with zero edges are omitted).
    """
    model.eval()
    with torch.no_grad():
        hidden = model.conv1(model.encoder(x.to(model.device)), edge_index.to(model.device))
        hidden = model.bn1(hidden).relu()
        _, (attn_edge_index, attn_weights) = model.conv2(
            hidden, edge_index.to(model.device), return_attention_weights=True
        )
        attn_mean = attn_weights.mean(dim=-1).cpu().numpy()

    aei = attn_edge_index.cpu().numpy()
    roles_flat = node_roles.ravel()
    src, dst = aei[0], aei[1]

    non_self_loop = src != dst
    src, dst, attn_mean = src[non_self_loop], dst[non_self_loop], attn_mean[non_self_loop]
    src_roles, dst_roles = roles_flat[src], roles_flat[dst]

    categories = {
        "ps_ps": (ROLE_TRUE_PS, ROLE_TRUE_PS),
        "ps_false_neighbor": (ROLE_TRUE_PS, ROLE_FALSE_NEIGHBOR),
        "ps_background": (ROLE_TRUE_PS, ROLE_BACKGROUND),
        "background_background": (ROLE_BACKGROUND, ROLE_BACKGROUND),
    }

    result = {}
    for name, (role_a, role_b) in categories.items():
        mask = ((src_roles == role_a) & (dst_roles == role_b)) | (
            (src_roles == role_b) & (dst_roles == role_a)
        )
        if mask.sum() > 0:
            result[name] = float(attn_mean[mask].mean())
    return result


def build_adi_baseline_mask(features: np.ndarray, n_select: int) -> np.ndarray:
    """A genuine classical baseline: the N pixels with the lowest ADI.

    This is what `run_sbas_validation`'s hypothesis check is actually
    meant to compare PS-GNN against -- classical Amplitude Dispersion
    Index thresholding -- not a random pixel sample.

    Parameters
    ----------
    features : np.ndarray, shape (19, H, W)
        Node feature stack; index 2 is ADI (see
        :data:`ps_gnn.data.preprocessing.FEATURE_NAMES`).
    n_select : int
        Number of lowest-ADI pixels to select.

    Returns
    -------
    np.ndarray, shape (H * W,), dtype bool
    """
    adi_flat = features[2].ravel()
    order = np.argsort(adi_flat)
    mask = np.zeros(adi_flat.shape[0], dtype=bool)
    mask[order[:n_select]] = True
    return mask


def main() -> None:
    logger.info("=" * 80)
    logger.info("HARDENED synthetic PS-GNN pipeline -- stress-testing the science")
    logger.info("=" * 80)
    output_dir = Path("synthetic_test_report_hardened")
    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: numpy's RNG below is seeded, but that alone does NOT make this
    # script reproducible -- PSGNN's weight initialization, GATConv's
    # internal randomness, and Trainer's dropout masks all draw from
    # PyTorch's own global RNG, which is otherwise seeded from OS entropy
    # and differs on every process invocation. Seed it explicitly too.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    rng = np.random.default_rng(42)
    t_steps, height, width = 30, 128, 128
    n_ps = 150

    # -------------------------------------------------------------------
    # Step 1: Background + realistic PS + false-neighbor injection
    # -------------------------------------------------------------------
    logger.info(
        "Step 1: Generating background SAR stack (%d scenes, %dx%d)...", t_steps, height, width
    )
    base_amp = rng.gamma(shape=2.0, scale=10.0, size=(t_steps, height, width)).astype(np.float32)
    base_phase = rng.uniform(-np.pi, np.pi, size=(t_steps, height, width)).astype(np.float32)
    worldcover = np.full((height, width), 30, dtype=np.int32)

    # Synthetic DEM (Gaussian hill + noise), for the 3D terrain visualization.
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    center_x, center_y = width // 2, height // 2
    dem = 100.0 + 50.0 * np.exp(
        -((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) / (2 * 30**2)
    )
    dem += rng.normal(0, 2.0, size=(height, width))
    dem = dem.astype(np.float32)

    logger.info(
        "Step 2: Injecting %d realistic (non-zero-variance) PS points, each with "
        "2 planted 'false neighbor' pixels...",
        n_ps,
    )
    amplitude, phase, node_roles = inject_realistic_ps_with_false_neighbors(
        base_amp,
        base_phase,
        n_ps=n_ps,
        min_separation_px=6,
        random_state=42,
        amplitude_percentile=90.0,
        amplitude_boost=3.0,
        amp_noise_std_frac=0.03,
        phase_noise_std=0.10,
        n_false_neighbors_per_ps=2,
        false_neighbor_max_offset_px=2,
        false_neighbor_adi=0.5,
        false_neighbor_extra_phase_std=0.15,
    )
    n_true_ps = int(np.sum(node_roles == ROLE_TRUE_PS))
    n_false_neighbors = int(np.sum(node_roles == ROLE_FALSE_NEIGHBOR))
    logger.info(
        "   -> Placed %d true PS points and %d false-neighbor (hard negative) pixels.",
        n_true_ps,
        n_false_neighbors,
    )

    # Labels: true PS -> 1, false neighbors -> 0 (the critical hard negative),
    # plus a broader sample of ordinary background -> 0, rest unlabeled (-1).
    y_flat = np.full(height * width, -1, dtype=np.int64)
    roles_flat = node_roles.ravel()
    y_flat[roles_flat == ROLE_TRUE_PS] = 1
    y_flat[roles_flat == ROLE_FALSE_NEIGHBOR] = 0

    background_idx = np.flatnonzero(roles_flat == ROLE_BACKGROUND)
    n_bg_negatives = min(400, len(background_idx))
    bg_negative_idx = rng.choice(background_idx, size=n_bg_negatives, replace=False)
    y_flat[bg_negative_idx] = 0

    logger.info(
        "   -> Labels: %d PS, %d false-neighbor hard negatives, %d background hard negatives, rest unlabeled.",
        int((y_flat == 1).sum()),
        n_false_neighbors,
        n_bg_negatives,
    )

    # -------------------------------------------------------------------
    # Step 2: Graph construction (denser, appropriate for non-degenerate signals)
    # -------------------------------------------------------------------
    logger.info("Step 3: Computing node features and building graph...")
    graph_config = GraphConstructionConfig(
        max_distance_m=25.0,
        min_phase_correlation=0.4,
        pixel_spacing_m=10.0,
        max_neighbors=12,
    )

    features = compute_node_features(amplitude, phase, 38.0, worldcover)
    x = torch.from_numpy(features.reshape(19, -1).T.astype(np.float32))

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    node_coords = np.stack(
        [cols.ravel() * graph_config.pixel_spacing_m, rows.ravel() * graph_config.pixel_spacing_m],
        axis=1,
    ).astype(np.float32)

    phase_flat = phase.reshape(t_steps, -1)
    edge_index_np, edge_attr_np = build_graph(
        node_coords, phase_flat, graph_config, show_progress=False
    )

    avg_degree = edge_index_np.shape[1] / (height * width)
    logger.info(
        "   -> Graph built: %d nodes, %d directed edges (%.3f avg degree, vs. 0.02 in the naive run).",
        height * width,
        edge_index_np.shape[1],
        avg_degree,
    )

    edge_stats = compute_edge_role_statistics(edge_index_np, node_roles)
    logger.info("   -> Edge composition by ground-truth role:")
    for category, count in edge_stats.items():
        logger.info("        %-32s %d", category, count)
    if edge_stats["ps_false_neighbor"] == 0:
        logger.warning(
            "   -> No ps<->false_neighbor edges formed! The attention diagnostic below "
            "will be uninformative. Consider raising max_distance_m or lowering "
            "min_phase_correlation."
        )

    data = Data(
        x=x,
        edge_index=torch.from_numpy(edge_index_np),
        edge_attr=torch.from_numpy(edge_attr_np),
        y=torch.from_numpy(y_flat),
        pos=torch.from_numpy(node_coords),
    )
    data.phase = torch.from_numpy(phase_flat)

    # -------------------------------------------------------------------
    # Step 3: Train
    # -------------------------------------------------------------------
    logger.info("Step 4: Training PS-GNN for 25 epochs...")
    model = PSGNN()
    trainer_config = TrainerConfig(
        n_epochs=25,
        warmup_epochs=3,
        n_spatial_blocks=4,
        log_dir=None,
        learning_rate=1e-3,
        # The default curriculum (epochs 1-20: correlation > 0.7) would
        # filter out almost every edge we just worked to create, since our
        # graph was built at min_phase_correlation=0.4. Start the
        # curriculum at that same threshold so the graph is actually used
        # from epoch 1, then relax to the full graph quickly.
        curriculum_schedule=[(1, 0.3), (6, -1.0)],
    )
    trainer = Trainer(model, trainer_config)
    history = trainer.fit(data)

    final_f1 = history[-1]["val_f1"]
    logger.info("   -> Training complete. Final Validation F1: %.4f", final_f1)
    if final_f1 >= 0.999:
        logger.warning(
            "   -> F1 is essentially perfect. With non-trivial noise now injected, this is "
            "a meaningfully stronger result than the naive run's F1=1.0 -- but still sanity-"
            "check against the baseline classifier below before trusting it fully."
        )

    fig_train = plot_training_curves(history)
    fig_train.write_html(str(output_dir / "training_curves.html"), include_plotlyjs="cdn")

    # -------------------------------------------------------------------
    # Step 4: Attention mechanism diagnostic (the actual point of this script)
    # -------------------------------------------------------------------
    logger.info("Step 5: Checking whether GAT attention learned to distrust false neighbors...")
    attention_by_role = compute_attention_by_edge_role(model, x, data.edge_index, node_roles)
    for category, mean_attn in attention_by_role.items():
        logger.info("   -> mean attention on %-20s edges: %.4f", category, mean_attn)

    if "ps_false_neighbor" in attention_by_role and "background_background" in attention_by_role:
        fn_attn = attention_by_role["ps_false_neighbor"]
        bg_attn = attention_by_role["background_background"]
        # Note: `ps_ps` edges are essentially guaranteed to be absent here --
        # true PS points are sparsely scattered (realistic; individual
        # stable structures aren't usually adjacent to each other), so they
        # rarely fall within max_distance_m of one another. We therefore
        # compare against background<->background edges (chance-correlated
        # noise pairs) as the "generic, uninformative edge" reference point
        # instead. Caveat: GATConv applies softmax normalization per target
        # node, so a node's per-edge attention share also depends on its
        # degree -- this comparison is a useful directional signal, not a
        # fully degree-controlled statistical test.
        if fn_attn < bg_attn:
            logger.info(
                "   PASS: mean attention on ps<->false_neighbor edges (%.4f) is LOWER than on "
                "generic background<->background edges (%.4f): the model is discounting the "
                "planted noisy neighbors below baseline, despite the edge existing.",
                fn_attn,
                bg_attn,
            )
        else:
            logger.warning(
                "   NOT CONFIRMED: mean attention on ps<->false_neighbor edges (%.4f) is NOT "
                "lower than generic background<->background edges (%.4f) -- no evidence the "
                "attention mechanism is discounting these deceptive neighbors (try more epochs, "
                "more false-neighbor examples via n_false_neighbors_per_ps, or a stronger ADI "
                "contrast).",
                fn_attn,
                bg_attn,
            )
    else:
        logger.warning(
            "   -> Not enough edges of both types to compare -- see edge composition above."
        )

    # -------------------------------------------------------------------
    # Step 5: Inference
    # -------------------------------------------------------------------
    logger.info("Step 6: Running tiled inference with Monte Carlo Dropout...")
    detector = PSDetector(
        model=model,
        tiling_config=TilingConfig(tile_size=64, overlap=16),
        graph_config=graph_config,
        mc_dropout_passes=5,
        confidence_threshold=0.85,
    )
    results = detector.detect(
        amplitude=amplitude, phase=phase, incidence_angle=38.0, worldcover=worldcover
    )
    n_detected = int(results["ps_mask"].sum())
    logger.info("   -> Detected %d PS points (true PS count was %d).", n_detected, n_true_ps)

    # -------------------------------------------------------------------
    # Step 6: SBAS validation against a REAL ADI baseline (not random)
    # -------------------------------------------------------------------
    logger.info("Step 7: Running SBAS validation vs. a genuine ADI-threshold baseline...")
    n_baseline = n_detected if n_detected > 0 else n_ps
    adi_baseline_mask = build_adi_baseline_mask(features, n_select=n_baseline)
    gnn_mask = results["ps_mask"].ravel()

    sbas_result = run_sbas_validation(
        phase=phase.reshape(t_steps, -1),
        ps_gnn_mask=gnn_mask,
        adi_baseline_mask=adi_baseline_mask,
        min_improvement_pct=10.0,
        strict=False,
    )
    logger.info(
        "   -> PS-GNN coherence=%.4f | ADI-threshold baseline coherence=%.4f | %.1f%% relative change",
        sbas_result.ps_gnn_mean_coherence,
        sbas_result.adi_baseline_mean_coherence,
        sbas_result.coherence_improvement_pct,
    )
    if sbas_result.hypothesis_passed:
        logger.info(
            "   PASS: hypothesis met against a genuine ADI baseline (not a random one this time)."
        )
    else:
        logger.warning("   NOT MET: hypothesis not satisfied against the ADI baseline.")

    # -------------------------------------------------------------------
    # Step 7: Visualizations & comprehensive HTML report
    # -------------------------------------------------------------------
    logger.info("Step 8: Generating visualizations and the comprehensive HTML report...")

    logger.info("   -> Generating 2D Folium map...")
    fmap = create_folium_map(
        ps_mask=results["ps_mask"],
        probability=results["probability"],
        cluster_labels=results["cluster_labels"],
        max_markers=300,
    )
    fmap.save(str(output_dir / "ps_detections_2d_map.html"))

    logger.info("   -> Generating 3D terrain plot...")
    fig_3d = create_3d_terrain_plot(
        dem=dem,
        ps_mask=results["ps_mask"],
        probability=results["probability"],
        pixel_spacing_m=graph_config.pixel_spacing_m,
        max_points=500,
    )
    fig_3d.write_html(str(output_dir / "ps_detections_3d_terrain.html"), include_plotlyjs="cdn")

    logger.info("   -> Generating comprehensive HTML report...")
    report_path = generate_html_report(
        results=results,
        output_dir=output_dir,
        training_history=history,
        dem=dem,
        pixel_spacing_m=graph_config.pixel_spacing_m,
        model_hash="hardened_v1",
    )

    logger.info("=" * 80)
    logger.info("HARDENED PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("Output directory: %s", output_dir.absolute())
    logger.info("  - Training curves:  %s", output_dir / "training_curves.html")
    logger.info("  - 2D Folium map:    %s", output_dir / "ps_detections_2d_map.html")
    logger.info("  - 3D terrain plot:  %s", output_dir / "ps_detections_3d_terrain.html")
    logger.info("  - Master report:    %s", report_path.absolute())
    logger.info("This run exercised: a genuinely connected graph, non-trivial noise, planted")
    logger.info("false-neighbor hard negatives with an attention-mechanism diagnostic, and a")
    logger.info("real classical-baseline comparison -- see Steps 3, 5, and 7 above.")


if __name__ == "__main__":
    main()