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
   they are. This script instead injects PS points with small, controlled,
   non-zero amplitude/phase noise (still well within the classical
   ADI<0.25 PS threshold), so genuine edges can form.

2. **Trivially separable classification task.** Exact-zero-ADI points
   are perfectly separable on a single feature. This script additionally
   injects "false neighbor" pixels (moderately noisy, vegetation-like
   scatterers next to each true PS point) and "borderline-ADI" PS clusters
   (genuine scatterers whose individual ADI exceeds the classical 0.25
   threshold, recoverable only via mutual phase corroboration) to create
   a genuinely hard, informative task.

3. **Baseline mislabeling.** Compares against a genuine fixed ADI<0.25
   classical threshold, not a random pixel sample.

Shared injection and diagnostic logic lives in ``synthetic_pipeline_utils``
(also used by the companion Jupyter notebook, ``ps_gnn_full_pipeline.ipynb``)
so both stay in sync. No changes are made to the ``ps_gnn`` package itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from ps_gnn.analytics.report_generator import generate_html_report
from ps_gnn.data.preprocessing import GraphConstructionConfig, build_graph, compute_node_features
from ps_gnn.inference.detector import PSDetector, TilingConfig
from ps_gnn.models.ps_gnn import PSGNN
from ps_gnn.training.trainer import Trainer, TrainerConfig
from ps_gnn.validation.sbas_check import run_sbas_validation
from ps_gnn.visualization.charts import plot_training_curves
from ps_gnn.visualization.maps import create_3d_terrain_plot, create_folium_map
from torch_geometric.data import Data

from synthetic_pipeline_utils import (
    ROLE_BACKGROUND,
    ROLE_BORDERLINE_PS,
    ROLE_FALSE_NEIGHBOR,
    ROLE_TRUE_PS,
    build_adi_baseline_mask,
    compute_attention_by_edge_role,
    compute_edge_role_statistics,
    compute_recall_by_role,
    diagnose_confidence_by_hit_miss,
    diagnose_spatial_contamination,
    inject_borderline_ps_clusters,
    inject_realistic_ps_with_false_neighbors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hardened_pipeline")


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

    n_false_neighbors_per_ps = (
        4  # bumped from 2: more statistical power for the attention diagnostic
    )
    logger.info(
        "Step 2: Injecting %d realistic (non-zero-variance) PS points, each with "
        "%d planted 'false neighbor' pixels...",
        n_ps,
        n_false_neighbors_per_ps,
    )
    amplitude, phase, node_roles, group_id = inject_realistic_ps_with_false_neighbors(
        base_amp,
        base_phase,
        n_ps=n_ps,
        min_separation_px=6,
        random_state=42,
        amplitude_percentile=90.0,
        amplitude_boost=3.0,
        amp_noise_std_frac=0.03,
        phase_noise_std=0.10,
        n_false_neighbors_per_ps=n_false_neighbors_per_ps,
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

    n_borderline_clusters = 10
    borderline_cluster_size = 4
    logger.info(
        "Step 2b: Injecting %d borderline-ADI PS clusters (%d points each, %d total) -- "
        "genuine scatterers whose individual ADI exceeds the classical 0.25 threshold, "
        "recoverable (if at all) only via mutual phase corroboration...",
        n_borderline_clusters,
        borderline_cluster_size,
        n_borderline_clusters * borderline_cluster_size,
    )
    amplitude, phase, node_roles, group_id = inject_borderline_ps_clusters(
        amplitude,
        phase,
        node_roles,
        group_id,
        n_clusters=n_borderline_clusters,
        cluster_size=borderline_cluster_size,
        random_state=43,
        min_cluster_separation_px=8,
        member_max_offset_px=2,
        borderline_amp_noise_std=0.30,
        cluster_shared_phase_std=0.15,
        member_individual_phase_std=0.05,
        amplitude_percentile=90.0,
        amplitude_boost=3.0,
    )
    n_borderline = int(np.sum(node_roles == ROLE_BORDERLINE_PS))
    logger.info("   -> Placed %d borderline-ADI PS points across clusters.", n_borderline)

    # Labels: true PS -> 1, borderline-cluster PS -> 1 (they ARE genuine
    # scatterers, just noisier), false neighbors -> 0 (the critical hard
    # negative), plus a broader sample of ordinary background -> 0, rest
    # unlabeled (-1).
    y_flat = np.full(height * width, -1, dtype=np.int64)
    roles_flat = node_roles.ravel()
    y_flat[roles_flat == ROLE_TRUE_PS] = 1
    y_flat[roles_flat == ROLE_BORDERLINE_PS] = 1
    y_flat[roles_flat == ROLE_FALSE_NEIGHBOR] = 0

    background_idx = np.flatnonzero(roles_flat == ROLE_BACKGROUND)
    n_bg_negatives = min(400, len(background_idx))
    bg_negative_idx = rng.choice(background_idx, size=n_bg_negatives, replace=False)
    y_flat[bg_negative_idx] = 0

    logger.info(
        "   -> Labels: %d easy PS + %d borderline PS = %d total PS, %d false-neighbor hard "
        "negatives, %d background hard negatives, rest unlabeled.",
        n_true_ps,
        n_borderline,
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
    logger.info("Step 4: Training PS-GNN for 50 epochs...")
    model = PSGNN()
    trainer_config = TrainerConfig(
        n_epochs=50,
        warmup_epochs=5,
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
    # Step 6: SBAS validation against a REAL (fixed-threshold) ADI baseline
    # -------------------------------------------------------------------
    logger.info("Step 7: Running SBAS validation vs. a genuine ADI-threshold baseline...")
    adi_baseline_mask = build_adi_baseline_mask(features, adi_threshold=0.25)
    gnn_mask = results["ps_mask"].ravel()
    logger.info(
        "   -> Fixed ADI<0.25 baseline selects %d pixels (vs. PS-GNN's %d detections).",
        int(adi_baseline_mask.sum()),
        n_detected,
    )

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
    # Step 7b: The actual point of this extension -- recall on the
    # deliberately hard, borderline-ADI points, GNN vs. classical baseline.
    # -------------------------------------------------------------------
    logger.info("Step 7b: Comparing recall on EASY vs. BORDERLINE-ADI PS points...")
    role_names = {"easy_ps": ROLE_TRUE_PS, "borderline_ps": ROLE_BORDERLINE_PS}
    gnn_recall = compute_recall_by_role(gnn_mask, node_roles, role_names)
    baseline_recall = compute_recall_by_role(adi_baseline_mask, node_roles, role_names)

    logger.info("   -> %-14s %12s %12s", "", "PS-GNN", "ADI<0.25")
    for name in role_names:
        logger.info(
            "   -> %-14s %11.1f%% %11.1f%%",
            name,
            gnn_recall[name] * 100,
            baseline_recall[name] * 100,
        )

    borderline_gain = gnn_recall["borderline_ps"] - baseline_recall["borderline_ps"]
    if borderline_gain > 0.05:
        logger.info(
            "   PASS: PS-GNN recovers %.1f%% more borderline-ADI points than the fixed ADI "
            "threshold (%.1f%% vs %.1f%%) -- graph corroboration is providing genuine value "
            "the classical method cannot access.",
            borderline_gain * 100,
            gnn_recall["borderline_ps"] * 100,
            baseline_recall["borderline_ps"] * 100,
        )
    else:
        logger.warning(
            "   NOT CONFIRMED: PS-GNN recall on borderline-ADI points (%.1f%%) is not "
            "meaningfully higher than the fixed ADI threshold's (%.1f%%) -- the model may "
            "not yet be leveraging cluster corroboration (try more epochs, larger clusters, "
            "or a stronger spatial-coherence loss weight).",
            gnn_recall["borderline_ps"] * 100,
            baseline_recall["borderline_ps"] * 100,
        )

    # Diagnose *why* recall for either role is below 100%: genuinely low
    # confidence, or borderline confidence pushed under threshold by
    # MC-Dropout variance?
    for role_name, role_code in role_names.items():
        if gnn_recall[role_name] >= 0.999:
            continue  # nothing to diagnose, recall is already ~100%
        diag = diagnose_confidence_by_hit_miss(
            results["probability"], results["uncertainty"], node_roles, role_code, 0.85
        )
        logger.info(
            "   -> %s confidence diagnostic: %d hits (mean p=%.3f, mean MC-var=%.5f) vs "
            "%d misses (mean p=%.3f, median p=%.3f, mean MC-var=%.5f)",
            role_name,
            diag["hit"]["count"],
            diag["hit"]["mean_probability"],
            diag["hit"]["mean_uncertainty"],
            diag["miss"]["count"],
            diag["miss"]["mean_probability"],
            diag["miss"]["median_probability"],
            diag["miss"]["mean_uncertainty"],
        )

        # Test the "nearby contaminant inflates local_var/edge_strength"
        # hypothesis directly, properly excluding each point's OWN by-design
        # neighbors (its own false neighbors, or its own cluster-mates) so
        # only genuine, incidental proximity to an UNRELATED point counts.
        spatial_diag = diagnose_spatial_contamination(
            node_roles,
            group_id,
            results["probability"],
            role_code,
            contaminant_roles=[ROLE_FALSE_NEIGHBOR, ROLE_BORDERLINE_PS],
            threshold=0.85,
            pixel_spacing_m=graph_config.pixel_spacing_m,
        )
        logger.info(
            "   -> %s spatial-contamination diagnostic (own-group excluded): hits are %.2fpx "
            "(median %.2fpx) from nearest UNRELATED false_neighbor/borderline pixel on average "
            "(%.0f%% within 5x5 local_var window, %.0f%% within 3x3 Sobel, %.0f%% exhausted); "
            "misses are %.2fpx (median %.2fpx) away (%.0f%% within 5x5, %.0f%% within 3x3, "
            "%.0f%% exhausted).",
            role_name,
            spatial_diag["hit"]["mean_distance_px"],
            spatial_diag["hit"]["median_distance_px"],
            spatial_diag["hit"]["frac_within_2px"] * 100,
            spatial_diag["hit"]["frac_within_1px"] * 100,
            spatial_diag["hit"]["frac_exhausted"] * 100,
            spatial_diag["miss"]["mean_distance_px"],
            spatial_diag["miss"]["median_distance_px"],
            spatial_diag["miss"]["frac_within_2px"] * 100,
            spatial_diag["miss"]["frac_within_1px"] * 100,
            spatial_diag["miss"]["frac_exhausted"] * 100,
        )
        if np.isnan(spatial_diag["hit"]["mean_distance_px"]) or np.isnan(
            spatial_diag["miss"]["mean_distance_px"]
        ):
            logger.warning(
                "   INCONCLUSIVE: %s hit or miss group had no valid (non-own-group) contaminant "
                "within k_query candidates -- increase k_query to draw a conclusion.",
                role_name,
            )
        elif spatial_diag["miss"]["mean_distance_px"] < spatial_diag["hit"]["mean_distance_px"]:
            logger.info(
                "   PASS: %s misses sit closer to UNRELATED contaminating pixels than hits do -- "
                "consistent with the local-feature-contamination hypothesis.",
                role_name,
            )
        else:
            logger.warning(
                "   NOT CONFIRMED: %s misses are NOT closer to unrelated contaminating pixels "
                "than hits -- the local-feature-contamination hypothesis is not supported by "
                "this data; the confidence drop likely has a different cause.",
                role_name,
            )

    # -------------------------------------------------------------------
    # Step 8: Visualizations & comprehensive HTML report
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
    logger.info("real classical-baseline comparison, and a borderline-ADI recall test -- see")
    logger.info("Steps 3, 5, 7, and 7b above.")


if __name__ == "__main__":
    main()
