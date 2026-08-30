"""
tests.test_preprocessing
===========================

Unit tests for :mod:`ps_gnn.data.preprocessing`: node feature engineering
on a tiny synthetic stack, and graph edge-construction logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from ps_gnn.data.preprocessing import (
    N_NODE_FEATURES,
    GraphConstructionConfig,
    build_graph,
    compute_node_features,
)


class TestComputeNodeFeatures:
    """Tests for :func:`compute_node_features` on a tiny (T=6, H=10, W=10) stack."""

    def test_output_shape(self, tiny_stack):
        features = compute_node_features(
            tiny_stack["amplitude"],
            tiny_stack["phase"],
            tiny_stack["incidence_angle"],
            tiny_stack["worldcover"],
        )
        assert features.shape == (19, 10, 10)
        assert features.dtype == np.float32

    def test_feature_count_matches_schema(self):
        assert N_NODE_FEATURES == 19

    def test_adi_calculation_matches_manual_formula(self, tiny_stack):
        """ADI (index 2) must equal std(amplitude) / mean(amplitude) per-pixel."""
        features = compute_node_features(
            tiny_stack["amplitude"],
            tiny_stack["phase"],
            tiny_stack["incidence_angle"],
            tiny_stack["worldcover"],
        )
        amp = tiny_stack["amplitude"]
        manual_adi = np.std(amp, axis=0) / (np.mean(amp, axis=0) + 1e-8)
        np.testing.assert_allclose(features[2], manual_adi, atol=1e-4)

    def test_adi_zero_for_constant_amplitude(self):
        """A pixel with perfectly stable amplitude should have ADI == 0."""
        t_steps, height, width = 5, 4, 4
        amplitude = np.full((t_steps, height, width), 50.0, dtype=np.float32)
        phase = np.zeros((t_steps, height, width), dtype=np.float32)
        worldcover = np.full((height, width), 50, dtype=np.int32)

        features = compute_node_features(amplitude, phase, 38.0, worldcover)
        np.testing.assert_allclose(features[2], 0.0, atol=1e-6)

    def test_temporal_coherence_is_one_for_constant_phase(self):
        """Perfectly stable phase should give temporal coherence == 1."""
        t_steps, height, width = 5, 4, 4
        amplitude = np.ones((t_steps, height, width), dtype=np.float32) * 10
        phase = np.zeros((t_steps, height, width), dtype=np.float32)
        worldcover = np.full((height, width), 50, dtype=np.int32)

        features = compute_node_features(amplitude, phase, 38.0, worldcover)
        np.testing.assert_allclose(features[5], 1.0, atol=1e-6)

    def test_worldcover_one_hot_sums_to_one(self, tiny_stack):
        features = compute_node_features(
            tiny_stack["amplitude"],
            tiny_stack["phase"],
            tiny_stack["incidence_angle"],
            tiny_stack["worldcover"],
        )
        worldcover_channels = features[9:19]
        np.testing.assert_allclose(worldcover_channels.sum(axis=0), 1.0, atol=1e-6)

    def test_scalar_vs_grid_incidence_angle(self, tiny_stack):
        """A scalar incidence angle should broadcast identically to a constant grid."""
        scalar_features = compute_node_features(
            tiny_stack["amplitude"], tiny_stack["phase"], 40.0, tiny_stack["worldcover"]
        )
        grid = np.full((10, 10), 40.0, dtype=np.float32)
        grid_features = compute_node_features(
            tiny_stack["amplitude"], tiny_stack["phase"], grid, tiny_stack["worldcover"]
        )
        np.testing.assert_allclose(scalar_features, grid_features, atol=1e-6)

    def test_raises_on_shape_mismatch(self, tiny_stack):
        with pytest.raises(ValueError):
            compute_node_features(
                tiny_stack["amplitude"],
                tiny_stack["phase"][:, :5, :5],  # mismatched shape
                tiny_stack["incidence_angle"],
                tiny_stack["worldcover"],
            )


class TestBuildGraph:
    """Tests for :func:`build_graph` edge-construction logic."""

    def test_edges_respect_distance_threshold(self):
        rng = np.random.default_rng(0)
        n_nodes, t_steps = 40, 8
        coords = rng.uniform(0, 100, size=(n_nodes, 2)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, n_nodes)).astype(np.float32)

        config = GraphConstructionConfig(max_distance_m=20.0, min_phase_correlation=-1.0)
        edge_index, _edge_attr = build_graph(coords, phase, config, show_progress=False)

        if edge_index.shape[1] > 0:
            src, dst = edge_index
            actual_dist = np.linalg.norm(coords[src] - coords[dst], axis=1)
            assert (actual_dist <= config.max_distance_m + 1e-4).all()

    def test_edges_respect_correlation_threshold(self):
        rng = np.random.default_rng(1)
        n_nodes, t_steps = 30, 10
        coords = rng.uniform(0, 50, size=(n_nodes, 2)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, n_nodes)).astype(np.float32)

        config = GraphConstructionConfig(max_distance_m=1000.0, min_phase_correlation=0.9)
        _edge_index, edge_attr = build_graph(coords, phase, config, show_progress=False)

        if edge_attr.shape[0] > 0:
            assert (edge_attr[:, 1] > 0.9).all()

    def test_graph_is_undirected_symmetric(self):
        """Every edge (i, j) should have a matching reverse edge (j, i)."""
        rng = np.random.default_rng(2)
        n_nodes, t_steps = 20, 8
        coords = rng.uniform(0, 30, size=(n_nodes, 2)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, n_nodes)).astype(np.float32)

        config = GraphConstructionConfig(max_distance_m=100.0, min_phase_correlation=-1.0)
        edge_index, _ = build_graph(coords, phase, config, show_progress=False)

        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for i, j in edge_set:
            assert (j, i) in edge_set, f"edge ({i},{j}) has no reverse edge"

    def test_no_self_loops(self):
        rng = np.random.default_rng(3)
        n_nodes, t_steps = 20, 8
        coords = rng.uniform(0, 30, size=(n_nodes, 2)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, n_nodes)).astype(np.float32)

        config = GraphConstructionConfig(max_distance_m=100.0, min_phase_correlation=-1.0)
        edge_index, _ = build_graph(coords, phase, config, show_progress=False)

        assert (edge_index[0] != edge_index[1]).all()

    def test_empty_graph_when_thresholds_impossible(self):
        rng = np.random.default_rng(4)
        n_nodes, t_steps = 15, 8
        coords = rng.uniform(0, 30, size=(n_nodes, 2)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, n_nodes)).astype(np.float32)

        config = GraphConstructionConfig(
            max_distance_m=1.0, min_phase_correlation=1.1
        )  # impossible
        edge_index, edge_attr = build_graph(coords, phase, config, show_progress=False)

        assert edge_index.shape == (2, 0)
        assert edge_attr.shape == (0, 2)

    def test_raises_on_node_count_mismatch(self):
        coords = np.random.rand(10, 2).astype(np.float32)
        phase = np.random.rand(5, 8).astype(np.float32)  # 8 != 10
        with pytest.raises(ValueError):
            build_graph(coords, phase, GraphConstructionConfig(), show_progress=False)


class TestBuildPygDataset:
    """Tests for :func:`build_pyg_dataset`, exercising the real GeoTIFF I/O path."""

    def test_builds_valid_graph_from_geotiffs(self, dummy_geotiffs):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from ps_gnn.data.preprocessing import build_pyg_dataset

        height, width = dummy_geotiffs["amplitude"].shape[1:]
        labels = np.full((height, width), -1, dtype=np.int64)
        labels[2:4, 2:4] = 1  # a couple of dummy positives
        labels[0:2, 0:2] = 0  # a couple of dummy negatives

        data = build_pyg_dataset(
            dummy_geotiffs["amplitude_path"],
            dummy_geotiffs["phase_path"],
            dummy_geotiffs["dem_path"],
            labels,
            graph_config=GraphConstructionConfig(
                max_distance_m=15.0, min_phase_correlation=-1.0, max_neighbors=6
            ),
        )

        assert data.x.shape == (height * width, N_NODE_FEATURES)
        assert data.edge_index.shape[0] == 2
        assert data.y.shape == (height * width,)
        assert data.pos.shape == (height * width, 2)
