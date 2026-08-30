"""
tests.test_pipeline
======================

End-to-end integration test exercising the full ``ps-gnn`` pipeline on
tiny synthetic data:

    Synthetic Data -> Label Generation -> Preprocessing -> Training (1 epoch)
    -> Inference (tiled, with MC Dropout) -> Report Generation

Marked ``@pytest.mark.slow`` since it touches every layer of the stack
(deep-learning, tiling, reporting) and is meaningfully heavier than the
focused unit tests elsewhere in the suite. Run the full suite including
slow tests with ``pytest``; skip them with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ps_gnn.data.label_generation import inject_synthetic_ps
from ps_gnn.data.preprocessing import GraphConstructionConfig, build_graph, compute_node_features
from ps_gnn.inference.detector import PSDetector, TilingConfig
from ps_gnn.models.ps_gnn import PSGNN
from ps_gnn.training.trainer import Trainer, TrainerConfig


@pytest.mark.slow
class TestFullPipeline:
    """The complete synthetic-data -> report-generation pipeline."""

    def test_synthetic_data_to_report(self, tmp_path):
        rng = np.random.default_rng(2024)
        t_steps, height, width = 8, 32, 32

        # --- 1. Synthetic data ---
        amplitude = rng.gamma(shape=4.0, scale=25.0, size=(t_steps, height, width)).astype(
            np.float32
        )
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, height, width)).astype(np.float32)

        # --- 2. Label generation ---
        synthetic = inject_synthetic_ps(
            amplitude, phase, n_ps=25, min_separation_px=2, random_state=1
        )
        assert synthetic.labels.sum() == 25
        amplitude, phase, labels = synthetic.amplitude, synthetic.phase, synthetic.labels

        # add a modest number of hard negatives so the classifier has signal
        # in both directions (inject_synthetic_ps only labels positives)
        neg_candidates = np.argwhere(labels == 0)
        neg_idx = rng.choice(len(neg_candidates), size=40, replace=False)
        neg_rows, neg_cols = neg_candidates[neg_idx, 0], neg_candidates[neg_idx, 1]

        y_flat = np.full(height * width, -1, dtype=np.int64)
        pos_rows, pos_cols = np.nonzero(labels)
        y_flat[pos_rows * width + pos_cols] = 1
        y_flat[neg_rows * width + neg_cols] = 0

        # --- 3. Preprocessing ---
        worldcover = np.full((height, width), 50, dtype=np.int32)
        features = compute_node_features(amplitude, phase, 38.0, worldcover)
        x = torch.from_numpy(features.reshape(19, -1).T.astype(np.float32))

        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        graph_config = GraphConstructionConfig(
            max_distance_m=20.0, min_phase_correlation=-1.0, pixel_spacing_m=5.0, max_neighbors=8
        )
        node_coords = np.stack(
            [
                cols.ravel() * graph_config.pixel_spacing_m,
                rows.ravel() * graph_config.pixel_spacing_m,
            ],
            axis=1,
        ).astype(np.float32)
        phase_flat = phase.reshape(t_steps, -1)
        edge_index_np, edge_attr_np = build_graph(
            node_coords, phase_flat, graph_config, show_progress=False
        )

        assert x.shape == (height * width, 19)
        assert edge_index_np.shape[0] == 2

        from torch_geometric.data import Data

        data = Data(
            x=x,
            edge_index=torch.from_numpy(edge_index_np),
            edge_attr=torch.from_numpy(edge_attr_np),
            y=torch.from_numpy(y_flat),
            pos=torch.from_numpy(node_coords),
        )
        data.phase = torch.from_numpy(phase_flat)

        # --- 4. Training (1 epoch) ---
        model = PSGNN()
        trainer_config = TrainerConfig(
            n_epochs=1, warmup_epochs=1, n_spatial_blocks=3, log_dir=None
        )
        trainer = Trainer(model, trainer_config)
        history = trainer.fit(data)

        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"])

        # --- 5. Inference (tiled, with MC Dropout) ---
        detector = PSDetector(
            model,
            tiling_config=TilingConfig(tile_size=20, overlap=6),
            graph_config=graph_config,
            mc_dropout_passes=3,
            confidence_threshold=0.5,
        )
        results = detector.detect(amplitude, phase, incidence_angle=38.0, worldcover=worldcover)

        for key in ("ps_mask", "probability", "uncertainty", "cluster_labels"):
            assert key in results
            assert results[key].shape == (height, width)
        assert results["probability"].min() >= 0.0
        assert results["probability"].max() <= 1.0

        # --- 6. Report generation ---
        report_dir = tmp_path / "report"

        from ps_gnn.analytics.report_generator import generate_html_report

        report_path = generate_html_report(results, output_dir=report_dir)

        assert report_path.exists()
        content = report_path.read_text()
        assert "__" not in content  # no unsubstituted template tokens
        assert len(content) > 1000

    def test_tiled_inference_no_memory_leak(self):
        """Repeated tiled inference should not grow Python-level memory unboundedly.

        Runs :meth:`PSDetector.detect` several times over a small scene and
        compares ``tracemalloc`` snapshots before/after, allowing normal
        allocator/interpreter overhead but failing on genuinely runaway
        growth (e.g. an accidentally-retained reference cycle across tiles).
        """
        rng = np.random.default_rng(0)
        t_steps, height, width = 6, 24, 24
        amplitude = rng.gamma(4.0, 25.0, size=(t_steps, height, width)).astype(np.float32)
        phase = rng.uniform(-np.pi, np.pi, size=(t_steps, height, width)).astype(np.float32)

        model = PSGNN()
        detector = PSDetector(
            model,
            tiling_config=TilingConfig(tile_size=12, overlap=4),
            graph_config=GraphConstructionConfig(
                max_distance_m=15.0, min_phase_correlation=-1.0, max_neighbors=6
            ),
            mc_dropout_passes=2,
            confidence_threshold=0.9,
        )

        # Warm-up run (first run often allocates persistent caches; exclude
        # it from the leak comparison).
        detector.detect(amplitude, phase, incidence_angle=38.0)
        gc.collect()

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(3):
            detector.detect(amplitude, phase, incidence_angle=38.0)
            gc.collect()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot_after.compare_to(snapshot_before, "lineno")
        total_growth_mb = sum(stat.size_diff for stat in diff) / (1024 * 1024)

        # Allow some growth (small caches, fragmentation) but fail on
        # clearly unbounded accumulation across repeated runs.
        assert total_growth_mb < 50, (
            f"Memory grew by {total_growth_mb:.1f} MB across 3 repeated tiled-inference "
            "runs; possible leak. Top allocations:\n" + "\n".join(str(s) for s in diff[:5])
        )
