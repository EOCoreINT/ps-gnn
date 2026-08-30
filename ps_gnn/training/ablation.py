"""
ps_gnn.training.ablation
===========================

Systematic ablation runner for PS-GNN. Sweeps over three independent
axes and reports validation metrics for every combination:

1. **Loss components** — ``--no_phase_loss`` / ``--no_spatial_loss``
   disable the corresponding term of
   :class:`ps_gnn.models.losses.PhysicsInformedPSLoss`, quantifying how
   much each physics-informed regularizer actually contributes.
2. **Backbone** — ``--model gnn`` trains :class:`PSGNN` on the graph via
   :class:`ps_gnn.training.trainer.Trainer`; ``--model vit`` instead
   tiles the raw amplitude/phase stack and trains :class:`PSViT` as a
   per-tile classifier (see :func:`_run_vit_ablation`), giving an
   apples-to-oranges-but-informative comparison between graph-structured
   and dense-attention approaches on the same scene.
3. **Graph construction thresholds** — ``--edge_dist`` / ``--edge_corr``
   rebuild the graph's edges (via
   :func:`ps_gnn.data.preprocessing.build_graph`) with different spatial
   and phase-correlation cutoffs before training, quantifying the
   sensitivity of PS-GNN to graph density/sparsity.

Usage
-----
.. code-block:: bash

    python -m ps_gnn.training.ablation \\
        --data data/mexico_city/graph.pt \\
        --output results/ablation_summary.csv \\
        --no_phase_loss --no_spatial_loss \\
        --model gnn vit \\
        --edge_dist 50 100 --edge_corr 0.5 0.7
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ps_gnn.data.preprocessing import GraphConstructionConfig, build_graph
from ps_gnn.models.losses import PhysicsInformedPSLossConfig
from ps_gnn.models.ps_gnn import PSGNN
from ps_gnn.models.ps_vit import PSViT
from ps_gnn.training.trainer import Trainer, TrainerConfig, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class AblationRun:
    """A single ablation configuration.

    Attributes
    ----------
    model : {"gnn", "vit"}
        Backbone to train.
    use_phase_loss : bool
        Whether the phase-stability loss term is enabled (GNN only; the
        ViT path always uses plain weighted cross-entropy since it has no
        graph structure to compute physics terms over).
    use_spatial_loss : bool
        Whether the spatial-coherence loss term is enabled (GNN only).
    edge_dist_m : float
        Max spatial edge distance, meters (GNN only).
    edge_corr : float
        Min phase-correlation edge threshold (GNN only).
    """

    model: str
    use_phase_loss: bool
    use_spatial_loss: bool
    edge_dist_m: float
    edge_corr: float

    def run_id(self) -> str:
        return (
            f"model={self.model}"
            f"_phase={int(self.use_phase_loss)}"
            f"_spatial={int(self.use_spatial_loss)}"
            f"_dist={self.edge_dist_m}"
            f"_corr={self.edge_corr}"
        )


def build_ablation_grid(
    models: list[str],
    phase_loss_options: list[bool],
    spatial_loss_options: list[bool],
    edge_dist_options: list[float],
    edge_corr_options: list[float],
) -> list[AblationRun]:
    """Build the Cartesian product of ablation axes.

    Loss-component axes are only meaningful for ``model="gnn"``; for
    ``model="vit"`` runs, the loss options are collapsed to a single
    entry each (ViT has no graph to compute physics terms over), avoiding
    duplicate, meaningless ViT runs.

    Parameters
    ----------
    models : list[str]
        Subset of ``{"gnn", "vit"}`` to run.
    phase_loss_options, spatial_loss_options : list[bool]
        Which settings of each loss toggle to sweep (GNN only).
    edge_dist_options, edge_corr_options : list[float]
        Which graph-construction thresholds to sweep (GNN only).

    Returns
    -------
    list[AblationRun]
    """
    runs: list[AblationRun] = []
    for model in models:
        if model == "vit":
            runs.append(
                AblationRun(
                    model="vit",
                    use_phase_loss=True,
                    use_spatial_loss=True,
                    edge_dist_m=edge_dist_options[0],
                    edge_corr=edge_corr_options[0],
                )
            )
            continue
        for phase_opt, spatial_opt, dist_opt, corr_opt in itertools.product(
            phase_loss_options, spatial_loss_options, edge_dist_options, edge_corr_options
        ):
            runs.append(
                AblationRun(
                    model="gnn",
                    use_phase_loss=phase_opt,
                    use_spatial_loss=spatial_opt,
                    edge_dist_m=dist_opt,
                    edge_corr=corr_opt,
                )
            )
    return runs


def _run_gnn_ablation(
    run: AblationRun,
    data,
    n_epochs: int,
) -> dict[str, Any]:
    """Rebuild the graph with ``run``'s thresholds and train/evaluate PS-GNN.

    Parameters
    ----------
    run : AblationRun
    data : torch_geometric.data.Data
        Base graph; must have ``pos`` and a ``phase`` attribute attached
        (see :meth:`ps_gnn.training.trainer.Trainer.fit`). Edges are
        rebuilt from ``pos``/``phase`` rather than reusing ``data.edge_index``,
        so ``--edge_dist``/``--edge_corr`` actually take effect.
    n_epochs : int
        Number of training epochs for this ablation run (typically much
        smaller than a full production run, to keep the sweep tractable).

    Returns
    -------
    dict[str, Any]
        Final-epoch validation metrics plus the run's configuration.
    """
    graph_cfg = GraphConstructionConfig(
        max_distance_m=run.edge_dist_m, min_phase_correlation=run.edge_corr
    )
    edge_index, edge_attr = build_graph(
        data.pos.numpy(), data.phase.numpy(), graph_cfg, show_progress=False
    )

    rebuilt = data.clone()
    rebuilt.edge_index = torch.from_numpy(edge_index)
    rebuilt.edge_attr = torch.from_numpy(edge_attr)
    rebuilt.phase = data.phase

    loss_cfg = PhysicsInformedPSLossConfig(
        use_phase_loss=run.use_phase_loss, use_spatial_loss=run.use_spatial_loss
    )
    trainer_cfg = TrainerConfig(
        n_epochs=n_epochs, warmup_epochs=min(2, n_epochs), loss_config=loss_cfg, log_dir=None
    )
    model = PSGNN()
    trainer = Trainer(model, trainer_cfg)
    history = trainer.fit(rebuilt)
    final = history[-1]

    return {
        "run_id": run.run_id(),
        "model": run.model,
        "use_phase_loss": run.use_phase_loss,
        "use_spatial_loss": run.use_spatial_loss,
        "edge_dist_m": run.edge_dist_m,
        "edge_corr": run.edge_corr,
        "n_edges": edge_index.shape[1],
        "val_f1": final["val_f1"],
        "val_precision": final["val_precision"],
        "val_recall": final["val_recall"],
        "val_roc_auc": final["val_roc_auc"],
        "train_loss": final["train_loss"],
    }


def _tile_grid_labels(
    amplitude: np.ndarray,
    phase: np.ndarray,
    labels: np.ndarray,
    tile_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a scene into fixed-size windows for PS-ViT training.

    Each tile is assigned a binary label by majority vote over its
    labeled pixels (ties/all-unlabeled tiles are dropped).

    Parameters
    ----------
    amplitude, phase : np.ndarray, shape (T, H, W)
    labels : np.ndarray, shape (H, W)
        Pixel labels, with ``-1`` for unlabeled.
    tile_size : int, default 32
        Tile side length in pixels; the scene is cropped (not padded) to
        the nearest multiple of ``tile_size``.

    Returns
    -------
    tiles : np.ndarray, shape (n_tiles, T, 2, tile_size, tile_size)
        Stacked amplitude/phase tiles.
    tile_labels : np.ndarray, shape (n_tiles,)
        Majority-vote binary label per tile.
    """
    _t_steps, height, width = amplitude.shape
    n_rows = height // tile_size
    n_cols = width // tile_size
    tiles, tile_labels = [], []

    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = r * tile_size, (r + 1) * tile_size
            c0, c1 = c * tile_size, (c + 1) * tile_size
            label_patch = labels[r0:r1, c0:c1]
            labeled = label_patch[label_patch != -1]
            if labeled.size == 0:
                continue
            tile_label = int(np.round(labeled.mean()))

            amp_tile = amplitude[:, r0:r1, c0:c1]
            phase_tile = phase[:, r0:r1, c0:c1]
            tiles.append(np.stack([amp_tile, phase_tile], axis=1))  # (T, 2, ts, ts)
            tile_labels.append(tile_label)

    if not tiles:
        raise ValueError(
            "No labeled tiles found; check that `labels` has non-(-1) entries "
            "within at least one full tile."
        )
    return np.stack(tiles, axis=0), np.array(tile_labels, dtype=np.int64)


def _run_vit_ablation(
    amplitude: np.ndarray,
    phase: np.ndarray,
    labels: np.ndarray,
    n_epochs: int,
    tile_size: int = 32,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train/evaluate PS-ViT as a per-tile binary classifier.

    A simple random tile-level train/val split is used (tiles, unlike
    individual pixels, are large enough that adjacent-tile leakage is a
    much smaller concern than adjacent-*pixel* leakage in the GNN case).

    Parameters
    ----------
    amplitude, phase : np.ndarray, shape (T, H, W)
    labels : np.ndarray, shape (H, W)
    n_epochs : int
    tile_size : int, default 32
    val_fraction : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    dict[str, Any]
        Final validation metrics plus run metadata.
    """
    tiles, tile_labels = _tile_grid_labels(amplitude, phase, labels, tile_size)
    n_tiles = tiles.shape[0]

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_tiles)
    n_val = max(1, round(val_fraction * n_tiles))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = PSViT()
    class_counts = np.bincount(tile_labels[train_idx], minlength=2).clip(min=1)
    class_weights = torch.tensor(
        class_counts.sum() / (2 * class_counts), dtype=torch.float32, device=model.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    x_train = torch.from_numpy(tiles[train_idx]).float()
    y_train = torch.from_numpy(tile_labels[train_idx]).long()
    x_val = torch.from_numpy(tiles[val_idx]).float()
    y_val = torch.from_numpy(tile_labels[val_idx]).long()

    for _epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = torch.nn.functional.cross_entropy(
            logits, y_train.to(model.device), weight=class_weights
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()

    metrics = compute_metrics(val_probs, y_val.numpy(), float(loss.item()))

    return {
        "run_id": AblationRun("vit", True, True, 0.0, 0.0).run_id(),
        "model": "vit",
        "use_phase_loss": None,
        "use_spatial_loss": None,
        "edge_dist_m": None,
        "edge_corr": None,
        "n_edges": None,
        "val_f1": metrics.f1,
        "val_precision": metrics.precision,
        "val_recall": metrics.recall,
        "val_roc_auc": metrics.roc_auc,
        "train_loss": metrics.loss,
    }


def run_ablation_study(
    data,
    output_path: str | Path,
    grid: list[AblationRun],
    n_epochs: int = 20,
    amplitude: np.ndarray | None = None,
    phase_raster: np.ndarray | None = None,
    labels_raster: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run every configuration in ``grid`` and save results to CSV.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph used for all ``model="gnn"`` runs (must have ``pos`` and a
        ``phase`` attribute; see :meth:`Trainer.fit`).
    output_path : str or pathlib.Path
        Where to write ``ablation_summary.csv``.
    grid : list[AblationRun]
        Configurations to run, typically from :func:`build_ablation_grid`.
    n_epochs : int, default 20
        Training epochs per run.
    amplitude, phase_raster, labels_raster : np.ndarray, optional
        Raw ``(T, H, W)`` stacks and ``(H, W)`` label grid, required only
        if ``grid`` contains any ``model="vit"`` runs.

    Returns
    -------
    pd.DataFrame
        One row per completed run; also written to ``output_path``.

    Raises
    ------
    ValueError
        If a ViT run is requested but the raw raster arguments are
        missing.
    """
    results: list[dict[str, Any]] = []

    for i, run in enumerate(grid):
        logger.info("Running ablation %d/%d: %s", i + 1, len(grid), run.run_id())
        try:
            if run.model == "gnn":
                result = _run_gnn_ablation(run, data, n_epochs)
            elif run.model == "vit":
                if amplitude is None or phase_raster is None or labels_raster is None:
                    raise ValueError(
                        "ViT ablation runs require `amplitude`, `phase_raster`, "
                        "and `labels_raster` to be provided to run_ablation_study()."
                    )
                result = _run_vit_ablation(amplitude, phase_raster, labels_raster, n_epochs)
            else:
                raise ValueError(f"Unknown model '{run.model}'")
        except Exception:
            logger.exception("Ablation run %s failed; recording NaNs.", run.run_id())
            result = {
                "run_id": run.run_id(),
                "model": run.model,
                "use_phase_loss": run.use_phase_loss,
                "use_spatial_loss": run.use_spatial_loss,
                "edge_dist_m": run.edge_dist_m,
                "edge_corr": run.edge_corr,
                "n_edges": np.nan,
                "val_f1": np.nan,
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_roc_auc": np.nan,
                "train_loss": np.nan,
            }
        results.append(result)

    df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved ablation summary (%d runs) to %s", len(df), output_path)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PS-GNN ablation studies across loss components, "
        "backbone architecture, and graph construction thresholds."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to a saved PyG Data object (.pt)."
    )
    parser.add_argument(
        "--output", type=str, default="results/ablation_summary.csv", help="Output CSV path."
    )
    parser.add_argument("--model", nargs="+", default=["gnn"], choices=["gnn", "vit"])
    parser.add_argument(
        "--no_phase_loss",
        action="store_true",
        help="Include a run with the phase-stability loss term disabled.",
    )
    parser.add_argument(
        "--no_spatial_loss",
        action="store_true",
        help="Include a run with the spatial-coherence loss term disabled.",
    )
    parser.add_argument("--edge_dist", nargs="+", type=float, default=[100.0])
    parser.add_argument("--edge_corr", nargs="+", type=float, default=[0.7])
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument(
        "--amplitude", type=str, default=None, help="Path to raw amplitude .npy (for --model vit)."
    )
    parser.add_argument(
        "--phase_raster", type=str, default=None, help="Path to raw phase .npy (for --model vit)."
    )
    parser.add_argument(
        "--labels_raster", type=str, default=None, help="Path to label grid .npy (for --model vit)."
    )
    return parser


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = torch.load(args.data, weights_only=False)

    phase_loss_options = [True, False] if args.no_phase_loss else [True]
    spatial_loss_options = [True, False] if args.no_spatial_loss else [True]

    grid = build_ablation_grid(
        models=args.model,
        phase_loss_options=phase_loss_options,
        spatial_loss_options=spatial_loss_options,
        edge_dist_options=args.edge_dist,
        edge_corr_options=args.edge_corr,
    )

    amplitude = np.load(args.amplitude) if args.amplitude else None
    phase_raster = np.load(args.phase_raster) if args.phase_raster else None
    labels_raster = np.load(args.labels_raster) if args.labels_raster else None

    run_ablation_study(
        data,
        args.output,
        grid,
        n_epochs=args.n_epochs,
        amplitude=amplitude,
        phase_raster=phase_raster,
        labels_raster=labels_raster,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
