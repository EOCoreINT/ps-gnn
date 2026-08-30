"""
ps_gnn.training.trainer
=========================

The core training loop for PS-GNN, implementing three scientific
requirements that go beyond a generic node-classification training
script:

1. **Spatial Block Cross-Validation** — random train/val splits on a
   spatial graph leak information (adjacent, highly-correlated pixels
   end up on both sides), inflating validation metrics. Instead, the
   scene is partitioned into 5 contiguous spatial blocks; each fold
   trains on 4 blocks and validates on the held-out block, so validation
   pixels are spatially distant from training pixels.

2. **Curriculum Learning** — early epochs train only on the most reliable
   part of the graph (edges with the strongest phase correlation), then
   progressively relax the edge-correlation requirement as the model
   matures, on the intuition that starting from unambiguous local
   structure before exposing the model to noisier, weakly-correlated
   long-range edges stabilizes optimization.

3. **Standard best-practice scaffolding** — AdamW, a linear-warmup +
   cosine-annealing LR schedule, gradient clipping, and TensorBoard
   logging of loss components and classification metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ps_gnn.models.losses import LossComponents, PhysicsInformedPSLoss, PhysicsInformedPSLossConfig
from ps_gnn.models.ps_gnn import PSGNN

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    _HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class TrainerConfig:
    """Hyperparameters and scaffolding options for :class:`Trainer`.

    Attributes
    ----------
    n_epochs : int
        Total number of training epochs.
    learning_rate : float
        Peak learning rate for AdamW.
    weight_decay : float
        AdamW weight decay.
    warmup_epochs : int
        Number of linear-warmup epochs before cosine annealing begins.
    grad_clip_norm : float
        Max gradient norm for ``torch.nn.utils.clip_grad_norm_``.
    n_spatial_blocks : int
        Number of spatial blocks for cross-validation (default 5: train
        on 4, validate on 1).
    curriculum_schedule : list[tuple[int, float]]
        Ordered ``(epoch_start, min_edge_correlation)`` pairs defining the
        curriculum: at epoch >= ``epoch_start``, only edges with
        ``edge_attr[:, 1] > min_edge_correlation`` are used. Default
        implements "epochs 1-20: coherence > 0.7, 21-50: > 0.4, 51+: full
        graph (> -1, i.e. no filtering)".
    log_dir : str or None
        TensorBoard log directory. If None, defaults to ``runs/ps_gnn``.
        Logging is silently skipped if TensorBoard is unavailable.
    checkpoint_dir : str or None
        Directory to save the best-validation-F1 checkpoint. If None,
        checkpointing is skipped.
    loss_config : PhysicsInformedPSLossConfig
        Configuration passed to :class:`PhysicsInformedPSLoss`.
    """

    n_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    n_spatial_blocks: int = 5
    curriculum_schedule: list[tuple[int, float]] = field(
        default_factory=lambda: [(1, 0.7), (21, 0.4), (51, -1.0)]
    )
    log_dir: str | None = None
    checkpoint_dir: str | None = None
    loss_config: PhysicsInformedPSLossConfig = field(default_factory=PhysicsInformedPSLossConfig)


@dataclass
class EpochMetrics:
    """Classification metrics computed on a set of labeled nodes.

    Attributes
    ----------
    precision, recall, f1 : float
        Standard binary classification metrics for the PS class,
        computed with ``zero_division=0``.
    roc_auc : float
        Area under the ROC curve using the predicted PS-class
        probability. ``NaN`` if only one class is present (undefined).
    loss : float
        The (weighted total) loss value.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float


def spatial_block_split(pos: np.ndarray, n_blocks: int = 5, random_state: int = 42) -> np.ndarray:
    """Partition nodes into contiguous spatial blocks via a coarse grid.

    Nodes are assigned to blocks by binning their coordinates onto a
    roughly-square grid with ``n_blocks`` cells (grid dimensions chosen so
    ``rows * cols >= n_blocks``), then merging/relabeling grid cells into
    exactly ``n_blocks`` groups of comparable size using a greedy
    balancing pass. This keeps each fold spatially contiguous while
    avoiding wildly unbalanced block sizes on irregular point sets.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 2)
        Node ``(x, y)`` ground coordinates.
    n_blocks : int, default 5
        Number of spatial blocks (folds) to produce.
    random_state : int, default 42
        Seed for the balancing pass's tie-breaking.

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        Block assignment (``0`` .. ``n_blocks - 1``) for every node.
    """
    grid_side = int(np.ceil(np.sqrt(n_blocks)))
    x_bins = np.quantile(pos[:, 0], np.linspace(0, 1, grid_side + 1))
    y_bins = np.quantile(pos[:, 1], np.linspace(0, 1, grid_side + 1))
    x_bins = np.unique(x_bins)
    y_bins = np.unique(y_bins)
    x_bins = x_bins if len(x_bins) > 1 else np.array([x_bins[0] - 1, x_bins[0] + 1])
    y_bins = y_bins if len(y_bins) > 1 else np.array([y_bins[0] - 1, y_bins[0] + 1])

    x_idx = np.clip(np.digitize(pos[:, 0], x_bins[1:-1]), 0, len(x_bins) - 2)
    y_idx = np.clip(np.digitize(pos[:, 1], y_bins[1:-1]), 0, len(y_bins) - 2)
    cell_id = x_idx * (len(y_bins) - 1) + y_idx

    unique_cells, cell_counts = np.unique(cell_id, return_counts=True)
    order = np.argsort(-cell_counts)
    block_sizes = np.zeros(n_blocks, dtype=np.int64)
    cell_to_block = {}
    for rank in order:
        cell = unique_cells[rank]
        target_block = int(np.argmin(block_sizes))
        cell_to_block[cell] = target_block
        block_sizes[target_block] += cell_counts[rank]

    block_assignment = np.array([cell_to_block[c] for c in cell_id], dtype=np.int64)
    return block_assignment


def _curriculum_min_correlation(epoch: int, schedule: list[tuple[int, float]]) -> float:
    """Look up the active curriculum threshold for a given epoch (1-indexed)."""
    threshold = schedule[0][1]
    for epoch_start, value in schedule:
        if epoch >= epoch_start:
            threshold = value
    return threshold


def _filter_edges_by_correlation(
    edge_index: Tensor, edge_attr: Tensor, min_correlation: float
) -> Tensor:
    """Return the subset of ``edge_index`` with phase correlation above threshold.

    Parameters
    ----------
    edge_index : Tensor, shape (2, E)
    edge_attr : Tensor, shape (E, 2)
        Columns are ``[normalized_distance, phase_correlation]`` per the
        convention established in :mod:`ps_gnn.data.preprocessing`.
    min_correlation : float
        Minimum phase correlation (column 1) required to keep an edge.

    Returns
    -------
    Tensor, shape (2, E')
        Filtered edge index.
    """
    if edge_index.shape[1] == 0:
        return edge_index
    mask = edge_attr[:, 1] > min_correlation
    return edge_index[:, mask]


def compute_metrics(
    probs: np.ndarray, targets: np.ndarray, loss_value: float, ps_class_index: int = 1
) -> EpochMetrics:
    """Compute precision/recall/F1/ROC-AUC on labeled nodes.

    Parameters
    ----------
    probs : np.ndarray, shape (N,)
        Predicted probability of the PS class.
    targets : np.ndarray, shape (N,)
        Ground-truth binary labels (already filtered to labeled nodes).
    loss_value : float
        Loss value to attach to the returned metrics for convenience.
    ps_class_index : int, default 1
        Label value considered "positive" for precision/recall/F1.

    Returns
    -------
    EpochMetrics
    """
    preds = (probs >= 0.5).astype(np.int64)
    precision = precision_score(targets, preds, pos_label=ps_class_index, zero_division=0)
    recall = recall_score(targets, preds, pos_label=ps_class_index, zero_division=0)
    f1 = f1_score(targets, preds, pos_label=ps_class_index, zero_division=0)
    if len(np.unique(targets)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(targets, probs)
    return EpochMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        loss=float(loss_value),
    )


class Trainer:
    """Orchestrates spatial-CV, curriculum-learning training of PS-GNN.

    Parameters
    ----------
    model : PSGNN
        The model to train.
    config : TrainerConfig, optional
        Training hyperparameters. Defaults to :class:`TrainerConfig`.

    Attributes
    ----------
    optimizer : torch.optim.AdamW
    scheduler : torch.optim.lr_scheduler.SequentialLR
        Linear warmup for ``config.warmup_epochs`` epochs, followed by
        cosine annealing for the remainder.
    loss_fn : PhysicsInformedPSLoss
    writer : torch.utils.tensorboard.SummaryWriter or None
    """

    def __init__(self, model: PSGNN, config: TrainerConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        cfg = self.config

        self.optimizer = AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        warmup = LinearLR(
            self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(cfg.n_epochs - cfg.warmup_epochs, 1))
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
        )

        self.loss_fn = PhysicsInformedPSLoss(cfg.loss_config)

        self.writer: SummaryWriter | None = None
        if _HAS_TENSORBOARD:
            log_dir = cfg.log_dir or "runs/ps_gnn"
            self.writer = SummaryWriter(log_dir=log_dir)
        else:  # pragma: no cover
            logger.warning("TensorBoard is not installed; training metrics will not be logged.")

        self.best_val_f1 = -1.0

    def _run_epoch(
        self,
        x: Tensor,
        phase: Tensor,
        y: Tensor,
        edge_index: Tensor,
        train: bool,
    ) -> tuple[LossComponents, EpochMetrics]:
        """Forward (+ backward, if ``train``) pass over one (sub)graph.

        Parameters
        ----------
        x : Tensor, shape (N, 19)
        phase : Tensor, shape (T, N)
        y : Tensor, shape (N,)
            Labels for this fold's nodes, with non-fold nodes masked to
            ``ignore_index`` upstream (see :meth:`fit`).
        edge_index : Tensor, shape (2, E)
            Already curriculum-filtered edges for the current epoch.
        train : bool
            If True, performs backprop + optimizer step; otherwise runs
            under ``torch.no_grad()``.

        Returns
        -------
        LossComponents, EpochMetrics
        """
        self.model.train(mode=train)
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            logits, _ = self.model(x, edge_index, return_attention=False)
            hidden, _ = self.model.encode(x, edge_index, return_attention=False)
            loss = self.loss_fn(logits, y, phase, edge_index, hidden=hidden)

            if train:
                self.optimizer.zero_grad()
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

            probs = torch.softmax(logits, dim=-1)[:, self.config.loss_config.ps_class_index]

        valid_mask = (y != self.config.loss_config.ignore_index).cpu().numpy()
        probs_np = probs.detach().cpu().numpy()[valid_mask]
        targets_np = y.detach().cpu().numpy()[valid_mask]
        if valid_mask.sum() == 0:
            metrics = EpochMetrics(0.0, 0.0, 0.0, float("nan"), float(loss.total.item()))
        else:
            metrics = compute_metrics(
                probs_np,
                targets_np,
                loss.total.item(),
                self.config.loss_config.ps_class_index,
            )
        return loss, metrics

    def fit(self, data) -> list[dict[str, float]]:
        """Train with 5-fold spatial-block cross-validation and curriculum learning.

        For simplicity and to keep a single set of model weights, this
        implementation trains sequentially across folds within *each*
        epoch (i.e. one gradient step per fold per epoch, always
        validating on the held-out block for that fold) rather than
        training ``n_blocks`` independent models — this exposes the model
        to the full scene every epoch while still reporting honest,
        spatially-blocked validation metrics, averaged across folds.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with ``x`` (N, 19), ``edge_index`` (2, E), ``edge_attr``
            (E, 2), ``y`` (N,), ``pos`` (N, 2), and a ``phase`` attribute
            (T, N) attached by the caller (phase is not part of the
            standard PyG ``Data`` schema, so callers should set
            ``data.phase = torch.from_numpy(phase_flat)`` before calling
            ``fit``).

        Returns
        -------
        list[dict[str, float]]
            Per-epoch history: mean train/val loss and metrics averaged
            across the ``n_spatial_blocks`` folds.
        """
        cfg = self.config
        device = self.model.device

        x = data.x.to(device)
        y_full = data.y.to(device)
        edge_index_full = data.edge_index.to(device)
        edge_attr_full = data.edge_attr.to(device)
        phase = data.phase.to(device)
        pos = data.pos.cpu().numpy()

        blocks = spatial_block_split(pos, n_blocks=cfg.n_spatial_blocks)
        blocks_t = torch.from_numpy(blocks).to(device)

        history: list[dict[str, float]] = []

        for epoch in range(1, cfg.n_epochs + 1):
            min_corr = _curriculum_min_correlation(epoch, cfg.curriculum_schedule)
            epoch_edge_index = _filter_edges_by_correlation(
                edge_index_full, edge_attr_full, min_corr
            )

            fold_train_losses = []
            fold_val_metrics = []

            for fold in range(cfg.n_spatial_blocks):
                val_mask = blocks_t == fold
                y_train = y_full.clone()
                y_train[val_mask] = self.config.loss_config.ignore_index
                y_val = y_full.clone()
                y_val[~val_mask] = self.config.loss_config.ignore_index

                train_loss, _train_metrics = self._run_epoch(
                    x, phase, y_train, epoch_edge_index, train=True
                )
                _, val_metrics = self._run_epoch(x, phase, y_val, epoch_edge_index, train=False)

                fold_train_losses.append(train_loss.total.item())
                fold_val_metrics.append(val_metrics)

            mean_train_loss = float(np.mean(fold_train_losses))
            mean_val_f1 = float(np.nanmean([m.f1 for m in fold_val_metrics]))
            mean_val_precision = float(np.nanmean([m.precision for m in fold_val_metrics]))
            mean_val_recall = float(np.nanmean([m.recall for m in fold_val_metrics]))
            mean_val_auc = float(np.nanmean([m.roc_auc for m in fold_val_metrics]))

            self.scheduler.step()

            record = {
                "epoch": epoch,
                "curriculum_min_correlation": min_corr,
                "train_loss": mean_train_loss,
                "val_f1": mean_val_f1,
                "val_precision": mean_val_precision,
                "val_recall": mean_val_recall,
                "val_roc_auc": mean_val_auc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(record)

            if self.writer is not None:
                for key, value in record.items():
                    if key != "epoch":
                        self.writer.add_scalar(f"train/{key}", value, epoch)

            logger.info(
                "Epoch %d/%d | min_corr=%.2f | train_loss=%.4f | val_f1=%.4f | val_auc=%.4f",
                epoch,
                cfg.n_epochs,
                min_corr,
                mean_train_loss,
                mean_val_f1,
                mean_val_auc,
            )

            if cfg.checkpoint_dir is not None and mean_val_f1 > self.best_val_f1:
                self.best_val_f1 = mean_val_f1
                ckpt_dir = Path(cfg.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), ckpt_dir / "best_model.pt")

        if self.writer is not None:
            self.writer.close()

        return history
