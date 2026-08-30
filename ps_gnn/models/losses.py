"""
ps_gnn.models.losses
======================

:class:`PhysicsInformedPSLoss` combines a standard classification
objective with two InSAR-physics-grounded regularization terms, so the
network is pushed not just towards the correct label, but towards
*physically self-consistent* predictions:

- **Phase stability**: a genuine PS pixel should have a temporally stable
  phase (that's the entire premise of PSI). If the model predicts "PS"
  for a pixel whose phase is actually noisy over time, that's physically
  suspicious — this term penalizes exactly that.
- **Spatial coherence**: neighboring pixels that share the same
  ground-truth label (both PS or both non-PS) should, after graph
  convolution, end up with *similar* learned representations — encoding
  the prior that PS-ness is spatially structured (buildings, not single
  isolated pixels), consistent with the graph's phase-correlation edges.

Total loss: ``0.7 * CE + 0.2 * phase_stability + 0.1 * spatial_coherence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LossComponents(NamedTuple):
    """Breakdown of the total loss into its constituent terms (for logging).

    Attributes
    ----------
    total : Tensor, scalar
        The final weighted sum, ``0.7 * ce + 0.2 * phase + 0.1 * spatial``.
    cross_entropy : Tensor, scalar
        Weighted cross-entropy classification loss.
    phase_stability : Tensor, scalar
        Phase-stability penalty term (before weighting).
    spatial_coherence : Tensor, scalar
        Spatial-coherence penalty term (before weighting).
    """

    total: Tensor
    cross_entropy: Tensor
    phase_stability: Tensor
    spatial_coherence: Tensor


@dataclass
class PhysicsInformedPSLossConfig:
    """Weights and toggles for :class:`PhysicsInformedPSLoss`.

    Attributes
    ----------
    ce_weight : float
        Weight of the cross-entropy term in the total loss.
    phase_weight : float
        Weight of the phase-stability term.
    spatial_weight : float
        Weight of the spatial-coherence term.
    ps_class_index : int
        Index of the "PS" class in the logits (default 1, matching the
        ``{0: non-PS, 1: PS}`` convention used throughout ``ps-gnn``).
    ignore_index : int
        Label value marking unlabeled nodes, excluded from the
        cross-entropy and spatial-coherence terms (but still usable as
        graph context).
    use_phase_loss : bool
        If False, the phase-stability term is computed but excluded from
        ``total`` (used by the ablation framework's ``--no_phase_loss``).
    use_spatial_loss : bool
        If False, the spatial-coherence term is excluded from ``total``
        (ablation: ``--no_spatial_loss``).
    """

    ce_weight: float = 0.7
    phase_weight: float = 0.2
    spatial_weight: float = 0.1
    ps_class_index: int = 1
    ignore_index: int = -1
    use_phase_loss: bool = True
    use_spatial_loss: bool = True


class PhysicsInformedPSLoss(nn.Module):
    """Multi-component, physics-informed loss for PS classification.

    Parameters
    ----------
    config : PhysicsInformedPSLossConfig, optional
        Component weights and toggles. Defaults to
        :class:`PhysicsInformedPSLossConfig`.

    Notes
    -----
    All three components tolerate the "no valid nodes/edges this batch"
    edge case (e.g. a spatial block with no same-label edges) by returning
    ``0.0`` for that term rather than raising or producing ``NaN``, so
    training remains stable across curriculum-learning stages and small
    spatial-CV folds.
    """

    def __init__(self, config: PhysicsInformedPSLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or PhysicsInformedPSLossConfig()

    @staticmethod
    def _class_weights(targets: Tensor, n_classes: int, ignore_index: int) -> Tensor:
        """Inverse-frequency class weights, upweighting the minority (PS) class.

        Parameters
        ----------
        targets : Tensor, shape (N,)
            Integer class labels (may contain ``ignore_index``).
        n_classes : int
            Number of classes.
        ignore_index : int
            Label value to exclude from frequency counting.

        Returns
        -------
        Tensor, shape (n_classes,)
            Per-class weights, normalized to mean 1.0 across present
            classes for numerical stability.
        """
        valid = targets[targets != ignore_index]
        weights = torch.ones(n_classes, device=targets.device)
        if valid.numel() == 0:
            return weights
        counts = torch.bincount(valid, minlength=n_classes).float()
        counts = counts.clamp(min=1.0)
        inv_freq = counts.sum() / (n_classes * counts)
        return inv_freq

    def _cross_entropy(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Weighted cross-entropy, ignoring unlabeled nodes.

        Parameters
        ----------
        predictions : Tensor, shape (N, n_classes)
            Raw logits.
        targets : Tensor, shape (N,)
            Integer labels, with unlabeled nodes marked
            ``self.config.ignore_index``.

        Returns
        -------
        Tensor, scalar
        """
        if (targets != self.config.ignore_index).sum() == 0:
            # No labeled nodes at all in this batch/fold (e.g. a curriculum
            # stage that filters down to an all-unlabeled subgraph);
            # F.cross_entropy would otherwise return NaN (mean of an empty
            # selection). A CE contribution of 0 correctly leaves the other
            # loss terms unaffected in that degenerate case.
            return torch.tensor(0.0, device=predictions.device)

        n_classes = predictions.shape[-1]
        weights = self._class_weights(targets, n_classes, self.config.ignore_index)
        return F.cross_entropy(
            predictions, targets, weight=weights, ignore_index=self.config.ignore_index
        )

    def _phase_stability(self, predictions: Tensor, phase: Tensor) -> Tensor:
        """Penalize high temporal phase *circular* variance for predicted-PS nodes.

        Uses the (differentiable) softmax probability of the PS class as a
        soft weight, rather than a hard argmax, so gradients can flow back
        into the classifier: nodes the model is confident are PS
        contribute their phase instability fully to the loss; nodes it is
        confident are non-PS contribute almost nothing.

        Why circular variance, not linear standard deviation
        -------------------------------------------------------
        Phase is an angular quantity wrapped to ``(-pi, pi]``, so its
        *linear* standard deviation is not a valid measure of dispersion:
        two phase samples of ``3.1`` and ``-3.1`` radians are physically
        almost identical (both lie right next to the wrap boundary, ~0.08
        rad apart on the unit circle), yet ``torch.std([3.1, -3.1])``
        reports a huge value (~4.38) because it measures distance along
        the real number line rather than along the circle. A model whose
        predicted-PS pixel happens to sit near the wrap boundary would
        then be *unfairly* penalized as if its phase were wildly unstable,
        even though it is not.

        Circular variance fixes this by measuring dispersion via the
        mean resultant vector length of the wrapped phases treated as unit
        vectors on the complex plane: ``V = 1 - |mean(exp(i * phase))|``.
        ``V`` is ``0`` for a perfectly stable (identical) phase and
        approaches ``1`` as phase becomes uniformly spread around the
        circle, correctly reporting near-zero dispersion for samples like
        ``[3.1, -3.1]`` that are close on the circle but far apart on the
        real line. It is smooth and differentiable in ``phase`` (composed
        of ``exp``, ``mean``, and ``abs``), so gradients still flow
        normally into any upstream computation of ``phase``.

        Parameters
        ----------
        predictions : Tensor, shape (N, n_classes)
            Raw logits.
        phase : Tensor, shape (T, N)
            Wrapped phase time series per node, radians.

        Returns
        -------
        Tensor, scalar
            Probability-weighted mean temporal circular phase variance.
        """
        probs = F.softmax(predictions, dim=-1)[:, self.config.ps_class_index]  # (N,)
        # Circular variance V = 1 - R, where R = |mean(exp(i * phase))| is the
        # mean resultant vector length of the phase treated as points on the
        # unit circle. Correctly reports near-zero dispersion for phase
        # samples that straddle the +/-pi wrap boundary, unlike a linear
        # standard deviation (see docstring above).
        circular_phase_variance = 1.0 - torch.abs(torch.mean(torch.exp(1j * phase), dim=0))  # (N,)
        weighted = probs * circular_phase_variance
        return weighted.mean()

    def _spatial_coherence(
        self,
        hidden: Tensor,
        targets: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Penalize embedding distance between same-label neighbors.

        For every edge ``(i, j)`` where both endpoints are labeled and
        ``y_i == y_j``, penalize ``||h_i - h_j||^2``. This encodes the
        prior that spatially/phase-correlated pixels sharing a ground
        truth label should also be close in the learned representation
        space.

        Parameters
        ----------
        hidden : Tensor, shape (N, D)
            Pre-classifier node embeddings (e.g. from
            :meth:`ps_gnn.models.ps_gnn.PSGNN.encode`).
        targets : Tensor, shape (N,)
            Integer labels, with unlabeled nodes marked
            ``self.config.ignore_index``.
        edge_index : Tensor, shape (2, E)
            Graph connectivity.

        Returns
        -------
        Tensor, scalar
        """
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=hidden.device)

        src, dst = edge_index[0], edge_index[1]
        both_labeled = (targets[src] != self.config.ignore_index) & (
            targets[dst] != self.config.ignore_index
        )
        same_label = both_labeled & (targets[src] == targets[dst])

        if same_label.sum() == 0:
            return torch.tensor(0.0, device=hidden.device)

        diff = hidden[src[same_label]] - hidden[dst[same_label]]
        return (diff.pow(2).sum(dim=-1)).mean()

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        phase: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
    ) -> LossComponents:
        """Compute the total physics-informed loss and its components.

        Parameters
        ----------
        predictions : Tensor, shape (N, n_classes)
            Raw (pre-softmax) model logits.
        targets : Tensor, shape (N,)
            Integer ground-truth labels; unlabeled nodes should be marked
            with ``self.config.ignore_index`` (default ``-1``).
        phase : Tensor, shape (T, N)
            Wrapped phase time series for every node.
        edge_index : Tensor, shape (2, E)
            Graph connectivity, used by the spatial-coherence term.
        hidden : Tensor, shape (N, D), optional
            Pre-classifier node embeddings, required (non-``None``) if
            ``config.use_spatial_loss`` is True.

        Returns
        -------
        LossComponents
            Named tuple with ``total`` and the three individual terms
            (all individual terms are returned un-weighted, for
            transparent logging; ``total`` applies the configured
            weights).

        Raises
        ------
        ValueError
            If ``use_spatial_loss`` is True but ``hidden`` is not provided.
        """
        cfg = self.config

        ce = self._cross_entropy(predictions, targets)
        phase_loss = self._phase_stability(predictions, phase)

        if cfg.use_spatial_loss:
            if hidden is None:
                raise ValueError(
                    "hidden embeddings must be provided when use_spatial_loss=True "
                    "(e.g. from PSGNN.encode())."
                )
            spatial_loss = self._spatial_coherence(hidden, targets, edge_index)
        else:
            spatial_loss = torch.tensor(0.0, device=predictions.device)

        total = cfg.ce_weight * ce
        if cfg.use_phase_loss:
            total = total + cfg.phase_weight * phase_loss
        if cfg.use_spatial_loss:
            total = total + cfg.spatial_weight * spatial_loss

        return LossComponents(
            total=total,
            cross_entropy=ce,
            phase_stability=phase_loss,
            spatial_coherence=spatial_loss,
        )