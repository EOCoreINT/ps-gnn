"""
ps_gnn.models.ps_gnn
======================

The core PS-GNN architecture: a heterogeneous stack of graph convolutions
that progressively refines per-pixel candidate-scatterer representations
by propagating information across the spatial/phase-correlation graph
built in :mod:`ps_gnn.data.preprocessing`.

Architecture
-------------
::

    Linear(19, 64) + ReLU                          (encoder)
    -> GCNConv(64, 128) + BatchNorm1d + ReLU        (layer 1: broad smoothing)
    -> GATConv(128, 128, heads=8, concat=False)     (layer 2: learned, sparse
       + Dropout(0.3) + residual                     attention over neighbors)
    -> SAGEConv(128, 64) + ReLU                     (layer 3: inductive aggregation)
    -> Linear(64, 32) + ReLU + Dropout(0.3)
    -> Linear(32, 2)                                (classifier: {non-PS, PS})

The GAT layer is the scientific centerpiece of the model: because it
learns per-edge attention coefficients, it gives the network a mechanism
to *down-weight* "false neighbor" edges (e.g. a stable building pixel
connected to a nearby, decorrelating patch of vegetation) rather than
uniformly averaging all neighbors, as GCN and mean-aggregation SAGE do.
Exposing these attention weights (see :meth:`PSGNN.forward`) is what
enables the explainability tooling in :mod:`ps_gnn.analytics.explainability`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


@dataclass
class PSGNNConfig:
    """Hyperparameters for :class:`PSGNN`.

    Attributes
    ----------
    in_channels : int
        Number of input node features (19 by default, see
        :mod:`ps_gnn.data.preprocessing`).
    encoder_hidden : int
        Width of the linear encoder applied before graph convolutions.
    gcn_hidden : int
        Output width of the first (GCN) graph convolution layer.
    gat_hidden : int
        Output width of the GAT layer (per-head width before averaging,
        since ``concat=False``).
    gat_heads : int
        Number of attention heads in the GAT layer.
    sage_hidden : int
        Output width of the final (SAGE) graph convolution layer.
    classifier_hidden : int
        Width of the hidden layer in the MLP classifier head.
    n_classes : int
        Number of output classes (2: non-PS / PS).
    dropout : float
        Dropout probability applied after the GAT layer and inside the
        classifier head.
    """

    in_channels: int = 19
    encoder_hidden: int = 64
    gcn_hidden: int = 128
    gat_hidden: int = 128
    gat_heads: int = 8
    sage_hidden: int = 64
    classifier_hidden: int = 32
    n_classes: int = 2
    dropout: float = 0.3


class PSGNN(nn.Module):
    """Graph Attention Network for Persistent Scatterer classification.

    Parameters
    ----------
    config : PSGNNConfig, optional
        Model hyperparameters. Defaults to :class:`PSGNNConfig` with the
        architecture described in the module docstring.

    Attributes
    ----------
    encoder : torch.nn.Sequential
        Linear + ReLU projection from raw node features into the model's
        working dimensionality.
    conv1 : torch_geometric.nn.GCNConv
        Broad, unweighted spatial smoothing layer.
    bn1 : torch.nn.BatchNorm1d
        Batch normalization applied after ``conv1``.
    conv2 : torch_geometric.nn.GATConv
        Multi-head attention layer; the interpretable core of the model.
    dropout2 : torch.nn.Dropout
        Dropout applied to the GAT output before the residual connection.
    residual_proj : torch.nn.Module
        Identity if ``gcn_hidden == gat_hidden``, else a Linear projection,
        so the residual connection around the GAT layer is always
        dimensionally valid.
    conv3 : torch_geometric.nn.SAGEConv
        Final inductive aggregation layer.
    classifier : torch.nn.Sequential
        MLP head producing class logits.

    Examples
    --------
    >>> model = PSGNN()
    >>> x = torch.randn(100, 19)
    >>> edge_index = torch.randint(0, 100, (2, 400))
    >>> logits, attn = model(x, edge_index)
    >>> logits.shape
    torch.Size([100, 2])
    """

    def __init__(self, config: PSGNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PSGNNConfig()
        cfg = self.config

        self.encoder = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.encoder_hidden),
            nn.ReLU(inplace=True),
        )

        self.conv1 = GCNConv(cfg.encoder_hidden, cfg.gcn_hidden)
        self.bn1 = nn.BatchNorm1d(cfg.gcn_hidden)

        self.conv2 = GATConv(
            cfg.gcn_hidden,
            cfg.gat_hidden,
            heads=cfg.gat_heads,
            concat=False,
            add_self_loops=True,
            dropout=0.0,  # attention-weight dropout handled separately below
        )
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.residual_proj: nn.Module = (
            nn.Identity()
            if cfg.gcn_hidden == cfg.gat_hidden
            else nn.Linear(cfg.gcn_hidden, cfg.gat_hidden)
        )

        self.conv3 = SAGEConv(cfg.gat_hidden, cfg.sage_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.sage_hidden, cfg.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, cfg.n_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
            Node feature matrix.
        edge_index : Tensor, shape (2, E), dtype long
            Graph connectivity in COO format.
        return_attention : bool, default True
            If True, also return the GAT layer's attention weights
            (edge-level, averaged over heads).

        Returns
        -------
        logits : Tensor, shape (N, n_classes)
            Raw (pre-softmax) class scores for every node.
        attention : Tensor or None, shape (E', 1)
            Attention coefficients from the GAT layer, one per (possibly
            self-loop-augmented) edge as returned by
            ``GATConv(..., return_attention_weights=True)``. ``None`` if
            ``return_attention`` is False. Note ``E'`` may exceed ``E``
            because of the added self-loops.
        """
        hidden, attention = self.encode(x, edge_index, return_attention=return_attention)
        logits = self.classifier(hidden)
        return logits, attention

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the graph-convolutional trunk, stopping before the classifier.

        Exposed separately from :meth:`forward` so that downstream
        components which need the pre-classifier node representations —
        notably the spatial-coherence term of
        :class:`ps_gnn.models.losses.PhysicsInformedPSLoss`, which compares
        hidden embeddings of same-label neighbors — can obtain them without
        duplicating the encoder/GCN/GAT/SAGE forward logic.

        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
            Node feature matrix.
        edge_index : Tensor, shape (2, E), dtype long
            Graph connectivity in COO format.
        return_attention : bool, default True
            If True, also return the GAT layer's attention weights.

        Returns
        -------
        hidden : Tensor, shape (N, sage_hidden)
            Node embeddings after the SAGE layer, immediately before the
            MLP classifier head.
        attention : Tensor or None, shape (E', 1)
            Same semantics as in :meth:`forward`.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        h = self.encoder(x)

        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = F.relu(h)

        if return_attention:
            h_gat, (_attn_edge_index, attn_weights) = self.conv2(
                h, edge_index, return_attention_weights=True
            )
            # Average across heads to get a single interpretable weight per edge.
            attention = attn_weights.mean(dim=-1, keepdim=True)
        else:
            h_gat = self.conv2(h, edge_index)
            attention = None

        h_gat = self.dropout2(h_gat)
        h = self.residual_proj(h) + h_gat  # residual connection around GAT

        h = self.conv3(h, edge_index)
        h = F.relu(h)
        return h, attention

    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Convenience wrapper returning softmax class probabilities.

        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
        edge_index : Tensor, shape (2, E)

        Returns
        -------
        Tensor, shape (N, n_classes)
            Softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index, return_attention=False)
            return F.softmax(logits, dim=-1)
