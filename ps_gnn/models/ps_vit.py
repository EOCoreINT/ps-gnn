"""
ps_gnn.models.ps_vit
======================

PS-ViT: a spatio-temporal Vision Transformer baseline for Persistent
Scatterer classification, used as an ablation counterpart to the
graph-based :class:`ps_gnn.models.ps_gnn.PSGNN`. Where PS-GNN reasons
explicitly over a sparse, physically-motivated graph, PS-ViT instead
tiles the scene into regular patches and lets self-attention discover
spatial relationships implicitly and densely — a useful comparison point
for quantifying how much the graph's inductive bias (sparsity, explicit
phase-correlation edges) actually helps versus a generic, powerful
sequence/attention model.

Architecture
-------------
::

    Input: SAR amplitude/phase stack, shape (B, T, C=2, H, W)
    -> Patch Embedding (8x8 patches, embed_dim=64) + learnable
       positional encoding                                  -> (B, T, P, 64)
    -> Bi-directional LSTM over time, per patch
       (hidden=128, 2 layers)                                -> (B, T, P, 256)
       -> temporal mean pool + project to 128                -> (B, P, 128)
    -> Transformer Encoder (6 layers, 8 heads, ff=512)
       over the patch sequence                                -> (B, P, 128)
    -> Global Average Pooling over patches -> Linear(128, 2)   -> (B, 2)

where ``P`` is the number of ``8x8`` patches tiling the (zero-padded, if
necessary) ``H x W`` frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PSViTConfig:
    """Hyperparameters for :class:`PSViT`.

    Attributes
    ----------
    in_channels : int
        Number of input channels per frame (2: amplitude + phase).
    patch_size : int
        Side length (pixels) of each square patch.
    embed_dim : int
        Dimensionality of the initial patch embedding.
    lstm_hidden : int
        Hidden size of the bidirectional LSTM temporal encoder (per
        direction; total output width is ``2 * lstm_hidden``).
    lstm_layers : int
        Number of stacked LSTM layers.
    transformer_dim : int
        Model (``d_model``) dimensionality used by the Transformer
        encoder; the ``2 * lstm_hidden``-wide temporal features are
        projected down/up to this size.
    transformer_layers : int
        Number of Transformer encoder layers.
    transformer_heads : int
        Number of self-attention heads.
    transformer_ff : int
        Width of the Transformer's position-wise feed-forward network.
    n_classes : int
        Number of output classes (2: non-PS / PS).
    dropout : float
        Dropout probability used throughout the model.
    """

    in_channels: int = 2
    patch_size: int = 8
    embed_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    transformer_dim: int = 128
    transformer_layers: int = 6
    transformer_heads: int = 8
    transformer_ff: int = 512
    n_classes: int = 2
    dropout: float = 0.1


class PatchEmbedding(nn.Module):
    """Splits each frame into non-overlapping patches and embeds them.

    Uses a strided 2D convolution (kernel = stride = ``patch_size``) as an
    efficient, learnable implementation of "flatten patch pixels + linear
    project", applied identically at every time step. A learnable
    positional encoding (one vector per patch *position*, shared across
    time) is added afterwards.

    Parameters
    ----------
    in_channels : int
        Number of input channels per frame.
    patch_size : int
        Side length of each square patch.
    embed_dim : int
        Output embedding dimensionality per patch.
    max_patches : int, default 4096
        Upper bound on the number of patches supported by the positional
        encoding table (safety cap for very large scenes).
    """

    def __init__(
        self, in_channels: int, patch_size: int, embed_dim: int, max_patches: int = 4096
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _pad_to_patch_multiple(self, x: Tensor) -> Tensor:
        """Zero-pad the spatial dims so H and W are multiples of patch_size."""
        _, _, height, width = x.shape
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Embed one batch of frames into patch tokens.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)
            A single time step's amplitude/phase frame(s).

        Returns
        -------
        Tensor, shape (B, P, embed_dim)
            Patch embeddings with positional encoding added, where
            ``P = (H // patch_size) * (W // patch_size)`` (after padding).
        """
        x = self._pad_to_patch_multiple(x)
        tokens = self.proj(x)  # (B, embed_dim, H', W')
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, P, embed_dim)

        n_patches = tokens.shape[1]
        if n_patches > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Scene produces {n_patches} patches, exceeding the positional "
                f"encoding capacity of {self.pos_embedding.shape[1]}. Increase "
                "max_patches when constructing PatchEmbedding."
            )
        tokens = tokens + self.pos_embedding[:, :n_patches, :]
        return tokens


class PSViT(nn.Module):
    """Spatio-temporal Vision Transformer for PS classification.

    Parameters
    ----------
    config : PSViTConfig, optional
        Model hyperparameters. Defaults to :class:`PSViTConfig`.

    Examples
    --------
    >>> model = PSViT()
    >>> x = torch.randn(2, 10, 2, 32, 32)  # (B, T, C, H, W)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([2, 2])
    """

    def __init__(self, config: PSViTConfig | None = None) -> None:
        super().__init__()
        self.config = config or PSViTConfig()
        cfg = self.config

        self.patch_embed = PatchEmbedding(cfg.in_channels, cfg.patch_size, cfg.embed_dim)

        self.temporal_encoder = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = cfg.lstm_hidden * 2  # bidirectional
        self.temporal_proj = nn.Linear(lstm_out_dim, cfg.transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.transformer_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.transformer_dim),
            nn.Linear(cfg.transformer_dim, cfg.n_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, T, C, H, W)
            Multi-temporal SAR amplitude/phase stack, ``C`` typically 2
            (amplitude and phase stacked as channels).

        Returns
        -------
        Tensor, shape (B, n_classes)
            Raw (pre-softmax) class scores for the tile/scene.
        """
        x = x.to(self.device)
        batch_size, n_steps, channels, height, width = x.shape

        # Embed every time step's frame into patch tokens: (B*T, P, embed_dim)
        frames = x.reshape(batch_size * n_steps, channels, height, width)
        patch_tokens = self.patch_embed(frames)
        n_patches = patch_tokens.shape[1]
        embed_dim = patch_tokens.shape[2]

        # Rearrange so each patch position has its own temporal sequence:
        # (B, T, P, D) -> (B, P, T, D) -> (B*P, T, D)
        patch_tokens = patch_tokens.reshape(batch_size, n_steps, n_patches, embed_dim)
        patch_tokens = patch_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * n_patches, n_steps, embed_dim
        )

        lstm_out, _ = self.temporal_encoder(patch_tokens)  # (B*P, T, 2*lstm_hidden)
        temporal_feat = lstm_out.mean(dim=1)  # temporal mean pooling -> (B*P, 2*lstm_hidden)
        temporal_feat = self.temporal_proj(temporal_feat)  # (B*P, transformer_dim)

        # Back to a per-scene sequence of patch tokens: (B, P, transformer_dim)
        tokens = temporal_feat.reshape(batch_size, n_patches, self.config.transformer_dim)

        encoded = self.transformer_encoder(tokens)  # (B, P, transformer_dim)
        pooled = encoded.mean(dim=1)  # Global Average Pooling over patches

        logits = self.head(pooled)
        return logits

    def predict_proba(self, x: Tensor) -> Tensor:
        """Convenience wrapper returning softmax class probabilities.

        Parameters
        ----------
        x : Tensor, shape (B, T, C, H, W)

        Returns
        -------
        Tensor, shape (B, n_classes)
            Softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
