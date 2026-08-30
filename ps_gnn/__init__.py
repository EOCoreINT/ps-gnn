"""
ps-gnn: AI-driven Persistent Scatterer identification for InSAR time series.

ps-gnn replaces the classical, per-pixel Amplitude Dispersion Index (ADI)
threshold with a Graph Attention Network (GAT) that reasons over the
*spatial relationships* between candidate scatterers. By propagating
information across a graph built from spatial proximity and phase
correlation, the model learns to down-weight noisy "false neighbors"
(e.g. vegetation, layover/shadow) and to up-weight mechanically stable,
coherent structures (buildings, infrastructure, exposed rock).

This top-level package intentionally stays lightweight: it re-exports the
package version and leaves heavy imports (torch, torch_geometric, rasterio,
...) to the submodules that actually need them, so that ``import ps_gnn``
stays fast and side-effect free.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
