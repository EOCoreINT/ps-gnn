# ps-gnn

**Graph Attention Networks for Persistent Scatterer identification in InSAR time series.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`ps-gnn` is a standalone, open-source Python package for AI-driven **Persistent
Scatterer (PS)** identification in multi-temporal InSAR (Interferometric
Synthetic Aperture Radar) stacks. It is designed as a companion to
[`pygeofetch`](https://github.com/your-org/pygeofetch) but works entirely
on its own with any Sentinel-1 SLC-derived amplitude/phase stack.

## Why a graph, not a threshold?

Classical PS selection (StaMPS, SqueeSAR, and most operational pipelines)
relies on the **Amplitude Dispersion Index (ADI)** — a per-pixel statistic
that ignores everything happening around the pixel. Two consequences:

- Isolated noisy pixels with low ADI by chance are false positives.
- Weakly stable but *spatially corroborated* pixels (e.g. the edge of a
  building next to many other stable pixels) are missed.

`ps-gnn` reframes PS selection as **node classification on a spatial
graph**. Each candidate pixel is a node with a rich feature vector
(amplitude statistics, temporal coherence, local texture, land cover,
...). Edges connect spatially close and phase-correlated pixels. A
**Graph Attention Network (GAT)** then learns *which neighbors to trust*,
propagating evidence of mechanical stability across structures while
attention weights automatically down-weight "false neighbors" such as
vegetation or layover-affected pixels.

## Key features

- 🧠 **PS-GNN**: GCN → GAT → GraphSAGE architecture with residual
  connections and an interpretable attention layer.
- 🛰️ **PS-ViT**: a spatio-temporal Vision Transformer baseline for
  ablation studies against the graph-based approach.
- ⚛️ **Physics-informed loss**: combines classification loss with phase
  stability and spatial coherence penalties grounded in InSAR physics.
- 📊 **Explainability & spatial statistics**: GraphSHAP, Moran's I,
  Ripley's K, and uncertainty calibration out of the box.
- 🗺️ **Rich reporting**: interactive Folium/Plotly maps, 3D DEM-draped
  scatterer clouds, and a self-contained HTML report generator.
- 🚀 **Production-ready inference**: tiled processing for huge scenes,
  Monte Carlo Dropout uncertainty, ONNX export with FP16 quantization,
  and automatic CPU (OpenVINO) fallback.
- ✅ **SBAS validation**: quantitatively checks that PS-GNN selections
  produce more coherent time-series inversions than ADI baselines.

## Installation

```bash
pip install -e ".[dev,geo-extra]"
```

`ps-gnn` targets Python 3.10+ and depends on PyTorch and PyTorch
Geometric. GPU acceleration is optional but recommended for training.

## Quickstart

```python
from ps_gnn.data.fetch_benchmarks import fetch_benchmark_dataset
from ps_gnn.data.label_generation import generate_stamps_pseudo_labels

# 1. Fetch a benchmark stack (Mexico City, Amatrice, Piton de la Fournaise,
#    Berlin, Jakarta) via pygeofetch, or use your own Sentinel-1 stack.
paths = fetch_benchmark_dataset("mexico_city", target_dir="data/mexico_city")

# 2. Generate high-confidence pseudo-labels from an existing StaMPS run.
labels = generate_stamps_pseudo_labels(
    stamps_output_path="data/mexico_city/stamps_ps2.mat",
    coherence_threshold=0.9,
    min_acquisitions=20,
    top_fraction=0.10,
)
```

Subsequent pipeline stages (graph construction, training, inference,
reporting) are documented in `docs/`.

## Project layout

```
ps_gnn/
├── data/            # Fetching, ground-truth / pseudo-label generation, preprocessing, graph construction
├── models/           # PS-GNN, PS-ViT, physics-informed losses
├── training/         # Trainer, spatial cross-validation, curriculum learning, ablation runner
├── inference/         # Tiled inference, Monte Carlo uncertainty, deployment helpers
├── analytics/         # Explainability (GraphSHAP), spatial statistics, HTML report generator
├── visualization/     # Interactive maps (Folium/PyVista) and charts (Plotly)
└── utils/             # Shared utilities (I/O, config, device management)
tests/                 # Unit, model, integration and validation tests
docs/                  # Documentation
```

## Status

This package is under active development as part of a Copernicus Master's
research project on AI-assisted InSAR processing. APIs may change between
minor versions until `1.0`.

## Citation

If you use `ps-gnn` in academic work, please cite this repository (a
`CITATION.cff` will be added once the associated paper is published).

## License

MIT — see [`LICENSE`](LICENSE).
