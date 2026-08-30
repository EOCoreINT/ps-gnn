# ps-gnn Documentation

This is a lightweight, markdown-based documentation index. Every module
in the package has a complete NumPy-style docstring at the top of its
file — this index is a map to help you find the right one, not a
duplicate of that reference material.

## Getting started

See the [README](../README.md) for installation and a quickstart example.

## Pipeline stages

The package is organized around the natural PS-GNN pipeline. Each stage
below links to its module and lists the key entry points.

### 1. Data acquisition & labeling — `ps_gnn.data`

| Module | Purpose |
|---|---|
| [`fetch_benchmarks`](../ps_gnn/data/fetch_benchmarks.py) | Download/organize the five benchmark InSAR sites (Mexico City, Amatrice, Piton de la Fournaise, Berlin, Jakarta) via `pygeofetch`, with a synthetic mock fallback. |
| [`label_generation`](../ps_gnn/data/label_generation.py) | `inject_synthetic_ps` (controlled-ADI synthetic PS) and `generate_stamps_pseudo_labels` (high-confidence pseudo-labels from a classical StaMPS run). |
| [`preprocessing`](../ps_gnn/data/preprocessing.py) | `compute_node_features` (19-D feature vector) and `build_graph` (spatial-proximity + phase-correlation edges) -> `build_pyg_dataset`. |

### 2. Models — `ps_gnn.models`

| Module | Purpose |
|---|---|
| [`ps_gnn`](../ps_gnn/models/ps_gnn.py) | `PSGNN`: GCN -> GAT (attention-exposing) -> SAGE -> MLP classifier. |
| [`ps_vit`](../ps_gnn/models/ps_vit.py) | `PSViT`: patch embedding + Bi-LSTM + Transformer, an ablation baseline. |
| [`losses`](../ps_gnn/models/losses.py) | `PhysicsInformedPSLoss`: weighted CE + phase-stability + spatial-coherence terms. |

### 3. Training — `ps_gnn.training`

| Module | Purpose |
|---|---|
| [`trainer`](../ps_gnn/training/trainer.py) | `Trainer`: 5-fold spatial-block CV, curriculum learning, AdamW + warmup/cosine schedule. |
| [`ablation`](../ps_gnn/training/ablation.py) | Sweeps loss components, backbone (GNN/ViT), and graph thresholds; also installed as the `ps-gnn-ablation` CLI command. |

### 4. Inference & deployment

| Module | Purpose |
|---|---|
| [`inference.detector`](../ps_gnn/inference/detector.py) | `PSDetector`: tiled inference, Monte Carlo Dropout uncertainty, clustering. |
| [`utils.deployment`](../ps_gnn/utils/deployment.py) | ONNX export, Zenodo weight caching, GPU/CPU/OpenVINO runtime fallback. |

### 5. Validation — `ps_gnn.validation`

| Module | Purpose |
|---|---|
| [`sbas_check`](../ps_gnn/validation/sbas_check.py) | From-scratch WLS SBAS inversion; `run_sbas_validation` compares PS-GNN vs. an ADI baseline. |

### 6. Analytics & reporting

| Module | Purpose |
|---|---|
| [`analytics.explainability`](../ps_gnn/analytics/explainability.py) | SHAP-based node and global feature attribution. |
| [`analytics.spatial_stats`](../ps_gnn/analytics/spatial_stats.py) | Moran's I, Ripley's K, uncertainty calibration. |
| [`visualization.maps`](../ps_gnn/visualization/maps.py) | Folium 2D map, Plotly 3D terrain projection. |
| [`visualization.charts`](../ps_gnn/visualization/charts.py) | Training curves, confusion matrix, PR curve, feature violin plots. |
| [`analytics.report_generator`](../ps_gnn/analytics/report_generator.py) | Assembles the above into a single self-contained HTML report. |

## CLI

```bash
ps-gnn-ablation --data path/to/graph.pt --output results/ablation_summary.csv \
    --model gnn vit --no_phase_loss --edge_dist 50 100 --edge_corr 0.5 0.7
```

## Testing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to run the test suite
and the project's coding conventions.
