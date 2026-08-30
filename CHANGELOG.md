# Changelog

All notable changes to `ps-gnn` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once it
reaches `1.0.0`. Before `1.0.0`, minor versions may include breaking changes.

## [Unreleased]

## [0.1.0] - 2026-08-29

### Added
- Initial package scaffold: `pyproject.toml`, `README`, package layout
  (`data`, `models`, `training`, `inference`, `validation`, `analytics`,
  `visualization`, `utils`).
- Benchmark dataset fetching (`ps_gnn.data.fetch_benchmarks`) with a
  synthetic mock fallback for offline development.
- Ground-truth generation: synthetic PS injection and StaMPS
  pseudo-labeling (`ps_gnn.data.label_generation`).
- Node feature engineering (19-D) and spatial-proximity +
  phase-correlation graph construction (`ps_gnn.data.preprocessing`).
- `PSGNN` (GCN -> GAT -> SAGE, with exposed attention weights) and
  `PSViT` (patch embedding + Bi-LSTM + Transformer) model architectures.
- `PhysicsInformedPSLoss` combining weighted cross-entropy, phase
  stability, and spatial coherence terms.
- `Trainer` with 5-fold spatial-block cross-validation, curriculum
  learning over graph edge correlation thresholds, and TensorBoard
  logging.
- Ablation study runner (`ps_gnn.training.ablation`) sweeping loss
  components, backbone architecture, and graph-construction thresholds.
- Tiled inference pipeline (`PSDetector`) with Monte Carlo Dropout
  uncertainty and connected-component clustering.
- Production deployment helpers: ONNX export, Zenodo-based weight
  caching, and a GPU/CPU/OpenVINO inference-runtime fallback chain.
- SBAS time-series inversion (from-scratch weighted least squares) for
  validating PS selections against an ADI baseline.
- Explainability (SHAP-based node/global feature attribution) and
  spatial statistics (Moran's I, Ripley's K, uncertainty calibration).
- Interactive visualization (Folium maps, 3D terrain plots, Plotly
  charts) and an automated, gracefully-degrading HTML report generator.
- Comprehensive test suite (unit, model, integration, validation) and a
  GitHub Actions CI workflow (lint, matrixed fast tests, slow/integration
  tests).

[Unreleased]: https://github.com/your-org/ps-gnn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/ps-gnn/releases/tag/v0.1.0
