"""
ps_gnn.analytics.report_generator
====================================

Generates a single, self-contained HTML report summarizing a PS-GNN
inference (or training) run: executive summary, model performance
charts, spatial analysis (Moran's I, Ripley's K, an interactive Folium
map), explainability (SHAP feature importance), and a provenance
footer (model hash, timestamp, package version).

Design notes
-------------
:func:`generate_html_report` is deliberately built to degrade
gracefully: it is called by
:meth:`ps_gnn.inference.detector.PSDetector.detect` with only the raw
inference ``results`` dict (``ps_mask``, ``probability``,
``uncertainty``, ``cluster_labels``) — no training history, ground
truth, or SHAP values are available at that point. Every optional
section (training curves, confusion matrix/PR curve, explainability) is
therefore skipped with a note in the report rather than raising, so a
minimal-but-valid report is always produced; richer reports are produced
when the optional keyword arguments are supplied by a caller with more
context (e.g. after a full training + evaluation run).

Plotly figures and the Folium map are embedded as standalone HTML
fragments saved under an ``assets/`` folder next to the report and
linked via ``<iframe>``, keeping the main report document itself small
and fast to open even though the embedded visualizations are rich.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PS-GNN Detection Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 0; background: #f7f8fa; color: #1a1a1a; }
  header { background: #0b3d91; color: white; padding: 24px 32px; }
  header h1 { margin: 0 0 4px 0; font-size: 1.6em; }
  header p { margin: 0; opacity: 0.85; font-size: 0.9em; }
  main { max-width: 1100px; margin: 0 auto; padding: 24px 32px 64px; }
  section { background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  section h2 { margin-top: 0; border-bottom: 2px solid #eef1f6; padding-bottom: 8px; }
  .stat-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }
  .stat-card { flex: 1 1 160px; background: #f2f5fa; border-radius: 6px; padding: 12px 16px; }
  .stat-card .value { font-size: 1.6em; font-weight: 600; color: #0b3d91; }
  .stat-card .label { font-size: 0.85em; color: #555; }
  iframe { width: 100%; border: none; border-radius: 6px; }
  .skipped { color: #888; font-style: italic; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 0.9em; }
  footer { color: #888; font-size: 0.8em; padding: 16px 32px; }
  code { background: #eef1f6; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>PS-GNN Detection Report</h1>
  <p>Generated __GENERATED_AT__</p>
</header>
<main>

<section>
  <h2>Executive Summary</h2>
  <div class="stat-grid">
    <div class="stat-card"><div class="value">__N_PS_POINTS__</div><div class="label">PS pixels detected</div></div>
    <div class="stat-card"><div class="value">__N_CLUSTERS__</div><div class="label">Clusters</div></div>
    <div class="stat-card"><div class="value">__COVERAGE_PCT__</div><div class="label">Scene coverage</div></div>
    <div class="stat-card"><div class="value">__MEAN_CONFIDENCE__</div><div class="label">Mean confidence</div></div>
  </div>
</section>

<section>
  <h2>Model Performance</h2>
  __PERFORMANCE_SECTION__
</section>

<section>
  <h2>Spatial Analysis</h2>
  __SPATIAL_SECTION__
</section>

<section>
  <h2>Explainability</h2>
  __EXPLAINABILITY_SECTION__
</section>

</main>
<footer>
  ps-gnn version __PACKAGE_VERSION__ &middot; model hash <code>__MODEL_HASH__</code> &middot; report generated __GENERATED_AT__
</footer>
</body>
</html>
"""


def _save_fragment(html_fragment: str, assets_dir: Path, filename: str) -> Path:
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / filename
    path.write_text(html_fragment)
    return path


def _iframe(relative_path: str, height: int = 500) -> str:
    return f'<iframe src="{relative_path}" height="{height}"></iframe>'


def generate_html_report(
    results: dict[str, np.ndarray],
    output_dir: str | Path = "reports",
    training_history: list[dict] | None = None,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
    dem: np.ndarray | None = None,
    transform=None,
    shap_summary=None,
    node_explanations: dict[tuple[int, int], list[tuple[str, float]]] | None = None,
    model_hash: str | None = None,
    pixel_spacing_m: float = 5.0,
) -> Path:
    """Generate a self-contained HTML report from inference (and optional training) results.

    Parameters
    ----------
    results : dict
        Inference results as returned by
        :meth:`ps_gnn.inference.detector.PSDetector.detect`: ``ps_mask``,
        ``probability``, ``uncertainty``, ``cluster_labels`` (all
        ``(H, W)`` arrays). This is the only required argument.
    output_dir : str or pathlib.Path, default "reports"
        Directory to write ``report.html`` and its ``assets/`` folder
        into.
    training_history : list[dict], optional
        Per-epoch training history (see
        :meth:`ps_gnn.training.trainer.Trainer.fit`). If provided, a
        training-curve chart is included.
    y_true, y_prob : np.ndarray, optional
        Ground-truth labels and predicted probabilities (flat arrays,
        same length), for the confusion matrix and PR curve. Both must
        be provided together.
    dem : np.ndarray, shape (H, W), optional
        Digital elevation model, for the 3D terrain plot.
    transform : optional
        Affine transform for the Folium map (see
        :func:`ps_gnn.visualization.maps.pixel_to_lonlat`).
    shap_summary : pd.DataFrame, optional
        Global feature-importance summary (see
        :func:`ps_gnn.analytics.explainability.aggregate_global_feature_importance`).
    node_explanations : dict, optional
        Per-node top-feature SHAP explanations, embedded in map popups.
    model_hash : str, optional
        A short identifier (e.g. a weights-file hash) for provenance.
    pixel_spacing_m : float, default 5.0
        Ground sampling distance, for the 3D terrain plot.

    Returns
    -------
    pathlib.Path
        Path to the generated ``report.html``.

    Raises
    ------
    KeyError
        If ``results`` is missing any of the four required keys.
    """
    required_keys = {"ps_mask", "probability", "uncertainty", "cluster_labels"}
    missing = required_keys - set(results.keys())
    if missing:
        raise KeyError(f"results is missing required keys: {sorted(missing)}")

    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    ps_mask = results["ps_mask"]
    probability = results["probability"]
    cluster_labels = results["cluster_labels"]

    n_ps_points = int(ps_mask.sum())
    n_clusters = len(set(cluster_labels[cluster_labels > 0].tolist()))
    coverage_pct = f"{100.0 * ps_mask.mean():.2f}%"
    mean_confidence = f"{probability[ps_mask].mean():.3f}" if n_ps_points > 0 else "n/a"

    # --- Model performance section ---
    performance_parts = []
    if training_history is not None:
        try:
            from ps_gnn.visualization.charts import plot_training_curves

            fig = plot_training_curves(training_history)
            frag_path = _save_fragment(
                fig.to_html(include_plotlyjs="cdn", full_html=True),
                assets_dir,
                "training_curves.html",
            )
            performance_parts.append(_iframe(f"assets/{frag_path.name}"))
        except Exception:
            logger.exception("Failed to render training curves; skipping.")
            performance_parts.append('<p class="skipped">Training curves unavailable.</p>')
    else:
        performance_parts.append(
            '<p class="skipped">No training history supplied; skipping training curves.</p>'
        )

    if y_true is not None and y_prob is not None:
        try:
            from ps_gnn.visualization.charts import plot_confusion_matrix, plot_pr_curve

            y_pred = (y_prob >= 0.5).astype(int)
            cm_fig = plot_confusion_matrix(y_true, y_pred)
            pr_fig = plot_pr_curve(y_true, y_prob)
            cm_path = _save_fragment(
                cm_fig.to_html(include_plotlyjs="cdn", full_html=True),
                assets_dir,
                "confusion_matrix.html",
            )
            pr_path = _save_fragment(
                pr_fig.to_html(include_plotlyjs="cdn", full_html=True), assets_dir, "pr_curve.html"
            )
            performance_parts.append(_iframe(f"assets/{cm_path.name}", height=450))
            performance_parts.append(_iframe(f"assets/{pr_path.name}", height=450))
        except Exception:
            logger.exception("Failed to render confusion matrix / PR curve; skipping.")
            performance_parts.append('<p class="skipped">Classification metrics unavailable.</p>')
    else:
        performance_parts.append(
            '<p class="skipped">No ground-truth labels supplied; skipping confusion matrix / PR curve.</p>'
        )

    # --- Spatial analysis section ---
    spatial_parts = []
    try:
        from ps_gnn.analytics.spatial_stats import morans_i_residuals, ripleys_k_clustering

        rows, cols = np.nonzero(ps_mask)
        if len(rows) >= 10:
            coords = np.stack([cols, rows], axis=1).astype(float) * pixel_spacing_m
            moran_result = morans_i_residuals(
                probability[ps_mask], coords, k_neighbors=min(8, len(rows) - 1)
            )
            spatial_parts.append(
                f"<p><b>Moran's I</b> on detection confidence: <code>{moran_result.I:.4f}</code> "
                f"(expected under CSR: <code>{moran_result.expected_I:.4f}</code>"
                + (f", p={moran_result.p_value:.4f}" if moran_result.p_value is not None else "")
                + f", backend=<code>{moran_result.backend}</code>). "
                "Positive values indicate confidence is spatially clustered, consistent with PS "
                "detections concentrating on physical structures rather than scattering randomly.</p>"
            )

            if len(rows) >= 3:
                ripley_result = ripleys_k_clustering(coords, n_distance_bins=8, n_simulations=49)
                n_sig_cluster = int(ripley_result.is_clustered.sum())
                spatial_parts.append(
                    f"<p><b>Ripley's K</b>: significant clustering (vs. CSR) detected at "
                    f"{n_sig_cluster}/{len(ripley_result.support)} evaluated distances.</p>"
                )
        else:
            spatial_parts.append(
                '<p class="skipped">Too few PS detections for spatial statistics.</p>'
            )
    except Exception:
        logger.exception("Failed to compute spatial statistics; skipping.")
        spatial_parts.append('<p class="skipped">Spatial statistics unavailable.</p>')

    try:
        from ps_gnn.visualization.maps import create_folium_map

        fmap = create_folium_map(
            ps_mask,
            probability,
            cluster_labels,
            transform=transform,
            node_explanations=node_explanations,
        )
        map_path = assets_dir / "ps_map.html"
        assets_dir.mkdir(parents=True, exist_ok=True)
        fmap.save(str(map_path))
        spatial_parts.append(_iframe(f"assets/{map_path.name}", height=500))
    except Exception:
        logger.exception("Failed to render Folium map; skipping.")
        spatial_parts.append('<p class="skipped">Interactive map unavailable.</p>')

    if dem is not None:
        try:
            from ps_gnn.visualization.maps import create_3d_terrain_plot

            terrain_fig = create_3d_terrain_plot(dem, ps_mask, probability, pixel_spacing_m)
            terrain_path = _save_fragment(
                terrain_fig.to_html(include_plotlyjs="cdn", full_html=True),
                assets_dir,
                "terrain_3d.html",
            )
            spatial_parts.append(_iframe(f"assets/{terrain_path.name}", height=550))
        except Exception:
            logger.exception("Failed to render 3D terrain plot; skipping.")
            spatial_parts.append('<p class="skipped">3D terrain plot unavailable.</p>')

    # --- Explainability section ---
    explainability_parts = []
    if shap_summary is not None:
        try:
            top_rows = shap_summary.head(10)
            table_rows = "".join(
                f"<tr><td>{row.feature}</td><td>{row.mean_abs_shap:.4f}</td>"
                f"<td>{row.mean_shap:+.4f}</td></tr>"
                for row in top_rows.itertuples()
            )
            explainability_parts.append(
                "<table><thead><tr><th>Feature</th><th>Mean |SHAP|</th>"
                f"<th>Mean SHAP</th></tr></thead><tbody>{table_rows}</tbody></table>"
            )
        except Exception:
            logger.exception("Failed to render SHAP summary table; skipping.")
            explainability_parts.append('<p class="skipped">SHAP summary unavailable.</p>')
    else:
        explainability_parts.append(
            '<p class="skipped">No SHAP summary supplied; skipping global feature importance. '
            "Run <code>ps_gnn.analytics.explainability.aggregate_global_feature_importance</code> "
            "and pass the result as <code>shap_summary</code> to include this section.</p>"
        )

    try:
        import ps_gnn

        package_version = ps_gnn.__version__
    except Exception:  # noqa: BLE001 -- provenance footer must never block report generation
        package_version = "unknown"

    substitutions = {
        "__GENERATED_AT__": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "__N_PS_POINTS__": str(n_ps_points),
        "__N_CLUSTERS__": str(n_clusters),
        "__COVERAGE_PCT__": coverage_pct,
        "__MEAN_CONFIDENCE__": mean_confidence,
        "__PERFORMANCE_SECTION__": "\n".join(performance_parts),
        "__SPATIAL_SECTION__": "\n".join(spatial_parts),
        "__EXPLAINABILITY_SECTION__": "\n".join(explainability_parts),
        "__PACKAGE_VERSION__": package_version,
        "__MODEL_HASH__": model_hash or "n/a",
    }
    html = _REPORT_TEMPLATE
    for token, value in substitutions.items():
        html = html.replace(token, value)

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    logger.info("Generated PS-GNN report at %s", report_path)
    return report_path
