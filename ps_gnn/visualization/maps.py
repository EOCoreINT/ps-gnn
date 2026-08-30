"""
ps_gnn.visualization.maps
============================

Interactive geospatial visualization of PS-GNN detections:

- :func:`create_folium_map` — a 2D Leaflet map (via ``folium``) with PS
  markers colored by confidence, rich popups (node ID, confidence,
  cluster ID, and optionally top SHAP features).
- :func:`create_3d_terrain_plot` — a 3D scene (via ``plotly``) draping
  detected PS points over the DEM surface, so vertical structure
  (subsidence bowls, volcanic edifices, building height) is visible
  alongside the horizontal detection pattern.

Coordinate handling
---------------------
Both functions accept an optional ``transform`` (a ``rasterio``-style
affine transform, or any callable ``(row, col) -> (x, y)``) to convert
pixel coordinates into real-world map coordinates. If omitted, a
placeholder linear mapping centered near ``(0, 0)`` is used instead (see
:func:`pixel_to_lonlat`) — sufficient to render a self-consistent,
correctly-shaped map for scenes without an attached geotransform (e.g.
synthetic/test stacks), but callers working with real Sentinel-1 scenes
should always pass the scene's actual affine transform for a
geographically accurate map.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# A conservative default: at mid-latitudes, ~0.00009 degrees per pixel
# corresponds roughly to a 10m Sentinel-1-derived pixel spacing. This is
# only used as a placeholder when no real transform is supplied.
_DEFAULT_PIXEL_SIZE_DEG = 0.00009


def pixel_to_lonlat(
    rows: np.ndarray,
    cols: np.ndarray,
    transform=None,
    origin_lon: float = 0.0,
    origin_lat: float = 0.0,
    pixel_size_deg: float = _DEFAULT_PIXEL_SIZE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel (row, col) coordinates to (lon, lat).

    Parameters
    ----------
    rows, cols : np.ndarray
        Pixel row/column coordinates.
    transform : optional
        A ``rasterio.transform.Affine``-like object (supports
        ``transform * (col, row)``), or any callable
        ``(rows, cols) -> (lon, lat)``. If provided, used directly and
        the placeholder parameters below are ignored.
    origin_lon, origin_lat : float, default 0.0
        Placeholder map origin, used only when ``transform`` is omitted.
    pixel_size_deg : float, default ~0.00009
        Placeholder degrees-per-pixel spacing, used only when
        ``transform`` is omitted.

    Returns
    -------
    lon, lat : np.ndarray
    """
    if transform is not None:
        if callable(transform) and not hasattr(transform, "__mul__"):
            return transform(rows, cols)
        lon, lat = transform * (cols, rows)  # rasterio Affine convention: (col, row)
        return np.asarray(lon), np.asarray(lat)

    lon = origin_lon + cols * pixel_size_deg
    lat = origin_lat - rows * pixel_size_deg  # rows increase downward -> latitude decreases
    return lon, lat


def create_folium_map(
    ps_mask: np.ndarray,
    probability: np.ndarray,
    cluster_labels: np.ndarray,
    transform=None,
    node_explanations: dict[tuple[int, int], list[tuple[str, float]]] | None = None,
    max_markers: int = 500,
):
    """Build an interactive 2D Folium map of PS detections.

    Parameters
    ----------
    ps_mask : np.ndarray, shape (H, W), dtype bool
        Binary PS detection mask (see
        :func:`ps_gnn.inference.detector.postprocess_predictions`).
    probability : np.ndarray, shape (H, W)
        Mean predicted PS-class probability per pixel.
    cluster_labels : np.ndarray, shape (H, W), dtype int
        Connected-component cluster ID per pixel (``0`` = non-PS).
    transform : optional
        See :func:`pixel_to_lonlat`.
    node_explanations : dict[(row, col), list[(feature_name, shap_value)]], optional
        Precomputed top-feature SHAP explanations (see
        :func:`ps_gnn.analytics.explainability.explain_node`) to embed in
        marker popups, keyed by pixel coordinate.
    max_markers : int, default 500
        If more than this many PS pixels are detected, only the
        ``max_markers`` highest-confidence ones are plotted (rendering
        every pixel of a large scene as an individual Leaflet marker is
        impractical and makes the map unusable).

    Returns
    -------
    folium.Map

    Raises
    ------
    ImportError
        If ``folium`` or ``branca`` is not installed.
    """
    try:
        import branca.colormap as cm
        import folium
    except ImportError as exc:
        raise ImportError(
            "create_folium_map requires `folium` and `branca`: `pip install folium branca`."
        ) from exc

    rows, cols = np.nonzero(ps_mask)
    if len(rows) == 0:
        logger.warning("create_folium_map: no PS pixels in ps_mask; returning an empty map.")
        rows, cols = np.array([ps_mask.shape[0] // 2]), np.array([ps_mask.shape[1] // 2])
        confidences = np.array([0.0])
        clusters = np.array([0])
    else:
        confidences = probability[rows, cols]
        clusters = cluster_labels[rows, cols]

        if len(rows) > max_markers:
            top_idx = np.argsort(-confidences)[:max_markers]
            rows, cols, confidences, clusters = (
                rows[top_idx],
                cols[top_idx],
                confidences[top_idx],
                clusters[top_idx],
            )
            logger.info(
                "create_folium_map: %d PS pixels found, plotting only the top %d by confidence.",
                ps_mask.sum(),
                max_markers,
            )

    lons, lats = pixel_to_lonlat(rows, cols, transform=transform)

    center = [float(np.mean(lats)), float(np.mean(lons))]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    colormap = cm.LinearColormap(
        colors=["#3182bd", "#fdae6b", "#e6550d", "#a63603"],
        vmin=float(confidences.min()),
        vmax=float(max(confidences.max(), confidences.min() + 1e-6)),
        caption="PS detection confidence",
    )
    colormap.add_to(fmap)

    for lon, lat, row, col, conf, cluster_id in zip(lons, lats, rows, cols, confidences, clusters):
        popup_lines = [
            f"<b>Node</b>: ({int(row)}, {int(col)})",
            f"<b>Confidence</b>: {conf:.3f}",
            f"<b>Cluster ID</b>: {int(cluster_id)}",
        ]
        if node_explanations is not None:
            top_feats = node_explanations.get((int(row), int(col)))
            if top_feats:
                popup_lines.append("<b>Top SHAP features</b>:")
                for name, val in top_feats:
                    popup_lines.append(f"&nbsp;&nbsp;{name}: {val:+.4f}")

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=4,
            color=colormap(float(conf)),
            fill=True,
            fill_color=colormap(float(conf)),
            fill_opacity=0.85,
            popup=folium.Popup("<br>".join(popup_lines), max_width=300),
        ).add_to(fmap)

    return fmap


def create_3d_terrain_plot(
    dem: np.ndarray,
    ps_mask: np.ndarray,
    probability: np.ndarray,
    pixel_spacing_m: float = 5.0,
    max_points: int = 5000,
):
    """Build a 3D Plotly scene projecting PS points onto the DEM surface.

    Parameters
    ----------
    dem : np.ndarray, shape (H, W)
        Digital elevation model, meters.
    ps_mask : np.ndarray, shape (H, W), dtype bool
        Binary PS detection mask.
    probability : np.ndarray, shape (H, W)
        Mean predicted PS-class probability per pixel, used to color the
        scattered PS points.
    pixel_spacing_m : float, default 5.0
        Ground sampling distance, used to scale the X/Y axes into meters.
    max_points : int, default 5000
        Cap on the number of PS points scattered (highest-confidence
        points kept), for rendering performance.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    ValueError
        If ``dem``, ``ps_mask``, and ``probability`` shapes disagree.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "create_3d_terrain_plot requires `plotly`: `pip install plotly`."
        ) from exc

    if not (dem.shape == ps_mask.shape == probability.shape):
        raise ValueError(
            f"Shape mismatch: dem={dem.shape}, ps_mask={ps_mask.shape}, "
            f"probability={probability.shape}"
        )

    height, width = dem.shape
    x_axis = np.arange(width) * pixel_spacing_m
    y_axis = np.arange(height) * pixel_spacing_m

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_axis,
            y=y_axis,
            z=dem,
            colorscale="earth",
            opacity=0.85,
            name="DEM",
            showscale=True,
            colorbar={"title": "Elevation (m)", "x": 1.0},
        )
    )

    rows, cols = np.nonzero(ps_mask)
    if len(rows) > 0:
        confidences = probability[rows, cols]
        elevations = dem[rows, cols] + 2.0  # small vertical offset so points sit above the surface

        if len(rows) > max_points:
            top_idx = np.argsort(-confidences)[:max_points]
            rows, cols, confidences, elevations = (
                rows[top_idx],
                cols[top_idx],
                confidences[top_idx],
                elevations[top_idx],
            )

        fig.add_trace(
            go.Scatter3d(
                x=cols * pixel_spacing_m,
                y=rows * pixel_spacing_m,
                z=elevations,
                mode="markers",
                marker={
                    "size": 3,
                    "color": confidences,
                    "colorscale": "Reds",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "colorbar": {"title": "PS confidence", "x": 1.15},
                },
                name="PS detections",
                text=[f"conf={c:.3f}" for c in confidences],
                hoverinfo="text",
            )
        )
    else:
        logger.warning("create_3d_terrain_plot: no PS pixels in ps_mask; DEM-only scene.")

    fig.update_layout(
        title="3D Persistent Scatterer Terrain Projection",
        scene={
            "xaxis_title": "Easting (m)",
            "yaxis_title": "Northing (m)",
            "zaxis_title": "Elevation (m)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig
