"""
ps_gnn.visualization.charts
==============================

Interactive Plotly charts for evaluating PS-GNN training and predictions:

- :func:`plot_training_curves` — training loss and validation F1 over
  epochs (see :meth:`ps_gnn.training.trainer.Trainer.fit`'s history
  output).
- :func:`plot_confusion_matrix` — a labeled confusion-matrix heatmap.
- :func:`plot_pr_curve` — precision-recall curve with the area under the
  curve annotated.
- :func:`plot_feature_violin` — violin plots comparing a feature's
  distribution (e.g. amplitude, coherence) between true PS detections and
  false positives, to visually diagnose systematic failure modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve


def plot_training_curves(history: list[dict] | pd.DataFrame):
    """Plot training loss and validation F1 (dual y-axes) over epochs.

    Parameters
    ----------
    history : list[dict] or pd.DataFrame
        Per-epoch records as returned by
        :meth:`ps_gnn.training.trainer.Trainer.fit`, with at least
        ``epoch``, ``train_loss``, and ``val_f1`` columns/keys.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    ValueError
        If required columns are missing.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plot_training_curves requires `plotly`: `pip install plotly`.") from exc

    df = pd.DataFrame(history) if not isinstance(history, pd.DataFrame) else history
    required = {"epoch", "train_loss", "val_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"history is missing required columns: {sorted(missing)}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Loss", line={"color": "#e6550d"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["epoch"], y=df["val_f1"], name="Validation F1", line={"color": "#3182bd"}),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Train Loss", secondary_y=False)
    fig.update_yaxes(title_text="Validation F1", range=[0, 1], secondary_y=True)
    fig.update_layout(title="Training Loss & Validation F1", hovermode="x unified")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
):
    """Plot a labeled confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N,)
        Ground-truth and predicted binary labels.
    class_names : list[str], optional
        Display names for the two classes. Defaults to
        ``["Non-PS", "PS"]``.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plot_confusion_matrix requires `plotly`: `pip install plotly`.") from exc

    class_names = class_names or ["Non-PS", "PS"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    text = [[f"{cm[i, j]}<br>({cm_normalized[i, j]:.1%})" for j in range(2)] for i in range(2)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis={"autorange": "reversed"},
    )
    return fig


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """Plot a precision-recall curve with AUC annotated.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth binary labels.
    y_prob : np.ndarray, shape (N,)
        Predicted probability of the positive (PS) class.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plot_pr_curve requires `plotly`: `pip install plotly`.") from exc

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = float(np.mean(y_true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision, mode="lines", name=f"PR curve (AUC={pr_auc:.3f})", fill="tozeroy"
        )
    )
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"No-skill baseline ({baseline:.3f})",
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.05]},
    )
    return fig


def plot_feature_violin(
    true_positive_values: np.ndarray,
    false_positive_values: np.ndarray,
    feature_name: str = "Feature",
):
    """Violin plots comparing a feature's distribution: true PS vs. false positives.

    Parameters
    ----------
    true_positive_values : np.ndarray
        Feature values (e.g. amplitude, temporal coherence) at pixels
        correctly predicted as PS.
    false_positive_values : np.ndarray
        Feature values at pixels incorrectly predicted as PS.
    feature_name : str, default "Feature"
        Display name for the compared feature (used in axis labels).

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plot_feature_violin requires `plotly`: `pip install plotly`.") from exc

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=true_positive_values,
            name="True PS",
            box_visible=True,
            meanline_visible=True,
            fillcolor="#3182bd",
            opacity=0.6,
            line_color="#08519c",
        )
    )
    fig.add_trace(
        go.Violin(
            y=false_positive_values,
            name="False Positives",
            box_visible=True,
            meanline_visible=True,
            fillcolor="#e6550d",
            opacity=0.6,
            line_color="#a63603",
        )
    )
    fig.update_layout(
        title=f"{feature_name}: True PS vs. False Positives",
        yaxis_title=feature_name,
    )
    return fig
