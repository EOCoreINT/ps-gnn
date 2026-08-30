"""
ps_gnn.analytics.explainability
==================================

Feature-level explainability for PS-GNN predictions, via SHAP
(SHapley Additive exPlanations).

Scope and honesty about the method
------------------------------------
A *complete* "GraphSHAP" (Duval & Malliaros, 2021) attributes a node's
prediction to both its own features **and** the contribution of every
other node/edge in its receptive field, by treating subgraph
inclusion/exclusion as the Shapley "players". That is a substantially
more involved (and expensive) method than wrapping ``shap``'s standard
:class:`~shap.KernelExplainer`.

This module implements the practical, still genuinely useful version
that fits the project's scope: for a target node, it holds the **graph
structure and every other node's features fixed**, and computes Shapley
values over perturbations of *only the target node's own 19-D feature
vector* (using a background distribution drawn from other nodes' feature
vectors, standard SHAP practice). This directly answers "which of this
pixel's own measured properties (ADI, coherence, land cover, ...) drove
the model's decision", which is the question domain scientists actually
ask when reviewing a candidate detection — it does not (and does not
claim to) attribute credit to specific *neighbors* or *edges*, which
would require the fuller graph-Shapley formulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ps_gnn.data.preprocessing import FEATURE_NAMES
from ps_gnn.models.ps_gnn import PSGNN

logger = logging.getLogger(__name__)


@dataclass
class NodeExplanation:
    """SHAP-based explanation for a single node's prediction.

    Attributes
    ----------
    node_id : int
        Index of the explained node.
    prediction : float
        Model's predicted PS-class probability for this node.
    base_value : float
        SHAP's expected model output over the background distribution
        (the "starting point" that feature contributions are added to).
    top_features : list[tuple[str, float]]
        The ``top_k`` features with the largest absolute SHAP value, as
        ``(feature_name, shap_value)``, sorted by |SHAP value| descending.
    all_shap_values : dict[str, float]
        SHAP value for every one of the 19 features, keyed by name.
    """

    node_id: int
    prediction: float
    base_value: float
    top_features: list[tuple[str, float]]
    all_shap_values: dict[str, float]


def _make_prediction_fn(
    model: PSGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_node: int,
    ps_class_index: int,
):
    """Build a ``feature_rows -> probability`` function for SHAP.

    The returned function replaces ``x[target_node]`` with each row of
    its input in turn, re-runs the full graph forward pass (all other
    nodes' features and the graph structure held fixed), and returns the
    target node's predicted PS-class probability for every row. This is
    the interface :class:`shap.KernelExplainer` expects.

    Device handling
    -----------------
    ``shap.KernelExplainer`` always works with plain CPU numpy arrays for
    its background distribution and perturbation samples, regardless of
    which device ``model`` lives on. The returned ``_predict`` function
    therefore explicitly moves each perturbed feature tensor to
    ``model.device`` before the forward pass, and moves the resulting
    probabilities back to CPU before extracting values — so this function
    works correctly whether ``model`` is on CPU or GPU, without relying
    on any particular device-handling behavior inside the model itself.

    Parameters
    ----------
    model : PSGNN
    x : Tensor, shape (N, 19)
    edge_index : Tensor, shape (2, E)
    target_node : int
    ps_class_index : int

    Returns
    -------
    callable
        ``f(feature_rows: np.ndarray, shape (n, 19)) -> np.ndarray, shape (n,)``
        Always returns a CPU numpy array, as required by
        :class:`shap.KernelExplainer`.
    """
    model.eval()

    def _predict(feature_rows: np.ndarray) -> np.ndarray:
        n_rows = feature_rows.shape[0]
        outputs = np.zeros(n_rows, dtype=np.float64)
        with torch.no_grad():
            for i in range(n_rows):
                x_perturbed = x.clone()
                x_perturbed[target_node] = torch.from_numpy(feature_rows[i].astype(np.float32))
                # SHAP's background/perturbation arrays are always plain CPU
                # numpy arrays, so both `x_perturbed` (built from `x` above)
                # and `edge_index` may be on CPU even when `model` lives on
                # a GPU. Move both explicitly to the model's device before
                # the forward pass, rather than relying on PSGNN.forward()'s
                # own internal `.to()` calls, so this function stays correct
                # even if that internal device handling ever changes.
                x_perturbed = x_perturbed.to(model.device)
                edge_index_on_device = edge_index.to(model.device)
                probs = model.predict_proba(x_perturbed, edge_index_on_device)
                # Move the result back to CPU before extracting values, so
                # the array returned to SHAP is always plain CPU/numpy data,
                # regardless of which device the model ran on.
                probs = probs.cpu()
                outputs[i] = probs[target_node, ps_class_index].item()
        return outputs

    return _predict


def explain_node(
    model: PSGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    target_node: int,
    feature_names: list[str] | None = None,
    n_background: int = 30,
    nsamples: int = 100,
    top_k: int = 5,
    random_state: int = 42,
) -> NodeExplanation:
    """Explain a single node's prediction via a SHAP KernelExplainer.

    Parameters
    ----------
    model : PSGNN
        A trained model.
    x : Tensor, shape (N, 19)
        Full node feature matrix (graph context for the target node).
    edge_index : Tensor, shape (2, E)
        Full graph connectivity.
    target_node : int
        Index of the node to explain.
    feature_names : list[str], optional
        Names for the 19 features. Defaults to
        :data:`ps_gnn.data.preprocessing.FEATURE_NAMES`.
    n_background : int, default 30
        Number of other nodes' feature vectors to use as the SHAP
        background distribution (randomly sampled, excluding
        ``target_node``).
    nsamples : int, default 100
        Number of coalition samples used by
        :class:`shap.KernelExplainer` (higher = more accurate, slower).
    top_k : int, default 5
        Number of top-magnitude features to report.
    random_state : int, default 42
        Seed for background sampling.

    Returns
    -------
    NodeExplanation

    Raises
    ------
    ImportError
        If ``shap`` is not installed.
    ValueError
        If ``target_node`` is out of range.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("explain_node requires the `shap` package: `pip install shap`.") from exc

    n_nodes = x.shape[0]
    if not (0 <= target_node < n_nodes):
        raise ValueError(f"target_node={target_node} out of range for {n_nodes} nodes")

    feature_names = feature_names or FEATURE_NAMES

    rng = np.random.default_rng(random_state)
    other_nodes = np.array([i for i in range(n_nodes) if i != target_node])
    background_idx = rng.choice(
        other_nodes, size=min(n_background, len(other_nodes)), replace=False
    )
    background = x[background_idx].numpy().astype(np.float64)

    predict_fn = _make_prediction_fn(model, x, edge_index, target_node, ps_class_index=1)

    explainer = shap.KernelExplainer(predict_fn, background, silent=True)
    target_features = x[target_node].numpy().reshape(1, -1).astype(np.float64)
    shap_values = explainer.shap_values(target_features, nsamples=nsamples, silent=True)
    shap_values = np.asarray(shap_values).reshape(-1)  # (19,)

    prediction = float(predict_fn(target_features)[0])
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    all_shap = {name: float(val) for name, val in zip(feature_names, shap_values)}
    order = np.argsort(-np.abs(shap_values))[:top_k]
    top_features = [(feature_names[i], float(shap_values[i])) for i in order]

    return NodeExplanation(
        node_id=target_node,
        prediction=prediction,
        base_value=base_value,
        top_features=top_features,
        all_shap_values=all_shap,
    )


def aggregate_global_feature_importance(
    model: PSGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_indices: list[int] | None = None,
    n_sample_nodes: int = 20,
    feature_names: list[str] | None = None,
    n_background: int = 30,
    nsamples: int = 100,
    random_state: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Aggregate per-node SHAP explanations into a global feature-importance summary.

    Parameters
    ----------
    model : PSGNN
    x : Tensor, shape (N, 19)
    edge_index : Tensor, shape (2, E)
    node_indices : list[int], optional
        Specific nodes to explain and aggregate over. If omitted,
        ``n_sample_nodes`` nodes are randomly sampled.
    n_sample_nodes : int, default 20
        Number of nodes to sample when ``node_indices`` is omitted.
    feature_names : list[str], optional
        See :func:`explain_node`.
    n_background, nsamples : int
        See :func:`explain_node`.
    random_state : int, default 42
    show_progress : bool, default True
        Show a tqdm progress bar over explained nodes.

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``mean_abs_shap``, ``mean_shap``,
        sorted by ``mean_abs_shap`` descending — the global importance
        ranking and the average signed effect direction of each feature.
    """
    feature_names = feature_names or FEATURE_NAMES
    n_nodes = x.shape[0]

    if node_indices is None:
        rng = np.random.default_rng(random_state)
        node_indices = rng.choice(
            n_nodes, size=min(n_sample_nodes, n_nodes), replace=False
        ).tolist()

    all_shap_rows = []
    iterator = tqdm(node_indices, desc="Explaining nodes") if show_progress else node_indices
    for node_id in iterator:
        explanation = explain_node(
            model,
            x,
            edge_index,
            node_id,
            feature_names=feature_names,
            n_background=n_background,
            nsamples=nsamples,
            random_state=random_state,
        )
        all_shap_rows.append(explanation.all_shap_values)

    shap_df = pd.DataFrame(all_shap_rows)  # (n_explained_nodes, 19)
    summary = pd.DataFrame(
        {
            "feature": shap_df.columns,
            "mean_abs_shap": shap_df.abs().mean(axis=0).values,
            "mean_shap": shap_df.mean(axis=0).values,
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    return summary