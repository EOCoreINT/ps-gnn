"""
synthetic_pipeline_utils
===========================
Shared, reusable helper functions for ps-gnn's synthetic stress-test
pipeline: realistic (non-degenerate) PS + false-neighbor injection,
borderline-ADI cluster injection, and the diagnostic functions used to
audit the resulting model (edge-role statistics, attention-by-edge-role,
a genuine classical ADI baseline, recall-by-role, and two hit/miss
diagnostics for investigating confidence gaps).

Extracted from ``run_synthetic_pipeline_hardened.py`` so both that script
and the companion Jupyter notebook (``ps_gnn_full_pipeline.ipynb``) share
a single, tested implementation rather than two copies that could drift
apart. No changes are made to the ``ps_gnn`` package itself -- everything
below uses only its existing public API.
"""

from __future__ import annotations

import numpy as np
import torch
from ps_gnn.data.label_generation import inject_synthetic_ps
from ps_gnn.models.ps_gnn import PSGNN

# Node role codes, used only within this script for diagnostics (not part
# of the ps_gnn package's schema).
ROLE_BACKGROUND = 0
ROLE_TRUE_PS = 1
ROLE_FALSE_NEIGHBOR = 2
ROLE_BORDERLINE_PS = 3


def inject_realistic_ps_with_false_neighbors(
    base_amplitude: np.ndarray,
    base_phase: np.ndarray,
    n_ps: int,
    min_separation_px: int,
    random_state: int,
    amplitude_percentile: float,
    amplitude_boost: float,
    amp_noise_std_frac: float = 0.03,
    phase_noise_std: float = 0.10,
    n_false_neighbors_per_ps: int = 2,
    false_neighbor_max_offset_px: int = 2,
    false_neighbor_adi: float = 0.5,
    false_neighbor_extra_phase_std: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject realistic (non-zero-variance) PS points plus adjacent 'false neighbors'.

    Starts from :func:`ps_gnn.data.label_generation.inject_synthetic_ps`
    (which places `n_ps` perfectly-stable points), then:

    1. Adds small, controlled Gaussian noise to those points' amplitude
       and phase, giving them realistic (low but non-zero) ADI and phase
       variance -- enough that :func:`ps_gnn.data.preprocessing.build_graph`
       can actually compute a meaningful phase correlation between them,
       rather than hitting the zero-variance degenerate case.
    2. For each true PS point, places `n_false_neighbors_per_ps` pixels
       within `false_neighbor_max_offset_px` pixels of it whose phase is
       the true PS's own phase series *plus* a small amount of additional
       noise (rather than a weighted blend -- see note below), but whose
       *amplitude* is genuinely unstable (moderate ADI). This models a
       specific, deceptive failure mode: a scatterer that is phase-coherent
       with (and therefore graph-connected to) a true PS -- e.g. via
       geometric coupling or layover -- but is not itself a stable
       reflector. A model relying on phase-correlation-driven proximity
       alone would be fooled by this edge; one that also attends to the
       full node feature vector (ADI, local texture, ...) should not be.

    Why additive noise, not a weighted blend
    -------------------------------------------
    An earlier version of this function built false-neighbor phase as
    ``blend * true_ps_phase + (1 - blend) * independent_noise``. This
    turned out to reliably produce *near-zero* correlation with the true
    PS even at ``blend=0.9``: Pearson correlation normalizes by each
    signal's own standard deviation, and a genuinely stable PS point has
    (by definition) a tiny phase standard deviation -- so any
    independent-noise component with variance comparable to or larger
    than that tiny value dominates the correlation estimate regardless of
    its blend weight. Adding a small amount of *absolute* extra noise on
    top of the true PS's own phase series (rather than mixing in an
    independent, full-range random signal) keeps the combined signal's
    variance in the same small regime as the true PS's own noise, which
    is what actually preserves a strong, edge-forming correlation.

    Parameters
    ----------
    base_amplitude, base_phase : np.ndarray, shape (T, H, W)
        Background SAR stack to inject into.
    n_ps : int
        Number of true PS points to place.
    min_separation_px : int
        Minimum spacing between true PS points.
    random_state : int
        Seed.
    amplitude_percentile, amplitude_boost : float
        Passed through to `inject_synthetic_ps` for the base PS amplitude.
    amp_noise_std_frac : float, default 0.03
        Relative amplitude noise (fraction of the base PS amplitude)
        added to true PS points, giving ADI ~= this value.
    phase_noise_std : float, default 0.10
        Phase noise (radians) added to true PS points.
    n_false_neighbors_per_ps : int, default 2
        Number of false-neighbor pixels planted next to each true PS.
    false_neighbor_max_offset_px : int, default 2
        Max pixel offset (Chebyshev) for false-neighbor placement.
    false_neighbor_adi : float, default 0.5
        Target ADI for false-neighbor pixels (moderate amplitude
        instability -- well above the classical ADI<0.25 PS threshold).
    false_neighbor_extra_phase_std : float, default 0.15
        Extra phase noise (radians), added on top of the true PS's own
        phase series, for each false neighbor. Calibrated (see module
        tests) to keep phase correlation with the true PS comfortably
        above typical graph-construction thresholds (~30% of pairs clear
        0.4 correlation, vs. ~1.5% for unrelated background pairs at the
        same threshold) while remaining a distinct, noisier signal.

    Returns
    -------
    amplitude, phase : np.ndarray, shape (T, H, W)
        Modified stacks.
    node_roles : np.ndarray, shape (H, W), dtype uint8
        ``ROLE_BACKGROUND`` (0), ``ROLE_TRUE_PS`` (1), or
        ``ROLE_FALSE_NEIGHBOR`` (2) for every pixel.
    group_id : np.ndarray, shape (H, W), dtype int32
        Each true-PS point and its own planted false neighbors share a
        unique, non-negative integer ID; all other pixels are ``-1``. Lets
        downstream diagnostics distinguish "this pixel's own by-design
        neighbor" from "an unrelated point's neighbor happens to be close
        by" -- the former is expected and uninformative, the latter is
        genuine incidental spatial contamination.
    """
    t_steps, height, width = base_amplitude.shape
    rng = np.random.default_rng(random_state)

    base_result = inject_synthetic_ps(
        base_amplitude,
        base_phase,
        n_ps=n_ps,
        min_separation_px=min_separation_px,
        random_state=random_state,
        amplitude_percentile=amplitude_percentile,
        amplitude_boost=amplitude_boost,
    )
    amplitude = base_result.amplitude.copy()
    phase = base_result.phase.copy()
    node_roles = np.zeros((height, width), dtype=np.uint8)
    group_id = np.full((height, width), -1, dtype=np.int32)

    ps_rows, ps_cols = np.nonzero(base_result.labels)
    occupied = {(int(r), int(c)) for r, c in zip(ps_rows, ps_cols)}

    for group_counter, (r, c) in enumerate(zip(ps_rows, ps_cols)):
        r, c = int(r), int(c)
        node_roles[r, c] = ROLE_TRUE_PS
        group_id[r, c] = group_counter

        # Add realistic (non-zero) noise so this point isn't a mathematically
        # degenerate zero-variance signal -- this is what lets it actually
        # form graph edges.
        base_high_amplitude = float(amplitude[0, r, c])  # constant across T pre-noise
        amp_noise = rng.normal(0.0, amp_noise_std_frac * base_high_amplitude, size=t_steps)
        amplitude[:, r, c] = np.clip(base_high_amplitude + amp_noise, a_min=1e-3, a_max=None)

        phase_noise = rng.normal(0.0, phase_noise_std, size=t_steps)
        phase[:, r, c] = ((phase_noise + np.pi) % (2 * np.pi)) - np.pi

        true_ps_phase_series = phase[:, r, c].copy()

        placed = 0
        attempts = 0
        while placed < n_false_neighbors_per_ps and attempts < 20:
            attempts += 1
            dr = int(rng.integers(-false_neighbor_max_offset_px, false_neighbor_max_offset_px + 1))
            dc = int(rng.integers(-false_neighbor_max_offset_px, false_neighbor_max_offset_px + 1))
            nr, nc = r + dr, c + dc
            if dr == 0 and dc == 0:
                continue
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in occupied:
                continue

            # Vegetation-like scatterer: moderate amplitude, high fractional
            # noise (target ADI ~= false_neighbor_adi).
            base_amp_here = float(np.mean(base_amplitude[:, nr, nc]))
            fn_amp = base_amp_here * (1.0 + rng.normal(0.0, false_neighbor_adi, size=t_steps))
            amplitude[:, nr, nc] = np.clip(fn_amp, a_min=1e-3, a_max=None)

            # Phase: the true PS's own (already-noisy) phase series, plus a
            # small amount of *additional* noise -- see the "Why additive
            # noise" note in this function's docstring for why a weighted
            # blend with independent noise does not work here.
            extra_noise = rng.normal(0.0, false_neighbor_extra_phase_std, size=t_steps)
            fn_phase = true_ps_phase_series + extra_noise
            phase[:, nr, nc] = ((fn_phase + np.pi) % (2 * np.pi)) - np.pi

            node_roles[nr, nc] = ROLE_FALSE_NEIGHBOR
            group_id[nr, nc] = group_counter
            occupied.add((nr, nc))
            placed += 1

    return amplitude, phase, node_roles, group_id


def inject_borderline_ps_clusters(
    amplitude: np.ndarray,
    phase: np.ndarray,
    node_roles: np.ndarray,
    group_id: np.ndarray,
    n_clusters: int,
    cluster_size: int,
    random_state: int,
    min_cluster_separation_px: int = 8,
    member_max_offset_px: int = 2,
    borderline_amp_noise_std: float = 0.30,
    cluster_shared_phase_std: float = 0.15,
    member_individual_phase_std: float = 0.05,
    amplitude_percentile: float = 90.0,
    amplitude_boost: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inject clusters of genuine-but-noisy 'borderline' PS points.

    This is the core scientific test the rest of this script's design was
    missing: every true PS point injected so far has near-zero ADI (~0.03),
    which a naive fixed classical threshold (ADI < 0.25) recovers just as
    well as PS-GNN does -- there was never a case where graph reasoning
    was actually *necessary*. This function creates that case.

    Each cluster is a small group of ``cluster_size`` genuinely real
    scatterers (labeled PS, ``y=1``) placed close together, whose:

    - **Amplitude** is calibrated (``borderline_amp_noise_std=0.30``) to
      give ADI ~0.29 on average -- comfortably *above* the classical 0.25
      cutoff, so a fixed-threshold baseline will usually reject them as
      false negatives.
    - **Phase** is built from one shared, cluster-level noise series (std
      ``cluster_shared_phase_std``) plus a small amount of independent
      noise per member (std ``member_individual_phase_std``) -- calibrated
      so that within-cluster pairs clear the graph's phase-correlation
      threshold ~99% of the time, giving the GAT genuine, informative
      same-label edges to learn from. This models real corroborating
      evidence: physically-plausible clustered structures (e.g. multiple
      reflectors on the same building) whose amplitude is individually
      too noisy to pass a naive threshold, but whose *mutual* phase
      consistency is a real, learnable signal that they belong together.

    A model with no mechanism for using spatial/graph context has no way
    to recover these points beyond what ADI alone tells it. PS-GNN, via
    message passing and the physics-informed spatial-coherence loss term,
    has a mechanism to use the cluster's mutual corroboration -- whether
    it actually learns to use it is exactly what this experiment measures.

    Parameters
    ----------
    amplitude, phase : np.ndarray, shape (T, H, W)
    node_roles : np.ndarray, shape (H, W), dtype uint8
        Existing role grid (mutated positions are skipped for collision
        avoidance).
    group_id : np.ndarray, shape (H, W), dtype int32
        Existing group-ID grid from
        :func:`inject_realistic_ps_with_false_neighbors`; cluster IDs are
        assigned starting from a large offset to avoid colliding with it.
    n_clusters : int
        Number of borderline clusters to place.
    cluster_size : int
        Number of pixels per cluster.
    random_state : int
        Seed.
    min_cluster_separation_px : int, default 8
        Minimum spacing between cluster seed points.
    member_max_offset_px : int, default 2
        Max pixel offset (Chebyshev) of other cluster members from the
        cluster's seed pixel.
    borderline_amp_noise_std, cluster_shared_phase_std,
    member_individual_phase_std : float
        See the calibration note above.
    amplitude_percentile, amplitude_boost : float
        Base "high amplitude" definition, matching
        :func:`inject_realistic_ps_with_false_neighbors` so borderline
        points are equally bright -- only their *stability* differs.

    Returns
    -------
    amplitude, phase : np.ndarray, shape (T, H, W)
        Modified stacks.
    node_roles : np.ndarray, shape (H, W), dtype uint8
        Updated in place at newly-injected pixels with ``ROLE_BORDERLINE_PS``.
    group_id : np.ndarray, shape (H, W), dtype int32
        Updated copy of the input ``group_id`` grid: every member of a
        given cluster shares a unique ID (offset well above any PS-group
        ID from :func:`inject_realistic_ps_with_false_neighbors` to avoid
        collisions), so a downstream diagnostic can exclude "this point's
        own cluster-mates" from a spatial-contamination distance query.
    """
    t_steps, height, width = amplitude.shape
    rng = np.random.default_rng(random_state)
    amplitude = amplitude.copy()
    phase = phase.copy()
    node_roles = node_roles.copy()
    group_id = group_id.copy()

    base_high_amplitude = float(np.percentile(amplitude, amplitude_percentile)) * amplitude_boost
    occupied = {(int(r), int(c)) for r, c in zip(*np.nonzero(node_roles))}
    cluster_seeds: list[tuple[int, int]] = []
    cluster_group_offset = 1_000_000  # guaranteed above any PS-group id

    n_placed_clusters = 0
    seed_attempts = 0
    while n_placed_clusters < n_clusters and seed_attempts < n_clusters * 50:
        seed_attempts += 1
        sr = int(rng.integers(member_max_offset_px, height - member_max_offset_px))
        sc = int(rng.integers(member_max_offset_px, width - member_max_offset_px))
        if (sr, sc) in occupied:
            continue
        if any(
            max(abs(sr - pr), abs(sc - pc)) < min_cluster_separation_px for pr, pc in cluster_seeds
        ):
            continue

        # Collect cluster_size member positions (seed + neighbors).
        members = [(sr, sc)]
        member_attempts = 0
        while len(members) < cluster_size and member_attempts < 30:
            member_attempts += 1
            dr = int(rng.integers(-member_max_offset_px, member_max_offset_px + 1))
            dc = int(rng.integers(-member_max_offset_px, member_max_offset_px + 1))
            mr, mc = sr + dr, sc + dc
            if (dr, dc) == (0, 0):
                continue
            if not (0 <= mr < height and 0 <= mc < width):
                continue
            if (mr, mc) in occupied or (mr, mc) in members:
                continue
            members.append((mr, mc))

        if len(members) < cluster_size:
            continue  # couldn't fit a full cluster here; try a different seed

        # One shared phase series for the whole cluster, plus small
        # independent noise per member -- see calibration note above.
        shared_phase = rng.normal(0.0, cluster_shared_phase_std, size=t_steps)
        this_cluster_group_id = cluster_group_offset + n_placed_clusters
        for mr, mc in members:
            amp_noise = rng.normal(0.0, borderline_amp_noise_std, size=t_steps)
            amplitude[:, mr, mc] = np.clip(
                base_high_amplitude * (1.0 + amp_noise), a_min=1e-3, a_max=None
            )

            individual_noise = rng.normal(0.0, member_individual_phase_std, size=t_steps)
            member_phase = shared_phase + individual_noise
            phase[:, mr, mc] = ((member_phase + np.pi) % (2 * np.pi)) - np.pi

            node_roles[mr, mc] = ROLE_BORDERLINE_PS
            group_id[mr, mc] = this_cluster_group_id
            occupied.add((mr, mc))

        cluster_seeds.append((sr, sc))
        n_placed_clusters += 1

    return amplitude, phase, node_roles, group_id


def compute_edge_role_statistics(edge_index: np.ndarray, node_roles: np.ndarray) -> dict[str, int]:
    """Categorize graph edges by the roles of their endpoints, for diagnostics.

    Parameters
    ----------
    edge_index : np.ndarray, shape (2, E)
    node_roles : np.ndarray, shape (H, W), dtype uint8
        Flattened internally to align with node indices.

    Returns
    -------
    dict[str, int]
        Counts of undirected edge pairs by category:
        ``ps_ps``, ``ps_false_neighbor``, ``ps_background``,
        ``false_neighbor_false_neighbor``, ``false_neighbor_background``,
        ``background_background``.
    """
    roles_flat = node_roles.ravel()
    src, dst = edge_index[0], edge_index[1]
    keep = src < dst  # count each undirected pair once
    src, dst = src[keep], dst[keep]
    src_roles, dst_roles = roles_flat[src], roles_flat[dst]

    pair_roles = np.stack(
        [np.minimum(src_roles, dst_roles), np.maximum(src_roles, dst_roles)], axis=1
    )

    def _count(role_a: int, role_b: int) -> int:
        return int(np.sum((pair_roles[:, 0] == role_a) & (pair_roles[:, 1] == role_b)))

    return {
        "ps_ps": _count(ROLE_TRUE_PS, ROLE_TRUE_PS),
        "ps_false_neighbor": _count(ROLE_TRUE_PS, ROLE_FALSE_NEIGHBOR),
        "ps_background": _count(ROLE_TRUE_PS, ROLE_BACKGROUND),
        "false_neighbor_false_neighbor": _count(ROLE_FALSE_NEIGHBOR, ROLE_FALSE_NEIGHBOR),
        "false_neighbor_background": _count(ROLE_FALSE_NEIGHBOR, ROLE_BACKGROUND),
        "background_background": _count(ROLE_BACKGROUND, ROLE_BACKGROUND),
        "borderline_borderline": _count(ROLE_BORDERLINE_PS, ROLE_BORDERLINE_PS),
        "borderline_background": _count(ROLE_BORDERLINE_PS, ROLE_BACKGROUND),
    }


def compute_attention_by_edge_role(
    model: PSGNN, x: torch.Tensor, edge_index: torch.Tensor, node_roles: np.ndarray
) -> dict[str, float]:
    """Run one forward pass and average GAT attention weight by edge role category.

    This is the key mechanistic check for the project's central hypothesis:
    a model that has actually learned to identify stable scatterers should
    assign *lower* mean attention to ``ps_false_neighbor`` edges (a true PS
    connected to a deliberately-planted unstable, superficially-similar
    neighbor) than to ``ps_ps`` edges (two genuinely stable points) or
    edges among consistently noisy background.

    Parameters
    ----------
    model : PSGNN
    x : Tensor, shape (N, 19)
    edge_index : Tensor, shape (2, E)
    node_roles : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    dict[str, float]
        Mean attention weight per edge-role category present in the graph
        (categories with zero edges are omitted).
    """
    model.eval()
    with torch.no_grad():
        hidden = model.conv1(model.encoder(x.to(model.device)), edge_index.to(model.device))
        hidden = model.bn1(hidden).relu()
        _, (attn_edge_index, attn_weights) = model.conv2(
            hidden, edge_index.to(model.device), return_attention_weights=True
        )
        attn_mean = attn_weights.mean(dim=-1).cpu().numpy()

    aei = attn_edge_index.cpu().numpy()
    roles_flat = node_roles.ravel()
    src, dst = aei[0], aei[1]

    non_self_loop = src != dst
    src, dst, attn_mean = src[non_self_loop], dst[non_self_loop], attn_mean[non_self_loop]
    src_roles, dst_roles = roles_flat[src], roles_flat[dst]

    categories = {
        "ps_ps": (ROLE_TRUE_PS, ROLE_TRUE_PS),
        "ps_false_neighbor": (ROLE_TRUE_PS, ROLE_FALSE_NEIGHBOR),
        "ps_background": (ROLE_TRUE_PS, ROLE_BACKGROUND),
        "background_background": (ROLE_BACKGROUND, ROLE_BACKGROUND),
    }

    result = {}
    for name, (role_a, role_b) in categories.items():
        mask = ((src_roles == role_a) & (dst_roles == role_b)) | (
            (src_roles == role_b) & (dst_roles == role_a)
        )
        if mask.sum() > 0:
            result[name] = float(attn_mean[mask].mean())
    return result


def build_adi_baseline_mask(features: np.ndarray, adi_threshold: float = 0.25) -> np.ndarray:
    """A genuine classical baseline: a fixed ADI cutoff (the StaMPS-style convention).

    Earlier versions of this script used a "top-K lowest ADI, K matched to
    the GNN's detection count" baseline. That framing is convenient but
    artificial: it doesn't reflect how classical PSI selection actually
    works (a fixed cutoff, independent of how many points any other method
    finds), and it can't reveal cases where PS-GNN correctly recovers
    genuinely-real but borderline-ADI points (see
    :func:`inject_borderline_ps_clusters`) that a fixed threshold would
    reject outright. A fixed threshold is the fairer, more realistic, and
    more informative comparison.

    Parameters
    ----------
    features : np.ndarray, shape (19, H, W)
        Node feature stack; index 2 is ADI (see
        :data:`ps_gnn.data.preprocessing.FEATURE_NAMES`).
    adi_threshold : float, default 0.25
        The classical PSI ADI cutoff (StaMPS convention).

    Returns
    -------
    np.ndarray, shape (H * W,), dtype bool
    """
    adi_flat = features[2].ravel()
    return adi_flat < adi_threshold


def compute_recall_by_role(
    detection_mask: np.ndarray, node_roles: np.ndarray, roles: dict[str, int]
) -> dict[str, float]:
    """Compute detection recall separately for each named ground-truth role.

    This is the key comparison for the borderline-ADI experiment: does
    PS-GNN recover a meaningfully higher fraction of the deliberately
    hard, borderline-ADI cluster points than a fixed classical ADI
    threshold does, while both still recover the easy points near-perfectly?

    Parameters
    ----------
    detection_mask : np.ndarray, shape (H * W,) or (H, W), dtype bool
        A method's detection mask (flattened internally).
    node_roles : np.ndarray, shape (H, W), dtype uint8
    roles : dict[str, int]
        Mapping of ``{display_name: role_code}`` to report recall for.

    Returns
    -------
    dict[str, float]
        Recall (fraction of that role's pixels present in
        ``detection_mask``) per named role. ``nan`` if a role has zero
        ground-truth pixels.
    """
    detection_flat = detection_mask.ravel()
    roles_flat = node_roles.ravel()
    result = {}
    for name, role_code in roles.items():
        role_mask = roles_flat == role_code
        n_role = int(role_mask.sum())
        if n_role == 0:
            result[name] = float("nan")
            continue
        result[name] = float((detection_flat & role_mask).sum() / n_role)
    return result


def diagnose_spatial_contamination(
    node_roles: np.ndarray,
    group_id: np.ndarray,
    probability: np.ndarray,
    target_role: int,
    contaminant_roles: list[int],
    threshold: float,
    pixel_spacing_m: float,
    k_query: int = 20,
) -> dict[str, dict[str, float]]:
    """Test whether missed detections sit closer to *unrelated* 'contaminant' pixels than hits do.

    Hypothesis under test: :func:`ps_gnn.data.preprocessing.compute_node_features`
    derives ``local_var`` from a 5x5-pixel window and ``edge_strength``
    from a 3x3 Sobel kernel around every pixel. A target-role pixel that
    happens to fall within that window of a spectrally "contaminating"
    pixel (e.g. a false neighbor or a borderline-ADI point) would have
    those two features inflated by the contaminant's presence, even
    though the target pixel's *own* ADI is unaffected -- this could
    plausibly explain lower model confidence for otherwise "easy" points
    with no other distinguishing weakness.

    Group-ID exclusion (why this version differs from a naive distance
    transform)
    -------------------------------------------------------------------------
    A true-PS point's *own* planted false neighbors sit within 1-2px of it
    by design (see :func:`inject_realistic_ps_with_false_neighbors`), and
    every borderline point's *own* cluster-mates sit within a few px of it
    by design too (see :func:`inject_borderline_ps_clusters`). A plain
    "distance to nearest contaminant pixel" query is therefore dominated
    by this expected, by-design proximity for essentially every target
    pixel regardless of hit/miss status, masking any genuine signal from
    *incidental* contamination by an unrelated point's neighbors. This
    version uses ``group_id`` to explicitly exclude a target pixel's own
    group from its nearest-contaminant query, isolating the effect this
    diagnostic is actually meant to measure.

    Parameters
    ----------
    node_roles : np.ndarray, shape (H, W), dtype uint8
    group_id : np.ndarray, shape (H, W), dtype int32
        From :func:`inject_realistic_ps_with_false_neighbors` /
        :func:`inject_borderline_ps_clusters`; ``-1`` for ungrouped pixels.
    probability : np.ndarray, shape (H, W)
        Mean MC-Dropout PS-class probability.
    target_role : int
        Role to analyze (e.g. ``ROLE_TRUE_PS``).
    contaminant_roles : list[int]
        Roles whose proximity is hypothesized to hurt ``target_role``'s
        confidence (e.g. ``[ROLE_FALSE_NEIGHBOR, ROLE_BORDERLINE_PS]``).
    threshold : float
        Confidence threshold defining hit vs. miss.
    pixel_spacing_m : float
        Ground sampling distance (unused in the current distance-in-pixels
        reporting, kept for API clarity/future use).
    k_query : int, default 20
        Number of nearest contaminant candidates to fetch per target pixel
        before filtering out same-group ones; increase if a target's
        entire k-nearest-neighborhood is ever exhausted by its own group
        (rare, but possible for very large groups).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"hit": {...}, "miss": {...}}``, each with ``count``,
        ``mean_distance_px``, ``median_distance_px``,
        ``frac_within_2px`` (inside the 5x5 local_var window),
        ``frac_within_1px`` (inside the 3x3 Sobel kernel),
        ``frac_exhausted`` (fraction where all ``k_query`` nearest
        candidates were same-group, i.e. no valid distance could be
        computed -- a non-zero value here means ``k_query`` should be
        increased).
    """
    from scipy.spatial import cKDTree

    del pixel_spacing_m  # currently unused; kept for API symmetry/future use

    contaminant_mask = np.isin(node_roles, contaminant_roles)
    c_rows, c_cols = np.nonzero(contaminant_mask)
    if len(c_rows) == 0:
        raise ValueError("No contaminant pixels found for the given contaminant_roles.")
    c_group_ids = group_id[c_rows, c_cols]
    c_coords = np.stack([c_rows, c_cols], axis=1).astype(np.float64)
    tree = cKDTree(c_coords)

    target_mask = node_roles == target_role
    target_rows, target_cols = np.nonzero(target_mask)
    target_probs = probability[target_rows, target_cols]
    target_group_ids = group_id[target_rows, target_cols]
    target_coords = np.stack([target_rows, target_cols], axis=1).astype(np.float64)

    k = min(k_query, len(c_rows))
    query_dists, query_idxs = tree.query(target_coords, k=k)
    # cKDTree.query returns 1D arrays when k=1 (not (n, 1) as np.atleast_2d
    # would produce -- atleast_2d would incorrectly give shape (1, n)).
    # Reshape explicitly so downstream indexing is always (n_targets, k).
    if query_dists.ndim == 1:
        query_dists = query_dists[:, None]
        query_idxs = query_idxs[:, None]

    target_dists = np.full(len(target_rows), np.nan)
    exhausted = np.zeros(len(target_rows), dtype=bool)
    for i in range(len(target_rows)):
        neighbor_groups = c_group_ids[query_idxs[i]]
        valid = neighbor_groups != target_group_ids[i]
        if valid.any():
            target_dists[i] = query_dists[i][valid][0]  # cKDTree returns results sorted ascending
        else:
            exhausted[i] = True

    hit_mask = target_probs >= threshold

    def _stats(mask: np.ndarray) -> dict[str, float]:
        n_total = int(mask.sum())
        frac_exhausted = float(exhausted[mask].mean()) if n_total else float("nan")
        valid_mask = mask & ~exhausted
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return {
                "count": 0,
                "mean_distance_px": float("nan"),
                "median_distance_px": float("nan"),
                "frac_within_2px": float("nan"),
                "frac_within_1px": float("nan"),
                "frac_exhausted": frac_exhausted,
            }
        d = target_dists[valid_mask]
        return {
            "count": n_valid,
            "mean_distance_px": float(d.mean()),
            "median_distance_px": float(np.median(d)),
            "frac_within_2px": float((d <= 2.0).mean()),
            "frac_within_1px": float((d <= 1.0).mean()),
            "frac_exhausted": frac_exhausted,
        }

    return {"hit": _stats(hit_mask), "miss": _stats(~hit_mask)}


def diagnose_confidence_by_hit_miss(
    probability: np.ndarray,
    uncertainty: np.ndarray,
    node_roles: np.ndarray,
    role_code: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Break down MC-Dropout probability/uncertainty for a role's hits vs. misses.

    Used to test *why* recall for a given role is below 100% -- e.g. is a
    "missed" point missing because the model's mean predicted probability
    is genuinely low (the model doesn't think it's PS), or because the
    mean probability is close to (just under) the confidence threshold and
    MC-Dropout variance pushed it under on this particular draw?

    Parameters
    ----------
    probability : np.ndarray, shape (H, W)
        Mean MC-Dropout PS-class probability (see
        :func:`ps_gnn.inference.detector.mc_dropout_predict`).
    uncertainty : np.ndarray, shape (H, W)
        MC-Dropout variance.
    node_roles : np.ndarray, shape (H, W), dtype uint8
    role_code : int
        Which role to analyze (e.g. ``ROLE_TRUE_PS``).
    threshold : float
        The confidence threshold used for detection.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"hit": {...}, "miss": {...}}``, each with ``count``,
        ``mean_probability``, ``median_probability``, ``mean_uncertainty``.
    """
    role_mask = node_roles == role_code
    probs = probability[role_mask]
    uncerts = uncertainty[role_mask]
    hit_mask = probs >= threshold

    def _stats(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return {
                "count": 0,
                "mean_probability": float("nan"),
                "median_probability": float("nan"),
                "mean_uncertainty": float("nan"),
            }
        return {
            "count": int(mask.sum()),
            "mean_probability": float(probs[mask].mean()),
            "median_probability": float(np.median(probs[mask])),
            "mean_uncertainty": float(uncerts[mask].mean()),
        }

    return {"hit": _stats(hit_mask), "miss": _stats(~hit_mask)}
