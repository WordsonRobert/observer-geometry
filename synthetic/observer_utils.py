"""
observer_utils.py
Utilities for freezing an observation ball B(v,r), computing classical centralities,
computing MULTIPLE mu/Phi variants on a fixed embedding (VECTORIZED), computing rich
hidden-region descriptors (before/after), and observer-level metadata.

base_layout / place_new_node provide a cheap Euclidean layout used only by the stability
panels' fallback path; the intervention benchmark embeds in the Poincare ball and
re-embeds after each intervention (reembed.py).

VECTORIZATION NOTE:
The centrality metrics (betweenness, eigenvector, pagerank, closeness) are networkx
graph algorithms with no vector/matrix form -- they dominate runtime and cannot be
vectorized away. The one part of this pipeline that IS pure numeric computation over
arrays is the mu/Phi kernel (distance + exponential decay over the hidden set), so
that part is fully vectorized below: one numpy broadcast per observer instead of a
per-node Python loop, with the distance array computed once and reused across both
delta variants.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import numpy as np
import networkx as nx

DELTA_DEFAULT = 1.0
DELTA_ALT = 2.0  # second delta used for the "weighted Phi" variant


def freeze_ball(G, v, r):
    """Return the frozen node set B(v,r) and its induced edge set. Never touch these again."""
    ball_nodes = set(nx.single_source_shortest_path_length(G, v, cutoff=r).keys())
    ball_edges = set(frozenset(e) for e in G.subgraph(ball_nodes).edges())
    return ball_nodes, ball_edges


def base_layout(G, seed=0):
    """Cheap Euclidean spring layout (fallback path; the benchmark itself uses reembed.py)."""
    pos = nx.spring_layout(G, seed=seed, dim=2)
    return {n: np.array(p) for n, p in pos.items()}


def place_new_node(coords, G, new_node, rng):
    """Approximate placement for a newly-added hidden node: mean of hidden neighbors + jitter."""
    neighbors = [nb for nb in G.neighbors(new_node) if nb in coords]
    if neighbors:
        base = np.mean([coords[nb] for nb in neighbors], axis=0)
    else:
        base = np.mean(list(coords.values()), axis=0)
    jitter = rng.normal(0, 0.05, size=base.shape)
    return base + jitter


def euclidean_dist(coords, u, v):
    return float(np.linalg.norm(coords[u] - coords[v]))


def _vectorized_hidden_distances(coords, v, hidden_nodes):
    """
    One numpy broadcast computing distance from observer v to every hidden node at once.
    Returns (hidden_list, dists) where hidden_list is the ordered node id list matching
    the dists array position-for-position (nodes missing from coords are dropped).
    """
    hidden_list = [u for u in hidden_nodes if u in coords]
    if not hidden_list:
        return [], np.array([])
    coord_matrix = np.stack([coords[u] for u in hidden_list])  # (n_hidden, dim)
    v_coord = coords[v]
    diffs = coord_matrix - v_coord  # broadcast subtraction
    dists = np.linalg.norm(diffs, axis=1)  # vectorized norm, one call for all hidden nodes
    return hidden_list, dists


def compute_phi_variants(G, coords, v, hidden_nodes):
    """
    Returns several descriptor variants, all derived from ONE vectorized distance
    computation (dists computed once, reused for both decay rates):
      - mu / Phi:       raw sum, delta = DELTA_DEFAULT
      - normalized_Phi: mu / |hidden_nodes|
      - weighted_Phi_delta2: raw sum at an alternate decay rate DELTA_ALT
      - raw_shell_weight_{mean,std,max}: summary stats of the per-node kernel weights
    """
    hidden_list, dists = _vectorized_hidden_distances(coords, v, hidden_nodes)

    if len(hidden_list) == 0:
        return {
            "mu": 0.0, "Phi": 0.0, "normalized_Phi": 0.0,
            "weighted_Phi_delta2": 0.0,
            "raw_shell_weight_mean": 0.0, "raw_shell_weight_std": 0.0,
            "raw_shell_weight_max": 0.0,
        }

    weights_default = np.exp(-DELTA_DEFAULT * dists)  # vectorized, one call for all nodes
    weights_alt = np.exp(-DELTA_ALT * dists)

    mu_val = float(weights_default.sum())
    weighted_val = float(weights_alt.sum())
    n_hidden = len(hidden_list)
    normalized = mu_val / n_hidden if n_hidden > 0 else 0.0

    return {
        "mu": mu_val,
        "Phi": mu_val,
        "normalized_Phi": normalized,
        "weighted_Phi_delta2": weighted_val,
        "raw_shell_weight_mean": float(weights_default.mean()),
        "raw_shell_weight_std": float(weights_default.std()),
        "raw_shell_weight_max": float(weights_default.max()),
    }


def safe_eigenvector_centrality(G, max_iter=500):
    try:
        return nx.eigenvector_centrality(G, max_iter=max_iter)
    except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution):
        return {n: float("nan") for n in G.nodes()}


def compute_classical_metrics(G, v, betweenness_k=None):
    """
    Returns a dict of {metric_name: value at observer v}.
    NOTE: Betweenness and Eigenvector are graph algorithms -- they cannot be vectorized
    and are the dominant cost here. Set betweenness_k=<int> to use networkx's sampled
    approximate betweenness (k random source nodes instead of all-pairs) if runtime is
    the binding constraint; leave None for the exact (slower) computation.
    """
    degree = dict(G.degree())
    pagerank = nx.pagerank(G)
    n_nodes = G.number_of_nodes()
    k_eff = None if betweenness_k is None else max(1, min(betweenness_k, n_nodes))
    betweenness = nx.betweenness_centrality(G, k=k_eff, seed=0 if k_eff else None)
    eigenvector = safe_eigenvector_centrality(G)
    closeness = nx.closeness_centrality(G)

    return {
        "Degree": float(degree[v]),
        "PageRank": float(pagerank[v]),
        "Betweenness": float(betweenness[v]),
        "Eigenvector": float(eigenvector[v]),
        "Closeness": float(closeness[v]),
    }


def hidden_change_magnitude(nodes_before, edges_before, nodes_after, edges_after):
    """Graph-edit-distance style proxy: nodes added/removed + edges added/removed, hidden-region only."""
    node_diff = len(nodes_before.symmetric_difference(nodes_after))
    edge_diff = len(edges_before.symmetric_difference(edges_after))
    return node_diff + edge_diff


def compute_hidden_descriptors(G, hidden_nodes):
    """
    Rich descriptors of the induced hidden-region subgraph. Computed BEFORE and AFTER
    every intervention so downstream analysis can ask "which hidden property drives Phi"
    rather than relying on the intervention label alone.
    """
    hidden_nodes = set(hidden_nodes)
    H = G.subgraph(hidden_nodes).copy()
    n = H.number_of_nodes()
    m = H.number_of_edges()

    if n == 0:
        return {
            "hidden_nodes": 0, "hidden_edges": 0, "hidden_density": 0.0,
            "hidden_components": 0, "hidden_diameter": None,
            "hidden_average_degree": 0.0, "hidden_clustering": 0.0,
            "hidden_assortativity": None, "hidden_average_shortest_path": None,
            "hidden_triangles": 0, "hidden_modularity": None, "hidden_entropy": 0.0,
        }

    degrees = [d for _, d in H.degree()]
    n_components = nx.number_connected_components(H)

    if n_components > 0:
        largest_cc_nodes = max(nx.connected_components(H), key=len)
        H_lcc = H.subgraph(largest_cc_nodes)
        try:
            diameter = nx.diameter(H_lcc) if H_lcc.number_of_nodes() < 1000 else None
        except nx.NetworkXError:
            diameter = None
        try:
            avg_sp = (nx.average_shortest_path_length(H_lcc)
                      if H_lcc.number_of_nodes() > 1 and H_lcc.number_of_nodes() < 1000 else None)
        except nx.NetworkXError:
            avg_sp = None
    else:
        diameter, avg_sp = None, None

    try:
        assort = nx.degree_assortativity_coefficient(H)
        if assort is not None and np.isnan(assort):
            assort = None
    except (nx.NetworkXError, ZeroDivisionError):
        assort = None

    triangles = sum(nx.triangles(H).values()) // 3

    try:
        communities = nx.algorithms.community.greedy_modularity_communities(H)
        modularity = nx.algorithms.community.quality.modularity(H, communities)
    except Exception:
        modularity = None

    if degrees:
        vals, counts = np.unique(degrees, return_counts=True)
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p)).sum())
    else:
        entropy = 0.0

    return {
        "hidden_nodes": n,
        "hidden_edges": m,
        "hidden_density": nx.density(H),
        "hidden_components": n_components,
        "hidden_diameter": diameter,
        "hidden_average_degree": float(np.mean(degrees)) if degrees else 0.0,
        "hidden_clustering": nx.average_clustering(H) if n > 0 else 0.0,
        "hidden_assortativity": assort,
        "hidden_average_shortest_path": avg_sp,
        "hidden_triangles": int(triangles),
        "hidden_modularity": modularity,
        "hidden_entropy": entropy,
    }


def compute_observer_metadata(G, v, ball_nodes, ball_edges, hidden_nodes, r):
    """Per-observer covariates, captured once (these never change since the ball is frozen)."""
    shell_counts = {}
    lengths = nx.single_source_shortest_path_length(G, v, cutoff=r)
    for node, dist in lengths.items():
        shell_counts[dist] = shell_counts.get(dist, 0) + 1

    return {
        "observer": int(v),
        "observer_degree": int(G.degree(v)),
        "observer_radius": r,
        "ball_size": len(ball_nodes),
        "ball_edges": len(ball_edges),
        "hidden_size": len(hidden_nodes),
        "observer_shell_counts": {str(k): v_ for k, v_ in shell_counts.items()},
    }
