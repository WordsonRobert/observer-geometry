"""
graph_families.py
Five synthetic graph family generators used in the Hidden Region Intervention Benchmark,
plus a graph-level metadata function (n, m, density, diameter, clustering, etc.) so every
raw record can carry full provenance about the base graph it came from.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import numpy as np
import networkx as nx


def balanced_tree_graph(seed, branching=3, depth=5):
    rng = np.random.RandomState(seed)
    b = rng.choice([2, 3])  # capped: b=4 with d=6 gives ~5461 nodes, a runtime long-tail risk
    d = rng.choice([4, 5])  # max size now (3^6-1)/2 = 364, typical case ~121 nodes
    G = nx.balanced_tree(int(b), int(d))
    return nx.convert_node_labels_to_integers(G)


def random_tree_graph(seed, n_nodes=250):
    rng = np.random.RandomState(seed)
    n = int(rng.randint(150, 350))
    try:
        G = nx.random_labeled_tree(n, seed=seed)  # networkx >= 3.4
    except AttributeError:
        G = nx.random_tree(n, seed=seed)  # networkx < 3.4
    return nx.convert_node_labels_to_integers(G)


def sbm_graph(seed, n_communities=3, community_size=80, p_in=0.15, p_out=0.01):
    rng = np.random.RandomState(seed)
    k = int(rng.choice([2, 3, 4]))
    size = int(rng.choice([60, 80, 100]))
    sizes = [size] * k
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        main = max(comps, key=len)
        for c in comps:
            if c is main:
                continue
            G.add_edge(next(iter(c)), next(iter(main)))
    return nx.convert_node_labels_to_integers(G)


def preferential_attachment_graph(seed, n_nodes=250, m=3):
    rng = np.random.RandomState(seed)
    n = int(rng.randint(150, 350))
    mm = int(rng.choice([2, 3, 4]))
    G = nx.barabasi_albert_graph(n, mm, seed=seed)
    return nx.convert_node_labels_to_integers(G)


def hyperbolic_random_graph(seed, n_nodes=250, R=10.0, alpha=0.75):
    """
    Simple hyperbolic random graph (popularity-similarity style):
    radial coordinate density ~ exp(alpha * r) truncated at R (via inverse-CDF sampling),
    angular coordinate uniform on [0, 2*pi). Connect u,v if hyperbolic distance < R.
    """
    rng = np.random.RandomState(seed)
    n = int(rng.randint(150, 350))

    u = rng.uniform(0, 1, size=n)
    r = (1.0 / alpha) * np.log(1 + u * (np.exp(alpha * R) - 1))
    theta = rng.uniform(0, 2 * np.pi, size=n)

    def hyp_dist(i, j):
        dtheta = np.pi - abs(np.pi - abs(theta[i] - theta[j]))
        cosh_d = (np.cosh(r[i]) * np.cosh(r[j])
                  - np.sinh(r[i]) * np.sinh(r[j]) * np.cos(dtheta))
        cosh_d = max(cosh_d, 1.0)
        return np.arccosh(cosh_d)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if hyp_dist(i, j) < R:
                G.add_edge(i, j)

    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        main = max(comps, key=len)
        for c in comps:
            if c is main:
                continue
            G.add_edge(next(iter(c)), next(iter(main)))
    return nx.convert_node_labels_to_integers(G)


GRAPH_FAMILIES = {
    "balanced_tree": balanced_tree_graph,
    "random_tree": random_tree_graph,
    "sbm": sbm_graph,
    "preferential_attachment": preferential_attachment_graph,
    "hyperbolic_random": hyperbolic_random_graph,
}


def generate_graph(family_name, seed):
    fn = GRAPH_FAMILIES[family_name]
    G = fn(seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        main = max(comps, key=len)
        for c in comps:
            if c is main:
                continue
            G.add_edge(next(iter(c)), next(iter(main)))
    return G


def graph_metadata(G):
    """Full provenance for the base graph -- cheap to compute once per graph instance."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    try:
        diameter = nx.diameter(G) if n < 2000 else None  # expensive on huge graphs
    except nx.NetworkXError:
        diameter = None
    try:
        avg_path = nx.average_shortest_path_length(G) if n < 2000 else None
    except nx.NetworkXError:
        avg_path = None
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        if assortativity is not None and np.isnan(assortativity):
            assortativity = None
    except (nx.NetworkXError, ZeroDivisionError):
        assortativity = None

    return {
        "n": n,
        "m": m,
        "density": nx.density(G),
        "diameter": diameter,
        "average_path_length": avg_path,
        "clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "assortativity": assortativity,
        "average_degree": float(np.mean(degrees)) if degrees else None,
        "max_degree": int(max(degrees)) if degrees else None,
    }
