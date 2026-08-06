"""
interventions_dose_response.py
Four controlled intervention axes. Each function takes the CURRENT hidden node set
and the graph, and returns:
    (G_new, hidden_nodes_new, intervention_meta)
where intervention_meta is a flat dict recording exactly what was changed --
edges_added, edges_removed, nodes_added, nodes_removed, cycles_added, branches_added,
rewired_edges -- so downstream analysis can regress on the actual realized change
rather than relying on the "level" label alone (levels can realize differently
depending on random placement, e.g. degenerate hidden regions with few candidates).

The frozen ball B(v,r) is NEVER touched by any of these functions.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import numpy as np
import networkx as nx


def axis_hidden_edges(G, ball_nodes, hidden_nodes, level, rng):
    """Level = number of new random edges added strictly within the hidden region."""
    G2 = G.copy()
    hidden_list = list(hidden_nodes)
    added = 0
    tries = 0
    max_tries = level * 20 + 50
    while added < level and tries < max_tries and len(hidden_list) >= 2:
        tries += 1
        a, b = rng.choice(hidden_list, size=2, replace=False)
        if a == b or G2.has_edge(a, b):
            continue
        G2.add_edge(int(a), int(b))
        added += 1

    meta = {
        "edges_added": added, "edges_removed": 0,
        "nodes_added": 0, "nodes_removed": 0,
        "cycles_added": 0, "branches_added": 0, "rewired_edges": 0,
    }
    return G2, set(hidden_nodes), meta


def axis_subtree_growth(G, ball_nodes, hidden_nodes, level, rng):
    """Level = number of new leaf nodes attached (as a growing subtree) to random hidden nodes."""
    G2 = G.copy()
    hidden_list = list(hidden_nodes)
    next_id = max(G2.nodes()) + 1
    new_hidden = set(hidden_nodes)
    nodes_added = 0
    edges_added = 0

    if hidden_list:
        anchor = int(rng.choice(hidden_list))
        frontier = [anchor]
        for _ in range(level):
            parent = rng.choice(frontier)
            new_node = next_id
            next_id += 1
            G2.add_node(new_node)
            G2.add_edge(int(parent), new_node)
            new_hidden.add(new_node)
            frontier.append(new_node)
            nodes_added += 1
            edges_added += 1

    meta = {
        "edges_added": edges_added, "edges_removed": 0,
        "nodes_added": nodes_added, "nodes_removed": 0,
        "cycles_added": 0, "branches_added": 0, "rewired_edges": 0,
    }
    return G2, new_hidden, meta


def axis_cycle_density(G, ball_nodes, hidden_nodes, level_fraction, rng):
    """
    Level = target fraction (0.0-1.0) of a fixed edge budget added among hidden nodes,
    realized as short random-walk closures so added edges actually form cycles.
    """
    G2 = G.copy()
    hidden_list = list(hidden_nodes)
    cycles_added = 0

    if len(hidden_list) >= 3:
        edge_budget = int(level_fraction * len(hidden_list))
        tries = 0
        max_tries = edge_budget * 10 + 50
        while cycles_added < edge_budget and tries < max_tries:
            tries += 1
            start = int(rng.choice(hidden_list))
            if start not in G2:
                continue
            walk = [start]
            cur = start
            for _ in range(rng.randint(3, 7)):
                nbrs = [n for n in G2.neighbors(cur) if n in hidden_nodes]
                if not nbrs:
                    break
                cur = int(rng.choice(nbrs))
                walk.append(cur)
            if len(walk) >= 3 and walk[0] != walk[-1] and not G2.has_edge(walk[0], walk[-1]):
                G2.add_edge(walk[0], walk[-1])
                cycles_added += 1

    meta = {
        "edges_added": cycles_added, "edges_removed": 0,
        "nodes_added": 0, "nodes_removed": 0,
        "cycles_added": cycles_added, "branches_added": 0, "rewired_edges": 0,
    }
    return G2, set(hidden_nodes), meta


def axis_hidden_branching(G, ball_nodes, hidden_nodes, level, rng):
    """
    Level = number of DISTINCT hidden nodes that each receive one new leaf child,
    increasing branching factor breadth-wise rather than growing a single subtree deep.
    """
    G2 = G.copy()
    hidden_list = list(hidden_nodes)
    new_hidden = set(hidden_nodes)
    branches_added = 0

    if hidden_list:
        next_id = max(G2.nodes()) + 1
        n_pick = min(level, len(hidden_list))
        picked_parents = rng.choice(hidden_list, size=n_pick, replace=False)
        for parent in picked_parents:
            new_node = next_id
            next_id += 1
            G2.add_node(new_node)
            G2.add_edge(int(parent), new_node)
            new_hidden.add(new_node)
            branches_added += 1

    meta = {
        "edges_added": branches_added, "edges_removed": 0,
        "nodes_added": branches_added, "nodes_removed": 0,
        "cycles_added": 0, "branches_added": branches_added, "rewired_edges": 0,
    }
    return G2, new_hidden, meta


INTERVENTION_AXES = {
    "hidden_edges": {
        "fn": axis_hidden_edges,
        "levels": [0, 10, 20, 40, 80, 160],
    },
    "subtree_growth": {
        "fn": axis_subtree_growth,
        "levels": [0, 20, 40, 80, 160],
    },
    "cycle_density": {
        "fn": axis_cycle_density,
        "levels": [0.0, 0.10, 0.20, 0.40, 0.80],
    },
    "hidden_branching": {
        "fn": axis_hidden_branching,
        "levels": [0, 10, 20, 40, 80],
    },
}
