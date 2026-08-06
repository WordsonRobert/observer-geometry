"""
reembed.py
==========
Warm-started hyperbolic re-embedding. This is the module that fixes the two
worst problems in the original dose-response study.

PROBLEM 1: THE OLD STUDY HAD NO HYPERBOLIC GEOMETRY IN IT.
    observer_utils.base_layout used nx.spring_layout(dim=2) with Euclidean
    distance. Figure 1 -- the sole empirical support for "captures structural
    information distinct from classical graph descriptors" -- contained no
    Poincare ball, no Lorentz model, no hyperbolic distance. Here the base
    graph is embedded into the Poincare ball with the same stress objective
    used by the stability benchmark.

PROBLEM 2: THE FROZEN EMBEDDING FORCED THE CONCLUSION.
    With coordinates frozen, an edge-only intervention cannot move any point,
    so mu = sum_{u in H} exp(-delta d(v,u)) is IDENTICALLY unchanged. The
    finding "Phi tracks extent, not connectivity" was therefore a property of
    the protocol, not of the descriptor -- it restates mu_stability_I, which
    is an exact identity proved in Lean. Re-embedding after each intervention
    lets connectivity changes propagate into the metric, so the question
    becomes empirical.

WHY WARM-STARTING MAKES THIS AFFORDABLE
    Re-embedding from random init costs ~300 epochs. But an intervention
    perturbs the graph only locally, so the post-intervention optimum is close
    to the pre-intervention one. Initialising from the base embedding (new
    nodes placed at the Mobius-free mean of their placed neighbours, pulled
    slightly toward the origin) and refining for ~50 epochs gets there at
    roughly 1/6 the cost.

    Set REEMBED_EPOCHS=0 to reproduce the old frozen behaviour for comparison.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import torch

from hetero_graph import poincare_dist, project_to_ball

torch.set_default_dtype(torch.float64)

DIM = 10
BASE_EPOCHS = 300
REEMBED_EPOCHS = 50
BASE_LR = 0.05
REEMBED_LR = 0.02
TARGET_SCALE = 4.0
PAIRS_PER_STEP = 2000


def _target_matrix(G, nodes):
    """Normalised all-pairs shortest-path distance, scaled for hyperbolic room."""
    index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    D = np.zeros((N, N))
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        if src not in index:
            continue
        i = index[src]
        for dst, d in lengths.items():
            j = index.get(dst)
            if j is not None:
                D[i, j] = d
    m = D.max()
    if m > 0:
        D = D / m
    return torch.tensor(D * TARGET_SCALE), index


def _optimise(z_init, target, epochs, lr, seed=0):
    if epochs <= 0:
        return project_to_ball(z_init).detach()
    N = z_init.size(0)
    g = torch.Generator().manual_seed(seed)
    z = project_to_ball(z_init.clone()).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    n_pairs = min(PAIRS_PER_STEP, max(N * 4, 64))
    for _ in range(epochs):
        opt.zero_grad()
        zc = project_to_ball(z)
        i = torch.randint(0, N, (n_pairs,), generator=g)
        j = torch.randint(0, N, (n_pairs,), generator=g)
        keep = i != j
        i, j = i[keep], j[keep]
        if i.numel() == 0:
            continue
        loss = ((poincare_dist(zc[i], zc[j]) - target[i, j]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return project_to_ball(z).detach()


def embed_base(G, seed=0, epochs=BASE_EPOCHS, dim=DIM):
    """
    Cold embedding of the base graph -- run ONCE per graph instance.
    Returns (z [N, dim], nodes list, index dict).
    """
    nodes = sorted(G.nodes())
    target, index = _target_matrix(G, nodes)
    torch.manual_seed(seed)
    z0 = torch.randn(len(nodes), dim) * 0.001
    z = _optimise(z0, target, epochs, BASE_LR, seed=seed)
    return z, nodes, index


def _init_new_nodes(G_new, nodes_new, z_old, index_old, rng, jitter=0.01):
    """
    Initial coordinates for the post-intervention graph: keep every existing
    node where it was, place each new node near its already-placed neighbours.

    The mean is taken in ambient coordinates then shrunk by 0.95 and projected.
    That is not a true Poincare barycentre, but it only has to be a good enough
    STARTING POINT -- the refinement below is what determines the final
    position, so the approximation does not bias the result.
    """
    dim = z_old.size(1)
    z = torch.zeros(len(nodes_new), dim)
    index_new = {n: i for i, n in enumerate(nodes_new)}

    placed = []
    for n in nodes_new:
        if n in index_old:
            z[index_new[n]] = z_old[index_old[n]]
            placed.append(n)
    placed_set = set(placed)

    for n in nodes_new:
        if n in placed_set:
            continue
        nbrs = [nb for nb in G_new.neighbors(n) if nb in placed_set]
        if nbrs:
            base = torch.stack([z[index_new[nb]] for nb in nbrs]).mean(dim=0) * 0.95
        else:
            base = torch.zeros(dim)
        noise = torch.tensor(rng.normal(0, jitter, size=dim))
        z[index_new[n]] = base + noise
        placed_set.add(n)

    return project_to_ball(z), index_new


def reembed_warm(G_new, z_old, index_old, rng, epochs=REEMBED_EPOCHS, seed=0):
    """
    Refine the embedding after an intervention. Returns (z, nodes, index).
    epochs=0 reproduces the frozen-embedding protocol exactly.
    """
    nodes_new = sorted(G_new.nodes())
    z_init, index_new = _init_new_nodes(G_new, nodes_new, z_old, index_old, rng)
    if epochs <= 0:
        return z_init, nodes_new, index_new
    target, _ = _target_matrix(G_new, nodes_new)
    z = _optimise(z_init, target, epochs, REEMBED_LR, seed=seed)
    return z, nodes_new, index_new


def phi_variants_hyperbolic(z, index, v, hidden_nodes,
                            delta_default=1.0, delta_alt=2.0):
    """
    Descriptor variants under genuine hyperbolic distance.

    Reports THREE normalisations, because the project currently uses different
    ones in different places and the paper needs to settle on one:
        mu                -- raw remaining mass  (Section 4.4 as written)
        Phi_normalized    -- mu / mu(V)          (what both simulators return)
        mu_per_node       -- mu / |H|            (the old dose-response 'Phi')
    """
    iv = index.get(v)
    if iv is None:
        return None

    hidden = [u for u in hidden_nodes if u in index]
    if not hidden:
        return {"mu": 0.0, "Phi_normalized": 0.0, "mu_per_node": 0.0,
                "mu_delta2": 0.0, "n_hidden": 0,
                "mean_hidden_dist": 0.0, "max_kernel_weight": 0.0}

    hi = torch.tensor([index[u] for u in hidden], dtype=torch.long)
    src = z[iv].unsqueeze(0).expand(hi.numel(), -1)
    with torch.no_grad():
        d_hidden = poincare_dist(src, z[hi])
        all_src = z[iv].unsqueeze(0).expand(z.size(0), -1)
        d_all = poincare_dist(all_src, z)

    w = torch.exp(-delta_default * d_hidden)
    mu = float(w.sum())
    mu_V = float(torch.exp(-delta_default * d_all).sum())

    return {
        "mu": mu,
        "Phi_normalized": mu / mu_V if mu_V > 0 else 0.0,
        "mu_per_node": mu / len(hidden),
        "mu_delta2": float(torch.exp(-delta_alt * d_hidden).sum()),
        "n_hidden": len(hidden),
        "mean_hidden_dist": float(d_hidden.mean()),
        "max_kernel_weight": float(w.max()),
    }
