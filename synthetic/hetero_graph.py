"""
Synthetic Heterogeneous Graph construction + Poincare-ball embedding + mu (Phi) descriptor.
Used as shared infrastructure for Panels A-D robustness experiments.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import numpy as np
import networkx as nx
import torch

torch.set_default_dtype(torch.float64)


def build_heterogeneous_graph(seed=0, community_size=80, n_communities=3,
                               p_in=0.18, p_out=0.015,
                               hub_degree=45, path_length=20,
                               n_shortcuts=15, n_cycles=8,
                               edge_removal_frac=0.05):
    rng = np.random.RandomState(seed)

    # 1. SBM communities
    sizes = [community_size] * n_communities
    probs = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.Graph(G)  # drop block attributes complications, keep as plain graph
    N0 = G.number_of_nodes()

    # 2. Hub: new node connected to ~hub_degree random existing nodes (favor spread across communities)
    hub = N0
    G.add_node(hub)
    hub_targets = rng.choice(N0, size=min(hub_degree, N0), replace=False)
    for t in hub_targets:
        G.add_edge(hub, int(t))

    # 3. Long path (length 20) attached to the hub at one end, to a random community node at the other
    path_nodes = list(range(hub + 1, hub + 1 + path_length))
    prev = hub
    for pn in path_nodes:
        G.add_node(pn)
        G.add_edge(prev, pn)
        prev = pn
    far_end = prev
    attach_target = int(rng.choice(N0))
    G.add_edge(far_end, attach_target)

    # 4. Random shortcuts (long-range edges between distant nodes)
    all_nodes = list(G.nodes())
    added = 0
    tries = 0
    while added < n_shortcuts and tries < n_shortcuts * 20:
        tries += 1
        a, b = rng.choice(all_nodes, size=2, replace=False)
        if not G.has_edge(a, b):
            G.add_edge(int(a), int(b))
            added += 1

    # 5. Random cycles: pick a short random walk and close it into a loop
    for _ in range(n_cycles):
        start = int(rng.choice(all_nodes))
        walk = [start]
        cur = start
        for _ in range(rng.randint(3, 7)):
            nbrs = list(G.neighbors(cur))
            if not nbrs:
                break
            cur = int(rng.choice(nbrs))
            walk.append(cur)
        if len(walk) >= 3 and walk[0] != walk[-1]:
            G.add_edge(walk[0], walk[-1])

    # 6. Remove ~5% of edges randomly (but keep graph connected)
    edges = list(G.edges())
    rng.shuffle(edges)
    n_remove = int(edge_removal_frac * len(edges))
    removed = 0
    for (u, v) in edges:
        if removed >= n_remove:
            break
        G.remove_edge(u, v)
        if not nx.is_connected(G):
            G.add_edge(u, v)  # revert if disconnects
        else:
            removed += 1

    # ensure connected (attach any stray components to hub)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        main = max(comps, key=len)
        for c in comps:
            if c is main:
                continue
            rep = next(iter(c))
            G.add_edge(rep, hub)

    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def poincare_dist(u, v, eps=1e-7):
    # u, v: (..., d) tensors in open unit ball
    diff_sq = ((u - v) ** 2).sum(-1)
    u_sq = (u ** 2).sum(-1)
    v_sq = (v ** 2).sum(-1)
    denom = (1 - u_sq).clamp_min(eps) * (1 - v_sq).clamp_min(eps)
    x = 1 + 2 * diff_sq / denom
    x = x.clamp_min(1 + eps)
    return torch.acosh(x)


def project_to_ball(z, max_norm=1 - 1e-5):
    norm = z.norm(dim=-1, keepdim=True)
    factor = torch.clamp(max_norm / norm, max=1.0)
    return z * factor


def train_hyperbolic_embedding(G, dim=10, epochs=400, lr=0.05, seed=0, verbose=False):
    N = G.number_of_nodes()
    sp = dict(nx.all_pairs_shortest_path_length(G))
    D = np.zeros((N, N))
    for i in range(N):
        for j, d in sp[i].items():
            D[i, j] = d
    D = torch.tensor(D)
    D = D / D.max()  # normalize graph distances to [0,1] scale, then stretch for hyperbolic target
    target = D * 4.0  # stretch factor so hyperbolic distances have room to differentiate

    torch.manual_seed(seed)
    z = (torch.randn(N, dim) * 0.001)
    z = project_to_ball(z)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        zc = project_to_ball(z)
        idx = torch.randint(0, N, (min(2000, N * 4),))
        idx2 = torch.randint(0, N, (min(2000, N * 4),))
        mask = idx != idx2
        idx, idx2 = idx[mask], idx2[mask]
        d_hyp = poincare_dist(zc[idx], zc[idx2])
        d_target = target[idx, idx2]
        loss = ((d_hyp - d_target) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"  epoch {ep}: loss={loss.item():.5f}")

    with torch.no_grad():
        z_final = project_to_ball(z).detach()
    return z_final


def all_pairs_hyp_dist(z):
    N = z.size(0)
    d = torch.zeros(N, N)
    for i in range(N):
        d[i] = poincare_dist(z[i:i+1].expand(N, -1), z)
    return d


def mu_from_dist_matrix(dist_matrix, delta):
    """mu_H(v) computed on the fly given a boolean hidden mask matrix or index sets."""
    return torch.exp(-delta * dist_matrix)
