"""
Bridges the trained HVGAE embeddings to the HBPSimulator: builds real edge weights
from Lorentz distances (replacing the graph-distance placeholder from earlier), and
computes Phi(K) curves for a set of source nodes.
"""
import torch
import geoopt


def load_embedding(embedding_path, curvature_path):
    z = torch.load(embedding_path, map_location="cpu")
    with open(curvature_path) as f:
        c = float(f.read().strip())
    manifold = geoopt.Lorentz(k=torch.tensor(c))
    return z, manifold, c


def compute_radii(z, manifold):
    """Hyperbolic distance from the manifold origin for every node -- used for
    stratified sampling, since Phi is fundamentally a geometric/radial quantity."""
    origin = manifold.origin(z.shape[-1])
    origin_batch = origin.unsqueeze(0).expand(z.size(0), -1)
    with torch.no_grad():
        radii = manifold.dist(origin_batch, z)
    return radii


def stratified_sample_by_radius(z, manifold, n_per_stratum=100, n_strata=3, seed=0):
    """
    Splits nodes into n_strata buckets by radius quantile (inner/middle/outer),
    samples n_per_stratum from each. Returns a flat list of node indices.
    """
    g = torch.Generator().manual_seed(seed)
    radii = compute_radii(z, manifold)
    N = radii.size(0)
    order = torch.argsort(radii)
    boundaries = [int(N * i / n_strata) for i in range(n_strata + 1)]

    sample = []
    for s in range(n_strata):
        stratum_nodes = order[boundaries[s]:boundaries[s + 1]]
        n_take = min(n_per_stratum, stratum_nodes.numel())
        perm = torch.randperm(stratum_nodes.numel(), generator=g)[:n_take]
        sample.extend(stratum_nodes[perm].tolist())
    return sample


def build_and_cache_kernel(edge_index, z, manifold, delta, num_nodes, cache_path):
    """Builds the CSR transition kernel once and caches it -- geometry never changes
    across the K*/sensitivity/full-pass stages, only K and eps do."""
    import os
    if os.path.exists(cache_path):
        row_ptr, col_idx, edge_weight = torch.load(cache_path)
        print(f"Loaded cached kernel from {cache_path}")
        return row_ptr, col_idx, edge_weight
    row_ptr, col_idx, edge_weight = build_csr_from_embedding(edge_index, z, manifold, delta, num_nodes)
    torch.save((row_ptr, col_idx, edge_weight), cache_path)
    print(f"Built and cached kernel to {cache_path}")
    return row_ptr, col_idx, edge_weight


def build_csr_from_embedding(edge_index, z, manifold, delta, num_nodes):
    """
    edge_index: [2, E] (assumed already symmetric/undirected, i.e. both directions present)
    Returns CSR (row_ptr, col_idx, edge_weight) with edge_weight = exp(-delta * d_H(u,v))
    """
    src, dst = edge_index
    with torch.no_grad():
        dists = manifold.dist(z[src], z[dst])
    weights = torch.exp(-delta * dists)

    order = torch.argsort(src)
    src_sorted = src[order]
    dst_sorted = dst[order]
    w_sorted = weights[order]

    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    counts = torch.bincount(src_sorted, minlength=num_nodes)
    row_ptr[1:] = torch.cumsum(counts, dim=0)

    return row_ptr, dst_sorted, w_sorted


def weights_by_distance(z, manifold, source_idx, delta):
    """w_u = exp(-delta * d_H(source, u)) for every node u. Vectorized over all nodes."""
    with torch.no_grad():
        src_point = z[source_idx].unsqueeze(0).expand(z.size(0), -1)
        dists = manifold.dist(src_point, z)
    return torch.exp(-delta * dists)


def phi_curves_for_nodes(sim, source_nodes, z, manifold, K, eps, delta, n_repeats=5, max_steps=2000):
    """
    Returns dict: {node_id: FloatTensor[K]} -- Phi(1)...Phi(K) for each requested source.
    """
    curves = {}
    for i, src in enumerate(source_nodes):
        w = weights_by_distance(z, manifold, src, delta)
        curve = sim.phi_curve(src, K, eps, w, n_repeats=n_repeats, max_steps=max_steps)
        curves[src] = curve
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(source_nodes)} source nodes done")
    return curves


def phi_at_K_all_nodes(sim, num_nodes, z, manifold, K, eps, delta, n_repeats=3, max_steps=2000):
    """Full-node pass -- only Phi(K) (not the whole curve) to save time. Use this ONCE,
    at the plateau K determined from the sample-based rank-stability analysis."""
    phi_final = torch.zeros(num_nodes)
    for src in range(num_nodes):
        w = weights_by_distance(z, manifold, src, delta)
        curve = sim.phi_curve(src, K, eps, w, n_repeats=n_repeats, max_steps=max_steps)
        phi_final[src] = curve[-1]
        if src % 200 == 0:
            print(f"  {src}/{num_nodes} nodes done")
    return phi_final
