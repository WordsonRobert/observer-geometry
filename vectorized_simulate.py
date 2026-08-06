"""
Vectorized replacement for simulate.py's HBPSimulator.

DESIGN NOTES (read before trusting this):

1. Preserves the exact walk semantics of the original: from a source node, run K
   independent non-backtracking weighted-random walks, each terminating when its
   cumulative path-probability PA drops below eps (or max_steps is hit). For each
   node v, "first_visiting_walk[v]" = the index (0..K-1) of the first walk (in
   some order) that reached v, or -1 if none did.

2. KEY EQUIVALENCE ASSUMPTION: the original runs the K walks SEQUENTIALLY, one
   full trajectory at a time. This version runs all K walks SYNCHRONOUSLY (all
   walks take one step per timestep, in lockstep) and uses each walk's fixed
   batch-index as its "order label". Because the K walks are i.i.d. (same source,
   independently resampled, non-backtracking is memoryless given (cur, prev)),
   they are exchangeable -- the distribution of "which k walks have reached v" is
   identical whether the K walks are labeled by sequential completion order or by
   an arbitrary fixed parallel index. This is what licenses full vectorization.
   If you need bit-exact reproduction of the ORIGINAL's specific random sequence
   (e.g. to exactly reproduce a cached phi_full_meanstd.pt to the last decimal),
   this will NOT do that -- it will produce a statistically equivalent but
   numerically different Phi. Re-validate against a small graph (e.g. the
   synthetic k-ary tree) by comparing distributions/correlations, not exact values.

3. Ragged-degree graphs are handled via a padded dense neighbor tensor
   [N, max_degree], padded with a sentinel weight of 0 (never sampled) and a
   sentinel neighbor index of -1 (masked out). This costs O(N * max_degree)
   memory -- fine for citation/PPI/GO graphs and for ogbn-arxiv unless max_degree
   is a large outlier (check with G.degree() before running at scale; a small
   number of very high-degree hub nodes would blow up memory disproportionately
   -- consider capping degree via a top-M-neighbor truncation for such hubs if so).

4. Early termination (PA < eps) is handled via a per-particle `active` boolean
   mask rather than literally breaking a shared loop -- inactive particles just
   stop updating state. This wastes some compute on already-terminated particles
   but is far faster than any Python-level per-particle branching.

5. Batches across (source nodes x repeats x K walks) as ONE flattened particle
   dimension W = B * R * K. Reduction ("first walk-index per node per group") is
   done via scatter_reduce with combined index (group_id * N + node), amin
   reduction, sentinel value K (meaning "never visited").

Usage mirrors phi_utils.phi_at_K_all_nodes / phi_curves_for_nodes but batched:
see `vectorized_phi_at_K_all_nodes` at the bottom.
"""
import torch


class VectorizedHBPSimulator:
    def __init__(self, edge_index, edge_weight, num_nodes, device="cuda", max_degree_cap=None):
        """
        edge_index: [2, E] (assumed symmetric/undirected -- both directions present)
        edge_weight: [E] transition weights (e.g. exp(-delta * d_H(u,v)))
        max_degree_cap: see _build_padded_neighbors docstring -- set this for graphs
            with extreme-degree hub nodes (e.g. ogbn-arxiv) to avoid OOM.
        """
        self.device = device
        self.N = num_nodes
        self.max_degree, self.nbr_idx, self.nbr_wt = self._build_padded_neighbors(
            edge_index, edge_weight, num_nodes, max_degree_cap=max_degree_cap
        )
        self.nbr_idx = self.nbr_idx.to(device)
        self.nbr_wt = self.nbr_wt.to(device)

    def _build_padded_neighbors(self, edge_index, edge_weight, num_nodes, max_degree_cap=None):
        """
        max_degree_cap: if set, any node with more neighbors than this keeps only
        its top-`max_degree_cap` HIGHEST-WEIGHT neighbors (i.e. closest in hyperbolic
        distance, since weight = exp(-delta*dist)). This is a real approximation for
        extreme-degree hub nodes (e.g. ogbn-arxiv has one node with degree 13161 vs
        mean degree ~13.7) -- without capping, the padded tensor scales with the MAX
        degree across ALL nodes, which can blow past available VRAM. Flag this
        truncation explicitly if you use it in the paper: it changes hub-node walk
        dynamics (they sample from their nearest max_degree_cap neighbors only,
        not their full neighbor set), though it should have negligible effect on
        the vast majority of non-hub nodes.
        """
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        src, dst = edge_index
        degrees = torch.bincount(src, minlength=num_nodes)
        true_max_degree = int(degrees.max().item())

        if max_degree_cap is not None and true_max_degree > max_degree_cap:
            print(f"  [WARN] true max_degree={true_max_degree} exceeds cap={max_degree_cap}; "
                  f"truncating hub nodes to their top-{max_degree_cap} highest-weight neighbors. "
                  f"This is an approximation -- see docstring.")
            max_degree = max_degree_cap
        else:
            max_degree = true_max_degree

        nbr_idx = torch.full((num_nodes, max_degree), -1, dtype=torch.long)
        nbr_wt = torch.zeros((num_nodes, max_degree), dtype=torch.float)

        order = torch.argsort(src)
        src_sorted = src[order]
        dst_sorted = dst[order]
        w_sorted = edge_weight[order]

        # position-within-row for each sorted edge (standard "ragged to padded" trick)
        counts = torch.bincount(src_sorted, minlength=num_nodes)
        row_start = torch.zeros(num_nodes, dtype=torch.long)
        row_start[1:] = torch.cumsum(counts, dim=0)[:-1]
        pos_in_row = torch.arange(src_sorted.size(0)) - row_start[src_sorted]

        if max_degree_cap is not None and true_max_degree > max_degree_cap:
            keep_mask = pos_in_row < max_degree
            over_cap_nodes = (counts > max_degree_cap).nonzero(as_tuple=True)[0]
            if over_cap_nodes.numel() > 0:
                print(f"  {over_cap_nodes.numel()} node(s) exceed the cap and will be truncated "
                      f"to their top-{max_degree_cap} highest-weight neighbors.")
            for node in over_cap_nodes.tolist():
                s, e = row_start[node].item(), row_start[node].item() + counts[node].item()
                node_w = w_sorted[s:e]
                node_d = dst_sorted[s:e]
                top_vals, top_idx = torch.topk(node_w, max_degree_cap)
                nbr_idx[node, :max_degree_cap] = node_d[top_idx]
                nbr_wt[node, :max_degree_cap] = top_vals
                keep_mask[s:e] = False

            nbr_idx[src_sorted[keep_mask], pos_in_row[keep_mask]] = dst_sorted[keep_mask]
            nbr_wt[src_sorted[keep_mask], pos_in_row[keep_mask]] = w_sorted[keep_mask]
        else:
            nbr_idx[src_sorted, pos_in_row] = dst_sorted
            nbr_wt[src_sorted, pos_in_row] = w_sorted

        print(f"  Padded neighbor tensor: [{num_nodes}, {max_degree}] "
              f"({num_nodes * max_degree * 8 / 1e6:.1f} MB idx + "
              f"{num_nodes * max_degree * 4 / 1e6:.1f} MB weights)")
        return max_degree, nbr_idx, nbr_wt

    @torch.no_grad()
    def run_batch(self, sources, K, eps, max_steps=2000):
        """
        sources: LongTensor [B] -- one source node per group.
        Returns: first_visit [B, N] LongTensor, values in 0..K-1 or K (sentinel
                 meaning "never visited by any of the K walks from this source").
        """
        device = self.device
        B = sources.size(0)
        W = B * K  # total flattened particle count

        # particle -> group (source) index, and particle -> walk-order label
        group_id = torch.arange(B, device=device).repeat_interleave(K)         # [W]
        walk_label = torch.arange(K, device=device).repeat(B)                  # [W]

        cur = sources.to(device).repeat_interleave(K)                          # [W]
        prev = torch.full((W,), -1, dtype=torch.long, device=device)
        PA = torch.ones(W, device=device)
        active = torch.ones(W, dtype=torch.bool, device=device)

        # first_visit[group, node] = min walk_label that reached it, else K (sentinel)
        first_visit_flat = torch.full((B * self.N,), K, dtype=torch.long, device=device)

        def record_visit(particle_mask):
            idx = group_id[particle_mask] * self.N + cur[particle_mask]
            vals = walk_label[particle_mask]
            first_visit_flat.scatter_reduce_(0, idx, vals, reduce="amin")

        record_visit(torch.ones(W, dtype=torch.bool, device=device))  # record source itself, step 0

        for step in range(max_steps):
            if not active.any():
                break

            nbrs = self.nbr_idx[cur]        # [W, max_degree]
            wts = self.nbr_wt[cur].clone()  # [W, max_degree]

            # non-backtracking: zero out the weight of the edge back to prev
            has_prev = prev >= 0
            if has_prev.any():
                back_mask = (nbrs == prev.unsqueeze(1)) & has_prev.unsqueeze(1)
                wts = wts.masked_fill(back_mask, 0.0)

            row_sums = wts.sum(dim=1)
            dead_end = row_sums <= 0
            active = active & ~dead_end

            # avoid div-by-zero for dead/inactive rows; they won't be used anyway
            safe_sums = row_sums.clamp(min=1e-12)
            probs = wts / safe_sums.unsqueeze(1)

            # sample next neighbor per active particle
            choice = torch.multinomial(probs.clamp(min=0) + 1e-12, 1).squeeze(1)  # [W]
            nxt = nbrs.gather(1, choice.unsqueeze(1)).squeeze(1)
            step_prob = probs.gather(1, choice.unsqueeze(1)).squeeze(1)

            new_PA = PA * step_prob
            below_eps = new_PA < eps
            active = active & ~below_eps

            # only advance particles that are still active
            advance = active
            prev = torch.where(advance, cur, prev)
            cur = torch.where(advance, nxt, cur)
            PA = torch.where(advance, new_PA, PA)

            if advance.any():
                record_visit(advance)

        return first_visit_flat.view(B, self.N)

    @torch.no_grad()
    def phi_from_first_visit(self, first_visit, weights_by_distance_batch, K):
        """
        first_visit: [B, N] from run_batch (values 0..K-1 or K=unvisited)
        weights_by_distance_batch: [B, N] -- per-source hidden-mass weights
        Returns: phi [B] -- final Phi(K) value per source (remaining unobserved mass / total)
        """
        observed = first_visit < K  # [B, N] bool -- visited within these K walks
        W_total = weights_by_distance_batch.sum(dim=1)  # [B]
        remaining_mass = (weights_by_distance_batch * (~observed)).sum(dim=1)
        return remaining_mass / W_total.clamp(min=1e-12)


def lorentz_pairwise_dist(batch_z, all_z, k):
    """
    Memory-efficient pairwise Lorentz distance via matmul, avoiding the
    [B, N, dim] broadcast/expand that OOMs for large N (e.g. ogbn-arxiv's
    169k nodes). Mathematically identical to manifold.dist(x,y) for every
    (batch, all) pair, just computed without materializing the expanded tensor.

    Lorentzian inner product: <u,v>_L = -u_0*v_0 + sum_{i>=1} u_i*v_i
    Distance: sqrt(k) * arcosh(-<u,v>_L / k)
    """
    negated = batch_z.clone()
    negated[:, 0] *= -1
    inner = negated @ all_z.t()          # [B, N] -- the expensive expand is GONE
    d = (-inner / k).clamp(min=1.0 + 1e-6)
    return torch.sqrt(k) * torch.acosh(d)


@torch.no_grad()
def vectorized_phi_at_K_all_nodes(edge_index, z, manifold, K, eps, delta, num_nodes,
                                    n_repeats=3, source_batch_size=256, device="cuda",
                                    max_steps=2000, max_degree_cap=None):
    """
    Drop-in vectorized replacement for phi_utils.phi_at_K_all_nodes.
    Processes nodes in batches of `source_batch_size`, each batch running
    n_repeats independent trials, each trial running K parallel walks --
    i.e. per source_batch, particle count = source_batch_size * n_repeats * K.

    Tune source_batch_size down if you hit OOM on the padded neighbor gather.
    """
    from phi_utils import build_csr_from_embedding  # reuse existing weight-building logic

    z = z.to(device)  # move embedding to target device once, upfront
    edge_index = edge_index.to(device)

    # manifold's curvature tensor may have been created on CPU in load_embedding();
    # rebuild it on the target device so manifold.dist() doesn't choke on mixed devices.
    import geoopt
    curvature_value = manifold.k.item() if hasattr(manifold, "k") else float(manifold.k)
    manifold = geoopt.Lorentz(k=torch.tensor(curvature_value, device=device))

    # Build edge weights once (same as build_csr_from_embedding but we only need
    # the flat edge_weight here, since VectorizedHBPSimulator builds its own
    # padded structure directly from edge_index + edge_weight).
    src, dst = edge_index
    with torch.no_grad():
        dists = manifold.dist(z[src], z[dst])
    edge_weight = torch.exp(-delta * dists)

    sim = VectorizedHBPSimulator(edge_index, edge_weight, num_nodes, device=device,
                                   max_degree_cap=max_degree_cap)

    phi_final = torch.zeros(num_nodes)
    all_nodes = torch.arange(num_nodes)

    for start in range(0, num_nodes, source_batch_size):
        batch_nodes = all_nodes[start:start + source_batch_size].to(device)
        Bsz = batch_nodes.size(0)

        # per-source hidden-mass weights: weight_by_distance for EVERY node in batch,
        # vs every node in the graph -- [Bsz, N]. Uses matmul-based lorentz_pairwise_dist
        # (NOT the naive [Bsz, N, dim] broadcast, which OOMs for large N like arxiv).
        with torch.no_grad():
            k_val = manifold.k
            batch_dists = lorentz_pairwise_dist(z[batch_nodes], z, k_val)  # [Bsz, N]
        weights_batch = torch.exp(-delta * batch_dists).to(device)

        phi_accum = torch.zeros(Bsz, device=device)
        for rep in range(n_repeats):
            first_visit = sim.run_batch(batch_nodes, K, eps, max_steps=max_steps)
            phi_rep = sim.phi_from_first_visit(first_visit, weights_batch, K)
            phi_accum += phi_rep
        phi_final[start:start + Bsz] = (phi_accum / n_repeats).cpu()

        print(f"  batch {start}-{start+Bsz}/{num_nodes} done")

    return phi_final
