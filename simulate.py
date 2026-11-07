"""Sequential reference simulator for the exploration process of the hyperbolic instantiation:
K i.i.d. non-backtracking walks from a source, transition probabilities proportional to
exp(-delta d_H), each walk stopping when its path probability would fall below eps.
"""
import torch


class WalkSimulator:
    def __init__(self, row_ptr, col_idx, edge_weight, device="cpu"):
        self.device = device
        self.row_ptr = row_ptr.to(device)
        self.col_idx = col_idx.to(device)
        self.edge_weight = edge_weight.to(device)
        self.N = row_ptr.numel() - 1

    def _neighbors(self, node):
        s, e = self.row_ptr[node].item(), self.row_ptr[node + 1].item()
        return self.col_idx[s:e], self.edge_weight[s:e]

    @torch.no_grad()
    def run(self, source, K, eps, max_steps=10_000):
        first_visiting_walk = torch.full((self.N,), -1, dtype=torch.long, device=self.device)

        for walk_idx in range(K):
            cur = source
            prev = -1
            PA = 1.0
            if first_visiting_walk[cur] == -1:
                first_visiting_walk[cur] = walk_idx
            for step in range(max_steps):
                nbrs, wts = self._neighbors(cur)
                if prev != -1:
                    mask = nbrs != prev
                    nbrs = nbrs[mask]
                    wts = wts[mask]
                if nbrs.numel() == 0:
                    break
                probs = wts / wts.sum()
                choice_idx = torch.multinomial(probs, 1).item()
                nxt = nbrs[choice_idx].item()
                step_prob = probs[choice_idx].item()
                newPA = PA * step_prob
                if newPA < eps:
                    break
                PA = newPA
                prev, cur = cur, nxt
                if first_visiting_walk[cur] == -1:
                    first_visiting_walk[cur] = walk_idx

        return first_visiting_walk

    @torch.no_grad()
    def phi_curve(self, source, K, eps, weights_by_distance, n_repeats=20, max_steps=10_000):
        W = weights_by_distance.sum()
        phi_accum = torch.zeros(K, device=self.device)

        for _ in range(n_repeats):
            fvw = self.run(source, K, eps, max_steps=max_steps)
            observed_mask = fvw.unsqueeze(0) < torch.arange(1, K + 1, device=self.device).unsqueeze(1)
            observed_mask &= (fvw.unsqueeze(0) != -1)
            remaining_mass = (weights_by_distance.unsqueeze(0) * (~observed_mask)).sum(dim=1)
            phi_accum += remaining_mass / W

        return phi_accum / n_repeats


HBPSimulator = WalkSimulator  # name used by older scripts
