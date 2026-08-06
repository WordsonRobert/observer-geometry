import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import json
import numpy as np
import torch
import networkx as nx
from hetero_graph import build_heterogeneous_graph, train_hyperbolic_embedding, all_pairs_hyp_dist

DELTA = 1.0
OBS_RADIUS = 2  # hidden region definition: H(v) = V \ B_r(v), FIXED r used for all observers
N_SEEDS = 5
N_SAMPLES_PER_RING = 5  # randomly sampled partners per ring per observer, then averaged
# ring bin edges in embedding (hyperbolic) distance space
RING_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def mu_H(hidden_idx, dist_row, delta):
    if len(hidden_idx) == 0:
        return 0.0
    return torch.exp(-delta * dist_row[hidden_idx]).sum().item()


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d_f = all_pairs_hyp_dist(z)

    # fixed hidden region per observer, H(v) = V \ B_r(v)
    hidden_sets = {}
    for v in range(N):
        ball = set(nx.single_source_shortest_path_length(G, v, cutoff=OBS_RADIUS).keys())
        hidden_sets[v] = np.array([u for u in range(N) if u not in ball])

    mu_vals = np.array([mu_H(hidden_sets[v], d_f[v], DELTA) for v in range(N)])

    # for every observer v, for every ring bin, sample N_SAMPLES_PER_RING random OTHER
    # observers v' whose embedding distance to v falls inside that ring bin, then average
    # |mu(v)-mu(v')| over the sampled set (rather than always taking the single nearest one,
    # which biases the estimate toward the low-distance edge of each bin).
    rng = np.random.RandomState(seed + 4000)
    bin_diffs = {i: [] for i in range(len(RING_EDGES) - 1)}
    bin_centers = [(RING_EDGES[i] + RING_EDGES[i + 1]) / 2 for i in range(len(RING_EDGES) - 1)]

    d_np = d_f.numpy()
    for v in range(N):
        dists = d_np[v].copy()
        dists[v] = np.inf
        for b in range(len(RING_EDGES) - 1):
            lo, hi = RING_EDGES[b], RING_EDGES[b + 1]
            in_bin = np.where((dists >= lo) & (dists < hi))[0]
            if len(in_bin) == 0:
                continue
            n_pick = min(N_SAMPLES_PER_RING, len(in_bin))
            picked = rng.choice(in_bin, size=n_pick, replace=False)
            for p in picked:
                bin_diffs[b].append(abs(mu_vals[v] - mu_vals[p]))

    return bin_diffs, bin_centers


def main():
    agg = {i: [] for i in range(len(RING_EDGES) - 1)}
    bin_centers = None
    for seed in range(N_SEEDS):
        print(f"[Panel D] seed {seed}...")
        bd, bc = run_one_graph(seed)
        bin_centers = bc
        for i in range(len(RING_EDGES) - 1):
            agg[i].extend(bd[i])

    output = {"delta": DELTA, "n_seeds": N_SEEDS, "obs_radius": OBS_RADIUS,
              "n_samples_per_ring": N_SAMPLES_PER_RING,
              "ring_edges": RING_EDGES,
              "note": "rings binned by hyperbolic embedding distance d_X(f(v),f(v')); "
                      f"for each v, {N_SAMPLES_PER_RING} random partners sampled (without replacement) "
                      "from each ring bin and averaged, rather than always taking the single nearest "
                      "partner -- avoids biasing each bin toward its low-distance edge",
              "points": []}
    for i, center in enumerate(bin_centers):
        vals = np.array(agg[i])
        output["points"].append({
            "ring_bin_lo": RING_EDGES[i],
            "ring_bin_hi": RING_EDGES[i + 1],
            "ring_bin_center": center,
            "mean_descriptor_diff": float(vals.mean()) if len(vals) else None,
            "std_descriptor_diff": float(vals.std()) if len(vals) else None,
            "n_pairs_pooled": int(len(vals)),
        })

    with open(str(_HERE / "results") + "/panelD_observer_perturbation.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved results/panelD_observer_perturbation.json")


if __name__ == "__main__":
    import os
    os.makedirs(str(_HERE / "results"), exist_ok=True)
    main()
