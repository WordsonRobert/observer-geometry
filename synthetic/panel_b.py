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
K_VALUES = [1, 2, 4, 8, 16, 24, 32]
N_REALIZATIONS = 100
OBS_RADIUS = 2
N_SEEDS = 5
N_OBSERVERS_PER_GRAPH = 20  # average over several observers per graph for stability


def mu_H(hidden_idx, dist_row, delta):
    if len(hidden_idx) == 0:
        return 0.0
    return torch.exp(-delta * dist_row[hidden_idx]).sum().item()


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d_f = all_pairs_hyp_dist(z)

    rng = np.random.RandomState(seed + 2000)
    observers = rng.choice(N, size=min(N_OBSERVERS_PER_GRAPH, N), replace=False)

    per_k_errs = {k: [] for k in K_VALUES}

    for v in observers:
        ball = set(nx.single_source_shortest_path_length(G, int(v), cutoff=OBS_RADIUS).keys())
        all_nodes = set(range(N))
        H = np.array(sorted(all_nodes - ball - {v}))
        outside_H = np.array(sorted(ball - {v}))  # candidates to ADD (currently observed, not in H)
        if len(H) < 40 or len(outside_H) < 40:
            continue  # skip tiny/degenerate hidden sets for this panel

        base_mu = mu_H(H, d_f[v], DELTA)

        for k in K_VALUES:
            k_del = k // 2
            k_add = k - k_del  # handles odd k too, still sums to k
            if k_del > len(H) or k_add > len(outside_H):
                continue
            diffs = []
            for rep in range(N_REALIZATIONS):
                del_idx = rng.choice(H, size=k_del, replace=False)
                add_idx = rng.choice(outside_H, size=k_add, replace=False)
                H_prime = np.array(sorted((set(H.tolist()) - set(del_idx.tolist())) | set(add_idx.tolist())))
                mu_prime = mu_H(H_prime, d_f[v], DELTA)
                diffs.append(abs(base_mu - mu_prime))
            per_k_errs[k].append(float(np.mean(diffs)))

    return per_k_errs


def main():
    agg = {k: [] for k in K_VALUES}
    for seed in range(N_SEEDS):
        print(f"[Panel B] seed {seed}...")
        res = run_one_graph(seed)
        for k in K_VALUES:
            agg[k].extend(res[k])

    output = {"delta": DELTA, "n_seeds": N_SEEDS, "obs_radius": OBS_RADIUS,
              "n_realizations_per_k": N_REALIZATIONS,
              "note": "symmetric difference |H triangle H'| = k exactly (k/2 deleted, k/2 added)",
              "k_values": K_VALUES, "points": []}
    for k in K_VALUES:
        vals = np.array(agg[k])
        output["points"].append({
            "k": k,
            "mean_descriptor_diff": float(vals.mean()) if len(vals) else None,
            "std_descriptor_diff": float(vals.std()) if len(vals) else None,
            "n_observers_pooled": int(len(vals)),
            "bound_y_equals_k": k,
        })

    with open(str(_HERE / "results") + "/panelB_hidden_region_perturbation.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved results/panelB_hidden_region_perturbation.json")


if __name__ == "__main__":
    import os
    os.makedirs(str(_HERE / "results"), exist_ok=True)
    main()
