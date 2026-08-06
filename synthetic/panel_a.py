import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import json
import numpy as np
import torch
import networkx as nx
from hetero_graph import (build_heterogeneous_graph, train_hyperbolic_embedding,
                           poincare_dist, project_to_ball, all_pairs_hyp_dist)

torch.manual_seed(0)

DELTA = 1.0
SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
N_PERTURB = 30
OBS_RADIUS = 2
N_SEEDS = 5


def hidden_mask(G, v, r):
    ball = set(nx.single_source_shortest_path_length(G, v, cutoff=r).keys())
    N = G.number_of_nodes()
    mask = np.ones(N, dtype=bool)
    for u in ball:
        mask[u] = False
    mask[v] = False
    return mask


def mu_H(v, hidden_idx, dist_row, delta):
    if len(hidden_idx) == 0:
        return 0.0
    return torch.exp(-delta * dist_row[hidden_idx]).sum().item()


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d_f = all_pairs_hyp_dist(z)

    hidden_sets = {v: np.where(hidden_mask(G, v, OBS_RADIUS))[0] for v in range(N)}

    per_sigma_errs = {s: [] for s in SIGMAS}
    per_sigma_eps = {s: [] for s in SIGMAS}

    rng = np.random.RandomState(seed + 1000)
    for sigma in SIGMAS:
        if sigma == 0.0:
            n_rep = 1
        else:
            n_rep = N_PERTURB
        for rep in range(n_rep):
            if sigma == 0.0:
                z_pert = z.clone()
            else:
                noise = torch.tensor(rng.normal(0, sigma, size=z.shape))
                z_pert = project_to_ball(z + noise)
            d_g = all_pairs_hyp_dist(z_pert)

            # sample subset of observers for speed (all if small)
            observers = range(N)
            mu_errs = []
            eps_realized = []
            for v in observers:
                H = hidden_sets[v]
                if len(H) == 0:
                    continue
                mu_f = mu_H(v, H, d_f[v], DELTA)
                mu_g = mu_H(v, H, d_g[v], DELTA)
                mu_errs.append(abs(mu_f - mu_g))
                eps_realized.append((d_f[v, H] - d_g[v, H]).abs().max().item())
            per_sigma_errs[sigma].append(float(np.mean(mu_errs)))
            per_sigma_eps[sigma].append(float(np.mean(eps_realized)))

    result = {}
    for s in SIGMAS:
        result[s] = {
            "mean_descriptor_error": float(np.mean(per_sigma_errs[s])),
            "std_descriptor_error": float(np.std(per_sigma_errs[s])),
            "mean_realized_epsilon": float(np.mean(per_sigma_eps[s])),
        }
    # mean |H| across observers for this graph, for the bound overlay delta*|H|*eps
    mean_H_size = float(np.mean([len(hidden_sets[v]) for v in range(N)]))
    return result, mean_H_size


def main():
    all_results = {s: {"errs": [], "eps": []} for s in SIGMAS}
    mean_H_sizes = []
    for seed in range(N_SEEDS):
        print(f"[Panel A] seed {seed}...")
        res, mean_H = run_one_graph(seed)
        mean_H_sizes.append(mean_H)
        for s in SIGMAS:
            all_results[s]["errs"].append(res[s]["mean_descriptor_error"])
            all_results[s]["eps"].append(res[s]["mean_realized_epsilon"])

    output = {"delta": DELTA, "n_seeds": N_SEEDS, "obs_radius": OBS_RADIUS,
              "mean_hidden_set_size": float(np.mean(mean_H_sizes)),
              "sigmas": SIGMAS, "points": []}
    for s in SIGMAS:
        errs = np.array(all_results[s]["errs"])
        eps = np.array(all_results[s]["eps"])
        output["points"].append({
            "sigma": s,
            "mean_descriptor_error": float(errs.mean()),
            "std_descriptor_error": float(errs.std()),
            "mean_realized_epsilon": float(eps.mean()),
            "std_realized_epsilon": float(eps.std()),
            "theoretical_bound_delta_H_eps": float(DELTA * np.mean(mean_H_sizes) * eps.mean()),
        })

    with open(str(_HERE / "results") + "/panelA_embedding_perturbation.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved results/panelA_embedding_perturbation.json")


if __name__ == "__main__":
    import os
    os.makedirs(str(_HERE / "results"), exist_ok=True)
    main()
