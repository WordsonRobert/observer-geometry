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
ALPHAS = [200, 100, 50, 20, 10, 5]
N_SAMPLES_PER_ALPHA = 40
N_SEEDS = 5
N_OBSERVERS_PER_GRAPH = 15
MAX_RADIUS_SUPPORT = 6  # budget "time" = observation radius t = 0..MAX_RADIUS_SUPPORT


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d_f = all_pairs_hyp_dist(z)

    rng = np.random.RandomState(seed + 3000)
    observers = rng.choice(N, size=min(N_OBSERVERS_PER_GRAPH, N), replace=False)

    T = MAX_RADIUS_SUPPORT + 1  # number of budget points {0,...,MAX_RADIUS_SUPPORT}
    P = np.ones(T) / T  # uniform base budget distribution

    points = []  # (l1_dist, descriptor_diff, J_V_for_this_observer)

    for v in observers:
        # J(V) for this observer = mu_V(v), i.e. hidden = everything except v itself (H = V \ {v})
        all_but_v = np.array([u for u in range(N) if u != v])
        J_V = torch.exp(-DELTA * d_f[v, all_but_v]).sum().item()

        # precompute mu_{H(v,t)}(v) for t = 0..MAX_RADIUS_SUPPORT, H(v,t) = V \ B_t(v)
        mu_by_t = []
        for t in range(T):
            ball = set(nx.single_source_shortest_path_length(G, int(v), cutoff=t).keys())
            hidden = np.array([u for u in range(N) if u not in ball])
            mu_t = torch.exp(-DELTA * d_f[v, hidden]).sum().item() if len(hidden) else 0.0
            mu_by_t.append(mu_t)
        mu_by_t = np.array(mu_by_t)

        Phi_P = float((P * mu_by_t).sum())

        for alpha in ALPHAS:
            for _ in range(N_SAMPLES_PER_ALPHA):
                Q = rng.dirichlet(alpha * P)
                l1 = float(np.abs(P - Q).sum())
                Phi_Q = float((Q * mu_by_t).sum())
                descriptor_diff = abs(Phi_P - Phi_Q)
                points.append({"alpha": alpha, "l1_dist": l1, "descriptor_diff": descriptor_diff, "J_V": J_V})

    return points


def main():
    all_points = []
    J_V_values = []
    for seed in range(N_SEEDS):
        print(f"[Panel C] seed {seed}...")
        pts = run_one_graph(seed)
        all_points.extend(pts)
        J_V_values.extend([p["J_V"] for p in pts])

    # aggregate per-alpha summary too, plus raw scatter points (subsampled if huge)
    per_alpha = {}
    for a in ALPHAS:
        sub = [p for p in all_points if p["alpha"] == a]
        l1s = np.array([p["l1_dist"] for p in sub])
        diffs = np.array([p["descriptor_diff"] for p in sub])
        per_alpha[a] = {
            "mean_l1_dist": float(l1s.mean()),
            "std_l1_dist": float(l1s.std()),
            "mean_descriptor_diff": float(diffs.mean()),
            "std_descriptor_diff": float(diffs.std()),
            "n": int(len(sub)),
        }

    mean_J_V = float(np.mean(J_V_values))
    max_J_V = float(np.max(J_V_values))

    output = {
        "delta": DELTA,
        "n_seeds": N_SEEDS,
        "alphas": ALPHAS,
        "budget_support_size": MAX_RADIUS_SUPPORT + 1,
        "mean_J_V_over_observers": mean_J_V,
        "max_J_V_over_observers": max_J_V,
        "note": "J(V) = mu_V(v) is observer-dependent; both mean and max over pooled observers reported for the bound overlay y = J(V) * x",
        "per_alpha_summary": per_alpha,
        "scatter_points": all_points[:2000],  # cap for file size, still representative
    }

    with open(str(_HERE / "results") + "/panelC_budget_perturbation.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved results/panelC_budget_perturbation.json")


if __name__ == "__main__":
    import os
    os.makedirs(str(_HERE / "results"), exist_ok=True)
    main()
