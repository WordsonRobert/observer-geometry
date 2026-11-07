"""phi_seed_variance.py
====================
Recomputes Phi under several independent simulator seeds at the same K* and reports, per
dataset, pairwise Spearman agreement between seeds, mean and maximum per-node standard
deviation, and the coefficient of variation. Writes sidecar files; phi_full_meanstd.pt is
not modified. (The exact evaluator, analysis/exact_phi_all.py, removes Monte Carlo error.)

Emits: <folder>/phi_seed_variance.json, results/phi_seed_variance.json
Usage:
    python analysis/phi_seed_variance.py --seeds 3
    python analysis/phi_seed_variance.py --datasets cora --seeds 5 --device cuda:0
"""
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from _common import (DISPLAY, MissingArtifact, RESULTS, folder, load_phi,
                     write_json, cli_datasets)


def arg(flag, default, cast=str):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def run_one(name, n_seeds, device):
    from run_everything import DATASET_CONFIG, get_graph
    from phi_utils import load_embedding
    from vectorized_simulate import vectorized_phi_at_K_all_nodes

    cfg = DATASET_CONFIG[name]
    f = folder(name)

    baseline, K_star = load_phi(name)
    if K_star is None:
        raise MissingArtifact(f"{name}: no K_star in phi_full_meanstd.pt")

    z, manifold, c = load_embedding(
        str(f / "hvgae_embedding" / "embedding.pt"),
        str(f / "hvgae_embedding" / "chosen_curvature.txt"))

    G, N, node_id_map, data = get_graph(name, cfg)
    if data is not None:
        edge_index = data.edge_index
    else:
        idx = {n: i for i, n in enumerate(node_id_map)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    print(f"\n=== phi seed variance: {DISPLAY[name]} (K*={K_star}, N={N}) ===")
    phis = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  seed {seed} ...")
        p = vectorized_phi_at_K_all_nodes(
            edge_index, z, manifold, K=K_star, eps=1 / 50, delta=1.0,
            num_nodes=N, n_repeats=3,
            source_batch_size=cfg["phi_batch_size"], device=device,
            max_degree_cap=cfg["max_degree_cap"])
        phis.append(p.numpy().astype(np.float64))

    P = np.vstack(phis)                       # [S, N]
    per_node_std = P.std(axis=0)
    per_node_mean = P.mean(axis=0)

    pair_rho = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            pair_rho.append(float(spearmanr(P[i], P[j])[0]))

    rho_vs_baseline = [float(spearmanr(P[i], baseline)[0]) for i in range(n_seeds)]

    denom = np.where(per_node_mean > 1e-12, per_node_mean, np.nan)
    cv = np.nanmean(per_node_std / denom)

    out = {
        "dataset": DISPLAY[name],
        "K_star": int(K_star),
        "n_nodes": int(N),
        "n_seeds": n_seeds,
        "pairwise_spearman": pair_rho,
        "pairwise_spearman_mean": float(np.mean(pair_rho)) if pair_rho else None,
        "pairwise_spearman_min": float(np.min(pair_rho)) if pair_rho else None,
        "spearman_vs_cached_phi": rho_vs_baseline,
        "per_node_std_mean": float(per_node_std.mean()),
        "per_node_std_max": float(per_node_std.max()),
        "per_node_mean_of_phi": float(per_node_mean.mean()),
        "coefficient_of_variation": float(cv),
    }
    write_json(out, folder(name) / "phi_seed_variance.json")

    # sidecar tensor -- does NOT overwrite phi_full_meanstd.pt
    torch.save({"mean": torch.tensor(per_node_mean),
                "std": torch.tensor(per_node_std),
                "seeds": list(range(n_seeds)),
                "K_star": K_star},
               folder(name) / "hvgae_embedding" / "phi_seed_variance.pt")

    print(f"  pairwise rho: mean={out['pairwise_spearman_mean']:.4f} "
          f"min={out['pairwise_spearman_min']:.4f}  CV={cv:.4f}")
    return out


def main():
    names = cli_datasets(sys.argv)
    n_seeds = arg("--seeds", 3, int)
    device = arg("--device", "cuda:1", str)

    pooled = {}
    for name in names:
        try:
            pooled[name] = run_one(name, n_seeds, device)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            pooled[name] = {"error": f"{type(e).__name__}: {e}"}

    write_json(pooled, RESULTS / "phi_seed_variance.json")


if __name__ == "__main__":
    main()
