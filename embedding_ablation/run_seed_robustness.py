"""
Run from ~/projects/niggtract2/embedding_ablation/ AFTER run_ablation.py.

Trains HVGAE 5 times at the SAME chosen curvature (Cora's already-selected c),
different random seeds only. Computes Phi for each, builds the 5x5 Spearman
correlation matrix. Answers: does optimization noise (not geometry, not encoder
choice) meaningfully affect Phi's rankings?
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys, json


import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch_geometric.datasets import Planetoid

from common_hvgae import HVGAE, negative_sampling
from run_ablation import compute_phi_for_embedding

SEEDS = [0, 1, 2]


def train_hvgae_seeded(data, in_dim, curvature, seed, epochs=300, lr=1e-3):
    torch.manual_seed(seed)
    model = HVGAE(in_dim, 64, 16, curvature=curvature)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    N = data.x.size(0)
    num_edges = data.edge_index.size(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pos_prob = model.decode_edges(z, data.edge_index)
        neg_edge_index = negative_sampling(data.edge_index, N, num_edges)
        neg_prob = model.decode_edges(z, neg_edge_index)
        loss = (-torch.log(pos_prob + 1e-8).mean() - torch.log(1 - neg_prob + 1e-8).mean())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"    seed={seed} epoch {epoch}: loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        z_final = model.encode(data.x, data.edge_index)
    return z_final, model.manifold()


def main():
    ds = Planetoid(root=str(_REPO / "cora" / "data"), name="Cora")
    data = ds[0]
    N = data.num_nodes

    with open(str(_REPO / "cora" / "hvgae_embedding") + "/chosen_curvature.txt") as f:
        c_hvgae = float(f.read().strip())
    print(f"Using Cora's chosen curvature c={c_hvgae} for all seeds")

    for seed in SEEDS:
        print(f"\n--- Training HVGAE with seed={seed} ---")
        z, manifold = train_hvgae_seeded(data, data.x.size(1), c_hvgae, seed)
        compute_phi_for_embedding(data, z, manifold, N, f"hvgae_seed_{seed}")

    # correlation matrix across seeds
    phi_dict = {s: torch.load(str(_HERE / "embeddings") + f"/phi_hvgae_seed_{s}.pt") for s in SEEDS}
    mat = np.zeros((len(SEEDS), len(SEEDS)))
    for i, si in enumerate(SEEDS):
        for j, sj in enumerate(SEEDS):
            rho, _ = spearmanr(phi_dict[si].numpy(), phi_dict[sj].numpy())
            mat[i, j] = rho

    labels = [f"seed {s}" for s in SEEDS]
    print("\nSeed robustness correlation matrix (Spearman rho):")
    print(f"{'':10}" + "".join(f"{l:>10}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:10}" + "".join(f"{mat[i,j]:10.3f}" for j in range(len(SEEDS))))

    off_diag = mat[~np.eye(len(SEEDS), dtype=bool)]
    print(f"\nMean off-diagonal rho = {off_diag.mean():.4f} (std={off_diag.std():.4f})")
    print("This isolates OPTIMIZATION NOISE specifically -- same architecture, same")
    print("curvature, same data, only random init/training order differs.")

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                      color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=8)
    plt.title("HVGAE seed robustness: Spearman(Phi_seed_i, Phi_seed_j)")
    plt.tight_layout()
    plt.savefig("seed_robustness_correlation.png", dpi=150)
    plt.close()

    with open("seed_robustness_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "matrix": mat.tolist(),
                    "mean_off_diagonal": float(off_diag.mean()),
                    "std_off_diagonal": float(off_diag.std())}, f, indent=2)
    print("\nSaved seed_robustness_correlation.png, seed_robustness_results.json")


if __name__ == "__main__":
    main()
