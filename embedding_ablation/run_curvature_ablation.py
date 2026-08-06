"""
Run from ~/projects/niggtract2/embedding_ablation/ AFTER run_ablation.py.

Experiment 2: retrain HVGAE at each of 6 fixed curvatures (no model selection --
this is different from curvature_search.py, which picks ONE best curvature for the
main paper results; here we KEEP all 6, to directly test how much Phi's ranking
depends on the curvature choice).
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys


import torch
from torch_geometric.datasets import Planetoid

from common_hvgae import HVGAE, negative_sampling
from run_ablation import compute_phi_for_embedding

CANDIDATE_CURVATURES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def train_hvgae_fixed_curvature(data, in_dim, curvature, epochs=300, lr=1e-3):
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
            print(f"    c={curvature} epoch {epoch}: loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        z_final = model.encode(data.x, data.edge_index)
    return z_final, model.manifold()


def main():
    ds = Planetoid(root=str(_REPO / "cora" / "data"), name="Cora")
    data = ds[0]
    N = data.num_nodes

    for c in CANDIDATE_CURVATURES:
        print(f"\n--- Training HVGAE at fixed curvature c={c} ---")
        z, manifold = train_hvgae_fixed_curvature(data, data.x.size(1), curvature=c)
        compute_phi_for_embedding(data, z, manifold, N, f"hvgae_curvature_{c}")

    print("\nAll curvature variants done. Run compare_results.py next.")


if __name__ == "__main__":
    main()
