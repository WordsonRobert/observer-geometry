"""
Run from ~/projects/niggtract2/embedding_ablation/

Produces Phi for:
  1. HVGAE (reused from cora/hvgae_embedding/, the main paper embedding)
  2. Shallow Poincare direct embedding
  3. HGCN-lite encoder
  4. Euclidean GraphSAGE baseline (no hyperbolic mapping)
  5. Random embedding (no training)
  6. HVGAE embedding + EUCLIDEAN kernel (isolates whether hyperbolic DISTANCE matters,
     holding the embedding itself fixed -- experiment 5)
  7. HVGAE retrained at each of 6 fixed curvatures (experiment 2)

IMPORTANT: set K_DEFAULT below to match the plateau K* you already found for Cora's
main analysis (cora/analysis.py's printed "Detected plateau K*") -- this should be
the SAME K used for the main paper's Cora results, so the ablation is comparing
Phi computed consistently, not introducing a second free parameter.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys, json


import torch
from torch_geometric.datasets import Planetoid
import geoopt

from simulate import HBPSimulator
from phi_utils import build_and_cache_kernel, phi_at_K_all_nodes, weights_by_distance
from embedding_variants import (train_poincare_direct, train_hgcn_lite,
                                 train_euclidean_sage, make_random_embedding, EuclideanDist)

# ---- config ----
K_DEFAULT = 50  # <-- SET THIS to Cora's actual plateau K* before treating results as final
EPS_DEFAULT = 1 / 50
DELTA_DEFAULT = 1.0
N_REPEATS = 3
EMBED_DIM = 16


def compute_phi_for_embedding(data, z, dist_wrapper, N, name):
    print(f"\nComputing Phi for: {name}")
    row_ptr, col_idx, edge_weight = build_and_cache_kernel(
        data.edge_index, z, dist_wrapper, DELTA_DEFAULT, N,
        cache_path=str(_HERE / "embeddings") + f"/kernel_{name}.pt")
    sim = HBPSimulator(row_ptr, col_idx, edge_weight)
    torch.manual_seed(0)
    phi = phi_at_K_all_nodes(sim, N, z, dist_wrapper, K_DEFAULT, EPS_DEFAULT, DELTA_DEFAULT,
                              n_repeats=N_REPEATS)
    torch.save(phi, str(_HERE / "embeddings") + f"/phi_{name}.pt")
    return phi


def main():
    ds = Planetoid(root=str(_REPO / "cora" / "data"), name="Cora")
    data = ds[0]
    N = data.num_nodes

    # ---------------- 1. HVGAE (main embedding, reused) ----------------
    z_hvgae = torch.load(str(_REPO / "cora" / "hvgae_embedding") + "/embedding.pt", map_location="cpu")
    with open(str(_REPO / "cora" / "hvgae_embedding") + "/chosen_curvature.txt") as f:
        c_hvgae = float(f.read().strip())
    manifold_hvgae = geoopt.Lorentz(k=torch.tensor(c_hvgae))
    compute_phi_for_embedding(data, z_hvgae, manifold_hvgae, N, "hvgae_main")

    # ---------------- 6. Same HVGAE embedding, EUCLIDEAN kernel (experiment 5) ----------------
    # holds the embedding fixed, swaps only the distance function -- isolates whether
    # hyperbolic distance itself (not just the embedding) is doing the work
    compute_phi_for_embedding(data, z_hvgae, EuclideanDist(), N, "hvgae_euclidean_kernel")

    # ---------------- 2. Shallow Poincare direct ----------------
    z_poincare, ball_poincare = train_poincare_direct(data, dim=EMBED_DIM, curvature=c_hvgae, epochs=300)
    compute_phi_for_embedding(data, z_poincare, ball_poincare, N, "poincare_direct")

    # ---------------- 3. HGCN-lite ----------------
    z_hgcn, ball_hgcn = train_hgcn_lite(data, in_dim=data.x.size(1), dim=EMBED_DIM,
                                         curvature=c_hvgae, epochs=300)
    compute_phi_for_embedding(data, z_hgcn, ball_hgcn, N, "hgcn_lite")

    # ---------------- 4. Euclidean GraphSAGE baseline ----------------
    z_euclidean, dist_euclidean = train_euclidean_sage(data, in_dim=data.x.size(1), dim=EMBED_DIM, epochs=300)
    compute_phi_for_embedding(data, z_euclidean, dist_euclidean, N, "euclidean_sage")

    # ---------------- 5. Random embedding ----------------
    z_random, dist_random = make_random_embedding(N, dim=EMBED_DIM, seed=0)
    compute_phi_for_embedding(data, z_random, dist_random, N, "random")

    print("\nAll variants done. Run compare_results.py next.")


if __name__ == "__main__":
    main()
