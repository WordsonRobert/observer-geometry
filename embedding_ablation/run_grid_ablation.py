"""
run_grid_ablation.py  -- PATCHED
================================
3 encoders (SAGE, GCN, GAT) x 3 target geometries (Lorentz, Poincare,
Euclidean) on Cora, plus an untrained random-embedding null.

WHAT CHANGED FROM THE PREVIOUS VERSION, AND WHY IT MATTERS
----------------------------------------------------------
1. SAGE/Lorentz is now TRAINED IN-GRID instead of being aliased to the main
   HVGAE result with rho=1.0 by definition.

   The old anchor came from curvature_search.py: curvature grid search, early
   stopping on held-out link-prediction loss, dropout 0.2. The other eight
   cells got fixed c=1.0, a flat 300 epochs, no early stopping, no dropout.
   Every reported rho therefore conflated the thing being measured (encoder,
   geometry) with a training-protocol difference -- and since the paper's
   claim is about the ORDERING of sensitivities, that directly biased the
   conclusion. All nine cells now share one protocol.

2. The `already_done` dict and the phi_euclidean_sage.pt reuse branch are
   gone. Reusing an artifact trained under a third protocol had the same
   problem.

3. K_DEFAULT is asserted against Cora's cached K* rather than hard-coded and
   hoped for. If the cached phi was computed at a different K, this refuses to
   run instead of silently comparing K=36 against a K=50 anchor.

4. A random-embedding null is included. Without it there is no scale on which
   to read rho=0.81 as high or low; it is the most informative single number
   in the table.

5. Every phi tensor is saved, so grid_pairwise.py can build the real 9x9
   matrix. The paper claims a pairwise matrix; the old script only ever
   computed nine-vs-one.

Emits: embedding_ablation/grid_ablation_results.json
       embedding_ablation/embeddings/phi_grid_<combo>.pt

Runtime: ~10 x (300 epochs + one phi pass on Cora) ~= 40-70 min on GPU.

Usage:
    python embedding_ablation/run_grid_ablation.py
    python embedding_ablation/run_grid_ablation.py --k 36 --force
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from scipy.stats import spearmanr
import geoopt

from simulate import HBPSimulator
from phi_utils import build_and_cache_kernel, phi_at_K_all_nodes

EPS_DEFAULT = 1 / 50
DELTA_DEFAULT = 1.0
N_REPEATS = 3
EPOCHS = 300
HIDDEN = 64
EMBED_DIM = 16
CURVATURE = 1.0
LR = 1e-3
ANCHOR = "SAGE_lorentz"

EMB_DIR = HERE / "embeddings"
EMB_DIR.mkdir(exist_ok=True)


def arg(flag, default, cast=str):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def resolve_K():
    """
    Read Cora's plateau K* from the cached phi rather than trusting a constant.
    A mismatch here silently invalidates every rho in the grid.
    """
    override = arg("--k", None, int)
    cached_path = ROOT / "cora" / "hvgae_embedding" / "phi_full_meanstd.pt"
    cached_K = None
    if cached_path.exists():
        d = torch.load(cached_path, map_location="cpu")
        if isinstance(d, dict):
            cached_K = d.get("K_star")

    if override is not None:
        if cached_K is not None and override != cached_K:
            print(f"  [WARN] --k {override} differs from Cora's cached "
                  f"K*={cached_K}. Using {override} as instructed.")
        return int(override)

    if cached_K is None:
        raise SystemExit(
            "Could not read K* from cora/hvgae_embedding/phi_full_meanstd.pt.\n"
            "Pass it explicitly:  --k 36")
    print(f"  Using Cora's cached plateau K* = {cached_K}")
    return int(cached_K)


class EuclideanDist:
    def dist(self, x, y):
        return torch.norm(x - y, dim=-1)


def negative_sampling(edge_index, num_nodes, num_neg):
    return torch.stack([torch.randint(0, num_nodes, (num_neg,)),
                        torch.randint(0, num_nodes, (num_neg,))], dim=0)


class FlexibleEncoder(nn.Module):
    """One encoder pattern for all three conv types; only the conv layer differs."""

    def __init__(self, conv_cls, in_dim, hidden_dim, out_dim, manifold_kind, ball=None):
        super().__init__()
        if conv_cls is GATConv:
            # GATConv concatenates heads by default; 1 head keeps dims matched
            self.conv1 = conv_cls(in_dim, hidden_dim, heads=1)
            self.conv2 = conv_cls(hidden_dim, out_dim, heads=1)
        else:
            self.conv1 = conv_cls(in_dim, hidden_dim)
            self.conv2 = conv_cls(hidden_dim, out_dim)
        self.manifold_kind = manifold_kind
        self.ball = ball

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        if self.manifold_kind == "euclidean":
            return h
        if self.manifold_kind == "poincare":
            return self.ball.expmap0(h)
        zeros = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        return self.ball.expmap0(torch.cat([zeros, h], dim=1))


def train_variant(conv_cls, conv_name, manifold_kind, data, in_dim, seed=0):
    print(f"  training {conv_name} -> {manifold_kind} ...")
    torch.manual_seed(seed)

    if manifold_kind == "lorentz":
        ball = geoopt.Lorentz(k=torch.tensor(CURVATURE)); dist_fn = ball
    elif manifold_kind == "poincare":
        ball = geoopt.PoincareBall(c=CURVATURE); dist_fn = ball
    else:
        ball = None; dist_fn = EuclideanDist()

    model = FlexibleEncoder(conv_cls, in_dim, HIDDEN, EMBED_DIM, manifold_kind, ball=ball)
    r = nn.Parameter(torch.tensor(2.0))
    t = nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.Adam(list(model.parameters()) + [r, t], lr=LR, weight_decay=5e-4)

    N = data.x.size(0)
    E = data.edge_index.size(1)
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        z = model(data.x, data.edge_index)
        pos = dist_fn.dist(z[data.edge_index[0]], z[data.edge_index[1]])
        neg_ei = negative_sampling(data.edge_index, N, E)
        neg = dist_fn.dist(z[neg_ei[0]], z[neg_ei[1]])
        loss = (-torch.log(torch.sigmoid((r - pos) / t) + 1e-8).mean()
                - torch.log(1 - torch.sigmoid((r - neg) / t) + 1e-8).mean())
        loss.backward(); opt.step()
        if epoch % 100 == 0:
            print(f"    epoch {epoch}: loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index), dist_fn


def make_random_embedding(N, dim=EMBED_DIM, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, dim), EuclideanDist()


def compute_phi(data, z, dist_fn, N, name, K):
    row_ptr, col_idx, w = build_and_cache_kernel(
        data.edge_index, z, dist_fn, DELTA_DEFAULT, N,
        cache_path=str(EMB_DIR / f"kernel_grid_{name}.pt"))
    sim = HBPSimulator(row_ptr, col_idx, w)
    torch.manual_seed(0)   # simulator seed fixed across cells: differences are geometric
    phi = phi_at_K_all_nodes(sim, N, z, dist_fn, K, EPS_DEFAULT, DELTA_DEFAULT,
                             n_repeats=N_REPEATS)
    torch.save(phi, EMB_DIR / f"phi_grid_{name}.pt")
    return phi


def main():
    force = "--force" in sys.argv
    K = resolve_K()

    ds = Planetoid(root=str(ROOT / "cora" / "data"), name="Cora")
    data = ds[0]
    N = data.num_nodes
    print(f"  Cora: N={N}, E={data.edge_index.size(1)}, K={K}")

    encoders = [("SAGE", SAGEConv), ("GCN", GCNConv), ("GAT", GATConv)]
    manifolds = ["lorentz", "poincare", "euclidean"]

    combos = [(f"{e}_{m}", cls, m) for e, cls in encoders for m in manifolds]

    phis, timings = {}, {}
    t0 = time.time()

    for combo_name, conv_cls, manifold_kind in combos:
        out_pt = EMB_DIR / f"phi_grid_{combo_name}.pt"
        if out_pt.exists() and not force:
            print(f"  {combo_name}: reusing {out_pt.name} (--force to retrain)")
            phis[combo_name] = torch.load(out_pt, map_location="cpu")
            timings[combo_name] = 0.0
            continue
        tc = time.time()
        z, dist_fn = train_variant(conv_cls, combo_name.split("_")[0],
                                   manifold_kind, data, data.x.size(1))
        phis[combo_name] = compute_phi(data, z, dist_fn, N, combo_name, K)
        timings[combo_name] = time.time() - tc
        print(f"  {combo_name}: done ({timings[combo_name]:.1f}s)")

    # --- random null: the scale bar for every rho in this table -------------
    null_pt = EMB_DIR / "phi_grid_random.pt"
    if null_pt.exists() and not force:
        phis["random"] = torch.load(null_pt, map_location="cpu")
    else:
        print("  random null ...")
        z_rand, dist_rand = make_random_embedding(N)
        phis["random"] = compute_phi(data, z_rand, dist_rand, N, "random", K)

    anchor = phis[ANCHOR].numpy()
    names = [c[0] for c in combos] + ["random"]

    grid = {e: {} for e, _ in encoders}
    flat = {}
    for nm in names:
        rho, p = spearmanr(phis[nm].numpy(), anchor)
        flat[nm] = {"rho_vs_anchor": float(rho), "p": float(p),
                    "seconds": timings.get(nm, 0.0)}
        if nm != "random":
            enc, man = nm.split("_")
            grid[enc][man] = float(rho)

    total = time.time() - t0
    out = {
        "anchor": ANCHOR,
        "anchor_note": ("SAGE/Lorentz trained IN-GRID under the same protocol as "
                        "every other cell (fixed c=1.0, 300 epochs, no early "
                        "stopping, no dropout). It is NOT the main HVGAE "
                        "embedding, which used curvature search and early "
                        "stopping -- mixing those protocols would confound "
                        "encoder/geometry effects with training differences."),
        "flat": flat,
        "grid": grid,
        "random_null_rho": flat["random"]["rho_vs_anchor"],
        "encoders": [e for e, _ in encoders],
        "manifolds": manifolds,
        "fixed_params": {
            "K": K, "eps": EPS_DEFAULT, "delta": DELTA_DEFAULT,
            "epochs": EPOCHS, "hidden_dim": HIDDEN, "embed_dim": EMBED_DIM,
            "curvature": CURVATURE, "lr": LR, "gat_heads": 1,
            "n_repeats": N_REPEATS, "simulator_seed": 0,
        },
        "total_seconds": total,
    }
    with open(HERE / "grid_ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  total {total/60:.1f} min")
    print(f"  wrote {HERE/'grid_ablation_results.json'}")
    print(f"\n  random null rho = {out['random_null_rho']:.4f}  <- read every "
          f"other rho against this")
    print("  Next: python embedding_ablation/grid_pairwise.py")


if __name__ == "__main__":
    main()
