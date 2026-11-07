"""
Run as: python obgn_arxiv/ogbn_arxiv_train_embedding.py

Requires: pip install ogb

NOTE ON SCALE: ogbn-arxiv has 169,343 nodes and 1,166,243 edges. Full-batch training
(as used here, same as Cora/Citeseer) may be tight on 6GB VRAM depending on hidden_dim.
If you hit CUDA OOM:
  1. First try device="cpu" just to confirm correctness (slow but will work).
  2. Then reduce hidden_dim (e.g. 64 -> 32) in the run_curvature_search call below.
  3. If still tight, the real fix is neighbor-sampling minibatches (torch_geometric.
     loader.NeighborLoader) instead of full-batch -- that's a bigger rewrite of
     common_hvgae.py's forward pass, so only do it if steps 1-2 aren't enough.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys


import torch
import torch_geometric

# PyTorch 2.6+ changed torch.load's default to weights_only=True, which breaks the
# ogb package's internal loading of cached datasets (it wasn't written with this in
# mind). Allowlist the specific torch_geometric classes ogb needs to unpickle.
torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])

from ogb.nodeproppred import PygNodePropPredDataset
from curvature_search import run_curvature_search

if __name__ == "__main__":
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(_HERE / "data"))
    data = dataset[0]
    # ogbn-arxiv is directed (citation direction) -- make undirected for embedding purposes,
    # consistent with how Cora/Citeseer are treated
    from torch_geometric.utils import to_undirected
    data.edge_index = to_undirected(data.edge_index)

    print(f"ogbn-arxiv: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results, best_c, best_model, best_z = run_curvature_search(
        data,
        in_dim=data.x.size(1),
        dataset_name="ogbn_arxiv",
        out_dir=str(_HERE / "hvgae_embedding"),
        epochs=300,
        device=device,
        verbose=True,
    )

    print("\nAll candidates:")
    for r in results:
        print(f"  c={r['curvature']:<6} val_loss={r['val_loss']:.4f}  AUC={r['val_auc']:.4f}  AP={r['val_ap']:.4f}")
    print(f"\nChosen curvature: {best_c}")
