"""
Run from ~/projects/niggtract2/cora/  (or adjust the sys.path line below).

Downloads Cora into ./data/ (cached, only downloads once), runs the 6-curvature
HVGAE search, saves results + chosen embedding into ./hvgae_embedding/.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys


from torch_geometric.datasets import Planetoid
from curvature_search import run_curvature_search

if __name__ == "__main__":
    ds = Planetoid(root=str(_HERE / "data"), name="Cora")
    data = ds[0]
    print(f"Cora: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    results, best_c, best_model, best_z = run_curvature_search(
        data,
        in_dim=data.x.size(1),
        dataset_name="cora",
        out_dir=str(_HERE / "hvgae_embedding"),
        epochs=300,
        verbose=True,
    )

    print("\nAll candidates:")
    for r in results:
        print(f"  c={r['curvature']:<6} val_loss={r['val_loss']:.4f}  AUC={r['val_auc']:.4f}  AP={r['val_ap']:.4f}")
    print(f"\nChosen curvature: {best_c}")
