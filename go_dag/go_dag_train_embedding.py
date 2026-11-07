"""
Run as: python go_dag/go_dag_train_embedding.py

Requires: pip install goatools

Downloads go-basic.obo, extracts the Biological Process namespace, builds the DAG
(is_a relationships), treats it as undirected for embedding purposes (same convention
as ogbn-arxiv), and runs the curvature search.

Also saves node_depths.pt -- shortest-path depth from the BP root for every term, since
used by the hierarchy validation. GO is a DAG (terms can have several parents); depth is
the SHORTEST-PATH depth from the root, so multi-parent terms take their shallowest depth.
Only primary term identifiers become vertices (goatools also keys terms by alternate ids).
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys, os


import requests
import torch
import networkx as nx
from curvature_search import run_curvature_search

GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
BP_ROOT = "GO:0008150"  # "biological_process" root term


def download_obo(data_dir=str(_HERE / "data")):
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "go-basic.obo")
    if not os.path.exists(local_path):
        print(f"Downloading {GO_OBO_URL} ...")
        r = requests.get(GO_OBO_URL, timeout=120)
        r.raise_for_status()
        with open(local_path, "w") as f:
            f.write(r.text)
    else:
        print("Using cached OBO file.")
    return local_path


def build_bp_dag(obo_path):
    from goatools.obo_parser import GODag
    godag = GODag(obo_path)

    G = nx.DiGraph()
    for go_id, term in godag.items():
        if term.namespace != "biological_process":
            continue
        if go_id != term.item_id:
            continue  # goatools also keys each term by its alt_ids; skip those duplicates
        G.add_node(go_id, name=term.name)
        for parent in term.parents:
            if parent.namespace == "biological_process":
                G.add_edge(go_id, parent.id)  # child -> parent

    return G


def to_pyg_data(G, feat_dim=64, seed=0):
    from torch_geometric.data import Data
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Placeholder features (GO terms have no natural numeric features): random + degree
    torch.manual_seed(seed)
    degrees = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float).unsqueeze(1)
    log_deg = torch.log1p(degrees)
    random_feats = torch.randn(len(nodes), feat_dim - 1)
    x = torch.cat([log_deg, random_feats], dim=1)

    data = Data(x=x, edge_index=edge_index_undirected)
    return data, nodes, idx


def compute_depths(G, root, nodes, idx):
    """Shortest-path depth from root, using the DAG in child->parent direction reversed
    (i.e. compute shortest path FROM root TO each node by following parent->child edges)."""
    G_rev = G.reverse()  # now edges point parent -> child
    if root not in G_rev:
        raise ValueError(f"Root {root} not found in graph -- check namespace filtering.")
    lengths = nx.single_source_shortest_path_length(G_rev, root)
    depths = torch.full((len(nodes),), -1, dtype=torch.long)  # -1 = unreachable from root
    for node, d in lengths.items():
        if node in idx:
            depths[idx[node]] = d
    return depths


if __name__ == "__main__":
    obo_path = download_obo(str(_HERE / "data"))
    print("Parsing GO DAG and extracting biological_process namespace...")
    G = build_bp_dag(obo_path)
    print(f"GO Biological Process DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    data, node_list, idx = to_pyg_data(G)
    depths = compute_depths(G, BP_ROOT, node_list, idx)
    torch.save(node_list, str(_HERE / "data" / "node_list.pt"))
    torch.save(depths, str(_HERE / "data" / "node_depths.pt"))
    print(f"Saved depths (shortest-path from {BP_ROOT}); "
          f"{(depths == -1).sum().item()} nodes unreachable from root (check namespace filtering if this is large)")

    results, best_c, best_model, best_z = run_curvature_search(
        data,
        in_dim=data.x.size(1),
        dataset_name="go_dag",
        out_dir=str(_HERE / "hvgae_embedding"),
        epochs=300,
        verbose=True,
    )

    print("\nAll candidates:")
    for r in results:
        print(f"  c={r['curvature']:<6} val_loss={r['val_loss']:.4f}  AUC={r['val_auc']:.4f}  AP={r['val_ap']:.4f}")
    print(f"\nChosen curvature: {best_c}")
