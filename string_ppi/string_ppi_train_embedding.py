"""
Run as: python string_ppi/string_ppi_train_embedding.py

Downloads the human STRING protein-protein interaction network (species 9606),
filters by combined confidence score, builds a graph, and runs the curvature search.

STRING has no natural node features (it's just protein IDs + interaction scores), so
this uses log-degree + a random projection as a simple placeholder feature vector.
Input features are log(1 + degree) and 63 i.i.d. N(0, 1) features (seed 0); they are not
biologically meaningful. Edges: combined score >= 700.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import sys, os, gzip, io


import requests
import torch
import networkx as nx
from curvature_search import run_curvature_search

STRING_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
CONFIDENCE_THRESHOLD = 700  # STRING combined_score is 0-1000; 700+ is "high confidence"


def download_string_ppi(data_dir=str(_HERE / "data")):
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "9606.protein.links.v12.0.txt.gz")
    if not os.path.exists(local_path):
        print(f"Downloading {STRING_URL} ...")
        r = requests.get(STRING_URL, timeout=120)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    else:
        print("Using cached STRING download.")
    return local_path


def build_graph(local_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    G = nx.Graph()
    with gzip.open(local_path, "rt") as f:
        header = f.readline()  # "protein1 protein2 combined_score"
        for line in f:
            p1, p2, score = line.strip().split(" ")
            score = int(score)
            if score >= confidence_threshold:
                G.add_edge(p1, p2, weight=score)
    return G


def to_pyg_data(G, feat_dim=64, seed=0):
    import torch_geometric
    from torch_geometric.data import Data
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

    # Placeholder features: log-degree broadcast + random projection (fixed seed for
    # reproducibility). Replace with real biological features if available.
    torch.manual_seed(seed)
    degrees = torch.tensor([G.degree(n) for n in nodes], dtype=torch.float).unsqueeze(1)
    log_deg = torch.log1p(degrees)
    random_feats = torch.randn(len(nodes), feat_dim - 1)
    x = torch.cat([log_deg, random_feats], dim=1)

    data = Data(x=x, edge_index=edge_index)
    return data, nodes  # nodes list needed later to map back to protein IDs for enrichment


if __name__ == "__main__":
    local_path = download_string_ppi(str(_HERE / "data"))
    print("Parsing STRING network (this can take a minute for the full human network)...")
    G = build_graph(local_path)
    print(f"STRING PPI (confidence >= {CONFIDENCE_THRESHOLD}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    data, node_list = to_pyg_data(G)
    torch.save(node_list, str(_HERE / "data" / "node_list.pt"))  # save protein ID <-> index mapping for later enrichment analysis

    results, best_c, best_model, best_z = run_curvature_search(
        data,
        in_dim=data.x.size(1),
        dataset_name="string_ppi",
        out_dir=str(_HERE / "hvgae_embedding"),
        epochs=300,
        verbose=True,
    )

    print("\nAll candidates:")
    for r in results:
        print(f"  c={r['curvature']:<6} val_loss={r['val_loss']:.4f}  AUC={r['val_auc']:.4f}  AP={r['val_ap']:.4f}")
    print(f"\nChosen curvature: {best_c}")
