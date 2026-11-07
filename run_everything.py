"""run_everything.py -- experimental pipeline for the five real-world graphs
=========================================================================
Regenerates the real-graph results of the paper (Section 6 and Appendix A) for Cora,
Citeseer, GO DAG, STRING-PPI and ogbn-arXiv from the cached embeddings
(<dataset>/hvgae_embedding/) and the raw data (<dataset>/data/). Embeddings and curvature
selection are fixed inputs; this script does not retrain them. The embedding-robustness
studies live in embedding_ablation/ and are not touched here.

Stages: phi, correlations, reconstruction, radius, classification, hierarchy, biology, ogd,
k_sweep, runtime. Each stage writes a completion marker to .checkpoint_<dataset>.json, and
reruns skip completed stages (--force to redo).

Phi is computed with the vectorized simulator (vectorized_simulate.py), which advances the K
i.i.d. trajectories synchronously; it agrees with the sequential reference simulator
(simulate.py) at Spearman 0.9963 (Cora) and 0.9977 (ogbn-arXiv). The exact evaluator in
exact_phi.py removes Monte Carlo error altogether (analysis/exact_phi_all.py).

Device: OG_DEVICE if set, else CUDA if available, else CPU.

Usage
    python run_everything.py                          # all datasets, all stages
    python run_everything.py --only cora citeseer
    python run_everything.py --stages phi radius
    python run_everything.py --force
    python run_everything.py --dry-run

Outputs (per dataset folder): correlation_summary.json, reconstruction_results.json,
radius_correlation.json, classification.json (citation graphs), hierarchy_validation.json
(GO DAG), biology_validation.json (STRING-PPI), ogd_experiment_results.json and
ogd_correlation_matrix.json (Cora, Citeseer), runtime_summary.json, k_sweep_full.json.

Notes
1. The biology stage downloads NCBI gene2go/gene_info on first use (10-20 min), then caches.
2. The GO DAG graph uses each term's primary identifier only; goatools also keys every term
   by its alternate identifiers, which must not become separate vertices.
"""


import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import argparse
import json
import time
import os
import sys
from pathlib import Path

import torch
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, kendalltau, pearsonr

sys.path.insert(0, ".")
from phi_utils import load_embedding, compute_radii
from vectorized_simulate import vectorized_phi_at_K_all_nodes
from reconstruction_utils import (
    compute_classical_features, run_reconstruction_experiment,
    run_mutual_information, run_feature_importance, FEATURE_NAMES,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PREFIX = ".checkpoint_"

ALL_STAGES = [
    "phi", "correlations", "reconstruction", "radius",
    "classification", "hierarchy", "biology", "ogd",
    "k_sweep", "runtime",
]


# ====================================================================================
# CHECKPOINTING
# ====================================================================================

def load_checkpoint(dataset_name):
    path = ROOT / f"{CHECKPOINT_PREFIX}{dataset_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(dataset_name, checkpoint):
    path = ROOT / f"{CHECKPOINT_PREFIX}{dataset_name}.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def stage_done(dataset_name, stage, force=False):
    if force:
        return False
    cp = load_checkpoint(dataset_name)
    return cp.get(stage, {}).get("done", False)


def mark_done(dataset_name, stage, elapsed=None, extra=None):
    cp = load_checkpoint(dataset_name)
    entry = {"done": True, "timestamp": time.time()}
    if elapsed is not None:
        entry["elapsed_seconds"] = elapsed
    if extra is not None:
        entry.update(extra)
    cp[stage] = entry
    save_checkpoint(dataset_name, cp)


# ====================================================================================
# DATASET LOADERS (graph structure + display name + reconstruction params)
# ====================================================================================

def load_cora():
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx
    ds = Planetoid(root="cora/data", name="Cora")
    data = ds[0]
    G = to_networkx(data, to_undirected=True)
    return G, data.num_nodes, None, data


def load_citeseer():
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx
    ds = Planetoid(root="citeseer/data", name="Citeseer")
    data = ds[0]
    G = to_networkx(data, to_undirected=True)
    return G, data.num_nodes, None, data


def load_go_dag():
    from goatools.obo_parser import GODag
    obo_path = "go_dag/data/go-basic.obo"
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"{obo_path} not found.")
    godag = GODag(obo_path)
    G_directed = nx.DiGraph()
    for go_id, term in godag.items():
        if term.namespace != "biological_process":
            continue
        if go_id != term.item_id:
            continue  # skip alt_id duplicate keys
        G_directed.add_node(go_id, name=term.name)
        for parent in term.parents:
            if parent.namespace == "biological_process":
                G_directed.add_edge(go_id, parent.id)
    node_list_path = "go_dag/data/node_list.pt"
    if not os.path.exists(node_list_path):
        raise FileNotFoundError(f"{node_list_path} not found.")
    node_list = torch.load(node_list_path)
    G = G_directed.to_undirected()
    return G, len(node_list), node_list, None


def load_arxiv():
    import torch_geometric
    from torch_geometric.utils import to_networkx, to_undirected
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ])
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="obgn_arxiv/data")
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.y = data.y.squeeze()
    G = to_networkx(data, to_undirected=True)
    return G, data.num_nodes, None, data


def load_string_ppi():
    node_list = torch.load("string_ppi/data/node_list.pt")
    pyg_path = "string_ppi/data/pyg_data.pt"
    if not os.path.exists(pyg_path):
        raise FileNotFoundError(f"{pyg_path} not found.")
    data = torch.load(pyg_path, weights_only=False)
    N = data.x.size(0)
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    G.add_nodes_from(range(N))
    return G, N, node_list, data


DATASET_CONFIG = {
    "cora": {
        "folder": "cora", "loader": load_cora, "display_name": "Cora",
        "betweenness_k": None, "closeness_harmonic_samples": None,
        "max_degree_cap": None, "phi_batch_size": 512,
    },
    "citeseer": {
        "folder": "citeseer", "loader": load_citeseer, "display_name": "Citeseer",
        "betweenness_k": None, "closeness_harmonic_samples": None,
        "max_degree_cap": None, "phi_batch_size": 512,
    },
    "go_dag": {
        "folder": "go_dag", "loader": load_go_dag, "display_name": "GO_DAG",
        "betweenness_k": 1000, "closeness_harmonic_samples": 1000,
        "max_degree_cap": 256, "phi_batch_size": 512,
    },
    "obgn_arxiv": {
        "folder": "obgn_arxiv", "loader": load_arxiv, "display_name": "ogbn_arxiv",
        "betweenness_k": 1000, "closeness_harmonic_samples": 1000,
        "max_degree_cap": 256, "phi_batch_size": 1024,
    },
    "string_ppi": {
        "folder": "string_ppi", "loader": load_string_ppi, "display_name": "STRING_PPI",
        "betweenness_k": 1000, "closeness_harmonic_samples": None,
        "max_degree_cap": 256, "phi_batch_size": 1024,
    },
}

_graph_cache = {}


def _default_device():
    """OG_DEVICE overrides; otherwise the first CUDA device if present, else CPU."""
    if os.environ.get("OG_DEVICE"):
        return os.environ["OG_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_graph(name, cfg):
    """Cache the loaded graph per dataset so we don't reload it for every stage."""
    if name not in _graph_cache:
        print(f"  Loading graph for {name}...")
        _graph_cache[name] = cfg["loader"]()
    return _graph_cache[name]


# ====================================================================================
# STAGE 1: Phi (re)computation via the validated vectorized simulator
# ====================================================================================

def stage_phi(name, cfg, folder, device=None):
    device = device or _default_device()
    """
    Recomputes Phi with the vectorized simulator at the cached budget K* (read from
    phi_full_meanstd.pt), delta = 1, eps = 1/50, one realization per call; overwrites
    phi_full_meanstd.pt and exports phi.csv.
    """
    old_data_path = folder / "hvgae_embedding" / "phi_full_meanstd.pt"
    if not old_data_path.exists():
        raise FileNotFoundError(f"{old_data_path} missing -- cannot determine K* to reuse.")
    old_data = torch.load(old_data_path)
    K_star = old_data["K_star"]

    embedding_path = folder / "hvgae_embedding" / "embedding.pt"
    curvature_path = folder / "hvgae_embedding" / "chosen_curvature.txt"
    z, manifold, c = load_embedding(str(embedding_path), str(curvature_path))

    G, N, node_id_map, data = get_graph(name, cfg)
    if data is not None:
        edge_index = data.edge_index
    else:
        idx = {n: i for i, n in enumerate(node_id_map)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    print(f"  Recomputing Phi with vectorized simulator (K*={K_star}, N={N})...")
    phi_new = vectorized_phi_at_K_all_nodes(
        edge_index, z, manifold, K=K_star, eps=1/50, delta=1.0, num_nodes=N,
        n_repeats=3, source_batch_size=cfg["phi_batch_size"], device=device,
        max_degree_cap=cfg["max_degree_cap"],
    )

    rho, _ = spearmanr(phi_new.numpy(), old_data["mean"].numpy())
    print(f"  Sanity check vs previous cached Phi: rho={rho:.4f}")
    if rho < 0.95:
        print(f"  [WARNING] rho={rho:.4f} is lower than expected (<0.95) -- investigate!")

    torch.save({"mean": phi_new, "std": torch.zeros_like(phi_new), "seeds": [0],
                "K_star": K_star, "recomputed_with": "vectorized_simulate.py"},
               folder / "hvgae_embedding" / "phi_full_meanstd.pt")

    ids = node_id_map if node_id_map is not None else list(range(N))
    with open(folder / "phi.csv", "w") as f:
        f.write("node_id,phi\n")
        for nid, val in zip(ids, phi_new.numpy()):
            f.write(f"{nid},{val}\n")

    return {"rho_vs_previous": float(rho), "K_star": K_star}


# ====================================================================================
# STAGE 2: Correlations (Spearman, Kendall, Pearson) vs classical centralities
# ====================================================================================

def _integer_indexed_graph(G, node_id_map):
    """
    Returns G relabeled so nodes are integers 0..N-1 matching node_id_map's order.
    Needed for go_dag specifically, whose graph nodes are GO ID strings (e.g.
    "GO:0008150") -- compute_classical_features() in reconstruction_utils.py
    assumes integer-indexed nodes 0..N-1 (true for cora/citeseer/arxiv/string,
    NOT true for go_dag), so calling it on the raw string-keyed graph raises
    KeyError on the first integer lookup.
    """
    if node_id_map is None:
        return G  # already integer-indexed (cora/citeseer/arxiv/string)
    idx = {n: i for i, n in enumerate(node_id_map)}
    return nx.relabel_nodes(G, idx)


_features_cache = {}  # in-memory cache, in case both stages run in the same process


def get_classical_features_cached(name, cfg, folder, G_int, N):
    """
    Classical centralities are expensive (hours on ogbn-arXiv), so they are cached in
    memory and on disk (<dataset>/_classical_features_cache.npy) and computed once.
    """
    if name in _features_cache:
        return _features_cache[name]

    cache_path = folder / "_classical_features_cache.npy"
    if cache_path.exists():
        print(f"  Loaded cached classical features from {cache_path}")
        X = np.load(cache_path)
        _features_cache[name] = X
        return X

    X = compute_classical_features(
        G_int, N, betweenness_k=cfg["betweenness_k"],
        closeness_harmonic_samples=cfg["closeness_harmonic_samples"],
    )
    np.save(cache_path, X)
    _features_cache[name] = X
    return X


def stage_correlations(name, cfg, folder):
    G, N, node_id_map, data = get_graph(name, cfg)
    G_int = _integer_indexed_graph(G, node_id_map)
    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi = phi_data["mean"].numpy()

    X = get_classical_features_cached(name, cfg, folder, G_int, N)

    results = {}
    for i, fname in enumerate(FEATURE_NAMES):
        vec = X[:, i]
        s_rho, s_p = spearmanr(phi, vec)
        k_tau, k_p = kendalltau(phi, vec)
        p_rho, p_p = pearsonr(phi, vec)
        results[fname] = {
            "spearman_rho": float(s_rho), "spearman_p": float(s_p),
            "kendall_tau": float(k_tau), "kendall_p": float(k_p),
            "pearson_rho": float(p_rho), "pearson_p": float(p_p),
        }

    with open(folder / "correlation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ====================================================================================
# STAGE 3: Reconstruction (Linear/RF/XGBoost) + feature importance + mutual info
# ====================================================================================

def stage_reconstruction(name, cfg, folder):
    G, N, node_id_map, data = get_graph(name, cfg)
    G_int = _integer_indexed_graph(G, node_id_map)
    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi = phi_data["mean"].numpy()

    X = get_classical_features_cached(name, cfg, folder, G_int, N)

    summary, best_model, last_split = run_reconstruction_experiment(
        X, phi, cfg["display_name"], out_dir=str(folder)
    )
    src = folder / f"reconstruction_{cfg['display_name']}.json"
    dst = folder / "reconstruction_results.json"
    if src.exists():
        with open(src) as f:
            content = json.load(f)
        with open(dst, "w") as f:
            json.dump(content, f, indent=2)

    run_mutual_information(X, phi, cfg["display_name"], out_dir=str(folder))
    run_feature_importance(best_model, cfg["display_name"], out_dir=str(folder))

    return summary


# ====================================================================================
# STAGE 4: Radius correlation -- REAL hyperbolic radius-from-origin
# ====================================================================================

def stage_radius(name, cfg, folder):
    embedding_path = folder / "hvgae_embedding" / "embedding.pt"
    curvature_path = folder / "hvgae_embedding" / "chosen_curvature.txt"
    z, manifold, c = load_embedding(str(embedding_path), str(curvature_path))
    radii = compute_radii(z, manifold)
    radii = radii.numpy() if hasattr(radii, "numpy") else np.array(radii)

    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi = phi_data["mean"].numpy()

    rho, p = spearmanr(radii, phi)
    result = {
        "rho": float(rho), "p": float(p),
        "definition": "hyperbolic distance from manifold origin (phi_utils.compute_radii)",
        "curvature": c,
    }
    with open(folder / "radius_correlation.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ====================================================================================
# STAGE 5: Node classification (citation graphs only: cora, citeseer, obgn_arxiv)
# ====================================================================================

def stage_classification(name, cfg, folder):
    if name not in ("cora", "citeseer", "obgn_arxiv"):
        print(f"  [skip] classification not applicable to {name}")
        return None

    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv

    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, edge_index)

    class GAT(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.5)
            self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.5)
        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, edge_index)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, edge_index)

    MODELS = {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE}

    def train_and_eval(model_cls, data, n_epochs=200, lr=0.01, weight_decay=5e-4):
        model = model_cls(data.x.size(1), 16, int(data.y.max().item()) + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        return acc

    def run_multiple_seeds(model_cls, data, n_seeds=5):
        accs = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            accs.append(train_and_eval(model_cls, data))
        accs = torch.tensor(accs)
        return accs.mean().item(), accs.std().item()

    G, N, node_id_map, data = get_graph(name, cfg)
    if not hasattr(data, "train_mask"):
        print(f"  [WARN] {name} has no train_mask/test_mask -- ogbn-arxiv uses OGB's "
              f"official split, not Planetoid masks -- check the split before use.")
        return None

    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi_mean = phi_data["mean"]
    phi_norm = (phi_mean.unsqueeze(1) - phi_mean.mean()) / (phi_mean.std() + 1e-8)
    data_with_phi = data.clone()
    data_with_phi.x = torch.cat([data.x, phi_norm], dim=1)

    results = {}
    for model_name, model_cls in MODELS.items():
        mean_base, std_base = run_multiple_seeds(model_cls, data)
        mean_phi, std_phi = run_multiple_seeds(model_cls, data_with_phi)
        results[model_name] = {
            "baseline_mean": mean_base, "baseline_std": std_base,
            "with_phi_mean": mean_phi, "with_phi_std": std_phi,
            "delta": mean_phi - mean_base,
        }
        print(f"  {model_name}: baseline={mean_base:.4f} with_phi={mean_phi:.4f} "
              f"delta={mean_phi - mean_base:+.4f}")

    with open(folder / "classification.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ====================================================================================
# STAGE 6: GO hierarchy validation (go_dag only) -- SEE MODULE DOCSTRING CAVEAT #1
# ====================================================================================

def stage_hierarchy(name, cfg, folder):
    """GO DAG hierarchy validation: Phi vs. log(1 + descendants) with and without
    controlling for depth, leaf composition of top-10% sets, parent-child differences."""
    if name != "go_dag":
        print(f"  [skip] hierarchy validation only applies to go_dag")
        return None

    G, N, node_id_map, data = get_graph(name, cfg)
    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi = phi_data["mean"].numpy()

    depths_path = folder / "data" / "node_depths.pt"
    depths = torch.load(depths_path).numpy()
    idx = {n: i for i, n in enumerate(node_id_map)}

    from goatools.obo_parser import GODag
    godag = GODag(str(folder / "data" / "go-basic.obo"))
    G_dag = nx.DiGraph()
    for go_id, term in godag.items():
        if term.namespace != "biological_process":
            continue
        if go_id != term.item_id:
            continue  # skip alt_id duplicate keys
        G_dag.add_node(go_id)
        for parent in term.parents:
            if parent.namespace == "biological_process":
                G_dag.add_edge(go_id, parent.id)  # child -> parent
    G_dag.add_nodes_from(node_id_map)

    # descendant counts via reversed (parent->child) graph, exactly as the original does
    print("  Computing descendant counts (this loop is the real cost, ~O(N) BFS calls)...")
    G_rev = G_dag.reverse()
    descendant_counts = np.zeros(N)
    for i, go_id in enumerate(node_id_map):
        if go_id in G_rev:
            descendant_counts[i] = len(nx.descendants(G_rev, go_id))
    log_desc = np.log1p(descendant_counts)  # heavy-tailed counts: correlate against log1p

    valid = depths >= 0
    rho_uncontrolled, p_uncontrolled = spearmanr(phi[valid], log_desc[valid])

    from scipy.stats import rankdata
    def partial_spearman(x, y, control):
        rx, ry, rc = rankdata(x), rankdata(y), rankdata(control)
        def residualize(a, c):
            c_mean, a_mean = c.mean(), a.mean()
            beta = np.sum((c - c_mean) * (a - a_mean)) / np.sum((c - c_mean) ** 2)
            return a - beta * (c - c_mean)
        return spearmanr(residualize(rx, rc), residualize(ry, rc))
    rho_partial, p_partial = partial_spearman(phi[valid], log_desc[valid], depths[valid])

    # leaf = no children = in-degree 0 in child->parent graph (matches original exactly)
    in_degrees = dict(G_dag.in_degree())
    is_leaf = np.array([in_degrees.get(go_id, 0) == 0 for go_id in node_id_map])

    leaf_baseline = is_leaf.mean()
    top_pct = 0.1
    n_top = int(N * top_pct)
    top_phi_idx = np.argsort(phi)[::-1][:n_top]
    top_degree = np.array([G.degree(n) for n in node_id_map])
    top_degree_idx = np.argsort(top_degree)[::-1][:n_top]
    top_pagerank_dict = nx.pagerank(G)
    top_pagerank = np.array([top_pagerank_dict[n] for n in node_id_map])
    top_pagerank_idx = np.argsort(top_pagerank)[::-1][:n_top]

    leaf_top_phi = is_leaf[top_phi_idx].mean()
    leaf_top_degree = is_leaf[top_degree_idx].mean()
    leaf_top_pagerank = is_leaf[top_pagerank_idx].mean()

    # parent-child delta on random EDGES (matches original -- not random nodes)
    import random
    random.seed(0)
    edges_list = list(G_dag.edges())
    sampled_edges = random.sample(edges_list, min(2000, len(edges_list)))
    deltas = []
    for child, parent in sampled_edges:
        if child in idx and parent in idx:
            deltas.append(phi[idx[child]] - phi[idx[parent]])
    deltas = np.array(deltas)

    result = {
        "descendant_correlation": {"rho": float(rho_uncontrolled), "p": float(p_uncontrolled)},
        "partial_correlation_depth_controlled": {"rho": float(rho_partial), "p": float(p_partial)},
        "leaf_baseline": float(leaf_baseline),
        "leaf_top_phi": float(leaf_top_phi),
        "leaf_top_degree": float(leaf_top_degree),
        "leaf_top_pagerank": float(leaf_top_pagerank),
        "parent_child_delta_mean": float(deltas.mean()),
        "parent_child_delta_std": float(deltas.std()),
        "fraction_child_greater": float((deltas > 0).mean()),
    }
    with open(folder / "hierarchy_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ====================================================================================
# STAGE 7: STRING biology validation (string_ppi only) -- SEE MODULE DOCSTRING CAVEAT #2
# ====================================================================================

def stage_biology(name, cfg, folder):
    """STRING-PPI biological validation: functional coherence of the top-5% sets by Phi and
    by degree against degree-preserving nulls, and GO enrichment on the set differences."""
    if name != "string_ppi":
        print(f"  [skip] biology validation only applies to string_ppi")
        return None

    import gzip
    STRING_SCORE_THRESHOLD = 700
    MAX_TERM_POP_COUNT = 500
    N_PERMUTATIONS = 1000
    TOP_PCT = 0.05

    G, N, node_id_map, data = get_graph(name, cfg)
    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi_mean = phi_data["mean"]
    phi = phi_mean.numpy()

    # --- Fix 1: rebuild graph with REAL STRING confidence weights ---
    print("  Rebuilding graph with real STRING confidence weights...")
    symbol_to_idx = {node_id_map[i]: i for i in range(N)}
    links_path = folder / "data" / "9606.protein.links.v12.0.txt.gz"
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(range(N))
    n_weighted = 0
    with gzip.open(links_path, "rt") as f:
        next(f)
        for line in f:
            p1, p2, score = line.strip().split(" ")
            score = int(score)
            if score < STRING_SCORE_THRESHOLD:
                continue
            i, j = symbol_to_idx.get(p1), symbol_to_idx.get(p2)
            if i is None or j is None:
                continue
            G_weighted.add_edge(i, j, weight=score / 1000.0)
            n_weighted += 1
    edge_list = data.edge_index.t().tolist()
    n_missing = 0
    for u, v in edge_list:
        if not G_weighted.has_edge(u, v):
            G_weighted.add_edge(u, v, weight=1.0)
            n_missing += 1
    print(f"  Rebuilt graph: {n_weighted} edges with real weights, {n_missing} defaulted to 1.0")

    degrees = np.array([G_weighted.degree(i) for i in range(N)])

    top_pct = 0.05
    n_top = max(1, int(top_pct * N))
    top_phi_idx = torch.argsort(phi_mean, descending=True)[:n_top].tolist()
    top_degree_idx = sorted(range(N), key=lambda i: degrees[i], reverse=True)[:n_top]

    top100_phi_idx = torch.argsort(phi_mean, descending=True)[:100].tolist()
    top100_degree_idx = sorted(range(N), key=lambda i: degrees[i], reverse=True)[:100]

    jaccard_5pct = len(set(top_phi_idx) & set(top_degree_idx)) / len(set(top_phi_idx) | set(top_degree_idx))
    jaccard_100 = len(set(top100_phi_idx) & set(top100_degree_idx)) / len(set(top100_phi_idx) | set(top100_degree_idx))

    # --- subgraph stats, now including conductance (Fix 4) ---
    def graph_stats(node_set):
        sub = G_weighted.subgraph(node_set)
        n = sub.number_of_nodes()
        density = nx.density(sub)
        avg_deg = sum(dict(sub.degree()).values()) / n if n else 0.0
        components = list(nx.connected_components(sub))
        n_components = len(components)
        largest_cc = max((len(c) for c in components), default=0)
        clustering = nx.average_clustering(sub)
        try:
            conductance = nx.algorithms.cuts.conductance(G_weighted, set(node_set))
        except Exception:
            conductance = float("nan")
        return {"density": float(density), "avg_degree": float(avg_deg),
                "n_components": int(n_components), "largest_component_size": int(largest_cc),
                "clustering_coefficient": float(clustering), "conductance": float(conductance)}

    stats_phi = graph_stats(top100_phi_idx)
    stats_deg = graph_stats(top100_degree_idx)

    # --- Fix 2 & 3: real gene mapping + GO enrichment, including set-difference ---
    print("  Loading gene symbol mapping + GO associations (cached after first run)...")
    from enrichment_utils import (load_string_id_to_symbol, load_go_associations,
                                    run_go_enrichment, summarize_enrichment)

    def filter_bp_not_generic(records, max_pop=MAX_TERM_POP_COUNT):
        out = []
        for r in records:
            if r.NS != "BP":
                continue
            pop_count = r.ratio_in_pop[0] if hasattr(r, "ratio_in_pop") else None
            if pop_count is not None and pop_count > max_pop:
                continue
            out.append(r)
        return out

    string_to_symbol = load_string_id_to_symbol(str(folder / "data"))
    idx_to_symbol = {i: string_to_symbol.get(node_id_map[i]) for i in range(N)}
    all_symbols = [s for s in idx_to_symbol.values() if s is not None]

    top_phi_genes = set(idx_to_symbol[i] for i in top_phi_idx if idx_to_symbol[i] is not None)
    top_deg_genes = set(idx_to_symbol[i] for i in top_degree_idx if idx_to_symbol[i] is not None)
    print(f"  Top-Phi mapped genes: {len(top_phi_genes)}, Top-Degree mapped genes: {len(top_deg_genes)}")

    godag, assoc = load_go_associations(str(folder / "data"), go_dag_data_dir=str(ROOT / "go_dag" / "data"))

    print("  Running GO enrichment: top-Phi, top-Degree, and set differences...")
    phi_enrichment = filter_bp_not_generic(run_go_enrichment(list(top_phi_genes), all_symbols, godag, assoc))
    deg_enrichment = filter_bp_not_generic(run_go_enrichment(list(top_deg_genes), all_symbols, godag, assoc))
    phi_summary = summarize_enrichment(phi_enrichment)
    deg_summary = summarize_enrichment(deg_enrichment)

    phi_only_genes = list(top_phi_genes - top_deg_genes)
    deg_only_genes = list(top_deg_genes - top_phi_genes)
    phi_minus_degree, degree_minus_phi = [], []
    if len(phi_only_genes) >= 5:
        phi_diff = filter_bp_not_generic(run_go_enrichment(phi_only_genes, all_symbols, godag, assoc))
        phi_minus_degree = summarize_enrichment(phi_diff)
    if len(deg_only_genes) >= 5:
        deg_diff = filter_bp_not_generic(run_go_enrichment(deg_only_genes, all_symbols, godag, assoc))
        degree_minus_phi = summarize_enrichment(deg_diff)

    # unique significant terms per method (against full filtered lists)
    phi_all = {r["go_id"]: r for r in summarize_enrichment(phi_enrichment, top_n=len(phi_enrichment))}
    deg_all = {r["go_id"]: r for r in summarize_enrichment(deg_enrichment, top_n=len(deg_enrichment))}
    unique_to_phi = sorted((phi_all[g] for g in phi_all.keys() - deg_all.keys()), key=lambda r: r["p_fdr_bh"])
    unique_to_degree = sorted((deg_all[g] for g in deg_all.keys() - phi_all.keys()), key=lambda r: r["p_fdr_bh"])

    # --- functional coherence permutation test, now using REAL weights ---
    def avg_pairwise_confidence(node_set):
        node_set = list(node_set)
        total, count = 0.0, 0
        for i in range(len(node_set)):
            for j in range(i + 1, len(node_set)):
                u, v = node_set[i], node_set[j]
                if G_weighted.has_edge(u, v):
                    total += G_weighted[u][v]["weight"]
                    count += 1
        return total / count if count > 0 else 0.0

    observed_coherence = avg_pairwise_confidence(top_phi_idx)
    import random
    random.seed(0)
    null_coherences = np.array([avg_pairwise_confidence(random.sample(range(N), n_top))
                                  for _ in range(N_PERMUTATIONS)])
    p_value = float((null_coherences >= observed_coherence).mean())

    result = {
        "go_enrichment": {
            "top_phi": phi_summary, "top_degree": deg_summary,
            "unique_to_phi": unique_to_phi, "unique_to_degree": unique_to_degree,
            "phi_minus_degree_enrichment": phi_minus_degree,
            "degree_minus_phi_enrichment": degree_minus_phi,
        },
        "jaccard_top5pct": float(jaccard_5pct),
        "jaccard_top100": float(jaccard_100),
        "n_top_phi_genes": len(top_phi_genes), "n_top_degree_genes": len(top_deg_genes),
        "phi_top100_subgraph_stats": stats_phi,
        "degree_top100_subgraph_stats": stats_deg,
        "functional_coherence": {
            "observed_coherence": float(observed_coherence),
            "null_mean": float(null_coherences.mean()),
            "null_std": float(null_coherences.std()),
            "p_value": p_value,
        },
    }
    with open(folder / "biology_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ====================================================================================
# STAGE 8: Dense K-sweep (K=5,10,...,100)
# ====================================================================================

def stage_k_sweep(name, cfg, folder, device=None):
    device = device or _default_device()
    embedding_path = folder / "hvgae_embedding" / "embedding.pt"
    curvature_path = folder / "hvgae_embedding" / "chosen_curvature.txt"
    z, manifold, c = load_embedding(str(embedding_path), str(curvature_path))

    G, N, node_id_map, data = get_graph(name, cfg)
    if data is not None:
        edge_index = data.edge_index
    else:
        idx = {n: i for i, n in enumerate(node_id_map)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    K_values = list(range(5, 101, 5))
    final_phi = vectorized_phi_at_K_all_nodes(
        edge_index, z, manifold, K=100, eps=1/50, delta=1.0, num_nodes=N,
        n_repeats=1, source_batch_size=cfg["phi_batch_size"], device=device,
        max_degree_cap=cfg["max_degree_cap"],
    )

    anchored = []
    for K in K_values:
        phi_k = vectorized_phi_at_K_all_nodes(
            edge_index, z, manifold, K=K, eps=1/50, delta=1.0, num_nodes=N,
            n_repeats=1, source_batch_size=cfg["phi_batch_size"], device=device,
            max_degree_cap=cfg["max_degree_cap"],
        )
        rho, _ = spearmanr(phi_k.numpy(), final_phi.numpy())
        anchored.append({"K": K, "rho_vs_K100": float(rho)})
        print(f"  K={K:3d}  rho vs K=100 = {rho:.4f}")

    plateau_K = 100
    for entry in anchored:
        if all(a["rho_vs_K100"] >= 0.99 for a in anchored if a["K"] >= entry["K"]):
            plateau_K = entry["K"]
            break

    with open(folder / "k_sweep_full.json", "w") as f:
        json.dump({"anchored_sweep": anchored, "plateau_K": plateau_K}, f, indent=2)
    return {"plateau_K": plateau_K}


def stage_ogd(name, cfg, folder):
    """
    Ported from cora_ogd_experiment.py / citeseer_ogd_experiment.py -- these were
    Cora/Citeseer only; this stage is a no-op for
    go_dag/obgn_arxiv/string_ppi. Runs the full 7-experiment OGD battery (single-
    descriptor ablation, full-descriptor comparison, classical-vs-observer,
    combined, permutation importance, dimensionality curve, descriptor correlation
    matrix) using ogd_utils.py.
    """
    if name not in ("cora", "citeseer"):
        print(f"  [skip] OGD experiment only applies to cora/citeseer (per original scope)")
        return None

    from ogd_utils import (compute_observer_descriptor_all, compute_classical_descriptor_all,
                             make_feature_tensor, run_multiple_seeds, permutation_importance)

    R_DESCRIPTOR = 2
    OBSERVER_NAMES = ["Phi", "H", "Reff", "Var", "C", "L", "B"]
    CLASSICAL_NAMES = ["Degree", "PageRank", "Betweenness", "Eigenvector", "HyperbolicRadius"]

    G, N, node_id_map, data = get_graph(name, cfg)
    embedding_path = folder / "hvgae_embedding" / "embedding.pt"
    curvature_path = folder / "hvgae_embedding" / "chosen_curvature.txt"
    z, manifold, c = load_embedding(str(embedding_path), str(curvature_path))
    phi_data = torch.load(folder / "hvgae_embedding" / "phi_full_meanstd.pt")
    phi = phi_data["mean"]

    print("  Computing Observer Geometry Descriptor for all nodes...")
    observer = compute_observer_descriptor_all(G, N, z, manifold, phi, r=R_DESCRIPTOR)
    print("  Computing classical descriptor...")
    radii = compute_radii(z, manifold)
    classical = compute_classical_descriptor_all(G, N, radii)
    all_features = {**observer, **classical}

    # Exp 7: descriptor correlation matrix
    mat = np.zeros((7, 7))
    for i, ni in enumerate(OBSERVER_NAMES):
        for j, nj in enumerate(OBSERVER_NAMES):
            if np.std(observer[ni]) < 1e-10 or np.std(observer[nj]) < 1e-10:
                mat[i, j] = np.nan
            else:
                rho, _ = spearmanr(observer[ni], observer[nj])
                mat[i, j] = rho
    with open(folder / "ogd_correlation_matrix.json", "w") as f:
        json.dump({"names": OBSERVER_NAMES, "matrix": np.nan_to_num(mat, nan=-999).tolist()}, f, indent=2)

    edge_index = data.edge_index
    y = data.y
    train_mask, test_mask = data.train_mask, data.test_mask

    base_mean, base_std, _ = run_multiple_seeds(data.x, edge_index, y, train_mask, test_mask)
    results = {"baseline": {"mean": base_mean, "std": base_std}}

    # Exp 1: single descriptor ablation
    for fname in OBSERVER_NAMES:
        x = make_feature_tensor(data.x, all_features, [fname])
        mean, std, _ = run_multiple_seeds(x, edge_index, y, train_mask, test_mask)
        results[f"single_{fname}"] = {"mean": mean, "std": std, "delta": mean - base_mean}
        print(f"  X + {fname:<8} : {mean:.4f} +- {std:.4f}  (delta={mean-base_mean:+.4f})")

    # Exp 2: Phi-only vs full observer descriptor
    x_phi = make_feature_tensor(data.x, all_features, ["Phi"])
    mean_phi, std_phi, _ = run_multiple_seeds(x_phi, edge_index, y, train_mask, test_mask)
    x_full_obs = make_feature_tensor(data.x, all_features, OBSERVER_NAMES)
    mean_full_obs, std_full_obs, _ = run_multiple_seeds(x_full_obs, edge_index, y, train_mask, test_mask)
    results["exp2_phi_only"] = {"mean": mean_phi, "std": std_phi}
    results["exp2_full_observer"] = {"mean": mean_full_obs, "std": std_full_obs}

    # Exp 3: classical descriptor
    x_classical = make_feature_tensor(data.x, all_features, CLASSICAL_NAMES)
    mean_classical, std_classical, _ = run_multiple_seeds(x_classical, edge_index, y, train_mask, test_mask)
    results["exp3_classical"] = {"mean": mean_classical, "std": std_classical}

    # Exp 4: combined
    x_combined = make_feature_tensor(data.x, all_features, CLASSICAL_NAMES + OBSERVER_NAMES)
    mean_combined, std_combined, _ = run_multiple_seeds(x_combined, edge_index, y, train_mask, test_mask)
    results["exp4_combined"] = {"mean": mean_combined, "std": std_combined,
                                  "delta_over_classical": mean_combined - mean_classical}

    # Exp 5: permutation importance
    _, importances = permutation_importance(data.x, edge_index, y, train_mask, test_mask,
                                              all_features, OBSERVER_NAMES)
    results["exp5_importances"] = importances

    # Exp 6: dimensionality curve
    dim_curve = []
    for k in range(1, len(OBSERVER_NAMES) + 1):
        subset = OBSERVER_NAMES[:k]
        x = make_feature_tensor(data.x, all_features, subset)
        mean, std, _ = run_multiple_seeds(x, edge_index, y, train_mask, test_mask)
        dim_curve.append({"n_features": k, "features": subset, "mean": mean, "std": std})
    results["exp6_dimensionality_curve"] = dim_curve

    with open(folder / "ogd_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ====================================================================================
# STAGE 9: Runtime/memory summary
# ====================================================================================

def stage_runtime(name, cfg, folder, checkpoint):
    G, N, node_id_map, data = get_graph(name, cfg)
    E = G.number_of_edges()

    times = {}
    for stage, entry in checkpoint.items():
        if isinstance(entry, dict) and "elapsed_seconds" in entry:
            times[stage] = entry["elapsed_seconds"]

    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6

    result = {
        "nodes": N, "edges": E,
        "stage_times_seconds": times,
        "total_seconds": sum(times.values()),
        "peak_gpu_mb_this_process": peak_gpu_mb,
    }
    with open(folder / "runtime_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ====================================================================================
# ORCHESTRATION
# ====================================================================================

STAGE_FUNCS = {
    "phi": stage_phi,
    "correlations": stage_correlations,
    "reconstruction": stage_reconstruction,
    "radius": stage_radius,
    "classification": stage_classification,
    "hierarchy": stage_hierarchy,
    "biology": stage_biology,
    "ogd": stage_ogd,
    "k_sweep": stage_k_sweep,
}


def run_dataset(name, cfg, stages, force, dry_run):
    folder = ROOT / cfg["folder"]
    print(f"\n{'='*70}\n{name.upper()}\n{'='*70}")

    for stage in stages:
        if stage == "runtime":
            continue

        if stage_done(name, stage, force=force):
            print(f"  [checkpoint] {stage} already done, skipping (--force to redo)")
            continue

        if dry_run:
            print(f"  [dry-run] would run stage: {stage}")
            continue

        print(f"  -- stage: {stage} --")
        t0 = time.time()
        try:
            extra = STAGE_FUNCS[stage](name, cfg, folder)
            elapsed = time.time() - t0
            mark_done(name, stage, elapsed=elapsed)
            print(f"  [done] {stage} in {elapsed:.1f}s")
        except Exception as e:
            print(f"  [ERROR] stage '{stage}' failed: {e}")
            import traceback
            traceback.print_exc()

    if "runtime" in stages and not dry_run:
        if stage_done(name, "runtime", force=force):
            print(f"  [checkpoint] runtime already done, skipping")
        else:
            checkpoint = load_checkpoint(name)
            stage_runtime(name, cfg, folder, checkpoint)
            mark_done(name, "runtime")
            print(f"  [done] runtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="ignore checkpoints, redo everything")
    parser.add_argument("--only", nargs="+", default=None, help="dataset names to run")
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES,
                         help=f"which stages to run, subset of {ALL_STAGES}")
    parser.add_argument("--dry-run", action="store_true", help="print plan without executing")
    args = parser.parse_args()

    targets = args.only if args.only else list(DATASET_CONFIG.keys())

    for name in targets:
        if name not in DATASET_CONFIG:
            print(f"[WARN] unknown dataset '{name}', skipping")
            continue
        run_dataset(name, DATASET_CONFIG[name], args.stages, args.force, args.dry_run)

    print("\n" + "=" * 70)
    print("ALL REQUESTED DATASETS/STAGES PROCESSED.")
    print("Checkpoints saved in .checkpoint_<dataset>.json -- rerun this script anytime")
    print("to resume; completed stages are skipped automatically unless --force is passed.")
    print("embedding_ablation/ was NOT touched by this script, per scope.")
    print("=" * 70)
