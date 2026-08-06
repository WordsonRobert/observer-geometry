"""
Observer Geometry Descriptor (OGD) utilities, shared by cora_ogd_experiment.py and
citeseer_ogd_experiment.py.

o(v) = (Phi, H, Reff, Var, C, L, B), derived directly from:
  - the hidden weighted-distance distribution at a fixed observation radius r:
      Phi  = the real walk-based metric (0th moment / hidden mass, but the actual
             trained Phi(K*), not raw mass)
      Reff = weighted mean hidden distance (1st moment)
      Var  = weighted hidden-distance variance (2nd moment)
      H    = entropy of the (normalized) hidden-distance distribution
  - the hidden graph topology:
      C = hidden connected component count
      L = largest hidden component, as a fraction of the hidden set
      B = branching estimate (mean degree of hidden nodes)

Classical descriptor c(v) = (Degree, PageRank, Betweenness, Eigenvector, HyperbolicRadius).
"""
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def bfs_ball(G, v, r):
    return set(nx.single_source_shortest_path_length(G, v, cutoff=r).keys())


def compute_observer_descriptor_all(G, N, z, manifold, phi, r=2, delta=1.0):
    names = ["Phi", "H", "Reff", "Var", "C", "L", "B"]
    values = {k: np.zeros(N) for k in names}

    for v in range(N):
        hidden = set(G.nodes()) - bfs_ball(G, v, r)
        values["Phi"][v] = phi[v].item()
        if not hidden:
            continue
        idx_list = list(hidden)
        with torch.no_grad():
            d = manifold.dist(z[v].unsqueeze(0).expand(len(idx_list), -1), z[idx_list])
        w = torch.exp(-delta * d)
        w_sum = w.sum().item()
        p = (w / w.sum()).numpy() if w_sum > 0 else np.ones(len(idx_list)) / len(idx_list)

        Reff = float((p * d.numpy()).sum())
        Var = float((p * (d.numpy() - Reff) ** 2).sum())
        H = float(-(p * np.log(p + 1e-12)).sum())

        sub = G.subgraph(hidden)
        components = list(nx.connected_components(sub))
        C = len(components)
        L = max(len(c) for c in components) / len(hidden)
        B = float(np.mean([G.degree(n) for n in hidden]))

        values["H"][v], values["Reff"][v], values["Var"][v] = H, Reff, Var
        values["C"][v], values["L"][v], values["B"][v] = C, L, B

        if v % 200 == 0:
            print(f"    descriptor: {v}/{N} nodes done")

    return values


def compute_classical_descriptor_all(G, N, radii):
    degree = dict(G.degree())
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G, k=min(500, N - 1), seed=0)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    return {
        "Degree": np.array([degree[i] for i in range(N)]),
        "PageRank": np.array([pagerank[i] for i in range(N)]),
        "Betweenness": np.array([betweenness[i] for i in range(N)]),
        "Eigenvector": np.array([eigenvector[i] for i in range(N)]),
        "HyperbolicRadius": radii.numpy() if hasattr(radii, "numpy") else np.array(radii),
    }


class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


def train_and_eval(x, edge_index, y, train_mask, test_mask, n_epochs=200, lr=0.01, weight_decay=5e-4):
    model = GraphSAGEClassifier(x.size(1), 16, int(y.max().item()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)
        acc = (pred[test_mask] == y[test_mask]).float().mean().item()
    return acc, model


def run_multiple_seeds(x, edge_index, y, train_mask, test_mask, n_seeds=5):
    accs = []
    last_model = None
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        acc, model = train_and_eval(x, edge_index, y, train_mask, test_mask)
        accs.append(acc)
        last_model = model
    accs = np.array(accs)
    return accs.mean(), accs.std(), last_model


def make_feature_tensor(base_x, feature_dict, feature_names):
    """Concatenate base_x with z-scored versions of the named features."""
    extra = []
    for name in feature_names:
        vec = feature_dict[name]
        vec = (vec - vec.mean()) / (vec.std() + 1e-8)
        extra.append(torch.tensor(vec, dtype=torch.float32).unsqueeze(1))
    if extra:
        return torch.cat([base_x] + extra, dim=1)
    return base_x


def permutation_importance(base_x, edge_index, y, train_mask, test_mask, feature_dict,
                            feature_names, n_repeats=10, n_seeds=3):
    """Trains X+full_descriptor once (averaged over n_seeds), then for each feature,
    shuffles just that column n_repeats times and measures the accuracy drop."""
    full_x = make_feature_tensor(base_x, feature_dict, feature_names)
    base_mean, base_std, _ = run_multiple_seeds(full_x, edge_index, y, train_mask, test_mask, n_seeds=n_seeds)
    print(f"  Full descriptor baseline accuracy: {base_mean:.4f} +- {base_std:.4f}")

    importances = {}
    n_base_cols = base_x.size(1)
    for i, name in enumerate(feature_names):
        col_idx = n_base_cols + i
        drops = []
        for rep in range(n_repeats):
            torch.manual_seed(rep)
            perm = torch.randperm(full_x.size(0))
            x_shuffled = full_x.clone()
            x_shuffled[:, col_idx] = full_x[perm, col_idx]
            acc, _, _ = run_multiple_seeds(x_shuffled, edge_index, y, train_mask, test_mask, n_seeds=n_seeds)
            drops.append(base_mean - acc)
        importances[name] = {"mean_drop": float(np.mean(drops)), "std_drop": float(np.std(drops))}
        print(f"  {name:<10} importance (accuracy drop when shuffled) = "
              f"{importances[name]['mean_drop']:.4f} +- {importances[name]['std_drop']:.4f}")

    return base_mean, importances
