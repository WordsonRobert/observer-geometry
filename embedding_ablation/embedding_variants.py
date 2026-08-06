"""
Embedding variants for the ablation study. Each function returns (z, dist_wrapper)
where dist_wrapper exposes .dist(x, y) with the same broadcasting behavior geoopt
manifolds use, so the existing phi_utils/simulate machinery works unchanged regardless
of which embedding produced z.
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import geoopt


class EuclideanDist:
    """Plain L2 distance, wrapped to match geoopt's manifold.dist(x, y) interface."""
    def dist(self, x, y):
        return torch.norm(x - y, dim=-1)


def negative_sampling(edge_index, num_nodes, num_neg):
    neg_src = torch.randint(0, num_nodes, (num_neg,))
    neg_dst = torch.randint(0, num_nodes, (num_neg,))
    return torch.stack([neg_src, neg_dst], dim=0)


def _train_shallow_decoder(z_param, manifold, edge_index, num_nodes, epochs, lr, dist_fn):
    """Shared training loop for shallow (lookup-table) embeddings: Poincare direct,
    and random-init-then-trained variants. z_param is a geoopt.ManifoldParameter or
    plain nn.Parameter; dist_fn(a, b) computes pairwise distance."""
    r = nn.Parameter(torch.tensor(2.0))
    t = nn.Parameter(torch.tensor(1.0))
    params = [z_param, r, t] if isinstance(z_param, nn.Parameter) else [z_param, r, t]
    if manifold is not None:
        optimizer = geoopt.optim.RiemannianAdam(params, lr=lr)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    num_edges = edge_index.size(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_dist = dist_fn(z_param[edge_index[0]], z_param[edge_index[1]])
        neg_edge_index = negative_sampling(edge_index, num_nodes, num_edges)
        neg_dist = dist_fn(z_param[neg_edge_index[0]], z_param[neg_edge_index[1]])

        pos_prob = torch.sigmoid((r - pos_dist) / t)
        neg_prob = torch.sigmoid((r - neg_dist) / t)
        loss = -torch.log(pos_prob + 1e-8).mean() - torch.log(1 - neg_prob + 1e-8).mean()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"    epoch {epoch}: loss {loss.item():.4f}")
    return z_param.detach()


def train_poincare_direct(data, dim=16, curvature=1.0, epochs=300, lr=1e-2):
    """Shallow Poincare embedding (Nickel & Kiela style) -- no encoder network, just
    a directly-optimized per-node embedding table on the Poincare ball."""
    print("  Training shallow Poincare embedding...")
    N = data.x.size(0)
    ball = geoopt.PoincareBall(c=curvature)
    init = torch.randn(N, dim) * 0.001
    z_param = geoopt.ManifoldParameter(init, manifold=ball)

    def dist_fn(a, b):
        return ball.dist(a, b)

    z_final = _train_shallow_decoder(z_param, ball, data.edge_index, N, epochs, lr, dist_fn)
    return z_final, ball


class HGCNLiteEncoder(nn.Module):
    """Simplified hyperbolic GCN layer: tangent-space linear transform + mean
    aggregation, then exp/log maps to move between manifold and tangent space at
    each layer. This is the same class of simplification as the HVGAE encoder
    (tangent-space ops + manifold maps, not full per-layer manifold-valued algebra)
    -- flag this the same way if a reviewer asks about "true" HGCN comparison."""
    def __init__(self, in_dim, hidden_dim, out_dim, ball):
        super().__init__()
        self.ball = ball
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # layer 1: Euclidean features -> tangent space transform -> map to manifold
        h = F.relu(self.lin1(x))
        # simple mean aggregation over neighbors (tangent space, origin-based)
        row, col = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        counts = torch.zeros(h.size(0), device=h.device).index_add_(0, row, torch.ones(row.size(0), device=h.device))
        counts = counts.clamp(min=1).unsqueeze(1)
        h = agg / counts

        h2 = self.lin2(h)
        zeros = torch.zeros(h2.size(0), 1, device=h2.device)
        tangent_ambient = torch.cat([zeros, h2], dim=1) if isinstance(self.ball, geoopt.Lorentz) else h2
        if isinstance(self.ball, geoopt.PoincareBall):
            z = self.ball.expmap0(h2)
        else:
            z = self.ball.expmap0(tangent_ambient)
        return z


def train_hgcn_lite(data, in_dim, dim=16, curvature=1.0, epochs=300, lr=1e-3):
    print("  Training HGCN-lite encoder...")
    ball = geoopt.PoincareBall(c=curvature)
    model = HGCNLiteEncoder(in_dim, 64, dim, ball)
    r = nn.Parameter(torch.tensor(2.0))
    t = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(list(model.parameters()) + [r, t], lr=lr)

    N = data.x.size(0)
    num_edges = data.edge_index.size(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        pos_dist = ball.dist(z[data.edge_index[0]], z[data.edge_index[1]])
        neg_edge_index = negative_sampling(data.edge_index, N, num_edges)
        neg_dist = ball.dist(z[neg_edge_index[0]], z[neg_edge_index[1]])
        pos_prob = torch.sigmoid((r - pos_dist) / t)
        neg_prob = torch.sigmoid((r - neg_dist) / t)
        loss = -torch.log(pos_prob + 1e-8).mean() - torch.log(1 - neg_prob + 1e-8).mean()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"    epoch {epoch}: loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z_final = model(data.x, data.edge_index)
    return z_final, ball


class EuclideanSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def train_euclidean_sage(data, in_dim, dim=16, epochs=300, lr=1e-3):
    """Plain Euclidean GraphSAGE autoencoder -- no hyperbolic mapping anywhere.
    Isolates whether the GNN architecture alone (vs. the geometry) explains results."""
    print("  Training Euclidean GraphSAGE baseline...")
    model = EuclideanSAGEEncoder(in_dim, 64, dim)
    dist = EuclideanDist()
    r = nn.Parameter(torch.tensor(2.0))
    t = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(list(model.parameters()) + [r, t], lr=lr)

    N = data.x.size(0)
    num_edges = data.edge_index.size(1)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        pos_dist = dist.dist(z[data.edge_index[0]], z[data.edge_index[1]])
        neg_edge_index = negative_sampling(data.edge_index, N, num_edges)
        neg_dist = dist.dist(z[neg_edge_index[0]], z[neg_edge_index[1]])
        pos_prob = torch.sigmoid((r - pos_dist) / t)
        neg_prob = torch.sigmoid((r - neg_dist) / t)
        loss = -torch.log(pos_prob + 1e-8).mean() - torch.log(1 - neg_prob + 1e-8).mean()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"    epoch {epoch}: loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z_final = model(data.x, data.edge_index)
    return z_final, dist


def make_random_embedding(num_nodes, dim=16, seed=0):
    """No training at all -- pure random Euclidean vectors. If Phi becomes
    meaningless here (low correlation with the real embedding's Phi, unstable
    rankings), that's evidence the geometry -- not just the pipeline mechanics --
    is doing the work."""
    torch.manual_seed(seed)
    z = torch.randn(num_nodes, dim)
    return z, EuclideanDist()
