"""Hyperbolic GraphSAGE autoencoder in the Lorentz model (deterministic; no KL term, since the
exploration process consumes only coordinates and pairwise distances).

Architecture: two GraphSAGE (mean) layers on tangent vectors at the origin, one exponential
map onto the hyperboloid <x, x>_L = -c (sectional curvature -1/c), Fermi-Dirac decoder on
hyperbolic distances. Message passing is in the tangent space (the common tangent-space
encoder + expmap pattern), not fully manifold-valued aggregation; only final pairwise
distances are consumed downstream.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import geoopt


class HVGAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=16, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, latent_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # tangent vector at origin, [N, latent_dim]


class FermiDiracDecoder(nn.Module):
    def __init__(self, r_init=2.0, t_init=1.0):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(float(r_init)))
        self.t = nn.Parameter(torch.tensor(float(t_init)))

    def forward(self, dist):
        return torch.sigmoid((self.r - dist) / self.t)


class HVGAE(nn.Module):
    """
    Deterministic Hyperbolic GraphSAGE Autoencoder, Lorentz model.

    Curvature is a fixed hyperparameter selected by grid search (curvature_search.py):
    in geoopt 0.5.1, Lorentz.expmap0 does not propagate gradients into k, so a learnable
    curvature would stay at its initialization.
    """
    def __init__(self, in_dim, hidden_dim=64, latent_dim=16, dropout=0.2, curvature=1.0):
        super().__init__()
        self.encoder = HVGAEEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.decoder = FermiDiracDecoder()
        self.fixed_curvature = float(curvature)  # NOT learned -- see note above
        self.latent_dim = latent_dim

    def curvature(self):
        return torch.tensor(self.fixed_curvature)

    def manifold(self):
        return geoopt.Lorentz(k=self.curvature())

    def encode(self, x, edge_index):
        """Returns points on the Lorentz hyperboloid, shape [N, latent_dim+1]."""
        tangent = self.encoder(x, edge_index)  # [N, latent_dim]
        man = self.manifold()
        # expmap0 needs tangent vectors in the AMBIENT space (latent_dim+1) with the
        # time component's tangent set to 0 (standard origin-tangent-space embedding)
        zeros = torch.zeros(tangent.size(0), 1, device=tangent.device, dtype=tangent.dtype)
        tangent_ambient = torch.cat([zeros, tangent], dim=1)
        z = man.expmap0(tangent_ambient)
        return z

    def pairwise_distance(self, z, edge_index):
        man = self.manifold()
        src, dst = edge_index
        return man.dist(z[src], z[dst])

    def decode_edges(self, z, edge_index):
        d = self.pairwise_distance(z, edge_index)
        return self.decoder(d)


def negative_sampling(edge_index, num_nodes, num_neg):
    """Simple random negative sampling (uniform), returns [2, num_neg]."""
    neg_src = torch.randint(0, num_nodes, (num_neg,))
    neg_dst = torch.randint(0, num_nodes, (num_neg,))
    return torch.stack([neg_src, neg_dst], dim=0)


def train_hvgae(data, in_dim, epochs=300, lr=1e-3, hidden_dim=64, latent_dim=16,
                 early_stop_patience=30, device="cpu", verbose=True):
    """
    data: torch_geometric Data object with .x and .edge_index
    Returns: trained model, final embeddings z [N, latent_dim+1] on the Lorentz manifold
    """
    model = HVGAE(in_dim, hidden_dim, latent_dim).to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, edge_index)

        pos_prob = model.decode_edges(z, edge_index)
        neg_edge_index = negative_sampling(edge_index, num_nodes, num_edges).to(device)
        neg_prob = model.decode_edges(z, neg_edge_index)

        pos_loss = -torch.log(pos_prob + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_prob + 1e-8).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  curvature {model.curvature().item():.4f}")

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"early stopping at epoch {epoch}")
            break

    model.eval()
    with torch.no_grad():
        z_final = model.encode(x, edge_index)

    return model, z_final
