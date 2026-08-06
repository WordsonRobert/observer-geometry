"""
Curvature model selection for HVGAE.

Trains HVGAE once per candidate curvature in {0.25, 0.5, 1, 2, 4, 8} on a held-out
train/val edge split, evaluates validation reconstruction loss + AUC + AP, and picks
the curvature with the best validation loss. Saves a results table with all six
candidates (loss/AUC/AP) plus the final chosen model's embeddings.

Usage (per dataset -- see individual train_embedding.py files):

    from curvature_search import run_curvature_search
    results, best_curvature, best_model, best_z = run_curvature_search(
        data, in_dim=data.x.size(1), dataset_name="cora", out_dir="cora/hvgae_embedding"
    )
"""
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from common_hvgae import HVGAE, negative_sampling

CANDIDATE_CURVATURES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def split_edges(edge_index, num_nodes, val_frac=0.1, seed=0):
    """
    Simple random edge split for link-prediction validation.
    Returns train_edge_index, val_pos_edge_index, val_neg_edge_index.
    NOTE: this is for embedding-quality validation only, not for the downstream
    HBP/BFC experiments -- the full edge set should be used for those once the
    embedding is chosen.
    """
    g = torch.Generator().manual_seed(seed)
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=g)
    n_val = int(num_edges * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_edge_index = edge_index[:, train_idx]
    val_pos_edge_index = edge_index[:, val_idx]
    val_neg_edge_index = negative_sampling(edge_index, num_nodes, n_val)

    return train_edge_index, val_pos_edge_index, val_neg_edge_index


def evaluate_link_prediction(model, z, pos_edge_index, neg_edge_index):
    """Returns (auc, ap) using Fermi-Dirac decoder probabilities as scores."""
    model.eval()
    with torch.no_grad():
        pos_prob = model.decode_edges(z, pos_edge_index)
        neg_prob = model.decode_edges(z, neg_edge_index)
        scores = torch.cat([pos_prob, neg_prob]).cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_prob.size(0)),
            torch.zeros(neg_prob.size(0))
        ]).cpu().numpy()
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


def train_one_curvature(data, in_dim, curvature, train_edge_index, val_pos, val_neg,
                         epochs=300, lr=1e-3, hidden_dim=64, latent_dim=16,
                         early_stop_patience=30, device="cpu", verbose=False):
    model = HVGAE(in_dim, hidden_dim, latent_dim, curvature=curvature).to(device)
    x = data.x.to(device)
    train_edge_index = train_edge_index.to(device)
    val_pos = val_pos.to(device)
    val_neg = val_neg.to(device)
    num_nodes = x.size(0)
    num_edges = train_edge_index.size(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, train_edge_index)
        pos_prob = model.decode_edges(z, train_edge_index)
        neg_edge_index = negative_sampling(train_edge_index, num_nodes, num_edges).to(device)
        neg_prob = model.decode_edges(z, neg_edge_index)

        pos_loss = -torch.log(pos_prob + 1e-8).mean()
        neg_loss = -torch.log(1 - neg_prob + 1e-8).mean()
        train_loss = pos_loss + neg_loss
        train_loss.backward()
        optimizer.step()

        # validation loss on held-out edges
        model.eval()
        with torch.no_grad():
            z_val = model.encode(x, train_edge_index)  # encode using train graph structure only
            vpos_prob = model.decode_edges(z_val, val_pos)
            vneg_prob = model.decode_edges(z_val, val_neg)
            val_loss = (-torch.log(vpos_prob + 1e-8).mean()
                        - torch.log(1 - vneg_prob + 1e-8).mean()).item()

        if verbose and epoch % 50 == 0:
            print(f"  c={curvature}  epoch {epoch:4d}  train_loss {train_loss.item():.4f}  val_loss {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_final = model.encode(x, train_edge_index)

    auc, ap = evaluate_link_prediction(model, z_final, val_pos, val_neg)
    return model, z_final, best_val_loss, auc, ap


def run_curvature_search(data, in_dim, dataset_name, out_dir, epochs=300,
                          candidate_curvatures=CANDIDATE_CURVATURES,
                          device="cpu", verbose=True):
    """
    Runs the full 6-curvature sweep, picks the best by validation loss, saves:
      out_dir/curvature_search_results.json  -- all 6 candidates' metrics
      out_dir/embedding.pt                   -- best model's final embeddings [N, latent_dim+1]
      out_dir/model_state.pt                 -- best model's state_dict
      out_dir/chosen_curvature.txt           -- just the winning curvature value
    Returns: results (list of dicts), best_curvature, best_model, best_z
    """
    num_nodes = data.x.size(0)
    train_edge_index, val_pos, val_neg = split_edges(data.edge_index, num_nodes)

    results = []
    best_val_loss = float("inf")
    best_curvature = None
    best_model = None
    best_z = None

    for c in candidate_curvatures:
        if verbose:
            print(f"\n--- training curvature c={c} ---")
        model, z, val_loss, auc, ap = train_one_curvature(
            data, in_dim, c, train_edge_index, val_pos, val_neg,
            epochs=epochs, device=device, verbose=verbose
        )
        results.append({"curvature": c, "val_loss": val_loss, "val_auc": auc, "val_ap": ap})
        if verbose:
            print(f"  -> c={c}  val_loss={val_loss:.4f}  AUC={auc:.4f}  AP={ap:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_curvature = c
            best_model = model
            best_z = z

    # re-encode best model on the FULL edge set (not just train split) for downstream use
    best_model.eval()
    with torch.no_grad():
        full_z = best_model.encode(data.x.to(device), data.edge_index.to(device))

    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/curvature_search_results.json", "w") as f:
        json.dump({"dataset": dataset_name, "results": results, "chosen_curvature": best_curvature}, f, indent=2)
    torch.save(full_z, f"{out_dir}/embedding.pt")
    torch.save(best_model.state_dict(), f"{out_dir}/model_state.pt")
    with open(f"{out_dir}/chosen_curvature.txt", "w") as f:
        f.write(str(best_curvature))

    if verbose:
        print(f"\n=== {dataset_name}: chosen curvature = {best_curvature} (val_loss={best_val_loss:.4f}) ===")
        print(f"Saved to {out_dir}/")

    return results, best_curvature, best_model, full_z
