"""R4: embedding-robustness study (Table 2) on further datasets, with exact Phi.

Same protocol as embedding_ablation/run_seed_robustness.py and run_grid_ablation.py:
  * seed floor: HVGAE (GraphSAGE + expmap, dropout 0.2) at the dataset's selected c,
    300 epochs on the full edge set, seeds 0-3;
  * grid: {SAGE, GCN, GAT} x {Lorentz, Poincare, Euclidean}, c = 1, 300 epochs,
    no early stopping, no dropout, seed 0;
  * random null: untrained N(0,1) coordinates in R^16 with Euclidean distance.
Phi is computed exactly (Proposition 2) at the dataset's K*, delta = 1, eps = 1/50.

Usage: python embedding_ablation/run_robustness_extension.py <dataset> [K*]
  (K* defaults to the dataset's cached plateau budget). Emits
  embedding_ablation/robust_<dataset>.json and embeddings/robust_<dataset>_phis.npz (resumable).
"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
DS = sys.argv[1]
if len(sys.argv) > 2:
    KSTAR = int(sys.argv[2])
else:
    KSTAR = int(__import__("torch").load(f"{REPO}/{DS}/hvgae_embedding/phi_full_meanstd.pt",
                                         weights_only=False, map_location="cpu")["K_star"])
OUT = f"{HERE}/robust_{DS}.json"
import geoopt
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from scipy.stats import spearmanr
from common_hvgae import HVGAE, negative_sampling
from exact_phi import exact_phi_generic
torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    pyg = f"{REPO}/{DS}/data/pyg_tensors.pt"
    if os.path.exists(pyg):
        from torch_geometric.data import Data
        d = torch.load(pyg, weights_only=False)
        return Data(x=d["x"], edge_index=d["edge_index"])
    if DS in ("cora", "citeseer"):
        from torch_geometric.datasets import Planetoid
        return Planetoid(root=f"{REPO}/{DS}/data", name={"cora": "Cora", "citeseer": "Citeseer"}[DS])[0]
    if DS == "go_dag":
        sys.path.insert(0, f"{REPO}/go_dag")  # rebuilds the graph exactly as go_dag_train_embedding.py
        from go_dag_train_embedding import build_bp_dag, to_pyg_data
        G = build_bp_dag(f"{REPO}/go_dag/data/go-basic.obo")
        data, nodes, idx = to_pyg_data(G)
        return data
    raise ValueError(DS)


data = load_data()
N = data.num_nodes
ei = data.edge_index.numpy()
data = data.to(DEV)
c_main = float(open(f"{REPO}/{DS}/hvgae_embedding/chosen_curvature.txt").read())
print(f"[{DS}] N={N} E={ei.shape[1]} c_main={c_main} K*={KSTAR}", flush=True)


# ---------------- distances -----------------
def lorentz_fns(z, c):
    z = z.double().numpy()
    J = z.copy(); J[:, 0] *= -1
    ed = lambda s, d: np.sqrt(c) * np.arccosh(np.clip(-(J[s] * z[d]).sum(1) / c, 1 + 1e-6, None))
    bd = lambda rows: np.sqrt(c) * np.arccosh(np.clip(-(J[rows] @ z.T) / c, 1 + 1e-6, None))
    return ed, bd


def poincare_fns(x, c):
    x = x.double().numpy(); n2 = (x ** 2).sum(1)
    def d(a, b, na, nb, sq):
        arg = 1 + 2 * c * sq / np.clip((1 - c * na) * (1 - c * nb), 1e-15, None)
        return np.arccosh(np.clip(arg, 1 + 1e-12, None)) / np.sqrt(c)
    ed = lambda s, t: d(None, None, n2[s], n2[t], ((x[s] - x[t]) ** 2).sum(1))
    bd = lambda rows: d(None, None, n2[rows][:, None], n2[None, :],
                        np.clip(n2[rows][:, None] + n2[None, :] - 2 * x[rows] @ x.T, 0, None))
    return ed, bd


def euclid_fns(x):
    x = x.double().numpy(); n2 = (x ** 2).sum(1)
    ed = lambda s, t: np.sqrt(((x[s] - x[t]) ** 2).sum(1))
    bd = lambda rows: np.sqrt(np.clip(n2[rows][:, None] + n2[None, :] - 2 * x[rows] @ x.T, 0, None))
    return ed, bd


def phi_of(ed, bd):
    out, _ = exact_phi_generic(ei, N, ed, bd, [KSTAR], block=1024)
    return out[KSTAR]


# ---------------- training (protocols copied from the ablation scripts) -----------------
def train_hvgae_seeded(seed, epochs=300, lr=1e-3):
    torch.manual_seed(seed)
    model = HVGAE(data.x.size(1), 64, 16, curvature=c_main).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    E = data.edge_index.size(1)
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pos = model.decode_edges(z, data.edge_index)
        neg = model.decode_edges(z, negative_sampling(data.edge_index, N, E).to(DEV))
        loss = -torch.log(pos + 1e-8).mean() - torch.log(1 - neg + 1e-8).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model.encode(data.x, data.edge_index).cpu()


class EuclideanDist:
    def dist(self, x, y):
        return torch.norm(x - y, dim=-1)


class FlexibleEncoder(nn.Module):
    def __init__(self, conv_cls, in_dim, hidden_dim, out_dim, kind, ball=None):
        super().__init__()
        if conv_cls is GATConv:
            self.conv1 = conv_cls(in_dim, hidden_dim, heads=1); self.conv2 = conv_cls(hidden_dim, out_dim, heads=1)
        else:
            self.conv1 = conv_cls(in_dim, hidden_dim); self.conv2 = conv_cls(hidden_dim, out_dim)
        self.kind, self.ball = kind, ball

    def forward(self, x, edge_index):
        h = self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)
        if self.kind == "euclidean":
            return h
        if self.kind == "poincare":
            return self.ball.expmap0(h)
        return self.ball.expmap0(torch.cat([torch.zeros(h.size(0), 1, dtype=h.dtype, device=h.device), h], 1))


def train_grid(conv_cls, kind, seed=0, epochs=300, lr=1e-3, c=1.0):
    torch.manual_seed(seed)
    if kind == "lorentz":
        ball = geoopt.Lorentz(k=torch.tensor(c)); dist_fn = ball
    elif kind == "poincare":
        ball = geoopt.PoincareBall(c=c); dist_fn = ball
    else:
        ball = None; dist_fn = EuclideanDist()
    model = FlexibleEncoder(conv_cls, data.x.size(1), 64, 16, kind, ball).to(DEV)
    r = nn.Parameter(torch.tensor(2.0, device=DEV)); t = nn.Parameter(torch.tensor(1.0, device=DEV))
    opt = torch.optim.Adam(list(model.parameters()) + [r, t], lr=lr, weight_decay=5e-4)
    E = data.edge_index.size(1)
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        z = model(data.x, data.edge_index)
        pos = dist_fn.dist(z[data.edge_index[0]], z[data.edge_index[1]])
        ne = negative_sampling(data.edge_index, N, E).to(DEV)
        neg = dist_fn.dist(z[ne[0]], z[ne[1]])
        loss = (-torch.log(torch.sigmoid((r - pos) / t) + 1e-8).mean()
                - torch.log(1 - torch.sigmoid((r - neg) / t) + 1e-8).mean())
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).cpu()


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
phis = {}
cache = f"{HERE}/embeddings/robust_{DS}_phis.npz"
if os.path.exists(cache):
    phis = dict(np.load(cache))


def save():
    np.savez_compressed(cache, **phis)


t0 = time.time()
for s in range(4):
    key = f"seed_{s}"
    if key in phis: continue
    z = train_hvgae_seeded(s)
    phis[key] = phi_of(*lorentz_fns(z, c_main)); save()
    print(f"[{DS}] {key} done {time.time()-t0:.0f}s", flush=True)
for enc_name, cls in [("SAGE", SAGEConv), ("GCN", GCNConv), ("GAT", GATConv)]:
    for kind in ["lorentz", "poincare", "euclidean"]:
        key = f"{enc_name}_{kind}"
        if key in phis: continue
        z = train_grid(cls, kind)
        fns = lorentz_fns(z, 1.0) if kind == "lorentz" else poincare_fns(z, 1.0) if kind == "poincare" else euclid_fns(z)
        phis[key] = phi_of(*fns); save()
        print(f"[{DS}] {key} done {time.time()-t0:.0f}s", flush=True)
if "random" not in phis:
    torch.manual_seed(0)
    phis["random"] = phi_of(*euclid_fns(torch.randn(N, 16))); save()


def rho(a, b): return float(spearmanr(phis[a], phis[b])[0])
encs, mans = ["SAGE", "GCN", "GAT"], ["lorentz", "poincare", "euclidean"]
cells = [f"{e}_{m}" for e in encs for m in mans]
seed_pairs = [rho(f"seed_{i}", f"seed_{j}") for i in range(4) for j in range(i + 1, 4)]
geo = [rho(f"{e}_{m1}", f"{e}_{m2}") for e in encs for i, m1 in enumerate(mans) for m2 in mans[i + 1:]]
enc = [rho(f"{e1}_{m}", f"{e2}_{m}") for m in mans for i, e1 in enumerate(encs) for e2 in encs[i + 1:]]
gcn_vs = [rho(f"GCN_{m1}", f"{e}_{m2}") for m1 in mans for e in ["SAGE", "GAT"] for m2 in mans]
null = [rho("random", cc) for cc in cells]
lor_poi = [rho(f"{e}_lorentz", f"{e}_poincare") for e in encs]
hyp_euc = [rho(f"{e}_{m}", f"{e}_euclidean") for e in encs for m in ["lorentz", "poincare"]]
summ = lambda v: {"n": len(v), "min": min(v), "max": max(v), "mean": float(np.mean(v))}
res = {"dataset": DS, "K": KSTAR, "c_main": c_main,
       "seed_floor": summ(seed_pairs), "geometry_encoder_fixed": summ(geo),
       "lorentz_vs_poincare": summ(lor_poi), "hyperbolic_vs_euclidean": summ(hyp_euc),
       "encoder_geometry_fixed": summ(enc), "gcn_vs_sage_gat": summ(gcn_vs), "random_null": summ(null),
       "matrix": {a: {b: rho(a, b) for b in cells + ["random"]} for a in cells + ["random"]},
       "seconds": time.time() - t0}
json.dump(res, open(OUT, "w"), indent=1)
for k in ["seed_floor", "geometry_encoder_fixed", "lorentz_vs_poincare", "hyperbolic_vs_euclidean",
          "encoder_geometry_fixed", "gcn_vs_sage_gat", "random_null"]:
    v = res[k]; print(f"[{DS}] {k:24s} n={v['n']:2d}  {v['min']:.3f}-{v['max']:.3f}  mean {v['mean']:.3f}")
