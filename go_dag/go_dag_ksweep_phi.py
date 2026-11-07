"""GO DAG (rebuilt without alternate IDs): Monte Carlo budget sweep and Monte Carlo Phi,
with the same protocol as run_everything.py --stages k_sweep / phi:
  delta = 1, eps = 1/50, step cap 2000, degree cap 256, vectorized simulator;
  sweep K = 5..100 (step 5), one realization each, anchored to K = 100;
  K* = smallest K from which agreement with K = 100 stays >= 0.98;
  Phi at K* = mean of 3 independent realizations (seeds 0, 1, 2).
Usage: python go_dag/go_dag_ksweep_phi.py
Writes: go_dag/k_sweep_full_go_dag.json, go_dag/hvgae_embedding/phi_full_meanstd.pt
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import torch, numpy as np
from scipy.stats import spearmanr
from phi_utils import load_embedding
from vectorized_simulate import vectorized_phi_at_K_all_nodes

OUT = HERE
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)
from go_dag_train_embedding import build_bp_dag, to_pyg_data
data, nodes, _ = to_pyg_data(build_bp_dag(f"{HERE}/data/go-basic.obo"))
ei = data.edge_index; N = data.num_nodes
z, man, c = load_embedding(f"{HERE}/hvgae_embedding/embedding.pt", f"{HERE}/hvgae_embedding/chosen_curvature.txt")
z = z.detach()
print(f"GO DAG: N={N}, directed edges={ei.shape[1]}, c={c}, device={dev}", flush=True)
kw = dict(eps=1/50, delta=1.0, num_nodes=N, n_repeats=1, source_batch_size=512,
          device=dev, max_degree_cap=256)
t0 = time.time()
torch.manual_seed(0)
final = vectorized_phi_at_K_all_nodes(ei, z, man, K=100, **kw).numpy()
sweep = []
for K in range(5, 101, 5):
    phik = vectorized_phi_at_K_all_nodes(ei, z, man, K=K, **kw).numpy()
    rho = float(spearmanr(phik, final)[0]); sweep.append({"K": K, "rho_vs_K100": rho})
    print(f"  K={K:3d}  rho vs K=100 = {rho:.4f}   ({time.time()-t0:.0f}s)", flush=True)
r = [e["rho_vs_K100"] for e in sweep]; ks = [e["K"] for e in sweep]
k98 = next(k for i, k in enumerate(ks) if all(x >= 0.98 for x in r[i:]))
k99 = next(k for i, k in enumerate(ks) if all(x >= 0.99 for x in r[i:]))
json.dump({"anchored_sweep": sweep, "K_star": k98, "plateau_K": k99}, open(f"{OUT}/k_sweep_full_go_dag.json", "w"), indent=2)
print(f"K* (0.98 criterion) = {k98};  0.99 criterion = {k99}", flush=True)
vals = []
for seed in range(3):
    torch.manual_seed(seed)
    vals.append(vectorized_phi_at_K_all_nodes(ei, z, man, K=k98, **kw))
    print(f"  Phi realization {seed+1}/3 done ({time.time()-t0:.0f}s)", flush=True)
V = torch.stack(vals)
torch.save({"mean": V.mean(0), "std": V.std(0), "seeds": [0, 1, 2], "K_star": k98,
            "computed_with": "vectorized_simulate.py, 3 seeds x 1 realization"}, f"{OUT}/hvgae_embedding/phi_full_meanstd.pt")
print(f"saved go_dag/hvgae_embedding/phi_full_meanstd.pt  (total {time.time()-t0:.0f}s)", flush=True)
