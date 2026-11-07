"""
panel_normalized.py
===================
Empirical tightness of the normalized stability bounds (Corollary on normalized stability;
Lean: muN_stability_embedding, muN_stability_observer, muN_stability_hidden).

Same benchmark, seeds and perturbations as panel_a.py and panel_d_shared_h.py, but the
quantity is the normalized remaining geometry mu(v,H)/mu(v,V) and the bounds are
  embedding:  tanh(delta * eta_v / 2),  eta_v = max_u |d_f(v,u) - d_g(v,u)| over all u in V
  observer:   tanh(delta * d(v,v') / 2), hidden region fixed
  hidden:     |H xor H'| / mu(v,V)
Emits synthetic/results/panel_normalized.json
Usage: python synthetic/panel_normalized.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import networkx as nx
import torch

from hetero_graph import (build_heterogeneous_graph, train_hyperbolic_embedding,
                          project_to_ball, all_pairs_hyp_dist)

torch.manual_seed(0)
DELTA = 1.0
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1]
N_PERTURB = 30
OBS_RADIUS = 2
N_SEEDS = 5
N_REFERENCE_OBSERVERS = 10
N_PARTNERS_PER_RING = 5
RING_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
KS = [1, 2, 4, 8, 16, 32]
OUT = HERE / "results" / "panel_normalized.json"


def ball(G, v, r):
    return set(nx.single_source_shortest_path_length(G, v, cutoff=r))


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d_f = all_pairs_hyp_dist(z).numpy()
    W_f = np.exp(-DELTA * d_f)
    tot_f = W_f.sum(1)
    hidden = [np.array(sorted(set(range(N)) - ball(G, v, OBS_RADIUS)), dtype=int) for v in range(N)]
    hidden = [h[h != v] for v, h in enumerate(hidden)]

    # embedding perturbation
    emb = {s: {"obs": [], "bound": [], "ratio": []} for s in SIGMAS}
    rng = np.random.RandomState(seed + 1000)
    for sigma in [0.0] + SIGMAS:
        reps = 1 if sigma == 0.0 else N_PERTURB
        for _ in range(reps):
            if sigma == 0.0:
                continue
            noise = torch.tensor(rng.normal(0, sigma, size=z.shape))
            d_g = all_pairs_hyp_dist(project_to_ball(z + noise)).numpy()
            W_g = np.exp(-DELTA * d_g); tot_g = W_g.sum(1)
            obs, bnd = [], []
            for v in range(N):
                H = hidden[v]
                if len(H) == 0 or len(H) == N:
                    continue
                a = W_f[v, H].sum() / tot_f[v]; b = W_g[v, H].sum() / tot_g[v]
                eta = np.abs(d_f[v] - d_g[v]).max()
                obs.append(abs(a - b)); bnd.append(np.tanh(DELTA * eta / 2))
            obs, bnd = np.array(obs), np.array(bnd)
            emb[sigma]["obs"].append(obs.mean()); emb[sigma]["bound"].append(bnd.mean())
            emb[sigma]["ratio"].append((obs / bnd).max())

    # observer perturbation, shared hidden region
    rng = np.random.RandomState(seed + 5000)
    refs = rng.choice(N, size=min(N_REFERENCE_OBSERVERS, N), replace=False)
    nb = len(RING_EDGES) - 1
    ob = {b: {"obs": [], "bound": [], "d": []} for b in range(nb)}
    for ref in refs:
        ref = int(ref)
        H = np.array([u for u in range(N) if u not in ball(G, ref, OBS_RADIUS)], dtype=int)
        if len(H) < 10:
            continue
        muN = W_f[:, H].sum(1) / tot_f
        dists = d_f[ref].copy(); dists[ref] = np.inf
        for b in range(nb):
            lo, hi = RING_EDGES[b], RING_EDGES[b + 1]
            in_bin = np.where((dists >= lo) & (dists < hi))[0]
            if len(in_bin) == 0:
                continue
            for p in rng.choice(in_bin, size=min(N_PARTNERS_PER_RING, len(in_bin)), replace=False):
                dvv = float(d_f[ref, p])
                ob[b]["obs"].append(abs(muN[ref] - muN[p]))
                ob[b]["bound"].append(np.tanh(DELTA * dvv / 2)); ob[b]["d"].append(dvv)

    # hidden-region perturbation
    rng = np.random.RandomState(seed + 3000)
    hid = {k: {"obs": [], "bound": []} for k in KS}
    for v in rng.choice(N, size=min(20, N), replace=False):
        H = set(hidden[int(v)].tolist())
        if len(H) < 40:
            continue
        rest = sorted(set(range(N)) - H - {int(v)})
        for k in KS:
            for _ in range(100):
                n_del = k // 2; n_add = k - n_del
                if n_add > len(rest):
                    continue
                dl = rng.choice(sorted(H), size=n_del, replace=False) if n_del else []
                ad = rng.choice(rest, size=n_add, replace=False)
                H2 = (H - set(np.atleast_1d(dl).tolist())) | set(ad.tolist())
                a = W_f[int(v), sorted(H)].sum() / tot_f[int(v)]
                b = W_f[int(v), sorted(H2)].sum() / tot_f[int(v)]
                hid[k]["obs"].append(abs(a - b)); hid[k]["bound"].append(k / tot_f[int(v)])
    return emb, ob, hid, float(np.mean([len(h) for h in hidden])), float(tot_f.mean())


def main():
    E = {s: {"obs": [], "bound": [], "ratio": []} for s in SIGMAS}
    nb = len(RING_EDGES) - 1
    O = {b: {"obs": [], "bound": [], "d": []} for b in range(nb)}
    Hh = {k: {"obs": [], "bound": []} for k in KS}
    sizes, tots = [], []
    for seed in range(N_SEEDS):
        print(f"[normalized panels] seed {seed}", flush=True)
        e, o, h, hs, tm = run_one_graph(seed)
        sizes.append(hs); tots.append(tm)
        for s in SIGMAS:
            for key in E[s]:
                E[s][key].extend(e[s][key])
        for b in range(nb):
            for key in O[b]:
                O[b][key].extend(o[b][key])
        for k in KS:
            for key in Hh[k]:
                Hh[k][key].extend(h[k][key])
    out = {"delta": DELTA, "n_seeds": N_SEEDS, "mean_hidden_size": float(np.mean(sizes)),
           "mean_total_mass": float(np.mean(tots)), "embedding": [], "observer": [], "hidden": []}
    for s in SIGMAS:
        out["embedding"].append({"sigma": s, "mean_obs": float(np.mean(E[s]["obs"])),
                                 "mean_bound": float(np.mean(E[s]["bound"])),
                                 "slack": float(np.mean(E[s]["bound"]) / np.mean(E[s]["obs"])),
                                 "max_ratio": float(np.max(E[s]["ratio"]))})
    viol = 0
    for b in range(nb):
        if not O[b]["obs"]:
            continue
        ob, bd = np.array(O[b]["obs"]), np.array(O[b]["bound"])
        viol += int((ob > bd + 1e-12).sum())
        out["observer"].append({"ring": [RING_EDGES[b], RING_EDGES[b + 1]], "n_pairs": int(len(ob)),
                                "mean_d": float(np.mean(O[b]["d"])), "mean_obs": float(ob.mean()),
                                "mean_bound": float(bd.mean()), "mean_tightness": float(np.mean(ob / bd)),
                                "max_tightness": float(np.max(ob / bd))})
    out["observer_violations"] = viol
    for k in KS:
        ob, bd = np.array(Hh[k]["obs"]), np.array(Hh[k]["bound"])
        out["hidden"].append({"k": k, "n": int(len(ob)), "mean_obs": float(ob.mean()),
                              "mean_bound": float(bd.mean()), "slack": float(bd.mean() / ob.mean()),
                              "violations": int((ob > bd + 1e-12).sum())})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
