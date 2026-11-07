"""
closer_baselines.py
===================
Compares Phi with measures closer to it than classical centralities:
  total kernel mass mu(s,V) ("distance-weighted volume"), expected hidden count E|H_K|
  (unweighted coverage), hyperbolic radius, random-walk accessibility exp(H(P^h_s)) for
  h = 2, 3 (Travencolo & Costa 2008; ordinary random walk, de Arruda et al. 2014), and
  eccentricity. Reports Spearman correlations, partial correlations given radius, the
  decomposition log(1 - Phi) = log C(s) - log mu(s,V) with C(s) the expected observed kernel
  mass, and reconstruction of Phi (XGBoost / linear; random 80/20 split and Louvain
  community holdout) from classical centralities, closer measures, and both.
Uses the exact descriptor at K* (analysis/exact_phi_all.py).
Accessibility and eccentricity use a radius-stratified sample of 12,000 nodes on graphs
with more than 100,000 nodes.

Emits <dataset>/closer_baselines.json
Usage: python analysis/closer_baselines.py [--datasets cora,citeseer]
"""
import sys, time
import numpy as np, scipy.sparse as sp, torch, networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from _common import (DISPLAY, cli_datasets, folder, load_exact, load_features, load_radius,
                     kstar, write_json)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def res(a):
        b = np.column_stack([np.ones_like(rz), rz]); return a - b @ np.linalg.lstsq(b, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def run_one(name):
    K = kstar(name); d = load_exact(name); N = len(d["W"])
    phi = d["phiK"][K - 1].astype(np.float64); W = d["W"]
    C = W * (1 - phi); hid_count = N - d["nvisK"][K - 1].astype(np.float64)
    radius = load_radius(name)
    rp, ci, _ = torch.load(folder(name) / "hvgae_embedding" / "transition_kernel.pt",
                           weights_only=False, map_location="cpu")
    rp, ci = rp.numpy(), ci.numpy()
    A = sp.csr_matrix((np.ones(len(ci)), ci, rp), shape=(N, N)); deg = np.asarray(A.sum(1)).ravel()
    if N > 100_000:
        rng = np.random.RandomState(0); order = np.argsort(radius)
        S = np.sort(np.concatenate([rng.choice(order[t * N // 3:(t + 1) * N // 3], 4000, replace=False)
                                    for t in range(3)]))
    else:
        S = np.arange(N)
    P = (sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1), 0.0)) @ A).tocsr()
    acc = {2: np.ones(len(S)), 3: np.ones(len(S))}
    for a in range(0, len(S), 256):
        rows = S[a:a + 256]; X = P[rows]
        for h in (2, 3):
            X = X @ P; Xd = X.toarray()
            with np.errstate(divide="ignore", invalid="ignore"):
                H = -np.nansum(np.where(Xd > 0, Xd * np.log(Xd), 0.0), axis=1)
            acc[h][a:a + 256] = np.where(deg[rows] == 0, 1.0, np.exp(H))
    ecc = np.zeros(len(S))
    for a in range(0, len(S), 512):
        D = shortest_path(A, method="D", unweighted=True, indices=S[a:a + 512])
        D[np.isinf(D)] = -1; ecc[a:a + 512] = D.max(1)
    full = {"total_kernel_mass": W, "expected_observed_mass": C, "expected_hidden_count": hid_count,
            "radius": radius, "degree": deg}
    samp = {"accessibility_h2": acc[2], "accessibility_h3": acc[3], "eccentricity": ecc}
    corr = {k: {"spearman": float(spearmanr(phi, v)[0]), "partial_given_radius": partial_spearman(phi, v, radius)}
            for k, v in full.items()}
    corr.update({k: {"spearman": float(spearmanr(phi[S], v)[0]),
                     "partial_given_radius": partial_spearman(phi[S], v, radius[S])} for k, v in samp.items()})
    y = np.log1p(-phi); lc, lw = np.log(C), np.log(W)
    decomp = {"var_logC_over_var": float(lc.var() / y.var()), "var_logW_over_var": float(lw.var() / y.var()),
              "corr_logC_logW": float(np.corrcoef(lc, lw)[0, 1]),
              "spearman_W_radius": float(spearmanr(W, radius)[0]), "spearman_C_radius": float(spearmanr(C, radius)[0])}
    # reconstruction
    Xc = np.nan_to_num(load_features(name))[S]; yS = phi[S]
    closer = np.column_stack([W[S], hid_count[S], radius[S], acc[2], acc[3], ecc])
    sets = {"classical7": Xc, "closer6": closer, "classical7+closer6": np.column_stack([Xc, closer])}
    G = nx.Graph(); G.add_nodes_from(range(N)); src = np.repeat(np.arange(N), np.diff(rp)); m = src < ci
    G.add_edges_from(zip(src[m].tolist(), ci[m].tolist()))
    comms = list(nx.community.louvain_communities(G, seed=0)); pos = {n: i for i, n in enumerate(S)}
    recon = {}
    for k, X in sets.items():
        rs, cs = [], []
        for seed in range(5):
            tr, te = train_test_split(np.arange(len(yS)), test_size=0.2, random_state=seed)
            order = np.random.RandomState(seed).permutation(len(comms)); test = set(); held = 0
            for c in order:
                if held >= int(0.2 * N): break
                test |= comms[c]; held += len(comms[c])
            te2 = np.array(sorted(pos[n] for n in test if n in pos)); msk = np.ones(len(yS), bool); msk[te2] = False
            for split, (a_tr, a_te), acc_list in [("random", (tr, te), rs), ("community", (np.where(msk)[0], te2), cs)]:
                lin = LinearRegression().fit(X[a_tr], yS[a_tr]).predict(X[a_te])
                xg = XGBRegressor(n_estimators=300, max_depth=6, random_state=seed, verbosity=0).fit(X[a_tr], yS[a_tr]).predict(X[a_te])
                acc_list.append((r2_score(yS[a_te], lin), r2_score(yS[a_te], xg), spearmanr(yS[a_te], xg)[0]))
        rs, cs = np.array(rs), np.array(cs)
        recon[k] = {"linear_r2": float(rs[:, 0].mean()), "xgb_r2": float(rs[:, 1].mean()), "xgb_r2_std": float(rs[:, 1].std()),
                    "xgb_rank_recovery": float(rs[:, 2].mean()), "community_xgb_r2": float(cs[:, 1].mean()),
                    "community_rank_recovery": float(cs[:, 2].mean())}
        print(f"  {DISPLAY[name]:<11} {k:<20} " + "  ".join(f"{a}={b:.3f}" for a, b in recon[k].items()), flush=True)
    out = {"dataset": DISPLAY[name], "K_star": K, "n_nodes": N, "n_sampled": int(len(S)),
           "correlations": corr, "log_decomposition": decomp, "reconstruction": recon}
    write_json(out, folder(name) / "closer_baselines.json")
    return out


if __name__ == "__main__":
    for n in cli_datasets(sys.argv):
        run_one(n)
