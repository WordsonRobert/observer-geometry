"""
reconstruction_plus.py
======================
Everything Table 1 is missing, computed in one pass over the cached features.

Answers three reviewer objections at once:

  1. "R^2 is about values; your claim is about rankings."
     -> reports Spearman(Phi, Phi_hat) beside every R^2.

  2. "Hyperbolic radius correlates -0.606 with Phi on Cora, better than any
     classical descriptor, and it's not in your feature set."
     -> reports reconstruction with and without radius as an 8th feature.

  3. "A random 80/20 node split leaks: centralities are spatially smooth, so
     test nodes neighbour training nodes."
     -> reports a community-held-out split (Louvain) beside the random one.

Emits per dataset:   <folder>/reconstruction_plus.json
Emits pooled:        results/reconstruction_plus.json

Runtime: minutes. Features are read from _classical_features_cache.npy;
nothing expensive is recomputed. The only new graph work is Louvain, which is
near-linear.

Usage:
    python analysis/reconstruction_plus.py
    python analysis/reconstruction_plus.py --datasets cora,citeseer
    python analysis/reconstruction_plus.py --no-community    # skip Louvain
"""
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from _common import (DATASETS, DISPLAY, FEATURE_NAMES, MissingArtifact,
                     RESULTS, folder, load_features, load_graph_int, load_phi,
                     load_radius, write_json, cli_datasets)

N_SEEDS = 5
TEST_FRAC = 0.2


def _fit_eval(X_tr, y_tr, X_te, y_te, seed):
    """Linear + XGB on one split. Returns both R^2 and rank recovery."""
    out = {}

    lin = LinearRegression().fit(X_tr, y_tr)
    p_lin = lin.predict(X_te)
    out["linear"] = {
        "r2": float(r2_score(y_te, p_lin)),
        "spearman": float(spearmanr(y_te, p_lin)[0]),
        "mae": float(mean_absolute_error(y_te, p_lin)),
    }

    xgb = XGBRegressor(n_estimators=300, max_depth=6,
                       random_state=seed, verbosity=0)
    xgb.fit(X_tr, y_tr)
    p_xgb = xgb.predict(X_te)
    out["xgboost"] = {
        "r2": float(r2_score(y_te, p_xgb)),
        "spearman": float(spearmanr(y_te, p_xgb)[0]),
        "mae": float(mean_absolute_error(y_te, p_xgb)),
    }
    return out


def _aggregate(runs):
    """mean/std over seeds for each model x metric."""
    agg = {}
    for model in ("linear", "xgboost"):
        for metric in ("r2", "spearman", "mae"):
            vals = np.array([r[model][metric] for r in runs], dtype=float)
            agg.setdefault(model, {})[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
            }
    return agg


def random_split_eval(X, y):
    runs = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=seed)
        runs.append(_fit_eval(X_tr, y_tr, X_te, y_te, seed))
    return _aggregate(runs)


def community_split_eval(X, y, G):
    """
    Louvain communities, then hold out whole communities until ~TEST_FRAC of
    nodes are covered. Removes the neighbour-leakage that a random node split
    allows. Falls back cleanly if Louvain is unavailable.
    """
    import networkx as nx
    try:
        comms = list(nx.community.louvain_communities(G, seed=0))
    except Exception as e:
        return {"error": f"louvain unavailable or failed: {e}"}

    rng = np.random.RandomState(0)
    order = rng.permutation(len(comms))
    N = X.shape[0]
    target = int(TEST_FRAC * N)

    runs = []
    for seed in range(N_SEEDS):
        order = np.random.RandomState(seed).permutation(len(comms))
        test_nodes, held = set(), 0
        for ci in order:
            if held >= target:
                break
            c = comms[ci]
            test_nodes |= set(c)
            held += len(c)
        te = np.array(sorted(n for n in test_nodes if 0 <= n < N), dtype=int)
        if len(te) < 10 or len(te) > N - 10:
            continue
        mask = np.ones(N, dtype=bool)
        mask[te] = False
        tr = np.where(mask)[0]
        runs.append(_fit_eval(X[tr], y[tr], X[te], y[te], seed))

    if not runs:
        return {"error": "no usable community split (communities too coarse)"}
    out = _aggregate(runs)
    out["n_communities"] = len(comms)
    out["n_splits_used"] = len(runs)
    return out


def run_one(name, do_community=True):
    print(f"\n=== reconstruction_plus: {DISPLAY[name]} ===")
    y, K_star = load_phi(name)
    X = load_features(name)
    if X.shape[0] != y.shape[0]:
        raise MissingArtifact(
            f"{name}: feature rows ({X.shape[0]}) != phi rows ({y.shape[0]}). "
            f"Delete {folder(name)}/_classical_features_cache.npy and rerun "
            f"the correlations stage.")

    result = {
        "dataset": DISPLAY[name],
        "n_nodes": int(X.shape[0]),
        "K_star": K_star,
        "feature_names": FEATURE_NAMES,
        "n_seeds": N_SEEDS,
        "test_frac": TEST_FRAC,
    }

    print("  [1/3] classical features only, random split")
    result["classical_random"] = random_split_eval(X, y)

    print("  [2/3] classical + hyperbolic radius, random split")
    try:
        r = load_radius(name)
        Xr = np.column_stack([X, r])
        result["radius_spearman_vs_phi"] = float(spearmanr(y, r)[0])
        result["classical_plus_radius_random"] = random_split_eval(Xr, y)
        result["feature_names_with_radius"] = FEATURE_NAMES + ["HyperbolicRadius"]
        # radius alone -- the number a reviewer will ask for
        result["radius_only_random"] = random_split_eval(r.reshape(-1, 1), y)
    except MissingArtifact as e:
        result["classical_plus_radius_random"] = {"error": str(e)}

    if do_community:
        print("  [3/3] classical features only, community-held-out split")
        try:
            G, N = load_graph_int(name)
            result["classical_community_holdout"] = community_split_eval(X, y, G)
        except Exception as e:
            result["classical_community_holdout"] = {"error": str(e)}
    else:
        result["classical_community_holdout"] = {"skipped": True}

    # one-line summary for the paper table
    cr = result["classical_random"]
    result["paper_row"] = {
        "linear_r2": cr["linear"]["r2"]["mean"],
        "nonlinear_r2": cr["xgboost"]["r2"]["mean"],
        "nonlinearity_gap": cr["xgboost"]["r2"]["mean"] - cr["linear"]["r2"]["mean"],
        "nonlinear_rank_recovery": cr["xgboost"]["spearman"]["mean"],
    }

    write_json(result, folder(name) / "reconstruction_plus.json")
    pr = result["paper_row"]
    print(f"  linear R2={pr['linear_r2']:.3f}  nonlinear R2={pr['nonlinear_r2']:.3f}  "
          f"gap={pr['nonlinearity_gap']:.3f}  rank rec={pr['nonlinear_rank_recovery']:.3f}")
    return result


def main():
    names = cli_datasets(sys.argv)
    do_community = "--no-community" not in sys.argv

    pooled = {}
    for name in names:
        try:
            pooled[name] = run_one(name, do_community=do_community)
        except MissingArtifact as e:
            print(f"[skip] {name}: {e}")
            pooled[name] = {"error": str(e)}
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            pooled[name] = {"error": f"{type(e).__name__}: {e}"}

    # the nonlinearity-gap ordering is the actual finding -- surface it
    rows = [(v["paper_row"]["nonlinearity_gap"], k, v["paper_row"])
            for k, v in pooled.items() if "paper_row" in v]
    rows.sort(reverse=True)
    pooled["_gap_ordering"] = [
        {"dataset": DISPLAY[k], **pr} for _, k, pr in rows
    ]

    write_json(pooled, RESULTS / "reconstruction_plus.json")

    print("\n--- nonlinearity gap ordering (the finding) ---")
    for gap, k, pr in rows:
        print(f"  {DISPLAY[k]:<12} linear={pr['linear_r2']:.3f}  "
              f"nonlinear={pr['nonlinear_r2']:.3f}  gap={gap:.3f}")


if __name__ == "__main__":
    main()
