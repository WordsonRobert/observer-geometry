"""
sensitivity.py
==============
Sensitivity of the exact descriptor to the decay rate delta and the termination threshold
eps. For every exact artifact available under <dataset>/exact/, reports
  * Spearman agreement of the node ranking with the default (delta = 1, eps = 1/50), and
  * the headline distinctiveness numbers at those parameters: best single classical
    centrality |rho|, and linear / XGBoost reconstruction from the seven centralities
    (3 random 80/20 splits).
delta = 1/sqrt(c) is the curvature-normalized decay (distances in units of the curvature
radius of the selected hyperboloid, whose sectional curvature is -1/c).

Emits results/sensitivity.json
Usage: python analysis/sensitivity.py [--datasets ...]
"""
import sys, re
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from _common import (RESULTS, DISPLAY, FEATURE_NAMES, cli_datasets, folder, load_features,
                     exact_path, kstar, write_json, CAPS)


def headline(y, X):
    rh = [abs(spearmanr(y, X[:, j])[0]) if X[:, j].std() > 0 else 0.0 for j in range(X.shape[1])]
    lin, xg, rr = [], [], []
    for seed in range(3):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        lin.append(r2_score(y[te], LinearRegression().fit(X[tr], y[tr]).predict(X[te])))
        p = XGBRegressor(n_estimators=300, max_depth=6, random_state=seed, verbosity=0).fit(X[tr], y[tr]).predict(X[te])
        xg.append(r2_score(y[te], p)); rr.append(spearmanr(y[te], p)[0])
    j = int(np.argmax(rh))
    return {"best_classical": FEATURE_NAMES[j], "best_classical_abs_rho": float(rh[j]),
            "linear_r2": float(np.mean(lin)), "xgb_r2": float(np.mean(xg)), "xgb_rank_recovery": float(np.mean(rr))}


def main():
    out = {}
    for name in cli_datasets(sys.argv):
        K = kstar(name); X = np.nan_to_num(load_features(name))
        base = np.load(exact_path(name))["phiK"][K - 1].astype(np.float64)
        rows = {}
        for p in sorted((folder(name) / "exact").glob(f"exact_eps*_cap{CAPS[name]}.npz")):
            m = re.match(r"exact_eps([\d.]+)_d([\d.]+)_cap", p.name)
            eps, delta = float(m.group(1)), float(m.group(2))
            y = np.load(p)["phiK"][K - 1].astype(np.float64)
            key = f"eps={eps:g},delta={delta:g}"
            rows[key] = {"eps": eps, "delta": delta, "rank_agreement_with_default": float(spearmanr(y, base)[0]),
                         **headline(y, X)}
            r = rows[key]
            print(f"  {DISPLAY[name]:<11} {key:<22} agree={r['rank_agreement_with_default']:.3f} "
                  f"best={r['best_classical_abs_rho']:.3f} ({r['best_classical']})  lin={r['linear_r2']:.3f} "
                  f"xgb={r['xgb_r2']:.3f} rank={r['xgb_rank_recovery']:.3f}", flush=True)
        out[name] = {"K_star": K, "variants": rows}
    write_json(out, RESULTS / "sensitivity.json")


if __name__ == "__main__":
    main()
