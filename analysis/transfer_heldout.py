"""
transfer_heldout.py
===================
Within-dataset held-out comparator for cross_dataset_transfer.py, under the same protocol
(features z-scored per dataset, Phi rank-normalized, XGBoost 300 trees depth 6): the
cross-dataset numbers are out-of-sample, so they are compared with held-out (80/20) rather
than in-sample fits.
Emits results/transfer_heldout.json
"""
import sys, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from _common import RESULTS, DISPLAY, cli_datasets, load_features, load_phi, write_json


def z(X):
    s = X.std(0, keepdims=True); s[s < 1e-12] = 1; return (X - X.mean(0, keepdims=True)) / s


def main():
    out = {}
    for name in cli_datasets(sys.argv):
        X = z(np.nan_to_num(load_features(name))); y, _ = load_phi(name)
        y = np.argsort(np.argsort(y)) / max(len(y) - 1, 1)
        rho, r2 = [], []
        for seed in range(3):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
            p = XGBRegressor(n_estimators=300, max_depth=6, random_state=seed, verbosity=0).fit(X[tr], y[tr]).predict(X[te])
            rho.append(spearmanr(y[te], p)[0]); r2.append(r2_score(y[te], p))
        out[name] = {"heldout_spearman": float(np.mean(rho)), "heldout_r2": float(np.mean(r2))}
        print(f"  {DISPLAY[name]:<11} held-out rho={out[name]['heldout_spearman']:.3f}")
    T = json.load(open(RESULTS / "cross_dataset_transfer.json")); keys = T["keys"]; M = T["matrix"]
    off = [M[a][b]["spearman_mean"] for a in keys for b in keys if a != b]
    out["summary"] = {"mean_in_sample": float(np.mean([M[a][a]["spearman_mean"] for a in keys])),
                      "mean_heldout_within": float(np.mean([out[k]["heldout_spearman"] for k in keys if k in out])),
                      "mean_cross": float(np.mean(off)), "best_cross": float(max(off))}
    print("  ", out["summary"])
    write_json(out, RESULTS / "transfer_heldout.json")


if __name__ == "__main__":
    main()
