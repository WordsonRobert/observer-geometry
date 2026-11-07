"""
cross_dataset_transfer.py
=========================
Is Phi the same function of the classical centralities across graphs? An XGBoost model
fit on one dataset is evaluated on the others. Features are z-scored per dataset and Phi is
rank-normalized, so scale differences between graphs cannot by themselves explain a drop.
Compare off-diagonal entries with held-out within-dataset fits
(analysis/transfer_heldout.py), not with the in-sample diagonal.

Emits: results/cross_dataset_transfer.json

Usage:
    python analysis/cross_dataset_transfer.py
"""
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from _common import (DISPLAY, FEATURE_NAMES, MissingArtifact, RESULTS,
                     load_features, load_phi, write_json, cli_datasets)

N_SEEDS = 3


def zscore(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def rank_normalize(y):
    """
    Phi's absolute scale is not comparable across graphs (different N, different
    mass). Map to [0,1] by rank so R^2 across datasets is meaningful.
    """
    order = np.argsort(np.argsort(y))
    return order / max(len(y) - 1, 1)


def main():
    names = cli_datasets(sys.argv)

    data = {}
    for name in names:
        try:
            y, _ = load_phi(name)
            X = load_features(name)
            if X.shape[0] != y.shape[0]:
                raise MissingArtifact(
                    f"feature rows {X.shape[0]} != phi rows {y.shape[0]}")
            data[name] = (zscore(X), rank_normalize(y))
            print(f"  loaded {DISPLAY[name]}: N={X.shape[0]}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    avail = list(data.keys())
    if len(avail) < 2:
        print("Need at least 2 datasets. Nothing to do.")
        return

    matrix = {a: {} for a in avail}
    for src in avail:
        X_tr, y_tr = data[src]
        models = []
        for seed in range(N_SEEDS):
            m = XGBRegressor(n_estimators=300, max_depth=6,
                             random_state=seed, verbosity=0)
            m.fit(X_tr, y_tr)
            models.append(m)

        for dst in avail:
            X_te, y_te = data[dst]
            r2s, rhos = [], []
            for m in models:
                p = m.predict(X_te)
                r2s.append(r2_score(y_te, p))
                rhos.append(spearmanr(y_te, p)[0])
            matrix[src][dst] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
                "spearman_mean": float(np.mean(rhos)),
                "spearman_std": float(np.std(rhos)),
                "is_self": src == dst,
            }
            tag = " (self, in-sample)" if src == dst else ""
            print(f"  {DISPLAY[src]:<12} -> {DISPLAY[dst]:<12} "
                  f"R2={np.mean(r2s):+.3f}  rho={np.mean(rhos):+.3f}{tag}")

    self_r2 = [matrix[a][a]["r2_mean"] for a in avail]
    cross_r2 = [matrix[a][b]["r2_mean"] for a in avail for b in avail if a != b]
    self_rho = [matrix[a][a]["spearman_mean"] for a in avail]
    cross_rho = [matrix[a][b]["spearman_mean"] for a in avail for b in avail if a != b]

    out = {
        "note": ("Features z-scored per dataset; Phi rank-normalized to [0,1] "
                 "so cross-graph R^2 is comparable. Diagonal is in-sample and "
                 "therefore optimistic -- compare off-diagonal to the "
                 "within-dataset held-out numbers in reconstruction_plus.json, "
                 "not to the diagonal here."),
        "datasets": [DISPLAY[a] for a in avail],
        "keys": avail,
        "matrix": matrix,
        "summary": {
            "mean_self_r2": float(np.mean(self_r2)),
            "mean_cross_r2": float(np.mean(cross_r2)),
            "r2_transfer_drop": float(np.mean(self_r2) - np.mean(cross_r2)),
            "mean_self_spearman": float(np.mean(self_rho)),
            "mean_cross_spearman": float(np.mean(cross_rho)),
            "spearman_transfer_drop": float(np.mean(self_rho) - np.mean(cross_rho)),
            "n_negative_r2_transfers": int(sum(1 for v in cross_r2 if v < 0)),
            "n_cross_pairs": len(cross_r2),
        },
        "feature_names": FEATURE_NAMES,
        "n_seeds": N_SEEDS,
    }
    write_json(out, RESULTS / "cross_dataset_transfer.json")

    s = out["summary"]
    print(f"\n  mean self R2   = {s['mean_self_r2']:+.3f}")
    print(f"  mean cross R2  = {s['mean_cross_r2']:+.3f}  "
          f"(drop {s['r2_transfer_drop']:.3f})")
    print(f"  negative-R2 transfers: {s['n_negative_r2_transfers']}/{s['n_cross_pairs']}")
    if s["mean_cross_r2"] < 0.2:
        print("\n  READ: transfer collapses. Phi is a different function of the "
              "classical set on each graph -> strong distinctiveness claim.")
    elif s["r2_transfer_drop"] < 0.15:
        print("\n  READ: transfer largely holds. Phi is close to a fixed nonlinear "
              "functional of classical descriptors -> frame the contribution as "
              "computational/geometric rather than informational.")
    else:
        print("\n  READ: partial transfer. Report the per-pair matrix, not a mean.")


if __name__ == "__main__":
    main()
