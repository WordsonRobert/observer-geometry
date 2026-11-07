"""
budget_value.py
===============
Budget distributions and value convergence, from the exact visit probabilities.
  * For a budget law P with probability generating function G_P,
      Phi_P(s) = sum_u w_u G_P(1 - p_u) / sum_u w_u.
    Compares the point mass at K* with geometric (mean K*), Poisson (mean K*) and uniform
    {1..100} laws by Spearman agreement.
  * Limit Phi_inf(s) = kernel-mass fraction of vertices no trajectory can reach (p_u = 0).
    Since every reachable vertex has p_u >= eps,
      0 <= Phi_K - Phi_inf <= (1 - eps)^K (1 - Phi_inf).
    Reports the mean absolute and relative gaps.
Emits results/budget_value.json
"""
import sys
import numpy as np
from scipy.stats import spearmanr
from _common import RESULTS, DISPLAY, cli_datasets, load_exact, kstar, write_json


def main():
    out = {}
    for name in cli_datasets(sys.argv):
        d = load_exact(name); K = kstar(name); N = len(d["W"])
        ptr, P, wU, W = d["ptr"], d["P"].astype(float), d["wU"].astype(float), d["W"]
        srcs = np.repeat(np.arange(N), np.diff(ptr)); q = 1 - P
        def phi_from_G(G):
            o = np.zeros(N); np.add.at(o, srcs, wU * (1 - G)); return 1 - o / W
        point = d["phiK"][K - 1].astype(float)
        a = 1 / K
        laws = {"geometric_mean_Kstar": phi_from_G(a * q / (1 - (1 - a) * q)),
                "poisson_mean_Kstar": phi_from_G(np.exp(K * (q - 1))),
                "uniform_1_100": d["phiK"][:100].astype(float).mean(0)}
        reach = np.zeros(N); np.add.at(reach, srcs, wU); phi_inf = 1 - reach / W
        gap = d["phiK"].astype(float) - phi_inf[None]
        rel = gap / np.maximum(1 - phi_inf[None], 1e-12)
        out[name] = {"K_star": K,
                     "rank_agreement_point_vs": {k: float(spearmanr(point, v)[0]) for k, v in laws.items()},
                     "rank_agreement_point_vs_limit": float(spearmanr(point, phi_inf)[0]),
                     "mean_gap_at_K": {str(k): float(gap[k - 1].mean()) for k in (1, 5, 10, 20, K, 50, 100)},
                     "mean_relative_gap_at_K": {str(k): float(rel[k - 1].mean()) for k in (1, 5, 10, 20, K, 50, 100)},
                     "max_gap_at_Kstar": float(gap[K - 1].max()),
                     "bound_at_Kstar": float((1 - 1 / 50) ** K),
                     "min_positive_visit_probability": float(d["pmin"][np.isfinite(d["pmin"])].min())}
        o = out[name]
        print(f"  {DISPLAY[name]:<11} K*={K} laws={ {k: round(v, 3) for k, v in o['rank_agreement_point_vs'].items()} } "
              f"gap@K*={o['mean_gap_at_K'][str(K)]:.4f} rel={o['mean_relative_gap_at_K'][str(K)]:.3f}")
    write_json(out, RESULTS / "budget_value.json")


if __name__ == "__main__":
    main()
