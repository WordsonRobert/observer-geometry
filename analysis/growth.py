"""
growth.py
=========
Finite-graph counterpart of the regular-tree threshold (Theorem 2(c)).
On the k-regular tree, mu(V) diverges iff delta <= log(k-1): shell counts grow like
exp(h r) with h = log(k-1). On a finite embedded graph the volume growth saturates, so
mu(V) is finite for every delta and only a crossover remains. For 300 random observers per
dataset this script estimates
  * h_hat: slope of log #{u : d(s,u) <= r} over the bulk of the distance distribution
    (2nd to 50th percentile), the embedded volume-growth rate;
  * delta_c: the smallest delta at which the kernel-weighted shell density N(r) e^{-delta r}
    peaks within the inner quartile of distances (near shells dominate mu(V)).
For GO DAG it also reports the depth-shell counts from the root term.
Emits results/growth.json
"""
import sys
import numpy as np, torch
from _common import RESULTS, DISPLAY, cli_datasets, folder, write_json


def main():
    out = {}
    deltas = np.linspace(0.05, 8, 160)
    for name in cli_datasets(sys.argv):
        emb = folder(name) / "hvgae_embedding"
        z = torch.load(emb / "embedding.pt", weights_only=False, map_location="cpu").double()
        c = float(open(emb / "chosen_curvature.txt").read()); N = z.shape[0]
        S = np.random.RandomState(0).choice(N, min(300, N), replace=False)
        J = z[S].clone(); J[:, 0] *= -1
        D = (np.sqrt(c) * torch.acosh(torch.clamp(-(J @ z.T) / c, min=1 + 1e-12))).numpy()
        hs, dc = [], []
        for row in D:
            d = np.sort(row[row > 1e-6])
            rr = np.linspace(np.quantile(d, 0.02), np.quantile(d, 0.5), 25)
            hs.append(np.polyfit(rr, np.log(np.maximum(np.searchsorted(d, rr, side="right"), 1)), 1)[0])
            hist, edges = np.histogram(d, bins=60); mids = 0.5 * (edges[1:] + edges[:-1]); q1 = np.quantile(d, 0.25)
            dc.append(next((dl for dl in deltas if mids[np.argmax(hist * np.exp(-dl * mids))] <= q1), np.nan))
        hs, dc = np.array(hs), np.array(dc)
        r = {"curvature_parameter_c": c, "sectional_curvature": -1 / c,
             "h_hat_median": float(np.median(hs)), "h_hat_iqr": [float(np.quantile(hs, .25)), float(np.quantile(hs, .75))],
             "delta_crossover_median": float(np.nanmedian(dc)),
             "delta_in_curvature_units_at_delta1": float(np.sqrt(c))}
        if name == "go_dag":
            depth = torch.load(folder(name) / "data" / "node_depths.pt", weights_only=False).numpy()
            cnt = np.bincount(depth[depth >= 0]); g = np.log(cnt[1:] / cnt[:-1]); k = int(np.argmax(cnt))
            r.update({"root_depth_shell_counts": cnt.tolist(), "root_peak_depth": k,
                      "root_mean_log_growth_to_peak": float(g[:k].mean())})
        out[name] = r; print(f"  {DISPLAY[name]:<11} h_hat={r['h_hat_median']:.2f} delta_c={r['delta_crossover_median']:.2f} c={c}")
    write_json(out, RESULTS / "growth.json")


if __name__ == "__main__":
    main()
