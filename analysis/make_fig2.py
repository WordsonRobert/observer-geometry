"""
make_fig2.py
============
Figure 2 of the paper: (a) agreement of the ranking at budget K with the ranking at K = 100,
for the Monte Carlo sweep (<dataset>/k_sweep_full_<dataset>.json; one realization per K) and
for the exact descriptor; (b) mean relative gap (Phi_K - Phi_inf) / (1 - Phi_inf) of the exact
descriptor to its limit, with the bound (1 - eps)^K.
Writes figures/k_sweep_convergence.pdf (+ .png) and results/fig2_summary.json.
"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _common import ROOT, RESULTS, exact_path, kstar as _kstar
R = str(ROOT); OUT = str(ROOT / "figures" / "k_sweep_convergence.pdf")
DS = [(k, n, None, str(exact_path(k)), f"{k}/k_sweep_full_{k}.json", None)
      for k, n in [("cora", "Cora"), ("citeseer", "Citeseer"), ("go_dag", "GO DAG"),
                   ("string_ppi", "STRING-PPI"), ("obgn_arxiv", "ogbn-arXiv")]]
COL = {"cora": "#2b6a99", "citeseer": "#c4561d", "go_dag": "#3d8b4f", "string_ppi": "#8460a8", "obgn_arxiv": "#d39a20"}
MK = {"cora": "o", "citeseer": "s", "go_dag": "^", "string_ppi": "D", "obgn_arxiv": "v"}
kstar = {k: _kstar(k) for k, *_ in DS}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
Ks = np.arange(1, 101); eps = 1 / 50
summary = {}
for key, name, tag, path, ks_json, _ in DS:
    d = np.load(path); phiK = d["phiK"].astype(float); W = d["W"]
    sw = json.load(open(f"{R}/{ks_json}"))["anchored_sweep"]
    kk = [e["K"] for e in sw]; rr = [e["rho_vs_K100"] for e in sw]
    a1.plot(kk, rr, color=COL[key], marker=MK[key], ms=3, lw=1.1, label=name)
    ex = [spearmanr(phiK[k - 1], phiK[99])[0] for k in kk]
    a1.plot(kk, ex, color=COL[key], lw=0.9, ls="--", alpha=0.8)
    K = kstar[key]
    a1.plot([K], [rr[kk.index(K)] if K in kk else np.interp(K, kk, rr)], marker="o", mfc="none", mec=COL[key], ms=8, mew=1.2)
    # value convergence: Phi_inf = mass fraction never reachable
    ptr = d["ptr"]; wU = d["wU"].astype(float); N = len(W)
    srcs = np.repeat(np.arange(N), np.diff(ptr)); reach = np.zeros(N); np.add.at(reach, srcs, wU)
    phi_inf = 1 - reach / W
    gap = (phiK - phi_inf[None, :]).mean(1)
    rel = ((phiK - phi_inf[None, :]) / np.maximum(1 - phi_inf[None, :], 1e-12)).mean(1)
    a2.semilogy(Ks, rel, color=COL[key], lw=1.2, label=name)
    summary[key] = {"mean_gap_at_Kstar": float(gap[K - 1]), "mean_rel_gap_at_Kstar": float(rel[K - 1]),
                    "exact_rho_Kstar_vs_100": float(spearmanr(phiK[K - 1], phiK[99])[0])}
a2.semilogy(Ks, (1 - eps) ** Ks, color="0.4", lw=1.0, ls=":", label=r"bound $(1-\varepsilon)^K$")
a1.axhline(0.98, color="0.5", lw=0.8, ls=":")
a1.set_xlabel(r"Observation budget $K$"); a1.set_ylabel(r"Rank agreement $\rho$ with $K=100$")
a1.set_title("(a) Ranking", fontsize=9); a1.set_ylim(0.6, 1.005); a1.grid(alpha=0.25, lw=0.5)
a1.plot([], [], color="0.3", lw=1.1, marker="o", ms=3, label="Monte Carlo")
a1.plot([], [], color="0.3", lw=0.9, ls="--", label="exact")
a1.legend(fontsize=6.5, loc="lower right", ncol=2, frameon=False)
a2.set_xlabel(r"Observation budget $K$")
a2.set_ylabel(r"$(\Phi_K-\Phi_\infty)/(1-\Phi_\infty)$, mean")
a2.set_title("(b) Value (exact)", fontsize=9); a2.grid(alpha=0.25, lw=0.5, which="both")
a2.legend(fontsize=6.5, loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
json.dump(summary, open(RESULTS / "fig2_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
