"""
panel_d_shared_h.py
===================
The observer-perturbation experiment, corrected.

WHY THIS EXISTS
    Section 5.1 states the bound for a SHARED hidden region:

        |mu_H(v) - mu_H(v')|  <=  |H| * delta * d_X(f(v), f(v'))

    The original panel_d.py gives every observer its OWN hidden region
    H(v) = V \\ B_r(v), so the differenced quantities are mu_{H(v)}(v) and
    mu_{H(v')}(v') -- both the observer and the hidden set move. The Lipschitz
    bound does not apply to that difference, so the panel never tested the
    theorem it was cited for.

    Here a single hidden region is FIXED and only the observer moves, which is
    exactly the theorem's setting, and the bound |H|*delta*d is overlaid.

    Keep the original panel too: varying H with the observer is a legitimate
    characterisation of descriptor smoothness under the natural
    observer-relative horizon. It just is not a test of mu_stability_V. Report
    them as two different things.

Emits: synthetic/results/panelD_shared_H.json

Runtime: a few minutes (5 graph instances, embedding dominates).

Usage:
    python synthetic/panel_d_shared_h.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import torch

from hetero_graph import (build_heterogeneous_graph, train_hyperbolic_embedding,
                          all_pairs_hyp_dist)

DELTA = 1.0
OBS_RADIUS = 2
N_SEEDS = 5
N_REFERENCE_OBSERVERS = 10      # how many distinct fixed hidden regions to test
N_PARTNERS_PER_RING = 5
RING_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

OUT = HERE / "results" / "panelD_shared_H.json"


def run_one_graph(seed):
    G = build_heterogeneous_graph(seed=seed)
    N = G.number_of_nodes()
    z = train_hyperbolic_embedding(G, dim=10, epochs=300, seed=seed)
    d = all_pairs_hyp_dist(z)
    d_np = d.numpy()

    rng = np.random.RandomState(seed + 5000)
    refs = rng.choice(N, size=min(N_REFERENCE_OBSERVERS, N), replace=False)

    n_bins = len(RING_EDGES) - 1
    bin_obs = {b: [] for b in range(n_bins)}     # |mu_H(v) - mu_H(v')|
    bin_bound = {b: [] for b in range(n_bins)}   # |H| * delta * d(v,v')
    bin_dist = {b: [] for b in range(n_bins)}

    for ref in refs:
        ref = int(ref)
        # ONE hidden region, fixed for every comparison below
        ball = set(nx.single_source_shortest_path_length(G, ref, cutoff=OBS_RADIUS))
        H = np.array([u for u in range(N) if u not in ball], dtype=int)
        if len(H) < 10:
            continue
        Hsize = len(H)

        # mu of the SAME H, evaluated from every observer position
        mu_from = np.array([
            float(np.exp(-DELTA * d_np[v, H]).sum()) for v in range(N)
        ])

        dists = d_np[ref].copy()
        dists[ref] = np.inf
        for b in range(n_bins):
            lo, hi = RING_EDGES[b], RING_EDGES[b + 1]
            in_bin = np.where((dists >= lo) & (dists < hi))[0]
            if len(in_bin) == 0:
                continue
            picked = rng.choice(in_bin,
                                size=min(N_PARTNERS_PER_RING, len(in_bin)),
                                replace=False)
            for p in picked:
                gap = abs(mu_from[ref] - mu_from[p])
                dvv = float(d_np[ref, p])
                bin_obs[b].append(gap)
                bin_bound[b].append(Hsize * DELTA * dvv)
                bin_dist[b].append(dvv)

    return bin_obs, bin_bound, bin_dist


def main():
    n_bins = len(RING_EDGES) - 1
    agg_obs = {b: [] for b in range(n_bins)}
    agg_bound = {b: [] for b in range(n_bins)}
    agg_dist = {b: [] for b in range(n_bins)}

    for seed in range(N_SEEDS):
        print(f"[panel D shared-H] seed {seed} ...")
        o, bd, ds = run_one_graph(seed)
        for b in range(n_bins):
            agg_obs[b].extend(o[b])
            agg_bound[b].extend(bd[b])
            agg_dist[b].extend(ds[b])

    points, violations, total = [], 0, 0
    for b in range(n_bins):
        obs = np.array(agg_obs[b], dtype=float)
        bnd = np.array(agg_bound[b], dtype=float)
        dst = np.array(agg_dist[b], dtype=float)
        if len(obs) == 0:
            points.append({"ring_bin_lo": RING_EDGES[b],
                           "ring_bin_hi": RING_EDGES[b + 1],
                           "n_pairs": 0})
            continue
        v = int((obs > bnd + 1e-9).sum())
        violations += v
        total += len(obs)
        slack = np.where(bnd > 0, obs / bnd, np.nan)
        points.append({
            "ring_bin_lo": RING_EDGES[b],
            "ring_bin_hi": RING_EDGES[b + 1],
            "ring_bin_center": (RING_EDGES[b] + RING_EDGES[b + 1]) / 2,
            "mean_observer_distance": float(dst.mean()),
            "mean_descriptor_diff": float(obs.mean()),
            "std_descriptor_diff": float(obs.std()),
            "max_descriptor_diff": float(obs.max()),
            "mean_theoretical_bound": float(bnd.mean()),
            "mean_tightness_ratio": float(np.nanmean(slack)),
            "max_tightness_ratio": float(np.nanmax(slack)),
            "n_bound_violations": v,
            "n_pairs": int(len(obs)),
        })

    out = {
        "experiment": "observer perturbation, SHARED hidden region",
        "theorem": "mu_stability_V: |mu_H(v)-mu_H(v')| <= |H|*delta*d_X(f(v),f(v'))",
        "difference_from_original_panel_d": (
            "panel_d.py gives each observer its own H(v)=V\\B_r(v), so both the "
            "observer and the hidden set vary and the bound does not apply. "
            "Here H is fixed per reference observer and only the observer moves."),
        "delta": DELTA,
        "obs_radius_defining_H": OBS_RADIUS,
        "n_seeds": N_SEEDS,
        "n_reference_observers_per_graph": N_REFERENCE_OBSERVERS,
        "n_partners_per_ring": N_PARTNERS_PER_RING,
        "ring_edges": RING_EDGES,
        "total_pairs": total,
        "total_bound_violations": violations,
        "bound_holds": violations == 0,
        "points": points,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")
    print(f"  pairs: {total}   violations: {violations}   "
          f"bound holds: {violations == 0}")
    if violations:
        print("  [!] A violation means a bug in the experiment or the bound -- "
              "investigate before reporting.")


if __name__ == "__main__":
    main()
