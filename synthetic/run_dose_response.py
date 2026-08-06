"""
run_dose_response.py -- PATCHED (hyperbolic + re-embedding + reduced grid)
==========================================================================
Generates the data behind Figure 1.

WHAT CHANGED AND WHY
--------------------
1. HYPERBOLIC GEOMETRY. The old version used nx.spring_layout(dim=2) with
   Euclidean distance, so Figure 1 contained no hyperbolic geometry at all
   despite the paper being about a hyperbolic instantiation. Now: Poincare
   ball, dim=10, same stress objective as the stability benchmark.

2. RE-EMBEDDING AFTER INTERVENTION. With frozen coordinates, edge-only
   interventions cannot change mu -- so "Phi tracks extent, not connectivity"
   was forced by the protocol rather than discovered. Warm-started refinement
   (50 epochs from the base embedding) lets connectivity propagate into the
   metric. Run with --frozen to reproduce the old behaviour for comparison;
   reporting BOTH is the strongest version of this experiment.

3. REDUCED GRID. 4.18M records bought false precision, not statistical power:
   100 realizations share an observer, 20 observers share a graph AND its
   embedding, 20 graphs share a family. Effective n for a family-level claim
   was always ~20, not ~800k. New grid is ~64k configs and gives the same
   information with honest error bars.

        graphs/family  20 -> 10        observers/graph  20 -> 8
        levels/axis   5-6 -> 4         realizations    100 -> 10

4. GRAPH-INSTANCE IDs on every record, so aggregate.py can cluster standard
   errors by instance instead of treating correlated records as independent.

Runtime: ~40-90 min on 7 workers (dominated by APSP + centralities per
realization, not by the embedding refinement). Measure first:
    python synthetic/run_dose_response.py --pilot

Emits: synthetic/results/raw_reembed/<family>/graph_<seed>.json
Resumable: completed graph files are skipped.

Usage:
    python synthetic/run_dose_response.py --pilot
    python synthetic/run_dose_response.py
    python synthetic/run_dose_response.py --frozen --out results/raw_frozen
    python synthetic/run_dose_response.py --workers 7 --graphs-per-family 10
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np
import networkx as nx

from graph_families import GRAPH_FAMILIES, generate_graph, graph_metadata
from observer_utils import (freeze_ball, compute_classical_metrics,
                            compute_hidden_descriptors, compute_observer_metadata)
from interventions_dose_response import INTERVENTION_AXES
from reembed import embed_base, reembed_warm, phi_variants_hyperbolic

# ----------------------------- CONFIG -----------------------------
GRAPHS_PER_FAMILY = 10
N_OBSERVERS_PER_GRAPH = 8
N_REALIZATIONS = 10
MAX_LEVELS_PER_AXIS = 4
OBS_RADIUS = 2
SEED_BASE = 0
N_WORKERS = 7
BETWEENNESS_K = 50
BASE_EPOCHS = 300
REEMBED_EPOCHS = 50          # 0 == frozen embedding (old protocol)
OUTPUT_DIR = HERE / "results" / "raw_reembed"
# -------------------------------------------------------------------


def arg(flag, default, cast=str):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def subsample_levels(levels, k=MAX_LEVELS_PER_AXIS):
    """Keep level 0 (control) plus a spread of the rest, always including the max."""
    if len(levels) <= k:
        return list(levels)
    rest = levels[1:]
    idx = np.unique(np.linspace(0, len(rest) - 1, k - 1).astype(int))
    return [levels[0]] + [rest[i] for i in idx]


def delta_dict(before, after):
    out = {}
    for key in before:
        a, b = before.get(key), after.get(key)
        if a is None or b is None:
            out[key] = None
        else:
            try:
                out[key] = float(b) - float(a)
            except (TypeError, ValueError):
                out[key] = None
    return out


def combined_metrics(G, z, index, v, hidden_nodes):
    m = compute_classical_metrics(G, v, betweenness_k=BETWEENNESS_K)
    pv = phi_variants_hyperbolic(z, index, v, hidden_nodes)
    if pv:
        m.update(pv)
    return m


def process_one_graph(job):
    family, seed, out_path, reembed_epochs = job
    out_path = Path(out_path)

    if out_path.exists():
        return f"SKIP (exists): {out_path.name}"

    t_start = time.time()
    G = generate_graph(family, seed)
    if G.number_of_nodes() < 30:
        return f"SKIP (too small): {family}/{seed}"

    g_meta = graph_metadata(G)

    # one cold embedding per graph instance; every intervention warm-starts from it
    z_base, nodes_base, index_base = embed_base(G, seed=seed, epochs=BASE_EPOCHS)

    rng_obs = np.random.RandomState(seed + 10_000)
    all_nodes = list(G.nodes())
    n_obs = min(N_OBSERVERS_PER_GRAPH, len(all_nodes))
    observers = rng_obs.choice(all_nodes, size=n_obs, replace=False)

    graph_instance_id = f"{family}::{seed}"
    records = []

    for v in observers:
        v = int(v)
        ball_nodes, ball_edges = freeze_ball(G, v, OBS_RADIUS)
        hidden_base = set(G.nodes()) - ball_nodes
        if len(hidden_base) < 10:
            continue

        obs_meta = compute_observer_metadata(G, v, ball_nodes, ball_edges,
                                             hidden_base, OBS_RADIUS)
        metrics_before = combined_metrics(G, z_base, index_base, v, hidden_base)
        hidden_before = compute_hidden_descriptors(G, hidden_base)

        rng = np.random.RandomState(int(abs(hash((seed, v))) % (2 ** 31)))

        for axis_name, cfg in INTERVENTION_AXES.items():
            for level in subsample_levels(cfg["levels"]):
                for realization in range(N_REALIZATIONS):
                    iseed = int(rng.randint(0, 2 ** 31))
                    local_rng = np.random.RandomState(iseed)

                    G_new, hidden_after, interv_meta = cfg["fn"](
                        G, ball_nodes, hidden_base, level, local_rng)

                    z_new, nodes_new, index_new = reembed_warm(
                        G_new, z_base, index_base, local_rng,
                        epochs=reembed_epochs, seed=iseed % (2 ** 31))

                    metrics_after = combined_metrics(
                        G_new, z_new, index_new, v, hidden_after)
                    hidden_after_desc = compute_hidden_descriptors(G_new, hidden_after)

                    records.append({
                        "family": family,
                        "graph_seed": seed,
                        "graph_instance_id": graph_instance_id,
                        "observer_id": f"{graph_instance_id}::{v}",
                        "graph_meta": g_meta,
                        "observer_meta": obs_meta,
                        "axis": axis_name,
                        "level": level,
                        "realization": realization,
                        "intervention_seed": iseed,
                        "intervention_meta": interv_meta,
                        "reembed_epochs": reembed_epochs,
                        "hidden_before": hidden_before,
                        "hidden_after": hidden_after_desc,
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                        "delta_metrics": delta_dict(metrics_before, metrics_after),
                    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)

    return (f"DONE {out_path.parent.name}/{out_path.name}: "
            f"{len(records)} records, {time.time()-t_start:.0f}s")


def build_jobs(out_dir, reembed_epochs, graphs_per_family):
    jobs, offset = [], 0
    for family in GRAPH_FAMILIES:
        for gi in range(graphs_per_family):
            seed = SEED_BASE + offset + gi
            jobs.append((family, seed,
                         str(Path(out_dir) / family / f"graph_{seed}.json"),
                         reembed_epochs))
        offset += graphs_per_family
    return jobs


def pilot():
    """Time one graph instance so the full run can be sized before launching."""
    print("PILOT -- timing one graph instance\n")
    t = time.time()
    G = generate_graph("sbm", 0)
    print(f"  generate:            {time.time()-t:.2f}s  (N={G.number_of_nodes()})")

    t = time.time()
    z, nodes, index = embed_base(G, seed=0, epochs=BASE_EPOCHS)
    t_base = time.time() - t
    print(f"  base embed (300ep):  {t_base:.2f}s")

    rng = np.random.RandomState(0)
    t = time.time()
    reembed_warm(G, z, index, rng, epochs=REEMBED_EPOCHS)
    t_warm = time.time() - t
    print(f"  warm re-embed (50ep):{t_warm:.2f}s")

    v = int(list(G.nodes())[0])
    ball, _ = freeze_ball(G, v, OBS_RADIUS)
    hidden = set(G.nodes()) - ball
    t = time.time()
    combined_metrics(G, z, index, v, hidden)
    t_metrics = time.time() - t
    print(f"  metrics per config:  {t_metrics:.2f}s")

    per_cfg = t_warm + t_metrics
    n_levels = sum(len(subsample_levels(c["levels"])) for c in INTERVENTION_AXES.values())
    per_graph = t_base + N_OBSERVERS_PER_GRAPH * n_levels * N_REALIZATIONS * per_cfg
    n_graphs = GRAPHS_PER_FAMILY * len(GRAPH_FAMILIES)
    total_h = per_graph * n_graphs / 3600

    print(f"\n  configs/graph: {N_OBSERVERS_PER_GRAPH} obs x {n_levels} levels "
          f"x {N_REALIZATIONS} reps = {N_OBSERVERS_PER_GRAPH*n_levels*N_REALIZATIONS}")
    print(f"  est per graph: {per_graph/60:.1f} min")
    print(f"  est total:     {total_h:.1f} CPU-h -> "
          f"{total_h/N_WORKERS:.1f} h on {N_WORKERS} workers")
    if total_h / N_WORKERS > 8:
        print("\n  TOO SLOW. Cut --graphs-per-family to 5 or N_REALIZATIONS to 5.")


def main():
    if "--pilot" in sys.argv:
        pilot()
        return

    reembed_epochs = 0 if "--frozen" in sys.argv else arg("--reembed-epochs",
                                                          REEMBED_EPOCHS, int)
    out_dir = Path(arg("--out", str(OUTPUT_DIR)))
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    workers = arg("--workers", N_WORKERS, int)
    gpf = arg("--graphs-per-family", GRAPHS_PER_FAMILY, int)

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(out_dir, reembed_epochs, gpf)

    mode = "FROZEN (old protocol)" if reembed_epochs == 0 else \
           f"RE-EMBEDDING ({reembed_epochs} warm epochs)"
    print(f"Mode:    {mode}")
    print(f"Output:  {out_dir}")
    print(f"Jobs:    {len(jobs)} graphs | workers: {workers}")
    print(f"Grid:    {gpf}/family x {N_OBSERVERS_PER_GRAPH} obs x "
          f"{N_REALIZATIONS} reps\n")

    with open(out_dir / "_run_config.json", "w") as f:
        json.dump({
            "reembed_epochs": reembed_epochs, "base_epochs": BASE_EPOCHS,
            "graphs_per_family": gpf, "observers_per_graph": N_OBSERVERS_PER_GRAPH,
            "realizations": N_REALIZATIONS, "max_levels_per_axis": MAX_LEVELS_PER_AXIS,
            "obs_radius": OBS_RADIUS, "betweenness_k": BETWEENNESS_K,
            "embedding": "poincare_ball_dim10_stress",
            "families": list(GRAPH_FAMILIES),
        }, f, indent=2)

    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one_graph, jobs), 1):
            print(f"[{i}/{len(jobs)}] {res}")

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min")
    print(f"Next: python synthetic/aggregate.py --raw {out_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
