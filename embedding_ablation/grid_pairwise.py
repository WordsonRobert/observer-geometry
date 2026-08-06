"""
grid_pairwise.py
================
Builds the FULL pairwise Spearman matrix across every saved Phi variant, plus
the three cluster summaries the paper actually quotes.

Section 7.1 currently claims "pairwise Spearman correlation between the nine
resulting descriptor rankings", but run_grid_ablation only ever computed
nine-vs-one-anchor. The ranges quoted in the text ("GCN internally consistent
0.90-0.95", "SAGE and GAT agree 0.86-0.96") require genuine pairwise values.
This produces them.

It also computes the three numbers the section should be organised around:

    within-encoder, across-geometry   -- how much does geometry matter?
    within-geometry, across-encoder   -- how much does the encoder matter?
    random null                       -- what does "no signal" look like?

and compares all of them against the seed-to-seed noise floor from
run_seed_robustness.py. Anything inside the noise band is indistinguishable,
not "in agreement" -- that distinction is what Section 7.1 currently gets
backwards.

Reads whatever phi_grid_*.pt / phi_hvgae_seed_*.pt files exist. Pure numpy;
runs in seconds.

Emits: embedding_ablation/grid_pairwise.json

Usage:
    python embedding_ablation/grid_pairwise.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EMB_DIR = HERE / "embeddings"

ENCODERS = ["SAGE", "GCN", "GAT"]
MANIFOLDS = ["lorentz", "poincare", "euclidean"]

LABEL = {
    "lorentz": "Lorentz",
    "poincare": "Poincare",
    "euclidean": "Euclidean",
}


def load_vec(path):
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict):
        d = d.get("mean", d)
    return np.asarray(d).astype(np.float64).ravel()


def main():
    vecs, names = {}, []
    for enc in ENCODERS:
        for man in MANIFOLDS:
            nm = f"{enc}_{man}"
            p = EMB_DIR / f"phi_grid_{nm}.pt"
            if p.exists():
                vecs[nm] = load_vec(p)
                names.append(nm)
            else:
                print(f"  [missing] {p.name}")

    null_path = EMB_DIR / "phi_grid_random.pt"
    if null_path.exists():
        vecs["random"] = load_vec(null_path)
        names.append("random")
    else:
        print("  [missing] phi_grid_random.pt -- run run_grid_ablation.py first")

    if len(names) < 2:
        raise SystemExit("Fewer than 2 variants found. Run run_grid_ablation.py.")

    lengths = {len(v) for v in vecs.values()}
    if len(lengths) > 1:
        raise SystemExit(f"Variants have differing lengths {lengths} -- these are "
                         f"not all Cora. Delete stale phi_grid_*.pt and rerun.")

    n = len(names)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(spearmanr(vecs[names[i]], vecs[names[j]])[0])
            M[i, j] = M[j, i] = r

    def band(pairs):
        if not pairs:
            return None
        a = np.array(pairs, dtype=float)
        return {"min": float(a.min()), "max": float(a.max()),
                "mean": float(a.mean()), "n": len(a)}

    idx = {nm: k for k, nm in enumerate(names)}
    trained = [nm for nm in names if nm != "random"]

    # geometry effect: same encoder, different manifold
    within_encoder = {}
    for enc in ENCODERS:
        got = [f"{enc}_{m}" for m in MANIFOLDS if f"{enc}_{m}" in idx]
        pairs = [M[idx[a], idx[b]]
                 for i, a in enumerate(got) for b in got[i + 1:]]
        if pairs:
            within_encoder[enc] = band(pairs)

    # encoder effect: same manifold, different encoder
    within_manifold = {}
    for man in MANIFOLDS:
        got = [f"{e}_{man}" for e in ENCODERS if f"{e}_{man}" in idx]
        pairs = [M[idx[a], idx[b]]
                 for i, a in enumerate(got) for b in got[i + 1:]]
        if pairs:
            within_manifold[LABEL[man]] = band(pairs)

    # SAGE/GAT vs GCN -- the cluster split the paper reports
    sg = [nm for nm in trained if nm.startswith(("SAGE", "GAT"))]
    gc = [nm for nm in trained if nm.startswith("GCN")]
    sage_gat_internal = band([M[idx[a], idx[b]]
                              for i, a in enumerate(sg) for b in sg[i + 1:]])
    gcn_internal = band([M[idx[a], idx[b]]
                         for i, a in enumerate(gc) for b in gc[i + 1:]])
    across_clusters = band([M[idx[a], idx[b]] for a in sg for b in gc])

    null_band = None
    if "random" in idx:
        null_band = band([M[idx["random"], idx[nm]] for nm in trained])

    # --- seed-to-seed noise floor, if available -----------------------------
    noise_floor = None
    seed_files = sorted(glob.glob(str(EMB_DIR / "phi_hvgae_seed_*.pt")))
    if len(seed_files) >= 2:
        sv = [load_vec(p) for p in seed_files]
        if len({len(v) for v in sv}) == 1:
            pr = [float(spearmanr(sv[i], sv[j])[0])
                  for i in range(len(sv)) for j in range(i + 1, len(sv))]
            noise_floor = band(pr)
    if noise_floor is None:
        srj = HERE / "seed_robustness_results.json"
        if srj.exists():
            with open(srj) as f:
                d = json.load(f)
            if "mean_off_diagonal" in d:
                noise_floor = {"mean": d["mean_off_diagonal"],
                               "min": d["mean_off_diagonal"] - d.get("std_off_diagonal", 0),
                               "max": d["mean_off_diagonal"] + d.get("std_off_diagonal", 0),
                               "n": None,
                               "source": "seed_robustness_results.json"}

    out = {
        "variants": names,
        "matrix": M.tolist(),
        "geometry_effect_within_encoder": within_encoder,
        "encoder_effect_within_geometry": within_manifold,
        "sage_gat_cluster_internal": sage_gat_internal,
        "gcn_cluster_internal": gcn_internal,
        "across_clusters_sage_gat_vs_gcn": across_clusters,
        "random_null_vs_trained": null_band,
        "seed_noise_floor": noise_floor,
        "reading_note": (
            "Compare every band against seed_noise_floor. A correlation BELOW "
            "the noise floor is a real effect, not agreement. Reporting a "
            "variant as 'insensitive' when its rho sits below the floor "
            "contradicts the seed measurement."),
    }

    with open(HERE / "grid_pairwise.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary ---------------------------------------------------
    w = max(len(x) for x in names) + 2
    print("\n" + " " * w + "".join(f"{x[:9]:>11}" for x in names))
    for i, a in enumerate(names):
        print(f"{a:<{w}}" + "".join(f"{M[i, j]:>11.3f}" for j in range(n)))

    def show(label, b):
        if b:
            print(f"  {label:<44} {b['min']:.3f} - {b['max']:.3f}  "
                  f"(mean {b['mean']:.3f}, n={b['n']})")

    print("\n  --- effect sizes ---")
    for enc, b in within_encoder.items():
        show(f"geometry effect, {enc} fixed", b)
    for man, b in within_manifold.items():
        show(f"encoder effect, {man} fixed", b)
    show("SAGE/GAT internal", sage_gat_internal)
    show("GCN internal", gcn_internal)
    show("SAGE/GAT vs GCN", across_clusters)
    show("random null vs trained", null_band)
    if noise_floor:
        print(f"\n  seed noise floor: mean {noise_floor['mean']:.4f}")
        print("  -> any band below this is a REAL effect, not agreement.")
    else:
        print("\n  [no seed noise floor found] run run_seed_robustness.py -- "
              "without it none of the above can be called large or small.")

    print(f"\n  wrote {HERE/'grid_pairwise.json'}")


if __name__ == "__main__":
    main()
