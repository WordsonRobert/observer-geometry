# `go_dag/`

Dataset workspace: cached embedding, descriptor and per-stage results for this graph.

## Expected contents

```
data/                            raw download cache (gitignored)
hvgae_embedding/
    embedding.pt                 Lorentz coordinates [N, 17]
    model_state.pt               trained encoder
    chosen_curvature.txt         selected curvature
    curvature_search_results.json all six candidates
    phi_full_meanstd.pt          {mean, std, seeds, K_star}
    phi_seed_variance.pt         sidecar from analysis/phi_seed_variance.py
_classical_features_cache.npy    [N, 7] centralities  <- EXPENSIVE, never delete casually
phi.csv                          node_id, phi
*.json                           per-stage results
exact/                           exact descriptor artifacts (git-ignored; analysis/exact_phi_all.py)
closer_baselines.json            comparison with coverage/accessibility/kernel-mass measures
```

## Graph

Biological-process terms of `go-basic.obo` (release 2026-06-15), primary identifiers only
(24,129 terms), `is_a` edges, symmetrized for embedding (40,863 edges). goatools also keys
every term by its alternate identifiers; those keys are skipped so they do not become
separate vertices.

## Regenerating

```bash
cd go_dag && python go_dag_train_embedding.py && cd ..     # embedding + curvature search
python go_dag/go_dag_ksweep_phi.py                      # budget sweep -> K*, Monte Carlo Phi
python run_everything.py --only go_dag                  # all pipeline stages
python analysis/reconstruction_plus.py --datasets go_dag
python analysis/exact_phi_all.py --datasets go_dag       # exact descriptor -> exact/
python analysis/closer_baselines.py --datasets go_dag
```

## Warning

`_classical_features_cache.npy` holds the centralities and takes hours to
rebuild on the large graphs. Deleting it forces a full recompute. If its row
count ever mismatches `phi_full_meanstd.pt`, the cache is stale and *must* be
deleted — `analysis/reconstruction_plus.py` detects this and says so.
