# `obgn_arxiv/`

Dataset workspace. **Nothing here is committed** — everything is regenerated.

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
```

## Regenerating

```bash
cd obgn_arxiv && python obgn_arxiv_train_embedding.py && cd ..     # embedding + curvature search
python run_everything.py --datasets obgn_arxiv               # all stages
python analysis/reconstruction_plus.py --datasets obgn_arxiv # new analyses
```

## Warning

`_classical_features_cache.npy` holds the centralities and takes hours to
rebuild on the large graphs. Deleting it forces a full recompute. If its row
count ever mismatches `phi_full_meanstd.pt`, the cache is stale and *must* be
deleted — `analysis/reconstruction_plus.py` detects this and says so.
