# `analysis/`

Analyses over already-cached artifacts. **Nothing here recomputes centralities
or embeddings** — everything reads `_classical_features_cache.npy` and
`phi_full_meanstd.pt`. All are minutes except `phi_seed_variance.py`.

| Script | Produces | Why |
|---|---|---|
| `reconstruction_plus.py` | rank recovery, radius-augmented $R^2$, community-held-out split | Table 1 is missing all three |
| `cross_dataset_transfer.py` | train-on-A / test-on-B matrix | sharpest distinctiveness test available |
| `phi_seed_variance.py` | measured std for $\Phi$ | `stage_phi` writes `std=zeros` |
| `collect_paper_numbers.py` | `results/paper_numbers.json` | one file with every reported number |

```bash
make analysis && make collect
```

`collect_paper_numbers.py` also prints a **MISSING** list — use it as the
pre-submission checklist.

## Note on `_common.py`

Loaders import `run_everything.DATASET_CONFIG` and `get_graph` directly rather
than reimplementing dataset loading, so node index alignment between $\Phi$,
the feature matrix, and the graph cannot drift. If a row count mismatches, the
error names the exact stale file.
