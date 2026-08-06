# `embedding_ablation/`

Does $\Phi$ measure graph geometry, or one embedding algorithm? Four factors,
isolated one at a time, all on Cora.

| Study | Script | Question |
|---|---|---|
| encoder × geometry | `run_grid_ablation.py` | architecture vs. curvature |
| pairwise + null | `grid_pairwise.py` | full matrix, read against the noise floor |
| curvature | `run_curvature_ablation.py` | sensitivity to the selected $c$ |
| optimization seed | `run_seed_robustness.py` | **the noise floor** |
| variants | `run_ablation.py` + `compare_results.py` | shallow / HGCN-lite / Euclidean / random |

## Read everything against the seed floor

`run_seed_robustness.py` gives the mean off-diagonal $\rho$ across seeds — the
resolution limit of every other study here. A correlation **below** it is a
real effect, not agreement. `grid_pairwise.py` flags this automatically; do not
describe a variant as "insensitive" when its $\rho$ sits below the floor.

## Run order

```bash
make seeds     # noise floor FIRST -- nothing else is interpretable without it
make grid      # 9 cells + random null, then the pairwise matrix
make curvature
```

## What changed in `run_grid_ablation.py`

The anchor (SAGE/Lorentz) is now trained **in-grid** under the same protocol as
every other cell. Previously it was aliased to the main HVGAE embedding at
$\rho = 1.0$ — but that embedding used curvature search, early stopping, and
dropout, while the eight other cells used none of those. Every reported $\rho$
therefore conflated encoder/geometry with training protocol, which biases
exactly the sensitivity ordering the paper reports.

A random-embedding null is also included. Without it there is no scale on which
to read $\rho = 0.81$ as high or low.
