# Reproducibility

Every number in the paper, the command that produces it, and the artifact it
lands in.

## Claim → command → artifact

| Paper location | Command | Artifact |
|---|---|---|
| Thm. 1 empirical tightness (4 panels) | `make panels` | `synthetic/results/panel{A,B,C,D}_*.json` |
| Thm. 1(e), shared hidden region | `python synthetic/panel_d_shared_h.py` | `synthetic/results/panelD_shared_H.json` |
| Figure 1, intervention benchmark | `make overnight` | `synthetic/results/paper_summary.json` |
| Frozen-coordinate control | `make dose-response-frozen` | `synthetic/results/summary_frozen.json` |
| Prop. 3 (exact descriptor), validation | `python analysis/exact_phi_all.py` | `results/exact_validation.json` |
| Figure 2, rank and value convergence | `python run_everything.py --stages k_sweep`; `make fig2` | `<ds>/k_sweep_full_<ds>.json`, `figures/k_sweep_convergence.pdf` |
| Budget laws, limit and bound | `make budget` | `results/budget_value.json` |
| Hub cap has no effect | `make capcheck` | `results/cap_inertness.json` |
| Table 2, Cora: encoder × geometry grid | `make grid` | `embedding_ablation/grid_pairwise.json` |
| Table 2, Cora: seed floor | `make seeds` | `embedding_ablation/seed_robustness_results.json` |
| Table 2, Cora: curvature | `make curvature` | `curvature_correlation.json` |
| Table 2, Citeseer and GO DAG | `make robustness-ext` | `embedding_ablation/robust_{citeseer,go_dag}.json` |
| Radius correlation | `python run_everything.py --stages radius` | `<ds>/radius_correlation.json` |
| Table 3, reconstruction from centralities | `make recon` | `<ds>/reconstruction_plus.json` |
| Closer measures (coverage, accessibility, kernel mass) | `make closer` | `<ds>/closer_baselines.json` |
| Sensitivity to δ and ε | `make exact-sweeps sensitivity` | `results/sensitivity.json` |
| Finite-graph crossover vs. tree threshold | `make growth` | `results/growth.json` |
| Cross-dataset transfer and held-out comparator | `make transfer transfer-heldout` | `results/cross_dataset_transfer.json`, `results/transfer_heldout.json` |
| Domain validation (GO, STRING) | `python run_everything.py --stages hierarchy biology` | `go_dag/hierarchy_validation.json`, `string_ppi/biology_validation.json` |
| Monte Carlo seed variance | `make phivar` | `<ds>/phi_seed_variance.json` |
| Everything, collected | `make collect` | `results/paper_numbers.json` |
| Lean development | `make lean` | 19 modules, no `sorry`, no project axioms |

## Fixed hyperparameters

| | |
|---|---|
| decay rate $\delta$ | 1.0 |
| termination threshold $\varepsilon$ | 1/50 |
| walk step cap | 2000 |
| latent dim | 16 (17 ambient Lorentz) |
| hidden dim | 64 |
| optimizer | Adam, lr $10^{-3}$, wd $5\times10^{-4}$ |
| epochs | 300, early-stop patience 30 |
| Monte Carlo realizations per source | 3 (the exact evaluator needs none) |
| curvature parameter grid $c$ (sectional curvature $-1/c$) | $\{0.25, 0.5, 1, 2, 4, 8\}$ |
| observation radius (synthetic) | 2 |

Note that with $\varepsilon = 1/50$ and mean degree ≈ 4, a trajectory
terminates after roughly $\log(0.02)/\log(1/3) \approx 3.6$ steps. The
observation region is therefore a shallow walk-weighted neighbourhood, closer
to a soft 3–4-hop ball than to a global exploration. This is why $K^\*$
plateaus early and why $\Phi$ is cheap.

## Hardware and expected timings

Developed on an RTX 4050 Laptop GPU (6 GB) with 16 CPU cores.

| Stage | Cold | Warm |
|---|---|---|
| dataset downloads | 30–90 min | cached |
| curvature search ×5 | 1.5–3 h | cached |
| classical centralities ×5 | 2–6 h | cached in `_classical_features_cache.npy` |
| $\Phi$ all nodes ×5 | 5–15 min | 5–15 min |
| K-sweep ×5 | 20–60 min | — |
| grid ablation | 40–70 min | — |
| dose-response (reduced grid) | 1–6 h | resumable |
| exact descriptor ×5 (incl. total kernel mass) | ~15 min CPU | — |
| Lean build | 3–8 min | 6 s |

**Full cold regeneration: 6–10 h wall clock with parallelism.**
**Warm (embeddings + centralities cached): 2–3.5 h.**

The two steps worth timing on new hardware first: centralities on
`ogbn-arxiv` (2000 BFS passes over 169k nodes in pure-Python networkx) and the
STRING permutation test. `scripts/triage.sh` times the synthetic pilot; time
one BFS with

```python
import time, networkx as nx
t = time.time(); nx.single_source_shortest_path_length(G, 0); print(time.time() - t)
```

and multiply by 2000 for the floor.

## Determinism

Embedding training, simulator sampling, and interventions are all seeded.
Exact bitwise reproduction is **not** expected across machines: the vectorized
simulator advances $K$ walks synchronously rather than sequentially. The walks
are i.i.d. and hence exchangeable, so the distribution of the observed union is
unchanged, but the specific random sequence differs. Validated at
$\rho = 0.9963$ (Cora) and $\rho = 0.9977$ (ogbn-arxiv) against the sequential
reference, at ~380× speedup. Compare distributions and correlations, not exact
values.

## Known limitations

Collected in one place; each is also documented in the relevant script header.

1. Tangent-space message passing with a single output exponential map, not
   fully manifold-valued aggregation.
2. Curvature fixed by grid search — `geoopt` 0.5.1 does not backpropagate into
   the curvature parameter.
3. Synthetic node features for GO DAG and STRING PPI: $\log(1+\deg)$ and 63 i.i.d.
   $\mathcal N(0,1)$ features (seed 0).
4. Synchronous walk parallelism: statistically but not bitwise equivalent. The exact
   evaluator (`exact_phi.py`) has no Monte Carlo error.
5. Hub degree capped at 256 in the vectorized simulator; no effect at $\varepsilon=1/50$
   (`results/cap_inertness.json`).
6. Landmark-approximated closeness/harmonic (1000 pivots) and sampled
   betweenness ($k = 1000$) on the large graphs; Katz omitted where power
   iteration fails to converge.
7. Embedding-robustness studies: full grid on Cora, Citeseer and GO DAG; curvature study on Cora only.
8. The regular-tree closed form is established at the root; no closed form for
   observers at general depth.
