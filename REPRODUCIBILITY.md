# Reproducibility

Every number in the paper, the command that produces it, and the artifact it
lands in.

## Claim → command → artifact

| Paper location | Command | Artifact |
|---|---|---|
| §5.1 stability bounds (4 panels) | `make panels` | `synthetic/results/panel{A,B,C,D}_*.json` |
| §5.1 observer bound, shared $H$ | `python synthetic/panel_d_shared_h.py` | `synthetic/results/panelD_shared_H.json` |
| §5.3 intervention benchmark | `make overnight` | `synthetic/results/paper_summary.json` |
| §5.3 frozen-vs-re-embedded | `make dose-response-frozen` | `synthetic/results/summary_frozen.json` |
| §7.1 encoder × geometry grid | `make grid` | `embedding_ablation/grid_pairwise.json` |
| §7.1 seed floor $\rho=0.959$ | `make seeds` | `embedding_ablation/seed_robustness_results.json` |
| §7.1 curvature $\rho=0.85$–$0.93$ | `make curvature` | `embedding_ablation/curvature_correlation.json` |
| §7.2 budget plateau $K^\*$ | `python run_everything.py --stages k_sweep` | `<ds>/k_sweep_full.json` |
| §7.2 radius correlation | `python run_everything.py --stages radius` | `<ds>/radius_correlation.json` |
| §7.3 Table 1 | `make recon` | `<ds>/reconstruction_plus.json` |
| §7.3 cross-dataset transfer | `make transfer` | `results/cross_dataset_transfer.json` |
| $\Phi$ seed variance | `make phivar` | `<ds>/phi_seed_variance.json` |
| Everything, collected | `make collect` | `results/paper_numbers.json` |
| Appendix A–B (Lean) | `cd lean && lake build` | zero `sorry`, zero project axioms |

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
| repeats per source | 3 |
| curvature grid | $\{0.25, 0.5, 1, 2, 4, 8\}$ |
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
| Lean build | 3–8 min | 6 s |

**Full cold regeneration: 6–10 h wall clock with parallelism.**
**Warm (embeddings + centralities cached): 2–3.5 h.**

The two lines to verify on your own hardware before committing: centralities on
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
3. Synthetic node features ($\log$-degree + fixed random projection) for GO DAG
   and STRING PPI.
4. Synchronous walk parallelism: statistically but not bitwise equivalent.
5. Hub degree capped at 256 on the three large graphs.
6. Landmark-approximated closeness/harmonic (1000 pivots) and sampled
   betweenness ($k = 1000$) on the large graphs; Katz omitted where power
   iteration fails to converge.
7. Embedding-robustness studies are Cora only.
8. The regular-tree closed form is established at the root; no closed form for
   observers at general depth.
