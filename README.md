# Observer Geometry

Code for *Observer Geometry: Describing Graph Structure Beyond the Observation
Horizon*.

Observer Geometry characterizes a node by the metric structure that remains
**unobserved** beyond a variable observation horizon, rather than by the
topology already visible from it. The observer descriptor is

$$\Phi(v) \;=\; \mathbb{E}_{I \sim P}\big[\,\mathcal{J}(v, \mathcal{H}_I(v))\,\big],
\qquad
\mathcal{J}(v, H) \;=\; \sum_{u \in H} e^{-\delta\, d_X(\varphi(v), \varphi(u))}.$$

The framework is metric-space agnostic; the instantiation here uses hyperbolic
graph embeddings. Core properties are mechanized in Lean 4.

> **Anonymized artifact for peer review.** No author, institution, or contact
> information is included.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/selftest.py   # verify standalone, seconds
bash scripts/triage.sh        # 20 min: inventory, K* check, timing pilot
make help
```

`triage.sh` is not optional. It catches the failure mode that costs the most
time: a stale `phi_*.pt` computed at a different observation budget silently
corrupting every correlation in the ablation grid.

---

## Layout

```
cora/  citeseer/  go_dag/  string_ppi/  obgn_arxiv/
    hvgae_embedding/
        embedding.pt              Lorentz coordinates [N, 17]
        chosen_curvature.txt      selected curvature
        phi_full_meanstd.pt       {mean, std, seeds, K_star}
    _classical_features_cache.npy [N, 7] centralities  <- expensive, cached
    *.json                        per-stage results

embedding_ablation/               encoder x geometry, curvature, seed studies
synthetic/                        stability panels + dose-response benchmark
analysis/                         new analyses over cached artifacts
results/                          pooled JSON, incl. paper_numbers.json
lean/                             Lean 4 formalization (see lean/README.md)
scripts/                          triage
```

Every script writes JSON. No script writes figures — build those from
`results/paper_numbers.json`, which collects every reported number in one file.

---

## Running order

| Command | Wall clock | What it produces |
|---|---|---|
| `make triage` | 20 min | inventory, K\* check, timing pilot |
| `make overnight` | 1–6 h | dose-response with hyperbolic re-embedding |
| `make morning` | ~2 h (GPU) | grid / seed / curvature ablations |
| `make analysis` | minutes | reconstruction+, transfer, stability panels |
| `make collect` | seconds | `results/paper_numbers.json` |

The only long job is `overnight`. Launch it, then do everything else while it
runs. `make triage` prints an estimate before you commit.

Two independent resources: the dose-response uses 7 CPU workers, the ablations
use the GPU. They can overlap.

---

## What each new script fixes

These were written to close specific gaps between what the paper claims and
what the code did. Each header documents the problem it addresses.

**`synthetic/run_dose_response.py`** — the old version used
`nx.spring_layout(dim=2)` with Euclidean distance, so Figure 1 contained no
hyperbolic geometry. It also froze coordinates, which makes edge-only
interventions *mathematically unable* to change $\mu$ — so "$\Phi$ tracks
extent, not connectivity" was forced by the protocol rather than discovered.
Now: Poincaré ball, and warm-started re-embedding after each intervention.
Run both modes (`make dose-response` and `make dose-response-frozen`) and
report the comparison.

**`embedding_ablation/run_grid_ablation.py`** — SAGE/Lorentz was aliased to the
main HVGAE embedding at $\rho = 1.0$ by definition, but that embedding used
curvature search, early stopping, and dropout while the other eight cells did
not. Every $\rho$ conflated encoder/geometry with training protocol. The anchor
is now trained in-grid. A random-embedding null is included, because without it
there is no scale on which to read $\rho = 0.81$ as high or low.

**`embedding_ablation/grid_pairwise.py`** — the paper quotes a pairwise matrix;
the old script computed nine-vs-one. This builds the real one and compares
every band against the seed-to-seed noise floor.

**`synthetic/panel_d_shared_h.py`** — the observer-perturbation bound is stated
for a *shared* hidden region, but the original panel gave each observer its own
$H(v)$, so both terms varied and the bound never applied. This fixes $H$ and
moves only the observer.

**`analysis/reconstruction_plus.py`** — adds rank recovery (the claim is about
rankings, $R^2$ is about values), hyperbolic radius as an 8th feature (it
correlates $-0.606$ on Cora, better than any classical descriptor, and was
absent), and a community-held-out split (a random node split leaks, because
centralities are spatially smooth).

**`analysis/cross_dataset_transfer.py`** — the sharpest distinctiveness test
available. If a model trained on one graph transfers to another, $\Phi$ is a
fixed nonlinear functional of classical descriptors. If transfer collapses, it
is a *different* function per graph.

**`synthetic/aggregate.py`** — reported $\pm 1.96\,\sigma/\sqrt{n}$ with $n$ =
record count, but realizations share observers, observers share graphs, graphs
share families. Now emits both record-level and instance-clustered intervals.

---

## Interpreting results

**Read every correlation against the noise floor.** `seed_robustness_results.json`
gives seed-to-seed agreement. Anything below it is a *real effect*, not
agreement — `grid_pairwise.json` flags this automatically.

**Use `cluster_level`, not `record_level`,** for any claim about a graph family.

**$\Phi$ is reported normalized** by total visible mass $\mu(V)$ in both
simulators. `reembed.py` emits all three conventions (`mu`, `Phi_normalized`,
`mu_per_node`) so the paper can settle on one.

---

## Caveats

- Message passing occurs in the tangent space with a single exponential map at
  the output, not fully manifold-valued aggregation.
- Curvature is fixed by grid search, not learned: in `geoopt` 0.5.1 the Lorentz
  exponential map does not propagate gradients into the curvature parameter.
- Hub nodes are capped at 256 nearest neighbours on `ogbn-arxiv`, GO DAG, and
  STRING PPI.
- GO DAG and STRING PPI have no native node features; inputs are
  $\log(1+\text{degree})$ plus a fixed random projection.
- Closeness/harmonic use a 1000-pivot landmark estimator, and betweenness is
  sampled at $k=1000$, on the three large graphs.
- Embedding-robustness studies are Cora only.

---

## License

MIT. See `LICENSE`.
