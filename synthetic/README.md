# `synthetic/`

Two independent experimental programs.

## 1. Stability validation (`panel_a..d`, `panel_d_shared_h`)

Validates the four bounds of §5.1 by applying exactly one controlled
perturbation at a time on a heterogeneous benchmark graph — SBM communities +
hub + long path + shortcuts + cycles + edge removal, embedded into the
10-dimensional Poincaré ball by stress minimization.

| Panel | Perturbs | Bound |
|---|---|---|
| A | embedding coordinates (Gaussian σ) | `mu_stability_III`: $\delta\|H\|\varepsilon$ |
| B | hidden region (symmetric difference $k$) | `mu_stability_II`: $k$ |
| C | budget distribution (Dirichlet α) | `Phi_stability_budget`: $\mathcal J(v,V)\|w_P-w_Q\|_1$ |
| D | observer position (embedding rings) | `mu_stability_V` — **see below** |

`panel_d.py` gives each observer its own $H(v)$, so both the observer and the
hidden set vary and the Lipschitz bound does not apply. It is a legitimate
smoothness characterization but **not** a test of the theorem.
`panel_d_shared_h.py` fixes $H$ and moves only the observer, which is the
theorem's actual setting. Report both, labelled differently.

`panel_normalized.py` repeats the embedding, observer and hidden-region panels for the
normalized functional mu(v,H)/mu(v,V) and the bounds of Corollary 1
(`tanh(delta*eta/2)`, `|H xor H'|/mu(v,V)`), with the same graphs, perturbations and seeds.

## 2. Dose-response benchmark (`run_dose_response`, `aggregate`)

Five graph families × four intervention axes × increasing dose, with the
observation ball frozen.

**Run both modes.** `--frozen` reproduces the original protocol, in which
coordinates never move — so an edge-only intervention *cannot* change $\mu$,
and "$\Phi$ tracks extent, not connectivity" is an identity rather than a
finding. The default mode re-embeds after each intervention, letting
connectivity propagate into the metric. The comparison between the two is the
experiment.

```bash
python synthetic/run_dose_response.py --pilot      # size it first
make overnight                                     # re-embedded
make dose-response-frozen                          # old protocol
```

Reduced grid (10 graphs/family × 8 observers × 10 realizations ≈ 64k configs)
replaces the original 4.18M. The old record count bought false precision, not
power: effective $n$ for a family-level claim was always the number of graph
instances. `aggregate.py` now emits instance-clustered intervals alongside the
record-level ones — **use `cluster_level` in the paper.**
