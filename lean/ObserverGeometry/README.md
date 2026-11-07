# ObserverGeometry (Lean 4)

Mechanized results of *Observer Geometry: Describing Graph Structure Beyond the Observation
Horizon*. Lean v4.32.0, Mathlib v4.32.0; 19 modules, 103 theorems, no `sorry`, no
project-introduced axioms (the axiom closure is `propext`, `Classical.choice`, `Quot.sound`).

```bash
lake exe cache get   # prebuilt Mathlib
lake build
```

| Module | Content (paper) |
|---|---|
| `Framework/Axioms` | observation processes, hidden regions, region functionals, the descriptor; boundedness and budget stability (Prop. 1, Thm. 1(b)) |
| `Framework/MetricLocality` | basepoint-relative locality and isometry invariance (Thm. 1(a)) |
| `Instantiation/Mu` | the exponential functional `mu` as a region functional |
| `Instantiation/Stability` | hidden-region, embedding and observer perturbation bounds for `mu` (Thm. 1(c)-(e)) |
| `Instantiation/NormalizedStability` | the same bounds for the normalized descriptor, independent of the hidden-region size: `tanh(δη/2)` |
| `Instantiation/Coverage` | coverage identity behind Proposition 2: expected remaining weight after K i.i.d. trajectories equals `Σ_u w_u (1 - p_u)^K` |
| `Instantiation/Sensitivity`, `Canonicality` | sensitivity, measure property, shell decomposition (Prop. 2) |
| `RegularTree/*` | k-regular tree as a metric space, closed form, threshold, identifiability, bridge to the framework descriptor (Thm. 2) |

Not formalized: the hyperbolic embedding, the specific transition law of the walk (the coverage
identity is proved for an arbitrary finite trajectory law), and the Monte Carlo estimator.
