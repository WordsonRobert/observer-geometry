# Manifest

Everything needed to reproduce the paper is in this repository; no script imports anything
outside it, and every script resolves paths from its own location, so any working directory
works. `python scripts/selftest.py` checks file presence, imports and compilation without
data (exit code 0 means later failures are about data, not paths).

```
ROOT (shared modules, imported flat)
  common_hvgae.py         hyperbolic GraphSAGE autoencoder (Lorentz)
  curvature_search.py     curvature selection by held-out link-prediction loss
  phi_utils.py            transition kernel, radii, sampling helpers
  simulate.py             sequential reference simulator of the observation process
  vectorized_simulate.py  batched simulator (degree cap 256 on the three large graphs)
  exact_phi.py            exact evaluator of the expected descriptor (Proposition 3)
  reconstruction_utils.py ogd_utils.py enrichment_utils.py
  run_everything.py       pipeline for the five real-world graphs

cora/ citeseer/ go_dag/ string_ppi/ obgn_arxiv/
  <name>_train_embedding.py      embedding + curvature search, writes into its own folder
  go_dag/go_dag_ksweep_phi.py    budget sweep and Monte Carlo Phi for the GO DAG
  hvgae_embedding/ _classical_features_cache.npy *.json  (cached artifacts)

synthetic/
  hetero_graph.py graph_families.py observer_utils.py
  interventions_dose_response.py reembed.py run_dose_response.py aggregate.py
  panel_a.py panel_b.py panel_c.py panel_d.py panel_d_shared_h.py

embedding_ablation/
  embedding_variants.py run_ablation.py run_curvature_ablation.py run_seed_robustness.py
  compare_results.py run_grid_ablation.py grid_pairwise.py run_robustness_extension.py

analysis/
  _common.py reconstruction_plus.py cross_dataset_transfer.py transfer_heldout.py
  phi_seed_variance.py exact_phi_all.py closer_baselines.py sensitivity.py
  budget_value.py growth.py cap_inertness.py make_fig2.py collect_paper_numbers.py

lean/ObserverGeometry/   Lean 4 + Mathlib, 19 modules
scripts/                 selftest.py triage.sh
```

Large regenerable artifacts (`*/exact/`, raw downloads) are git-ignored.
