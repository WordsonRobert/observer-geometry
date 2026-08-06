# MANIFEST — fully standalone

Nothing here imports anything outside this folder. Every script runs correctly
from any working directory. Verify with:

```bash
python scripts/selftest.py
```

That checks file presence, absence of outward imports, absence of cwd
dependence, and that every module compiles and imports. It needs no data and
runs in seconds. **Exit code 0 means any later failure is about data, not
paths.**

## Everything is vendored

```
ROOT (shared modules — imported flat, as your code expects)
  common_hvgae.py  curvature_search.py  phi_utils.py
  simulate.py  vectorized_simulate.py
  reconstruction_utils.py  ogd_utils.py  enrichment_utils.py
  run_everything.py

cora/ citeseer/ go_dag/ string_ppi/ obgn_arxiv/
  <name>_train_embedding.py            anchored: writes into its own folder
  hvgae_embedding/                     you supply, or regenerate
  _classical_features_cache.npy        you supply, or regenerate (expensive)

synthetic/
  hetero_graph.py  graph_families.py  observer_utils.py
  interventions_dose_response.py
  panel_a.py  panel_b.py  panel_c.py  panel_d.py
  panel_d_shared_h.py     NEW    correct observer-perturbation test
  reembed.py              NEW    hyperbolic warm re-embedding
  run_dose_response.py    PATCHED hyperbolic + reduced grid + instance ids
  aggregate.py            PATCHED cluster-clustered intervals

embedding_ablation/
  embedding_variants.py  run_ablation.py
  run_curvature_ablation.py  run_seed_robustness.py  compare_results.py
  run_grid_ablation.py    PATCHED in-grid anchor + random null + K* guard
  grid_pairwise.py        NEW    real pairwise matrix vs noise floor

analysis/                 ALL NEW
  _common.py  reconstruction_plus.py  cross_dataset_transfer.py
  phi_seed_variance.py  collect_paper_numbers.py

lean/
  lakefile.lean  lean-toolchain
  ObserverGeometry/{Framework,Instantiation,RegularTree}/   17 modules

scripts/
  selftest.py  triage.sh
  _patch_paths_ALREADY_APPLIED.py    provenance only; do not rerun
```

## What I changed in YOUR files (and why)

Path anchoring only — no logic touched. 19 files had one of these:

| Was | Now | Why it mattered |
|---|---|---|
| `sys.path.insert(0, "..")` | `_REPO` from `__file__` | broke when run from the repo root |
| `ROOT = Path(".")` | `Path(__file__).resolve().parent` | `run_everything.py` resolved against cwd |
| `Planetoid(root="data")` | `str(_HERE / "data")` | downloaded into whatever cwd you were in |
| `os.makedirs("results")` | `str(_HERE / "results")` | panels wrote outside `synthetic/` |
| `out_dir="hvgae_embedding"` | `str(_HERE / "hvgae_embedding")` | same |

`scripts/_patch_paths_ALREADY_APPLIED.py` is the script that did it, kept so you
can see exactly what was rewritten. It has already run; do not run it again.

The ablation scripts now read Cora from `cora/data/` and the main embedding from
`cora/hvgae_embedding/`, so there is one dataset cache rather than two.

## You still need to supply

Only data, never code:

- `<dataset>/hvgae_embedding/{embedding.pt, chosen_curvature.txt, phi_full_meanstd.pt}`
- `<dataset>/_classical_features_cache.npy`

Or regenerate: `python cora/cora_train_embedding.py` then
`python run_everything.py --datasets cora`.

## Before pushing

- [ ] Pin the mathlib commit in `lean/lakefile.lean`
- [ ] Set `lean/lean-toolchain` to your actual version (`lake env lean --version`)
- [ ] `grep -ri "iiserkol\|wordson" .` returns nothing
- [ ] `python scripts/selftest.py` exits 0
- [ ] `python analysis/collect_paper_numbers.py` — clear the MISSING list

## Untested against real data

Path resolution, imports, and compilation are verified. Behaviour against your
actual tensors is not — I have no access to them. Failures will name the exact
missing path (`MissingArtifact`), not produce silently wrong numbers.
