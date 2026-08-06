# Observer Geometry -- experiment driver
#
# The schedule is built around one fact: exactly one job needs hours
# (the synthetic dose-response). Everything else is minutes. So:
#
#   make triage      20 min, do this first
#   make overnight   launch before sleeping
#   make morning     GPU jobs, unattended ~2h
#   make analysis    the cheap stuff, minutes
#   make collect     one JSON with every paper number
#
# Override the GPU:  make morning DEVICE=cuda:0

PY      ?= python
DEVICE  ?= cuda:1
WORKERS ?= 7
SEEDS   ?= 3

.PHONY: help selftest triage overnight morning analysis collect \
        dose-response dose-response-frozen grid seeds curvature \
        panels recon transfer phivar clean-checkpoints

help:
	@echo "make selftest    -- verify the package is standalone (seconds, no data)"
	@echo "make triage      -- selftest + K* check + inventory + timing pilot"
	@echo "make overnight   -- dose-response rerun (hours) -- launch and sleep"
	@echo "make morning     -- grid + seed + curvature ablations (GPU, ~2h)"
	@echo "make analysis    -- reconstruction+, transfer, panels (minutes)"
	@echo "make collect     -- results/paper_numbers.json"
	@echo ""
	@echo "Individual: dose-response dose-response-frozen grid seeds"
	@echo "            curvature panels recon transfer phivar"

selftest:
	$(PY) scripts/selftest.py

triage: selftest
	bash scripts/triage.sh

# ---------------------------------------------------------------- overnight
overnight: dose-response

dose-response:
	$(PY) synthetic/run_dose_response.py --workers $(WORKERS)
	$(PY) synthetic/aggregate.py --raw results/raw_reembed \
	      --out results/paper_summary.json

# the old protocol, for the side-by-side that makes Section 5.3 defensible
dose-response-frozen:
	$(PY) synthetic/run_dose_response.py --frozen --workers $(WORKERS) \
	      --out results/raw_frozen
	$(PY) synthetic/aggregate.py --raw results/raw_frozen \
	      --out results/summary_frozen.json

# ------------------------------------------------------------------ morning
morning: grid seeds curvature

grid:
	$(PY) embedding_ablation/run_grid_ablation.py
	$(PY) embedding_ablation/grid_pairwise.py

seeds:
	$(PY) embedding_ablation/run_seed_robustness.py

curvature:
	$(PY) embedding_ablation/run_curvature_ablation.py
	$(PY) embedding_ablation/compare_results.py

# ----------------------------------------------------------------- analysis
analysis: recon transfer panels

recon:
	$(PY) analysis/reconstruction_plus.py

transfer:
	$(PY) analysis/cross_dataset_transfer.py

phivar:
	$(PY) analysis/phi_seed_variance.py --seeds $(SEEDS) --device $(DEVICE)

panels:
	$(PY) synthetic/panel_a.py
	$(PY) synthetic/panel_b.py
	$(PY) synthetic/panel_c.py
	$(PY) synthetic/panel_d.py
	$(PY) synthetic/panel_d_shared_h.py

collect:
	$(PY) analysis/collect_paper_numbers.py

clean-checkpoints:
	rm -f .checkpoint_*.json
	@echo "Stage checkpoints cleared. Embeddings and Phi are untouched."
