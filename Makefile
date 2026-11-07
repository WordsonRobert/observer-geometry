# Observer Geometry -- experiment driver
#
#   make selftest     package check (seconds, no data)
#   make triage       selftest + K* check + inventory + timing pilot
#   make overnight    intervention benchmark, re-embedded protocol (hours)
#   make morning      encoder x geometry grid, seed and curvature ablations (GPU)
#   make analysis     reconstruction, transfer, stability panels (minutes)
#   make exact        exact descriptor (Proposition 2) and the analyses built on it
#   make collect      results/paper_numbers.json
#   make lean         build the Lean development
#
# Device for the simulator and training scripts:  make morning DEVICE=cuda:0

PY      ?= python
DEVICE  ?= cuda
WORKERS ?= 7
SEEDS   ?= 3

.PHONY: help selftest triage overnight morning analysis collect lean \
        dose-response dose-response-frozen grid seeds curvature robustness-ext \
        panels recon transfer transfer-heldout phivar exact exact-sweeps closer \
        sensitivity budget growth capcheck fig2 clean-checkpoints

help:
	@echo "make selftest | triage | overnight | morning | analysis | exact | collect | lean"
	@echo "individual: dose-response dose-response-frozen grid seeds curvature robustness-ext"
	@echo "            panels recon transfer transfer-heldout phivar exact-sweeps closer"
	@echo "            sensitivity budget growth capcheck fig2"

selftest:
	$(PY) scripts/selftest.py

triage: selftest
	bash scripts/triage.sh

# ------------------------------------------------- intervention benchmark
overnight: dose-response

dose-response:
	$(PY) synthetic/run_dose_response.py --workers $(WORKERS)
	$(PY) synthetic/aggregate.py --raw results/raw_reembed --out results/paper_summary.json

# frozen-coordinate control (edge-only interventions cannot change mu here)
dose-response-frozen:
	$(PY) synthetic/run_dose_response.py --frozen --workers $(WORKERS) --out results/raw_frozen
	$(PY) synthetic/aggregate.py --raw results/raw_frozen --out results/summary_frozen.json

# ------------------------------------------------- embedding robustness
morning: grid seeds curvature

grid:
	$(PY) embedding_ablation/run_grid_ablation.py
	$(PY) embedding_ablation/grid_pairwise.py

seeds:
	$(PY) embedding_ablation/run_seed_robustness.py

curvature:
	$(PY) embedding_ablation/run_curvature_ablation.py
	$(PY) embedding_ablation/compare_results.py

# same protocol on Citeseer and GO DAG, exact Phi
robustness-ext:
	$(PY) embedding_ablation/run_robustness_extension.py citeseer
	$(PY) embedding_ablation/run_robustness_extension.py go_dag

# ------------------------------------------------- reconstruction and transfer
analysis: recon transfer transfer-heldout panels

recon:
	$(PY) analysis/reconstruction_plus.py

transfer:
	$(PY) analysis/cross_dataset_transfer.py

transfer-heldout:
	$(PY) analysis/transfer_heldout.py

phivar:
	$(PY) analysis/phi_seed_variance.py --seeds $(SEEDS) --device $(DEVICE)

panels:
	$(PY) synthetic/panel_a.py
	$(PY) synthetic/panel_b.py
	$(PY) synthetic/panel_c.py
	$(PY) synthetic/panel_d.py
	$(PY) synthetic/panel_d_shared_h.py
	$(PY) synthetic/panel_normalized.py

# ------------------------------------------------- exact descriptor
exact:
	$(PY) analysis/exact_phi_all.py
	$(MAKE) exact-sweeps closer sensitivity budget growth capcheck fig2

exact-sweeps:
	for d in 0.5 2; do $(PY) analysis/exact_phi_all.py --delta $$d; done
	for e in 0.05 0.01; do $(PY) analysis/exact_phi_all.py --eps $$e; done
	$(PY) analysis/exact_phi_all.py --datasets string_ppi,obgn_arxiv --cap none
	# curvature-normalized decay delta = 1/sqrt(c), and the wider Cora sweep
	$(PY) analysis/exact_phi_all.py --datasets citeseer --delta 0.5
	$(PY) analysis/exact_phi_all.py --datasets string_ppi --delta 0.7071
	$(PY) analysis/exact_phi_all.py --datasets obgn_arxiv --delta 0.3536
	for d in 0.25 4; do $(PY) analysis/exact_phi_all.py --datasets cora --delta $$d; done
	$(PY) analysis/exact_phi_all.py --datasets cora --eps 0.005

closer:
	$(PY) analysis/closer_baselines.py

sensitivity:
	$(PY) analysis/sensitivity.py

budget:
	$(PY) analysis/budget_value.py

growth:
	$(PY) analysis/growth.py

capcheck:
	$(PY) analysis/cap_inertness.py

fig2:
	$(PY) analysis/make_fig2.py

collect:
	$(PY) analysis/collect_paper_numbers.py

lean:
	cd lean/ObserverGeometry && lake exe cache get && lake build

clean-checkpoints:
	rm -f .checkpoint_*.json
