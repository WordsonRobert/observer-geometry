#!/usr/bin/env bash
# triage.sh -- run this FIRST. Twenty minutes here prevents a wasted day.
#
#   1. K* consistency between the ablation grid and the cached Cora Phi
#   2. which expensive artifacts already exist (never recompute those)
#   3. how long one synthetic graph instance takes (sizes the overnight run)
#
# Safe to run from anywhere; resolves everything relative to this file.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}

echo "=============================================="
echo " 0. SELF-TEST (imports resolve inside folder?)"
echo "=============================================="
$PY scripts/selftest.py || echo "  [!] selftest failed -- fix before running experiments"

echo
echo "=============================================="
echo " 1. K* CONSISTENCY"
echo "=============================================="
$PY - <<'PY'
import glob, os
try:
    import torch
except ImportError:
    print("  [!] torch not importable -- pip install -r requirements.txt"); raise SystemExit
p = "cora/hvgae_embedding/phi_full_meanstd.pt"
if os.path.exists(p):
    d = torch.load(p, map_location="cpu")
    print(f"  cora cached K* = {d.get('K_star') if isinstance(d, dict) else None}")
else:
    print(f"  [ ] {p} missing -- copy the artifacts in, or run:")
    print("        python cora/cora_train_embedding.py")
    print("        python run_everything.py --datasets cora")
files = sorted(glob.glob("embedding_ablation/embeddings/phi_*.pt"))
if files:
    print("\n  existing ablation phi tensors:")
    for f in files:
        print(f"    {f}")
    print("\n  Stale phi_*.pt from a different K* silently corrupts every rho.")
    print("  If unsure:  rm embedding_ablation/embeddings/phi_grid_*.pt")
else:
    print("\n  no ablation phi tensors yet (clean slate)")
PY

echo
echo "=============================================="
echo " 2. EXISTING ARTIFACTS"
echo "=============================================="
mark() { if [ -f "$1" ]; then printf '%s' "$2 "; else printf '%s' "-- "; fi; }
for d in cora citeseer go_dag string_ppi obgn_arxiv; do
  printf '  %-12s' "$d"
  mark "$d/hvgae_embedding/embedding.pt"        "emb"
  mark "$d/hvgae_embedding/phi_full_meanstd.pt" "phi"
  mark "$d/_classical_features_cache.npy"       "cent"
  mark "$d/correlation_summary.json"            "corr"
  echo
done
echo "  'cent' present => do NOT rerun the correlations stage (hours saved)"

echo
echo "=============================================="
echo " 3. SYNTHETIC TIMING PILOT"
echo "=============================================="
$PY synthetic/run_dose_response.py --pilot || \
  echo "  [!] pilot failed -- see error above"

echo
echo "=============================================="
echo " NEXT"
echo "=============================================="
echo "  make overnight   # dose-response, hours, resumable"
echo "  make morning     # seeds -> grid -> curvature (GPU)"
echo "  make analysis    # minutes"
echo "  make collect     # results/paper_numbers.json"
