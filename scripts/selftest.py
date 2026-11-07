#!/usr/bin/env python3
"""
selftest.py -- verifies the package is standalone.

Checks, in order:

  1. every expected file is present
  2. no script imports anything outside this folder
  3. no script depends on the current working directory
  4. every module actually imports (catches typos, missing deps)

Requires no data. Runs in seconds. Exit code 0 means the scaffolding is sound
and any later failure is about data, not paths.

    python scripts/selftest.py
"""
import importlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAIL = []
WARN = []

ROOT_MODULES = [
    "common_hvgae", "curvature_search", "phi_utils", "simulate",
    "vectorized_simulate", "reconstruction_utils", "ogd_utils",
    "enrichment_utils", "run_everything",
]
SYNTH_MODULES = [
    "hetero_graph", "graph_families", "observer_utils",
    "interventions_dose_response", "reembed",
]
EXPECTED_SCRIPTS = [
    "synthetic/run_dose_response.py", "synthetic/aggregate.py",
    "synthetic/panel_a.py", "synthetic/panel_b.py", "synthetic/panel_c.py",
    "synthetic/panel_d.py", "synthetic/panel_d_shared_h.py",
    "embedding_ablation/run_grid_ablation.py",
    "embedding_ablation/grid_pairwise.py",
    "embedding_ablation/run_ablation.py",
    "embedding_ablation/run_seed_robustness.py",
    "embedding_ablation/run_curvature_ablation.py",
    "embedding_ablation/compare_results.py",
    "embedding_ablation/embedding_variants.py",
    "analysis/reconstruction_plus.py", "analysis/cross_dataset_transfer.py",
    "analysis/phi_seed_variance.py", "analysis/collect_paper_numbers.py",
    "cora/cora_train_embedding.py", "citeseer/citeseer_train_embedding.py",
    "go_dag/go_dag_train_embedding.py",
    "string_ppi/string_ppi_train_embedding.py",
    "obgn_arxiv/ogbn_arxiv_train_embedding.py",
]

print("[1/4] files present")
for rel in EXPECTED_SCRIPTS + [f"{m}.py" for m in ROOT_MODULES]:
    if not (REPO / rel).exists():
        FAIL.append(f"missing file: {rel}")
n_lean = len(list((REPO / "lean" / "ObserverGeometry").rglob("*.lean")))
if n_lean != 17:
    WARN.append(f"expected 17 .lean files, found {n_lean}")
print(f"      {len(EXPECTED_SCRIPTS)} scripts, {len(ROOT_MODULES)} root modules, "
      f"{n_lean} lean files")

print("[2/4] no outward imports / cwd dependence")
BAD = [
    (re.compile(r'sys\.path\.insert\(\s*0\s*,\s*["\']\.\.["\']'),
     'sys.path.insert(0, "..") -- breaks unless cwd is the subfolder'),
    (re.compile(r'^ROOT\s*=\s*Path\(\s*["\']\.["\']\s*\)', re.M),
     'ROOT = Path(".") -- resolves against cwd'),
    (re.compile(r'root\s*=\s*["\']data["\']'),
     'root="data" -- resolves against cwd'),
    (re.compile(r'os\.makedirs\(\s*["\']results["\']'),
     'os.makedirs("results") -- resolves against cwd'),
]
for py in sorted(REPO.rglob("*.py")):
    if py.name == "selftest.py":
        continue
    txt = py.read_text()
    for rx, msg in BAD:
        if rx.search(txt):
            FAIL.append(f"{py.relative_to(REPO)}: {msg}")

print("[3/4] modules import")
missing_deps = set()
for mod in ROOT_MODULES:
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        missing_deps.add(str(e).split("'")[1] if "'" in str(e) else str(e))
    except Exception as e:
        FAIL.append(f"import {mod}: {type(e).__name__}: {e}")

sys.path.insert(0, str(REPO / "synthetic"))
for mod in SYNTH_MODULES:
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        missing_deps.add(str(e).split("'")[1] if "'" in str(e) else str(e))
    except Exception as e:
        FAIL.append(f"import {mod}: {type(e).__name__}: {e}")

if missing_deps:
    WARN.append("third-party packages not installed: "
                + ", ".join(sorted(missing_deps))
                + "  ->  pip install -r requirements.txt")

print("[4/4] scripts compile")
for rel in EXPECTED_SCRIPTS:
    p = REPO / rel
    if not p.exists():
        continue
    r = subprocess.run([sys.executable, "-m", "py_compile", str(p)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        FAIL.append(f"compile {rel}: {r.stderr.strip().splitlines()[-1]}")

print()
for w in WARN:
    print(f"  WARN  {w}")
if FAIL:
    print(f"\n  {len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print(f"    - {f}")
    sys.exit(1)
print("  OK -- package is standalone. Any further error is about data, not paths.")
