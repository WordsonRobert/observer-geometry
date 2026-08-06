#!/usr/bin/env python3
"""
_patch_paths.py -- build-time only, not part of the shipped runtime.

The vendored scripts were written assuming a specific working directory:

    sys.path.insert(0, "..")          # only correct if cwd == the subfolder
    Planetoid(root="data")            # only correct if cwd == the dataset folder
    os.makedirs("results")            # writes wherever you happened to be
    ROOT = Path(".")                  # run_everything.py

That is what broke `bash scripts/triage.sh` from the repo root. This rewrites
every such site to be anchored on __file__, so any script runs correctly from
any cwd and resolves imports only inside this folder.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BOOTSTRAP = (
    "import sys as _sys, pathlib as _pl\n"
    "_REPO = _pl.Path(__file__).resolve().parent{up}\n"
    "_sys.path.insert(0, str(_REPO))\n"
    "_HERE = _pl.Path(__file__).resolve().parent\n"
    "_sys.path.insert(0, str(_HERE))\n"
)

changed = []


def patch(path, up_levels, data_anchor=None, results_anchor=None):
    p = Path(path)
    src = p.read_text()
    orig = src

    # 1. kill cwd-relative sys.path hacks
    src = re.sub(r'^\s*sys\.path\.insert\(\s*0\s*,\s*["\']\.\.["\']\s*\).*$',
                 '', src, flags=re.M)
    src = re.sub(r'^\s*sys\.path\.insert\(\s*0\s*,\s*str\(ROOT\)\s*\).*$',
                 '', src, flags=re.M)

    # 2. inject the anchored bootstrap after the docstring
    up = "".join(".parent" for _ in range(up_levels))
    boot = BOOTSTRAP.format(up=up)
    m = re.match(r'\s*(?:"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')\s*', src)
    if "_REPO = _pl.Path" not in src:
        if m:
            src = src[:m.end()] + "\n" + boot + "\n" + src[m.end():]
        else:
            src = boot + "\n" + src

    # 3. anchor dataset download roots
    if data_anchor:
        src = re.sub(r'root\s*=\s*["\']data["\']',
                     'root=str(_HERE / "data")', src)
        src = re.sub(r'root\s*=\s*["\']\./data["\']',
                     'root=str(_HERE / "data")', src)
        src = src.replace('"data"', 'str(_HERE / "data")') \
            if 'os.path.join(str(_HERE / "data")' not in src else src
        # undo over-eager replacement inside already-anchored joins
        src = src.replace('str(_HERE / str(_HERE / "data"))', 'str(_HERE / "data")')

    # 4. anchor result output dirs
    if results_anchor:
        src = re.sub(r'os\.makedirs\(\s*["\']results["\']',
                     'os.makedirs(str(_HERE / "results")', src)
        src = re.sub(r'["\']results/', 'str(_HERE / "results") + "/', src)
        src = src.replace('str(_HERE / "results") + "/', 'str(_HERE / "results") + "/')

    # 5. run_everything.py: ROOT must be the repo, not cwd
    src = re.sub(r'^ROOT\s*=\s*Path\(\s*["\']\.["\']\s*\)\s*$',
                 'ROOT = Path(__file__).resolve().parent', src, flags=re.M)

    if src != orig:
        p.write_text(src)
        changed.append(str(p.relative_to(ROOT)))


# repo root modules: no sys.path games needed, but fix ROOT in run_everything
for f in ["run_everything.py"]:
    patch(ROOT / f, 0)

# synthetic/: deps live beside them
for f in ["panel_a.py", "panel_b.py", "panel_c.py", "panel_d.py"]:
    patch(ROOT / "synthetic" / f, 1, results_anchor=True)
for f in ["hetero_graph.py", "graph_families.py", "observer_utils.py",
          "interventions_dose_response.py"]:
    patch(ROOT / "synthetic" / f, 1)

# embedding_ablation/: needs repo root for simulate/phi_utils
for f in ["embedding_variants.py", "run_ablation.py", "run_curvature_ablation.py",
          "run_seed_robustness.py", "compare_results.py"]:
    patch(ROOT / "embedding_ablation" / f, 1)

# dataset folders: need repo root for curvature_search, and anchored data/
for folder, f in [("cora", "cora_train_embedding.py"),
                  ("citeseer", "citeseer_train_embedding.py"),
                  ("go_dag", "go_dag_train_embedding.py"),
                  ("string_ppi", "string_ppi_train_embedding.py"),
                  ("obgn_arxiv", "ogbn_arxiv_train_embedding.py")]:
    patch(ROOT / folder / f, 1, data_anchor=True)

print(f"patched {len(changed)} files:")
for c in sorted(changed):
    print("  " + c)
