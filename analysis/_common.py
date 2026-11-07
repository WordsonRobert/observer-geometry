"""
_common.py -- shared plumbing for every script in analysis/.

Design rule: this module NEVER recomputes anything expensive. It only reads
artifacts that run_everything.py has already produced:

    <folder>/hvgae_embedding/phi_full_meanstd.pt     dict: mean, std, seeds, K_star
    <folder>/hvgae_embedding/embedding.pt            Lorentz coords [N, 17]
    <folder>/hvgae_embedding/chosen_curvature.txt    float
    <folder>/_classical_features_cache.npy           [N, 7], FEATURE_NAMES order

If a dataset is missing an artifact the caller gets a clear error naming the
exact missing path, rather than a traceback from three imports deep.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

# repo root == parent of analysis/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Must match reconstruction_utils.FEATURE_NAMES column order exactly.
FEATURE_NAMES = ["Degree", "PageRank", "Betweenness", "Eigenvector",
                 "Closeness", "Katz", "Harmonic"]

# Keys here match DATASET_CONFIG in run_everything.py, including the
# "obgn_arxiv" spelling used by the existing folder layout.
DATASETS = ["cora", "citeseer", "go_dag", "string_ppi", "obgn_arxiv"]

DISPLAY = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "go_dag": "GO DAG",
    "string_ppi": "STRING-PPI",
    "obgn_arxiv": "ogbn-arXiv",
}


class MissingArtifact(Exception):
    pass


def folder(name):
    return ROOT / name


def _require(path, hint):
    if not Path(path).exists():
        raise MissingArtifact(f"{path} not found.\n  -> {hint}")
    return Path(path)


def load_phi(name):
    """Returns (phi [N] float64 ndarray, K_star)."""
    p = _require(folder(name) / "hvgae_embedding" / "phi_full_meanstd.pt",
                 f"run: python run_everything.py --datasets {name} --stages phi")
    d = torch.load(p, map_location="cpu")
    if isinstance(d, dict):
        return d["mean"].numpy().astype(np.float64), d.get("K_star")
    return d.numpy().astype(np.float64), None


def load_features(name):
    """Returns X [N, 7] ndarray in FEATURE_NAMES order."""
    p = _require(folder(name) / "_classical_features_cache.npy",
                 f"run: python run_everything.py --datasets {name} "
                 f"--stages correlations   (this is the expensive centrality pass)")
    return np.load(p).astype(np.float64)


def load_radius(name):
    """Hyperbolic distance from manifold origin, [N]. Computed on demand (cheap)."""
    from phi_utils import load_embedding, compute_radii
    emb = _require(folder(name) / "hvgae_embedding" / "embedding.pt",
                   "train the embedding first (curvature_search.py)")
    cur = _require(folder(name) / "hvgae_embedding" / "chosen_curvature.txt",
                   "chosen_curvature.txt should sit beside embedding.pt")
    z, manifold, c = load_embedding(str(emb), str(cur))
    r = compute_radii(z, manifold)
    return (r.numpy() if hasattr(r, "numpy") else np.asarray(r)).astype(np.float64)


def load_graph_int(name):
    """
    Integer-indexed networkx graph aligned with the Phi / feature row order.
    Reuses run_everything.py's own loaders so index alignment cannot drift.
    """
    from run_everything import DATASET_CONFIG, get_graph, _integer_indexed_graph
    cfg = DATASET_CONFIG[name]
    G, N, node_id_map, data = get_graph(name, cfg)
    return _integer_indexed_graph(G, node_id_map), N


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    print(f"  wrote {path}")


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def each_dataset(names=None):
    """
    Iterate datasets, skipping (with a clear message) any whose artifacts are
    missing. One broken dataset must never abort a whole analysis run.
    """
    for name in (names or DATASETS):
        try:
            yield name
        except MissingArtifact as e:
            print(f"[skip] {name}: {e}")


def cli_datasets(argv):
    """--datasets a,b,c  ->  list. Absent means all."""
    for i, a in enumerate(argv):
        if a == "--datasets" and i + 1 < len(argv):
            return [s.strip() for s in argv[i + 1].split(",") if s.strip()]
    return list(DATASETS)


# ---------------------------------------------------------------------------
# Exact descriptor artifacts (analysis/exact_phi_all.py). Stored per dataset in
# <dataset>/exact/ (large, git-ignored); summaries go to results/.
# ---------------------------------------------------------------------------
CAPS = {"cora": None, "citeseer": None, "go_dag": 256, "string_ppi": 256, "obgn_arxiv": 256}


def exact_tag(name, eps=1 / 50, delta=1.0, cap="config"):
    if cap == "config":
        cap = CAPS[name]
    return f"eps{eps:.4g}_d{delta:g}_cap{cap}"


def exact_path(name, eps=1 / 50, delta=1.0, cap="config"):
    d = folder(name) / "exact"
    d.mkdir(exist_ok=True)
    return d / f"exact_{exact_tag(name, eps, delta, cap)}.npz"


def load_exact(name, eps=1 / 50, delta=1.0, cap="config"):
    p = _require(exact_path(name, eps, delta, cap),
                 f"run: python analysis/exact_phi_all.py --datasets {name} "
                 f"--eps {eps} --delta {delta}")
    return np.load(p)


def kstar(name):
    return int(load_phi(name)[1])
