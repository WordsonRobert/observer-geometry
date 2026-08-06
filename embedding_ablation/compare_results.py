"""
Run from ~/projects/niggtract2/embedding_ablation/ AFTER run_ablation.py and
(optionally) run_curvature_ablation.py.

Builds Spearman correlation matrices:
  1. Across embedding variants (HVGAE, Poincare, HGCN-lite, Euclidean SAGE, random,
     HVGAE-with-Euclidean-kernel)
  2. Across the 6 fixed curvatures (if run_curvature_ablation.py was run)
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO))
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE))

import glob, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

VARIANT_ORDER = ["hvgae_main", "hvgae_euclidean_kernel", "poincare_direct",
                  "hgcn_lite", "euclidean_sage", "random"]
VARIANT_LABELS = {
    "hvgae_main": "HVGAE (main)",
    "hvgae_euclidean_kernel": "HVGAE + Euclidean kernel",
    "poincare_direct": "Poincare direct",
    "hgcn_lite": "HGCN-lite",
    "euclidean_sage": "Euclidean SAGE",
    "random": "Random",
}


def build_correlation_matrix(names, phi_dict):
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = spearmanr(phi_dict[names[i]].numpy(), phi_dict[names[j]].numpy())
            mat[i, j] = rho
    return mat


def plot_matrix(mat, labels, title, path):
    plt.figure(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    plt.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                      color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    # ---------------- embedding variant comparison ----------------
    available = [n for n in VARIANT_ORDER if os.path.exists(str(_HERE / "embeddings") + f"/phi_{n}.pt")]
    missing = [n for n in VARIANT_ORDER if n not in available]
    if missing:
        print(f"Warning: missing Phi files for: {missing} -- run run_ablation.py first")

    phi_dict = {n: torch.load(str(_HERE / "embeddings") + f"/phi_{n}.pt") for n in available}
    labels = [VARIANT_LABELS[n] for n in available]
    mat = build_correlation_matrix(available, phi_dict)

    print("\nEmbedding variant correlation matrix (Spearman rho):")
    print(f"{'':30}" + "".join(f"{l[:12]:>14}" for l in labels))
    for i, name in enumerate(available):
        print(f"{labels[i]:30}" + "".join(f"{mat[i,j]:14.3f}" for j in range(len(available))))

    plot_matrix(mat, labels, "Embedding variant robustness: Spearman(Phi_A, Phi_B)",
                "embedding_variant_correlation.png")
    print("\nSaved embedding_variant_correlation.png")

    results = {"variants": available, "matrix": mat.tolist()}
    with open("embedding_variant_correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    if "hvgae_main" in available:
        print(f"\n--- Correlation vs HVGAE (main) ---")
        i_main = available.index("hvgae_main")
        for j, name in enumerate(available):
            if name == "hvgae_main":
                continue
            print(f"  rho(HVGAE, {VARIANT_LABELS[name]}) = {mat[i_main, j]:.4f}")

    # ---------------- curvature comparison ----------------
    curvature_files = sorted(glob.glob(str(_HERE / "embeddings") + "/phi_hvgae_curvature_*.pt"))
    if curvature_files:
        curv_names = [os.path.basename(f).replace("phi_hvgae_curvature_", "").replace(".pt", "")
                       for f in curvature_files]
        curv_labels = [f"c={c}" for c in curv_names]
        curv_phi = {name: torch.load(f) for name, f in zip(curv_names, curvature_files)}
        curv_mat = np.zeros((len(curv_names), len(curv_names)))
        for i in range(len(curv_names)):
            for j in range(len(curv_names)):
                rho, _ = spearmanr(curv_phi[curv_names[i]].numpy(), curv_phi[curv_names[j]].numpy())
                curv_mat[i, j] = rho

        print("\nCurvature correlation matrix (Spearman rho):")
        print(f"{'':10}" + "".join(f"{l:>10}" for l in curv_labels))
        for i, name in enumerate(curv_names):
            print(f"{curv_labels[i]:10}" + "".join(f"{curv_mat[i,j]:10.3f}" for j in range(len(curv_names))))

        plot_matrix(curv_mat, curv_labels, "Curvature sensitivity: Spearman(Phi_c1, Phi_c2)",
                    "curvature_correlation.png")
        print("Saved curvature_correlation.png")

        with open("curvature_correlation.json", "w") as f:
            json.dump({"curvatures": curv_names, "matrix": curv_mat.tolist()}, f, indent=2)
    else:
        print("\nNo curvature ablation results found -- run run_curvature_ablation.py first if needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
