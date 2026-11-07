"""
cap_inertness.py
================
The vectorized simulator keeps only the 256 highest-weight neighbours of a vertex whose
degree exceeds 256 (ogbn-arXiv, STRING-PPI; GO DAG has no such vertex). A step is taken
only if the path probability stays >= eps, and path probability is at most 1, so a vertex
all of whose transition probabilities are below eps ends every trajectory that reaches it.
This script checks that every over-cap vertex is such a vertex, with and without the cap and
after removing the previous vertex, and compares the exact descriptor with and without the
cap when both artifacts exist.
Emits results/cap_inertness.json
"""
import sys
import numpy as np, torch
from _common import RESULTS, DISPLAY, cli_datasets, folder, exact_path, kstar, write_json


def main():
    out = {}
    for name in cli_datasets(sys.argv):
        rp, ci, ew = torch.load(folder(name) / "hvgae_embedding" / "transition_kernel.pt",
                                weights_only=False, map_location="cpu")
        rp, ew = rp.numpy(), ew.numpy().astype(float); deg = np.diff(rp)
        over = np.where(deg > 256)[0]
        full, capped, worst = [], [], []
        for i in over:
            w = np.sort(ew[rp[i]:rp[i + 1]])[::-1]
            full.append(w[0] / w.sum()); capped.append(w[0] / w[:256].sum()); worst.append(w[0] / (w[:256].sum() - w[1]))
        r = {"max_degree": int(deg.max()), "n_over_cap": int(len(over))}
        if len(over):
            r.update({"max_transition_prob_full": float(max(full)), "max_transition_prob_capped": float(max(capped)),
                      "max_transition_prob_capped_prev_removed": float(max(worst))})
        a, b = exact_path(name, cap=256), exact_path(name, cap=None)
        if len(over) and a.exists() and b.exists():
            K = kstar(name)
            x, y = np.load(a)["phiK"][K - 1], np.load(b)["phiK"][K - 1]
            r["max_abs_diff_exact_cap_vs_nocap"] = float(np.abs(x - y).max())
        out[name] = r; print(f"  {DISPLAY[name]:<11} {r}")
    write_json(out, RESULTS / "cap_inertness.json")


if __name__ == "__main__":
    main()
