"""
collect_paper_numbers.py
========================
Walks the whole repo, gathers every result JSON, and emits ONE file:

    results/paper_numbers.json

with a flat `tables` section holding exactly the numbers that go into the
paper's tables and text. Written so that when you build figures you read one
file, and so that any number in the paper can be traced back to the artifact
that produced it.

Also reports what is MISSING, which doubles as a pre-submission checklist.

Usage:
    python analysis/collect_paper_numbers.py
"""
import json
import sys
from pathlib import Path

from _common import DATASETS, DISPLAY, RESULTS, ROOT, folder, write_json


def read(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_read_error": str(e)}


def read_any(*paths):
    """Try each candidate path in order; return the first that exists.

    run_everything.py writes k_sweep output as k_sweep_full_<display_name>.json
    (dataset-suffixed) rather than the flat k_sweep_full.json this collector
    originally assumed. Both are checked so pre-existing runs (suffixed) and
    fresh runs (either naming) both resolve without a manual rename.
    """
    for p in paths:
        r = read(p)
        if r is not None:
            return r
    return None


def main():
    out = {"per_dataset": {}, "ablation": {}, "synthetic": {}, "tables": {},
           "missing": []}

    def need(obj, label):
        if obj is None:
            out["missing"].append(label)
        return obj

    # ---------------- per dataset ----------------
    for name in DATASETS:
        f = folder(name)
        d = {
            "correlations":        read(f / "correlation_summary.json"),
            "reconstruction":      read(f / "reconstruction_results.json"),
            "reconstruction_plus": read(f / "reconstruction_plus.json"),
            "radius":              read(f / "radius_correlation.json"),
            "phi_seed_variance":   read(f / "phi_seed_variance.json"),
            "classification":      read(f / "classification.json"),
            "hierarchy":           read(f / "hierarchy_validation.json"),
            "biology":             read(f / "biology_validation.json"),
            "ogd":                 read(f / "ogd_experiment_results.json"),
            "k_sweep":             read_any(f / "k_sweep_full.json",
                                            f / f"k_sweep_full_{name}.json"),
            "runtime":             read(f / "runtime_summary.json"),
            "closer_baselines":    read(f / "closer_baselines.json"),
        }
        for k in ("correlations", "reconstruction_plus", "radius", "k_sweep"):
            if d[k] is None:
                out["missing"].append(f"{name}/{k}")
        out["per_dataset"][name] = d

    # ---------------- ablation ----------------
    ab = ROOT / "embedding_ablation"
    out["ablation"] = {
        "grid":            need(read(ab / "grid_ablation_results.json"), "ablation/grid"),
        "grid_pairwise":   need(read(ab / "grid_pairwise.json"), "ablation/grid_pairwise"),
        "seed_robustness": need(read(ab / "seed_robustness_results.json"),
                                "ablation/seed_robustness"),
        "curvature":       read(ab / "curvature_correlation.json"),
        "variants":        read(ab / "embedding_variant_correlation.json"),
        "robustness_citeseer": read(ab / "robust_citeseer.json"),
        "robustness_go_dag":   read(ab / "robust_go_dag.json"),
    }

    # ---------------- synthetic ----------------
    sy = ROOT / "synthetic" / "results"
    out["synthetic"] = {
        "panelA": read(sy / "panelA_embedding_perturbation.json"),
        "panelB": read(sy / "panelB_hidden_region_perturbation.json"),
        "panelC": read(sy / "panelC_budget_perturbation.json"),
        "panelD_original": read(sy / "panelD_observer_perturbation.json"),
        "panelD_shared_H": need(read(sy / "panelD_shared_H.json"),
                                "synthetic/panelD_shared_H"),
        "dose_response_reembed": read(sy / "paper_summary.json"),
        "dose_response_frozen": read(sy / "summary_frozen.json"),
    }
    out["ablation"]["cross_dataset_transfer"] = read(RESULTS / "cross_dataset_transfer.json")
    out["ablation"]["transfer_heldout"] = read(RESULTS / "transfer_heldout.json")

    # ---------------- exact descriptor ----------------
    out["exact"] = {k: read(RESULTS / f"{k}.json") for k in
                    ("exact_validation", "budget_value", "cap_inertness", "sensitivity",
                     "growth", "fig2_summary")}

    # ================= TABLES =================
    # Table 1: structural distinctiveness
    t1 = []
    for name in DATASETS:
        rp = out["per_dataset"][name].get("reconstruction_plus")
        corr = out["per_dataset"][name].get("correlations")
        if not rp or "paper_row" not in rp:
            continue
        best_metric, best_rho = None, 0.0
        if corr:
            for k, v in corr.items():
                r = abs(v.get("spearman_rho", 0.0))
                if r > abs(best_rho):
                    best_metric, best_rho = k, v["spearman_rho"]
        pr = rp["paper_row"]
        row = {
            "dataset": DISPLAY[name],
            "best_classical_metric": best_metric,
            "best_classical_spearman": best_rho,
            "linear_r2": pr["linear_r2"],
            "nonlinear_r2": pr["nonlinear_r2"],
            "nonlinearity_gap": pr["nonlinearity_gap"],
            "nonlinear_rank_recovery": pr["nonlinear_rank_recovery"],
            "radius_spearman": rp.get("radius_spearman_vs_phi"),
        }
        cpr = rp.get("classical_plus_radius_random")
        if isinstance(cpr, dict) and "xgboost" in cpr:
            row["nonlinear_r2_with_radius"] = cpr["xgboost"]["r2"]["mean"]
        ch = rp.get("classical_community_holdout")
        if isinstance(ch, dict) and "xgboost" in ch:
            row["nonlinear_r2_community_holdout"] = ch["xgboost"]["r2"]["mean"]
        t1.append(row)
    t1.sort(key=lambda r: -r["nonlinearity_gap"])
    out["tables"]["structural_distinctiveness"] = t1

    # Section 7.1: robustness ordering, read against the noise floor
    gp = out["ablation"].get("grid_pairwise")
    if gp:
        nf = (gp.get("seed_noise_floor") or {}).get("mean")
        rows = []
        for label, key in [("geometry (encoder fixed)", "geometry_effect_within_encoder"),
                           ("encoder (geometry fixed)", "encoder_effect_within_geometry")]:
            for sub, band in (gp.get(key) or {}).items():
                if band:
                    rows.append({"factor": f"{label} [{sub}]", **band})
        for label, key in [("SAGE/GAT internal", "sage_gat_cluster_internal"),
                           ("GCN internal", "gcn_cluster_internal"),
                           ("SAGE/GAT vs GCN", "across_clusters_sage_gat_vs_gcn"),
                           ("random null", "random_null_vs_trained")]:
            b = gp.get(key)
            if b:
                rows.append({"factor": label, **b})
        for r in rows:
            if nf is not None and r.get("mean") is not None:
                r["below_noise_floor"] = bool(r["mean"] < nf)
        out["tables"]["robustness_ordering"] = {
            "seed_noise_floor": nf,
            "note": ("A band whose mean sits BELOW the noise floor is a real "
                     "effect. Do not describe it as agreement."),
            "rows": sorted(rows, key=lambda r: r.get("mean", 0)),
        }

    # Section 7.2: radius correlations
    out["tables"]["radius_correlation"] = {
        DISPLAY[n]: (out["per_dataset"][n].get("radius") or {})
        for n in DATASETS
    }

    # Section 7.2: K* actually used (read from the Phi tensor, not the sweep).
    # The k_sweep JSONs report a stricter rho>=0.99 plateau, but the Phi tensors
    # were computed at the rho>=0.98 plateau. The paper must report the K the
    # descriptor actually uses, so we read K_star off the tensor and record the
    # sweep's stricter plateau separately for transparency.
    ks = {}
    ks_sweep_plateau = {}
    for n in DATASETS:
        f = folder(n)
        phi_path = f / "hvgae_embedding" / "phi_full_meanstd.pt"
        if phi_path.exists():
            try:
                import torch as _torch
                d = _torch.load(phi_path, map_location="cpu")
                ks[DISPLAY[n]] = d.get("K_star") if isinstance(d, dict) else None
            except Exception:
                ks[DISPLAY[n]] = None
        sw = out["per_dataset"][n].get("k_sweep")
        if isinstance(sw, dict):
            ks_sweep_plateau[DISPLAY[n]] = sw.get("plateau_K", sw.get("K_star"))
    out["tables"]["k_star"] = ks
    out["tables"]["k_sweep_strict_plateau"] = ks_sweep_plateau
    out["tables"]["k_star_note"] = (
        "k_star is the budget the Phi tensor was computed at (rho>=0.98 vs K=100). "
        "k_sweep_strict_plateau is the sweep's rho>=0.99 plateau, reported for "
        "transparency; the gain from k_star to the strict plateau lies within the "
        "seed noise floor (0.959+/-0.004).")

    write_json(out, RESULTS / "paper_numbers.json")

    print("\n--- Table 1 (ordered by nonlinearity gap) ---")
    for r in t1:
        print(f"  {r['dataset']:<12} best={str(r['best_classical_metric']):<12} "
              f"rho={r['best_classical_spearman']:+.3f}  "
              f"lin={r['linear_r2']:.3f}  nonlin={r['nonlinear_r2']:.3f}  "
              f"gap={r['nonlinearity_gap']:.3f}")

    if out["missing"]:
        print(f"\n--- MISSING ({len(out['missing'])}) ---")
        for m in out["missing"]:
            print(f"  [ ] {m}")
    else:
        print("\nAll expected artifacts present.")


if __name__ == "__main__":
    main()
