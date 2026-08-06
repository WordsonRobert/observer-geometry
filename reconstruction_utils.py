"""
Shared core for the "Can classical centralities reconstruct Phi?" experiment.
Run on REAL graphs only (Cora, Citeseer, ogbn-arxiv, STRING, GO DAG) -- deliberately
NOT on synthetic graphs, since Phi's synthetic behavior is already validated
separately (theorem visualization, sensitivity, structural controls), and testing
reconstructability there would be circular (Phi is generated from the same
construction as the synthetic graph itself).

Requires: pip install xgboost scikit-learn
"""
import json
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor

FEATURE_NAMES = ["Degree", "PageRank", "Betweenness", "Eigenvector",
                  "Closeness", "Katz", "Harmonic"]


def approximate_closeness_and_harmonic(G, N, n_samples=1000, seed=0):
    """
    Exact closeness/harmonic centrality require a full BFS from EVERY node --
    O(N*(N+E)), which is intractable at ogbn-arxiv's scale (169K nodes, 2.3M edges
    -- roughly 4*10^11 edge traversals, realistically 1-3+ days in networkx).

    This uses the standard landmark/pivot approximation: sample n_samples random
    "pivot" nodes, BFS from each (cost O(n_samples*(N+E)) instead of O(N*(N+E))),
    and for every node v estimate its average distance / average inverse distance
    to a random node using only distances observed from the pivots. This is a
    Monte Carlo estimator -- each pivot's distance to v is one sample of "distance
    from a random node to v", so averaging over many pivots estimates the true
    population mean.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    n_samples = min(n_samples, len(nodes))
    pivots = rng.sample(nodes, n_samples)
    pivots_set = set(pivots)

    sum_dist = {v: 0.0 for v in nodes}       # sum over REACHED pivots only
    sum_inv_dist = {v: 0.0 for v in nodes}   # sum over REACHED pivots only
    reached_count = {v: 0 for v in nodes}

    for p in pivots:
        lengths = nx.single_source_shortest_path_length(G, p)
        for v, d in lengths.items():
            if v == p:
                continue  # exclude self-distance -- NetworkX's exact definitions do too
            sum_dist[v] += d
            sum_inv_dist[v] += 1.0 / d
            reached_count[v] += 1

    closeness, harmonic = {}, {}
    for v in nodes:
        if reached_count[v] == 0:
            closeness[v], harmonic[v] = 0.0, 0.0
            continue
        avg_dist_reached = sum_dist[v] / reached_count[v]
        closeness[v] = 1.0 / avg_dist_reached if avg_dist_reached > 0 else 0.0

        # effective_n excludes v itself from the pivot count, if v happened to be
        # sampled as one of its own pivots (it contributes nothing -- self-distance
        # is skipped above -- so it must not be counted in the averaging denominator)
        effective_n = n_samples - (1 if v in pivots_set else 0)
        harmonic[v] = sum_inv_dist[v] / effective_n if effective_n > 0 else 0.0

    return closeness, harmonic


def compute_classical_features(G, N, betweenness_k=None, closeness_harmonic_samples=None):
    """
    betweenness_k: None for exact betweenness, or an int sample size for approximate.
    closeness_harmonic_samples: None for EXACT closeness/harmonic (fine up to roughly
        Cora/Citeseer/GO-DAG/STRING scale), or an int landmark-sample size for
        approximate versions -- REQUIRED for ogbn-arxiv scale (exact is intractable).
    """
    print("  Computing Degree, PageRank, Betweenness, Eigenvector, Closeness, Katz, Harmonic...")
    degree = dict(G.degree())
    pagerank = nx.pagerank(G)
    betweenness = (nx.betweenness_centrality(G, k=betweenness_k, seed=0) if betweenness_k
                   else nx.betweenness_centrality(G))
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

    if closeness_harmonic_samples is not None:
        print(f"  Using APPROXIMATE closeness/harmonic (landmark samples={closeness_harmonic_samples}) "
              f"-- exact is intractable at this graph size.")
        closeness, harmonic = approximate_closeness_and_harmonic(
            G, N, n_samples=closeness_harmonic_samples, seed=0)
    else:
        closeness = nx.closeness_centrality(G)
        harmonic = nx.harmonic_centrality(G)
        harmonic = {k: v / (N - 1) for k, v in harmonic.items()}  # normalize for comparability

    try:
        katz = nx.katz_centrality(G, alpha=0.1, max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        print("  Katz centrality didn't converge at alpha=0.1, retrying at alpha=0.01...")
        try:
            katz = nx.katz_centrality(G, alpha=0.01, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            print("  [WARN] Katz still didn't converge at alpha=0.01/max_iter=500 -- "
                  "filling with zeros so the rest of the pipeline can proceed. "
                  "Flag this in the paper: Katz centrality is not reported for this dataset "
                  "(power iteration instability, likely due to graph structure at this scale).")
            katz = {i: 0.0 for i in G.nodes()}

    X = np.zeros((N, 7))
    for i in range(N):
        X[i] = [degree[i], pagerank[i], betweenness[i], eigenvector[i],
                closeness[i], katz[i], harmonic[i]]
    return X


def run_reconstruction_experiment(X, y, dataset_name, n_seeds=5, test_frac=0.2, out_dir="."):
    print(f"\n{'='*60}\nRECONSTRUCTION EXPERIMENT: {dataset_name}\n{'='*60}")

    model_results = {"LinearRegression": [], "RandomForest": [], "XGBoost": []}
    best_model_overall, best_r2_overall = None, -np.inf
    last_split = None

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=seed)

        lr = LinearRegression().fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        rf = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        xgb = XGBRegressor(n_estimators=300, max_depth=6, random_state=seed, verbosity=0)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)

        for name, pred in [("LinearRegression", pred_lr), ("RandomForest", pred_rf), ("XGBoost", pred_xgb)]:
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            model_results[name].append({"r2": r2, "mae": mae, "rmse": rmse})

        if model_results["XGBoost"][-1]["r2"] > best_r2_overall:
            best_r2_overall = model_results["XGBoost"][-1]["r2"]
            best_model_overall = xgb
            last_split = (X_test, y_test, pred_xgb)

    summary = {}
    print(f"\n{'Model':<18}{'R2':>10}{'MAE':>10}{'RMSE':>10}")
    for name, runs in model_results.items():
        r2s = [r["r2"] for r in runs]
        maes = [r["mae"] for r in runs]
        rmses = [r["rmse"] for r in runs]
        summary[name] = {"r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                          "mae_mean": np.mean(maes), "rmse_mean": np.mean(rmses)}
        print(f"{name:<18}{np.mean(r2s):>10.4f}{np.mean(maes):>10.4f}{np.mean(rmses):>10.4f}")

    with open(f"{out_dir}/reconstruction_{dataset_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary, best_model_overall, last_split


def run_mutual_information(X, y, dataset_name, out_dir="."):
    print(f"\nMutual information ({dataset_name}):")
    mi = mutual_info_regression(X, y, random_state=0)
    result = {}
    for name, val in zip(FEATURE_NAMES, mi):
        result[name] = float(val)
        print(f"  {name:<15} MI = {val:.4f}")
    with open(f"{out_dir}/mutual_info_{dataset_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_residual_analysis(G, X, y, best_model, dataset_name, node_id_map=None, top_n=20, out_dir="."):
    print(f"\nResidual analysis ({dataset_name}, top {top_n})...")
    pred = best_model.predict(X)
    residuals = y - pred
    top_idx = np.argsort(np.abs(residuals))[::-1][:top_n]

    records = []
    for idx in top_idx:
        node = node_id_map[idx] if node_id_map is not None else idx
        records.append({
            "node": str(node), "true_phi": float(y[idx]), "predicted_phi": float(pred[idx]),
            "residual": float(residuals[idx]), "degree": float(X[idx, 0]), "pagerank": float(X[idx, 1]),
        })
        print(f"  node={node}  true_Phi={y[idx]:.4f}  pred_Phi={pred[idx]:.4f}  "
              f"residual={residuals[idx]:+.4f}  degree={X[idx,0]:.0f}")

    with open(f"{out_dir}/residual_analysis_{dataset_name}.json", "w") as f:
        json.dump(records, f, indent=2)

    # ego-network visualization for the single worst-residual node
    worst_idx = top_idx[0]
    worst_node = node_id_map[worst_idx] if node_id_map is not None else worst_idx
    if worst_node in G:
        ego = nx.ego_graph(G, worst_node, radius=2)
        plt.figure(figsize=(7, 7))
        pos = nx.spring_layout(ego, seed=0)
        colors = ["red" if n == worst_node else "lightgray" for n in ego.nodes()]
        nx.draw_networkx(ego, pos, node_color=colors, node_size=100, with_labels=False, edge_color="gray")
        plt.title(f"{dataset_name}: worst-residual node {worst_node}\n"
                  f"true Phi={y[worst_idx]:.3f}, predicted={pred[worst_idx]:.3f}, "
                  f"degree={X[worst_idx,0]:.0f}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/residual_ego_{dataset_name}.png", dpi=150)
        plt.close()
        print(f"  Saved {out_dir}/residual_ego_{dataset_name}.png")

    return records


def run_feature_importance(best_model, dataset_name, out_dir="."):
    print(f"\nFeature importance ({dataset_name}, XGBoost):")
    importances = best_model.feature_importances_
    result = {}
    for name, val in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        result[name] = float(val)
        print(f"  {name:<15} importance = {val:.4f}")
    with open(f"{out_dir}/feature_importance_{dataset_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    plt.figure(figsize=(7, 5))
    names_sorted = list(result.keys())
    vals_sorted = list(result.values())
    plt.barh(names_sorted[::-1], vals_sorted[::-1])
    plt.xlabel("feature importance"); plt.title(f"{dataset_name}: XGBoost feature importance for Phi")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance_{dataset_name}.png", dpi=150)
    plt.close()
    return result


def run_prediction_scatter(last_split, dataset_name, out_dir="."):
    X_test, y_test, pred_xgb = last_split
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_xgb, s=8, alpha=0.4)
    lims = [min(y_test.min(), pred_xgb.min()), max(y_test.max(), pred_xgb.max())]
    plt.plot(lims, lims, "r--", label="y=x (perfect reconstruction)")
    plt.xlabel("True Phi"); plt.ylabel("Predicted Phi (XGBoost)")
    plt.title(f"{dataset_name}: true vs predicted Phi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction_scatter_{dataset_name}.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir}/prediction_scatter_{dataset_name}.png")


def run_full_reconstruction_battery(G, N, phi, dataset_name, betweenness_k=None,
                                      closeness_harmonic_samples=None, node_id_map=None, out_dir="."):
    """One call runs all five experiments for a given dataset."""
    X = compute_classical_features(G, N, betweenness_k=betweenness_k,
                                     closeness_harmonic_samples=closeness_harmonic_samples)
    y = phi.numpy() if hasattr(phi, "numpy") else np.array(phi)

    summary, best_model, last_split = run_reconstruction_experiment(X, y, dataset_name, out_dir=out_dir)
    run_mutual_information(X, y, dataset_name, out_dir=out_dir)
    run_residual_analysis(G, X, y, best_model, dataset_name, node_id_map=node_id_map, out_dir=out_dir)
    run_feature_importance(best_model, dataset_name, out_dir=out_dir)
    run_prediction_scatter(last_split, dataset_name, out_dir=out_dir)

    return summary
