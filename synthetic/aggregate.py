"""aggregate.py
============
Streams results/raw*/<family>/graph_*.json into dose-response tables. Records share
observers, observers share a graph and its embedding, and graphs share a family, so the
effective sample size for a family-level claim is the number of graph instances. Both are
emitted:
    record_level   -- Welford accumulator over records (descriptive)
    cluster_level  -- per-graph-instance means; intervals clustered on instance (reported)
Requires graph_instance_id on each record (falls back to family::graph_seed).

Usage:
    python synthetic/aggregate.py
    python synthetic/aggregate.py --raw results/raw_frozen --out results/summary_frozen.json
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

METRICS = ["Degree", "PageRank", "Betweenness", "Eigenvector", "Closeness",
           "mu", "Phi_normalized", "mu_per_node", "mu_delta2"]
HIDDEN_DESCRIPTORS = ["hidden_nodes", "hidden_edges", "hidden_density",
                      "hidden_clustering", "hidden_components",
                      "hidden_average_degree", "hidden_modularity",
                      "hidden_entropy"]


def arg(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


class RunningStat:
    """Welford. O(1) memory, exact mean/variance."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        if x is None:
            return
        try:
            x = float(x)
        except (TypeError, ValueError):
            return
        if math.isnan(x) or math.isinf(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def finalize(self):
        if self.n == 0:
            return {"mean": None, "std": None, "n": 0}
        if self.n == 1:
            return {"mean": self.mean, "std": 0.0, "n": 1,
                    "ci_low": self.mean, "ci_high": self.mean}
        std = math.sqrt(self.M2 / (self.n - 1))
        se = std / math.sqrt(self.n)
        return {"mean": self.mean, "std": std, "n": self.n,
                "ci_low": self.mean - 1.96 * se,
                "ci_high": self.mean + 1.96 * se}


def cluster_interval(instance_means):
    """
    Interval over graph-instance means. n here is the number of independent
    graphs, which is the correct denominator for a family-level claim.
    """
    vals = [v for v in instance_means if v is not None and not math.isnan(v)]
    k = len(vals)
    if k == 0:
        return {"mean": None, "n_instances": 0}
    m = sum(vals) / k
    if k == 1:
        return {"mean": m, "n_instances": 1, "std": 0.0,
                "ci_low": m, "ci_high": m}
    var = sum((v - m) ** 2 for v in vals) / (k - 1)
    se = math.sqrt(var / k)
    return {"mean": m, "std": math.sqrt(var), "n_instances": k,
            "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se}


def iter_records(raw_dir):
    for fam_dir in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        if not os.path.isdir(fam_dir):
            continue
        family = os.path.basename(fam_dir)
        for fp in sorted(glob.glob(os.path.join(fam_dir, "graph_*.json"))):
            with open(fp) as f:
                recs = json.load(f)
            for r in recs:
                yield family, r
            del recs


def main():
    raw_dir = arg("--raw", str(HERE / "results" / "raw_reembed"))
    if not os.path.isabs(raw_dir):
        raw_dir = str(HERE / raw_dir)
    out_path = arg("--out", str(HERE / "results" / "paper_summary.json"))
    if not os.path.isabs(out_path):
        out_path = str(HERE / out_path)

    if not os.path.isdir(raw_dir):
        raise SystemExit(f"{raw_dir} not found. Run run_dose_response.py first.")

    # record-level accumulators
    rec = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(RunningStat))))
    pooled_rec = defaultdict(lambda: defaultdict(lambda: defaultdict(RunningStat)))
    # per-instance accumulators, for clustered intervals
    inst = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(RunningStat))))

    n_seen = 0
    for family, r in iter_records(raw_dir):
        axis, level = r["axis"], r["level"]
        gid = r.get("graph_instance_id") or f"{family}::{r.get('graph_seed')}"
        n_seen += 1

        def push(key, val):
            rec[family][axis][level][key].update(val)
            pooled_rec[axis][level][key].update(val)
            inst[family][axis][level].setdefault(key, defaultdict(RunningStat))
            inst[family][axis][level][key][gid].update(val)

        for m in METRICS:
            if m in r.get("metrics_after", {}):
                push(f"after_{m}", r["metrics_after"][m])
            if m in r.get("delta_metrics", {}):
                push(f"delta_{m}", r["delta_metrics"][m])
        for d in HIDDEN_DESCRIPTORS:
            if d in r.get("hidden_after", {}):
                push(f"hidden_{d}", r["hidden_after"][d])

        if n_seen % 100_000 == 0:
            print(f"  ...{n_seen:,} records")

    print(f"Total records: {n_seen:,}")

    out = {"n_total_records": n_seen, "raw_dir": raw_dir,
           "per_family": {}, "pooled_record_level": {}}

    cfg_path = Path(raw_dir) / "_run_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            out["run_config"] = json.load(f)

    for family, axes in rec.items():
        out["per_family"][family] = {}
        for axis, levels in axes.items():
            out["per_family"][family][axis] = {}
            for level, metrics in levels.items():
                cell = {}
                for key, stat in metrics.items():
                    means = [s.finalize()["mean"]
                             for s in inst[family][axis][level][key].values()]
                    cell[key] = {
                        "record_level": stat.finalize(),
                        "cluster_level": cluster_interval(means),
                    }
                out["per_family"][family][axis][str(level)] = cell

    for axis, levels in pooled_rec.items():
        out["pooled_record_level"][axis] = {
            str(level): {k: s.finalize() for k, s in metrics.items()}
            for level, metrics in levels.items()
        }

    out["interval_note"] = (
        "Use cluster_level for any claim about a family or an axis. "
        "record_level treats correlated records as independent and its "
        "intervals are optimistic by roughly sqrt(records per instance).")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")

    # quick read on the headline question
    print("\n--- delta_Phi_normalized by axis (cluster-level, max level) ---")
    for family, axes in out["per_family"].items():
        for axis, levels in axes.items():
            keys = sorted(levels, key=lambda s: float(s))
            top = levels[keys[-1]].get("delta_Phi_normalized")
            if top and top["cluster_level"].get("mean") is not None:
                c = top["cluster_level"]
                print(f"  {family:<24} {axis:<18} level={keys[-1]:<6} "
                      f"{c['mean']:+.5f}  [{c['ci_low']:+.5f}, {c['ci_high']:+.5f}]"
                      f"  (n={c['n_instances']} graphs)")


if __name__ == "__main__":
    main()
