"""
exact_phi_all.py
================
Exact expected descriptor E[Phi_K(s)] for every source s and K = 1..100 (Proposition 2),
via exact enumeration of the single-trajectory visit probabilities (exact_phi.py).

Writes <dataset>/exact/exact_<tag>.npz with
  phiK [100, N]  E[Phi_K] (normalized), K = 1..100
  nvisK [100, N] expected number of distinct observed vertices E|O_K|
  W [N]          total kernel mass mu(s, V)
  ptr, U, P, wU  CSR over sources: reached vertices, visit probabilities, kernel weights
  pmin [N]       smallest positive visit probability per source
and, for the default parameters, a validation entry in results/exact_validation.json
(agreement with the cached Monte Carlo estimate, frames enumerated, reached vertices).

Usage:
  python analysis/exact_phi_all.py                         # all datasets, delta=1, eps=1/50
  python analysis/exact_phi_all.py --datasets cora --delta 2
  python analysis/exact_phi_all.py --datasets cora --eps 0.01 --cap none
"""
import sys, time, json
import numpy as np
import torch
from scipy.stats import spearmanr

from _common import (ROOT, RESULTS, DISPLAY, cli_datasets, folder, exact_path, CAPS, load_phi)
sys.path.insert(0, str(ROOT))
from exact_phi import cap_csr, all_sources


def arg(flag, default, cast=str):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def lorentz_inner(a, b):
    return -a[..., 0] * b[..., 0] + (a[..., 1:] * b[..., 1:]).sum(-1)


def run(name, eps, delta, cap):
    emb = folder(name) / "hvgae_embedding"
    z = torch.load(emb / "embedding.pt", weights_only=False, map_location="cpu").double()
    c = float(open(emb / "chosen_curvature.txt").read().strip())
    rp, ci, _ = torch.load(emb / "transition_kernel.pt", weights_only=False, map_location="cpu")
    rp, ci = rp.numpy().astype(np.int64), ci.numpy().astype(np.int64)
    N = len(rp) - 1
    zs = z.numpy()
    src = np.repeat(np.arange(N), np.diff(rp))
    d_e = np.sqrt(c) * np.arccosh(np.clip(-lorentz_inner(zs[src], zs[ci]) / c, 1 + 1e-6, None))
    w = np.exp(-delta * d_e)
    rp2, ci2, w2 = cap_csr(rp, ci, w, cap)
    t0 = time.time()
    ptr, U, P, frames, n_over = all_sources(np.arange(N, dtype=np.int64), rp2, ci2, w2,
                                            eps, 2000, N, 200000)
    print(f"  [{name}] enumeration {time.time()-t0:.1f}s, frames mean {frames.mean():.1f} "
          f"max {frames.max()}, overflow {n_over}", flush=True)
    # total kernel mass mu(s, V), blockwise
    zt = torch.from_numpy(zs); J = zt.clone(); J[:, 0] *= -1
    W = np.zeros(N); B = 256 if N > 50000 else 2048
    for a in range(0, N, B):
        blk = (J[a:a + B] @ zt.T).neg_().div_(c).clamp_(min=1 + 1e-6)
        W[a:a + B] = blk.acosh_().mul_(-delta * np.sqrt(c)).exp_().sum(1).numpy()
    srcs = np.repeat(np.arange(N), np.diff(ptr))
    wU = np.exp(-delta * np.sqrt(c) * np.arccosh(
        np.clip(-lorentz_inner(zs[srcs], zs[U]) / c, 1 + 1e-6, None)))
    Ks = np.arange(1, 101)
    logq = np.log1p(-np.clip(P, 0, 1 - 1e-15))
    phiK = np.zeros((len(Ks), N), np.float32); nvisK = np.zeros((len(Ks), N), np.float32)
    for i, K in enumerate(Ks):
        hit = -np.expm1(K * logq)
        o = np.zeros(N); np.add.at(o, srcs, wU * hit); phiK[i] = 1 - o / W
        o = np.zeros(N); np.add.at(o, srcs, hit); nvisK[i] = o
    pmin = np.full(N, np.inf); np.minimum.at(pmin, srcs, P)
    out = exact_path(name, eps, delta, cap)
    np.savez_compressed(out, Ks=Ks, phiK=phiK, nvisK=nvisK, W=W, frames=frames, ptr=ptr,
                        U=U.astype(np.int32), P=P.astype(np.float32), wU=wU.astype(np.float32),
                        pmin=pmin)
    print(f"  [{name}] wrote {out.relative_to(ROOT)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def validate(name):
    """Agreement of the exact descriptor with the cached Monte Carlo estimate."""
    d = np.load(exact_path(name))
    mc, K = load_phi(name)
    ex = d["phiK"][K - 1].astype(np.float64)
    return {"K_star": int(K), "spearman_exact_vs_mc": float(spearmanr(ex, mc)[0]),
            "mean_abs_diff": float(np.abs(ex - mc).mean()),
            "spearman_exact_Kstar_vs_K100": float(spearmanr(ex, d["phiK"][99])[0]),
            "frames_mean": float(d["frames"].mean()), "frames_max": int(d["frames"].max()),
            "reached_mean": float(np.diff(d["ptr"]).mean()),
            "pmin_min": float(d["pmin"][np.isfinite(d["pmin"])].min())}


def main():
    eps = arg("--eps", 1 / 50, float); delta = arg("--delta", 1.0, float)
    cap_arg = arg("--cap", "config")
    for name in cli_datasets(sys.argv):
        cap = CAPS[name] if cap_arg == "config" else (None if cap_arg == "none" else int(cap_arg))
        p = exact_path(name, eps, delta, cap)
        if p.exists() and "--force" not in sys.argv:
            print(f"  [{name}] {p.name} exists (--force to recompute)")
        else:
            run(name, eps, delta, cap)
    if eps == 1 / 50 and delta == 1.0 and cap_arg == "config":
        vpath = RESULTS / "exact_validation.json"
        val = json.load(open(vpath)) if vpath.exists() else {}
        for name in cli_datasets(sys.argv):
            val[name] = validate(name)
            print(f"  [{DISPLAY[name]}] exact vs MC rho = {val[name]['spearman_exact_vs_mc']:.4f}")
        json.dump(val, open(vpath, "w"), indent=2)


if __name__ == "__main__":
    main()
