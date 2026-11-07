"""Exact evaluation of the walk-based observer descriptor (Proposition 3 of the paper).

The exploration process of the hyperbolic instantiation (``simulate.py`` /
``vectorized_simulate.py``) runs K i.i.d. trajectories from a source s. Each trajectory
  * starts at s (s is recorded as visited);
  * moves to a neighbour of the current vertex other than the previous one (non-backtracking),
    with probability proportional to the edge weight exp(-delta * d_H);
  * stops at a dead end, after ``max_steps`` steps, or when the next step would bring its path
    probability below eps -- in that case the would-be next vertex is NOT recorded;
  * optionally (vectorized simulator only) sees only the ``cap`` highest-weight neighbours of a
    vertex whose degree exceeds ``cap``.

Live paths of a given length are disjoint events of probability >= eps, so there are at most
1/eps of them per length, and every reached vertex has visit probability p_u >= eps. The
enumerator below lists all live paths (breadth first, with parent pointers) and returns the
single-trajectory visit probabilities p_u(s). The expected normalized descriptor is then

    E[Phi_K(s)] = sum_u w_u (1 - p_u)^K / sum_u w_u,   w_u = exp(-delta d(s, u)),

with no Monte Carlo error (Lean: ``Instantiation.expected_muN_coverage``).
"""
import numpy as np
import numba as nb


def cap_csr(row_ptr, col_idx, w, cap):
    N = len(row_ptr) - 1
    deg = np.diff(row_ptr)
    if cap is None or deg.max() <= cap:
        return row_ptr, col_idx, w
    new_rp = np.zeros(N + 1, dtype=np.int64)
    cols, ws = [], []
    for i in range(N):
        a, b = row_ptr[i], row_ptr[i + 1]
        c, ww = col_idx[a:b], w[a:b]
        if b - a > cap:
            top = np.argsort(-ww, kind="stable")[:cap]
            c, ww = c[top], ww[top]
        cols.append(c); ws.append(ww)
        new_rp[i + 1] = new_rp[i] + len(c)
    return new_rp, np.concatenate(cols), np.concatenate(ws)


@nb.njit(cache=True)
def visit_probs_one(s, row_ptr, col_idx, w, eps, max_steps, p_acc, stamp, stamp_ctr,
                    touched, pool_cap):
    # frame pool (BFS order); parent pointers reconstruct paths
    f_cur = np.empty(pool_cap, np.int64)
    f_prev = np.empty(pool_cap, np.int64)
    f_pa = np.empty(pool_cap, np.float64)
    f_par = np.empty(pool_cap, np.int64)
    f_dep = np.empty(pool_cap, np.int64)
    n = 1
    f_cur[0] = s; f_prev[0] = -1; f_pa[0] = 1.0; f_par[0] = -1; f_dep[0] = 0
    n_touched = 0
    overflow = False
    i = 0
    while i < n:
        cur = f_cur[i]; prev = f_prev[i]; pa = f_pa[i]
        term = 0.0
        if f_dep[i] >= max_steps:
            term = pa
        else:
            a = row_ptr[cur]; b = row_ptr[cur + 1]
            tot = 0.0
            for e in range(a, b):
                if col_idx[e] != prev:
                    tot += w[e]
            if tot <= 0.0:
                term = pa
            else:
                for e in range(a, b):
                    v = col_idx[e]
                    if v == prev:
                        continue
                    q = w[e] / tot
                    npa = pa * q
                    if npa < eps:
                        term += npa
                    else:
                        # deterministic-loop check: identical state & PA on the ancestor chain
                        loop = False
                        if q == 1.0:
                            j = i
                            while j >= 0:
                                if f_cur[j] == v and f_prev[j] == cur and f_pa[j] == npa:
                                    loop = True
                                    break
                                j = f_par[j]
                        if loop:
                            term += npa
                            continue
                        if n >= pool_cap:
                            overflow = True
                            term += npa
                            continue
                        f_cur[n] = v; f_prev[n] = cur; f_pa[n] = npa
                        f_par[n] = i; f_dep[n] = f_dep[i] + 1
                        n += 1
        if term > 0.0:
            stamp_ctr += 1
            j = i
            while j >= 0:
                u = f_cur[j]
                if stamp[u] != stamp_ctr:
                    stamp[u] = stamp_ctr
                    if p_acc[u] == 0.0:
                        touched[n_touched] = u
                        n_touched += 1
                    p_acc[u] += term
                j = f_par[j]
        i += 1
    return n_touched, stamp_ctr, n, overflow


@nb.njit(cache=True)
def all_sources(sources, row_ptr, col_idx, w, eps, max_steps, N, pool_cap):
    """Returns CSR over sources: ptr, node ids, p values; plus frame counts."""
    p_acc = np.zeros(N, np.float64)
    stamp = np.zeros(N, np.int64)
    touched = np.empty(N, np.int64)
    stamp_ctr = 0
    out_ptr = np.zeros(len(sources) + 1, np.int64)
    cap_out = 1 << 20
    out_u = np.empty(cap_out, np.int64)
    out_p = np.empty(cap_out, np.float64)
    frames = np.zeros(len(sources), np.int64)
    n_over = 0
    m = 0
    for k in range(len(sources)):
        s = sources[k]
        nt, stamp_ctr, nf, ov = visit_probs_one(s, row_ptr, col_idx, w, eps, max_steps,
                                                p_acc, stamp, stamp_ctr, touched, pool_cap)
        frames[k] = nf
        if ov:
            n_over += 1
        if m + nt > cap_out:
            new_cap = max(cap_out * 2, m + nt)
            nu = np.empty(new_cap, np.int64); nu[:m] = out_u[:m]; out_u = nu
            npp = np.empty(new_cap, np.float64); npp[:m] = out_p[:m]; out_p = npp
            cap_out = new_cap
        for t in range(nt):
            u = touched[t]
            out_u[m] = u
            out_p[m] = min(p_acc[u], 1.0)
            m += 1
            p_acc[u] = 0.0
        out_ptr[k + 1] = m
    return out_ptr, out_u[:m], out_p[:m], frames, n_over


def csr_from_edge_index(edge_index, N):
    src, dst = edge_index
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    rp = np.zeros(N + 1, dtype=np.int64)
    rp[1:] = np.cumsum(np.bincount(src, minlength=N))
    return rp, dst.astype(np.int64), order


def exact_phi_generic(edge_index, N, edge_dist, block_dist, K_list, eps=1/50, delta=1.0,
                      cap=None, max_steps=2000, block=2048):
    """edge_index: int array [2,E] (both directions); edge_dist(src,dst)->[E] distances;
    block_dist(rows)->[len(rows),N] distances. Returns dict K -> Phi_K (normalized, exact)."""
    rp, ci, order = csr_from_edge_index(np.asarray(edge_index), N)
    ed = np.asarray(edge_dist(np.asarray(edge_index)[0][order], np.asarray(edge_index)[1][order]), dtype=np.float64)
    w = np.exp(-delta * ed)
    rp2, ci2, w2 = cap_csr(rp, ci, w, cap)
    ptr, U, P, frames, n_over = all_sources(np.arange(N, dtype=np.int64), rp2, ci2, w2, eps, max_steps, N, 200000)
    srcs = np.repeat(np.arange(N), np.diff(ptr))
    W = np.zeros(N); wU = np.zeros(len(U))
    for a in range(0, N, block):
        rows = np.arange(a, min(N, a + block))
        D = np.asarray(block_dist(rows), dtype=np.float64)
        E = np.exp(-delta * D)
        W[rows] = E.sum(1)
        m = (srcs >= a) & (srcs < a + block)
        wU[m] = E[srcs[m] - a, U[m]]
    logq = np.log1p(-np.clip(P, 0, 1 - 1e-15))
    out = {}
    for K in K_list:
        hit = -np.expm1(K * logq)
        o = np.zeros(N); np.add.at(o, srcs, wU * hit)
        out[K] = 1 - o / W
    return out, dict(W=W, ptr=ptr, U=U, P=P, wU=wU, overflow=n_over)
