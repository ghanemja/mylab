"""Boolean operations on polygons with holes and several regions (used by polyshape).

The method is the edge-arrangement scheme behind the Martinez-Rueda-Feito algorithm
(F. Martinez, A. J. Rueda, F. R. Feito, "A new algorithm for computing Boolean operations on
polygons", Computers & Geosciences 35, 2009):

1. every edge of every operand is split at all the points where it meets another edge (proper
   crossings, T-junctions and the ends of collinear overlaps), and points closer than a small
   tolerance are merged, so that two elementary edges meet only at their ends or coincide;
2. coinciding edges are merged into one edge that remembers how much it adds to the winding
   number of each operand (shared edges, opposite edges);
3. for each elementary edge the winding number of every operand just to its left and just to
   its right is computed by casting a ray from its midpoint (the half-open crossing rule keeps
   vertices and vertical edges consistent) -- this is the "in/out" labelling of the algorithm;
4. an edge belongs to the result when the operation's predicate differs on its two sides; it is
   directed so that the result lies on its left;
5. the selected edges are chained into loops; at a vertex where several loops touch, the
   sharpest left turn is taken, so touching regions and holes come out as separate loops.

Loops come back with the inside on the left: outer boundaries counterclockwise, holes clockwise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Loop = np.ndarray  # n-by-2 vertices, not closed


# -- small geometry ---------------------------------------------------------------------------------------------

def signed_area(b: np.ndarray) -> float:
    """Shoelace area: positive for a counterclockwise boundary."""
    if b.shape[0] < 3:
        return 0.0
    x, y = b[:, 0], b[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    x0, y0 = x.mean(), y.mean()  # shift for accuracy
    return 0.5 * float(np.sum((x - x0) * (ys - y0) - (xs - x0) * (y - y0)))


def scale_of(pts: np.ndarray) -> float:
    if not pts.size:
        return 1.0
    span = float(np.max(np.ptp(pts, axis=0)))
    mag = float(np.max(np.abs(pts)))
    return max(span, 1e-3 * mag, 1e-300)


def remove_collinear(b: np.ndarray, rel: float = 1e-10) -> np.ndarray:
    """Drop vertices where the boundary goes straight on (or doubles back on itself)."""
    changed = True
    while changed and b.shape[0] >= 3:
        prev, nxt = np.roll(b, 1, axis=0), np.roll(b, -1, axis=0)
        u, v = b - prev, nxt - b
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        nu, nv = np.hypot(u[:, 0], u[:, 1]), np.hypot(v[:, 0], v[:, 1])
        flat = np.abs(cross) <= rel * nu * nv
        if not flat.any():
            break
        # remove every other flagged vertex at a time, so that runs shrink safely
        idx = np.flatnonzero(flat)
        keep = np.ones(b.shape[0], dtype=bool)
        last = -2
        for i in idx:
            if i != last + 1:
                keep[i] = False
                last = i
        if not keep[0] and not keep[-1] and b.shape[0] > 1:
            keep[-1] = True
        changed = not keep.all()
        b = b[keep]
    return b


def dedupe(b: np.ndarray) -> np.ndarray:
    """Remove consecutive repeated vertices (and a closing copy of the first one)."""
    if b.shape[0] < 2:
        return b
    same = np.all(b[1:] == b[:-1], axis=1)
    b = b[np.concatenate([[True], ~same])]
    while b.shape[0] > 1 and np.all(b[-1] == b[0]):
        b = b[:-1]
    return b


def point_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance from every point (rows of P) to the nearest of the segments A[k]-B[k]."""
    out = np.full(P.shape[0], np.inf)
    if not A.size or not P.size:
        return out
    d = B - A
    L2 = np.einsum("ij,ij->i", d, d)
    L2[L2 == 0] = 1.0
    step = max(1, int(2e6 // max(A.shape[0], 1)))
    for s in range(0, P.shape[0], step):
        p = P[s:s + step, None, :]
        t = np.clip(np.einsum("qkj,kj->qk", p - A[None], d) / L2, 0.0, 1.0)
        q = A[None] + t[..., None] * d[None]
        out[s:s + step] = np.sqrt(np.min(np.sum((p - q) ** 2, axis=2), axis=1))
    return out


def winding(P: np.ndarray, A: np.ndarray, B: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Winding number of each point about the directed edges A[k]->B[k] (weights w, default 1)."""
    out = np.zeros(P.shape[0])
    if not A.size or not P.size:
        return out
    w = np.ones(A.shape[0]) if w is None else w
    ax, ay, bx, by = A[:, 0], A[:, 1], B[:, 0], B[:, 1]
    xmin, xmax = np.minimum(ax, bx), np.maximum(ax, bx)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (by - ay) / (bx - ax)
    sgn = np.where(bx > ax, 1.0, -1.0) * w
    step = max(1, int(2e6 // max(A.shape[0], 1)))
    for s in range(0, P.shape[0], step):
        px, py = P[s:s + step, 0:1], P[s:s + step, 1:2]
        span = (xmin <= px) & (px < xmax)
        with np.errstate(invalid="ignore"):
            yk = ay + (px - ax) * slope
        below = span & (yk < py)
        out[s:s + step] = below @ sgn
    return out


# -- the arrangement ------------------------------------------------------------------------------------------

def _candidate_pairs(A: np.ndarray, B: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    xmin = np.minimum(A[:, 0], B[:, 0]) - tol
    xmax = np.maximum(A[:, 0], B[:, 0]) + tol
    ymin = np.minimum(A[:, 1], B[:, 1]) - tol
    ymax = np.maximum(A[:, 1], B[:, 1]) + tol
    order = np.argsort(xmin, kind="stable")
    xs = xmin[order]
    hi = np.searchsorted(xs, xmax[order], side="right")
    counts = np.maximum(hi - np.arange(n) - 1, 0)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    I = np.repeat(np.arange(n), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    J = np.arange(total) - starts + I + 1
    i, j = order[I], order[J]
    keep = (ymin[i] <= ymax[j]) & (ymin[j] <= ymax[i])
    return i[keep], j[keep]


def _splits(A: np.ndarray, B: np.ndarray, tol: float):
    """Points where segments must be split: (segment, parameter t, x, y) arrays."""
    i, j = _candidate_pairs(A, B, tol)
    segs, ts, pts = [], [], []
    if not i.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros((0, 2))
    P, r = A[i], B[i] - A[i]
    Q, s = A[j], B[j] - A[j]
    rr = np.einsum("ij,ij->i", r, r)
    ss = np.einsum("ij,ij->i", s, s)
    lr, ls = np.sqrt(rr), np.sqrt(ss)

    def on_segment(E, P0, d, dd, ld):
        t = np.einsum("ij,ij->i", E - P0, d) / dd
        dist = np.abs(d[:, 0] * (E - P0)[:, 1] - d[:, 1] * (E - P0)[:, 0]) / ld
        tt = tol / ld
        return t, (dist <= tol) & (t > tt) & (t < 1 - tt)

    for E, P0, d, dd, ld, seg in ((Q, P, r, rr, lr, i), (Q + s, P, r, rr, lr, i),
                                  (P, Q, s, ss, ls, j), (P + r, Q, s, ss, ls, j)):
        t, ok = on_segment(E, P0, d, dd, ld)
        segs.append(seg[ok])
        ts.append(t[ok])
        pts.append(E[ok])
    # proper crossings
    qp = Q - P
    denom = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    o1 = r[:, 0] * qp[:, 1] - r[:, 1] * qp[:, 0]
    o2 = r[:, 0] * (qp + s)[:, 1] - r[:, 1] * (qp + s)[:, 0]
    pq = -qp
    o3 = s[:, 0] * pq[:, 1] - s[:, 1] * pq[:, 0]
    o4 = s[:, 0] * (pq + r)[:, 1] - s[:, 1] * (pq + r)[:, 0]
    cross = (o1 * o2 < 0) & (o3 * o4 < 0) & (denom != 0)
    if cross.any():
        k = np.flatnonzero(cross)
        t = (qp[k, 0] * s[k, 1] - qp[k, 1] * s[k, 0]) / denom[k]
        u = (qp[k, 0] * r[k, 1] - qp[k, 1] * r[k, 0]) / denom[k]
        tt, uu = tol / lr[k], tol / ls[k]
        ok = (t > tt) & (t < 1 - tt) & (u > uu) & (u < 1 - uu)
        k, t, u = k[ok], t[ok], u[ok]
        X = P[k] + t[:, None] * r[k]
        segs += [i[k], j[k]]
        ts += [t, u]
        pts += [X, X]
    return np.concatenate(segs), np.concatenate(ts), np.concatenate(pts)


def _cluster(pts: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge points closer than tol: (vertex id of each point, vertex coordinates)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = pts.shape[0]
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray") if n > 1 else np.zeros((0, 2), dtype=np.int64)
    if pairs.size:
        g = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
    else:
        lab = np.arange(n)
    # number clusters by their first point; that point gives the coordinates
    first = np.full(lab.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, lab, np.arange(n))
    order = np.argsort(first)
    renum = np.empty_like(order)
    renum[order] = np.arange(order.size)
    return renum[lab], pts[first[order]]


def _side_windings(V: np.ndarray, lo: np.ndarray, hi: np.ndarray, M: np.ndarray):
    """Winding numbers (one column per operand) just left and just right of each edge lo->hi."""
    E, G = M.shape
    WL = np.zeros((E, G))
    WR = np.zeros((E, G))
    a, b = V[lo], V[hi]
    d = b - a
    steep = np.abs(d[:, 1]) > np.abs(d[:, 0])
    for swap in (False, True):
        sel = np.flatnonzero(steep == swap)
        if not sel.size:
            continue
        aa, bb = (a[:, ::-1], b[:, ::-1]) if swap else (a, b)
        ax, ay, bx, by = aa[:, 0], aa[:, 1], bb[:, 0], bb[:, 1]
        xmin, xmax = np.minimum(ax, bx), np.maximum(ax, bx)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (by - ay) / (bx - ax)
        sgn = np.where(bx > ax, 1.0, -1.0)
        contrib = sgn[:, None] * M
        below = np.zeros((sel.size, G))
        step = max(1, int(2e6 // max(E, 1)))
        for s0 in range(0, sel.size, step):
            q = sel[s0:s0 + step]
            px = (0.5 * (ax[q] + bx[q]))[:, None]
            py = (0.5 * (ay[q] + by[q]))[:, None]
            span = (xmin <= px) & (px < xmax)
            with np.errstate(invalid="ignore"):
                yk = ay + (px - ax) * slope
            hit = span & (yk < py)
            hit[np.arange(q.size), q] = False
            below[s0:s0 + step] = hit @ contrib
        above = below + contrib[sel]
        right_going = (sgn[sel] > 0)[:, None]
        left = np.where(right_going, above, below)
        right = np.where(right_going, below, above)
        if swap:  # a reflection swaps the sides and the sign of winding numbers
            left, right = -right, -left
        WL[sel] = left
        WR[sel] = right
    return WL, WR


def arrangement(groups: list[list[np.ndarray]], tol: float | None = None):
    """Elementary edges of all boundaries: vertices V, edges lo->hi, multiplicity per group M."""
    A_, B_, G_ = [], [], []
    for g, bounds in enumerate(groups):
        for b in bounds:
            if b.shape[0] < 2:
                continue
            A_.append(b)
            B_.append(np.roll(b, -1, axis=0))
            G_.append(np.full(b.shape[0], g, dtype=np.int64))
    ng = len(groups)
    if not A_:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, ng))
    A, B, G = np.concatenate(A_), np.concatenate(B_), np.concatenate(G_)
    keep = np.any(A != B, axis=1)
    A, B, G = A[keep], B[keep], G[keep]
    if tol is None:
        tol = 1e-10 * scale_of(np.vstack([A, B]) if A.size else A)
    S = A.shape[0]
    sseg, st, spts = _splits(A, B, tol)
    allpts = np.vstack([A, B, spts])
    vid, V = _cluster(allpts, tol)
    seg = np.concatenate([np.arange(S), np.arange(S), sseg])
    tpar = np.concatenate([np.zeros(S), np.ones(S), st])
    order = np.lexsort((tpar, seg))
    seg, vid = seg[order], vid[order]
    same = seg[1:] == seg[:-1]
    u, v = vid[:-1][same], vid[1:][same]
    g = G[seg[:-1][same]]
    ok = u != v
    u, v, g = u[ok], v[ok], g[ok]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    sign = np.where(u < v, 1.0, -1.0)
    nv = V.shape[0]
    key = lo * nv + hi
    uk, inv = np.unique(key, return_inverse=True)
    M = np.zeros((uk.size, ng))
    np.add.at(M, (np.asarray(inv).reshape(-1), g), sign)
    lo, hi = uk // nv, uk % nv
    nz = np.any(M != 0, axis=1)
    return V, lo[nz], hi[nz], M[nz]


def _chain(V: np.ndarray, U: np.ndarray, W: np.ndarray) -> list[Loop]:
    """Join directed edges U->W into loops, taking the sharpest left turn where loops touch."""
    out_of: dict[int, list[int]] = {}
    for k in range(U.size):
        out_of.setdefault(int(U[k]), []).append(k)
    used = np.zeros(U.size, dtype=bool)
    ang = np.arctan2(V[W, 1] - V[U, 1], V[W, 0] - V[U, 0])
    loops = []
    for start in range(U.size):
        if used[start]:
            continue
        seq = []
        cur = start
        guard = 0
        while True:
            guard += 1
            used[cur] = True
            seq.append(int(U[cur]))
            v = int(W[cur])
            cands = out_of.get(v, [])
            back = ang[cur] + np.pi  # direction from v back along the incoming edge
            best, best_turn, best_free, best_free_turn = None, 9.0, None, 9.0
            for c in cands:
                turn = (back - ang[c]) % (2 * np.pi)
                if turn <= 1e-15:
                    turn += 2 * np.pi
                if turn < best_turn:
                    best, best_turn = c, turn
                if not used[c] and turn < best_free_turn:
                    best_free, best_free_turn = c, turn
            if best == start or best_free is None or guard > U.size + 1:
                break
            cur = best_free
        for part in _split_walk(seq):
            if len(part) >= 3:
                loops.append(V[np.array(part)])
    return loops


def _split_walk(seq: list[int]) -> list[list[int]]:
    """Cut a closed walk that passes a vertex more than once into simple loops."""
    out, path, pos = [], [], {}
    for v in seq:
        if v in pos:
            k = pos[v]
            sub = path[k:]
            out.append(sub)
            for x in sub[1:]:
                pos.pop(x, None)
            path = path[:k + 1]
        else:
            pos[v] = len(path)
            path.append(v)
    out.append(path)
    return out


def overlay(groups: list[list[np.ndarray]], predicate: Callable[[np.ndarray], np.ndarray],
            tol: float | None = None) -> list[Loop]:
    """Boundaries of the region where ``predicate`` (of the per-group winding numbers) is true."""
    V, lo, hi, M = arrangement(groups, tol)
    if not lo.size:
        return []
    WL, WR = _side_windings(V, lo, hi, M)
    inL = np.asarray(predicate(np.rint(WL)), dtype=bool)
    inR = np.asarray(predicate(np.rint(WR)), dtype=bool)
    sel = inL != inR
    U = np.where(inL[sel], lo[sel], hi[sel])
    W = np.where(inL[sel], hi[sel], lo[sel])
    return _chain(V, U, W)


def crossings_with_polyline(bounds: list[np.ndarray], line: np.ndarray, tol: float) -> np.ndarray:
    """Split the polyline (rows of ``line``) at the points where it meets the boundaries."""
    if line.shape[0] < 2:
        return line
    A = np.vstack([b for b in bounds]) if bounds else np.zeros((0, 2))
    B = np.vstack([np.roll(b, -1, axis=0) for b in bounds]) if bounds else np.zeros((0, 2))
    LA, LB = line[:-1], line[1:]
    nb = A.shape[0]
    allA, allB = np.vstack([A, LA]), np.vstack([B, LB])
    seg, t, pts = _splits(allA, allB, tol)
    out = []
    for k in range(LA.shape[0]):
        mine = seg == nb + k
        o = np.argsort(t[mine])
        out.append(LA[k:k + 1])
        out.append(pts[mine][o])
    out.append(line[-1:])
    res = np.vstack(out)
    return dedupe_open(res)


def dedupe_open(b: np.ndarray) -> np.ndarray:
    if b.shape[0] < 2:
        return b
    same = np.all(b[1:] == b[:-1], axis=1)
    return b[np.concatenate([[True], ~same])]
