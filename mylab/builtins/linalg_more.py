"""More linear algebra: Schur/QZ forms, factor updates, least squares variants, matrix functions,
page-wise operations, test matrices, polynomial matrix helpers and extra iterative solvers.

SciPy is imported inside the functions that need it, so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, as_2d, char, simplify, to_int
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, fnum, is_text, nargs, text

_EPS = float(np.finfo(float).eps)


# -- small helpers ------------------------------------------------------------------------------

def _sla():
    import scipy.linalg as sla
    return sla


def _mat(v: Any, name: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim != 2:
        raise MatlabError(f"{name}: input must be a 2-D matrix.")
    return a


def _sq(v: Any, name: str) -> np.ndarray:
    a = _mat(v, name)
    if a.shape[0] != a.shape[1]:
        raise MatlabError(f"{name}: the matrix must be square.")
    return a


def _dbl(a: np.ndarray) -> np.ndarray:
    return a.astype(complex if a.dtype.kind == "c" else float)


def _single(*arrays) -> bool:
    return any(isinstance(a, np.ndarray) and a.dtype in (np.float32, np.complex64) for a in arrays)


def _fin(r, single: bool = False) -> np.ndarray:
    r = simplify(np.asarray(r))
    if r.ndim < 2:
        r = r.reshape(1, -1) if r.ndim == 1 else r.reshape(1, 1)
    if single and r.dtype.kind in "fc":
        r = r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)
    return r


def _flags(args: list, allowed: tuple, name: str) -> tuple[list, list]:
    rest, flags = [], []
    for a in args:
        if is_text(a):
            s = text(a).lower()
            if s not in allowed:
                raise MatlabError(f"{name}: unknown option '{text(a)}'.")
            flags.append(s)
        else:
            rest.append(a)
    return rest, flags


def _vec(v: Any, name: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim != 2 or (a.size and 1 not in a.shape):
        raise MatlabError(f"{name}: input must be a vector.")
    return a.flatten()


def _scalar_real(v: Any, name: str, what: str) -> float:
    a = fnum(v)
    if a.size != 1 or a.dtype.kind == "c":
        raise MatlabError(f"{name}: {what} must be a real scalar.")
    return float(a.flat[0])


def _finite_check(name: str, *arrays) -> None:
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise MatlabError(f"{name}: input must not contain NaN or Inf.")


# -- reduced row echelon form, adjugate --------------------------------------------------------------

def rref_array(a: np.ndarray, tol: float | None) -> tuple[np.ndarray, list[int]]:
    """Gauss-Jordan elimination with partial pivoting. Returns the reduced matrix and pivot columns."""
    a = _dbl(a).copy()
    m, n = a.shape
    if tol is None:
        tol = max(m, n) * _EPS * (np.abs(a).sum(axis=1).max() if a.size else 0.0)
    pivots: list[int] = []
    i = 0
    for j in range(n):
        if i >= m:
            break
        p = int(np.argmax(np.abs(a[i:, j]))) + i
        if abs(a[p, j]) <= tol:
            a[i:, j] = 0.0
            continue
        if p != i:
            a[[i, p], j:] = a[[p, i], j:]
        a[i, j:] = a[i, j:] / a[i, j]
        for k in range(m):
            if k != i and a[k, j] != 0:
                a[k, j:] = a[k, j:] - a[k, j] * a[i, j:]
        pivots.append(j)
        i += 1
    return a, pivots


@builtin("rref")
def _rref(interp, args, nargout):
    nargs(args, 1, 2, "rref")
    a = _mat(args[0], "rref")
    tol = _scalar_real(args[1], "rref", "the tolerance") if len(args) > 1 else None
    if tol is not None and tol < 0:
        raise MatlabError("rref: the tolerance must be nonnegative.")
    r, piv = rref_array(a, tol)
    r = _fin(r + 0.0, _single(a))
    if nargout > 1:
        return r, np.array([p + 1.0 for p in piv]).reshape(1, -1)
    return r


@builtin("adjoint")
def _adjoint(interp, args, nargout):
    nargs(args, 1, 1, "adjoint")
    a = _dbl(_sq(args[0], "adjoint"))
    n = a.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    if n == 1:
        return np.ones((1, 1))
    # adj(A) = det(A) * inv(A), computed from the SVD so that singular matrices work too:
    # adj(U S V') = det(U V') * V * adj(S) * U'
    u, s, vh = np.linalg.svd(a)
    adj_s = np.array([np.prod(np.delete(s, i)) for i in range(n)])
    scale = np.linalg.det(u) * np.linalg.det(vh)
    return _fin(scale * (vh.conj().T * adj_s) @ u.conj().T)


# -- Schur and QZ forms ------------------------------------------------------------------------------

@builtin("schur")
def _schur(interp, args, nargout):
    nargs(args, 1, 2, "schur")
    if nargout > 2:
        raise MatlabError("schur: too many output arguments.")
    rest, flags = _flags(args, ("real", "complex"), "schur")
    if len(rest) != 1 or len(flags) > 1:
        raise MatlabError("schur: expected a square matrix and at most one of 'real' and 'complex'.")
    a = _sq(rest[0], "schur")
    single = _single(a)
    w = _dbl(a)
    _finite_check("schur", w)
    if w.size == 0:
        return (np.zeros((0, 0)), np.zeros((0, 0))) if nargout > 1 else np.zeros((0, 0))
    mode = "complex" if (w.dtype.kind == "c" or "complex" in flags) else "real"
    t, z = _sla().schur(w, output=mode)
    if mode == "complex":
        t = np.triu(t)
    if nargout <= 1:
        return _fin(t, single)
    return _fin(z, single), _fin(t, single)


def _block_starts(t: np.ndarray) -> list[tuple[int, int]]:
    """(start, size) of the diagonal blocks of a quasi-triangular matrix."""
    n = t.shape[0]
    out = []
    i = 0
    while i < n:
        if i + 1 < n and t[i + 1, i] != 0:
            out.append((i, 2))
            i += 2
        else:
            out.append((i, 1))
            i += 1
    return out


def _ordeig(t: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Eigenvalues in the order in which they appear on the diagonal of a (quasi-)triangular form."""
    n = t.shape[0]
    e = np.zeros(n, dtype=complex)
    for i, size in _block_starts(t):
        if size == 1:
            if b is None:
                e[i] = t[i, i]
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    e[i] = np.inf if b[i, i] == 0 and t[i, i] != 0 else t[i, i] / b[i, i]
        else:
            blk = t[i:i + 2, i:i + 2]
            if b is None:
                ev = np.linalg.eigvals(blk)
            else:
                ev = _sla().eigvals(blk, b[i:i + 2, i:i + 2])
            ev = np.asarray(ev, dtype=complex)
            # the eigenvalue with positive imaginary part first, as LAPACK orders a pair
            ev = ev[np.argsort(-ev.imag)]
            e[i:i + 2] = ev
    return e


@builtin("ordeig")
def _ordeig_fn(interp, args, nargout):
    nargs(args, 1, 2, "ordeig")
    t = _dbl(_sq(args[0], "ordeig"))
    b = None
    if len(args) > 1:
        b = _dbl(_sq(args[1], "ordeig"))
        if b.shape != t.shape:
            raise MatlabError("ordeig: AA and BB must have the same size.")
    return _fin(_ordeig(t, b).reshape(-1, 1), _single(args[0]))


_REGIONS = {
    "lhp": lambda e: e.real < 0,
    "rhp": lambda e: e.real > 0,
    "udi": lambda e: np.abs(e) < 1,
    "udo": lambda e: np.abs(e) > 1,
}


def _selection(sel: Any, e: np.ndarray, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Parse a select argument: (boolean mask, None) or (None, cluster indices)."""
    n = e.size
    if is_text(sel):
        key = text(sel).lower()
        if key not in _REGIONS:
            raise MatlabError(f"{name}: unknown region '{text(sel)}'. Use 'lhp', 'rhp', 'udi' or 'udo'.")
        return _REGIONS[key](e), None
    v = np.asarray(sel)
    if not isinstance(sel, np.ndarray) or v.size != n:
        raise MatlabError(f"{name}: the selection must have one entry per eigenvalue ({n}).")
    flat = v.flatten(order="F")
    if v.dtype == bool:
        return flat.copy(), None
    vals = np.real(fnum(sel)).flatten(order="F")
    if np.all((vals == 0) | (vals == 1)):
        return vals != 0, None
    return None, vals


def _pair_fix(mask: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Both members of a complex pair (a 2-by-2 block) move together."""
    mask = mask.copy()
    for i, size in _block_starts(t):
        if size == 2:
            mask[i] = mask[i + 1] = bool(mask[i] or mask[i + 1])
    return mask


def _trsen(t, u, mask):
    sla = _sla()
    trsen = sla.lapack.get_lapack_funcs("trsen", (t, u))
    res = trsen(mask.astype(np.int32), t, u, job="N")
    if res[-1] not in (0, 1):
        raise MatlabError("ordschur: the reordering failed.")
    if res[-1] == 1:
        raise MatlabError("ordschur: reordering failed because some eigenvalues are too close to separate.")
    return res[0], res[1]


@builtin("ordschur")
def _ordschur(interp, args, nargout):
    nargs(args, 3, 3, "ordschur")
    if nargout > 2:
        raise MatlabError("ordschur: too many output arguments.")
    u = _dbl(_sq(args[0], "ordschur"))
    t = _dbl(_sq(args[1], "ordschur"))
    if u.shape != t.shape:
        raise MatlabError("ordschur: U and T must have the same size.")
    single = _single(args[0], args[1])
    if t.shape[0] == 0:
        return u, t
    if u.dtype != t.dtype:
        u, t = u.astype(complex), t.astype(complex)
    if np.any(np.tril(t, -2) != 0) or (t.dtype.kind == "c" and np.any(np.tril(t, -1) != 0)):
        raise MatlabError("ordschur: T must be a Schur form (upper triangular or quasi-triangular).")
    e = _ordeig(t)
    mask, clusters = _selection(args[2], e, "ordschur")
    if mask is not None:
        ts, us = _trsen(t, u, _pair_fix(mask, t))
    else:
        ts, us = t, u
        labels = clusters.copy()
        for c in np.unique(labels):  # ascending: the largest cluster index ends up first
            for i, size in _block_starts(ts):
                if size == 2 and labels[i] != labels[i + 1]:
                    raise MatlabError("ordschur: both eigenvalues of a complex pair must be in the same cluster.")
            m = labels == c
            ts, us = _trsen(ts, us, m)
            labels = np.concatenate([labels[m], labels[~m]])
    us, ts = _fin(us, single), _fin(ts, single)
    return us, ts


def _qz_mode(flags: list, complex_data: bool) -> str:
    if len(flags) > 1:
        raise MatlabError("qz: give at most one of 'real' and 'complex'.")
    if complex_data or not flags or flags[0] == "complex":
        return "complex"
    return "real"


@builtin("qz")
def _qz(interp, args, nargout):
    nargs(args, 2, 3, "qz")
    if nargout > 6:
        raise MatlabError("qz: too many output arguments.")
    rest, flags = _flags(args, ("real", "complex"), "qz")
    if len(rest) != 2:
        raise MatlabError("qz: expected two square matrices A and B.")
    a, b = _dbl(_sq(rest[0], "qz")), _dbl(_sq(rest[1], "qz"))
    if a.shape != b.shape:
        raise MatlabError("qz: A and B must be square matrices of the same size.")
    single = _single(rest[0], rest[1])
    _finite_check("qz", a, b)
    n = a.shape[0]
    if n == 0:
        z0 = np.zeros((0, 0))
        return tuple([z0] * max(nargout, 1)) if nargout > 1 else z0
    sla = _sla()
    mode = _qz_mode(flags, a.dtype.kind == "c" or b.dtype.kind == "c")
    aa, bb, q, z = sla.qz(a, b, output=mode)  # A = q aa z^H
    if mode == "complex":
        aa, bb = np.triu(aa), np.triu(bb)
    outs = [_fin(aa, single), _fin(bb, single), _fin(q.conj().T, single), _fin(z, single)]
    if nargout > 4:
        with np.errstate(divide="ignore", invalid="ignore"):
            _, wl, vr = sla.eig(a, b, left=True, right=True)
        from .linalg import _ggev_scale
        outs.append(_fin(_ggev_scale(np.asarray(vr, dtype=complex)), single))
        outs.append(_fin(_ggev_scale(np.asarray(wl, dtype=complex)), single))
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


@builtin("ordqz")
def _ordqz(interp, args, nargout):
    nargs(args, 5, 5, "ordqz")
    if nargout > 4:
        raise MatlabError("ordqz: too many output arguments.")
    mats = [_dbl(_sq(x, "ordqz")) for x in args[:4]]
    if len({m.shape for m in mats}) != 1:
        raise MatlabError("ordqz: AA, BB, Q and Z must be square matrices of the same size.")
    single = _single(*args[:4])
    cplx = any(m.dtype.kind == "c" for m in mats)
    if cplx:
        mats = [m.astype(complex) for m in mats]
    aa, bb, q, z = mats
    n = aa.shape[0]
    if n == 0:
        return aa, bb, q, z
    if np.any(np.tril(aa, -2) != 0) or np.any(np.tril(bb, -1) != 0) or (cplx and np.any(np.tril(aa, -1) != 0)):
        raise MatlabError("ordqz: AA and BB must be a QZ form (upper triangular, AA possibly quasi-triangular).")
    sla = _sla()
    tgsen = sla.lapack.get_lapack_funcs("tgsen", (aa, bb))
    ql = q.conj().T  # LAPACK convention: A = Q S Z^H

    def reorder(aa, bb, ql, z, mask):
        res = tgsen(mask.astype(np.int32), aa, bb, ql, z, ijob=0)
        info = res[-1]
        if info != 0:
            raise MatlabError("ordqz: reordering failed because some eigenvalues are too close to separate.")
        if cplx:
            return res[0], res[1], res[4], res[5]
        return res[0], res[1], res[5], res[6]

    e = _ordeig(aa, bb)
    mask, clusters = _selection(args[4], e, "ordqz")
    if mask is not None:
        aa, bb, ql, z = reorder(aa, bb, ql, z, _pair_fix(mask, aa))
    else:
        labels = clusters.copy()
        for c in np.unique(labels):
            for i, size in _block_starts(aa):
                if size == 2 and labels[i] != labels[i + 1]:
                    raise MatlabError("ordqz: both eigenvalues of a complex pair must be in the same cluster.")
            m = labels == c
            aa, bb, ql, z = reorder(aa, bb, ql, z, m)
            labels = np.concatenate([labels[m], labels[~m]])
    outs = (_fin(aa, single), _fin(bb, single), _fin(ql.conj().T, single), _fin(z, single))
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


def _rot(f, g):
    """Plane rotation [c s; -conj(s) c] that maps (f, g) to (r, 0), with c real."""
    if g == 0:
        return 1.0, 0.0
    if f == 0:
        return 0.0, np.conj(g) / abs(g)
    r = np.hypot(abs(f), abs(g))
    c = abs(f) / r
    s = (f / abs(f)) * np.conj(g) / r
    return c, s


@builtin("hess")
def _hess(interp, args, nargout):
    nargs(args, 1, 2, "hess")
    a = _dbl(_sq(args[0], "hess"))
    single = _single(*args)
    _finite_check("hess", a)
    sla = _sla()
    n = a.shape[0]
    if len(args) == 1:
        if nargout > 2:
            raise MatlabError("hess: too many output arguments.")
        if n == 0:
            return (np.zeros((0, 0)), np.zeros((0, 0))) if nargout > 1 else np.zeros((0, 0))
        if nargout <= 1:
            return _fin(sla.hessenberg(a), single)
        h, p = sla.hessenberg(a, calc_q=True)
        return _fin(p, single), _fin(h, single)
    if nargout > 4:
        raise MatlabError("hess: too many output arguments.")
    b = _dbl(_sq(args[1], "hess"))
    if b.shape != a.shape:
        raise MatlabError("hess: A and B must be square matrices of the same size.")
    _finite_check("hess", b)
    # Hessenberg-triangular reduction: Q*A*Z upper Hessenberg, Q*B*Z upper triangular.
    dt = complex if (a.dtype.kind == "c" or b.dtype.kind == "c") else float
    q0, r = sla.qr(b.astype(dt))
    q = q0.conj().T.astype(dt)
    aa = q @ a.astype(dt)
    bb = np.triu(r).astype(dt)
    z = np.eye(n, dtype=dt)
    for j in range(n - 2):
        for i in range(n - 1, j + 1, -1):
            # rotate rows i-1, i to remove aa[i, j]
            x, y = aa[i - 1, j], aa[i, j]
            rho = np.hypot(abs(x), abs(y))
            if rho != 0 and y != 0:
                g = np.array([[np.conj(x), np.conj(y)], [-y, x]], dtype=dt) / rho
                rows = [i - 1, i]
                aa[rows, :] = g @ aa[rows, :]
                bb[rows, :] = g @ bb[rows, :]
                q[rows, :] = g @ q[rows, :]
            aa[i, j] = 0.0
            # rotate columns i-1, i to remove the fill-in bb[i, i-1]
            x, y = bb[i, i - 1], bb[i, i]
            rho = np.hypot(abs(x), abs(y))
            if rho != 0 and x != 0:
                g = np.array([[y, np.conj(x)], [-x, np.conj(y)]], dtype=dt) / rho
                cols = [i - 1, i]
                aa[:, cols] = aa[:, cols] @ g
                bb[:, cols] = bb[:, cols] @ g
                z[:, cols] = z[:, cols] @ g
            bb[i, i - 1] = 0.0
    outs = (_fin(aa, single), _fin(bb, single), _fin(q, single), _fin(z, single))
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


@builtin("balance")
def _balance(interp, args, nargout):
    nargs(args, 1, 2, "balance")
    if nargout > 3:
        raise MatlabError("balance: too many output arguments.")
    a = _dbl(_sq(args[0], "balance"))
    single = _single(args[0])
    permute = True
    if len(args) > 1:
        if not is_text(args[1]) or text(args[1]).lower() != "noperm":
            raise MatlabError("balance: the only option is 'noperm'.")
        permute = False
    _finite_check("balance", a)
    n = a.shape[0]
    if n == 0:
        z = np.zeros((0, 0))
        return (z, z, z)[:nargout] if nargout > 1 else z
    b, t = _sla().matrix_balance(a, permute=permute, scale=True)
    if nargout <= 1:
        return _fin(b, single)
    if nargout == 2:
        return _fin(t, single), _fin(b, single)
    # T(:, P) = diag(S): row k of T has its only nonzero S(k) in column P(k)
    p = np.argmax(np.abs(t) > 0, axis=1)
    s = t[np.arange(n), p]
    return _fin(s.reshape(-1, 1), single), (p + 1.0).reshape(-1, 1), _fin(b, single)


# -- updating factorizations -----------------------------------------------------------------------

@builtin("cholupdate")
def _cholupdate(interp, args, nargout):
    nargs(args, 2, 3, "cholupdate")
    if nargout > 2:
        raise MatlabError("cholupdate: too many output arguments.")
    r = _dbl(_sq(args[0], "cholupdate"))
    single = _single(args[0], args[1])
    x = fnum(args[1])
    n = r.shape[0]
    if x.ndim != 2 or x.shape != (n, 1):
        raise MatlabError("cholupdate: x must be a column vector with one element per column of R.")
    sign = "+"
    if len(args) > 2:
        sign = text(args[2]) if is_text(args[2]) else ""
        if sign not in ("+", "-"):
            raise MatlabError("cholupdate: the third argument must be '+' or '-'.")
    dt = complex if (r.dtype.kind == "c" or x.dtype.kind == "c") else float
    r1 = np.triu(r).astype(dt)  # only the upper triangle is used
    x = x.flatten().astype(dt)
    d = np.diag(r1)
    if n and (np.any(d.imag != 0) or np.any(d.real <= 0) or not np.all(np.isfinite(r1))):
        if nargout > 1:
            return _fin(np.triu(r), single), 2.0
        raise MatlabError("cholupdate: R must be a valid upper triangular Cholesky factor "
                          "(positive real diagonal).")
    if sign == "+":
        row = np.conj(x)  # [R; x'] has the Gram matrix R'*R + x*x'
        for k in range(n):
            rho = np.hypot(r1[k, k].real, abs(row[k]))
            if rho == 0:
                continue
            c = r1[k, k].real / rho
            s = np.conj(row[k]) / rho
            rk = r1[k, k:].copy()
            r1[k, k:] = c * rk + s * row[k:]
            row[k:] = -np.conj(s) * rk + c * row[k:]
            r1[k, k] = rho
        out = _fin(r1, single)
        return (out, 0.0) if nargout > 1 else out
    # downdate: solve R'*p = x, then rotate [p; alpha] onto the last coordinate
    sla = _sla()
    p = sla.solve_triangular(r1, x, trans=2, lower=False, check_finite=False) if n else x
    alpha2 = 1.0 - float(np.real(np.vdot(p, p)))
    if not alpha2 > 0:
        if nargout > 1:
            return _fin(np.triu(r), single), 1.0
        raise MatlabError("Downdated matrix must be positive definite.", "MATLAB:cholupdate:notPosDef")
    alpha = np.sqrt(alpha2)
    extra = np.zeros(n, dtype=dt)
    for k in range(n - 1, -1, -1):
        rho = np.hypot(alpha, abs(p[k]))
        c = alpha / rho
        s = p[k] / rho
        rk = r1[k, k:].copy()
        r1[k, k:] = c * rk - s * extra[k:]
        extra[k:] = np.conj(s) * rk + c * extra[k:]
        alpha = rho
    out = _fin(r1, single)
    return (out, 0.0) if nargout > 1 else out


def _qr_pair(args, name):
    q, r = _dbl(_mat(args[0], name)), _dbl(_mat(args[1], name))
    if q.shape[0] != r.shape[0] and q.shape[1] != r.shape[0]:
        raise MatlabError(f"{name}: Q and R do not have compatible sizes.")
    if q.shape[1] != r.shape[0]:
        raise MatlabError(f"{name}: the number of columns of Q must equal the number of rows of R.")
    if q.dtype != r.dtype:
        q, r = q.astype(complex), r.astype(complex)
    return q, r


@builtin("qrupdate")
def _qrupdate(interp, args, nargout):
    nargs(args, 4, 4, "qrupdate")
    q, r = _qr_pair(args, "qrupdate")
    u, v = fnum(args[2]), fnum(args[3])
    if u.shape != (q.shape[0], 1) or v.shape != (r.shape[1], 1):
        raise MatlabError("qrupdate: u and v must be column vectors that match the rows and columns of A.")
    single = _single(*args)
    if u.dtype.kind == "c" or v.dtype.kind == "c" or q.dtype.kind == "c":
        q, r, u, v = (x.astype(complex) for x in (q, r, u, v))
    q1, r1 = _sla().qr_update(q, r, _dbl(u).flatten(), _dbl(v).flatten())
    return _fin(q1, single), _fin(r1, single)


def _orient(args, pos, name):
    if len(args) > pos:
        o = text(args[pos]).lower() if is_text(args[pos]) else ""
        if o not in ("col", "row"):
            raise MatlabError(f"{name}: the orientation must be 'col' or 'row'.")
        return o
    return "col"


@builtin("qrinsert")
def _qrinsert(interp, args, nargout):
    nargs(args, 4, 5, "qrinsert")
    q, r = _qr_pair(args, "qrinsert")
    j = to_int(args[2], "j")
    x = fnum(args[3])
    which = _orient(args, 4, "qrinsert")
    m, n = r.shape
    single = _single(args[0], args[1], args[3])
    if which == "col":
        if not 1 <= j <= n + 1:
            raise MatlabError("qrinsert: the column index is out of range.")
        if x.shape != (q.shape[0], 1):
            raise MatlabError("qrinsert: x must be a column vector with one element per row of A.")
    else:
        if not 1 <= j <= q.shape[0] + 1:
            raise MatlabError("qrinsert: the row index is out of range.")
        if x.shape != (1, n):
            raise MatlabError("qrinsert: x must be a row vector with one element per column of A.")
    if x.dtype.kind == "c" or q.dtype.kind == "c":
        q, r, x = q.astype(complex), r.astype(complex), x.astype(complex)
    q1, r1 = _sla().qr_insert(q, r, _dbl(x).flatten(), j - 1, which=which)
    return _fin(q1, single), _fin(r1, single)


@builtin("qrdelete")
def _qrdelete(interp, args, nargout):
    nargs(args, 3, 4, "qrdelete")
    q, r = _qr_pair(args, "qrdelete")
    j = to_int(args[2], "j")
    which = _orient(args, 3, "qrdelete")
    limit = r.shape[1] if which == "col" else q.shape[0]
    if not 1 <= j <= limit:
        raise MatlabError("qrdelete: the index is out of range.")
    single = _single(args[0], args[1])
    q1, r1 = _sla().qr_delete(q, r, j - 1, which=which)
    return _fin(q1, single), _fin(r1, single)


@builtin("planerot")
def _planerot(interp, args, nargout):
    nargs(args, 1, 1, "planerot")
    x = fnum(args[0])
    if x.shape != (2, 1):
        raise MatlabError("planerot: x must be a 2-by-1 column vector.")
    single = _single(x)
    x = _dbl(x)
    if x[1, 0] != 0:
        r = np.linalg.norm(x)
        g = np.array([[np.conj(x[0, 0]), np.conj(x[1, 0])], [-x[1, 0], x[0, 0]]]) / r
        y = np.array([[r], [0.0]])
    else:
        g = np.eye(2)
        y = x
    return _fin(g, single), _fin(y, single)


@builtin("givens")
def _givens(interp, args, nargout):
    nargs(args, 2, 2, "givens")
    x, y = fnum(args[0]), fnum(args[1])
    if x.size != 1 or y.size != 1:
        raise MatlabError("givens: x and y must be scalars.")
    f, g = _dbl(x).flat[0], _dbl(y).flat[0]
    if g == 0:
        c, s = 1.0, 0.0
    elif f == 0:
        c, s = 0.0, 1.0
    else:
        r = np.hypot(abs(f), abs(g))
        c = abs(f) / r
        s = (f / abs(f)) * np.conj(g) / r
    if nargout > 1:
        return float(c), _fin(np.array([[s]]))
    return _fin(np.array([[c, s], [-np.conj(s), c]]))


# -- subspaces and least squares -------------------------------------------------------------------

def _orth_basis(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros((a.shape[0], 0))
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    tol = max(a.shape) * np.spacing(s[0]) if s.size else 0.0
    return u[:, : int(np.sum(s > tol))]


@builtin("subspace")
def _subspace(interp, args, nargout):
    nargs(args, 2, 2, "subspace")
    a, b = _dbl(_mat(args[0], "subspace")), _dbl(_mat(args[1], "subspace"))
    if a.shape[0] != b.shape[0]:
        raise MatlabError("subspace: A and B must have the same number of rows.")
    qa, qb = _orth_basis(a), _orth_basis(b)
    if qa.shape[1] < qb.shape[1]:
        qa, qb = qb, qa
    # the largest principal angle, from the part of the smaller space outside the larger one
    resid = qb - qa @ (qa.conj().T @ qb)
    s = np.linalg.norm(resid, 2) if resid.size else 0.0
    return float(np.arcsin(min(1.0, s)))


def _basic_lstsq(a: np.ndarray, b: np.ndarray, interp=None, name: str = "lscov"):
    """Least squares by pivoted QR. Returns x, the rank and (R, permutation) of the factorization."""
    sla = _sla()
    m, n = a.shape
    q, r, p = sla.qr(a, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    tol = max(m, n) * _EPS * (d[0] if d.size else 0.0)
    rank = int(np.sum(d > tol))
    c = q.conj().T @ b
    x = np.zeros((n, b.shape[1]), dtype=np.result_type(r.dtype, c.dtype))
    if rank:
        x[p[:rank], :] = sla.solve_triangular(r[:rank, :rank], c[:rank, :], check_finite=False)
    if rank < n and interp is not None:
        interp.warn(f"Rank deficient, rank = {rank}, tol = {tol:.6e}.", "MATLAB:lscov:RankDefDesignMat")
    return x, rank, r, p


@builtin("lscov")
def _lscov(interp, args, nargout):
    nargs(args, 2, 4, "lscov")
    if nargout > 4:
        raise MatlabError("lscov: too many output arguments.")
    a = _dbl(_mat(args[0], "lscov"))
    b = _dbl(_mat(args[1], "lscov"))
    m, n = a.shape
    if b.shape[0] != m:
        raise MatlabError("lscov: B must have one row per row of A.")
    if m < n:
        raise MatlabError("lscov: A must have at least as many rows as columns.")
    alg = "chol"
    if len(args) > 3:
        alg = text(args[3]).lower() if is_text(args[3]) else ""
        if alg not in ("chol", "orth"):
            raise MatlabError("lscov: the algorithm must be 'chol' or 'orth'.")
    sla = _sla()
    wv = None
    cov = None
    if len(args) > 2 and not (isinstance(args[2], np.ndarray) and args[2].size == 0):
        w = _dbl(fnum(args[2]))
        if w.ndim == 2 and 1 in w.shape and w.size == m and not (m == 1):
            wv = np.real(w).flatten()
            if np.any(wv < 0):
                raise MatlabError("lscov: weights must be nonnegative.")
        elif w.shape == (m, m):
            cov = w
            if not np.allclose(cov, cov.conj().T, rtol=1e-12, atol=0):
                raise MatlabError("lscov: the covariance matrix must be symmetric.")
        elif w.size == 1 and m == 1:
            wv = np.real(w).flatten()
        else:
            raise MatlabError("lscov: the third argument must be a weight vector with one element per "
                              "row of A, or a square covariance matrix of that size.")
    # Transform to an ordinary least-squares problem  At*x ~ bt.
    if wv is not None:
        sw = np.sqrt(wv).reshape(-1, 1)
        at, bt = a * sw, b * sw
    elif cov is not None:
        try:
            low = sla.cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            if alg == "chol":
                raise MatlabError("lscov: the covariance matrix must be positive definite "
                                  "(use the 'orth' algorithm for a semidefinite one).")
            low = None
        if low is not None:
            at = sla.solve_triangular(low, a, lower=True, check_finite=False)
            bt = sla.solve_triangular(low, b, lower=True, check_finite=False)
        else:
            # semidefinite covariance: generalized least squares through its eigen-decomposition
            ev, evec = np.linalg.eigh(cov)
            keep = ev > max(m, 1) * _EPS * max(ev.max(), 0.0)
            if np.any(ev < -max(m, 1) * _EPS * abs(ev).max()):
                raise MatlabError("lscov: the covariance matrix must be positive semidefinite.")
            t = evec[:, keep] / np.sqrt(ev[keep])
            null = evec[:, ~keep]
            # constraints null'*(A*x - b) = 0 are imposed with a large weight
            big = 1.0 / np.sqrt(_EPS)
            at = np.vstack([t.conj().T @ a, big * (null.conj().T @ a)])
            bt = np.vstack([t.conj().T @ b, big * (null.conj().T @ b)])
    else:
        at, bt = a, b
    x, rank, r, p = _basic_lstsq(at, bt, interp)
    outs = [_fin(x, _single(args[0], args[1]))]
    if nargout > 1:
        resid = bt - at @ x
        dof = m - rank
        with np.errstate(divide="ignore", invalid="ignore"):
            mse = np.real(np.sum(np.abs(resid) ** 2, axis=0)) / dof if dof > 0 else np.full(b.shape[1], np.nan)
        # covariance of the estimate: inv(At'*At), from the triangular factor
        s_unscaled = np.zeros((n, n), dtype=x.dtype)
        if rank:
            ri = sla.solve_triangular(r[:rank, :rank], np.eye(rank), check_finite=False)
            s_unscaled[np.ix_(p[:rank], p[:rank])] = ri @ ri.conj().T
        diag = np.real(np.diag(s_unscaled)).reshape(-1, 1)
        stdx = np.sqrt(diag * mse.reshape(1, -1))
        outs.append(_fin(stdx))
        outs.append(_fin(mse.reshape(1, -1)))
        if nargout > 3:
            if b.shape[1] == 1:
                outs.append(_fin(s_unscaled * mse[0]))
            else:
                pages = np.stack([s_unscaled * v for v in mse], axis=2)
                outs.append(as_2d(simplify(pages)))
    return tuple(outs[:max(nargout, 1)]) if nargout > 1 else outs[0]


def lsqminnorm_array(a: np.ndarray, b: np.ndarray, tol: float | None, interp=None, warn: bool = False):
    """Minimum-norm least-squares solution through a complete orthogonal decomposition."""
    sla = _sla()
    m, n = a.shape
    if m == 0 or n == 0:
        return np.zeros((n, b.shape[1])), 0
    q, r, p = sla.qr(a, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    if tol is None:
        tol = max(m, n) * np.spacing(d[0]) if d.size else 0.0
    rank = int(np.sum(d > tol))
    x = np.zeros((n, b.shape[1]), dtype=np.result_type(r.dtype, b.dtype))
    if rank:
        c = (q.conj().T @ b)[:rank, :]
        r1 = r[:rank, :]
        if rank == n:
            y = sla.solve_triangular(r1, c, check_finite=False)
        else:
            # R1 = T' * Z'  (T upper triangular):  y = Z * (T' \ c)  has minimum norm
            z, t = sla.qr(r1.conj().T, mode="economic")
            y = z @ sla.solve_triangular(t.conj().T, c, lower=True, check_finite=False)
        x[p, :] = y
    if warn and rank < min(m, n) and interp is not None:
        interp.warn(f"Rank deficient, rank = {rank}, tol = {tol:.6e}.", "MATLAB:rankDeficientMatrix")
    return x, rank


@builtin("lsqminnorm")
def _lsqminnorm(interp, args, nargout):
    nargs(args, 2, 4, "lsqminnorm")
    rest, flags = _flags(args, ("warn", "nowarn"), "lsqminnorm")
    if len(rest) < 2 or len(rest) > 3:
        raise MatlabError("lsqminnorm: expected A, B and an optional tolerance.")
    a, b = _dbl(_mat(rest[0], "lsqminnorm")), _dbl(_mat(rest[1], "lsqminnorm"))
    if a.shape[0] != b.shape[0]:
        raise MatlabError("lsqminnorm: A and B must have the same number of rows.")
    tol = None
    if len(rest) > 2:
        tol = _scalar_real(rest[2], "lsqminnorm", "the tolerance")
        if tol < 0:
            raise MatlabError("lsqminnorm: the tolerance must be nonnegative.")
    _finite_check("lsqminnorm", a, b)
    x, _ = lsqminnorm_array(a, b, tol, interp, "warn" in flags)
    return _fin(x, _single(rest[0], rest[1]))


# -- decomposition objects ---------------------------------------------------------------------------

_DECOMP_TYPES = ("auto", "qr", "cod", "lu", "ldl", "chol", "triangular", "permutedtriangular", "banded",
                 "hessenberg", "diagonal")
_DECOMP_NAMES = {"permutedtriangular": "permutedTriangular"}


class DecompositionObject(MObject):
    """A stored matrix factorization that solves linear systems with ``\\`` and ``/``."""

    def __init__(self, a: np.ndarray, kind: str, lower: bool | None, check: bool, rank_tol: float | None,
                 sparse: bool, single: bool):
        self.a = a
        self.kind = kind
        self.lower = lower
        self.check = check
        self.rank_tol = rank_tol
        self.sparse = sparse
        self.single = single
        self.transposed = False
        self.scale = 1.0
        self._fac: dict = {}
        self._factor()

    # -- construction ------------------------------------------------------------------------
    @staticmethod
    def pick_type(a: np.ndarray, sparse: bool = False) -> str:
        m, n = a.shape
        if m != n:
            return "qr"
        if n == 0:
            return "triangular"
        if sparse:
            off = a.copy()
            off[np.arange(n), np.arange(n)] = 0
            if not off.any():
                return "diagonal"
        if not np.tril(a, -1).any() or not np.triu(a, 1).any():
            return "triangular"
        try:
            _permuted_triangular(a)
            return "permutedtriangular"
        except MatlabError:
            pass
        if np.array_equal(a, a.conj().T):
            d = np.real(np.diag(a))
            if np.all(d > 0):
                try:
                    _sla().cholesky(a, lower=False)
                    return "chol"
                except np.linalg.LinAlgError:
                    pass
            return "ldl"
        if n > 2 and not np.tril(a, -2).any():
            return "hessenberg"
        return "lu"

    def _factor(self) -> None:
        sla = _sla()
        a, k = self.a, self.kind
        m, n = a.shape
        if k in ("lu", "ldl", "chol", "triangular", "permutedtriangular", "hessenberg", "diagonal", "banded") \
                and m != n:
            raise MatlabError(f"decomposition: type '{_DECOMP_NAMES.get(k, k)}' needs a square matrix.")
        if k in ("lu", "hessenberg", "banded"):
            self._fac["lu"] = sla.lu_factor(a, check_finite=False) if n else None
        elif k == "chol":
            if not np.array_equal(a, a.conj().T):
                tri = np.tril(a) if self.lower else np.triu(a)
                a = tri + tri.conj().T - np.diag(np.real(np.diag(tri)))
                self.a = a
            try:
                self._fac["chol"] = sla.cho_factor(a, lower=bool(self.lower), check_finite=False)
            except np.linalg.LinAlgError:
                raise MatlabError("decomposition: the matrix must be positive definite for type 'chol'.")
        elif k == "ldl":
            if not np.array_equal(a, a.conj().T):
                tri = np.tril(a) if self.lower else np.triu(a)
                a = tri + tri.conj().T - np.diag(np.real(np.diag(tri)))
                self.a = a
            self._fac["lu"] = sla.lu_factor(a, check_finite=False) if n else None
        elif k == "triangular":
            if self.lower is None:
                self.lower = bool(np.tril(a, -1).any() and not np.triu(a, 1).any())
            self.a = np.tril(a) if self.lower else np.triu(a)
        elif k == "permutedtriangular":
            self._fac["perm"] = _permuted_triangular(a)
        elif k == "diagonal":
            self.a = np.diag(np.diag(a)) if n else a
        elif k in ("qr", "cod"):
            q, r, p = sla.qr(a, mode="economic", pivoting=True) if a.size else (a, a, np.arange(n))
            d = np.abs(np.diag(r)) if a.size else np.zeros(0)
            tol = self.rank_tol
            if tol is None:
                tol = max(m, n) * _EPS * (d[0] if d.size else 0.0)
            self._fac["rank"] = int(np.sum(d > tol))
            self._fac["tol"] = tol

    # -- identity ------------------------------------------------------------------------------
    def class_name(self) -> str:
        return "decomposition"

    def property_names(self) -> list[str]:
        return list(self._props())

    def isa_class(self, name: str) -> bool:
        return name == "decomposition"

    def matlab_shape(self) -> tuple:
        return (1, 1)

    def _size(self) -> tuple:
        m, n = self.a.shape
        return (n, m) if self.transposed else (m, n)

    def _props(self) -> dict:
        return {
            "MatrixSize": np.array([[float(d) for d in self._size()]]),
            "Type": char(_DECOMP_NAMES.get(self.kind, self.kind)),
            "CheckCondition": bool_scalar(self.check),
            "Datatype": char("single" if self.single else "double"),
            "IsConjugateTransposed": bool_scalar(self.transposed),
            "IsReal": bool_scalar(self.a.dtype.kind != "c"),
            "IsSparse": bool_scalar(self.sparse),
            "ScaleFactor": np.array([[self.scale]]),
        }

    _METHODS = {"mldivide", "mrdivide", "ctranspose", "transpose", "isIllConditioned", "rank", "rcond",
                "isreal", "issparse", "size", "uminus", "mtimes", "times", "rdivide"}

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def get_field(self, interp, name: str) -> Any:
        props = self._props()
        if name in props:
            return props[name]
        raise MatlabError(f"Unrecognized property '{name}' for class 'decomposition'.")

    def set_field(self, interp, name: str, value: Any) -> "DecompositionObject":
        if name != "CheckCondition":
            raise MatlabError(f"Cannot set read-only property '{name}' of class 'decomposition'.")
        new = self._copy()
        new.check = bool(np.asarray(value).size and np.all(np.asarray(value) != 0))
        return new

    def _copy(self) -> "DecompositionObject":
        new = object.__new__(DecompositionObject)
        new.__dict__.update(self.__dict__)
        return new

    # -- solving -------------------------------------------------------------------------------
    def _rcond(self) -> float:
        a = self.a
        if a.shape[0] != a.shape[1]:
            raise MatlabError("rcond is only available for decompositions of square matrices.")
        if a.size == 0:
            return float("inf")
        if self.kind == "diagonal":
            d = np.abs(np.diag(a))
            return float(d.min() / d.max()) if d.max() > 0 else 0.0
        return float(np.asarray(BUILTINS["rcond"](None, [a], 1)).flat[0])

    def _ill(self) -> bool:
        if self.kind in ("qr", "cod"):
            return self._fac["rank"] < min(self.a.shape)
        return self._rcond() < _EPS

    def _solve(self, interp, b: np.ndarray, transposed: bool) -> np.ndarray:
        """Solve op(A) * X = B, where op is the conjugate transpose when ``transposed``."""
        sla = _sla()
        a, k = self.a, self.kind
        op_rows = a.shape[1] if transposed else a.shape[0]
        if b.shape[0] != op_rows:
            raise MatlabError("Matrix dimensions must agree.")
        if b.dtype.kind not in "fc":
            b = b.astype(float)
        b = _dbl(b)
        if self.check and interp is not None:
            if k in ("qr", "cod"):
                if self._fac["rank"] < min(a.shape):
                    interp.warn(f"Rank deficient, rank = {self._fac['rank']}, tol = {self._fac['tol']:.6e}.",
                                "MATLAB:rankDeficientMatrix")
            elif a.size:
                rc = self._rcond()
                if rc == 0:
                    interp.warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
                elif rc < _EPS:
                    interp.warn("Matrix is close to singular or badly scaled. Results may be inaccurate. "
                                f"RCOND = {rc:e}.", "MATLAB:nearlySingularMatrix")
        with np.errstate(all="ignore"):
            if a.size == 0:
                return np.zeros(((a.shape[0] if transposed else a.shape[1]), b.shape[1]))
            if k in ("lu", "hessenberg", "banded", "ldl"):
                x = sla.lu_solve(self._fac["lu"], b, trans=2 if transposed else 0, check_finite=False)
            elif k == "chol":
                x = sla.cho_solve(self._fac["chol"], b, check_finite=False)
            elif k == "triangular":
                x = sla.solve_triangular(a, b, lower=bool(self.lower), trans=2 if transposed else 0,
                                         check_finite=False)
            elif k == "permutedtriangular":
                rows, cols, lower = self._fac["perm"]
                tri = a[np.ix_(rows, cols)]
                if not transposed:
                    y = sla.solve_triangular(tri, b[rows, :], lower=lower, check_finite=False)
                    x = np.zeros_like(y)
                    x[cols, :] = y
                else:
                    y = sla.solve_triangular(tri, b[cols, :], lower=lower, trans=2, check_finite=False)
                    x = np.zeros_like(y)
                    x[rows, :] = y
            elif k == "diagonal":
                d = np.diag(a).reshape(-1, 1)
                x = b / (np.conj(d) if transposed else d)
            else:
                op = a.conj().T if transposed else a
                if k == "cod":
                    x, _ = lsqminnorm_array(op, b, self.rank_tol)
                else:
                    x = _basic_lstsq(op, b)[0]
        return x

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op == "\\" and left is self:
            if isinstance(right, DecompositionObject):
                raise MatlabError("Both operands of \\ cannot be decomposition objects.")
            b = fnum(right)
            x = self._solve(interp, b, self.transposed) / self.scale
            return _fin(x, self.single or _single(b))
        if op == "/" and right is self:
            if isinstance(left, DecompositionObject):
                raise MatlabError("Both operands of / cannot be decomposition objects.")
            b = fnum(left)
            # X * op(A) = B   <=>   op(A)' * X' = B'
            x = self._solve(interp, b.conj().T, not self.transposed).conj().T / self.scale
            return _fin(x, self.single or _single(b))
        if op in ("*", ".*") and (left is self or right is self):
            other = right if left is self else left
            v = fnum(other)
            if v.size != 1:
                raise MatlabError("A decomposition object can only be multiplied by a scalar.")
            new = self._copy()
            new.scale = self.scale * (complex(v.flat[0]) if v.dtype.kind == "c" else float(v.flat[0]))
            return new
        if op in ("/", "./") and left is self:
            v = fnum(right)
            if v.size != 1:
                raise MatlabError("A decomposition object can only be divided by a scalar.")
            new = self._copy()
            new.scale = self.scale / (complex(v.flat[0]) if v.dtype.kind == "c" else float(v.flat[0]))
            return new
        raise MatlabError(f"Operator '{op}' is not supported for decomposition objects. Use \\ or / to solve "
                          "a linear system.")

    def unary_op(self, interp, op: str) -> Any:
        if op == "'":
            new = self._copy()
            new.transposed = not self.transposed
            new.scale = np.conj(self.scale) if isinstance(self.scale, complex) else self.scale
            return new
        if op == ".'":
            if self.a.dtype.kind == "c" or isinstance(self.scale, complex):
                raise MatlabError("The nonconjugate transpose of a complex decomposition is not supported; "
                                  "use the conjugate transpose (').")
            return self.unary_op(interp, "'")
        if op == "-":
            new = self._copy()
            new.scale = -self.scale
            return new
        if op == "+":
            return self
        return NotImplemented

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "mldivide":
            return [self.binary_op(interp, "\\", args[0], args[1])]
        if name == "mrdivide":
            return [self.binary_op(interp, "/", args[0], args[1])]
        if name in ("mtimes", "times"):
            return [self.binary_op(interp, "*", args[0], args[1])]
        if name == "rdivide":
            return [self.binary_op(interp, "./", args[0], args[1])]
        if name == "ctranspose":
            return [self.unary_op(interp, "'")]
        if name == "transpose":
            return [self.unary_op(interp, ".'")]
        if name == "uminus":
            return [self.unary_op(interp, "-")]
        if name == "isIllConditioned":
            return [bool_scalar(self._ill())]
        if name == "rcond":
            return [np.array([[self._rcond()]])]
        if name == "rank":
            if self.kind not in ("qr", "cod"):
                raise MatlabError("rank is only available for decompositions of type 'qr' and 'cod'.")
            return [np.array([[float(self._fac["rank"])]])]
        if name == "isreal":
            return [bool_scalar(self.a.dtype.kind != "c")]
        if name == "issparse":
            return [bool_scalar(self.sparse)]
        if name == "size":
            sz = self._size()
            if len(args) > 1:
                d = to_int(args[1], "dimension")
                return [np.array([[float(sz[d - 1]) if d <= 2 else 1.0]])]
            if nargout <= 1:
                return [np.array([[float(sz[0]), float(sz[1])]])]
            return [np.array([[float(sz[0])]]), np.array([[float(sz[1])]])][:nargout]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        m, n = self._size()
        return ["  decomposition with properties:", "",
                f"    MatrixSize: [{m} {n}]",
                f"          Type: '{_DECOMP_NAMES.get(self.kind, self.kind)}'", "",
                "  Show all properties"]


def _permuted_triangular(a: np.ndarray):
    """Row and column orders that make ``a`` triangular, or an error if there are none."""
    n = a.shape[0]
    nz = a != 0
    # try to peel off rows with a single remaining nonzero (gives a lower triangular form)
    rows, cols = [], []
    remaining_rows = set(range(n))
    remaining_cols = set(range(n))
    counts = nz.sum(axis=1).astype(int)
    work = nz.copy()
    for _ in range(n):
        pick = next((r for r in sorted(remaining_rows) if counts[r] == 1), None)
        if pick is None:
            break
        c = int(np.argmax(work[pick]))
        rows.append(pick)
        cols.append(c)
        remaining_rows.discard(pick)
        remaining_cols.discard(c)
        for r in remaining_rows:
            if work[r, c]:
                counts[r] -= 1
        work[:, c] = False
    if len(rows) != n:
        raise MatlabError("decomposition: the matrix is not a permuted triangular matrix.")
    return np.array(rows), np.array(cols), True


@builtin("decomposition")
def _decomposition(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for decomposition.")
    src = args[0]
    sparse = hasattr(src, "to_dense")
    a = fnum(src)
    if a.ndim != 2:
        raise MatlabError("decomposition: the input must be a 2-D matrix.")
    single = _single(a)
    a = _dbl(a)
    rest = list(args[1:])
    kind = "auto"
    lower = None
    check = True
    rank_tol = None
    if rest and is_text(rest[0]) and text(rest[0]).lower() in _DECOMP_TYPES:
        kind = text(rest.pop(0)).lower()
    if rest and is_text(rest[0]) and text(rest[0]).lower() in ("upper", "lower"):
        lower = text(rest.pop(0)).lower() == "lower"
    if len(rest) % 2:
        raise MatlabError("decomposition: options must be given as name-value pairs.")
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError("decomposition: option names must be text.")
        name = text(rest[k]).lower()
        val = rest[k + 1]
        if name == "checkcondition":
            check = bool(np.asarray(val).size and np.all(np.asarray(val) != 0))
        elif name == "ranktolerance":
            rank_tol = _scalar_real(val, "decomposition", "RankTolerance")
        elif name in ("banddensity", "ldlpivottolerance", "lupivottolerance"):
            _scalar_real(val, "decomposition", text(rest[k]))  # tuning for sparse factorizations
        else:
            raise MatlabError(f"decomposition: unknown option '{text(rest[k])}'.")
    if not np.all(np.isfinite(a)):
        raise MatlabError("decomposition: the matrix must not contain NaN or Inf.")
    if kind == "auto":
        kind = DecompositionObject.pick_type(a, sparse)
    if lower is not None and kind not in ("triangular", "chol", "ldl"):
        raise MatlabError("decomposition: 'upper' and 'lower' apply to the types 'triangular', 'chol' and 'ldl'.")
    if lower is None and kind in ("chol", "ldl"):
        lower = False
    return DecompositionObject(a, kind, lower, check, rank_tol, sparse, single)


# -- functions of a matrix ---------------------------------------------------------------------------

def _named_derivatives(name: str):
    """f(x, k): the k-th derivative of a named scalar function, or None."""
    from math import factorial
    if name == "exp":
        return lambda x, k: np.exp(x)
    if name == "log":
        return lambda x, k: np.log(x) if k == 0 else (-1.0) ** (k - 1) * factorial(k - 1) * x ** (-float(k))
    if name == "sin":
        return lambda x, k: np.sin(x + (k % 4) * np.pi / 2)
    if name == "cos":
        return lambda x, k: np.cos(x + (k % 4) * np.pi / 2)
    if name == "sinh":
        return lambda x, k: np.sinh(x) if k % 2 == 0 else np.cosh(x)
    if name == "cosh":
        return lambda x, k: np.cosh(x) if k % 2 == 0 else np.sinh(x)
    return None


def _clusters(d: np.ndarray, delta: float) -> np.ndarray:
    """Group eigenvalues: two belong together when a chain of gaps <= delta joins them."""
    n = d.size
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if abs(d[i] - d[j]) <= delta:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    order: dict[int, int] = {}
    labels = np.zeros(n, dtype=int)
    for i, r in enumerate(roots):
        labels[i] = order.setdefault(r, len(order))
    return labels


def _atom(tb: np.ndarray, f, tol: float, maxterms: int) -> tuple[np.ndarray, int, bool]:
    """f of a triangular block with close eigenvalues, by a Taylor series about their mean."""
    from math import factorial
    m = tb.shape[0]

    def fval(x, k):
        return complex(np.asarray(f(x, k)).reshape(-1)[0])

    if m == 1:
        return np.array([[fval(tb[0, 0], 0)]]), 1, True
    sigma = np.trace(tb) / m
    shift = tb - sigma * np.eye(m)
    total = fval(sigma, 0) * np.eye(m, dtype=complex)
    power = np.eye(m, dtype=complex)
    small = 0
    for k in range(1, maxterms + 1):
        power = power @ shift
        if not power.any():  # the shifted block is nilpotent: the series has ended
            return total, k, True
        term = (fval(sigma, k) / float(factorial(k))) * power
        total = total + term
        if np.linalg.norm(term, np.inf) <= tol * np.linalg.norm(total, np.inf):
            small += 1
            if small >= 2:  # two small terms in a row (a single one can be a zero derivative)
                return total, k + 1, True
        else:
            small = 0
    return total, maxterms, False


def funm_array(a: np.ndarray, f, delta: float = 0.1, tol: float = _EPS, maxterms: int = 250):
    """Schur-Parlett evaluation of a matrix function. Returns F, a success flag and details."""
    sla = _sla()
    n = a.shape[0]
    info = {"terms": [], "ind": [], "ord": np.zeros(0), "T": np.zeros((0, 0))}
    if n == 0:
        return np.zeros((0, 0)), True, info
    off = a.copy()
    off[np.arange(n), np.arange(n)] = 0
    if not off.any():
        vals = np.array([complex(np.asarray(f(x, 0)).reshape(-1)[0]) for x in np.diag(a)])
        info.update(terms=[1] * n, ind=[[i] for i in range(n)], ord=np.arange(1, n + 1.0), T=a)
        return np.diag(vals), True, info
    t, u = sla.schur(a.astype(complex), output="complex")
    t = np.triu(t)
    d = np.diag(t).copy()
    labels = _clusters(d, delta)
    info["ord"] = labels + 1.0
    q = int(labels.max()) + 1
    if q > 1 and np.any(np.diff(labels) < 0):
        trsen = sla.lapack.get_lapack_funcs("trsen", (t, u))
        for c in range(q - 1, -1, -1):  # after the loop the clusters appear in the order 0, 1, ...
            mask = labels == c
            res = trsen(mask.astype(np.int32), t, u, job="N")
            t, u = np.triu(res[0]), res[1]
            labels = np.concatenate([labels[mask], labels[~mask]])
    blocks = [np.nonzero(labels == c)[0] for c in range(q)]
    info["ind"] = [list(b) for b in blocks]
    info["T"] = t
    fm = np.zeros((n, n), dtype=complex)
    ok = True
    for j, bj in enumerate(blocks):
        fb, terms, conv = _atom(t[np.ix_(bj, bj)], f, tol, maxterms)
        ok = ok and conv
        info["terms"].append(terms)
        fm[np.ix_(bj, bj)] = fb
        for i in range(j - 1, -1, -1):
            bi = blocks[i]
            rhs = fm[np.ix_(bi, bi)] @ t[np.ix_(bi, bj)] - t[np.ix_(bi, bj)] @ fm[np.ix_(bj, bj)]
            for k in range(i + 1, j):
                bk = blocks[k]
                rhs += fm[np.ix_(bi, bk)] @ t[np.ix_(bk, bj)] - t[np.ix_(bi, bk)] @ fm[np.ix_(bk, bj)]
            fm[np.ix_(bi, bj)] = sla.solve_sylvester(t[np.ix_(bi, bi)], -t[np.ix_(bj, bj)], rhs)
    out = u @ fm @ u.conj().T
    return out, ok, info


@builtin("funm")
def _funm(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for funm.")
    if nargout > 3:
        raise MatlabError("funm: too many output arguments.")
    a = _dbl(_sq(args[0], "funm"))
    single = _single(args[0])
    fun = args[1]
    extra = list(args[3:])
    opts = {"tolblk": 0.1, "toltay": _EPS, "maxterms": 250, "display": "off"}
    if len(args) > 2 and not (isinstance(args[2], np.ndarray) and args[2].size == 0):
        o = args[2]
        if not isinstance(o, Struct):
            raise MatlabError("funm: the options must be a structure (fields TolBlk, TolTay, MaxTerms, Display).")
        for fld in o.fields:
            val = o.get_field(0, fld)
            key = fld.lower()
            if isinstance(val, np.ndarray) and val.size == 0:
                continue
            if key in ("tolblk", "toltay"):
                opts[key] = _scalar_real(val, "funm", fld)
            elif key in ("maxterms", "maxsqrt"):
                opts[key] = to_int(val, fld)
            elif key == "display":
                opts[key] = text(val).lower()
                if opts[key] not in ("off", "on", "verbose"):
                    raise MatlabError("funm: Display must be 'off', 'on' or 'verbose'.")
            elif key == "ord":
                raise MatlabError("funm: a user-specified ordering (the Ord option) is not supported.")
            else:
                raise MatlabError(f"funm: unknown option '{fld}'.")
    named = None
    if isinstance(fun, FunctionHandle) and not fun.is_anonymous and not extra:
        named = _named_derivatives(str(fun.name))
    elif is_text(fun) and not extra:
        named = _named_derivatives(text(fun))
    if named is not None:
        f = named
    else:
        if not isinstance(fun, FunctionHandle) and not is_text(fun):
            raise MatlabError("funm: the second argument must be a function handle.")

        def f(x, k):
            xv = complex(x)
            arg = np.array([[xv]]) if xv.imag != 0 else np.array([[xv.real]])
            res = interp.call_function_value(fun, [arg, np.array([[float(k)]]), *extra], 1)
            return fnum(res[0])
    _finite_check("funm", a)
    out, ok, info = funm_array(a, f, opts["tolblk"], opts["toltay"], opts["maxterms"])
    n = a.shape[0]
    if a.dtype.kind != "c" and n:
        if np.linalg.norm(out.imag, 1) <= 10 * n * _EPS * max(np.linalg.norm(out, 1), np.finfo(float).tiny):
            out = out.real
    out = _fin(out, single)
    flag = 0.0 if ok else 1.0
    if not ok and nargout < 2:
        interp.warn("funm: the Taylor series of a block failed to converge; the result may be inaccurate.",
                    "MATLAB:funm:TaylorNonConvergence")
    if opts["display"] != "off":
        sizes = ", ".join(str(len(b)) for b in info["ind"])
        interp.out(f"funm: {len(info['ind'])} block(s) of size {sizes}; Taylor terms per block: "
                   f"{', '.join(str(t) for t in info['terms'])}\n")
    if nargout <= 1:
        return out
    if nargout == 2:
        return out, flag
    details = Struct.scalar({
        "terms": np.array([float(t) for t in info["terms"]]).reshape(1, -1),
        "ind": Cell.from_list([np.array([float(i + 1) for i in b]).reshape(1, -1) for b in info["ind"]]),
        "ord": np.asarray(info["ord"], dtype=float).reshape(1, -1),
        "T": simplify(np.asarray(info["T"])),
    })
    return out, flag, details


# -- tensor products and page-wise functions -----------------------------------------------------------

@builtin("tensorprod")
def _tensorprod(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for tensorprod.")
    a, b = fnum(args[0]), fnum(args[1])
    rest = list(args[2:])
    nda = None
    if len(rest) >= 2 and is_text(rest[-2]) and text(rest[-2]).lower() == "numdimensionsa":
        nda = to_int(rest[-1], "NumDimensionsA")
        rest = rest[:-2]
    dim_a: list[int] = []
    dim_b: list[int] = []
    if len(rest) == 1 and is_text(rest[0]):
        if text(rest[0]).lower() != "all":
            raise MatlabError("tensorprod: the dimension must be numeric or \"all\".")
        if a.shape != b.shape:
            raise MatlabError("tensorprod: A and B must have the same size for the \"all\" option.")
        return _fin(np.asarray(np.sum(a * b)).reshape(1, 1))
    if len(rest) == 1:
        dim_a = dim_b = [int(d) for d in np.real(fnum(rest[0])).flatten()]
    elif len(rest) == 2:
        dim_a = [int(d) for d in np.real(fnum(rest[0])).flatten()]
        dim_b = [int(d) for d in np.real(fnum(rest[1])).flatten()]
    elif len(rest) > 2:
        raise MatlabError("Too many input arguments for tensorprod.")
    if len(dim_a) != len(dim_b):
        raise MatlabError("tensorprod: dimA and dimB must have the same number of elements.")
    for dims in (dim_a, dim_b):
        if any(d < 1 for d in dims) or len(set(dims)) != len(dims):
            raise MatlabError("tensorprod: dimensions must be unique positive integers.")
    na = max([a.ndim] + dim_a)
    if nda is not None:
        if nda < a.ndim and any(s != 1 for s in a.shape[nda:]):
            raise MatlabError("tensorprod: NumDimensionsA is smaller than the number of dimensions of A.")
        if dim_a and max(dim_a) > nda:
            raise MatlabError("tensorprod: dimA exceeds NumDimensionsA.")
        na = nda
    nb = max([b.ndim] + dim_b)
    a2 = a.reshape(tuple(a.shape[:na]) + (1,) * (na - a.ndim))
    b2 = b.reshape(tuple(b.shape) + (1,) * (nb - b.ndim))
    ax_a = [d - 1 for d in dim_a]
    ax_b = [d - 1 for d in dim_b]
    for x, y in zip(ax_a, ax_b):
        if a2.shape[x] != b2.shape[y]:
            raise MatlabError("tensorprod: the contracted dimensions of A and B must have the same length.")
    r = np.tensordot(a2, b2, axes=(ax_a, ax_b))
    if r.ndim == 0:
        r = r.reshape(1, 1)
    elif r.ndim == 1:
        r = r.reshape(-1, 1)
    return as_2d(simplify(np.asarray(r)))


def _page_pair(arrs: list) -> tuple[list, tuple]:
    """Expand the page dimensions (3 and up) of several arrays to a common shape.

    Returns each array as (rows, cols, npages) and the common page shape.
    """
    nd = max(a.ndim for a in arrs)
    shapes = [tuple(a.shape[2:]) + (1,) * (nd - a.ndim) for a in arrs]
    try:
        common = np.broadcast_shapes(*shapes) if shapes else ()
    except ValueError:
        raise MatlabError("Page dimensions (dimension 3 and higher) must match or be 1.")
    out = []
    for a, s in zip(arrs, shapes):
        x = np.broadcast_to(a.reshape(a.shape[:2] + s), a.shape[:2] + common)
        out.append(x.reshape(a.shape[:2] + (-1,), order="F"))
    return out, tuple(common)


def _stack_pages(pages: list, common: tuple) -> np.ndarray:
    out = np.stack(pages, axis=2)
    return as_2d(out.reshape(out.shape[:2] + tuple(common), order="F"))


def _pages_of(v: Any) -> np.ndarray:
    a = fnum(v)
    return np.atleast_2d(a) if a.ndim < 2 else a


@builtin("pagetranspose")
def _pagetranspose(interp, args, nargout):
    nargs(args, 1, 1, "pagetranspose")
    a = args[0] if isinstance(args[0], np.ndarray) else fnum(args[0])
    return np.swapaxes(a, 0, 1).copy()


@builtin("pagectranspose")
def _pagectranspose(interp, args, nargout):
    nargs(args, 1, 1, "pagectranspose")
    a = args[0] if isinstance(args[0], np.ndarray) else fnum(args[0])
    t = np.swapaxes(a, 0, 1)
    return np.conj(t) if t.dtype.kind == "c" else t.copy()


@builtin("pageinv")
def _pageinv(interp, args, nargout):
    nargs(args, 1, 1, "pageinv")
    a = _pages_of(args[0])
    if a.shape[0] != a.shape[1]:
        raise MatlabError("pageinv: each page must be a square matrix.")
    (x,), common = _page_pair([a])
    inv = BUILTINS["inv"]
    pages = [np.asarray(inv(interp, [np.ascontiguousarray(x[:, :, k])], 1)) for k in range(x.shape[2])]
    if not pages:
        return a.copy()
    if any(p.dtype.kind == "c" for p in pages):
        pages = [p.astype(complex) for p in pages]
    return _stack_pages(pages, common)


def _page_solve(a, b, name, right: bool):
    from ..ops import left_divide, right_divide
    (x, y), common = _page_pair([a, b])
    if right:
        if x.shape[1] != y.shape[1]:
            raise MatlabError(f"{name}: the pages of B and A must have the same number of columns.")
    elif x.shape[0] != y.shape[0]:
        raise MatlabError(f"{name}: the pages of A and B must have the same number of rows.")
    pages = []
    for k in range(x.shape[2]):
        p, q = np.ascontiguousarray(x[:, :, k]), np.ascontiguousarray(y[:, :, k])
        pages.append(np.asarray(right_divide(q, p) if right else left_divide(p, q)))
    if not pages:
        shape = (y.shape[0], x.shape[0]) if right else (x.shape[1], y.shape[1])
        return np.zeros(shape + tuple(common))
    if any(p.dtype.kind == "c" for p in pages):
        pages = [p.astype(complex) for p in pages]
    return _stack_pages(pages, common)


@builtin("pagemldivide")
def _pagemldivide(interp, args, nargout):
    nargs(args, 2, 2, "pagemldivide")
    return _page_solve(_pages_of(args[0]), _pages_of(args[1]), "pagemldivide", False)


@builtin("pagemrdivide")
def _pagemrdivide(interp, args, nargout):
    nargs(args, 2, 2, "pagemrdivide")
    # B / A  page by page
    return _page_solve(_pages_of(args[1]), _pages_of(args[0]), "pagemrdivide", True)


@builtin("pagenorm")
def _pagenorm(interp, args, nargout):
    nargs(args, 1, 2, "pagenorm")
    a = _pages_of(args[0])
    single = _single(a)
    p: Any = 2
    if len(args) > 1:
        if is_text(args[1]):
            s = text(args[1]).lower()
            if s == "fro":
                p = "fro"
            elif s == "inf":
                p = np.inf
            else:
                raise MatlabError("pagenorm: the norm type must be 1, 2, Inf or \"fro\".")
        else:
            pv = fnum(args[1])
            if pv.size != 1 or pv.flat[0] not in (1, 2, np.inf):
                raise MatlabError("pagenorm: the norm type must be 1, 2, Inf or \"fro\".")
            p = np.inf if np.isinf(pv.flat[0]) else int(pv.flat[0])
    (x,), common = _page_pair([_dbl(a)])
    vals = []
    for k in range(x.shape[2]):
        pg = x[:, :, k]
        if pg.size == 0:
            vals.append(0.0)
        elif np.isnan(pg).any():
            vals.append(np.nan)
        else:
            vals.append(float(np.linalg.norm(pg, p)))
    out = np.array(vals, dtype=np.float32 if single else float).reshape((1, 1) + tuple(common), order="F")
    return as_2d(out)


def _per_page(interp, fname: str, a: np.ndarray, opts: list, nargout: int):
    (x,), common = _page_pair([a])
    fn = BUILTINS[fname]
    outs: list[list] = [[] for _ in range(max(nargout, 1))]
    for k in range(x.shape[2]):
        res = fn(interp, [np.ascontiguousarray(x[:, :, k]), *opts], nargout)
        res = res if isinstance(res, tuple) else (res,)
        for lst, r in zip(outs, res):
            lst.append(np.asarray(r))
    if not outs[0]:
        raise MatlabError(f"page{fname}: the input must have at least one page.")
    fixed = []
    for lst in outs:
        if any(p.dtype.kind == "c" for p in lst):
            lst = [p.astype(complex) for p in lst]
        fixed.append(_stack_pages(lst, common))
    return tuple(fixed) if nargout > 1 else fixed[0]


@builtin("pagesvd")
def _pagesvd(interp, args, nargout):
    nargs(args, 1, 3, "pagesvd")
    if nargout > 3:
        raise MatlabError("pagesvd: too many output arguments.")
    a = _pages_of(args[0])
    opts = list(args[1:])
    if not any(is_text(o) and text(o).lower() in ("vector", "matrix") for o in opts):
        opts.append(char("vector" if nargout <= 1 else "matrix"))
    return _per_page(interp, "svd", a, opts, nargout)


@builtin("pageeig")
def _pageeig(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for pageeig.")
    if nargout > 3:
        raise MatlabError("pageeig: too many output arguments.")
    a = _pages_of(args[0])
    if a.shape[0] != a.shape[1]:
        raise MatlabError("pageeig: each page must be a square matrix.")
    return _per_page(interp, "eig", a, list(args[1:]), nargout)


@builtin("pagepinv")
def _pagepinv(interp, args, nargout):
    nargs(args, 1, 2, "pagepinv")
    a = _pages_of(args[0])
    if a.size == 0 or (a.ndim > 2 and 0 in a.shape[2:]):
        return np.zeros((a.shape[1], a.shape[0]) + tuple(a.shape[2:]))
    return _per_page(interp, "pinv", a, list(args[1:]), 1)


@builtin("pagelsqminnorm")
def _pagelsqminnorm(interp, args, nargout):
    nargs(args, 2, 3, "pagelsqminnorm")
    a, b = _pages_of(args[0]), _pages_of(args[1])
    tol = _scalar_real(args[2], "pagelsqminnorm", "the tolerance") if len(args) > 2 else None
    (x, y), common = _page_pair([_dbl(a), _dbl(b)])
    if x.shape[0] != y.shape[0]:
        raise MatlabError("pagelsqminnorm: the pages of A and B must have the same number of rows.")
    pages = [lsqminnorm_array(np.ascontiguousarray(x[:, :, k]), np.ascontiguousarray(y[:, :, k]), tol)[0]
             for k in range(x.shape[2])]
    if not pages:
        return np.zeros((x.shape[1], y.shape[1]) + tuple(common))
    if any(p.dtype.kind == "c" for p in pages):
        pages = [p.astype(complex) for p in pages]
    return simplify(_stack_pages(pages, common))


# -- test matrices -----------------------------------------------------------------------------------

def _bucky_geometry():
    """Vertices and edges of the truncated icosahedron (C60): the even permutations of
    (0, +-1, +-3p), (+-1, +-(2+p), +-2p) and (+-p, +-2, +-(2p+1)), p the golden ratio; edges join
    the vertices at the smallest distance (2)."""
    import itertools
    phi = (1 + 5 ** 0.5) / 2
    base = [(0.0, 1.0, 3 * phi), (1.0, 2 + phi, 2 * phi), (phi, 2.0, 2 * phi + 1)]
    even = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    pts = set()
    for b in base:
        for perm in even:
            v = [b[i] for i in perm]
            for signs in itertools.product((1, -1), repeat=3):
                pts.add(tuple(s * x + 0.0 for s, x in zip(signs, v)))  # exact; + 0.0 makes -0 into 0
    V = np.array(sorted(pts, key=lambda q: (-q[2], np.arctan2(q[1], q[0]))))
    d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    A = (np.abs(d - 2.0) < 1e-9).astype(float)
    return A, V / np.linalg.norm(V[0])


@builtin("bucky")
def _bucky(interp, args, nargout):
    """[B, V] = bucky: the graph of the C60 molecule (a soccer ball)."""
    if args:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    from .sparse import SparseMatrix
    import scipy.sparse as sp
    A, V = _bucky_geometry()
    B = SparseMatrix(sp.csc_matrix(A))
    return (B, V) if nargout > 1 else B


@builtin("gallery")
def _gallery(interp, args, nargout):
    if not args:
        raise MatlabError("gallery: expected the name of a matrix family, for example gallery('minij', 5).")
    from . import _gallery as g
    first = args[0]
    if not is_text(first):
        # gallery(3) and gallery(5): two small fixed test matrices
        n = to_int(first, "n")
        if n == 3:
            return np.array([[-149.0, -50.0, -154.0], [537.0, 180.0, 546.0], [-27.0, -9.0, -25.0]])
        if n == 5:
            return np.array([[-9.0, 11.0, -21.0, 63.0, -252.0],
                             [70.0, -69.0, 141.0, -421.0, 1684.0],
                             [-575.0, 575.0, -1149.0, 3451.0, -13801.0],
                             [3891.0, -3891.0, 7782.0, -23345.0, 93365.0],
                             [1024.0, -1024.0, 2048.0, -6144.0, 24572.0]])
        raise MatlabError("gallery(n) is defined only for n = 3 and n = 5.")
    name = text(first).lower()
    fam = g.FAMILIES.get(name)
    if fam is None:
        raise MatlabError(f"gallery: unknown matrix family '{text(first)}'.")
    rest = list(args[1:])
    cls = None
    if name not in ("integerdata", "uniformdata", "normaldata") and rest and is_text(rest[-1]):
        cls = text(rest.pop())
        if cls not in ("double", "single"):
            raise MatlabError("gallery: the class must be 'double' or 'single'.")
    if not rest:
        raise MatlabError(f"gallery('{name}', ...): not enough input arguments.")
    out = fam(interp, rest, nargout) if name in g.MULTI_OUTPUT else fam(interp, rest)

    def conv(x):
        if isinstance(x, np.ndarray) and x.dtype.kind in "fc":
            x = simplify(x)
            if cls == "single":
                x = x.astype(np.complex64 if x.dtype.kind == "c" else np.float32)
        return x

    if isinstance(out, tuple):
        return tuple(conv(x) for x in out)
    return conv(out)


@builtin("hankel")
def _hankel(interp, args, nargout):
    nargs(args, 1, 2, "hankel")
    c = fnum(args[0]).flatten(order="F")
    n = c.size
    i, j = np.indices((n, n)) if len(args) == 1 else (None, None)
    if len(args) == 1:
        k = i + j
        return np.where(k < n, c[np.minimum(k, n - 1)], 0) if n else np.zeros((0, 0))
    r = fnum(args[1]).flatten(order="F")
    m = r.size
    if n and m and c[-1] != r[0]:
        interp.warn("Last element of input column does not match first element of input row. "
                    "Column wins anti-diagonal conflict.", "MATLAB:hankel:AntiDiagonalConflict")
    both = np.concatenate([c, r[1:]])
    i, j = np.indices((n, m))
    return both[i + j] if n and m else np.zeros((n, m))


@builtin("rosser")
def _rosser(interp, args, nargout):
    nargs(args, 0, 1, "rosser")
    r = np.array([[611, 196, -192, 407, -8, -52, -49, 29],
                  [196, 899, 113, -192, -71, -43, -8, -44],
                  [-192, 113, 899, 196, 61, 49, 8, 52],
                  [407, -192, 196, 611, 8, 44, 59, -23],
                  [-8, -71, 61, 8, 411, -599, 208, 208],
                  [-52, -43, 49, 44, -599, 411, 208, 208],
                  [-49, -8, 8, 59, 208, 208, 99, -911],
                  [29, -44, 52, -23, 208, 208, -911, 99]], dtype=float)
    if args:
        cls = text(args[0])
        if cls == "single":
            return r.astype(np.float32)
        if cls != "double":
            raise MatlabError("rosser: the class must be 'double' or 'single'.")
    return r


@builtin("wilkinson")
def _wilkinson(interp, args, nargout):
    nargs(args, 1, 2, "wilkinson")
    n = to_int(args[0], "n")
    if n < 0:
        raise MatlabError("wilkinson: n must be a nonnegative integer.")
    m = (n - 1) / 2.0
    w = np.diag(np.abs(np.arange(-m, m + 1))) if n else np.zeros((0, 0))
    if n > 1:
        w = w + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    if len(args) > 1:
        cls = text(args[1])
        if cls == "single":
            return w.astype(np.float32)
        if cls != "double":
            raise MatlabError("wilkinson: the class must be 'double' or 'single'.")
    return w


def _paley(q: int, backward: bool = False) -> np.ndarray:
    """Normalized Hadamard matrix of order q + 1 for a prime q = 3 (mod 4).

    The core is a circulant of quadratic-residue signs, or (``backward``) the symmetric
    anti-circulant arrangement of the same sequence.
    """
    residues = {(x * x) % q for x in range(1, q)}
    chi = np.array([-1.0] + [1.0 if k in residues else -1.0 for k in range(1, q)])
    i, j = np.indices((q, q))
    h = np.ones((q + 1, q + 1))
    h[1:, 1:] = chi[(i + j + 2) % q] if backward else chi[(i - j) % q]
    return h


@builtin("hadamard")
def _hadamard(interp, args, nargout):
    nargs(args, 1, 2, "hadamard")
    n = to_int(args[0], "n")
    base = None
    makers = ((1, lambda: np.ones((1, 1))), (12, lambda: _paley(11)), (20, lambda: _paley(19, backward=True)))
    for start, make in makers:
        k = n / start if n > 0 else 0
        if k >= 1 and k == int(k) and (int(k) & (int(k) - 1)) == 0:
            base, e = make(), int(k).bit_length() - 1
            break
    if base is None:
        raise MatlabError("hadamard: n must be a power of 2, or 12 or 20 times a power of 2.")
    h = base
    for _ in range(e):
        h = np.block([[h, h], [h, -h]])
    if len(args) > 1:
        cls = text(args[1])
        if cls == "single":
            return h.astype(np.float32)
        if cls != "double":
            raise MatlabError("hadamard: the class must be 'double' or 'single'.")
    return h


@builtin("condeig")
def _condeig(interp, args, nargout):
    nargs(args, 1, 1, "condeig")
    if nargout > 3:
        raise MatlabError("condeig: too many output arguments.")
    a = _dbl(_sq(args[0], "condeig"))
    single = _single(args[0])
    _finite_check("condeig", a)
    n = a.shape[0]
    if n == 0:
        z = np.zeros((0, 0))
        return (z, z, np.zeros((0, 1))) if nargout > 1 else np.zeros((0, 1))
    if np.array_equal(a, a.conj().T):
        w, v = np.linalg.eigh(a)
        s = np.ones(n)
    else:
        w, vl, v = _sla().eig(a, left=True, right=True)
        vl = vl / np.linalg.norm(vl, axis=0)
        v = v / np.linalg.norm(v, axis=0)
        with np.errstate(divide="ignore"):
            s = 1.0 / np.abs(np.sum(np.conj(vl) * v, axis=0))
    s = s.reshape(-1, 1)
    if nargout <= 1:
        return _fin(s, single)
    return _fin(v, single), _fin(np.diag(w), single), _fin(s, single)


def _normest1_core(interp, matmat, rmatmat, n: int, t: int, x0, real: bool):
    """Block 1-norm estimator (Higham and Tisseur). Returns est, v, w, iterations, products."""
    rand = BUILTINS["rand"]

    def signs(rows, cols):
        u = np.asarray(rand(interp, [np.array([[float(rows)]]), np.array([[float(cols)]])], 1))
        return np.where(u < 0.5, -1.0, 1.0)

    def parallel(col, mat):
        # +-1 vectors are parallel exactly when |col' * other| = n
        return mat.shape[1] > 0 and bool(np.any(np.abs(col @ mat) == n))

    if x0 is None:
        x = np.ones((n, t))
        if t > 1:
            x[:, 1:] = signs(n, t - 1)
            for j in range(1, t):
                tries = 0
                while parallel(x[:, j], x[:, :j]) and tries < 10:
                    x[:, j] = signs(n, 1).flatten()
                    tries += 1
        x = x / n
    else:
        x = x0
    itmax = 5
    ind_hist: list[int] = []
    est_old = 0.0
    ind = np.zeros(n, dtype=int)
    s = np.zeros((n, t))
    ind_best = 0
    w = np.zeros(n)
    k = 0
    nprod = 0
    est = 0.0
    while True:
        k += 1
        y = matmat(x)
        nprod += 1
        norms = np.sum(np.abs(y), axis=0)
        j = int(np.argmax(norms))
        est = float(norms[j])
        if est > est_old or k == 2:
            if k >= 2:
                ind_best = int(ind[j])
            w = y[:, j].copy()
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        s_old = s
        if k > itmax:
            break
        if real:
            s = np.where(y >= 0, 1.0, -1.0)
            if k >= 2 and all(parallel(s[:, c], s_old) for c in range(s.shape[1])):
                break
            if t > 1:
                for c in range(s.shape[1]):
                    tries = 0
                    while (parallel(s[:, c], s[:, :c]) or (k >= 2 and parallel(s[:, c], s_old))) and tries < 10:
                        s[:, c] = signs(n, 1).flatten()
                        tries += 1
        else:
            mag = np.abs(y)
            s = np.where(mag == 0, 1.0, y / np.where(mag == 0, 1.0, mag))
        z = rmatmat(s)
        nprod += 1
        h = np.max(np.abs(z), axis=1)
        if k >= 2 and h.max() == h[ind_best]:
            break
        ind = np.argsort(-h, kind="stable")
        if t > 1:
            if all(int(i) in ind_hist for i in ind[:t]):
                break
            fresh = [int(i) for i in ind if int(i) not in ind_hist]
            seen = [int(i) for i in ind if int(i) in ind_hist]
            ind = np.array(fresh + seen, dtype=int)
        cols = min(t, n)
        x = np.zeros((n, cols))
        x[ind[:cols], np.arange(cols)] = 1.0
        ind_hist.extend(int(i) for i in ind[:cols])
    v = np.zeros(n)
    v[ind_best] = 1.0
    return est, v, w, k, nprod


@builtin("normest1")
def _normest1(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for normest1.")
    if nargout > 4:
        raise MatlabError("normest1: too many output arguments.")
    src = args[0]
    extra = list(args[3:])
    if isinstance(src, FunctionHandle) or is_text(src):
        def call(flag, x=None):
            a = [char(flag)] + ([x] if x is not None else [np.zeros((0, 0))]) + extra
            return interp.call_function_value(src, a, 1)[0]
        n = to_int(call("dim"), "the dimension returned by afun('dim')")
        real = bool(np.all(fnum(call("real")) != 0))

        def matmat(x):
            return _dbl(fnum(call("notransp", x)))

        def rmatmat(x):
            return _dbl(fnum(call("transp", x)))
        mat = None
    else:
        mat = _dbl(_sq(src, "normest1"))
        if extra:
            raise MatlabError("normest1: extra parameters are only passed on to a function handle.")
        n = mat.shape[0]
        real = mat.dtype.kind != "c"

        def matmat(x):
            return mat @ x

        def rmatmat(x):
            return mat.conj().T @ x
    t = 2
    if len(args) > 1 and not (isinstance(args[1], np.ndarray) and args[1].size == 0):
        t = to_int(args[1], "t")
        if t == 0 or abs(t) > max(n, 1):
            raise MatlabError("normest1: t must be a nonzero integer no larger than the matrix size in magnitude.")
    trace = t < 0
    t = min(abs(t), max(n, 1))
    x0 = None
    if len(args) > 2 and not (isinstance(args[2], np.ndarray) and args[2].size == 0):
        x0 = _dbl(fnum(args[2]))
        if x0.shape != (n, t):
            raise MatlabError("normest1: X0 must have n rows and t columns.")
        if np.any(np.abs(np.sum(np.abs(x0), axis=0) - 1) > 1e-10):
            raise MatlabError("normest1: the columns of X0 must have unit 1-norm.")
    if n == 0:
        outs = (0.0, np.zeros((0, 1)), np.zeros((0, 1)), np.array([[0.0, 0.0]]))
        return outs[:nargout] if nargout > 1 else 0.0
    if n <= 4 or t == n:
        # small problems are done exactly
        full = matmat(np.eye(n))
        norms = np.sum(np.abs(full), axis=0)
        j = int(np.argmax(norms))
        est, v, w, its, nprod = float(norms[j]), np.eye(n)[:, j], full[:, j], 0, 1
    else:
        est, v, w, its, nprod = _normest1_core(interp, matmat, rmatmat, n, t, x0, real)
    if trace:
        interp.out(f"normest1: estimate {est:g} after {its} iteration(s) and {nprod} matrix product(s).\n")
    outs = (est, v.reshape(-1, 1), _fin(w.reshape(-1, 1)), np.array([[float(its), float(nprod)]]))
    return outs[:nargout] if nargout > 1 else est


# -- polynomials --------------------------------------------------------------------------------------

def _poly_strip(p: np.ndarray) -> np.ndarray:
    """Remove leading zero coefficients."""
    nz = np.nonzero(p)[0]
    return p[nz[0]:] if nz.size else p[:0]


def _polyvec(v: Any, name: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim != 2 or (a.size and 1 not in a.shape):
        raise MatlabError(f"{name}: polynomial coefficients must be given as vectors.")
    return _dbl(a.flatten())


def mpoles_groups(p: np.ndarray, tol: float = 1e-3, reorder: bool = True):
    """Group nearly equal poles. Returns (multiplicity counter per pole, ordering index)."""
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    order = np.argsort(-np.abs(p), kind="stable") if reorder else np.arange(n)
    remaining = list(order)
    mults: list[int] = []
    idx: list[int] = []
    while remaining:
        first = remaining[0]
        ref = p[first]
        scale = max(abs(ref), 1.0) if abs(ref) > 0 else 1.0
        group = [k for k in remaining if abs(p[k] - ref) <= tol * scale]
        for c, k in enumerate(group):
            mults.append(c + 1)
            idx.append(k)
        remaining = [k for k in remaining if k not in group]
    return np.array(mults, dtype=int), np.array(idx, dtype=int)


@builtin("mpoles")
def _mpoles(interp, args, nargout):
    nargs(args, 1, 3, "mpoles")
    p = _dbl(fnum(args[0])).flatten(order="F")
    tol = 1e-3
    if len(args) > 1 and not (isinstance(args[1], np.ndarray) and args[1].size == 0):
        tol = _scalar_real(args[1], "mpoles", "the tolerance")
    reorder = True
    if len(args) > 2:
        reorder = bool(np.all(np.asarray(fnum(args[2])) != 0))
    m, idx = mpoles_groups(p, tol, reorder)
    return m.astype(float).reshape(-1, 1), (idx + 1.0).reshape(-1, 1)


def _taylor_shift(c_desc: np.ndarray, x0) -> np.ndarray:
    """Coefficients (ascending powers of (x - x0)) of a polynomial given in descending powers of x."""
    c = np.array(c_desc, dtype=complex)
    n = c.size
    out = np.zeros(n, dtype=complex)
    for k in range(n):
        # synthetic division by (x - x0): the remainder is the next Taylor coefficient
        for i in range(1, n - k):
            c[i] += c[i - 1] * x0
        out[k] = c[n - k - 1]
    return out


def _conj_closed(vals: np.ndarray, tol: float = 1e-8) -> bool:
    """True when the multiset is closed under complex conjugation (so a real result is expected)."""
    v = list(vals)
    for x in vals:
        if abs(x.imag) <= tol * max(abs(x), 1.0):
            continue
        hit = next((k for k, y in enumerate(v) if abs(y - np.conj(x)) <= tol * max(abs(x), 1.0)), None)
        if hit is None:
            return False
    return True


def _real_if_close(a: np.ndarray, force: bool) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind == "c" and a.size:
        if force or np.max(np.abs(a.imag)) <= 1e-12 * max(np.max(np.abs(a)), np.finfo(float).tiny):
            return a.real.copy()
    return simplify(a)


def _cluster_poles(poles: np.ndarray, tol: float, real_coeffs: bool):
    """Sort poles as mpoles does and move every group of nearly equal poles to its mean.

    Returns the poles and the start index of each group. For a polynomial with real coefficients
    a group that lies on the real axis (up to the tolerance) gets a real location.
    """
    mult, idx = mpoles_groups(poles, tol)
    p = poles[idx].astype(complex)
    starts = np.nonzero(mult == 1)[0]
    for gi, s0 in enumerate(starts):
        e = starts[gi + 1] if gi + 1 < starts.size else p.size
        p0 = np.mean(p[s0:e])
        if real_coeffs and abs(p0.imag) <= tol * max(abs(p0), 1.0) and (e - s0 > 1 or p0.imag == 0):
            p0 = complex(p0.real, 0.0)
        if e - s0 > 1:
            p[s0:e] = p0
        elif real_coeffs and p0.imag == 0:
            p[s0] = p0
    return p, starts


def _pf_expand(b_rem: np.ndarray, a: np.ndarray, tol: float = 1e-3):
    """Residues and poles of b_rem/a (deg b_rem < deg a), both in descending powers."""
    real_coeffs = not (np.any(np.imag(a) != 0) or np.any(np.imag(b_rem) != 0))
    # the roots of a real polynomial from the real eigensolver: real roots come out exactly real
    # (the complex solver can leave +-1e-17i on them, depending on the LAPACK library)
    poles = np.roots(np.real(a) if real_coeffs else a).astype(complex)
    p_sorted, starts = _cluster_poles(poles, tol, real_coeffs)
    res = np.zeros(p_sorted.size, dtype=complex)
    for gi, s in enumerate(starts):
        e = starts[gi + 1] if gi + 1 < starts.size else p_sorted.size
        m = e - s
        p0 = p_sorted[s]
        others = np.concatenate([p_sorted[:s], p_sorted[e:]])
        q = a[0] * np.poly(others) if others.size else np.array([a[0]], dtype=complex)
        num_t = _taylor_shift(b_rem, p0) if b_rem.size else np.zeros(1, dtype=complex)
        den_t = _taylor_shift(q, p0)
        coef = np.zeros(m, dtype=complex)
        for k in range(m):
            acc = num_t[k] if k < num_t.size else 0.0
            for i in range(1, k + 1):
                if i < den_t.size:
                    acc -= den_t[i] * coef[k - i]
            coef[k] = acc / den_t[0]
        # coefficient of (s - p0)^(-j) is coef[m - j]
        for j in range(1, m + 1):
            res[s + j - 1] = coef[m - j]
        if real_coeffs and p0.imag == 0:
            res[s:e] = res[s:e].real  # a real pole of a real fraction has real residues
    return res, p_sorted


@builtin("residue")
def _residue(interp, args, nargout):
    nargs(args, 2, 3, "residue")
    if len(args) == 2:
        b = _poly_strip(_polyvec(args[0], "residue"))
        a = _poly_strip(_polyvec(args[1], "residue"))
        if a.size == 0:
            raise MatlabError("residue: the denominator must not be zero.")
        if b.size >= a.size:
            k, rem = np.polydiv(b, a)
            k = np.atleast_1d(k)
            rem = np.atleast_1d(rem)
            # keep the remainder aligned to degree < deg(a)
            rem = rem[-(a.size - 1):] if a.size > 1 else rem[:0]
        else:
            k, rem = np.zeros(0), b
        if a.size == 1:
            r, p = np.zeros(0), np.zeros(0)
        else:
            r, p = _pf_expand(rem.astype(complex), a.astype(complex))
        r = simplify(np.asarray(r).reshape(-1, 1))
        p = simplify(np.asarray(p).reshape(-1, 1))
        kk = simplify(np.asarray(k).reshape(1, -1)) if np.asarray(k).size else np.zeros((0, 0))
        return (r, p, kk)[:max(nargout, 1)] if nargout > 1 else r
    # inverse: from residues, poles and the direct term back to b/a
    r = _dbl(fnum(args[0])).flatten(order="F").astype(complex)
    p = _dbl(fnum(args[1])).flatten(order="F").astype(complex)
    k = _dbl(fnum(args[2])).flatten(order="F").astype(complex)
    if r.size != p.size:
        raise MatlabError("residue: R and P must have the same number of elements.")
    mult, idx = mpoles_groups(p, 1e-3, reorder=False)
    order = np.empty(p.size, dtype=int)
    order[:] = idx
    a = np.poly(p) if p.size else np.array([1.0 + 0j])
    b = np.zeros(max(p.size, 1), dtype=complex)
    used = np.zeros(p.size, dtype=int)
    for pos, orig in enumerate(idx):
        used[orig] = mult[pos]
    for j in range(p.size):
        m = used[j]
        # drop m copies of this pole from the list (members of its cluster)
        close = [q for q in range(p.size) if abs(p[q] - p[j]) <= 1e-3 * max(abs(p[j]), 1.0)]
        drop = set(close[:m])
        others = np.array([p[q] for q in range(p.size) if q not in drop], dtype=complex)
        term = r[j] * (np.poly(others) if others.size else np.array([1.0 + 0j]))
        b[b.size - term.size:] += term
    if k.size:
        b = np.polyadd(b, np.convolve(k, a))
    b = _poly_strip(np.atleast_1d(b)) if np.any(b != 0) else np.zeros(1, dtype=complex)
    real = _conj_closed(p) and (np.all(np.abs(k.imag) == 0) if k.size else True)
    b = _real_if_close(b, False)
    a = _real_if_close(a, real and np.max(np.abs(np.asarray(a).imag)) < 1e-8 * max(np.max(np.abs(a)), 1.0))
    return b.reshape(1, -1), a.reshape(1, -1)


@builtin("residuez")
def _residuez(interp, args, nargout):
    nargs(args, 2, 3, "residuez")
    if len(args) == 2:
        b = _polyvec(args[0], "residuez")
        a = _polyvec(args[1], "residuez")
        while a.size > 1 and a[-1] == 0:
            a = a[:-1]
        while b.size > 1 and b[-1] == 0:
            b = b[:-1]
        if a.size == 0 or a[0] == 0:
            raise MatlabError("residuez: the first denominator coefficient must be nonzero.")
        # in x = z^-1 the coefficient vectors are ascending polynomials
        if b.size >= a.size:
            kq, rem = np.polydiv(b[::-1], a[::-1])
            k = np.atleast_1d(kq)[::-1]
            rem = np.atleast_1d(rem)
            rem = rem[-(a.size - 1):] if a.size > 1 else rem[:0]
        else:
            k, rem = np.zeros(0), b[::-1]
        if a.size == 1:
            r, p = np.zeros(0), np.zeros(0)
        else:
            # B(x)/A(x) with A(x) = a0 * prod(1 - p_i x): poles of the x-fraction are 1/p_i
            real_coeffs = a.dtype.kind != "c" and b.dtype.kind != "c"
            p, starts = _cluster_poles(np.roots(a).astype(complex), 1e-3, real_coeffs)
            r = np.zeros(p.size, dtype=complex)
            for gi, s in enumerate(starts):
                e = starts[gi + 1] if gi + 1 < starts.size else p.size
                m = e - s
                p0 = p[s]
                others = np.concatenate([p[:s], p[e:]])
                q = np.array([a[0]], dtype=complex)
                for o in others:
                    q = np.convolve(q, np.array([-o, 1.0]))  # (1 - o x) in descending powers of x
                x0 = 1.0 / p0
                num_t = _taylor_shift(rem.astype(complex), x0) if rem.size else np.zeros(1, dtype=complex)
                den_t = _taylor_shift(q, x0)
                # u = 1 - p0 x  =>  (x - x0) = -u / p0
                g = np.zeros(m, dtype=complex)
                for kk in range(m):
                    acc = (num_t[kk] if kk < num_t.size else 0.0) * (-1.0 / p0) ** kk
                    for i in range(1, kk + 1):
                        if i < den_t.size:
                            acc -= den_t[i] * (-1.0 / p0) ** i * g[kk - i]
                    g[kk] = acc / den_t[0]
                for j in range(1, m + 1):
                    r[s + j - 1] = g[m - j]
                if real_coeffs and p0.imag == 0:
                    r[s:e] = r[s:e].real
        r = simplify(np.asarray(r).reshape(-1, 1))
        p = simplify(np.asarray(p).reshape(-1, 1))
        kk = simplify(np.asarray(k).reshape(1, -1)) if np.asarray(k).size else np.zeros((0, 0))
        return (r, p, kk)[:max(nargout, 1)] if nargout > 1 else r
    r = _dbl(fnum(args[0])).flatten(order="F").astype(complex)
    p = _dbl(fnum(args[1])).flatten(order="F").astype(complex)
    k = _dbl(fnum(args[2])).flatten(order="F").astype(complex)
    if r.size != p.size:
        raise MatlabError("residuez: R and P must have the same number of elements.")
    mult, idx = mpoles_groups(p, 1e-3, reorder=False)
    used = np.zeros(p.size, dtype=int)
    for pos, orig in enumerate(idx):
        used[orig] = mult[pos]
    a = np.poly(p) if p.size else np.array([1.0 + 0j])  # ascending in x = z^-1
    b = np.zeros(max(p.size, 1), dtype=complex)        # ascending in x
    for j in range(p.size):
        m = used[j]
        close = [q for q in range(p.size) if abs(p[q] - p[j]) <= 1e-3 * max(abs(p[j]), 1.0)]
        drop = set(close[:m])
        others = np.array([p[q] for q in range(p.size) if q not in drop], dtype=complex)
        term = r[j] * (np.poly(others) if others.size else np.array([1.0 + 0j]))
        b[:term.size] += term
    if k.size:
        extra = np.convolve(k, a)
        if extra.size > b.size:
            b = np.concatenate([b, np.zeros(extra.size - b.size, dtype=complex)])
        b[:extra.size] += extra
    while b.size > 1 and b[-1] == 0:
        b = b[:-1]
    real = _conj_closed(p)
    b = _real_if_close(b, False)
    a = _real_if_close(a, real and np.max(np.abs(np.asarray(a).imag)) < 1e-8 * max(np.max(np.abs(a)), 1.0))
    return b.reshape(1, -1), a.reshape(1, -1)


@builtin("polyvalm")
def _polyvalm(interp, args, nargout):
    nargs(args, 2, 2, "polyvalm")
    p = _polyvec(args[0], "polyvalm")
    x = _dbl(_sq(args[1], "polyvalm"))
    n = x.shape[0]
    y = np.zeros((n, n), dtype=np.result_type(p.dtype, x.dtype))
    eye = np.eye(n)
    for c in p:  # Horner's rule with matrix products
        y = x @ y + c * eye
    return _fin(y, _single(args[0], args[1]))


@builtin("polyeig")
def _polyeig(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for polyeig.")
    if nargout > 3:
        raise MatlabError("polyeig: too many output arguments.")
    mats = [_dbl(_sq(a, "polyeig")) for a in args]
    n = mats[0].shape[0]
    if any(m.shape != (n, n) for m in mats):
        raise MatlabError("polyeig: all coefficient matrices must be square and of the same size.")
    p = len(mats) - 1
    sla = _sla()
    if p == 0 or n == 0:
        # degree 0: every vector in the null space; MATLAB treats this as the pencil A0 = 0
        e = np.zeros((0, 1))
        return (np.zeros((n, 0)), e, np.zeros((0, 1)))[:max(nargout, 1)] if nargout > 1 else e
    dt = complex if any(m.dtype.kind == "c" for m in mats) else float
    big = n * p
    a = np.zeros((big, big), dtype=dt)
    b = np.eye(big, dtype=dt)
    # companion pencil:  A z = lambda B z  with  z = [x; lambda x; ...; lambda^(p-1) x]
    for k in range(p - 1):
        a[k * n:(k + 1) * n, (k + 1) * n:(k + 2) * n] = np.eye(n)
    for k in range(p):
        a[(p - 1) * n:, k * n:(k + 1) * n] = -mats[k]
    b[(p - 1) * n:, (p - 1) * n:] = mats[p]
    with np.errstate(divide="ignore", invalid="ignore"):
        if nargout <= 1:
            e = sla.eigvals(a, b)
            return simplify(np.asarray(e).reshape(-1, 1))
        e, z = sla.eig(a, b)
    e = np.asarray(e, dtype=complex)
    x = np.zeros((n, big), dtype=complex)
    for j in range(big):
        # take the block of z with the largest norm: the most accurate copy of the eigenvector
        blocks = z[:, j].reshape(p, n)
        k = int(np.argmax(np.linalg.norm(blocks, axis=1)))
        v = blocks[k]
        nv = np.linalg.norm(v)
        x[:, j] = v / nv if nv else v
    outs = [simplify(x), simplify(e.reshape(-1, 1))]
    if nargout > 2:
        # normwise condition number of each eigenvalue (Tisseur's formula)
        norms = np.array([np.linalg.norm(m, 2) for m in mats])
        s = np.zeros(big)
        for j in range(big):
            lam = e[j]
            if not np.isfinite(lam):
                s[j] = np.inf
                continue
            pl = sum(lam ** k * mats[k] for k in range(p + 1))
            dpl = sum(k * lam ** (k - 1) * mats[k] for k in range(1, p + 1))
            u, sv, vh = np.linalg.svd(pl)
            yv, xv = u[:, -1], vh[-1, :].conj()
            denom = abs(lam) * abs(np.vdot(yv, dpl @ xv))
            alpha = float(np.sum(np.abs(lam) ** np.arange(p + 1) * norms))
            with np.errstate(divide="ignore", invalid="ignore"):
                s[j] = alpha * np.linalg.norm(yv) * np.linalg.norm(xv) / denom if denom else np.inf
        outs.append(s.reshape(-1, 1))
    return tuple(outs[:nargout])


# -- filtering helpers ------------------------------------------------------------------------------------

@builtin("filtic")
def _filtic(interp, args, nargout):
    nargs(args, 3, 4, "filtic")
    b = _polyvec(args[0], "filtic")
    a = _polyvec(args[1], "filtic")
    y = _dbl(fnum(args[2])).flatten(order="F")
    x = _dbl(fnum(args[3])).flatten(order="F") if len(args) > 3 else np.zeros(0)
    if a.size == 0 or a[0] == 0:
        raise MatlabError("filtic: the first denominator coefficient must be nonzero.")
    b, a = b / a[0], a / a[0]
    order = max(a.size, b.size) - 1
    dt = np.result_type(b.dtype, a.dtype, y.dtype, x.dtype, float)
    bp = np.concatenate([b, np.zeros(order + 1 - b.size)]).astype(dt)
    ap = np.concatenate([a, np.zeros(order + 1 - a.size)]).astype(dt)
    yp = np.concatenate([y, np.zeros(max(order - y.size, 0))]).astype(dt)
    xp = np.concatenate([x, np.zeros(max(order - x.size, 0))]).astype(dt)
    z = np.zeros(order, dtype=dt)
    for m in range(order):
        k = np.arange(order - m)
        z[m] = np.sum(bp[m + 1 + k] * xp[k]) - np.sum(ap[m + 1 + k] * yp[k])
    return simplify(z.reshape(-1, 1)) if order else np.zeros((0, 1))


@builtin("filter2")
def _filter2(interp, args, nargout):
    nargs(args, 2, 3, "filter2")
    h = fnum(args[0])
    x = fnum(args[1])
    if h.ndim != 2 or x.ndim != 2:
        raise MatlabError("filter2: the filter and the data must be 2-D matrices.")
    shape = "same"
    if len(args) > 2:
        shape = text(args[2]).lower() if is_text(args[2]) else ""
        if shape not in ("same", "full", "valid"):
            raise MatlabError("filter2: the shape must be 'same', 'full' or 'valid'.")
    # correlation: convolution with the filter rotated by 180 degrees
    out = BUILTINS["conv2"](interp, [x, np.ascontiguousarray(h[::-1, ::-1]), char(shape)], 1)
    if any(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in args[:2]):
        out = np.asarray(out).astype(np.complex64 if np.iscomplexobj(out) else np.float32)  # single in, single out
    return out


# -- more iterative solvers ---------------------------------------------------------------------------

def _solver_setup(interp, args: list, name: str, need_transpose: bool = False):
    """Common argument handling for  x = solver(A, b, tol, maxit, M1, M2, x0)."""
    from . import sparse as sm
    if len(args) < 2:
        raise MatlabError(f"Not enough input arguments for {name}.")
    if len(args) > 7:
        raise MatlabError(f"Too many input arguments for {name}.")
    b = sm._dense(args[1])
    if b.ndim != 2 or b.shape[1] != 1:
        raise MatlabError(f"{name}: the right-hand side must be a column vector.")
    n = b.shape[0]
    src = args[0]
    rmatvec = None
    if isinstance(src, FunctionHandle) or is_text(src):
        if need_transpose:
            def matvec(x):
                res = interp.call_function_value(src, [np.asarray(x).reshape(-1, 1), char("notransp")], 1)
                return sm._dense(res[0]).reshape(-1)

            def rmatvec(x):
                res = interp.call_function_value(src, [np.asarray(x).reshape(-1, 1), char("transp")], 1)
                return sm._dense(res[0]).reshape(-1)
        else:
            def matvec(x):
                res = interp.call_function_value(src, [np.asarray(x).reshape(-1, 1)], 1)
                return sm._dense(res[0]).reshape(-1)
        cplx = False
    else:
        m = sm._num_csc(src) if isinstance(src, sm.SparseMatrix) else sm._dense(src)
        if m.shape != (n, n):
            raise MatlabError(f"{name}: A must be a square matrix whose size matches the right-hand side.")
        mh = m.conj().T

        def matvec(x):
            return np.asarray(m @ x).reshape(-1)

        def rmatvec(x):
            return np.asarray(mh @ x).reshape(-1)
        cplx = m.dtype.kind == "c"
    tol_v, maxit_v = sm._opt(args, 2), sm._opt(args, 3)
    tol = 1e-6 if tol_v is None else _scalar_real(tol_v, name, "the tolerance")
    maxit = min(n, 20) if maxit_v is None else to_int(maxit_v, "maxit")
    if maxit < 0:
        raise MatlabError(f"{name}: maxit must be nonnegative.")
    m1 = sm._precond_solver(interp, sm._opt(args, 4), n)
    m2 = sm._precond_solver(interp, sm._opt(args, 5), n)
    x0v = sm._opt(args, 6)
    cplx = cplx or b.dtype.kind == "c"
    dtype = complex if cplx else float
    x0 = np.zeros(n, dtype=dtype) if x0v is None else sm._dense(x0v).reshape(-1).astype(dtype)
    if x0.size != n:
        raise MatlabError(f"{name}: the initial guess must have one element per unknown.")

    def psolve(v):
        y = np.asarray(v).reshape(-1)
        if m1 is not None:
            y = np.asarray(m1(y)).reshape(-1)
        if m2 is not None:
            y = np.asarray(m2(y)).reshape(-1)
        return y

    has_m = m1 is not None or m2 is not None
    return dict(n=n, b=b.reshape(-1).astype(dtype), matvec=matvec, rmatvec=rmatvec, tol=tol, maxit=maxit,
                psolve=psolve if has_m else None, x0=x0, dtype=dtype)


def _solver_finish(interp, name, nargout, x, flag, relres, it, resvec, tol):
    x = simplify(np.asarray(x).reshape(-1, 1))
    if nargout <= 1:
        if flag == 0:
            interp.out(f"{name} converged at iteration {it:g} to a solution with relative residual "
                       f"{relres:.2g}.\n")
        else:
            why = {1: "the maximum number of iterations was reached", 3: "the method stagnated",
                   4: "a scalar quantity became too small or too large to continue"}.get(int(flag), "it broke down")
            interp.out(f"{name} stopped at iteration {it:g} without converging to the desired tolerance "
                       f"{tol:.2g} because {why}.\nThe iterate returned has relative residual {relres:.2g}.\n")
        return x
    outs = (x, np.array([[float(flag)]]), np.array([[float(relres)]]), np.array([[float(it)]]),
            np.array(resvec, dtype=float).reshape(-1, 1))
    return outs[:nargout]


@builtin("bicgstabl")
def _bicgstabl(interp, args, nargout):
    s = _solver_setup(interp, args, "bicgstabl")
    n, b, matvec, tol, maxit, dtype = s["n"], s["b"], s["matvec"], s["tol"], s["maxit"], s["dtype"]
    psolve = s["psolve"]
    ell = 2

    def op(v):  # right-preconditioned operator A * inv(M)
        return matvec(psolve(v)) if psolve is not None else matvec(v)

    bnorm = float(np.linalg.norm(b))
    x = s["x0"].copy()
    if bnorm == 0:
        return _solver_finish(interp, "bicgstabl", nargout, np.zeros(n), 0, 0.0, 0, [0.0], tol)
    y = np.zeros(n, dtype=dtype)  # correction in the preconditioned variable
    r = np.zeros((ell + 1, n), dtype=dtype)
    u = np.zeros((ell + 1, n), dtype=dtype)
    r[0] = b - matvec(x)
    resvec = [float(np.linalg.norm(r[0]))]
    flag = 1
    it = 0.0
    if resvec[0] <= tol * bnorm:
        return _solver_finish(interp, "bicgstabl", nargout, x, 0, resvec[0] / bnorm, 0, resvec, tol)
    rt = r[0].copy()
    rho0, alpha, omega = 1.0, 0.0, 1.0
    tiny = np.finfo(float).tiny
    best = (resvec[0], x.copy(), 0.0)
    for outer in range(maxit):
        rho0 = -omega * rho0
        broke = False
        for j in range(ell):
            rho1 = np.vdot(rt, r[j])
            if abs(rho0) < tiny:
                broke = True
                break
            beta = alpha * rho1 / rho0
            rho0 = rho1
            for i in range(j + 1):
                u[i] = r[i] - beta * u[i]
            u[j + 1] = op(u[j])
            gamma = np.vdot(rt, u[j + 1])
            if abs(gamma) < tiny:
                broke = True
                break
            alpha = rho0 / gamma
            for i in range(j + 1):
                r[i] = r[i] - alpha * u[i + 1]
            r[j + 1] = op(r[j])
            y = y + alpha * u[0]
            it = outer + (j + 1) / (2.0 * ell)
            resvec.append(float(np.linalg.norm(r[0])))
            if resvec[-1] < best[0]:
                best = (resvec[-1], y.copy(), it)
        if broke:
            flag = 4
            break
        # minimal-residual part: orthogonalize r[1..ell] and take the best combination
        tau = np.zeros((ell + 1, ell + 1), dtype=dtype)
        sigma = np.zeros(ell + 1)
        gp = np.zeros(ell + 1, dtype=dtype)
        for j in range(1, ell + 1):
            for i in range(1, j):
                tau[i, j] = np.vdot(r[i], r[j]) / sigma[i]
                r[j] = r[j] - tau[i, j] * r[i]
            sigma[j] = float(np.real(np.vdot(r[j], r[j])))
            if sigma[j] < tiny:
                broke = True
                break
            gp[j] = np.vdot(r[j], r[0]) / sigma[j]
        if broke:
            flag = 4
            break
        g = np.zeros(ell + 1, dtype=dtype)
        gpp = np.zeros(ell + 1, dtype=dtype)
        g[ell] = gp[ell]
        omega = g[ell]
        for j in range(ell - 1, 0, -1):
            g[j] = gp[j] - sum(tau[j, i] * g[i] for i in range(j + 1, ell + 1))
        for j in range(1, ell):
            gpp[j] = g[j + 1] + sum(tau[j, i] * g[i + 1] for i in range(j + 1, ell))
        y = y + g[1] * r[0]
        r[0] = r[0] - gp[ell] * r[ell]
        u[0] = u[0] - g[ell] * u[ell]
        for j in range(1, ell):
            u[0] = u[0] - g[j] * u[j]
            y = y + gpp[j] * r[j]
            r[0] = r[0] - gp[j] * r[j]
        it = float(outer + 1)
        resvec.append(float(np.linalg.norm(r[0])))
        if resvec[-1] < best[0]:
            best = (resvec[-1], y.copy(), it)
        if resvec[-1] <= tol * bnorm:
            flag = 0
            break
        if abs(omega) < tiny:
            flag = 4
            break
    if flag != 0:
        y, it = best[1], best[2]
    xs = x + (psolve(y) if psolve is not None else y)
    relres = float(np.linalg.norm(b - matvec(xs)) / bnorm)
    if flag == 0 and relres > tol:  # the recursive residual drifted from the true one
        flag = 3 if relres > 10 * tol else 0
    return _solver_finish(interp, "bicgstabl", nargout, xs, flag, relres, it, resvec, tol)


def _scipy_krylov(name: str):
    def f(interp, args, nargout):
        import warnings

        import scipy.sparse.linalg as spla
        s = _solver_setup(interp, args, name, need_transpose=(name == "qmr"))
        n, b, tol, maxit, dtype = s["n"], s["b"], s["tol"], s["maxit"], s["dtype"]
        bnorm = float(np.linalg.norm(b))
        if bnorm == 0:
            return _solver_finish(interp, name, nargout, np.zeros(n), 0, 0.0, 0, [0.0], tol)
        psolve = s["psolve"]
        a_op = spla.LinearOperator((n, n), matvec=s["matvec"], rmatvec=s["rmatvec"], dtype=dtype)
        resvec = [float(np.linalg.norm(b - s["matvec"](s["x0"])))]
        count = [0]

        def cb(xk):
            count[0] += 1
            resvec.append(float(np.linalg.norm(b - s["matvec"](np.asarray(xk).reshape(-1)))))

        if resvec[0] <= tol * bnorm:
            return _solver_finish(interp, name, nargout, s["x0"], 0, resvec[0] / bnorm, 0, resvec, tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "qmr":
                m1 = None
                if psolve is not None:
                    if any(isinstance(a, FunctionHandle) for a in args[4:6]):
                        raise MatlabError("qmr: a preconditioner given as a function handle is not supported; "
                                          "pass the preconditioner matrices.")
                    from . import sparse as sm
                    mats = [sm._dense(a) for a in args[4:6]
                            if not (isinstance(a, np.ndarray) and a.size == 0)]
                    full = mats[0] if len(mats) == 1 else mats[0] @ mats[1]
                    fac = _sla().lu_factor(full.astype(dtype))
                    m1 = spla.LinearOperator(
                        (n, n), matvec=lambda v: _sla().lu_solve(fac, v),
                        rmatvec=lambda v: _sla().lu_solve(fac, v, trans=2), dtype=dtype)
                x, info = spla.qmr(a_op, b, x0=s["x0"], rtol=tol, atol=0.0, maxiter=maxit, M1=m1, callback=cb)
            else:
                m_op = None if psolve is None else spla.LinearOperator((n, n), matvec=psolve, dtype=dtype)
                x, info = spla.tfqmr(a_op, b, x0=s["x0"], rtol=tol, atol=0.0, maxiter=maxit, M=m_op, callback=cb)
        relres = float(np.linalg.norm(b - s["matvec"](np.asarray(x).reshape(-1))) / bnorm)
        flag = 0 if relres <= tol * (1 + 1e-9) else (1 if info > 0 else 4)
        return _solver_finish(interp, name, nargout, x, flag, relres, count[0], resvec, tol)
    return f


builtin("qmr")(_scipy_krylov("qmr"))
builtin("tfqmr")(_scipy_krylov("tfqmr"))


@builtin("symmlq")
def _symmlq(interp, args, nargout):
    s = _solver_setup(interp, args, "symmlq")
    n, b, matvec, tol, maxit = s["n"], s["b"], s["matvec"], s["tol"], s["maxit"]
    psolve = s["psolve"]
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0:
        return _solver_finish(interp, "symmlq", nargout, np.zeros(n), 0, 0.0, 0, [0.0], tol)
    x0 = s["x0"]
    r0 = b - matvec(x0)
    resvec = [float(np.linalg.norm(r0))]
    if resvec[0] <= tol * bnorm:
        return _solver_finish(interp, "symmlq", nargout, x0, 0, resvec[0] / bnorm, 0, resvec, tol)
    # Lanczos in the inner product of the (symmetric positive definite) preconditioner M:
    # z_k are the Lanczos vectors, w_k = M \ z_k, and the iterates lie in span{w_1, ..., w_k}.
    z = r0.copy()
    w = psolve(z) if psolve is not None else z.copy()
    beta1 = np.sqrt(float(np.real(np.vdot(z, w))))
    if not np.isfinite(beta1) or beta1 <= 0:
        raise MatlabError("symmlq: the preconditioner must be symmetric positive definite.")
    zs = [z / beta1]
    ws = [w / beta1]
    alphas: list[float] = []
    betas: list[float] = [beta1]
    flag = 1
    x = x0.copy()
    it = 0
    best = (resvec[0], x0.copy(), 0)
    for k in range(1, maxit + 1):
        v = matvec(ws[-1])
        alpha = float(np.real(np.vdot(ws[-1], v)))
        v = v - alpha * zs[-1]
        if k > 1:
            v = v - betas[-1] * zs[-2]
        wn = psolve(v) if psolve is not None else v.copy()
        beta = np.sqrt(max(float(np.real(np.vdot(v, wn))), 0.0))
        alphas.append(alpha)
        # T_k is k-by-k; its first k-1 rows give the underdetermined system of the LQ iterate,
        # and the full matrix gives the conjugate-gradient point when it is nonsingular
        t = np.diag(alphas)
        for i in range(k - 1):
            t[i, i + 1] = t[i + 1, i] = betas[i + 1]
        rhs = np.zeros(k)
        rhs[0] = beta1
        cg = None
        try:
            if abs(np.linalg.det(t)) > 0 and np.linalg.cond(t) < 1.0 / _EPS:
                cg = np.linalg.solve(t, rhs)
        except np.linalg.LinAlgError:
            cg = None
        if k > 1:
            lq = np.linalg.lstsq(t[:k - 1, :], rhs[:k - 1], rcond=None)[0]
        else:
            lq = np.zeros(1)
        wk = np.array(ws).T
        cand = [x0 + wk @ lq]
        if cg is not None:
            cand.append(x0 + wk @ cg)
        res = [float(np.linalg.norm(b - matvec(c))) for c in cand]
        j = int(np.argmin(res))
        x, it = cand[j], k
        resvec.append(res[j])
        if res[j] < best[0]:
            best = (res[j], x.copy(), k)
        if res[j] <= tol * bnorm:
            flag = 0
            break
        if beta <= _EPS * max(abs(alpha), 1.0):
            flag = 4 if res[j] > tol * bnorm else 0  # the Krylov space is exhausted
            break
        betas.append(beta)
        zs.append(v / beta)
        ws.append(wn / beta)
    if flag != 0:
        x, it = best[1], best[2]
    relres = float(np.linalg.norm(b - matvec(x)) / bnorm)
    return _solver_finish(interp, "symmlq", nargout, x, flag, relres, it, resvec, tol)


def _register_sparse_aware() -> None:
    """Functions here that take sparse matrices as they are (no conversion to full)."""
    from . import sparse as sm
    sm._OWN.update({"bicgstabl", "qmr", "tfqmr", "symmlq", "decomposition"})


_register_sparse_aware()
