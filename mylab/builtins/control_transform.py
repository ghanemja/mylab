"""Model transformations: reduction, discretization, delays, canonical forms.

minreal, sminreal, balreal, balred, hsvd, modred, stabsep, gram, covar, pade, c2d, d2c, d2d,
ctrb, obsv, ctrbf, obsvf, canon, ss2ss and xperm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char
from . import _ctrl_num as cn
from .control import (
    FRD, LTI, PID, SS, TF, ZPK, _carry, _clean_poly, _grid, _no_delay_copy, absorb_delay, as_matrix, as_scalar,
    as_vector, back_to, col, is_empty_num, lti_append, lti_mul, lti_times, ltifn, need_lti, pid_to_tf, row,
    split_options, to_ss, to_tf, to_zpk,
)
from ._util import is_text, text


def _need_dynamic(sys: LTI, name: str) -> LTI:
    if isinstance(sys, FRD):
        raise MatlabError(f"{name} does not apply to frequency-response data.")
    return pid_to_tf(sys) if isinstance(sys, PID) else sys


def _ss_like(src: SS, a, b, c, d, ts: float | None = None) -> SS:
    out = SS(a, b, c, d, src.ts if ts is None else ts)
    _carry(out, src)
    return out


# -- minimal realizations ---------------------------------------------------------------------

def _cancel(z: np.ndarray, p: np.ndarray, tol: float):
    z, p = list(np.asarray(z, dtype=complex)), list(np.asarray(p, dtype=complex))
    kept_z = []
    for v in z:
        if p:
            dist = [abs(v - q) for q in p]
            k = int(np.argmin(dist))
            if dist[k] <= tol * max(1.0, abs(v)):
                p.pop(k)
                continue
        kept_z.append(v)
    return cn.pair_conjugates(np.array(kept_z, dtype=complex)), cn.pair_conjugates(np.array(p, dtype=complex))


def _merge_multiple_roots(r: np.ndarray, tol: float) -> np.ndarray:
    """Roots computed from a polynomial, with each cluster of nearly equal roots replaced by its mean.

    A double root comes out of the eigenvalue solver split by about sqrt(eps) (-1 +- 1e-8 for
    s^2 + 2s + 1), by an amount that depends on the LAPACK library; the mean of the cluster is
    accurate to working precision. Without this, cancelling one of the pair against a zero left
    the other slightly off on some platforms (1/(s + 1.00000001)).
    """
    r = np.asarray(r, dtype=complex).ravel()
    if r.size < 2:
        return r
    out = r.copy()
    used = np.zeros(r.size, dtype=bool)
    for i in range(r.size):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, r.size):
            if not used[j] and abs(r[j] - r[i]) <= tol * max(1.0, abs(r[i])):
                group.append(j)
        if len(group) > 1:
            out[group] = r[group].mean()
            used[group] = True
    return out


def minreal_model(sys: LTI, tol: float | None = None):
    """The model without cancelling pole/zero pairs or unreachable states; also the number removed."""
    sys = _need_dynamic(sys, "minreal")
    if isinstance(sys, SS):
        a, b, c, d = cn.ss_minreal(sys.a, sys.b, sys.c, sys.d, tol)
        return _ss_like(sys, a, b, c, d), sys.nx - a.shape[0]
    g = to_zpk(sys)
    eff = np.sqrt(cn.EPS) if tol is None else tol
    z = _grid(g.ny, g.nu, lambda i, j: None)
    p = _grid(g.ny, g.nu, lambda i, j: None)
    removed = 0
    from_polynomials = isinstance(sys, TF)  # zeros and poles computed as roots, not given
    for i in range(g.ny):
        for j in range(g.nu):
            gz, gp = g.z[i][j], g.p[i][j]
            if from_polynomials:
                gz, gp = _merge_multiple_roots(gz, 10 * eff), _merge_multiple_roots(gp, 10 * eff)
            z[i][j], p[i][j] = _cancel(gz, gp, eff)
            removed += len(g.p[i][j]) - len(p[i][j])
    out = ZPK(z, p, g.k, g.ts, g.var)
    out.display_format = g.display_format
    _carry(out, g)
    if isinstance(sys, TF):
        res = to_tf(out)
        res.var = sys.var
        return res, removed
    return out, removed


@ltifn("minreal")
def _minreal(interp, args, nargout):
    sys = need_lti(args, "minreal")
    tol = as_scalar(args[1], "tolerance") if len(args) > 1 and not is_empty_num(args[1]) else None
    show = True
    if len(args) > 2:
        show = bool(np.asarray(args[2]).any())
    if nargout > 1:
        raise MatlabError("minreal returns the reduced model only; the transformation matrix is not available.")
    out, removed = minreal_model(sys, tol)
    if isinstance(out, SS) and removed and show:
        interp.out(f"{removed} state{'s' if removed != 1 else ''} removed.\n")
    return out


@ltifn("sminreal")
def _sminreal(interp, args, nargout):
    sys = need_lti(args, "sminreal")
    if not isinstance(sys, SS):
        return sys
    idx = cn.sminreal_states(sys.a, sys.b, sys.c)
    out = _ss_like(sys, sys.a[np.ix_(idx, idx)], sys.b[idx, :], sys.c[:, idx], sys.d)
    out.state_name = [sys.state_name[i] for i in idx]
    return out


# -- gramians, balancing, order reduction -----------------------------------------------------

def _stable_ss(sys: LTI, name: str) -> SS:
    s = to_ss(_no_delay_copy(_need_dynamic(sys, name)))
    if not cn.is_stable(cn.eigvals(s.a), s.is_dt):
        raise MatlabError(f"{name}: the model must be stable (all poles in the open left half-plane, or "
                          "inside the unit circle for discrete time).")
    return s


def _chol_factor(w: np.ndarray) -> np.ndarray:
    """L with W = L L' (via eigenvalues when W is only positive semidefinite)."""
    try:
        return np.linalg.cholesky(w)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cn.sym(w))
        return vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]


@ltifn("gram")
def _gram(interp, args, nargout):
    if len(args) != 2 or not is_text(args[1]):
        raise MatlabError("Use gram(sys, 'c') for the controllability gramian or gram(sys, 'o') for the "
                          "observability gramian.")
    kind = text(args[1]).lower()
    if kind not in ("c", "o", "cf", "of"):
        raise MatlabError("gram: the type must be 'c', 'o', 'cf' or 'of'.")
    s = _stable_ss(need_lti(args, "gram"), "gram")
    if kind[0] == "c":
        w = cn.gram(s.a, s.b, s.is_dt)
    else:
        w = cn.gram(s.a.conj().T, s.c.conj().T, s.is_dt)
    if len(kind) == 2:
        try:
            return np.linalg.cholesky(w).conj().T
        except np.linalg.LinAlgError:
            raise MatlabError("gram: the gramian is not positive definite, so it has no Cholesky factor "
                              "(the model is not controllable/observable).") from None
    return w


@ltifn("covar")
def _covar(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("Use [P, Q] = covar(sys, W) with the noise intensity W.")
    sys = _need_dynamic(need_lti(args, "covar"), "covar")
    s = to_ss(_no_delay_copy(sys))
    w = as_matrix(args[1], "noise intensity W")
    if w.size == 1:
        w = w.flat[0] * np.eye(s.nu)
    if w.shape != (s.nu, s.nu):
        raise MatlabError(f"covar: W must be a {s.nu}-by-{s.nu} matrix (or a scalar).")
    if not cn.is_stable(cn.eigvals(s.a), s.is_dt):
        return np.full((s.ny, s.ny), np.inf), np.full((s.nx, s.nx), np.inf)
    bw = s.b @ w @ s.b.conj().T
    q = cn.sym(cn.lyap_d(s.a, bw) if s.is_dt else cn.lyap_c(s.a, bw))
    p = s.c @ q @ s.c.conj().T
    if s.is_dt:
        p = p + s.d @ w @ s.d.conj().T
    elif np.any(s.d @ w @ s.d.conj().T != 0):
        p = np.where(np.eye(s.ny, dtype=bool) & (np.diag(s.d @ w @ s.d.conj().T) != 0)[:, None], np.inf, p)
    return cn.sym(p) if np.all(np.isfinite(p)) else p, q


def stab_split(a, b, c, disc: bool):
    """Decouple the stable modes from the others: (stable part), (other part), T with x = T [xs; xu]."""
    n = a.shape[0]
    if n == 0:
        return (a, b, c), (a, b, c), np.zeros((0, 0))
    from scipy.linalg import schur, solve_sylvester
    t, z, ns = schur(a, output="real", sort="iuc" if disc else "lhp")
    # eigenvalues on the boundary count as unstable; re-sort with a margin if needed
    ev = np.linalg.eigvals(t[:ns, :ns]) if ns else np.zeros(0)
    edge = (np.abs(ev) >= 1 - 1e-10) if disc else (ev.real >= -1e-10 * np.maximum(1.0, np.abs(ev)))
    if np.any(edge):
        if disc:
            t, z, ns = schur(a, output="real", sort=lambda re, im: np.hypot(re, im) < 1 - 1e-10)
        else:
            t, z, ns = schur(a, output="real", sort=lambda re, im: re < -1e-10 * max(1.0, np.hypot(re, im)))
    a11, a12, a22 = t[:ns, :ns], t[:ns, ns:], t[ns:, ns:]
    if ns and ns < n:
        x = solve_sylvester(a11, -a22, -a12)
    else:
        x = np.zeros((ns, n - ns))
    w = np.eye(n)
    w[:ns, ns:] = x
    winv = np.eye(n)
    winv[:ns, ns:] = -x
    tfull = z @ w
    b2 = winv @ z.T @ b
    c2 = c @ tfull
    return (a11, b2[:ns], c2[:, :ns]), (a22, b2[ns:], c2[:, ns:]), tfull


def _balance(a, b, c, disc: bool):
    """Balanced realization of a stable minimal model: A, B, C, Hankel values, T and its inverse."""
    n = a.shape[0]
    if n == 0:
        return a, b, c, np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0))
    lc = _chol_factor(cn.gram(a, b, disc))
    lo = _chol_factor(cn.gram(a.conj().T, c.conj().T, disc))
    u, s, vt = np.linalg.svd(lo.conj().T @ lc)
    if s.min() <= 1e-13 * max(s.max(), 1e-300):
        raise MatlabError("The model is not minimal (a Hankel singular value is zero), so it cannot be "
                          "balanced; remove the extra states first with minreal(sys).")
    sq = 1.0 / np.sqrt(s)
    t = (sq[:, None] * u.conj().T) @ lo.conj().T
    ti = lc @ vt.conj().T * sq[None, :]
    return t @ a @ ti, t @ b, c @ ti, s, t, ti


def _balance_full(s: SS):
    """Balance the stable part; unstable modes come first with infinite Hankel values."""
    (a1, b1, c1), (a2, b2, c2), tsplit = stab_split(s.a, s.b, s.c, s.is_dt)
    ab, bb, cb, g, t, ti = _balance(a1, b1, c1, s.is_dt)
    nu_ = a2.shape[0]
    n = s.nx
    a = cn.blkdiag(a2, ab)
    b = np.vstack([b2, bb])
    c = np.hstack([c2, cb])
    hsv = np.concatenate([np.full(nu_, np.inf), g])
    # x = tsplit [xs; xu],  xs = ti xb   =>   x = tsplit [ti 0; 0 I] [xb; xu]
    ns = a1.shape[0]
    big = np.zeros((n, n))
    big[:ns, nu_:] = ti
    big[ns:, :nu_] = np.eye(nu_)
    tinv_total = tsplit @ big                      # x = tinv_total * [xu; xb]
    t_total = np.linalg.inv(tinv_total) if n else tinv_total
    return a, b, c, hsv, t_total, tinv_total


@ltifn("balreal")
def _balreal(interp, args, nargout):
    sys = _need_dynamic(need_lti(args, "balreal"), "balreal")
    if len(args) > 1:
        raise MatlabError("balreal(sys) takes no options here.")
    s = to_ss(_no_delay_copy(sys))
    a, b, c, hsv, t, ti = _balance_full(s)
    out = _ss_like(s, a, b, c, s.d)
    out.in_delay, out.out_delay = sys.in_delay, sys.out_delay
    return (out, col(hsv), t, ti)[:max(nargout, 1)]


@ltifn("hsvd")
def _hsvd(interp, args, nargout):
    sys = _need_dynamic(need_lti(args, "hsvd"), "hsvd")
    s = to_ss(_no_delay_copy(sys))
    a, b, c, d = cn.ss_minreal(s.a, s.b, s.c, s.d)
    extra = s.nx - a.shape[0]
    hsv = _balance_full(SS(a, b, c, d, s.ts))[3]
    return col(np.concatenate([hsv, np.zeros(extra)]))


def _eliminate(a, b, c, d, keep: np.ndarray, drop: np.ndarray, method: str, disc: bool):
    a11, a12 = a[np.ix_(keep, keep)], a[np.ix_(keep, drop)]
    a21, a22 = a[np.ix_(drop, keep)], a[np.ix_(drop, drop)]
    b1, b2, c1, c2 = b[keep], b[drop], c[:, keep], c[:, drop]
    if method == "truncate" or drop.size == 0:
        return a11, b1, c1, d
    m = (np.eye(drop.size) - a22) if disc else -a22
    if np.linalg.cond(m) > 1e13:
        raise MatlabError("The states to eliminate contain an integrator (or a pole at z = 1), so the DC "
                          "gain cannot be matched; use the 'Truncate' method.")
    mi = np.linalg.inv(m)
    return a11 + a12 @ mi @ a21, b1 + a12 @ mi @ b2, c1 + c2 @ mi @ a21, d + c2 @ mi @ b2


def _elim_method(v: Any, name: str) -> str:
    m = text(v).lower()
    if m in ("matchdc", "mdc"):
        return "matchdc"
    if m in ("truncate", "del"):
        return "truncate"
    raise MatlabError(f"{name}: the state elimination method must be 'MatchDC' or 'Truncate'.")


@ltifn("balredOptions")
def _balred_options(interp, args, nargout):
    vals = {"StateElimMethod": char("MatchDC"), "StateProjection": char("MatchDC")}
    _pos, pairs = split_options(args)
    if _pos:
        raise MatlabError("balredOptions takes name/value pairs only.")
    for nm, v in pairs:
        if nm.lower() not in ("stateelimmethod", "stateprojection"):
            raise MatlabError(f"balredOptions: the option '{nm}' is not supported; only 'StateProjection' "
                              "(or its older name 'StateElimMethod') is.")
        m = "MatchDC" if _elim_method(v, "balredOptions") == "matchdc" else "Truncate"
        vals["StateElimMethod"] = vals["StateProjection"] = char(m)
    return Struct.scalar(vals)


@ltifn("balred")
def _balred(interp, args, nargout):
    sys = _need_dynamic(need_lti(args, "balred"), "balred")
    if len(args) < 2:
        raise MatlabError("Use balred(sys, order): the number of states to keep.")
    order = as_scalar(args[1], "order")
    if order != int(order) or order < 0:
        raise MatlabError("balred: the order must be a non-negative whole number.")
    order = int(order)
    method = "matchdc"
    rest = list(args[2:])
    if len(rest) == 1 and isinstance(rest[0], Struct):
        f = {k.lower(): rest[0].get_field(0, k) for k in rest[0].fields}
        key = "stateprojection" if "stateprojection" in f else "stateelimmethod"
        if key not in f:
            raise MatlabError("balred: unknown options structure; create it with balredOptions.")
        method = _elim_method(f[key], "balred")
    elif rest:
        _pos, pairs = split_options(rest)
        if _pos:
            raise MatlabError("balred: options must be name/value pairs or a balredOptions structure.")
        for nm, v in pairs:
            if nm.lower() not in ("stateelimmethod", "stateprojection"):
                raise MatlabError(f"balred: the option '{nm}' is not supported.")
            method = _elim_method(v, "balred")
    s0 = to_ss(_no_delay_copy(sys))
    a, b, c, d = cn.ss_minreal(s0.a, s0.b, s0.c, s0.d)
    s = SS(a, b, c, d, s0.ts)
    ab, bb, cb, hsv, _t, _ti = _balance_full(s)
    nunst = int(np.sum(np.isinf(hsv)))
    if order < nunst:
        raise MatlabError(f"balred: the order must be at least {nunst}, the number of unstable poles (they "
                          "are always kept).")
    order = min(order, s.nx)
    keep, drop = np.arange(order), np.arange(order, s.nx)
    ar, br, cr, dr = _eliminate(ab, bb, cb, s.d, keep, drop, method, s.is_dt)
    out = _ss_like(s0, ar, br, cr, dr)
    out.state_name, out.state_unit = [""] * out.nx, [""] * out.nx
    out.in_delay, out.out_delay, out.io_delay = sys.in_delay, sys.out_delay, np.zeros((sys.ny, sys.nu))
    res = back_to(out, sys)
    if not isinstance(res, SS):
        res.io_delay = sys.io_delay
    return res


@ltifn("modred")
def _modred(interp, args, nargout):
    sys = need_lti(args, "modred")
    if not isinstance(sys, SS):
        raise MatlabError("modred works on state-space models; convert with ss(sys) first.")
    if len(args) < 2:
        raise MatlabError("Use modred(sys, elim) with the indices of the states to eliminate.")
    e = np.asarray(args[1])
    if e.dtype == bool:
        drop = np.flatnonzero(e.ravel())
    else:
        drop = as_vector(args[1], "state indices").real.astype(int) - 1
    if drop.size and (drop.min() < 0 or drop.max() >= sys.nx):
        raise MatlabError(f"modred: state indices must lie between 1 and {sys.nx}.")
    method = _elim_method(args[2], "modred") if len(args) > 2 else "matchdc"
    keep = np.array([k for k in range(sys.nx) if k not in set(drop.tolist())], dtype=int)
    a, b, c, d = _eliminate(sys.a, sys.b, sys.c, sys.d, keep, np.unique(drop), method, sys.is_dt)
    out = _ss_like(sys, a, b, c, d)
    out.state_name = [sys.state_name[k] for k in keep]
    out.state_unit = [sys.state_unit[k] for k in keep]
    return out


@ltifn("stabsep")
def _stabsep(interp, args, nargout):
    sys = _need_dynamic(need_lti(args, "stabsep"), "stabsep")
    if len(args) > 1:
        raise MatlabError("stabsep(sys) takes no options here.")
    if sys.has_delay():
        raise MatlabError("stabsep does not support models with delays.")
    s = to_ss(sys)
    (a1, b1, c1), (a2, b2, c2), _t = stab_split(s.a, s.b, s.c, s.is_dt)
    gs = back_to(_ss_like(s, a1, b1, c1, s.d), sys)
    gu = back_to(_ss_like(s, a2, b2, c2, np.zeros_like(s.d)), sys)
    for g in (gs, gu):
        if isinstance(g, SS):
            g.state_name, g.state_unit = [""] * g.nx, [""] * g.nx
    return (gs, gu)[:max(nargout, 1)]


# -- delays -----------------------------------------------------------------------------------

def _pade_orders(v: Any, n: int, what: str) -> np.ndarray:
    o = as_vector(v, what).real.astype(float)
    if o.size == 1:
        o = np.full(n, o[0])
    if o.size != n:
        raise MatlabError(f"pade: {what} needs one order per channel ({n}) or a scalar.")
    if np.any((o < 0) | ((o != np.round(o)) & np.isfinite(o))):
        raise MatlabError("pade: approximation orders must be non-negative whole numbers (or Inf to keep "
                          "a delay).")
    return o


def pade_model(sys: LTI, nu_order, ny_order, nio_order) -> LTI:
    sys = _need_dynamic(sys, "pade")
    if not sys.has_delay():
        return sys
    if sys.is_dt:
        return absorb_delay(sys)
    ou = _pade_orders(nu_order, sys.nu, "the input order")
    oy = _pade_orders(ny_order, sys.ny, "the output order")
    core = _no_delay_copy(sys)

    def approx(tau: float, order: float) -> TF:
        if tau == 0 or np.isinf(order):
            return TF.siso([1.0], [1.0])
        num, den = cn.pade_poly(tau, int(order))
        return TF.siso(num, den)

    out = core
    keep_io = np.zeros((sys.ny, sys.nu))
    if np.any(sys.io_delay):
        oio = np.broadcast_to(np.atleast_2d(as_matrix(nio_order, "the I/O order").real), (sys.ny, sys.nu)) \
            if np.size(nio_order) in (1, sys.ny * sys.nu) else None
        if oio is None:
            raise MatlabError("pade: the I/O delay order must be a scalar or an ny-by-nu matrix.")
        rows = []
        for i in range(sys.ny):
            rows.append([approx(sys.io_delay[i, j], oio[i, j]) for j in range(sys.nu)])
            for j in range(sys.nu):
                if np.isinf(oio[i, j]):
                    keep_io[i, j] = sys.io_delay[i, j]
        from .control import lti_hcat, lti_vcat
        pio = lti_vcat([lti_hcat(r) for r in rows])
        out = lti_times(out, pio)
    if np.any(sys.in_delay):
        out = lti_mul(out, lti_append([approx(sys.in_delay[j], ou[j]) for j in range(sys.nu)]))
    if np.any(sys.out_delay):
        out = lti_mul(lti_append([approx(sys.out_delay[i], oy[i]) for i in range(sys.ny)]), out)
    out = out.clone()
    out.copy_meta(sys)
    out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
    out.in_delay = np.where(np.isinf(ou), sys.in_delay, 0.0)
    out.out_delay = np.where(np.isinf(oy), sys.out_delay, 0.0)
    if not isinstance(out, SS):
        out.io_delay = keep_io
    if hasattr(sys, "var") and hasattr(out, "var"):
        out.var = sys.var
    return out


@ltifn("pade")
def _pade(interp, args, nargout):
    if not args:
        raise MatlabError("Use [num, den] = pade(T, N) for a delay T, or sysx = pade(sys, N).")
    if isinstance(args[0], LTI):
        sys = args[0]
        if len(args) == 2:
            return pade_model(sys, args[1], args[1], args[1])
        if len(args) == 4:
            return pade_model(sys, args[1], args[2], args[3])
        raise MatlabError("Use pade(sys, N) or pade(sys, NU, NY, NINT).")
    if len(args) != 2:
        raise MatlabError("Use [num, den] = pade(T, N).")
    tau = as_scalar(args[0], "delay T")
    n = as_scalar(args[1], "order N")
    if tau < 0 or n < 0 or n != int(n):
        raise MatlabError("pade: the delay must not be negative and the order must be a whole number >= 0.")
    num, den = cn.pade_poly(tau, int(n))
    return (row(num), row(den))[:max(nargout, 1)] if nargout > 1 else row(num)


# -- discretization ---------------------------------------------------------------------------

_C2D_METHODS = {"zoh": "zoh", "foh": "foh", "impulse": "impulse", "imp": "impulse", "tustin": "tustin",
                "bilinear": "tustin", "matched": "matched", "prewarp": "prewarp"}


def _method_args(args: list, name: str, allowed: tuple):
    """Method and prewarp frequency from the trailing arguments of c2d / d2c / d2d."""
    method, w0 = "zoh", None
    if args:
        a = args[0]
        if isinstance(a, Struct):
            f = {k.lower(): a.get_field(0, k) for k in a.fields}
            if "method" not in f:
                raise MatlabError(f"{name}: unknown options structure; create it with c2dOptions or d2cOptions.")
            method = text(f["method"]).lower()
            pw = f.get("prewarpfrequency")
            if pw is not None and not is_empty_num(pw) and as_scalar(pw, "PrewarpFrequency") > 0:
                w0 = as_scalar(pw, "PrewarpFrequency")
            if len(args) > 1:
                raise MatlabError(f"Too many arguments for {name}.")
        elif is_text(a):
            method = text(a).lower()
            if len(args) > 2:
                raise MatlabError(f"Too many arguments for {name}.")
            if len(args) == 2:
                w0 = as_scalar(args[1], "prewarp frequency")
        else:
            raise MatlabError(f"{name}: the method must be text such as 'zoh' or 'tustin'.")
    if method not in _C2D_METHODS:
        if method in ("least-squares", "damped"):
            raise MatlabError(f"{name}: the method '{method}' is not supported; use 'zoh', 'foh', 'impulse', "
                              "'tustin' or 'matched'.")
        raise MatlabError(f"{name}: unknown method '{method}'. Use 'zoh', 'foh', 'impulse', 'tustin' or "
                          "'matched'.")
    method = _C2D_METHODS[method]
    if method == "prewarp":
        if w0 is None:
            raise MatlabError(f"{name}: the 'prewarp' method needs the critical frequency, e.g. "
                              f"{name}(sys, Ts, 'prewarp', w0).")
        method = "tustin"
    elif w0 is not None and method != "tustin":
        raise MatlabError(f"{name}: a prewarp frequency only applies to the 'tustin' method.")
    if method not in allowed:
        raise MatlabError(f"{name}: the method '{method}' is not available here; use one of "
                          + ", ".join(f"'{m}'" for m in allowed) + ".")
    if w0 is not None and w0 <= 0:
        raise MatlabError(f"{name}: the prewarp frequency must be positive.")
    return method, w0


def _tustin_alpha(ts: float, w0: float | None) -> float:
    if w0 is None:
        return 2.0 / ts
    if w0 * ts >= np.pi:
        raise MatlabError("The prewarp frequency must be below the Nyquist frequency pi/Ts.")
    return w0 / np.tan(w0 * ts / 2)


def c2d_matrices(a, b, c, d, ts: float, method: str, alpha: float | None = None):
    n = a.shape[0]
    if method == "zoh":
        phi, gam = cn.zoh_matrices(a, b, ts)
        return phi, gam, c, d
    if method == "foh":
        phi, g1, g2 = cn.foh_matrices(a, b, ts)
        return phi, g1 + phi @ g2 - g2, c, d + c @ g2
    if method == "impulse":
        if np.any(d != 0):
            raise MatlabError("c2d: the impulse-invariant method needs a strictly proper model (no direct "
                              "feedthrough).")
        phi = cn.expm(a * ts)
        return phi, ts * (phi @ b), c, ts * (c @ b)
    if method == "tustin":
        m = alpha * np.eye(n) - a
        if n and np.linalg.cond(m) > 1e14:
            raise MatlabError("c2d: the Tustin method fails because the model has a pole at s = 2/Ts.")
        mi = np.linalg.inv(m) if n else m
        r = np.sqrt(2 * alpha)
        return (alpha * np.eye(n) + a) @ mi, r * (mi @ b), r * (c @ mi), d + c @ mi @ b
    raise MatlabError(f"c2d: unknown method '{method}'.")


def d2c_matrices(a, b, c, d, ts: float, method: str, alpha: float | None = None):
    n, m = a.shape[0], b.shape[1]
    if method == "zoh":
        if n == 0:
            return a, b, c, d
        ev = np.linalg.eigvals(a)
        if np.any(np.abs(ev) < 1e-12):
            raise MatlabError("d2c: a model with poles at z = 0 has no continuous-time equivalent under "
                              "the zero-order hold; remove pure delays first or use 'tustin'.")
        if np.any((np.abs(ev.imag) < 1e-12 * np.abs(ev)) & (ev.real < 0)):
            raise MatlabError("d2c: a pole on the negative real axis has no real continuous-time "
                              "equivalent of the same order under the zero-order hold; use 'tustin'.")
        from scipy.linalg import logm
        big = np.zeros((n + m, n + m))
        big[:n, :n] = a
        big[:n, n:] = b
        big[n:, n:] = np.eye(m)
        lg = logm(big) / ts
        lg = np.real(lg) if np.max(np.abs(np.imag(lg))) <= 1e-9 * max(1.0, float(np.max(np.abs(lg)))) else lg
        if np.iscomplexobj(lg):
            raise MatlabError("d2c: the matrix logarithm is complex, so there is no real continuous-time "
                              "equivalent; use 'tustin'.")
        return lg[:n, :n], lg[:n, n:], c, d
    if method == "tustin":
        mm = a + np.eye(n)
        if n and np.linalg.cond(mm) > 1e14:
            raise MatlabError("d2c: the Tustin method fails because the model has a pole at z = -1.")
        mi = np.linalg.inv(mm) if n else mm
        r = np.sqrt(2 * alpha)
        return alpha * (a - np.eye(n)) @ mi, r * (mi @ b), r * (c @ mi), d - c @ mi @ b
    raise MatlabError(f"d2c: unknown method '{method}'.")


def _poly_sub(p: np.ndarray, top: np.ndarray, bottom: np.ndarray, big: int) -> np.ndarray:
    """p(top/bottom) * bottom^big for polynomials top, bottom."""
    out = np.zeros(1)
    deg = p.size - 1
    for idx, coef in enumerate(p):
        k = deg - idx
        term = np.array([coef])
        for _ in range(k):
            term = np.convolve(term, top)
        for _ in range(big - k):
            term = np.convolve(term, bottom)
        out = np.polyadd(out, term)
    return out


def _tustin_poly(num, den, alpha: float, forward: bool):
    big = max(num.size, den.size) - 1
    if forward:    # s = alpha (z - 1)/(z + 1)
        top, bottom = alpha * np.array([1.0, -1.0]), np.array([1.0, 1.0])
    else:          # z = (alpha + s)/(alpha - s)
        top, bottom = np.array([1.0, alpha]), np.array([-1.0, alpha])
    n, d = _poly_sub(num, top, bottom, big), _poly_sub(den, top, bottom, big)
    d = cn.ptrim(d)
    return n / d[0], d / d[0]


def _matched_zpk(z, p, k, ts: float, forward: bool):
    z, p = np.asarray(z, dtype=complex), np.asarray(p, dtype=complex)
    if forward:
        zd, pd = np.exp(z * ts), np.exp(p * ts)
        extra = max(len(p) - len(z) - 1, 0)
        zd = np.concatenate([zd, -np.ones(extra)])
        z0, p0, origin_new = z == 0, p == 0, 1.0
        m = int(np.sum(z0)) - int(np.sum(p0))
        dc = k * np.prod(-z[~z0]) / np.prod(-p[~p0])
        zn, pn = zd[np.abs(zd - 1) > 1e-12], pd[np.abs(pd - 1) > 1e-12]
        kd = dc / (np.prod(origin_new - zn) / np.prod(origin_new - pn)) / ts ** m
        return cn.pair_conjugates(zd), cn.pair_conjugates(pd), float(np.real(kd))
    if np.any(np.abs(p) < 1e-12) or np.any(np.abs(z) < 1e-12):
        raise MatlabError("d2c: the matched method cannot map poles or zeros at z = 0.")
    keep = np.abs(z + 1) > 1e-9
    zc, pc = np.log(z[keep]) / ts, np.log(p) / ts
    z1, p1 = np.abs(z - 1) <= 1e-12, np.abs(p - 1) <= 1e-12
    m = int(np.sum(z1)) - int(np.sum(p1))
    dc = k * np.prod(1 - z[~z1]) / np.prod(1 - p[~p1])
    zn, pn = zc[np.abs(zc) > 1e-12], pc[np.abs(pc) > 1e-12]
    kc = dc / (np.prod(-zn) / np.prod(-pn)) * ts ** m
    return cn.pair_conjugates(zc), cn.pair_conjugates(pc), float(np.real(kc))


def _frac_delay_zoh(a, b, c, d, ts: float, tau_in: np.ndarray, tau_out: np.ndarray):
    """ZOH discretization with input/output delays that are fractions of the sample time (< Ts)."""
    n, m, p = a.shape[0], b.shape[1], c.shape[0]
    phi, gam = cn.zoh_matrices(a, b, ts)
    jin = np.flatnonzero(tau_in > 0)
    g0, g1 = gam.copy(), np.zeros((n, jin.size))
    for col_, j in enumerate(jin):
        # u_j arrives tau late: the old value acts for tau, the new one for Ts - tau
        _phi_a, gam_a = cn.zoh_matrices(a, b[:, [j]], ts - tau_in[j])
        phi_b, gam_b = cn.zoh_matrices(a, b[:, [j]], tau_in[j])
        g0[:, j] = gam_a[:, 0]
        g1[:, col_] = (cn.expm(a * (ts - tau_in[j])) @ gam_b)[:, 0]
        del phi_b
    ad = np.block([[phi, g1], [np.zeros((jin.size, n)), np.zeros((jin.size, jin.size))]])
    sel = np.zeros((jin.size, m))
    sel[np.arange(jin.size), jin] = 1.0
    bd = np.vstack([g0, sel])
    dcur = d.copy()
    dold = np.zeros((p, jin.size))
    for col_, j in enumerate(jin):
        dold[:, col_] = d[:, j]
        dcur[:, j] = 0.0
    cd = np.hstack([c, dold])
    dd = dcur
    iout = np.flatnonzero(tau_out > 0)
    if iout.size:
        # output i shows the response at a time (Ts - tau) after the previous sample: an extra state
        # holds that value, which already accounts for the fraction of a sample
        na = ad.shape[0]
        rows_c, rows_d = [], []
        for i in iout:
            sig = ts - tau_out[i]
            # continuous state and held inputs at that intermediate time
            phis, gams = cn.zoh_matrices(a, b, sig)
            xrow = np.zeros((1, na))
            urow = np.zeros((1, m))
            g_now = gams.copy()
            g_old = np.zeros((n, jin.size))
            d_now = d[[i], :].copy()
            d_old = np.zeros((1, jin.size))
            for col_, j in enumerate(jin):
                if sig <= tau_in[j]:
                    g_now[:, j] = 0.0
                    g_old[:, col_] = gams[:, j]
                    d_now[0, j] = 0.0
                    d_old[0, col_] = d[i, j]
                else:
                    _pa, ga = cn.zoh_matrices(a, b[:, [j]], sig - tau_in[j])
                    _pb, gb = cn.zoh_matrices(a, b[:, [j]], tau_in[j])
                    g_now[:, j] = ga[:, 0]
                    g_old[:, col_] = (cn.expm(a * (sig - tau_in[j])) @ gb)[:, 0]
            xrow[0, :n] = (c[[i], :] @ phis)[0]
            xrow[0, n:] = (c[[i], :] @ g_old + d_old)[0]
            urow[0, :] = (c[[i], :] @ g_now + d_now)[0]
            rows_c.append(xrow)
            rows_d.append(urow)
        k = iout.size
        ad2 = np.block([[ad, np.zeros((na, k))], [np.vstack(rows_c), np.zeros((k, k))]])
        bd2 = np.vstack([bd, np.vstack(rows_d)])
        cd2 = np.hstack([cd, np.zeros((p, k))])
        dd2 = dd.copy()
        for col_, i in enumerate(iout):
            cd2[i, :] = 0.0
            cd2[i, na + col_] = 1.0
            dd2[i, :] = 0.0
        ad, bd, cd, dd = ad2, bd2, cd2, dd2
    return ad, bd, cd, dd


def _split_delay(tau: np.ndarray, ts: float):
    """Whole samples and the remaining fraction (in time units) of each delay."""
    ratio = tau / ts
    whole = np.floor(ratio + 1e-9)
    frac = ratio - whole
    frac[np.abs(frac) < 1e-9] = 0.0
    return whole, frac * ts


def _c2d_siso_tf(num, den, ts: float, method: str, alpha: float | None, tau: float, interp):
    """One transfer function entry: discrete numerator, denominator and delay in samples."""
    whole, frac = _split_delay(np.array([tau]), ts)
    delay = float(whole[0])
    if cn.piszero(num):
        return np.zeros(1), np.ones(1), 0.0
    if method == "tustin":
        if frac[0]:
            delay = float(np.round(tau / ts))
            _warn_round(interp)
        n, d = _tustin_poly(num, den, alpha, True)
        return n, d, delay
    if method == "matched":
        if frac[0]:
            delay = float(np.round(tau / ts))
            _warn_round(interp)
        if num.size > den.size:
            raise MatlabError("c2d: the matched method needs a proper model.")
        zd, pd, kd = _matched_zpk(cn.proots(num), cn.proots(den), num[0] / den[0], ts, True)
        return kd * cn.rpoly(zd), cn.rpoly(pd), delay
    a, b, c, d = cn.tf2ss_siso(num, den)
    if frac[0] and method == "zoh":
        ad, bd, cd, dd = _frac_delay_zoh(a, b, c, d, ts, frac, np.zeros(1))
        nd, dn = cn.ss_numden(ad, bd, cd, dd)
        n, dnn = _strip_common_z(nd[0, 0], dn)
        return n, dnn, delay
    if frac[0]:
        delay = float(np.round(tau / ts))
        _warn_round(interp)
    ad, bd, cd, dd = c2d_matrices(a, b, c, d, ts, method)
    nd, dn = cn.ss_numden(ad, bd, cd, dd)
    return _clean_poly(nd[0, 0]), dn, delay


def _strip_common_z(num: np.ndarray, den: np.ndarray):
    """Cancel factors of z shared by numerator and denominator."""
    num, den = np.asarray(num), np.asarray(den)
    scale = max(float(np.max(np.abs(num))), 1e-300)
    while num.size > 1 and den.size > 1 and abs(num[-1]) <= 1e-14 * scale and den[-1] == 0:
        num, den = num[:-1], den[:-1]
    return num, den


def _warn_round(interp) -> None:
    if interp is not None:
        interp.warn("A delay that is not a multiple of the sample time was rounded to the nearest number of "
                    "samples; only the 'zoh' method treats fractional delays exactly.",
                    "Control:transformation:C2DFractionalDelay")


def c2d_model(sys: LTI, ts: float, method: str, w0: float | None, interp=None) -> LTI:
    if isinstance(sys, FRD):
        raise MatlabError("c2d does not apply to frequency-response data.")
    if sys.is_dt:
        raise MatlabError("c2d needs a continuous-time model; this one is already discrete (use d2d to "
                          "change its sample time).")
    alpha = _tustin_alpha(ts, w0) if method == "tustin" else None
    if isinstance(sys, PID):
        if method == "matched":
            raise MatlabError("c2d: the matched method is not available for pid objects; convert with "
                              "tf(C) first.")
        formula = {"zoh": "ForwardEuler", "foh": "Trapezoidal", "tustin": "Trapezoidal",
                   "impulse": "BackwardEuler"}[method]
        out = sys.clone()
        out.ts = ts
        out.iformula = out.dformula = formula
        for d, nm in ((sys.in_delay, "in_delay"), (sys.out_delay, "out_delay")):
            setattr(out, nm, np.round(d / ts))
        return out
    if isinstance(sys, SS):
        if method == "matched":
            if not sys.is_siso:
                raise MatlabError("c2d: the matched method works for SISO models only.")
            return to_ss(c2d_model(to_zpk(sys), ts, method, w0, interp))
        win, fin = _split_delay(sys.in_delay, ts)
        wout, fout = _split_delay(sys.out_delay, ts)
        if (np.any(fin) or np.any(fout)) and method == "zoh":
            ad, bd, cd, dd = _frac_delay_zoh(sys.a, sys.b, sys.c, sys.d, ts, fin, fout)
        else:
            if np.any(fin) or np.any(fout):
                win, wout = np.round(sys.in_delay / ts), np.round(sys.out_delay / ts)
                _warn_round(interp)
            ad, bd, cd, dd = c2d_matrices(sys.a, sys.b, sys.c, sys.d, ts, method, alpha)
        out = SS(ad, bd, cd, dd, ts)
        out.copy_meta(sys)
        out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
        if out.nx == sys.nx:
            out.state_name, out.state_unit = list(sys.state_name), list(sys.state_unit)
        out.in_delay, out.out_delay = win, wout
        return out
    g = to_tf(sys)
    total = g.total_delay()
    num = _grid(g.ny, g.nu, lambda i, j: None)
    den = _grid(g.ny, g.nu, lambda i, j: None)
    delays = np.zeros((g.ny, g.nu))
    for i in range(g.ny):
        for j in range(g.nu):
            n_ij, d_ij = g.num[i][j], g.den[i][j]
            if np.iscomplexobj(n_ij) or np.iscomplexobj(d_ij):
                raise MatlabError("c2d needs a model with real coefficients.")
            if n_ij.size > d_ij.size and method != "tustin":
                raise MatlabError("c2d: the model is improper (more zeros than poles); only the 'tustin' "
                                  "method can discretize it.")
            num[i][j], den[i][j], delays[i, j] = _c2d_siso_tf(n_ij, d_ij, ts, method, alpha, total[i, j], interp)
    out = TF(num, den, ts)
    out.copy_meta(sys)
    out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
    from .control import nonzero_mask, split_delays
    out.in_delay, out.out_delay, out.io_delay = split_delays(delays, nonzero_mask(out))
    if isinstance(sys, ZPK):
        res = to_zpk(out)
        if method in ("tustin", "matched") and not np.any(delays):
            res = _c2d_zpk_exact(sys, ts, method, alpha, res)
        return res
    return out


def _c2d_zpk_exact(sys: ZPK, ts: float, method: str, alpha: float | None, fallback: ZPK) -> ZPK:
    """Tustin and matched discretization of zero-pole-gain data root by root."""
    z = _grid(sys.ny, sys.nu, lambda i, j: None)
    p = _grid(sys.ny, sys.nu, lambda i, j: None)
    k = np.zeros((sys.ny, sys.nu))
    for i in range(sys.ny):
        for j in range(sys.nu):
            zz = np.asarray(sys.z[i][j], dtype=complex)
            pp = np.asarray(sys.p[i][j], dtype=complex)
            kk = sys.k[i, j]
            if kk == 0:
                z[i][j], p[i][j] = np.zeros(0), np.zeros(0)
                continue
            if method == "matched":
                z[i][j], p[i][j], k[i, j] = _matched_zpk(zz, pp, kk, ts, True)
                continue
            if np.any(np.abs(alpha - pp) < 1e-12) or len(zz) > len(pp):
                return fallback
            zd = np.concatenate([(alpha + zz) / (alpha - zz), -np.ones(len(pp) - len(zz))])
            pd = (alpha + pp) / (alpha - pp)
            z[i][j], p[i][j] = cn.pair_conjugates(zd), cn.pair_conjugates(pd)
            k[i, j] = float(np.real(kk * np.prod(alpha - zz) / np.prod(alpha - pp)))
    out = ZPK(z, p, k, ts)
    _carry(out, fallback)
    return out


@ltifn("c2d")
def _c2d(interp, args, nargout):
    sys = need_lti(args, "c2d")
    if len(args) < 2:
        raise MatlabError("Use c2d(sys, Ts) or c2d(sys, Ts, method).")
    ts = as_scalar(args[1], "sample time Ts")
    if not (ts > 0) or not np.isfinite(ts):
        raise MatlabError("c2d: the sample time must be positive and finite.")
    method, w0 = _method_args(list(args[2:]), "c2d", ("zoh", "foh", "impulse", "tustin", "matched"))
    out = c2d_model(sys, ts, method, w0, interp)
    if nargout > 1:
        if not isinstance(sys, SS) or method not in ("zoh", "foh") or out.nx != sys.nx:
            raise MatlabError("c2d: the matrix that maps initial conditions is available for state-space "
                              "models with the 'zoh' and 'foh' methods (without fractional delays).")
        g = np.hstack([np.eye(sys.nx), np.zeros((sys.nx, sys.nu))])
        if method == "foh":
            g[:, sys.nx:] = -cn.foh_matrices(sys.a, sys.b, ts)[2]
        return out, g
    return out


def d2c_model(sys: LTI, method: str, w0: float | None) -> LTI:
    if isinstance(sys, FRD):
        raise MatlabError("d2c does not apply to frequency-response data.")
    if not sys.is_dt:
        raise MatlabError("d2c needs a discrete-time model; this one is continuous.")
    if sys.ts < 0:
        raise MatlabError("d2c: the sample time of the model is unspecified (Ts = -1); set sys.Ts first.")
    ts = sys.ts
    alpha = _tustin_alpha(ts, w0) if method == "tustin" else None
    if isinstance(sys, PID):
        out = sys.clone()
        out.ts = 0.0
        out.iformula = out.dformula = "ForwardEuler"
        out.in_delay, out.out_delay = sys.in_delay * ts, sys.out_delay * ts
        return out
    if isinstance(sys, SS):
        if method == "matched":
            if not sys.is_siso:
                raise MatlabError("d2c: the matched method works for SISO models only.")
            return to_ss(d2c_model(to_zpk(sys), method, w0))
        a, b, c, d = d2c_matrices(sys.a, sys.b, sys.c, sys.d, ts, method, alpha)
        out = SS(a, b, c, d, 0.0)
        out.copy_meta(sys)
        out.state_name, out.state_unit = list(sys.state_name), list(sys.state_unit)
        out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
        out.in_delay, out.out_delay = sys.in_delay * ts, sys.out_delay * ts
        return out
    g = to_tf(sys)
    num = _grid(g.ny, g.nu, lambda i, j: None)
    den = _grid(g.ny, g.nu, lambda i, j: None)
    for i in range(g.ny):
        for j in range(g.nu):
            n_ij, d_ij = g.num[i][j], g.den[i][j]
            if cn.piszero(n_ij):
                num[i][j], den[i][j] = np.zeros(1), np.ones(1)
            elif method == "tustin":
                num[i][j], den[i][j] = _tustin_poly(n_ij, d_ij, alpha, False)
            elif method == "matched":
                zc, pc, kc = _matched_zpk(cn.proots(n_ij), cn.proots(d_ij), n_ij[0] / d_ij[0], ts, False)
                num[i][j], den[i][j] = kc * cn.rpoly(zc), cn.rpoly(pc)
            else:
                if n_ij.size > d_ij.size:
                    raise MatlabError("d2c: the model is improper (not causal).")
                a, b, c, d = d2c_matrices(*cn.tf2ss_siso(n_ij, d_ij), ts, "zoh")
                nd, dn = cn.ss_numden(a, b, c, d)
                num[i][j], den[i][j] = _clean_poly(nd[0, 0]), dn
    out = TF(num, den, 0.0)
    out.copy_meta(sys)
    out.name, out.notes, out.userdata = sys.name, sys.notes, sys.userdata
    out.in_delay, out.out_delay, out.io_delay = sys.in_delay * ts, sys.out_delay * ts, sys.io_delay * ts
    return to_zpk(out) if isinstance(sys, ZPK) else out


@ltifn("d2c")
def _d2c(interp, args, nargout):
    sys = need_lti(args, "d2c")
    method, w0 = _method_args(list(args[1:]), "d2c", ("zoh", "tustin", "matched"))
    return d2c_model(sys, method, w0)


@ltifn("d2d")
def _d2d(interp, args, nargout):
    sys = need_lti(args, "d2d")
    if len(args) < 2:
        raise MatlabError("Use d2d(sys, Ts) or d2d(sys, Ts, method).")
    ts2 = as_scalar(args[1], "sample time Ts")
    if not (ts2 > 0):
        raise MatlabError("d2d: the new sample time must be positive.")
    method, w0 = _method_args(list(args[2:]), "d2d", ("zoh", "tustin"))
    if isinstance(sys, FRD) or not sys.is_dt or sys.ts < 0:
        raise MatlabError("d2d needs a discrete-time model with a known sample time.")
    ts1 = sys.ts
    if abs(ts2 - ts1) <= 1e-12 * ts1:
        return sys
    src = pid_to_tf(sys) if isinstance(sys, PID) else sys
    ratio = ts2 / ts1
    core = _no_delay_copy(src)
    if method == "zoh" and abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
        k = int(round(ratio))
        s = to_ss(core)
        ak = np.linalg.matrix_power(s.a, k) if s.nx else s.a
        acc = np.zeros_like(s.a)
        pw = np.eye(s.nx)
        for _ in range(k):
            acc = acc + pw
            pw = pw @ s.a
        res: LTI = back_to(_ss_like(s, ak, acc @ s.b, s.c, s.d, ts2), src)
    else:
        res = c2d_model(d2c_model(core, method, w0), ts2, method, w0, interp)
    res = res.clone()
    res.ts = ts2
    for nm in ("in_delay", "out_delay", "io_delay"):
        d = getattr(src, nm) * ts1 / ts2
        if np.any(np.abs(d - np.round(d)) > 1e-9):
            _warn_round(interp)
        if nm != "io_delay" or not isinstance(res, SS):
            setattr(res, nm, np.round(d))
    return res


@ltifn("c2dOptions", "d2cOptions", "d2dOptions")
def _c2d_options(interp, args, nargout):
    vals = {"Method": char("zoh"), "PrewarpFrequency": np.zeros((1, 1))}
    _pos, pairs = split_options(args)
    if _pos:
        raise MatlabError("c2dOptions takes name/value pairs only.")
    for nm, v in pairs:
        key = nm.lower()
        if key == "method":
            m = text(v).lower()
            if m not in _C2D_METHODS or m == "prewarp":
                raise MatlabError("c2dOptions: Method must be 'zoh', 'foh', 'impulse', 'tustin' or 'matched'.")
            vals["Method"] = char(_C2D_METHODS[m])
        elif key == "prewarpfrequency":
            vals["PrewarpFrequency"] = np.array([[as_scalar(v, "PrewarpFrequency")]])
        else:
            raise MatlabError(f"c2dOptions: the option '{nm}' is not supported; the options are 'Method' and "
                              "'PrewarpFrequency'.")
    return Struct.scalar(vals)


# -- controllability, observability, canonical forms ------------------------------------------

def _ab_args(args: list, name: str, second: str):
    """(A, B) or (A, C) from a model or from two matrices."""
    if args and isinstance(args[0], LTI):
        if len(args) > 1:
            raise MatlabError(f"{name}(sys) takes one model.")
        s = to_ss(_need_dynamic(args[0], name))
        return s.a, (s.b if second == "B" else s.c)
    if len(args) != 2:
        raise MatlabError(f"Use {name}(A, {second}) or {name}(sys).")
    a, m = as_matrix(args[0], "A"), as_matrix(args[1], second)
    n = a.shape[0]
    if a.shape != (n, n):
        raise MatlabError(f"{name}: A must be square.")
    if (second == "B" and m.shape[0] != n) or (second == "C" and m.shape[1] != n):
        raise MatlabError(f"{name}: A and {second} have incompatible sizes.")
    return a, m


@ltifn("ctrb")
def _ctrb(interp, args, nargout):
    a, b = _ab_args(args, "ctrb", "B")
    blocks, cur = [], b
    for _ in range(a.shape[0]):
        blocks.append(cur)
        cur = a @ cur
    return np.hstack(blocks) if blocks else np.zeros((0, 0))


@ltifn("obsv")
def _obsv(interp, args, nargout):
    a, c = _ab_args(args, "obsv", "C")
    blocks, cur = [], c
    for _ in range(a.shape[0]):
        blocks.append(cur)
        cur = cur @ a
    return np.vstack(blocks) if blocks else np.zeros((0, 0))


def _abc_args(args: list, name: str):
    if len(args) < 3:
        raise MatlabError(f"Use [Abar, Bbar, Cbar, T, k] = {name}(A, B, C).")
    a, b, c = as_matrix(args[0], "A"), as_matrix(args[1], "B"), as_matrix(args[2], "C")
    n = a.shape[0]
    if a.shape != (n, n) or b.shape[0] != n or c.shape[1] != n:
        raise MatlabError(f"{name}: A, B and C have incompatible sizes.")
    tol = as_scalar(args[3], "tolerance") if len(args) > 3 and not is_empty_num(args[3]) else None
    return a, b, c, tol


def _ctrbf(a, b, c, tol):
    n = a.shape[0]
    eff = None if tol is None else tol
    t, nc, sizes = cn.ctrb_staircase(a, b, eff)
    perm = np.concatenate([np.arange(nc, n), np.arange(nc)]).astype(int)
    t = t[perm, :]
    k = np.zeros(n)
    k[:len(sizes)] = sizes
    return t @ a @ t.conj().T, t @ b, c @ t.conj().T, t, k


@ltifn("ctrbf")
def _ctrbf_fn(interp, args, nargout):
    a, b, c, tol = _abc_args(args, "ctrbf")
    ab, bb, cb, t, k = _ctrbf(a, b, c, tol)
    return (ab, bb, cb, t, row(k))[:max(nargout, 1)]


@ltifn("obsvf")
def _obsvf_fn(interp, args, nargout):
    a, b, c, tol = _abc_args(args, "obsvf")
    at, ct, bt, t, k = _ctrbf(a.conj().T, c.conj().T, b.conj().T, tol)
    return (at.conj().T, t @ b, c @ t.conj().T, t, row(k))[:max(nargout, 1)]


def modal_transform(a: np.ndarray, condt: float = 1e8):
    """Real T^-1 whose columns turn A into real modal form (2-by-2 blocks for complex pairs)."""
    n = a.shape[0]
    w, v = np.linalg.eig(a)
    cols_, used = [], np.zeros(n, dtype=bool)
    order = np.argsort(np.abs(w), kind="stable")
    for k in order:
        if used[k]:
            continue
        used[k] = True
        if abs(w[k].imag) <= 1e-12 * max(1.0, abs(w[k])):
            vec = v[:, k]
            ph = vec[np.argmax(np.abs(vec))]
            cols_.append(np.real(vec / ph * abs(ph)))
        else:
            partner = [j for j in range(n) if not used[j] and abs(w[j] - np.conj(w[k])) <= 1e-8 * abs(w[k])]
            if not partner:
                raise MatlabError("canon: complex eigenvalues of a real matrix must come in conjugate pairs.")
            used[partner[0]] = True
            vec = v[:, k] if w[k].imag > 0 else v[:, partner[0]]
            cols_ += [np.real(vec), np.imag(vec)]
    tinv = np.column_stack(cols_) if cols_ else np.zeros((0, 0))
    if n and np.linalg.cond(tinv) > condt:
        raise MatlabError("canon: the A matrix is defective or nearly so (repeated eigenvalues without "
                          "independent eigenvectors), so a modal form is not available.")
    return tinv


@ltifn("canon")
def _canon(interp, args, nargout):
    sys = _need_dynamic(need_lti(args, "canon"), "canon")
    kind = text(args[1]).lower() if len(args) > 1 else "modal"
    condt = as_scalar(args[2], "condt") if len(args) > 2 else 1e8
    s = to_ss(sys)
    if any(np.iscomplexobj(m) for m in s.abcd):
        raise MatlabError("canon needs a model with real coefficients.")
    if kind == "modal":
        tinv = modal_transform(s.a, condt)
    elif kind == "companion":
        if s.nu < 1:
            raise MatlabError("canon: the companion form needs at least one input.")
        cur, blocks = s.b[:, [0]], []
        for _ in range(s.nx):
            blocks.append(cur)
            cur = s.a @ cur
        tinv = np.hstack(blocks) if blocks else np.zeros((0, 0))
        if s.nx and np.linalg.matrix_rank(tinv) < s.nx:
            raise MatlabError("canon: the companion form needs a model that is controllable from its "
                              "first input.")
    else:
        raise MatlabError("canon: the form must be 'modal' or 'companion'.")
    t = np.linalg.inv(tinv) if s.nx else tinv
    a = t @ s.a @ tinv
    if kind == "companion":
        a[np.abs(a) < 1e-12 * max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)] = 0.0
    else:
        a = _clean_modal(a)
    out = _ss_like(s, a, t @ s.b, s.c @ tinv, s.d)
    return (out, t)[:max(nargout, 1)]


def _clean_modal(a: np.ndarray) -> np.ndarray:
    """Zero the off-block entries of a modal form that are rounding noise."""
    out = a.copy()
    scale = max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(out[i, j]) < 1e-9 * scale:
                out[i, j] = 0.0
            elif abs(i - j) == 1 and abs(out[i, j]) < 1e-12 * scale:
                out[i, j] = 0.0
    return out


@ltifn("ss2ss")
def _ss2ss(interp, args, nargout):
    sys = need_lti(args, "ss2ss")
    if not isinstance(sys, SS) or len(args) != 2:
        raise MatlabError("Use ss2ss(sys, T) with a state-space model and a square, invertible matrix T.")
    t = as_matrix(args[1], "T")
    if t.shape != (sys.nx, sys.nx) or (sys.nx and np.linalg.matrix_rank(t) < sys.nx):
        raise MatlabError(f"ss2ss: T must be an invertible {sys.nx}-by-{sys.nx} matrix.")
    ti = np.linalg.inv(t) if sys.nx else t
    return _ss_like(sys, t @ sys.a @ ti, t @ sys.b, sys.c @ ti, sys.d)


@ltifn("xperm")
def _xperm(interp, args, nargout):
    sys = need_lti(args, "xperm")
    if not isinstance(sys, SS) or len(args) != 2:
        raise MatlabError("Use xperm(sys, P) with a state-space model and a permutation P of its states.")
    p = as_vector(args[1], "permutation").real.astype(int) - 1
    if sorted(p.tolist()) != list(range(sys.nx)):
        raise MatlabError(f"xperm: P must be a permutation of 1..{sys.nx}.")
    out = _ss_like(sys, sys.a[np.ix_(p, p)], sys.b[p, :], sys.c[:, p], sys.d)
    out.state_name = [sys.state_name[k] for k in p]
    out.state_unit = [sys.state_unit[k] for k in p]
    return out
