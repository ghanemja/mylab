"""Control design: Lyapunov and Riccati equations, LQR/LQG, pole placement, estimators, PID tuning.

lyap, dlyap, lyapchol, dlyapchol, care, dare, icare, idare; lqr, dlqr, lqrd, lqry, lqi, lqe, dlqe,
kalman, lqg, lqgreg; place, acker, estim, reg; pidtune and pidtuneOptions; and the conversion
helpers ss2tf, tf2ss, zp2tf, tf2zp, zp2ss, ss2zp.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char
from . import _ctrl_num as cn
from .control import (
    FRD, LTI, PID, PIDSTD, SS, _no_delay_copy, as_matrix, as_scalar, as_vector, col, is_empty_num, is_num,
    lti_freqresp, ltifn, need_lti, pid_to_std, pid_to_tf, row, split_options, to_ss,
)
from ._util import bool_scalar, is_text, text


def _opt(args: list, k: int) -> Any:
    """Argument k, or None when it is missing or the empty placeholder []."""
    if len(args) > k and not is_empty_num(args[k]):
        return args[k]
    return None


def _mat(v: Any, what: str) -> np.ndarray:
    m = as_matrix(v, what)
    return m.real.copy() if np.iscomplexobj(m) and not np.any(m.imag) else m


def _sym_weight(v: Any, n: int, what: str) -> np.ndarray:
    m = _mat(v, what)
    if m.size == 1 and n != 1:
        m = m.flat[0] * np.eye(n)
    if m.shape != (n, n):
        raise MatlabError(f"{what} must be a {n}-by-{n} matrix.")
    return m


def _plant(args: list, name: str, need_d: bool = False):
    """(A, B, rest, discrete, sys) from either (A, B, ...) or (sys, ...)."""
    if args and isinstance(args[0], LTI):
        s = to_ss(args[0])
        if s.has_delay():
            raise MatlabError(f"{name} does not support models with time delays; approximate them with pade "
                              "or absorb them (discrete time) first.")
        return s, list(args[1:])
    return None, list(args)


# -- Lyapunov equations -----------------------------------------------------------------------

@ltifn("lyap")
def _lyap(interp, args, nargout):
    if len(args) == 2:
        return cn.lyap_c(_mat(args[0], "A"), _mat(args[1], "Q"))
    if len(args) == 3:
        return cn.sylvester(_mat(args[0], "A"), _mat(args[1], "B"), _mat(args[2], "C"))
    if len(args) == 4:
        if not is_empty_num(args[2]):
            raise MatlabError("lyap(A, Q, [], E): the third argument must be empty.")
        a, q, e = _mat(args[0], "A"), _mat(args[1], "Q"), _mat(args[3], "E")
        if e.shape != a.shape or np.linalg.matrix_rank(e) < e.shape[0]:
            raise MatlabError("lyap: E must be square, of the size of A, and nonsingular.")
        ei = np.linalg.inv(e)
        return cn.lyap_c(ei @ a, ei @ q @ ei.conj().T)
    raise MatlabError("Use lyap(A, Q), lyap(A, B, C) or lyap(A, Q, [], E).")


@ltifn("dlyap")
def _dlyap(interp, args, nargout):
    if len(args) == 2:
        return cn.lyap_d(_mat(args[0], "A"), _mat(args[1], "Q"))
    if len(args) == 3:
        return cn.sylvester_d(_mat(args[0], "A"), _mat(args[1], "B"), _mat(args[2], "C"))
    if len(args) == 4:
        if not is_empty_num(args[2]):
            raise MatlabError("dlyap(A, Q, [], E): the third argument must be empty.")
        a, q, e = _mat(args[0], "A"), _mat(args[1], "Q"), _mat(args[3], "E")
        if e.shape != a.shape or np.linalg.matrix_rank(e) < e.shape[0]:
            raise MatlabError("dlyap: E must be square, of the size of A, and nonsingular.")
        ei = np.linalg.inv(e)
        return cn.lyap_d(ei @ a, ei @ q @ ei.conj().T)
    raise MatlabError("Use dlyap(A, Q), dlyap(A, B, C) or dlyap(A, Q, [], E).")


def _chol_upper(x: np.ndarray, name: str) -> np.ndarray:
    try:
        return np.linalg.cholesky(cn.sym(x)).conj().T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cn.sym(x))
        if vals.min() < -1e-10 * max(1.0, abs(vals.max())):
            raise MatlabError(f"{name}: the solution is not positive semidefinite (A must be stable).") from None
        _q, r = np.linalg.qr((vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]).conj().T)
        return r


@ltifn("lyapchol")
def _lyapchol(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("Use R = lyapchol(A, B): the factor R of X = R'*R with A*X + X*A' + B*B' = 0.")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    if not cn.is_stable(cn.eigvals(a), False):
        raise MatlabError("lyapchol: A must be stable (all eigenvalues in the open left half-plane).")
    return _chol_upper(cn.lyap_c(a, b @ b.conj().T), "lyapchol")


@ltifn("dlyapchol")
def _dlyapchol(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("Use R = dlyapchol(A, B): the factor R of X = R'*R with A*X*A' - X + B*B' = 0.")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    if not cn.is_stable(cn.eigvals(a), True):
        raise MatlabError("dlyapchol: A must be stable (all eigenvalues inside the unit circle).")
    return _chol_upper(cn.lyap_d(a, b @ b.conj().T), "dlyapchol")


# -- Riccati equations ------------------------------------------------------------------------

def _riccati_args(args: list, name: str):
    if len(args) < 3:
        raise MatlabError(f"Use {name}(A, B, Q) or {name}(A, B, Q, R, S, E).")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    n, m = a.shape[0], b.shape[1]
    if a.shape != (n, n) or b.shape[0] != n:
        raise MatlabError(f"{name}: A must be square and B must have as many rows as A.")
    q = _sym_weight(args[2], n, "Q") if _opt(args, 2) is not None else np.zeros((n, n))
    r = _sym_weight(args[3], m, "R") if _opt(args, 3) is not None else np.eye(m)
    s = _mat(args[4], "S") if _opt(args, 4) is not None else None
    e = _mat(args[5], "E") if _opt(args, 5) is not None else None
    if s is not None and s.shape != (n, m):
        raise MatlabError(f"{name}: S must be a {n}-by-{m} matrix.")
    if e is not None and e.shape != (n, n):
        raise MatlabError(f"{name}: E must be a {n}-by-{n} matrix.")
    if e is not None and np.array_equal(e, np.eye(n)):
        e = None
    if s is not None and not np.any(s):
        s = None
    return a, b, q, r, s, e


def _lam(v: np.ndarray) -> np.ndarray:
    v = cn.pair_conjugates(v, 1e-10)
    return col(v)


@ltifn("care")
def _care(interp, args, nargout):
    if len(args) > 6:
        raise MatlabError("Too many arguments for care.")
    x, g, lam = cn.care(*_riccati_args(args, "care"))
    return (x, _lam(lam), g)[:max(nargout, 1)]


@ltifn("dare")
def _dare(interp, args, nargout):
    if len(args) > 6:
        raise MatlabError("Too many arguments for dare.")
    x, g, lam = cn.dare(*_riccati_args(args, "dare"))
    return (x, _lam(lam), g)[:max(nargout, 1)]


def _implicit(args: list, name: str, discrete: bool):
    if len(args) > 7:
        raise MatlabError(f"Too many arguments for {name}.")
    if len(args) > 6 and _opt(args, 6) is not None and np.any(_mat(args[6], "G")):
        raise MatlabError(f"{name}: a nonzero G term is not supported.")
    a, b, q, r, s, e = _riccati_args(args, name)
    x, k, lam = (cn.dare if discrete else cn.care)(a, b, q, r, s, e)
    n = a.shape[0]
    info = Struct.scalar({"Sx": np.ones((n, 1)), "Sr": np.ones((b.shape[1], 1)), "U": np.eye(n), "V": x,
                          "W": -k, "Report": np.zeros((1, 1))})
    return x, k, _lam(lam), info


@ltifn("icare")
def _icare(interp, args, nargout):
    return _implicit(args, "icare", False)[:max(nargout, 1)]


@ltifn("idare")
def _idare(interp, args, nargout):
    return _implicit(args, "idare", True)[:max(nargout, 1)]


# -- linear-quadratic design ------------------------------------------------------------------

def _lq_weights(rest: list, n: int, m: int, name: str):
    if len(rest) < 2:
        raise MatlabError(f"{name} needs the weights Q and R.")
    if len(rest) > 3:
        raise MatlabError(f"Too many arguments for {name}.")
    q = _sym_weight(rest[0], n, "Q")
    r = _sym_weight(rest[1], m, "R")
    nn = _mat(rest[2], "N") if len(rest) > 2 and not is_empty_num(rest[2]) else None
    if nn is not None and nn.shape != (n, m):
        raise MatlabError(f"{name}: N must be a {n}-by-{m} matrix (states by inputs).")
    ev = np.linalg.eigvalsh(cn.sym(r))
    if ev.size and ev.min() <= 0:
        raise MatlabError(f"{name}: R must be positive definite (every input needs a positive weight).")
    return q, r, nn


def _ab(args: list, name: str):
    s, rest = _plant(args, name)
    if s is not None:
        return s.a, s.b, rest, s.is_dt, s
    if len(args) < 2:
        raise MatlabError(f"{name} needs the matrices A and B, or a model.")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    if a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0]:
        raise MatlabError(f"{name}: A must be square and B must have as many rows as A.")
    return a, b, list(args[2:]), False, None


def lqr_gain(a, b, q, r, n, discrete: bool):
    x, k, lam = (cn.dare if discrete else cn.care)(a, b, q, r, n)
    return k, x, lam


@ltifn("lqr")
def _lqr(interp, args, nargout):
    a, b, rest, disc, _s = _ab(args, "lqr")
    q, r, n = _lq_weights(rest, a.shape[0], b.shape[1], "lqr")
    k, x, lam = lqr_gain(a, b, q, r, n, disc)
    return (k, x, _lam(lam))[:max(nargout, 1)]


@ltifn("dlqr")
def _dlqr(interp, args, nargout):
    a, b, rest, _disc, _s = _ab(args, "dlqr")
    q, r, n = _lq_weights(rest, a.shape[0], b.shape[1], "dlqr")
    k, x, lam = lqr_gain(a, b, q, r, n, True)
    return (k, x, _lam(lam))[:max(nargout, 1)]


@ltifn("lqrd")
def _lqrd(interp, args, nargout):
    if len(args) not in (5, 6):
        raise MatlabError("Use lqrd(A, B, Q, R, Ts) or lqrd(A, B, Q, R, N, Ts).")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    n, m = a.shape[0], b.shape[1]
    q, r, nn = _lq_weights(list(args[2:-1]), n, m, "lqrd")
    ts = as_scalar(args[-1], "sample time Ts")
    if ts <= 0:
        raise MatlabError("lqrd: the sample time must be positive.")
    nn = np.zeros((n, m)) if nn is None else nn
    # discrete equivalent of the continuous cost (Van Loan's integral of exp(M't) Qc exp(Mt))
    ac = np.zeros((n + m, n + m))
    ac[:n, :n], ac[:n, n:] = a, b
    qc = np.block([[q, nn], [nn.T, r]])
    big = np.block([[-ac.T, qc], [np.zeros((n + m, n + m)), ac]])
    e = cn.expm(big * ts)
    f22 = e[n + m:, n + m:]
    qd = cn.sym(f22.T @ e[:n + m, n + m:])
    ad, bd = f22[:n, :n], f22[:n, n:]
    k, x, lam = lqr_gain(ad, bd, qd[:n, :n], qd[n:, n:], qd[:n, n:], True)
    return (k, x, _lam(lam))[:max(nargout, 1)]


@ltifn("lqry")
def _lqry(interp, args, nargout):
    s, rest = _plant(args, "lqry")
    if s is None:
        raise MatlabError("lqry needs a model: lqry(sys, Q, R) weights the outputs y.")
    q, r, nn = _lq_weights(rest, s.ny, s.nu, "lqry")
    nn = np.zeros((s.ny, s.nu)) if nn is None else nn
    qx = s.c.T @ q @ s.c
    ru = r + s.d.T @ q @ s.d + nn.T @ s.d + s.d.T @ nn
    nx = s.c.T @ (q @ s.d + nn)
    k, x, lam = lqr_gain(s.a, s.b, cn.sym(qx), cn.sym(ru), nx, s.is_dt)
    return (k, x, _lam(lam))[:max(nargout, 1)]


@ltifn("lqi")
def _lqi(interp, args, nargout):
    s, rest = _plant(args, "lqi")
    if s is None:
        raise MatlabError("lqi needs a model: lqi(sys, Q, R).")
    n, m, p = s.nx, s.nu, s.ny
    if s.is_dt:
        ts = s.tsv()
        a = np.block([[s.a, np.zeros((n, p))], [-ts * s.c, np.eye(p)]])
        b = np.vstack([s.b, -ts * s.d])
    else:
        a = np.block([[s.a, np.zeros((n, p))], [-s.c, np.zeros((p, p))]])
        b = np.vstack([s.b, -s.d])
    q, r, nn = _lq_weights(rest, n + p, m, "lqi")
    k, x, lam = lqr_gain(a, b, q, r, nn, s.is_dt)
    return (k, x, _lam(lam))[:max(nargout, 1)]


def _lqe(a, g, c, q, r, n, discrete: bool, name: str):
    nw, ny = g.shape[1], c.shape[0]
    q = _sym_weight(q, nw, "Q")
    r = _sym_weight(r, ny, "R")
    nmat = np.zeros((nw, ny)) if n is None else _mat(n, "N")
    if nmat.shape != (nw, ny):
        raise MatlabError(f"{name}: N must be a {nw}-by-{ny} matrix (noise inputs by measurements).")
    qb = cn.sym(g @ q @ g.T)
    nb = g @ nmat
    solver = cn.dare if discrete else cn.care
    try:
        p, _k, _lam_ = solver(a.T, c.T, qb, r, nb if np.any(nb) else None)
    except MatlabError as err:
        raise MatlabError(f"{name}: the filter Riccati equation has no stabilizing solution; (C, A) must be "
                          f"detectable and R positive definite. ({err})") from None
    return p, qb, nb, r


@ltifn("lqe")
def _lqe_fn(interp, args, nargout):
    if len(args) < 5 or len(args) > 6:
        raise MatlabError("Use [L, P, E] = lqe(A, G, C, Q, R) or lqe(A, G, C, Q, R, N).")
    a, g, c = _mat(args[0], "A"), _mat(args[1], "G"), _mat(args[2], "C")
    p, _qb, nb, r = _lqe(a, g, c, args[3], args[4], _opt(args, 5), False, "lqe")
    gain = np.linalg.solve(r.T, (p @ c.T + nb).T).T
    return (gain, p, _lam(np.linalg.eigvals(a - gain @ c)))[:max(nargout, 1)]


@ltifn("dlqe")
def _dlqe(interp, args, nargout):
    if len(args) != 5:
        raise MatlabError("Use [M, P, Z, E] = dlqe(A, G, C, Q, R).")
    a, g, c = _mat(args[0], "A"), _mat(args[1], "G"), _mat(args[2], "C")
    p, _qb, _nb, r = _lqe(a, g, c, args[3], args[4], None, True, "dlqe")
    s = c @ p @ c.T + r
    m = np.linalg.solve(s.T, (p @ c.T).T).T
    z = cn.sym((np.eye(a.shape[0]) - m @ c) @ p)
    return (m, p, z, _lam(np.linalg.eigvals(a - a @ m @ c)))[:max(nargout, 1)]


def _index(v: Any, n: int, what: str) -> np.ndarray:
    idx = as_vector(v, what).real
    if np.any(idx != np.round(idx)) or (idx.size and (idx.min() < 1 or idx.max() > n)):
        raise MatlabError(f"The {what} must be whole numbers between 1 and {n}.")
    return idx.astype(int) - 1


def _named(sys: SS, idx: np.ndarray, inputs: bool) -> list:
    names = sys.in_name if inputs else sys.out_name
    return [names[i] for i in idx]


@ltifn("kalman")
def _kalman(interp, args, nargout):
    s, rest = _plant(args, "kalman")
    if s is None or len(rest) < 2:
        raise MatlabError("Use [kest, L, P] = kalman(sys, Qn, Rn) with a state-space plant whose last inputs "
                          "are the process noise.")
    qn = _mat(rest[0], "Qn")
    nw = qn.shape[0]
    kind = "current"
    rest = list(rest)
    if rest and is_text(rest[-1]):
        kind = text(rest[-1]).lower()
        if kind not in ("current", "delayed"):
            raise MatlabError("kalman: the estimator type must be 'current' or 'delayed'.")
        rest = rest[:-1]
    sensors = _index(rest[3], s.ny, "sensor indices") if len(rest) > 3 and not is_empty_num(rest[3]) \
        else np.arange(s.ny)
    if len(rest) > 4 and not is_empty_num(rest[4]):
        known = _index(rest[4], s.nu, "known input indices")
        noise = np.array([j for j in range(s.nu) if j not in set(known.tolist())], dtype=int)
    else:
        if nw > s.nu:
            raise MatlabError(f"kalman: Qn is {nw}-by-{nw} but the plant has only {s.nu} input(s).")
        known, noise = np.arange(s.nu - nw), np.arange(s.nu - nw, s.nu)
    if noise.size != nw or qn.shape != (nw, nw):
        raise MatlabError(f"kalman: Qn must be square with one row per noise input ({noise.size}).")
    a, bk, g = s.a, s.b[:, known], s.b[:, noise]
    c, dk, h = s.c[sensors, :], s.d[np.ix_(sensors, known)], s.d[np.ix_(sensors, noise)]
    ny = sensors.size
    rn = _sym_weight(rest[1], ny, "Rn")
    nn = _mat(rest[2], "Nn") if len(rest) > 2 and not is_empty_num(rest[2]) else np.zeros((nw, ny))
    if nn.shape != (nw, ny):
        raise MatlabError(f"kalman: Nn must be a {nw}-by-{ny} matrix.")
    qb = cn.sym(g @ qn @ g.T)
    rb = cn.sym(rn + h @ nn + nn.T @ h.T + h @ qn @ h.T)
    nb = g @ (qn @ h.T + nn)
    if np.linalg.eigvalsh(rb).min() <= 0:
        raise MatlabError("kalman: the measurement noise covariance (Rn plus the part of the process noise "
                          "that reaches the outputs directly) must be positive definite.")
    solver = cn.dare if s.is_dt else cn.care
    try:
        p, _k, _l = solver(a.T, c.T, qb, rb, nb if np.any(nb) else None)
    except MatlabError as err:
        raise MatlabError("kalman: the filter Riccati equation has no stabilizing solution; (C, A) must be "
                          f"detectable. ({err})") from None
    n = s.nx
    if not s.is_dt:
        gain = np.linalg.solve(rb.T, (p @ c.T + nb).T).T
        ak = a - gain @ c
        bkk = np.hstack([bk - gain @ dk, gain])
        ck = np.vstack([c, np.eye(n)])
        dkk = np.block([[dk, np.zeros((ny, ny))], [np.zeros((n, known.size)), np.zeros((n, ny))]])
        m = z = None
    else:
        sm = c @ p @ c.T + rb
        gain = np.linalg.solve(sm.T, (a @ p @ c.T + nb).T).T
        m = np.linalg.solve(sm.T, (p @ c.T).T).T
        z = cn.sym((np.eye(n) - m @ c) @ p)
        ak = a - gain @ c
        bkk = np.hstack([bk - gain @ dk, gain])
        if kind == "current":
            ck = np.vstack([(np.eye(ny) - c @ m) @ c, np.eye(n) - m @ c])
            dkk = np.block([[(np.eye(ny) - c @ m) @ dk, c @ m], [-m @ dk, m]])
        else:
            ck = np.vstack([c, np.eye(n)])
            dkk = np.block([[dk, np.zeros((ny, ny))], [np.zeros((n, known.size)), np.zeros((n, ny))]])
    kest = SS(ak, bkk, ck, dkk, s.ts)
    kest.in_name = _named(s, known, True) + _named(s, sensors, False)
    kest.out_name = [f"{nm}_e" if nm else "" for nm in _named(s, sensors, False)] + \
        [f"{nm}_e" if nm else "" for nm in s.state_name]
    kest.state_name = [f"{nm}_e" if nm else "" for nm in s.state_name]
    out = [kest, gain, p]
    if s.is_dt:
        out += [m, z]
    else:
        out += [np.zeros((0, 0)), np.zeros((0, 0))]
    return tuple(out[:max(nargout, 1)])


@ltifn("estim")
def _estim(interp, args, nargout):
    s, rest = _plant(args, "estim")
    if s is None or not rest:
        raise MatlabError("Use est = estim(sys, L) with a state-space model and an estimator gain L.")
    gain = _mat(rest[0], "L")
    sensors = _index(rest[1], s.ny, "sensor indices") if len(rest) > 1 and not is_empty_num(rest[1]) \
        else np.arange(s.ny)
    known = _index(rest[2], s.nu, "known input indices") if len(rest) > 2 and not is_empty_num(rest[2]) \
        else np.zeros(0, dtype=int)
    if len(rest) > 3:
        raise MatlabError("Too many arguments for estim.")
    n, ny = s.nx, sensors.size
    if gain.shape != (n, ny):
        raise MatlabError(f"estim: L must be a {n}-by-{ny} matrix (states by measured outputs).")
    c, bk, dk = s.c[sensors, :], s.b[:, known], s.d[np.ix_(sensors, known)]
    est = SS(s.a - gain @ c, np.hstack([bk - gain @ dk, gain]), np.vstack([c, np.eye(n)]),
             np.block([[dk, np.zeros((ny, ny))], [np.zeros((n, known.size)), np.zeros((n, ny))]]), s.ts)
    est.in_name = _named(s, known, True) + _named(s, sensors, False)
    return est


@ltifn("reg")
def _reg(interp, args, nargout):
    s, rest = _plant(args, "reg")
    if s is None or len(rest) < 2:
        raise MatlabError("Use rsys = reg(sys, K, L) with a state-space model, a state-feedback gain K and an "
                          "estimator gain L.")
    k, gain = _mat(rest[0], "K"), _mat(rest[1], "L")
    sensors = _index(rest[2], s.ny, "sensor indices") if len(rest) > 2 and not is_empty_num(rest[2]) \
        else np.arange(s.ny)
    known = _index(rest[3], s.nu, "known input indices") if len(rest) > 3 and not is_empty_num(rest[3]) \
        else np.zeros(0, dtype=int)
    if len(rest) > 4 and not is_empty_num(rest[4]):
        controls = _index(rest[4], s.nu, "control input indices")
    else:
        controls = np.array([j for j in range(s.nu) if j not in set(known.tolist())], dtype=int)
    if len(rest) > 5:
        raise MatlabError("Too many arguments for reg.")
    n = s.nx
    if k.shape != (controls.size, n):
        raise MatlabError(f"reg: K must be a {controls.size}-by-{n} matrix (control inputs by states).")
    if gain.shape != (n, sensors.size):
        raise MatlabError(f"reg: L must be a {n}-by-{sensors.size} matrix (states by measured outputs).")
    c = s.c[sensors, :]
    bc, dc = s.b[:, controls], s.d[np.ix_(sensors, controls)]
    bd, dd = s.b[:, known], s.d[np.ix_(sensors, known)]
    a = s.a - bc @ k - gain @ c + gain @ dc @ k
    out = SS(a, np.hstack([bd - gain @ dd, gain]), -k, np.zeros((controls.size, known.size + sensors.size)), s.ts)
    out.in_name = _named(s, known, True) + _named(s, sensors, False)
    out.out_name = _named(s, controls, True)
    return out


@ltifn("lqgreg")
def _lqgreg(interp, args, nargout):
    if len(args) != 2 or not isinstance(args[0], LTI):
        raise MatlabError("Use rlqg = lqgreg(kest, K) with a Kalman estimator and a state-feedback gain.")
    kest = to_ss(args[0])
    k = _mat(args[1], "K")
    n, nu = kest.nx, k.shape[0]
    if k.shape[1] != n:
        raise MatlabError(f"lqgreg: K must have {n} columns, one per estimated state.")
    ny = kest.nu - nu
    if ny < 1 or kest.ny < n:
        raise MatlabError("lqgreg: the estimator must have the control inputs first, then the measurements, "
                          "and the state estimates as its last outputs.")
    cx, dxu, dxy = kest.c[-n:, :], kest.d[-n:, :nu], kest.d[-n:, nu:]
    m = np.eye(nu) + k @ dxu
    f1 = -np.linalg.solve(m, k @ cx)
    f2 = -np.linalg.solve(m, k @ dxy)
    bu, by = kest.b[:, :nu], kest.b[:, nu:]
    out = SS(kest.a + bu @ f1, by + bu @ f2, f1, f2, kest.ts)
    out.in_name = kest.in_name[nu:]
    out.out_name = kest.in_name[:nu]
    return out


@ltifn("lqg")
def _lqg(interp, args, nargout):
    s, rest = _plant(args, "lqg")
    if s is None or len(rest) < 2:
        raise MatlabError("Use reg = lqg(sys, QXU, QWV) with weights for [x; u] and covariances for [w; v].")
    if len(rest) > 2:
        raise MatlabError("lqg: only the regulator form lqg(sys, QXU, QWV) is supported (no integral action "
                          "or 1-/2-degree-of-freedom options); use lqi with kalman and lqgreg for those.")
    n, m, p = s.nx, s.nu, s.ny
    qxu = _sym_weight(rest[0], n + m, "QXU")
    qwv = _sym_weight(rest[1], n + p, "QWV")
    q, r, nn = qxu[:n, :n], qxu[n:, n:], qxu[:n, n:]
    if np.linalg.eigvalsh(cn.sym(r)).min() <= 0:
        raise MatlabError("lqg: the input block of QXU must be positive definite.")
    k, _x, _lam_ = lqr_gain(s.a, s.b, cn.sym(q), cn.sym(r), nn if np.any(nn) else None, s.is_dt)
    w, v, nwv = qwv[:n, :n], qwv[n:, n:], qwv[:n, n:]
    if np.linalg.eigvalsh(cn.sym(v)).min() <= 0:
        raise MatlabError("lqg: the measurement-noise block of QWV must be positive definite.")
    solver = cn.dare if s.is_dt else cn.care
    pm, _g, _e = solver(s.a.T, s.c.T, cn.sym(w), cn.sym(v), nwv if np.any(nwv) else None)
    if not s.is_dt:
        gain = np.linalg.solve(v.T, (pm @ s.c.T + nwv).T).T
        a = s.a - s.b @ k - gain @ s.c + gain @ s.d @ k
        out = SS(a, gain, -k, np.zeros((m, p)), 0.0)
    else:
        sm = s.c @ pm @ s.c.T + v
        gain = np.linalg.solve(sm.T, (s.a @ pm @ s.c.T + nwv).T).T
        mm = np.linalg.solve(sm.T, (pm @ s.c.T).T).T
        # u = -K (xhat + M (y - C xhat - D u))
        t = np.eye(m) - k @ mm @ s.d
        f1 = -np.linalg.solve(t, k @ (np.eye(n) - mm @ s.c))
        f2 = -np.linalg.solve(t, k @ mm)
        bu = s.b - gain @ s.d
        out = SS(s.a - gain @ s.c + bu @ f1, gain + bu @ f2, f1, f2, s.ts)
    out.in_name, out.out_name = list(s.out_name), list(s.in_name)
    return out


# -- pole placement ---------------------------------------------------------------------------

def _place_args(args: list, name: str):
    if len(args) != 3:
        raise MatlabError(f"Use K = {name}(A, B, p) with the desired closed-loop poles p.")
    a, b = _mat(args[0], "A"), _mat(args[1], "B")
    p = np.asarray(as_vector(args[2], "pole vector"))
    n = a.shape[0]
    if a.shape != (n, n) or b.shape[0] != n:
        raise MatlabError(f"{name}: A must be square and B must have as many rows as A.")
    if p.size != n:
        raise MatlabError(f"{name}: give exactly {n} poles, one per state.")
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise MatlabError(f"{name} needs real matrices A and B.")
    pp = cn.pair_conjugates(p, 1e-10)
    if np.iscomplexobj(pp):
        rest = sorted(pp[np.abs(pp.imag) > 0], key=lambda v: (v.real, abs(v.imag), v.imag))
        if len(rest) % 2 or any(abs(rest[i] - np.conj(rest[i + 1])) > 1e-9 * max(1.0, abs(rest[i]))
                                for i in range(0, len(rest), 2)):
            raise MatlabError(f"{name}: complex poles must come in conjugate pairs.")
    return a, b, pp


@ltifn("acker")
def _acker(interp, args, nargout):
    a, b, p = _place_args(args, "acker")
    if b.shape[1] != 1:
        raise MatlabError("acker works for single-input systems; use place for several inputs.")
    n = a.shape[0]
    co = np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(n)]) if n else np.zeros((0, 0))
    if n and np.linalg.matrix_rank(co) < n:
        raise MatlabError("acker: (A, B) is not controllable, so the poles cannot be placed freely.")
    coef = np.real(cn.rpoly(p))
    phi = np.zeros((n, n))
    for cft in coef:
        phi = phi @ a + cft * np.eye(n)
    k = np.linalg.solve(co.T, np.eye(n)[:, -1]) @ phi
    return row(k)


@ltifn("place")
def _place(interp, args, nargout):
    a, b, p = _place_args(args, "place")
    n, m = a.shape[0], b.shape[1]
    rank_b = np.linalg.matrix_rank(b)
    vals, counts = np.unique(np.round(np.asarray(p, dtype=complex), 10), return_counts=True)
    if counts.max() > rank_b:
        raise MatlabError(f"place: a pole cannot be repeated more than rank(B) = {rank_b} times; use acker "
                          "for repeated poles of a single-input system.")
    co = np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(n)]) if n else np.zeros((0, 0))
    if n and np.linalg.matrix_rank(co) < n:
        raise MatlabError("place: (A, B) is not controllable, so the poles cannot be placed freely.")
    if m == 1:
        coef = np.real(cn.rpoly(p))
        phi = np.zeros((n, n))
        for cft in coef:
            phi = phi @ a + cft * np.eye(n)
        k = (np.linalg.solve(co.T, np.eye(n)[:, -1]) @ phi).reshape(1, n)
    else:
        from scipy.signal import place_poles
        try:
            res = place_poles(a, b, np.asarray(p), method="YT")
        except ValueError:
            try:
                res = place_poles(a, b, np.asarray(p), method="KNV0")
            except ValueError as err:
                raise MatlabError(f"place: the poles cannot be placed: {err}") from None
        k = np.real(res.gain_matrix)
    if nargout > 1:
        got = np.linalg.eigvals(a - b @ k)
        want = np.asarray(p, dtype=complex)
        err = 0.0
        for v in want:
            j = int(np.argmin(np.abs(got - v)))
            err = max(err, abs(got[j] - v) / max(1.0, abs(v)))
            got = np.delete(got, j)
        prec = float(np.floor(-np.log10(max(err, 1e-17))))
        return (k, min(prec, 15.0), char(""))[:nargout]
    return k


# -- PID tuning -------------------------------------------------------------------------------

_PID_TYPES = ("P", "I", "PI", "PD", "PDF", "PID", "PIDF")
_FOCUS = {"balanced": 4.0, "reference-tracking": 8.0, "disturbance-rejection": 2.0}


@ltifn("pidtuneOptions")
def _pidtune_options(interp, args, nargout):
    vals = {"PhaseMargin": np.array([[60.0]]), "DesignFocus": char("balanced"),
            "CrossoverFrequency": np.zeros((0, 0)), "NumUnstablePoles": np.zeros((1, 1))}
    _pos, pairs = split_options(args)
    if _pos:
        raise MatlabError("pidtuneOptions takes name/value pairs only.")
    lower = {k.lower(): k for k in vals}
    for nm, v in pairs:
        key = lower.get(nm.lower())
        if key is None:
            raise MatlabError(f"pidtuneOptions: unknown option '{nm}'. The options are: " + ", ".join(vals) + ".")
        if key == "DesignFocus":
            f = text(v).lower()
            if f not in _FOCUS:
                raise MatlabError("pidtuneOptions: DesignFocus must be 'balanced', 'reference-tracking' or "
                                  "'disturbance-rejection'.")
            vals[key] = char(f)
        elif key == "PhaseMargin":
            pm = as_scalar(v, "PhaseMargin")
            if not 0 < pm < 90:
                raise MatlabError("pidtuneOptions: PhaseMargin must be between 0 and 90 degrees.")
            vals[key] = np.array([[pm]])
        elif key == "CrossoverFrequency":
            if not is_empty_num(v) and as_scalar(v, "CrossoverFrequency") <= 0:
                raise MatlabError("pidtuneOptions: CrossoverFrequency must be positive.")
            vals[key] = v if is_empty_num(v) else np.array([[as_scalar(v, "CrossoverFrequency")]])
        else:
            vals[key] = np.array([[as_scalar(v, key)]])
    return Struct.scalar(vals)


class _Tuner:
    """Frequency-domain PID design for one SISO plant."""

    def __init__(self, plant: LTI, kind: str, pm: float, ratio: float, iform: str, dform: str):
        self.plant, self.kind, self.pm, self.ratio = plant, kind, pm, ratio
        self.ts = plant.ts
        self.h = plant.tsv()
        self.iform, self.dform = iform, dform
        self.has_i = "I" in kind
        self.has_d = "D" in kind
        self.has_p = "P" in kind
        self.filt = kind.endswith("F")
        self.sign = self._plant_sign()

    def g(self, w: float) -> complex:
        with np.errstate(all="ignore"):
            return complex(lti_freqresp(self.plant, np.array([w]))[0, 0, 0])

    def _plant_sign(self) -> float:
        if isinstance(self.plant, FRD):
            w = np.sort(self.plant.rad_freq())
            ph = np.degrees(np.angle(self.g(w[0])))
            return -1.0 if abs(abs(ph) - 180) < 45 else 1.0
        from .control import to_zpk
        g = to_zpk(_no_delay_copy(self.plant))
        z, p, k = np.asarray(g.z[0][0], dtype=complex), np.asarray(g.p[0][0], dtype=complex), g.k[0, 0]
        o = 1.0 if self.plant.is_dt else 0.0
        val = complex(k)
        for v in z[np.abs(z - o) > 1e-9]:
            val *= (o - v)
        for v in p[np.abs(p - o) > 1e-9]:
            val /= (o - v)
        # unstable poles flip the sign of the low-frequency gain without changing the feedback sign needed
        nunst = int(np.sum((np.abs(p) > 1 + 1e-9) if self.plant.is_dt else (p.real > 1e-9)))
        s = 1.0 if val.real >= 0 else -1.0
        return s * (-1.0) ** nunst

    def integ(self, w: float) -> complex:
        if self.ts == 0:
            return 1.0 / (1j * w)
        z = np.exp(1j * w * self.h)
        return {"ForwardEuler": self.h / (z - 1), "BackwardEuler": self.h * z / (z - 1),
                "Trapezoidal": self.h / 2 * (z + 1) / (z - 1)}[self.iform]

    def deriv(self, w: float, tf: float) -> complex:
        if self.ts == 0:
            return 1j * w / (1 + tf * 1j * w)
        z = np.exp(1j * w * self.h)
        f = {"ForwardEuler": self.h / (z - 1), "BackwardEuler": self.h * z / (z - 1),
             "Trapezoidal": self.h / 2 * (z + 1) / (z - 1)}[self.dform]
        return 1.0 / (tf + f)

    def filter_time(self, wc: float) -> float:
        if not self.filt:
            return 0.0
        tf = 0.01 / wc
        if self.ts != 0:
            tf = max(tf, self.h)
        return tf

    def gains(self, wc: float):
        """(Kp, Ki, Kd, Tf) that put the crossover at wc with the target phase margin, if possible."""
        gw = self.g(wc) * self.sign
        if not np.isfinite(gw) or gw == 0:
            return None
        mag = abs(gw)
        phi_c = np.radians(-180.0 + self.pm) - np.angle(gw)
        phi_c = (phi_c + np.pi) % (2 * np.pi) - np.pi       # controller phase needed, in (-180, 180]
        tf = self.filter_time(wc)
        ii, dd = self.integ(wc), self.deriv(wc, tf)
        kind = self.kind
        kp = ki = kd = 0.0
        lim = np.radians(1.0)
        if kind == "P":
            kp = 1.0 / mag
        elif kind == "I":
            ki = 1.0 / (mag * abs(ii))
        elif kind == "PI":
            lo, hi = np.angle(ii) + lim, -lim           # between pure integral and pure proportional
            ph = min(max(phi_c, lo), hi)
            target = np.exp(1j * ph) / mag
            sol = np.linalg.solve(np.array([[1.0, ii.real], [0.0, ii.imag]]), [target.real, target.imag])
            kp, ki = float(sol[0]), float(sol[1])
        elif kind in ("PD", "PDF"):
            lo, hi = lim, np.angle(dd) - lim
            ph = min(max(phi_c, lo), hi)
            target = np.exp(1j * ph) / mag
            sol = np.linalg.solve(np.array([[1.0, dd.real], [0.0, dd.imag]]), [target.real, target.imag])
            kp, kd = float(sol[0]), float(sol[1])
        else:  # PID, PIDF: Ti = ratio * Td fixes the third degree of freedom
            ph = min(max(phi_c, np.radians(-88.0)), np.radians(88.0))

            def shape(td: float) -> complex:
                return 1.0 + ii / (self.ratio * td) + td * self.deriv(wc, tf)

            from scipy.optimize import brentq
            lo_t, hi_t = 1e-6 / wc, 1e6 / wc

            def f(lt: float) -> float:
                return float(np.angle(shape(np.exp(lt))) - ph)

            if f(np.log(lo_t)) * f(np.log(hi_t)) > 0:
                return None
            td = float(np.exp(brentq(f, np.log(lo_t), np.log(hi_t), xtol=1e-12)))
            kp = 1.0 / (mag * abs(shape(td)))
            ki, kd = kp / (self.ratio * td), kp * td
        if min(kp, ki, kd) < 0:
            return None
        return self.sign * kp, self.sign * ki, self.sign * kd, (tf if kd != 0 else 0.0)

    def controller(self, vals) -> PID:
        kp, ki, kd, tf = vals
        c = PID(kp, ki, kd, tf, self.ts)
        c.iformula, c.dformula = self.iform, self.dform
        return c

    def assess(self, vals):
        """Closed-loop stability and the smallest phase margin of the loop with these gains."""
        from .control import lti_mul
        from .control_freq import allmargin_data
        loop = lti_mul(self.plant, self.controller(vals))
        try:
            m = allmargin_data(loop, "pidtune")
        except MatlabError:
            return False, float("nan"), float("nan"), np.inf
        pm, wp = m["PhaseMargin"], m["PMFrequency"]
        stable = m["Stable"]
        gm = m["GainMargin"]
        gmin = float(np.min(np.abs(np.log(gm[gm > 0])))) if np.any(gm > 0) else np.inf
        if pm.size == 0:
            return stable, np.inf, float("nan"), gmin
        k = int(np.argmin(np.abs(pm)))
        return stable, float(pm[k]), float(wp[np.argmax(wp)]), gmin


def _candidate_range(plant: LTI):
    from .control_freq import auto_freqs
    w = auto_freqs(plant)
    return float(w[0]), float(w[-1])


def pid_tune(plant: LTI, kind: str, wc: float | None, pm: float, focus: str, iform: str, dform: str):
    tuner = _Tuner(plant, kind, pm, _FOCUS[focus], iform, dform)
    if wc is not None:
        if plant.is_dt and wc >= np.pi / plant.tsv():
            raise MatlabError("pidtune: the crossover frequency must be below the Nyquist frequency pi/Ts.")
        vals = tuner.gains(wc)
        if vals is None:
            raise MatlabError(f"pidtune: no {kind} controller with positive gains gives a crossover at "
                              f"{wc:g} rad/s; try another frequency or controller type.")
        stable, apm, awc, _g = tuner.assess(vals)
        return tuner.controller(vals), stable, wc if np.isnan(awc) else awc, apm
    lo, hi = _candidate_range(plant)
    grid = np.logspace(np.log10(lo), np.log10(hi), 160)
    if plant.is_dt:
        grid = grid[grid < 0.9 * np.pi / plant.tsv()]
    cands = []
    for w in grid:
        vals = tuner.gains(float(w))
        if vals is None:
            continue
        stable, apm, awc, gmin = tuner.assess(vals)
        cands.append((float(w), vals, stable, apm, awc, gmin))
    good = [c for c in cands if c[2] is True and c[3] >= pm - 1.0]
    if not good:
        stable_c = [c for c in cands if c[2] is True]
        if not stable_c:
            if isinstance(plant, FRD) and cands:
                best = max(cands, key=lambda c: c[3] if np.isfinite(c[3]) else -1.0)
                return tuner.controller(best[1]), float("nan"), best[0], best[3]
            raise MatlabError(f"pidtune: no stabilizing {kind} controller was found for this plant; try a "
                              "controller type with more freedom (for example 'PID' or 'PIDF').")
        best = max(stable_c, key=lambda c: (c[3] if np.isfinite(c[3]) else 180.0))
        return tuner.controller(best[1]), True, best[4] if not np.isnan(best[4]) else best[0], best[3]

    def score(c) -> float:
        kp, ki, kd, _tf = c[1]
        if tuner.has_i:
            return abs(ki)             # integral gain: the speed of disturbance rejection
        return c[0]                    # otherwise the bandwidth

    if focus == "reference-tracking" and tuner.has_i and kind in ("PI", "I"):
        calm = []
        for c in good:
            gw = tuner.g(c[0]) * tuner.sign
            lag = np.degrees(abs(np.angle(np.exp(1j * (np.radians(-180.0 + pm) - np.angle(gw))))))
            if lag <= 20.0:
                calm.append(c)
        if calm:
            good = calm
    best = max(good, key=score)
    top = good[-1]
    if best is top and top[0] >= grid[-1] * 0.8:
        # the score keeps growing with bandwidth: there is no natural limit, so use a moderate
        # multiple of the slowest plant dynamics instead of an arbitrarily fast loop
        from .control_freq import equivalent_roots
        mags = np.abs(equivalent_roots(plant))
        mags = mags[mags > 1e-9]
        w_ref = 2.0 * (float(mags.min()) if mags.size else 1.0)
        near = min(good, key=lambda c: abs(np.log(c[0] / w_ref)))
        best = near
    # polish the chosen crossover between its neighbours
    idx = good.index(best)
    if 0 < idx < len(good) - 1 and best is not top:
        from scipy.optimize import minimize_scalar

        def neg(lw: float) -> float:
            v = tuner.gains(float(np.exp(lw)))
            return -(abs(v[1]) if tuner.has_i else float(np.exp(lw))) if v is not None else 0.0

        if tuner.has_i:
            res = minimize_scalar(neg, bounds=(np.log(good[idx - 1][0]), np.log(good[idx + 1][0])),
                                  method="bounded", options={"xatol": 1e-6})
            w_new = float(np.exp(res.x))
            v_new = tuner.gains(w_new)
            if v_new is not None:
                st, apm, awc, gmin = tuner.assess(v_new)
                if st is True and apm >= pm - 1.0 and abs(v_new[1]) >= abs(best[1][1]):
                    best = (w_new, v_new, st, apm, awc, gmin)
    return tuner.controller(best[1]), True, best[4] if not np.isnan(best[4]) else best[0], best[3]


@ltifn("pidtune")
def _pidtune(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Use C = pidtune(sys, type) with type 'P', 'I', 'PI', 'PD', 'PDF', 'PID' or 'PIDF'.")
    plant = need_lti(args, "pidtune")
    if not plant.is_siso:
        raise MatlabError("pidtune needs a SISO plant (one input and one output).")
    if isinstance(plant, PID):
        plant = pid_to_tf(plant)
    template = None
    if isinstance(args[1], PID):
        template = args[1]
        kind = template.type_name()
        iform, dform = template.iformula, template.dformula
        if template.ts != plant.ts and not (template.ts == 0 and not plant.is_dt):
            raise MatlabError("pidtune: the controller and the plant must have the same sample time.")
    elif is_text(args[1]):
        kind = text(args[1]).upper()
        iform = dform = "ForwardEuler"
        if kind.endswith("2") or kind.startswith("I-") or "-" in kind:
            raise MatlabError(f"pidtune: 2-degree-of-freedom controllers ('{text(args[1])}') are not supported; "
                              "use 'P', 'I', 'PI', 'PD', 'PDF', 'PID' or 'PIDF'.")
        if kind not in _PID_TYPES:
            raise MatlabError(f"pidtune: unknown controller type '{text(args[1])}'. Use 'P', 'I', 'PI', 'PD', "
                              "'PDF', 'PID' or 'PIDF'.")
    else:
        raise MatlabError("pidtune: the second argument is the controller type (for example 'PI') or a pid "
                          "object whose structure is to be kept.")
    wc, pm, focus = None, 60.0, "balanced"
    for a in args[2:]:
        if isinstance(a, Struct):
            f = {k.lower(): a.get_field(0, k) for k in a.fields}
            if "phasemargin" not in f:
                raise MatlabError("pidtune: unknown options structure; create it with pidtuneOptions.")
            pm = as_scalar(f["phasemargin"], "PhaseMargin")
            focus = text(f.get("designfocus", char("balanced"))).lower()
            cf = f.get("crossoverfrequency")
            if cf is not None and not is_empty_num(cf):
                wc = as_scalar(cf, "CrossoverFrequency")
        elif is_num(a) and np.asarray(a).size == 1:
            wc = as_scalar(a, "crossover frequency")
            if wc <= 0:
                raise MatlabError("pidtune: the crossover frequency must be positive.")
        else:
            raise MatlabError("pidtune: after the type give a crossover frequency and/or a pidtuneOptions "
                              "structure.")
    if focus not in _FOCUS:
        raise MatlabError("pidtune: DesignFocus must be 'balanced', 'reference-tracking' or "
                          "'disturbance-rejection'.")
    if plant.is_dt and kind in ("PD", "PID") and dform == "ForwardEuler":
        # an unfiltered forward-Euler derivative is not causal; use the backward formula as MATLAB does
        dform = "BackwardEuler"
    c, stable, wcross, apm = pid_tune(plant, kind, wc, pm, focus, iform, dform)
    if isinstance(template, PIDSTD):
        kp, ki, kd, tf = c.gains()
        std = pid_to_std(kp, ki, kd, tf, c.ts)
        std.iformula, std.dformula = c.iformula, c.dformula
        c = std
    c.time_unit = plant.time_unit
    info = Struct.scalar({"Stable": bool_scalar(bool(stable)) if isinstance(stable, bool)
                          else np.array([[float("nan")]]),
                          "CrossoverFrequency": float(wcross), "PhaseMargin": float(apm)})
    return (c, info)[:max(nargout, 1)]


def _no_app(name: str, hint: str):
    def f(interp, args, nargout):
        raise MatlabError(f"{name} is an interactive design app and is not available. {hint}")
    return f


for _nm, _hint in (
    ("sisotool", "Use rlocus, bode, margin and pidtune from the command line instead."),
    ("rltool", "Use rlocus(sys) and rlocus(sys, k) from the command line instead."),
    ("controlSystemDesigner", "Use rlocus, bode, margin and pidtune from the command line instead."),
    ("pidTuner", "Use C = pidtune(sys, type) from the command line instead."),
    ("pidtool", "Use C = pidtune(sys, type) from the command line instead."),
    ("ltiview", "Use step, impulse, bode, nyquist and pzmap from the command line instead."),
    ("linearSystemAnalyzer", "Use step, impulse, bode, nyquist and pzmap from the command line instead."),
):
    ltifn(_nm)(_no_app(_nm, _hint))


# -- conversions between polynomial, zero-pole and state-space data ---------------------------

@ltifn("ss2tf")
def _ss2tf(interp, args, nargout):
    if len(args) not in (4, 5):
        raise MatlabError("Use [b, a] = ss2tf(A, B, C, D) or ss2tf(A, B, C, D, iu).")
    s = SS(_mat(args[0], "A"), _mat(args[1], "B"), _mat(args[2], "C"), _full_d(args))
    iu = 0
    if len(args) == 5:
        iu = int(as_scalar(args[4], "input index")) - 1
        if iu < 0 or iu >= s.nu:
            raise MatlabError(f"ss2tf: the input index must lie between 1 and {s.nu}.")
    elif s.nu > 1:
        raise MatlabError("ss2tf: the system has several inputs; give the input number as fifth argument.")
    if s.nu == 0:
        return np.zeros((s.ny, s.nx + 1)), row(cn.charpoly(s.a))
    num, den = cn.ss_numden(s.a, s.b[:, [iu]], s.c, s.d[:, [iu]])
    return np.real_if_close(num[:, 0, :]) if not np.iscomplexobj(num) else num[:, 0, :], row(den)


def _full_d(args: list) -> np.ndarray:
    d = _mat(args[3], "D")
    b, c = _mat(args[1], "B"), _mat(args[2], "C")
    if d.size == 1 and d.flat[0] == 0 and (c.shape[0], b.shape[1]) != (1, 1):
        return np.zeros((c.shape[0], b.shape[1]))
    return d


@ltifn("tf2ss")
def _tf2ss(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("Use [A, B, C, D] = tf2ss(num, den).")
    num, den = _mat(args[0], "numerator"), as_vector(args[1], "denominator")
    den = cn.ptrim(den)
    if cn.piszero(den):
        raise MatlabError("tf2ss: the denominator cannot be zero.")
    n = den.size - 1
    rows = []
    for r in num:
        r = cn.ptrim(r)
        if r.size > den.size:
            raise MatlabError("tf2ss: the numerator has a higher order than the denominator; the transfer "
                              "function must be proper.")
        rows.append(np.concatenate([np.zeros(den.size - r.size), r]) / den[0])
    nm = np.array(rows) if rows else np.zeros((0, n + 1))
    dn = den / den[0]
    a = np.zeros((n, n))
    b = np.zeros((n, 1))
    if n:
        a[0, :] = -dn[1:]
        if n > 1:
            a[1:, :-1] = np.eye(n - 1)
        b[0, 0] = 1.0
    d = nm[:, [0]] if nm.size else np.zeros((nm.shape[0], 1))
    c = nm[:, 1:] - d * dn[None, 1:] if n else np.zeros((nm.shape[0], 0))
    return (a, b, c, d)[:max(nargout, 1)]


def _zp_columns(z: Any, ny: int) -> list:
    if is_empty_num(z):
        return [np.zeros(0)] * ny
    zm = np.asarray(z)
    if zm.ndim == 2 and zm.shape[0] == 1 and ny == 1:
        zm = zm.T
    if zm.ndim == 1:
        zm = zm.reshape(-1, 1)
    if zm.shape[1] != ny:
        raise MatlabError(f"The zeros must have one column per output ({ny}).")
    return [zm[:, i][np.isfinite(zm[:, i])] for i in range(ny)]


@ltifn("zp2tf")
def _zp2tf(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("Use [num, den] = zp2tf(z, p, k).")
    k = np.asarray(as_vector(args[2], "gain k"))
    p = np.asarray(as_vector(args[1], "poles")) if not is_empty_num(args[1]) else np.zeros(0)
    zs = _zp_columns(args[0], k.size)
    den = cn.rpoly(p)
    n = max([den.size] + [z.size + 1 for z in zs])
    num = np.zeros((k.size, n), dtype=complex)
    for i, z in enumerate(zs):
        pz = k[i] * cn.rpoly(z)
        num[i, n - pz.size:] = pz
    num = num.real.copy() if not np.any(num.imag) else num
    return num, row(den)


@ltifn("tf2zp")
def _tf2zp(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("Use [z, p, k] = tf2zp(num, den).")
    num, den = _mat(args[0], "numerator"), cn.ptrim(as_vector(args[1], "denominator"))
    if cn.piszero(den):
        raise MatlabError("tf2zp: the denominator cannot be zero.")
    p = cn.proots(den)
    zs, ks = [], []
    for r in num:
        r = cn.ptrim(r)
        zs.append(cn.proots(r) if not cn.piszero(r) else np.zeros(0))
        ks.append(0.0 if cn.piszero(r) else r[0] / den[0])
    nz = max([z.size for z in zs] + [0])
    zmat = np.full((nz, len(zs)), np.inf, dtype=complex)
    for i, z in enumerate(zs):
        zmat[:z.size, i] = z
    zmat = zmat.real.copy() if not np.any(np.where(np.isfinite(zmat), zmat.imag, 0.0)) else zmat
    pp = p.real.copy() if np.iscomplexobj(p) and not np.any(p.imag) else p
    return (zmat, col(pp), col(np.array(ks)))[:max(nargout, 1)]


@ltifn("zp2ss")
def _zp2ss(interp, args, nargout):
    num, den = _zp2tf(interp, args, 2)
    return _tf2ss(interp, [num, den], nargout)


@ltifn("ss2zp")
def _ss2zp(interp, args, nargout):
    if len(args) not in (4, 5):
        raise MatlabError("Use [z, p, k] = ss2zp(A, B, C, D) or ss2zp(A, B, C, D, iu).")
    s = SS(_mat(args[0], "A"), _mat(args[1], "B"), _mat(args[2], "C"), _full_d(args))
    iu = 0
    if len(args) == 5:
        iu = int(as_scalar(args[4], "input index")) - 1
        if iu < 0 or iu >= s.nu:
            raise MatlabError(f"ss2zp: the input index must lie between 1 and {s.nu}.")
    elif s.nu > 1:
        raise MatlabError("ss2zp: the system has several inputs; give the input number as fifth argument.")
    p = cn.pair_conjugates(cn.eigvals(s.a), 1e-12)
    zs, ks = [], []
    for i in range(s.ny):
        z, k = cn.siso_zeros_gain(s.a, s.b[:, [iu]], s.c[[i], :], s.d[i, iu])
        zs.append(np.asarray(z))
        ks.append(k)
    nz = max([z.size for z in zs] + [0])
    zmat = np.full((nz, s.ny), np.inf, dtype=complex)
    for i, z in enumerate(zs):
        zmat[:z.size, i] = z
    zmat = zmat.real.copy() if not np.any(np.where(np.isfinite(zmat), zmat.imag, 0.0)) else zmat
    return (zmat, col(p), col(np.array(ks)))[:max(nargout, 1)]
