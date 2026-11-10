"""Linear algebra."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..errors import MatlabError
from ..ops import left_divide
from ..values import simplify, to_int
from ._registry import builtin
from ._util import fnum, is_text, text


def _square(a: np.ndarray, name: str) -> None:
    if a.shape[0] != a.shape[1]:
        raise MatlabError(f"Matrix must be square for {name}.")


@builtin("inv")
def _inv(interp, args, nargout):
    a = fnum(args[0])
    _square(a, "inv")
    if a.size and (np.linalg.cond(a) > 1 / np.finfo(float).eps):
        interp.warn("Matrix is singular to working precision.")
        with np.errstate(all="ignore"):
            try:
                return simplify(np.linalg.inv(a))
            except np.linalg.LinAlgError:
                return np.full(a.shape, np.inf)
    return simplify(np.linalg.inv(a))


@builtin("pinv")
def _pinv(interp, args, nargout):
    return simplify(np.linalg.pinv(fnum(args[0])))


@builtin("det")
def _det(interp, args, nargout):
    a = fnum(args[0])
    _square(a, "det")
    return simplify(np.asarray(np.linalg.det(a)))


@builtin("rank")
def _rank(interp, args, nargout):
    a = fnum(args[0])
    if a.size == 0:
        return 0.0
    return float(np.linalg.matrix_rank(a))


@builtin("trace")
def _trace(interp, args, nargout):
    a = fnum(args[0])
    _square(a, "trace")
    return simplify(np.asarray(np.trace(a)))


@builtin("norm")
def _norm(interp, args, nargout):
    a = fnum(args[0])
    p = args[1] if len(args) > 1 else None
    if a.size == 0:
        return 0.0
    is_vec = a.shape[0] == 1 or a.shape[1] == 1
    if p is None:
        order = 2
    elif is_text(p):
        s = text(p).lower()
        order = {"fro": "fro", "inf": np.inf, "-inf": -np.inf}.get(s)
        if order is None:
            raise MatlabError(f"Unknown norm type '{s}'.")
    else:
        order = float(np.real(fnum(p).flat[0]))
    if is_vec:
        v = a.flatten()
        if order == "fro":
            order = 2
        return float(np.linalg.norm(v, ord=order))
    return float(np.linalg.norm(a, ord=order))


@builtin("eig")
def _eig(interp, args, nargout):
    a = fnum(args[0])
    _square(a, "eig")
    if len(args) > 1 and not is_text(args[1]):
        b = fnum(args[1])
        w, v = sla.eig(a, b)
    elif np.allclose(a, a.conj().T):
        w, v = np.linalg.eigh(a)
    else:
        w, v = np.linalg.eig(a)
    w = simplify(np.asarray(w))
    if nargout <= 1:
        return w.reshape(-1, 1)
    v = simplify(np.asarray(v))
    return v, np.diag(w)


@builtin("svd")
def _svd(interp, args, nargout):
    a = fnum(args[0])
    econ = len(args) > 1
    if nargout <= 1:
        return np.linalg.svd(a, compute_uv=False).reshape(-1, 1)
    u, s, vh = np.linalg.svd(a, full_matrices=not econ)
    S = np.zeros((u.shape[1], vh.shape[0]))
    S[: s.size, : s.size] = np.diag(s)
    return simplify(u), S, simplify(vh.conj().T)


@builtin("qr")
def _qr(interp, args, nargout):
    a = fnum(args[0])
    mode = "economic" if len(args) > 1 else "full"
    q, r = sla.qr(a, mode=mode)
    if nargout <= 1:
        return r
    return q, r


@builtin("lu")
def _lu(interp, args, nargout):
    a = fnum(args[0])
    p, l, u = sla.lu(a)
    if nargout <= 2:
        return p @ l, u
    return l, u, p.T


@builtin("chol")
def _chol(interp, args, nargout):
    a = fnum(args[0])
    lower = len(args) > 1 and text(args[1]).lower() == "lower"
    try:
        r = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        if nargout > 1:
            return np.zeros((0, 0)), 1.0
        raise MatlabError("Matrix must be positive definite.")
    out = r if lower else r.conj().T
    return (out, 0.0) if nargout > 1 else out


@builtin("expm")
def _expm(interp, args, nargout):
    return simplify(sla.expm(fnum(args[0])))


@builtin("logm")
def _logm(interp, args, nargout):
    return simplify(np.asarray(sla.logm(fnum(args[0]))))


@builtin("sqrtm")
def _sqrtm(interp, args, nargout):
    return simplify(np.asarray(sla.sqrtm(fnum(args[0]))))


@builtin("null")
def _null(interp, args, nargout):
    return sla.null_space(fnum(args[0]))


@builtin("orth")
def _orth(interp, args, nargout):
    return sla.orth(fnum(args[0]))


@builtin("linsolve")
def _linsolve(interp, args, nargout):
    return left_divide(fnum(args[0]), fnum(args[1]))


@builtin("cond")
def _cond(interp, args, nargout):
    return float(np.linalg.cond(fnum(args[0])))


@builtin("vander")
def _vander(interp, args, nargout):
    return np.vander(fnum(args[0]).flatten())


@builtin("hilb")
def _hilb(interp, args, nargout):
    return sla.hilbert(to_int(args[0], "n"))


@builtin("toeplitz")
def _toeplitz(interp, args, nargout):
    c = fnum(args[0]).flatten()
    r = fnum(args[1]).flatten() if len(args) > 1 else None
    return sla.toeplitz(c, r)

