"""Polynomials and interpolation.

Integration and the ODE suite live in ``ode_more``; root finding and optimization in
``optim_more`` and ``_optim_local``."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char, simplify, to_int
from ._registry import builtin
from ._util import fnum, is_text, nargs, text


# -- polynomials ----------------------------------------------------------------------

@builtin("polyfit")
def _polyfit(interp, args, nargout):
    nargs(args, 3, 3, "polyfit")
    x, y = fnum(args[0]).flatten(order="F"), fnum(args[1]).flatten(order="F")
    if x.size != y.size:
        raise MatlabError("The first two inputs to polyfit must have the same number of elements.")
    n = to_int(args[2], "degree")
    if n < 0:
        raise MatlabError("The polynomial degree must be a nonnegative integer.")
    mu = None
    xs = x
    if nargout > 2:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        mu = np.array([[float(np.mean(x))], [sd]])
        xs = (x - mu[0, 0]) / (sd if sd else 1.0)
    A = np.vander(xs, n + 1)
    p = np.linalg.lstsq(A, y, rcond=None)[0]
    p_row = simplify(p.reshape(1, -1))
    if nargout <= 1:
        return p_row
    R = np.linalg.qr(A, mode="r") if A.shape[0] >= A.shape[1] else np.linalg.qr(A)[1]
    resid = y - A @ p
    normr = float(np.linalg.norm(resid))
    ss_tot = float(np.sum(np.abs(y - np.mean(y)) ** 2)) if y.size else 0.0
    s = Struct.scalar({"R": R, "df": float(max(0, x.size - (n + 1))), "normr": normr,
                       "rsquared": 1.0 - normr ** 2 / ss_tot if ss_tot else 1.0})
    if nargout == 2:
        return p_row, s
    return p_row, s, mu


@builtin("polyval")
def _polyval(interp, args, nargout):
    nargs(args, 2, 4, "polyval")
    p = fnum(args[0]).flatten()
    x = fnum(args[1])
    S = args[2] if len(args) > 2 and isinstance(args[2], Struct) else None
    if len(args) > 3:
        mu = fnum(args[3]).flatten()
        if mu.size < 2:
            raise MatlabError("mu must have two elements: [mean; standard deviation].")
        x = (x - mu[0]) / (mu[1] if mu[1] else 1.0)
    y = simplify(np.polyval(p, x)) if p.size else np.zeros(x.shape)
    if nargout < 2:
        return y
    if S is None:
        raise MatlabError("The structure S from polyfit is needed to compute delta.")
    R = fnum(S.get_field(0, "R"))
    df = float(fnum(S.get_field(0, "df")).flat[0])
    normr = float(fnum(S.get_field(0, "normr")).flat[0])
    A = np.vander(x.flatten(order="F"), p.size)
    E = np.linalg.lstsq(R.T, A.T, rcond=None)[0].T  # A / R
    e = np.sqrt(1.0 + np.sum(np.abs(E) ** 2, axis=1))
    delta = (normr / np.sqrt(df) * e) if df > 0 else np.full(e.shape, np.inf)
    return y, delta.reshape(x.shape, order="F")


@builtin("roots")
def _roots(interp, args, nargout):
    p = fnum(args[0])
    if p.size and not (p.ndim == 2 and 1 in p.shape):
        raise MatlabError("roots: the input must be a vector of polynomial coefficients.")
    p = p.flatten()
    if p.size and not np.all(np.isfinite(p)):
        raise MatlabError("roots: the coefficients must not contain NaN or Inf.")
    r = np.roots(p)
    if not np.any(p):
        return np.zeros((0, 0))  # empty or all-zero polynomial
    return simplify(r.reshape(-1, 1))


@builtin("poly")
def _poly(interp, args, nargout):
    a = fnum(args[0])
    if a.size == 0:
        return np.array([[1.0]])
    if a.shape[0] == a.shape[1] and a.shape[0] > 1:
        return simplify(np.poly(a).reshape(1, -1))
    return simplify(np.poly(a.flatten()).reshape(1, -1))


@builtin("polyder")
def _polyder(interp, args, nargout):
    nargs(args, 1, 2, "polyder")

    def trim(p):
        """Drop leading zeros; the zero polynomial is [0]."""
        p = np.atleast_1d(p)
        nz = np.nonzero(p)[0]
        return p[nz[0]:] if nz.size else np.zeros(1, dtype=p.dtype)

    def der(p):
        p = np.atleast_1d(p)
        return np.polyder(p) if p.size > 1 else np.zeros(1)

    a = fnum(args[0]).flatten()
    if len(args) == 1:
        return simplify(der(a).reshape(1, -1))
    b = fnum(args[1]).flatten()
    if nargout < 2:  # derivative of the product a*b
        return simplify(trim(der(np.convolve(a, b))).reshape(1, -1))
    # derivative of the quotient a/b = q/d
    q = np.polysub(np.convolve(der(a), b), np.convolve(a, der(b)))
    d = np.convolve(b, b)
    return simplify(trim(q).reshape(1, -1)), simplify(trim(d).reshape(1, -1))


@builtin("polyint")
def _polyint(interp, args, nargout):
    k = float(fnum(args[1]).flat[0]) if len(args) > 1 else 0.0
    return np.polyint(fnum(args[0]).flatten(), k=k).reshape(1, -1)


def _conv_shape(args, pos, name):
    if len(args) <= pos:
        return "full"
    shape = text(args[pos]).lower() if is_text(args[pos]) else ""
    if shape not in ("full", "same", "valid"):
        raise MatlabError(f"{name}: the shape must be 'full', 'same' or 'valid'.")
    return shape


def _conv_vec(a, b, shape):
    """Convolution of two 1-D arrays with MATLAB's 'same' and 'valid' conventions (sized like a)."""
    if a.size == 0 or b.size == 0:
        full = np.zeros(max(a.size + b.size - 1, 0))
    else:
        full = np.convolve(a, b, mode="full")
    if shape == "same":
        start = int(np.ceil((b.size - 1) / 2))
        return full[start:start + a.size]
    if shape == "valid":
        n = max(a.size - max(b.size - 1, 0), 0) if b.size else 0
        return full[b.size - 1:b.size - 1 + n] if n else np.zeros(0, dtype=full.dtype)
    return full


@builtin("conv")
def _conv(interp, args, nargout):
    nargs(args, 2, 3, "conv")
    a, b = fnum(args[0]), fnum(args[1])
    for v in (a, b):
        if v.ndim != 2 or (v.size and 1 not in v.shape):
            raise MatlabError("conv: A and B must be vectors.")
    r = _conv_vec(a.flatten(), b.flatten(), _conv_shape(args, 2, "conv"))
    col = a.shape[1] == 1 and a.shape[0] > 1
    return simplify(r.reshape(-1, 1) if col else r.reshape(1, -1))


@builtin("deconv")
def _deconv(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for deconv.")
    y, h = fnum(args[0]), fnum(args[1])
    for v in (y, h):
        if v.ndim != 2 or (v.size and 1 not in v.shape):
            raise MatlabError("deconv: inputs must be vectors.")
    rest = list(args[2:])
    shape = "full"
    if rest and is_text(rest[0]) and text(rest[0]).lower() in ("full", "same", "valid"):
        shape = text(rest.pop(0)).lower()
    if len(rest) % 2:
        raise MatlabError("deconv: options must be name-value pairs.")
    method, alpha = "long-division", 0.0
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower() if is_text(rest[k]) else ""
        if key == "method":
            method = text(rest[k + 1]).lower() if is_text(rest[k + 1]) else ""
            if method not in ("long-division", "least-squares"):
                raise MatlabError("deconv: Method must be 'long-division' or 'least-squares'.")
        elif key == "regularizationfactor":
            alpha = float(np.real(fnum(rest[k + 1])).flat[0])
            if alpha < 0:
                raise MatlabError("deconv: RegularizationFactor must be nonnegative.")
        else:
            raise MatlabError(f"deconv: unknown option '{text(rest[k]) if is_text(rest[k]) else '?'}'.")
    yv, hv = y.flatten(), h.flatten()
    col = y.shape[1] == 1 and y.shape[0] > 1

    def orient(v):
        v = simplify(np.asarray(v))
        return v.reshape(-1, 1) if col else v.reshape(1, -1)

    if method == "long-division":
        if shape != "full" or alpha:
            raise MatlabError("deconv: the shape and RegularizationFactor options need Method 'least-squares'.")
        if hv.size == 0 or hv[0] == 0:
            raise MatlabError("First coefficient of the divisor must be nonzero.")
        if yv.size < hv.size:
            q, r = np.zeros(1), yv
        else:
            q, r = np.polydiv(yv, hv)
            q, r = np.atleast_1d(q), np.atleast_1d(r)
        full_r = np.zeros(yv.size, dtype=np.result_type(r.dtype, float))
        if r.size:
            full_r[-r.size:] = r
        return orient(q), orient(full_r)
    # least squares: x minimizes ||conv(x, h, shape) - y||^2 + alpha ||x||^2
    ny, nh = yv.size, hv.size
    nx = {"full": ny - nh + 1, "same": ny, "valid": ny + nh - 1}[shape]
    if nx < 1 or nh == 0:
        raise MatlabError("deconv: the lengths of the inputs do not fit the requested shape.")
    cmat = np.zeros((ny, nx), dtype=np.result_type(yv.dtype, hv.dtype, float))
    for k in range(nx):
        e = np.zeros(nx)
        e[k] = 1.0
        cmat[:, k] = _conv_vec(e, hv, shape)
    if alpha:
        lhs = cmat.conj().T @ cmat + alpha * np.eye(nx)
        x = np.linalg.solve(lhs, cmat.conj().T @ yv)
    else:
        x = np.linalg.lstsq(cmat, yv, rcond=None)[0]
    return orient(x), orient(yv - cmat @ x)


@builtin("filter")
def _filter(interp, args, nargout):
    from scipy.signal import lfilter
    nargs(args, 3, 5, "filter")
    if nargout > 2:
        raise MatlabError("filter: too many output arguments.")
    b, a = fnum(args[0]), fnum(args[1])
    for v in (b, a):
        if v.ndim != 2 or (v.size and 1 not in v.shape) or v.size == 0:
            raise MatlabError("filter: the coefficients must be nonempty vectors.")
    b, a = b.flatten(), a.flatten()
    if a[0] == 0:
        raise MatlabError("First denominator filter coefficient must be non-zero.")
    x = fnum(args[2])
    single = any(isinstance(v, np.ndarray) and v.dtype in (np.float32, np.complex64) for v in args[:3])
    dim = None
    if len(args) > 4 and not (isinstance(args[4], np.ndarray) and args[4].size == 0):
        dim = to_int(args[4], "dimension")
        if dim < 1:
            raise MatlabError("Dimension argument must be a positive integer.")
    if dim is not None:
        axis = dim - 1
    else:
        axis = next((k for k, n in enumerate(x.shape) if n != 1), 0)
    order = max(a.size, b.size) - 1
    if axis >= x.ndim:
        x = x.reshape(x.shape + (1,) * (axis + 1 - x.ndim))
    zi_given = len(args) > 3 and not (isinstance(args[3], np.ndarray) and args[3].size == 0)
    want_zf = nargout > 1
    shape = list(x.shape)
    shape[axis] = order
    zi = None
    if zi_given:
        z = fnum(args[3])
        if z.size == order and (z.ndim == 2 and 1 in z.shape or order <= 1):
            # one set of initial conditions shared by every column
            zshape = [1] * x.ndim
            zshape[axis] = order
            zi = np.broadcast_to(z.reshape(zshape), shape).astype(np.result_type(z.dtype, float))
        elif list(z.shape) + [1] * (x.ndim - z.ndim) == shape:
            zi = z.reshape(shape)
        else:
            raise MatlabError("filter: the initial conditions must be a vector of length "
                              f"max(length(a), length(b)) - 1 = {order}, or an array with that length along "
                              "the operating dimension.")
    elif want_zf:
        zi = np.zeros(shape)
    if x.size == 0:
        y = x.copy()
        zf = zi if zi is not None else np.zeros(shape)
    elif zi is not None:
        y, zf = lfilter(b, a, x, axis=axis, zi=zi)
    else:
        y = lfilter(b, a, x, axis=axis)
        zf = None
    from ..values import as_2d
    y = as_2d(simplify(np.asarray(y)))
    if single and y.dtype.kind in "fc":
        y = y.astype(np.complex64 if y.dtype.kind == "c" else np.float32)
    if not want_zf:
        return y
    zf = as_2d(simplify(np.asarray(zf)))
    if x.ndim == 2 and 1 in x.shape:
        zf = zf.reshape(-1, 1)  # the final conditions of a vector are a column
    return y, zf


# -- interpolation ----------------------------------------------------------------------

_SMOOTH_METHODS = ("spline", "pchip", "cubic", "makima", "v5cubic")


def _interp_text_tail(args: list) -> tuple[list, str | None, Any]:
    """Split ``(..., method, extrap)`` from the numeric arguments."""
    rest = list(args)
    method = None
    extrap = None
    strs = [k for k, a in enumerate(rest) if is_text(a)]
    if strs:
        k = strs[0]
        method = text(rest[k]).lower()
        tail = rest[k + 1:]
        rest = rest[:k]
        if tail:
            if is_text(tail[0]):
                extrap = text(tail[0]).lower()
            else:
                extrap = float(np.real(fnum(tail[0]).flat[0]))
    return rest, method, extrap


def _interp1_object(method: str, x: np.ndarray, yv: np.ndarray):
    """A callable that interpolates ``yv`` (rows = sample points) with ``method``."""
    from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
    n = x.size
    if method in ("linear", "nearest", "previous", "next"):
        if n < 2 and method == "linear":
            raise MatlabError("Linear interpolation needs at least two sample points.")
        kind = "nearest-up" if method == "nearest" else method
        if n == 1:
            return lambda q: np.repeat(yv[:1], np.size(q), axis=0)
        return interp1d(x, yv, kind=kind, axis=0, bounds_error=False, fill_value="extrapolate",
                        assume_sorted=True)
    if n < 2:
        raise MatlabError(f"Interpolation method '{method}' needs at least two sample points.")
    if method == "spline":
        return CubicSpline(x, yv, axis=0, bc_type="not-a-knot", extrapolate=True)
    if method in ("pchip", "cubic", "v5cubic"):
        return PchipInterpolator(x, yv, axis=0, extrapolate=True)
    if method == "makima":
        from .._compat import makima
        return makima(x, yv, axis=0, extrapolate=True)
    raise MatlabError(f"Unknown interpolation method '{method}'.")


@builtin("interp1")
def _interp1(interp, args, nargout):
    nargs(args, 2, 5, "interp1")
    rest, method, extrap = _interp_text_tail(args)
    method = method or "linear"
    if len(rest) == 2:
        y = fnum(rest[0])
        xq = fnum(rest[1])
        npts = y.size if (y.shape[0] == 1 or y.shape[1] == 1) else y.shape[0]
        x = np.arange(1, npts + 1, dtype=float)
    elif len(rest) == 3:
        x, y, xq = fnum(rest[0]).flatten(order="F"), fnum(rest[1]), fnum(rest[2])
    else:
        raise MatlabError("interp1 expects (y, xq) or (x, y, xq), optionally followed by a method.")
    if np.iscomplexobj(x):
        x = np.real(x)
    yvec = y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1)
    yv = y.flatten(order="F") if yvec else y
    if yv.shape[0] != x.size:
        raise MatlabError("The sample points x and values y must have matching lengths.")
    order = np.argsort(x, kind="stable")
    x = x[order]
    yv = yv[order, ...]
    if x.size > 1 and np.any(np.diff(x) == 0):
        raise MatlabError("The sample points must be unique.")
    xqf = np.real(xq).flatten(order="F") if np.iscomplexobj(xq) else xq.flatten(order="F")
    f = _interp1_object(method, x, yv)
    out = np.array(f(xqf))
    if out.dtype.kind not in "fc":
        out = out.astype(float)
    do_extrap = extrap == "extrap" or (extrap is None and method in _SMOOTH_METHODS)
    if x.size and (not do_extrap or isinstance(extrap, float)):
        outside = (xqf < x[0]) | (xqf > x[-1])
        if outside.any():
            out[outside, ...] = extrap if isinstance(extrap, float) else np.nan
    out[np.isnan(xqf), ...] = np.nan
    if yvec:
        return simplify(out.reshape(xq.shape, order="F"))
    if xq.ndim == 2 and (xq.shape[0] == 1 or xq.shape[1] == 1):
        return simplify(out.reshape(xqf.size, -1))
    return simplify(out.reshape(xq.shape + yv.shape[1:], order="F"))


def _pp_struct(breaks: np.ndarray, coefs: np.ndarray, d: int) -> Struct:
    pieces = breaks.size - 1
    return Struct.scalar({
        "form": char("pp"), "breaks": breaks.reshape(1, -1).astype(float), "coefs": coefs,
        "pieces": float(pieces), "order": float(coefs.shape[1]), "dim": float(d),
    })


def _pp_from_ppoly(obj, d: int) -> Struct:
    c = np.asarray(obj.c)  # (order, pieces[, d]) highest power first
    k, L = c.shape[0], c.shape[1]
    if c.ndim == 2:
        coefs = c.T.copy()
    else:
        coefs = np.transpose(c, (1, 2, 0)).reshape(L * d, k)
    return _pp_struct(np.asarray(obj.x), simplify(coefs), d)


def _pp_parts(pp) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    if not isinstance(pp, Struct) or "breaks" not in pp.fields or "coefs" not in pp.fields:
        raise MatlabError("Expected a piecewise polynomial structure (from mkpp, spline or pchip).")
    breaks = fnum(pp.get_field(0, "breaks")).flatten()
    coefs = fnum(pp.get_field(0, "coefs"))
    d = int(fnum(pp.get_field(0, "dim")).flat[0]) if "dim" in pp.fields else 1
    L = breaks.size - 1
    k = coefs.size // max(L * d, 1)
    return breaks, coefs.reshape(L * d, k), L, k, d


def _ppval_eval(pp, xq: np.ndarray) -> np.ndarray:
    breaks, coefs, L, k, d = _pp_parts(pp)
    xf = np.real(xq).flatten(order="F").astype(float)
    idx = np.clip(np.searchsorted(breaks, xf, side="right") - 1, 0, L - 1)
    s = xf - breaks[idx]
    out = np.zeros((d, xf.size), dtype=complex if np.iscomplexobj(coefs) else float)
    for comp in range(d):
        rows = coefs[idx * d + comp, :]
        acc = np.zeros(xf.size, dtype=out.dtype)
        for j in range(k):
            acc = acc * s + rows[:, j]
        out[comp] = acc
    out[:, np.isnan(xf)] = np.nan
    out = simplify(out)
    if d == 1:
        return out.reshape(xq.shape, order="F")
    if xq.ndim == 2 and (xq.shape[0] == 1 or xq.shape[1] == 1):
        return out
    return out.reshape((d,) + xq.shape, order="F")


def _spline_like(interp, args, nargout, kind: str):
    from scipy.interpolate import CubicSpline, PchipInterpolator
    nargs(args, 2, 3, kind)
    x = np.real(fnum(args[0])).flatten(order="F").astype(float)
    y = fnum(args[1])
    if y.shape[0] == 1 or y.shape[1] == 1:
        yv = y.flatten(order="F")
        d = 1
    else:
        yv = y.T  # column j of y holds the value at x(j)
        d = y.shape[0]
    bc = "not-a-knot"
    if kind == "spline" and yv.shape[0] == x.size + 2:
        bc = ((1, yv[0]), (1, yv[-1]))
        yv = yv[1:-1]
    if yv.shape[0] != x.size:
        raise MatlabError("The number of sites and values must match.")
    order = np.argsort(x, kind="stable")
    x, yv = x[order], yv[order, ...]
    if x.size < 2:
        raise MatlabError("At least two data points are needed.")
    if kind == "spline":
        obj = CubicSpline(x, yv, axis=0, bc_type=bc)
    elif kind == "pchip":
        obj = PchipInterpolator(x, yv, axis=0)
    else:
        from .._compat import makima
        obj = makima(x, yv, axis=0, extrapolate=True)
    pp = _pp_from_ppoly(obj, d)
    if len(args) > 2:
        return _ppval_eval(pp, fnum(args[2]))
    return pp


builtin("spline")(lambda interp, args, nargout: _spline_like(interp, args, nargout, "spline"))
builtin("pchip")(lambda interp, args, nargout: _spline_like(interp, args, nargout, "pchip"))
builtin("makima")(lambda interp, args, nargout: _spline_like(interp, args, nargout, "makima"))


@builtin("ppval")
def _ppval(interp, args, nargout):
    nargs(args, 2, 2, "ppval")
    pp, xq = args[0], args[1]
    if not isinstance(pp, Struct) and isinstance(xq, Struct):
        pp, xq = xq, pp
    return _ppval_eval(pp, fnum(xq))


@builtin("mkpp")
def _mkpp(interp, args, nargout):
    nargs(args, 2, 3, "mkpp")
    breaks = fnum(args[0]).flatten()
    coefs = fnum(args[1])
    d = int(np.prod(fnum(args[2]).flatten())) if len(args) > 2 else 1
    L = breaks.size - 1
    if L < 1:
        raise MatlabError("mkpp needs at least two breaks.")
    if coefs.size % (L * d):
        raise MatlabError("The number of coefficients does not match the number of pieces.")
    k = coefs.size // (L * d)
    if coefs.shape != (L * d, k):
        coefs = coefs.flatten(order="F").reshape((L * d, k), order="F")
    return _pp_struct(breaks, coefs, d)


@builtin("unmkpp")
def _unmkpp(interp, args, nargout):
    nargs(args, 1, 1, "unmkpp")
    breaks, coefs, L, k, d = _pp_parts(args[0])
    return breaks.reshape(1, -1), coefs, float(L), float(k), float(d)


def _grid_vector(G: np.ndarray, along_cols: bool) -> np.ndarray:
    if G.ndim == 2 and G.shape[0] > 1 and G.shape[1] > 1:
        return (G[0, :] if along_cols else G[:, 0]).astype(float)
    return G.flatten().astype(float)


@builtin("interp2")
def _interp2(interp, args, nargout):
    from scipy.interpolate import RegularGridInterpolator
    nargs(args, 1, 7, "interp2")
    rest, method, extrap = _interp_text_tail(args)
    method = method or "linear"
    fill = extrap if isinstance(extrap, float) else np.nan
    if len(rest) in (1, 2):
        Z = fnum(rest[0])
        k = to_int(rest[1], "refinement") if len(rest) == 2 else 1
        m, n = Z.shape
        x = np.arange(1, n + 1, dtype=float)
        y = np.arange(1, m + 1, dtype=float)
        Xq, Yq = np.meshgrid(np.linspace(1, n, (n - 1) * 2 ** k + 1), np.linspace(1, m, (m - 1) * 2 ** k + 1))
    elif len(rest) in (3, 5):
        if len(rest) == 3:
            Z = fnum(rest[0])
            x = np.arange(1, Z.shape[1] + 1, dtype=float)
            y = np.arange(1, Z.shape[0] + 1, dtype=float)
            Xq, Yq = fnum(rest[1]), fnum(rest[2])
        else:
            Z = fnum(rest[2])
            x = _grid_vector(fnum(rest[0]), True)
            y = _grid_vector(fnum(rest[1]), False)
            Xq, Yq = fnum(rest[3]), fnum(rest[4])
        def is_row(a):
            return a.ndim == 2 and a.shape[0] == 1 and a.shape[1] > 1

        def is_col(a):
            return a.ndim == 2 and a.shape[1] == 1 and a.shape[0] > 1

        if (is_row(Xq) and is_col(Yq)) or (is_col(Xq) and is_row(Yq)) \
                or (Xq.ndim == 2 and Yq.ndim == 2 and Xq.shape[0] == 1 and Yq.shape[1] == 1 and Xq.shape != Yq.shape):
            # vectors of different orientation are grid vectors: the result is length(Yq)-by-length(Xq)
            Xq, Yq = np.meshgrid(Xq.flatten(), Yq.flatten())
        elif Xq.shape != Yq.shape:
            Xq, Yq = np.broadcast_arrays(Xq, Yq)
    else:
        raise MatlabError("interp2 expects (Z), (Z, k), (Z, Xq, Yq) or (X, Y, Z, Xq, Yq).")
    if Z.shape != (y.size, x.size):
        raise MatlabError("The grid vectors do not match the size of Z.")
    if x.size > 1 and x[0] > x[-1]:
        x, Z = x[::-1], Z[:, ::-1]
    if y.size > 1 and y[0] > y[-1]:
        y, Z = y[::-1], Z[::-1, :]
    xqf = np.real(Xq).flatten(order="F").astype(float)
    yqf = np.real(Yq).flatten(order="F").astype(float)
    if method == "makima":
        from scipy.interpolate import Akima1DInterpolator
        along_x = Akima1DInterpolator(x, Z.T, axis=0, method="makima")
        out = np.empty(xqf.size, dtype=complex if np.iscomplexobj(Z) else float)
        for p in range(xqf.size):
            out[p] = Akima1DInterpolator(y, along_x(xqf[p]), method="makima")(yqf[p])
        outside = (xqf < x[0]) | (xqf > x[-1]) | (yqf < y[0]) | (yqf > y[-1])
        out[outside] = fill
    else:
        rgi_method = {"linear": "linear", "nearest": "nearest", "cubic": "cubic", "spline": "cubic"}.get(method)
        if rgi_method is None:
            raise MatlabError(f"Unknown interpolation method '{method}'.")
        if rgi_method == "cubic" and (x.size < 4 or y.size < 4):
            rgi_method = "linear"
        if rgi_method == "nearest":
            def nearest(grid, q):
                # index of the closest grid point; a query halfway between two points takes the upper one
                hi = np.clip(np.searchsorted(grid, q, side="right"), 1, max(grid.size - 1, 1))
                lo = hi - 1
                if grid.size == 1:
                    return np.zeros(q.shape, dtype=int)
                return np.where(q - grid[lo] >= grid[hi] - q, hi, lo)

            inside = (xqf >= x[0]) & (xqf <= x[-1]) & (yqf >= y[0]) & (yqf <= y[-1])
            safe_x, safe_y = np.where(inside, xqf, x[0]), np.where(inside, yqf, y[0])
            picked = Z[nearest(y, safe_y), nearest(x, safe_x)]
            out = np.where(inside, picked, fill)
        else:
            f = RegularGridInterpolator((y, x), Z, method=rgi_method, bounds_error=False, fill_value=fill)
            out = f(np.stack([yqf, xqf], axis=-1))
    return simplify(np.asarray(out).reshape(Xq.shape, order="F"))


# -- numerical gradient and 2-D convolution ------------------------------------------

def _gradient_axis(f: np.ndarray, axis: int, h) -> np.ndarray:
    F = np.moveaxis(f, axis, 0)
    n = F.shape[0]
    G = np.zeros(F.shape, dtype=complex if np.iscomplexobj(F) else float)
    if n < 2:
        return np.moveaxis(G, 0, axis)
    hv = np.asarray(h, dtype=float).flatten()
    x = np.arange(n) * float(hv[0]) if hv.size == 1 else hv
    if x.size != n:
        raise MatlabError("The spacing vector must have one element per point along that dimension.")
    shape = (-1,) + (1,) * (F.ndim - 1)
    G[0] = (F[1] - F[0]) / (x[1] - x[0])
    G[-1] = (F[-1] - F[-2]) / (x[-1] - x[-2])
    if n > 2:
        G[1:-1] = (F[2:] - F[:-2]) / (x[2:] - x[:-2]).reshape(shape)
    return np.moveaxis(G, 0, axis)


@builtin("gradient")
def _gradient(interp, args, nargout):
    nargs(args, 1, None, "gradient")
    f = fnum(args[0])
    spacing = [fnum(a) for a in args[1:]]
    if f.ndim == 2 and (f.shape[0] == 1 or f.shape[1] == 1):
        axis = 1 if f.shape[0] == 1 else 0
        return simplify(_gradient_axis(f, axis, spacing[0] if spacing else 1.0))
    axes = [1, 0] + list(range(2, f.ndim))
    outs = []
    for k in range(max(nargout, 1)):
        if k >= len(axes):
            raise MatlabError("Too many output arguments for gradient.")
        if len(spacing) == 1:
            h = spacing[0]
        elif k < len(spacing):
            h = spacing[k]
        else:
            h = 1.0
        outs.append(simplify(_gradient_axis(f, axes[k], h)))
    return tuple(outs) if len(outs) > 1 else outs[0]


@builtin("conv2")
def _conv2(interp, args, nargout):
    from scipy.signal import convolve2d
    nargs(args, 2, 4, "conv2")
    shape = "full"
    nums = list(args)
    if is_text(nums[-1]):
        shape = text(nums.pop()).lower()
    if shape not in ("full", "same", "valid"):
        raise MatlabError("The shape option must be 'full', 'same' or 'valid'.")
    if len(nums) == 3:
        u, v, A = (fnum(a) for a in nums)
        kr, kc = u.size, v.size
        if A.size and kr and kc:
            full = convolve2d(convolve2d(A, u.reshape(-1, 1)), v.reshape(1, -1))
    elif len(nums) == 2:
        A, B = fnum(nums[0]), fnum(nums[1])
        kr, kc = B.shape
        if A.size and B.size:
            full = convolve2d(A, B)
    else:
        raise MatlabError("conv2 expects (A, B) or (u, v, A).")
    ma, na = A.shape
    if not (A.size and kr and kc):
        full = np.zeros((max(ma + kr - 1, 0), max(na + kc - 1, 0)))
    if shape == "same":
        r0, c0 = kr // 2, kc // 2
        out = full[r0:r0 + ma, c0:c0 + na]
    elif shape == "valid":
        out = full[kr - 1:ma, kc - 1:na] if ma >= kr and na >= kc else \
            np.zeros((max(ma - kr + 1, 0), max(na - kc + 1, 0)))
    else:
        out = full
    return simplify(np.array(out))
