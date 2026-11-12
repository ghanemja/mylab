"""Polynomials, interpolation, root finding, optimization, integration, ODE solvers."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char, simplify, to_int
from ._registry import builtin
from ._util import fnum, is_text, nargs, text


def _call_scalar(interp, f, *xs) -> float:
    vals = interp.call_function_value(f, [np.array([[float(x)]]) for x in xs], 1)
    if not vals:
        raise MatlabError("The function must return a value.")
    a = fnum(vals[0])
    return float(np.real(a.flat[0]))


def _call_vector(interp, f, args) -> np.ndarray:
    vals = interp.call_function_value(f, args, 1)
    if not vals:
        raise MatlabError("The function must return a value.")
    return fnum(vals[0])


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


@builtin("conv")
def _conv(interp, args, nargout):
    a, b = fnum(args[0]), fnum(args[1])
    shape = text(args[2]) if len(args) > 2 else "full"
    r = np.convolve(a.flatten(), b.flatten(), mode="full")
    if shape == "same":
        start = int(np.ceil((b.size - 1) / 2))
        r = r[start:start + a.size]
    elif shape == "valid":
        r = np.convolve(a.flatten(), b.flatten(), mode="valid") if a.size >= b.size else np.zeros(0)
    col = a.shape[1] == 1 and a.shape[0] > 1
    return simplify(r.reshape(-1, 1) if col else r.reshape(1, -1))


@builtin("deconv")
def _deconv(interp, args, nargout):
    q, r = np.polydiv(fnum(args[0]).flatten(), fnum(args[1]).flatten())
    full_r = np.zeros(fnum(args[0]).size)
    full_r[-r.size:] = r
    return q.reshape(1, -1), full_r.reshape(1, -1)


@builtin("filter")
def _filter(interp, args, nargout):
    from scipy.signal import lfilter
    b, a, x = fnum(args[0]).flatten(), fnum(args[1]).flatten(), fnum(args[2])
    axis = 1 if x.shape[0] == 1 else 0
    return lfilter(b, a, x, axis=axis)


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
    from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d
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
        return Akima1DInterpolator(x, yv, axis=0, method="makima", extrapolate=True)
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
    from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
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
        obj = Akima1DInterpolator(x, yv, axis=0, method="makima", extrapolate=True)
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
        if (Xq.ndim == 2 and Yq.ndim == 2 and Xq.shape[0] == 1 and Yq.shape[1] == 1
                and Xq.shape != Yq.shape):
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


# -- root finding and optimization ----------------------------------------------------

@builtin("optimset", "optimoptions")
def _optimset(interp, args, nargout):
    s = Struct()
    start = 1 if args and is_text(args[0]) and len(args) % 2 == 1 else 0
    for k in range(start, len(args) - 1, 2):
        s.set_field(0, text(args[k]), args[k + 1])
    return s


def _opt(options, name, default):
    if isinstance(options, Struct) and name in options.fields:
        v = options.data.flat[0][name]
        if isinstance(v, np.ndarray) and v.size:
            return float(np.real(fnum(v).flat[0]))
    return default


def _opt_output(iterations, func_count, algorithm: str, message: str) -> Struct:
    return Struct.scalar({"iterations": float(iterations), "funcCount": float(func_count),
                          "algorithm": char(algorithm), "message": char(message)})


def _outs(nargout: int, vals: tuple):
    if nargout <= 1:
        return vals[0]
    if nargout > len(vals):
        raise MatlabError("Too many output arguments.")
    return vals[:nargout]


@builtin("fzero")
def _fzero(interp, args, nargout):
    from scipy.optimize import brentq
    nargs(args, 2, None, "fzero")
    f = args[0]
    x0 = fnum(args[1]).flatten()
    opts = args[2] if len(args) > 2 else None
    tol = _opt(opts, "TolX", 2.2e-16)
    count = [0]

    def g(x):
        count[0] += 1
        return _call_scalar(interp, f, x)

    def done(x, fx, iters):
        out = _opt_output(iters, count[0], "bisection, interpolation",
                          "Zero found in the interval.")
        return _outs(nargout, (x, fx, 1.0, out))

    if x0.size == 2:
        a, b = float(x0[0]), float(x0[1])
        fa, fb = g(a), g(b)
        if np.sign(fa) == np.sign(fb) and fa != 0 and fb != 0:
            raise MatlabError("The function values at the interval endpoints must differ in sign.")
        search = 0
    else:
        a = b = float(x0[0])
        fa = fb = g(a)
        if fa == 0:
            return done(a, 0.0, 0)
        dx = abs(a) / 50 if a != 0 else 1 / 50
        found = False
        search = 0
        for _ in range(200):
            search += 1
            a, b = float(x0[0]) - dx, float(x0[0]) + dx
            fa, fb = g(a), g(b)
            if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                found = True
                break
            dx *= np.sqrt(2)
        if not found:
            raise MatlabError("fzero could not find an interval with a sign change around the starting point.")
    if fa == 0:
        return done(a, 0.0, search)
    if fb == 0:
        return done(b, 0.0, search)
    x, res = brentq(g, a, b, xtol=max(tol, 1e-300), rtol=4 * np.finfo(float).eps, maxiter=500,
                    full_output=True)
    return done(x, g(x), res.iterations)


@builtin("fminsearch")
def _fminsearch(interp, args, nargout):
    from scipy.optimize import minimize
    nargs(args, 2, None, "fminsearch")
    f = args[0]
    x0 = fnum(args[1])
    opts = args[2] if len(args) > 2 else None
    shape = x0.shape

    def g(x):
        return float(np.real(_call_vector(interp, f, [x.reshape(shape)]).flat[0]))

    res = minimize(g, x0.flatten(order="F").astype(float), method="Nelder-Mead",
                   options={"xatol": _opt(opts, "TolX", 1e-4), "fatol": _opt(opts, "TolFun", 1e-4),
                            "maxiter": int(_opt(opts, "MaxIter", 200 * x0.size)),
                            "maxfev": int(_opt(opts, "MaxFunEvals", 200 * x0.size))})
    x = res.x.reshape(shape, order="F")
    msg = "Optimization terminated: the current x satisfies the termination criteria." if res.success \
        else "Exiting: the maximum number of iterations or function evaluations was reached."
    out = _opt_output(res.nit, res.nfev, "Nelder-Mead simplex direct search", msg)
    return _outs(nargout, (x, float(res.fun), 1.0 if res.success else 0.0, out))


@builtin("fminbnd")
def _fminbnd(interp, args, nargout):
    from scipy.optimize import minimize_scalar
    nargs(args, 3, None, "fminbnd")
    f = args[0]
    a, b = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])
    opts = args[3] if len(args) > 3 else None
    res = minimize_scalar(lambda x: _call_scalar(interp, f, x), bounds=(a, b), method="bounded",
                          options={"xatol": _opt(opts, "TolX", 1e-4),
                                   "maxiter": int(_opt(opts, "MaxIter", 500))})
    msg = "Optimization terminated: the current x satisfies the termination criteria." if res.success \
        else "Exiting: the maximum number of iterations was reached."
    out = _opt_output(getattr(res, "nit", 0), res.nfev, "golden section search, parabolic interpolation", msg)
    return _outs(nargout, (float(res.x), float(res.fun), 1.0 if res.success else 0.0, out))


@builtin("fminunc")
def _fminunc(interp, args, nargout):
    from scipy.optimize import minimize
    nargs(args, 2, None, "fminunc")
    f = args[0]
    x0 = fnum(args[1])
    shape = x0.shape
    res = minimize(lambda x: float(np.real(_call_vector(interp, f, [x.reshape(shape, order="F")]).flat[0])),
                   x0.flatten(order="F").astype(float), method="BFGS")
    x = res.x.reshape(shape, order="F")
    msg = "Local minimum found." if res.success else str(res.message)
    out = _opt_output(res.nit, res.nfev, "quasi-newton", msg)
    grad = np.asarray(res.jac, dtype=float).reshape(-1, 1)
    try:
        hess = np.linalg.inv(np.asarray(res.hess_inv, dtype=float))
    except np.linalg.LinAlgError:
        hess = np.full((x.size, x.size), np.nan)
    return _outs(nargout, (x, float(res.fun), 1.0 if res.success else 0.0, out, grad, hess))


@builtin("fsolve")
def _fsolve(interp, args, nargout):
    from scipy.optimize import root
    f = args[0]
    x0 = fnum(args[1])
    shape = x0.shape
    res = root(lambda x: _call_vector(interp, f, [x.reshape(shape)]).flatten(), x0.flatten(), method="hybr")
    x = res.x.reshape(shape)
    fval = _call_vector(interp, f, [x])
    outs = (x, fval, 1.0 if res.success else 0.0)
    return outs[:nargout] if nargout > 1 else x


# -- integration ----------------------------------------------------------------------------

@builtin("integral", "quad", "quadgk", "quadl")
def _integral(interp, args, nargout):
    from scipy.integrate import quad
    f = args[0]
    a, b = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])
    rest = args[3:]
    opts = {}
    for k in range(0, len(rest) - 1, 2):
        if is_text(rest[k]):
            opts[text(rest[k]).lower()] = float(fnum(rest[k + 1]).flat[0])
    if opts.get("arrayvalued"):
        from scipy.integrate import quad_vec
        shape = []

        def g(x):
            v = _call_vector(interp, f, [np.array([[float(x)]])])
            if not shape:
                shape.append(v.shape)
            elif v.shape != shape[0]:
                raise MatlabError("integral: the function must return arrays of the same size.")
            return v.flatten(order="F")

        val, _err = quad_vec(g, a, b, epsabs=opts.get("abstol", 1e-10), epsrel=opts.get("reltol", 1e-6))
        return simplify(np.asarray(val).reshape(shape[0], order="F"))
    val, _err = quad(lambda x: _call_scalar(interp, f, x), a, b,
                     epsabs=opts.get("abstol", 1e-10), epsrel=opts.get("reltol", 1e-6), limit=200)
    return val


@builtin("integral2")
def _integral2(interp, args, nargout):
    from scipy.integrate import dblquad
    f = args[0]
    xmin, xmax = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])

    def bound(v):
        if isinstance(v, np.ndarray):
            c = float(fnum(v).flat[0])
            return lambda x: c
        return lambda x: _call_scalar(interp, v, x)

    ymin, ymax = bound(args[3]), bound(args[4])
    val, _ = dblquad(lambda y, x: _call_scalar(interp, f, x, y), xmin, xmax, ymin, ymax)
    return val


# -- ODE solvers ------------------------------------------------------------------------------

# Continuous extensions of solutions returned as ``sol = ode45(...)``, keyed by sol.idata.id so
# that deval can evaluate the solver's own interpolant. Only the most recent ones are kept.
_ODE_SOLUTIONS: "OrderedDict[int, dict]" = OrderedDict()
_ODE_IDS = itertools.count(1)
_ODE_KEEP = 256
_SOLVER_CLASSES: dict = {}


def _solver_class(method: str):
    """Explicit Runge-Kutta pairs with a max-norm error test and a more cautious step-size
    controller (safety factor 0.8 rather than 0.9), which keeps the error at the default
    tolerances close to what MATLAB-language programs expect."""
    if method not in ("RK45", "RK23"):
        return method
    if method not in _SOLVER_CLASSES:
        from scipy.integrate import RK23, RK45
        base = RK45 if method == "RK45" else RK23
        factor = (0.9 / 0.8) ** (5 if method == "RK45" else 3)

        class Cautious(base):  # type: ignore[misc, valid-type]
            def _estimate_error_norm(self, K, h, scale):
                err = self._estimate_error(K, h) / scale
                return float(np.max(np.abs(err))) * factor if err.size else 0.0

        Cautious.__name__ = f"Matlab{method}"
        _SOLVER_CLASSES[method] = Cautious
    return _SOLVER_CLASSES[method]


def _opt_value(options, name):
    if isinstance(options, Struct) and options.numel() == 1:
        for k, v in options.data.flat[0].items():
            if k.lower() == name.lower():
                if isinstance(v, np.ndarray) and v.size == 0:
                    return None
                return v
    return None


def _ode(method, refine_default, solver_name):
    def f(interp, args, nargout):
        from scipy.integrate import solve_ivp
        nargs(args, 3, None, solver_name)
        fn = args[0]
        tspan = np.real(fnum(args[1])).flatten().astype(float)
        if tspan.size < 2:
            raise MatlabError("tspan must have at least two elements: [t0 tfinal].")
        y0 = fnum(args[2]).flatten(order="F")
        opts = args[3] if len(args) > 3 else None
        extra = list(args[4:])
        rtol = _opt(opts, "RelTol", 1e-3)
        atol_v = _opt_value(opts, "AbsTol")
        atol = fnum(atol_v).flatten().astype(float) if atol_v is not None else 1e-6
        if isinstance(atol, np.ndarray) and atol.size == 1:
            atol = float(atol[0])
        refine = int(_opt(opts, "Refine", refine_default))
        kw = {}
        max_step = _opt(opts, "MaxStep", None)
        if max_step is not None:
            kw["max_step"] = max_step
        first = _opt(opts, "InitialStep", None)
        if first is not None:
            kw["first_step"] = first
        is_complex = np.iscomplexobj(y0)
        y0 = y0.astype(complex if is_complex else float)

        def rhs(t, y):
            out = _call_vector(interp, fn, [np.array([[t]]), y.reshape(-1, 1), *extra])
            if out.size != y.size:
                raise MatlabError(f"The ODE function must return a column vector of length {y.size}.")
            return out.flatten(order="F")

        ev_fn = _opt_value(opts, "Events")
        events = None
        if ev_fn is not None:
            events = _make_events(interp, ev_fn, tspan[0], y0, extra)
        sol = solve_ivp(rhs, (float(tspan[0]), float(tspan[-1])), y0, method=_solver_class(method),
                        rtol=rtol, atol=atol, dense_output=True, events=events, **kw)
        if sol.status == -1:
            raise MatlabError(f"ODE solver failed: {sol.message}")
        t_steps, y_steps = sol.t, sol.y
        te = ye = ie = None
        if events is not None:
            rows = []
            for k, (tk, yk) in enumerate(zip(sol.t_events, sol.y_events)):
                for j in range(len(tk)):
                    rows.append((float(tk[j]), yk[j], k + 1))
            rows.sort(key=lambda r: r[0] * np.sign(tspan[-1] - tspan[0]))
            te = np.array([r[0] for r in rows]).reshape(-1, 1) if rows else np.zeros((0, 1))
            ye = np.vstack([r[1].reshape(1, -1) for r in rows]) if rows else np.zeros((0, y0.size))
            ie = np.array([float(r[2]) for r in rows]).reshape(-1, 1) if rows else np.zeros((0, 1))
        if nargout <= 1:
            sid = next(_ODE_IDS)
            _ODE_SOLUTIONS[sid] = {"sol": sol.sol, "t0": float(t_steps[0]), "t1": float(t_steps[-1]),
                                   "rhs": rhs}
            while len(_ODE_SOLUTIONS) > _ODE_KEEP:
                _ODE_SOLUTIONS.popitem(last=False)
            fields = {
                "solver": char(solver_name), "x": t_steps.reshape(1, -1), "y": simplify(y_steps),
            }
            if te is not None:
                fields.update({"xe": te.reshape(1, -1), "ye": simplify(ye.T), "ie": ie.reshape(1, -1)})
            fields["stats"] = Struct.scalar({"nsteps": float(max(t_steps.size - 1, 0)),
                                             "nfevals": float(sol.nfev)})
            fields["idata"] = Struct.scalar({"id": float(sid)})
            return Struct.scalar(fields)
        if tspan.size > 2:
            forward = tspan[-1] >= tspan[0]
            tend = t_steps[-1]
            keep = tspan <= tend if forward else tspan >= tend
            tout = tspan[keep]
            if sol.status == 1 and (tout.size == 0 or tout[-1] != tend):
                tout = np.append(tout, tend)
            yout = sol.sol(tout) if tout.size else np.zeros((y0.size, 0))
            if tout.size:
                yout[:, 0] = y0
        else:
            tout = t_steps
            if refine > 1 and t_steps.size > 1:
                fine = [t_steps[:1]]
                for k in range(t_steps.size - 1):
                    fine.append(np.linspace(t_steps[k], t_steps[k + 1], refine + 1)[1:])
                tout = np.concatenate(fine)
                yout = sol.sol(tout)
                step_pos = np.arange(0, tout.size, refine)
                yout[:, step_pos] = y_steps  # exact solver values at the steps
            else:
                yout = y_steps
        outs = [tout.reshape(-1, 1), simplify(yout.T)]
        if nargout > 2:
            if te is None:
                te, ye, ie = np.zeros((0, 1)), np.zeros((0, y0.size)), np.zeros((0, 1))
            outs += [te, simplify(ye), ie]
        return tuple(outs[:nargout])
    return f


def _make_events(interp, ev_fn, t0, y0, extra):
    """Translate a MATLAB event function [value, isterminal, direction] = f(t, y)."""
    targs = [np.array([[float(t0)]]), np.real(y0).reshape(-1, 1) if not np.iscomplexobj(y0) else y0.reshape(-1, 1)]
    try:
        first = interp.call_function_value(ev_fn, targs + extra, 3)
    except MatlabError:
        first = interp.call_function_value(ev_fn, targs + extra, 1)
    if not first:
        raise MatlabError("The event function must return a value.")
    n = fnum(first[0]).size
    terminal = fnum(first[1]).flatten() if len(first) > 1 else np.zeros(n)
    direction = fnum(first[2]).flatten() if len(first) > 2 else np.zeros(n)
    terminal = np.broadcast_to(terminal, (n,)) if terminal.size in (1, n) else np.zeros(n)
    direction = np.broadcast_to(direction, (n,)) if direction.size in (1, n) else np.zeros(n)
    cache: dict = {}

    def values(t, y):
        key = (float(t), y.tobytes())
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            vals = interp.call_function_value(ev_fn, [np.array([[float(t)]]), y.reshape(-1, 1), *extra], 1)
            cache[key] = np.real(fnum(vals[0])).flatten()
        return cache[key]

    funcs = []
    for k in range(n):
        def ev(t, y, _k=k):
            return float(values(t, y)[_k])
        ev.terminal = bool(terminal[k])
        ev.direction = float(np.sign(direction[k]))
        funcs.append(ev)
    return funcs


builtin("ode45")(_ode("RK45", 4, "ode45"))
builtin("ode23")(_ode("RK23", 1, "ode23"))
builtin("ode113")(_ode("DOP853", 1, "ode113"))
builtin("ode78", "ode89")(_ode("DOP853", 1, "ode78"))
for _stiff in ("ode15s", "ode23s", "ode23t", "ode23tb"):
    builtin(_stiff)(_ode("BDF" if _stiff != "ode23s" else "Radau", 1, _stiff))


@builtin("odeset")
def _odeset(interp, args, nargout):
    s = Struct()
    start = 0
    if args and isinstance(args[0], Struct):
        s = args[0].copy()
        start = 1
    rest = args[start:]
    if len(rest) % 2:
        raise MatlabError("odeset expects name-value pairs.")
    for k in range(0, len(rest) - 1, 2):
        s.set_field(0, text(rest[k]), rest[k + 1])
    return s


@builtin("odeget")
def _odeget(interp, args, nargout):
    nargs(args, 2, 3, "odeget")
    v = _opt_value(args[0], text(args[1]))
    if v is None:
        return args[2] if len(args) > 2 else np.zeros((0, 0))
    return v


@builtin("deval")
def _deval(interp, args, nargout):
    nargs(args, 2, 3, "deval")
    if isinstance(args[0], Struct):
        sol, xq = args[0], fnum(args[1])
    else:
        xq, sol = fnum(args[0]), args[1]
    if not isinstance(sol, Struct) or "x" not in sol.fields:
        raise MatlabError("deval needs a solution structure returned by an ODE solver.")
    xf = np.real(xq).flatten(order="F").astype(float)
    entry = None
    if "idata" in sol.fields:
        idata = sol.get_field(0, "idata")
        if isinstance(idata, Struct) and "id" in idata.fields:
            entry = _ODE_SOLUTIONS.get(int(fnum(idata.get_field(0, "id")).flat[0]))
    t = fnum(sol.get_field(0, "x")).flatten()
    y = fnum(sol.get_field(0, "y"))
    lo, hi = min(t[0], t[-1]), max(t[0], t[-1])
    span = hi - lo
    if xf.size and (xf.min() < lo - 1e-12 * max(span, 1) or xf.max() > hi + 1e-12 * max(span, 1)):
        raise MatlabError(f"deval: points must lie in the solution interval [{lo:g}, {hi:g}].")
    if entry is not None:
        Y = np.asarray(entry["sol"](xf)).reshape(y.shape[0], -1)
    else:
        from scipy.interpolate import CubicSpline
        order = np.argsort(t)
        if t.size >= 2:
            Y = CubicSpline(t[order], y[:, order], axis=1)(xf)
        else:
            Y = np.repeat(y[:, :1], xf.size, axis=1)
    rows = None
    if len(args) > 2:
        rows = fnum(args[2]).flatten().astype(int) - 1
        Y = Y[rows, :]
    Y = simplify(Y)
    if nargout > 1:
        if entry is None:
            raise MatlabError("deval cannot compute derivatives for this solution structure.")
        full = np.asarray(entry["sol"](xf)).reshape(y.shape[0], -1)
        dY = np.column_stack([entry["rhs"](xf[k], full[:, k]) for k in range(xf.size)]) if xf.size else \
            np.zeros((y.shape[0], 0))
        if rows is not None:
            dY = dY[rows, :]
        return Y, simplify(dY)
    return Y
