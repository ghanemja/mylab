"""Smaller numeric functions: special matrices, rational approximation, NaN tests."""

from __future__ import annotations

import math

import numpy as np

from ..errors import MatlabError
from ..values import char, to_int
from ._registry import builtin
from ._util import default_dim, fnum, nargs


# -- special matrices -------------------------------------------------------------------

@builtin("invhilb")
def _invhilb(interp, args, nargout):
    nargs(args, 1, 2, "invhilb")
    n = to_int(args[0], "n")
    if n < 0:
        raise MatlabError("invhilb: n must be a nonnegative integer.")
    out = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            v = (i + j - 1) * math.comb(n + i - 1, n - j) * math.comb(n + j - 1, n - i) \
                * math.comb(i + j - 2, i - 1) ** 2
            out[i - 1, j - 1] = -v if (i + j) % 2 else v
    return out


@builtin("compan")
def _compan(interp, args, nargout):
    nargs(args, 1, 1, "compan")
    p = fnum(args[0])
    if p.size and not (p.ndim == 2 and 1 in p.shape):
        raise MatlabError("compan: the input must be a vector of polynomial coefficients.")
    p = p.flatten()
    n = p.size - 1
    if n < 1:
        return np.zeros((0, 0))
    if p[0] == 0:
        raise MatlabError("compan: the first coefficient must not be zero.")
    out = np.zeros((n, n), dtype=p.dtype)
    out[0, :] = -p[1:] / p[0]
    if n > 1:
        out[np.arange(1, n), np.arange(0, n - 1)] = 1
    return out


def _second_difference(u: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Second derivative along one axis; the two end values are extrapolated linearly."""
    m = np.moveaxis(u, axis, 0)
    n = m.shape[0]
    d = np.zeros(m.shape, dtype=m.dtype)
    if n >= 3:
        d[1:-1] = (m[2:] - 2 * m[1:-1] + m[:-2]) / (h * h)
        if n == 3:
            d[0] = d[1]
            d[-1] = d[1]
        else:
            d[0] = 2 * d[1] - d[2]
            d[-1] = 2 * d[-2] - d[-3]
    return np.moveaxis(d, 0, axis)


@builtin("del2")
def _del2(interp, args, nargout):
    nargs(args, 1, None, "del2")
    u = fnum(args[0])
    u = u.astype(complex if u.dtype.kind == "c" else float)
    if u.size == 0:
        return u
    is_vector = u.ndim == 2 and 1 in u.shape
    axes = [default_dim(u)] if is_vector else list(range(u.ndim))
    steps = [fnum(a) for a in args[1:]]
    if any(s.size != 1 for s in steps):
        raise MatlabError("del2: only scalar spacings are supported, for example del2(U, h) "
                          "or del2(U, hx, hy).")
    hs = [float(np.real(s.flat[0])) for s in steps]
    if len(hs) == 0:
        hs = [1.0] * len(axes)
    elif len(hs) == 1:
        hs = hs * len(axes)
    elif len(hs) != len(axes):
        raise MatlabError("del2: give one spacing, or one spacing for every dimension.")
    elif not is_vector:
        hs[0], hs[1] = hs[1], hs[0]  # the first spacing is for the columns (x direction)
    total = np.zeros(u.shape, dtype=u.dtype)
    for axis, h in zip(axes, hs):
        total = total + _second_difference(u, axis, h)
    return total / (4.0 if len(axes) <= 2 else 2.0 * len(axes))


# -- rational approximation ---------------------------------------------------------------

def _rat_terms(x: float, tol: float) -> tuple[float, float, list[float]]:
    """Continued fraction expansion of x: numerator, denominator and the terms used."""
    if not math.isfinite(x):
        return (x if not math.isnan(x) else 0.0), (1.0 if not math.isnan(x) else 0.0), [x]
    n0, d0, n1, d1 = 1.0, 0.0, 0.0, 1.0  # columns of the convergent matrix
    y = x
    terms = []
    while True:
        t = float(math.floor(abs(y) + 0.5)) * (1 if y >= 0 else -1)
        terms.append(t)
        frac = y - t
        n0, n1 = n0 * t + n1, n0
        d0, d1 = d0 * t + d1, d0
        if frac == 0 or abs(n0 / d0 - x) <= max(tol, np.spacing(abs(x))) or len(terms) > 60:
            break
        y = 1 / frac
    if d0 < 0:
        n0, d0 = -n0, -d0
    return n0, d0, terms


@builtin("rat")
def _rat(interp, args, nargout):
    nargs(args, 1, 2, "rat")
    x = fnum(args[0])
    if x.dtype.kind == "c":
        raise MatlabError("rat: the input must be real.")
    x = x.astype(float)
    finite = x[np.isfinite(x)]
    tol = float(np.real(fnum(args[1]).flat[0])) if len(args) > 1 else 1e-6 * float(np.sum(np.abs(finite)))
    flat = x.flatten(order="F")
    parts = [_rat_terms(float(v), tol) for v in flat]
    if nargout >= 2:
        n = np.array([p[0] for p in parts]).reshape(x.shape, order="F")
        d = np.array([p[1] for p in parts]).reshape(x.shape, order="F")
        return n, d

    def fmt(t):
        return f"{t:.0f}" if math.isfinite(t) else ("Inf" if t > 0 else "-Inf" if t < 0 else "NaN")

    def expansion(terms):
        s = fmt(terms[0])
        for t in terms[1:-1]:
            s += f" + 1/({fmt(t)}"
        if len(terms) > 1:
            last = fmt(terms[-1])
            s += f" + 1/({last})" if terms[-1] < 0 else f" + 1/{last}"
        return s + ")" * max(len(terms) - 2, 0)

    rows = [expansion(p[2]) for p in parts]
    if not rows:
        return char("")
    width = max(len(r) for r in rows)
    return np.array([list(r.ljust(width)) for r in rows], dtype="<U1").reshape(len(rows), width)


# -- missing values and outliers ------------------------------------------------------------

