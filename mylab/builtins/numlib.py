"""Smaller numeric functions: special matrices, rational approximation, outliers, tolerant unique."""

from __future__ import annotations

import math

import numpy as np

from ..errors import MatlabError
from ..values import Cell, as_2d, char, is_char, to_int, to_str
from ._registry import builtin
from ._util import bool_scalar, default_dim, flag_present, fnum, is_text, nargs, num, text


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

def _nan_test(name, np_fn, combine):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        a = num(args[0])
        if a.dtype.kind not in "fc":
            return bool_scalar(name == "allfinite")
        return bool_scalar(bool(combine(np_fn(a))))
    return f


builtin("anynan")(_nan_test("anynan", np.isnan, np.any))
builtin("allfinite")(_nan_test("allfinite", np.isfinite, np.all))


@builtin("standardizeMissing")
def _standardize_missing(interp, args, nargout):
    nargs(args, 2, 2, "standardizeMissing")
    v, ind = args
    if isinstance(v, Cell):
        marks = [to_str(x) for x in ind.values_f()] if isinstance(ind, Cell) else \
            ([to_str(ind)] if is_text(ind) else [])
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            item = v.data[idx]
            data[idx] = char("") if is_text(item) and to_str(item) in marks else item
        return Cell(data)
    if not isinstance(v, np.ndarray) or is_char(v):
        return v
    values = []
    for item in (ind.values_f() if isinstance(ind, Cell) else [ind]):
        if isinstance(item, np.ndarray) and item.dtype.kind in "biuf":
            values.extend(float(x) for x in item.flatten())
    if v.dtype.kind not in "fc" or not values:
        return v  # only floating-point arrays have a standard missing value (NaN)
    out = v.copy()
    out[np.isin(np.real(v), values) & (np.imag(v) == 0)] = np.nan
    return out


@builtin("isoutlier")
def _isoutlier(interp, args, nargout):
    nargs(args, 1, 6, "isoutlier")
    a = fnum(args[0])
    if a.dtype.kind == "c":
        raise MatlabError("isoutlier: the input must be real.")
    a = a.astype(float)
    method, factor, dim = "median", None, None
    rest = list(args[1:])
    k = 0
    while k < len(rest):
        r = rest[k]
        if is_text(r):
            name = text(r).lower()
            if name in ("median", "mean", "quartiles"):
                method = name
            elif name == "thresholdfactor" and k + 1 < len(rest):
                factor = float(np.real(fnum(rest[k + 1]).flat[0]))
                k += 1
            else:
                raise MatlabError(f"isoutlier: the method '{text(r)}' is not supported. Use 'median', "
                                  "'mean' or 'quartiles'.")
        else:
            dim = to_int(r, "dimension")
        k += 1
    if a.size == 0:
        return np.zeros(a.shape, dtype=bool)
    axis = dim - 1 if dim is not None else default_dim(a)
    if axis < 0:
        raise MatlabError("Dimension argument must be a positive integer.")
    if axis >= a.ndim:
        return np.zeros(a.shape, dtype=bool)
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "median":
            centre = np.nanmedian(a, axis=axis, keepdims=True)
            from scipy.special import erfcinv
            scale = -1 / (math.sqrt(2) * erfcinv(1.5)) * np.nanmedian(np.abs(a - centre), axis=axis, keepdims=True)
            lo, hi = centre - (factor or 3.0) * scale, centre + (factor or 3.0) * scale
        elif method == "mean":
            centre = np.nanmean(a, axis=axis, keepdims=True)
            n = np.sum(~np.isnan(a), axis=axis, keepdims=True)
            sd = np.nanstd(a, axis=axis, keepdims=True, ddof=0)
            scale = np.where(n > 1, sd * np.sqrt(n / np.maximum(n - 1, 1)), 0)
            lo, hi = centre - (factor or 3.0) * scale, centre + (factor or 3.0) * scale
        else:
            q1 = _quartile(a, axis, 0.25)
            q3 = _quartile(a, axis, 0.75)
            lo, hi = q1 - (factor or 1.5) * (q3 - q1), q3 + (factor or 1.5) * (q3 - q1)
        return (a < lo) | (a > hi)


def _quartile(a: np.ndarray, axis: int, p: float) -> np.ndarray:
    """Quantile with the (k - 0.5)/n plotting positions that MATLAB uses."""
    m = np.sort(np.moveaxis(a, axis, -1), axis=-1)
    out = np.full(m.shape[:-1] + (1,), np.nan)
    for idx in np.ndindex(m.shape[:-1]):
        v = m[idx]
        v = v[~np.isnan(v)]
        if v.size:
            pos = np.clip(p * v.size - 0.5, 0, v.size - 1)
            k = int(np.floor(pos))
            j = min(k + 1, v.size - 1)
            out[idx] = v[k] + (pos - k) * (v[j] - v[k])
    return np.moveaxis(out, -1, axis)


@builtin("uniquetol")
def _uniquetol(interp, args, nargout):
    rest, flags = flag_present(args, "stable")
    nargs(rest, 1, None, "uniquetol")
    a = fnum(rest[0])
    if a.dtype.kind == "c":
        raise MatlabError("uniquetol: the input must be real.")
    single = a.dtype == np.float32
    tol = 1e-6 if single else 1e-12
    scale = None
    opts = rest[1:]
    if opts and not is_text(opts[0]):
        tol = float(np.real(fnum(opts[0]).flat[0]))
        opts = opts[1:]
    if len(opts) % 2:
        raise MatlabError("uniquetol: options must be name-value pairs.")
    for k in range(0, len(opts), 2):
        name = text(opts[k]).lower()
        if name == "datascale":
            scale = float(np.real(fnum(opts[k + 1]).flat[0]))
        else:
            raise MatlabError(f"uniquetol: the option '{text(opts[k])}' is not supported.")
    flat = a.flatten(order="F").astype(float)
    if scale is None:
        finite = np.abs(flat[np.isfinite(flat)])
        scale = float(finite.max()) if finite.size else 1.0
    order = np.argsort(flat, kind="stable")  # NaNs last; each NaN is its own group
    limit = tol * scale
    reps: list[int] = []
    group = np.zeros(flat.size, dtype=int)
    start = None
    for pos in order:
        v = flat[pos]
        if start is None or np.isnan(v) or not (v - start <= limit):
            reps.append(int(pos))
            start = v
        group[pos] = len(reps) - 1
    ia = np.array(reps, dtype=int)
    if "stable" in flags:
        first = np.array([np.nonzero(group == g)[0][0] for g in range(len(reps))], dtype=int)
        pos = np.argsort(first, kind="stable")
        remap = np.empty(len(reps), dtype=int)
        remap[pos] = np.arange(len(reps))
        ia, group = first[pos], remap[group]
    vals = a.flatten(order="F")[ia]
    if a.shape == (0, 0):
        out = a
    elif a.ndim == 2 and a.shape[0] == 1 and a.size:
        out = vals.reshape(1, -1)
    else:
        out = vals.reshape(-1, 1)
    out = as_2d(out)
    if nargout > 1:
        return out, ia.reshape(-1, 1) + 1.0, group.reshape(-1, 1) + 1.0
    return out
