"""Elementwise math, rounding, reductions, sorting and set operations."""

from __future__ import annotations


import math
from typing import Any

import numpy as np
import scipy.special as sp

from ..errors import MatlabError
from ..ops import binop, check_integer_mix, unary
from ..values import Cell, MString, StringArray, char, is_char, simplify, string_result, to_int, to_str
from ._registry import builtin
from ._util import (
    as_row_or_col, default_dim, flag_present, fnum, get_dim, is_text, keep2d, nargs, num, text,
)


# -- elementwise ------------------------------------------------------------------------

_F64 = np.dtype(float)


def _scalar_arg(v):
    """The float value of a 1x1 double array, else None (for fast scalar paths)."""
    if type(v) is np.ndarray and v.dtype is _F64 and v.size == 1:
        return v.item()
    return None


def _unary_math(fn, *, allow_int=False, scalar=None):
    def f(interp, args, nargout):
        if scalar is not None and len(args) == 1:
            x = _scalar_arg(args[0])
            if x is not None:
                try:
                    r = scalar(x)
                except (ValueError, OverflowError, ZeroDivisionError):
                    r = None
                if r is not None:
                    return np.array([[r]])
        nargs(args, 1, 1, "function")
        a = num(args[0])
        if a.dtype.kind in "iu" and not allow_int:
            a = a.astype(float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = fn(a)
        return simplify(np.asarray(r))
    return f


def _round_half_away(a):
    if a.dtype.kind == "c":
        return _round_half_away(a.real) + 1j * _round_half_away(a.imag)
    if a.dtype.kind != "f":
        return a.astype(float)
    # floor(|a| + 0.5) is wrong where the sum rounds up (0.49999999999999994, odd values near 2^53).
    m = np.abs(a)
    r = np.floor(m)
    with np.errstate(invalid="ignore"):
        r = r + (m - r >= 0.5)
    return np.copysign(r, a)  # a zero result keeps the sign of the input, as in MATLAB


_ELEMENTWISE = {
    "sqrt": np.emath.sqrt, "exp": np.exp, "log": np.emath.log, "log2": np.emath.log2,
    "log10": np.emath.log10, "log1p": np.log1p, "expm1": np.expm1,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.emath.arcsin, "acos": np.emath.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": lambda a: np.emath.log(a + np.emath.sqrt(a * a - 1)),
    "atanh": lambda a: 0.5 * np.emath.log((1 + a) / (1 - a)),
    "sec": lambda a: 1 / np.cos(a), "csc": lambda a: 1 / np.sin(a), "cot": lambda a: 1 / np.tan(a),
    "sind": lambda a: _sind(a), "cosd": lambda a: _sind(np.asarray(a) + 90.0),
    "tand": lambda a: _sind(a) / _sind(np.asarray(a) + 90.0),
    "asind": lambda a: np.degrees(np.emath.arcsin(a)), "acosd": lambda a: np.degrees(np.emath.arccos(a)),
    "atand": lambda a: np.degrees(np.arctan(a)),
    "deg2rad": np.radians, "rad2deg": np.degrees,
    "floor": np.floor, "ceil": np.ceil, "fix": np.trunc,
    "real": np.real, "imag": np.imag, "conj": np.conj, "angle": np.angle,
    "gamma": sp.gamma, "gammaln": sp.gammaln, "erf": sp.erf, "erfc": sp.erfc,
    "erfinv": sp.erfinv, "erfcinv": sp.erfcinv,
}


def _sind(a):
    a = np.asarray(a, dtype=float)
    r = np.sin(np.radians(a))
    exact = np.mod(a, 180) == 0
    return np.where(exact, 0.0, r)


_SCALAR = {
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else None,
    "exp": lambda x: math.exp(x) if x < 700 else None,
    "log": lambda x: math.log(x) if x > 0 else None,
    "log2": lambda x: math.log2(x) if x > 0 else None,
    "log10": lambda x: math.log10(x) if x > 0 else None,
    "sin": lambda x: math.sin(x) if math.isfinite(x) else None,
    "cos": lambda x: math.cos(x) if math.isfinite(x) else None,
    "tan": lambda x: math.tan(x) if math.isfinite(x) else None,
    "atan": math.atan,
    "floor": lambda x: math.copysign(float(math.floor(x)), x) if math.isfinite(x) else None,
    "ceil": lambda x: math.copysign(float(math.ceil(x)), x) if math.isfinite(x) else None,
    "fix": lambda x: math.copysign(float(math.trunc(x)), x) if math.isfinite(x) else None,
}

for _name, _fn in _ELEMENTWISE.items():
    builtin(_name)(_unary_math(_fn, scalar=_SCALAR.get(_name)))


@builtin("abs")
def _abs(interp, args, nargout):
    a = num(args[0])
    if a.dtype.kind == "i":
        from ..values import saturate_int
        return saturate_int(np.abs(a.astype(float)), a.dtype)
    return np.abs(a)


@builtin("sign")
def _sign(interp, args, nargout):
    a = num(args[0])
    if a.dtype.kind == "c":
        with np.errstate(invalid="ignore"):
            return np.where(a == 0, 0, a / np.abs(a))
    return np.sign(a).astype(a.dtype if a.dtype.kind in "iu" else float)


@builtin("round")
def _round(interp, args, nargout):
    a = num(args[0])
    if a.dtype.kind in "iu":
        return a
    if len(args) > 1:
        n = to_int(args[1], "digits")
        f = 10.0 ** n
        return _round_half_away(a * f) / f
    return _round_half_away(a)


@builtin("isnan")
def _isnan(interp, args, nargout):
    a = num(args[0])
    return np.isnan(a) if a.dtype.kind in "fc" else np.zeros(a.shape, bool)


@builtin("isinf")
def _isinf(interp, args, nargout):
    a = num(args[0])
    return np.isinf(a) if a.dtype.kind in "fc" else np.zeros(a.shape, bool)


@builtin("isfinite")
def _isfinite(interp, args, nargout):
    a = num(args[0])
    return np.isfinite(a) if a.dtype.kind in "fc" else np.ones(a.shape, bool)


def _binary_math(fn, scalar=None):
    def f(interp, args, nargout):
        if scalar is not None and len(args) == 2:
            x, y = _scalar_arg(args[0]), _scalar_arg(args[1])
            if x is not None and y is not None:
                try:
                    r = scalar(x, y)
                except (ValueError, OverflowError, ZeroDivisionError):
                    r = None
                if r is not None:
                    return np.array([[r]])
        nargs(args, 2, 2, "function")
        a, b = fnum(args[0]), fnum(args[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return simplify(np.asarray(fn(a, b)))
    return f


def _mod(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a - np.floor(a / b) * b
    r = np.where(b == 0, a, r)
    # Results within roundoff of b are reported as 0, as MATLAB does.
    close = (b != 0) & np.isclose(r, b, rtol=0, atol=np.finfo(float).eps * np.maximum(np.abs(a), 1) * 4)
    return np.where(close, 0.0, r)


def _rem(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a - np.trunc(a / b) * b
    return np.where(b == 0, np.where(np.isinf(a), np.nan, np.nan), r)


builtin("atan2")(_binary_math(np.arctan2, math.atan2))
builtin("atan2d")(_binary_math(lambda a, b: np.degrees(np.arctan2(a, b))))
builtin("hypot")(_binary_math(np.hypot, math.hypot))
def _mod_scalar(a, b):
    if b == 0:
        return a
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    r = a - math.floor(a / b) * b
    return 0.0 if abs(r - b) <= 4 * 2.220446049250313e-16 * max(abs(a), 1.0) else r


def _rem_scalar(a, b):
    if b == 0 or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return a - math.trunc(a / b) * b


builtin("mod")(_binary_math(_mod, _mod_scalar))
builtin("rem")(_binary_math(_rem, _rem_scalar))
def _nthroot(a, n):
    bad = (a < 0) & ~((n == np.round(n)) & (np.mod(n, 2) == 1))
    if np.any(bad):
        raise MatlabError("If X is negative, N must be an odd integer.")
    return np.sign(a) * np.abs(a) ** (1.0 / n)


builtin("nthroot")(_binary_math(_nthroot))
builtin("beta")(_binary_math(sp.beta))


@builtin("complex")
def _complex(interp, args, nargout):
    re = fnum(args[0])
    im = fnum(args[1]) if len(args) > 1 else np.zeros_like(re)
    return np.asarray(re + 1j * im, dtype=complex)


@builtin("power")
def _power(interp, args, nargout):
    return binop(".^", args[0], args[1])


_FUNCTIONAL_OPS = {
    "plus": "+", "minus": "-", "times": ".*", "rdivide": "./", "ldivide": ".\\",
    "mtimes": "*", "mrdivide": "/", "mldivide": "\\", "mpower": "^",
    "eq": "==", "ne": "~=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "and": "&", "or": "|",
}


def _op_fn(op):
    def f(interp, args, nargout):
        nargs(args, 2, 2, op)
        return binop(op, args[0], args[1])
    return f


for _name, _op in _FUNCTIONAL_OPS.items():
    builtin(_name)(_op_fn(_op))


@builtin("not")
def _not(interp, args, nargout):
    return unary("~", args[0])


@builtin("uminus")
def _uminus(interp, args, nargout):
    return unary("-", args[0])


@builtin("uplus")
def _uplus(interp, args, nargout):
    return args[0]


@builtin("xor")
def _xor(interp, args, nargout):
    return np.logical_xor(num(args[0]) != 0, num(args[1]) != 0)


@builtin("transpose")
def _transpose(interp, args, nargout):
    from ..ops import transpose
    return transpose(args[0], False)


@builtin("ctranspose")
def _ctranspose(interp, args, nargout):
    from ..ops import transpose
    return transpose(args[0], True)


# -- integer / number theory --------------------------------------------------------

@builtin("factorial")
def _factorial(interp, args, nargout):
    a = fnum(args[0])
    return sp.gamma(a + 1).round()


@builtin("nchoosek")
def _nchoosek(interp, args, nargout):
    n, k = args
    nv = fnum(n)
    kk = to_int(k, "k")
    if nv.size == 1:
        return float(sp.comb(int(nv.flat[0]), kk, exact=True))
    from itertools import combinations
    items = nv.flatten()
    combos = list(combinations(items, kk))
    return np.array(combos, dtype=float).reshape(len(combos), kk)


@builtin("primes")
def _primes(interp, args, nargout):
    n = int(fnum(args[0]).flat[0])
    if n < 2:
        return np.zeros((1, 0))
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0].astype(float).reshape(1, -1)


@builtin("isprime")
def _isprime(interp, args, nargout):
    a = fnum(args[0])

    def isp(x):
        if x < 2 or x != int(x):
            return False
        x = int(x)
        if x < 4:
            return True
        if x % 2 == 0:
            return False
        for d in range(3, int(x ** 0.5) + 1, 2):
            if x % d == 0:
                return False
        return True

    return np.vectorize(isp, otypes=[bool])(a) if a.size else np.zeros(a.shape, bool)


@builtin("factor")
def _factor(interp, args, nargout):
    n = int(fnum(args[0]).flat[0])
    out = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1 or not out:
        out.append(n)
    return np.array(out, dtype=float).reshape(1, -1)


@builtin("gcd")
def _gcd(interp, args, nargout):
    a, b = np.broadcast_arrays(fnum(args[0]), fnum(args[1]))
    return np.gcd(a.astype(np.int64), b.astype(np.int64)).astype(float)


@builtin("lcm")
def _lcm(interp, args, nargout):
    a, b = np.broadcast_arrays(fnum(args[0]), fnum(args[1]))
    return np.lcm(a.astype(np.int64), b.astype(np.int64)).astype(float)


# -- reductions ---------------------------------------------------------------------------

def _reduce(np_fn, empty_value, name, *, nan_fn=None, int_ok=True):
    def f(interp, args, nargout):
        nargs(args, 1, 4, name)
        rest, flags = flag_present(args, "omitnan", "includenan", "all", "double", "native", "default")
        a = num(rest[0])
        if a.dtype == bool or (a.dtype.kind in "iu" and not int_ok):
            a = a.astype(float)
        fn = nan_fn if ("omitnan" in flags and nan_fn is not None) else np_fn
        int_class = a.dtype if (a.dtype.kind in "iu" and "double" not in flags) else None
        if int_class is not None:
            a = a.astype(float)  # integer sums keep their class and saturate, as in MATLAB

        def finish(r):
            if int_class is None:
                return r
            from ..values import saturate_int
            return saturate_int(np.asarray(r, dtype=float), int_class)

        if "all" in flags:
            return finish(fn(a.flatten())) if a.size else empty_value
        if a.shape == (0, 0) and len(rest) < 2:
            return empty_value
        axis = get_dim(rest, 1, a)
        if axis >= a.ndim:
            return finish(a)
        return finish(keep2d(np.asarray(fn(a, axis=axis)), axis, a.ndim))
    return f


builtin("sum")(_reduce(np.sum, 0.0, "sum", nan_fn=np.nansum))
builtin("prod")(_reduce(np.prod, 1.0, "prod", nan_fn=np.nanprod))
builtin("mean")(_reduce(np.mean, np.nan, "mean", nan_fn=np.nanmean, int_ok=False))
builtin("median")(_reduce(np.median, np.nan, "median", nan_fn=np.nanmedian, int_ok=False))


def _anyall(np_fn, empty_value):
    def f(interp, args, nargout):
        rest, flags = flag_present(args, "all")
        a = num(rest[0]) != 0
        if "all" in flags:
            return bool(np_fn(a))
        if a.shape == (0, 0) and len(rest) < 2:
            return np.array([[empty_value]])
        axis = get_dim(rest, 1, a)
        return keep2d(np_fn(a, axis=axis), axis, a.ndim)
    return f


builtin("any")(_anyall(np.any, False))
builtin("all")(_anyall(np.all, True))


def _std_var(power):
    def f(interp, args, nargout):
        rest, flags = flag_present(args, "omitnan", "includenan")
        a = fnum(rest[0])
        w = 0
        if len(rest) > 1 and not (isinstance(rest[1], np.ndarray) and rest[1].size == 0):
            w = to_int(rest[1], "weight")
        if a.size == 0:
            return np.nan
        axis = get_dim(rest, 2, a)
        n = a.shape[axis]
        ddof = 0 if (w == 1 or n == 1) else 1
        fn = np.nanvar if "omitnan" in flags else np.var
        v = keep2d(np.asarray(fn(a, axis=axis, ddof=ddof)), axis, a.ndim)
        return np.sqrt(v) if power == 0.5 else v
    return f


builtin("std")(_std_var(0.5))
builtin("var")(_std_var(1))


@builtin("mode")
def _mode(interp, args, nargout):
    a = fnum(args[0])
    if a.size == 0:
        return np.nan
    axis = get_dim(args, 1, a)

    def mode1(v):
        vals, counts = np.unique(v[~np.isnan(v)], return_counts=True)
        if vals.size == 0:
            return np.nan
        return vals[np.argmax(counts)]

    r = np.apply_along_axis(mode1, axis, a)
    return keep2d(np.asarray(r), axis, a.ndim)


def _cumulative(np_fn):
    def f(interp, args, nargout):
        rest, flags = flag_present(args, "reverse", "forward", "omitnan", "includenan")
        a = num(rest[0])
        if a.dtype == bool:
            a = a.astype(float)
        int_class = a.dtype if a.dtype.kind in "iu" else None
        if int_class is not None:
            a = a.astype(float)
        axis = get_dim(rest, 1, a)
        if "reverse" in flags:
            r = np.flip(np_fn(np.flip(a, axis), axis=axis), axis)
        else:
            r = np_fn(a, axis=axis)
        if int_class is not None:
            from ..values import saturate_int
            r = saturate_int(r, int_class)
        return r
    return f


builtin("cumsum")(_cumulative(np.cumsum))
builtin("cumprod")(_cumulative(np.cumprod))
builtin("cummax")(_cumulative(np.maximum.accumulate))
builtin("cummin")(_cumulative(np.minimum.accumulate))


def _minmax(is_max):
    np_arg = np.nanargmax if is_max else np.nanargmin
    np_pair = np.fmax if is_max else np.fmin

    def f(interp, args, nargout):
        nargs(args, 1, 4, "max" if is_max else "min")
        rest, flags = flag_present(args, "omitnan", "includenan", "all")
        a = num(rest[0])
        if len(rest) >= 2 and not (isinstance(rest[1], np.ndarray) and rest[1].size == 0):
            b = num(rest[1])
            check_integer_mix(a, b)
            int_class = a.dtype if a.dtype.kind in "iu" else (b.dtype if b.dtype.kind in "iu" else None)
            r = np_pair(a.astype(float) if a.dtype.kind in "iu" else a,
                        b.astype(float) if b.dtype.kind in "iu" else b)
            if int_class is not None:
                from ..values import saturate_int
                r = saturate_int(r, int_class)
            return r
        if a.size == 0:
            return (np.zeros((0, 0)), np.zeros((0, 0))) if nargout > 1 else np.zeros((0, 0))
        cmp = np.real(a) if a.dtype.kind != "c" else np.abs(a)
        if "all" in flags:
            flat = cmp.flatten(order="F")
            if np.all(np.isnan(flat)) if flat.dtype.kind == "f" else False:
                return (np.nan, 1.0) if nargout > 1 else np.nan
            k = int(np_arg(flat))
            v = a.flatten(order="F")[k]
            return (v, float(k + 1)) if nargout > 1 else v
        axis = get_dim(rest, 2, a)
        if axis >= a.ndim:
            return (a, np.ones(a.shape)) if nargout > 1 else a
        with np.errstate(invalid="ignore"):
            if cmp.dtype.kind == "f":
                allnan = np.all(np.isnan(cmp), axis=axis)
                safe = np.where(np.isnan(cmp), -np.inf if is_max else np.inf, cmp)
                idx = (np.argmax if is_max else np.argmin)(safe, axis=axis)
                idx = np.where(allnan, 0, idx)
            else:
                idx = (np.argmax if is_max else np.argmin)(cmp, axis=axis)
        vals = np.take_along_axis(a, np.expand_dims(idx, axis), axis)
        if cmp.dtype.kind == "f":
            vals = np.where(np.expand_dims(allnan, axis), np.nan, vals)
        if nargout > 1:
            return vals, keep2d(idx.astype(float) + 1, axis, a.ndim)
        return vals
    return f


builtin("max")(_minmax(True))
builtin("min")(_minmax(False))


@builtin("bounds")
def _bounds(interp, args, nargout):
    a = fnum(args[0])
    return (np.nanmin(a), np.nanmax(a))


@builtin("range")
def _range(interp, args, nargout):
    a = fnum(args[0])
    axis = get_dim(args, 1, a)
    return keep2d(np.ptp(a, axis=axis), axis, a.ndim)


@builtin("diff")
def _diff(interp, args, nargout):
    a = num(args[0])
    n = to_int(args[1], "n") if len(args) > 1 else 1
    if a.size == 0:
        return np.zeros((0, 0))
    axis = get_dim(args, 2, a)
    if a.shape[axis] <= n:
        shape = list(a.shape)
        shape[axis] = 0
        return np.zeros(shape)
    return np.diff(a.astype(float) if a.dtype == bool else a, n=n, axis=axis)


@builtin("trapz")
def _trapz(interp, args, nargout):
    if len(args) == 1:
        y = fnum(args[0])
        x = None
    else:
        x, y = fnum(args[0]).flatten(), fnum(args[1])
    axis = default_dim(y)
    trap = getattr(np, "trapezoid", None) or np.trapz
    r = trap(y, x=x, axis=axis) if x is not None else trap(y, axis=axis)
    return keep2d(np.asarray(r), axis, y.ndim)


@builtin("cumtrapz")
def _cumtrapz(interp, args, nargout):
    from scipy.integrate import cumulative_trapezoid
    if len(args) == 1:
        y = fnum(args[0])
        x = None
    else:
        x, y = fnum(args[0]).flatten(), fnum(args[1])
    axis = default_dim(y)
    r = cumulative_trapezoid(y, x=x, axis=axis, initial=0)
    return r


@builtin("dot")
def _dot(interp, args, nargout):
    a, b = num(args[0]), num(args[1])
    if a.size != b.size:
        raise MatlabError("A and B must be the same size.")
    if (a.shape[0] == 1 or a.shape[1] == 1) and (b.shape[0] == 1 or b.shape[1] == 1):
        return simplify(np.asarray(np.vdot(a.flatten(), b.flatten())))
    return keep2d(np.sum(np.conj(a) * b, axis=0), 0)


@builtin("cross")
def _cross(interp, args, nargout):
    a, b = num(args[0]), num(args[1])
    if a.size == 3 and b.size == 3:
        return as_row_or_col(np.cross(a.flatten(), b.flatten()), a)
    if a.shape != b.shape:
        raise MatlabError("cross: A and B must be the same size.")
    if len(args) > 2:
        axis = to_int(args[2], "dim") - 1
    else:
        sizes = [d for d, n in enumerate(a.shape) if n == 3]
        if not sizes:
            raise MatlabError("cross: A and B must have a dimension of length 3.")
        axis = sizes[0]
    if a.shape[axis] != 3:
        raise MatlabError("cross: the chosen dimension must have length 3.")
    return np.cross(a, b, axis=axis)


@builtin("kron")
def _kron(interp, args, nargout):
    return np.kron(num(args[0]), num(args[1]))


# -- find / sort / set operations ----------------------------------------------------

@builtin("find")
def _find(interp, args, nargout):
    a = num(args[0])
    flat = a.flatten(order="F")
    idx = np.nonzero(flat)[0]
    if len(args) > 1:
        k = to_int(args[1], "n")
        direction = text(args[2]) if len(args) > 2 else "first"
        idx = idx[-k:] if direction == "last" else idx[:k]
    if nargout <= 1:
        if a.shape[0] == 1 and a.ndim == 2 and a.shape != (1, 1) or (a.shape == (1, 1)):
            return (idx + 1.0).reshape(1, -1)
        return (idx + 1.0).reshape(-1, 1)
    rows = idx % a.shape[0] if a.shape[0] else idx
    cols = idx // a.shape[0] if a.shape[0] else idx
    shape = (1, -1) if a.shape[0] == 1 else (-1, 1)
    outs = [(rows + 1.0).reshape(shape), (cols + 1.0).reshape(shape)]
    if nargout > 2:
        outs.append(flat[idx].reshape(shape))
    return tuple(outs)


@builtin("nnz")
def _nnz(interp, args, nargout):
    return float(np.count_nonzero(num(args[0])))


@builtin("sort")
def _sort(interp, args, nargout):
    rest, flags = flag_present(args, "ascend", "descend")
    v = rest[0]
    if isinstance(v, StringArray):
        from .strings import _as_cell_if_strings
        res = _sort(interp, [_as_cell_if_strings(v), *rest[1:], *(char(f) for f in flags)], nargout)
        cell = res[0] if isinstance(res, tuple) else res
        out = string_result(np.vectorize(to_str, otypes=[object])(cell.data))
        return (out, res[1]) if isinstance(res, tuple) else out
    if isinstance(v, Cell):
        strs = [to_str(x) for x in v.values_f()]
        order = sorted(range(len(strs)), key=lambda k: strs[k], reverse="descend" in flags)
        data = np.empty(v.shape, dtype=object)
        flat = [char(strs[k]) for k in order]
        for k, s in enumerate(flat):
            data[k % v.shape[0], k // v.shape[0]] = s
        idx = np.array(order, dtype=float).reshape(v.shape, order="F") + 1
        return (Cell(data), idx) if nargout > 1 else Cell(data)
    a = v if is_char(v) else num(v)
    work = num(a) if is_char(a) else a
    if work.size == 0:
        return (a, np.zeros(a.shape)) if nargout > 1 else a
    axis = get_dim(rest, 1, work)
    key = np.abs(work) if work.dtype.kind == "c" else work
    if "descend" in flags:
        # stable descending sort; NaNs first, as in MATLAB
        fkey = key.astype(float) if key.dtype.kind in "iub" else key
        idx = np.argsort(-np.where(np.isnan(fkey), np.inf, fkey), axis=axis, kind="stable")
    else:
        idx = np.argsort(key, axis=axis, kind="stable")
    out = np.take_along_axis(a, idx, axis)
    if nargout > 1:
        return out, idx.astype(float) + 1
    return out


@builtin("sortrows")
def _sortrows(interp, args, nargout):
    nargs(args, 1, 3, "sortrows")
    v = args[0]
    descend = any(is_text(x) and text(x).lower() == "descend" for x in args[1:])
    col_arg = next((x for x in args[1:] if not is_text(x) and not isinstance(x, Cell)), None)
    if isinstance(v, (Cell, StringArray)):
        data = v.data
        ncols = data.shape[1]
        cols = [int(c) for c in num(col_arg).flatten()] if col_arg is not None and np.size(col_arg) else \
            list(range(1, ncols + 1))
        order = np.arange(data.shape[0])
        for c in reversed(cols):
            j = abs(c) - 1
            if j >= ncols:
                raise MatlabError("sortrows: column index exceeds the number of columns.")
            desc = (c < 0) ^ descend
            column = [data[i, j] for i in order]
            if all(x is None or isinstance(x, str) or is_text(x) for x in column):
                keys = ["" if x is None else to_str(x) for x in column]
            elif all(isinstance(x, np.ndarray) and x.size == 1 and x.dtype.kind in "biuf" for x in column):
                keys = [float(x.flat[0]) for x in column]
            else:
                raise MatlabError("sortrows on a cell array needs columns of text or of numeric scalars.")
            # stable in both directions: equal keys keep their order
            o = sorted(range(len(keys)), key=(lambda i: (_neg_key(keys[i]), i)) if desc else
                       (lambda i: (keys[i], i)))
            order = order[np.array(o, dtype=int)] if o else order
        out = type(v)(data[order, :])
        return (out, (order + 1.0).reshape(-1, 1)) if nargout > 1 else out
    a = num(v)
    cols = [int(c) for c in num(col_arg).flatten()] if col_arg is not None and np.size(col_arg) else \
        list(range(1, a.shape[1] + 1))
    order = np.arange(a.shape[0])
    for c in reversed(cols):
        if abs(c) - 1 >= a.shape[1] or c == 0:
            raise MatlabError("sortrows: column index out of range.")
        col = a[order, abs(c) - 1]
        if col.dtype.kind == "c":
            col = np.abs(col)
        desc = (c < 0) ^ descend
        if desc:
            fcol = col.astype(float)
            o = np.argsort(-np.where(np.isnan(fcol), np.inf, fcol), kind="stable")
        else:
            o = np.argsort(col, kind="stable")
        order = order[o]
    out = v[order, :]
    return (out, (order + 1.0).reshape(-1, 1)) if nargout > 1 else out


def _neg_key(k):
    if isinstance(k, str):
        return tuple(-ord(ch) for ch in k) + (1,)
    return -k


@builtin("unique")
def _unique(interp, args, nargout):
    rest, flags = flag_present(args, "stable", "sorted", "rows", "first", "last")
    v = rest[0]
    if isinstance(v, StringArray):
        from .strings import _as_cell_if_strings
        res = _unique(interp, [_as_cell_if_strings(v), *(char(f) for f in flags)], nargout)
        cell = res[0] if isinstance(res, tuple) else res
        out = string_result(np.vectorize(to_str, otypes=[object])(cell.data))
        return (out, *res[1:]) if isinstance(res, tuple) else out
    if isinstance(v, Cell):
        strs = [to_str(x) for x in v.values_f()]
        if "stable" in flags:
            seen: dict[str, int] = {}
            for k, s in enumerate(strs):
                seen.setdefault(s, k)
            uniq = list(seen)
            ia = [seen[s] for s in uniq]
        else:
            uniq = sorted(set(strs))
            ia = [strs.index(s) for s in uniq]
        ic = [uniq.index(s) for s in strs]
        out = Cell.from_list([char(s) for s in uniq], column=v.shape[0] != 1 or v.shape[1] == 1)
        if nargout > 1:
            return out, np.array(ia, float).reshape(-1, 1) + 1, np.array(ic, float).reshape(-1, 1) + 1
        return out
    a = num(v) if not is_char(v) else v
    if "rows" in flags:
        uniq, ia, ic = np.unique(a, axis=0, return_index=True, return_inverse=True)
        if nargout > 1:
            return uniq, ia.reshape(-1, 1) + 1.0, ic.reshape(-1, 1) + 1.0
        return uniq
    flat = a.flatten(order="F")
    uniq, ia, ic = np.unique(flat, return_index=True, return_inverse=True)
    if "stable" in flags:
        order = np.argsort(ia)
        uniq = uniq[order]
        ia = ia[order]
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        ic = remap[ic]
    if "last" in flags:
        ia = np.array([np.nonzero(flat == u)[0][-1] for u in uniq])
    is_row = a.shape[0] == 1 and a.size > 0
    out = uniq.reshape(1, -1) if is_row else uniq.reshape(-1, 1)
    if a.size == 0:
        out = a.reshape(0, 0) if a.shape == (0, 0) else out
    if nargout > 1:
        return out, ia.reshape(-1, 1) + 1.0, ic.reshape(-1, 1) + 1.0
    return out


def _set_values(v):
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()], True
    return list(num(v).flatten(order="F")), False


def _set_result(vals, is_cell, like_row=True):
    if is_cell:
        return Cell.from_list([char(s) for s in vals])
    arr = np.array(vals, dtype=float)
    return arr.reshape(1, -1) if like_row else arr.reshape(-1, 1)


def _set_op(kind):
    def f(interp, args, nargout):
        rest, flags = flag_present(args, "stable", "sorted", "rows")
        a, ca = _set_values(rest[0])
        b, cb = _set_values(rest[1])
        sb = set(b)
        sa = set(a)
        if kind == "intersect":
            vals = [x for x in a if x in sb]
        elif kind == "union":
            vals = a + b
        elif kind == "setdiff":
            vals = [x for x in a if x not in sb]
        else:  # setxor
            vals = [x for x in a if x not in sb] + [x for x in b if x not in sa]
        if "stable" in flags:
            seen = []
            for x in vals:
                if x not in seen:
                    seen.append(x)
            vals = seen
        else:
            vals = sorted(set(vals))
        row = not (isinstance(rest[0], np.ndarray) and rest[0].shape[1] == 1 and rest[0].shape[0] > 1)
        out = _set_result(vals, ca or cb, row)
        if nargout > 1:
            ia = [a.index(x) + 1.0 for x in vals if x in a]
            outs = [out, np.array(ia).reshape(-1, 1)]
            if nargout > 2:
                ib = [b.index(x) + 1.0 for x in vals if x in b]
                outs.append(np.array(ib).reshape(-1, 1))
            return tuple(outs)
        return out
    return f


for _k in ("intersect", "union", "setdiff", "setxor"):
    builtin(_k)(_set_op(_k))


@builtin("ismember")
def _ismember(interp, args, nargout):
    from .strings import _as_cell_if_strings
    args, flags = flag_present(args, "rows", "legacy")
    nargs(args, 2, 2, "ismember")
    a, b = _as_cell_if_strings(args[0]), _as_cell_if_strings(args[1])
    if "rows" in flags and not isinstance(a, Cell):
        av, bv = num(a), num(b)
        if av.shape[1] != bv.shape[1] and av.size and bv.size:
            raise MatlabError("ismember(A, B, 'rows') needs A and B with the same number of columns.")
        lookup: dict = {}
        for k in range(bv.shape[0]):
            lookup.setdefault(tuple(bv[k].tolist()), k + 1.0)
        locs = [lookup.get(tuple(av[k].tolist()), 0.0) for k in range(av.shape[0])]
        loc = np.array(locs, dtype=float).reshape(-1, 1)
        tf = loc != 0
        return (tf, loc) if nargout > 1 else tf
    if isinstance(a, MString) and isinstance(b, Cell):
        a = char(str(a))
    if is_text(a) and isinstance(b, Cell):
        strs = [to_str(x) for x in b.values_f()]
        s = to_str(a)
        found = s in strs
        return (found, float(strs.index(s) + 1) if found else 0.0) if nargout > 1 else found
    if isinstance(a, Cell):
        strs = [to_str(x) for x in b.values_f()] if isinstance(b, Cell) else [to_str(b)]
        res = np.array([to_str(x) in strs for x in a.data.flatten(order="F")]).reshape(a.shape, order="F")
        if nargout > 1:
            loc = np.array([float(strs.index(to_str(x)) + 1) if to_str(x) in strs else 0.0
                            for x in a.data.flatten(order="F")]).reshape(a.shape, order="F")
            return res, loc
        return res
    av, bv = num(a), num(b).flatten(order="F")
    res = np.isin(av, bv)
    if nargout > 1:
        lookup: dict = {}
        for k, x in enumerate(bv):
            lookup.setdefault(x, k + 1.0)
        loc = np.vectorize(lambda x: lookup.get(x, 0.0), otypes=[float])(av) if av.size else np.zeros(av.shape)
        return res, loc
    return res


# -- reshaping helpers ------------------------------------------------------------------

def _wrap_like(v, data):
    if isinstance(v, Cell):
        return Cell(data)
    return data


def _arr(v):
    return v.data if isinstance(v, Cell) else v


@builtin("fliplr")
def _fliplr(interp, args, nargout):
    return _wrap_like(args[0], np.fliplr(_arr(args[0])).copy())


@builtin("flipud")
def _flipud(interp, args, nargout):
    return _wrap_like(args[0], np.flipud(_arr(args[0])).copy())


@builtin("flip")
def _flip(interp, args, nargout):
    a = _arr(args[0])
    axis = get_dim(args, 1, a)
    return _wrap_like(args[0], np.flip(a, axis).copy())


@builtin("rot90")
def _rot90(interp, args, nargout):
    k = to_int(args[1], "k") if len(args) > 1 else 1
    return np.rot90(args[0], k).copy()


@builtin("circshift")
def _circshift(interp, args, nargout):
    a = _arr(args[0])
    s = num(args[1]).flatten().astype(int)
    if len(args) > 2:
        axis = to_int(args[2], "dim") - 1
        return _wrap_like(args[0], np.roll(a, int(s[0]), axis=axis))
    if s.size == 1:
        axis = default_dim(a)
        return _wrap_like(args[0], np.roll(a, int(s[0]), axis=axis))
    return _wrap_like(args[0], np.roll(a, tuple(int(x) for x in s[:2]), axis=(0, 1)))


@builtin("triu")
def _triu(interp, args, nargout):
    k = to_int(args[1], "k") if len(args) > 1 else 0
    return np.triu(args[0], k)


@builtin("tril")
def _tril(interp, args, nargout):
    k = to_int(args[1], "k") if len(args) > 1 else 0
    return np.tril(args[0], k)


def _nice_width(w: float) -> float:
    """Round a raw bin width to 1, 2, 2.5 or 5 times a power of ten."""
    if not np.isfinite(w) or w <= 0:
        return 1.0
    p = 10.0 ** np.floor(np.log10(w))
    m = w / p
    # the candidate closest to m on a logarithmic scale
    return p * min((1.0, 2.0, 2.5, 5.0, 10.0), key=lambda step: abs(np.log(m / step)))


def _auto_edges(x: np.ndarray, width: float | None = None, nbins: int | None = None,
                limits: tuple[float, float] | None = None) -> np.ndarray:
    """Bin edges chosen automatically: round limits and a round bin width."""
    finite = x[np.isfinite(x)]
    if limits is not None:
        lo, hi = limits
    elif finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = 0.0, 1.0
    if nbins is not None:
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return np.linspace(lo, hi, nbins + 1)
    if width is None:
        span = hi - lo
        integers = finite.size > 0 and bool(np.all(finite == np.round(finite)))
        raw = 3.5 * float(np.std(finite)) / finite.size ** (1 / 3) if finite.size > 1 else 0.0
        if limits is None and integers and span <= 50 and (raw <= 2.0 or span <= 10):
            # integer data over a small range: unit bins centred on the integers
            return np.arange(lo - 0.5, hi + 1.0, 1.0)
        if span == 0:
            return np.array([lo - 0.5, lo + 0.5])
        width = _nice_width(raw if raw > 0 else span)
        if integers:
            width = max(1.0, float(np.ceil(width)))
    if limits is not None:
        n = max(int(np.ceil((hi - lo) / width - 1e-9)), 1)
        return lo + width * np.arange(n + 1)
    left = np.floor(lo / width) * width
    n = max(int(np.ceil((hi - left) / width - 1e-9)), 1)
    edges = left + width * np.arange(n + 1)
    if edges[-1] < hi:
        edges = np.append(edges, edges[-1] + width)
    return edges


@builtin("histcounts")
def _histcounts(interp, args, nargout):
    nargs(args, 1, None, "histcounts")
    x = np.real(fnum(args[0])).flatten(order="F").astype(float)
    rest = list(args[1:])
    edges = None
    nbins = width = limits = None
    norm = "count"
    if rest and not is_text(rest[0]):
        b = fnum(rest.pop(0)).flatten()
        if b.size == 1:
            nbins = int(b[0])
        else:
            edges = b.astype(float)
    if len(rest) % 2:
        raise MatlabError("histcounts options must be name-value pairs.")
    for k in range(0, len(rest), 2):
        name = text(rest[k]).lower()
        val = rest[k + 1]
        if name == "binedges":
            edges = fnum(val).flatten().astype(float)
        elif name == "numbins":
            nbins = to_int(val, "NumBins")
        elif name == "binwidth":
            width = float(fnum(val).flat[0])
        elif name == "binlimits":
            lim = fnum(val).flatten()
            limits = (float(lim[0]), float(lim[1]))
        elif name == "normalization":
            norm = text(val).lower()
        elif name == "binmethod":
            meth = text(val).lower()
            if meth == "integers":
                f = x[np.isfinite(x)]
                lo, hi = (np.floor(f.min()), np.ceil(f.max())) if f.size else (0.0, 0.0)
                edges = np.arange(lo - 0.5, hi + 1.0, 1.0)
        else:
            raise MatlabError(f"Unknown histcounts option '{text(rest[k])}'.")
    if edges is None:
        if nbins is not None and nbins < 1:
            raise MatlabError("The number of bins must be a positive integer.")
        edges = _auto_edges(x, width, nbins, limits)
    if edges.size < 2:
        raise MatlabError("Bin edges must have at least two elements.")
    counts, _ = np.histogram(x[np.isfinite(x) | np.isinf(x)], bins=edges)
    counts = counts.astype(float)
    n_total = float(np.count_nonzero(~np.isnan(x)))
    widths = np.diff(edges)
    if norm == "count":
        out = counts
    elif norm == "probability":
        out = counts / n_total if n_total else counts * np.nan
    elif norm == "pdf":
        out = counts / (n_total * widths) if n_total else counts * np.nan
    elif norm == "cdf":
        out = np.cumsum(counts) / n_total if n_total else counts * np.nan
    elif norm == "countdensity":
        out = counts / widths
    elif norm == "cumcount":
        out = np.cumsum(counts)
    else:
        raise MatlabError(f"Unknown normalization '{norm}'.")
    out = out.reshape(1, -1)
    if nargout <= 1:
        return out
    outs = [out, edges.reshape(1, -1)]
    if nargout > 2:
        idx = np.searchsorted(edges, x, side="right").astype(float)
        idx[x == edges[-1]] = edges.size - 1
        idx[(x < edges[0]) | (x > edges[-1]) | np.isnan(x)] = 0
        outs.append(idx.reshape(args[0].shape if isinstance(args[0], np.ndarray) else (1, -1), order="F"))
    return tuple(outs)


def _accum_out_shape(subs: np.ndarray, sz) -> tuple[int, ...]:
    need = tuple(int(m) for m in subs.max(axis=0)) if subs.size else (0,) * subs.shape[1]
    if sz is not None and np.asarray(sz).size:
        shape = tuple(int(v) for v in np.asarray(sz).flatten())
        if subs.shape[1] == 1 and len(shape) == 2 and 1 in shape:
            if max(shape) < need[0]:
                raise MatlabError("accumarray: subscripts exceed the requested size.")
            return shape
        if len(shape) != subs.shape[1] or any(s < n for s, n in zip(shape, need)):
            raise MatlabError("accumarray: the size must have one element per subscript column "
                              "and be at least as large as the subscripts.")
        return shape
    if subs.shape[1] == 1:
        return (need[0], 1)
    return need


@builtin("accumarray")
def _accumarray(interp, args, nargout):
    nargs(args, 2, 6, "accumarray")
    subs_v = args[0]
    if isinstance(subs_v, Cell):
        subs = np.column_stack([num(c).flatten(order="F") for c in subs_v.values_f()])
    else:
        subs = num(subs_v)
        if subs.ndim == 2 and subs.shape[0] == 1 and subs.shape[1] != 2:
            subs = subs.reshape(-1, 1)
    subs = np.real(subs).astype(float)
    if subs.size and (np.any(subs < 1) or np.any(subs != np.round(subs))):
        raise MatlabError("accumarray: subscripts must be positive integers.")
    subs = subs.astype(int)
    vals_v = args[1]
    vals = fnum(vals_v).flatten(order="F")
    m = subs.shape[0]
    if vals.size == 1:
        vals = np.repeat(vals, m)
    elif vals.size != m:
        raise MatlabError("accumarray: the number of values must match the number of subscript rows.")
    sz = num(args[2]) if len(args) > 2 and not (isinstance(args[2], np.ndarray) and args[2].size == 0) \
        else None
    fun = args[3] if len(args) > 3 and not (isinstance(args[3], np.ndarray) and args[3].size == 0) else None
    fill = args[4] if len(args) > 4 and not (isinstance(args[4], np.ndarray) and args[4].size == 0) else None
    shape = _accum_out_shape(subs, sz)
    lin = np.ravel_multi_index(tuple(subs[:, k] - 1 for k in range(subs.shape[1])),
                               shape if subs.shape[1] > 1 else (int(np.prod(shape)),), order="F") \
        if m else np.zeros(0, dtype=int)
    total = int(np.prod(shape))
    is_logical = isinstance(vals_v, np.ndarray) and vals_v.dtype == bool
    if fun is None:
        out = np.zeros(total, dtype=complex if np.iscomplexobj(vals) else float)
        np.add.at(out, lin, vals)
        filled = np.zeros(total, dtype=bool)
        filled[lin] = True
        if fill is not None:
            out[~filled] = float(fnum(fill).flat[0])
        return simplify(out.reshape(shape, order="F"))
    groups: dict[int, list[int]] = {}
    for k, p in enumerate(lin):
        groups.setdefault(int(p), []).append(k)
    results: dict[int, Any] = {}
    for p, members in groups.items():
        piece = vals[members].reshape(-1, 1)
        if is_logical:
            piece = piece.astype(bool)
        res = interp.call_function_value(fun, [piece], 1)
        if not res:
            raise MatlabError("accumarray: the function must return a value.")
        results[p] = res[0]
    if any(isinstance(r, Cell) for r in results.values()):
        data = np.empty(total, dtype=object)
        empty_fill = fill if fill is not None else np.zeros((0, 0))
        for p in range(total):
            r = results.get(p)
            if r is None:
                data[p] = empty_fill
            elif isinstance(r, Cell):
                if r.data.size != 1:
                    raise MatlabError("accumarray: the function must return a scalar cell.")
                data[p] = r.data.flat[0]
            else:
                data[p] = r
        return Cell(data.reshape(shape, order="F"))
    fill_val = float(np.real(fnum(fill).flat[0])) if fill is not None else 0.0
    cplx = any(np.iscomplexobj(fnum(r)) for r in results.values())
    out = np.full(total, fill_val, dtype=complex if cplx else float)
    all_bool = bool(results) and all(isinstance(r, np.ndarray) and r.dtype == bool for r in results.values())
    for p, r in results.items():
        a = fnum(r)
        if a.size != 1:
            raise MatlabError("accumarray: the function must return a scalar for each group; "
                              "wrap non-scalar results in a cell, e.g. @(x) {x}.")
        out[p] = a.flat[0]
    out = out.reshape(shape, order="F")
    if all_bool and fill is None:
        return out.astype(bool)
    return simplify(out)


@builtin("repelem")
def _repelem(interp, args, nargout):
    nargs(args, 2, None, "repelem")
    v = args[0]
    data = v.data if isinstance(v, (Cell, StringArray)) else v
    if not isinstance(data, np.ndarray):
        data = num(v)

    def counts(arg, n):
        c = num(arg).flatten().astype(int)
        if c.size == 1:
            return np.repeat(c, n)
        if c.size != n:
            raise MatlabError("repelem: the repetition vector must match the size of the input.")
        if np.any(c < 0):
            raise MatlabError("repelem: repetition counts must be nonnegative.")
        return c

    if len(args) == 2:
        if not (data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1)):
            raise MatlabError("repelem(v, n) needs a vector v; use repelem(A, r, c) for matrices.")
        res = np.repeat(data.flatten(order="F"), counts(args[1], data.size))
        res = res.reshape(1, -1) if data.shape[0] == 1 else res.reshape(-1, 1)
    else:
        res = data
        for axis, arg in enumerate(args[1:]):
            while res.ndim <= axis:
                res = res[..., np.newaxis]
            res = np.repeat(res, counts(arg, res.shape[axis]), axis=axis)
    if isinstance(v, Cell):
        return Cell(res)
    if isinstance(v, StringArray):
        return string_result(res)
    return res


@builtin("issorted")
def _issorted(interp, args, nargout):
    nargs(args, 1, 3, "issorted")
    rest, flags = flag_present(args, "ascend", "descend", "monotonic", "strictascend",
                               "strictdescend", "strictmonotonic", "rows")
    v = rest[0]
    if isinstance(v, (Cell, StringArray)):
        from .strings import _as_cell_if_strings
        keys = [to_str(x) for x in _as_cell_if_strings(v).values_f()]
    elif "rows" in flags:
        a = num(v)
        keys = [tuple(r) for r in a]
    else:
        a = num(v)
        keys = list(a.flatten(order="F"))
    pairs = list(zip(keys, keys[1:]))
    asc = all(not (b < a) for a, b in pairs)
    desc = all(not (b > a) for a, b in pairs)
    if "descend" in flags:
        ok = desc
    elif "monotonic" in flags:
        ok = asc or desc
    elif "strictascend" in flags:
        ok = all(a < b for a, b in pairs)
    elif "strictdescend" in flags:
        ok = all(a > b for a, b in pairs)
    elif "strictmonotonic" in flags:
        ok = all(a < b for a, b in pairs) or all(a > b for a, b in pairs)
    else:
        ok = asc
    return np.array([[bool(ok)]])


def _topk(largest: bool):
    def f(interp, args, nargout):
        nargs(args, 2, 4, "maxk" if largest else "mink")
        a = num(args[0])
        k = to_int(args[1], "k")
        if k < 0:
            raise MatlabError("k must be a nonnegative integer.")
        axis = get_dim(args, 2, a)
        work = a.astype(float) if a.dtype.kind in "biu" else a
        key = np.abs(work) if work.dtype.kind == "c" else work
        if largest:
            key = np.where(np.isnan(key), -np.inf, -key) if key.dtype.kind == "f" else -key
        else:
            key = np.where(np.isnan(key), np.inf, key) if key.dtype.kind == "f" else key
        idx = np.argsort(key, axis=axis, kind="stable")
        idx = np.take(idx, np.arange(min(k, a.shape[axis] if a.ndim > axis else 1)), axis=axis)
        out = np.take_along_axis(a, idx, axis)
        return (out, idx.astype(float) + 1) if nargout > 1 else out
    return f


builtin("maxk")(_topk(True))
builtin("mink")(_topk(False))


@builtin("dec2bin")
def _dec2bin(interp, args, nargout):
    n = int(fnum(args[0]).flat[0])
    width = to_int(args[1], "digits") if len(args) > 1 else 0
    return char(format(n, "b").zfill(width))


@builtin("bin2dec")
def _bin2dec(interp, args, nargout):
    return float(int(text(args[0]).replace(" ", ""), 2))


@builtin("dec2hex")
def _dec2hex(interp, args, nargout):
    n = int(fnum(args[0]).flat[0])
    width = to_int(args[1], "digits") if len(args) > 1 else 0
    return char(format(n, "X").zfill(width))


@builtin("hex2dec")
def _hex2dec(interp, args, nargout):
    return float(int(text(args[0]), 16))


def _bit(fn):
    def f(interp, args, nargout):
        a, b = np.broadcast_arrays(fnum(args[0]).astype(np.uint64), fnum(args[1]).astype(np.uint64))
        return fn(a, b).astype(float)
    return f


builtin("bitand")(_bit(np.bitwise_and))
builtin("bitor")(_bit(np.bitwise_or))
builtin("bitxor")(_bit(np.bitwise_xor))


@builtin("bitshift")
def _bitshift(interp, args, nargout):
    a = fnum(args[0]).astype(np.int64)
    n = int(fnum(args[1]).flat[0])
    return (a << n if n >= 0 else a >> -n).astype(float)




@builtin("bitget")
def _bitget(interp, args, nargout):
    a, b = np.broadcast_arrays(fnum(args[0]).astype(np.uint64), fnum(args[1]).astype(np.int64))
    if np.any(b < 1) or np.any(b > 64):
        raise MatlabError("bitget: bit positions must be between 1 and 64.")
    return ((a >> (b - 1).astype(np.uint64)) & np.uint64(1)).astype(float)


@builtin("bitset")
def _bitset(interp, args, nargout):
    a, b = np.broadcast_arrays(fnum(args[0]).astype(np.uint64), fnum(args[1]).astype(np.int64))
    v = fnum(args[2]) if len(args) > 2 else np.ones((1, 1))
    v = np.broadcast_to(v, a.shape) != 0
    mask = np.left_shift(np.uint64(1), (b - 1).astype(np.uint64))
    return np.where(v, a | mask, a & ~mask).astype(float)


@builtin("bitcmp")
def _bitcmp(interp, args, nargout):
    a = num(args[0])
    bits = {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16, np.dtype(np.uint32): 32}.get(a.dtype, 53)
    if len(args) > 1:
        from ..values import to_str
        bits = {"uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64}.get(to_str(args[1]), bits)
    full = (1 << bits) - 1
    return np.vectorize(lambda x: float(full ^ int(x)), otypes=[float])(a)
