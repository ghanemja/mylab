"""Elementwise math, rounding, reductions, sorting and set operations."""

from __future__ import annotations


import math
from typing import Any

import numpy as np
import scipy.special as sp

from ..errors import MatlabError
from ..ops import binop, unary
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
    mag = np.abs(a)
    with np.errstate(invalid="ignore"):
        low = np.floor(mag)
        r = np.where(mag - low >= 0.5, low + 1, low)  # exact, unlike floor(|a| + 0.5)
    return np.copysign(r, a).astype(a.dtype, copy=False)


def _per_part(fn):
    """Apply a real rounding function to the real and imaginary parts of complex input."""
    def f(a):
        if a.dtype.kind == "c":
            return fn(a.real) + 1j * fn(a.imag)
        return fn(a)
    return f


def _asin(a):
    if a.dtype.kind == "c":
        return np.arcsin(a)
    out = np.abs(a) > 1
    if not np.any(out):
        return np.arcsin(a)
    # outside [-1, 1]: asin(x) = sign(x) * (pi/2 - i*acosh(|x|))
    s = np.sign(a)
    big = s * (np.pi / 2) - 1j * s * np.arccosh(np.where(out, np.abs(a), 1.0))
    return np.where(out, big, np.arcsin(np.where(out, 0.0, a)))


def _acos(a):
    if a.dtype.kind == "c":
        return np.arccos(a)
    if not np.any(np.abs(a) > 1):
        return np.arccos(a)
    return np.pi / 2 - _asin(a)


def _acosh(a):
    if a.dtype.kind != "c" and np.any(a < 1):
        a = a.astype(np.complex64 if a.dtype == np.float32 else complex)
    return np.arccosh(a)


def _gamma(a):
    r = sp.gamma(a)
    if a.dtype.kind == "f":
        r = np.where((a <= 0) & (a == np.floor(a)), np.inf, r)  # poles: 0 and the negative integers
    return r


_ELEMENTWISE = {
    "sqrt": np.emath.sqrt, "exp": np.exp, "log": np.emath.log, "log2": np.emath.log2,
    "log10": np.emath.log10, "log1p": np.log1p, "expm1": np.expm1,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": _asin, "acos": _acos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": _acosh,
    "atanh": lambda a: 0.5 * np.emath.log((1 + a) / (1 - a)),
    "sec": lambda a: 1 / np.cos(a), "csc": lambda a: 1 / np.sin(a), "cot": lambda a: 1 / np.tan(a),
    "sind": lambda a: _sind(a), "cosd": lambda a: _sind(np.asarray(a) + 90.0),
    "tand": lambda a: _sind(a) / _sind(np.asarray(a) + 90.0),
    "asind": lambda a: np.degrees(_asin(a)), "acosd": lambda a: np.degrees(_acos(a)),
    "atand": lambda a: np.degrees(np.arctan(a)),
    "deg2rad": np.radians, "rad2deg": np.degrees,
    "floor": _per_part(np.floor), "ceil": _per_part(np.ceil), "fix": _per_part(np.trunc),
    "real": np.real, "imag": np.imag, "conj": np.conj, "angle": np.angle,
    "gamma": _gamma, "gammaln": sp.gammaln, "erf": sp.erf, "erfc": sp.erfc,
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
    "floor": lambda x: math.copysign(math.floor(x), x) if math.isfinite(x) else None,
    "ceil": lambda x: math.copysign(math.ceil(x), x) if math.isfinite(x) else None,
    "fix": lambda x: math.copysign(math.trunc(x), x) if math.isfinite(x) else None,
}

for _name, _fn in _ELEMENTWISE.items():
    builtin(_name)(_unary_math(_fn, scalar=_SCALAR.get(_name)))

_log2_one = _unary_math(np.emath.log2, scalar=_SCALAR["log2"])


@builtin("log2")
def _log2(interp, args, nargout):
    if nargout < 2:
        return _log2_one(interp, args, nargout)
    nargs(args, 1, 1, "log2")
    a = fnum(args[0])
    if a.dtype.kind == "c":
        a = a.real
    f, e = np.frexp(a)
    return f, e.astype(a.dtype)


@builtin("pow2")
def _pow2(interp, args, nargout):
    nargs(args, 1, 2, "pow2")
    two = np.array([[2.0]])
    if len(args) == 1:
        return binop(".^", two, args[0])
    return binop(".*", args[0], binop(".^", two, args[1]))


@builtin("realsqrt")
def _realsqrt(interp, args, nargout):
    nargs(args, 1, 1, "realsqrt")
    a = fnum(args[0])
    if a.dtype.kind == "c" or (a.size and np.any(a < 0)):
        raise MatlabError("realsqrt produced complex result.")
    return np.sqrt(a)


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
    return np.sign(a).astype(a.dtype if a.dtype.kind in "iuf" else float)


def _round_scaled(a, n):
    """Round to n decimal places; n is an integer array that broadcasts with a."""
    n = np.asarray(n, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        up = _round_half_away(a * 10.0 ** np.maximum(n, 0)) / 10.0 ** np.maximum(n, 0)
        down = _round_half_away(a / 10.0 ** np.maximum(-n, 0)) * 10.0 ** np.maximum(-n, 0)
    r = np.where(n >= 0, up, down)
    return np.where(np.isfinite(a), r, a).astype(a.dtype, copy=False)


@builtin("round")
def _round(interp, args, nargout):
    nargs(args, 1, 3, "round")
    a = num(args[0])
    if len(args) > 1:
        n = to_int(args[1], "digits")
        mode = text(args[2]).lower() if len(args) > 2 else "decimals"
        if mode not in ("significant", "decimals"):
            raise MatlabError("round: the rounding type must be 'significant' or 'decimals'.")
        int_class = a.dtype if a.dtype.kind in "iu" else None
        if int_class is not None:
            if n >= 0 and mode == "decimals":
                return a
            a = a.astype(float)
        if mode == "significant":
            if n < 1:
                raise MatlabError("round: the number of significant digits must be a positive integer.")
            if a.dtype.kind == "c":
                raise MatlabError("round: 'significant' is not supported for complex values.")
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.floor(np.log10(np.abs(a)))
            e = np.where(np.isfinite(e), e, 0.0)
            r = _round_scaled(a, n - 1 - e)
        elif a.dtype.kind == "c":
            r = _round_scaled(a.real, n) + 1j * _round_scaled(a.imag, n)
        else:
            r = _round_scaled(a, n)
        if int_class is not None:
            from ..values import saturate_int
            return saturate_int(r, int_class)
        return r
    if a.dtype.kind in "iu":
        return a
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


def _pair_class(a, b, name):
    """Result dtype of a two-operand numeric function: integer class, single, or None for double."""
    ia, ib = a.dtype.kind in "iu", b.dtype.kind in "iu"
    if ia and ib and a.dtype != b.dtype:
        raise MatlabError(f"{name}: integers can only be combined with integers of the same class "
                          "or with doubles.")
    if ia or ib:
        return a.dtype if ia else b.dtype
    if a.dtype in (np.float32, np.complex64) or b.dtype in (np.float32, np.complex64):
        return np.dtype(np.float32)
    return None


def _int_exact(fn):
    """Elementwise Python-integer function for 64-bit classes (doubles would lose precision)."""
    def g(a, b):
        a, b = np.broadcast_arrays(a.astype(object), b.astype(object))
        out = [fn(int(x), int(y)) for x, y in zip(a.reshape(-1), b.reshape(-1))]
        return np.array(out, dtype=object).reshape(a.shape)
    return g


def _py_rem(x, y):
    if y == 0:
        return 0
    return (abs(x) % abs(y)) * (1 if x >= 0 else -1)


def _mod_rem(float_fn, scalar_fn, exact_fn, name):
    plain = _binary_math(float_fn, scalar_fn)

    def f(interp, args, nargout):
        if len(args) == 2 and not (type(args[0]) is np.ndarray and args[0].dtype is _F64
                                   and type(args[1]) is np.ndarray and args[1].dtype is _F64):
            a, b = num(args[0]), num(args[1])
            cls = _pair_class(a, b, name)
            if cls is not None:
                if cls.kind in "iu":
                    from ..values import saturate_int
                    # a double operand is converted to the integer class first
                    a = a if a.dtype.kind in "iu" else saturate_int(a, cls)
                    b = b if b.dtype.kind in "iu" else saturate_int(b, cls)
                    if cls.itemsize == 8:
                        return exact_fn(a, b).astype(cls)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return saturate_int(float_fn(a.astype(float), b.astype(float)), cls)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.asarray(float_fn(a.astype(float), b.astype(float))).astype(np.float32)
        return plain(interp, args, nargout)
    return f


builtin("mod")(_mod_rem(_mod, _mod_scalar, _int_exact(lambda x, y: x if y == 0 else x % y), "mod"))
builtin("rem")(_mod_rem(_rem, _rem_scalar, _int_exact(_py_rem), "rem"))


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
    nargs(args, 2, 2, "nchoosek")
    n, k = args
    kk = to_int(k, "k")
    if kk < 0:
        raise MatlabError("nchoosek: k must be a nonnegative integer.")
    src = n if isinstance(n, np.ndarray) else num(n)
    if src.size == 1 and not is_char(src):
        nn = to_int(n, "n")
        if nn < 0:
            raise MatlabError("nchoosek: n must be a nonnegative integer.")
        r = float(sp.comb(nn, kk, exact=True))
        if src.dtype.kind in "iu" or src.dtype == np.float32:
            from ..values import class_name
            from ._util import cast_class
            return cast_class(np.array([[r]]), class_name(src))
        return r
    from itertools import combinations
    items = src.flatten(order="F")
    combos = list(combinations(range(items.size), kk))
    idx = np.array(combos, dtype=int).reshape(len(combos), kk)
    return items[idx]


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


def _gcd_lcm(fn, name):
    def f(interp, args, nargout):
        nargs(args, 2, 2, name)
        a, b = num(args[0]), num(args[1])
        if a.dtype.kind == "c" or b.dtype.kind == "c":
            raise MatlabError(f"{name}: inputs must be real integers.")
        cls = _pair_class(a, b, name)
        for x in (a, b):
            if x.dtype.kind == "f" and x.size and not np.all(np.isfinite(x) & (x == np.round(x))):
                raise MatlabError(f"{name}: inputs must be integers.")
        a, b = np.broadcast_arrays(a, b)
        r = [fn(abs(int(x)), abs(int(y))) for x, y in zip(a.reshape(-1), b.reshape(-1))]
        if cls is not None and cls.kind in "iu":
            info = np.iinfo(cls)
            return np.array([min(x, info.max) for x in r], dtype=cls).reshape(a.shape)
        return np.array(r, dtype=float).reshape(a.shape).astype(np.float32 if cls is not None else float)
    return f


builtin("gcd")(_gcd_lcm(math.gcd, "gcd"))
builtin("lcm")(_gcd_lcm(lambda x, y: 0 if x == 0 or y == 0 else x * y // math.gcd(x, y), "lcm"))


# -- reductions ---------------------------------------------------------------------------

def _no_arg(v) -> bool:
    """True for the [] placeholder argument."""
    return isinstance(v, np.ndarray) and v.size == 0


def _dims_arg(v):
    """A dim or vecdim argument as a list of 0-based axes."""
    d = np.real(num(v)).flatten()
    if d.size == 0 or np.any(d < 1) or np.any(d != np.round(d)) or len(set(d.tolist())) != d.size:
        raise MatlabError("Dimension argument must be a positive integer scalar or a vector of "
                          "unique positive integers.")
    return [int(x) - 1 for x in d]


def _red_dims(rest, pos, a, flags=()):
    """Axes to reduce: 'all', an explicit dim/vecdim at rest[pos], or the first non-singleton one."""
    if "all" in flags:
        return list(range(a.ndim))
    if len(rest) > pos and not _no_arg(rest[pos]):
        return _dims_arg(rest[pos])
    if a.shape == (0, 0):
        return [0, 1]  # sum([]) is a scalar
    return [default_dim(a)]


def _merge(a, dims):
    """Move the reduced axes to one trailing axis (column-major order inside it).

    Returns the rearranged array and the MATLAB shape of the reduced result. Axes beyond
    ndims are singleton, so reducing only along them reduces slices of one element.
    """
    if len(dims) == 1 and a.ndim == 2:  # the common cases need no rearranging
        if dims[0] == 1:
            return a, [a.shape[0], 1]
        if dims[0] == 0:
            return a.T, [1, a.shape[1]]
    dims = sorted({d for d in dims if d < a.ndim})
    other = [d for d in range(a.ndim) if d not in dims]
    count = 1
    for d in dims:
        count *= a.shape[d]
    b = np.transpose(a, other + dims[::-1]).reshape([a.shape[d] for d in other] + [count])
    return b, [1 if d in dims else a.shape[d] for d in range(a.ndim)]


def _unmerge(r, shape):
    from ..values import as_2d
    return as_2d(np.asarray(r).reshape(shape))


def _saturate(r, cls):
    """Round and clamp a float or Python-int (object) array to an integer class."""
    from ..values import saturate_int
    r = np.asarray(r)
    if r.dtype == object:
        info = np.iinfo(cls)
        flat = [min(max(int(x), info.min), info.max) for x in r.reshape(-1)]
        return np.array(flat, dtype=cls).reshape(r.shape)
    return saturate_int(r, cls)


def _wide(a):
    """Integers as a type that cannot overflow: float64, or Python ints for 64-bit classes."""
    if a.dtype.kind in "iu":
        return a.astype(object) if a.dtype.itemsize == 8 else a.astype(float)
    return a


def _sum_like(name, np_fn, identity):
    def f(interp, args, nargout):
        nargs(args, 1, 5, name)
        rest, flags = flag_present(args, "omitnan", "includenan", "all", "double", "native", "default")
        if len(rest) > 2:
            raise MatlabError(f"Too many input arguments for {name}.")
        src = rest[0]
        a = num(src)
        if "native" in flags:
            out = bool if (isinstance(src, np.ndarray) and src.dtype == bool) else a.dtype
        elif "double" in flags:
            out = np.dtype(complex if a.dtype.kind == "c" else float)
        else:
            out = a.dtype
        out = np.dtype(out)
        dims = _red_dims(rest, 1, a, flags)
        work = _wide(a)
        if out.kind in "fc" and work.dtype != object and work.dtype.itemsize < out.itemsize:
            work = work.astype(out)
        if "omitnan" in flags and work.dtype.kind in "fc":
            work = np.where(np.isnan(work), identity, work)
        b, shape = _merge(work, dims)
        with np.errstate(over="ignore", invalid="ignore"):
            r = np_fn(b, axis=-1)
        if out.kind == "b":
            r = np.asarray(r) != 0
        elif out.kind in "iu":
            r = _saturate(r, out)
        else:
            r = np.asarray(r).astype(out, copy=False) if np.asarray(r).dtype == object else np.asarray(r)
        return _unmerge(r, shape)
    return f


builtin("sum")(_sum_like("sum", np.sum, 0.0))
builtin("prod")(_sum_like("prod", np.prod, 1.0))


def _float_out(a, flags):
    """Result dtype of mean-like functions: single stays single, everything else is double."""
    if "native" in flags:
        return a.dtype
    if "double" in flags:
        return np.dtype(complex if a.dtype.kind == "c" else float)
    if a.dtype in (np.float32, np.complex64):
        return a.dtype
    return np.dtype(complex if a.dtype.kind == "c" else float)


@builtin("mean")
def _mean(interp, args, nargout):
    nargs(args, 1, 5, "mean")
    rest, flags = flag_present(args, "omitnan", "includenan", "all", "double", "native", "default")
    if len(rest) > 2:
        raise MatlabError("Too many input arguments for mean.")
    src = rest[0]
    a = num(src)
    if "native" in flags and isinstance(src, np.ndarray) and src.dtype == bool:
        raise MatlabError("mean: the 'native' option is not supported for logical input.")
    out = _float_out(a, flags)
    dims = _red_dims(rest, 1, a, flags)
    work = a.astype(float) if a.dtype.kind in "iu" else a
    if work.dtype == np.float32 or work.dtype == np.complex64:
        work = work.astype(complex if work.dtype.kind == "c" else float)
    b, shape = _merge(work, dims)
    if "omitnan" in flags and b.dtype.kind in "fc":
        keep = ~np.isnan(b)
        total = np.where(keep, b, 0.0).sum(axis=-1)
        count = keep.sum(axis=-1)
    else:
        total = b.sum(axis=-1)
        count = b.shape[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(total / count)
    if out.kind in "iu":
        r = _saturate(r, out)
    else:
        r = r.astype(out, copy=False)
    return _unmerge(r, shape)


@builtin("median")
def _median(interp, args, nargout):
    nargs(args, 1, 4, "median")
    rest, flags = flag_present(args, "omitnan", "includenan", "all")
    if len(rest) > 2:
        raise MatlabError("Too many input arguments for median.")
    a = num(rest[0])
    if a.dtype.kind == "c":
        raise MatlabError("median: complex input is not supported.")
    out = a.dtype
    dims = _red_dims(rest, 1, a, flags)
    b, shape = _merge(a.astype(float) if a.dtype.kind in "iu" and a.dtype.itemsize < 8 else a, dims)
    b = np.sort(b, axis=-1)  # NaNs go last
    n = b.shape[-1]
    lead = b.shape[:-1]
    if b.dtype.kind == "f":
        nans = np.isnan(b).sum(axis=-1)
        count = n - nans if "omitnan" in flags else np.where(nans > 0, 0, n)
    else:
        count = np.full(lead, n)
    count = np.asarray(count)
    safe = np.maximum(count, 1)
    if n == 0:
        r = np.full(lead, np.nan)
    else:
        lo = np.take_along_axis(b, np.expand_dims((safe - 1) // 2, -1), -1)[..., 0]
        hi = np.take_along_axis(b, np.expand_dims(safe // 2, -1), -1)[..., 0]
        if b.dtype.kind in "iu":  # 64-bit integers: exact midpoint, halves away from zero
            mid = [(int(x) + int(y)) for x, y in zip(lo.reshape(-1), hi.reshape(-1))]
            r = np.array([int(math.copysign((abs(m) + 1) // 2, m)) for m in mid], dtype=object).reshape(lo.shape)
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                r = np.where(lo == hi, lo, lo + (hi - lo) / 2)
            r = np.where(count == 0, np.nan, r)
    if out.kind in "iu":
        r = _saturate(r, out)
    else:
        r = np.asarray(r).astype(np.float32 if out == np.float32 else float)
    return _unmerge(r, shape)


def _anyall(np_fn, name):
    def f(interp, args, nargout):
        nargs(args, 1, 2, name)
        rest, flags = flag_present(args, "all")
        if len(rest) > 2:
            raise MatlabError(f"Too many input arguments for {name}.")
        a = num(rest[0]) != 0
        b, shape = _merge(a, _red_dims(rest, 1, a, flags))
        return _unmerge(np_fn(b, axis=-1), shape)
    return f


builtin("any")(_anyall(np.any, "any"))
builtin("all")(_anyall(np.all, "all"))


def _std_var(power, name):
    def f(interp, args, nargout):
        nargs(args, 1, 4, name)
        rest, flags = flag_present(args, "omitnan", "includenan", "all")
        if len(rest) > 3:
            raise MatlabError(f"Too many input arguments for {name}.")
        a = fnum(rest[0])
        single = a.dtype in (np.float32, np.complex64)
        a = a.astype(complex if a.dtype.kind == "c" else float)
        w = 0
        weights = None
        if len(rest) > 1 and not _no_arg(rest[1]):
            wv = np.real(fnum(rest[1])).astype(float)
            if wv.size == 1:
                w = wv.flat[0]
                if w not in (0, 1):
                    raise MatlabError(f"{name}: the weight must be 0, 1 or a vector of nonnegative weights.")
            else:
                weights = wv.flatten()
                if np.any(weights < 0) or not np.all(np.isfinite(weights)):
                    raise MatlabError(f"{name}: weights must be nonnegative and finite.")
        dims = _red_dims(rest, 2, a, flags)
        b, shape = _merge(a, dims)
        n = b.shape[-1]
        if weights is not None:
            if len(dims) != 1 or "all" in flags:
                raise MatlabError(f"{name}: a weight vector needs a single operating dimension.")
            if weights.size != n:
                raise MatlabError(f"{name}: the weight vector must have one element per element "
                                  "along the operating dimension.")
        keep = ~np.isnan(b) if "omitnan" in flags else np.ones(b.shape, dtype=bool)
        x = np.where(keep, b, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            if weights is not None:
                wk = np.where(keep, weights, 0.0)
                tot = wk.sum(axis=-1, keepdims=True)
                mu = (wk * x).sum(axis=-1, keepdims=True) / tot
                v = (wk * np.abs(np.where(keep, b - mu, 0.0)) ** 2).sum(axis=-1) / tot[..., 0]
            else:
                cnt = keep.sum(axis=-1, keepdims=True)
                mu = x.sum(axis=-1, keepdims=True) / cnt
                ss = (np.abs(np.where(keep, b - mu, 0.0)) ** 2).sum(axis=-1)
                c = cnt[..., 0].astype(float)
                den = np.where((c > 1) & (w == 0), c - 1, c)
                v = np.where(c == 0, np.nan, ss / np.where(den == 0, 1, den))
            r = np.sqrt(v) if power == 0.5 else v
        r = np.asarray(r, dtype=float)
        return _unmerge(r.astype(np.float32) if single else r, shape)
    return f


builtin("std")(_std_var(0.5, "std"))
builtin("var")(_std_var(1, "var"))


@builtin("mode")
def _mode(interp, args, nargout):
    nargs(args, 1, 2, "mode")
    rest, flags = flag_present(args, "all")
    src = rest[0]
    a = src if isinstance(src, np.ndarray) and src.dtype.kind in "bU" else num(src)
    if a.dtype.kind == "c":
        raise MatlabError("mode: complex input is not supported.")
    dims = _red_dims(rest, 1, a, flags)
    b, shape = _merge(a, dims)
    rows = b.reshape(int(np.prod(b.shape[:-1])), b.shape[-1])
    is_float = a.dtype.kind == "f"
    vals = np.full(rows.shape[0], np.nan) if is_float or rows.shape[1] == 0 else np.zeros(rows.shape[0], a.dtype)
    freq = np.zeros(rows.shape[0])
    ties = np.empty(rows.shape[0], dtype=object)
    for k, row in enumerate(rows):
        row = row[~np.isnan(row)] if is_float else row
        if row.size == 0:
            ties[k] = (np.full((1, 1), np.nan) if rows.shape[1] else np.zeros((0, 1))).astype(vals.dtype)
            continue
        u, c = np.unique(row, return_counts=True)
        top = c.max()
        vals[k] = u[np.argmax(c)]
        freq[k] = top
        ties[k] = u[c == top].reshape(-1, 1)
    out = _unmerge(vals, shape)
    if nargout <= 1:
        return out
    outs = [out, _unmerge(freq, shape)]
    if nargout > 2:
        outs.append(Cell(_unmerge(ties, shape)))
    return tuple(outs)


def _running_int(a, axis, op, cls):
    """Cumulative sum/product of an integer class, saturating at every step."""
    info = np.iinfo(cls)
    m = np.moveaxis(a.astype(object), axis, 0)
    out = np.empty(m.shape, dtype=object)
    acc = None
    for k in range(m.shape[0]):
        acc = m[k] if acc is None else np.clip(op(acc, m[k]), info.min, info.max)
        out[k] = acc
    return np.moveaxis(out, 0, axis).astype(cls)


def _cumulative(name, np_fn, identity=None, nan_fn=None, op=None):
    def f(interp, args, nargout):
        nargs(args, 1, 4, name)
        rest, flags = flag_present(args, "reverse", "forward", "omitnan", "includenan")
        if len(rest) > 2:
            raise MatlabError(f"Too many input arguments for {name}.")
        a = num(rest[0])
        if len(rest) > 1 and not _no_arg(rest[1]):
            axis = to_int(rest[1], "dimension") - 1
            if axis < 0:
                raise MatlabError("Dimension argument must be a positive integer.")
        else:
            axis = default_dim(a)
        if a.size == 0:
            return a
        fn = np_fn
        if a.dtype.kind in "fc":
            if identity is not None and "omitnan" in flags:
                a = np.where(np.isnan(a), np.asarray(identity, dtype=a.dtype), a)
            elif nan_fn is not None and "includenan" in flags:
                fn = nan_fn
        if axis >= a.ndim:
            return a
        if "reverse" in flags:
            a = np.flip(a, axis)
        if a.dtype.kind in "iu" and op is not None:
            r = _running_int(a, axis, op, a.dtype)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                r = fn(a, axis=axis)
        return np.flip(r, axis) if "reverse" in flags else r
    return f


builtin("cumsum")(_cumulative("cumsum", np.cumsum, identity=0, op=lambda x, y: x + y))
builtin("cumprod")(_cumulative("cumprod", np.cumprod, identity=1, op=lambda x, y: x * y))
builtin("cummax")(_cumulative("cummax", np.fmax.accumulate, nan_fn=np.maximum.accumulate))
builtin("cummin")(_cumulative("cummin", np.fmin.accumulate, nan_fn=np.minimum.accumulate))


def _minmax(is_max):
    name = "max" if is_max else "min"

    def pair(src_a, src_b, flags):
        a, b = num(src_a), num(src_b)
        both_logical = all(isinstance(s, np.ndarray) and s.dtype == bool for s in (src_a, src_b))
        cls = _pair_class(a, b, name)
        if cls is not None and cls.kind in "iu":
            a = a if a.dtype.kind in "iu" else _saturate(np.real(a), cls)
            b = b if b.dtype.kind in "iu" else _saturate(np.real(b), cls)
            return (np.maximum if is_max else np.minimum)(a, b)
        if a.dtype.kind == "c" or b.dtype.kind == "c":
            ka, kb = np.abs(a), np.abs(b)
            a, b = np.broadcast_arrays(a.astype(complex), b.astype(complex))
            ka, kb = np.broadcast_arrays(ka, kb)
            pick_a = (ka >= kb) if is_max else (ka <= kb)
            if "includenan" in flags:
                r = np.where(np.isnan(ka), a, np.where(np.isnan(kb), b, np.where(pick_a, a, b)))
            else:
                r = np.where(np.isnan(kb), a, np.where(np.isnan(ka), b, np.where(pick_a, a, b)))
        elif "includenan" in flags:
            r = (np.maximum if is_max else np.minimum)(a, b)
        else:
            r = (np.fmax if is_max else np.fmin)(a, b)
        if cls is not None:
            r = r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)
        return r.astype(bool) if both_logical else r

    def f(interp, args, nargout):
        nargs(args, 1, 5, name)
        rest, flags = flag_present(args, "omitnan", "includenan", "all")
        if len(rest) >= 2 and not _no_arg(rest[1]):
            if len(rest) > 2 or "all" in flags:
                raise MatlabError(f"{name} with two arrays to compare does not take a dimension.")
            if nargout > 1:
                raise MatlabError(f"{name} with two arrays to compare returns only one output.")
            return pair(rest[0], rest[1], flags)
        src = rest[0]
        a = src if isinstance(src, np.ndarray) and src.dtype == bool else num(src)
        if a.shape == (0, 0) and len(rest) < 3 and "all" not in flags:
            return (a, np.zeros((0, 0))) if nargout > 1 else a
        dims = _red_dims(rest, 2, a, flags)
        b, shape = _merge(a, dims)
        if b.shape[-1] == 0:
            # nothing to compare along the operating dimension: the result stays empty
            if "all" in flags or len(dims) > 1:
                empty = np.zeros((0, 0), dtype=a.dtype)
            else:
                empty = np.zeros(shape if 0 in shape else a.shape, dtype=a.dtype)
            return (empty, np.zeros(empty.shape)) if nargout > 1 else empty
        key = np.abs(b) if b.dtype.kind == "c" else b
        nan = np.isnan(key) if key.dtype.kind == "f" else None
        if nan is not None and not nan.any():
            nan = None
        if nan is None and nargout <= 1 and b.dtype.kind != "c":
            return _unmerge((np.max if is_max else np.min)(b, axis=-1), shape)
        if nan is not None:
            if "includenan" in flags:
                safe = np.where(nan, np.inf if is_max else -np.inf, key)
                first_nan = np.argmax(nan, axis=-1)
                idx = np.where(nan.any(axis=-1), first_nan, (np.argmax if is_max else np.argmin)(safe, axis=-1))
            else:
                safe = np.where(nan, -np.inf if is_max else np.inf, key)
                idx = (np.argmax if is_max else np.argmin)(safe, axis=-1)
                # among the non-NaN elements; an all-NaN slice gives its first element
                masked = np.where(nan, False, safe == np.take_along_axis(safe, idx[..., None], -1))
                idx = np.where(nan.all(axis=-1), 0, np.argmax(masked, axis=-1))
        else:
            idx = (np.argmax if is_max else np.argmin)(key, axis=-1)
        vals = np.take_along_axis(b, np.expand_dims(idx, -1), -1)[..., 0]
        vals = _unmerge(vals, shape)
        if nargout > 1:
            return vals, _unmerge(idx.astype(float) + 1, shape)
        return vals
    return f


builtin("max")(_minmax(True))
builtin("min")(_minmax(False))


@builtin("bounds")
def _bounds(interp, args, nargout):
    from ._registry import BUILTINS
    full = [args[0], np.zeros((0, 0)), *args[1:]]
    lo = BUILTINS["min"](interp, full, 1)
    hi = BUILTINS["max"](interp, full, 1)
    return lo, hi


@builtin("range")
def _range(interp, args, nargout):
    a = fnum(args[0])
    if a.size == 0:
        return np.zeros((0, 0))
    b, shape = _merge(a, _red_dims(args, 1, a))
    return _unmerge(np.ptp(b, axis=-1), shape)


@builtin("diff")
def _diff(interp, args, nargout):
    nargs(args, 1, 3, "diff")
    a = num(args[0])
    n = to_int(args[1], "n") if len(args) > 1 and not _no_arg(args[1]) else 1
    if n < 0:
        raise MatlabError("diff: the order must be a nonnegative integer.")
    fixed = None
    if len(args) > 2 and not _no_arg(args[2]):
        fixed = to_int(args[2], "dimension") - 1
        if fixed < 0:
            raise MatlabError("Dimension argument must be a positive integer.")
    for _ in range(n):
        if a.size <= 1 and fixed is None and a.shape in ((0, 0), (1, 1)):
            return np.zeros((0, 0), dtype=a.dtype)
        axis = fixed if fixed is not None else default_dim(a)
        if axis >= a.ndim:
            return np.zeros(a.shape[:2] + (0,) * (axis - 1) if axis > 1 else a.shape, dtype=a.dtype)
        if a.shape[axis] == 0:
            return a
        if a.dtype.kind in "iu":
            a = _saturate(np.diff(_wide(a), axis=axis), a.dtype)
        else:
            a = np.diff(a, axis=axis)
    return a


def _trapz_args(args, name):
    """(x, y, axis): x is None (unit spacing), a scalar, a vector or an array shaped like y."""
    nargs(args, 1, 3, name)
    if len(args) == 1:
        x, y, dim = None, fnum(args[0]), None
    elif len(args) == 2:
        first, second = fnum(args[0]), fnum(args[1])
        if second.size == 1 and first.size != 1:
            x, y, dim = None, first, to_int(args[1], "dimension")  # (Y, dim)
        else:
            x, y, dim = first, second, None
    else:
        x, y, dim = fnum(args[0]), fnum(args[1]), to_int(args[2], "dimension")
    if dim is not None and dim < 1:
        raise MatlabError("Dimension argument must be a positive integer.")
    if dim is not None:
        axis = dim - 1
    elif x is not None and x.size > 1 and x.size == y.shape[0] and y.ndim == 2 and y.shape[0] != 1:
        axis = 0
    else:
        axis = default_dim(y)
    return x, y, axis


def _trapz_pieces(x, y, axis, name):
    """Areas of the individual trapezoids along the last axis of the rearranged y."""
    ym = np.moveaxis(y, axis, -1)
    n = ym.shape[-1]
    if x is None:
        dx = 1.0
    elif x.size == 1:
        dx = x.flat[0]
    elif x.shape == y.shape and not (x.ndim == 2 and 1 in x.shape):
        dx = np.diff(np.moveaxis(x, axis, -1), axis=-1)
    else:
        if x.size != n:
            raise MatlabError(f"{name}: the length of X must match the size of Y along the "
                              "operating dimension.")
        dx = np.diff(x.flatten(order="F"))
    return dx * (ym[..., 1:] + ym[..., :-1]) / 2


@builtin("trapz")
def _trapz(interp, args, nargout):
    x, y, axis = _trapz_args(args, "trapz")
    if y.shape == (0, 0):
        return 0.0
    if axis >= y.ndim:
        return np.zeros(y.shape)
    r = _trapz_pieces(x, y, axis, "trapz").sum(axis=-1)
    return keep2d(np.asarray(r), axis, y.ndim)


@builtin("cumtrapz")
def _cumtrapz(interp, args, nargout):
    x, y, axis = _trapz_args(args, "cumtrapz")
    if y.size == 0 or axis >= y.ndim:
        return np.zeros(y.shape)
    pieces = _trapz_pieces(x, y, axis, "cumtrapz")
    zero = np.zeros(pieces.shape[:-1] + (1,), dtype=pieces.dtype)
    r = np.concatenate([zero, np.cumsum(pieces, axis=-1)], axis=-1)
    return np.moveaxis(r, -1, axis)


@builtin("dot")
def _dot(interp, args, nargout):
    nargs(args, 2, 3, "dot")
    a, b = num(args[0]), num(args[1])
    single = _pair_class(fnum(a), fnum(b), "dot") is not None
    a, b = fnum(a), fnum(b)

    def vec(x):
        return x.ndim == 2 and 1 in x.shape

    if len(args) < 3 and vec(a) and vec(b):
        if a.size != b.size:
            raise MatlabError("A and B must be the same size.")
        r = simplify(np.asarray(np.vdot(a.flatten(), b.flatten())).reshape(1, 1))
    else:
        if a.shape != b.shape:
            raise MatlabError("A and B must be the same size.")
        axis = to_int(args[2], "dimension") - 1 if len(args) > 2 else default_dim(a)
        if axis < 0:
            raise MatlabError("Dimension argument must be a positive integer.")
        prod = np.conj(a) * b
        r = prod if axis >= prod.ndim else keep2d(np.sum(prod, axis=axis), axis, prod.ndim)
        r = simplify(r)
    if single:
        r = r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)
    return r


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
    nargs(args, 1, 3, "find")
    src = args[0]
    a = src if isinstance(src, np.ndarray) and src.dtype == bool else num(src)
    flat = a.flatten(order="F")
    idx = np.nonzero(flat)[0]
    if len(args) > 1:
        k = to_int(args[1], "n")
        if k < 1:
            raise MatlabError("find: the number of indices must be a positive integer.")
        direction = text(args[2]).lower() if len(args) > 2 else "first"
        if direction not in ("first", "last"):
            raise MatlabError("find: the direction must be 'first' or 'last'.")
        idx = idx[-k:] if direction == "last" else idx[:k]
    if idx.size == 0 and (a.shape == (0, 0) or a.shape == (1, 1)):
        shape = (0, 0)
    elif a.ndim == 2 and a.shape[0] == 1 and a.shape[1] != 1:
        shape = (1, -1)
    else:
        shape = (-1, 1)
    if nargout <= 1:
        return (idx + 1.0).reshape(shape)
    rows = idx % a.shape[0] if a.shape[0] else idx
    cols = idx // a.shape[0] if a.shape[0] else idx
    outs = [(rows + 1.0).reshape(shape), (cols + 1.0).reshape(shape)]
    if nargout > 2:
        outs.append(flat[idx].reshape(shape))
    return tuple(outs)


@builtin("nnz")
def _nnz(interp, args, nargout):
    return float(np.count_nonzero(num(args[0])))


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


def _sort_index(work, descend):
    """Stable sort permutation along the last axis, with MATLAB's NaN and complex ordering."""
    kind = work.dtype.kind
    if kind in "biu":
        return np.argsort(~work if descend else work, axis=-1, kind="stable")
    if kind == "c":
        mag, ang = np.abs(work), np.angle(work)
        nan = np.isnan(mag)
        if descend:
            return np.lexsort((-ang, -mag, ~nan), axis=-1)
        return np.lexsort((ang, mag, nan), axis=-1)
    if descend:
        return np.lexsort((-work, ~np.isnan(work)), axis=-1)  # NaNs first, then large to small
    return np.argsort(work, axis=-1, kind="stable")  # NaNs last


@builtin("sort")
def _sort(interp, args, nargout):
    rest, flags = flag_present(args, "ascend", "descend")
    nargs(rest, 1, 2, "sort")
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
    a = v if isinstance(v, np.ndarray) and v.dtype.kind in "bU" else num(v)
    work = num(a) if is_char(a) else a
    if len(rest) > 1 and not _no_arg(rest[1]):
        axis = to_int(rest[1], "dimension") - 1
        if axis < 0:
            raise MatlabError("Dimension argument must be a positive integer.")
    else:
        axis = default_dim(work)
    if work.size == 0 or axis >= work.ndim:
        return (a, np.ones(a.shape)) if nargout > 1 else a
    if axis == work.ndim - 1:
        idx = _sort_index(work, "descend" in flags)
    else:
        idx = np.moveaxis(_sort_index(np.moveaxis(work, axis, -1), "descend" in flags), -1, axis)
    out = np.take_along_axis(a, idx, axis)
    if nargout > 1:
        return out, idx.astype(float) + 1
    return out


def _row_groups(m):
    """Group equal rows of a 2-D numeric array.

    Returns (order, group): ``order`` sorts the rows (stable, NaN last, complex by magnitude
    then angle) and ``group[k]`` is the group number of the sorted row k. NaN never equals NaN,
    so every row with a NaN is a group of its own.
    """
    n = m.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    keys = []
    for c in range(m.shape[1] - 1, -1, -1):  # lexsort: the last key is the most significant
        col = m[:, c]
        if col.dtype.kind == "c":
            keys += [np.angle(col), np.abs(col)]
        else:
            keys.append(col)
    order = np.lexsort(tuple(keys)) if keys else np.arange(n)
    ms = m[order]
    new = np.any(ms[1:] != ms[:-1], axis=1) if m.shape[1] else np.zeros(n - 1, dtype=bool)
    return order, np.concatenate([[0], np.cumsum(new)]).astype(int)


def _set_matrix(v, rows):
    """A set argument as (2-D array of numeric keys with one item per row, original array)."""
    a = v if isinstance(v, np.ndarray) and v.dtype.kind in "bU" else num(v)
    work = num(a) if a.dtype.kind in "bU" else a
    if rows:
        if work.ndim != 2:
            raise MatlabError("The 'rows' option needs 2-D inputs.")
        return work, a
    return work.reshape(-1, 1, order="F"), a


def _set_restore(m, like, rows, as_row):
    """Turn selected key rows back into the class of ``like`` and the right orientation."""
    if like.dtype.kind == "U":
        from ..values import codes_to_char
        out = codes_to_char(m) if m.size else np.zeros(m.shape, dtype="<U1")
    elif like.dtype == bool:
        out = m != 0
    else:
        out = m.astype(like.dtype, copy=False) if like.dtype.kind in "iuf" and m.dtype.kind != "c" else m
    if rows:
        return out
    return out.reshape(1, -1) if as_row else out.reshape(-1, 1)


def _is_row(a):
    return a.ndim == 2 and a.shape[0] == 1


@builtin("unique")
def _unique(interp, args, nargout):
    rest, flags = flag_present(args, "stable", "sorted", "rows", "first", "last", "legacy")
    nargs(rest, 1, 1, "unique")
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
    rows = "rows" in flags
    m, a = _set_matrix(v, rows)
    order, group = _row_groups(m)
    n = m.shape[0]
    ngroups = int(group[-1]) + 1 if n else 0
    starts = np.concatenate([[True], group[1:] != group[:-1]]) if n else np.zeros(0, dtype=bool)
    if "last" in flags:
        ends = np.concatenate([group[1:] != group[:-1], [True]])
        ia = order[ends]
    else:
        ia = order[starts]
    ic = np.empty(n, dtype=int)
    ic[order] = group
    if "stable" in flags:
        pos = np.argsort(ia, kind="stable")
        ia = ia[pos]
        remap = np.empty(ngroups, dtype=int)
        remap[pos] = np.arange(ngroups)
        ic = remap[ic]
    if a.shape == (0, 0):
        out = a
    else:
        out = _set_restore(m[ia], a, rows, _is_row(a) and a.size > 0)
    if nargout > 1:
        return out, ia.reshape(-1, 1) + 1.0, ic.reshape(-1, 1) + 1.0
    return out


def _set_values(v):
    return [to_str(x) for x in v.values_f()]


def _cell_set_op(kind, rest, flags, nargout):
    a, b = _set_values(rest[0]), _set_values(rest[1])
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
        vals = list(dict.fromkeys(vals))
    else:
        vals = sorted(set(vals))
    out = Cell.from_list([char(s) for s in vals])
    if nargout > 1:
        ia = [a.index(x) + 1.0 for x in vals if x in sa]
        outs = [out, np.array(ia).reshape(-1, 1)]
        if nargout > 2:
            if kind == "intersect":
                ib = [b.index(x) + 1.0 for x in vals]
            else:
                ib = [b.index(x) + 1.0 for x in vals if x not in sa]
            outs.append(np.array(ib).reshape(-1, 1))
        return tuple(outs)
    return out


def _set_op(kind):
    def f(interp, args, nargout):
        from .strings import _as_cell_if_strings
        rest, flags = flag_present(args, "stable", "sorted", "rows", "legacy")
        nargs(rest, 2, 2, kind)
        va, vb = _as_cell_if_strings(rest[0]), _as_cell_if_strings(rest[1])
        if isinstance(va, Cell) or isinstance(vb, Cell):
            va = va if isinstance(va, Cell) else Cell.from_list([va] if is_text(va) else [])
            vb = vb if isinstance(vb, Cell) else Cell.from_list([vb] if is_text(vb) else [])
            out = _cell_set_op(kind, [va, vb], flags, nargout)
            if isinstance(rest[0], (MString, StringArray)) or isinstance(rest[1], (MString, StringArray)):
                first = out[0] if isinstance(out, tuple) else out
                first = string_result(np.vectorize(to_str, otypes=[object])(first.data)) \
                    if first.data.size else first
                out = (first, *out[1:]) if isinstance(out, tuple) else first
            return out
        rows = "rows" in flags
        ma, a = _set_matrix(va, rows)
        mb, b = _set_matrix(vb, rows)
        if rows and ma.shape[1] != mb.shape[1]:
            if ma.size and mb.size:
                raise MatlabError(f"{kind}: with 'rows', A and B must have the same number of columns.")
            width = ma.shape[1] if ma.size else mb.shape[1]
            ma = ma.reshape(ma.shape[0] if ma.size else 0, width)
            mb = mb.reshape(mb.shape[0] if mb.size else 0, width)
        # one class for the keys: an integer class wins, as in concatenation
        like = a
        for x, other in ((a, b), (b, a)):
            if x.dtype.kind in "iu":
                like = x
                if other.dtype.kind in "iu" and other.dtype != x.dtype:
                    raise MatlabError(f"{kind}: integer inputs must have the same class.")
                break
        else:
            if a.dtype.kind == "U" or b.dtype.kind == "U":
                like = a if a.dtype.kind == "U" else b
            elif a.dtype == bool and b.dtype != bool:
                like = b
            elif np.float32 in (a.dtype, b.dtype):
                like = a if a.dtype == np.float32 else b
        if kind == "setdiff":
            like = a if like.dtype.kind not in "iu" else like
        if like.dtype.kind in "iu":
            ma = ma if ma.dtype == like.dtype else _saturate(np.real(ma), like.dtype)
            mb = mb if mb.dtype == like.dtype else _saturate(np.real(mb), like.dtype)
        na = ma.shape[0]
        m = np.concatenate([ma, mb], axis=0)
        order, group = _row_groups(m)
        ngroups = int(group[-1]) + 1 if m.shape[0] else 0
        gid = np.empty(m.shape[0], dtype=int)
        gid[order] = group
        big = m.shape[0] + 1
        first_a = np.full(ngroups, big)
        first_b = np.full(ngroups, big)
        np.minimum.at(first_a, gid[:na], np.arange(na))
        np.minimum.at(first_b, gid[na:], np.arange(m.shape[0] - na))
        in_a, in_b = first_a < big, first_b < big
        if kind == "intersect":
            keep = in_a & in_b
        elif kind == "union":
            keep = in_a | in_b
        elif kind == "setdiff":
            keep = in_a & ~in_b
        else:
            keep = in_a ^ in_b
        groups = np.nonzero(keep)[0]
        rep = np.where(in_a[groups], first_a[groups], na + first_b[groups])
        if "stable" in flags:
            pos = np.argsort(rep, kind="stable")
            groups, rep = groups[pos], rep[pos]
        if kind == "setdiff":
            as_row = _is_row(a)
        else:
            as_row = (_is_row(a) or a.shape == (0, 0)) and (_is_row(b) or b.shape == (0, 0)) \
                and not (a.shape == (0, 0) and b.shape == (0, 0))
        if a.shape == (0, 0) and (b.shape == (0, 0) or kind == "setdiff") and not rows:
            out = np.zeros((0, 0), dtype=like.dtype)
        else:
            out = _set_restore(m[rep], like, rows, as_row)
        if nargout <= 1:
            return out
        from_a = in_a[groups]
        outs = [out, (first_a[groups][from_a] + 1.0).reshape(-1, 1)]
        if nargout > 2:
            if kind == "intersect":
                outs.append((first_b[groups] + 1.0).reshape(-1, 1))
            elif kind == "setdiff":
                raise MatlabError("setdiff returns at most two outputs.")
            else:
                outs.append((first_b[groups][~from_a] + 1.0).reshape(-1, 1))
        return tuple(outs)
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


def _bit_class(a, b, name):
    """Result dtype of a bit function: the integer class of the inputs, or None for double."""
    ia, ib = a.dtype.kind in "iu", b.dtype.kind in "iu"
    if ia and ib and a.dtype != b.dtype:
        raise MatlabError(f"{name}: integer inputs must have the same class.")
    return a.dtype if ia else (b.dtype if ib else None)


def _bit_ints(a):
    """Object array of Python ints (two's complement handled by the caller)."""
    if a.dtype.kind in "iu":
        return a.astype(object)
    a = np.real(a)
    if a.size and not np.all(np.isfinite(a)):
        raise MatlabError("Bit functions need finite integer inputs.")
    if a.size and np.all(np.abs(a) < 2.0 ** 62):
        return a.astype(np.int64).astype(object)
    return np.vectorize(int, otypes=[object])(a) if a.size else np.zeros(a.shape, dtype=object)


def _bit_finish(r, cls, shape):
    """Wrap an object array of Python ints into the result class."""
    if cls is None:
        return np.array(r, dtype=float).reshape(shape)
    bits = cls.itemsize * 8
    mask = (1 << bits) - 1

    def wrap(x):
        x = int(x) & mask
        if cls.kind == "i" and x >= 1 << (bits - 1):
            x -= 1 << bits
        return x

    flat = [wrap(x) for x in np.asarray(r, dtype=object).reshape(-1)]
    return np.array(flat, dtype=cls).reshape(shape)


def _bit(fn, name):
    def f(interp, args, nargout):
        nargs(args, 2, 3, name)
        a, b = num(args[0]), num(args[1])
        cls = _bit_class(a, b, name)
        a, b = np.broadcast_arrays(_bit_ints(a), _bit_ints(b))
        r = [fn(int(x), int(y)) for x, y in zip(a.reshape(-1), b.reshape(-1))]
        return _bit_finish(r, cls, a.shape)
    return f


builtin("bitand")(_bit(lambda x, y: x & y, "bitand"))
builtin("bitor")(_bit(lambda x, y: x | y, "bitor"))
builtin("bitxor")(_bit(lambda x, y: x ^ y, "bitxor"))


@builtin("bitshift")
def _bitshift(interp, args, nargout):
    nargs(args, 2, 3, "bitshift")
    a0 = num(args[0])
    cls = a0.dtype if a0.dtype.kind in "iu" else None
    a, n = np.broadcast_arrays(_bit_ints(a0), _bit_ints(num(args[1])))
    r = [(int(x) << int(k)) if k >= 0 else (int(x) >> -int(k)) for x, k in zip(a.reshape(-1), n.reshape(-1))]
    if cls is None:
        r = [x & ((1 << 53) - 1) for x in r]
    return _bit_finish(r, cls, a.shape)


@builtin("bitget")
def _bitget(interp, args, nargout):
    a0 = num(args[0])
    cls = a0.dtype if a0.dtype.kind in "iu" else None
    a, b = np.broadcast_arrays(_bit_ints(a0), _bit_ints(num(args[1])))
    top = cls.itemsize * 8 if cls is not None else 53
    if any(k < 1 or k > top for k in b.reshape(-1)):
        raise MatlabError(f"bitget: bit positions must be between 1 and {top}.")
    r = [(int(x) >> (int(k) - 1)) & 1 for x, k in zip(a.reshape(-1), b.reshape(-1))]
    return _bit_finish(r, cls, a.shape)


@builtin("bitset")
def _bitset(interp, args, nargout):
    a0 = num(args[0])
    cls = a0.dtype if a0.dtype.kind in "iu" else None
    v = num(args[2]) if len(args) > 2 else np.ones((1, 1))
    a, b, v = np.broadcast_arrays(_bit_ints(a0), _bit_ints(num(args[1])), v != 0)
    top = cls.itemsize * 8 if cls is not None else 53
    if any(k < 1 or k > top for k in b.reshape(-1)):
        raise MatlabError(f"bitset: bit positions must be between 1 and {top}.")
    r = [(int(x) | (1 << (int(k) - 1))) if on else (int(x) & ~(1 << (int(k) - 1)))
         for x, k, on in zip(a.reshape(-1), b.reshape(-1), v.reshape(-1))]
    return _bit_finish(r, cls, a.shape)


@builtin("bitcmp")
def _bitcmp(interp, args, nargout):
    a = num(args[0])
    cls = a.dtype if a.dtype.kind in "iu" else None
    bits = cls.itemsize * 8 if cls is not None else 53
    if len(args) > 1 and cls is None:
        bits = {"uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64}.get(to_str(args[1]), bits)
    full = (1 << bits) - 1
    r = [full ^ (int(x) & full) for x in _bit_ints(a).reshape(-1)]
    return _bit_finish(r, cls, a.shape)


@builtin("swapbytes")
def _swapbytes(interp, args, nargout):
    nargs(args, 1, 1, "swapbytes")
    a = args[0]
    if not (isinstance(a, np.ndarray) and a.dtype.kind in "iu"):
        raise MatlabError("swapbytes: the input must be an integer array (for example uint16).")
    return a.byteswap()
