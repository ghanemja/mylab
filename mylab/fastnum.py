"""Fast numeric base layer under the operators and the hottest built-ins.

NumPy's inner loops are compiled and use the same BLAS as other systems; what costs time is how
they are reached.  A call on a 1x1 or 3-element array costs about a microsecond before any
arithmetic happens, and the general operator code adds class checks, size checks and error-state
handling on top.  This module holds the shortcuts, in two groups:

*Small values.*  Real double scalars and vectors of up to eight elements are handled with plain
Python floats (arithmetic, comparisons, ``sum``, ``prod``, ``max``, ``min``, ``any``, ``all``,
``abs``, ``round``, ``mod``, ``rem``, ``numel``, ``cumsum``), and plain double arrays of any size
go straight to the NumPy function the general code would end up calling.

*Large arrays.*  A better algorithm or back end where one exists: ``mod`` without full-size
temporaries, value-only ``sort``/``unique``, ``ismember`` by binary search, FFTs through
``scipy.fft`` with several workers, and LAPACK drivers chosen by the structure of the matrix.

Rules every shortcut follows:

* It returns exactly what the general code returns: the same values (to the last bit), class and
  shape.  The only exceptions are the LAPACK/FFT back-end choices (``posv``, triangular solves,
  values-only ``eig``, ``scipy.fft``), which are a different but equally valid computation and
  agree with the general code to rounding error.
* When it is not sure it applies - another class, complex numbers, a NaN or signed-zero corner it
  cannot reproduce, a shape that needs expansion - it returns ``None`` and the general code runs.
* Summation and product order are checked against NumPy the first time they are used
  (:func:`_verified`); if NumPy ever changes its order the shortcut switches itself off.

``MYLAB_FASTNUM=0`` (or :func:`set_enabled`) turns the whole layer off.  Nothing heavy is imported
here: SciPy is loaded on first use, and everything has a pure-NumPy fallback.
"""

from __future__ import annotations

import math
import operator
import os
import sys
from itertools import accumulate
from typing import Any, Callable

import numpy as np

from .values import Cell, StringArray, Struct

_CONTAINERS = (Cell, Struct, StringArray)
ENABLED = os.environ.get("MYLAB_FASTNUM", "1") != "0"

_ND = np.ndarray
_F64 = np.dtype(np.float64)
_C128 = np.dtype(np.complex128)
_BOOL = np.dtype(np.bool_)
_S11 = (1, 1)
_empty = np.empty
_INF = math.inf
_EPS = 2.220446049250313e-16
TINY = 8             # largest vector handled element by element in Python
FFT_MIN = 1 << 14    # elements; smaller transforms stay with numpy.fft


def set_enabled(on: bool) -> None:
    """Switch the layer on or off at run time (the environment variable sets the default)."""
    global ENABLED
    ENABLED = bool(on)


def scalar(x: float) -> np.ndarray:
    """A 1x1 double array (about half the cost of ``np.array([[x]])``)."""
    r = _empty(_S11)
    r[0, 0] = x
    return r


def logical(x: bool) -> np.ndarray:
    r = _empty(_S11, _BOOL)
    r[0, 0] = x
    return r


# -- floating-point warnings ---------------------------------------------------------------------
# The general code wraps each operation in np.errstate (about 0.8 microseconds a time).  The
# interpreter instead silences NumPy's floating-point warnings once around each run, so the
# shortcuts below can call ufuncs bare.

class _Quiet:
    __slots__ = ("depth", "state")

    def __init__(self):
        self.depth = 0
        self.state = None


try:
    import threading
    _QUIET_LOCAL = threading.local()
except Exception:  # pragma: no cover - no thread support
    _QUIET_LOCAL = _Quiet()


def quiet_enter() -> None:
    q = getattr(_QUIET_LOCAL, "q", None)
    if q is None:
        q = _QUIET_LOCAL.q = _Quiet()
    if q.depth == 0:
        q.state = np.errstate(all="ignore")
        q.state.__enter__()
    q.depth += 1


def quiet_leave() -> None:
    q = getattr(_QUIET_LOCAL, "q", None)
    if q is None or q.depth == 0:
        return
    q.depth -= 1
    if q.depth == 0:
        state, q.state = q.state, None
        try:
            state.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - left from another context: nothing to restore
            pass


# -- operators -----------------------------------------------------------------------------------

def _s_div(x, y):
    return x / y if y != 0.0 else None


def _s_ldiv(x, y):
    return y / x if x != 0.0 else None


def _s_pow(x, y):
    if x < 0.0 and y == y and y not in (_INF, -_INF) and y != int(y):
        return None  # a complex result: the general code computes it
    try:
        r = x ** y
    except (OverflowError, ZeroDivisionError):
        return None
    return r if type(r) is float else None


_SC_ARITH = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, ".*": operator.mul,
    "/": _s_div, "./": _s_div, "\\": _s_ldiv, ".\\": _s_ldiv, "^": _s_pow, ".^": _s_pow,
}
_SC_COMPARE = {
    "==": operator.eq, "~=": operator.ne, "<": operator.lt, "<=": operator.le,
    ">": operator.gt, ">=": operator.ge,
}
_UFUNC = {
    "+": np.add, "-": np.subtract, ".*": np.multiply, "./": np.divide,
    "==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal,
    ">": np.greater, ">=": np.greater_equal,
}


def binop_f64(op: str, a: np.ndarray, b: np.ndarray):
    """``a op b`` for two plain double arrays, or None when the general code must do it."""
    sa, sb = a.shape, b.shape
    if sa == _S11 and sb == _S11:
        f = _SC_ARITH.get(op)
        if f is not None:
            r = f(a.item(), b.item())
            if r is None:
                return None
            out = _empty(_S11)
            out[0, 0] = r
            return out
        f = _SC_COMPARE.get(op)
        if f is not None:
            out = _empty(_S11, _BOOL)
            out[0, 0] = f(a.item(), b.item())
            return out
        return None
    if sa != sb and sa != _S11 and sb != _S11:
        return None  # implicit expansion or a size error
    u = _UFUNC.get(op)
    if u is not None:
        return u(a, b)
    if op == "*":
        if sa == _S11 or sb == _S11:
            return np.multiply(a, b)
    elif op == "/":
        if sb == _S11:
            return np.divide(a, b)
    elif op == "\\" or op == ".\\":
        if sa == _S11 or op == ".\\":
            return np.divide(b, a)
    elif op == ".^":
        if sb == _S11:
            y = b.item()
            if y == y and y not in (_INF, -_INF) and y == math.floor(y):
                return np.power(a, b)  # an integer exponent never gives a complex result
    return None


# Integer classes of up to 32 bits: every value is exact in a double, so arithmetic with a double, a
# logical or the same class is the double operation (Python's, on ints and floats, gives the same double),
# rounded halves away from zero and saturated, as saturate_int does it.
_SMALL_INT = {np.dtype(t): (int(np.iinfo(t).min), int(np.iinfo(t).max))
              for t in ("int8", "uint8", "int16", "uint16", "int32", "uint32")}
_INT_ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, ".*": operator.mul,
              "/": _s_div, "./": _s_div}


def saturate(x, lo: int, hi: int) -> int:
    """A double (or exact int) as an integer class: NaN is 0, halves round away from zero, saturated."""
    if x != x:
        return 0
    if x >= hi:
        return hi
    if x <= lo:
        return lo
    m = abs(x)
    r = math.floor(m)
    if m - r >= 0.5:
        r += 1
    return -r if x < 0 else r


def binop_int(op: str, a: np.ndarray, b: np.ndarray):
    """``a op b`` for 1x1 operands, one of an integer class of up to 32 bits and the other a double, a
    logical or the same class; None when the general code must do it."""
    ta, tb = a.dtype, b.dtype
    lim = _SMALL_INT.get(ta)
    if lim is not None:
        cls = ta
        if tb != ta and tb is not _F64 and tb is not _BOOL:
            return None
    else:
        lim = _SMALL_INT.get(tb)
        if lim is None or (ta is not _F64 and ta is not _BOOL):
            return None
        cls = tb
    f = _INT_ARITH.get(op)
    if f is not None:
        r = f(a.item(), b.item())
        if r is None:
            return None  # division by zero: the general code gives the saturated infinity or 0
        out = _empty(_S11, cls)
        out[0, 0] = saturate(r, *lim)
        return out
    f = _SC_COMPARE.get(op)
    if f is not None:  # Python compares ints, bools and floats by their exact values
        out = _empty(_S11, _BOOL)
        out[0, 0] = f(a.item(), b.item())
        return out
    return None


def binop_logical_double(op: str, a: np.ndarray, b: np.ndarray):
    """``a op b`` for a 1x1 logical and a 1x1 double (either order): the logical counts as 0 or 1, the
    result is a double (arithmetic) or a logical (comparison); None when the general code must do it."""
    f = _SC_ARITH.get(op)
    if f is not None:
        r = f(float(a.item()), float(b.item()))
        if r is None:
            return None
        out = _empty(_S11)
        out[0, 0] = r
        return out
    f = _SC_COMPARE.get(op)
    if f is not None:
        out = _empty(_S11, _BOOL)
        out[0, 0] = f(a.item(), b.item())
        return out
    return None


_BOOL_UFUNC = {"&": np.logical_and, "|": np.logical_or, "==": np.equal, "~=": np.not_equal}


def binop_bool(op: str, a: np.ndarray, b: np.ndarray):
    """``a op b`` for two logical arrays (``&``, ``|``, ``==``, ``~=``), or None."""
    u = _BOOL_UFUNC.get(op)
    if u is None:
        return None
    sa, sb = a.shape, b.shape
    if sa != sb and sa != _S11 and sb != _S11:
        return None
    return u(a, b)


def _hcat(items: list):
    n = len(items)
    if n == 0:
        return None
    if n == 1:
        first = items[0]
        if type(first) is _ND and first.dtype is _F64 and first.ndim == 2 and first.size:
            return first
        return None
    scalars = True
    rows = -1
    for i in items:
        if type(i) is not _ND or i.dtype is not _F64 or i.ndim != 2:
            return None
        s = i.shape
        if s != _S11:
            if s[0] == 0 or s[1] == 0:
                return None  # empty pieces follow their own rules
            scalars = False
        if rows < 0:
            rows = s[0]
        elif s[0] != rows:
            return None
    if scalars:
        return np.array([[i.item() for i in items]])
    return np.concatenate(items, axis=1)


def build_matrix(rows: list):
    """``[a, b; c, d]`` when every piece is a non-empty plain double matrix, else None."""
    if len(rows) == 1:
        return _hcat(rows[0])
    parts = []
    cols = -1
    for r in rows:
        p = _hcat(r)
        if p is None:
            return None
        if cols < 0:
            cols = p.shape[1]
        elif p.shape[1] != cols:
            return None
        parts.append(p)
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


# -- small vectors -------------------------------------------------------------------------------

def _tiny_vec(a) -> list | None:
    """The elements of a real double vector with 1..TINY elements, else None."""
    if type(a) is _ND and a.dtype is _F64 and a.ndim == 2:
        s = a.shape
        if s[0] == 1:
            if 0 < s[1] <= TINY:
                return a.tolist()[0]
        elif s[1] == 1 and 0 < s[0] <= TINY:
            return a.ravel().tolist()
    return None


def _sum_list(v: list) -> float:
    # NumPy starts from +0 (so a sum of negative zeros is +0) and adds blocks of eight pairwise.
    if len(v) < 8:
        r = 0.0
        for x in v:
            r += x
        return r
    return 0.0 + (((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])))


def _prod_list(v: list) -> float:
    r = v[0]
    for x in v[1:]:
        r *= x
    return r


_CHECKED: dict[str, bool] = {}
_SLOW: dict[str, Callable] = {}   # the general built-ins, filled in by install()


def _probes(n: int, count: int = 6) -> list[list[float]]:
    """Deterministic test vectors with mixed signs and magnitudes (they expose a change of order)."""
    out = []
    state = 12345 + n
    for _ in range(count):
        v = []
        for _ in range(n):
            state = (state * 1103515245 + 12345) % 2147483648
            mant = state / 2147483648.0 - 0.5
            state = (state * 1103515245 + 12345) % 2147483648
            v.append(mant * 10.0 ** (state % 17 - 8))
        out.append(v)
    out.append([-0.0] * n)                                   # signed zeros show where a sum starts
    out.append([0.0 if k % 2 else -0.0 for k in range(n)])
    return out


def _same_bits(x: np.ndarray, y: np.ndarray) -> bool:
    return x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def _verified(family: str) -> bool:
    """Whether the Python version of ``family`` reproduces the general built-in on this NumPy."""
    ok = _CHECKED.get(family)
    if ok is None:
        _CHECKED[family] = True  # so that the shortcut under test runs during the check
        try:
            ok = _run_check(family)
        except Exception:  # noqa: BLE001 - any surprise: do not use the shortcut
            ok = False
        _CHECKED[family] = ok
    return ok


def _run_check(family: str) -> bool:
    slow = _SLOW.get(family)
    if slow is None:
        return False
    fast = _FAST[family]
    for n in range(1, TINY + 1):
        for v in _probes(n):
            for arr in (np.array([v]), np.array([v]).T.copy()):
                args = [arr, arr[::-1].copy()] if family == "dot" else [arr]
                want = np.asarray(slow(None, list(args), 1), dtype=float).reshape(-1)
                got = fast(list(args), 1)
                if got is None or not _same_bits(np.asarray(got, dtype=float).reshape(-1), want):
                    return False
    return True


def _f_sum(args, nargout):
    if len(args) != 1:
        return None
    v = _tiny_vec(args[0])
    if v is None or not (_CHECKED.get("sum") or _verified("sum")):
        return None
    return scalar(_sum_list(v))


def _f_prod(args, nargout):
    if len(args) != 1:
        return None
    v = _tiny_vec(args[0])
    if v is None or not (_CHECKED.get("prod") or _verified("prod")):
        return None
    return scalar(_prod_list(v))


def _f_cumsum(args, nargout):
    if len(args) != 1:
        return None
    a = args[0]
    v = _tiny_vec(a)
    if v is None or not (_CHECKED.get("cumsum") or _verified("cumsum")):
        return None
    return np.array(list(accumulate(v))).reshape(a.shape)


def _minmax(is_max: bool):
    pick = max if is_max else min

    def f(args, nargout):
        if nargout > 1:
            return None
        n = len(args)
        if n == 1:
            v = _tiny_vec(args[0])
            if v is None:
                return None
            if len(v) == 1:
                return scalar(v[0])
            for x in v:
                if x != x:
                    return None  # NaNs are skipped by rules the general code knows
            r = pick(v)
            if r == 0.0:
                return None  # +0 or -0: let NumPy pick, as before
            return scalar(r)
        if n == 2:
            a, b = args
            if type(a) is _ND and type(b) is _ND and a.dtype is _F64 and b.dtype is _F64 \
                    and a.shape == _S11 and b.shape == _S11:
                x, y = a.item(), b.item()
                if x != x or y != y:
                    return None
                if x == y:
                    return None if x == 0.0 else scalar(x)
                if is_max:
                    return scalar(x if x > y else y)
                return scalar(x if x < y else y)
        return None
    return f


def _anyall(is_any: bool):
    test = any if is_any else all

    def f(args, nargout):
        if len(args) != 1:
            return None
        a = args[0]
        if type(a) is not _ND or a.ndim != 2 or not (a.dtype is _F64 or a.dtype is _BOOL):
            return None
        s = a.shape
        if not (0 < a.size <= TINY and (s[0] == 1 or s[1] == 1)):
            return None
        return logical(test(a.ravel().tolist()))  # NaN counts as non-zero, as in the general code
    return f


def _f_abs(args, nargout):
    if len(args) == 1:
        a = args[0]
        if type(a) is _ND and a.dtype is _F64:
            if a.shape == _S11:
                return scalar(abs(a.item()))
            return np.abs(a)
    return None


def _f_round(args, nargout):
    if len(args) != 1:
        return None
    a = args[0]
    if type(a) is _ND and a.dtype is _F64 and a.shape == _S11:
        x = a.item()
        if x != x or x in (_INF, -_INF):
            return None
        m = abs(x)
        low = float(math.floor(m))
        return scalar(math.copysign(low + 1.0 if m - low >= 0.5 else low, x))
    if type(a) is _ND and a.dtype is _F64 and a.size:
        if a.size <= TINY:
            out = []
            for x in a.ravel().tolist():
                if x != x or x in (_INF, -_INF):
                    out = None
                    break
                m = abs(x)
                low = float(math.floor(m))
                out.append(math.copysign(low + 1.0 if m - low >= 0.5 else low, x))
            if out is not None:
                return np.array(out).reshape(a.shape)
        # the general formula (halves away from zero) with its temporaries reused
        mag = np.abs(a)
        low = np.floor(mag)
        np.subtract(mag, low, out=mag)
        np.add(low, 1.0, out=low, where=mag >= 0.5)
        return np.copysign(low, a, out=low)
    return None


def _f_numel(args, nargout):
    if len(args) == 1:
        a = args[0]
        if type(a) is _ND:
            return scalar(float(a.size))
        if type(a) in _CONTAINERS:
            return scalar(float(a.data.size))
    return None


def _f_length(args, nargout):
    if len(args) == 1:
        a = args[0]
        if type(a) is _ND:
            return scalar(float(max(a.shape) if a.size else 0))
    return None


def _f_size(args, nargout):
    if nargout <= 1 and len(args) == 2:
        a, k = args
        if type(a) is _ND and type(k) is _ND and k.dtype is _F64 and k.shape == _S11:
            d = k.item()
            if d >= 1 and d == math.floor(d):
                d = int(d)
                return scalar(float(a.shape[d - 1]) if d <= a.ndim else 1.0)
    return None


def _f_norm(args, nargout):
    if len(args) != 1:
        return None
    a = args[0]
    if type(a) is _ND and a.dtype is _F64 and a.ndim == 2 and a.size and (a.shape[0] == 1 or a.shape[1] == 1) \
            and (_CHECKED.get("norm") or _verified("norm")):
        v = a.ravel()
        s = v.dot(v)
        if s != s:
            return None  # NaN in the data: the general path decides which NaN comes back
        return scalar(math.sqrt(s))  # what numpy.linalg.norm computes for a real vector
    return None


def _f_dot(args, nargout):
    if len(args) != 2:
        return None
    a, b = args
    if type(a) is _ND and type(b) is _ND and a.dtype is _F64 and b.dtype is _F64 and a.ndim == 2 \
            and b.ndim == 2 and a.size == b.size and a.size \
            and (a.shape[0] == 1 or a.shape[1] == 1) and (b.shape[0] == 1 or b.shape[1] == 1) \
            and (_CHECKED.get("dot") or _verified("dot")):
        r = 0.0 + float(a.ravel().dot(b.ravel()))  # (vdot starts its sum from +0)
        return scalar(r) if r == r else None  # a NaN may differ in its sign bit: general code
    return None


_MOD_SCALAR: dict[str, Callable] = {}   # mod/rem of two Python floats (from mathfns, set by install())


def _mod_large(a: np.ndarray, y: float):
    """``mod(a, y)`` for a real double array and a finite non-zero scalar, or None.

    The same operations as the general formula with one reused temporary; the pass that rounds
    results within roundoff of ``y`` to zero is skipped when the extremes show it cannot apply.
    """
    r = np.divide(a, y)
    np.floor(r, out=r)
    np.multiply(r, y, out=r)
    np.subtract(a, r, out=r)
    lo, hi = float(a.min()), float(a.max())
    rlo, rhi = float(r.min()), float(r.max())
    amax = max(abs(lo), abs(hi))  # NaN anywhere makes the tests below fail: general code
    if not amax < _INF:
        return None
    atol = _EPS * max(amax, 1.0) * 4  # the largest tolerance any element gets
    if (rhi < y and y - rhi > atol) or (rlo > y and rlo - y > atol):
        return r
    return None


def _mod_elem(x: float, y: float):
    """One element of the array formula for mod (finite x, finite y), or None."""
    if y == 0.0:
        return x
    q = x / y
    f = math.floor(q)
    f = float(f) if f else math.copysign(0.0, q)  # floor keeps the sign of a zero, like NumPy's
    r = x - f * y
    return 0.0 if abs(r - y) <= _EPS * max(abs(x), 1.0) * 4 else r


def _rem_elem(x: float, y: float):
    """One element of the array formula for rem (finite x, finite non-zero y)."""
    q = x / y
    t = math.trunc(q)
    t = float(t) if t else math.copysign(0.0, q)
    return x - t * y


def _modrem(name: str):
    elem = _mod_elem if name == "mod" else _rem_elem

    def f(args, nargout):
        if len(args) != 2:
            return None
        a, b = args
        if type(a) is not _ND or type(b) is not _ND or a.dtype is not _F64 or b.dtype is not _F64 \
                or b.shape != _S11:
            return None
        y = b.item()
        n = a.size
        if n <= TINY:
            if n == 0:
                return None
            one = _MOD_SCALAR[name]
            if a.shape == _S11:
                r = one(a.item(), y)
                return None if r is None else scalar(r)
            if y != y or y in (_INF, -_INF) or (y == 0.0 and name == "rem"):
                return None
            out = []
            for x in a.ravel().tolist():
                if x != x or x in (_INF, -_INF):
                    return None
                out.append(elem(x, y))
            return np.array(out, dtype=float).reshape(a.shape)
        if y == 0.0 or y != y or y in (_INF, -_INF):
            return None
        if name == "mod":
            return _mod_large(a, y)
        r = np.divide(a, y)  # rem: the general formula, reusing one temporary
        np.trunc(r, out=r)
        np.multiply(r, y, out=r)
        np.subtract(a, r, out=r)
        return r
    return f


# Element functions that are real for every real input: the general code calls the same ufunc.
_REAL_UFUNCS = {
    "floor": np.floor, "ceil": np.ceil, "fix": np.trunc, "exp": np.exp, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "expm1": np.expm1,
}
# Real only where the argument is not negative.
_NONNEG_UFUNCS = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
_SCALAR_FUNCS: dict[str, Callable] = {}   # name -> f(float) from mathfns, set by install()


def _elementwise(name: str):
    ufunc = _REAL_UFUNCS.get(name)
    guarded = _NONNEG_UFUNCS.get(name)

    def f(args, nargout):
        if len(args) != 1 or nargout > 1:
            return None
        a = args[0]
        if type(a) is not _ND or a.dtype is not _F64:
            return None
        if a.shape == _S11:
            one = _SCALAR_FUNCS.get(name)
            if one is None:
                return None
            r = one(a.item())
            return None if r is None else scalar(r)
        if ufunc is not None:
            return ufunc(a)
        n = a.size
        if n == 0:
            return None
        if (min(a.ravel().tolist()) if n <= 64 else a.min()) >= 0:  # (a NaN is not negative; if it
            return guarded(a)                                       # hides the minimum: general code)
        return None
    return f


# -- large arrays --------------------------------------------------------------------------------

def _restore_tie_order(flat: np.ndarray, sorted_flat: np.ndarray, descend: bool) -> None:
    """Make a value sort of doubles agree bit for bit with a stable sort.

    Equal doubles are indistinguishable except +0/-0, and NaNs can differ in sign or payload; a
    stable sort leaves each such group in its original order (the fast sort may even change how
    many zeros carry each sign), so those two groups are rewritten from the input.
    ``sorted_flat`` is ascending with NaNs last, or its reverse when ``descend``.
    """
    n = sorted_flat.size
    asc = sorted_flat[::-1] if descend else sorted_flat
    lo = int(np.searchsorted(asc, 0.0, side="left"))
    hi = int(np.searchsorted(asc, 0.0, side="right"))
    if hi - lo > 1:
        block = flat[flat == 0]
        if descend:
            sorted_flat[n - hi:n - lo] = block
        else:
            sorted_flat[lo:hi] = block
    if asc[n - 1] != asc[n - 1]:  # NaNs present (they sort last)
        first = int(np.searchsorted(asc, np.nan, side="left"))
        block = flat[np.isnan(flat)]
        if descend:
            sorted_flat[:n - first] = block
        else:
            sorted_flat[first:] = block


def _flag(v) -> str | None:
    """A char-vector option in lower case, or None."""
    if type(v) is _ND and v.dtype.kind == "U" and v.ndim == 2 and v.shape[0] == 1:
        return "".join(v[0].tolist()).lower()
    return None


def _f_sort(args, nargout):
    """``sort(v)`` / ``sort(v, 'descend')`` of a real vector when only the values are wanted."""
    if nargout > 1 or not 1 <= len(args) <= 2:
        return None
    a = args[0]
    if type(a) is not _ND or a.ndim != 2 or a.size < 2 or (a.shape[0] != 1 and a.shape[1] != 1):
        return None
    kind = a.dtype.kind
    if kind not in "fiub" or a.dtype.itemsize > 8:
        return None
    descend = False
    if len(args) == 2:
        flag = _flag(args[1])
        if flag == "descend":
            descend = True
        elif flag != "ascend":
            return None
    if a.size <= TINY and a.dtype is _F64:
        v = a.ravel().tolist()
        if not any(x != x for x in v):
            return np.array(sorted(v, reverse=descend)).reshape(a.shape)  # (stable in both directions)
    out = np.sort(a, axis=None)
    if descend:
        out = out[::-1].copy()
    if kind == "f":
        _restore_tie_order(a.reshape(-1), out, descend)
    return out.reshape(a.shape)


def _f_unique(args, nargout):
    """``unique(x)`` of a real numeric array when only the sorted values are wanted."""
    if nargout > 1 or len(args) != 1:
        return None
    a = args[0]
    if type(a) is not _ND or a.size < 2 or a.dtype.kind not in "fiu" or a.dtype.itemsize > 8:
        return None
    row = a.ndim == 2 and a.shape[0] == 1
    if a.size <= TINY and a.dtype is _F64 and (row or a.shape[1] == 1):
        v = sorted(a.ravel().tolist())
        if v[-1] == v[-1] and v[0] == v[0] and not any(x != x for x in v):
            u = [v[0]]
            for x in v:
                if x != u[-1]:
                    u.append(x)  # the first of equal values (so +0 or -0 as in the input)
            return np.array([u]) if row else np.array(u).reshape(-1, 1)
    col = a.reshape(-1, order="F")
    s = np.sort(col)
    keep = np.empty(s.size, dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    u = s[keep]
    if a.dtype.kind == "f":
        k = int(np.searchsorted(u, 0.0))
        if k < u.size and u[k] == 0:
            u[k] = col[col == 0][0]  # +0 or -0: whichever comes first in the input
        _restore_tie_order(col, u, False)
    return u.reshape(1, -1) if row else u.reshape(-1, 1)


def _f_ismember(args, nargout):
    """``ismember(a, b)`` for real double arrays: binary search instead of merging both sets."""
    if len(args) != 2 or nargout > 2:
        return None
    a, b = args
    if type(a) is not _ND or type(b) is not _ND or a.dtype is not _F64 or b.dtype is not _F64 \
            or b.size == 0 or a.size == 0:
        return None
    bv = b.reshape(-1, order="F")
    if nargout > 1:
        order = np.argsort(bv, kind="stable")  # equal values stay in input order: the first one is found
        bs = bv[order]
    else:
        order = None
        bs = np.sort(bv)
    idx = np.searchsorted(bs, a, side="left")
    np.minimum(idx, bs.size - 1, out=idx)
    hit = bs[idx] == a  # (never true for NaN)
    if order is None:
        return hit
    return hit, np.where(hit, order[idx] + 1.0, 0.0)


# -- matrix products of mostly-zero integer matrices ----------------------------------------------

SPARSE_PRODUCT_MIN = 1 << 26   # multiply-adds (m * k * n) below which the dense product is always used
_EXACT_SUM = 2.0 ** 52


def _integer_sample(a: np.ndarray, count: int = 1024):
    """(fraction of non-zeros, whether all are integers) in about ``count`` entries spread over ``a``."""
    step = max(1, a.size // count)
    s = a.flat[::step] if a.size > count else a.ravel()
    return float(np.count_nonzero(s)) / max(1, s.size), bool(np.all(s == np.floor(s)))


def _integer_bound(a: np.ndarray) -> float:
    """max |a| when every entry is a finite integer, else inf (NaN counts as not finite)."""
    m = max(float(a.max()), -float(a.min())) if a.size else 0.0
    if not m < _EXACT_SUM or not np.array_equal(a, np.floor(a)):
        return math.inf
    return m


def _sparse_integers(a: np.ndarray):
    """(SciPy CSR matrix, max |entry|) of a matrix that is at most 5% non-zeros, all finite integers; else None.

    Built straight from the non-zeros, which are also the only entries whose values need checking.
    """
    import scipy.sparse as sp
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    m, k = a.shape
    flat = a.reshape(-1)
    idx = np.flatnonzero(flat)  # (NaN counts as non-zero)
    if idx.size > 0.05 * flat.size:
        return None
    vals = flat[idx]
    top = max(float(vals.max()), -float(vals.min())) if vals.size else 0.0
    if not top < _EXACT_SUM or not np.array_equal(vals, np.floor(vals)):
        return None
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(idx // k, minlength=m), out=indptr[1:])
    return sp.csr_matrix((vals, idx % k, indptr), shape=(m, k)), top


def matmul_exact(a: np.ndarray, b: np.ndarray):
    """``a @ b`` for large real matrices of integers of which one is mostly zeros, as a sparse product.

    When every entry is an integer and max|a| * max|b| * k stays below 2^52, every product and every
    partial sum of every dot product is an exact integer, so the order of the additions cannot change
    a single bit: the sparse product is exactly the dense one (BLAS), only without the zeros.  Count
    and indicator matrices (histograms, adjacency, one-hot labels) are of this kind.  None otherwise.
    """
    m, k = a.shape
    n = b.shape[1]
    if m * k * n < SPARSE_PRODUCT_MIN or a.dtype is not _F64 or b.dtype is not _F64:
        return None
    da, ia = _integer_sample(a)
    db, ib = _integer_sample(b)
    if not (ia and ib) or min(da, db) > 0.05:
        return None
    # the sparser operand first: (b' a')' when that is b
    first, other = (a, b) if da <= db else (b.T, a.T)
    try:
        sf = _sparse_integers(first)
        if sf is None:
            return None
        sf, top_f = sf
        so = _sparse_integers(other) if max(da, db) <= 0.05 else None
        if so is not None:
            so, top_o = so
            if not top_f * top_o * k < _EXACT_SUM:
                return None
            r = (sf @ so).toarray()
        else:
            top_o = _integer_bound(other)
            if not top_f * top_o * k < _EXACT_SUM:
                return None
            r = sf @ other
    except ImportError:  # no SciPy: the dense product
        return None
    if first is not a:
        r = r.T
    r = np.ascontiguousarray(r, dtype=_F64)
    r += 0.0  # (a sparse product has no negative zeros either; this keeps the result writable and owned)
    return r


# -- FFT ------------------------------------------------------------------------------------------

_THREADS: int | None = None


def threads() -> int:
    """Number of cores this process may use (``MYLAB_THREADS`` overrides)."""
    global _THREADS
    if _THREADS is None:
        try:
            n = int(os.environ.get("MYLAB_THREADS", "0"))
        except ValueError:
            n = 0
        if n <= 0:
            try:
                n = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                n = os.cpu_count() or 1
        _THREADS = max(1, n)
    return _THREADS


_SCIPY_FFT: Any = None
FFT_THREADED_MIN = 1 << 18  # a single transform this long uses several cores


def _scipy_fft():
    global _SCIPY_FFT
    if _SCIPY_FFT is None:
        try:
            import scipy.fft as module
        except Exception:  # noqa: BLE001 - SciPy missing or broken (the browser build may lack it)
            module = False
        _SCIPY_FFT = module
    return _SCIPY_FFT or None


def fft(x: np.ndarray, n: int | None, axis: int, inverse: bool = False) -> np.ndarray:
    """Discrete Fourier transform along ``axis`` (the forward transform as complex double).

    Large transforms go to ``scipy.fft`` (real input is transformed as real, and several vectors
    are transformed on several cores); small ones stay with ``numpy.fft``.
    """
    size = x.size if n is None else (x.size // max(x.shape[axis], 1)) * n
    if ENABLED and size >= FFT_MIN:
        sf = _scipy_fft()
        if sf is not None:
            lanes = size // max(n if n is not None else x.shape[axis], 1)
            # several vectors are shared out between cores; one long vector also gains from several (SciPy
            # splits a large transform's work), as MATLAB's multithreaded fft does
            workers = min(threads(), lanes) if lanes > 1 else (threads() if size >= FFT_THREADED_MIN else 1)
            if lanes == 1 and workers > 1:
                # one vector: SciPy splits the work of a long transform between cores only for a 1-D array
                shape = list(x.shape)
                r = (sf.ifft if inverse else sf.fft)(x.reshape(-1), n=n, workers=workers)
                shape[axis] = r.size
                return r.reshape(shape).astype(complex, copy=False)
            if inverse:
                return sf.ifft(x, n=n, axis=axis, workers=workers)
            return sf.fft(x, n=n, axis=axis, workers=workers).astype(complex, copy=False)
    if inverse:
        return np.fft.ifft(x, n=n, axis=axis)
    return np.fft.fft(x, n=n, axis=axis).astype(complex, copy=False)


# -- LAPACK ---------------------------------------------------------------------------------------
# ``A \ b`` factorises and solves, then estimates the reciprocal condition number from the factors
# to decide whether to warn.  NumPy keeps the factors to itself, so the drivers are called
# directly: first choice is the OpenBLAS library NumPy itself is linked against (through ctypes:
# the general solve then gives exactly the bits numpy.linalg.solve gives, and SciPy need not be
# imported), second choice is scipy.linalg.lapack.

_LAPACK: Any = None


def _find_numpy_openblas() -> str | None:
    """Path of the OpenBLAS shared library bundled with NumPy wheels, if there is one."""
    candidates: list[str] = []
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    path = line.rsplit(" ", 1)[-1].strip()
                    if "scipy_openblas64_" in os.path.basename(path) and "numpy" in path:
                        candidates.append(path)
                        break
        except OSError:
            pass
    if not candidates:
        import glob
        site = os.path.dirname(os.path.dirname(os.path.abspath(np.__file__)))
        for pattern in ("numpy.libs/libscipy_openblas64_*.so*", "numpy/.dylibs/libscipy_openblas64_*.dylib",
                        "numpy.libs/libscipy_openblas64_*.dll"):
            candidates += sorted(glob.glob(os.path.join(site, pattern)))
    return candidates[0] if candidates else None


class _NumpyOpenBLAS:
    """LAPACK drivers from NumPy's own 64-bit-integer OpenBLAS."""

    name = "numpy-openblas"

    def __init__(self, path: str):
        import ctypes
        self.ct = ctypes
        lib = ctypes.CDLL(path)
        self.fn = {}
        for routine in ("dgesv", "zgesv", "dgecon", "zgecon", "dgetrs", "zgetrs", "dpotrf", "dpotrs", "dpocon",
                        "dtrtrs", "dtrcon"):
            f = getattr(lib, f"scipy_{routine}_64_")  # AttributeError: this back end is not usable
            f.restype = None
            self.fn[routine] = f
        for routine in ("dlange", "zlange"):
            f = getattr(lib, f"scipy_{routine}_64_")
            f.restype = ctypes.c_double
            self.fn[routine] = f
        self.one = ctypes.c_size_t(1)
        try:
            self.get_threads = lib.scipy_openblas_get_num_threads64_
            self.get_threads.restype = ctypes.c_int
            self.set_threads = lib.scipy_openblas_set_num_threads64_
            self.set_threads.argtypes = [ctypes.c_int]
            self.set_threads.restype = None
        except AttributeError:
            self.get_threads = self.set_threads = None

    def _ptr(self, arr):
        return self.ct.c_void_p(arr.ctypes.data)

    def _chr(self, c: bytes):
        return self.ct.byref(self.ct.c_char(c))

    def general(self, a: np.ndarray, b: np.ndarray):
        """LU solve: (x, info, rcond); info > 0 is an exactly zero pivot."""
        ct, ref = self.ct, self.ct.byref
        cplx = a.dtype.kind == "c" or b.dtype.kind == "c"
        dtype = _C128 if cplx else _F64
        p = "z" if cplx else "d"
        lu = np.array(a, dtype=dtype, order="F")      # copies: the factorisation overwrites its input
        x = np.array(b, dtype=dtype, order="F")
        n = lu.shape[0]
        c_n, c_nrhs, info = ct.c_int64(n), ct.c_int64(x.shape[1]), ct.c_int64(0)
        ipiv = np.empty(n, dtype=np.int64)
        work = np.empty(4 * n, dtype=dtype)
        a_ptr, x_ptr, ipiv_ptr = self._ptr(lu), self._ptr(x), self._ptr(ipiv)
        anorm = ct.c_double(self.fn[p + "lange"](self._chr(b"1"), ref(c_n), ref(c_n), a_ptr, ref(c_n),
                                                 self._ptr(work), self.one))
        # gesv is the routine numpy.linalg.solve calls: factorise and solve in one go.
        self.fn[p + "gesv"](ref(c_n), ref(c_nrhs), a_ptr, ref(c_n), ipiv_ptr, x_ptr, ref(c_n), ref(info))
        fact = info.value
        if fact < 0:
            raise ValueError("illegal argument to LAPACK gesv")
        if fact > 0:
            # gesv stops after the factorisation; do the triangular solves anyway, so the result
            # holds the Inf/NaN values that plain elimination gives.
            self.fn[p + "getrs"](self._chr(b"N"), ref(c_n), ref(c_nrhs), a_ptr, ref(c_n), ipiv_ptr, x_ptr,
                                 ref(c_n), ref(info), self.one)
            return x, fact, 0.0
        rcond = ct.c_double(0.0)
        aux = np.empty(2 * n if cplx else n, dtype=np.float64 if cplx else np.int64)
        self.fn[p + "gecon"](self._chr(b"1"), ref(c_n), a_ptr, ref(c_n), ref(anorm), ref(rcond),
                             self._ptr(work), self._ptr(aux), ref(info), self.one)
        return x, 0, rcond.value

    def cholesky(self, a: np.ndarray, b: np.ndarray):
        """Cholesky solve of a real symmetric matrix: (x, rcond), or None if not positive definite."""
        ct, ref = self.ct, self.ct.byref
        c = np.array(a, dtype=_F64, order="F")
        n = c.shape[0]
        c_n, c_nrhs, info = ct.c_int64(n), ct.c_int64(b.shape[1]), ct.c_int64(0)
        work = np.empty(3 * n)
        a_ptr = self._ptr(c)
        anorm = ct.c_double(self.fn["dlange"](self._chr(b"1"), ref(c_n), ref(c_n), a_ptr, ref(c_n),
                                              self._ptr(work), self.one))
        self.fn["dpotrf"](self._chr(b"U"), ref(c_n), a_ptr, ref(c_n), ref(info), self.one)
        if info.value != 0:
            return None
        x = np.array(b, dtype=_F64, order="F")
        self.fn["dpotrs"](self._chr(b"U"), ref(c_n), ref(c_nrhs), a_ptr, ref(c_n), self._ptr(x), ref(c_n),
                          ref(info), self.one)
        rcond = ct.c_double(0.0)
        iwork = np.empty(n, dtype=np.int64)
        self.fn["dpocon"](self._chr(b"U"), ref(c_n), a_ptr, ref(c_n), ref(anorm), ref(rcond), self._ptr(work),
                          self._ptr(iwork), ref(info), self.one)
        return x, rcond.value

    def triangular(self, a: np.ndarray, b: np.ndarray, lower: bool):
        """Substitution for a real triangular matrix with a non-zero diagonal: (x, rcond)."""
        ct, ref = self.ct, self.ct.byref
        t = np.asfortranarray(a, dtype=_F64)          # read only: no copy needed when already suitable
        x = np.array(b, dtype=_F64, order="F")
        n = t.shape[0]
        c_n, c_nrhs, info = ct.c_int64(n), ct.c_int64(x.shape[1]), ct.c_int64(0)
        uplo = self._chr(b"L" if lower else b"U")
        a_ptr = self._ptr(t)
        self.fn["dtrtrs"](uplo, self._chr(b"N"), self._chr(b"N"), ref(c_n), ref(c_nrhs), a_ptr, ref(c_n),
                          self._ptr(x), ref(c_n), ref(info), self.one, self.one, self.one)
        if info.value != 0:
            return None
        rcond = ct.c_double(0.0)
        work = np.empty(3 * n)
        iwork = np.empty(n, dtype=np.int64)
        self.fn["dtrcon"](self._chr(b"1"), uplo, self._chr(b"N"), ref(c_n), a_ptr, ref(c_n), ref(rcond),
                          self._ptr(work), self._ptr(iwork), ref(info), self.one, self.one, self.one)
        return x, rcond.value


class _ScipyLapack:
    """The same drivers through scipy.linalg.lapack."""

    name = "scipy"
    get_threads = set_threads = None

    def __init__(self):
        import scipy.linalg.lapack as lapack
        self.lapack = lapack

    def general(self, a: np.ndarray, b: np.ndarray):
        lapack = self.lapack
        cplx = a.dtype.kind == "c" or b.dtype.kind == "c"
        p = "z" if cplx else "d"
        dtype = _C128 if cplx else _F64
        a = np.asarray(a, dtype=dtype)
        b = np.asarray(b, dtype=dtype)
        anorm = float(np.abs(a).sum(axis=0).max())
        lu, piv, x, fact = getattr(lapack, p + "gesv")(a, b)
        if fact < 0:
            raise ValueError("illegal argument to LAPACK gesv")
        if fact > 0:
            x, _ = getattr(lapack, p + "getrs")(lu, piv, b)
            return x, int(fact), 0.0
        rcond, _ = getattr(lapack, p + "gecon")(lu, anorm, norm="1")
        return x, 0, float(rcond)

    def cholesky(self, a: np.ndarray, b: np.ndarray):
        lapack = self.lapack
        anorm = float(np.abs(a).sum(axis=0).max())
        c, info = lapack.dpotrf(a, lower=0, clean=0)
        if info != 0:
            return None
        x, info = lapack.dpotrs(c, b, lower=0)
        rcond, info = lapack.dpocon(c, anorm, uplo="U")
        return x, float(rcond)

    def triangular(self, a: np.ndarray, b: np.ndarray, lower: bool):
        lapack = self.lapack
        x, info = lapack.dtrtrs(a, b, lower=int(lower))
        if info != 0:
            return None
        from ._compat import triangular_rcond
        return x, triangular_rcond(a, lower)


def _lapack_self_test(be) -> bool:
    """True if the back end reproduces numpy.linalg.solve bit for bit on a few systems."""
    rng = np.random.RandomState(12345)
    for n, k in ((2, 1), (3, 1), (7, 2), (24, 1), (65, 3)):
        a = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal((n, k))
        x, fact, rcond = be.general(a, b)
        if fact != 0 or not (0.0 < rcond <= 1.0) or not np.array_equal(x, np.linalg.solve(a, b)):
            return False
        spd = a @ a.T
        res = be.cholesky(spd, b)
        if res is None or not np.allclose(res[0], np.linalg.solve(spd, b), rtol=1e-9, atol=0):
            return False
        tri = np.triu(a)
        res = be.triangular(tri, b, False)
        if res is None or not np.allclose(res[0], np.linalg.solve(tri, b), rtol=1e-9, atol=0):
            return False
    a = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    b = rng.standard_normal((5, 2)) + 0j
    x, fact, rcond = be.general(a, b)
    if fact != 0 or not np.array_equal(x, np.linalg.solve(a, b)):
        return False
    x, fact, _ = be.general(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([[1.0], [1.0]]))
    return fact > 0


def lapack_backend():
    """The LAPACK back end in use, or None when only NumPy is available."""
    global _LAPACK
    if _LAPACK is None:
        chosen: Any = False
        if os.environ.get("MYLAB_LAPACK", "") != "scipy":
            try:
                path = _find_numpy_openblas()
                if path is not None:
                    candidate = _NumpyOpenBLAS(path)
                    with np.errstate(all="ignore"):
                        if _lapack_self_test(candidate):
                            chosen = candidate
            except Exception:  # noqa: BLE001 - any problem: use the next back end
                chosen = False
        if chosen is False:
            try:
                chosen = _ScipyLapack()
            except Exception:  # noqa: BLE001 - SciPy missing or broken
                chosen = False
        _LAPACK = chosen
    return _LAPACK or None


def _rcond_warning(rcond: float, warn) -> None:
    if rcond != rcond:
        warn("Matrix is singular, close to singular or badly scaled. Results may be inaccurate. RCOND = NaN.",
             "MATLAB:illConditionedMatrix")
    elif rcond < _EPS:
        warn(f"Matrix is close to singular or badly scaled. Results may be inaccurate. RCOND =  {rcond:e}.",
             "MATLAB:nearlySingularMatrix")


def solve(a: np.ndarray, b: np.ndarray, warn: Callable[[str, str], None]):
    """``a \\ b`` for a square full matrix (n >= 2), or None without a LAPACK back end (see _solve)."""
    if a.shape[0] > SMALL_SOLVE:
        return _solve(a, b, warn)
    saved = one_thread()
    try:
        return _solve(a, b, warn)
    finally:
        restore_threads(saved)


def _solve(a: np.ndarray, b: np.ndarray, warn: Callable[[str, str], None]):
    """``a \\ b`` for a square full matrix (n >= 2), or None when no LAPACK back end is available.

    The driver follows the structure of ``a``: substitution when it is exactly triangular,
    Cholesky when it is exactly symmetric with a positive diagonal and the factorisation
    succeeds, LU with partial pivoting otherwise.  The warnings are MATLAB's: an exactly zero
    pivot is "singular to working precision", a reciprocal condition estimate below eps is
    "close to singular or badly scaled".
    """
    be = lapack_backend()
    if be is None:
        return None
    n = a.shape[0]
    with np.errstate(all="ignore"):
        if a.dtype is _F64 and b.dtype is _F64:
            res = None
            lower_zero = a[1, 0] == 0.0 and a[n - 1, 0] == 0.0
            upper_zero = a[0, 1] == 0.0 and a[0, n - 1] == 0.0
            if (lower_zero or upper_zero) and a.diagonal().all():  # (a NaN on the diagonal counts as non-zero)
                if lower_zero and not np.tril(a, -1).any():
                    res = be.triangular(a, b, False)
                elif upper_zero and not np.triu(a, 1).any():
                    res = be.triangular(a, b, True)
            if res is None and a[0, 1] == a[1, 0] and a[0, n - 1] == a[n - 1, 0] \
                    and (a.diagonal() > 0).all() and np.array_equal(a, a.T):
                res = be.cholesky(a, b)
            if res is not None:
                x, rcond = res
                _rcond_warning(rcond, warn)
                x += 0.0  # no negative zeros from the solver
                return x
        x, info, rcond = be.general(a, b)
        if info > 0:
            warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
        else:
            _rcond_warning(rcond, warn)
        x += 0.0
    return x


def _blas_threads(n: int):
    """Limit NumPy's OpenBLAS to ``n`` threads; returns the previous count (None if not possible)."""
    be = lapack_backend()
    if be is None or be.set_threads is None:
        return None
    old = be.get_threads()
    if old != n:
        be.set_threads(n)
    return old


# -- one BLAS thread for small factorisations ------------------------------------------------------
# NumPy and SciPy each ship an OpenBLAS with its own pool of threads. Factorisations and matrix functions
# of small and medium matrices gain little from several threads, and when the cores are shared with other
# work (parallel runs, other processes) waking the pools costs far more than the arithmetic: a 3x3 expm
# took 20 ms instead of 5 us on a loaded machine, a 1000x1000 svd 36 s instead of 0.8 s. Such calls run
# on one thread, as the values-only eig and svd above already do.
#
# Up to about 50 rows OpenBLAS computes the same bits on one thread as on several; beyond that its
# threaded factorisations order some sums differently (last-bit differences, as between machines with
# different core counts). LU and Cholesky based functions therefore stay with the default threads above
# that size, so that, e.g., A\b is exactly what numpy.linalg.solve gives; the eigenvalue, singular value and
# matrix function (expm ...) drivers use one thread up to SMALL_FACTOR, like the values-only ones.

_THREAD_CONTROLS: list = []
_THREADS_LOOKED: set = set()
SMALL_FACTOR = 2000   # largest dimension handled on one thread by svd, eig, expm ...
SMALL_SOLVE = 48      # ... and by LU, QR and Cholesky based functions (where one thread changes no bits)


def _add_control(key: str, find: Callable) -> None:
    if key in _THREADS_LOOKED:
        return
    _THREADS_LOOKED.add(key)
    try:
        pair = find()
    except Exception:  # noqa: BLE001 - a library without these entry points: leave its threads alone
        pair = None
    if pair is not None:
        count = pair[0]()  # read once: nothing else in mylab changes it for longer than a call
        if count > 1:
            _THREAD_CONTROLS.append((pair[1], count))


def _numpy_threads():
    be = lapack_backend()
    if be is None or be.set_threads is None:
        return None
    return be.get_threads, be.set_threads


def _scipy_threads():
    import ctypes
    import glob
    import importlib.util
    spec = importlib.util.find_spec("scipy")
    if spec is None or not spec.origin:
        return None
    site = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    paths: list[str] = []
    for pattern in ("scipy.libs/libscipy_openblas*.so*", "scipy/.dylibs/libscipy_openblas*.dylib",
                    "scipy.libs/libscipy_openblas*.dll"):
        paths += sorted(glob.glob(os.path.join(site, pattern)))
    if not paths:
        return None
    lib = ctypes.CDLL(paths[0])
    get, set_ = lib.scipy_openblas_get_num_threads, lib.scipy_openblas_set_num_threads
    get.restype = ctypes.c_int
    set_.argtypes = [ctypes.c_int]
    set_.restype = None
    return get, set_


def one_thread() -> list:
    """Run the BLAS libraries on one thread; returns what restore_threads needs."""
    _add_control("numpy", _numpy_threads)
    if "scipy.linalg" in sys.modules:  # (before SciPy's linear algebra is loaded its threads are idle)
        _add_control("scipy", _scipy_threads)
    for set_, _count in _THREAD_CONTROLS:
        set_(1)
    return _THREAD_CONTROLS


def restore_threads(saved: list) -> None:
    for set_, n in saved:
        set_(n)


def _small(args, limit: int) -> bool:
    for a in args:
        if type(a) is _ND and a.ndim and max(a.shape) > limit:
            return False
    return True


def _on_one_thread(slow: Callable, limit: int) -> Callable:
    def f(interp, args, nargout):
        if not _small(args, limit):
            return slow(interp, args, nargout)
        saved = one_thread()
        try:
            return slow(interp, args, nargout)
        finally:
            restore_threads(saved)

    f.__name__ = getattr(slow, "__name__", "builtin")
    f.__doc__ = getattr(slow, "__doc__", None)
    f.__wrapped__ = slow  # type: ignore[attr-defined]
    f.one_thread = True  # type: ignore[attr-defined]
    names = getattr(slow, "matlab_names", None)
    if names is not None:
        f.matlab_names = names  # type: ignore[attr-defined]
    return f


_ONE_THREAD = {
    **dict.fromkeys("svd eig eigs expm logm sqrtm funm schur hess qz gsvd balance null orth pinv rank cond "
                    "svds ordschur ordeig".split(), SMALL_FACTOR),
    **dict.fromkeys("inv det lu qr chol ldl linsolve lsqminnorm rcond condest mldivide mrdivide decomposition"
                    .split(), SMALL_SOLVE),
}


def _close_to_symmetric(a: np.ndarray) -> bool:
    """``np.allclose(a, a.T)`` (the test the general eig uses), rejecting most matrices cheaply."""
    if not np.allclose(a[0, :], a[:, 0]):
        return False
    return bool(np.allclose(a, a.T))


def _f_eig(args, nargout):
    """``eig(A)`` of a real square matrix when only the eigenvalues are wanted."""
    if nargout > 1 or len(args) != 1:
        return None
    a = args[0]
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        return None
    # These drivers gain nothing from several threads at ordinary sizes and lose badly when the
    # cores are shared, so they run on one.
    old = _blas_threads(1) if a.shape[0] <= 2000 else None
    try:
        w = np.linalg.eigvalsh(a) if _close_to_symmetric(a) else np.linalg.eigvals(a)
    finally:
        if old is not None and old != 1:
            _blas_threads(old)
    if w.dtype.kind == "c" and not np.any(w.imag):
        w = w.real.copy()
    return w.reshape(-1, 1)


def _f_svd(args, nargout):
    """``svd(A)`` of a real matrix when only the singular values are wanted."""
    if nargout > 1 or len(args) != 1:
        return None
    a = args[0]
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2 or min(a.shape) < 2:
        return None
    old = _blas_threads(1) if max(a.shape) <= 2000 else None
    try:
        s = np.linalg.svd(a, compute_uv=False)
    finally:
        if old is not None and old != 1:
            _blas_threads(old)
    return s.reshape(-1, 1)


# -- installation --------------------------------------------------------------------------------

_FAST: dict[str, Callable] = {
    "sum": _f_sum, "prod": _f_prod, "cumsum": _f_cumsum, "max": _minmax(True), "min": _minmax(False),
    "any": _anyall(True), "all": _anyall(False), "abs": _f_abs, "round": _f_round,
    "numel": _f_numel, "length": _f_length, "size": _f_size, "norm": _f_norm, "dot": _f_dot,
    "mod": _modrem("mod"), "rem": _modrem("rem"),
    "sort": _f_sort, "unique": _f_unique, "ismember": _f_ismember, "eig": _f_eig, "svd": _f_svd,
}
for _name in list(_REAL_UFUNCS) + list(_NONNEG_UFUNCS):
    _FAST[_name] = _elementwise(_name)

_FALLBACK = (ValueError, TypeError, OverflowError, ZeroDivisionError, IndexError)


def _wrap(fast: Callable, slow: Callable) -> Callable:
    def f(interp, args, nargout):
        if ENABLED:
            try:
                r = fast(args, nargout)
            except _FALLBACK:
                r = None
            if r is not None:
                return r
        return slow(interp, args, nargout)

    f.__name__ = getattr(slow, "__name__", "builtin")
    f.__doc__ = getattr(slow, "__doc__", None)
    f.__wrapped__ = slow  # type: ignore[attr-defined]
    names = getattr(slow, "matlab_names", None)
    if names is not None:
        f.matlab_names = names  # type: ignore[attr-defined]
    return f


def install(builtins: dict) -> None:
    """Put the shortcuts in front of the general built-ins (called once, after all are registered)."""
    from .builtins import mathfns
    _MOD_SCALAR["mod"] = mathfns._mod_scalar
    _MOD_SCALAR["rem"] = mathfns._rem_scalar
    _SCALAR_FUNCS.update(mathfns._SCALAR)
    for name, fast in _FAST.items():
        slow = builtins.get(name)
        if slow is None or getattr(slow, "__wrapped__", None) is _SLOW.get(name) is not None:
            continue
        _SLOW[name] = slow
        builtins[name] = _wrap(fast, slow)
    for name, limit in _ONE_THREAD.items():
        fn = builtins.get(name)
        if fn is not None and not getattr(fn, "one_thread", False):
            builtins[name] = _on_one_thread(fn, limit)
