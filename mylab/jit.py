"""Loop and function compiler: runs simple loops and numeric functions as plain Python.

Most of the time in loop-heavy scripts goes to interpreter overhead: every scalar is a 1x1 array
and every statement walks the syntax tree.  This module generates an equivalent Python function
that works on raw floats where it can, which is typically 20-60x faster.

What is compiled
----------------
* ``for k = a:s:b`` and ``while`` loops whose body is made of ``x = <expr>;``, ``A(i) = <expr>;``,
  ``A(i, j) = <expr>;``, ``s.field = <expr>;``, ``s.field(i) = <expr>;``, ``c{i} = <expr>;``,
  ``c{end + 1} = <expr>;``, ``if/elseif/else``, ``break`` and ``continue``.
* Whole functions (one output, no nested functions) built from the same statements plus nested
  ``for``/``while`` loops and ``return`` - provided they are *pure*: they only compute a result
  from their inputs.  They are compiled when called from a compiled loop, from each other
  (recursion included), or for the second time from the interpreter.
* Expressions built from real scalars, element reads of double arrays, arithmetic, comparisons,
  logical operators, a fixed list of side-effect-free math functions, small matrix literals,
  text literals, cell and struct-field reads, calls to compiled user functions and calls to a
  list of built-ins that are known to have no side effects (``sum``, ``numel``, ``sprintf``,
  ``strrep``, ``strsplit``, ``str2double`` ...).

Values come in two forms.  Real scalars are Python floats and bools; everything else stays a
normal MATLAB value ("boxed") and is handled by the same routines the interpreter uses, so the
results are the interpreter's results.

Safety model
------------
* Variable kinds (double scalar, logical scalar, double array and its orientation, row cell,
  scalar struct and the kinds of its fields, other) are checked on entry; the compiled code is
  specialised for them.  The functions a loop calls are part of that check: if a file changes or
  a new file shadows a name, the loop is compiled again.
* Anything unusual at run time (an index out of range or not an integer, a result that would be
  complex, division by zero, NaN used as a condition, a value that is not the scalar the code
  expected, an error inside a built-in or a called function, too deep a recursion, a Python error
  of any kind) makes the compiled code *bail out*: the variables are written back and the
  interpreter resumes at exactly the statement that bailed, so the user sees the normal result or
  the normal error message.  Bail-outs can only happen while evaluating an expression, before the
  statement has changed anything, so resuming is exact.  A compiled function that bails is simply
  run again by the interpreter: it is pure, so nothing has to be undone.

Set the environment variable ``MYLAB_JIT=0`` (or ``interp.jit_enabled = False``) to turn the
compiler off.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

from . import ast as A
from . import fastnum
from . import indexing as ix
from . import ops
from ._refcount import CALLER_LOCAL as _RC_CALLER_LOCAL
from ._refcount import DICT_AND_LOCAL as _RC_DICT_AND_LOCAL
from ._refcount import ELEMENT as _RC_ELEMENT
from .errors import BreakSignal, ContinueSignal
from .values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_true, to_matlab, to_numeric

_F64 = np.dtype(float)
_BOOL = np.dtype(bool)
_ND = np.ndarray
_S11 = (1, 1)
MIN_ITERATIONS = 4
MAX_BAILS = 3
FUNCTION_CALLS_BEFORE_COMPILE = 2
_PLAIN = (np.ndarray, Cell, Struct, MString, StringArray, FunctionHandle)


class Bail(Exception):
    """Raised inside compiled code to hand control back to the interpreter."""


class Reject(Exception):
    """The loop or function cannot be compiled."""


class _Unset:
    """Stands for a variable without a value; any use of it bails out."""

    __slots__ = ()

    def _bail(self, *args):
        raise Bail

    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _bail
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _bail
    __pow__ = __rpow__ = __neg__ = __pos__ = __abs__ = __float__ = __bool__ = __int__ = _bail
    __hash__ = None  # type: ignore[assignment]


_UNSET = _Unset()
_MISSING = object()


# -- run-time helpers used by generated code ---------------------------------------------------

def _ix(i, n):
    k = int(i)
    if k != i or k < 1 or k > n:
        raise Bail
    return k - 1


def _lin(i, n, rows):
    k = int(i)
    if k != i or k < 1 or k > n:
        raise Bail
    k -= 1
    return (k % rows, k // rows)


def _tr(x):
    if x != x:
        raise Bail  # NaN cannot be converted to logical: let the interpreter report it
    return x != 0.0


def _pow(a, b):
    if a < 0.0 and b != int(b):
        raise Bail  # complex result
    r = a ** b
    if type(r) is not float:
        raise Bail
    return r


def _round(x):
    if x != x or x in (math.inf, -math.inf):
        return x
    m = abs(x)
    low = float(math.floor(m))  # (not floor(|x| + 0.5): that rounds 0.49999999999999994 up)
    return math.copysign(low + 1.0 if m - low >= 0.5 else low, x)  # the sign of zero is kept


# Element functions of a real scalar.  They call the same NumPy routines as the interpreter (on a
# Python float NumPy runs the very loop it uses for arrays), not the math module: the C library's
# exp, log, tan, sinh ... can differ from NumPy's in the last bit, and a loop must give exactly
# the interpreter's answer.  sqrt is correctly rounded everywhere.
def _np1(ufunc, nonneg=False):
    if nonneg:
        def f(x):
            if x < 0.0:
                raise Bail  # a complex result
            return float(ufunc(x))
    else:
        def f(x):
            return float(ufunc(x))
    return f


def _np2(ufunc):
    def f(x, y):
        return float(ufunc(x, y))
    return f


_ELEMENT_FUNCS = {
    "_sin": _np1(np.sin), "_cos": _np1(np.cos), "_tan": _np1(np.tan), "_atan": _np1(np.arctan),
    "_sinh": _np1(np.sinh), "_cosh": _np1(np.cosh), "_tanh": _np1(np.tanh), "_exp": _np1(np.exp),
    "_log": _np1(np.log, True), "_log2": _np1(np.log2, True), "_log10": _np1(np.log10, True),
    "_atan2": _np2(np.arctan2), "_hypot": _np2(np.hypot),
}


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else (0.0 if x == 0 else x))


def _min2(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


def _max2(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def _mod(a, b):
    r = _MOD_SCALAR(a, b)
    if r is None:
        raise Bail
    return float(r)


def _rem(a, b):
    r = _REM_SCALAR(a, b)
    if r is None:
        raise Bail
    return float(r)


def _late_mod(a, b):
    global _MOD_SCALAR, _REM_SCALAR
    from .builtins.mathfns import _mod_scalar, _rem_scalar
    _MOD_SCALAR, _REM_SCALAR = _mod_scalar, _rem_scalar
    return _mod_scalar(a, b)


def _late_rem(a, b):
    _late_mod(1.0, 1.0)
    return _REM_SCALAR(a, b)


_MOD_SCALAR, _REM_SCALAR = _late_mod, _late_rem


def _floor(x):
    if x != x or x in (math.inf, -math.inf):
        return x
    return math.copysign(math.floor(x), x)  # floor(-0) and floor of -0.5 .. -0 keep their sign


def _ceil(x):
    if x != x or x in (math.inf, -math.inf):
        return x
    return math.copysign(math.ceil(x), x)  # ceil(-0.5) is -0


def _fix(x):
    if x != x or x in (math.inf, -math.inf):
        return x
    return math.copysign(math.trunc(x), x)


def _d(v):
    """A boxed value that must be a real scalar, as a float (logical scalars count as 0 or 1)."""
    if type(v) is _ND and v.shape == _S11 and (v.dtype is _F64 or v.dtype is _BOOL):
        return float(v.item())
    raise Bail


def _df(v):
    """A built-in's result that must be a real double scalar (a logical one would change class)."""
    if type(v) is _ND and v.shape == _S11 and v.dtype is _F64:
        return v.item()
    raise Bail


def _bd(x):
    r = np.empty(_S11)
    r[0, 0] = x
    return r


def _bb(x):
    r = np.empty(_S11, _BOOL)
    r[0, 0] = x
    return r


def _plain(v):
    if type(v) not in _PLAIN:
        raise Bail  # objects dispatch to their own methods: the interpreter handles them
    return v


def _truth(v):
    return is_true(_plain(v))


def _sc(v):
    """Operand of && or ||, as the interpreter evaluates it."""
    a = to_numeric(_plain(v))
    if a.size != 1:
        raise Bail
    return bool(np.real(a.flat[0]) != 0)


def _binop(op, a, b):
    if type(a) is not _ND or type(b) is not _ND:
        _plain(a)
        _plain(b)
    return ops.binop(op, a, b)


def _unary(op, v):
    return ops.unary(op, _plain(v))


def _transpose(v, conjugate):
    return ops.transpose(_plain(v), conjugate)


def _arr(rows):
    return np.array(rows)


def _build(rows):
    for row in rows:
        for v in row:
            if type(v) not in _PLAIN:
                raise Bail
    return ops.build_matrix(rows)


def _get1(a, i):
    """``a(i)`` for a real double array held as a boxed value."""
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2:
        raise Bail
    k = int(i)
    if k != i or k < 1 or k > a.size:
        raise Bail
    rows = a.shape[0]
    if rows == 1 or a.shape[1] == 1:
        return a.item(k - 1)
    k -= 1
    return a.item(k % rows, k // rows)


def _get2(a, i, j):
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2:
        raise Bail
    r, c = int(i), int(j)
    if r != i or c != j or r < 1 or c < 1 or r > a.shape[0] or c > a.shape[1]:
        raise Bail
    return a.item(r - 1, c - 1)


def _set1(a, i, x):
    """``a(i) = x`` in place, for a local real double array that nothing else refers to."""
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2 or not a.flags.writeable or a.base is not None \
            or sys.getrefcount(a) > _RC_CALLER_LOCAL:
        raise Bail
    k = int(i)
    if k != i or k < 1 or k > a.size:
        raise Bail
    rows = a.shape[0]
    k -= 1
    if rows == 1:
        a[0, k] = x
    else:
        a[k % rows, k // rows] = x


def _set2(a, i, j, x):
    if type(a) is not _ND or a.dtype is not _F64 or a.ndim != 2 or not a.flags.writeable or a.base is not None \
            or sys.getrefcount(a) > _RC_CALLER_LOCAL:
        raise Bail
    r, c = int(i), int(j)
    if r != i or c != j or r < 1 or c < 1 or r > a.shape[0] or c > a.shape[1]:
        raise Bail
    a[r - 1, c - 1] = x


def _copy(v):
    return v.copy() if type(v) is _ND else v


def _cellget(c, i):
    """``c{i}`` for a cell held as a boxed value (a vector, or linear index into a matrix)."""
    if type(c) is not Cell:
        raise Bail
    data = c.data
    k = int(i)
    if k != i or k < 1 or k > data.size or data.ndim != 2:
        raise Bail
    rows = data.shape[0]
    k -= 1
    return data[k % rows, k // rows] if rows != 1 else data[0, k]


def _lset(lst, i, v):
    """``c{i} = v`` on the list form of a row cell: inside it, or one past the end."""
    n = len(lst)
    k = int(i)
    if k != i or k < 1 or k > n + 1:
        raise Bail
    if k > n:
        lst.append(v)
    else:
        lst[k - 1] = v


def _field(s, name):
    if type(s) is not Struct or s.data.size != 1 or name not in s.fields:
        raise Bail
    return s.data.flat[0][name]


def _numelf(v):
    if type(v) is _ND:
        return float(v.size)
    if type(v) in (Cell, Struct, StringArray):
        return float(v.data.size)
    raise Bail


def _endf(v, pos, count):
    if type(v) is _ND:
        return float(ix.end_value(v.shape, pos, count))
    if type(v) in (Cell, Struct, StringArray):
        return float(ix.end_value(v.data.shape, pos, count))
    raise Bail


def _index(v, args):
    """``v(args)`` for a plain value, as the interpreter evaluates it."""
    from .interpreter import _scalar_index, index_value
    if type(v) is _ND:
        pos = _scalar_index(v, args)
        if pos is not None:
            return v[pos].reshape(1, 1).copy()
        return index_value(v, args)
    if type(v) in (Cell, Struct, MString, StringArray):
        return index_value(v, args)
    raise Bail


def _range(a, s, b):
    from .interpreter import colon_range
    return colon_range(_bd(a), None if s is None else _bd(s), _bd(b))


def _count(a, s, b):
    """Number of iterations of ``for k = a:s:b`` (the interpreter's formula)."""
    if s == 0 or a != a or b != b or s != s or abs(a) == math.inf or abs(b) == math.inf:
        raise Bail
    span = (b - a) / s
    if span < 0:
        return 0
    return int(math.floor(span + 1e-10 * max(1.0, abs(span)))) + 1


# -- integer classes ----------------------------------------------------------------------------
# A scalar of an integer class is held as a Python int, under a one-character type code ("1" for
# int8 ... "8" for uint64).  Python ints are exact, so the only rounding is the one the general code
# does: arithmetic with a double happens in double precision, then rounds halves away from zero and
# saturates (saturate_int).  For 8- to 32-bit classes every int is exact in a double, so Python's
# own operators on ints, bools and floats compute exactly what NumPy computes on the arrays.  The
# 64-bit classes are computed exactly with Python ints where that is provably what the general code
# gives (integer operands); everything else goes through the operators' general code (_ig).

_INT_NAMES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
_ICODE = {name: str(k + 1) for k, name in enumerate(_INT_NAMES)}     # class name -> type code
_IDT = {code: np.dtype(name) for name, code in _ICODE.items()}       # type code -> dtype
_DT_CODE = {dt: code for code, dt in _IDT.items()}                    # dtype -> type code
_ILIM = {code: (int(np.iinfo(dt).min), int(np.iinfo(dt).max)) for code, dt in _IDT.items()}
_WIDE = frozenset(code for code, dt in _IDT.items() if dt.itemsize == 8)
_EXACT_DOUBLE = 2 ** 53   # every integer up to this magnitude is a double


def _int_helpers(code: str) -> dict:
    lo, hi = _ILIM[code]
    dt = _IDT[code]

    def sat(x):
        """A double (or an exact int) as this class: NaN is 0, halves round away from zero, saturated."""
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

    def box(x):
        r = np.empty(_S11, dt)
        r[0, 0] = x
        return r

    def neg(x):
        return hi if x == lo else -x

    def unbox(v):
        """A built-in's result that must be a scalar of this class."""
        if type(v) is _ND and v.shape == _S11 and v.dtype == dt:
            return v.item()
        raise Bail

    def clamp(x):
        return lo if x < lo else (hi if x > hi else x)

    def operand(v):
        """A boxed scalar next to a value of this class: a double or logical as a float, this class as an int."""
        if type(v) is _ND and v.shape == _S11:
            t = v.dtype
            if t is _F64 or t is _BOOL:
                return float(v.item())
            if t == dt:
                return v.item()
        raise Bail

    out = {f"_s{code}": sat, f"_b{code}": box, f"_n{code}": neg, f"_u{code}": unbox, f"_c{code}": clamp,
           f"_q{code}": operand}
    if code in _WIDE:
        def wide(op, x, y):
            """``x op y`` (+, - or *) for this 64-bit class and an int, bool or double operand."""
            for v in (x, y):
                if type(v) is float and not v.is_integer():
                    # a fraction: rounding in the wide type may differ from exact arithmetic
                    return _ig(op, _ibox(x, code), _ibox(y, code), code)
            a, b = int(x), int(y)
            r = a + b if op == "+" else (a - b if op == "-" else a * b)
            return lo if r < lo else (hi if r > hi else r)
        out[f"_w{code}"] = wide
    return out


def _ibox(x, code):
    """A Python value used by integer arithmetic as a MATLAB value (int: of class code)."""
    t = type(x)
    if t is bool:
        return _bb(x)
    if t is float:
        return _bd(x)
    if t is int:
        return _NS["_b" + code](x)
    raise Bail


def _ig(op, x, y, code):
    """``x op y`` on boxed operands through the operators' general code: a scalar of class code."""
    r = ops.binop(op, x, y)
    if type(r) is _ND and r.shape == _S11 and r.dtype == _IDT[code]:
        return r.item()
    raise Bail


def _dw(x):
    """double() of a 64-bit integer: exact up to 2^53; beyond that the interpreter converts it."""
    if -_EXACT_DOUBLE <= x <= _EXACT_DOUBLE:
        return float(x)
    raise Bail


def _exact_lit(node) -> int | None:
    """The exact value of an integer literal too large for a double (optionally negated), else None."""
    if type(node) is A.UnaryOp and node.op == "-":
        v = _exact_lit(node.operand)
        return None if v is None else -v
    if type(node) is A.Number and isinstance(node.value, A.ExactInt):
        return node.value.exact
    return None


# Built-ins that keep the integer class of a scalar argument (their result is checked at run time).
_INT_KEEPING = frozenset("abs sign floor ceil round fix mod rem min max idivide bitand bitor bitxor".split())


_INTERP: list = [None]   # the interpreter running compiled code (set on every entry)


def _callb(fn, args):
    """Call a side-effect-free built-in with one output, as ``Interpreter.call_builtin`` does."""
    for a in args:
        if type(a) not in _PLAIN:
            raise Bail  # an object argument may overload the function
    r = fn(_INTERP[0], args, 1)
    if type(r) is _ND and r.ndim == 2:
        return r
    if r is None:
        raise Bail
    if type(r) is tuple:
        r = r[0]
    return to_matlab(r)


def _sumf(v):
    lst = fastnum._tiny_vec(v) if fastnum.ENABLED else None
    if lst is not None and (fastnum._CHECKED.get("sum") or fastnum._verified("sum")):
        return fastnum._sum_list(lst)
    return _d(_callb(_NS["_bi_sum"], [v]))


def _prodf(v):
    lst = fastnum._tiny_vec(v) if fastnum.ENABLED else None
    if lst is not None and (fastnum._CHECKED.get("prod") or fastnum._verified("prod")):
        return fastnum._prod_list(lst)
    return _d(_callb(_NS["_bi_prod"], [v]))


def _extreme(name, pick):
    def f(v):
        lst = fastnum._tiny_vec(v) if fastnum.ENABLED else None
        if lst is not None:
            if len(lst) == 1:
                return lst[0]
            r = pick(lst)
            if r != 0.0 and r == r:
                for x in lst:
                    if x != x:
                        break
                else:
                    return r
        return _d(_callb(_NS["_bi_" + name], [v]))
    return f


def _normf(v):
    if fastnum.ENABLED and type(v) is _ND and v.dtype is _F64 and v.ndim == 2 and v.size \
            and (v.shape[0] == 1 or v.shape[1] == 1) and (fastnum._CHECKED.get("norm") or fastnum._verified("norm")):
        flat = v.ravel()
        return math.sqrt(flat.dot(flat))
    return _d(_callb(_NS["_bi_norm"], [v]))


_FLOAT_CALLS = {"sum": "_sumf", "prod": "_prodf", "max": "_maxf", "min": "_minf", "norm": "_normf"}

# The one namespace all generated code lives in: helpers, constants and compiled functions.
_NS: dict[str, Any] = {
    "_ix": _ix, "_lin": _lin, "_tr": _tr, "_pow": _pow, "_round": _round, "_sign": _sign,
    "_min2": _min2, "_max2": _max2, "_mod": _mod, "_rem": _rem, "_floor": _floor, "_ceil": _ceil,
    "_fix": _fix, "_math": math, "_UNSET": _UNSET, "_Bail": Bail, "_inf": math.inf, "_nan": math.nan,
    "_d": _d, "_df": _df, "_bd": _bd, "_bb": _bb, "_truth": _truth, "_sc": _sc, "_binop": _binop, "_unary": _unary,
    "_transpose": _transpose, "_arr": _arr, "_build": _build, "_get1": _get1, "_get2": _get2,
    "_set1": _set1, "_set2": _set2, "_copy": _copy, "_cellget": _cellget, "_lset": _lset, "_field": _field,
    "_numelf": _numelf, "_endf": _endf, "_index": _index, "_range": _range, "_count": _count,
    "_callb": _callb, "_sumf": _sumf, "_prodf": _prodf, "_maxf": _extreme("max", max),
    "_minf": _extreme("min", min), "_normf": _normf, "_COLON": ix.COLON, "_lim": [500],
    "_EMPTY": np.zeros((0, 0)),
}
_NS.update(_ELEMENT_FUNCS)
_NS["_EMPTY"].setflags(write=False)
for _code in _IDT:
    _NS.update(_int_helpers(_code))
_NS.update({"_ig": _ig, "_dw": _dw})
_CONSTS: dict = {}
_counter = [0]


def _unique(prefix: str) -> str:
    _counter[0] += 1
    return f"{prefix}{_counter[0]}"


# MATLAB function -> (number of arguments, Python expression template)
_FUNCS = {
    "sin": (1, "_sin({0})"), "cos": (1, "_cos({0})"), "tan": (1, "_tan({0})"),
    "atan": (1, "_atan({0})"), "sinh": (1, "_sinh({0})"), "cosh": (1, "_cosh({0})"),
    "tanh": (1, "_tanh({0})"), "exp": (1, "_exp({0})"), "sqrt": (1, "_math.sqrt({0})"),
    "log": (1, "_log({0})"), "log2": (1, "_log2({0})"), "log10": (1, "_log10({0})"),
    "abs": (1, "abs({0})"), "floor": (1, "_floor({0})"), "ceil": (1, "_ceil({0})"), "fix": (1, "_fix({0})"),
    "round": (1, "_round({0})"), "sign": (1, "_sign({0})"), "mod": (2, "_mod({0}, {1})"),
    "rem": (2, "_rem({0}, {1})"), "min": (2, "_min2({0}, {1})"), "max": (2, "_max2({0}, {1})"),
    "atan2": (2, "_atan2({0}, {1})"), "hypot": (2, "_hypot({0}, {1})"),
}
_CONSTANTS = {"pi": ("_math.pi", "d"), "Inf": ("_inf", "d"), "inf": ("_inf", "d"), "NaN": ("_nan", "d"),
              "nan": ("_nan", "d"), "eps": (repr(sys.float_info.epsilon), "d"), "true": ("True", "b"),
              "false": ("False", "b")}
_ARITH = {"+": "+", "-": "-", "*": "*", ".*": "*", "/": "/", "./": "/"}
_COMPARE = {"==": "==", "~=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
_GENERIC_OPS = frozenset(_ARITH) | frozenset(_COMPARE) | {"^", ".^", "&", "|", "\\", ".\\"}

# Built-ins that only compute a result from their arguments: no output, no files, no random
# numbers, no look at the workspace or the clock, and they do not call back into user code.
_PURE = frozenset("""
abs acos acosd acosh all angle any asin asind asinh atan atan2 atand atanh bitand bitor bitshift bitxor
blanks cat ceil cell cell2mat char circshift colon columns complex conj contains cos cosd cosh
cot cross cummax cummin cumprod cumsum deblank deg2rad det diag diff dot double endsWith eps erf
erfc exp expm1 eye factorial false fieldnames find fix fliplr flipud floor gamma gcd horzcat hypot idivide
imag int16 int2str int32 int64 int8 isa iscell iscellstr ischar iscolumn isempty isequal isfield isfinite
isfloat isinf isinteger islogical ismatrix ismember isnan isnumeric isreal isrow isscalar isspace isstring
isstruct isvector kron lcm length linspace log log10 log1p log2 logical lower mat2str max mean median min
mod nchoosek ndims nnz norm not num2cell num2str numel ones or pi power prod rad2deg real regexp regexprep
rem repmat reshape round sec sign sin sind single sinh size sort sprintf sqrt squeeze startsWith std
str2double strcat strcmp strcmpi strfind strjoin strjust strlength strncmp strncmpi strrep strsplit strtrim
sum tan tand tanh times trace transpose tril triu true uint16 uint32 uint64 uint8 unique upper var vertcat
xor zeros plus minus rdivide ldivide mtimes and lt gt le ge eq ne uminus numel cumtrapz trapz polyval fix
""".split())
# ... whose result is a real scalar often enough to expect one (the code bails out if it is not)
_LIKELY_SCALAR = frozenset("""
sum prod max min norm dot numel length ndims nnz str2double mean median std var det trace abs sqrt exp log
floor ceil round fix mod rem sin cos tan atan atan2 sign real imag gcd lcm factorial nchoosek strlength
polyval trapz double
""".split())
# ... and those that may see a whole array that the loop also writes to (their result never shares
# memory with the argument)
_NO_ALIAS = frozenset("""
sum prod max min norm dot numel length ndims nnz mean median std var any all isempty size find
""".split())


# -- static scan --------------------------------------------------------------------------------

class _Scan:
    """Names a piece of code uses and how; raises Reject for statements that cannot be compiled."""

    def __init__(self, allow_loops: bool):
        self.allow_loops = allow_loops
        self.names: set[str] = set()            # every identifier
        self.assigned: set[str] = set()         # x = ...
        self.index_written: set[str] = set()    # x(i) = ...
        self.brace_used: set[str] = set()       # x{i} read or written
        self.brace_written: set[str] = set()
        self.fields: set[tuple[str, str]] = set()          # s.f read or written
        self.fields_written: set[tuple[str, str]] = set()  # s.f = ..., s.f(i) = ...
        self.loop_vars: set[str] = set()
        self.reads: dict[str, int] = {}         # occurrences as a plain identifier

    def block(self, body: list) -> None:
        for stmt in body:
            t = type(stmt)
            if t is A.Assign:
                if stmt.show or len(stmt.targets) != 1:
                    raise Reject
                self.target(stmt.targets[0])
                self.expr(stmt.value)
            elif t is A.If:
                for cond, clause in stmt.clauses:
                    self.expr(cond)
                    self.block(clause)
                if stmt.orelse is not None:
                    self.block(stmt.orelse)
            elif t is A.Break or t is A.Continue:
                pass
            elif self.allow_loops and t is A.For:
                if type(stmt.iterable) is not A.Range or type(stmt.var) is not A.Ident:
                    raise Reject
                self.names.add(stmt.var.name)
                self.assigned.add(stmt.var.name)
                self.loop_vars.add(stmt.var.name)
                self.expr(stmt.iterable)
                self.block(stmt.body)
            elif self.allow_loops and t is A.While:
                self.expr(stmt.cond)
                self.block(stmt.body)
            elif self.allow_loops and t is A.Return:
                pass
            else:
                raise Reject

    def target(self, target) -> None:
        t = type(target)
        if t is A.Ident:
            if target.name == "~":
                raise Reject
            self.names.add(target.name)
            self.assigned.add(target.name)
        elif t is A.Index and type(target.base) is A.Ident and 1 <= len(target.args) <= 2:
            self.names.add(target.base.name)
            self.index_written.add(target.base.name)
            for a in target.args:
                self.expr(a)
        elif t is A.Field and type(target.base) is A.Ident:
            self.names.add(target.base.name)
            self.fields.add((target.base.name, target.name))
            self.fields_written.add((target.base.name, target.name))
        elif t is A.Index and type(target.base) is A.Field and type(target.base.base) is A.Ident \
                and 1 <= len(target.args) <= 2:
            pair = (target.base.base.name, target.base.name)
            self.names.add(pair[0])
            self.fields.add(pair)
            self.fields_written.add(pair)
            for a in target.args:
                self.expr(a)
        elif t is A.CellIndex and type(target.base) is A.Ident and len(target.args) == 1:
            self.names.add(target.base.name)
            self.brace_used.add(target.base.name)
            self.brace_written.add(target.base.name)
            self.expr(target.args[0])
        else:
            raise Reject

    def expr(self, node) -> None:
        t = type(node)
        if t is A.Number:
            if isinstance(node.value, complex):
                raise Reject
        elif t is A.String or t is A.End or t is A.Colon or t is A.IntLiteral:
            pass
        elif t is A.Ident:
            self.names.add(node.name)
            self.reads[node.name] = self.reads.get(node.name, 0) + 1
        elif t is A.BinOp:
            if node.op not in _GENERIC_OPS and node.op not in ("&&", "||"):
                raise Reject
            self.expr(node.left)
            self.expr(node.right)
        elif t is A.UnaryOp:
            self.expr(node.operand)
        elif t is A.Postfix:
            self.expr(node.operand)
        elif t is A.Index or t is A.CellIndex:
            base = node.base
            if type(base) is A.Ident:
                self.names.add(base.name)
                if t is A.CellIndex:
                    self.brace_used.add(base.name)
                elif base.name in ("numel", "length", "size") and node.args and type(node.args[0]) is A.Ident:
                    self.names.add(node.args[0].name)  # asking for a size is not a use of the whole value
                    for a in node.args[1:]:
                        self.expr(a)
                    return
            elif type(base) is A.Field and type(base.base) is A.Ident and t is A.Index:
                self.names.add(base.base.name)
                self.fields.add((base.base.name, base.name))
            else:
                self.expr(base)
            for a in node.args:
                self.expr(a)
        elif t is A.Field:
            if type(node.base) is A.Ident:
                self.names.add(node.base.name)
                self.fields.add((node.base.name, node.name))
            else:
                self.expr(node.base)
        elif t is A.Matrix:
            for row in node.rows:
                for e in row:
                    self.expr(e)
        elif t is A.Range:
            self.expr(node.start)
            if node.step is not None:
                self.expr(node.step)
            self.expr(node.stop)
        else:
            raise Reject


def _static(stmt) -> dict | None:
    cached = stmt.__dict__.get("_jit_static")
    if cached is not None:
        return cached or None
    scan = _Scan(allow_loops=False)
    try:
        if type(stmt) is A.While:
            scan.expr(stmt.cond)
        scan.block(stmt.body)
    except Reject:
        stmt._jit_static = False
        return None
    info = {"scan": scan, "bails": 0, "compiled": {}}
    stmt._jit_static = info
    return info


# -- name resolution ----------------------------------------------------------------------------

def _resolve(interp, name: str, functions: dict):
    """What calling ``name`` means: ('u', function, its file's functions), ('b', builtin) or (kind,)."""
    fn = functions.get(name)
    if fn is not None:
        return ("u", fn, functions) if type(fn) is A.Function else ("n",)
    if interp.private_or_imported(name, interp.frame):
        return ("?",)  # an import or a private function may give the name another meaning
    if name in interp.classes:
        return ("c",)
    found = interp.find_file_function(name)
    if found is not None:
        program = found[0]
        if program.classdefs or program.body or not program.functions:
            return ("s",)
        return ("u", next(iter(program.functions.values())), program.functions)
    builtin = interp.builtins.get(name)
    if builtin is not None:
        return ("b", builtin)
    return ("?",)


def _deps_valid(interp, deps: dict) -> bool:
    for (name, _fid), (functions, kind, target) in deps.items():
        now = _resolve(interp, name, functions)
        if now[0] != kind or (len(now) > 1 and now[1] is not target):
            return False
    return True


# -- code generation ----------------------------------------------------------------------------

def _unify(types: set) -> str:
    if len(types) == 1:
        return next(iter(types))
    if "O" in types:
        return "O"
    return "o"


class _Gen:
    def __init__(self, interp, functions: dict, kinds: dict, scan: _Scan, mode: str):
        self.interp = interp
        self.functions = functions
        self.kinds = kinds          # name (or 'struct.field') -> entry kind
        self.scan = scan
        self.mode = mode            # 'loop' or 'func'
        self.nargin = 0
        self.arrays: set[str] = set()     # double arrays read and written element by element
        self.lists: set[str] = set()      # row cells held as Python lists
        self.structs: set[str] = set()    # scalar structs whose fields are held as variables
        self.variables: set[str] = set()  # scalars and boxed values
        self.types: dict[str, set] = {}
        self.final: dict[str, str] = {}
        self.lines: list[str] = []
        self.paths: list[tuple] = []
        self.deps: dict = {}
        self.inline = None
        self.consts: dict[str, tuple] = {}
        self.blocked: frozenset = frozenset()
        self.whole_ok = False
        self.loop_id = 0
        self.return_code = ""

    # -- names ------------------------------------------------------------------------------
    @staticmethod
    def py(name: str, prefix: str = "v") -> str:
        if "." in name:
            s, f = name.split(".", 1)
            return f"{prefix}f{len(s)}_{s}_{f}"
        return f"{prefix}_{name}"

    def var_type(self, name: str) -> str:
        t = self.final.get(name)
        if t is not None:
            return t
        k = self.kinds.get(name, "-")
        if k == "d" or k == "b" or k in _IDT:
            return k
        return "O" if k == "x" or k.startswith("h:") else "d"

    def const(self, value, key=None) -> str:
        key = id(value) if key is None else key
        hit = _CONSTS.get(key)
        if hit is not None and hit[1] is (value if key == id(value) else hit[1]):
            return hit[0]
        name = _unique("_k")
        _NS[name] = value
        _CONSTS[key] = (name, value)
        return name

    def resolve(self, name: str):
        res = _resolve(self.interp, name, self.functions)
        self.deps[(name, id(self.functions))] = (self.functions, res[0], res[1] if len(res) > 1 else None)
        return res

    def is_builtin(self, name: str) -> bool:
        return self.resolve(name)[0] == "b"

    # -- conversions ------------------------------------------------------------------------
    @staticmethod
    def as_float(code: str, typ: str) -> str:
        if typ == "d":
            return code
        if typ in _IDT:
            raise Reject  # an integer-class value is not a double: only the contexts that allow one take it
        return f"float({code})" if typ == "b" else f"_d({code})"

    def as_index(self, code: str, typ: str) -> str:
        """An index: a double, a logical or an integer-class scalar (whose value the index helpers check)."""
        return code if typ in _IDT else self.as_float(code, typ)

    @staticmethod
    def as_truth(code: str, typ: str) -> str:
        if typ == "b":
            return code
        if typ in _IDT:
            return f"({code} != 0)"
        return f"_tr({code})" if typ == "d" else f"_truth({code})"

    @staticmethod
    def as_box(code: str, typ: str) -> str:
        if typ == "d":
            return f"_bd({code})"
        if typ in _IDT:
            return f"_b{typ}({code})"
        return f"_bb({code})" if typ == "b" else code

    def convert(self, code: str, typ: str, want: str) -> str:
        if want in "oO":
            return self.as_box(code, typ)
        if want == typ:
            return code
        if want == "d":
            return self.as_float(code, typ)
        raise Reject  # a logical variable assigned something else: sorted out by the next round

    # -- expressions ------------------------------------------------------------------------
    def expr(self, node, ctx=None) -> tuple[str, str]:
        t = type(node)
        if t is A.Number:
            v = float(node.value)
            if v != v:
                return "_nan", "d"
            if v in (math.inf, -math.inf):
                return ("_inf" if v > 0 else "(-_inf)"), "d"
            return repr(v), "d"
        if t is A.IntLiteral:
            code = _ICODE.get(node.cls)
            if code is None or not _ILIM[code][0] <= node.value <= _ILIM[code][1]:
                raise Reject
            return repr(int(node.value)), code
        if t is A.String:
            name = node.__dict__.get("_jit_const")
            if name is None:
                value = MString(node.value) if node.double else _readonly(char(node.value))
                name = node._jit_const = self.const(value)
            return name, "O"
        if t is A.Ident:
            return self.ident(node.name)
        if t is A.End:
            if ctx is None:
                raise Reject
            return ctx, "d"
        if t is A.UnaryOp:
            code, typ = self.expr(node.operand, ctx)
            if typ in "oO":
                return f"_unary({node.op!r}, {code})", typ
            if typ in _IDT:
                if node.op == "-":  # saturates: -intmin is intmax, and unsigned values become 0
                    return (f"_n{typ}({code})" if _ILIM[typ][0] else f"(0 * {code})"), typ
                if node.op == "+":
                    return code, typ
                return f"(not {self.as_truth(code, typ)})", "b"
            if node.op == "-":
                return f"(-{self.as_float(code, typ)})", "d"
            if node.op == "+":
                return self.as_float(code, typ), "d"
            return f"(not {self.as_truth(code, typ)})", "b"
        if t is A.Postfix:
            code, typ = self.expr(node.operand, ctx)
            if typ in "db" or typ in _IDT:
                return code, typ
            return f"_transpose({code}, {node.op == chr(39)!r})", typ
        if t is A.BinOp:
            return self.binop(node, ctx)
        if t is A.Matrix:
            return self.matrix(node, ctx)
        if t is A.Range:
            a = self.as_float(*self.expr(node.start, ctx))
            s = "None" if node.step is None else self.as_float(*self.expr(node.step, ctx))
            b = self.as_float(*self.expr(node.stop, ctx))
            return f"_range({a}, {s}, {b})", "O"
        if t is A.Field:
            if type(node.base) is A.Ident and node.base.name in self.structs:
                return self.ident(f"{node.base.name}.{node.name}")
            code, typ = self.expr(node.base, ctx)
            if typ not in "oO":
                raise Reject
            return f"_field({code}, {node.name!r})", "o"
        if t is A.CellIndex:
            return self.cell_read(node, ctx)
        if t is A.Index:
            return self.index_read(node, ctx)
        raise Reject

    def ident(self, name: str) -> tuple[str, str]:
        if name in self.consts:
            return self.consts[name]
        if name in self.blocked or name in self.lists or name in self.structs:
            raise Reject
        if name in self.arrays:
            if name in self.written_arrays and not self.whole_ok:
                raise Reject  # the whole array would be visible while it is being changed
            return self.py(name, "a"), "O"
        if name in self.variables:
            return self.py(name), self.var_type(name)
        if "." in name:
            raise Reject
        if name == "nargin" and self.mode == "func" and self.is_builtin(name):
            return repr(float(self.nargin)), "d"
        if name in _CONSTANTS and self.is_builtin(name):
            return _CONSTANTS[name]
        raise Reject  # a function call without parentheses, or undefined

    def binop(self, node, ctx) -> tuple[str, str]:
        op = node.op
        a, ta = self.expr(node.left, ctx)
        b, tb = self.expr(node.right, ctx)
        if op in ("&&", "||"):
            joiner = "and" if op == "&&" else "or"
            left = f"_sc({a})" if ta in "oO" else self.as_truth(a, ta)
            right = f"_sc({b})" if tb in "oO" else self.as_truth(b, tb)
            return f"({left} {joiner} {right})", "b"
        boxed_a, boxed_b = ta in "oO", tb in "oO"
        if boxed_a or boxed_b:
            if ta == "O" or tb == "O" or (boxed_a and boxed_b):
                kind = "O" if "O" in (ta, tb) else "o"
                return f"_binop({op!r}, {self.as_box(a, ta)}, {self.as_box(b, tb)})", kind
            # one boxed value of unknown shape next to a scalar: expect a scalar (bails otherwise)
            other = tb if boxed_a else ta
            if other in _IDT:
                # next to an integer: a double or logical scalar (as a float), or one of the same class (as an int)
                pyop = _ARITH.get(op)
                if not (op in _COMPARE or op in ("&", "|")
                        or (pyop is not None and (other not in _WIDE or pyop != "/"))):
                    return f"_ig({op!r}, {self.as_box(a, ta)}, {self.as_box(b, tb)}, {other!r})", other
                unbox = f"_q{other}"
            else:
                unbox = "_d"
            if boxed_a:
                a, ta = f"{unbox}({a})", "d"
            else:
                b, tb = f"{unbox}({b})", "d"
        if ta in _IDT or tb in _IDT:
            return self.int_binop(op, a, ta, b, tb)
        if op in _ARITH:
            return f"({self.as_float(a, ta)} {_ARITH[op]} {self.as_float(b, tb)})", "d"
        if op in ("^", ".^"):
            right = node.right
            if type(right) is A.Number and not isinstance(right.value, complex) \
                    and float(right.value) == int(right.value) and 0 <= right.value <= 64:
                return f"({self.as_float(a, ta)} ** {float(right.value)!r})", "d"
            return f"_pow({self.as_float(a, ta)}, {self.as_float(b, tb)})", "d"
        if op in _COMPARE:
            return f"({a} {_COMPARE[op]} {b})", "b"
        if op in ("&", "|"):
            # element-wise: both sides are evaluated (an error on the right must not be skipped);
            # Python's & and | on two bools do that and give a bool
            return f"(({self.as_truth(a, ta)}) {op} ({self.as_truth(b, tb)}))", "b"
        raise Reject  # left division of scalars: rare enough to leave to the interpreter

    def int_binop(self, op: str, a: str, ta: str, b: str, tb: str) -> tuple[str, str]:
        """An operator with an integer-class scalar operand (the other: a double, a logical or an integer)."""
        code = ta if ta in _IDT else tb
        if op in _COMPARE:
            # Python compares ints, bools and floats by their exact values, as the general code does
            return f"({a} {_COMPARE[op]} {b})", "b"
        if op in ("&", "|"):
            return f"(({self.as_truth(a, ta)}) {op} ({self.as_truth(b, tb)}))", "b"
        same = ta == tb or ta in "db" or tb in "db"
        pyop = _ARITH.get(op)
        if same and pyop is not None:
            if code not in _WIDE:
                # every value of these classes is exact in a double: Python computes in doubles (or exactly,
                # which gives the same double), then the result is rounded and saturated
                return f"_s{code}({a} {pyop} {b})", code
            if pyop != "/":
                return f"_w{code}({op!r}, {a}, {b})", code
        # powers, divisions of 64-bit integers, left divisions, mixed integer classes (an error):
        # the operators' general code
        return f"_ig({op!r}, {self.as_box(a, ta)}, {self.as_box(b, tb)}, {code!r})", code

    def matrix(self, node, ctx) -> tuple[str, str]:
        rows = [[self.expr(e, ctx) for e in row] for row in node.rows]
        count = sum(len(r) for r in rows)
        if count == 0:
            return "_EMPTY", "O"
        if count == 1:
            return rows[0][0] if rows[0] else next(r for r in rows if r)[0]
        if all(typ == "d" for r in rows for _, typ in r) and len({len(r) for r in rows}) == 1:
            body = ", ".join("[" + ", ".join(code for code, _ in r) + "]" for r in rows)
            return f"_arr([{body}])", "O"
        body = ", ".join("[" + ", ".join(self.as_box(code, typ) for code, typ in r) + "]" for r in rows)
        return f"_build([{body}])", "O"

    def cell_read(self, node, ctx) -> tuple[str, str]:
        if len(node.args) != 1 or type(node.args[0]) is A.Colon:
            raise Reject
        base = node.base
        if type(base) is A.Ident and base.name in self.lists:
            lst = self.py(base.name, "l")
            i = self.as_index(*self.expr(node.args[0], f"float(len({lst}))"))
            return f"{lst}[_ix({i}, len({lst}))]", "o"
        code, typ = self.expr(base, ctx)
        if typ not in "oO":
            raise Reject
        i = self.as_index(*self.expr(node.args[0], f"_numelf({code})"))
        return f"_cellget({code}, {i})", "o"

    def index_read(self, node, ctx) -> tuple[str, str]:
        base, args = node.base, node.args
        if type(base) is A.Field and type(base.base) is A.Ident and base.base.name in self.structs:
            name = f"{base.base.name}.{base.name}"
        elif type(base) is A.Ident:
            name = base.name
        else:
            raise Reject
        if name in self.arrays:
            if not 1 <= len(args) <= 2:
                raise Reject
            return f"{self.py(name, 'a')}.item({self.index(name, args)})", "d"
        if name in self.lists or name in self.structs or name in self.blocked:
            raise Reject
        if name in self.variables or name in self.consts:
            if self.kinds.get(name, "-") == "H":
                raise Reject  # a captured function handle
            if self.kinds.get(name, "-").startswith("h:"):
                inlined = None
                if self.inline is not None and name not in self.scan.assigned:
                    inlined = self.inline(name, args, ctx, self)
                if inlined is None:
                    raise Reject  # a function handle that cannot be inlined
                return inlined
            code, typ = self.ident(name)
            if typ not in "oO":
                raise Reject  # indexing into a scalar variable
            return self.boxed_index(code, args)
        if "." in name:
            raise Reject
        return self.call(name, args, ctx)

    def boxed_index(self, code: str, args: list) -> tuple[str, str]:
        count = len(args)
        if 1 <= count <= 2 and all(type(a) not in (A.Colon, A.Range, A.Matrix, A.String) for a in args):
            parts = [self.expr(a, f"_endf({code}, {k}, {count})") for k, a in enumerate(args)]
            if all(typ in "dbo" or typ in _IDT for _, typ in parts):
                idx = ", ".join(self.as_index(c, typ) for c, typ in parts)
                return f"_get{count}({code}, {idx})", "d"
        items = []
        for k, a in enumerate(args):
            if type(a) is A.Colon:
                items.append("_COLON")
            else:
                items.append(self.as_box(*self.expr(a, f"_endf({code}, {k}, {count})")))
        return f"_index({code}, [{', '.join(items)}])", "o"

    def call(self, name: str, args: list, ctx) -> tuple[str, str]:
        res = self.resolve(name)
        if res[0] == "u":
            return self.user_call(res[1], res[2], args, ctx)
        if res[0] != "b":
            raise Reject
        if name in ("numel", "length", "size") and args and type(args[0]) is A.Ident:
            target = args[0].name
            if target in self.arrays:
                arr = self.py(target, "")
                if name == "numel" and len(args) == 1:
                    return f"float(n{arr})", "d"
                if name == "length" and len(args) == 1:
                    return f"float(max(r{arr}, c{arr}))", "d"
                if name == "size" and len(args) == 2 and type(args[1]) is A.Number and args[1].value in (1, 2):
                    return f"float({'r' if args[1].value == 1 else 'c'}{arr})", "d"
            elif target in self.lists and name in ("numel", "length") and len(args) == 1:
                return f"float(len({self.py(target, 'l')}))", "d"
        saved = self.whole_ok
        self.whole_ok = name in _NO_ALIAS and len(args) == 1
        try:
            parts = [self.expr(a, ctx) for a in args]
        finally:
            self.whole_ok = saved
        if name in _ICODE and len(parts) == 1:
            conv = self.int_conversion(name, args[0], *parts[0], res[1])
            if conv is not None:
                return conv
        if any(typ in _IDT for _, typ in parts):
            return self.int_call(name, res[1], parts)
        spec = _FUNCS.get(name)
        if spec is not None and spec[0] == len(parts) and all(typ != "O" for _, typ in parts) \
                and not (name in ("min", "max") and any(typ == "o" for _, typ in parts)):
            if name in ("min", "max") and all(typ == "b" for _, typ in parts):
                # of two logical values, logical
                return f"bool({spec[1].format(*(self.as_float(c, typ) for c, typ in parts))})", "b"
            return spec[1].format(*(self.as_float(c, typ) for c, typ in parts)), "d"
        if name not in _PURE:
            raise Reject
        boxed = ", ".join(self.as_box(c, typ) for c, typ in parts)
        helper = _FLOAT_CALLS.get(name)
        if helper is not None and len(parts) == 1:
            _NS.setdefault("_bi_" + name, res[1])
            if _NS["_bi_" + name] is res[1]:
                return f"{helper}({boxed})", "d"
        fn = self.const(res[1])
        if name in ("numel", "length", "ndims") or (name == "size" and len(parts) == 2):
            return f"_d(_callb({fn}, [{boxed}]))", "d"
        if name in _LIKELY_SCALAR and all(typ != "O" for _, typ in parts):
            return f"_df(_callb({fn}, [{boxed}]))", "d"
        return f"_callb({fn}, [{boxed}])", ("o" if name in _LIKELY_SCALAR else "O")

    def int_conversion(self, name: str, node, code: str, typ: str, builtin) -> tuple[str, str] | None:
        """int8(x) ... uint64(x) of a scalar, as saturate_int converts it; None for the general call."""
        target = _ICODE[name]
        lo, hi = _ILIM[target]
        if target in _WIDE:
            exact = _exact_lit(node)
            if exact is not None:  # int64(<literal>) keeps every digit (the interpreter does the same)
                return repr(min(max(exact, lo), hi)), target
        if typ == "d":
            return f"_s{target}({code})", target
        if typ == "b":
            return f"int({code})", target
        if typ == target:
            return code, target
        if typ in _IDT:
            return f"_c{target}({code})", target
        if typ == "o":
            return f"_u{target}(_callb({self.const(builtin)}, [{code}]))", target
        return None

    def int_call(self, name: str, builtin, parts: list) -> tuple[str, str]:
        """A built-in called with an integer-class scalar among its arguments."""
        if name not in _PURE:
            raise Reject
        if name == "double" and len(parts) == 1:
            c, typ = parts[0]
            return (f"_dw({c})" if typ in _WIDE else f"float({c})"), "d"
        boxed = ", ".join(self.as_box(c, typ) for c, typ in parts)
        fn = self.const(builtin)
        if name in ("numel", "length", "ndims") or (name == "size" and len(parts) == 2):
            return f"_d(_callb({fn}, [{boxed}]))", "d"
        codes = {typ for _, typ in parts if typ in _IDT}
        if name in _INT_KEEPING and len(codes) == 1 and all(typ in "db" or typ in _IDT for _, typ in parts):
            code = codes.pop()
            return f"_u{code}(_callb({fn}, [{boxed}]))", code  # bails if the class is not the expected one
        return f"_callb({fn}, [{boxed}])", ("o" if name in _LIKELY_SCALAR else "O")

    def user_call(self, fn, functions, args: list, ctx) -> tuple[str, str]:
        parts = [self.expr(a, ctx) for a in args]
        sig = tuple("o" if typ in "oO" else typ for _, typ in parts)
        record = _compile_function(self.interp, fn, functions, sig)
        self.deps.update(record.deps)
        actual = "".join(f", {c}" for c, _ in parts)
        return f"{record.name}(_dp + 1{actual})", ("o" if record.ret in "oO" else record.ret)

    def index(self, name: str, args: list) -> str:
        kind = self.kinds[name]
        n = self.py(name, "")
        if len(args) == 1:
            i = self.as_index(*self.expr(args[0], f"float(n{n})"))
            if kind in "rc":
                return f"_ix({i}, n{n})"
            return f"_lin({i}, n{n}, r{n})"
        i = self.as_index(*self.expr(args[0], f"float(r{n})"))
        j = self.as_index(*self.expr(args[1], f"float(c{n})"))
        return f"(_ix({i}, r{n}), _ix({j}, c{n}))"

    def store_index(self, name: str, args: list) -> str:
        kind = self.kinds[name]
        if len(args) == 1 and kind in "rc":
            n = self.py(name, "")
            i = self.as_index(*self.expr(args[0], f"float(n{n})"))
            k = f"_ix({i}, n{n})"
            return f"0, {k}" if kind == "r" else f"{k}, 0"
        return self.index(name, args)

    # -- statements -------------------------------------------------------------------------
    def emit(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)

    def assign_var(self, indent: int, name: str, value) -> None:
        if name not in self.variables:
            raise Reject
        code, typ = self.expr(value)
        self.types.setdefault(name, set()).add(typ)
        want = self.var_type(name)
        self.emit(indent, f"{self.py(name)} = {self.convert(code, typ, want)}")

    def block(self, body: list, indent: int, path: tuple) -> None:
        if not body:
            self.emit(indent, "pass")
        for k, stmt in enumerate(body):
            here = path + (k,)
            t = type(stmt)
            if t is A.Break:
                self.emit(indent, "break")
                continue
            if t is A.Continue:
                self.emit(indent, "continue")
                continue
            if t is A.Return:
                self.emit(indent, self.return_code)
                continue
            if self.mode == "loop":
                self.paths.append(here)
                self.emit(indent, f"_p = {len(self.paths) - 1}")
            if t is A.Assign:
                self.assign(indent, stmt)
            elif t is A.If:
                for c, (cond, clause) in enumerate(stmt.clauses):
                    code, typ = self.expr(cond)
                    self.emit(indent, f"{'if' if c == 0 else 'elif'} {self.as_truth(code, typ)}:")
                    self.block(clause, indent + 1, here + (c,))
                if stmt.orelse is not None:
                    self.emit(indent, "else:")
                    self.block(stmt.orelse, indent + 1, here + (len(stmt.clauses),))
            elif t is A.For:
                self.for_loop(indent, stmt)
            elif t is A.While:
                self.emit(indent, "while True:")
                code, typ = self.expr(stmt.cond)
                self.emit(indent + 1, f"if not {self.as_truth(code, typ)}:")
                self.emit(indent + 2, "break")
                self.block(stmt.body, indent + 1, ())
            else:
                raise Reject

    def for_loop(self, indent: int, stmt) -> None:
        self.loop_id += 1
        k = self.loop_id
        name = stmt.var.name
        if name not in self.variables:
            raise Reject
        rng = stmt.iterable
        a = self.as_float(*self.expr(rng.start))
        s = "1.0" if rng.step is None else self.as_float(*self.expr(rng.step))
        b = self.as_float(*self.expr(rng.stop))
        self.types.setdefault(name, set()).add("d")
        self.emit(indent, f"_a{k} = {a}")
        self.emit(indent, f"_s{k} = {s}")
        self.emit(indent, f"_n{k} = _count(_a{k}, _s{k}, {b})")
        if self.read_outside(name, stmt):
            self.emit(indent, f"if _n{k} == 0:")
            self.emit(indent + 1, "raise _Bail")  # the loop variable becomes empty: interpreter's job
        self.emit(indent, f"for _i{k} in range(_n{k}):")
        self.emit(indent + 1, f"{self.py(name)} = {self.convert(f'(_a{k} + _s{k} * _i{k})', 'd', self.var_type(name))}")
        self.block(stmt.body, indent + 1, ())

    def read_outside(self, name: str, loop) -> bool:
        inner = _Scan(allow_loops=True)
        try:
            inner.block(loop.body)
        except Reject:
            return True
        return self.scan.reads.get(name, 0) > inner.reads.get(name, 0) or name == self.output

    output = ""

    def assign(self, indent: int, stmt) -> None:
        target = stmt.targets[0]
        t = type(target)
        if t is A.Ident:
            self.assign_var(indent, target.name, stmt.value)
            return
        if t is A.Field:
            name = f"{target.base.name}.{target.name}"
            if target.base.name not in self.structs or name in self.arrays:
                raise Reject
            self.assign_var(indent, name, stmt.value)
            return
        if t is A.CellIndex:
            name = target.base.name
            if name not in self.lists:
                raise Reject
            lst = self.py(name, "l")
            value = self.as_box(*self.expr(stmt.value))
            arg = target.args[0]
            if type(arg) is A.BinOp and arg.op == "+" and type(arg.left) is A.End \
                    and type(arg.right) is A.Number and arg.right.value == 1:
                self.emit(indent, f"{lst}.append({value})")
            else:
                i = self.as_index(*self.expr(arg, f"float(len({lst}))"))
                self.emit(indent, f"_lset({lst}, {i}, {value})")
            self.emit(indent, f"{self.py(name, 'm')} = True")
            return
        # x(i) = value  /  s.f(i) = value
        base = target.base
        name = f"{base.base.name}.{base.name}" if type(base) is A.Field else base.name
        if "." in name and base.base.name not in self.structs:
            raise Reject
        code, typ = self.expr(stmt.value)
        value = self.as_float(code, typ)
        if name in self.arrays:
            self.emit(indent, f"{self.py(name, 'a')}[{self.store_index(name, target.args)}] = {value}")
            return
        if name not in self.variables or self.var_type(name) not in "oO":
            raise Reject
        var = self.py(name)
        count = len(target.args)
        idx = ", ".join(self.as_index(*self.expr(a, f"_endf({var}, {k}, {count})"))
                        for k, a in enumerate(target.args))
        self.emit(indent, f"_set{count}({var}, {idx}, {value})")

    written_arrays: set = set()


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _kind(value: Any) -> str:
    if value is None:
        return "-"
    t = type(value)
    if t is FunctionHandle:
        if not value.is_anonymous:
            return "x"
        # anonymous function: identified by its body and the scalar values it captured
        caught = tuple(sorted((k, v.item()) for k, v in value.closure.items()
                              if type(v) is np.ndarray and v.shape == (1, 1) and v.dtype.kind in "fb"))
        return f"h:{id(value.body)}:{caught!r}"
    if t is Cell:
        shape = value.data.shape
        return "L" if shape == (0, 0) or (len(shape) == 2 and shape[0] == 1) else "x"
    if t is Struct:
        return "S" if value.data.shape == (1, 1) else "x"
    if t is not np.ndarray or value.ndim != 2:
        return "x"
    if value.shape == (1, 1):
        if value.dtype is _F64:
            return "d"
        if value.dtype == bool:
            return "b"
        return _DT_CODE.get(value.dtype, "x")
    if value.dtype is not _F64 or value.size == 0:
        return "x"
    return "r" if value.shape[0] == 1 else ("c" if value.shape[1] == 1 else "m")


def _roles(gen: _Gen, scan: _Scan, kinds: dict, declared: set) -> None:
    """Decide how each name is held in the compiled code."""
    structs = {s for s, _f in scan.fields}
    for name in scan.names:
        kind = kinds.get(name, "-")
        if name in structs:
            if kind != "S" or name in scan.assigned or name in scan.index_written or name in scan.brace_used \
                    or scan.reads.get(name, 0):
                raise Reject  # the struct is also used as a whole
            gen.structs.add(name)
        elif name in scan.brace_written or (name in scan.brace_used and kind == "L"):
            if kind != "L" or name in scan.assigned or name in scan.index_written or scan.reads.get(name, 0):
                raise Reject
            gen.lists.add(name)
        elif kind in ("r", "c", "m") and name not in scan.assigned:
            gen.arrays.add(name)
        elif kind != "-" or name in scan.assigned or name in declared:
            if kind in ("r", "c", "m", "L", "S") or kind.startswith("h:"):
                kinds[name] = "x" if not kind.startswith("h:") else kind
            gen.variables.add(name)
    for s, f in scan.fields:
        if s not in gen.structs:
            continue
        name = f"{s}.{f}"
        kind = kinds.get(name, "-")
        if kind == "-":
            raise Reject  # a field that does not exist yet
        if kind in ("r", "c", "m"):
            gen.arrays.add(name)
        else:
            if kind not in "db":
                kinds[name] = "x"
            gen.variables.add(name)
    gen.written_arrays = {n for n in gen.arrays if n in scan.index_written} | \
        {f"{s}.{f}" for s, f in scan.fields_written if f"{s}.{f}" in gen.arrays}
    for name in scan.index_written:
        if name in gen.lists or name in gen.structs:
            raise Reject


def _infer(gen: _Gen, kinds: dict, generate) -> None:
    """Run the generator until the variable types it infers stop changing."""
    final: dict[str, str] = {}
    for _attempt in range(12):
        gen.final = final
        gen.types = {}
        gen.lines = []
        gen.paths = []
        gen.loop_id = 0
        try:
            generate()
            ok = True
        except Reject:
            ok = False  # perhaps a variable had the wrong provisional type: see what was learned
        new_final = dict(final)
        for name, types in gen.types.items():
            entry = kinds.get(name, "-")
            seen = set(types)
            if entry == "d" or entry == "b" or entry in _IDT:
                seen.add(entry)
            elif entry != "-":
                seen.add("O")
            if name in final:
                seen.add(final[name])
            new_final[name] = _unify(seen)
        if new_final == final:
            if ok:
                return
            raise Reject
        final = new_final
    raise Reject


class _Function:
    __slots__ = ("name", "ret", "deps", "py", "bails", "sig", "captured")

    def __init__(self, name: str, ret: str):
        self.name = name
        self.ret = ret
        self.deps: dict = {}
        self.py = None
        self.bails = 0
        self.sig = ()
        self.captured: tuple = ()   # names of the workspace values an anonymous function captured


_COMPILING: dict = {}
_SESSION: list = []


def _compile_function(interp, fn, functions: dict, sig: tuple) -> _Function:
    """The compiled form of ``fn`` for arguments of the given types; raises Reject if not possible."""
    state = fn.__dict__.get("_jit_fn")
    if state is None:
        state = fn._jit_fn = {"calls": 0, "sigs": {}}
    key = (id(fn), sig)
    if key in _COMPILING:
        return _COMPILING[key]  # a recursive call: use the record being built
    record = state["sigs"].get(sig, _MISSING)
    if record is not _MISSING:
        if record is None:
            raise Reject
        if _deps_valid(interp, record.deps):
            return record
    if not _COMPILING:
        _SESSION.clear()
    state["sigs"][sig] = None  # until proven otherwise
    if fn.subfunctions or fn.arguments or len(fn.outputs) != 1 or len(sig) > len(fn.params) \
            or len(set(fn.params)) != len(fn.params) or "varargin" in fn.params or "~" in fn.params \
            or fn.outputs[0] == "varargout":
        raise Reject
    scan = _Scan(allow_loops=True)
    scan.block(fn.body)
    if scan.fields or scan.brace_written or "nargout" in scan.names:
        raise Reject
    out = fn.outputs[0]
    if out not in scan.assigned:
        raise Reject
    done = None
    outermost = not _COMPILING
    for assumed in ("d", "b", "o"):  # what a recursive call is taken to return while compiling
        record = _Function(_unique("_f"), assumed)
        record.sig = sig
        _COMPILING[key] = record
        try:
            kinds = {p: ("x" if k == "o" else k) for p, k in zip(fn.params, sig)}
            gen = _Gen(interp, functions, kinds, scan, "func")
            gen.nargin = len(sig)
            gen.output = out
            _roles(gen, scan, kinds, set(fn.params) | {out})
            if gen.arrays or gen.lists or gen.structs:
                raise Reject
            gen.return_code = f"return _ret({gen.py(out)})"
            _infer(gen, kinds, lambda gen=gen: gen.block(fn.body, 1, ()))
        finally:
            _COMPILING.pop(key, None)
        actual = gen.var_type(out)
        actual = "o" if actual in "oO" else actual
        if actual == assumed or not any(record.name + "(" in line for line in gen.lines):
            record.ret = actual
            done = (gen, record)
            break
    if done is None:
        raise Reject
    gen, record = done
    params = "".join(f", {gen.py(p)}=_UNSET" for p in fn.params)
    src = [f"def {record.name}(_dp{params}):", "    if _dp > _lim[0]:", "        raise _Bail"]
    for name in sorted(gen.variables):
        if name not in fn.params:
            src.append(f"    {gen.py(name)} = _UNSET")
    for p, k in zip(fn.params, sig):  # arguments arrive as floats/bools; box those the body keeps boxed
        if (k in "db" or k in _IDT) and gen.var_type(p) in "oO":
            src.append(f"    {gen.py(p)} = {gen.as_box(gen.py(p), k)}")
        elif k == "o" and p in scan.index_written:
            src.append(f"    {gen.py(p)} = _copy({gen.py(p)})")  # the caller keeps its own array
    src += gen.lines
    src.append(f"    {gen.return_code}")
    _NS.setdefault("_ret", _ret)
    exec(compile("\n".join(src), f"<mylab-jit {fn.name}>", "exec"), _NS)  # noqa: S102 - code generated above
    record.py = _NS[record.name]
    record.deps = gen.deps
    state["sigs"][sig] = record
    _SESSION.append(record)
    if outermost:  # functions that call each other share what they depend on
        merged: dict = {}
        for r in _SESSION:
            merged.update(r.deps)
        for r in _SESSION:
            r.deps = merged
        _SESSION.clear()
    return record


def _ret(v):
    if v is _UNSET:
        raise Bail
    return v


def call_function(interp, fn, args: list, nargout: int, functions: dict):
    """Run a simple numeric user function compiled; None if the interpreter must run it."""
    state = fn.__dict__.get("_jit_fn")
    if state is None:
        state = fn._jit_fn = {"calls": 0, "sigs": {}}
    if state.get("never"):
        return None
    state["calls"] += 1
    if state["calls"] < FUNCTION_CALLS_BEFORE_COMPILE or nargout > 1:
        return None
    sig = []
    values = []
    for v in args:
        if type(v) is _ND and v.shape == _S11 and v.dtype is _F64:
            sig.append("d")
            values.append(v.item())
        elif type(v) is _ND and v.shape == _S11 and v.dtype is _BOOL:
            sig.append("b")
            values.append(v.item())
        elif type(v) is _ND and v.shape == _S11 and v.dtype in _DT_CODE:
            sig.append(_DT_CODE[v.dtype])
            values.append(v.item())
        elif type(v) in _PLAIN:
            sig.append("o")
            values.append(v)
        else:
            return None
    key = tuple(sig)
    record = state["sigs"].get(key, _MISSING)
    if record is None:
        return None
    if record is _MISSING or record.py is None or not _deps_valid(interp, record.deps):
        try:
            record = _compile_function(interp, fn, functions, key)
        except (Reject, KeyError, SyntaxError, RecursionError):
            state["sigs"][key] = None
            if all(v is None for v in state["sigs"].values()) and len(state["sigs"]) >= 3:
                state["never"] = True
            return None
    _NS["_lim"][0] = interp.max_depth
    _INTERP[0] = interp
    try:
        r = record.py(interp.depth + 1, *values)
    except Exception:  # noqa: BLE001 - a bail-out: the function is pure, so the interpreter just runs it
        record.bails += 1
        # Without arguments a function meets the same values every time, so one bail-out means it would
        # always bail (and do part of its work twice).
        if record.bails >= MAX_BAILS or not values:
            state["sigs"][key] = None
        return None
    if record.ret == "d":
        return [_bd(r)]
    if record.ret == "b":
        return [_bb(r)]
    if record.ret in _IDT:
        return [_NS["_b" + record.ret](r)]
    return [r]


def _value_kind(v):
    """How a value is passed to compiled code: 'd', 'b', 'o' (boxed), or None if it cannot be."""
    t = type(v)
    if t is _ND:
        if v.shape == _S11:
            if v.dtype is _F64:
                return "d"
            if v.dtype is _BOOL:
                return "b"
        return "o"
    return "o" if t in _PLAIN else None


_PRIVATE_DIRS: dict = {}


def _has_private(file) -> bool:
    """True when the folder of ``file`` has a private/ folder (whose functions only that folder's code sees)."""
    if not file:
        return False
    found = _PRIVATE_DIRS.get(file)
    if found is None:
        folder = os.path.dirname(os.path.abspath(str(file)))
        found = _PRIVATE_DIRS[file] = os.path.isdir(os.path.join(folder, "private"))
    return found


def _compile_anonymous(interp, handle, sig: tuple):
    """The compiled form of an anonymous function for arguments of the given types, or None."""
    body = handle.body
    state = body.__dict__.get("_jit_fn")
    if state is None:
        state = body._jit_fn = {"calls": 0, "sigs": {}}
    closure = handle.closure
    functions = closure.get("__functions__", {})
    params = handle.params
    # Entries named __x__ are where the handle was made, not variables (MATLAB names cannot start with _).
    # Imports and a class's access change what a name means, so those handles stay with the interpreter; the
    # file matters only when its folder has private functions.
    internal = [k for k in closure if k.startswith("__")]
    if any(k not in ("__functions__", "__file__") for k in internal) or _has_private(closure.get("__file__")):
        return None
    captured = tuple(sorted(k for k in closure if not k.startswith("__") and k not in params))
    ckinds = []
    for k in captured:
        v = closure[k]
        kind = "H" if type(v) is FunctionHandle else _value_kind(v)
        if kind is None:
            return None
        ckinds.append(kind)
    key = (sig, tuple(ckinds), id(functions))
    record = state["sigs"].get(key, _MISSING)
    if record is None:
        return None
    if record is not _MISSING and _deps_valid(interp, record.deps):
        return record
    state["sigs"][key] = None
    if len(sig) > len(params) or "varargin" in params or "~" in params or len(set(params)) != len(params):
        return None
    try:
        scan = _Scan(allow_loops=False)
        scan.expr(body)
        kinds = {p: ("x" if k == "o" else k) for p, k in zip(params, sig)}
        kinds.update({c: ("x" if k == "o" else k) for c, k in zip(captured, ckinds)})
        gen = _Gen(interp, functions, kinds, scan, "func")
        gen.nargin = len(sig)
        gen.variables = set(params) | set(captured)
        code, typ = gen.expr(body)
        if typ in _IDT:
            code, typ = gen.as_box(code, typ), "o"
    except (Reject, KeyError, RecursionError):
        return None
    record = _Function(_unique("_f"), "o" if typ in "oO" else typ)
    record.sig = sig
    record.captured = captured
    names = "".join(f", {gen.py(p)}=_UNSET" for p in params) + "".join(f", {gen.py(c)}=_UNSET" for c in captured)
    src = f"def {record.name}(_dp{names}):\n    if _dp > _lim[0]:\n        raise _Bail\n    return {code}"
    try:
        exec(compile(src, "<mylab-jit anonymous>", "exec"), _NS)  # noqa: S102 - code generated above
    except SyntaxError:
        return None
    record.py = _NS[record.name]
    record.deps = gen.deps
    state["sigs"][key] = record
    state.setdefault("keep", []).append(functions)  # so that id(functions) is never reused
    return record


def _captured_values(handle, record) -> list:
    out = []
    for name in record.captured:
        v = handle.closure[name]
        out.append(v.item() if type(v) is _ND and v.shape == _S11 and (v.dtype is _F64 or v.dtype is _BOOL) else v)
    return out


def call_anonymous(interp, handle, args: list, nargout: int):
    """Evaluate a side-effect-free anonymous function compiled; None if the interpreter must."""
    body = handle.body
    state = body.__dict__.get("_jit_fn")
    if state is None:
        state = body._jit_fn = {"calls": 0, "sigs": {}}
    state["calls"] += 1
    if state["calls"] < FUNCTION_CALLS_BEFORE_COMPILE or nargout > 1 or state.get("never"):
        return None
    sig = []
    values = []
    for v in args:
        kind = _value_kind(v)
        if kind is None:
            return None
        sig.append(kind)
        values.append(v.item() if kind != "o" else v)
    record = _compile_anonymous(interp, handle, tuple(sig))
    if record is None:
        state["misses"] = state.get("misses", 0) + 1
        if state["misses"] >= 4:
            state["never"] = True
        return None
    _NS["_lim"][0] = interp.max_depth
    _INTERP[0] = interp
    if len(values) < len(handle.params):
        values += [_UNSET] * (len(handle.params) - len(values))
    try:
        r = record.py(interp.depth + 1, *values, *_captured_values(handle, record))
    except Exception:  # noqa: BLE001 - a bail-out: the expression is pure, the interpreter evaluates it
        record.bails += 1
        if record.bails >= 8 * MAX_BAILS:
            state["never"] = True
        return None
    if record.ret == "d":
        return [_bd(r)]
    if record.ret == "b":
        return [_bb(r)]
    return [r]


def map_function(interp, f, columns: list, kinds: tuple):
    """Apply a function to each element (arrayfun/cellfun): (result type, list of results) or None.

    ``columns`` holds one list of values per input, floats where ``kinds`` says 'd' and MATLAB
    values where it says 'o'.  Only compiled user functions, compiled anonymous functions and
    side-effect-free built-ins are handled; a failure on any element gives None, and the caller
    then does the whole job the general way (the function is pure, so nothing was changed).
    """
    if not interp.jit_enabled or interp.trace_hook is not None or type(f) is not FunctionHandle:
        return None
    extra: list = []
    if f.is_anonymous:
        record = _compile_anonymous(interp, f, kinds)
        if record is None:
            return None
        extra = _captured_values(f, record)
        missing = len(f.params) - len(kinds)
        if missing:
            extra = [_UNSET] * missing + extra
        py, ret = record.py, record.ret
    else:
        res = _resolve(interp, f.name, f.closure.get("__functions__", {}))
        if res[0] == "u":
            try:
                record = _compile_function(interp, res[1], res[2], kinds)
            except (Reject, KeyError, SyntaxError, RecursionError):
                return None
            py, ret = record.py, record.ret
        elif res[0] == "b" and f.name in _PURE and all(k == "o" for k in kinds):
            fn = res[1]

            def py(_dp, *args):
                return _callb(fn, list(args))
            ret = "o"
        else:
            return None
    _NS["_lim"][0] = interp.max_depth
    _INTERP[0] = interp
    depth = interp.depth + 1
    try:
        if len(columns) == 1 and not extra:
            out = [py(depth, x) for x in columns[0]]
        else:
            out = [py(depth, *xs, *extra) for xs in zip(*columns)]
    except Exception:  # noqa: BLE001 - some element needs the interpreter: let it do them all
        return None
    if ret in _IDT:
        box = _NS["_b" + ret]
        return "o", [box(x) for x in out]
    return ret, out


# -- loops --------------------------------------------------------------------------------------

class _Loop:
    __slots__ = ("fn", "variables", "arrays", "lists", "paths", "final", "deps", "struct_fields")


def _compile(stmt, info: dict, kinds: dict, interp, frame, loop_var: str | None) -> _Loop:
    scan: _Scan = info["scan"]
    kinds = dict(kinds)
    gen = _Gen(interp, frame.functions, kinds, scan, "loop")
    declared = {loop_var} if loop_var else set()
    loop_kind = kinds.get(loop_var, "d") if loop_var else "d"   # 'd' or an integer class's type code
    if loop_var:
        if (loop_kind != "d" and loop_kind not in _IDT) or loop_var in scan.index_written \
                or loop_var in scan.brace_used or any(s == loop_var for s, _ in scan.fields):
            raise Reject
    _roles(gen, scan, kinds, declared)
    if loop_var:
        gen.variables.add(loop_var)
        gen.arrays.discard(loop_var)

    def inline_source(name: str):
        """(parameter names, body expression, captured values) of an anonymous function variable."""
        kind = kinds.get(name, "-")
        if not kind.startswith("h:"):
            return None
        handle = frame.vars.get(name)
        if type(handle) is not FunctionHandle or not handle.is_anonymous:
            return None
        return list(handle.params), handle.body, handle.closure

    def make_inline(depth: int):
        def inline(name, args, ctx, outer):
            src = inline_source(name) if depth == 0 else None
            if src is None:
                return None
            params, body, closure = src
            if len(params) != len(args) or "varargin" in params or "~" in params \
                    or len(set(params)) != len(params):
                raise Reject
            child_scan = _Scan(allow_loops=False)
            child_scan.expr(body)
            child = _Gen(interp, closure.get("__functions__", {}), {p: "d" for p in params}, child_scan, "loop")
            child.variables = set(params)
            consts, blocked = {}, set()
            for k, v in closure.items():
                if k in params or k == "__functions__":
                    continue
                if type(v) is np.ndarray and v.shape == (1, 1) and v.dtype.kind in "fb":
                    x = v.item()
                    if v.dtype.kind == "b":
                        consts[k] = (repr(bool(x)), "b")
                    elif x != x:
                        consts[k] = ("_nan", "d")
                    elif x in (math.inf, -math.inf):
                        consts[k] = ("_inf" if x > 0 else "(-_inf)", "d")
                    else:
                        consts[k] = (f"({float(x)!r})", "d")
                else:
                    blocked.add(k)
            child.consts, child.blocked = consts, frozenset(blocked)
            child.inline = make_inline(depth + 1)
            code, typ = child.expr(body)
            if typ in "oO" or "_dp" in code:
                raise Reject
            outer.deps.update(child.deps)
            actual = ", ".join(outer.as_float(*outer.expr(a, ctx)) for a in args)
            return f"(lambda {', '.join('v_' + p for p in params)}: {code})({actual})", typ
        return inline

    gen.inline = make_inline(0)
    is_for = loop_var is not None

    def generate():
        if is_for:
            gen.types.setdefault(loop_var, set()).add(loop_kind)
            gen.emit(2, "for _i in range(_it0, _n):")
            # an integer range is computed in doubles and then converted, element by element (colon_range)
            value = "(_start + _step * _i)" if loop_kind == "d" else f"_s{loop_kind}(_start + _step * _i)"
            gen.emit(3, f"{gen.py(loop_var)} = {gen.convert(value, loop_kind, gen.var_type(loop_var))}")
            gen.block(stmt.body, 3, ())
        else:
            gen.emit(2, "while True:")
            gen.emit(3, "_p = -1")
            code, typ = gen.expr(stmt.cond)
            gen.emit(3, f"if not {gen.as_truth(code, typ)}:")
            gen.emit(4, "break")
            gen.block(stmt.body, 3, ())

    _infer(gen, kinds, generate)
    final = gen.final
    names = sorted(gen.variables)
    arr_names = sorted(gen.arrays)
    list_names = sorted(gen.lists)
    params = ["_it0", "_n", "_start", "_step", "_dp"] + [gen.py(n) for n in names]
    for n in arr_names:
        suffix = gen.py(n, "")
        params += [f"a{suffix}", f"n{suffix}", f"r{suffix}", f"c{suffix}"]
    params += [gen.py(n, "l") for n in list_names]
    values = "(" + "".join(f"{gen.py(n)}, " for n in names) + \
        "".join(f"{gen.py(n, 'l')}, {gen.py(n, 'm')}, " for n in list_names) + ")"
    name = _unique("_loop")
    src = [f"def {name}({', '.join(params)}):", "    _p = 0", "    _i = _it0"]
    src += [f"    {gen.py(n, 'm')} = False" for n in list_names]
    src += ["    try:"] + gen.lines
    src += ["    except BaseException as _e:",
            f"        return (1, _i, _p, {values}, _e)",
            f"    return (0, _i, 0, {values}, None)"]
    exec(compile("\n".join(src), "<mylab-jit>", "exec"), _NS)  # noqa: S102 - code generated above
    loop = _Loop()
    loop.fn = _NS.pop(name)
    loop.variables = [(n, gen.var_type(n)) for n in names]
    loop.arrays = [(n, n in gen.written_arrays) for n in arr_names]
    loop.lists = list_names
    loop.paths = gen.paths
    loop.final = final
    loop.deps = gen.deps
    loop.struct_fields = {f"{s}.{f}" for s, f in scan.fields_written}
    return loop


# -- running ----------------------------------------------------------------------------------

def _entry_value(frame, name: str):
    if "." in name:
        s, f = name.split(".", 1)
        struct = frame.vars.get(s)
        if type(struct) is not Struct or struct.data.shape != _S11:
            return None
        return struct.data[0, 0].get(f)
    return frame.vars.get(name)


def _prepare(interp, stmt, frame, loop_var, loop_kind="d"):
    """Compiled function and call arguments for this loop in its current state, or None.

    ``loop_kind``: the type of a for loop's variable ('d', or an integer class's type code).  What the
    variable held before the loop does not matter: the loop runs at least once and sets it first.
    """
    info = _static(stmt)
    if info is None or info["bails"] >= MAX_BAILS:
        return None
    scan: _Scan = info["scan"]
    if frame.globals and (any(n in frame.globals for n in scan.names) or loop_var in frame.globals):
        return None
    fvars = frame.vars
    kinds = {n: _kind(fvars.get(n)) for n in scan.names}
    for s, f in scan.fields:
        kinds[f"{s}.{f}"] = _kind(_entry_value(frame, f"{s}.{f}")) if kinds.get(s) == "S" else "-"
    for n, k in kinds.items():
        if k.startswith("h:"):
            body = fvars[n].body  # keep the body alive so its id() is never reused
            info.setdefault("keep", {})[id(body)] = body
    if loop_var:
        kinds[loop_var] = loop_kind
    sig = (id(frame.functions), tuple(sorted(kinds.items())))
    compiled = info["compiled"].get(sig, _MISSING)
    if compiled is not _MISSING and compiled is not None and not _deps_valid(interp, compiled.deps):
        compiled = _MISSING  # a function this loop calls now means something else
    if compiled is _MISSING:
        try:
            compiled = _compile(stmt, info, kinds, interp, frame, loop_var)
        except (Reject, KeyError, SyntaxError, RecursionError):
            compiled = None
        info["compiled"][sig] = compiled
        info.setdefault("keep_functions", []).append(frame.functions)
    if compiled is None:
        return None
    # structs whose fields are written get a private copy first (copy-on-write)
    if compiled.struct_fields:
        for s in {n.split(".", 1)[0] for n in compiled.struct_fields}:
            struct = fvars[s]
            if sys.getrefcount(struct) > _RC_DICT_AND_LOCAL or sys.getrefcount(struct.data[0, 0]) > _RC_ELEMENT:
                fvars[s] = struct.copy()
    args = []
    for n, typ in compiled.variables:
        v = None if n == loop_var else _entry_value(frame, n)
        if v is None:
            args.append(_UNSET)
        elif typ in "db" or typ in _IDT:
            args.append(v.item())
        else:
            args.append(v)
    for n, written in compiled.arrays:
        arr = _entry_value(frame, n)
        if written and (not arr.flags.writeable or arr.base is not None or sys.getrefcount(arr) > _RC_DICT_AND_LOCAL):
            arr = arr.copy()  # copy-on-write: this workspace gets its own array before modifying it
            if "." in n:
                s, f = n.split(".", 1)
                fvars[s].data[0, 0][f] = arr
            else:
                fvars[n] = arr
        args += [arr, arr.size, arr.shape[0], arr.shape[1]]
    for n in compiled.lists:
        data = fvars[n].data
        args.append(data[0].tolist() if data.size else [])
    _NS["_lim"][0] = interp.max_depth
    _INTERP[0] = interp
    return compiled, args, info


def _write_back(frame, compiled: _Loop, values) -> None:
    fvars = frame.vars
    k = 0
    for n, typ in compiled.variables:
        v = values[k]
        k += 1
        if v is _UNSET:
            continue
        if "." in n:
            if n not in compiled.struct_fields:
                continue
            s, f = n.split(".", 1)
            store, key = fvars[s].data[0, 0], f
        else:
            store, key = fvars, n
        if typ in _IDT:
            store[key] = _NS["_b" + typ](v)
        elif typ == "b" or type(v) is bool:
            store[key] = _bb(v)
        elif typ == "d" or type(v) is float:
            store[key] = _bd(v)
        else:
            store[key] = v
    for n in compiled.lists:
        lst, modified = values[k], values[k + 1]
        k += 2
        if modified:
            data = np.empty((1, len(lst)), dtype=object)
            data[0, :] = np.fromiter(lst, dtype=object, count=len(lst))
            fvars[n] = Cell(data)


def _exec_from(interp, body: list, path: tuple, frame) -> None:
    """Run ``body`` starting at the statement ``path`` points to (inside nested ifs if needed)."""
    k = path[0]
    if len(path) > 1:
        stmt = body[k]
        c = path[1]
        clause = stmt.clauses[c][1] if c < len(stmt.clauses) else stmt.orelse
        _exec_from(interp, clause, path[2:], frame)
        k += 1
    for stmt in body[k:]:
        interp.exec_stmt(stmt, frame)


def run_for(interp, stmt, frame) -> bool:
    """Try to run a ``for k = a:s:b`` loop compiled. Returns False if the interpreter must run it."""
    rng = stmt.iterable
    try:
        start = interp.eval(rng.start, frame)
        stop = interp.eval(rng.stop, frame)
        step = interp.eval(rng.step, frame) if rng.step is not None else None
    except Exception:  # noqa: BLE001 - let the interpreter evaluate it again and report the error
        return False
    kind = "d"  # or the type code of the integer class the range has (colon_range)
    for v in (start, stop, step):
        if v is None:
            continue
        if type(v) is not np.ndarray or v.shape != (1, 1):
            return False
        if v.dtype is not _F64:
            code = _DT_CODE.get(v.dtype)
            if code is None or (kind != "d" and code != kind):
                return False  # another class, or two integer classes (an error the interpreter reports)
            kind = code
    if kind in _WIDE and any(v is not None and abs(v.item()) > _EXACT_DOUBLE // 2 for v in (start, stop, step)):
        return False  # the range is computed in doubles: leave values that do not fit one to the interpreter
    a, b = float(start.item()), float(stop.item())
    s = 1.0 if step is None else float(step.item())
    if s == 0 or a != a or b != b or s != s or abs(a) == math.inf or abs(b) == math.inf:
        return False
    span = (b - a) / s
    if span < 0:
        return False
    n = int(math.floor(span + 1e-10 * max(1.0, abs(span)))) + 1
    if n < MIN_ITERATIONS:
        return False
    name = stmt.var.name
    prepared = _prepare(interp, stmt, frame, name, kind)
    if prepared is None:
        return False
    compiled, args, info = prepared
    status, i, p, values, exc = compiled.fn(0, n, a, s, interp.depth, *args)
    _write_back(frame, compiled, values)
    if status == 0:
        return True
    if not isinstance(exc, Exception):
        raise exc  # KeyboardInterrupt and the like
    # Bail-out: finish iteration i in the interpreter from statement p, then the remaining iterations.
    info["bails"] += 1
    try:
        try:
            _exec_from(interp, stmt.body, compiled.paths[p], frame)
        except ContinueSignal:
            pass
        for j in range(i + 1, n):
            if kind == "d":
                frame.set(name, np.array([[a + s * j]]))
            else:
                frame.set(name, _NS["_b" + kind](_NS["_s" + kind](a + s * j)))
            try:
                interp.exec_block(stmt.body, frame)
            except ContinueSignal:
                continue
    except BreakSignal:
        pass
    return True


def run_while(interp, stmt, frame) -> bool:
    """Try to run a ``while`` loop compiled. Returns False if the interpreter must run it."""
    prepared = _prepare(interp, stmt, frame, None)
    if prepared is None:
        return False
    compiled, args, info = prepared
    status, _i, p, values, exc = compiled.fn(0, 0, 0.0, 0.0, interp.depth, *args)
    _write_back(frame, compiled, values)
    if status == 0:
        return True
    if not isinstance(exc, Exception):
        raise exc
    info["bails"] += 1
    if p >= 0:  # bailed inside the body: finish this iteration first
        try:
            _exec_from(interp, stmt.body, compiled.paths[p], frame)
        except BreakSignal:
            return True
        except ContinueSignal:
            pass
    return False  # the interpreter's own while loop carries on from the current state
