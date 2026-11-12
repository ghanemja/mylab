"""Operators: arithmetic, comparison, logical and concatenation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MatlabError
from .values import (
    Cell, FunctionHandle, MString, StringArray, Struct, as_2d, codes_to_char, is_char,
    WIDE_FLOAT, saturate_int, simplify, string_data, string_result, to_numeric, to_str,
)


def _is_str(v: Any) -> bool:
    return isinstance(v, (MString, StringArray))


def _string_binop(op: str, a: Any, b: Any) -> Any:
    """Elementwise + (join) and comparisons on string arrays, with implicit expansion."""
    x, y = align(string_data(a), string_data(b))
    _check_broadcast(x, y, op)
    xb, yb = np.broadcast_arrays(x, y)
    out_shape = xb.shape
    if op == "+":
        out = np.empty(out_shape, dtype=object)
        for idx in np.ndindex(out_shape):
            p, q = xb[idx], yb[idx]
            out[idx] = None if p is None or q is None else p + q
        return string_result(out)
    fns = {"==": lambda p, q: p == q, "~=": lambda p, q: p != q, "<": lambda p, q: p < q,
           ">": lambda p, q: p > q, "<=": lambda p, q: p <= q, ">=": lambda p, q: p >= q}
    if op not in fns:
        raise MatlabError(f"Operator '{op}' is not supported for operands of type 'string'.")
    res = np.zeros(out_shape, dtype=bool)
    for idx in np.ndindex(out_shape):
        p, q = xb[idx], yb[idx]
        res[idx] = (op == "~=") if (p is None or q is None) else fns[op](p, q)
    return as_2d(res)

ELEMENTWISE = {"+", "-", ".*", "./", ".\\", ".^"}
COMPARE = {"==", "~=", "<", "<=", ">", ">="}


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the lower-dimensional operand with trailing singleton dims (MATLAB expansion rule)."""
    if a.ndim == b.ndim:
        return a, b
    if a.ndim < b.ndim:
        return a.reshape(a.shape + (1,) * (b.ndim - a.ndim)), b
    return a, b.reshape(b.shape + (1,) * (a.ndim - b.ndim))


def _dims(shape) -> str:
    return "x".join(str(d) for d in shape)


def _check_broadcast(a: np.ndarray, b: np.ndarray, op: str) -> None:
    for da, db in zip(a.shape, b.shape):
        if da != db and da != 1 and db != 1:
            raise MatlabError(
                f"Arrays have incompatible sizes for this operation ({_dims(a.shape)} {op} {_dims(b.shape)}).",
                "MATLAB:sizeDimensionsMustMatch",
            )


_SINGLE = (np.dtype(np.float32), np.dtype(np.complex64))
_INT_MIX_MSG = "Integers can only be combined with integers of the same class, or scalar doubles."


def _result_type(a: np.ndarray, b: np.ndarray):
    """Pick the MATLAB result class for an arithmetic op (ints dominate, then single)."""
    ka, kb = a.dtype.kind, b.dtype.kind
    if ka in "iu" or kb in "iu":
        if ka in "iu" and kb in "iu":
            if a.dtype != b.dtype:
                raise MatlabError(_INT_MIX_MSG, "MATLAB:mixedClasses")
        return a.dtype if ka in "iu" else b.dtype
    if a.dtype in _SINGLE or b.dtype in _SINGLE:
        return np.float32
    return None


def check_integer_mix(a: np.ndarray, b: np.ndarray) -> None:
    """Raise MATLAB's error for an integer combined with another integer class."""
    ka, kb = a.dtype.kind, b.dtype.kind
    if ka in "iu" and kb in "iu" and a.dtype != b.dtype:
        raise MatlabError(_INT_MIX_MSG, "MATLAB:mixedClasses")


def _exact(a: np.ndarray):
    """Object array of exact Python numbers (ints and Fractions), or None if not possible."""
    if a.dtype.kind in "iub":
        return a.astype(object)
    if a.dtype.kind != "f" or not np.all(np.isfinite(a)):
        return None
    from fractions import Fraction
    out = np.empty(a.shape, dtype=object)
    flat = out.reshape(-1)
    for i, v in enumerate(a.reshape(-1).tolist()):
        flat[i] = int(v) if v == int(v) else Fraction(v)
    return out


def _round_half_away(v) -> int:
    n, d = (v.numerator, v.denominator) if not isinstance(v, int) else (v, 1)
    q, r = divmod(abs(n), d)
    q += 2 * r >= d
    return q if n >= 0 else -q


def _exact_int64(op: str, a: np.ndarray, b: np.ndarray, rtype):
    """64-bit integer arithmetic without a wide float type: exact rational arithmetic."""
    from fractions import Fraction
    x, y = _exact(a), _exact(b)
    if x is None or y is None:
        return None
    if op == ".\\":
        x, y, op = y, x, "./"
    if op == "./":
        if any(v == 0 for v in y.flat):
            return None
        x = np.frompyfunc(Fraction, 1, 1)(x)
    r = x + y if op == "+" else x - y if op == "-" else x * y if op == ".*" else x / y
    info = np.iinfo(rtype)
    out = np.empty(np.shape(r), dtype=rtype)
    flat = out.reshape(-1)
    for i, v in enumerate(np.asarray(r, dtype=object).reshape(-1)):
        flat[i] = min(max(_round_half_away(v), info.min), info.max)
    return out


def to_single(a: np.ndarray) -> np.ndarray:
    if a.dtype in _SINGLE:
        return a
    return a.astype(np.complex64 if a.dtype.kind == "c" else np.float32)


def _finish(result: np.ndarray, rtype) -> np.ndarray:
    if rtype is None:
        return simplify(np.asarray(result))
    if np.dtype(rtype).kind in "iu":
        return saturate_int(result, rtype)
    r = simplify(np.asarray(result))
    return r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)


def _power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Negative base with fractional exponent gives a complex result in MATLAB.
    if a.dtype.kind != "c" and b.dtype.kind != "c":
        if np.any((a < 0) & (np.mod(b, 1) != 0)):
            return np.power(a.astype(complex), b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.power(a if a.dtype.kind == "f" else a.astype(float), b)
    return np.power(a.astype(complex), b)


def _divide(a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.divide(a, b)


def elementwise(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = align(a, b)
    rtype = _result_type(a, b)
    if (rtype is not None and WIDE_FLOAT is np.float64 and op != ".^"
            and np.dtype(rtype).kind in "iu" and np.dtype(rtype).itemsize == 8):
        _check_broadcast(a, b, op)
        exact = _exact_int64(op, a, b, rtype)
        if exact is not None:
            return exact
    if rtype is not None:
        if np.dtype(rtype).kind in "iu":
            # Integer arithmetic is done in floating point and then rounded and saturated;
            # 64-bit integers need the wider float type to keep every digit.
            wide = WIDE_FLOAT if np.dtype(rtype).itemsize == 8 else float
            a = a.astype(wide) if a.dtype.kind in "iu" or (wide is not float and a.dtype.kind == "f") else a
            b = b.astype(wide) if b.dtype.kind in "iu" or (wide is not float and b.dtype.kind == "f") else b
        else:
            a, b = to_single(a), to_single(b)
    _check_broadcast(a, b, op)
    with np.errstate(over="ignore", invalid="ignore"):
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == ".*":
            r = a * b
        elif op == "./":
            r = _divide(a, b)
        elif op == ".\\":
            r = _divide(b, a)
        elif op == ".^":
            r = _power(a, b)
        else:  # pragma: no cover
            raise MatlabError(f"Unknown operator {op}")
    return _finish(r, rtype)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 1 or b.size == 1:
        return elementwise(".*", a, b)
    if a.ndim > 2 or b.ndim > 2:
        raise MatlabError("Matrix multiplication is only defined for 2-D arrays; use .* or pagemtimes.")
    if a.shape[1] != b.shape[0]:
        raise MatlabError(
            "Incorrect dimensions for matrix multiplication. Check that the number of columns in the "
            "first matrix matches the number of rows in the second matrix. To operate on each element "
            "of the matrix individually, use TIMES (.*) for elementwise multiplication.",
            "MATLAB:innerdim",
        )
    rtype = _result_type(a, b)
    return _finish(np.asarray(a, dtype=None if a.dtype.kind in "fc" else float)
                   @ np.asarray(b, dtype=None if b.dtype.kind in "fc" else float), rtype)


def _warn(msg: str, identifier: str = "") -> None:
    from .interpreter import current_interpreter
    interp = current_interpreter()
    if interp is not None:
        interp.warn(msg, identifier)


def left_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a \\ b``: solve ``a * x = b``."""
    if a.size == 1:
        return elementwise(".\\", a, b)
    if a.shape[0] != b.shape[0]:
        raise MatlabError("Matrix dimensions must agree.")
    af = a.astype(complex if a.dtype.kind == "c" else float)
    bf = b.astype(complex if b.dtype.kind == "c" else float)
    if a.shape[0] == a.shape[1]:
        try:
            cond = np.linalg.cond(af)
            if not np.isfinite(cond) or cond > 1 / np.finfo(float).eps:
                _warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
                with np.errstate(all="ignore"):
                    return simplify(np.linalg.lstsq(af, bf, rcond=None)[0])
            return simplify(np.linalg.solve(af, bf) + 0.0)  # + 0.0: no negative zeros from the solver
        except np.linalg.LinAlgError:
            _warn("Matrix is singular to working precision.", "MATLAB:singularMatrix")
            return np.full((a.shape[1], b.shape[1]), np.inf)
    return simplify(np.linalg.lstsq(af, bf, rcond=None)[0])


def right_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a / b``: solve ``x * b = a``."""
    if b.size == 1:
        return elementwise("./", a, b)
    if a.shape[1] != b.shape[1]:
        raise MatlabError("Matrix dimensions must agree.")
    return left_divide(b.T, a.T).T


def matpower(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 1 and b.size == 1:
        return elementwise(".^", a, b)
    if b.size == 1 and a.shape[0] == a.shape[1]:
        p = b.flat[0]
        if np.isreal(p) and float(np.real(p)) == int(np.real(p)):
            p = int(np.real(p))
            if p >= 0:
                return simplify(np.linalg.matrix_power(a.astype(float) if a.dtype.kind != "c" else a, p))
            return simplify(np.linalg.matrix_power(np.linalg.inv(a), -p))
        import scipy.linalg
        return simplify(np.asarray(scipy.linalg.fractional_matrix_power(a, p)))
    if a.size == 1 and b.shape[0] == b.shape[1]:
        import scipy.linalg
        return simplify(scipy.linalg.expm(np.log(complex(a.flat[0])) * b))
    raise MatlabError("Incorrect dimensions for raising a matrix to a power. Check that the matrix is "
                      "square and the power is a scalar. To operate on each element of the matrix "
                      "individually, use POWER (.^) for elementwise power.")


def compare(op: str, a: Any, b: Any) -> np.ndarray:
    if _is_str(a) or _is_str(b):
        return _string_binop(op, a, b)
    x, y = align(to_numeric(a), to_numeric(b))
    _check_broadcast(x, y, op)
    if x.dtype != y.dtype:
        x, y = _common_for_compare(x, y)
    if op in ("<", "<=", ">", ">="):
        x, y = np.real(x), np.real(y)
    with np.errstate(invalid="ignore"):
        r = {
            "==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal,
            ">": np.greater, ">=": np.greater_equal,
        }[op](x, y)
    return np.asarray(r, dtype=bool)


def _common_for_compare(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bring operands of different classes to one class the way MATLAB compares them."""
    kx, ky = x.dtype.kind, y.dtype.kind
    if kx in "iu" or ky in "iu":
        # Relational operators accept any mix of numeric classes and compare the exact values.
        if any(v.dtype.kind in "iu" and v.dtype.itemsize == 8 for v in (x, y)):
            if WIDE_FLOAT is not np.float64:
                return x.astype(WIDE_FLOAT), y.astype(WIDE_FLOAT)  # exact for 64-bit integers
            ex, ey = _exact(x), _exact(y)
            if ex is not None and ey is not None:
                return ex, ey
        return x, y
    if x.dtype in _SINGLE or y.dtype in _SINGLE:
        return to_single(x), to_single(y)  # a double is rounded to single before the comparison
    return x, y


def _logical_operand(v: Any) -> np.ndarray:
    a = to_numeric(v)
    if a.dtype.kind == "f" and np.isnan(a).any():
        raise MatlabError("NaN's cannot be converted to logicals.")
    return a != 0


_F64 = np.dtype(float)


def _scalar_fast(op: str, x: float, y: float):
    """Plain-float arithmetic for 1x1 doubles; returns None to fall back to numpy."""
    if op in ("+",):
        return x + y
    if op == "-":
        return x - y
    if op in ("*", ".*"):
        return x * y
    if op in ("/", "./"):
        return x / y if y != 0 else None
    if op in ("\\", ".\\"):
        return y / x if x != 0 else None
    if op in ("^", ".^"):
        if x < 0 and y != int(y):
            return None
        try:
            r = x ** y
        except (OverflowError, ZeroDivisionError):
            return None
        return r if isinstance(r, float) else None
    if op == "==":
        return x == y
    if op == "~=":
        return x != y
    if op == "<":
        return x < y
    if op == "<=":
        return x <= y
    if op == ">":
        return x > y
    if op == ">=":
        return x >= y
    return None


def binop(op: str, a: Any, b: Any) -> Any:
    if (type(a) is np.ndarray and type(b) is np.ndarray and a.shape == (1, 1) and b.shape == (1, 1)
            and a.dtype is _F64 and b.dtype is _F64):
        r = _scalar_fast(op, float(a[0, 0]), float(b[0, 0]))
        if r is not None:
            return np.array([[r]], dtype=bool if isinstance(r, bool) else float)
    if op in COMPARE:
        return compare(op, a, b)
    if op in ("&", "|"):
        x, y = align(_logical_operand(a), _logical_operand(b))
        _check_broadcast(x, y, op)
        return (x & y) if op == "&" else (x | y)
    if _is_str(a) or _is_str(b):
        if op == "+":
            return _string_binop("+", a, b)
        raise MatlabError(f"Operator '{op}' is not supported for operands of type 'string'.")
    x, y = to_numeric(a), to_numeric(b)
    if op in ELEMENTWISE:
        return elementwise(op, x, y)
    if op == "*":
        return matmul(x, y)
    if op == "/":
        return right_divide(x, y)
    if op == "\\":
        return left_divide(x, y)
    if op == "^":
        return matpower(x, y)
    raise MatlabError(f"Unknown operator '{op}'")


def _to_string_piece(v: Any) -> str:
    if isinstance(v, MString):
        return str(v)
    if is_char(v):
        return to_str(v)
    if isinstance(v, np.ndarray) and v.size == 1:
        from .display import format_scalar_plain
        return format_scalar_plain(v.flat[0])
    raise MatlabError("Cannot convert value to string.")


def unary(op: str, v: Any) -> Any:
    if op == "~":
        return ~_logical_operand(v)
    a = to_numeric(v)
    if op == "-":
        if a.dtype.kind == "u":
            return np.zeros_like(a)
        if a.dtype.kind == "i":
            info = np.iinfo(a.dtype)
            return np.where(a == info.min, info.max, -a).astype(a.dtype)
        return -a
    return a  # unary plus


def transpose(v: Any, conjugate: bool) -> Any:
    if isinstance(v, Cell):
        return Cell(v.data.T.copy())
    if isinstance(v, Struct):
        return Struct(v.data.T.copy(), v.fields)
    if isinstance(v, (MString, FunctionHandle)):
        return v
    if isinstance(v, StringArray):
        if v.data.ndim > 2:
            raise MatlabError("Transpose on N-D arrays is not defined. Use permute instead.")
        return StringArray(v.data.T.copy())
    a = v
    if a.ndim > 2:
        raise MatlabError("Transpose on N-D arrays is not defined. Use permute instead.")
    if conjugate and a.dtype.kind == "c":
        return a.conj().T.copy()
    return a.T.copy()


# -- concatenation --------------------------------------------------------------

def _concat_arrays(items: list[np.ndarray], axis: int) -> np.ndarray:
    if not items:
        return np.zeros((0, 0))
    # The class of the result depends on every operand, the empty ones included:
    # char wins, then the leftmost integer class, then single; all-logical stays logical.
    any_char = any(i.dtype.kind == "U" for i in items)
    int_dt = next((i.dtype for i in items if i.dtype.kind in "iu"), None)
    if any_char:
        conv = lambda i: codes_to_char(i) if i.dtype.kind != "U" else i  # noqa: E731
    elif int_dt is not None:
        conv = lambda i: saturate_int(i, int_dt) if i.dtype != int_dt else i  # noqa: E731
    elif any(i.dtype in _SINGLE for i in items):
        conv = to_single
    elif all(i.dtype == bool for i in items):
        conv = lambda i: i  # noqa: E731
    else:
        conv = lambda i: i.astype(float) if i.dtype.kind not in "fc" else i  # noqa: E731
    # Empty operands are dropped when anything non-empty is present ([zeros(1,0); 5] is 5).
    # When everything is empty, all but [] (0x0) take part, so [zeros(0,1) zeros(0,1)] is 0x2.
    nonempty = [i for i in items if i.size > 0]
    if not nonempty:
        nonempty = [i for i in items if i.shape != (0, 0)]
        if not nonempty:
            return conv(items[0])
    nd = max([axis + 1] + [i.ndim for i in nonempty])
    nonempty = [i.reshape(i.shape + (1,) * (nd - i.ndim)) if i.ndim < nd else i for i in nonempty]
    nonempty = [conv(i) for i in nonempty]
    first = nonempty[0].shape[:axis] + nonempty[0].shape[axis + 1:]
    for i in nonempty[1:]:
        if i.shape[:axis] + i.shape[axis + 1:] != first:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch")
    return as_2d(np.concatenate(nonempty, axis=axis))


def _to_cell(v: Any) -> Cell:
    if isinstance(v, Cell):
        return v
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = v
    return Cell(data)


def _concat_cells(items: list[Cell], axis: int) -> Cell:
    nonempty = [c.data for c in items if c.data.size > 0]
    if not nonempty:
        # Only empties: 0x0 ones are ignored, the rest keep their shape ([cell(1,0), cell(1,0)] is 1x0).
        shaped = [c.data for c in items if c.data.shape != (0, 0)]
        try:
            return Cell(np.concatenate(shaped, axis=axis)) if shaped else Cell(np.empty((0, 0), dtype=object))
        except ValueError:
            return Cell(np.empty((0, 0), dtype=object))
    nd = max([axis + 1] + [d.ndim for d in nonempty])
    nonempty = [d.reshape(d.shape + (1,) * (nd - d.ndim)) if d.ndim < nd else d for d in nonempty]
    first = nonempty[0].shape[:axis] + nonempty[0].shape[axis + 1:]
    for d in nonempty[1:]:
        if d.shape[:axis] + d.shape[axis + 1:] != first:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch")
    return Cell(as_2d(np.concatenate(nonempty, axis=axis)))


def _concat_structs(items: list[Struct], axis: int) -> Struct:
    nonempty = [s for s in items if s.data.size > 0]
    if not nonempty:
        return items[0]
    fields = nonempty[0].fields
    for s in nonempty[1:]:
        if set(s.fields) != set(fields):
            raise MatlabError("Concatenation of structs requires the same field names.")
    data = np.concatenate([s.copy().data for s in nonempty], axis=axis)
    return Struct(data, fields)


def concat(items: list[Any], axis: int) -> Any:
    """Concatenate along numpy ``axis`` (1 = horizontal, 0 = vertical, 2 = third dim, ...)."""
    if len(items) == 1:
        return items[0]
    if not items:
        return np.zeros((0, 0))
    for i in items:
        if hasattr(type(i), "concat") and hasattr(i, "isa_class"):
            return type(i).concat(items, axis)
    if any(hasattr(i, "isa_class") for i in items):
        kinds = sorted({i.class_name() for i in items if hasattr(i, "isa_class")})
        raise MatlabError(f"Objects of class {', '.join(kinds)} cannot be concatenated into an array; "
                          "store them in a cell array instead, e.g. {obj1, obj2}.")
    if any(isinstance(i, Cell) for i in items):
        return _concat_cells([_to_cell(i) for i in items], axis)
    if any(isinstance(i, Struct) for i in items):
        real = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if not all(isinstance(i, Struct) for i in real):
            raise MatlabError("Cannot concatenate a struct with a non-struct value.")
        return _concat_structs(real, axis)
    if any(_is_str(i) for i in items):
        pieces = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0 and not is_char(i))]
        if any(isinstance(i, (Cell, Struct, FunctionHandle)) for i in pieces):
            raise MatlabError("Cannot concatenate a string with a cell, struct or function handle.")
        datas = [string_data(i) for i in pieces]
        nd = max([axis + 1] + [d.ndim for d in datas])
        datas = [d.reshape(d.shape + (1,) * (nd - d.ndim)) if d.ndim < nd else d for d in datas]
        first = datas[0].shape[:axis] + datas[0].shape[axis + 1:]
        for d in datas[1:]:
            if d.shape[:axis] + d.shape[axis + 1:] != first:
                raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch")
        return string_result(as_2d(np.concatenate(datas, axis=axis)))
    if any(isinstance(i, FunctionHandle) for i in items):
        raise MatlabError("Function handles cannot be concatenated; use a cell array {...}.")
    return _concat_arrays([np.asarray(i) for i in items], axis)


def build_matrix(rows: list[list[Any]]) -> Any:
    row_values = [concat(r, 1) for r in rows]
    if not row_values:
        return np.zeros((0, 0))
    return concat(row_values, 0)

