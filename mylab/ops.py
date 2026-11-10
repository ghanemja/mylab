"""Operators: arithmetic, comparison, logical and concatenation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MatlabError
from .values import (
    Cell, FunctionHandle, MString, Struct, codes_to_char, is_char,
    saturate_int, simplify, to_numeric, to_str,
)

ELEMENTWISE = {"+", "-", ".*", "./", ".\\", ".^"}
COMPARE = {"==", "~=", "<", "<=", ">", ">="}


def _check_broadcast(a: np.ndarray, b: np.ndarray, op: str) -> None:
    for da, db in zip(a.shape, b.shape):
        if da != db and da != 1 and db != 1:
            raise MatlabError(
                f"Arrays have incompatible sizes for this operation ({a.shape[0]}x{a.shape[1]} "
                f"{op} {b.shape[0]}x{b.shape[1]})."
            )


def _result_type(a: np.ndarray, b: np.ndarray):
    """Pick the MATLAB result class for an arithmetic op (ints dominate, then single)."""
    ka, kb = a.dtype.kind, b.dtype.kind
    if ka in "iu" and kb in "iu" and a.dtype != b.dtype:
        raise MatlabError("Integers can only be combined with integers of the same class, or scalar doubles.")
    if ka in "iu":
        return a.dtype
    if kb in "iu":
        return b.dtype
    if a.dtype in (np.float32, np.complex64) or b.dtype in (np.float32, np.complex64):
        return np.float32
    return None


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
            return np.power(a.astype(float), b)
    return np.power(a.astype(complex), b)


def _divide(a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.divide(a, b)


def elementwise(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rtype = _result_type(a, b)
    if rtype is not None and np.dtype(rtype).kind in "iu":
        a = a.astype(float) if a.dtype.kind in "iu" else a
        b = b.astype(float) if b.dtype.kind in "iu" else b
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
    if a.shape[1] != b.shape[0]:
        raise MatlabError(
            "Incorrect dimensions for matrix multiplication. Check that the number of columns in the "
            "first matrix matches the number of rows in the second matrix. To operate on each element "
            "of the matrix individually, use TIMES (.*) for elementwise multiplication."
        )
    rtype = _result_type(a, b)
    return _finish(np.asarray(a, dtype=None if a.dtype.kind in "fc" else float)
                   @ np.asarray(b, dtype=None if b.dtype.kind in "fc" else float), rtype)


def _warn(msg: str) -> None:
    from .interpreter import current_interpreter
    interp = current_interpreter()
    if interp is not None:
        interp.warn(msg)


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
                _warn("Matrix is singular to working precision.")
                with np.errstate(all="ignore"):
                    return simplify(np.linalg.lstsq(af, bf, rcond=None)[0])
            return simplify(np.linalg.solve(af, bf))
        except np.linalg.LinAlgError:
            _warn("Matrix is singular to working precision.")
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
    if isinstance(a, MString) or isinstance(b, MString):
        if op not in ("==", "~=", "<", ">", "<=", ">="):
            raise MatlabError(f"Operator {op} not supported for strings.")
        sa, sb = to_str(a), to_str(b)
        r = {"==": sa == sb, "~=": sa != sb, "<": sa < sb, ">": sa > sb,
             "<=": sa <= sb, ">=": sa >= sb}[op]
        return np.array([[r]])
    x, y = to_numeric(a), to_numeric(b)
    _check_broadcast(x, y, op)
    if op in ("<", "<=", ">", ">="):
        x, y = np.real(x), np.real(y)
    with np.errstate(invalid="ignore"):
        r = {
            "==": np.equal, "~=": np.not_equal, "<": np.less, "<=": np.less_equal,
            ">": np.greater, ">=": np.greater_equal,
        }[op](x, y)
    return np.asarray(r, dtype=bool)


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
        x, y = _logical_operand(a), _logical_operand(b)
        _check_broadcast(x, y, op)
        return (x & y) if op == "&" else (x | y)
    if isinstance(a, MString) or isinstance(b, MString):
        if op == "+":
            return MString(_to_string_piece(a) + _to_string_piece(b))
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
        return -a
    return a  # unary plus


def transpose(v: Any, conjugate: bool) -> Any:
    if isinstance(v, Cell):
        return Cell(v.data.T.copy())
    if isinstance(v, Struct):
        return Struct(v.data.T.copy(), v.fields)
    if isinstance(v, (MString, FunctionHandle)):
        return v
    a = v
    if conjugate and a.dtype.kind == "c":
        return a.conj().T.copy()
    return a.T.copy()


# -- concatenation --------------------------------------------------------------

def _concat_arrays(items: list[np.ndarray], axis: int) -> np.ndarray:
    if not items:
        return np.zeros((0, 0))
    any_char = any(i.dtype.kind == "U" for i in items)
    nonempty = [i for i in items if i.size > 0]
    if not nonempty:
        return np.zeros((0, 0), dtype="<U1") if any_char else items[0]
    if any_char:
        nonempty = [codes_to_char(i) if i.dtype.kind != "U" else i for i in nonempty]
    else:
        int_items = [i for i in nonempty if i.dtype.kind in "iu"]
        if int_items:
            dt = int_items[0].dtype
            nonempty = [saturate_int(i, dt) if i.dtype != dt else i for i in nonempty]
        elif not all(i.dtype == bool for i in nonempty):
            nonempty = [i.astype(float) if i.dtype == bool else i for i in nonempty]
    other = 1 - axis
    first = nonempty[0].shape[other]
    for i in nonempty[1:]:
        if i.shape[other] != first:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.")
    return np.concatenate(nonempty, axis=axis)


def _to_cell(v: Any) -> Cell:
    if isinstance(v, Cell):
        return v
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = v
    return Cell(data)


def _concat_cells(items: list[Cell], axis: int) -> Cell:
    nonempty = [c.data for c in items if c.data.size > 0]
    if not nonempty:
        return Cell(np.empty((0, 0), dtype=object))
    other = 1 - axis
    first = nonempty[0].shape[other]
    for d in nonempty[1:]:
        if d.shape[other] != first:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.")
    return Cell(np.concatenate(nonempty, axis=axis))


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
    """Concatenate along ``axis`` (1 = horizontal, 0 = vertical)."""
    if len(items) == 1:
        return items[0]
    if not items:
        return np.zeros((0, 0))
    if any(isinstance(i, Cell) for i in items):
        return _concat_cells([_to_cell(i) for i in items], axis)
    if any(isinstance(i, Struct) for i in items):
        real = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if not all(isinstance(i, Struct) for i in real):
            raise MatlabError("Cannot concatenate a struct with a non-struct value.")
        return _concat_structs(real, axis)
    if any(isinstance(i, MString) for i in items):
        pieces = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if len(pieces) == 1:
            return pieces[0]
        raise MatlabError("String arrays are not supported yet; use + to join strings or a cell array {...}.")
    if any(isinstance(i, FunctionHandle) for i in items):
        raise MatlabError("Function handles cannot be concatenated; use a cell array {...}.")
    return _concat_arrays([np.asarray(i) for i in items], axis)


def build_matrix(rows: list[list[Any]]) -> Any:
    row_values = [concat(r, 1) for r in rows]
    if not row_values:
        return np.zeros((0, 0))
    return concat(row_values, 0)

