"""Runtime value types and conversions.

Every MATLAB value is represented by one of:

* ``numpy.ndarray`` (always 2-D) for numeric, logical and char arrays.
  Char arrays use dtype ``<U1`` - one character per element - so indexing,
  concatenation and reshaping work exactly like numeric arrays.
* :class:`MString` for ``"double quoted"`` string scalars.
* :class:`Cell` for cell arrays.
* :class:`Struct` for structs and struct arrays.
* :class:`FunctionHandle` for ``@name`` and ``@(x) ...`` values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MatlabError

INT_TYPES = {
    "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64,
}


class MString(str):
    """A MATLAB string scalar ("double quoted")."""


class StringArray:
    """A non-scalar string array: an object ndarray of Python str (None = <missing>).

    1x1 string arrays are represented by :class:`MString`; use :func:`string_result`
    to build a value from data so that scalars collapse to ``MString``.
    """

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.data = data

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def values_f(self) -> list:
        return list(self.data.flatten(order="F"))


def string_result(data: np.ndarray):
    """Wrap an object array of str as a MATLAB string value (scalar -> MString)."""
    data = np.asarray(data, dtype=object)
    if data.ndim < 2:
        data = data.reshape(1, -1) if data.ndim == 1 else data.reshape(1, 1)
    if data.size == 1 and data.flat[0] is not None:
        return MString(data.flat[0])
    return StringArray(data)


def string_data(v) -> np.ndarray:
    """Object array of str for a string scalar/array, char row, or cellstr."""
    if isinstance(v, StringArray):
        return v.data
    if isinstance(v, MString):
        d = np.empty((1, 1), dtype=object)
        d[0, 0] = str(v)
        return d
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        d = np.empty((1, 1), dtype=object)
        if v.ndim == 2 and v.shape[0] > 1:
            d = np.empty((v.shape[0], 1), dtype=object)
            for i in range(v.shape[0]):
                d[i, 0] = "".join(v[i])
            return d
        d[0, 0] = "".join(v.flatten(order="F")) if v.size else ""
        return d
    if isinstance(v, Cell):
        d = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            d[idx] = to_str(v.data[idx], "cell element")
        return d
    if isinstance(v, np.ndarray):
        from .display import format_scalar_plain
        d = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            x = v[idx]
            d[idx] = None if (v.dtype.kind == "f" and np.isnan(x)) else format_scalar_plain(x)
        return d
    raise MatlabError(f"Cannot convert a value of type {class_name(v)} to string.")


class Cell:
    """A cell array: a 2-D numpy object array holding arbitrary values."""

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.data = data

    @classmethod
    def empty(cls, rows: int = 0, cols: int = 0) -> "Cell":
        data = np.empty((rows, cols), dtype=object)
        for idx in np.ndindex(data.shape):
            data[idx] = empty_double()
        return cls(data)

    @classmethod
    def from_list(cls, items: list, column: bool = False) -> "Cell":
        data = np.empty((len(items), 1) if column else (1, len(items)), dtype=object)
        for i, v in enumerate(items):
            data.flat[i] = v
        return cls(data)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def values_f(self) -> list:
        """Elements in column-major order."""
        return list(self.data.flatten(order="F"))


class Struct:
    """A struct array. A plain ``s.a = 1`` struct is a 1x1 struct array."""

    __slots__ = ("data", "fields")

    def __init__(self, data: np.ndarray | None = None, fields: list[str] | None = None):
        if data is None:
            data = np.empty((1, 1), dtype=object)
            data[0, 0] = {}
        self.data = data
        self.fields = list(fields or [])

    @classmethod
    def scalar(cls, mapping: dict | None = None) -> "Struct":
        s = cls()
        if mapping:
            for k, v in mapping.items():
                s.set_field(0, k, v)
        return s

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def numel(self) -> int:
        return self.data.size

    def add_field(self, name: str) -> None:
        if name not in self.fields:
            self.fields.append(name)
            for idx in np.ndindex(self.data.shape):
                self.data[idx].setdefault(name, empty_double())

    def get_field(self, i: int, name: str) -> Any:
        if name not in self.fields:
            raise MatlabError(f"Unrecognized field name \"{name}\".")
        return self._elem_f(i)[name]

    def set_field(self, i: int, name: str, value: Any) -> None:
        self.add_field(name)
        if isinstance(value, (bool, int, float, complex, str, np.generic)) and not isinstance(value, MString):
            value = to_matlab(value)
        self._elem_f(i)[name] = value

    def _elem_f(self, i: int) -> dict:
        rows = self.data.shape[0]
        return self.data[i % rows, i // rows]

    def copy(self) -> "Struct":
        data = np.empty(self.data.shape, dtype=object)
        for idx in np.ndindex(self.data.shape):
            data[idx] = dict(self.data[idx])
        return Struct(data, self.fields)


class FunctionHandle:
    """``@name`` or an anonymous function ``@(x) expr``."""

    def __init__(self, name: str | None = None, params: list[str] | None = None,
                 body: Any = None, closure: dict | None = None, text: str = ""):
        self.name = name
        self.params = params or []
        self.body = body
        self.closure = closure or {}
        self.text = text

    @property
    def is_anonymous(self) -> bool:
        return self.name is None

    def __repr__(self) -> str:
        return self.text if self.is_anonymous else "@" + str(self.name)


# -- constructors ---------------------------------------------------------------

def empty_double() -> np.ndarray:
    return np.zeros((0, 0))


def scalar(x: Any) -> np.ndarray:
    return np.array([[x]], dtype=complex if isinstance(x, complex) else float)


def logical(x: Any) -> np.ndarray:
    return np.array(x, dtype=bool).reshape((1, 1)) if np.ndim(x) == 0 else np.atleast_2d(np.array(x, dtype=bool))


def char(s: str) -> np.ndarray:
    if s == "":
        return np.zeros((0, 0), dtype="<U1")
    return np.array([list(s)], dtype="<U1")


def to_matlab(x: Any) -> Any:
    """Convert a Python value into a MATLAB value."""
    if hasattr(x, "isa_class"):  # MObject instances pass through unchanged
        return x
    if isinstance(x, (np.ndarray, Cell, Struct, FunctionHandle, MString, StringArray)):
        if isinstance(x, np.ndarray):
            return as_2d(x)
        return x
    if isinstance(x, bool) or isinstance(x, np.bool_):
        return np.array([[bool(x)]])
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([[float(x)]])
    if isinstance(x, (complex, np.complexfloating)):
        return np.array([[complex(x)]])
    if isinstance(x, str):
        return char(x)
    if isinstance(x, (list, tuple)):
        return as_2d(np.array(x, dtype=float))
    if x is None:
        return empty_double()
    raise MatlabError(f"Cannot convert Python value of type {type(x).__name__}")


def as_2d(a: np.ndarray) -> np.ndarray:
    """Canonical MATLAB shape: at least 2-D, no trailing singleton dims beyond the 2nd."""
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim > 2 and a.shape[-1] == 1:
        shape = list(a.shape)
        while len(shape) > 2 and shape[-1] == 1:
            shape.pop()
        return a.reshape(shape)
    return a


# -- predicates -----------------------------------------------------------------

def is_char(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind == "U"


def is_numeric_array(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind in "biufc"


def is_logical(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.dtype == bool


def is_string_like(v: Any) -> bool:
    return isinstance(v, MString) or (is_char(v) and (v.shape[0] <= 1))


def is_empty(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        return v.size == 0
    if isinstance(v, (Cell, Struct, StringArray)):
        return v.data.size == 0
    if hasattr(v, "matlab_shape"):
        return 0 in v.matlab_shape()
    return False


def shape_of(v: Any) -> tuple[int, ...]:
    """Full MATLAB size (at least two dimensions)."""
    if isinstance(v, np.ndarray):
        return v.shape
    if isinstance(v, (Cell, Struct, StringArray)):
        return v.data.shape
    if hasattr(v, "matlab_shape"):
        return v.matlab_shape()
    return (1, 1)


def numel(v: Any) -> int:
    n = 1
    for d in shape_of(v):
        n *= d
    return n


def bool_array(flags, shape) -> np.ndarray:
    return np.array(flags, dtype=bool).reshape(shape, order="F")


def class_name(v: Any) -> str:
    if isinstance(v, (MString, StringArray)):
        return "string"
    if hasattr(v, "isa_class"):  # MObject (Map, table, classdef instances)
        return v.class_name()
    if isinstance(v, Cell):
        return "cell"
    if isinstance(v, Struct):
        return "struct"
    if isinstance(v, FunctionHandle):
        return "function_handle"
    if isinstance(v, np.ndarray):
        k = v.dtype
        if k.kind == "b":
            return "logical"
        if k.kind == "U":
            return "char"
        if k == np.float32 or k == np.complex64:
            return "single"
        if k.kind in "iu":
            return k.name
        return "double"
    return type(v).__name__


# -- conversions -----------------------------------------------------------------

def to_str(v: Any, what: str = "argument") -> str:
    """Convert a char row vector or string scalar to a Python string."""
    if isinstance(v, MString):
        return str(v)
    if is_char(v):
        if v.size == 0:
            return ""
        if v.shape[0] != 1:
            return "".join(v.flatten(order="F"))
        return "".join(v[0])
    if isinstance(v, str):
        return v
    raise MatlabError(f"Expected {what} to be a character vector or string.")


def char_codes(v: np.ndarray) -> np.ndarray:
    """Char array -> float array of character codes (same shape)."""
    if v.size == 0:
        return np.zeros(v.shape)
    # numpy drops trailing NULs, so char(0) is stored as '' - map it back to 0.
    return np.vectorize(lambda c: ord(c) if c else 0, otypes=[float])(v)


def codes_to_char(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.size == 0:
        return np.zeros(a.shape, dtype="<U1")
    if a.dtype.kind == "U":
        return a
    a = np.real(a)
    return np.vectorize(lambda c: chr(int(c)) if c >= 0 else "\0", otypes=["<U1"])(a)


def to_numeric(v: Any, what: str = "operand") -> np.ndarray:
    """Convert any array-like value to a numeric ndarray (chars -> codes)."""
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return char_codes(v)
        if v.dtype == bool:
            return v.astype(float)
        return v
    if isinstance(v, (MString, StringArray)):
        raise MatlabError("Arithmetic on strings is not supported; use double() or str2double().")
    if isinstance(v, Cell):
        raise MatlabError(f"Undefined operation for a cell array {what}. Use {{}} to get the contents.")
    if isinstance(v, Struct):
        raise MatlabError(f"Undefined operation for a struct {what}.")
    if isinstance(v, FunctionHandle):
        raise MatlabError(f"Undefined operation for a function handle {what}.")
    if hasattr(v, "to_dense"):  # objects that stand for a numeric array (sparse matrices)
        return to_numeric(v.to_dense(), what)
    if hasattr(v, "numeric_error"):  # objects that must not be silently treated as numbers
        raise MatlabError(v.numeric_error(what))
    return to_matlab(v)


def to_float_array(v: Any) -> np.ndarray:
    a = to_numeric(v)
    if a.dtype.kind in "iu":
        return a.astype(float)
    return a


def to_scalar(v: Any, what: str = "value") -> float:
    a = to_numeric(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    x = a.flat[0]
    if isinstance(x, complex) or np.iscomplexobj(a):
        return float(np.real(x))
    return float(x)


def to_int(v: Any, what: str = "value") -> int:
    x = to_scalar(v, what)
    if not np.isfinite(x) or x != int(x):
        raise MatlabError(f"Expected {what} to be an integer.")
    return int(x)


def is_true(v: Any) -> bool:
    """MATLAB truthiness used by ``if``/``while``: non-empty and all nonzero."""
    if type(v) is np.ndarray and v.size == 1 and v.dtype.kind in "bf":
        x = v.item()
        if x != x:  # NaN
            raise MatlabError("NaN's cannot be converted to logicals.")
        return bool(x)
    if isinstance(v, (MString, StringArray)):
        raise MatlabError("Conditional expressions must be numeric or logical, not string.")
    if isinstance(v, (Cell, Struct, FunctionHandle)):
        raise MatlabError(f"Conditional expressions cannot be of type {class_name(v)}.")
    a = to_numeric(v)
    if a.size == 0:
        return False
    if np.iscomplexobj(a):
        a = np.real(a)
    if np.isnan(a).any() if a.dtype.kind == "f" else False:
        raise MatlabError("NaN's cannot be converted to logicals.")
    return bool(np.all(a != 0))


def simplify(a: np.ndarray) -> np.ndarray:
    """Drop all-zero imaginary parts, as MATLAB does after most operations."""
    if isinstance(a, np.ndarray) and a.dtype.kind == "c" and a.size and not np.any(a.imag):
        return a.real.copy()
    return a


def saturate_int(a: np.ndarray, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    a = np.real(np.asarray(a, dtype=complex if np.iscomplexobj(a) else float))
    a = np.where(np.isnan(a), 0, a)
    rounded = np.sign(a) * np.floor(np.abs(a) + 0.5)  # MATLAB rounds halves away from zero
    out = np.clip(rounded, info.min, info.max).astype(dtype)
    if info.max > 2 ** 53:
        # float(info.max) rounds up to 2**63 or 2**64, which overflows in the cast: fix the top end.
        out = np.where(rounded >= float(info.max), info.max, out).astype(dtype)
    return out


def values_equal(a: Any, b: Any) -> bool:
    """``isequal`` semantics."""
    if isinstance(a, StringArray) or isinstance(b, StringArray):
        if not (isinstance(a, StringArray) and isinstance(b, StringArray)):
            return False
        return a.shape == b.shape and all(x == y for x, y in zip(a.data.flat, b.data.flat))
    if isinstance(a, MString) or isinstance(b, MString):
        try:
            return to_str(a) == to_str(b)
        except MatlabError:
            return False
    if isinstance(a, Cell) and isinstance(b, Cell):
        return a.shape == b.shape and all(values_equal(x, y) for x, y in zip(a.data.flat, b.data.flat))
    if isinstance(a, Struct) and isinstance(b, Struct):
        if a.shape != b.shape or set(a.fields) != set(b.fields):
            return False
        return all(values_equal(x[f], y[f]) for x, y in zip(a.data.flat, b.data.flat) for f in a.fields)
    if isinstance(a, FunctionHandle) or isinstance(b, FunctionHandle):
        return a is b
    for x, y in ((a, b), (b, a)):
        if hasattr(x, "object_equals"):  # array-like objects (sparse, datetime, duration) compare by value
            return bool(x.object_equals(y))
    if hasattr(a, "isa_class") or hasattr(b, "isa_class"):
        from .builtins.objfuncs import objects_equal
        return objects_equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        try:
            return bool(np.all(to_numeric(a) == to_numeric(b)))
        except MatlabError:
            return False
    return False
