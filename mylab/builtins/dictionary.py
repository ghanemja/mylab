"""The dictionary type: a map from keys of one type to values of one type.

A dictionary remembers the order in which its entries were added. Keys and values are arrays:
``d(keys)`` looks up every element of ``keys`` and ``d(keys) = values`` sets one entry per
element. Text keys and values are strings (char vectors are converted). A dictionary is a
value: assigning it to another variable copies it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    INT_TYPES, Cell, FunctionHandle, MString, StringArray, Struct, class_name, is_char, saturate_int,
    string_result, to_str, values_equal,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, nargs, text

ARROW = "⟼"  # the long arrow MATLAB prints between a key and its value


def _hashable(v: Any) -> Any:
    """A hashable stand-in for a scalar value, equal for values that isequal says are equal."""
    if isinstance(v, MString):
        return ("s", str(v))
    if isinstance(v, StringArray):
        return ("S", v.data.shape, tuple(v.values_f()))
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return ("c", v.shape, "".join(v.flatten(order="F")))
        if v.dtype.kind == "c":
            return ("n", v.shape, tuple(complex(x) for x in v.flatten(order="F")))
        return ("n", v.shape, tuple(float(x) for x in v.flatten(order="F")))
    if isinstance(v, Cell):
        return ("C", v.data.shape, tuple(_hashable(x) for x in v.values_f()))
    if isinstance(v, Struct):
        return ("T", v.data.shape, tuple(sorted(v.fields)),
                tuple(tuple((f, _hashable(d[f])) for f in sorted(v.fields)) for d in v.data.flatten(order="F")))
    if isinstance(v, FunctionHandle):
        return ("f", v.name or id(v))
    data = getattr(v, "data", None)
    if isinstance(data, np.ndarray) and data.dtype.kind in "fiub":
        extra = tuple(getattr(v, "categories", ())) if hasattr(v, "categories") else ()
        names = tuple(v.names_f()) if hasattr(v, "names_f") else tuple(float(x) for x in data.flatten(order="F"))
        return ("o", v.class_name(), data.shape, names, extra if not hasattr(v, "names_f") else ())
    return ("id", id(v))


def _type_of(v: Any) -> str:
    name = class_name(v)
    return "string" if name == "char" else name


def _elements(v: Any, what: str) -> tuple[list, tuple]:
    """The scalar elements of an array value (column-major) and its shape; char becomes one string."""
    if isinstance(v, MString):
        return [v], (1, 1)
    if isinstance(v, StringArray):
        out = []
        for x in v.values_f():
            if x is None:
                d = np.empty((1, 1), dtype=object)
                d[0, 0] = None
                out.append(StringArray(d))
            else:
                out.append(MString(x))
        return out, v.data.shape
    if is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return [MString("".join(r)) for r in v], (v.shape[0], 1)
        return [MString(to_str(v))], (1, 1)
    if isinstance(v, Cell):
        out = []
        for x in v.values_f():
            d = np.empty((1, 1), dtype=object)
            d[0, 0] = x
            out.append(Cell(d))
        return out, v.data.shape
    if isinstance(v, Struct):
        out = []
        for d in v.data.flatten(order="F"):
            arr = np.empty((1, 1), dtype=object)
            arr[0, 0] = dict(d)
            out.append(Struct(arr, v.fields))
        return out, v.data.shape
    if isinstance(v, np.ndarray):
        flat = v.flatten(order="F")
        return [flat[k:k + 1].reshape(1, 1) for k in range(flat.size)], v.shape
    if isinstance(v, FunctionHandle):
        return [v], (1, 1)
    data = getattr(v, "data", None)
    if isinstance(data, np.ndarray) and hasattr(v, "like"):
        flat = data.flatten(order="F")
        return [v.like(flat[k:k + 1].reshape(1, 1)) for k in range(flat.size)], data.shape
    if hasattr(v, "isa_class"):
        return [v], (1, 1)
    raise MatlabError(f"A value of class '{class_name(v)}' cannot be used as dictionary {what}.",
                      "MATLAB:dictionary:InvalidType")


def _convert(v: Any, kind: str, what: str) -> Any:
    """Convert a scalar element to the key or value type of a dictionary."""
    have = _type_of(v)
    if have == kind:
        return v
    if kind == "cell":
        d = np.empty((1, 1), dtype=object)
        d[0, 0] = v
        return Cell(d)
    if kind == "string":
        if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size == 1:
            from ..values import string_data
            return string_result(string_data(v))
        inner = v.data.flat[0] if isinstance(v, Cell) and v.data.size == 1 else None
        if inner is not None and (is_char(inner) or isinstance(inner, MString)):
            return MString(to_str(inner))
        if hasattr(v, "has_method") and v.has_method("string"):
            return v.call_method(None, "string", [v], 1)[0]
    numeric = {"double": float, "single": np.float32, "logical": bool}
    if kind in numeric or kind in INT_TYPES:
        if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            if kind in INT_TYPES:
                return saturate_int(v, INT_TYPES[kind])
            if kind == "logical":
                return v != 0
            return v.astype(numeric[kind])
        if isinstance(v, MString):
            try:
                x = float(str(v))
            except ValueError:
                x = float("nan")
            if kind == "double":
                return np.array([[x]])
    raise MatlabError(f"Unable to use a value of class '{have}' as a {what} of this dictionary, whose "
                      f"{what}s are of class '{kind}'.", "MATLAB:dictionary:InvalidConversion")


def _join(values: list, shape: tuple, kind: str) -> Any:
    """Array of the given shape from scalar elements."""
    n = len(values)
    if kind == "string":
        data = np.empty(n, dtype=object)
        for k, v in enumerate(values):
            data[k] = None if isinstance(v, StringArray) else str(v)
        data = data.reshape(shape, order="F")
        return string_result(data) if n != 0 else StringArray(data)
    if kind == "cell":
        data = np.empty(n, dtype=object)
        for k, v in enumerate(values):
            data[k] = v.data.flat[0]
        return Cell(data.reshape(shape, order="F"))
    if kind == "struct":
        fields = list(values[0].fields) if values else []
        data = np.empty(n, dtype=object)
        for k, v in enumerate(values):
            if list(v.fields) != fields:
                raise MatlabError("The struct values of the dictionary have different fields.")
            data[k] = dict(v.data.flat[0])
        return Struct(data.reshape(shape, order="F"), fields)
    if values and all(isinstance(v, np.ndarray) for v in values):
        arr = np.empty(n, dtype=values[0].dtype)
        for k, v in enumerate(values):
            arr[k] = v.flat[0]
        return arr.reshape(shape, order="F")
    if not values:
        if kind == "double":
            return np.zeros(shape)
        if kind in INT_TYPES:
            return np.zeros(shape, dtype=INT_TYPES[kind])
        if kind == "logical":
            return np.zeros(shape, dtype=bool)
        if kind == "single":
            return np.zeros(shape, dtype=np.float32)
        return np.zeros(shape)
    first = values[0]
    data = getattr(first, "data", None)
    if isinstance(data, np.ndarray) and hasattr(first, "like") and not hasattr(first, "categories"):
        arr = np.array([float(v.data.flat[0]) for v in values]).reshape(shape, order="F")
        return first.like(arr)
    if n == 1:
        return first
    from ..ops import concat
    joined = concat(values, 0)
    if hasattr(joined, "reshape") and len(shape) == 2 and shape[0] == 1:
        joined = concat(values, 1)
    return joined


class DictionaryObject(MObject):
    """A MATLAB dictionary."""

    def __init__(self, key_type: str | None = None, value_type: str | None = None):
        self.key_type = key_type
        self.value_type = value_type
        self.store: dict = {}  # hashable key -> (key element, value element), in insertion order

    def class_name(self) -> str:
        return "dictionary"

    def isa_class(self, name: str) -> bool:
        return name == "dictionary"

    def matlab_shape(self) -> tuple:
        return (1, 1)

    def copy(self) -> "DictionaryObject":
        d = DictionaryObject(self.key_type, self.value_type)
        d.store = dict(self.store)
        return d

    @property
    def configured(self) -> bool:
        return self.key_type is not None

    # -- keys and values ----------------------------------------------------------------------------
    def key_elements(self, keys: Any, configure: bool = False) -> tuple[list, tuple]:
        elems, shape = _elements(keys, "key")
        if self.key_type is None:
            if not configure:
                raise MatlabError("Unable to perform a lookup in a dictionary with unset key and value types.",
                                  "MATLAB:dictionary:UnconfiguredLookupNotSupported")
            self.key_type = _type_of(keys)
        return [_convert(e, self.key_type, "key") for e in elems], shape

    def set_entries(self, keys: Any, values: Any) -> None:
        kelems, kshape = self.key_elements(keys, configure=True)
        if self.value_type == "cell" and not isinstance(values, Cell):
            raise MatlabError("The values of this dictionary are cells: assign with d(key) = {value} "
                              "or d{key} = value.", "MATLAB:dictionary:CellValueRequired")
        velems, _vshape = _elements(values, "value")
        if self.value_type is None:
            self.value_type = _type_of(values)
        if len(velems) == 1 and len(kelems) != 1:
            velems = velems * len(kelems)
        if len(velems) != len(kelems):
            raise MatlabError("The number of values must be 1 or match the number of keys.",
                              "MATLAB:dictionary:NumelKeysMustMatchNumelValues")
        for k, v in zip(kelems, velems):
            self.store[_hashable(k)] = (k, _convert(v, self.value_type, "value"))

    def lookup(self, keys: Any, fallback: Any = None, has_fallback: bool = False) -> Any:
        kelems, shape = self.key_elements(keys)
        out = []
        for k in kelems:
            hit = self.store.get(_hashable(k))
            if hit is None:
                if not has_fallback:
                    raise MatlabError("Key not found.", "MATLAB:dictionary:ScalarKeyNotFound"
                                      if len(kelems) == 1 else "MATLAB:dictionary:KeyNotFound")
                out.append(_convert(_elements(fallback, "value")[0][0], self.value_type, "value"))
            else:
                out.append(hit[1])
        return _join(out, shape, self.value_type)

    # -- protocol -------------------------------------------------------------------------------------
    _METHODS = {"keys", "values", "entries", "numEntries", "isKey", "insert", "remove", "lookup", "types",
                "isConfigured", "isequal"}

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        from ..values import to_matlab
        res = _DICT_FUNCS[name](interp, args, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def get_field(self, interp, name: str) -> Any:
        if name in self._METHODS:
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method or property '{name}' for class 'dictionary'.",
                          "MATLAB:noSuchMethodOrField")

    def index_get(self, interp, args: list) -> Any:
        if len(args) != 1:
            raise MatlabError("A dictionary is indexed with one array of keys: d(keys).",
                              "MATLAB:dictionary:InvalidIndexing")
        return self.lookup(self._key_arg(args[0]))

    def _key_arg(self, a: Any) -> Any:
        if a is ix.COLON:
            raise MatlabError("A colon cannot be used to index a dictionary. Use keys(d) or values(d).",
                              "MATLAB:dictionary:ColonIndexingNotSupported")
        return a

    def index_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 1:
            raise MatlabError("A dictionary is indexed with one array of keys: d(keys) = values.",
                              "MATLAB:dictionary:InvalidIndexing")
        d = self.copy()
        d.set_entries(self._key_arg(args[0]), value)
        return d

    def index_delete(self, interp, args: list) -> Any:
        """d(keys) = [] removes the entries."""
        if len(args) != 1:
            raise MatlabError("A dictionary is indexed with one array of keys.", "MATLAB:dictionary:InvalidIndexing")
        return _remove(interp, [self, args[0]], 1)

    def brace_get(self, interp, args: list) -> list:
        if self.value_type != "cell":
            raise MatlabError("Brace indexing needs a dictionary whose values are cells.",
                              "MATLAB:dictionary:BraceIndexingNotSupported")
        if len(args) != 1:
            raise MatlabError("A dictionary is indexed with one array of keys: d{key}.",
                              "MATLAB:dictionary:InvalidIndexing")
        return self.lookup(args[0]).values_f()

    def brace_set(self, interp, args: list, value: Any) -> Any:
        if self.value_type not in (None, "cell"):
            raise MatlabError("Brace indexing needs a dictionary whose values are cells.",
                              "MATLAB:dictionary:BraceIndexingNotSupported")
        if len(args) != 1:
            raise MatlabError("A dictionary is indexed with one array of keys: d{key} = value.",
                              "MATLAB:dictionary:InvalidIndexing")
        wrapped = np.empty((1, 1), dtype=object)
        wrapped[0, 0] = value
        d = self.copy()
        d.set_entries(args[0], Cell(wrapped))
        return d

    def object_equals(self, other: Any) -> bool:
        if not isinstance(other, DictionaryObject) or other.key_type != self.key_type \
                or other.value_type != self.value_type or set(other.store) != set(self.store):
            return False
        return all(values_equal(v[1], other.store[k][1]) for k, v in self.store.items())

    # -- display -----------------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        return "1×1 dictionary" if in_cell else "[1×1 dictionary]"

    def display_lines(self) -> list[str]:
        from ..display import summary
        if not self.configured:
            return ["  dictionary with unset key and value types."]
        n = len(self.store)
        head = f"  dictionary ({self.key_type} {ARROW} {self.value_type}) with "
        if n == 0:
            return [head + "no entries."]
        lines = [head + (f"{n} entries:" if n != 1 else "1 entry:"), ""]
        keys = [summary(k, False) for k, _ in self.store.values()]
        vals = [summary(v, False) if self.value_type != "cell" else "{" + summary(v.data.flat[0], True) + "}"
                for _, v in self.store.values()]
        width = max(len(k) for k in keys)
        numeric_keys = self.key_type != "string"
        for k, v in zip(keys, vals):
            lines.append(f"    {k.rjust(width) if numeric_keys else k.ljust(width)} {ARROW} {v}")
        return lines

    def format_display(self, name: str) -> str:
        return f"{name} =\n\n" + "\n".join(self.display_lines()) + "\n\n"


def _need(args: list, name: str) -> DictionaryObject:
    if not args or not isinstance(args[0], DictionaryObject):
        raise MatlabError(f"{name}: the first argument must be a dictionary.", "MATLAB:dictionary:InvalidInput")
    return args[0]


# -- construction ---------------------------------------------------------------------------------------

@builtin("dictionary")
def _dictionary(interp, args, nargout):
    """dictionary, dictionary(keys, values), dictionary(k1, v1, ..., kN, vN)."""
    d = DictionaryObject()
    if not args:
        return d
    if len(args) == 1 and isinstance(args[0], DictionaryObject):
        return args[0].copy()
    if len(args) % 2:
        raise MatlabError("dictionary needs keys and values in pairs: dictionary(keys, values) or "
                          "dictionary(key1, value1, ..., keyN, valueN).", "MATLAB:dictionary:IncorrectNumInputs")
    for k in range(0, len(args), 2):
        keys, values = args[k], args[k + 1]
        if len(args) > 2 and is_char(values) and values.ndim == 2 and values.shape[0] <= 1:
            values = MString(to_str(values))
        d.set_entries(keys, values)
    return d


_TYPE_NAMES = {"double", "single", "logical", "string", "cell", "struct", "datetime", "duration", "categorical",
               "function_handle", *INT_TYPES}


@builtin("configureDictionary")
def _configure(interp, args, nargout):
    """configureDictionary(keyType, valueType): an empty dictionary with the given types."""
    nargs(args, 2, 2, "configureDictionary")
    kt, vt = text(args[0]), text(args[1])
    kt, vt = ("string" if kt == "char" else kt), ("string" if vt == "char" else vt)
    for t in (kt, vt):
        if t not in _TYPE_NAMES and (interp is None or interp.find_class(t) is None):
            raise MatlabError(f"'{t}' is not a known class name.", "MATLAB:dictionary:UnknownType")
    return DictionaryObject(kt, vt)


@builtin("isConfigured")
def _is_configured(interp, args, nargout):
    return bool_scalar(_need(args, "isConfigured").configured)


@builtin("numEntries")
def _num_entries(interp, args, nargout):
    return float(len(_need(args, "numEntries").store))


@builtin("types")
def _types(interp, args, nargout):
    d = _need(args, "types")
    kt = MString(d.key_type) if d.key_type else StringArray(np.array([[None]], dtype=object))
    vt = MString(d.value_type) if d.value_type else StringArray(np.array([[None]], dtype=object))
    if nargout > 1:
        return kt, vt
    data = np.empty((1, 2), dtype=object)
    data[0, 0], data[0, 1] = d.key_type, d.value_type
    return StringArray(data)


def _keys(interp, args, nargout):
    d = _need(args, "keys")
    if not d.configured:
        raise MatlabError("The dictionary has unset key and value types.",
                          "MATLAB:dictionary:UnconfiguredLookupNotSupported")
    as_cell = len(args) > 1 and is_text(args[1]) and text(args[1]).lower() == "cell"
    items = [k for k, _ in d.store.values()]
    if as_cell:
        return Cell.from_list(items, column=True)
    return _join(items, (len(items), 1), d.key_type)


def _values(interp, args, nargout):
    d = _need(args, "values")
    if not d.configured:
        raise MatlabError("The dictionary has unset key and value types.",
                          "MATLAB:dictionary:UnconfiguredLookupNotSupported")
    as_cell = len(args) > 1 and is_text(args[1]) and text(args[1]).lower() == "cell"
    items = [v for _, v in d.store.values()]
    if as_cell:
        if d.value_type == "cell":
            return _join(items, (len(items), 1), "cell")
        return Cell.from_list(items, column=True)
    return _join(items, (len(items), 1), d.value_type)


def _is_key(interp, args, nargout):
    d = _need(args, "isKey")
    nargs(args, 2, 2, "isKey")
    if not d.configured:
        elems, shape = _elements(args[1], "key")
        return np.zeros(shape, dtype=bool)
    try:
        kelems, shape = d.key_elements(args[1])
    except MatlabError:
        _elems, shape = _elements(args[1], "key")
        return np.zeros(shape, dtype=bool)
    return np.array([_hashable(k) in d.store for k in kelems], dtype=bool).reshape(shape, order="F")


def _insert(interp, args, nargout):
    d = _need(args, "insert")
    rest = list(args[1:])
    overwrite = True
    if len(rest) >= 4 and is_text(rest[-2]) and text(rest[-2]).lower() == "overwrite":
        v = rest[-1]
        overwrite = text(v).lower() in ("true", "on") if is_text(v) else bool(np.asarray(v).flat[0])
        del rest[-2:]
    if len(rest) != 2:
        raise MatlabError("insert(d, keys, values) needs keys and values.", "MATLAB:minrhs")
    out = d.copy()
    if not overwrite and out.configured:
        kelems, _ = out.key_elements(rest[0])
        if any(_hashable(k) in out.store for k in kelems):
            raise MatlabError("A key is already in the dictionary and 'Overwrite' is false.",
                              "MATLAB:dictionary:KeyAlreadyExists")
    out.set_entries(rest[0], rest[1])
    return out


def _remove(interp, args, nargout):
    d = _need(args, "remove")
    nargs(args, 2, 2, "remove")
    out = d.copy()
    kelems, _ = out.key_elements(args[1])
    for k in kelems:
        h = _hashable(k)
        if h not in out.store:
            raise MatlabError("Key not found.", "MATLAB:dictionary:ScalarKeyNotFound")
        del out.store[h]
    return out


def _lookup(interp, args, nargout):
    d = _need(args, "lookup")
    rest = list(args[1:])
    fallback, has = None, False
    if len(rest) >= 3 and is_text(rest[-2]) and text(rest[-2]).lower() == "fallbackvalue":
        fallback, has = rest[-1], True
        del rest[-2:]
    if len(rest) != 1:
        raise MatlabError("lookup(d, keys) needs the keys to look up.", "MATLAB:minrhs")
    return d.lookup(rest[0], fallback, has)


def _entries(interp, args, nargout):
    d = _need(args, "entries")
    form = text(args[1]).lower() if len(args) > 1 else "table"
    keys = [k for k, _ in d.store.values()]
    vals = [v for _, v in d.store.values()]
    n = len(keys)
    if form == "cell":
        data = np.empty((n, 2), dtype=object)
        for i in range(n):
            data[i, 0] = keys[i]
            data[i, 1] = vals[i].data.flat[0] if d.value_type == "cell" else vals[i]
        return Cell(data)
    if form == "struct":
        data = np.empty((n, 1), dtype=object)
        for i in range(n):
            data[i, 0] = {"Key": keys[i], "Value": vals[i].data.flat[0] if d.value_type == "cell" else vals[i]}
        return Struct(data, ["Key", "Value"])
    if form != "table":
        raise MatlabError("entries: the format must be \"table\", \"struct\" or \"cell\".",
                          "MATLAB:dictionary:InvalidEntriesFormat")
    from .table import TableObject
    return TableObject(["Key", "Value"], [_column(_join(keys, (n, 1), d.key_type or "double")),
                                          _column(_join(vals, (n, 1), d.value_type or "double"))])


def _column(v: Any) -> Any:
    if isinstance(v, MString):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = str(v)
        return StringArray(data)
    return v


def _isequal(interp, args, nargout):
    return bool_scalar(all(values_equal(args[0], a) for a in args[1:]))


_DICT_FUNCS = {"keys": _keys, "values": _values, "entries": _entries, "numEntries": _num_entries,
               "isKey": _is_key, "insert": _insert, "remove": _remove, "lookup": _lookup, "types": _types,
               "isConfigured": _is_configured, "isequal": _isequal}

builtin("entries")(_entries)
builtin("insert")(_insert)
builtin("lookup")(_lookup)


@builtin("isdictionary")
def _isdictionary(interp, args, nargout):
    nargs(args, 1, 1, "isdictionary")
    return bool_scalar(isinstance(args[0], DictionaryObject))
