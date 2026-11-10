"""Object protocol shared by containers.Map, tables and user-defined classdef classes.

An :class:`MObject` decides how it responds to field access (``obj.name``),
method calls (``obj.method(...)`` or ``method(obj, ...)``), paren indexing,
brace indexing and display. The interpreter calls these hooks; it does not
need to know about each concrete type.

Value objects return a *new* object from the ``*_set`` hooks; handle objects
(``is_handle = True``) mutate themselves and return ``self``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import MatlabError


class MObject:
    is_handle = False

    # -- identity ----------------------------------------------------------------------
    def class_name(self) -> str:
        return type(self).__name__

    def isa_class(self, name: str) -> bool:
        return name == self.class_name() or (self.is_handle and name == "handle")

    def matlab_shape(self) -> tuple:
        return (1, 1)

    # -- fields and methods -------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        return False

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        raise MatlabError(f"No method '{name}' for class '{self.class_name()}'.")

    def get_field(self, interp, name: str) -> Any:
        if self.has_method(name):
            return _first(self.call_method(interp, name, [self], 1), name)
        raise MatlabError(f"Unrecognized property or method '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name: str, value: Any) -> "MObject":
        raise MatlabError(f"Cannot set property '{name}' of class '{self.class_name()}'.")

    # -- indexing --------------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        if all(_is_one(a) for a in args):
            return self
        raise MatlabError(f"Indexing into a {self.class_name()} object is not supported.")

    def index_set(self, interp, args: list, value: Any) -> Any:
        raise MatlabError(f"Index assignment into a {self.class_name()} object is not supported. "
                          "Object arrays are not supported yet.")

    def index_delete(self, interp, args: list) -> Any:
        raise MatlabError(f"Cannot delete elements of a {self.class_name()} object.")

    def brace_get(self, interp, args: list) -> list:
        raise MatlabError(f"Brace indexing is not supported for variables of type '{self.class_name()}'.")

    def brace_set(self, interp, args: list, value: Any) -> Any:
        raise MatlabError(f"Brace indexing is not supported for variables of type '{self.class_name()}'.")

    # -- display ----------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        return [f"  {self.class_name()} object"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines()) + "\n\n"


def _first(vals: list, name: str) -> Any:
    if not vals:
        raise MatlabError(f"'{name}' does not return a value.")
    return vals[0]


def _is_one(a: Any) -> bool:
    return isinstance(a, np.ndarray) and a.size == 1 and a.dtype != bool and float(np.real(a.flat[0])) == 1


# -- containers.Map --------------------------------------------------------------------------

_NUMERIC_KEYS = {"double", "single", "int32", "uint32", "int64", "uint64"}


class MapObject(MObject):
    """containers.Map: a handle object mapping char/numeric keys to values."""

    is_handle = True

    def __init__(self, key_type: str = "char", value_type: str = "any"):
        self.key_type = key_type
        self.value_type = value_type
        self.store: dict = {}

    def class_name(self) -> str:
        return "containers.Map"

    def isa_class(self, name: str) -> bool:
        return name in ("containers.Map", "handle")

    def matlab_shape(self) -> tuple:
        return (len(self.store), 1)

    # keys -------------------------------------------------------------------------------
    def norm_key(self, k: Any) -> Any:
        from .values import MString, is_char, to_str
        if self.key_type == "char":
            if isinstance(k, MString) or is_char(k):
                return to_str(k)
            raise MatlabError("Specified key type does not match the type expected for this container.")
        if isinstance(k, np.ndarray) and k.dtype.kind in "biuf" and k.size == 1:
            return float(k.flat[0])
        raise MatlabError("Specified key type does not match the type expected for this container.")

    def key_value(self, k: Any) -> Any:
        from .values import char
        return char(k) if self.key_type == "char" else np.array([[k]])

    def key_list(self, keys_val: Any) -> list:
        from .values import Cell, MString, StringArray, is_char
        if isinstance(keys_val, Cell):
            return [self.norm_key(x) for x in keys_val.values_f()]
        if isinstance(keys_val, StringArray):
            return [self.norm_key(MString(x)) for x in keys_val.values_f()]
        if isinstance(keys_val, MString) or is_char(keys_val):
            return [self.norm_key(keys_val)]
        if isinstance(keys_val, np.ndarray):
            return [self.norm_key(np.array([[x]])) for x in keys_val.flatten(order="F")]
        raise MatlabError("Invalid key.")

    def sorted_keys(self) -> list:
        return sorted(self.store)

    def get(self, k: Any) -> Any:
        key = self.norm_key(k)
        if key not in self.store:
            shown = key if isinstance(key, str) else f"{key:g}"
            raise MatlabError(f"The specified key '{shown}' is not present in this container.")
        return self.store[key]

    # protocol -----------------------------------------------------------------------------
    _METHODS = {"keys", "values", "isKey", "remove", "length", "Count", "KeyType", "ValueType"}

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def get_field(self, interp, name: str) -> Any:
        if name == "Count":
            return np.array([[float(len(self.store))]])
        if name == "KeyType":
            from .values import char
            return char(self.key_type)
        if name == "ValueType":
            from .values import char
            return char(self.value_type)
        if name in self._METHODS:
            return _first(self.call_method(interp, name, [self], 1), name)
        raise MatlabError(f"Unrecognized property '{name}' for class 'containers.Map'.")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        from .values import Cell, bool_array
        rest = args[1:]
        if name == "keys":
            return [Cell.from_list([self.key_value(k) for k in self.sorted_keys()])]
        if name == "values":
            if rest:
                return [Cell.from_list([self.get(self.key_value(k)) for k in self.key_list(rest[0])])]
            return [Cell.from_list([self.store[k] for k in self.sorted_keys()])]
        if name == "isKey":
            keys = self.key_list(rest[0])
            from .values import Cell as _Cell
            if isinstance(rest[0], _Cell):
                flags = [k in self.store for k in keys]
                return [bool_array(flags, rest[0].data.shape)]
            return [np.array([[keys[0] in self.store]])]
        if name == "remove":
            for k in self.key_list(rest[0]):
                if k not in self.store:
                    raise MatlabError("The specified key is not present in this container.")
                del self.store[k]
            return [self]
        if name in ("length", "Count"):
            return [np.array([[float(len(self.store))]])]
        if name in ("KeyType", "ValueType"):
            return [self.get_field(interp, name)]
        return super().call_method(interp, name, args, nargout)

    def index_get(self, interp, args: list) -> Any:
        if len(args) != 1:
            raise MatlabError("Only one key may be used to index a containers.Map.")
        return self.get(args[0])

    def index_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 1:
            raise MatlabError("Only one key may be used to index a containers.Map.")
        self.store[self.norm_key(args[0])] = value
        return self

    def display_lines(self) -> list[str]:
        return ["  Map with properties:", "",
                f"        Count: {len(self.store)}",
                f"      KeyType: {self.key_type}",
                f"    ValueType: {self.value_type}"]


def make_map(args: list) -> MapObject:
    """containers.Map(...) constructor."""
    from .values import Cell, MString, StringArray, is_char, to_str
    opts = {}
    positional = list(args)
    option_names = {"KeyType", "ValueType", "UniformValues"}
    i = 0
    while i < len(positional):
        a = positional[i]
        if (isinstance(a, MString) or is_char(a)) and to_str(a) in option_names and i + 1 < len(positional) \
                and (i > 0 or len(positional) % 2 == 0):
            opts[to_str(a)] = positional[i + 1]
            del positional[i:i + 2]
            continue
        i += 1
    key_type = to_str(opts["KeyType"]) if "KeyType" in opts else None
    value_type = to_str(opts["ValueType"]) if "ValueType" in opts else "any"
    if not positional:
        return MapObject(key_type or "char", value_type)
    if len(positional) != 2:
        raise MatlabError("containers.Map expects keys and values, e.g. containers.Map({'a','b'}, {1, 2}).")
    keys_val, vals_val = positional
    if key_type is None:
        numeric = isinstance(keys_val, np.ndarray) and keys_val.dtype.kind in "biuf"
        if isinstance(keys_val, Cell) and keys_val.data.size and all(
                isinstance(x, np.ndarray) and x.dtype.kind in "biuf" for x in keys_val.values_f()):
            numeric = True
        key_type = "double" if numeric else "char"
    m = MapObject(key_type, value_type)
    keys = m.key_list(keys_val)
    if isinstance(vals_val, Cell):
        values = vals_val.values_f()
    elif isinstance(vals_val, StringArray):
        values = [MString(x) for x in vals_val.values_f()]
    elif len(keys) == 1:
        values = [vals_val]
    elif isinstance(vals_val, np.ndarray):
        values = [vals_val.flatten(order="F")[k:k + 1].reshape(1, 1) for k in range(vals_val.size)]
    else:
        values = [vals_val]
    if len(values) != len(keys):
        raise MatlabError("The number of keys and values must be the same.")
    for k, v in zip(keys, values):
        m.store[k] = v
    if value_type == "any" and isinstance(vals_val, np.ndarray) and vals_val.dtype.kind in "biuf":
        m.value_type = "double"
    elif value_type == "any" and len(keys) == 1 and is_char(vals_val):
        m.value_type = "char"
    return m


# -- MException -----------------------------------------------------------------------------

class MExceptionObject(MObject):
    """The error object caught by ``catch err`` (and created with MException)."""

    def __init__(self, identifier: str, message: str, stack: Any = None, causes: list | None = None):
        self.identifier = identifier
        self.message = message
        self.stack = stack
        self.causes = list(causes or [])

    def class_name(self) -> str:
        return "MException"

    def isa_class(self, name: str) -> bool:
        return name == "MException"

    _METHODS = {"getReport", "rethrow", "throw", "throwAsCaller", "addCause"}

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def to_error(self) -> MatlabError:
        err = MatlabError(self.message, self.identifier)
        err.mexception = self  # type: ignore[attr-defined]
        return err

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        from .values import char
        if name in ("rethrow", "throw", "throwAsCaller"):
            raise self.to_error()
        if name == "getReport":
            report = f"Error: {self.message}"
            for c in self.causes:
                if isinstance(c, MExceptionObject):
                    report += f"\nCaused by:\n    {c.message}"
            return [char(report)]
        if name == "addCause":
            other = MExceptionObject(self.identifier, self.message, self.stack, self.causes + [args[1]])
            return [other]
        return super().call_method(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        from .values import Cell, Struct, char
        if name == "message":
            return char(self.message)
        if name == "identifier":
            return char(self.identifier)
        if name == "stack":
            return self.stack if self.stack is not None else Struct(np.empty((0, 1), dtype=object), [])
        if name == "cause":
            return Cell.from_list(self.causes, column=True)
        return super().get_field(interp, name)

    def set_field(self, interp, name: str, value: Any) -> "MExceptionObject":
        raise MatlabError(f"Cannot set read-only property '{name}' of MException.")

    def display_lines(self) -> list[str]:
        return ["  MException with properties:", "",
                f"    identifier: '{self.identifier}'",
                f"       message: '{self.message}'",
                f"         cause: {{{len(self.causes)}×1 cell}}",
                "         stack: [1×1 struct]"]
