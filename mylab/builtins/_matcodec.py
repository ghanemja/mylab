"""Turning mylab values into what a MAT-file can hold, and back.

A MAT-file stores numeric, logical and char arrays, sparse matrices, cells and structs. Values of
any other kind (string arrays, tables, datetimes, durations, categoricals, containers.Map,
timeseries, function handles, classdef objects) are written as a scalar struct whose first field,
``mylab_class_``, names the kind. MATLAB and Octave read such a variable as an ordinary struct;
mylab turns it back into the original value.

Reading also understands what MATLAB itself writes for objects: old-style objects, function
handles and the "MCOS" object data kept in the file's subsystem (string arrays, datetime,
duration, categorical, table, containers.Map and plain classdef objects).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char, string_result, to_str,
)
from . import _matio

MARK = "mylab_class_"


def _marker(kind: str, **fields: Any) -> Struct:
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = {MARK: char(kind), **fields}
    return Struct(data, [MARK, *fields])


def _cellstr(items: list, column: bool = False) -> Cell:
    return Cell.from_list([char(x) for x in items], column=column)


def _texts(v: Any) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) if (is_char(x) or isinstance(x, MString)) else "" for x in v.values_f()]
    if isinstance(v, StringArray):
        return ["" if x is None else x for x in v.values_f()]
    if isinstance(v, np.ndarray) and v.dtype.kind == "U" and v.ndim == 2 and v.shape[0] > 1:
        return ["".join(r).rstrip() for r in v]
    if is_char(v) or isinstance(v, MString):
        return [to_str(v)]
    return []


def _num(x: float) -> np.ndarray:
    return np.array([[float(x)]])


def _dims(shape: tuple) -> np.ndarray:
    return np.array([[float(d) for d in shape]])


# -- encoding ------------------------------------------------------------------------------------

class Encoder:
    """Encodes the variables of one save call.

    A handle object (or containers.Map) is written in full where the call first meets it; later
    occurrences, in the same variable or in another one, are ``object_ref`` markers naming its id.
    The ids of one call start at a random multiple of 2^22, so variables that a later call adds
    with -append (another encoder) do not reuse the ids already in the file. They stay below 2^52,
    so a double holds them exactly.
    """

    def __init__(self, interp):
        import secrets
        self.interp = interp
        self.handles: dict[int, int] = {}
        self.keep: list = []  # keeps encoded handle objects alive so their id() stays unique
        self.first_id = (secrets.randbits(30) << 22) + 1

    def handle_id(self, v: Any) -> tuple[int, bool]:
        """The id of a handle value, and whether this call has written it already."""
        key = id(v)
        if key in self.handles:
            return self.handles[key], True
        self.handles[key] = self.first_id + len(self.handles)
        self.keep.append(v)
        return self.handles[key], False

    def fail(self, var: str, v: Any, why: str = "") -> MatlabError:
        detail = f" ({why})" if why else ""
        return MatlabError(f"save: variable '{var}' contains a value of class '{class_name(v)}', which cannot "
                           f"be stored in a MAT-file{detail}.", "MATLAB:save:unsupportedType")

    def encode(self, v: Any, var: str) -> Any:
        from .sparse import SparseMatrix
        if isinstance(v, np.ndarray):
            if v.dtype.kind == "O":
                raise self.fail(var, v)
            return v
        if isinstance(v, SparseMatrix):
            return v
        if isinstance(v, Cell):
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = self.encode(v.data[idx], var)
            return Cell(data)
        if isinstance(v, Struct):
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = {f: self.encode(v.data[idx][f], var) for f in v.fields}
            return Struct(data, list(v.fields))
        if isinstance(v, (MString, StringArray)):
            d = np.asarray(v.data if isinstance(v, StringArray) else np.array([[str(v)]], dtype=object))
            flat = d.flatten(order="F")
            return _marker("string", size=_dims(d.shape),
                           text=_cellstr(["" if x is None else x for x in flat]),
                           missing=np.array([[x is None for x in flat]], dtype=bool).reshape(1, -1))
        if isinstance(v, FunctionHandle):
            return self._function(v, var)
        return self._object(v, var)

    def _function(self, f: FunctionHandle, var: str) -> Struct:
        from ..display import function_handle_text
        if not f.is_anonymous:
            return _marker("function_handle", function=char(str(f.name)), workspace=Struct())
        ws = {k: self.encode(x, var) for k, x in f.closure.items() if not k.startswith("__")}  # skip engine entries
        return _marker("function_handle", function=char(function_handle_text(f)), workspace=Struct.scalar(ws))

    def _object(self, v: Any, var: str) -> Struct:
        from ..classdef import ObjectArray, UserObject
        from ..objects import MapObject, MExceptionObject
        from .categorical import CategoricalArray
        from .datetimes import CalendarDuration, DatetimeArray, DurationArray
        from .dictionary import DictionaryObject
        from .table import TableObject
        if isinstance(v, TableObject):
            return self._table(v, var)
        if isinstance(v, DatetimeArray):
            return _marker("datetime", seconds=v.data, format=char(v.fmt or ""), timezone=char(v.zone or ""))
        if isinstance(v, DurationArray):
            return _marker("duration", seconds=v.data, format=char(v.fmt or ""))
        if isinstance(v, CalendarDuration):
            return _marker("calendarDuration", months=v.months, days=v.days, seconds=v.secs)
        if isinstance(v, CategoricalArray):
            return _marker("categorical", codes=v.data, categories=_cellstr(v.categories),
                           ordinal=np.array([[bool(v.ordinal)]]))
        if isinstance(v, MapObject):  # a handle: shared maps stay shared
            ident, seen = self.handle_id(v)
            if seen:
                return _marker("object_ref", id=_num(ident))
            keys = v.sorted_keys()
            return _marker("containers.Map", keytype=char(v.key_type), valuetype=char(v.value_type),
                           keys=Cell.from_list([v.key_value(k) for k in keys]),
                           values=Cell.from_list([self.encode(v.store[k], var) for k in keys]), id=_num(ident))
        if isinstance(v, DictionaryObject):
            entries = list(v.store.values())
            return _marker("dictionary", keytype=char(v.key_type or ""), valuetype=char(v.value_type or ""),
                           keys=Cell.from_list([self.encode(k, var) for k, _ in entries]),
                           values=Cell.from_list([self.encode(x, var) for _, x in entries]),
                           configured=np.array([[v.key_type is not None]]))
        if isinstance(v, MExceptionObject):
            extra = {"causes": Cell.from_list([self.encode(c, var) for c in v.causes])}
            if isinstance(v.stack, Struct):
                extra["stack"] = self.encode(v.stack, var)
            return _marker("MException", identifier=char(v.identifier), message=char(v.message), **extra)
        from .timeseries import TimeSeries, to_saved
        if isinstance(v, TimeSeries):
            return _marker("timeseries", **{k: self.encode(x, var) for k, x in to_saved(v).items()})
        if isinstance(v, ObjectArray):
            return _marker("object_array", name=char(v.cls.full_name if v.cls else ""), size=_dims(v.data.shape),
                           items=Cell.from_list([self.encode(x, var) for x in v.elements()]))
        if isinstance(v, UserObject):
            return self._user_object(v, var)
        if getattr(v, "is_sym", False):
            data = np.asarray(v.data, dtype=object)
            return _marker("sym", size=_dims(data.shape), text=_cellstr([str(e) for e in data.flatten(order="F")]))
        raise self.fail(var, v)

    def _user_object(self, obj: Any, var: str) -> Struct:
        cls = obj.cls
        if obj._enum_name is not None:
            return _marker("enumeration", name=char(cls.full_name), member=char(obj._enum_name))
        if getattr(obj, "_deleted", False):
            raise self.fail(var, obj, "the handle object was deleted")
        ident = 0
        if obj.is_handle:
            ident, seen = self.handle_id(obj)
            if seen:
                return _marker("object_ref", id=_num(ident))
        if "saveobj" in cls.methods:
            res = cls.call(self.interp, "saveobj", [obj], 1)
            if not res:
                raise MatlabError(f"save: the saveobj method of class '{cls.name}' returned nothing "
                                  f"(variable '{var}').")
            saved = res[0]
            if not (saved is obj or (isinstance(saved, type(obj)) and getattr(saved, "cls", None) is cls)):
                return _marker("object", name=char(cls.full_name), id=_num(ident), saveobj=np.array([[True]]),
                               data=self.encode(saved, var))
            obj = saved
        return _marker("object", name=char(cls.full_name), id=_num(ident), saveobj=np.array([[False]]),
                       data=self._props(obj, var), **self._object_extras(obj, var))

    def _props(self, obj: Any, var: str) -> Struct:
        """The stored property values: not Transient, Dependent or Constant ones (old-style objects: every
        field, including the parent objects)."""
        cls = obj.cls
        out = {}
        names = list(obj.props) if cls.old_style else cls.instance_props()
        for name in names:
            entry = cls.prop_defs.get(name)
            if entry is not None and (entry[0].attrs.get("Transient") or entry[0].attrs.get("Dependent")):
                continue
            if name in obj.props:
                out[name] = self.encode(obj.props[name], var)
        return Struct.scalar(out) if out else Struct()

    def _object_extras(self, obj: Any, var: str) -> dict:
        """The built-in value of an object of a class derived from double, char, ..., and the values of
        its dynamic properties (addprop), when it has them."""
        out: dict = {}
        if obj._data is not None:
            out["builtin"] = self.encode(obj._data, var)
        dyn = {n: self.encode(d.stored, var) for n, d in (obj._dynprops or {}).items()
               if not (d.attrs.get("Transient") or d.attrs.get("Dependent"))}
        if dyn:
            out["dynamic"] = Struct.scalar(dyn)
        return out

    def _table(self, t: Any, var: str) -> Struct:
        """A table or timetable with its properties (units, descriptions, user data, dimension and
        custom properties)."""
        from .timetables import TimetableObject

        def per_var(store: dict) -> Cell:
            return _cellstr([store.get(n, "") for n in t.names])

        fields: dict = {"varnames": _cellstr(t.names), "vars": Cell.from_list([self.encode(c, var) for c in t.columns]),
                        "rownames": _cellstr(t.row_names or []), "description": char(t.description or ""),
                        "units": per_var(t.units), "vardescriptions": per_var(t.var_descriptions),
                        "dimnames": _cellstr(t.dim_names),
                        "custom": Struct.scalar({k: self.encode(x, var) for k, x in t.custom.items()}),
                        "customvar": _cellstr(sorted(t.custom_var))}
        if t.user_data is not None:
            fields["userdata"] = self.encode(t.user_data, var)
        if isinstance(t, TimetableObject):
            return _marker("timetable", rowtimes=self.encode(t.row_times, var), **fields)
        return _marker("table", **fields)


# -- decoding ------------------------------------------------------------------------------------

class Decoder:
    """Decodes the variables of one load call.

    ``everything``, when given, returns the stored (not yet decoded) values of all variables of the
    file. It is called when a variable refers to a handle object that is written in full in another
    variable which this call does not read (load of some variables, matfile, whos -file with names).

    A ``passive`` decoder makes values only to describe them (whos -file): objects get their saved
    property values directly, and no code of their class runs (no constructor, set method, loadobj,
    and no delete method when the object is dropped).
    """

    def __init__(self, interp, subsystem: bytes | None = None, everything=None, passive: bool = False):
        self.interp = interp
        self.passive = passive
        self.handles: dict[int, Any] = {}
        self.mcos = Mcos(subsystem) if subsystem else None
        self.var = ""
        self._in_object = 0  # inside MATLAB object data, uint32 reference arrays stand for objects
        self._everything = everything
        self._definitions: dict[int, dict] | None = None
        self._made: dict[int, Any] = {}  # id() of a stored handle object's marker -> the object made from it
        self._resolving: set[int] = set()
        self._quiet = 0  # inside an object array of a missing class, whose variable is warned about once
        self.made_handles = False  # whether a handle object or containers.Map was made
        self.made_objects = False  # whether any object of a class was made

    def warn(self, msg: str, ident: str = "MATLAB:load:classNotFound") -> None:
        self.interp.warn(msg, ident)

    def _handle(self, ident: int) -> Any:
        """The handle object (or containers.Map) with this id, decoding its definition if needed."""
        if ident not in self.handles and self._everything is not None and ident not in self._resolving:
            if self._definitions is None:
                self._definitions = _handle_definitions(self._everything())
            d = self._definitions.get(ident)
            if d is not None:
                self._resolving.add(ident)  # (saveobj data that refers back to its own object: an error)
                try:
                    self._marked(d)  # registers it
                finally:
                    self._resolving.discard(ident)
        if ident not in self.handles:
            raise MatlabError(f"load: variable '{self.var}' refers to a handle object that is not in the file.")
        return self.handles[ident]

    def decode(self, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            if self._in_object and v.dtype == np.uint32 and self.mcos.is_reference(v):
                return self._mcos_value(v)
            return v
        if isinstance(v, Cell):
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = self.decode(v.data[idx])
            return Cell(data)
        if isinstance(v, Struct):
            if v.data.size == 1 and v.fields and v.fields[0] == MARK:
                return self._marked(v.data.flat[0])
            data = np.empty(v.data.shape, dtype=object)
            for idx in np.ndindex(v.data.shape):
                data[idx] = {f: self.decode(v.data[idx][f]) for f in v.fields}
            return Struct(data, list(v.fields))
        if isinstance(v, _matio.OldObject):
            return self._make_object(v.class_name, self.decode(v.fields))
        if isinstance(v, _matio.StoredFunction):
            return self._stored_function(v.info)
        if isinstance(v, _matio.Opaque):
            return self._opaque(v)
        return v

    # values written by mylab ------------------------------------------------------------------
    def _marked(self, d: dict) -> Any:
        kind = to_str(d[MARK])
        if kind == "string":
            shape = tuple(int(x) for x in np.asarray(d["size"]).flatten())
            texts = _texts(d["text"])
            missing = np.asarray(d["missing"]).flatten().astype(bool)
            data = np.empty(len(texts), dtype=object)
            for k, t in enumerate(texts):
                data[k] = None if (k < missing.size and missing[k]) else t
            return string_result(data.reshape(shape, order="F"))
        if kind == "function_handle":
            ws = d.get("workspace")
            captured = {}
            if isinstance(ws, Struct) and ws.data.size:
                captured = {k: self.decode(x) for k, x in ws.data.flat[0].items()}
            return self._function(to_str(d["function"]), captured)
        if kind in ("table", "timetable"):
            return self._table(d, kind)
        if kind == "datetime":
            from .datetimes import DatetimeArray
            return DatetimeArray(np.asarray(d["seconds"], dtype=float), _opt_text(d["format"]),
                                 _opt_text(d["timezone"]) or "")
        if kind == "duration":
            from .datetimes import DurationArray
            return DurationArray(np.asarray(d["seconds"], dtype=float), _opt_text(d["format"]))
        if kind == "calendarDuration":
            from .datetimes import CalendarDuration
            return CalendarDuration(d["months"], d["days"], d["seconds"])
        if kind == "categorical":
            from .categorical import CategoricalArray
            return CategoricalArray(np.asarray(d["codes"], dtype=float), _texts(d["categories"]),
                                    bool(np.asarray(d["ordinal"]).any()))
        if kind == "containers.Map":
            from ..objects import MapObject
            if id(d) in self._made:
                return self._made[id(d)]
            m = MapObject(to_str(d["keytype"]), to_str(d["valuetype"]))
            self.made_handles = self.made_objects = True
            ident = _ident(d)
            if ident:  # registered first: its values may refer back to it
                self.handles[ident] = self._made[id(d)] = m
            for k, x in zip(d["keys"].values_f(), d["values"].values_f()):
                m.store[m.norm_key(k)] = self.decode(x)
            return m
        if kind == "dictionary":
            from .dictionary import DictionaryObject, _hashable
            out = DictionaryObject(_opt_text(d["keytype"]), _opt_text(d["valuetype"]))
            if not bool(np.asarray(d["configured"]).any()):
                out.key_type = out.value_type = None
            for k, x in zip(d["keys"].values_f(), d["values"].values_f()):
                key = self.decode(k)
                out.store[_hashable(key)] = (key, self.decode(x))
            return out
        if kind == "timeseries":
            from .timeseries import from_saved
            return from_saved({k: self.decode(x) for k, x in d.items() if k != MARK})
        if kind == "MException":
            from ..objects import MExceptionObject
            causes = [self.decode(c) for c in d["causes"].values_f()] if isinstance(d.get("causes"), Cell) else []
            stack = self.decode(d["stack"]) if "stack" in d else None
            return MExceptionObject(_opt_text(d["identifier"]) or "", _opt_text(d["message"]) or "",
                                    stack if isinstance(stack, Struct) else None,
                                    [c for c in causes if isinstance(c, MExceptionObject)])
        if kind == "enumeration":
            cls = self.interp.find_class(to_str(d["name"]))
            if cls is None:
                if not self._quiet:
                    self.warn(f"Variable '{self.var}': the enumeration class '{to_str(d['name'])}' was not found; "
                              "the member name was loaded as text.")
                return d["member"]
            return cls.enum_lookup(self.interp, d["member"])
        if kind == "object_ref":
            return self._handle(_ident(d))
        if kind == "object":
            return self._user_object(d)
        if kind == "object_array":
            return self._object_array(d)
        if kind == "sym":
            return self._sym(d)
        self.warn(f"Variable '{self.var}' holds a value of the unknown kind '{kind}'; it was loaded as a struct.")
        return Struct.scalar({k: self.decode(x) for k, x in d.items()})

    def _function(self, text_: str, captured: dict) -> FunctionHandle:
        interp = self.interp
        t = text_.strip()
        if not t.startswith("@"):
            return FunctionHandle(name=t, closure={"__functions__": interp.frame.functions})
        from .. import ast as A
        from ..interpreter import Frame
        from ..parser import parse
        try:
            program = parse(t)
            node = program.body[0].expr
            if not isinstance(node, A.Lambda):
                raise MatlabError("not an anonymous function")
        except (MatlabError, IndexError, AttributeError):
            raise MatlabError(f"load: cannot rebuild the function handle '{t}' of variable '{self.var}'.") from None
        frame = Frame(interp, interp.frame.functions)
        frame.vars.update(captured)
        f = interp.make_lambda(node, frame)
        f.text = t
        return f

    def _user_object(self, d: dict) -> Any:
        if id(d) in self._made:  # a handle object already made for a reference to it
            return self._made[id(d)]
        name = to_str(d["name"])
        ident = _ident(d)
        cls = self.interp.find_class(name)
        self.made_objects = True
        if cls is None:
            if not self._quiet:
                self.warn(f"Variable '{self.var}': the class '{name}' was not found, so the object was loaded as a "
                          "struct of its saved properties.")
            value = self.decode(d["data"])
            if ident:  # references to it get the same struct
                self.handles[ident] = self._made[id(d)] = value
            return value
        if bool(np.asarray(d["saveobj"]).any()):
            obj = self._blank(cls) if self.passive else self._loadobj(cls, self.decode(d["data"]))
        else:
            saved = d["data"]
            items = list(saved.data.flat[0].items()) if isinstance(saved, Struct) and saved.data.size else []
            obj = self._restore(cls, items, stored=True, ident=ident, key=id(d), builtin=d.get("builtin"),
                                dynamic=d.get("dynamic"))
        if ident:
            self.handles[ident] = self._made[id(d)] = obj
        return obj

    def _blank(self, cls: Any) -> Any:
        """The object load fills in: default property values, or what the constructor makes when the class
        has ConstructOnLoad (old-style objects: no fields, they all come from the file)."""
        from ..classdef import UserObject
        if cls.old_style:
            return UserObject(cls)
        attrs = getattr(getattr(cls, "cd", None), "attrs", None) or {}
        if attrs.get("ConstructOnLoad") and not getattr(cls, "is_abstract", False) and not self.passive:
            return cls.construct(self.interp, [], 1)[0]  # the constructor runs with no arguments
        return cls.new_object(self.interp)

    def _restore(self, cls: Any, items: list, stored: bool, ident: int = 0, key: int | None = None,
                 builtin: Any = None, dynamic: Any = None) -> Any:
        """An object of class ``cls`` from its saved property values (``items``: name, value pairs, still in
        their stored form when ``stored``).

        Each value is assigned through the property's validation and set method, in the saved order. A
        value that no longer fits the class (the property was removed or renamed, became Constant, fails
        its validation, or its set method fails) makes load hand loadobj a struct of the saved values,
        when the class has a loadobj method. Otherwise a removed property is dropped, a value that fails
        validation leaves the default value, and a value whose set method fails is stored as saved; the
        last two with a warning.
        """
        from ..classdef import UserObject, adopt_old_style_parent
        from .classes_more import DynamicProperty
        interp = self.interp
        obj = self._blank(cls)
        self.made_objects = True
        self.made_handles = self.made_handles or bool(cls.is_handle)
        if ident:
            self.handles[ident] = obj  # registered first: its properties may refer back to it
            if key is not None:
                self._made[key] = obj
        values = {k: self.decode(x) if stored else x for k, x in items}
        if builtin is not None and isinstance(obj, UserObject) and cls.numeric_base:
            obj._data = self.decode(builtin) if stored else builtin
        if isinstance(dynamic, Struct) and dynamic.data.size and isinstance(obj, UserObject):
            obj._dynprops = obj._dynprops or {}
            for k, x in dynamic.data.flat[0].items():
                prop = obj._dynprops[k] = DynamicProperty(obj, k)
                prop.stored = self.decode(x) if stored else x
        has_loadobj = "loadobj" in cls.methods
        misfit = False
        where = f"Variable '{self.var}': the saved value of property"
        for k, x in values.items():
            if cls.old_style:
                obj.props[k] = x
                if isinstance(x, UserObject) and x.cls.old_style and x.cls.name == k:
                    adopt_old_style_parent(cls, x.cls)  # the parent part of the object
                continue
            entry = cls.prop_defs.get(k)
            attrs = entry[0].attrs if entry is not None else {}
            if entry is None or attrs.get("Constant") or (attrs.get("Dependent") and "set." + k not in cls.methods):
                misfit = True
                continue
            if self.passive:
                if not attrs.get("Dependent"):
                    obj.props[k] = x
                continue
            try:
                x = obj.validated(interp, k, x)
            except MatlabError as e:
                misfit = True
                if not has_loadobj:
                    self.warn(f"{where} '{k}' of class '{cls.name}' is not valid for the class now, so the property "
                              f"has its default value. {e.message}", "mylab:load:propertyNotRestored")
                continue
            try:
                obj = obj.restore_field(interp, k, x)
            except MatlabError as e:
                misfit = True
                if not has_loadobj:
                    self.warn(f"{where} '{k}' of class '{cls.name}' was stored without its set method, which "
                              f"failed: {e.message}", "mylab:load:propertyNotRestored")
                    obj.props[k] = x
        if self.passive:
            return obj
        if misfit and has_loadobj:
            return self._loadobj(cls, Struct.scalar(values) if values else Struct())
        cls.arm(obj)
        return self._loadobj(cls, obj)

    def _loadobj(self, cls: Any, value: Any) -> Any:
        if "loadobj" in cls.methods:
            res = cls.call(self.interp, "loadobj", [value], 1)
            if not res:
                raise MatlabError(f"load: the loadobj method of class '{cls.name}' returned nothing "
                                  f"(variable '{self.var}').")
            return res[0]
        return value

    def _object_array(self, d: dict) -> Any:
        from ..classdef import ObjectArray
        shape = tuple(int(x) for x in np.asarray(d["size"]).flatten())
        name = to_str(d["name"]) if d["name"].size else ""
        cls = self.interp.find_class(name) if name else None
        missing = bool(name) and cls is None
        self._quiet += missing  # the elements are not warned about one by one
        try:
            items = [self.decode(x) for x in d["items"].values_f()]
        finally:
            self._quiet -= missing
        data = np.empty(len(items), dtype=object)
        for k, x in enumerate(items):
            data[k] = x
        data = data.reshape(shape, order="F")
        if cls is not None and all(hasattr(x, "cls") for x in items):
            return ObjectArray(data, cls)
        out = _struct_array(data)
        if missing:
            self.warn(f"Variable '{self.var}': the class '{name}' was not found, so the object array was loaded as "
                      + ("a struct array of the saved properties." if out is not None else
                         "a cell array of what was saved for each element."))
        return out if out is not None else Cell(data)

    def _make_object(self, name: str, fields: Any) -> Any:
        """An object from a class name and a struct of property values (old-style and MCOS objects)."""
        cls = self.interp.find_class(name) if name else None
        if cls is None or not isinstance(fields, Struct):
            self.warn(f"Variable '{self.var}' was saved as an object of class '{name}', which is not available; "
                      "it was loaded as a struct of its properties.")
            return fields
        from ..classdef import ObjectArray
        objs = [self._restore(cls, list(d.items()), stored=False) for d in fields.data.flatten(order="F")]
        if len(objs) == 1:
            return objs[0]
        data = np.empty(len(objs), dtype=object)
        for k, o in enumerate(objs):
            data[k] = o
        return ObjectArray(data.reshape(fields.data.shape, order="F"), cls)

    def _table(self, d: dict, kind: str) -> Any:
        from .table import TableObject
        from .timetables import TimetableObject
        names = _texts(d["varnames"])
        cols = [self.decode(c) for c in d["vars"].values_f()]
        desc = to_str(d["description"]) if d["description"].size else ""
        if kind == "timetable":
            t = TimetableObject(names, cols, self.decode(d["rowtimes"]), desc)
        else:
            t = TableObject(names, cols, desc, _texts(d["rownames"]) or None)
        if "units" in d:  # files of older mylab versions store only the names, data and row names
            t.units = {n: u for n, u in zip(names, _texts(d["units"])) if u}
            t.var_descriptions = {n: u for n, u in zip(names, _texts(d["vardescriptions"])) if u}
            dims = _texts(d["dimnames"])
            if len(dims) == 2:
                t.dim_names = dims
            custom = d["custom"]
            if isinstance(custom, Struct) and custom.data.size:
                t.custom = {k: self.decode(x) for k, x in custom.data.flat[0].items()}
            t.custom_var = set(_texts(d["customvar"])) & set(t.custom)
            if "userdata" in d:
                t.user_data = self.decode(d["userdata"])
        return t

    def _sym(self, d: dict) -> Any:
        fn = self.interp.builtins.get("str2sym")
        shape = tuple(int(x) for x in np.asarray(d["size"]).flatten())
        texts = _texts(d["text"])
        if fn is None:
            raise MatlabError(f"load: variable '{self.var}' is symbolic, and symbolic math is not available.")
        from .symbolic import SymObject
        data = np.empty(len(texts), dtype=object)
        for k, t in enumerate(texts):
            data[k] = fn(self.interp, [char(t)], 1).data.flat[0]
        return SymObject(data.reshape(shape, order="F"))

    # values written by MATLAB ------------------------------------------------------------------
    def _stored_function(self, info: Any) -> Any:
        try:
            fh = info.data.flat[0]["function_handle"].data.flat[0]
            text_ = to_str(fh["function"])
            kind = to_str(fh["type"]) if "type" in fh else "simple"
        except (AttributeError, KeyError, IndexError, MatlabError):
            raise MatlabError(f"load: the function handle in variable '{self.var}' cannot be read.") from None
        if kind != "anonymous":
            return self._function(text_.split("/")[-1], {})
        captured: dict = {}
        ws = fh.get("workspace")
        if isinstance(ws, _matio.Opaque):
            ws = self._opaque(ws)
        for item in (ws.values_f() if isinstance(ws, Cell) else [ws]):
            if isinstance(item, Struct) and item.data.size:
                captured.update(item.data.flat[0])
        at = text_.find("@")
        return self._function(text_[at:] if at >= 0 else text_, captured)

    def _opaque(self, v: _matio.Opaque) -> Any:
        if v.type_system != "MCOS" or self.mcos is None or not isinstance(v.payload, np.ndarray):
            self.warn(f"Variable '{self.var}' is a '{v.class_name}' object stored in a form mylab cannot read "
                      f"({v.type_system}); it was loaded as an empty array.")
            return np.zeros((0, 0))
        return self._mcos_value(v.payload, v.class_name)

    def _mcos_value(self, ref: np.ndarray, class_hint: str = "") -> Any:
        try:
            shape, ids, cls_name = self.mcos.reference(ref)
            if not any(ids):
                return np.zeros((0, 0))  # a reference to no object
            objs = [self.mcos.properties(i) for i in ids]
        except (IndexError, ValueError, KeyError, AttributeError, MatlabError):
            self.warn(f"Variable '{self.var}': the stored '{class_hint}' object cannot be read; it was loaded as "
                      "an empty array.")
            return np.zeros((0, 0))
        self._in_object += 1
        try:
            decoded = [{k: self.decode(x) for k, x in props.items()} for props in objs]
        finally:
            self._in_object -= 1
        if len(decoded) == 1:
            return self._from_matlab(cls_name, decoded[0])
        items = [self._from_matlab(cls_name, p) for p in decoded]
        from ..classdef import ObjectArray, UserObject
        data = np.empty(len(items), dtype=object)
        for k, x in enumerate(items):
            data[k] = x
        data = data.reshape(shape, order="F")
        if items and all(isinstance(x, UserObject) for x in items):
            return ObjectArray(data, items[0].cls)
        return Cell(data)

    def _from_matlab(self, name: str, p: dict) -> Any:
        """A mylab value from the saved properties of one MATLAB object of class ``name``."""
        try:
            return self._known_class(name, p)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            self.warn(f"Variable '{self.var}': the stored form of the '{name}' object was not understood; it was "
                      "loaded as a struct of its saved properties.")
            return Struct.scalar(p)

    def _known_class(self, name: str, p: dict) -> Any:
        if name == "string":
            return _unpack_strings(p["any"])
        if name == "function_handle_workspace":
            return p["any"]
        if name == "datetime":
            from .datetimes import DatetimeArray
            raw = np.asarray(p["data"])
            ms = np.real(raw).astype(float) + (np.imag(raw).astype(float) if np.iscomplexobj(raw) else 0.0)
            zone = _opt_text(p.get("tz")) or ""
            secs = ms / 1000.0
            if zone:
                from .datetimes import _rezone
                secs = _rezone(secs, "UTC", zone)
            return DatetimeArray(secs, _opt_text(p.get("fmt")), zone)
        if name == "duration":
            from .datetimes import DurationArray
            return DurationArray(np.asarray(p["millis"], dtype=float) / 1000.0, _opt_text(p.get("fmt")))
        if name == "calendarDuration":
            from .datetimes import CalendarDuration
            c = p["components"].data.flat[0]
            return CalendarDuration(c["months"], c["days"], np.asarray(c["millis"], dtype=float) / 1000.0)
        if name == "categorical":
            from .categorical import CategoricalArray
            codes = np.asarray(p["codes"]).astype(float)
            codes[codes == 0] = np.nan
            ordinal = bool(np.asarray(p.get("isOrdinal", False)).any())
            return CategoricalArray(codes, _texts(p["categoryNames"]), ordinal)
        if name in ("table", "timetable"):
            return self._matlab_table(p)
        if name == "containers.Map":
            from ..objects import MapObject
            s = p["serialization"].data.flat[0]
            m = MapObject("char" if to_str(s["keyType"]) == "char" else to_str(s["keyType"]), to_str(s["valueType"]))
            keys = s["keys"].values_f() if isinstance(s["keys"], Cell) else list(np.asarray(s["keys"]).flatten())
            vals = s["values"].values_f() if isinstance(s["values"], Cell) else \
                [np.array([[x]]) for x in np.asarray(s["values"]).flatten()]
            for k, x in zip(keys, vals):
                m.store[m.norm_key(k if isinstance(k, np.ndarray) else np.array([[k]]))] = x
            return m
        return self._make_object(name, Struct.scalar(p))

    def _matlab_table(self, p: dict) -> Any:
        from .table import TableObject
        cols = p["data"].values_f()
        names = _texts(p["varnames"])
        rows = _texts(p["rownames"]) if isinstance(p.get("rownames"), Cell) else []
        desc = ""
        props = p.get("props")
        if isinstance(props, Struct) and props.data.size and "Description" in props.fields:
            desc = _opt_text(props.data.flat[0]["Description"]) or ""
        return TableObject(names, cols, desc, rows or None)


def _ident(d: dict) -> int:
    """The handle id of a stored object or containers.Map (0: none)."""
    return int(np.asarray(d["id"]).flat[0]) if "id" in d and np.asarray(d["id"]).size else 0


def _handle_definitions(values) -> dict[int, dict]:
    """The stored handle objects and maps in ``values`` (stored values of variables), by id, in file
    order: each is written in full once, where its save call first met it."""
    out: dict[int, dict] = {}
    stack = list(values)[::-1]
    while stack:
        v = stack.pop()
        if isinstance(v, Cell):
            stack.extend(v.data.flatten(order="F")[::-1])
        elif isinstance(v, Struct):
            children = []
            for d in v.data.flatten(order="F"):
                if v.data.size == 1 and v.fields and v.fields[0] == MARK and is_char(d[MARK]) \
                        and to_str(d[MARK]) in ("object", "containers.Map"):
                    ident = _ident(d)
                    if ident:
                        out.setdefault(ident, d)
                children.extend(d.values())
            stack.extend(children[::-1])
    return out


def _struct_array(data: np.ndarray) -> Struct | None:
    """One struct array from an array of scalar structs with the same fields (None if they differ)."""
    items = list(data.flat)
    if not items or not all(isinstance(x, Struct) and x.data.size == 1 for x in items):
        return None
    fields = list(items[0].fields)
    if any(set(x.fields) != set(fields) for x in items):
        return None
    out = np.empty(data.shape, dtype=object)
    for idx in np.ndindex(data.shape):
        out[idx] = dict(data[idx].data.flat[0])
    return Struct(out, fields)


def _opt_text(v: Any) -> str | None:
    if isinstance(v, MString) or is_char(v):
        return to_str(v) or None
    return None


def _unpack_strings(raw: Any) -> Any:
    """A MATLAB string array from its saved form: a uint64 vector of sizes, lengths and UTF-16 text."""
    a = np.asarray(raw).astype(np.uint64).flatten()
    nd = int(a[1])
    shape = tuple(int(x) for x in a[2:2 + nd])
    n = int(np.prod(shape, dtype=np.int64))
    lengths = [int(x) for x in a[2 + nd:2 + nd + n]]
    blob = a[2 + nd + n:].astype("<u8").tobytes()
    out = np.empty(n, dtype=object)
    pos = 0
    for k, ln in enumerate(lengths):
        if ln == 0xFFFFFFFFFFFFFFFF:  # <missing>
            out[k] = None
            continue
        out[k] = blob[pos:pos + 2 * ln].decode("utf-16-le", errors="replace")
        pos += 2 * ln
    return string_result(out.reshape(shape, order="F"))


class Mcos:
    """The object data of a MATLAB-written file (the "FileWrapper__" cell array of its subsystem).

    The first cell is a table of names, classes and objects; the cells after the second hold
    property values. A variable that is an object is stored as a small uint32 array that starts
    with 0xDD000000 and names the object ids and the class.
    """

    REF = 0xDD000000

    def __init__(self, subsystem: bytes):
        endian = "<" if subsystem[2:4] == b"IM" else ">"
        reader = _matio.Reader(subsystem, endian)
        _, top, _, _ = reader.matrix(8)
        wrapper = top.data.flat[0]["MCOS"]
        cells = wrapper.payload.values_f()
        meta = np.asarray(cells[0]).astype(np.uint8).tobytes()
        u4 = np.frombuffer(meta[:len(meta) // 4 * 4], dtype=endian + "u4")
        self.version = int(u4[0])
        count = int(u4[1])
        offs = [int(x) for x in u4[2:10]]
        names = meta[40:offs[0]].split(b"\0")
        self.names = [n.decode("utf-8", errors="replace") for n in names[:count]]
        seg = lambda k: np.frombuffer(meta[offs[k]:offs[k + 1]], dtype=endian + "u4")
        self.classes = seg(0).reshape(-1, 4)
        self.objects = seg(2).reshape(-1, 6)
        self.props1 = self._blocks(seg(1))
        self.props2 = self._blocks(seg(3))
        self.cells = cells
        extra = 3 if self.version >= 4 else 1
        defaults = cells[len(cells) - extra] if len(cells) > extra + 1 else None
        self.defaults = defaults.values_f() if isinstance(defaults, Cell) else []

    @staticmethod
    def _blocks(words: np.ndarray) -> list[list[tuple[int, int, int]]]:
        out: list[list[tuple[int, int, int]]] = [[]]  # ids count from 1
        p = 2
        while p < words.size:
            n = int(words[p])
            p += 1
            out.append([(int(words[p + 3 * k]), int(words[p + 3 * k + 1]), int(words[p + 3 * k + 2]))
                        for k in range(n)])
            p += 3 * n
            p += p % 2  # blocks are padded to a multiple of eight bytes
        return out

    def is_reference(self, v: np.ndarray) -> bool:
        a = v.flatten(order="F")
        return a.size >= 6 and int(a[0]) == self.REF and a.size == 3 + int(a[1]) + int(np.prod(a[2:2 + int(a[1])]))

    def reference(self, v: np.ndarray) -> tuple[tuple, list[int], str]:
        a = [int(x) for x in np.asarray(v).flatten(order="F")]
        if a[0] != self.REF:
            raise ValueError("not an object reference")
        nd = a[1]
        shape = tuple(a[2:2 + nd])
        n = int(np.prod(shape, dtype=np.int64))
        ids = a[2 + nd:2 + nd + n]
        return shape, ids, self.class_name(a[2 + nd + n])

    def class_name(self, class_id: int) -> str:
        ns, name = int(self.classes[class_id][0]), int(self.classes[class_id][1])
        text_ = self.names[name - 1]
        return f"{self.names[ns - 1]}.{text_}" if ns else text_

    def properties(self, object_id: int) -> dict:
        if object_id <= 0 or object_id >= len(self.objects):
            raise IndexError("no such object")
        class_id, _, _, id1, id2, _ = (int(x) for x in self.objects[object_id])
        out: dict = {}
        if class_id < len(self.defaults) and isinstance(self.defaults[class_id], Struct) \
                and self.defaults[class_id].data.size:
            out.update(self.defaults[class_id].data.flat[0])
        blocks = self.props1[id1] if id1 else (self.props2[id2] if id2 else [])
        for name, kind, value in blocks:
            key = self.names[name - 1]
            if kind == 1:
                out[key] = self.cells[2 + value]
            elif kind == 0:
                out[key] = char(self.names[value - 1])
            else:
                out[key] = np.array([[bool(value)]])
        return out
