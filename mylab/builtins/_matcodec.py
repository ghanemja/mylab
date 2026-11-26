"""Turning mylab values into what a MAT-file can hold, and back.

A MAT-file stores numeric, logical and char arrays, sparse matrices, cells and structs. Values of
any other kind (string arrays, tables, datetimes, durations, categoricals, containers.Map,
function handles, classdef objects) are written as a scalar struct whose first field,
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
    """Encodes the variables of one save call (handle objects are written once and then referred to)."""

    def __init__(self, interp):
        self.interp = interp
        self.handles: dict[int, int] = {}
        self.keep: list = []  # keeps encoded handle objects alive so their id() stays unique

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
        ws = {k: self.encode(x, var) for k, x in f.closure.items() if k != "__functions__"}
        return _marker("function_handle", function=char(function_handle_text(f)), workspace=Struct.scalar(ws))

    def _object(self, v: Any, var: str) -> Struct:
        from ..classdef import ObjectArray, UserObject
        from ..objects import MapObject, MExceptionObject
        from .categorical import CategoricalArray
        from .datetimes import CalendarDuration, DatetimeArray, DurationArray
        from .table import TableObject
        if isinstance(v, TableObject):
            return _marker("table", varnames=_cellstr(v.names),
                           vars=Cell.from_list([self.encode(c, var) for c in v.columns]),
                           rownames=_cellstr(v.row_names or []), description=char(v.description or ""))
        if isinstance(v, DatetimeArray):
            return _marker("datetime", seconds=v.data, format=char(v.fmt or ""), timezone=char(v.zone or ""))
        if isinstance(v, DurationArray):
            return _marker("duration", seconds=v.data, format=char(v.fmt or ""))
        if isinstance(v, CalendarDuration):
            return _marker("calendarDuration", months=v.months, days=v.days, seconds=v.secs)
        if isinstance(v, CategoricalArray):
            return _marker("categorical", codes=v.data, categories=_cellstr(v.categories),
                           ordinal=np.array([[bool(v.ordinal)]]))
        if isinstance(v, MapObject):
            keys = v.sorted_keys()
            return _marker("containers.Map", keytype=char(v.key_type), valuetype=char(v.value_type),
                           keys=Cell.from_list([v.key_value(k) for k in keys]),
                           values=Cell.from_list([self.encode(v.store[k], var) for k in keys]))
        if isinstance(v, MExceptionObject):
            return _marker("MException", identifier=char(v.identifier), message=char(v.message))
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
        key = id(obj)
        if obj.is_handle:
            if key in self.handles:
                return _marker("object_ref", id=_num(self.handles[key]))
            self.handles[key] = len(self.handles) + 1
            self.keep.append(obj)
        ident = _num(self.handles.get(key, 0))
        if "saveobj" in cls.methods:
            res = cls.call(self.interp, "saveobj", [obj], 1)
            if not res:
                raise MatlabError(f"save: the saveobj method of class '{cls.name}' returned nothing "
                                  f"(variable '{var}').")
            saved = res[0]
            if saved is obj or (isinstance(saved, type(obj)) and getattr(saved, "cls", None) is cls):
                return _marker("object", name=char(cls.full_name), id=ident, saveobj=np.array([[False]]),
                               data=self._props(saved, var))
            return _marker("object", name=char(cls.full_name), id=ident, saveobj=np.array([[True]]),
                           data=self.encode(saved, var))
        return _marker("object", name=char(cls.full_name), id=ident, saveobj=np.array([[False]]),
                       data=self._props(obj, var))

    def _props(self, obj: Any, var: str) -> Struct:
        out = {}
        for name in obj.cls.instance_props():
            attrs = obj.cls.prop_defs[name][0].attrs
            if attrs.get("Transient") or attrs.get("Dependent"):
                continue
            if name in obj.props:
                out[name] = self.encode(obj.props[name], var)
        return Struct.scalar(out) if out else Struct()


# -- decoding ------------------------------------------------------------------------------------

class Decoder:
    """Decodes the variables of one load call."""

    def __init__(self, interp, subsystem: bytes | None = None):
        self.interp = interp
        self.handles: dict[int, Any] = {}
        self.mcos = Mcos(subsystem) if subsystem else None
        self.var = ""
        self._in_object = 0  # inside MATLAB object data, uint32 reference arrays stand for objects

    def warn(self, msg: str) -> None:
        self.interp.warn(msg, "MATLAB:load:classNotFound")

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
        if kind == "table":
            from .table import TableObject
            cols = [self.decode(c) for c in d["vars"].values_f()]
            return TableObject(_texts(d["varnames"]), cols, to_str(d["description"]) if d["description"].size else "",
                               _texts(d["rownames"]) or None)
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
            m = MapObject(to_str(d["keytype"]), to_str(d["valuetype"]))
            for k, x in zip(d["keys"].values_f(), d["values"].values_f()):
                m.store[m.norm_key(k)] = self.decode(x)
            return m
        if kind == "MException":
            from ..objects import MExceptionObject
            return MExceptionObject(_opt_text(d["identifier"]) or "", _opt_text(d["message"]) or "")
        if kind == "enumeration":
            cls = self.interp.find_class(to_str(d["name"]))
            if cls is None:
                self.warn(f"Variable '{self.var}': the enumeration class '{to_str(d['name'])}' was not found; "
                          "the member name was loaded as text.")
                return d["member"]
            return cls.enum_lookup(self.interp, d["member"])
        if kind == "object_ref":
            ident = int(np.asarray(d["id"]).flat[0])
            if ident not in self.handles:
                raise MatlabError(f"load: variable '{self.var}' refers to a handle object that is not in the file.")
            return self.handles[ident]
        if kind == "object":
            return self._user_object(d)
        if kind == "object_array":
            from ..classdef import ObjectArray
            items = [self.decode(x) for x in d["items"].values_f()]
            shape = tuple(int(x) for x in np.asarray(d["size"]).flatten())
            data = np.empty(len(items), dtype=object)
            for k, x in enumerate(items):
                data[k] = x
            cls = self.interp.find_class(to_str(d["name"])) if d["name"].size else None
            if cls is None or any(not hasattr(x, "cls") for x in items):
                return Cell(data.reshape(shape, order="F"))
            return ObjectArray(data.reshape(shape, order="F"), cls)
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
        name = to_str(d["name"])
        ident = int(np.asarray(d["id"]).flat[0]) if "id" in d else 0
        by_saveobj = bool(np.asarray(d["saveobj"]).any())
        cls = self.interp.find_class(name)
        if cls is None:
            self.warn(f"Variable '{self.var}': the class '{name}' was not found, so the object was loaded as a "
                      "struct of its saved properties.")
            return self.decode(d["data"])
        if by_saveobj:
            data = self.decode(d["data"])
            obj = self._loadobj(cls, data)
        else:
            from ..classdef import UserObject
            obj = UserObject(cls)
            obj.props = dict(cls.defaults(self.interp))
            if ident:
                self.handles[ident] = obj  # registered first: its properties may refer back to it
            saved = d["data"]
            if isinstance(saved, Struct) and saved.data.size:
                for k, x in saved.data.flat[0].items():
                    if k in cls.prop_defs and not cls.is_constant(k):
                        obj.props[k] = self.decode(x)
            obj = self._loadobj(cls, obj)
        if ident:
            self.handles[ident] = obj
        return obj

    def _loadobj(self, cls: Any, value: Any) -> Any:
        if "loadobj" in cls.methods:
            res = cls.call(self.interp, "loadobj", [value], 1)
            if not res:
                raise MatlabError(f"load: the loadobj method of class '{cls.name}' returned nothing "
                                  f"(variable '{self.var}').")
            return res[0]
        return value

    def _make_object(self, name: str, fields: Any) -> Any:
        """An object from a class name and a struct of property values (old-style and MCOS objects)."""
        cls = self.interp.find_class(name) if name else None
        if cls is None or not isinstance(fields, Struct):
            self.warn(f"Variable '{self.var}' was saved as an object of class '{name}', which is not available; "
                      "it was loaded as a struct of its properties.")
            return fields
        from ..classdef import ObjectArray, UserObject
        objs = []
        for d in fields.data.flatten(order="F"):
            obj = UserObject(cls)
            obj.props = dict(cls.defaults(self.interp))
            for k, x in d.items():
                if k in cls.prop_defs and not cls.is_constant(k):
                    obj.props[k] = x
            objs.append(self._loadobj(cls, obj))
        if len(objs) == 1:
            return objs[0]
        data = np.empty(len(objs), dtype=object)
        for k, o in enumerate(objs):
            data[k] = o
        return ObjectArray(data.reshape(fields.data.shape, order="F"), cls)

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
