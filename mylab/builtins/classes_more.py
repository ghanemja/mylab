"""Class introspection and the built-in mixin base classes.

- ``metaclass``, ``?Name`` (``meta.class.fromName``), ``superclasses``, ``findprop``: meta objects
  describing a class, its properties, methods, events and enumeration members.
- Methods of the base classes the engine provides: ``handle`` (findobj, findprop, eq, ...),
  ``matlab.mixin.Copyable`` (copy, copyElement), ``matlab.mixin.SetGet`` / ``hgsetget`` (set, get),
  ``dynamicprops`` (addprop), ``matlab.mixin.CustomDisplay`` (the display hooks) and
  ``matlab.mixin.Heterogeneous``.
- Helpers for classes: ``substruct``, ``subsref``, ``subsasgn``, ``inferiorto``, ``superiorto``,
  ``localfunctions``, ``mlock`` / ``munlock`` / ``mislocked``, more ``mustBe...`` validators, and
  packing of objects for ``save`` / ``load`` (``saveobj`` / ``loadobj``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..classdef import (
    BUILTIN_BASES, INDEXING_CONTEXT, NUMERIC_BASES, ClassInfo, NativeArray, NativeItem, ObjectArray,
    UserObject, access_names, base_method, class_list,
)
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char, shape_of, to_str,
    values_equal,
)
from ._registry import builtin
from ._util import bool_scalar, is_text, nargs, text

# -- meta objects -------------------------------------------------------------------------------------


def _empty() -> np.ndarray:
    return np.zeros((0, 0))


def _col(items: list, kind: str) -> Any:
    """A column array of meta objects (0-by-1 when there are none)."""
    data = np.empty((len(items), 1), dtype=object)
    for k, item in enumerate(items):
        data[k, 0] = item
    return NativeArray(data, kind)


def _cellcol(items: list) -> Cell:
    return Cell.from_list(items, column=True)


class MetaItem(NativeItem):
    """Shared behaviour of the meta.* objects: read-only properties listed in ``FIELDS``."""

    is_handle = True
    FIELDS: tuple = ()

    def value(self, interp, name: str) -> Any:
        raise NotImplementedError

    def get_field(self, interp, name: str) -> Any:
        if name in self.FIELDS or name in getattr(self, "EXTRA", ()):
            return self.value(interp, name)
        if self.array_method(name):
            return self.call_array_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> Any:
        raise MatlabError(f"Cannot set the read-only property '{name}' of a {self.kind} object.",
                          "MATLAB:class:SetProhibited")

    def array_props(self) -> list[str]:
        return list(self.FIELDS)

    def has_method(self, name: str) -> bool:
        return self.array_method(name)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return self.call_array_method(interp, name, args, nargout)

    def array_method(self, name: str) -> bool:
        return name in ("findobj", "isvalid")

    def call_array_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "isvalid":
            return [bool_scalar(True)]
        return [_findobj(interp, args)]

    def display_lines(self) -> list[str]:
        from ..display import summary
        from ..interpreter import current_interpreter
        interp = current_interpreter()
        w = max(len(n) for n in self.FIELDS)
        short = self.kind.rsplit(".", 1)[-1]
        rows = []
        for n in self.FIELDS:
            try:
                rows.append(f"    {n:>{w}}: {summary(self.value(interp, n), False)}")
            except MatlabError:
                rows.append(f"    {n:>{w}}: []")
        return [f"  {short} with properties:", ""] + rows


def _access_value(interp, level: Any) -> Any:
    """An access attribute as MATLAB reports it: 'public', ... or a cell of meta.class objects."""
    names = access_names(level)
    if names is not None:
        return _cellcol([meta_class(interp, n) for n in names])
    return char(str(level).lower() if level not in (None, True) else "public")


class MetaProperty(MetaItem):
    kind = "meta.property"
    FIELDS = ("Name", "Description", "DetailedDescription", "GetAccess", "SetAccess", "Dependent",
              "Constant", "Abstract", "Transient", "Hidden", "GetObservable", "SetObservable", "AbortSet",
              "NonCopyable", "GetMethod", "SetMethod", "HasDefault", "DefaultValue", "DefiningClass")
    EXTRA = ("Validation",)

    def __init__(self, cls: ClassInfo, name: str):
        self.info = cls
        self.name = name
        self.prop, self.owner = cls.prop_defs[name]

    def same_as(self, other: Any) -> bool:
        return isinstance(other, MetaProperty) and other.owner is self.owner and other.name == self.name

    def value(self, interp, name: str) -> Any:
        attrs = self.prop.attrs
        if name == "Name":
            return char(self.name)
        if name in ("Description", "DetailedDescription"):
            return char("")
        if name == "GetAccess":
            return _access_value(interp, attrs.get("GetAccess", attrs.get("Access")))
        if name == "SetAccess":
            return _access_value(interp, attrs.get("SetAccess", attrs.get("Access")))
        if name in ("GetMethod", "SetMethod"):
            key = ("get." if name == "GetMethod" else "set.") + self.name
            if key in self.owner.methods:
                return FunctionHandle(name=f"{self.owner.name}.{key}", closure={"__functions__": {}})
            return _empty()
        if name == "HasDefault":
            return bool_scalar(self.prop.default is not None)
        if name == "DefaultValue":
            if self.prop.default is None:
                raise MatlabError(f"The property '{self.name}' of class '{self.owner.name}' has no default "
                                  "value.", "MATLAB:class:NoDefaultDefined")
            if attrs.get("Constant"):
                return self.owner.constant(interp, self.name)
            if attrs.get("Dependent"):
                return self.owner._eval_default(interp, self.prop, self.owner)
            return self.owner.defaults(interp)[self.name]
        if name == "DefiningClass":
            return meta_class(interp, self.owner)
        if name == "Validation":
            return _empty()
        return bool_scalar(bool(attrs.get(name)))


class MetaMethod(MetaItem):
    kind = "meta.method"
    FIELDS = ("Name", "Description", "DetailedDescription", "Access", "Static", "Abstract", "Sealed",
              "ExplicitConversion", "Hidden", "InputNames", "OutputNames", "DefiningClass")

    def __init__(self, cls: ClassInfo, name: str, base: str | None = None):
        self.info = cls
        self.name = name
        self.base = base  # set for methods that come from a built-in base class such as handle
        entry = cls.methods.get(name)
        self.fn = entry[0] if entry else None
        self.owner = entry[1] if entry else cls

    def same_as(self, other: Any) -> bool:
        return isinstance(other, MetaMethod) and other.owner is self.owner and other.name == self.name

    def value(self, interp, name: str) -> Any:
        attrs = self.info.method_attrs.get(self.name, {}) if self.base is None else {}
        if name == "Name":
            return char(self.name)
        if name in ("Description", "DetailedDescription"):
            return char("")
        if name == "Access":
            return _access_value(interp, attrs.get("Access"))
        if name == "Static":
            return bool_scalar(self.name in self.info.static or self.name == "empty")
        if name == "Abstract":
            return bool_scalar(self.name in self.info.abstract_methods)
        if name == "ExplicitConversion":
            return bool_scalar(False)
        if name == "Hidden":
            return bool_scalar(bool(attrs.get("Hidden")) or self.name == "empty")
        if name == "InputNames":
            params = list(self.fn.params) if self.fn is not None else (["varargin"] if self.base else [])
            return _cellcol([char(p) for p in params])
        if name == "OutputNames":
            outs = list(self.fn.outputs) if self.fn is not None else []
            return _cellcol([char(p) for p in outs])
        if name == "DefiningClass":
            return meta_class(interp, self.base or self.owner)
        return bool_scalar(bool(attrs.get(name)))


class MetaEvent(MetaItem):
    kind = "meta.event"
    FIELDS = ("Name", "Description", "DetailedDescription", "Hidden", "NotifyAccess", "ListenAccess",
              "DefiningClass")

    def __init__(self, cls: ClassInfo, name: str):
        self.info = cls
        self.name = name

    def same_as(self, other: Any) -> bool:
        return isinstance(other, MetaEvent) and other.name == self.name and other.info is self.info

    def value(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name in ("Description", "DetailedDescription"):
            return char("")
        if name == "Hidden":
            return bool_scalar(False)
        if name in ("NotifyAccess", "ListenAccess"):
            return char("public")
        owner: Any = next((c for c in reversed(self.info.lineage()) if self.name in c.cd.events), None)
        return meta_class(interp, owner if owner is not None else "handle")


class MetaEnum(MetaItem):
    kind = "meta.EnumeratedValue"
    FIELDS = ("Name", "Description", "DetailedDescription", "Hidden")

    def __init__(self, name: str):
        self.name = name

    def value(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name == "Hidden":
            return bool_scalar(False)
        return char("")


class MetaFunction(MetaItem):
    """A package function in meta.package.FunctionList (only its name is known)."""

    kind = "meta.method"
    FIELDS = ("Name", "Description", "DetailedDescription")

    def __init__(self, name: str):
        self.name = name

    def value(self, interp, name: str) -> Any:
        return char(self.name if name == "Name" else "")


_HANDLE_METHOD_NAMES = ("addlistener", "delete", "eq", "findobj", "findprop", "ge", "gt", "isvalid", "le",
                        "listener", "lt", "ne", "notify")


class MetaClass(MetaItem):
    kind = "meta.class"
    FIELDS = ("Name", "Description", "DetailedDescription", "Hidden", "Sealed", "Abstract", "Enumeration",
              "ConstructOnLoad", "HandleCompatible", "InferiorClasses", "ContainingPackage",
              "RestrictsSubclassing", "PropertyList", "MethodList", "EventList", "EnumerationMemberList",
              "SuperclassList")
    EXTRA = ("Properties", "Methods", "Events", "SuperClasses", "Package", "Aliases")

    def __init__(self, name: str, info: ClassInfo | None = None):
        self.name = name
        self.info = info

    def same_as(self, other: Any) -> bool:
        return isinstance(other, MetaClass) and other.name == self.name

    def supers(self) -> list[str]:
        """Every superclass name, nearest first (classdef parents and built-in bases)."""
        if self.info is None:
            spec = BUILTIN_BASES.get(self.name)
            return list(spec.parents) if spec else []
        return superclass_names(self.info)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if not (isinstance(left, MetaClass) and isinstance(right, MetaClass)):
            return NotImplemented
        same = left.name == right.name
        if op == "==":
            return bool_scalar(same)
        if op == "~=":
            return bool_scalar(not same)
        if op in ("<", "<="):  # is a subclass of
            return bool_scalar(right.name in left.supers() or (op == "<=" and same))
        if op in (">", ">="):
            return bool_scalar(left.name in right.supers() or (op == ">=" and same))
        return NotImplemented

    def value(self, interp, name: str) -> Any:
        info = self.info
        if name == "Name":
            return char(self.name)
        if name in ("Description", "DetailedDescription"):
            return char("")
        attrs = info.cd.attrs if info is not None else {}
        if name == "Abstract":
            return bool_scalar(info.is_abstract if info is not None else self.name == "handle")
        if name == "Enumeration":
            return bool_scalar(bool(info and info.enums))
        if name == "HandleCompatible":
            spec = BUILTIN_BASES.get(self.name)
            return bool_scalar(bool(info.is_handle or attrs.get("HandleCompatible")) if info is not None
                               else bool(spec is not None))
        if name == "RestrictsSubclassing":
            return bool_scalar("AllowedSubclasses" in attrs)
        if name == "InferiorClasses":
            return _cellcol([meta_class(interp, n) for n in sorted(info.inferior)] if info else [])
        if name in ("ContainingPackage", "Package"):
            if "." not in self.name:
                return _empty()
            return MetaPackage(self.name.rsplit(".", 1)[0])
        if name in ("PropertyList", "Properties"):
            items = [MetaProperty(info, n) for n in ordered_props(info)] if info else []
            return _col(items, "meta.property") if name == "PropertyList" else _cellcol(items)
        if name in ("MethodList", "Methods"):
            items = self.method_list()
            return _col(items, "meta.method") if name == "MethodList" else _cellcol(items)
        if name in ("EventList", "Events"):
            items = []
            if info is not None:
                names = [e for c in info.lineage() for e in c.cd.events]
                if info.is_handle:
                    names.append("ObjectBeingDestroyed")
                items = [MetaEvent(info, n) for n in dict.fromkeys(names)]
            return _col(items, "meta.event") if name == "EventList" else _cellcol(items)
        if name == "EnumerationMemberList":
            return _col([MetaEnum(n) for n in info.enums] if info else [], "meta.EnumeratedValue")
        if name in ("SuperclassList", "SuperClasses"):
            direct = list(info.cd.superclasses) if info is not None else self.supers()[:1]
            items = [meta_class(interp, n) for n in direct]
            return _col(items, "meta.class") if name == "SuperclassList" else _cellcol(items)
        if name == "Aliases":
            return StringArray(np.empty((0, 1), dtype=object))
        return bool_scalar(bool(attrs.get(name)))

    def method_list(self) -> list:
        info = self.info
        if info is None:
            return []
        own = [n for n, (_, o) in info.methods.items() if o is info and "." not in n]
        if info.name not in own and not info.old_style:
            own.append(info.name)  # the default constructor
        inherited = [n for n, (_, o) in info.methods.items() if o is not info and "." not in n]
        items = [MetaMethod(info, n) for n in own + inherited]
        items.append(MetaMethod(info, "empty", base=None))
        if info.is_handle:
            items += [MetaMethod(info, n, base="handle") for n in _HANDLE_METHOD_NAMES
                      if n not in info.methods]
        return items


class MetaPackage(MetaItem):
    kind = "meta.package"
    FIELDS = ("Name", "Description", "DetailedDescription", "ClassList", "FunctionList", "PackageList",
              "ContainingPackage")

    def __init__(self, name: str):
        self.name = name

    def same_as(self, other: Any) -> bool:
        return isinstance(other, MetaPackage) and other.name == self.name

    def folders(self, interp) -> list[str]:
        import os
        dirs = list(interp.path)
        if os.getcwd() not in dirs:
            dirs.append(os.getcwd())
        out = []
        for d in dirs:
            folder = os.path.join(d, *("+" + part for part in self.name.split(".")))
            if os.path.isdir(folder):
                out.append(folder)
        return out

    def members(self, interp) -> tuple[list[str], list[str], list[str]]:
        """Names of the classes, functions and sub-packages in the package's folders."""
        import os
        classes: list[str] = []
        functions: list[str] = []
        packages: list[str] = []
        for folder in self.folders(interp):
            for entry in sorted(os.listdir(folder)):
                full = os.path.join(folder, entry)
                if entry.startswith("+") and os.path.isdir(full):
                    packages.append(entry[1:])
                elif entry.startswith("@") and os.path.isdir(full):
                    classes.append(entry[1:])
                elif entry.endswith(".m"):
                    program = interp._load_file(full)
                    (classes if program is not None and program.classdefs else functions).append(entry[:-2])
        return classes, functions, packages

    def value(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name in ("Description", "DetailedDescription"):
            return char("")
        if name == "ContainingPackage":
            return MetaPackage(self.name.rsplit(".", 1)[0]) if "." in self.name else _empty()
        classes, functions, packages = self.members(interp)
        if name == "ClassList":
            return _col([meta_class(interp, f"{self.name}.{c}") for c in classes], "meta.class")
        if name == "FunctionList":
            return _col([MetaFunction(f) for f in functions], "meta.method")
        return _col([MetaPackage(f"{self.name}.{p}") for p in packages], "meta.package")


def ordered_props(info: ClassInfo) -> list[str]:
    """Property names in MATLAB's listing order: the class's own first, then inherited ones."""
    own = [n for n, (_, o) in info.prop_defs.items() if o is info]
    return own + [n for n, (_, o) in info.prop_defs.items() if o is not info]


def superclass_names(info: ClassInfo) -> list[str]:
    names: list[str] = []

    def add(n: str) -> None:
        if n not in names:
            names.append(n)

    def walk(c: ClassInfo) -> None:
        for sup in c.cd.superclasses:
            add(sup)
        for p in c.parents:
            add(p.full_name)
        for p in c.parents:
            walk(p)
        for sup in c.cd.superclasses:
            spec = BUILTIN_BASES.get(sup)
            for b in (spec.parents if spec else ()):
                add(b)

    walk(info)
    return names


def meta_class(interp, what: Any) -> Any:
    """The meta.class object of a ClassInfo or class name (one object per class)."""
    cache = interp.state.setdefault("meta_classes", {})
    if isinstance(what, ClassInfo):
        info: ClassInfo | None = what
        name = what.full_name
    else:
        name = str(what)
        info = interp.find_class(name)
        if info is not None:
            name = info.full_name
    made = cache.get(name)
    if made is None or made.info is not info:
        made = cache[name] = MetaClass(name, info)
    return made


_BUILTIN_CLASSES = {"double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                    "uint64", "logical", "char", "cell", "struct", "function_handle", "string", "table",
                    "datetime", "duration", "categorical", "containers.Map", "MException", "inputParser"}


def _class_info(interp, v: Any, what: str = "class") -> ClassInfo | None:
    if isinstance(v, (UserObject, ObjectArray)) and not isinstance(v, NativeArray):
        return v.cls
    if is_text(v):
        return interp.find_class(text(v))
    return None


@builtin("metaclass")
def _metaclass(interp, args, nargout):
    nargs(args, 1, 1, "metaclass")
    v = args[0]
    if isinstance(v, (UserObject, ObjectArray)) and not isinstance(v, NativeArray) and v.cls is not None:
        return meta_class(interp, v.cls)
    return meta_class(interp, class_name(v))


@builtin("meta.class.fromName")
def _meta_from_name(interp, args, nargout):
    nargs(args, 1, 1, "meta.class.fromName")
    name = text(args[0], "class name")
    if interp.find_class(name) is None and name not in BUILTIN_BASES and name not in _BUILTIN_CLASSES:
        return NativeArray(np.empty((0, 0), dtype=object), "meta.class")  # no such class: empty
    return meta_class(interp, name)


@builtin("meta.package.fromName")
def _meta_package(interp, args, nargout):
    nargs(args, 1, 1, "meta.package.fromName")
    pkg = MetaPackage(text(args[0], "package name"))
    if not pkg.folders(interp):
        return NativeArray(np.empty((0, 0), dtype=object), "meta.package")
    return pkg


@builtin("superclasses")
def _superclasses(interp, args, nargout):
    nargs(args, 1, 1, "superclasses")
    v = args[0]
    info = _class_info(interp, v)
    if info is None:
        name = text(v) if is_text(v) else class_name(v)
        if is_text(v) and name not in BUILTIN_BASES and name not in _BUILTIN_CLASSES:
            raise MatlabError(f"Class '{name}' was not found.", "MATLAB:class:InvalidClass")
        spec = BUILTIN_BASES.get(name)
        names = list(spec.parents) if spec else []
    else:
        names = superclass_names(info)
    if nargout:
        return _cellcol([char(n) for n in names])
    label = info.full_name if info is not None else (text(v) if is_text(v) else class_name(v))
    if not names:
        interp.out(f"\nNo superclasses for class {label}.\n\n")
    else:
        interp.out(f"\nSuperclasses for class {label}:\n\n" + "".join(f"    {n}\n" for n in names) + "\n")
    return None


def _findprop(interp, args, nargout):
    nargs(args, 2, 2, "findprop")
    obj = args[0]
    name = text(args[1], "property name")
    if isinstance(obj, ObjectArray) and obj.data.size:
        obj = obj.data.flat[0]
    if isinstance(obj, UserObject):
        if obj._dynprops and name in obj._dynprops:
            return obj._dynprops[name]
        if name in obj.cls.prop_defs:
            return MetaProperty(obj.cls, name)
        return NativeArray(np.empty((0, 0), dtype=object), "meta.property")
    raise MatlabError("findprop needs a handle object and a property name.")


# -- findobj for handle objects ---------------------------------------------------------------------------

def _prop_value(interp, obj: Any, name: str) -> tuple[bool, Any]:
    try:
        return True, obj.get_field(interp, name)
    except MatlabError:
        return False, None


def _findobj(interp, args: list) -> Any:
    """findobj(H, 'Prop', value, ...) on handle objects, with -and / -or / -not / -isa / -property."""
    if not args:
        raise MatlabError("findobj needs an array of handle objects.")
    src = args[0]
    if isinstance(src, ObjectArray):
        items = src.elements()
    elif isinstance(src, MObject):
        items = [src]
    else:
        raise MatlabError("findobj needs an array of handle objects.")
    rest = list(args[1:])

    def test(obj: Any) -> bool:
        result: bool | None = None
        joiner = "-and"
        negate = False
        k = 0
        while k < len(rest):
            a = rest[k]
            key = text(a) if is_text(a) else None
            if key is None:
                raise MatlabError("findobj: expected a property name or an option such as '-isa'.")
            low = key.lower()
            if low in ("-and", "-or", "-xor"):
                joiner = low
                k += 1
                continue
            if low == "-not":
                negate = not negate
                k += 1
                continue
            if low in ("-isa", "-property", "-method", "-event", "-function", "-depth", "-regexp"):
                if k + 1 >= len(rest):
                    raise MatlabError(f"findobj: the option '{key}' needs a value.")
                arg = rest[k + 1]
                if low == "-isa":
                    ok = obj.isa_class(text(arg))
                elif low == "-property":
                    ok = _has_prop(obj, text(arg))
                elif low == "-method":
                    ok = obj.has_method(text(arg))
                elif low == "-event":
                    ok = isinstance(obj, UserObject) and text(arg) in obj.cls.events
                elif low == "-function":
                    res = interp.call_function_value(arg, [obj], 1)
                    ok = bool(res and np.all(np.asarray(res[0])))
                elif low == "-depth":
                    ok = True
                else:
                    raise MatlabError("findobj: '-regexp' is not supported for handle objects.")
                k += 2
            else:
                if k + 1 >= len(rest):
                    raise MatlabError(f"findobj: no value was given for the property '{key}'.")
                found, value = _prop_value(interp, obj, key)
                ok = found and values_equal(value, rest[k + 1])
                k += 2
            if negate:
                ok, negate = not ok, False
            if result is None:
                result = ok
            elif joiner == "-and":
                result = result and ok
            elif joiner == "-or":
                result = result or ok
            else:
                result = result != ok
            joiner = "-and"
        return True if result is None else result

    picked = [o for o in items if not getattr(o, "_deleted", False) and test(o)]
    if isinstance(src, NativeArray) or (items and not isinstance(items[0], UserObject)):
        kind = src.kind if isinstance(src, NativeArray) else items[0].class_name()
        return _col(picked, kind) if len(picked) != 1 else picked[0]
    data = np.empty((len(picked), 1), dtype=object)
    for k, o in enumerate(picked):
        data[k, 0] = o
    klass = src.cls if isinstance(src, (ObjectArray, UserObject)) else None
    return ObjectArray._unwrap(data, klass) if picked else ObjectArray(np.empty((0, 0), dtype=object), klass)


def _has_prop(obj: Any, name: str) -> bool:
    if isinstance(obj, UserObject):
        return name in obj.cls.prop_defs or bool(obj._dynprops and name in obj._dynprops)
    return name in getattr(obj, "FIELDS", ())


@base_method("handle", "findobj")
def _handle_findobj(interp, args, nargout):
    return _findobj(interp, list(args))


@base_method("handle", "findprop")
def _handle_findprop(interp, args, nargout):
    return _findprop(interp, args, nargout)


def _handle_compare(op: str):
    def f(interp, args, nargout):
        nargs(args, 2, 2, op)
        from ..classdef import raw_builtin
        return raw_builtin(interp, op, list(args), 1)[0]
    return f


for _name in ("eq", "ne"):
    base_method("handle", _name)(_handle_compare(_name))


# -- matlab.mixin.Copyable ----------------------------------------------------------------------------------

def shallow_copy(interp, obj: UserObject) -> UserObject:
    """A new handle object with the same property values (NonCopyable ones get their defaults)."""
    other = UserObject(obj.cls)
    other.props = dict(obj.props)
    defaults = obj.cls.defaults(interp)
    for name, (prop, _) in obj.cls.prop_defs.items():
        if prop.attrs.get("NonCopyable") and name in other.props:
            other.props[name] = defaults.get(name, _empty())
    other._data = obj._data
    obj.cls.arm(other)
    if obj._dynprops:
        other._dynprops = {}
        for name, dyn in obj._dynprops.items():
            if not dyn.attrs["NonCopyable"]:
                made = DynamicProperty(other, name)
                made.attrs = dict(dyn.attrs)
                made.stored = dyn.stored
                other._dynprops[name] = made
    return other


@base_method("matlab.mixin.Copyable", "copyElement")
def _copy_element(interp, args, nargout):
    nargs(args, 1, 1, "copyElement")
    if not isinstance(args[0], UserObject):
        raise MatlabError("copyElement needs one object.")
    return shallow_copy(interp, args[0])


@base_method("matlab.mixin.Copyable", "copy")
def _copy(interp, args, nargout):
    nargs(args, 1, 1, "copy")
    src = args[0]

    def one(obj: UserObject) -> UserObject:
        obj._check_valid()
        if "copyElement" in obj.cls.methods:  # the class customises copying
            res = obj.cls.call(interp, "copyElement", [obj], 1)
            if not res or not isinstance(res[0], UserObject):
                raise MatlabError(f"copyElement of class '{obj.cls.name}' must return the copy.")
            return res[0]
        return shallow_copy(interp, obj)

    if isinstance(src, ObjectArray):
        data = np.empty(src.data.shape, dtype=object)
        for idx in np.ndindex(src.data.shape):
            data[idx] = one(src.data[idx])
        return ObjectArray(data, src.cls)
    return one(src)


# -- matlab.mixin.SetGet / hgsetget ---------------------------------------------------------------------------

def _settable(obj: UserObject, public_only: bool = True) -> list[str]:
    names = [n for n in ordered_props(obj.cls) if not obj.cls.is_constant(n)]
    out = []
    for n in names:
        attrs = obj.cls.prop_defs[n][0].attrs
        level = str(attrs.get("SetAccess", attrs.get("Access", "public"))).lower()
        if level == "public" or not public_only:
            out.append(n)
    if obj._dynprops:
        out += list(obj._dynprops)
    return out


def _match_prop(obj: UserObject, key: str, names: list[str]) -> str:
    """The property ``key`` names: exact for SetGetExactNames, else case-insensitive and abbreviated."""
    if key in names:
        return key
    if "matlab.mixin.SetGetExactNames" not in obj.cls.bases:
        low = key.lower()
        exact = [n for n in names if n.lower() == low]
        if len(exact) == 1:
            return exact[0]
        partial = [n for n in names if n.lower().startswith(low)] if low else []
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise MatlabError(f"The name '{key}' is ambiguous for class '{obj.cls.name}': it matches "
                              + ", ".join(partial) + ".", "MATLAB:class:AmbiguousProperty")
    raise MatlabError(f"The name '{key}' is not an accessible property for an instance of class "
                      f"'{obj.cls.name}'.", "MATLAB:class:InvalidProperty")


def _readable(obj: UserObject) -> list[str]:
    names = [n for n in obj.cls.public_props() if not obj.cls.is_constant(n)]
    own = [n for n in ordered_props(obj.cls) if n in names]
    if obj._dynprops:
        own += [n for n, d in obj._dynprops.items() if d.attrs["GetAccess"] == "public"]
    return own


def _objects(v: Any, what: str) -> list:
    if isinstance(v, ObjectArray) and not isinstance(v, NativeArray):
        return v.elements()
    if isinstance(v, UserObject):
        return [v]
    raise MatlabError(f"{what}: the first argument must be an object of a class derived from "
                      "matlab.mixin.SetGet.")


def _names_list(v: Any) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x, "property name") for x in v.values_f()]
    if isinstance(v, StringArray):
        return [x or "" for x in v.values_f()]
    return [text(v, "property name")]


@base_method("matlab.mixin.SetGet", "set")
@base_method("hgsetget", "set")
def _set(interp, args, nargout):
    if not args:
        raise MatlabError("set needs an object.")
    objs = _objects(args[0], "set")
    rest = list(args[1:])
    if not rest:  # set(obj): the properties that can be set
        info = Struct.scalar({n: Cell(np.empty((0, 0), dtype=object)) for n in _settable(objs[0])})
        if nargout:
            return info
        for n in info.fields:
            interp.out(f"    {n}: {{}}\n")
        return None
    if len(rest) == 1 and not isinstance(rest[0], Struct):  # set(obj, 'Name'): possible values
        _match_prop(objs[0], _names_list(rest[0])[0], _settable(objs[0]))
        return Cell(np.empty((0, 0), dtype=object)) if nargout else None
    pairs: list[tuple[str, Any]] = []
    k = 0
    while k < len(rest):
        a = rest[k]
        if isinstance(a, Struct):
            if a.numel() != 1:
                raise MatlabError("set: a struct of property values must be 1-by-1.")
            pairs += [(f, a.data.flat[0][f]) for f in a.fields]
            k += 1
        elif isinstance(a, Cell) and k + 1 < len(rest) and isinstance(rest[k + 1], Cell):
            names, values = _names_list(a), rest[k + 1]
            if values.data.shape[-1] != len(names) or values.data.ndim != 2:
                raise MatlabError("set: the cell array of values needs one column per property name.")
            if values.data.shape[0] not in (1, len(objs)):
                raise MatlabError("set: the cell array of values needs one row, or one row per object.")
            for row in range(values.data.shape[0]):
                targets = objs if values.data.shape[0] == 1 else [objs[row]]
                for obj in targets:
                    for j, n in enumerate(names):
                        obj.set_field(interp, _match_prop(obj, n, _settable(obj, False)), values.data[row, j])
            k += 2
        else:
            if k + 1 >= len(rest):
                raise MatlabError("set: property names and values must come in pairs.",
                                  "MATLAB:class:BadParamValuePairs")
            pairs.append((text(a, "property name"), rest[k + 1]))
            k += 2
    for obj in objs:
        for key, value in pairs:
            obj.set_field(interp, _match_prop(obj, key, _settable(obj, False)), value)
    return None


@base_method("matlab.mixin.SetGet", "get")
@base_method("hgsetget", "get")
def _get(interp, args, nargout):
    if not args:
        raise MatlabError("get needs an object.")
    nargs(args, 1, 2, "get")
    objs = _objects(args[0], "get")
    if len(args) == 1:  # get(obj): every readable property
        data = np.empty((len(objs), 1), dtype=object)
        fields = _readable(objs[0]) if objs else []
        for k, obj in enumerate(objs):
            data[k, 0] = {n: obj.get_field(interp, n) for n in fields}
        s = Struct(data, fields)
        if nargout:
            return s
        from ..display import display_lines
        interp.out("\n".join(display_lines(s)) + "\n")
        return None
    names = _names_list(args[1])
    as_cell = isinstance(args[1], (Cell, StringArray)) or len(objs) != 1
    out = np.empty((len(objs), len(names)), dtype=object)
    for i, obj in enumerate(objs):
        for j, n in enumerate(names):
            out[i, j] = obj.get_field(interp, _match_prop(obj, n, _readable(obj)))
    return Cell(out) if as_cell else out[0, 0]


for _base in ("matlab.mixin.SetGet", "hgsetget"):
    @base_method(_base, "setdisp")
    def _setdisp(interp, args, nargout):
        return _set(interp, args[:1], 0)

    @base_method(_base, "getdisp")
    def _getdisp(interp, args, nargout):
        return _get(interp, args[:1], 0)


# -- dynamicprops ------------------------------------------------------------------------------------------

class DynamicProperty(NativeItem):
    """What addprop returns: the description of a property added to one object. Deleting it removes it."""

    kind = "meta.DynamicProperty"
    is_handle = True
    FIELDS = ("Name", "Description", "DetailedDescription", "GetAccess", "SetAccess", "Dependent",
              "Constant", "Abstract", "Transient", "Hidden", "GetObservable", "SetObservable", "AbortSet",
              "NonCopyable", "GetMethod", "SetMethod", "HasDefault", "DefiningClass")
    _FLAGS = ("Dependent", "Transient", "Hidden", "GetObservable", "SetObservable", "AbortSet",
              "NonCopyable")

    def __init__(self, owner: UserObject, name: str):
        import weakref
        self._owner = weakref.ref(owner)  # the object owns its dynamic properties, not the reverse
        self.name = name
        self.stored: Any = _empty()
        self.attrs: dict[str, Any] = {f: False for f in self._FLAGS}
        self.attrs.update({"GetAccess": "public", "SetAccess": "public", "GetMethod": _empty(),
                           "SetMethod": _empty(), "Description": "", "DetailedDescription": ""})
        self.busy = False

    @property
    def owner(self) -> Any:
        return self._owner()

    @property
    def hidden(self) -> bool:
        return bool(self.attrs["Hidden"])

    @property
    def get_access(self) -> str:
        return str(self.attrs["GetAccess"])

    def _allowed(self, interp, kind: str) -> bool:
        level = self.attrs[kind]
        owner = self.owner
        return level == "public" or (owner is not None and owner.cls.allowed(interp, level, owner.cls))

    def read(self, interp, obj: UserObject) -> Any:
        if not self._allowed(interp, "GetAccess"):
            raise MatlabError(f"You cannot access the '{self.name}' property of {obj.cls.name}.",
                              "MATLAB:class:GetProhibited")
        getter = self.attrs["GetMethod"]
        if isinstance(getter, FunctionHandle) and not self.busy:
            self.busy = True
            try:
                res = interp.call_function_value(getter, [obj], 1)
            finally:
                self.busy = False
            if not res:
                raise MatlabError(f"The GetMethod of '{self.name}' did not return a value.")
            return res[0]
        return self.stored

    def write(self, interp, obj: UserObject, value: Any) -> None:
        if not self._allowed(interp, "SetAccess"):
            raise MatlabError(f"You cannot set the read-only property '{self.name}' of {obj.cls.name}.",
                              "MATLAB:class:SetProhibited")
        if self.attrs["AbortSet"] and values_equal(self.stored, value):
            return
        from .events import property_event
        if obj._listeners:
            property_event(interp, obj, self.name, "PreSet")
        setter = self.attrs["SetMethod"]
        if isinstance(setter, FunctionHandle) and not self.busy:
            self.busy = True
            try:
                interp.call_function_value(setter, [obj, value], 0)
            finally:
                self.busy = False
        else:
            self.stored = value
        if obj._listeners:
            property_event(interp, obj, self.name, "PostSet")

    def get_field(self, interp, name: str) -> Any:
        if name == "Name":
            return char(self.name)
        if name in ("Constant", "Abstract", "HasDefault"):
            return bool_scalar(False)
        if name == "DefiningClass":
            return _empty()
        if name in self._FLAGS:
            return bool_scalar(bool(self.attrs[name]))
        if name in self.attrs:
            v = self.attrs[name]
            return char(v) if isinstance(v, str) else v
        raise MatlabError(f"Unrecognized property '{name}' for class 'meta.DynamicProperty'.")

    def set_field(self, interp, name: str, value: Any) -> "DynamicProperty":
        self._check()
        if name in self._FLAGS:
            self.attrs[name] = bool(np.asarray(value).flat[0])
        elif name in ("GetAccess", "SetAccess"):
            level = text(value, name).lower()
            if level not in ("public", "protected", "private"):
                raise MatlabError(f"{name} must be 'public', 'protected' or 'private'.")
            self.attrs[name] = level
        elif name in ("GetMethod", "SetMethod"):
            if not isinstance(value, FunctionHandle) and not (isinstance(value, np.ndarray) and value.size == 0):
                raise MatlabError(f"{name} must be a function handle.")
            self.attrs[name] = value
        elif name in ("Description", "DetailedDescription"):
            self.attrs[name] = text(value, name)
        else:
            raise MatlabError(f"Cannot set the property '{name}' of a meta.DynamicProperty.",
                              "MATLAB:class:SetProhibited")
        return self

    def _check(self) -> None:
        if getattr(self, "_deleted", False):
            raise MatlabError("Invalid or deleted object.", "MATLAB:class:InvalidHandle")

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "isvalid":
            return [bool_scalar(not getattr(self, "_deleted", False))]
        owner = self.owner
        if owner is not None and owner._dynprops and owner._dynprops.get(self.name) is self:
            del owner._dynprops[self.name]  # delete(p): the object loses the property
        self._deleted = True
        return []

    def array_props(self) -> list[str]:
        return list(self.FIELDS)

    def display_lines(self) -> list[str]:
        from ..display import summary
        w = max(len(n) for n in self.FIELDS)
        return ["  DynamicProperty with properties:", ""] + \
            [f"    {n:>{w}}: {summary(self.get_field(None, n), False)}" for n in self.FIELDS]


@base_method("dynamicprops", "addprop")
def _addprop(interp, args, nargout):
    nargs(args, 2, 2, "addprop")
    targets = args[0].elements() if isinstance(args[0], ObjectArray) else [args[0]]
    name = text(args[1], "property name")
    if not name.isidentifier():
        raise MatlabError(f"'{name}' is not a valid property name.", "MATLAB:class:InvalidPropertyName")
    made = []
    for obj in targets:
        if not isinstance(obj, UserObject):
            raise MatlabError("addprop needs an object of a class derived from dynamicprops.")
        obj._check_valid()
        if name in obj.cls.prop_defs or name in obj.cls.methods or (obj._dynprops and name in obj._dynprops):
            raise MatlabError(f"The name '{name}' is already used by a property or method of class "
                              f"'{obj.cls.name}'.", "MATLAB:class:PropertyInUse")
        if obj._dynprops is None:
            obj._dynprops = {}
        dyn = obj._dynprops[name] = DynamicProperty(obj, name)
        made.append(dyn)
    if not nargout:
        return None
    return made[0] if len(made) == 1 else NativeArray.wrap(made, "meta.DynamicProperty")


# -- matlab.mixin.CustomDisplay ------------------------------------------------------------------------------

class PropertyGroup(NativeItem):
    """matlab.mixin.util.PropertyGroup: a titled list of properties (or name/value pairs) to show."""

    kind = "matlab.mixin.util.PropertyGroup"

    def __init__(self, plist: Any, title: str = ""):
        self.plist = plist
        self.title = title

    def get_field(self, interp, name: str) -> Any:
        if name == "Title":
            return char(self.title)
        if name == "PropertyList":
            return self.plist
        if name == "NumProperties":
            return np.array([[float(len(self.names()))]])
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")

    def set_field(self, interp, name: str, value: Any) -> "PropertyGroup":
        other = PropertyGroup(self.plist, self.title)
        if name == "Title":
            other.title = text(value, "Title")
        elif name == "PropertyList":
            other.plist = value
        else:
            raise MatlabError(f"Cannot set the property '{name}' of a PropertyGroup.")
        return other

    def names(self) -> list[str]:
        if isinstance(self.plist, Struct):
            return list(self.plist.fields)
        if isinstance(self.plist, Cell):
            return [to_str(x) for x in self.plist.values_f()]
        if isinstance(self.plist, StringArray):
            return [x or "" for x in self.plist.values_f()]
        if isinstance(self.plist, np.ndarray) and self.plist.size == 0:
            return []
        return [text(self.plist, "property list")]

    def pairs(self, interp, obj: Any) -> list[tuple[str, Any]]:
        if isinstance(self.plist, Struct):
            return [(f, self.plist.data.flat[0][f]) for f in self.plist.fields]
        return [(n, obj.get_field(interp, n)) for n in self.names()]

    def array_props(self) -> list[str]:
        return ["Title", "NumProperties", "PropertyList"]

    def display_lines(self) -> list[str]:
        return ["  PropertyGroup with properties:", "", f"            Title: '{self.title}'",
                f"    NumProperties: {len(self.names())}"]


@builtin("matlab.mixin.util.PropertyGroup")
def _property_group(interp, args, nargout):
    nargs(args, 0, 2, "matlab.mixin.util.PropertyGroup")
    plist = args[0] if args else Cell(np.empty((0, 0), dtype=object))
    if not isinstance(plist, (Struct, Cell, StringArray)) and not is_text(plist) \
            and not (isinstance(plist, np.ndarray) and plist.size == 0):
        raise MatlabError("A PropertyGroup takes a cell array of property names or a struct of values.")
    return PropertyGroup(plist, text(args[1], "title") if len(args) > 1 else "")


def _size_string(shape: tuple) -> str:
    return "×".join(str(d) for d in shape)


def _class_link(obj: Any) -> str:
    return obj.cls.full_name if getattr(obj, "cls", None) is not None else obj.class_name()


def _custom(interp, obj: Any, name: str, *extra: Any) -> list:
    """Call display hook ``name``: the class's override when there is one, else the default."""
    if name in obj.cls.methods:
        return obj.cls.call(interp, name, [obj, *extra], 1 if name.startswith("get") else 0)
    return interp.call_builtin(name, _CUSTOM_DEFAULTS[name], [obj, *extra], 1)


def _default_header(interp, args, nargout):
    obj = args[0]
    name = _class_link(obj)
    if isinstance(obj, ObjectArray):
        if obj.data.size == 0:
            head = f"  {_size_string(obj.data.shape)} empty {name} array"
        else:
            head = f"  {_size_string(obj.data.shape)} {name} array"
        props = [p for p in obj.cls.public_props() if not obj.cls.is_constant(p)]
        return char(head + (" with properties:" if props else " with no properties.") + "\n")
    if getattr(obj, "_deleted", False):
        return char(f"  handle to deleted {name}\n")
    return char(f"  {name} with properties:\n" if obj.shown_props() else f"  {name} with no properties.\n")


def _default_groups(interp, args, nargout):
    obj = args[0]
    if isinstance(obj, ObjectArray):
        names = [p for p in obj.cls.public_props() if not obj.cls.is_constant(p)] if obj.cls else []
    else:
        names = obj.shown_props()
    return PropertyGroup(Cell.from_list([char(n) for n in names]))


def _default_footer(interp, args, nargout):
    return char("")


def _group_lines(interp, obj: Any, groups: Any) -> list[str]:
    from ..display import summary
    items = groups.elements() if isinstance(groups, NativeArray) else [groups]
    lines: list[str] = []
    scalar = isinstance(obj, UserObject)
    width = 0
    rows: list[list[tuple[str, Any]]] = []
    for g in items:
        if not isinstance(g, PropertyGroup):
            raise MatlabError("getPropertyGroups must return matlab.mixin.util.PropertyGroup objects.")
        if scalar or isinstance(g.plist, Struct):
            pairs = g.pairs(interp, obj)
        else:
            pairs = [(n, None) for n in g.names()]
        rows.append(pairs)
        width = max([width] + [len(n) for n, _ in pairs])
    for g, pairs in zip(items, rows):
        if g.title:
            lines += [f"  {g.title}", ""] if lines else [f"   {g.title}"]
        for n, v in pairs:
            if scalar or isinstance(g.plist, Struct):
                lines.append(f"    {n:>{width}}: {summary(v, False)}")
            else:
                lines.append(f"    {n}")
        lines.append("")
    return lines


def _display_groups(interp, args, nargout):
    """matlab.mixin.CustomDisplay.displayPropertyGroups(obj, groups)."""
    nargs(args, 2, 2, "displayPropertyGroups")
    lines = _group_lines(interp, args[0], args[1])
    interp.out("\n".join(lines) + ("\n" if lines else ""))
    return None


def _write(interp, textv: Any) -> None:
    s = to_str(textv) if (is_char(textv) or isinstance(textv, MString)) else ""
    if s:
        interp.out(s if s.endswith("\n") else s + "\n")


def _display_scalar(interp, args, nargout):
    obj = args[0]
    header = _custom(interp, obj, "getHeader")
    _write(interp, header[0] if header else char(""))
    groups = _custom(interp, obj, "getPropertyGroups")
    if groups:
        lines = _group_lines(interp, obj, groups[0])
        if lines:
            interp.out("\n" + "\n".join(lines) + "\n")
    footer = _custom(interp, obj, "getFooter")
    _write(interp, footer[0] if footer else char(""))
    return None


def _display_deleted(interp, args, nargout):
    header = _custom(interp, args[0], "getHeader")
    _write(interp, header[0] if header else char(""))
    return None


_CUSTOM_DEFAULTS = {
    "getHeader": _default_header, "getPropertyGroups": _default_groups, "getFooter": _default_footer,
    "displayScalarObject": _display_scalar, "displayNonScalarObject": _display_scalar,
    "displayEmptyObject": _display_scalar, "displayScalarHandleToDeletedObject": _display_deleted,
    "displayPropertyGroups": _display_groups,
}
for _name, _fn in _CUSTOM_DEFAULTS.items():
    base_method("matlab.mixin.CustomDisplay", _name)(_fn)


@base_method("matlab.mixin.CustomDisplay", "disp")
def _custom_disp(interp, args, nargout):
    obj = args[0]
    if isinstance(obj, ObjectArray):
        hook = "displayEmptyObject" if obj.data.size == 0 else "displayNonScalarObject"
    elif getattr(obj, "_deleted", False):
        hook = "displayScalarHandleToDeletedObject"
    else:
        hook = "displayScalarObject"
    _custom(interp, obj, hook)
    return None


@base_method("matlab.mixin.CustomDisplay", "display")
def _custom_display(interp, args, nargout):
    name = text(args[1]) if len(args) > 1 else "ans"
    interp.out(f"{name} = \n\n")
    _custom_disp(interp, args[:1], 0)
    interp.out("\n")
    return None


@builtin("matlab.mixin.CustomDisplay.getClassNameForHeader")
def _class_for_header(interp, args, nargout):
    nargs(args, 1, 1, "getClassNameForHeader")
    return char(_class_link(args[0]))


@builtin("matlab.mixin.CustomDisplay.getSimpleHeader", "matlab.mixin.CustomDisplay.getDetailedHeader")
def _simple_header(interp, args, nargout):
    nargs(args, 1, 1, "getSimpleHeader")
    return _default_header(interp, args, 1)


@builtin("matlab.mixin.CustomDisplay.getDeletedHandleText")
def _deleted_text(interp, args, nargout):
    return char("handle to deleted")


@builtin("matlab.mixin.CustomDisplay.getHandleText")
def _handle_text(interp, args, nargout):
    return char("handle")


@builtin("matlab.mixin.CustomDisplay.getDetailedFooter", "matlab.mixin.CustomDisplay.getSimpleFooter")
def _detailed_footer(interp, args, nargout):
    return char("")


@builtin("matlab.mixin.CustomDisplay.convertDimensionsToString")
def _dims_string(interp, args, nargout):
    nargs(args, 1, 1, "convertDimensionsToString")
    return char(_size_string(shape_of(args[0])))


@builtin("matlab.mixin.CustomDisplay.displayPropertyGroups")
def _display_groups_static(interp, args, nargout):
    return _display_groups(interp, args, nargout)


# -- matlab.mixin.Heterogeneous -----------------------------------------------------------------------------

@base_method("matlab.mixin.Heterogeneous", "getDefaultScalarElement")
def _default_scalar(interp, args, nargout):
    raise MatlabError("getDefaultScalarElement has no default: define it as a Static, Sealed, protected "
                      "method of the root class when that class is abstract.",
                      "MATLAB:class:DefaultObjectSubstitution")


_default_scalar.static = True  # type: ignore[attr-defined]


# -- matlab.mixin.util.IndexingContext and matlab.indexing.IndexingOperationType ---------------------------

class EnumValue(NativeItem):
    """A member of a small built-in enumeration (compares by name; char/string give the name)."""

    def __init__(self, kind: str, name: str):
        self.kind = kind
        self.name = name

    def same_as(self, other: Any) -> bool:
        return isinstance(other, EnumValue) and other.kind == self.kind and other.name == self.name

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("==", "~="):
            return NotImplemented
        other = right if left is self else left
        if isinstance(other, EnumValue):
            same = self.same_as(other)
        elif is_text(other):
            same = text(other) == self.name
        else:
            return NotImplemented
        return bool_scalar(same if op == "==" else not same)

    def has_method(self, name: str) -> bool:
        return name in ("char", "string")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return [char(self.name) if name == "char" else MString(self.name)]

    def summary_text(self, in_cell: bool) -> str:
        return self.name

    def display_lines(self) -> list[str]:
        return [f"  {self.kind.rsplit('.', 1)[-1]} enumeration", "", f"    {self.name}"]


def _enum_group(kind: str, names: tuple, key: str = "") -> None:
    for n in names:
        value = EnumValue(kind, n)
        INDEXING_CONTEXT[key + n] = value
        builtin(f"{kind}.{n}")(lambda interp, args, nargout, _v=value: _v)


_enum_group("matlab.mixin.util.IndexingContext", ("Statement", "Expression", "Assignment", "Unknown"))
_enum_group("matlab.indexing.IndexingOperationType", ("Paren", "Dot", "Brace"), "Type:")


@builtin("numArgumentsFromSubscript")
def _num_args(interp, args, nargout):
    nargs(args, 2, 3, "numArgumentsFromSubscript")
    from ..classdef import raw_builtin
    return raw_builtin(interp, "numArgumentsFromSubscript", list(args), 1)[0]


@builtin("listLength")
def _list_length(interp, args, nargout):
    """listLength(obj, indexOp, indexingContext): values a forwarded indexing operation produces."""
    nargs(args, 3, 3, "listLength")
    from ..classdef import index_op_levels
    levels = index_op_levels(args[1])
    if levels is None:
        raise MatlabError("listLength needs the indexing operations a parenReference, dotReference or "
                          "braceReference method received.")
    if not levels or levels[-1][0] == "()":
        return 1.0
    value = args[0]
    if len(levels) > 1:
        from ..classdef import _chain_frame
        tmp, chain = _chain_frame(interp, value, levels[:-1], raw=False)
        value = interp.eval(chain, tmp)
    kind, subs = levels[-1]
    if kind == "." and isinstance(value, (Struct, ObjectArray)):
        return float(int(np.prod(shape_of(value))))
    if kind == "{}" and isinstance(value, Cell):
        from .. import indexing as ix
        picked = ix.get(value.data, [ix.COLON if (is_char(a) and a.size == 1 and a.flat[0] == ":") else a
                                     for a in subs])
        return float(picked.size)
    return 1.0


def _size_of(interp, obj: Any) -> tuple:
    """The size a class derived from RedefinesParen reports through its size method."""
    res = obj.cls.call(interp, "size", [obj], 1) if "size" in obj.cls.methods else []
    if not res:
        return tuple(shape_of(obj))
    return tuple(int(x) for x in np.real(np.asarray(res[0])).flatten())


def _from_size(name: str, fn):
    @base_method("matlab.mixin.indexing.RedefinesParen", name)
    def method(interp, args, nargout):
        nargs(args, 1, 1, name)
        return fn(_size_of(interp, args[0]))
    return method


_from_size("numel", lambda d: float(int(np.prod(d))))
_from_size("ndims", lambda d: float(len(d)))
_from_size("length", lambda d: float(0 if 0 in d else max(d)))
_from_size("isempty", lambda d: bool_scalar(0 in d))
_from_size("isscalar", lambda d: bool_scalar(all(x == 1 for x in d)))
_from_size("isvector", lambda d: bool_scalar(len(d) == 2 and 1 in d))
_from_size("isrow", lambda d: bool_scalar(len(d) == 2 and d[0] == 1))
_from_size("iscolumn", lambda d: bool_scalar(len(d) == 2 and d[1] == 1))
_from_size("ismatrix", lambda d: bool_scalar(len(d) == 2))


# -- old-style classes: inferiorto / superiorto -----------------------------------------------------------

def _rank(interp, args, superior: bool, what: str) -> None:
    owner = getattr(interp.frame, "owner_class", None)
    if owner is None:
        raise MatlabError(f"{what} can only be called from a class constructor.",
                          "MATLAB:class:InvalidContext")
    for a in args:
        name = text(a, "class name")
        if superior:
            owner.inferior.add(name)
        else:
            owner.superior_to_me.add(name)


@builtin("superiorto")
def _superiorto(interp, args, nargout):
    _rank(interp, args, True, "superiorto")
    return None


@builtin("inferiorto")
def _inferiorto(interp, args, nargout):
    _rank(interp, args, False, "inferiorto")
    return None


# -- function utilities: localfunctions, mlock ---------------------------------------------------------------

@builtin("localfunctions")
def _localfunctions(interp, args, nargout):
    nargs(args, 0, 0, "localfunctions")
    frame = interp.frame
    from ..interpreter import NestedRef
    file = frame.file
    names = []
    main = None
    if file:
        program = interp._load_file(file)
        if program is not None and program.functions and not program.body and not program.classdefs:
            main = next(iter(program.functions))  # the file's main function is not a local function
    for name, fn in frame.functions.items():
        if isinstance(fn, NestedRef) or name == main:
            continue
        if file and getattr(fn, "file", file) != file:
            continue
        names.append(name)
    scope = interp._handle_scope(frame)
    return _cellcol([FunctionHandle(name=n, closure=scope) for n in names])


@builtin("mlock")
def _mlock(interp, args, nargout):
    nargs(args, 0, 0, "mlock")
    if interp.frame is interp.base:
        raise MatlabError("mlock can only be called from a function.")
    interp.state.setdefault("mlocked", set()).add(interp.frame.name)
    return None


@builtin("munlock")
def _munlock(interp, args, nargout):
    nargs(args, 0, 1, "munlock")
    name = text(args[0], "function name") if args else interp.frame.name
    interp.state.setdefault("mlocked", set()).discard(name)
    return None


@builtin("mislocked")
def _mislocked(interp, args, nargout):
    nargs(args, 0, 1, "mislocked")
    name = text(args[0], "function name") if args else interp.frame.name
    return bool_scalar(name in interp.state.get("mlocked", set()))


# -- more validators -----------------------------------------------------------------------------------------

def _fail(name: str, message: str):
    raise MatlabError(message, f"MATLAB:validators:{name}")


def _dims(v: Any) -> tuple:
    return tuple(shape_of(v))


@builtin("mustBeScalarOrEmpty")
def _must_scalar_or_empty(interp, args, nargout):
    nargs(args, 1, 1, "mustBeScalarOrEmpty")
    d = _dims(args[0])
    if int(np.prod(d)) > 1:
        _fail("mustBeScalarOrEmpty", "Value must be scalar or empty.")
    return None


@builtin("mustBeRow")
def _must_row(interp, args, nargout):
    nargs(args, 1, 1, "mustBeRow")
    d = _dims(args[0])
    if not (len(d) == 2 and d[0] == 1):
        _fail("mustBeRow", "Value must be a row vector.")
    return None


@builtin("mustBeColumn")
def _must_column(interp, args, nargout):
    nargs(args, 1, 1, "mustBeColumn")
    d = _dims(args[0])
    if not (len(d) == 2 and d[1] == 1):
        _fail("mustBeColumn", "Value must be a column vector.")
    return None


@builtin("mustBeMatrix")
def _must_matrix(interp, args, nargout):
    nargs(args, 1, 1, "mustBeMatrix")
    if len(_dims(args[0])) != 2:
        _fail("mustBeMatrix", "Value must be a matrix (two dimensions).")
    return None


@builtin("mustBeFloat")
def _must_float(interp, args, nargout):
    nargs(args, 1, 1, "mustBeFloat")
    v = args[0]
    if isinstance(v, UserObject) and v._data is not None:
        v = v._data
    if not (isinstance(v, np.ndarray) and v.dtype.kind in "fc"):
        _fail("mustBeFloat", "Value must be a floating-point array (double or single).")
    return None


@builtin("mustBeUnderlyingType")
def _must_underlying(interp, args, nargout):
    nargs(args, 2, 2, "mustBeUnderlyingType")
    v = args[0]
    names = _names_list(args[1])
    if isinstance(v, UserObject) and v._data is not None:
        v = v._data
    if class_name(v) not in names:
        _fail("mustBeUnderlyingType", "Value must have one of these underlying types: "
              + ", ".join(names) + ".")
    return None


@builtin("mustBeNonmissing")
def _must_nonmissing(interp, args, nargout):
    nargs(args, 1, 1, "mustBeNonmissing")
    v = args[0]
    missing = False
    if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
        missing = bool(np.isnan(v).any())
    elif isinstance(v, StringArray):
        missing = any(x is None for x in v.data.flat)
    elif isinstance(v, MObject) and "ismissing" in interp.builtins and not isinstance(v, (UserObject, ObjectArray)):
        res = interp.call_builtin("ismissing", interp.builtins["ismissing"], [v], 1)
        missing = bool(res and np.any(np.asarray(res[0])))
    if missing:
        _fail("mustBeNonmissing", "Value must not contain missing values.")
    return None


@builtin("mustBeNonsparse")
def _must_nonsparse(interp, args, nargout):
    nargs(args, 1, 1, "mustBeNonsparse")
    if getattr(args[0], "is_sparse", False) or type(args[0]).__name__.lower().startswith("sparse"):
        _fail("mustBeNonsparse", "Value must not be sparse.")
    return None


@builtin("mustBeSparse")
def _must_sparse(interp, args, nargout):
    nargs(args, 1, 1, "mustBeSparse")
    if not (getattr(args[0], "is_sparse", False) or type(args[0]).__name__.lower().startswith("sparse")):
        _fail("mustBeSparse", "Value must be sparse.")
    return None


@builtin("mustBeValidVariableName")
def _must_varname(interp, args, nargout):
    nargs(args, 1, 1, "mustBeValidVariableName")
    import re
    from ..lexer import KEYWORDS
    v = args[0]
    ok = is_text(v)
    if ok:
        s = text(v)
        ok = bool(re.fullmatch(r"[A-Za-z]\w{0,62}", s)) and s not in KEYWORDS
    if not ok:
        _fail("mustBeValidVariableName", "Value must be a valid variable name.")
    return None


@builtin("mustBeBetween")
def _must_between(interp, args, nargout):
    nargs(args, 3, 4, "mustBeBetween")
    v, lo, hi = (np.real(np.asarray(a, dtype=float)) for a in args[:3])
    kind = text(args[3]).lower() if len(args) > 3 else "closed"
    if kind not in ("closed", "open", "openleft", "openright", "closedleft", "closedright"):
        raise MatlabError(f"mustBeBetween: unknown interval type '{kind}'.")
    left_open = kind in ("open", "openleft", "closedright")
    right_open = kind in ("open", "openright", "closedleft")
    ok = np.all((v > lo) if left_open else (v >= lo)) and np.all((v < hi) if right_open else (v <= hi))
    if not ok:
        _fail("mustBeBetween", "Value must be within the given interval.")
    return None


# -- saving and loading objects ------------------------------------------------------------------------------

_STREAM_FIELD = "MylabObjectStream"


def has_objects(v: Any) -> bool:
    """Whether a value is (or holds) a classdef object, so that save needs :func:`pack_value`."""
    if isinstance(v, (UserObject, ObjectArray)) and not isinstance(v, NativeArray):
        return True
    if isinstance(v, Cell):
        return any(has_objects(x) for x in v.data.flat)
    if isinstance(v, Struct):
        return any(has_objects(x) for d in v.data.flat for x in d.values())
    return False


def object_to_saved(interp, obj: UserObject, memo: dict) -> dict:
    """What save stores for one object: its class and property values (after the saveobj hook)."""
    key = id(obj)
    if obj.is_handle and key in memo:
        return {"k": "ref", "id": memo[key]}
    cls = obj.cls
    if obj.is_handle:
        memo[key] = len(memo) + 1
    ident = memo.get(key, 0)
    if obj._enum_name is not None:
        return {"k": "enum", "cls": cls.full_name, "name": obj._enum_name}
    if "saveobj" in cls.methods:
        res = cls.call(interp, "saveobj", [obj], 1)
        if not res:
            raise MatlabError(f"saveobj of class '{cls.name}' did not return a value.")
        if res[0] is not obj:
            return {"k": "saveobj", "cls": cls.full_name, "id": ident, "value": _encode(interp, res[0], memo)}
    props = {}
    for name, value in obj.props.items():
        entry = cls.prop_defs.get(name)
        if entry is not None and (entry[0].attrs.get("Transient") or entry[0].attrs.get("Dependent")
                                  or entry[0].attrs.get("Constant")):
            continue
        props[name] = _encode(interp, value, memo)
    out = {"k": "obj", "cls": cls.full_name, "id": ident, "props": props, "old": bool(cls.old_style)}
    if obj._data is not None:
        out["data"] = _encode(interp, obj._data, memo)
    if obj._dynprops:
        out["dyn"] = {n: _encode(interp, d.stored, memo) for n, d in obj._dynprops.items()
                      if not d.attrs["Transient"]}
    return out


def _encode(interp, v: Any, memo: dict) -> Any:
    import base64
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return {"k": "char", "shape": list(v.shape), "text": "".join(v.flatten(order="F").tolist())}
        return {"k": "num", "dtype": v.dtype.str, "shape": list(v.shape),
                "b": base64.b64encode(np.asfortranarray(v).tobytes(order="F")).decode("ascii")}
    if isinstance(v, MString):
        return {"k": "str", "text": str(v)}
    if isinstance(v, StringArray):
        return {"k": "strs", "shape": list(v.data.shape), "items": list(v.data.flatten(order="F"))}
    if isinstance(v, Cell):
        return {"k": "cell", "shape": list(v.data.shape),
                "items": [_encode(interp, x, memo) for x in v.data.flatten(order="F")]}
    if isinstance(v, Struct):
        return {"k": "struct", "shape": list(v.data.shape), "fields": list(v.fields),
                "items": [{f: _encode(interp, d.get(f, _empty()), memo) for f in v.fields}
                          for d in v.data.flatten(order="F")]}
    if isinstance(v, FunctionHandle):
        from ..display import function_handle_text
        return {"k": "fh", "text": function_handle_text(v) if v.is_anonymous else "@" + str(v.name)}
    if isinstance(v, NativeArray):
        raise MatlabError(f"Objects of class '{v.kind}' cannot be saved.")
    if isinstance(v, ObjectArray):
        return {"k": "objs", "cls": v.cls.full_name if v.cls else "", "shape": list(v.data.shape),
                "items": [object_to_saved(interp, e, memo) for e in v.data.flatten(order="F")]}
    if isinstance(v, UserObject):
        return object_to_saved(interp, v, memo)
    if isinstance(v, MObject) and v.class_name() == "containers.Map":
        keys = v.sorted_keys()
        return {"k": "map", "kt": v.key_type, "vt": v.value_type, "keys": keys,
                "values": [_encode(interp, v.store[key], memo) for key in keys]}
    raise MatlabError(f"Values of class '{class_name(v)}' inside an object cannot be saved.")


def pack_value(interp, v: Any) -> np.ndarray:
    """Encode a value holding objects as bytes (a uint8 row) that :func:`unpack_value` restores."""
    import json
    payload = json.dumps({"v": 1, "value": _encode(interp, v, {})})
    return np.frombuffer(payload.encode("utf-8"), dtype=np.uint8).reshape(1, -1).copy()


def _shape(d: dict) -> tuple:
    return tuple(int(x) for x in d["shape"])


def _decode(interp, d: Any, memo: dict) -> Any:
    import base64
    k = d["k"]
    if k == "num":
        raw = np.frombuffer(base64.b64decode(d["b"]), dtype=np.dtype(d["dtype"]))
        return raw.reshape(_shape(d), order="F").copy()
    if k == "char":
        return np.array(list(d["text"]), dtype="<U1").reshape(_shape(d), order="F")
    if k == "str":
        return MString(d["text"])
    if k in ("strs", "cell", "struct", "objs"):
        if k == "strs":
            items = list(d["items"])
        elif k == "struct":
            items = [{f: _decode(interp, x[f], memo) for f in d["fields"]} for x in d["items"]]
        else:
            items = [_decode(interp, x, memo) for x in d["items"]]
        flat = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            flat[i] = item
        data = flat.reshape(_shape(d), order="F")
        if k == "strs":
            return StringArray(data)
        if k == "cell":
            return Cell(data)
        if k == "struct":
            return Struct(data, list(d["fields"]))
        return ObjectArray(data, interp.find_class(d["cls"]) if d["cls"] else None)
    if k == "fh":
        return interp.eval_expression(d["text"]) if d["text"].startswith("@(") else \
            FunctionHandle(name=d["text"][1:], closure={"__functions__": {}})
    if k == "map":
        from ..objects import MapObject
        m = MapObject(d["kt"], d["vt"])
        for key, value in zip(d["keys"], d["values"]):
            m.store[key] = _decode(interp, value, memo)
        return m
    if k == "ref":
        return memo[d["id"]]
    cls = interp.find_class(d["cls"])
    if cls is None:
        interp.warn(f"Class '{d['cls']}' was not found; its saved object is loaded as a struct.",
                    "MATLAB:load:classNotFound")
        return Struct.scalar({n: _decode(interp, x, memo) for n, x in d.get("props", {}).items()})
    if k == "enum":
        return cls.enum_members(interp)[d["name"]]
    if k == "saveobj":
        value = _decode(interp, d["value"], memo)
        if "loadobj" in cls.methods:
            res = cls.call(interp, "loadobj", [value], 1)
            if not res:
                raise MatlabError(f"loadobj of class '{cls.name}' did not return a value.")
            value = res[0]
        if d.get("id"):
            memo[d["id"]] = value
        return value
    if cls.old_style:
        obj = UserObject(cls)
    elif cls.cd.attrs.get("ConstructOnLoad") and not cls.is_abstract:
        obj = cls.construct(interp, [], 1)[0]  # ConstructOnLoad: the constructor runs with no arguments
    else:
        obj = cls.new_object(interp)
    if d.get("id"):
        memo[d["id"]] = obj
    for name, value in d.get("props", {}).items():
        if name in cls.prop_defs or cls.old_style or name in obj.props:
            obj.props[name] = _decode(interp, value, memo)
    if "data" in d:
        obj._data = _decode(interp, d["data"], memo)
    for name, value in d.get("dyn", {}).items():
        if obj._dynprops is None:
            obj._dynprops = {}
        obj._dynprops[name] = DynamicProperty(obj, name)
        obj._dynprops[name].stored = _decode(interp, value, memo)
    cls.arm(obj)
    if "loadobj" in cls.methods:
        res = cls.call(interp, "loadobj", [obj], 1)
        if res:
            return res[0]
    return obj


def unpack_value(interp, raw: Any) -> Any:
    """The value :func:`pack_value` encoded."""
    import json
    data = bytes(np.asarray(raw, dtype=np.uint8).flatten().tolist()).decode("utf-8")
    return _decode(interp, json.loads(data)["value"], {})


STREAM_FIELD = _STREAM_FIELD
__all__ = ["meta_class", "pack_value", "unpack_value", "has_objects", "STREAM_FIELD", "NUMERIC_BASES",
           "class_list"]
