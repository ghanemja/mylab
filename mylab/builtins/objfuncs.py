"""Validation functions (mustBe*) and object utilities (isvalid, methods, properties, ...)."""

from __future__ import annotations

import os

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, StringArray, char, class_name, is_char, to_str, values_equal
from ._registry import builtin
from ._util import bool_scalar, is_text, num, text


def _fail(name: str, message: str):
    raise MatlabError(message, f"MATLAB:validators:{name}")


def _numeric(v, name: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biufc":
        _fail(name, "Value must be numeric or logical.")
    return v.astype(float) if v.dtype == bool else v


def _check(name: str, test, message: str, need_numeric: bool = True):
    def f(interp, args, nargout):
        v = args[0]
        a = _numeric(v, name) if need_numeric else v
        if not test(a, *args[1:]):
            _fail(name, message)
        return None
    builtin(name)(f)


_check("mustBePositive", lambda a: bool(np.all(np.real(a) > 0)), "Value must be positive.")
_check("mustBeNonnegative", lambda a: bool(np.all(np.real(a) >= 0)), "Value must be nonnegative.")
_check("mustBeNegative", lambda a: bool(np.all(np.real(a) < 0)), "Value must be negative.")
_check("mustBeNonpositive", lambda a: bool(np.all(np.real(a) <= 0)), "Value must be nonpositive.")
_check("mustBeNonzero", lambda a: bool(np.all(a != 0)), "Value must not be zero.")
_check("mustBeInteger", lambda a: bool(np.all(np.isfinite(a)) and np.all(np.real(a) == np.round(np.real(a)))),
       "Value must be integer.")
_check("mustBeFinite", lambda a: bool(np.all(np.isfinite(a))), "Value must be finite.")
_check("mustBeNonNan", lambda a: not bool(np.any(np.isnan(a))), "Value must not be NaN.")
_check("mustBeReal", lambda a: not np.iscomplexobj(a), "Value must be real.")
_check("mustBeNumeric", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "iufc",
       "Value must be numeric.", need_numeric=False)
_check("mustBeNumericOrLogical", lambda a: isinstance(a, np.ndarray) and a.dtype.kind in "biufc",
       "Value must be numeric or logical.", need_numeric=False)
_check("mustBeNonempty", lambda a: not (hasattr(a, "size") and a.size == 0) and
       not (isinstance(a, (Cell, StringArray)) and a.data.size == 0), "Value must not be empty.", need_numeric=False)
_check("mustBeVector", lambda a: len(np.shape(getattr(a, "data", a))) == 2 and 1 in np.shape(getattr(a, "data", a)),
       "Value must be a vector.", need_numeric=False)
_check("mustBeText", lambda a: isinstance(a, (MString, StringArray)) or is_char(a) or
       (isinstance(a, Cell) and all(is_char(x) for x in a.data.flat)), "Value must be text.", need_numeric=False)
_check("mustBeTextScalar", lambda a: isinstance(a, MString) or (is_char(a) and a.ndim == 2 and a.shape[0] <= 1),
       "Value must be a single piece of text.", need_numeric=False)
_check("mustBeNonzeroLengthText", lambda a: (isinstance(a, MString) and len(a) > 0) or (is_char(a) and a.size > 0),
       "Value must be text with one or more characters.", need_numeric=False)
_check("mustBeGreaterThan", lambda a, c: bool(np.all(np.real(a) > num(c))), "Value must be greater than the limit.")
_check("mustBeGreaterThanOrEqual", lambda a, c: bool(np.all(np.real(a) >= num(c))),
       "Value must be greater than or equal to the limit.")
_check("mustBeLessThan", lambda a, c: bool(np.all(np.real(a) < num(c))), "Value must be less than the limit.")
_check("mustBeLessThanOrEqual", lambda a, c: bool(np.all(np.real(a) <= num(c))),
       "Value must be less than or equal to the limit.")


def _in_range(a, lo, hi, *flags) -> bool:
    """mustBeInRange(value, lower, upper, flags...): flags move the boundaries out of the range."""
    lower_open = upper_open = False
    for f in flags:
        key = text(f, "boundary flag").lower()
        if key == "exclusive":
            lower_open = upper_open = True
        elif key == "exclude-lower":
            lower_open = True
        elif key == "exclude-upper":
            upper_open = True
        elif key != "inclusive":
            raise MatlabError(f"mustBeInRange: unknown boundary flag '{key}'; use 'inclusive', 'exclusive', "
                              "'exclude-lower' or 'exclude-upper'.")
    v = np.real(a)
    low = (v > num(lo)) if lower_open else (v >= num(lo))
    high = (v < num(hi)) if upper_open else (v <= num(hi))
    return bool(np.all(low & high))


_check("mustBeInRange", _in_range, "Value must be within the allowed range.")
_check("mustBeFile", lambda a: os.path.isfile(to_str(a)), "The file must exist.", need_numeric=False)
_check("mustBeFolder", lambda a: os.path.isdir(to_str(a)), "The folder must exist.", need_numeric=False)


@builtin("mustBeMember")
def _must_be_member(interp, args, nargout):
    v, allowed = args[0], args[1]
    if isinstance(allowed, (Cell, StringArray)):
        options = [to_str(x) if not isinstance(x, str) else x for x in allowed.data.flatten(order="F")]
        values = [to_str(x) for x in v.data.flatten(order="F")] if isinstance(v, (Cell,)) else \
            ([x for x in v.data.flatten(order="F")] if isinstance(v, StringArray) else [to_str(v)])
        ok = all(x in options for x in values)
    else:
        ok = bool(np.all(np.isin(num(v), num(allowed))))
    if not ok:
        _fail("mustBeMember", "Value must be a member of the allowed set.")
    return None


@builtin("mustBeA")
def _must_be_a(interp, args, nargout):
    v, classes = args[0], args[1]
    names = [to_str(x) for x in classes.data.flatten(order="F")] if isinstance(classes, Cell) else \
        ([x for x in classes.data.flatten(order="F")] if isinstance(classes, StringArray) else [text(classes)])
    ok = any((v.isa_class(n) if hasattr(v, "isa_class") else class_name(v) == n) for n in names)
    if not ok:
        _fail("mustBeA", f"Value must be one of these classes: {', '.join(names)}.")
    return None


# -- object utilities -------------------------------------------------------------------------

@builtin("isvalid")
def _isvalid(interp, args, nargout):
    v = args[0]
    return bool_scalar(isinstance(v, MObject) and not getattr(v, "_deleted", False))


def _class_of(v):
    from ..classdef import NativeArray, ObjectArray, UserObject
    if isinstance(v, (UserObject, ObjectArray)) and not isinstance(v, NativeArray):
        return v.cls
    return None


def _public(level) -> bool:
    return str(level if level not in (None, True) else "public").lower() == "public"


def public_property_names(interp, v) -> list[str]:
    """Property names ``properties`` lists: public, not Hidden, the class's own before inherited ones."""
    from ..classdef import ObjectArray, UserObject
    cls = _class_of(v)
    if cls is None:
        return []
    if cls.old_style:
        first = v.data.flat[0] if isinstance(v, ObjectArray) and v.data.size else v
        return list(first.props) if isinstance(first, UserObject) else []
    own = [n for n, (_, o) in cls.prop_defs.items() if o is cls]
    names = own + [n for n in cls.prop_defs if n not in own]
    out = []
    for n in names:
        attrs = cls.prop_defs[n][0].attrs
        if _public(attrs.get("GetAccess", attrs.get("Access"))) and not attrs.get("Hidden"):
            out.append(n)
    if isinstance(v, UserObject) and v._dynprops:
        out += [n for n, d in v._dynprops.items() if not d.hidden and d.get_access == "public"]
    return out


_HANDLE_PUBLIC = ("addlistener", "delete", "eq", "findobj", "findprop", "ge", "gt", "isvalid", "le",
                  "listener", "lt", "ne", "notify")
_PROTECTED_BASE = {"copyElement", "getHeader", "getFooter", "getPropertyGroups", "displayScalarObject",
                   "displayNonScalarObject", "displayEmptyObject", "displayScalarHandleToDeletedObject",
                   "displayPropertyGroups", "getDefaultScalarElement", "setdisp", "getdisp"}


def public_method_names(cls) -> list[str]:
    """Method names ``methods`` lists: public, not Hidden, with the constructor and inherited ones."""
    names = {n for n in cls.methods if "." not in n
             and _public(cls.method_attrs.get(n, {}).get("Access"))
             and not cls.method_attrs.get(n, {}).get("Hidden")}
    if _public(cls.method_attrs.get(cls.name, {}).get("Access")):
        names.add(cls.name)  # the constructor (the default one when the class does not write its own)
    names |= {n for n in cls.base_methods if n not in _PROTECTED_BASE}
    if cls.is_handle:
        names |= set(_HANDLE_PUBLIC)
    return sorted(names, key=lambda n: (n.lower(), n))


def _member_lists(interp, v):
    cls = _class_of(v)
    if cls is None and is_text(v):
        cls = interp.find_class(text(v))
        if cls is not None:
            from ..classdef import UserObject
            return cls.full_name, public_property_names(interp, UserObject(cls)), public_method_names(cls)
    if cls is not None:
        return cls.full_name, public_property_names(interp, v), public_method_names(cls)
    if isinstance(v, MObject):
        methods = sorted(getattr(v, "_METHODS", []))
        props = ["Count", "KeyType", "ValueType"] if v.class_name() == "containers.Map" else \
            list(getattr(v, "FIELDS", []) or (v.array_props() if hasattr(v, "array_props") else []))
        return v.class_name(), props, methods
    raise MatlabError("Expected an object or a class name.")


@builtin("properties")
def _properties(interp, args, nargout):
    if not args:
        raise MatlabError("properties needs an object or a class name.")
    try:
        name, props, _ = _member_lists(interp, args[0])
    except MatlabError:
        if is_text(args[0]):
            raise MatlabError(f"Class '{text(args[0])}' was not found.", "MATLAB:class:InvalidClass") from None
        name, props = class_name(args[0]), []  # a built-in type has no properties
    if nargout:
        return Cell.from_list([char(p) for p in props], column=True)
    if not props:
        interp.out(f"\nNo properties for class {name}.\n\n")
    else:
        interp.out(f"\nProperties for class {name}:\n\n" + "".join(f"    {p}\n" for p in props) + "\n")
    return None


@builtin("methods")
def _methods(interp, args, nargout):
    if not args:
        raise MatlabError("methods needs an object or a class name.")
    full = len(args) > 1 and is_text(args[1]) and text(args[1]) == "-full"
    if len(args) > 1 and not full:
        raise MatlabError("methods: the only option is '-full'.")
    try:
        name, _, methods = _member_lists(interp, args[0])
    except MatlabError:
        if is_text(args[0]):
            raise MatlabError(f"Class '{text(args[0])}' was not found.", "MATLAB:class:InvalidClass") from None
        name, methods = class_name(args[0]), []
    if full:
        cls = _class_of(args[0]) or (interp.find_class(text(args[0])) if is_text(args[0]) else None)
        described = []
        for n in methods:
            entry = cls.methods.get(n) if cls is not None else None
            if entry is None:
                described.append(n)
                continue
            fn = entry[0]
            outs = ", ".join(fn.outputs)
            lhs = f"[{outs}] = " if len(fn.outputs) > 1 else (f"{outs} = " if outs else "")
            static = "Static " if n in cls.static else ""
            described.append(f"{static}{lhs}{n}({', '.join(fn.params)})")
        methods = described
    if nargout:
        return Cell.from_list([char(p) for p in methods], column=True)
    interp.out(f"\nMethods for class {name}:\n\n" + ("\n".join(methods) if full else "  ".join(methods))
               + "\n\n")
    return None


@builtin("isprop")
def _isprop(interp, args, nargout):
    from ..classdef import ObjectArray, UserObject
    if len(args) != 2:
        raise MatlabError("isprop needs an object and a property name.")
    v, name = args[0], text(args[1], "property name")

    def one(obj) -> bool:
        if isinstance(obj, UserObject):
            return (name in obj.cls.prop_defs and not obj.cls.old_style) or bool(
                obj._dynprops and name in obj._dynprops)
        try:
            return name in _member_lists(interp, obj)[1]
        except MatlabError:
            return False

    if isinstance(v, ObjectArray) and _class_of(v) is not None:
        flags = [one(e) for e in v.elements()]
        return np.array(flags, dtype=bool).reshape(v.data.shape, order="F")
    return bool_scalar(one(v))


@builtin("ismethod")
def _ismethod(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("ismethod needs an object and a method name.")
    try:
        _, _, methods = _member_lists(interp, args[0])
    except MatlabError:
        return bool_scalar(False)
    return bool_scalar(text(args[1], "method name") in methods)


@builtin("events")
def _events(interp, args, nargout):
    if not args:
        raise MatlabError("events needs an object or a class name.")
    cls = _class_of(args[0])
    if cls is None and is_text(args[0]):
        cls = interp.find_class(text(args[0]))
        if cls is None:
            raise MatlabError(f"Class '{text(args[0])}' was not found.", "MATLAB:class:InvalidClass")
    names: list[str] = []
    if cls is not None:
        for c in cls.lineage():
            names += [e for e in c.cd.events if e not in names]
        if cls.is_handle:
            names.append("ObjectBeingDestroyed")
    label = cls.full_name if cls is not None else class_name(args[0])
    if nargout:
        return Cell.from_list([char(n) for n in names], column=True)
    if not names:
        interp.out(f"\nNo events for class {label}.\n\n")
    else:
        interp.out(f"\nEvents for class {label}:\n\n" + "".join(f"    {n}\n" for n in names) + "\n")
    return None


def delete_object(interp, obj) -> None:
    """delete(h) for handle objects: run the class's delete method, then invalidate it."""
    from ..classdef import ObjectArray, UserObject
    if isinstance(obj, ObjectArray):
        for element in obj.elements():
            delete_object(interp, element)
        return
    if getattr(obj, "_deleted", False):
        return
    if isinstance(obj, UserObject):
        listeners = (obj._listeners or {}).get("ObjectBeingDestroyed")
        if listeners:
            from .events import EventData, _fire
            _fire(interp, listeners, [obj, EventData(obj, "ObjectBeingDestroyed")])
        if "delete" in obj.cls.methods:
            obj.cls.call(interp, "delete", [obj], 0)
    elif obj.has_method("delete"):
        obj.call_method(interp, "delete", [obj], 0)
        return
    obj._deleted = True


def objects_equal(a, b, _seen=None) -> bool:
    """isequal for two objects: same class and equal property values (handles need not be identical)."""
    from ..classdef import UserObject
    if a is b:
        return True
    if isinstance(a, UserObject) and isinstance(b, UserObject):
        if a.cls is not b.cls:
            return False
        if a._enum_name is not None or b._enum_name is not None:
            return a._enum_name == b._enum_name
        if a.is_handle:
            _seen = _seen if _seen is not None else set()
            key = (id(a), id(b))
            if key in _seen:
                return True  # already being compared further up (objects that refer to each other)
            _seen.add(key)
        if (a._data is None) != (b._data is None) or (a._data is not None and not values_equal(a._data, b._data)):
            return False
        if a.props.keys() != b.props.keys():
            return False
        for k in a.props:
            x, y = a.props[k], b.props[k]
            if isinstance(x, UserObject) and isinstance(y, UserObject):
                if not objects_equal(x, y, _seen):
                    return False
            elif not values_equal(x, y):
                return False
        return True
    return False
