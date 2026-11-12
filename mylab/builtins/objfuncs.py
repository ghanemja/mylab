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
_check("mustBeInRange", lambda a, lo, hi, *_: bool(np.all((np.real(a) >= num(lo)) & (np.real(a) <= num(hi)))),
       "Value must be within the allowed range.")
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
    from ..classdef import ObjectArray, UserObject
    if isinstance(v, (UserObject, ObjectArray)):
        return v.cls
    return None


def _member_lists(interp, v):
    cls = _class_of(v)
    if cls is None and is_text(v):
        cls = interp.find_class(text(v))
    if cls is not None:
        props = [n for n, (p, _) in cls.prop_defs.items()
                 if str(p.attrs.get("GetAccess", p.attrs.get("Access", "public"))).lower() == "public"]
        methods = sorted(n for n in cls.methods if "." not in n
                         and str(cls.method_attrs.get(n, {}).get("Access", "public")).lower() == "public")
        return cls.name, props, methods
    if isinstance(v, MObject):
        methods = sorted(getattr(v, "_METHODS", []))
        props = ["Count", "KeyType", "ValueType"] if v.class_name() == "containers.Map" else []
        return v.class_name(), props, methods
    raise MatlabError("Expected an object or a class name.")


@builtin("properties")
def _properties(interp, args, nargout):
    name, props, _ = _member_lists(interp, args[0])
    if nargout:
        return Cell.from_list([char(p) for p in props], column=True)
    interp.out(f"\nProperties for class {name}:\n\n" + "".join(f"    {p}\n" for p in props) + "\n")
    return None


@builtin("methods")
def _methods(interp, args, nargout):
    name, _, methods = _member_lists(interp, args[0])
    if nargout:
        return Cell.from_list([char(p) for p in methods], column=True)
    interp.out(f"\nMethods for class {name}:\n\n" + "  ".join(methods) + "\n\n")
    return None


@builtin("isprop")
def _isprop(interp, args, nargout):
    try:
        _, props, _ = _member_lists(interp, args[0])
    except MatlabError:
        return bool_scalar(False)
    return bool_scalar(text(args[1]) in props)


@builtin("ismethod")
def _ismethod(interp, args, nargout):
    try:
        _, _, methods = _member_lists(interp, args[0])
    except MatlabError:
        return bool_scalar(False)
    return bool_scalar(text(args[1]) in methods)


def delete_object(interp, obj) -> None:
    """delete(h) for handle objects: run the class's delete method, then invalidate it."""
    from ..classdef import ObjectArray, UserObject
    if isinstance(obj, ObjectArray):
        for element in obj.elements():
            delete_object(interp, element)
        return
    if isinstance(obj, UserObject) and "delete" in obj.cls.methods and not getattr(obj, "_deleted", False):
        obj.cls.call(interp, "delete", [obj], 0)
    obj._deleted = True


def objects_equal(a, b) -> bool:
    from ..classdef import UserObject
    if a is b:
        return True
    if isinstance(a, UserObject) and isinstance(b, UserObject):
        if a.cls is not b.cls or a.is_handle:
            return False
        return a.props.keys() == b.props.keys() and all(values_equal(a.props[k], b.props[k]) for k in a.props)
    return False
