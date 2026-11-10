"""Runtime support for user-defined classes (``classdef``).

Supported: properties with default values, Constant properties, methods,
constructors, Static methods, handle classes (``< handle``), single and
multiple inheritance from other classdef classes, superclass constructor and
method calls (``obj@Parent(...)``), get./set. property methods, operator
overloading (plus, minus, mtimes, eq, ...) and custom disp/display.

Not supported (yet): object arrays, events/listeners, enumerations,
property validation, access restrictions (all members behave as public).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import ast as A
from .errors import MatlabError
from .objects import MObject

HANDLE_BASES = {"handle", "matlab.mixin.Copyable", "dynamicprops", "hgsetget", "matlab.mixin.SetGet",
                "matlab.mixin.Heterogeneous", "matlab.mixin.CustomDisplay"}

OPERATOR_METHODS = {
    "+": "plus", "-": "minus", ".*": "times", "*": "mtimes", "./": "rdivide", "/": "mrdivide",
    ".\\": "ldivide", "\\": "mldivide", ".^": "power", "^": "mpower", "==": "eq", "~=": "ne",
    "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "&": "and", "|": "or",
}
UNARY_METHODS = {"-": "uminus", "+": "uplus", "~": "not", "'": "ctranspose", ".'": "transpose"}


class ClassInfo:
    def __init__(self, cd: A.ClassDef, path: str | None, interp, functions: dict | None = None):
        self.cd = cd
        self.name = cd.name
        self.path = path
        self.functions = dict(functions or {})  # local functions in the classdef file
        self.parents: list[ClassInfo] = []
        self.is_handle = False
        for sup in cd.superclasses:
            if sup in HANDLE_BASES:
                self.is_handle = True
                continue
            parent = interp.find_class(sup)
            if parent is None:
                raise MatlabError(f"Superclass '{sup}' of class '{self.name}' was not found.")
            self.parents.append(parent)
            self.is_handle = self.is_handle or parent.is_handle
        # Properties and methods, parents first so the child overrides.
        self.prop_defs: dict[str, tuple[A.PropertyDef, ClassInfo]] = {}
        self.methods: dict[str, tuple[A.Function, ClassInfo]] = {}
        self.static: set[str] = set()
        self.method_attrs: dict[str, dict] = {}
        abstract: set[str] = set()
        for parent in self.parents:
            self.prop_defs.update(parent.prop_defs)
            for mname, entry in parent.methods.items():
                if mname != parent.name:  # constructors are not inherited
                    self.methods[mname] = entry
            self.static |= parent.static
            self.method_attrs.update(parent.method_attrs)
            abstract |= parent.abstract_methods
        for p in cd.properties:
            self.prop_defs[p.name] = (p, self)
        for mname, fn in cd.methods.items():
            self.methods[mname] = (fn, self)
        self.static |= set(cd.static_methods)
        self.method_attrs.update(cd.method_attrs)
        self.abstract_methods = (abstract | set(cd.abstract_methods)) - {
            m for m, (_, owner) in self.methods.items() if owner is self}
        self.is_abstract = bool(cd.attrs.get("Abstract")) or bool(self.abstract_methods)
        self._defaults: dict[str, Any] | None = None
        self._constants: dict[str, Any] = {}

    # -- class hierarchy -------------------------------------------------------------------
    def ancestors(self) -> list[str]:
        names = [self.name]
        for p in self.parents:
            names += p.ancestors()
        return names

    def instance_props(self) -> list[str]:
        return [n for n, (p, _) in self.prop_defs.items() if not p.attrs.get("Constant")]

    def is_constant(self, name: str) -> bool:
        entry = self.prop_defs.get(name)
        return bool(entry and entry[0].attrs.get("Constant"))

    # -- defaults and constants ------------------------------------------------------------
    def _eval_default(self, interp, prop: A.PropertyDef, owner: "ClassInfo") -> Any:
        if prop.default is None:
            return np.zeros((0, 0))
        from .interpreter import Frame
        frame = Frame(interp, owner.functions, name=f"{owner.name} property default", file=owner.path)
        return interp.eval(prop.default, frame)

    def defaults(self, interp) -> dict[str, Any]:
        # Like MATLAB, default values are evaluated once, when the class is first used.
        if self._defaults is None:
            self._defaults = {}
            for name in self.instance_props():
                prop, owner = self.prop_defs[name]
                self._defaults[name] = self._eval_default(interp, prop, owner)
        return self._defaults

    def constant(self, interp, name: str) -> Any:
        if name not in self._constants:
            prop, owner = self.prop_defs[name]
            self._constants[name] = self._eval_default(interp, prop, owner)
        return self._constants[name]

    # -- construction ---------------------------------------------------------------------------
    def construct(self, interp, args: list, nargout: int) -> list:
        if self.is_abstract:
            missing = ", ".join(sorted(self.abstract_methods))
            detail = f" (abstract methods: {missing})" if missing else ""
            raise MatlabError(f"Cannot instantiate the abstract class '{self.name}'{detail}.")
        obj = UserObject(self)
        obj.props = dict(self.defaults(interp))
        return [self.run_constructor(interp, obj, args)]

    def run_constructor(self, interp, obj: "UserObject", args: list) -> "UserObject":
        ctor = self.cd.methods.get(self.name)
        if ctor is None:
            if args:
                raise MatlabError(f"Too many input arguments for the default constructor of '{self.name}'.")
            for parent in self.parents:
                obj = parent.run_constructor(interp, obj, [])
            return obj
        # Parents whose constructor is not called explicitly get a no-argument call first.
        called = _super_calls(ctor)
        for parent in self.parents:
            if parent.name not in called and parent.cd.methods.get(parent.name) is not None:
                obj = parent.run_constructor(interp, obj, [])
        out_name = ctor.outputs[0] if ctor.outputs else "obj"
        res = interp.call_user_function(ctor, args, 1, self.functions, self.path, preset={out_name: obj},
                                        owner=self)
        if not res or not isinstance(res[0], UserObject):
            raise MatlabError(f"The constructor of '{self.name}' must return the object.")
        return res[0]

    # -- calls ----------------------------------------------------------------------------------
    def call(self, interp, name: str, args: list, nargout: int) -> list:
        fn, owner = self.methods[name]
        return interp.call_user_function(fn, args, nargout, owner.functions, owner.path, owner=owner)

    # -- access control --------------------------------------------------------------------
    @staticmethod
    def _caller(interp) -> "ClassInfo | None":
        return getattr(interp.frame, "owner_class", None)

    def allowed(self, interp, level: Any, owner: "ClassInfo") -> bool:
        """Whether the running code may use a member with access ``level`` defined in ``owner``."""
        level = str(level).lower() if level is not None else "public"
        if level == "public" or level == "true":
            return True
        caller = self._caller(interp)
        if caller is None:
            return False
        if level in ("private", "immutable"):
            return caller is owner or caller.name == owner.name
        return owner.name in caller.ancestors()  # protected

    def static_access(self, interp, member: str, args: list, nargout: int) -> list:
        if member in self.methods and member in self.static:
            return self.call(interp, member, args, nargout)
        if self.is_constant(member):
            value = self.constant(interp, member)
            if args:
                from .interpreter import index_value
                return [index_value(value, args)]
            return [value]
        if member in self.methods:
            raise MatlabError(f"Method '{member}' of class '{self.name}' is not Static; call it on an object.")
        raise MatlabError(f"Unrecognized method, property, or field '{member}' for class '{self.name}'.")


def _super_calls(fn: A.Function) -> set[str]:
    found: set[str] = set()

    def visit(n: Any) -> None:
        if isinstance(n, A.SuperCall):
            found.add(n.parent)
        if isinstance(n, A.Node):
            for v in vars(n).values():
                visit(v)
        elif isinstance(n, (list, tuple)):
            for item in n:
                visit(item)

    visit(fn.body)
    return found


class UserObject(MObject):
    """An instance of a classdef class."""

    def __init__(self, cls: ClassInfo):
        self.cls = cls
        self.props: dict[str, Any] = {}
        self._in_accessor: set[str] = set()

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return self.cls.is_handle

    def class_name(self) -> str:
        return self.cls.name

    def isa_class(self, name: str) -> bool:
        return name in self.cls.ancestors() or (name == "handle" and self.cls.is_handle)

    def copy(self) -> "UserObject":
        other = UserObject(self.cls)
        other.props = dict(self.props)
        return other

    # -- members ---------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        # delete(obj) goes through the built-in, which runs the class's delete and invalidates the handle.
        return name in self.cls.methods and name != self.cls.name and "." not in name and name != "delete"

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        self._check_valid()
        access = self.cls.method_attrs.get(name, {}).get("Access")
        if access is not None and not self.cls.allowed(interp, access, self.cls.methods[name][1]):
            raise MatlabError(f"Cannot access method '{name}' in class '{self.cls.name}'.")
        if name in self.cls.static and args and args[0] is self:
            args = args[1:]
        return self.cls.call(interp, name, args, nargout)

    def _check_valid(self) -> None:
        if getattr(self, "_deleted", False):
            raise MatlabError("Invalid or deleted object.")

    def _check_access(self, interp, name: str, kind: str) -> None:
        entry = self.cls.prop_defs.get(name)
        if entry is None:
            return
        prop, owner = entry
        level = prop.attrs.get(kind, prop.attrs.get("Access"))
        if level is not None and not self.cls.allowed(interp, level, owner):
            verb = "set" if kind == "SetAccess" else "access"
            raise MatlabError(f"You cannot {verb} the '{name}' property of {self.cls.name}.")

    def validate(self, interp, name: str, value: Any) -> Any:
        """Apply a property's class, size and validator functions (as declared in the classdef)."""
        prop = self.cls.prop_defs[name][0]
        cls = prop.attrs.get("_class")
        if cls:
            value = _convert_class(value, cls, name)
        size = prop.attrs.get("_size")
        if size:
            from .values import shape_of
            shape = list(shape_of(value))
            ok = len(size) == len(shape) and all(d is None or d == s for d, s in zip(size, shape))
            if not ok:
                want = "x".join(":" if d is None else str(d) for d in size)
                got = "x".join(str(d) for d in shape)
                raise MatlabError(f"Error setting property '{name}' of class '{self.cls.name}'. "
                                  f"Value must be {want} (it is {got}).")
        for v in prop.attrs.get("_validators", []):
            from .interpreter import Frame
            frame = Frame(interp, self.cls.functions, name=f"{self.cls.name}.{name} validator")
            frame.vars[name] = value
            try:
                if isinstance(v, A.Ident) and not frame.has(v.name):
                    interp.call_named(v.name, [value], 0, frame)
                else:
                    interp.frames.append(frame)
                    try:
                        interp.eval_multi(v, frame, 0)
                    finally:
                        interp.frames.pop()
            except MatlabError as e:
                raise MatlabError(f"Error setting property '{name}' of class '{self.cls.name}'. {e.message}",
                                  e.identifier)
        return value

    def get_field(self, interp, name: str) -> Any:
        self._check_valid()
        self._check_access(interp, name, "GetAccess")
        getter = "get." + name
        if getter in self.cls.methods and name not in self._in_accessor:
            self._in_accessor.add(name)
            try:
                res = self.cls.call(interp, getter, [self], 1)
            finally:
                self._in_accessor.discard(name)
            if not res:
                raise MatlabError(f"Get method for '{name}' did not return a value.")
            return res[0]
        if name in self.props:
            return self.props[name]
        if self.cls.is_constant(name):
            return self.cls.constant(interp, name)
        if self.has_method(name):
            res = self.call_method(interp, name, [self], 1)
            if not res:
                raise MatlabError(f"Method '{name}' does not return a value.")
            return res[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.cls.name}'.")

    def set_field(self, interp, name: str, value: Any) -> "UserObject":
        if self.cls.is_constant(name):
            raise MatlabError(f"Cannot set the Constant property '{name}' of class '{self.cls.name}'.")
        if name not in self.cls.prop_defs:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.cls.name}'.")
        self._check_valid()
        self._check_access(interp, name, "SetAccess")
        value = self.validate(interp, name, value)
        setter = "set." + name
        if setter in self.cls.methods and name not in self._in_accessor:
            target = self if self.is_handle else self.copy()
            target._in_accessor.add(name)
            try:
                res = self.cls.call(interp, setter, [target, value], 1)
            finally:
                target._in_accessor.discard(name)
            if self.is_handle:
                return self
            if not res or not isinstance(res[0], UserObject):
                raise MatlabError(f"Set method for '{name}' must return the object.")
            return res[0]
        target = self if self.is_handle else self.copy()
        target.props[name] = value
        return target

    def index_get(self, interp, args: list) -> Any:
        if "subsref" in self.cls.methods:
            raise MatlabError("Overloading subsref is not supported.")
        return super().index_get(interp, args)

    # -- display -------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        from .display import summary
        names = [n for n in self.cls.instance_props()
                 if str(self.cls.prop_defs[n][0].attrs.get("GetAccess",
                        self.cls.prop_defs[n][0].attrs.get("Access", "public"))).lower() == "public"]
        if not names:
            return [f"  {self.cls.name} with no properties."]
        from .interpreter import current_interpreter
        interp = current_interpreter()
        w = max(len(n) for n in names)

        def value(n: str) -> Any:
            if interp is not None and "get." + n in self.cls.methods:
                try:
                    return self.get_field(interp, n)
                except MatlabError:
                    pass
            return self.props.get(n, np.zeros((0, 0)))

        return [f"  {self.cls.name} with properties:", ""] + \
            [f"    {n:>{w}}: {summary(value(n), False)}" for n in names]


def call_operator(interp, op: str, left: Any, right: Any) -> Any:
    """Dispatch a binary operator to a class method, or return NotImplemented."""
    name = OPERATOR_METHODS.get(op)
    for obj in (left, right):
        if isinstance(obj, UserObject) and name and name in obj.cls.methods:
            res = obj.cls.call(interp, name, [left, right], 1)
            if not res:
                raise MatlabError(f"Operator method '{name}' did not return a value.")
            return res[0]
    if op in ("==", "~=") and isinstance(left, MObject) and isinstance(right, MObject) \
            and left.is_handle and right.is_handle:
        same = left is right
        return np.array([[same if op == "==" else not same]])
    if isinstance(left, MObject) or isinstance(right, MObject):
        cls = (left if isinstance(left, MObject) else right).class_name()
        raise MatlabError(f"Operator '{op}' is not supported for operands of type '{cls}'.")
    return NotImplemented


def call_unary_operator(interp, op: str, operand: Any) -> Any:
    name = UNARY_METHODS.get(op)
    if isinstance(operand, UserObject) and name in operand.cls.methods:
        res = operand.cls.call(interp, name, [operand], 1)
        if not res:
            raise MatlabError(f"Operator method '{name}' did not return a value.")
        return res[0]
    if isinstance(operand, MObject):
        raise MatlabError(f"Operator '{op}' is not supported for operands of type '{operand.class_name()}'.")
    return NotImplemented


def _convert_class(value: Any, cls: str, name: str) -> Any:
    from .values import (
        INT_TYPES, Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, saturate_int,
        string_data, string_result, to_numeric,
    )
    numeric_ok = isinstance(value, np.ndarray)
    if cls in ("double", "single") or cls in INT_TYPES or cls == "logical":
        if not numeric_ok:
            raise MatlabError(f"Value of property '{name}' must be convertible to {cls}.")
        a = to_numeric(value)
        if cls == "double":
            return a.astype(complex if a.dtype.kind == "c" else float)
        if cls == "single":
            return a.astype(np.float32)
        if cls == "logical":
            if a.dtype.kind == "f" and np.isnan(a).any():
                raise MatlabError("NaN's cannot be converted to logicals.")
            return a != 0
        return saturate_int(a, INT_TYPES[cls])
    if cls == "string":
        if isinstance(value, (MString, StringArray)):
            return value
        try:
            return string_result(string_data(value))
        except MatlabError:
            raise MatlabError(f"Value of property '{name}' must be convertible to string.") from None
    if cls == "char":
        if is_char(value):
            return value
        if isinstance(value, MString):
            return char(str(value))
        raise MatlabError(f"Value of property '{name}' must be text.")
    checks = {"cell": Cell, "struct": Struct, "function_handle": FunctionHandle}
    if cls in checks:
        if not isinstance(value, checks[cls]):
            raise MatlabError(f"Value of property '{name}' must be a {cls}.")
        return value
    from .values import class_name
    if hasattr(value, "isa_class"):
        if value.isa_class(cls):
            return value
    elif class_name(value) == cls:
        return value
    if isinstance(value, np.ndarray) and value.size == 0:
        return value
    raise MatlabError(f"Value of property '{name}' must be of class '{cls}'.")
