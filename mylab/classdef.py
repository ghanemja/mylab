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
        for parent in self.parents:
            self.prop_defs.update(parent.prop_defs)
            for mname, entry in parent.methods.items():
                if mname != parent.name:  # constructors are not inherited
                    self.methods[mname] = entry
            self.static |= parent.static
        for p in cd.properties:
            self.prop_defs[p.name] = (p, self)
        for mname, fn in cd.methods.items():
            self.methods[mname] = (fn, self)
        self.static |= set(cd.static_methods)
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
        res = interp.call_user_function(ctor, args, 1, self.functions, self.path, preset={out_name: obj})
        if not res or not isinstance(res[0], UserObject):
            raise MatlabError(f"The constructor of '{self.name}' must return the object.")
        return res[0]

    # -- calls ----------------------------------------------------------------------------------
    def call(self, interp, name: str, args: list, nargout: int) -> list:
        fn, owner = self.methods[name]
        return interp.call_user_function(fn, args, nargout, owner.functions, owner.path)

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
        return name in self.cls.methods and name != self.cls.name and "." not in name

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name in self.cls.static and args and args[0] is self:
            args = args[1:]
        return self.cls.call(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
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
        names = self.cls.instance_props()
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
    if op in ("==", "~=") and isinstance(left, UserObject) and isinstance(right, UserObject) \
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
