"""Runtime support for user-defined classes (``classdef``).

Supported: properties with default values, Constant properties, methods,
constructors, Static methods, handle classes (``< handle``), single and
multiple inheritance from other classdef classes, superclass constructor and
method calls (``obj@Parent(...)``), get./set. property methods, operator
overloading (plus, minus, mtimes, eq, ...) and custom disp/display.

Also supported: object arrays (:class:`ObjectArray`), ``enumeration`` blocks,
``events`` with ``notify``/``addlistener``, and methods kept in separate files
of an ``@ClassName`` folder.

Not supported (yet): subsref/subsasgn overloading.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import ast as A
from .errors import MatlabError
from .objects import MObject

HANDLE_BASES = {"handle", "matlab.mixin.Copyable", "dynamicprops", "hgsetget", "matlab.mixin.SetGet",
                "matlab.mixin.Heterogeneous", "matlab.mixin.CustomDisplay", "event.EventData",
                "matlab.System", "matlab.unittest.TestCase", "matlab.apps.AppBase"}
# Built-in value classes an enumeration class may derive from (classdef Level < uint8).
NUMERIC_BASES = {"double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                 "uint64", "logical", "char"}
# Methods every handle object has; they run the built-in function of the same name.
HANDLE_METHODS = {"notify", "addlistener", "listener", "isvalid", "delete"}

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
        self.numeric_base: str | None = None
        self.full_name = cd.name  # 'pkg.Name' for a class in a +pkg folder
        for sup in cd.superclasses:
            if sup in HANDLE_BASES:
                self.is_handle = True
                continue
            if sup in NUMERIC_BASES:
                self.numeric_base = sup
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
            self.numeric_base = self.numeric_base or parent.numeric_base
        self.events: set[str] = set(cd.events)
        for parent in self.parents:
            self.events |= parent.events
        self.enums: dict[str, list] = dict(cd.enums)
        self._enum_objects: dict[str, UserObject] | None = None
        for p in cd.properties:
            self.prop_defs[p.name] = (p, self)
        for mname, fn in cd.methods.items():
            self.methods[mname] = (fn, self)
        self.static |= set(cd.static_methods)
        self.method_attrs.update(cd.method_attrs)
        self.abstract_methods = (abstract | set(cd.abstract_methods)) - {
            m for m, (_, owner) in self.methods.items() if owner is self}
        self._load_folder_methods()
        self.is_abstract = bool(cd.attrs.get("Abstract")) or bool(self.abstract_methods)
        self._defaults: dict[str, Any] | None = None
        self._constants: dict[str, Any] = {}

    def _load_folder_methods(self) -> None:
        """Methods stored as separate files next to the classdef in an ``@ClassName`` folder."""
        if not self.path:
            return
        folder = os.path.dirname(self.path)
        if os.path.basename(folder) != "@" + self.name:
            return
        from .parser import parse
        for entry in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(entry)
            if ext != ".m" or stem == self.name or stem in self.cd.methods:
                continue
            try:
                with open(os.path.join(folder, entry), encoding="utf-8", errors="replace") as fh:
                    program = parse(fh.read())
            except MatlabError as e:
                raise MatlabError(f"Error in method file {os.path.join(folder, entry)}: {e.message}") from None
            if not program.functions:
                continue
            main = next(iter(program.functions.values()))
            self.methods[stem] = (main, self)
            for fname, fn in program.functions.items():
                if fn is not main:
                    self.functions.setdefault(fname, fn)
            attrs = self.cd.declared_methods.get(stem, {})
            self.method_attrs[stem] = dict(attrs)
            if attrs.get("Static"):
                self.static.add(stem)
            self.abstract_methods.discard(stem)

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

    # -- enumerations ---------------------------------------------------------------------
    def enum_members(self, interp) -> dict[str, "UserObject"]:
        """The enumeration members, created once (each member is a single shared object)."""
        if self._enum_objects is None:
            from .interpreter import Frame
            self._enum_objects = {}
            has_ctor = self.cd.methods.get(self.name) is not None
            for member, arg_nodes in self.enums.items():
                frame = Frame(interp, self.functions, name=f"{self.name} enumeration", file=self.path)
                args = [interp.eval(a, frame) for a in arg_nodes]
                obj = UserObject(self)
                obj.props = dict(self.defaults(interp))
                obj._enum_name = member
                if self.numeric_base and args:
                    obj._enum_value = args[0]
                if has_ctor or not args:
                    obj = self.run_constructor(interp, obj, args if has_ctor else [])
                    obj._enum_name = member
                elif not self.numeric_base:
                    raise MatlabError(f"Enumeration member '{member}' of class '{self.name}' has arguments, "
                                      "but the class has no constructor.")
                self._enum_objects[member] = obj
        return self._enum_objects

    def enum_lookup(self, interp, value: Any) -> "UserObject":
        """Color('Red') or Level(2): the member with that name or underlying value."""
        from .values import MString, is_char, to_str
        members = self.enum_members(interp)
        if isinstance(value, UserObject) and value.cls is self:
            return value
        if isinstance(value, MString) or is_char(value):
            name = to_str(value)
            if name in members:
                return members[name]
            raise MatlabError(f"'{name}' is not a member of the enumeration class '{self.name}'.")
        if isinstance(value, np.ndarray) and value.size == 1:
            for obj in members.values():
                if obj._enum_value is not None and np.asarray(obj._enum_value).flat[0] == value.flat[0]:
                    return obj
        raise MatlabError(f"No member of the enumeration class '{self.name}' has that value.")

    # -- construction ---------------------------------------------------------------------------
    def construct(self, interp, args: list, nargout: int) -> list:
        if self.enums:
            if len(args) != 1:
                raise MatlabError(f"Use {self.name}.MemberName to get a member of the enumeration class "
                                  f"'{self.name}'.")
            return [self.enum_lookup(interp, args[0])]
        if self.is_abstract:
            missing = ", ".join(sorted(self.abstract_methods))
            detail = f" (abstract methods: {missing})" if missing else ""
            raise MatlabError(f"Cannot instantiate the abstract class '{self.name}'{detail}.")
        obj = UserObject(self)
        obj.props = dict(self.defaults(interp))
        made = self.run_constructor(interp, obj, args, nargout)
        return [] if made is None else [made]

    def run_constructor(self, interp, obj: "UserObject", args: list, nargout: int = 1) -> "UserObject | None":
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
        res = interp.call_user_function(ctor, args, min(nargout, 1), self.functions, self.path,
                                        preset={out_name: obj}, owner=self)
        if not res and nargout == 0:
            return None  # the constructor cleared its output (App Designer: if nargout == 0, clear app)
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
        if member in self.enums:
            return [self.enum_members(interp)[member]]
        if member == "empty":
            dims = [int(np.real(np.asarray(a).flat[0])) for a in args] or [0, 0]
            if len(dims) == 1:
                dims = [dims[0], dims[0]]
            if 0 not in dims:
                raise MatlabError("At least one dimension of an empty array must be zero.")
            return [ObjectArray(np.empty(tuple(dims), dtype=object), self)]
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

    _enum_name: str | None = None   # set for members of an enumeration class
    _enum_value: Any = None         # underlying value (classdef Level < uint8)
    _listeners: dict | None = None  # event name -> listeners (see builtins/events.py)

    def __init__(self, cls: ClassInfo):
        self.cls = cls
        self.props: dict[str, Any] = {}
        self._in_accessor: set[str] = set()

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return self.cls.is_handle

    def class_name(self) -> str:
        return self.cls.full_name

    def isa_class(self, name: str) -> bool:
        return name in self.cls.ancestors() or (name == "handle" and self.cls.is_handle) \
            or name == self.cls.full_name or (name == self.cls.numeric_base and name is not None)

    def copy(self) -> "UserObject":
        other = UserObject(self.cls)
        other.props = dict(self.props)
        other._enum_name = self._enum_name
        other._enum_value = self._enum_value
        return other

    # -- members ---------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        # delete(obj) goes through the built-in, which runs the class's delete and invalidates the handle.
        if name in self.cls.methods:
            return name != self.cls.name and "." not in name and (name != "delete" or self.cls.is_handle)
        if self._enum_name is not None and (name in _ENUM_METHODS or (
                self.cls.numeric_base and name in NUMERIC_BASES)):
            return True
        if name in self.cls.prop_defs:
            return False  # a property with the name of a built-in method wins
        if name == "step":
            return "stepImpl" in self.cls.methods  # System objects
        return name in _ARRAY_METHODS or (self.cls.is_handle and name in HANDLE_METHODS)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete" and self.cls.is_handle:  # obj.delete(): the built-in runs the class's delete
            return interp.call_builtin(name, interp.builtins[name], list(args), nargout)
        if name not in self.cls.methods:
            if self._enum_name is not None and (name in _ENUM_METHODS or name in NUMERIC_BASES):
                return [_enum_method(interp, name, args)]
            if name in _ARRAY_METHODS:
                return [_ARRAY_METHODS[name](interp, args)]
            if (self.cls.is_handle and name in HANDLE_METHODS) or name == "step":
                return interp.call_builtin(name, interp.builtins[name], list(args), nargout)
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
        return validate_value(interp, prop.attrs, value, name, self.cls.functions,
                              f"Error setting property '{name}' of class '{self.cls.name}'.")

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
        if self._listeners:
            from .builtins.events import property_event
            property_event(interp, self, name, "PreSet")
            target.props[name] = value
            property_event(interp, self, name, "PostSet")
            return target
        target.props[name] = value
        return target

    def index_get(self, interp, args: list) -> Any:
        if "subsref" in self.cls.methods:
            raise MatlabError("Overloading subsref is not supported.")
        return ObjectArray.of(self).index_get(interp, args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        return ObjectArray.of(self).index_set(interp, args, value)

    def index_delete(self, interp, args: list) -> Any:
        return ObjectArray.of(self).index_delete(interp, args)

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        return ObjectArray.concat(items, axis)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("==", "~=") and (self._enum_name is not None or not self.is_handle):
            other = right if left is self else left
            if isinstance(other, ObjectArray):
                return NotImplemented
            same = _same_object(self, other)
            if same is None:
                return NotImplemented
            return np.array([[same if op == "==" else not same]])
        return NotImplemented

    def summary_text(self, in_cell: bool) -> str:
        if self._enum_name is not None:
            return self._enum_name
        return f"1×1 {self.cls.name}" if in_cell else f"[1×1 {self.cls.name}]"

    def format_display(self, name: str) -> str:
        if self._enum_name is not None:
            return f"{name} = \n\n  {self.cls.name} enumeration\n\n    {self._enum_name}\n\n"
        return super().format_display(name)

    # -- display -------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        from .display import summary
        if self._enum_name is not None:
            return [f"    {self._enum_name}"]
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


_ENUM_METHODS = {"char", "string", "cellstr", "num2str", "strcmp", "strcmpi"}


def _same_object(a: Any, b: Any) -> bool | None:
    """Equality of two scalar objects for ==: enumeration members by name, handles by identity."""
    from .values import MString, is_char, to_str
    if isinstance(a, UserObject) and a._enum_name is not None:
        if isinstance(b, UserObject):
            return b.cls is a.cls and b._enum_name == a._enum_name
        if isinstance(b, MString) or is_char(b):
            return to_str(b) == a._enum_name
        if isinstance(b, np.ndarray) and a._enum_value is not None and b.size == 1:
            return bool(np.asarray(a._enum_value).flat[0] == b.flat[0])
        return None
    if isinstance(b, UserObject) and b._enum_name is not None:
        return _same_object(b, a)
    if isinstance(a, UserObject) and isinstance(b, UserObject):
        if a.is_handle or b.is_handle:
            return a is b
        from .builtins.objfuncs import objects_equal
        return objects_equal(a, b)
    return None


def _as_array(v: Any) -> "ObjectArray":
    if isinstance(v, ObjectArray):
        return v
    if isinstance(v, UserObject):
        return ObjectArray.of(v)
    raise MatlabError(f"Expected an object or an array of objects, not a {_class_name(v)}.")


def _array_ismember(interp, args: list) -> Any:
    from .values import Cell
    first, pool = args[0], args[1]
    if isinstance(pool, Cell):
        items = pool.values_f()
    elif isinstance(pool, (ObjectArray, UserObject)):
        items = _as_array(pool).elements()
    else:
        items = [pool]
    if isinstance(first, ObjectArray):
        flags = [any(bool(_same_object(x, y)) for y in items) for x in first.elements()]
        return np.array(flags, dtype=bool).reshape(first.data.shape, order="F")
    return np.array([[any(bool(_same_object(first, y)) for y in items)]])


def _array_repmat(interp, args: list) -> Any:
    arr = _as_array(args[0])
    reps = [int(np.real(x)) for a in args[1:] for x in np.asarray(a).flatten()]
    if len(reps) == 1:
        reps = [reps[0], reps[0]]
    reps = [max(r, 0) for r in reps]
    data = arr.data.reshape(arr.data.shape + (1,) * (len(reps) - arr.data.ndim))
    return ObjectArray._unwrap(np.tile(data, reps + [1] * (data.ndim - len(reps))), arr.cls)


def _array_reshape(interp, args: list) -> Any:
    arr = _as_array(args[0])
    dims: list = []
    for a in args[1:]:
        a = np.asarray(a)
        dims += [None] if a.size == 0 else [int(np.real(x)) for x in a.flatten()]
    if dims.count(None) == 1:
        known = int(np.prod([d for d in dims if d is not None]))
        dims[dims.index(None)] = arr.data.size // known if known else 0
    if None in dims or int(np.prod(dims)) != arr.data.size:
        raise MatlabError("reshape: the number of elements must not change.")
    return ObjectArray._unwrap(arr.data.reshape(dims, order="F"), arr.cls)


def _array_num2cell(interp, args: list) -> Any:
    from .values import Cell
    arr = _as_array(args[0])
    return Cell(arr.data.copy())


def _shape_test(test):
    return lambda interp, args: np.array([[bool(test(_as_array(args[0]).data.shape))]])


def _array_flip(axis: int):
    return lambda interp, args: ObjectArray._unwrap(np.flip(_as_array(args[0]).data, axis=axis).copy(),
                                                    _as_array(args[0]).cls)


def _array_isvalid(interp, args: list) -> Any:
    arr = _as_array(args[0])
    flags = [not getattr(e, "_deleted", False) for e in arr.elements()]
    return np.array(flags, dtype=bool).reshape(arr.data.shape, order="F")


_ARRAY_METHODS = {
    "ismember": _array_ismember, "repmat": _array_repmat, "reshape": _array_reshape,
    "num2cell": _array_num2cell, "fliplr": _array_flip(1), "flipud": _array_flip(0),
    "isrow": _shape_test(lambda s: len(s) == 2 and s[0] == 1),
    "iscolumn": _shape_test(lambda s: len(s) == 2 and s[1] == 1),
    "isvector": _shape_test(lambda s: len(s) == 2 and 1 in s and max(s) >= 1),
    "isscalar": _shape_test(lambda s: all(d == 1 for d in s)),
    "isvalid": _array_isvalid,
}


def _enum_method(interp, name: str, args: list) -> Any:
    from .values import Cell, MString, char, is_char, to_str
    obj = next(a for a in args if isinstance(a, UserObject) and a._enum_name is not None)
    if name in ("char", "num2str"):
        return char(obj._enum_name)
    if name == "string":
        return MString(obj._enum_name)
    if name == "cellstr":
        return Cell.from_list([char(obj._enum_name)])
    if name in ("strcmp", "strcmpi"):
        texts = []
        for a in args[:2]:
            if isinstance(a, UserObject) and a._enum_name is not None:
                texts.append(a._enum_name)
            elif isinstance(a, MString) or is_char(a):
                texts.append(to_str(a))
            else:
                return np.array([[False]])
        if name == "strcmpi":
            texts = [t.lower() for t in texts]
        return np.array([[texts[0] == texts[1]]])
    if obj._enum_value is None:
        raise MatlabError(f"Members of '{obj.cls.name}' have no underlying numeric value.")
    return interp.call_builtin(name, interp.builtins[name], [obj._enum_value], 1)[0]


class ObjectArray(MObject):
    """An array of classdef objects, e.g. ``objs(3) = Point(1, 2)`` or ``[a, b, c]``."""

    def __init__(self, data: np.ndarray, cls: ClassInfo | None = None):
        self.data = data
        self.cls = cls

    @classmethod
    def of(cls, obj: UserObject) -> "ObjectArray":
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = obj
        return cls(data, obj.cls)

    @staticmethod
    def _unwrap(data: np.ndarray, cls: ClassInfo | None) -> Any:
        if data.size == 1:
            return data.flat[0]
        return ObjectArray(data, cls)

    def elements(self) -> list:
        return list(self.data.flatten(order="F"))

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return bool(self.cls and self.cls.is_handle)

    def class_name(self) -> str:
        return self.cls.full_name if self.cls else "double"

    def isa_class(self, name: str) -> bool:
        if self.data.size:
            return all(e.isa_class(name) for e in self.data.flat)
        return bool(self.cls) and (name in self.cls.ancestors() or name == self.cls.full_name
                                   or (name == "handle" and self.cls.is_handle))

    def matlab_shape(self) -> tuple:
        return tuple(self.data.shape)

    def empty_like(self) -> "ObjectArray":
        return ObjectArray(np.empty((0, 0), dtype=object), self.cls)

    # -- members: methods receive the whole array; properties are read element by element --------
    def has_method(self, name: str) -> bool:
        if self.cls is None:
            return False
        if name in self.cls.methods:
            return name != self.cls.name and "." not in name and name != "delete"
        return name in _ARRAY_METHODS and name not in self.cls.prop_defs

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name not in self.cls.methods and name in _ARRAY_METHODS:
            return [_ARRAY_METHODS[name](interp, args)]
        if name in self.cls.static and args and args[0] is self:
            args = args[1:]
        return self.cls.call(interp, name, args, nargout)

    def field_list(self, interp, name: str) -> list:
        """``objs.name`` as a comma-separated list: one value per element."""
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)
        return [e.get_field(interp, name) for e in self.elements()]

    def get_field(self, interp, name: str) -> Any:
        vals = self.field_list(interp, name)
        if len(vals) != 1:
            raise MatlabError(f"Reading '{name}' from an array of {self.data.size} objects gives "
                              f"{len(vals)} values. Index one object first, e.g. objs(1).{name}, or collect "
                              f"them with [objs.{name}].")
        return vals[0]

    def set_field(self, interp, name: str, value: Any) -> Any:
        if self.data.size != 1:
            raise MatlabError(f"Cannot set '{name}' on an array of objects; index one object first, "
                              f"e.g. objs(1).{name} = value.")
        return self.data.flat[0].set_field(interp, name, value)

    # -- indexing ---------------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        from . import indexing as ix
        return self._unwrap(ix.get(self.data, list(args)), self.cls)

    def _fill(self, interp) -> Any:
        """The object used for elements an assignment creates but does not name."""
        if self.cls is None:
            raise MatlabError("Cannot grow an array of objects of unknown class.")
        if self.cls.enums:
            return next(iter(self.cls.enum_members(interp).values()))
        try:
            return self.cls.construct(interp, [], 1)[0]
        except MatlabError as e:
            raise MatlabError(f"Growing the array needs a default {self.cls.name} object, but calling "
                              f"{self.cls.name}() with no arguments failed: {e.message}") from None

    def index_set(self, interp, args: list, value: Any) -> Any:
        from . import indexing as ix
        if isinstance(value, UserObject):
            rhs = ObjectArray.of(value).data
        elif isinstance(value, ObjectArray):
            rhs = value.data
        else:
            raise MatlabError(f"Cannot store a {_class_name(value)} in an array of "
                              f"{self.class_name()} objects.")
        base = self if self.cls is not None else ObjectArray(self.data, value.cls)
        return self._unwrap(ix.set_(base.data.copy(), list(args), rhs, lambda: base._fill(interp)), base.cls)

    def index_delete(self, interp, args: list) -> Any:
        from . import indexing as ix
        return ObjectArray(ix.delete(self.data, list(args)), self.cls)

    def brace_get(self, interp, args: list) -> list:
        raise MatlabError("Brace indexing is not supported for object arrays; use parentheses: objs(k).")

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        datas = []
        klass = None
        for i in items:
            if isinstance(i, UserObject):
                datas.append(ObjectArray.of(i).data)
                klass = klass or i.cls
            elif isinstance(i, ObjectArray):
                klass = klass or i.cls
                if i.data.size:
                    datas.append(i.data)
            elif isinstance(i, np.ndarray) and i.size == 0:
                continue
            else:
                raise MatlabError(f"Cannot concatenate a {_class_name(i)} with objects of class "
                                  f"'{klass.name if klass else 'object'}'. Use a cell array {{...}} to mix "
                                  "types.")
        if not datas:
            return ObjectArray(np.empty((0, 0), dtype=object), klass)
        nd = max([axis + 1] + [d.ndim for d in datas])
        datas = [d.reshape(d.shape + (1,) * (nd - d.ndim)) for d in datas]
        try:
            data = np.concatenate(datas, axis=axis)
        except ValueError:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch") from None
        return cls._unwrap(data, klass)

    def object_equals(self, other: Any) -> bool:
        from .builtins.objfuncs import objects_equal
        if isinstance(other, UserObject):
            other = ObjectArray.of(other)
        if not isinstance(other, ObjectArray) or other.data.shape != self.data.shape:
            return False
        return all(objects_equal(a, b) for a, b in zip(self.data.flat, other.data.flat))

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("==", "~="):
            return NotImplemented
        other = right if left is self else left
        if isinstance(other, ObjectArray):
            if other.data.shape != self.data.shape:
                raise MatlabError("Arrays have incompatible sizes for this operation.")
            pairs = zip(self.data.flatten(order="F"), other.data.flatten(order="F"))
            flags = [bool(_same_object(a, b)) for a, b in pairs]
        else:
            flags = [bool(_same_object(a, other)) for a in self.data.flatten(order="F")]
        out = np.array(flags, dtype=bool).reshape(self.data.shape, order="F")
        return out if op == "==" else ~out

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return ObjectArray(self.data.T.copy(), self.cls)
        return NotImplemented

    # -- display -----------------------------------------------------------------------------
    def _size_text(self) -> str:
        return "×".join(str(d) for d in self.data.shape)

    def summary_text(self, in_cell: bool) -> str:
        s = f"{self._size_text()} {self.class_name()}"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        name = self.class_name()
        if self.cls is not None and self.cls.enums and self.data.size:
            head = [f"  {self._size_text()} {name} enumeration array", ""]
            names = np.array([e._enum_name for e in self.data.flat], dtype=object).reshape(self.data.shape)
            if names.ndim > 2:
                return head[:1]
            w = max(len(n) for n in names.flat)
            return head + ["    " + "    ".join(n.ljust(w) for n in row).rstrip() for row in names]
        kind = "empty " if self.data.size == 0 else ""
        props = self.cls.instance_props() if self.cls else []
        if not props:
            return [f"  {self._size_text()} {kind}{name} array with no properties."]
        return [f"  {self._size_text()} {kind}{name} array with properties:", ""] + [f"    {p}" for p in props]


def _class_name(v: Any) -> str:
    from .values import class_name
    return class_name(v)


def call_operator(interp, op: str, left: Any, right: Any) -> Any:
    """Dispatch a binary operator to a class method, or return NotImplemented."""
    name = OPERATOR_METHODS.get(op)
    for obj in (left, right):
        if isinstance(obj, (UserObject, ObjectArray)) and obj.cls is not None and name \
                and name in obj.cls.methods:
            res = obj.cls.call(interp, name, [left, right], 1)
            if not res:
                raise MatlabError(f"Operator method '{name}' did not return a value.")
            return res[0]
    for obj in (left, right):
        if isinstance(obj, MObject):
            res = obj.binary_op(interp, op, left, right)
            if res is not NotImplemented:
                return res
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
    if isinstance(operand, (UserObject, ObjectArray)) and operand.cls is not None \
            and name in operand.cls.methods:
        res = operand.cls.call(interp, name, [operand], 1)
        if not res:
            raise MatlabError(f"Operator method '{name}' did not return a value.")
        return res[0]
    if isinstance(operand, MObject):
        res = operand.unary_op(interp, op)
        if res is not NotImplemented:
            return res
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


def validate_value(interp, attrs: dict, value: Any, name: str, functions: dict, context: str) -> Any:
    """Check (and convert) a value against declared class, size and validator functions.

    Shared by classdef property declarations and function ``arguments`` blocks.
    """
    cls = attrs.get("_class")
    if cls:
        try:
            value = _convert_class(value, cls, name)
        except MatlabError as e:
            raise MatlabError(f"{context} {e.message}", e.identifier) from None
    size = attrs.get("_size")
    if size:
        from .values import shape_of
        shape = list(shape_of(value))
        ok = len(size) == len(shape) and all(d is None or d == s for d, s in zip(size, shape))
        if not ok:
            want = "x".join(":" if d is None else str(d) for d in size)
            got = "x".join(str(d) for d in shape)
            raise MatlabError(f"{context} Value must be {want} (it is {got}).")
    for v in attrs.get("_validators", []):
        from .interpreter import Frame
        frame = Frame(interp, functions, name=f"{name} validator")
        frame.vars[name.split(".")[-1]] = value
        if "." in name:
            frame.vars[name.split(".")[0]] = value
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
            raise MatlabError(f"{context} {e.message}", e.identifier) from None
    return value
