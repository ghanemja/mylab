"""Runtime support for user-defined classes (``classdef``).

Supported: properties with default values, Constant properties, methods,
constructors, Static methods, handle classes (``< handle``), single and
multiple inheritance from other classdef classes, superclass constructor and
method calls (``obj@Parent(...)``), get./set. property methods, operator
overloading (plus, minus, mtimes, eq, ...) and custom disp/display.

Also supported: object arrays (:class:`ObjectArray`), ``enumeration`` blocks,
``events`` with ``notify``/``addlistener``, methods kept in separate files
of an ``@ClassName`` folder, indexing overloads (``subsref``, ``subsasgn``,
``subsindex``, ``end``, ``numArgumentsFromSubscript``), classes derived from
built-in types (``classdef Meters < double``), the built-in mixin base classes
(see :data:`BUILTIN_BASES`) and old-style classes whose constructor calls
``class(s, 'name')`` in an ``@name`` folder.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from . import ast as A
from .errors import MatlabError
from .objects import MObject
from .sourcefile import read_source


class BaseSpec:
    """A base class the engine provides itself (``handle``, ``matlab.mixin.Copyable``, ...)."""

    def __init__(self, handle: bool, parents: tuple = ()):
        self.handle = handle
        self.parents = parents
        self.methods: dict[str, Callable] = {}  # name -> f(interp, args, nargout), like a built-in


BUILTIN_BASES: dict[str, BaseSpec] = {
    "handle": BaseSpec(True),
    "matlab.mixin.Copyable": BaseSpec(True, ("handle",)),
    "dynamicprops": BaseSpec(True, ("handle",)),
    "hgsetget": BaseSpec(True, ("handle",)),
    "matlab.mixin.SetGet": BaseSpec(True, ("handle",)),
    "matlab.mixin.SetGetExactNames": BaseSpec(True, ("matlab.mixin.SetGet", "handle")),
    "matlab.mixin.Heterogeneous": BaseSpec(False),
    "matlab.mixin.CustomDisplay": BaseSpec(False),
    "matlab.mixin.CustomCompactDisplayProvider": BaseSpec(False),
    "matlab.mixin.Scalar": BaseSpec(False),
    "matlab.mixin.indexing.RedefinesParen": BaseSpec(False),
    "matlab.mixin.indexing.RedefinesDot": BaseSpec(False),
    "matlab.mixin.indexing.RedefinesBrace": BaseSpec(False),
    "event.EventData": BaseSpec(True, ("handle",)),
    "matlab.System": BaseSpec(True, ("handle",)),
    "matlab.unittest.TestCase": BaseSpec(True, ("handle",)),
    "matlab.unittest.fixtures.Fixture": BaseSpec(True, ("handle",)),
    "matlab.apps.AppBase": BaseSpec(True, ("handle",)),
}
HANDLE_BASES = {n for n, b in BUILTIN_BASES.items() if b.handle}


def base_method(base: str, *names: str):
    """Register a Python function as a method of a built-in base class (used by built-in modules)."""
    def deco(fn: Callable) -> Callable:
        for n in names:
            BUILTIN_BASES[base].methods[n] = fn
        return fn
    return deco


# Built-in value classes a class may derive from (classdef Meters < double, classdef Level < uint8).
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
# Class methods that take over indexing; a class defining one makes the interpreter look for them.
_INDEX_METHODS = ("subsref", "subsasgn", "subsindex", "numArgumentsFromSubscript", "end",
                  "parenReference", "parenAssign", "dotReference", "dotAssign", "braceReference",
                  "braceAssign")


def access_names(level: Any) -> list[str] | None:
    """Class names of an access list such as ``?A`` or ``{?A, ?B}``; None for a plain level."""
    if not isinstance(level, str) or "?" not in level:
        return None
    return [part.strip().lstrip("?") for part in level.strip("{}[] ").split(",") if part.strip()]


def class_list(value: Any) -> list[str]:
    """Names in a class-list attribute: ``?A``, ``{?A, ?B}`` or ``{'A', 'B'}``."""
    if not isinstance(value, str):
        return []
    return [part.strip().lstrip("?").strip("'\"") for part in value.strip("{}[] ").split(",") if part.strip()]


class ClassInfo:
    old_style = False  # True for a class made by class(s, 'name') in an @name folder

    def __init__(self, cd: A.ClassDef, path: str | None, interp, functions: dict | None = None):
        self.cd = cd
        self.name = cd.name
        self.path = path
        self.functions = dict(functions or {})  # local functions in the classdef file
        self.parents: list[ClassInfo] = []
        self.is_handle = False
        self.numeric_base: str | None = None
        self.full_name = cd.name  # 'pkg.Name' for a class in a +pkg folder
        self.bases: list[str] = []  # built-in base classes, direct and inherited (BUILTIN_BASES)
        for sup in cd.superclasses:
            spec = BUILTIN_BASES.get(sup)
            if spec is not None:
                for b in (sup, *spec.parents):
                    if b not in self.bases:
                        self.bases.append(b)
                self.is_handle = self.is_handle or spec.handle
                continue
            if sup in NUMERIC_BASES:
                self.numeric_base = sup
                continue
            parent = interp.find_class(sup)
            if parent is None:
                raise MatlabError(f"Superclass '{sup}' of class '{self.name}' was not found.")
            if parent.cd.attrs.get("Sealed"):
                raise MatlabError(f"Class '{parent.name}' is Sealed and may not be used as a superclass.",
                                  "MATLAB:class:sealed")
            allowed = parent.cd.attrs.get("AllowedSubclasses")
            if allowed is not None and self.name not in class_list(allowed):
                raise MatlabError(f"Class '{self.name}' is not on the AllowedSubclasses list of "
                                  f"'{parent.name}'.", "MATLAB:class:InvalidSuperClass")
            self.parents.append(parent)
            self.is_handle = self.is_handle or parent.is_handle
            for b in parent.bases:
                if b not in self.bases:
                    self.bases.append(b)
        if self.is_handle and "handle" not in self.bases:
            self.bases.append("handle")
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
            inherited = self.methods.get(mname)
            if inherited is not None and self.method_attrs.get(mname, {}).get("Sealed") \
                    and "." not in mname:
                raise MatlabError(f"Method '{mname}' is Sealed in class '{inherited[1].name}' and cannot be "
                                  f"redefined in '{self.name}'.", "MATLAB:class:SealedMethod")
            self.methods[mname] = (fn, self)
        self.static |= set(cd.static_methods)
        self.method_attrs.update(cd.method_attrs)
        self.abstract_methods = (abstract | set(cd.abstract_methods)) - {
            m for m, (_, owner) in self.methods.items() if owner is self}
        self._load_folder_methods()
        self.abstract_props = {n for n, (p, _) in self.prop_defs.items() if p.attrs.get("Abstract")}
        self.is_abstract = bool(cd.attrs.get("Abstract")) or bool(self.abstract_methods) \
            or bool(self.abstract_props)
        self._defaults: dict[str, Any] | None = None
        self._constants: dict[str, Any] = {}
        # Methods of the built-in base classes (copy, set, get, addprop, verifyEqual, ...).
        self.base_methods: dict[str, Callable] = {}
        for b in reversed(self.bases):
            self.base_methods.update(BUILTIN_BASES[b].methods)
        # Classes this one is superior to in operator dispatch (the InferiorClasses attribute).
        self.inferior: set[str] = set(class_list(cd.attrs.get("InferiorClasses")))
        self.superior_to_me: set[str] = set()
        self.overloads_ref = any(m in self.methods for m in ("subsref", "parenReference", "dotReference",
                                                               "braceReference"))
        self.overloads_asgn = any(m in self.methods for m in ("subsasgn", "parenAssign", "dotAssign",
                                                                "braceAssign"))
        if not getattr(interp, "index_overloads", False) and any(m in self.methods for m in _INDEX_METHODS):
            interp.index_overloads = True  # the interpreter now checks for indexing overloads
        # A handle class with a delete method: its objects run it when the last reference goes away.
        self.has_destructor = self.is_handle and "delete" in self.methods

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
                program = parse(read_source(os.path.join(folder, entry)))
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

    def lineage(self) -> list["ClassInfo"]:
        """This class and every classdef superclass, nearest first."""
        out = [self]
        for p in self.parents:
            for c in p.lineage():
                if c not in out:
                    out.append(c)
        return out

    def isa(self, name: str) -> bool:
        if name == self.full_name or name in self.bases or name in self.ancestors():
            return True
        if any(c.full_name == name for c in self.lineage()):
            return True
        base = self.numeric_base
        if base is None:
            return False
        if name == base:
            return True
        if name == "numeric":
            return base not in ("logical", "char")
        if name == "float":
            return base in ("double", "single")
        return name == "integer" and "int" in base

    def instance_props(self) -> list[str]:
        return [n for n, (p, _) in self.prop_defs.items() if not p.attrs.get("Constant")]

    def is_constant(self, name: str) -> bool:
        entry = self.prop_defs.get(name)
        return bool(entry and entry[0].attrs.get("Constant"))

    def public_props(self, interp=None, hidden: bool = False) -> list[str]:
        """Names of properties code outside the class may read (Hidden ones only on request)."""
        out = []
        for n, (p, owner) in self.prop_defs.items():
            level = p.attrs.get("GetAccess", p.attrs.get("Access", "public"))
            if p.attrs.get("Hidden") and not hidden:
                continue
            if str(level).lower() == "public" or (interp is not None and self.allowed(interp, level, owner)):
                out.append(n)
        return out

    # -- defaults and constants ------------------------------------------------------------
    def _eval_default(self, interp, prop: A.PropertyDef, owner: "ClassInfo") -> Any:
        if prop.default is None:
            return typed_default(interp, prop.attrs)
        from .interpreter import Frame
        frame = Frame(interp, owner.functions, name=f"{owner.name} property default", file=owner.path)
        frame.owner_class = owner
        value = interp.eval(prop.default, frame)
        if prop.attrs.get("_class") or prop.attrs.get("_size") or prop.attrs.get("_validators"):
            value = validate_value(interp, prop.attrs, value, prop.name, owner.functions,
                                   f"Invalid default value for property '{prop.name}' of class "
                                   f"'{owner.name}'.")
        return value

    def defaults(self, interp) -> dict[str, Any]:
        # Like MATLAB, default values are evaluated once, when the class is first used.
        if self._defaults is None:
            values: dict[str, Any] = {}
            for name in self.instance_props():
                prop, owner = self.prop_defs[name]
                if prop.attrs.get("Dependent"):
                    continue  # computed by its get method; nothing is stored
                values[name] = self._eval_default(interp, prop, owner)
            self._defaults = values
        return self._defaults

    def constant(self, interp, name: str) -> Any:
        prop, owner = self.prop_defs[name]
        if owner is not self:
            return owner.constant(interp, name)  # one value shared with the defining class
        if name not in self._constants:
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
    def new_object(self, interp) -> "UserObject":
        """An object holding the default property values, before any constructor has run."""
        obj = UserObject(self)
        obj.props = dict(self.defaults(interp))
        if self.numeric_base and not self.enums:
            obj._data = _empty_of(self.numeric_base)
        return obj

    def construct(self, interp, args: list, nargout: int) -> list:
        if self.old_style:  # the constructor is the function file @name/name.m
            program = interp._load_file(self.path)
            if program is None or not program.functions:
                raise MatlabError(f"The constructor of class '{self.name}' was not found.")
            main = next(iter(program.functions.values()))
            return interp.call_user_function(main, list(args), nargout, program.functions, self.path,
                                             owner=self)
        if self.enums:
            if len(args) != 1:
                raise MatlabError(f"Use {self.name}.MemberName to get a member of the enumeration class "
                                  f"'{self.name}'.")
            value = args[0]
            if isinstance(value, ObjectArray) and value.cls is self:
                return [value]
            from .values import Cell, StringArray
            if isinstance(value, (Cell, StringArray)) or (isinstance(value, np.ndarray) and value.size > 1
                                                           and value.dtype.kind != "U"):
                items = value.data if isinstance(value, (Cell, StringArray)) else value
                data = np.empty(items.shape, dtype=object)
                for idx in np.ndindex(items.shape):
                    item = items[idx]
                    if isinstance(item, str):
                        from .values import char
                        item = char(item)
                    elif not isinstance(item, np.ndarray):
                        item = np.array([[item]])
                    data[idx] = self.enum_lookup(interp, item if item.ndim == 2 else item.reshape(1, 1))
                return [ObjectArray._unwrap(data, self)]
            return [self.enum_lookup(interp, value)]
        if self.is_abstract:
            missing = ", ".join(sorted(self.abstract_methods | self.abstract_props))
            detail = f" (abstract members: {missing})" if missing else ""
            raise MatlabError(f"Cannot instantiate the abstract class '{self.name}'{detail}.",
                              "MATLAB:class:abstract")
        access = self.method_attrs.get(self.name, {}).get("Access")
        if access is not None and not self.allowed(interp, access, self):
            raise MatlabError(f"Cannot access the constructor of class '{self.name}'.",
                              "MATLAB:class:MethodRestricted")
        made = self.run_constructor(interp, self.new_object(interp), args, nargout)
        if self.has_destructor and made is not None:
            self.arm(made)
        return [] if made is None else [made]

    def arm(self, made: Any) -> None:
        """Mark finished handle objects: from now on losing the last reference runs their delete method."""
        if not self.has_destructor:
            return
        for obj in (made.data.flat if isinstance(made, ObjectArray) else (made,)):
            if isinstance(obj, UserObject):
                obj._alive = True

    def run_constructor(self, interp, obj: "UserObject", args: list, nargout: int = 1) -> Any:
        ctor = self.cd.methods.get(self.name)
        if ctor is None:
            if args and self.numeric_base and not self.parents:
                if len(args) > 1:
                    raise MatlabError(f"Too many input arguments for the default constructor of '{self.name}'.")
                obj._data = _to_base(interp, self.numeric_base, args[0])
                return obj
            if args and not self.parents:
                raise MatlabError(f"Too many input arguments for the default constructor of '{self.name}'.")
            for k, parent in enumerate(self.parents):
                # A class without a constructor hands its arguments to its (first) superclass.
                obj = parent.run_constructor(interp, obj, args if k == 0 else [])
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
        if not res or not isinstance(res[0], (UserObject, ObjectArray)):
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
        if level is None or level is True:
            return True
        text = str(level).lower()
        if text == "public" or text == "true":
            return True
        caller = self._caller(interp)
        if caller is None:
            return False
        if caller is owner or caller.name == owner.name:
            return True
        names = access_names(level)
        if names is not None:  # an access list: those classes and their subclasses
            return any(caller.isa(n) for n in names)
        if text in ("private", "immutable", "none"):
            return False
        return owner.name in caller.ancestors()  # protected

    def check_method_access(self, interp, name: str) -> None:
        access = self.method_attrs.get(name, {}).get("Access")
        if access is not None and not self.allowed(interp, access, self.methods[name][1]):
            raise MatlabError(f"Cannot access method '{name}' in class '{self.name}'.",
                              "MATLAB:class:MethodRestricted")

    def static_access(self, interp, member: str, args: list, nargout: int) -> list:
        if member in self.methods and member in self.static:
            self.check_method_access(interp, member)
            return self.call(interp, member, args, nargout)
        if member in self.enums:
            return [self.enum_members(interp)[member]]
        if member == "empty":
            dims = [int(np.real(x)) for a in args for x in np.asarray(a).flatten()] or [0, 0]
            if len(dims) == 1:
                dims = [dims[0], dims[0]]
            if 0 not in dims:
                raise MatlabError("At least one dimension of an empty array must be zero.")
            return [ObjectArray(np.empty(tuple(dims), dtype=object), self)]
        if self.is_constant(member):
            prop, owner = self.prop_defs[member]
            level = prop.attrs.get("GetAccess", prop.attrs.get("Access"))
            if level is not None and not self.allowed(interp, level, owner):
                raise MatlabError(f"You cannot access the '{member}' property of {self.name}.",
                                  "MATLAB:class:GetProhibited")
            value = self.constant(interp, member)
            if args:
                from .interpreter import index_value
                if isinstance(value, MObject):
                    return [value.index_get(interp, args)]
                return [index_value(value, args)]
            return [value]
        if member in self.methods:
            raise MatlabError(f"Method '{member}' of class '{self.name}' is not Static; call it on an object.")
        fn = self.base_methods.get(member)
        if fn is not None and getattr(fn, "static", False):
            return interp.call_builtin(member, fn, list(args), nargout)
        raise MatlabError(f"Unrecognized method, property, or field '{member}' for class '{self.name}'.")


def _empty_of(base: str) -> np.ndarray:
    if base == "char":
        return np.zeros((0, 0), dtype="<U1")
    if base == "logical":
        return np.zeros((0, 0), dtype=bool)
    return np.zeros((0, 0), dtype=np.float32 if base == "single" else (float if base == "double" else base))


def _to_base(interp, base: str, value: Any) -> np.ndarray:
    """Convert a value to the built-in type a class derives from (``o = o@double(v)``)."""
    if isinstance(value, UserObject) and value._data is not None:
        value = value._data
    res = interp.call_builtin(base, interp.builtins[base], [value], 1)
    return res[0]


def typed_default(interp, attrs: dict) -> Any:
    """The value a property has when its declaration gives a class or size but no default."""
    from .values import INT_TYPES, Cell, MString, StringArray, Struct
    cls, size = attrs.get("_class"), attrs.get("_size")
    if not cls and not size:
        return np.zeros((0, 0))
    dims = tuple(0 if d is None else d for d in size) if size else (0, 0)
    if len(dims) == 1:
        dims = (dims[0], dims[0])
    cls = cls or "double"
    if cls == "double":
        return np.zeros(dims)
    if cls == "single":
        return np.zeros(dims, dtype=np.float32)
    if cls in INT_TYPES:
        return np.zeros(dims, dtype=INT_TYPES[cls])
    if cls == "logical":
        return np.zeros(dims, dtype=bool)
    if cls == "char":
        return np.full(dims, "\0", dtype="<U1")
    if cls == "string":
        if all(d == 1 for d in dims):
            return MString("")
        data = np.empty(dims, dtype=object)
        data.fill("")
        return StringArray(data)
    if cls == "cell":
        data = np.empty(dims, dtype=object)
        for idx in np.ndindex(dims):
            data[idx] = np.zeros((0, 0))
        return Cell(data)
    if cls == "struct":
        data = np.empty(dims, dtype=object)
        for idx in np.ndindex(dims):
            data[idx] = {}
        return Struct(data, [])
    klass = interp.find_class(cls) if interp is not None else None
    if klass is not None:
        if int(np.prod(dims)) == 0:
            return ObjectArray(np.empty(dims, dtype=object), klass)
        first = next(iter(klass.enum_members(interp).values())) if klass.enums else None
        data = np.empty(dims, dtype=object)
        for idx in np.ndindex(dims):
            data[idx] = first if first is not None else klass.construct(interp, [], 1)[0]
        return ObjectArray._unwrap(data, klass)
    if all(d == 1 for d in dims) and interp is not None and cls in interp.builtins \
            and cls != "function_handle":
        try:
            return interp.call_builtin(cls, interp.builtins[cls], [], 1)[0]  # datetime, duration, table, ...
        except MatlabError:
            pass
    return np.zeros((0, 0))


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
    _data: Any = None               # the built-in value of an object of a class derived from double, char, ...
    _dynprops: dict | None = None   # properties added with addprop: name -> meta.DynamicProperty

    def __init__(self, cls: ClassInfo):
        self.cls = cls
        self.props: dict[str, Any] = {}
        self._in_accessor: set[str] = set()

    def __del__(self) -> None:
        # MATLAB destroys a handle object when nothing refers to it any more (clear, reassignment,
        # the end of a function) and runs the class's delete method at that moment.
        d = self.__dict__
        if not d.get("_alive") or d.get("_deleted"):
            return
        d["_alive"] = False
        try:
            from .interpreter import current_interpreter
            interp = current_interpreter()
            if interp is None:
                return
            from .builtins.objfuncs import delete_object
            try:
                delete_object(interp, self)
            except MatlabError as e:
                interp.warn(f"The following error was caught while executing '{self.cls.name}' class "
                            f"destructor: {e.message}", "MATLAB:class:DestructorError")
        except Exception:  # never let a destructor disturb the code that dropped the object
            pass

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return self.cls.is_handle

    def class_name(self) -> str:
        return self.cls.full_name

    def isa_class(self, name: str) -> bool:
        return self.cls.isa(name)

    def matlab_shape(self) -> tuple:
        return self._data.shape if self._data is not None else (1, 1)

    def copy(self) -> "UserObject":
        other = UserObject(self.cls)
        other.props = dict(self.props)
        other._enum_name = self._enum_name
        other._enum_value = self._enum_value
        other._data = self._data
        return other

    def to_struct(self) -> Any:
        """struct(obj): the property values as a struct (old-style classes: the fields)."""
        from .interpreter import current_interpreter
        from .values import Struct
        interp = current_interpreter()
        if self.cls.old_style:
            return Struct.scalar(dict(self.props))
        if interp is not None:
            interp.warn("Calling struct on an object exposes its private and hidden properties; prefer the "
                        "object's own methods and properties.", "MATLAB:structOnObject")
        out = {}
        for n in self.cls.instance_props():
            try:
                out[n] = self.get_field(interp, n) if interp is not None and \
                    self.cls.prop_defs[n][0].attrs.get("Dependent") else self.props.get(n, np.zeros((0, 0)))
            except MatlabError:
                out[n] = np.zeros((0, 0))
        for n, d in (self._dynprops or {}).items():
            out[n] = d.stored
        return Struct.scalar(out)

    def with_data(self, data: Any) -> "UserObject":
        """An object like this one holding other built-in data (classes derived from double, ...)."""
        other = self.copy()
        other._data = data
        return other

    # -- members ---------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        cls = self.cls
        # delete(obj) goes through the built-in, which runs the class's delete and invalidates the handle.
        if name in cls.methods:
            return name != cls.name and "." not in name and (name != "delete" or cls.is_handle)
        if self._enum_name is not None and (name in _ENUM_METHODS or (
                cls.numeric_base and name in NUMERIC_BASES)):
            return True
        if name in cls.prop_defs:
            return False  # a property with the name of a built-in method wins
        if name in cls.base_methods:
            return True
        if name == "step":
            return "stepImpl" in cls.methods  # System objects
        if name in _ARRAY_METHODS or (cls.is_handle and name in HANDLE_METHODS):
            return True
        return self._data is not None and _works_on_data(name)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        cls = self.cls
        if name == "delete" and cls.is_handle:  # obj.delete(): the built-in runs the class's delete
            return interp.call_builtin(name, interp.builtins[name], list(args), nargout)
        if name not in cls.methods:
            if self._enum_name is not None and (name in _ENUM_METHODS or name in NUMERIC_BASES):
                return [_enum_method(interp, name, args)]
            fn = cls.base_methods.get(name)
            if fn is not None:
                return interp.call_builtin(name, fn, list(args), nargout)
            if self._data is not None and _works_on_data(name):
                return call_on_data(interp, name, args, nargout)
            if name in _ARRAY_METHODS:
                return [_ARRAY_METHODS[name](interp, args)]
            if (cls.is_handle and name in HANDLE_METHODS) or name == "step":
                return interp.call_builtin(name, interp.builtins[name], list(args), nargout)
            raise MatlabError(f"Unrecognized method '{name}' for class '{cls.name}'.")
        self._check_valid()
        cls.check_method_access(interp, name)
        if name in cls.static and args and args[0] is self:
            args = args[1:]
        return cls.call(interp, name, args, nargout)

    def _check_valid(self) -> None:
        if getattr(self, "_deleted", False):
            raise MatlabError("Invalid or deleted object.", "MATLAB:class:InvalidHandle")

    def _check_access(self, interp, name: str, kind: str) -> None:
        entry = self.cls.prop_defs.get(name)
        if entry is None:
            return
        prop, owner = entry
        level = prop.attrs.get(kind, prop.attrs.get("Access"))
        if level is not None and not self.cls.allowed(interp, level, owner):
            if kind == "SetAccess":
                raise MatlabError(f"You cannot set the read-only property '{name}' of {self.cls.name}.",
                                  "MATLAB:class:SetProhibited")
            raise MatlabError(f"You cannot access the '{name}' property of {self.cls.name}.",
                              "MATLAB:class:GetProhibited")

    def validate(self, interp, name: str, value: Any) -> Any:
        """Apply a property's class, size and validator functions (as declared in the classdef)."""
        prop, owner = self.cls.prop_defs[name]
        attrs = prop.attrs
        if "_class" not in attrs and "_size" not in attrs and "_validators" not in attrs:
            return value
        return validate_value(interp, attrs, value, name, owner.functions,
                              f"Error setting property '{name}' of class '{self.cls.name}'.")

    def get_field(self, interp, name: str) -> Any:
        cls = self.cls
        entry = cls.prop_defs.get(name)
        if entry is None:
            return self._get_other(interp, name)
        self._check_valid()
        prop, owner = entry
        attrs = prop.attrs
        level = attrs.get("GetAccess", attrs.get("Access"))
        if level is not None and not cls.allowed(interp, level, owner):
            raise MatlabError(f"You cannot access the '{name}' property of {cls.name}.",
                              "MATLAB:class:GetProhibited")
        if attrs.get("Constant"):
            return cls.constant(interp, name)
        listeners = self._listeners
        if listeners:
            from .builtins.events import property_event
            property_event(interp, self, name, "PreGet")
        getter = "get." + name
        if getter in cls.methods and name not in self._in_accessor:
            self._in_accessor.add(name)
            try:
                res = cls.call(interp, getter, [self], 1)
            finally:
                self._in_accessor.discard(name)
            if not res:
                raise MatlabError(f"Get method for '{name}' did not return a value.")
            value = res[0]
        else:
            value = self.props.get(name)
            if value is None:
                value = np.zeros((0, 0))  # a Dependent property without a get method
        if listeners:
            property_event(interp, self, name, "PostGet")
        return value

    def method_statement(self, interp, name: str, nargout: int) -> list | None:
        """``obj.name`` naming a method, called with ``nargout`` outputs (None when it names no method).

        As a statement (``obj.update;``) the method is called with no outputs, so one that returns
        nothing works; reading a value calls it with one.
        """
        if name in self.cls.prop_defs or (self._dynprops and name in self._dynprops) or name in self.props \
                or self.cls.old_style or not self.has_method(name):
            return None
        return self.call_method(interp, name, [self], nargout)

    def _get_other(self, interp, name: str) -> Any:
        """``obj.name`` when ``name`` is not a declared property: dynamic property, field or method."""
        cls = self.cls
        if self._dynprops and name in self._dynprops:
            self._check_valid()
            return self._dynprops[name].read(interp, self)
        if cls.old_style:
            holder = self._field_holder(name)
            if holder is not None:
                self._old_access(interp)
                return holder.props[name]
        elif name in self.props:
            return self.props[name]
        if self.has_method(name):
            res = self.call_method(interp, name, [self], 1)
            if not res:
                raise MatlabError(f"Method '{name}' does not return a value.")
            return res[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{cls.name}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "UserObject":
        cls = self.cls
        entry = cls.prop_defs.get(name)
        if entry is None:
            return self._set_other(interp, name, value)
        prop, owner = entry
        attrs = prop.attrs
        if attrs.get("Constant"):
            raise MatlabError(f"Cannot set the Constant property '{name}' of class '{cls.name}'.",
                              "MATLAB:class:SetProhibited")
        self._check_valid()
        level = attrs.get("SetAccess", attrs.get("Access"))
        if level is not None and not cls.allowed(interp, level, owner):
            raise MatlabError(f"You cannot set the read-only property '{name}' of {cls.name}.",
                              "MATLAB:class:SetProhibited")
        setter = "set." + name
        has_setter = setter in cls.methods and name not in self._in_accessor
        if attrs.get("Dependent") and not has_setter and setter not in cls.methods:
            raise MatlabError(f"In class '{cls.name}', no set method is defined for Dependent property "
                              f"'{name}'. A Dependent property needs a set method to assign its value.",
                              "MATLAB:class:noSetMethod")
        if "_class" in attrs or "_size" in attrs or "_validators" in attrs:
            value = validate_value(interp, attrs, value, name, owner.functions,
                                   f"Error setting property '{name}' of class '{cls.name}'.")
        if attrs.get("AbortSet") and name in self.props:
            from .values import values_equal
            if values_equal(self.props[name], value):
                return self  # AbortSet: the value does not change, so nothing runs
        listeners = self._listeners
        if listeners:
            from .builtins.events import property_event
            property_event(interp, self, name, "PreSet")
        if has_setter:
            target = self if cls.is_handle else self.copy()
            target._in_accessor.add(name)
            try:
                res = cls.call(interp, setter, [target, value], 1 if cls.methods[setter][0].outputs else 0)
            finally:
                target._in_accessor.discard(name)
            if not cls.is_handle:
                if not res or not isinstance(res[0], UserObject):
                    raise MatlabError(f"Set method for '{name}' must return the object.")
                target = res[0]
        else:
            target = self if cls.is_handle else self.copy()
            target.props[name] = value
        if listeners:
            property_event(interp, self, name, "PostSet")
        return target

    def validated(self, interp, name: str, value: Any) -> Any:
        """``value`` after the size, class and validation functions of declared property ``name``
        (converted as an assignment converts it); raises their error."""
        prop, owner = self.cls.prop_defs[name]
        attrs = prop.attrs
        if "_class" in attrs or "_size" in attrs or "_validators" in attrs:
            value = validate_value(interp, attrs, value, name, owner.functions,
                                   f"Error setting property '{name}' of class '{self.cls.name}'.")
        return value

    def restore_field(self, interp, name: str, value: Any) -> "UserObject":
        """Give a declared property its saved (and :meth:`validated`) value while load rebuilds this
        object: its set method runs, as for an assignment, but access attributes do not apply and no
        listeners exist yet. Raises the error of a set method that fails. Returns the object (a new
        one for a value class whose set method returns it).
        """
        cls = self.cls
        setter = "set." + name
        if setter not in cls.methods:
            self.props[name] = value  # the object is still being built: nothing else refers to it yet
            return self
        self._in_accessor.add(name)
        try:
            res = cls.call(interp, setter, [self, value], 1 if cls.methods[setter][0].outputs else 0)
        finally:
            self._in_accessor.discard(name)
        if cls.is_handle:
            return self
        if not res or not isinstance(res[0], UserObject):
            raise MatlabError(f"Set method for '{name}' must return the object.")
        return res[0]

    def _set_other(self, interp, name: str, value: Any) -> "UserObject":
        cls = self.cls
        if self._dynprops and name in self._dynprops:
            self._check_valid()
            self._dynprops[name].write(interp, self, value)
            return self
        if cls.old_style:
            self._old_access(interp)
            target = self.copy()
            if name in self.props:
                target.props[name] = value
                return target
            for key, part in self.props.items():  # a field of the parent part of the object
                if isinstance(part, UserObject) and part.cls.old_style and part._field_holder(name) is not None:
                    target.props[key] = part._set_other(interp, name, value)
                    return target
        raise MatlabError(f"Unrecognized property '{name}' for class '{cls.name}'.",
                          "MATLAB:noPublicFieldForClass")

    def _field_holder(self, name: str) -> "UserObject | None":
        """The object (this one, or the parent part inside it) that stores field ``name``."""
        if name in self.props:
            return self
        for part in self.props.values():
            if isinstance(part, UserObject) and part.cls.old_style:
                holder = part._field_holder(name)
                if holder is not None:
                    return holder
        return None

    def _old_access(self, interp) -> None:
        caller = ClassInfo._caller(interp)
        if caller is None or not (caller in self.cls.lineage() or self.cls in caller.lineage()):
            raise MatlabError(f"Access to the fields of a '{self.cls.name}' object is only permitted within "
                              "its methods.", "MATLAB:class:fieldAccess")

    # -- indexing: one object behaves as a 1-by-1 array -------------------------------------
    def index_get(self, interp, args: list) -> Any:
        if self._data is not None:
            from .interpreter import index_value
            return self.with_data(index_value(self._data, args))
        return ObjectArray.of(self).index_get(interp, args)

    def index_peek(self, interp, args: list) -> Any:
        if self._data is not None:
            return self.index_get(interp, args)
        return ObjectArray.of(self).index_peek(interp, args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        if self._data is not None:
            from . import indexing as ix
            from .interpreter import coerce_for_assignment
            rhs = value._data if isinstance(value, UserObject) and value._data is not None else value
            if not isinstance(rhs, np.ndarray):
                raise MatlabError(f"Cannot store a {_class_name(rhs)} in a {self.cls.name} array.")
            cur, val = coerce_for_assignment(self._data, rhs)
            fill = (lambda: "\0") if cur.dtype.kind == "U" else ((lambda: False) if cur.dtype == bool
                                                                 else (lambda: 0))
            return self.with_data(ix.set_(cur, list(args), val, fill))
        return ObjectArray.of(self).index_set(interp, args, value)

    def index_delete(self, interp, args: list) -> Any:
        if self._data is not None:
            from . import indexing as ix
            return self.with_data(ix.delete(self._data, list(args)))
        return ObjectArray.of(self).index_delete(interp, args)

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        return ObjectArray.concat(items, axis)

    def colon_range(self, start: Any, step: Any, stop: Any) -> Any:
        return object_colon(start, step, stop)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if self._data is not None or (self._enum_value is not None and op not in ("==", "~=")):
            return interp.binop_values(op, _plain(left), _plain(right))
        if op in ("==", "~=") and (self._enum_name is not None or not self.is_handle):
            other = right if left is self else left
            if isinstance(other, ObjectArray):
                return NotImplemented
            same = _same_object(self, other)
            if same is None:
                return NotImplemented
            return np.array([[same if op == "==" else not same]])
        return NotImplemented

    def unary_op(self, interp, op: str) -> Any:
        if self._data is not None:
            from . import ops
            if op in ("'", ".'"):
                return self.with_data(ops.transpose(self._data, op == "'"))
            return ops.unary(op, self._data)
        if op in ("'", ".'"):
            return self  # the transpose of one object is the object
        if self._enum_value is not None:
            from . import ops
            return ops.unary(op, self._enum_value)
        return NotImplemented

    def summary_text(self, in_cell: bool) -> str:
        if self._enum_name is not None:
            return self._enum_name
        size = "×".join(str(d) for d in self.matlab_shape())
        return f"{size} {self.cls.name}" if in_cell else f"[{size} {self.cls.name}]"

    def format_display(self, name: str) -> str:
        if self._enum_name is not None:
            return f"{name} = \n\n  {self.cls.name} enumeration\n\n    {self._enum_name}\n\n"
        return super().format_display(name)

    # -- display -------------------------------------------------------------------------------
    def shown_props(self, interp=None) -> list[str]:
        names = self.cls.public_props(interp)
        names = [n for n in names if not self.cls.prop_defs[n][0].attrs.get("Constant")] + \
            [n for n in names if self.cls.prop_defs[n][0].attrs.get("Constant")]
        if self._dynprops:
            names += [n for n, d in self._dynprops.items() if not d.hidden and d.get_access == "public"]
        if self.cls.old_style:
            names = []
        return names

    def shown_value(self, interp, n: str) -> Any:
        if interp is not None:
            try:
                return self.get_field(interp, n)
            except MatlabError:
                pass
        return self.props.get(n, np.zeros((0, 0)))

    def display_lines(self) -> list[str]:
        from .display import display_lines as value_lines
        from .display import summary
        from .interpreter import current_interpreter
        cls = self.cls
        if self._enum_name is not None:
            return [f"    {self._enum_name}"]
        if getattr(self, "_deleted", False):
            return [f"  handle to deleted {cls.name}"]
        if cls.old_style:
            return [f"\t{cls.name} object: 1-by-1"]
        interp = current_interpreter()
        names = self.shown_props()
        lines: list[str] = []
        if self._data is not None:
            size = "×".join(str(d) for d in self._data.shape)
            head = f"  {cls.name}" if self._data.size == 1 else f"  {size} {cls.name}"
            lines = [head + (" with properties:" if names else ":")]
            if names:
                w = max(len(n) for n in names)
                lines += [""] + [f"    {n:>{w}}: {summary(self.shown_value(interp, n), False)}" for n in names]
                lines.append("")
            lines.append(f"  {cls.numeric_base} data:")
            return lines + value_lines(self._data)
        if not names:
            return [f"  {cls.name} with no properties."]
        w = max(len(n) for n in names)
        return [f"  {cls.name} with properties:", ""] + \
            [f"    {n:>{w}}: {summary(self.shown_value(interp, n), False)}" for n in names]


_ENUM_METHODS = {"char", "string", "cellstr", "num2str", "strcmp", "strcmpi"}

# Built-in functions that must see the object itself, not the built-in data of a class derived from
# double, char, ... (every other built-in function works on that data).
_KEEP_OBJECT = frozenset({
    "class", "isa", "isobject", "disp", "display", "methods", "properties", "metaclass", "isprop",
    "ismethod", "superclasses", "events", "enumeration", "isenum", "builtin", "cellfun", "arrayfun",
    "feval", "struct", "deal", "num2cell", "setfield", "getfield", "cell2struct", "containers.Map",
    "save", "assignin", "evalin", "eval", "evalc", "inputname", "func2str", "fieldnames", "isfield",
    "table", "subsref", "subsasgn", "substruct", "copy", "isvalid", "delete", "onCleanup", "who", "whos",
    "exist", "clear", "clearvars", "validateattributes", "mustBeA", "iscell", "isstruct", "isstring",
    "is_function_handle", "details", "inputParser", "varargin", "isequal", "isequaln", "jsonencode",
    "set", "get", "findobj", "addprop", "numArgumentsFromSubscript", "functions", "localfunctions",
})
# Functions whose result has the class of the object they were given (the others give the built-in type).
_KEEP_CLASS = frozenset({
    "reshape", "permute", "ipermute", "transpose", "ctranspose", "sort", "repmat", "circshift", "flip",
    "fliplr", "flipud", "rot90", "squeeze", "horzcat", "vertcat", "cat", "shiftdim",
})
_ALL_BUILTINS: dict | None = None


def _works_on_data(name: str) -> bool:
    global _ALL_BUILTINS
    if _ALL_BUILTINS is None:
        from .builtins import BUILTINS
        _ALL_BUILTINS = BUILTINS
    return name in _ALL_BUILTINS and name not in _KEEP_OBJECT


def _plain(v: Any) -> Any:
    """The built-in value behind an object of a class derived from a built-in type."""
    if type(v) is UserObject:
        if v._data is not None:
            return v._data
        if v._enum_value is not None:
            return v._enum_value
    return v


def call_on_data(interp, name: str, args: list, nargout: int) -> list:
    """Run a built-in function on the built-in data of objects derived from double, char, ..."""
    first = next(a for a in args if type(a) is UserObject and a._data is not None)
    res = interp.call_builtin(name, interp.builtins[name], [_plain(a) for a in args], nargout)
    if name in _KEEP_CLASS and res and isinstance(res[0], np.ndarray):
        res[0] = first.with_data(res[0])
    return res


def object_colon(start: Any, step: Any, stop: Any) -> Any:
    """``a:b`` or ``a:d:b`` with an object operand: the class's colon method, or its built-in data."""
    from .interpreter import colon_range, current_interpreter
    interp = current_interpreter()
    operands = [v for v in (start, step, stop) if v is not None]
    for v in operands:
        if isinstance(v, (UserObject, ObjectArray)) and v.cls is not None and "colon" in v.cls.methods:
            res = v.cls.call(interp, "colon", operands, 1)
            if not res:
                raise MatlabError("The colon method did not return a value.")
            return res[0]
    if any(type(v) is UserObject and (v._data is not None or v._enum_value is not None) for v in operands):
        return colon_range(_plain(start), None if step is None else _plain(step), _plain(stop))
    kind = next(v.class_name() for v in operands if isinstance(v, MObject))
    raise MatlabError(f"Operator ':' is not supported for operands of type '{kind}'.",
                      "MATLAB:UndefinedFunction")


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
    from .builtins.data import num2cell_dims, split_to_cells
    from .values import Cell
    arr = _as_array(args[0])
    if len(args) > 2:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    if len(args) == 1:
        return Cell(arr.data.copy())
    return split_to_cells(arr.data, num2cell_dims(args[1]), lambda d: ObjectArray._unwrap(d, arr.cls))


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


_GAP = object()  # marks the elements a growing assignment leaves unset


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
        return ObjectArray(data, common_class(data.flat, cls))

    def elements(self) -> list:
        return list(self.data.flatten(order="F"))

    def mixed(self) -> bool:
        """True for a heterogeneous array (elements of different classes)."""
        first = None
        for e in self.data.flat:
            if first is None:
                first = e.cls
            elif e.cls is not first:
                return True
        return False

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return bool(self.cls and self.cls.is_handle)

    def class_name(self) -> str:
        return self.cls.full_name if self.cls else "double"

    def isa_class(self, name: str) -> bool:
        if self.cls is not None and self.cls.isa(name):
            return True
        return bool(self.data.size) and all(e.isa_class(name) for e in self.data.flat)

    def matlab_shape(self) -> tuple:
        return tuple(self.data.shape)

    def empty_like(self) -> "ObjectArray":
        return ObjectArray(np.empty((0, 0), dtype=object), self.cls)

    def to_struct(self) -> Any:
        from .values import Struct
        data = np.empty(self.data.shape, dtype=object)
        fields: list[str] = []
        for idx in np.ndindex(self.data.shape):
            one = self.data[idx].to_struct()
            data[idx] = one.data.flat[0]
            fields = one.fields
        if not self.data.size and self.cls is not None:
            fields = self.cls.instance_props()
        return Struct(data, fields)

    # -- members: methods receive the whole array; properties are read element by element --------
    def has_method(self, name: str) -> bool:
        cls = self.cls
        if cls is None:
            return False
        if name in cls.methods:
            return name != cls.name and "." not in name and name != "delete"
        if name in cls.prop_defs:
            return False
        return name in cls.base_methods or name in _ARRAY_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        cls = self.cls
        if name not in cls.methods:
            fn = cls.base_methods.get(name)
            if fn is not None:
                return interp.call_builtin(name, fn, list(args), nargout)
            if name in _ARRAY_METHODS:
                return [_ARRAY_METHODS[name](interp, args)]
            raise MatlabError(f"Unrecognized method '{name}' for class '{cls.name}'.")
        if not cls.method_attrs.get(name, {}).get("Sealed") and self.mixed():
            raise MatlabError(f"Method '{name}' is not Sealed, so it cannot be called on a heterogeneous "
                              f"array of class '{cls.name}'. Call it on one element, e.g. objs(k).{name}().",
                              "MATLAB:class:NonSealedMethodHeterogeneous")
        cls.check_method_access(interp, name)
        if name in cls.static and args and args[0] is self:
            args = args[1:]
        return cls.call(interp, name, args, nargout)

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

    def index_peek(self, interp, args: list) -> Any:
        """The element ``objs(k).name = value`` changes: a default object when ``k`` is past the end."""
        try:
            return self.index_get(interp, args)
        except MatlabError:
            if self.cls is None or not all(isinstance(a, np.ndarray) and a.size == 1 and a.dtype != bool
                                           for a in args):
                raise
            return self._fill(interp)

    def _fill(self, interp) -> Any:
        """The object used for elements an assignment creates but does not name."""
        cls = self.cls
        if cls is None:
            raise MatlabError("Cannot grow an array of objects of unknown class.")
        if cls.enums:
            return next(iter(cls.enum_members(interp).values()))
        if "getDefaultScalarElement" in cls.methods and "getDefaultScalarElement" in cls.static:
            res = cls.call(interp, "getDefaultScalarElement", [], 1)
            if res and isinstance(res[0], UserObject):
                return res[0]
            raise MatlabError(f"getDefaultScalarElement of class '{cls.name}' must return one object.")
        try:
            return cls.construct(interp, [], 1)[0]
        except MatlabError as e:
            raise MatlabError(f"Growing the array needs a default {cls.name} object, but calling "
                              f"{cls.name}() with no arguments failed: {e.message}") from None

    def index_set(self, interp, args: list, value: Any) -> Any:
        from . import indexing as ix
        if self.cls is not None and not isinstance(value, MObject):
            value = convert_to_class(interp, value, self.cls)  # objs(k) = 5 calls the constructor
        if isinstance(value, UserObject):
            rhs = ObjectArray.of(value).data
        elif isinstance(value, ObjectArray) and not isinstance(value, NativeArray):
            rhs = value.data
        else:
            raise MatlabError(f"Cannot store a {_class_name(value)} in an array of "
                              f"{self.class_name()} objects.")
        base = self
        if self.cls is None or (self.data.size == 0 and not _compatible(self.cls, value.cls)):
            base = ObjectArray(self.data, value.cls)  # an empty array takes the class of what it gets
        # Growing makes new elements; only those the assignment leaves unset need a default object,
        # so objs(end + 1) = obj never calls the constructor with no arguments.
        gap = _GAP
        data = ix.set_(base.data.copy(), list(args), rhs, lambda: gap)
        if data.size > self.data.size:
            for idx in np.ndindex(data.shape):
                if data[idx] is gap:
                    data[idx] = base._fill(interp)
        klass = common_class(data.flat, base.cls)
        _check_mix(data.flat, klass)
        return ObjectArray(data, klass) if data.size != 1 else data.flat[0]

    def index_delete(self, interp, args: list) -> Any:
        from . import indexing as ix
        data = ix.delete(self.data, list(args))
        return ObjectArray(data, common_class(data.flat, self.cls))

    def brace_get(self, interp, args: list) -> list:
        raise MatlabError("Brace indexing is not supported for object arrays; use parentheses: objs(k).")

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        """``[a, b]`` / ``[a; b]`` with at least one object: the class's own method, or the default."""
        from .interpreter import current_interpreter
        interp = current_interpreter()
        if interp is not None:
            name = "horzcat" if axis == 1 else ("vertcat" if axis == 0 else "")
            for i in items:
                if isinstance(i, (UserObject, ObjectArray)) and i.cls is not None:
                    if name in i.cls.methods:
                        res = i.cls.call(interp, name, list(items), 1)
                    elif "cat" in i.cls.methods:
                        res = i.cls.call(interp, "cat", [np.array([[axis + 1.0]]), *items], 1)
                    else:
                        continue
                    if not res:
                        raise MatlabError(f"The concatenation method of class '{i.cls.name}' did not return "
                                          "a value.")
                    return res[0]
        return default_concat(items, axis)

    def object_equals(self, other: Any) -> bool:
        from .builtins.objfuncs import objects_equal
        if isinstance(other, UserObject):
            other = ObjectArray.of(other)
        if not isinstance(other, ObjectArray) or other.data.shape != self.data.shape:
            return False
        return all(objects_equal(a, b) for a, b in zip(self.data.flat, other.data.flat))

    def colon_range(self, start: Any, step: Any, stop: Any) -> Any:
        return object_colon(start, step, stop)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("==", "~="):
            return NotImplemented
        other = right if left is self else left
        if isinstance(other, ObjectArray):
            if other.data.shape != self.data.shape:
                raise MatlabError("Arrays have incompatible sizes for this operation.",
                                  "MATLAB:sizeDimensionsMustMatch")
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
        if self.cls is not None and self.cls.old_style:
            return [f"\t{name} object: {'-by-'.join(str(d) for d in self.data.shape)}"]
        kind = "empty " if self.data.size == 0 else ""
        if self.data.size and self.mixed():
            kinds = []
            for e in self.elements():
                if e.cls.name not in kinds:
                    kinds.append(e.cls.name)
            name = f"heterogeneous {name} ({', '.join(kinds)})"
        props = [p for p in self.cls.public_props() if not self.cls.is_constant(p)] if self.cls else []
        if not props:
            return [f"  {self._size_text()} {kind}{name} array with no properties."]
        return [f"  {self._size_text()} {kind}{name} array with properties:", ""] + [f"    {p}" for p in props]


def common_class(elements: Any, fallback: ClassInfo | None) -> ClassInfo | None:
    """The class of an array holding ``elements``: their class, or the nearest shared superclass."""
    classes: list[ClassInfo] = []
    for e in elements:
        if e.cls not in classes:
            classes.append(e.cls)
    if not classes:
        return fallback
    if len(classes) == 1:
        return classes[0]
    for cand in classes[0].lineage():
        if all(cand in c.lineage() for c in classes[1:]):
            return cand
    return fallback or classes[0]


def _compatible(a: ClassInfo | None, b: ClassInfo | None) -> bool:
    if a is None or b is None or a is b:
        return True
    return any(c in b.lineage() and "matlab.mixin.Heterogeneous" in c.bases for c in a.lineage()) \
        or a in b.lineage()


def _check_mix(elements: Any, klass: ClassInfo | None) -> None:
    """Objects of different classes share an array only below a matlab.mixin.Heterogeneous class."""
    classes: list[ClassInfo] = []
    for e in elements:
        if e.cls not in classes:
            classes.append(e.cls)
    if len(classes) < 2:
        return
    if klass is not None and all(klass in c.lineage() for c in classes) \
            and "matlab.mixin.Heterogeneous" in klass.bases:
        return
    names = ", ".join(c.name for c in classes)
    raise MatlabError(f"Objects of the classes {names} cannot be combined into one array: they have no "
                      "common superclass derived from matlab.mixin.Heterogeneous. Use a cell array "
                      "{...} to mix types.", "MATLAB:UnableToConvert")


def convert_to_class(interp, value: Any, cls: ClassInfo) -> Any:
    """Turn ``value`` into an object of ``cls`` the way MATLAB does: converter method, else constructor."""
    if isinstance(value, (UserObject, ObjectArray)):
        if value.cls is cls or (value.cls is not None and cls in value.cls.lineage()):
            return value
        if value.cls is not None and cls.name in value.cls.methods:  # a converter method named after cls
            res = value.cls.call(interp, cls.name, [value], 1)
            if res:
                return res[0]
    if cls.enums:
        return cls.construct(interp, [value], 1)[0]
    try:
        res = cls.construct(interp, [value], 1)
    except MatlabError as e:
        raise MatlabError(f"The following error occurred converting from {_class_name(value)} to "
                          f"{cls.name}: {e.message}", "MATLAB:UnableToConvert") from None
    return res[0]


def default_concat(items: list, axis: int) -> Any:
    """Built-in concatenation of objects (what ``builtin('horzcat', ...)`` does)."""
    if any(type(i) is UserObject and i._data is not None for i in items):
        from . import ops
        first = next(i for i in items if type(i) is UserObject and i._data is not None)
        return first.with_data(ops.concat([_plain(i) for i in items], axis))
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
    klass = common_class(data.flat, klass)
    _check_mix(data.flat, klass)
    return data.flat[0] if data.size == 1 else ObjectArray(data, klass)


def _class_name(v: Any) -> str:
    from .values import class_name
    return class_name(v)


def _is_superior(a: ClassInfo, b: ClassInfo) -> bool:
    """Whether class ``a`` outranks ``b`` when both appear in one operation."""
    return b.name in a.inferior or a.name in b.superior_to_me


def dominant_first(left: Any, right: Any) -> tuple:
    """The two operands, the one whose class decides which method runs first."""
    lc = left.cls if isinstance(left, (UserObject, ObjectArray)) else None
    rc = right.cls if isinstance(right, (UserObject, ObjectArray)) else None
    if lc is not None and rc is not None and lc is not rc and _is_superior(rc, lc):
        return right, left
    if lc is None and rc is not None:
        return right, left
    return left, right


def call_operator(interp, op: str, left: Any, right: Any) -> Any:
    """Dispatch a binary operator to a class method, or return NotImplemented."""
    name = OPERATOR_METHODS.get(op)
    first, second = dominant_first(left, right)
    for obj in (first, second):
        if isinstance(obj, (UserObject, ObjectArray)) and obj.cls is not None and name \
                and name in obj.cls.methods:
            res = obj.cls.call(interp, name, [left, right], 1)
            if not res:
                raise MatlabError(f"Operator method '{name}' did not return a value.")
            return res[0]
    for obj in (first, second):
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
        raise MatlabError(f"Operator '{op}' is not supported for operands of type '{cls}'.",
                          "MATLAB:UndefinedFunction")
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
        raise MatlabError(f"Operator '{op}' is not supported for operands of type '{operand.class_name()}'.",
                          "MATLAB:UndefinedFunction")
    return NotImplemented


# -- indexing overloads: subsref, subsasgn, subsindex, end, numArgumentsFromSubscript -----------------

_CHAIN = (A.Index, A.CellIndex, A.Field, A.DynField)
INDEXING_CONTEXT: dict[str, Any] = {}  # 'Statement' / 'Expression' / 'Assignment' -> value for the class


def _inside(frame, cls: ClassInfo) -> bool:
    """Code of the class itself (or a related class) uses built-in indexing on its objects."""
    owner = frame.owner_class
    return owner is not None and (owner is cls or owner.name in cls.ancestors()
                                  or cls.name in owner.ancestors())


def subs_struct(levels: list) -> Any:
    """The struct array ``S`` handed to subsref/subsasgn: fields ``type`` and ``subs``."""
    from .values import Cell, Struct, char
    data = np.empty((1, len(levels)), dtype=object)
    for k, (kind, subs) in enumerate(levels):
        data[0, k] = {"type": char(kind), "subs": char(subs) if kind == "." else Cell.from_list(list(subs))}
    return Struct(data, ["type", "subs"])


def subs_levels(s: Any) -> list:
    """The reverse of :func:`subs_struct`: [(type, subs)] from a substruct value."""
    from .values import Cell, Struct, is_char, to_str
    if not isinstance(s, Struct) or "type" not in s.fields or "subs" not in s.fields:
        raise MatlabError("The subscript argument must be a struct with the fields 'type' and 'subs' "
                          "(see substruct).", "MATLAB:subsArgNotStruc")
    levels = []
    for d in s.data.flatten(order="F"):
        kind = to_str(d["type"], "subscript type")
        subs = d["subs"]
        if kind == ".":
            levels.append((kind, to_str(subs, "field name")))
        elif kind in ("()", "{}"):
            if isinstance(subs, Cell):
                levels.append((kind, subs.values_f()))
            elif is_char(subs) and to_str(subs) == ":":
                levels.append((kind, [subs]))
            else:
                raise MatlabError("The 'subs' field of a '()' or '{}' subscript must be a cell array.",
                                  "MATLAB:subsArgNotCell")
        else:
            raise MatlabError(f"Unknown subscript type '{kind}'; use '()', '{{}}' or '.'.",
                              "MATLAB:subsTypeMustBeSquigglyOrSmooth")
    return levels


def _chain_frame(interp, value: Any, levels: list, raw: bool = True):
    """A scratch workspace and the syntax tree of ``value`` followed by the subscript ``levels``."""
    from .interpreter import Frame
    from .values import is_char
    frame = Frame(interp, interp.frame.functions, name=interp.frame.name, file=interp.frame.file)
    frame.owner_class = interp.frame.owner_class
    frame.raw_index = raw  # raw: the first level uses built-in indexing, not the class's overload
    frame.vars["__v"] = value
    node: A.Node = A.Ident("__v")
    for i, (kind, subs) in enumerate(levels):
        if kind == ".":
            node = A.Field(node, subs)
            continue
        arg_nodes: list[A.Node] = []
        for j, a in enumerate(subs):
            if is_char(a) and a.size == 1 and a.flat[0] == ":":
                arg_nodes.append(A.Colon())
            else:
                key = f"__a{i}_{j}"
                frame.vars[key] = a
                arg_nodes.append(A.Ident(key))
        node = A.Index(node, arg_nodes) if kind == "()" else A.CellIndex(node, arg_nodes)
    return frame, node


def builtin_subsref(interp, value: Any, levels: list, nargout: int) -> list:
    """Standard indexing of ``value`` by subscript levels (``builtin('subsref', ...)``)."""
    if not levels:
        return [value]
    frame, node = _chain_frame(interp, value, levels)
    return interp.eval_multi(node, frame, nargout)


def builtin_subsasgn(interp, value: Any, levels: list, rhs: Any) -> Any:
    """Standard indexed assignment into ``value`` (``builtin('subsasgn', ...)``)."""
    if not levels:
        return rhs
    frame, node = _chain_frame(interp, value, levels)
    if value is None:
        del frame.vars["__v"]
    interp.assign(node, rhs, frame)
    return frame.vars["__v"]


def _ref_value(interp, obj: Any, levels: list) -> Any:
    """One value of ``obj`` indexed by ``levels`` (through the class's subsref when it has one)."""
    if not levels:
        return obj
    cls = getattr(obj, "cls", None)
    if type(obj) in (UserObject, ObjectArray) and cls is not None and cls.overloads_ref:
        res = _subsref(interp, obj, levels, 1)
    else:
        res = builtin_subsref(interp, obj, levels, 1)
    if not res:
        raise MatlabError("Indexing produced no value.")
    return res[0]


class _EndProxy:
    """Stands for the indexed value on the interpreter's ``end`` stack until ``end`` is really used."""

    __slots__ = ("get",)

    def __init__(self, get: Callable[[], Any]):
        self.get = get

    def value(self, interp, pos: int, count: int) -> np.ndarray:
        return np.array([[float(end_of(interp, self.get(), pos, count))]])


def end_of(interp, v: Any, pos: int, count: int) -> int:
    """``end`` in index ``pos`` (0-based) of ``count`` indices of ``v``, honouring end/size methods."""
    from . import indexing as ix
    from .values import shape_of
    cls = getattr(v, "cls", None)
    if type(v) in (UserObject, ObjectArray) and cls is not None:
        if "end" in cls.methods:
            res = cls.call(interp, "end", [v, np.array([[pos + 1.0]]), np.array([[float(count)]])], 1)
            if not res:
                raise MatlabError(f"The end method of class '{cls.name}' did not return a value.")
            return int(np.real(np.asarray(_plain(res[0])).flat[0]))
        if "size" in cls.methods:
            res = cls.call(interp, "size", [v], 1)
            if res:
                dims = tuple(int(x) for x in np.real(np.asarray(res[0])).flatten())
                return ix.end_value(dims, pos, count)
        if "numel" in cls.methods and count == 1:
            res = cls.call(interp, "numel", [v], 1)
            if res:
                return int(np.real(np.asarray(res[0]).flat[0]))
    return ix.end_value(shape_of(v), pos, count)


def _levels(interp, obj: Any, nodes: list, frame) -> list:
    """Evaluate the subscripts of the syntax nodes applied to ``obj``: [(type, subs)]."""
    from .values import char, to_str
    out: list = []
    for n in nodes:
        t = type(n)
        if t is A.Field:
            out.append((".", n.name))
            continue
        if t is A.DynField:
            out.append((".", to_str(interp.eval(n.expr, frame), "dynamic field name")))
            continue
        before = list(out)
        cache: list = []

        def base(before=before, cache=cache):
            if not cache:
                cache.append(_ref_value(interp, obj, before))
            return cache[0]

        proxy = _EndProxy(base)
        count = len(n.args)
        args: list = []
        for i, a in enumerate(n.args):
            if type(a) is A.Colon:
                args.append(char(":"))
                continue
            interp.end_stack.append((proxy, i, count))
            try:
                vals = interp.eval_expanding(a, frame)
            finally:
                interp.end_stack.pop()
            args.extend(vals)
        out.append(("()" if t is A.Index else "{}", args))
    return out


def _list_length(interp, obj: Any, levels: list, context: str) -> int:
    """How many values ``obj{...}`` or ``obj.name`` gives where a comma-separated list is expected."""
    cls = obj.cls
    if "numArgumentsFromSubscript" in cls.methods:
        ctx = INDEXING_CONTEXT.get(context)
        if ctx is None:
            from .values import char
            ctx = char(context)
        res = cls.call(interp, "numArgumentsFromSubscript", [obj, subs_struct(levels), ctx], 1)
        if not res:
            raise MatlabError("numArgumentsFromSubscript did not return a value.")
        return int(np.real(np.asarray(res[0]).flat[0]))
    kind, subs = levels[0]
    if kind == "()":
        return 1
    if "numel" in cls.methods:
        res = cls.call(interp, "numel", [obj, *(subs if kind == "{}" else [])], 1)
        if res:
            return int(np.real(np.asarray(res[0]).flat[0]))
    if isinstance(obj, ObjectArray) and kind == ".":
        return int(obj.data.size)
    return 1


def _subsref(interp, obj: Any, levels: list, nargout: int) -> list:
    cls = obj.cls
    if "subsref" not in cls.methods:
        return _redefined_ref(interp, obj, levels, nargout)
    n = _list_length(interp, obj, levels, "Expression") if nargout < 0 else nargout
    return cls.call(interp, "subsref", [obj, subs_struct(levels)], n)


def _subsasgn(interp, obj: Any, levels: list, value: Any) -> Any:
    cls = obj.cls
    if "subsasgn" not in cls.methods:
        return _redefined_assign(interp, obj, levels, value)
    res = cls.call(interp, "subsasgn", [obj, subs_struct(levels), value], 1)
    if not res:
        if cls.is_handle:
            return obj
        raise MatlabError(f"The subsasgn method of class '{cls.name}' must return the object.")
    return res[0]


def _peek(interp, value: Any, lvl: A.Node, frame) -> Any:
    """The value one indexing level deeper when that is free of side effects, else None."""
    from . import indexing as ix
    from .values import Cell, Struct
    t = type(lvl)
    if t is A.Field:
        if isinstance(value, Struct):
            return value.data.flat[0].get(lvl.name) if value.data.size == 1 else None
        if type(value) is UserObject:
            entry = value.cls.prop_defs.get(lvl.name)
            if entry is None or value._listeners or ("get." + lvl.name) in value.cls.methods:
                return None
            attrs = entry[0].attrs
            if "GetAccess" in attrs or "Access" in attrs or attrs.get("Constant"):
                return None
            return value.props.get(lvl.name)
        return None
    if t is A.DynField or not all(interp._pure(a, frame) for a in lvl.args):
        return None
    try:
        if t is A.CellIndex and isinstance(value, Cell):
            picked = ix.get(value.data, interp.eval_index_args(lvl.args, value, frame))
            return picked.flat[0] if picked.size == 1 else None
        if t is A.Index and isinstance(value, Struct):
            return Struct(ix.get(value.data, interp.eval_index_args(lvl.args, value, frame)), value.fields)
        if t is A.Index and type(value) is ObjectArray and value.cls is not None \
                and not value.cls.overloads_ref:
            return value.index_get(interp, interp.eval_index_args(lvl.args, value, frame))
    except MatlabError:
        return None
    return None


def _find_overloaded(interp, node: A.Node, frame, attr: str):
    """Walk ``root.a(1).b`` from the root variable to the first object whose class overloads indexing.

    Returns (root name, nodes, k, object) or None when standard indexing applies.
    """
    nodes = []
    n = node
    while type(n) in _CHAIN:
        nodes.append(n)
        n = n.base
    if type(n) is not A.Ident:
        return None
    value = frame.lookup(n.name)
    if value is None:
        return None
    nodes.reverse()
    raw = frame.__dict__.get("raw_index", False)
    k, total = 0, len(nodes)
    while True:
        tv = type(value)
        if (tv is UserObject or tv is ObjectArray) and value.cls is not None \
                and getattr(value.cls, attr) and not (raw and k == 0) and not _inside(frame, value.cls):
            return n.name, nodes, k, value
        if k >= total - 1:
            return None
        value = _peek(interp, value, nodes[k], frame)
        if value is None:
            return None
        k += 1


def _forwarded_levels(interp, node: A.Node, frame) -> list | None:
    """Levels of ``x.(indexOp)`` when ``indexOp`` is an IndexingOperation value, else None."""
    if type(node) is not A.DynField:
        return None
    expr = node.expr
    root = expr.base if type(expr) is A.Index else expr
    if type(root) is not A.Ident:
        return None
    held = frame.lookup(root.name)
    if not (isinstance(held, IndexOp) or (isinstance(held, NativeArray)
                                          and held.kind == "matlab.indexing.IndexingOperation")):
        return None
    return index_op_levels(interp.eval(expr, frame))


def overloaded_ref(interp, node: A.Node, frame, nargout: int) -> list | None:
    """Evaluate an indexing expression through a class's subsref; None when no class takes it over."""
    levels = _forwarded_levels(interp, node, frame)
    if levels is not None:  # value.(indexOp): apply the forwarded indexing operations
        value = interp.eval(node.base, frame)
        if not levels:
            return [value]
        tmp, chain = _chain_frame(interp, value, levels, raw=False)
        return interp.eval_multi(chain, tmp, nargout)
    found = _find_overloaded(interp, node, frame, "overloads_ref")
    if found is None:
        return None
    _, nodes, k, obj = found
    return _subsref(interp, obj, _levels(interp, obj, nodes[k:], frame), nargout)


def overloaded_assign(interp, target: A.Node, value: Any, frame) -> bool:
    """Run an indexed assignment through a class's subsasgn; False when no class takes it over."""
    levels = _forwarded_levels(interp, target, frame)
    if levels is not None:  # value.(indexOp) = rhs
        current = interp.lookup_existing(target.base, frame)
        if not levels:
            new = value
        else:
            tmp, chain = _chain_frame(interp, current, levels, raw=False)
            if current is None:
                del tmp.vars["__v"]
            interp.assign(chain, value, tmp)
            new = tmp.vars["__v"]
        interp.assign(target.base, new, frame)
        return True
    found = _find_overloaded(interp, target, frame, "overloads_asgn")
    if found is None:
        return False
    root, nodes, k, obj = found
    new = _subsasgn(interp, obj, _levels(interp, obj, nodes[k:], frame), value)
    if k == 0:
        frame.set(root, new)
    elif new is not obj:
        interp.assign(nodes[k - 1], new, frame)
    return True


def object_to_builtin(interp, value: Any, kind: str) -> Any:
    """Convert an object stored into a ``kind`` array (``A(k) = obj``): its converter method or data."""
    cls = getattr(value, "cls", None)
    if type(value) in (UserObject, ObjectArray) and cls is not None:
        if kind in cls.methods:
            res = cls.call(interp, kind, [value], 1)
            if res and isinstance(res[0], np.ndarray):
                return res[0]
        if type(value) is UserObject and (value._data is not None or value._enum_value is not None):
            return _plain(value)
    if kind == "double" and getattr(value, "is_handle", False) and hasattr(value, "has_method") \
            and value.has_method("double") and type(value).__name__ in ("GraphicsHandle", "GraphicsArray"):
        res = value.call_method(interp, "double", [value], 1)  # graphics objects stored as numeric handles
        if res and isinstance(res[0], np.ndarray):
            return res[0]
    raise MatlabError(f"The following error occurred converting from {_class_name(value)} to {kind}: "
                      f"Conversion to {kind} from {_class_name(value)} is not possible.",
                      "MATLAB:UnableToConvert")


def subsindex_value(interp, v: Any) -> Any:
    """An object used as an index: its subsindex method gives zero-based positions."""
    if type(v) in (UserObject, ObjectArray) and v.cls is not None:
        if "subsindex" in v.cls.methods:
            res = v.cls.call(interp, "subsindex", [v], 1)
            if not res:
                raise MatlabError("subsindex did not return a value.")
            return np.asarray(_plain(res[0]), dtype=float) + 1.0
        if type(v) is UserObject and (v._data is not None or v._enum_value is not None):
            return _plain(v)
        raise MatlabError(f"Function 'subsindex' is not defined for values of class '{v.cls.name}'.",
                          "MATLAB:UndefinedFunction")
    return v


def index_args_overloaded(interp, args: list, base: Any, frame, shape: Any) -> list:
    """Index arguments once some class overloads indexing: end methods and objects used as indices."""
    from . import indexing as ix
    from .values import is_char
    cls = getattr(base, "cls", None)
    if cls is not None and type(base) in (UserObject, ObjectArray) and (
            "end" in cls.methods or "size" in cls.methods or "numel" in cls.methods):
        shape = _EndProxy(lambda: base)
    plain = not isinstance(base, MObject)
    count = len(args)
    out: list = []
    for i, arg in enumerate(args):
        if type(arg) is A.Colon:
            out.append(ix.COLON)
            continue
        interp.end_stack.append((shape, i, count))
        try:
            vals = interp.eval_expanding(arg, frame)
        finally:
            interp.end_stack.pop()
        for v in vals:
            if is_char(v) and v.size == 1 and v.flat[0] == ":":
                out.append(ix.COLON)
            elif plain and type(v) in (UserObject, ObjectArray):
                out.append(subsindex_value(interp, v))
            else:
                out.append(v)
    return out


def show_object(interp, name: str, value: Any) -> bool:
    """Print ``name = ...`` through the class's own display or disp method; False if it has none."""
    cls = getattr(value, "cls", None)
    if cls is None or type(value) not in (UserObject, ObjectArray):
        return False
    from .values import char
    if "display" in cls.methods:
        fn = cls.methods["display"][0]
        cls.call(interp, "display", [value, char(name)] if len(fn.params) > 1 else [value], 0)
        return True
    if "disp" in cls.methods:
        from .display import finish_layout, object_header
        interp.out(object_header(name))
        cls.call(interp, "disp", [value], 0)
        interp.out(finish_layout("\n"))
        return True
    fn = cls.base_methods.get("display")
    if fn is not None:
        fn(interp, [value, char(name)], 0)
        return True
    return False


class NativeItem(MObject):
    """A built-in object that can sit in a :class:`NativeArray` (``[a, b]``, ``a(2)``, ``numel``)."""

    kind = "object"

    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        return name == self.kind or (self.is_handle and name == "handle")

    def array_method(self, name: str) -> bool:
        return False

    def call_array_method(self, interp, name: str, args: list, nargout: int) -> list:
        raise MatlabError(f"Unrecognized method '{name}' for class '{self.kind}'.")

    def array_props(self) -> list[str]:
        return []

    def index_get(self, interp, args: list) -> Any:
        return NativeArray(_one(self), self.kind).index_get(interp, args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        return NativeArray(_one(self), self.kind).index_set(interp, args, value)

    def index_delete(self, interp, args: list) -> Any:
        return NativeArray(_one(self), self.kind).index_delete(interp, args)

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        kind = next(i.class_name() for i in items if isinstance(i, MObject))
        return NativeArray.concat_native(items, axis, kind)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("==", "~=") and isinstance(left, NativeItem) and isinstance(right, NativeItem):
            same = left is right or left.same_as(right)
            return np.array([[same if op == "==" else not same]])
        return NotImplemented

    def same_as(self, other: Any) -> bool:
        return False


def _one(item: Any) -> np.ndarray:
    data = np.empty((1, 1), dtype=object)
    data[0, 0] = item
    return data


# -- matlab.mixin.indexing.RedefinesParen / RedefinesDot / RedefinesBrace ------------------------------

class IndexOp(NativeItem):
    """One level of a matlab.indexing.IndexingOperation array: Type and Indices or Name."""

    kind = "matlab.indexing.IndexingOperation"

    def __init__(self, op: str, subs: Any):
        self.op = op  # '()', '{}' or '.'
        self.subs = subs

    def array_props(self) -> list[str]:
        return ["Type", "Indices", "Name"]

    def get_field(self, interp, name: str) -> Any:
        from .values import Cell, char
        if name == "Type":
            label = {"()": "Paren", "{}": "Brace", ".": "Dot"}[self.op]
            made = INDEXING_CONTEXT.get("Type:" + label)
            return made if made is not None else char(label)
        if name == "Indices":
            if self.op == ".":
                raise MatlabError("A dot indexing operation has a Name, not Indices.")
            return Cell.from_list(list(self.subs))
        if name == "Name":
            if self.op != ".":
                raise MatlabError("Only a dot indexing operation has a Name.")
            return char(self.subs)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def display_lines(self) -> list[str]:
        return ["  IndexingOperation with properties:", "",
                f"    Type: {({'()': 'Paren', '{}': 'Brace', '.': 'Dot'})[self.op]}"]


def index_ops(levels: list) -> Any:
    return NativeArray.wrap([IndexOp(kind, subs) for kind, subs in levels],
                            "matlab.indexing.IndexingOperation")


def index_op_levels(v: Any) -> list | None:
    """[(type, subs)] of an IndexingOperation value (used by ``obj.(indexOp)`` forwarding)."""
    if isinstance(v, IndexOp):
        return [(v.op, v.subs)]
    if isinstance(v, NativeArray) and v.kind == "matlab.indexing.IndexingOperation":
        return [(e.op, e.subs) for e in v.elements()]
    return None


_REDEFINED = {"()": ("parenReference", "parenAssign", "parenListLength"),
              ".": ("dotReference", "dotAssign", "dotListLength"),
              "{}": ("braceReference", "braceAssign", "braceListLength")}


def _redefined_ref(interp, obj: Any, levels: list, nargout: int) -> list:
    cls = obj.cls
    ref = _REDEFINED[levels[0][0]][0]
    if ref not in cls.methods or (levels[0][0] == "." and (levels[0][1] in cls.prop_defs
                                                           or levels[0][1] in cls.methods)):
        # this kind of indexing is not redefined (or names a real member): standard behaviour,
        # with the levels after it still free to reach a redefined one
        return builtin_subsref(interp, obj, levels, nargout)
    n = nargout
    if n < 0:
        length = _REDEFINED[levels[0][0]][2]
        n = 1
        if length in cls.methods:
            ctx = INDEXING_CONTEXT.get("Expression")
            res = cls.call(interp, length, [obj, index_ops(levels), ctx], 1)
            n = int(np.real(np.asarray(res[0]).flat[0])) if res else 1
    return cls.call(interp, ref, [obj, index_ops(levels)], n)


def _redefined_assign(interp, obj: Any, levels: list, value: Any) -> Any:
    cls = obj.cls
    kind = levels[0][0]
    name = _REDEFINED[kind][1]
    if name not in cls.methods or (kind == "." and (levels[0][1] in cls.prop_defs)):
        return builtin_subsasgn(interp, obj, levels, value)
    if kind == "()" and len(levels) == 1 and isinstance(value, np.ndarray) and value.shape == (0, 0) \
            and value.dtype == float and "parenDelete" in cls.methods:
        res = cls.call(interp, "parenDelete", [obj, index_ops(levels)], 1)
    else:
        res = cls.call(interp, name, [obj, index_ops(levels), value], 1)
    if not res:
        if cls.is_handle:
            return obj
        raise MatlabError(f"The {name} method of class '{cls.name}' must return the object.")
    return res[0]


class NativeArray(ObjectArray):
    """An array of built-in objects (meta.property, TestResult, ...): indexing, ``[a.Name]``, numel."""

    def __init__(self, data: np.ndarray, kind: str):
        self.data = data
        self.cls = None
        self.kind = kind

    @classmethod
    def wrap(cls, items: list, kind: str, column: bool = False) -> Any:
        """One item as itself, several (or none) as an array of the built-in class ``kind``."""
        if len(items) == 1:
            return items[0]
        data = np.empty((len(items), 1) if column else (1, len(items)), dtype=object)
        for k, item in enumerate(items):
            data.flat[k] = item
        if not items:
            data = np.empty((0, 0) if not column else (0, 1), dtype=object)
        return cls(data, kind)

    def _same(self, data: np.ndarray) -> Any:
        return data.flat[0] if data.size == 1 else type(self)(data, self.kind)

    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        if self.data.size:
            return all(e.isa_class(name) for e in self.data.flat)
        return name == self.kind

    @property
    def is_handle(self) -> bool:  # type: ignore[override]
        return bool(self.data.size) and bool(self.data.flat[0].is_handle)

    def empty_like(self) -> "NativeArray":
        return type(self)(np.empty((0, 0), dtype=object), self.kind)

    prototypes: dict[str, Any] = {}  # kind -> an object that answers for an empty array of that kind

    def _proto(self) -> Any:
        return self.data.flat[0] if self.data.size else self.prototypes.get(self.kind)

    def has_method(self, name: str) -> bool:
        proto = self._proto()
        if proto is not None and getattr(proto, "array_method", lambda n: False)(name):
            return True
        return name in _ARRAY_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        proto = self._proto()
        if proto is not None and proto.array_method(name):
            return proto.call_array_method(interp, name, args, nargout)
        if name in _ARRAY_METHODS:
            return [_ARRAY_METHODS[name](interp, args)]
        raise MatlabError(f"Unrecognized method '{name}' for class '{self.kind}'.")

    def field_list(self, interp, name: str) -> list:
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)
        return [e.get_field(interp, name) for e in self.elements()]

    def index_get(self, interp, args: list) -> Any:
        from . import indexing as ix
        return self._same(ix.get(self.data, list(args)))

    def index_peek(self, interp, args: list) -> Any:
        return self.index_get(interp, args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        from . import indexing as ix
        if isinstance(value, NativeArray):
            rhs = value.data
        elif isinstance(value, MObject) and value.class_name() == self.kind:
            rhs = np.empty((1, 1), dtype=object)
            rhs[0, 0] = value
        else:
            raise MatlabError(f"Cannot store a {_class_name(value)} in an array of {self.kind} objects.")

        def fill():
            raise MatlabError(f"An array of {self.kind} objects cannot grow with gaps.")

        return self._same(ix.set_(self.data.copy(), list(args), rhs, fill))

    def index_delete(self, interp, args: list) -> Any:
        from . import indexing as ix
        return type(self)(ix.delete(self.data, list(args)), self.kind)

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        kind = next(i.class_name() for i in items if isinstance(i, MObject))
        return cls.concat_native(items, axis, kind)

    @classmethod
    def concat_native(cls, items: list, axis: int, kind: str) -> Any:
        datas = []
        for i in items:
            if isinstance(i, NativeArray):
                if i.data.size:
                    datas.append(i.data)
            elif isinstance(i, MObject):
                one = np.empty((1, 1), dtype=object)
                one[0, 0] = i
                datas.append(one)
            elif isinstance(i, np.ndarray) and i.size == 0:
                continue
            else:
                raise MatlabError(f"Cannot concatenate a {_class_name(i)} with {kind} objects.")
        if not datas:
            return cls(np.empty((0, 0), dtype=object), kind)
        try:
            data = np.concatenate(datas, axis=axis)
        except ValueError:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch") from None
        return data.flat[0] if data.size == 1 else cls(data, kind)

    def object_equals(self, other: Any) -> bool:
        if not isinstance(other, NativeArray) or other.data.shape != self.data.shape:
            return False
        return all(a is b for a, b in zip(self.data.flat, other.data.flat))

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op not in ("==", "~="):
            return NotImplemented
        other = right if left is self else left
        if isinstance(other, NativeArray):
            if other.data.shape != self.data.shape:
                raise MatlabError("Arrays have incompatible sizes for this operation.")
            flags = [a is b for a, b in zip(self.data.flatten(order="F"), other.data.flatten(order="F"))]
        else:
            flags = [a is other for a in self.data.flatten(order="F")]
        out = np.array(flags, dtype=bool).reshape(self.data.shape, order="F")
        return out if op == "==" else ~out

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return type(self)(self.data.T.copy(), self.kind)
        return NotImplemented

    def display_lines(self) -> list[str]:
        short = self.kind.rsplit(".", 1)[-1]
        kind = "empty " if self.data.size == 0 else ""
        proto = self._proto()
        names = proto.array_props() if proto is not None and hasattr(proto, "array_props") else []
        if not names:
            return [f"  {self._size_text()} {kind}{short} array"]
        lines = [f"  {self._size_text()} {kind}{short} array with properties:", ""] + [f"    {p}" for p in names]
        extra = getattr(proto, "array_footer", None)
        if extra is not None:
            lines += extra(self)
        return lines


# -- superclass calls into built-in bases, and old-style classes ---------------------------------------

SUPER_DEFAULTS: dict[str, Callable] = {}  # method name -> default behaviour for name@BuiltinBase(...)


def super_default(*names: str):
    def deco(fn: Callable) -> Callable:
        for n in names:
            SUPER_DEFAULTS[n] = fn
        return fn
    return deco


def builtin_super_call(interp, node: A.SuperCall, frame, nargout: int, args: list) -> list:
    """``name@double(...)`` or ``name@handle(...)``: a call into a base class the engine provides."""
    parent, name = node.parent, node.name
    current = frame.lookup(name) if frame.has(name) else None
    if parent in NUMERIC_BASES:
        if isinstance(current, UserObject):  # o = o@double(v) in a constructor
            if current.cls.numeric_base is None:
                raise MatlabError(f"Class '{current.cls.name}' is not derived from '{parent}'.")
            if len(args) > 1:
                raise MatlabError(f"Too many input arguments for the {parent} constructor.")
            made = current.with_data(_to_base(interp, parent, args[0]) if args else _empty_of(parent))
            frame.set(name, made)
            return [] if nargout == 0 else [made]
        fn = interp.builtins.get(name)
        if fn is None:
            raise MatlabError(f"'{name}' is not a method of the built-in class '{parent}'.")
        if name in ("subsref", "subsasgn", "horzcat", "vertcat", "cat", "disp", "display"):
            return raw_builtin(interp, name, args, nargout)
        return call_on_data(interp, name, args, nargout) if any(
            type(a) is UserObject and a._data is not None for a in args) \
            else interp.call_builtin(name, fn, list(args), nargout)
    spec = BUILTIN_BASES.get(parent)
    if spec is None:
        raise MatlabError(f"Superclass '{parent}' was not found.")
    if isinstance(current, UserObject):
        return [] if nargout == 0 else [current]  # obj@handle(...): the base has nothing to construct
    fn = SUPER_DEFAULTS.get(name) or spec.methods.get(name)
    if fn is None:
        for b in spec.parents:
            fn = BUILTIN_BASES[b].methods.get(name)
            if fn is not None:
                break
    if fn is not None:
        return interp.call_builtin(name, fn, list(args), nargout)
    if name in interp.builtins:
        return raw_builtin(interp, name, args, nargout)
    raise MatlabError(f"No method '{name}' in superclass '{parent}'.")


def raw_builtin(interp, name: str, args: list, nargout: int) -> list:
    """Run the standard behaviour of a function for objects, skipping any class method of that name."""
    from .values import to_str
    if name == "subsref":
        if len(args) != 2:
            raise MatlabError("subsref needs a value and a subscript struct: subsref(A, S).")
        return builtin_subsref(interp, args[0], subs_levels(args[1]), nargout)
    if name == "subsasgn":
        if len(args) != 3:
            raise MatlabError("subsasgn needs a value, a subscript struct and the new value.")
        return [builtin_subsasgn(interp, args[0], subs_levels(args[1]), args[2])]
    if name in ("horzcat", "vertcat", "cat") and any(isinstance(a, (UserObject, ObjectArray)) for a in args):
        if name == "cat":
            dim = int(np.real(np.asarray(args[0]).flat[0]))
            return [default_concat(list(args[1:]), dim - 1)]
        return [default_concat(list(args), 1 if name == "horzcat" else 0)]
    if name in ("disp", "display") and args and isinstance(args[0], (UserObject, ObjectArray)):
        from .display import format_display
        lines = args[0].display_lines()
        if name == "disp":
            interp.out("\n".join(lines) + "\n")
        else:
            label = to_str(args[1]) if len(args) > 1 else "ans"
            interp.out(format_display(label, args[0]) if not lines else
                       f"{label} = \n\n" + "\n".join(lines) + "\n\n")
        return []
    if name in ("numel", "size", "length", "ndims", "isempty", "isscalar") and args \
            and type(args[0]) is UserObject and args[0]._data is None:
        pass  # the built-in shape functions already ignore class methods
    if name == "end" and len(args) == 3:
        from . import indexing as ix
        from .values import shape_of
        k = int(np.real(np.asarray(args[1]).flat[0]))
        n = int(np.real(np.asarray(args[2]).flat[0]))
        return [np.array([[float(ix.end_value(shape_of(args[0]), k - 1, n))]])]
    if name == "numArgumentsFromSubscript" and len(args) >= 2:
        obj, levels = args[0], subs_levels(args[1])
        n = 1
        if levels and levels[0][0] != "()" and isinstance(obj, ObjectArray):
            n = int(obj.data.size)
        return [np.array([[float(n)]])]
    op = next((o for o, m in OPERATOR_METHODS.items() if m == name), None)
    if op is not None and len(args) == 2 and any(isinstance(a, MObject) for a in args):
        for obj in args:
            if isinstance(obj, MObject):
                res = obj.binary_op(interp, op, args[0], args[1])
                if res is not NotImplemented:
                    return [res]
        if op in ("==", "~=") and all(isinstance(a, MObject) and a.is_handle for a in args):
            same = args[0] is args[1]
            return [np.array([[same if op == "==" else not same]])]
    fn = interp.builtins.get(name)
    if fn is None:
        raise MatlabError(f"'{name}' is not a built-in function.", "MATLAB:UndefinedFunction")
    if any(type(a) is UserObject and a._data is not None for a in args) and _works_on_data(name):
        return call_on_data(interp, name, list(args), nargout)
    return interp.call_builtin(name, fn, list(args), nargout)


def old_style_class(interp, name: str, ctor_path: str) -> ClassInfo:
    """The class behind an ``@name`` folder whose constructor is a function calling class(s, 'name')."""
    cls = interp.classes.get(name)
    if cls is not None:
        return cls
    cd = A.ClassDef(name, [], {}, [], {})
    cls = ClassInfo(cd, ctor_path, interp, {})
    cls.old_style = True
    interp.classes[name] = cls
    return cls


def adopt_old_style_parent(owner: ClassInfo, parent: ClassInfo) -> None:
    """Make ``parent`` a parent class of the old-style class ``owner``: it inherits its methods."""
    if parent in owner.parents:
        return
    owner.parents.append(parent)
    for mname, entry in parent.methods.items():
        if mname != parent.name:
            owner.methods.setdefault(mname, entry)
    owner.inferior |= parent.inferior
    owner.superior_to_me |= parent.superior_to_me
    owner.overloads_ref = owner.overloads_ref or parent.overloads_ref
    owner.overloads_asgn = owner.overloads_asgn or parent.overloads_asgn


def make_old_style(interp, s: Any, name: str, parents: list) -> Any:
    """``class(s, 'name', parent1, ...)``: turn the struct ``s`` into an object of class ``name``."""
    from .values import Struct
    frame = interp.frame
    path = frame.file or ""
    owner = frame.owner_class
    if owner is None or owner.name != name or not owner.old_style:
        if os.path.basename(os.path.dirname(path)) != "@" + name:
            raise MatlabError(f"class(s, '{name}') can only be called from the constructor {name}.m in a "
                              f"folder named @{name}.", "MATLAB:class:InvalidContext")
        owner = old_style_class(interp, name, path)
    if not isinstance(s, Struct):
        raise MatlabError("The first argument of class(s, 'name') must be a struct holding the fields "
                          "of the object.", "MATLAB:class:mustBeStruct")
    for p in parents:
        if not isinstance(p, UserObject):
            raise MatlabError("Parent arguments of class(s, 'name', parent) must be objects.")
        adopt_old_style_parent(owner, p.cls)

    def one(fields: dict) -> UserObject:
        obj = UserObject(owner)
        obj.props = {f: fields[f] for f in s.fields}
        for p in parents:
            obj.props[p.cls.name] = p
        return obj

    if s.data.size == 1:
        return one(s.data.flat[0])
    data = np.empty(s.data.shape, dtype=object)
    for idx in np.ndindex(s.data.shape):
        data[idx] = one(s.data[idx])
    return ObjectArray(data, owner)


# -- validation of property values and function arguments ----------------------------------------------

def _unable(name: str, cls: str) -> MatlabError:
    return MatlabError(f"Value must be of type {cls} or be convertible to {cls}.",
                       "MATLAB:validation:UnableToConvert")


def _convert_class(interp, value: Any, cls: str, name: str) -> Any:
    from .values import (
        INT_TYPES, Cell, FunctionHandle, MString, StringArray, Struct, char, class_name, is_char,
        saturate_int, string_data, string_result, to_numeric, to_str,
    )
    if type(value) is UserObject and value.cls.numeric_base and cls in NUMERIC_BASES:
        value = _plain(value)
    if cls in ("double", "single") or cls in INT_TYPES or cls == "logical":
        if isinstance(value, (MString, StringArray)) and cls != "logical" and interp is not None:
            value = interp.call_builtin("double", interp.builtins["double"], [value], 1)[0]
        if hasattr(value, "to_dense"):  # sparse: kept when it already has the class
            if class_name(value) == cls:
                return value
            value = value.to_dense()
        if not isinstance(value, np.ndarray):
            raise _unable(name, cls)
        a = to_numeric(value)
        if cls == "double":
            return a.astype(complex if a.dtype.kind == "c" else float)
        if cls == "single":
            return a.astype(np.complex64 if a.dtype.kind == "c" else np.float32)
        if cls == "logical":
            if is_char(value) or a.dtype.kind == "c":  # as logical('a') and logical(1i), an error
                raise _unable(name, cls)
            if a.dtype.kind == "f" and np.isnan(a).any():
                raise MatlabError("NaN's cannot be converted to logicals.", "MATLAB:nologicalnan")
            return a != 0
        return saturate_int(a, INT_TYPES[cls])
    if cls == "string":
        if isinstance(value, (MString, StringArray)):
            return value
        try:
            if interp is not None and not is_char(value):
                return interp.call_builtin("string", interp.builtins["string"], [value], 1)[0]
            return string_result(string_data(value))
        except MatlabError:
            raise _unable(name, cls) from None
    if cls == "char":
        if is_char(value):
            return value
        if isinstance(value, MString):
            return char(str(value))
        if isinstance(value, StringArray) and value.data.size == 1:
            return char(value.data.flat[0] or "")
        if isinstance(value, np.ndarray) and value.dtype.kind in "iuf" and interp is not None:
            return interp.call_builtin("char", interp.builtins["char"], [value], 1)[0]
        raise _unable(name, cls)
    checks = {"cell": Cell, "struct": Struct, "function_handle": FunctionHandle}
    if cls in checks:
        if not isinstance(value, checks[cls]):
            raise _unable(name, cls)
        return value
    if hasattr(value, "isa_class"):
        if value.isa_class(cls):
            return value
    elif class_name(value) == cls:
        return value
    klass = interp.find_class(cls) if interp is not None else None
    if isinstance(value, np.ndarray) and value.size == 0 and value.dtype.kind == "f":
        if klass is not None and not klass.numeric_base:
            return ObjectArray(np.empty(value.shape, dtype=object), klass)  # [] is an empty array of the class
        return value
    if klass is not None:
        if klass.enums and (isinstance(value, MString) or (is_char(value) and value.ndim == 2
                                                           and value.shape[0] == 1)):
            member = _enum_by_text(interp, klass, to_str(value))
            if member is not None:
                return member
        try:
            return convert_to_class(interp, value, klass)
        except MatlabError:
            raise _unable(name, cls) from None
    if interp is not None and cls in interp.builtins:  # datetime, duration, categorical, table, ...
        try:
            res = interp.call_builtin(cls, interp.builtins[cls], [value], 1)
        except MatlabError:
            raise _unable(name, cls) from None
        if res:
            return res[0]
    raise _unable(name, cls)


def _enum_by_text(interp, klass: ClassInfo, text: str) -> Any:
    """The enumeration member a text names when assigned to a property or argument of that class.

    The match is inexact: case is ignored, and a leading part of one member's name is enough.
    """
    members = klass.enum_members(interp)
    low = text.lower()
    exact = [m for n, m in members.items() if n.lower() == low]
    if exact:
        return exact[0]
    found = [m for n, m in members.items() if low and n.lower().startswith(low)]
    return found[0] if len(found) == 1 else None


def size_text(dims) -> str:
    """A size as MATLAB writes it, with ':' for a dimension of any length (1×:, 3×4)."""
    return "×".join(":" if d is None else str(d) for d in dims)


def _size_error(size: list, shape: list) -> MatlabError:
    got = size_text(shape)
    if all(d == 1 for d in size):
        return MatlabError("Value must be a scalar.", "MATLAB:validation:IncompatibleSize")
    want = size_text(size)
    if len(size) == 2 and size[0] == 1 and size[1] is None:
        msg = f"Value must be a row vector (size {want}), but its size is {got}."
    elif len(size) == 2 and size[0] is None and size[1] == 1:
        msg = f"Value must be a column vector (size {want}), but its size is {got}."
    elif None in size:
        msg = f"Value must have size {want}, where ':' allows any length, but its size is {got}."
    else:
        msg = f"Value must have size {want} (or be a scalar, which fills that size), but its size is {got}."
    return MatlabError(msg, "MATLAB:validation:IncompatibleSize")


def fit_size(value: Any, size: list) -> Any:
    """Make ``value`` fit a declared size such as (1,1), (1,:) or (:,3), as MATLAB does.

    Besides a value that already fits, MATLAB accepts and converts
      - an empty value, when the size allows an empty array (a ':' or a 0 in it): it becomes an
        empty array of that size, so [] for (1,:) is 1×0 and for (:,3) is 0×3;
      - a scalar, which is repeated to fill the size (a ':' dimension counts as 1);
      - a vector with the right number of elements in another orientation, when the size describes
        a vector: a column for (1,:) or (1,3) becomes a row, a row for (:,1) becomes a column.
    Anything else is an error that names the expected size.
    """
    from .values import MString, StringArray, shape_of
    shape = list(shape_of(value))
    size = list(size)
    while len(shape) < len(size):
        shape.append(1)
    padded = size + [1] * (len(shape) - len(size))
    if all(d is None or d == s for d, s in zip(padded, shape)):
        return value
    count = int(np.prod(shape))
    if hasattr(value, "to_dense") and hasattr(value, "m"):  # sparse: converted as full, kept sparse
        import scipy.sparse as sp
        full = fit_size(value.to_dense(), size)
        return type(value)(sp.csc_matrix(full)) if full.ndim == 2 else full
    if isinstance(value, np.ndarray):
        data = value
    elif isinstance(value, MString):
        data = np.array([[str(value)]], dtype=object)
    elif type(value) is UserObject and value._data is not None:
        data = value._data
    elif type(value) is UserObject:
        data = np.array([[value]], dtype=object)
    else:
        data = getattr(value, "data", None)
    if not isinstance(data, np.ndarray) or data.size != count:
        raise _size_error(size, shape)

    def rebuild(new: np.ndarray) -> Any:
        if isinstance(value, np.ndarray):
            return new
        if isinstance(value, MString):
            return StringArray(new)
        if type(value) is UserObject and value._data is None:
            return ObjectArray._unwrap(new, value.cls)
        return _with_data(value, new)

    if count == 0:
        if len(shape) > len(size) and any(s != 1 for s in shape[len(size):]) and 0 not in shape[:len(size)]:
            raise _size_error(size, shape)
        target = [s if d is None else d for d, s in zip(size, shape)]
        if int(np.prod(target)) != 0:
            target = [0 if d is None else d for d in size]
        if int(np.prod(target)) == 0:
            return rebuild(data.reshape(target, order="F"))
        raise _size_error(size, shape)
    if len(shape) > len(size) and any(s != 1 for s in shape[len(size):]):
        raise _size_error(size, shape)
    shape = shape[:len(size)]
    if count == 1:
        fixed = [1 if d is None else d for d in size]
        return rebuild(np.tile(data.reshape((1,) * len(fixed)), fixed))
    free = [i for i, d in enumerate(size) if d != 1]
    is_vector = sum(1 for s in shape if s != 1) == 1
    if is_vector and len(free) == 1 and size[free[0]] in (None, count):
        target = [1] * len(size)
        target[free[0]] = count
        return rebuild(data.reshape(target, order="F"))
    raise _size_error(size, shape)


def _with_data(value: Any, data: np.ndarray) -> Any:
    from .values import Cell, StringArray, Struct
    if isinstance(value, Cell):
        return Cell(data)
    if isinstance(value, StringArray):
        return StringArray(data)
    if isinstance(value, Struct):
        return Struct(data, value.fields)
    if isinstance(value, NativeArray):
        return type(value)(data, value.kind)
    if isinstance(value, ObjectArray):
        return ObjectArray(data, value.cls)
    if type(value) is UserObject and value._data is not None:
        return value.with_data(data)
    raise MatlabError("Value has the wrong size.", "MATLAB:validation:IncompatibleSize")


def validate_value(interp, attrs: dict, value: Any, name: str, functions: dict, context: str) -> Any:
    """Check (and convert) a value against declared class, size and validator functions.

    Shared by classdef property declarations and function ``arguments`` blocks.
    """
    cls = attrs.get("_class")
    if cls:
        try:
            value = _convert_class(interp, value, cls, name)
        except MatlabError as e:
            raise MatlabError(f"{context} {e.message}", e.identifier) from None
    size = attrs.get("_size")
    if size:
        try:
            value = fit_size(value, size)
        except MatlabError as e:
            raise MatlabError(f"{context} {e.message}", e.identifier) from None
    for v in attrs.get("_validators", ()):
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
