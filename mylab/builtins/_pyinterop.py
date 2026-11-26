"""Calling Python from MATLAB code: py.module.name(...), Python objects as MATLAB values, pyrun.

Conversions follow MATLAB's: Python float, complex and bool results become double, complex double
and logical; int becomes int64 when it fits; everything else (str, list, tuple, dict, objects)
stays a Python object that MATLAB code converts with double, char, string, cell or struct.
Going the other way, scalars become float, int, bool or complex, text becomes str, a real row
vector becomes an array.array, other arrays become NumPy arrays, a cell vector becomes a tuple
and a scalar struct or containers.Map becomes a dict.
"""

from __future__ import annotations

import builtins as _py_builtins
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, Struct, as_2d, char, is_char, string_result, to_str,
)

_INT_CLASSES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
_CONVERSIONS = {"double", "single", "logical", "char", "string", "cell", "struct", "numel", "length", "isempty",
                "cellstr", "py.len", *_INT_CLASSES}
_ARRAY_CODES = {"float64": "d", "float32": "f", "int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
                "int32": "i", "uint32": "I", "int64": "q", "uint64": "Q"}


def check_enabled(interp) -> None:
    if not getattr(interp, "python_enabled", True):
        raise MatlabError("Calling Python is turned off in this session (the host set python_enabled to false or "
                          "started mylab with MYLAB_PYTHON=0).", "MATLAB:Python:Disabled")


class PyKeywords(MObject):
    """pyargs('name', value, ...): keyword arguments for a Python call."""

    def __init__(self, pairs: dict):
        self.pairs = pairs

    def class_name(self) -> str:
        return "pyargs"

    def display_lines(self) -> list[str]:
        return ["  Python keyword arguments: " + ", ".join(self.pairs)]


class _Output:
    """Sends what Python code prints to the command window of the session."""

    def __init__(self, interp):
        import sys
        self.interp = interp
        self.outer = sys.stdout

    def write(self, s: str) -> int:
        import sys
        if s:
            inner = sys.stdout
            sys.stdout = self.outer  # the session's own output function may print to stdout
            try:
                self.interp.out(s)
            finally:
                sys.stdout = inner
        return len(s)

    def flush(self) -> None:
        pass


def _redirect(interp):
    import contextlib
    return contextlib.redirect_stdout(_Output(interp))


def py_error(e: BaseException) -> MatlabError:
    return MatlabError(f"Python Error: {type(e).__name__}: {e}", "MATLAB:Python:PyException")


def _type_name(obj: Any) -> str:
    t = type(obj)
    if t.__module__ in ("builtins", "__main__"):
        return t.__qualname__
    return f"{t.__module__}.{t.__qualname__}"


# -- MATLAB -> Python ------------------------------------------------------------------------------

def to_python(interp, v: Any) -> Any:
    from ..objects import MapObject
    if isinstance(v, PyObject):
        return v.obj
    if isinstance(v, MString):
        return str(v)
    if isinstance(v, StringArray):
        return [x for x in v.data.flatten(order="F")]
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            if v.ndim == 2 and v.shape[0] > 1:
                return ["".join(r) for r in v]
            return to_str(v)
        if v.size == 1:
            x = v.flat[0]
            if v.dtype == bool:
                return bool(x)
            if v.dtype.kind in "iu":
                return int(x)
            if v.dtype.kind == "c":
                return complex(x)
            return float(x)
        if v.ndim == 2 and v.shape[0] == 1 and v.dtype.name in _ARRAY_CODES:
            import array
            return array.array(_ARRAY_CODES[v.dtype.name], v.reshape(-1).tolist())
        return np.array(v)  # a copy: Python code must not change MATLAB values in place
    if isinstance(v, Cell):
        items = [to_python(interp, x) for x in v.data.flatten(order="F")]
        if v.data.ndim == 2 and 1 in v.data.shape or v.data.size == 0:
            return tuple(items)
        return [tuple(to_python(interp, x) for x in row) for row in v.data]
    if isinstance(v, Struct):
        if v.data.size == 1:
            return {k: to_python(interp, x) for k, x in v.data.flat[0].items()}
        return [{k: to_python(interp, x) for k, x in d.items()} for d in v.data.flatten(order="F")]
    if isinstance(v, MapObject):
        return {k: to_python(interp, x) for k, x in v.store.items()}
    if isinstance(v, FunctionHandle):
        def call(*args):
            outs = interp.call_function_value(v, [from_python(a) for a in args], 1)
            return to_python(interp, outs[0]) if outs else None
        return call
    if hasattr(v, "to_dense"):
        return v.m.copy()  # a sparse matrix stays a SciPy sparse matrix
    raise MatlabError(f"A value of class '{_class(v)}' cannot be passed to Python.",
                      "MATLAB:Python:UnsupportedInputArgumentType")


def _class(v: Any) -> str:
    from ..values import class_name
    return class_name(v)


def split_arguments(interp, args: list) -> tuple[list, dict]:
    kwargs: dict = {}
    positional = []
    for a in args:
        if isinstance(a, PyKeywords):
            kwargs.update({k: to_python(interp, x) for k, x in a.pairs.items()})
        else:
            positional.append(to_python(interp, a))
    return positional, kwargs


# -- Python -> MATLAB ------------------------------------------------------------------------------

def from_python(x: Any) -> Any:
    """The MATLAB value a Python result becomes on its own (only numbers and booleans convert)."""
    if isinstance(x, (bool, np.bool_)):
        return np.array([[bool(x)]])
    if isinstance(x, (int, np.integer)):
        if -2 ** 63 <= int(x) < 2 ** 63:
            return np.array([[int(x)]], dtype=np.int64)
        return PyObject(x)
    if isinstance(x, (float, np.floating)):
        return np.array([[float(x)]])
    if isinstance(x, (complex, np.complexfloating)):
        return np.array([[complex(x)]])
    if isinstance(x, MObject) or isinstance(x, (Cell, Struct, MString, StringArray, FunctionHandle)):
        return x
    return PyObject(x)


def _numeric(obj: Any, dtype) -> np.ndarray:
    if isinstance(obj, str):
        raise MatlabError("A Python str cannot be converted to a number; use char or string, then str2double.")
    try:
        if hasattr(obj, "toarray") and not isinstance(obj, np.ndarray):
            obj = obj.toarray()
        a = np.array(obj) if not isinstance(obj, np.ndarray) else obj
        if a.dtype == object:
            a = np.array([[_numeric(x, dtype).item() for x in row] if isinstance(row, (list, tuple)) else
                          _numeric(row, dtype).item() for row in obj])
        if a.dtype.kind == "c" and np.dtype(dtype).kind == "f":
            a = a.astype(np.complex64 if dtype is np.float32 else complex)
        elif np.dtype(dtype).kind in "iu" and a.dtype.kind == "f":
            from ..values import saturate_int
            a = saturate_int(a, dtype)
        else:
            a = a.astype(dtype)
    except (TypeError, ValueError) as e:
        raise MatlabError(f"The Python {_type_name(obj)} cannot be converted to a numeric array ({e}).") from None
    return as_2d(a)


class PyObject(MObject):
    """A Python value held by MATLAB code."""

    def __init__(self, obj: Any):
        self.obj = obj

    def class_name(self) -> str:
        return "py." + _type_name(self.obj)

    def isa_class(self, name: str) -> bool:
        if not name.startswith("py."):
            return False
        want = name[3:]
        for t in type(self.obj).__mro__:
            full = t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"
            if want == full:
                return True
        return False

    def matlab_shape(self) -> tuple:
        if isinstance(self.obj, (list, tuple, str, bytes, dict, set, frozenset, range)):
            return (1, len(self.obj))
        return (1, 1)

    # methods and attributes --------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        if name in _CONVERSIONS:
            return True
        try:
            return callable(getattr(self.obj, name))
        except AttributeError:
            return False

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        check_enabled(interp)
        rest = list(args)
        for k, a in enumerate(rest):
            if a is self:
                del rest[k]
                break
        if name in _CONVERSIONS and not (rest and hasattr(self.obj, name)):
            return [self.convert(interp, name)]
        return [call(interp, getattr(self.obj, name), rest)]

    def convert(self, interp, name: str) -> Any:
        obj = self.obj
        if name == "double":
            return _numeric(obj, np.float64)
        if name == "single":
            return _numeric(obj, np.float32)
        if name in _INT_CLASSES:
            return _numeric(obj, np.dtype(name).type)
        if name == "logical":
            return _numeric(obj, np.float64) != 0
        if name in ("numel", "length", "py.len"):
            try:
                return np.array([[float(len(obj))]])
            except TypeError:
                return np.array([[1.0]])
        if name == "isempty":
            try:
                return np.array([[len(obj) == 0]])
            except TypeError:
                return np.array([[obj is None]])
        if name == "char":
            if isinstance(obj, (bytes, bytearray)):
                return char(bytes(obj).decode("latin-1"))
            if isinstance(obj, str):
                return char(obj)
            return char(str(obj))
        if name == "string":
            if isinstance(obj, (list, tuple)):
                data = np.empty((1, len(obj)), dtype=object)
                for k, x in enumerate(obj):
                    data[0, k] = None if x is None else str(x)
                return string_result(data)
            if isinstance(obj, (bytes, bytearray)):
                return MString(bytes(obj).decode("latin-1"))
            return MString(str(obj))
        if name in ("cell", "cellstr"):
            if isinstance(obj, dict) or isinstance(obj, (str, bytes)) or not hasattr(obj, "__iter__"):
                raise MatlabError(f"A Python {_type_name(obj)} cannot be converted to a cell array; "
                                  "cell works on lists, tuples and other sequences.")
            items = list(obj)
            if name == "cellstr":
                return Cell.from_list([char(str(x)) for x in items])
            return Cell.from_list([from_python(x) for x in items])
        if name == "struct":
            if not isinstance(obj, dict):
                raise MatlabError(f"A Python {_type_name(obj)} cannot be converted to a struct; struct works "
                                  "on dict objects.")
            out = {}
            for k, x in obj.items():
                key = str(k)
                if not key.isidentifier():
                    raise MatlabError(f"The dict key '{key}' is not a valid field name.")
                out[key] = from_python(x)
            return Struct.scalar(out) if out else Struct()
        raise MatlabError(f"Cannot convert a Python {_type_name(obj)} with {name}.")

    def get_field(self, interp, name: str) -> Any:
        check_enabled(interp)
        try:
            attr = getattr(self.obj, name)
        except AttributeError:
            raise MatlabError(f"Unrecognized method, property, or field '{name}' for class "
                              f"'{self.class_name()}'.") from None
        if callable(attr) and not isinstance(attr, type):
            return call(interp, attr, [])  # obj.method without parentheses calls it, as in MATLAB
        return from_python(attr)

    def set_field(self, interp, name: str, value: Any) -> "PyObject":
        check_enabled(interp)
        try:
            setattr(self.obj, name, to_python(interp, value))
        except Exception as e:  # any Python failure is reported as a MATLAB error
            raise py_error(e) from None
        return self

    # calls and indexing --------------------------------------------------------------------------
    def call_object(self, interp, args: list, nargout: int) -> list:
        if callable(self.obj):
            return [call(interp, self.obj, args)]
        return [self.index_get(interp, args)]

    def _position(self, a: Any) -> Any:
        """A MATLAB subscript as a Python key: numbers count from one, text is a dict key."""
        if isinstance(a, PyObject):
            return a.obj
        if isinstance(a, MString) or is_char(a):
            return to_str(a)
        if isinstance(a, np.ndarray) and a.size == 1:
            if isinstance(self.obj, dict):
                x = a.flat[0]
                return int(x) if float(x) == int(x) else float(x)
            k = int(a.flat[0])
            if k < 1 or k > len(self.obj):
                raise MatlabError(f"Index {k} is out of range for a Python {_type_name(self.obj)} with "
                                  f"{len(self.obj)} elements (indices count from 1).")
            return k - 1
        raise MatlabError("Use one index or key to index a Python object.")

    def index_get(self, interp, args: list) -> Any:
        if len(args) != 1 or not hasattr(self.obj, "__getitem__"):
            raise MatlabError(f"A Python {_type_name(self.obj)} cannot be indexed with parentheses.")
        a = args[0]
        if isinstance(a, np.ndarray) and a.size != 1 and not isinstance(self.obj, dict):
            picked = [self.obj[self._position(np.array([[x]]))] for x in a.flatten(order="F")]
            try:
                return PyObject(type(self.obj)(picked))
            except TypeError:
                return PyObject(picked)
        try:
            item = self.obj[self._position(a)]
        except (KeyError, IndexError, TypeError) as e:
            raise py_error(e) from None
        if isinstance(self.obj, (list, tuple)):
            return PyObject(type(self.obj)([item]))  # parentheses keep the container, as for a cell array
        return from_python(item)

    def brace_get(self, interp, args: list) -> list:
        if len(args) != 1 or not hasattr(self.obj, "__getitem__"):
            raise MatlabError(f"A Python {_type_name(self.obj)} cannot be indexed with braces.")
        a = args[0]
        try:
            if isinstance(a, np.ndarray) and a.size != 1 and not isinstance(self.obj, dict):
                return [from_python(self.obj[self._position(np.array([[x]]))]) for x in a.flatten(order="F")]
            return [from_python(self.obj[self._position(a)])]
        except (KeyError, IndexError, TypeError) as e:
            raise py_error(e) from None

    def brace_set(self, interp, args: list, value: Any) -> Any:
        if len(args) != 1 or not hasattr(self.obj, "__setitem__"):
            raise MatlabError(f"Elements of a Python {_type_name(self.obj)} cannot be assigned.")
        a = args[0]
        new = to_python(interp, value)
        try:
            if isinstance(self.obj, list) and isinstance(a, np.ndarray) and a.size == 1 \
                    and int(a.flat[0]) == len(self.obj) + 1:
                self.obj.append(new)  # assigning one past the end grows the list
            else:
                self.obj[self._position(a)] = new
        except (KeyError, IndexError, TypeError) as e:
            raise py_error(e) from None
        return self

    def index_set(self, interp, args: list, value: Any) -> Any:
        return self.brace_set(interp, args, value)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        import operator
        table = {"==": operator.eq, "~=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
                 ">=": operator.ge, "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
                 ".*": operator.mul, "./": operator.truediv}
        fn = table.get(op)
        if fn is None:
            return NotImplemented
        try:
            return from_python(fn(to_python(interp, left), to_python(interp, right)))
        except MatlabError:
            return NotImplemented
        except Exception as e:  # any Python failure is reported as a MATLAB error
            raise py_error(e) from None

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, PyObject) and bool(self.obj == other.obj)

    # display --------------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        try:
            shown = repr(self.obj) if not isinstance(self.obj, str) else self.obj
        except Exception:  # a broken __repr__ must not break the display
            shown = f"<{_type_name(self.obj)} object>"
        lines = shown.split("\n")
        return [f"  Python {_type_name(self.obj)} with no properties.", ""] + ["    " + ln for ln in lines]

    def summary_text(self, in_cell: bool) -> str:
        t = f"1×{self.matlab_shape()[1]} py.{_type_name(self.obj)}"
        return t if in_cell else f"[{t}]"


def call(interp, target: Any, args: list) -> Any:
    check_enabled(interp)
    positional, kwargs = split_arguments(interp, args)
    try:
        with _redirect(interp):
            return from_python(target(*positional, **kwargs))
    except MatlabError:
        raise
    except Exception as e:  # any Python failure is reported as a MATLAB error
        raise py_error(e) from None


def resolve(dotted: str) -> Any:
    """The Python object named by 'py.a.b.c': a built-in, or the longest importable module then attributes."""
    import importlib
    parts = dotted.split(".")[1:]
    if not parts or not all(parts):
        raise MatlabError("Name the Python function or module after 'py.', for example py.math.sqrt(2).")
    if hasattr(_py_builtins, parts[0]):
        obj = getattr(_py_builtins, parts[0])
        rest = parts[1:]
    else:
        obj = None
        rest = parts
        for k in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:k]))
                rest = parts[k:]
                break
            except ImportError:
                continue
            except Exception as e:  # a module that fails while importing
                raise py_error(e) from None
        if obj is None:
            raise MatlabError(f"Unable to resolve the name '{dotted}': Python has no built-in or module named "
                              f"'{parts[0]}'.", "MATLAB:undefinedVarOrClass")
    for name in rest:
        try:
            obj = getattr(obj, name)
        except AttributeError:
            raise MatlabError(f"Unable to resolve the name '{dotted}': '{name}' was not found in Python.",
                              "MATLAB:undefinedVarOrClass") from None
    return obj


def dispatch(interp, dotted: str, args: list, nargout: int) -> Any:
    """py.name(...) — call a Python function or class, or read a module attribute."""
    check_enabled(interp)
    target = resolve(dotted)
    if callable(target):
        return call(interp, target, args)
    if args:
        raise MatlabError(f"'{dotted}' is not a function, so it cannot be called with arguments.")
    return from_python(target)


def namespace(interp) -> dict:
    ns = interp.state.get("py_namespace")
    if ns is None:
        ns = interp.state["py_namespace"] = {"__name__": "__main__", "__builtins__": _py_builtins}
    return ns


def _names(v: Any) -> list[str]:
    if isinstance(v, MString) or (is_char(v) and v.shape[0] <= 1):
        return [to_str(v)]
    if isinstance(v, StringArray):
        return [x for x in v.data.flatten(order="F")]
    if isinstance(v, Cell):
        return [to_str(x) for x in v.values_f()]
    raise MatlabError("The output variable names must be text.")


def _code_text(v: Any) -> str:
    if isinstance(v, StringArray):
        return "\n".join("" if x is None else x for x in v.data.flatten(order="F"))
    if isinstance(v, Cell):
        return "\n".join(to_str(x) for x in v.values_f())
    return to_str(v, "the Python code")


def _split_run_args(interp, rest: list, fn: str) -> tuple[list[str], dict]:
    outvars: list[str] = []
    if len(rest) % 2:
        outvars = _names(rest[0])
        rest = rest[1:]
    inputs = {}
    for k in range(0, len(rest), 2):
        key = to_str(rest[k], "the variable name")
        if not key.isidentifier():
            raise MatlabError(f"{fn}: '{key}' is not a valid Python variable name.")
        inputs[key] = to_python(interp, rest[k + 1])
    return outvars, inputs


def _collect(ns: dict, outvars: list[str], nargout: int, fn: str) -> Any:
    if nargout > len(outvars):
        raise MatlabError(f"{fn}: {nargout} outputs were asked for but {len(outvars)} variable names were given.")
    vals = []
    for name in outvars[:max(nargout, 1)] if outvars else []:
        if name not in ns:
            raise MatlabError(f"{fn}: the Python code did not define a variable named '{name}'.")
        vals.append(from_python(ns[name]))
    if not vals:
        return None
    return tuple(vals) if len(vals) > 1 else vals[0]


def pyrun(interp, args: list, nargout: int) -> Any:
    check_enabled(interp)
    if not args:
        raise MatlabError("pyrun needs Python code, for example pyrun(\"x = 2 + 3\", \"x\").")
    code = _code_text(args[0])
    outvars, inputs = _split_run_args(interp, list(args[1:]), "pyrun")
    ns = namespace(interp)
    ns.update(inputs)
    try:
        with _redirect(interp):
            exec(compile(code, "<pyrun>", "exec"), ns)  # noqa: S102 - running the user's Python is the feature
    except Exception as e:  # any Python failure is reported as a MATLAB error
        raise py_error(e) from None
    return _collect(ns, outvars, nargout, "pyrun")


def pyrunfile(interp, args: list, nargout: int) -> Any:
    import os
    import shlex
    import sys
    check_enabled(interp)
    if not args:
        raise MatlabError("pyrunfile needs a file name, for example pyrunfile(\"script.py\", \"result\").")
    words = shlex.split(to_str(args[0], "the file name"))
    if not words:
        raise MatlabError("pyrunfile: the file name is empty.")
    path = words[0]
    if not os.path.isfile(path):
        found = next((os.path.join(d, path) for d in getattr(interp, "path", [])
                      if os.path.isfile(os.path.join(d, path))), None)
        if found is None:
            raise MatlabError(f"pyrunfile: the file '{path}' was not found.")
        path = found
    outvars, inputs = _split_run_args(interp, list(args[1:]), "pyrunfile")
    ns = {"__name__": "__main__", "__file__": os.path.abspath(path), "__builtins__": _py_builtins}
    ns.update(inputs)
    old_argv = sys.argv
    sys.argv = [path, *words[1:]]
    try:
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        with _redirect(interp):
            exec(compile(source, path, "exec"), ns)  # noqa: S102 - running the user's Python is the feature
    except SystemExit:
        pass
    except Exception as e:  # any Python failure is reported as a MATLAB error
        raise py_error(e) from None
    finally:
        sys.argv = old_argv
    return _collect(ns, outvars, nargout, "pyrunfile")


class PyEnvironment(MObject):
    """pyenv: the Python that mylab itself runs on."""

    def class_name(self) -> str:
        return "matlab.pyclient.PythonEnvironment"

    def _values(self) -> dict:
        import os
        import sys
        import sysconfig
        lib = sysconfig.get_config_var("LDLIBRARY") or ""
        libdir = sysconfig.get_config_var("LIBDIR") or ""
        return {"Version": MString(f"{sys.version_info[0]}.{sys.version_info[1]}"),
                "Executable": MString(sys.executable or ""),
                "Library": MString(os.path.join(libdir, lib) if lib else ""),
                "Home": MString(sys.prefix), "Status": MString("Loaded"), "ExecutionMode": MString("InProcess"),
                "ProcessID": MString(str(os.getpid())),
                "ProcessName": MString(os.path.basename(sys.executable or "python"))}

    def get_field(self, interp, name: str) -> Any:
        vals = self._values()
        if name not in vals:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")
        return vals[name]

    def set_field(self, interp, name: str, value: Any) -> "PyEnvironment":
        raise MatlabError("The Python environment cannot be changed: mylab uses the Python it runs on.")

    def display_lines(self) -> list[str]:
        vals = self._values()
        w = max(len(k) for k in vals)
        return ["  PythonEnvironment with properties:", ""] + [f"    {k:>{w}}: \"{v}\"" for k, v in vals.items()]
