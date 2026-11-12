"""Loop compiler: runs simple numeric loops as plain Python instead of interpreting them.

Most of the time in loop-heavy scripts goes to interpreter overhead: every
scalar is a 1x1 array and every statement walks the syntax tree. For loops
that only do scalar arithmetic and element reads/writes on numeric arrays,
this module generates an equivalent Python function that works on raw floats,
which is typically 20-60x faster.

Safety model
------------
* A loop is compiled only if every statement in its body is one of:
  ``x = <expr>;``, ``A(i) = <expr>;``, ``A(i, j) = <expr>;``, ``if/elseif/else``,
  ``break``, ``continue`` - with expressions built from real scalars, element
  reads of double arrays, arithmetic, comparisons, logical operators and a
  fixed list of side-effect-free math functions.
* Variable kinds (double scalar, logical scalar, double array and its
  orientation) are checked on entry; the compiled code is specialised for them.
* Anything unusual at run time (an index out of range or not an integer, a
  result that would be complex, division by zero, NaN used as a condition, a
  Python error of any kind) makes the compiled code *bail out*: the variables
  are written back and the interpreter resumes at exactly the statement that
  bailed, so the user sees the normal result or the normal error message.
  Bail-outs can only happen while evaluating an expression, before the
  statement has changed anything, so resuming is exact.

Set the environment variable ``MYLAB_JIT=0`` (or ``interp.jit_enabled = False``)
to turn the compiler off.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from . import ast as A
from .errors import BreakSignal, ContinueSignal
from .values import FunctionHandle

_F64 = np.dtype(float)
_UNSET = object()
MIN_ITERATIONS = 4
MAX_BAILS = 3


class Bail(Exception):
    """Raised inside compiled code to hand control back to the interpreter."""


class Reject(Exception):
    """The loop cannot be compiled."""


# -- run-time helpers used by generated code ---------------------------------------------------

def _ix(i, n):
    k = int(i)
    if k != i or k < 1 or k > n:
        raise Bail
    return k - 1


def _lin(i, n, rows):
    k = int(i)
    if k != i or k < 1 or k > n:
        raise Bail
    k -= 1
    return (k % rows, k // rows)


def _tr(x):
    if x != x:
        raise Bail  # NaN cannot be converted to logical: let the interpreter report it
    return x != 0.0


def _pow(a, b):
    if a < 0.0 and b != int(b):
        raise Bail  # complex result
    r = a ** b
    if type(r) is not float:
        raise Bail
    return r


def _round(x):
    if x != x or x in (math.inf, -math.inf):
        return x
    return float(math.floor(abs(x) + 0.5)) * (1.0 if x >= 0 else -1.0)


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else (0.0 if x == 0 else x))


def _min2(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


def _max2(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def _mod(a, b):
    from .builtins.mathfns import _mod_scalar
    r = _mod_scalar(a, b)
    if r is None:
        raise Bail
    return float(r)


def _rem(a, b):
    from .builtins.mathfns import _rem_scalar
    r = _rem_scalar(a, b)
    if r is None:
        raise Bail
    return float(r)


def _floor(x):
    return float(math.floor(x))


def _ceil(x):
    return float(math.ceil(x))


def _fix(x):
    return float(math.trunc(x))


_HELPERS = {
    "_ix": _ix, "_lin": _lin, "_tr": _tr, "_pow": _pow, "_round": _round, "_sign": _sign,
    "_min2": _min2, "_max2": _max2, "_mod": _mod, "_rem": _rem, "_floor": _floor, "_ceil": _ceil,
    "_fix": _fix, "_math": math, "_UNSET": _UNSET, "_Bail": Bail, "_inf": math.inf, "_nan": math.nan,
}

# MATLAB function -> (number of arguments, Python expression template, result type)
_FUNCS = {
    "sin": (1, "_math.sin({0})"), "cos": (1, "_math.cos({0})"), "tan": (1, "_math.tan({0})"),
    "atan": (1, "_math.atan({0})"), "sinh": (1, "_math.sinh({0})"), "cosh": (1, "_math.cosh({0})"),
    "tanh": (1, "_math.tanh({0})"), "exp": (1, "_math.exp({0})"), "sqrt": (1, "_math.sqrt({0})"),
    "log": (1, "_math.log({0})"), "log2": (1, "_math.log2({0})"), "log10": (1, "_math.log10({0})"),
    "abs": (1, "abs({0})"), "floor": (1, "_floor({0})"), "ceil": (1, "_ceil({0})"), "fix": (1, "_fix({0})"),
    "round": (1, "_round({0})"), "sign": (1, "_sign({0})"), "mod": (2, "_mod({0}, {1})"),
    "rem": (2, "_rem({0}, {1})"), "min": (2, "_min2({0}, {1})"), "max": (2, "_max2({0}, {1})"),
    "atan2": (2, "_math.atan2({0}, {1})"), "hypot": (2, "_math.hypot({0}, {1})"),
}
_CONSTANTS = {"pi": ("_math.pi", "d"), "Inf": ("_inf", "d"), "inf": ("_inf", "d"), "NaN": ("_nan", "d"),
              "nan": ("_nan", "d"), "eps": (repr(sys.float_info.epsilon), "d"), "true": ("True", "b"),
              "false": ("False", "b")}
_ARITH = {"+": "+", "-": "-", "*": "*", ".*": "*", "/": "/", "./": "/"}
_COMPARE = {"==": "==", "~=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


# -- static analysis ---------------------------------------------------------------------------

def _collect(body: list, info: dict) -> None:
    """Names used by the loop body; raises Reject for unsupported statements."""
    for stmt in body:
        t = type(stmt)
        if t is A.Assign:
            if stmt.show or len(stmt.targets) != 1:
                raise Reject
            target = stmt.targets[0]
            if type(target) is A.Ident:
                info["scalar_writes"].add(target.name)
            elif type(target) is A.Index and type(target.base) is A.Ident and 1 <= len(target.args) <= 2:
                info["array_writes"].add(target.base.name)
                info["arrays"].add(target.base.name)
                for a in target.args:
                    _collect_expr(a, info)
            else:
                raise Reject
            _collect_expr(stmt.value, info)
        elif t is A.If:
            for cond, clause in stmt.clauses:
                _collect_expr(cond, info)
                _collect(clause, info)
            if stmt.orelse is not None:
                _collect(stmt.orelse, info)
        elif t is A.Break or t is A.Continue:
            pass
        else:
            raise Reject


def _collect_expr(node, info: dict) -> None:
    t = type(node)
    if t is A.Number:
        if isinstance(node.value, complex):
            raise Reject
    elif t is A.Ident:
        info["scalar_reads"].add(node.name)
    elif t is A.BinOp:
        if node.op not in _ARITH and node.op not in _COMPARE and node.op not in ("^", ".^", "&&", "||", "&", "|"):
            raise Reject
        _collect_expr(node.left, info)
        _collect_expr(node.right, info)
    elif t is A.UnaryOp:
        _collect_expr(node.operand, info)
    elif t is A.Index and type(node.base) is A.Ident:
        info["calls"].add((node.base.name, len(node.args)))
        for a in node.args:
            if type(a) is A.Ident and node.base.name in ("numel", "length", "size"):
                info["arrays"].add(a.name)
            else:
                _collect_expr(a, info)
    elif t is A.End:
        pass
    else:
        raise Reject


def _static(stmt) -> dict | None:
    cached = stmt.__dict__.get("_jit_static")
    if cached is not None:
        return cached or None
    info = {"scalar_reads": set(), "scalar_writes": set(), "arrays": set(), "array_writes": set(),
            "calls": set(), "bails": 0, "compiled": {}}
    try:
        if type(stmt) is A.While:
            _collect_expr(stmt.cond, info)
        _collect(stmt.body, info)
    except Reject:
        stmt._jit_static = False
        return None
    stmt._jit_static = info
    return info


# -- code generation ---------------------------------------------------------------------------

class _Gen:
    def __init__(self, kinds: dict, scalars: set, arrays: set, funcs_ok, writes: set):
        self.kinds = kinds          # name -> entry kind: 'd', 'b', 'r', 'c', 'm' or '-'
        self.writes = writes        # scalar names assigned somewhere in the loop
        self.scalars = scalars
        self.arrays = arrays
        self.funcs_ok = funcs_ok
        self.types: dict[str, set] = {}
        self.final: dict[str, str] = {}
        self.lines: list[str] = []
        self.paths: list[tuple] = []
        self.inline = None          # callable(name, args, ctx, gen) -> (code, type) or None
        self.consts: dict[str, tuple] = {}   # captured scalar values inside an inlined function
        self.blocked: frozenset = frozenset()

    def var_type(self, name: str) -> str:
        if name in self.final:
            return self.final[name]
        k = self.kinds.get(name, "-")
        return k if k in "db" else "d"

    def as_float(self, code: str, typ: str) -> str:
        return f"float({code})" if typ == "b" else code

    def as_truth(self, code: str, typ: str) -> str:
        return code if typ == "b" else f"_tr({code})"

    def expr(self, node, ctx=None) -> tuple[str, str]:
        t = type(node)
        if t is A.Number:
            v = float(node.value)
            if v != v:
                return "_nan", "d"
            if v in (math.inf, -math.inf):
                return ("_inf" if v > 0 else "(-_inf)"), "d"
            return repr(v), "d"
        if t is A.Ident:
            name = node.name
            if name in self.consts:
                return self.consts[name]
            if name in self.arrays or name in self.blocked:
                raise Reject  # whole-array use
            if self.kinds.get(name, "-") in ("d", "b") or name in self.writes:
                return f"v_{name}", self.var_type(name)
            if name in _CONSTANTS and self.funcs_ok(name):
                return _CONSTANTS[name]
            raise Reject  # not a variable: a function call without parentheses, or undefined
        if t is A.End:
            if ctx is None:
                raise Reject
            return ctx, "d"
        if t is A.UnaryOp:
            code, typ = self.expr(node.operand, ctx)
            if node.op == "-":
                return f"(-{self.as_float(code, typ)})", "d"
            if node.op == "+":
                return self.as_float(code, typ), "d"
            return f"(not {self.as_truth(code, typ)})", "b"
        if t is A.BinOp:
            op = node.op
            a, ta = self.expr(node.left, ctx)
            b, tb = self.expr(node.right, ctx)
            if op in _ARITH:
                return f"({self.as_float(a, ta)} {_ARITH[op]} {self.as_float(b, tb)})", "d"
            if op in ("^", ".^"):
                right = node.right
                if type(right) is A.Number and not isinstance(right.value, complex) \
                        and float(right.value) == int(right.value) and 0 <= right.value <= 64:
                    return f"({self.as_float(a, ta)} ** {float(right.value)!r})", "d"
                return f"_pow({self.as_float(a, ta)}, {self.as_float(b, tb)})", "d"
            if op in _COMPARE:
                return f"({a} {_COMPARE[op]} {b})", "b"
            joiner = "and" if op in ("&&", "&") else "or"
            return f"({self.as_truth(a, ta)} {joiner} {self.as_truth(b, tb)})", "b"
        if t is A.Index and type(node.base) is A.Ident:
            name = node.base.name
            if name in self.arrays:
                if name in ("numel", "length", "size"):
                    raise Reject
                return f"a_{name}.item({self.index(name, node.args)})", "d"
            if self.inline is not None and name not in self.writes and name not in self.consts:
                inlined = self.inline(name, node.args, ctx, self)
                if inlined is not None:
                    return inlined
            if self.kinds.get(name, "-") != "-" or name in self.writes or name in self.consts \
                    or name in self.blocked:
                raise Reject  # indexing into a scalar variable
            if name in ("numel", "length", "size") and node.args and type(node.args[0]) is A.Ident \
                    and node.args[0].name in self.arrays and self.funcs_ok(name):
                arr = node.args[0].name
                if name == "numel" and len(node.args) == 1:
                    return f"float(n_{arr})", "d"
                if name == "length" and len(node.args) == 1:
                    return f"float(max(r_{arr}, c_{arr}))", "d"
                if name == "size" and len(node.args) == 2 and type(node.args[1]) is A.Number \
                        and node.args[1].value in (1, 2):
                    return f"float({'r' if node.args[1].value == 1 else 'c'}_{arr})", "d"
                raise Reject
            spec = _FUNCS.get(name)
            if spec is None or spec[0] != len(node.args) or not self.funcs_ok(name):
                raise Reject
            args = [self.as_float(*self.expr(a, ctx)) for a in node.args]
            return spec[1].format(*args), "d"
        raise Reject

    def index(self, name: str, args: list) -> str:
        kind = self.kinds[name]
        if len(args) == 1:
            i = self.as_float(*self.expr(args[0], f"float(n_{name})"))
            if kind in "rc":
                return f"_ix({i}, n_{name})"
            return f"_lin({i}, n_{name}, r_{name})"
        i = self.as_float(*self.expr(args[0], f"float(r_{name})"))
        j = self.as_float(*self.expr(args[1], f"float(c_{name})"))
        return f"(_ix({i}, r_{name}), _ix({j}, c_{name}))"

    def store_index(self, name: str, args: list) -> str:
        kind = self.kinds[name]
        if len(args) == 1 and kind in "rc":
            i = self.as_float(*self.expr(args[0], f"float(n_{name})"))
            k = f"_ix({i}, n_{name})"
            return f"0, {k}" if kind == "r" else f"{k}, 0"
        return self.index(name, args)

    def emit(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)

    def block(self, body: list, indent: int, path: tuple) -> None:
        if not body:
            self.emit(indent, "pass")
        for k, stmt in enumerate(body):
            here = path + (k,)
            t = type(stmt)
            if t is A.Break:
                self.emit(indent, "break")
                continue
            if t is A.Continue:
                self.emit(indent, "continue")
                continue
            self.paths.append(here)
            self.emit(indent, f"_p = {len(self.paths) - 1}")
            if t is A.Assign:
                code, typ = self.expr(stmt.value)
                target = stmt.targets[0]
                if type(target) is A.Ident:
                    name = target.name
                    if name in self.arrays or self.kinds.get(name, "-") not in "-db":
                        raise Reject
                    self.types.setdefault(name, set()).add(typ)
                    self.emit(indent, f"v_{name} = {code}")
                else:
                    name = target.base.name
                    if self.kinds.get(name) not in ("r", "c", "m"):
                        raise Reject
                    self.emit(indent, f"a_{name}[{self.store_index(name, target.args)}] = {self.as_float(code, typ)}")
            else:  # If
                for c, (cond, clause) in enumerate(stmt.clauses):
                    code, typ = self.expr(cond)
                    self.emit(indent, f"{'if' if c == 0 else 'elif'} {self.as_truth(code, typ)}:")
                    self.block(clause, indent + 1, here + (c,))
                if stmt.orelse is not None:
                    self.emit(indent, "else:")
                    self.block(stmt.orelse, indent + 1, here + (len(stmt.clauses),))


def _kind(value: Any) -> str:
    if value is None:
        return "-"
    if type(value) is FunctionHandle:
        if not value.is_anonymous:
            return "x"
        # anonymous function: identified by its body and the scalar values it captured
        caught = tuple(sorted((k, v.item()) for k, v in value.closure.items()
                              if type(v) is np.ndarray and v.shape == (1, 1) and v.dtype.kind in "fb"))
        return f"h:{id(value.body)}:{caught!r}"
    if type(value) is not np.ndarray or value.ndim != 2:
        return "x"
    if value.shape == (1, 1):
        if value.dtype is _F64:
            return "d"
        return "b" if value.dtype == bool else "x"
    if value.dtype is not _F64 or value.size == 0:
        return "x"
    return "r" if value.shape[0] == 1 else ("c" if value.shape[1] == 1 else "m")


def _compile(stmt, info: dict, kinds: dict, interp, frame, loop_var: str | None):
    writes = set(info["scalar_writes"]) | ({loop_var} if loop_var else set())
    # Names indexed with parentheses are arrays if they are array variables on entry, else functions.
    arrays = set(info["arrays"]) | {n for n, _ in info["calls"] if kinds.get(n) in ("r", "c", "m")}
    if arrays & (info["scalar_reads"] | writes):
        raise Reject  # a name used both as a whole value and as an indexed array
    scalars = info["scalar_reads"] | writes
    for name in arrays:
        if kinds.get(name) not in ("r", "c", "m"):
            raise Reject
    for name in scalars:
        if kinds.get(name, "-") not in "-db":
            raise Reject

    shadow_cache: dict[str, bool] = {}

    def funcs_ok(name: str) -> bool:
        ok = shadow_cache.get(name)
        if ok is None:
            ok = not (name in frame.functions or name in interp.classes
                      or interp.find_file_function(name) is not None)
            shadow_cache[name] = ok
        return ok

    def inline_source(name: str):
        """(parameter names, body expression, captured values) of a function simple enough to inline."""
        kind = kinds.get(name, "-")
        if kind.startswith("h:"):
            handle = frame.vars.get(name)
            if type(handle) is not FunctionHandle or not handle.is_anonymous:
                return None
            return list(handle.params), handle.body, handle.closure
        if not kind.startswith("f:"):
            return None
        fn = frame.functions.get(name)
        if type(fn) is not A.Function or fn.subfunctions or fn.arguments or len(fn.outputs) != 1 \
                or len(fn.body) != 1:
            return None
        stmt0 = fn.body[0]
        if type(stmt0) is not A.Assign or stmt0.show or len(stmt0.targets) != 1 \
                or type(stmt0.targets[0]) is not A.Ident or stmt0.targets[0].name != fn.outputs[0]:
            return None
        return list(fn.params), stmt0.value, {}

    def make_inline(depth: int):
        def inline(name, args, ctx, outer):
            if depth > 0:
                return None  # calls inside an inlined body must be built-in math functions
            src = inline_source(name)
            if src is None:
                return None
            params, body, closure = src
            if len(params) != len(args) or "varargin" in params or "~" in params \
                    or len(set(params)) != len(params):
                raise Reject
            child = _Gen({p: "d" for p in params}, set(params), set(), funcs_ok, set())
            consts, blocked = {}, set()
            for k, v in closure.items():
                if k in params:
                    continue
                if type(v) is np.ndarray and v.shape == (1, 1) and v.dtype.kind in "fb":
                    x = v.item()
                    if v.dtype.kind == "b":
                        consts[k] = (repr(bool(x)), "b")
                    elif x != x:
                        consts[k] = ("_nan", "d")
                    elif x in (math.inf, -math.inf):
                        consts[k] = ("_inf" if x > 0 else "(-_inf)", "d")
                    else:
                        consts[k] = (f"({float(x)!r})", "d")
                else:
                    blocked.add(k)
            child.consts, child.blocked = consts, frozenset(blocked)
            child.inline = make_inline(depth + 1)
            code, typ = child.expr(body)
            actual = ", ".join(outer.as_float(*outer.expr(a, ctx)) for a in args)
            return f"(lambda {', '.join('v_' + p for p in params)}: {code})({actual})", typ
        return inline

    for name, _n in info["calls"]:
        if name in arrays or (name not in writes and inline_source(name) is not None):
            continue
        if kinds.get(name, "-") != "-" or name in writes or (
                name not in _FUNCS and name not in ("numel", "length", "size")):
            raise Reject

    gen = None
    final: dict[str, str] = {}
    for _attempt in range(3):  # iterate until the inferred variable types are stable
        gen = _Gen(kinds, scalars, arrays, funcs_ok, writes)
        gen.final = dict(final)
        gen.inline = make_inline(0)
        is_for = loop_var is not None
        if is_for:
            gen.types.setdefault(loop_var, set()).add("d")
            gen.emit(2, "for _i in range(_it0, _n):")
            gen.emit(3, f"v_{loop_var} = _start + _step * _i")
            gen.block(stmt.body, 3, ())
        else:
            gen.emit(2, "while True:")
            gen.emit(3, "_p = -1")
            code, typ = gen.expr(stmt.cond)
            gen.emit(3, f"if not {gen.as_truth(code, typ)}:")
            gen.emit(4, "break")
            gen.block(stmt.body, 3, ())
        new_final = {}
        for name, types in gen.types.items():
            entry = kinds.get(name, "-")
            all_types = set(types) | ({entry} if entry in "db" else set())
            if len(all_types) != 1:
                raise Reject
            new_final[name] = next(iter(all_types))
        if new_final == final:
            break
        final = new_final
    else:
        raise Reject

    names = sorted(n for n in scalars if kinds.get(n, "-") in ("d", "b") or n in writes)
    arr_names = sorted(arrays)
    params = ["_it0", "_n", "_start", "_step"] + [f"v_{n}" for n in names]
    for n in arr_names:
        params += [f"a_{n}", f"n_{n}", f"r_{n}", f"c_{n}"]
    values = "(" + "".join(f"v_{n}, " for n in names) + ")"
    src = [f"def _loop({', '.join(params)}):", "    _p = 0", "    _i = _it0", "    try:"]
    src += gen.lines
    src += ["    except BaseException as _e:",
            f"        return (1, _i, _p, {values}, _e)",
            f"    return (0, _i, 0, {values}, None)"]
    namespace = dict(_HELPERS)
    exec(compile("\n".join(src), "<mylab-jit>", "exec"), namespace)  # noqa: S102 - code generated above
    return namespace["_loop"], names, arr_names, gen.paths, final


# -- running ----------------------------------------------------------------------------------

def _prepare(interp, stmt, frame, loop_var):
    """Compiled function and call arguments for this loop in its current state, or None."""
    info = _static(stmt)
    if info is None or info["bails"] >= MAX_BAILS:
        return None
    names = info["scalar_reads"] | info["scalar_writes"] | info["arrays"] | {n for n, _ in info["calls"]}
    if loop_var:
        names = names | {loop_var}
    if frame.globals and any(n in frame.globals for n in names):
        return None
    fvars = frame.vars
    kinds = {n: _kind(fvars.get(n)) for n in names}
    for n, _nargs in info["calls"]:  # local functions are part of the signature too
        if kinds[n] == "-":
            fn = frame.functions.get(n)
            if type(fn) is A.Function:
                kinds[n] = f"f:{id(fn)}"
                info.setdefault("keep", {})[id(fn)] = fn
        elif kinds[n].startswith("h:"):
            body = fvars[n].body  # keep the body alive so its id() is never reused
            info.setdefault("keep", {})[id(body)] = body
    if loop_var:
        kinds[loop_var] = "d" if kinds[loop_var] in "-db" else kinds[loop_var]
    sig = tuple(sorted(kinds.items()))
    compiled = info["compiled"].get(sig, _UNSET)
    if compiled is _UNSET:
        try:
            compiled = _compile(stmt, info, kinds, interp, frame, loop_var)
        except (Reject, KeyError, SyntaxError, RecursionError):
            compiled = None
        info["compiled"][sig] = compiled
    if compiled is None:
        return None
    fn, scalar_names, arr_names, paths, final = compiled
    args = []
    for n in scalar_names:
        v = fvars.get(n)
        args.append(_UNSET if v is None else v.item())
    for n in arr_names:
        arr = fvars[n]
        if n in info["array_writes"] and (not arr.flags.writeable or arr.base is not None
                                          or sys.getrefcount(arr) > 3):
            arr = arr.copy()  # copy-on-write: this workspace gets its own array before modifying it
            fvars[n] = arr
        args += [arr, arr.size, arr.shape[0], arr.shape[1]]
    return fn, args, scalar_names, paths, final, info


def _write_back(frame, names, values, final) -> None:
    fvars = frame.vars
    for n, v in zip(names, values):
        if v is _UNSET:
            continue
        if final.get(n) == "b" or type(v) is bool:
            fvars[n] = np.array([[bool(v)]])
        else:
            fvars[n] = np.array([[float(v)]])


def _exec_from(interp, body: list, path: tuple, frame) -> None:
    """Run ``body`` starting at the statement ``path`` points to (inside nested ifs if needed)."""
    k = path[0]
    if len(path) > 1:
        stmt = body[k]
        c = path[1]
        clause = stmt.clauses[c][1] if c < len(stmt.clauses) else stmt.orelse
        _exec_from(interp, clause, path[2:], frame)
        k += 1
    for stmt in body[k:]:
        interp.exec_stmt(stmt, frame)


def run_for(interp, stmt, frame) -> bool:
    """Try to run a ``for k = a:s:b`` loop compiled. Returns False if the interpreter must run it."""
    rng = stmt.iterable
    try:
        start = interp.eval(rng.start, frame)
        stop = interp.eval(rng.stop, frame)
        step = interp.eval(rng.step, frame) if rng.step is not None else None
    except Exception:  # noqa: BLE001 - let the interpreter evaluate it again and report the error
        return False
    for v in (start, stop, step):
        if v is not None and (type(v) is not np.ndarray or v.shape != (1, 1) or v.dtype is not _F64):
            return False
    a, b = start.item(), stop.item()
    s = 1.0 if step is None else step.item()
    if s == 0 or a != a or b != b or s != s or abs(a) == math.inf or abs(b) == math.inf:
        return False
    span = (b - a) / s
    if span < 0:
        return False
    n = int(math.floor(span + 1e-10 * max(1.0, abs(span)))) + 1
    if n < MIN_ITERATIONS:
        return False
    name = stmt.var.name
    prepared = _prepare(interp, stmt, frame, name)
    if prepared is None:
        return False
    fn, args, names, paths, final, info = prepared
    status, i, p, values, exc = fn(0, n, a, s, *args)
    _write_back(frame, names, values, final)
    if status == 0:
        return True
    if not isinstance(exc, Exception):
        raise exc  # KeyboardInterrupt and the like
    # Bail-out: finish iteration i in the interpreter from statement p, then the remaining iterations.
    info["bails"] += 1
    try:
        try:
            _exec_from(interp, stmt.body, paths[p], frame)
        except ContinueSignal:
            pass
        for j in range(i + 1, n):
            frame.set(name, np.array([[a + s * j]]))
            try:
                interp.exec_block(stmt.body, frame)
            except ContinueSignal:
                continue
    except BreakSignal:
        pass
    return True


def run_while(interp, stmt, frame) -> bool:
    """Try to run a ``while`` loop compiled. Returns False if the interpreter must run it."""
    prepared = _prepare(interp, stmt, frame, None)
    if prepared is None:
        return False
    fn, args, names, paths, final, info = prepared
    status, _i, p, values, exc = fn(0, 0, 0.0, 0.0, *args)
    _write_back(frame, names, values, final)
    if status == 0:
        return True
    if not isinstance(exc, Exception):
        raise exc
    info["bails"] += 1
    if p >= 0:  # bailed inside the body: finish this iteration first
        try:
            _exec_from(interp, stmt.body, paths[p], frame)
        except BreakSignal:
            return True
        except ContinueSignal:
            pass
    return False  # the interpreter's own while loop carries on from the current state
