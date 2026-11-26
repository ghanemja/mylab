"""Fused evaluation of large elementwise expressions.

``y = sin(x).^2 + 3*x - x./z`` normally makes one full-size temporary array per
operation.  When the optional ``numexpr`` package is installed and the arrays
are large, the whole expression tree is handed to numexpr instead, which
evaluates it in cache-sized blocks on several threads without temporaries.

Only a conservative subset is fused, so results keep MATLAB semantics:

* operators ``+ - .* ./`` and unary minus; ``*`` and ``/`` with a literal
  number on one side; ``.^`` with a literal exponent of -1, 0, 1 or 2;
* functions that are real for every real input (``sin``, ``exp``, ``abs`` ...),
  each only after a check that numexpr computes it bit for bit like NumPy;
* operands that are real double arrays of one common shape, or scalars.

The result must be the ordinary one to the last bit.  numexpr rewrites some
operations (``x / c`` as ``x * (1/c)``, ``x.^3`` as repeated products, a
literal ``-0`` as ``0``) and uses its own exp, tan, sinh ... on some systems, so
numbers are passed as variables, other powers are left to NumPy, and each
function is compared with NumPy's on test values the first time it is needed.

Anything else (integer classes, complex numbers, implicit expansion, objects,
a variable named ``sin`` ...) takes the ordinary path.  Set ``MYLAB_FUSE=0`` or
``interp.fuse_enabled = False`` to switch fusion off.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import ast as A

from ._lazy import lazy_module

try:  # optional accelerator; found now, imported the first time a large expression is evaluated
    import importlib.util
    _ne = lazy_module("numexpr", globals(), "_ne") if importlib.util.find_spec("numexpr") is not None else None
except Exception:  # pragma: no cover - a broken installation
    _ne = None

AVAILABLE = _ne is not None
ENABLED_BY_DEFAULT = AVAILABLE and os.environ.get("MYLAB_FUSE", "1") != "0"
MIN_SIZE = int(os.environ.get("MYLAB_FUSE_MIN", "100000"))  # elements; below this NumPy is just as fast
MIN_OPS = 2             # a single operation gains too little
_F64 = np.dtype(float)

_OPS = {"+": "+", "-": "-", ".*": "*", "./": "/"}
# MATLAB name -> numexpr name; all are real-valued for every real input
_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp", "abs": "abs",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "atan": "arctan", "expm1": "expm1",
}
NOT_FUSED = object()


class Plan:
    __slots__ = ("root", "leaves", "funcs", "expr", "skip", "misses", "consts")

    def __init__(self, root, leaves, funcs, expr):
        self.root = root        # nested tuples mirroring the expression
        self.leaves = leaves    # AST nodes evaluated the ordinary way
        self.funcs = funcs      # function names that must still mean the built-in
        self.expr = expr        # numexpr source using v0, v1, ...
        self.skip = 0           # evaluations to skip after repeated small inputs
        self.misses = 0
        self.consts: dict = {}  # numbers in the expression, passed to numexpr as variables


def _is_int_literal(node) -> bool:
    return type(node) is A.Number and isinstance(node.value, float) and node.value == int(node.value) \
        and abs(node.value) <= 64


def _const(node) -> str | None:
    if type(node) is A.Number and type(node.value) is float and np.isfinite(node.value):
        return repr(node.value)
    return None


# x.^k for these k is exact in numexpr (x*x, 1/x, x, 1); other powers become repeated products
_EXACT_POWERS = (-1, 0, 1, 2)
_VERIFIED: dict[str, bool] = {}


def _probe_values() -> np.ndarray:
    rng = np.random.default_rng(12345)
    special = [0.0, -0.0, 5e-324, -5e-324, 1e-310, 2.2250738585072014e-308, 1e-300, 0.5, -0.5, 1.0, -1.0,
               np.pi / 2, np.pi, 10.0, 20.0, 50.0, 100.0, 700.0, 709.7, 710.0, -745.0, -746.0, 1e10, 1e22,
               1e300, 1.7976931348623157e308, np.inf, -np.inf, np.nan]
    return np.concatenate([special, np.negative(special), rng.uniform(-1, 1, 2000), rng.uniform(-30, 30, 2000),
                           rng.uniform(-750, 750, 1000), 10.0 ** rng.uniform(-320, 308, 1000)])


def _function_ok(name: str) -> bool:
    """Whether numexpr's version of a function agrees with NumPy's on test values."""
    ok = _VERIFIED.get(name)
    if ok is None:
        x = _probe_values()
        try:
            with np.errstate(all="ignore"):
                got = _ne.evaluate(f"{_FUNCS[name]}(x)", local_dict={"x": x}, global_dict={})
                ref = getattr(np, _FUNCS[name])(x)
            same = (got == ref) | (np.isnan(got) & np.isnan(ref))
            ok = bool(same.all() and (np.signbit(got) == np.signbit(ref))[~np.isnan(ref)].all())
        except Exception:  # noqa: BLE001 - numexpr cannot compute it: never fuse it
            ok = False
        _VERIFIED[name] = ok
    return ok


def plan(node) -> Plan | None:
    """Build a fusion plan for an expression node, or None when there is nothing to gain."""
    if _ne is None:
        return None
    leaves: list = []
    funcs: set = set()
    consts: list = []
    count = [0]

    def walk(n):
        t = type(n)
        if t is A.BinOp:
            op = n.op
            sym = _OPS.get(op)
            if sym is None:
                if op == "*" and (_const(n.left) or _const(n.right)):
                    sym, op = "*", ".*"
                elif op == "/" and _const(n.right):
                    sym, op = "/", "./"
                elif op == ".^" and _is_int_literal(n.right) and int(n.right.value) in _EXACT_POWERS:
                    count[0] += 1
                    tree, text = walk(n.left)
                    k = int(n.right.value)
                    return ("pow", tree, k), f"({text})**{k}"
            if sym is not None:
                count[0] += 1
                lt, ltext = walk(n.left)
                rt, rtext = walk(n.right)
                return ("bin", op, lt, rt), f"({ltext}{sym}{rtext})"
        elif t is A.UnaryOp and n.op == "-":
            count[0] += 1
            tree, text = walk(n.operand)
            return ("neg", tree), f"(-{text})"
        elif t is A.Index and type(n.base) is A.Ident and n.base.name in _FUNCS and len(n.args) == 1 \
                and type(n.args[0]) not in (A.Colon,):
            count[0] += 1
            funcs.add(n.base.name)
            tree, text = walk(n.args[0])
            return ("fn", n.base.name, tree, n), f"{_FUNCS[n.base.name]}({text})"
        if _const(n) is not None:
            # passed as a variable: numexpr turns x / c into x * (1 / c), which rounds differently
            consts.append(float(n.value))
            return ("const", n), f"c{len(consts) - 1}"
        leaves.append(n)
        i = len(leaves) - 1
        return ("leaf", i), f"v{i}"

    root, expr = walk(node)
    if count[0] < MIN_OPS or not leaves:
        return None
    p = Plan(root, leaves, tuple(funcs), expr)
    p.consts = {f"c{i}": c for i, c in enumerate(consts)}
    return p


def run(interp, p: Plan, frame) -> Any:
    """Evaluate a planned expression; returns NOT_FUSED when the plan cannot apply right now."""
    if p.skip:
        p.skip -= 1
        return NOT_FUSED
    for name in p.funcs:  # a variable or user function with the same name wins
        if name in frame.vars or frame.has(name) or name in frame.functions or name in interp.classes \
                or interp.find_file_function(name) is not None or interp.private_or_imported(name, frame):
            p.skip = 1 << 30
            return NOT_FUSED
    # Leaves are evaluated in order while they are double arrays of one shape or scalars: no
    # operation between such values can fail, so nothing is reordered.  At the first other value
    # the ordinary evaluation takes over, and it evaluates the remaining leaves where it meets them
    # (an error in an operation must come before the evaluation of a later operand).
    vals: list = []
    shape = None
    for n in p.leaves:
        v = interp.eval(n, frame)
        vals.append(v)
        if type(v) is not np.ndarray or v.dtype is not _F64:
            return _apply(interp, p, p.root, vals, frame)
        if v.size != 1:
            if shape is None:
                shape = v.shape
            elif v.shape != shape:
                return _apply(interp, p, p.root, vals, frame)
    if shape is not None and int(np.prod(shape)) >= MIN_SIZE:
        try:
            evaluate = _ne.evaluate
            usable = all(_function_ok(name) for name in p.funcs)
        except ImportError:  # pragma: no cover - numexpr is installed but cannot be imported
            usable = False
        if not usable:
            p.skip = 1 << 30
            return _apply(interp, p, p.root, vals, frame)
        local = {f"v{i}": (v if v.size != 1 else float(v.flat[0])) for i, v in enumerate(vals)}
        local.update(p.consts)
        with np.errstate(all="ignore"):
            out = evaluate(p.expr, local_dict=local, global_dict={})
        p.misses = 0
        return out
    p.misses += 1
    if p.misses >= 3:  # small inputs here: stop trying for a while
        p.skip = 256
        p.misses = 0
    return _apply(interp, p, p.root, vals, frame)


def _apply(interp, p: Plan, tree, vals, frame):
    """Ordinary evaluation of the plan; leaves not evaluated yet are evaluated where they occur."""
    kind = tree[0]
    if kind == "leaf":
        i = tree[1]
        if i == len(vals):  # (leaves are numbered in evaluation order)
            vals.append(interp.eval(p.leaves[i], frame))
        return vals[i]
    if kind == "const":
        return interp.eval(tree[1], frame)
    if kind == "bin":
        left = _apply(interp, p, tree[2], vals, frame)
        return interp.binop_values(tree[1], left, _apply(interp, p, tree[3], vals, frame))
    if kind == "pow":
        return interp.binop_values(".^", _apply(interp, p, tree[1], vals, frame), np.array([[float(tree[2])]]))
    if kind == "neg":
        return interp.unary_values("-", _apply(interp, p, tree[1], vals, frame))
    # kind == "fn"
    arg = _apply(interp, p, tree[2], vals, frame)
    out = interp.call_named(tree[1], [arg], 1, frame, tree[3])
    return out[0]
