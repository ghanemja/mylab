"""Fused evaluation of large elementwise expressions.

``y = sin(x).^2 + 3*x - x./z`` normally makes one full-size temporary array per
operation.  When the optional ``numexpr`` package is installed and the arrays
are large, the whole expression tree is handed to numexpr instead, which
evaluates it in cache-sized blocks on several threads without temporaries.

Only a conservative subset is fused, so results keep MATLAB semantics:

* operators ``+ - .* ./`` and unary minus; ``*`` and ``/`` with a literal
  number on one side; ``.^`` with a literal integer exponent;
* functions that are real for every real input (``sin``, ``exp``, ``abs`` ...);
* operands that are real double arrays of one common shape, or scalars.

Anything else (integer classes, complex numbers, implicit expansion, objects,
a variable named ``sin`` ...) takes the ordinary path.  Set ``MYLAB_FUSE=0`` or
``interp.fuse_enabled = False`` to switch fusion off.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import ast as A

try:  # optional accelerator
    import numexpr as _ne
except Exception:  # pragma: no cover - numexpr missing or broken
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
    __slots__ = ("root", "leaves", "funcs", "expr", "skip", "misses")

    def __init__(self, root, leaves, funcs, expr):
        self.root = root        # nested tuples mirroring the expression
        self.leaves = leaves    # AST nodes evaluated the ordinary way
        self.funcs = funcs      # function names that must still mean the built-in
        self.expr = expr        # numexpr source using v0, v1, ...
        self.skip = 0           # evaluations to skip after repeated small inputs
        self.misses = 0


def _is_int_literal(node) -> bool:
    return type(node) is A.Number and isinstance(node.value, float) and node.value == int(node.value) \
        and abs(node.value) <= 64


def _const(node) -> str | None:
    if type(node) is A.Number and type(node.value) is float and np.isfinite(node.value):
        return repr(node.value)
    return None


def plan(node) -> Plan | None:
    """Build a fusion plan for an expression node, or None when there is nothing to gain."""
    if _ne is None:
        return None
    leaves: list = []
    funcs: set = set()
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
                elif op == ".^" and _is_int_literal(n.right):
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
        c = _const(n)
        if c is not None:
            return ("const", n), c if not c.startswith("-") else f"({c})"
        leaves.append(n)
        i = len(leaves) - 1
        return ("leaf", i), f"v{i}"

    root, expr = walk(node)
    if count[0] < MIN_OPS or not leaves:
        return None
    return Plan(root, leaves, tuple(funcs), expr)


def run(interp, p: Plan, frame) -> Any:
    """Evaluate a planned expression; returns NOT_FUSED when the plan cannot apply right now."""
    if p.skip:
        p.skip -= 1
        return NOT_FUSED
    for name in p.funcs:  # a variable or user function with the same name wins
        if name in frame.vars or frame.has(name) or name in frame.functions or name in interp.classes \
                or interp.find_file_function(name) is not None:
            p.skip = 1 << 30
            return NOT_FUSED
    vals = [interp.eval(n, frame) for n in p.leaves]
    shape = None
    ok = True
    for v in vals:
        if type(v) is not np.ndarray or v.dtype is not _F64:
            ok = False
            break
        if v.size != 1:
            if shape is None:
                shape = v.shape
            elif v.shape != shape:
                ok = False
                break
    if ok and shape is not None and int(np.prod(shape)) >= MIN_SIZE:
        local = {f"v{i}": (v if v.size != 1 else float(v.flat[0])) for i, v in enumerate(vals)}
        with np.errstate(all="ignore"):
            out = _ne.evaluate(p.expr, local_dict=local, global_dict={})
        p.misses = 0
        return out
    p.misses += 1
    if p.misses >= 3:  # small inputs here: stop trying for a while
        p.skip = 256
        p.misses = 0
    return _apply(interp, p.root, vals, frame)


def _apply(interp, tree, vals, frame):
    """Ordinary evaluation of the plan from already-computed leaf values."""
    kind = tree[0]
    if kind == "leaf":
        return vals[tree[1]]
    if kind == "const":
        return interp.eval(tree[1], frame)
    if kind == "bin":
        return interp.binop_values(tree[1], _apply(interp, tree[2], vals, frame), _apply(interp, tree[3], vals, frame))
    if kind == "pow":
        return interp.binop_values(".^", _apply(interp, tree[1], vals, frame), np.array([[float(tree[2])]]))
    if kind == "neg":
        return interp.unary_values("-", _apply(interp, tree[1], vals, frame))
    # kind == "fn"
    arg = _apply(interp, tree[2], vals, frame)
    out = interp.call_named(tree[1], [arg], 1, frame, tree[3])
    return out[0]
