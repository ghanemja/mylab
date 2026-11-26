"""Problem-based optimisation: variables, expressions, constraints, problems and solve.

An expression array is a tree of operations on optimisation variables. Each node can be
evaluated numerically and, when it is a polynomial of degree at most two, turned into
coefficient arrays, which is how linear and quadratic problems reach the matrix solvers."""

from __future__ import annotations

import itertools

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, MString, Struct, char, empty_double
from ._registry import builtin
from ._util import fnum, is_text, nargs, text
from . import optim_more as om

_IDS = itertools.count(1)


def _prod(shape) -> int:
    return int(np.prod(shape)) if len(shape) else 1


class VarInfo:
    """An optimisation variable array (shared by every expression that mentions it)."""

    def __init__(self, name, shape, vtype, lb, ub, index_names=None):
        self.id = next(_IDS)
        self.name, self.shape, self.type = name, tuple(shape), vtype
        self.size = _prod(shape)
        self.lb = np.broadcast_to(lb, self.shape).astype(float).copy()
        self.ub = np.broadcast_to(ub, self.shape).astype(float).copy()
        self.index_names = index_names or [None] * len(self.shape)

    def element_name(self, k: int) -> str:
        if self.size == 1:
            return self.name
        sub = np.unravel_index(k, self.shape, order="F")
        if sum(1 for d in self.shape if d != 1) <= 1:
            dim = next((i for i, d in enumerate(self.shape) if d != 1), 0)
            names = self.index_names[dim] if dim < len(self.index_names) else None
            return f"{self.name}('{names[k]}')" if names else f"{self.name}({k + 1})"
        return f"{self.name}({', '.join(str(i + 1) for i in sub)})"


# -- polynomial form ------------------------------------------------------------------------------

class Poly:
    """Polynomial of degree <= 2 for m expression elements (rows, column-major order).

    c: (m,), L[var]: (m, nv), Q[(va, vb)]: (m, na, nb) with va.id <= vb.id."""

    __slots__ = ("c", "L", "Q")

    def __init__(self, c, L=None, Q=None):
        self.c = np.asarray(c, dtype=float)
        self.L = L or {}
        self.Q = Q or {}

    @property
    def m(self):
        return self.c.size

    def degree(self) -> int:
        if any(np.any(v) for v in self.Q.values()):
            return 2
        if any(np.any(v) for v in self.L.values()):
            return 1
        return 0

    def rows(self, idx) -> "Poly":
        idx = np.asarray(idx, dtype=int).reshape(-1)
        return Poly(self.c[idx], {k: v[idx] for k, v in self.L.items()}, {k: v[idx] for k, v in self.Q.items()})

    def scaled(self, a) -> "Poly":
        a = np.asarray(a, dtype=float).reshape(-1)
        return Poly(self.c * a, {k: v * a[:, None] for k, v in self.L.items()},
                    {k: v * a[:, None, None] for k, v in self.Q.items()})

    def plus(self, other: "Poly", sign=1.0) -> "Poly":
        L = dict(self.L)
        for k, v in other.L.items():
            L[k] = L[k] + sign * v if k in L else sign * v
        Q = dict(self.Q)
        for k, v in other.Q.items():
            Q[k] = Q[k] + sign * v if k in Q else sign * v
        return Poly(self.c + sign * other.c, L, Q)

    def times(self, other: "Poly"):
        """Element-wise product (same m), or None when the degree would exceed two."""
        if self.degree() + other.degree() > 2:
            return None
        out = Poly(self.c * other.c)
        for p, q in ((self, other), (other, self)):
            for k, v in q.L.items():
                out.L[k] = out.L.get(k, 0) + v * p.c[:, None]
            for k, v in q.Q.items():
                out.Q[k] = out.Q.get(k, 0) + v * p.c[:, None, None]
        for ka, va in self.L.items():
            for kb, vb in other.L.items():
                _add_quad(out.Q, ka, kb, va[:, :, None] * vb[:, None, :])
        return out

    def linear_map(self, fn, m_out: int) -> "Poly":
        """Apply a linear map on rows; ``fn`` takes an array with rows first and keeps trailing axes."""
        return Poly(fn(self.c).reshape(m_out), {k: fn(v).reshape(m_out, v.shape[1]) for k, v in self.L.items()},
                    {k: fn(v).reshape(m_out, v.shape[1], v.shape[2]) for k, v in self.Q.items()})


def _add_quad(Q: dict, ka: VarInfo, kb: VarInfo, block):
    if ka.id > kb.id:
        ka, kb, block = kb, ka, np.swapaxes(block, 1, 2)
    Q[(ka, kb)] = Q[(ka, kb)] + block if (ka, kb) in Q else block


def _bcast(sa, sb, what="Arrays"):
    """Broadcast two shapes the MATLAB way: (shape, flat indices into a, flat indices into b)."""
    try:
        so = np.broadcast_shapes(sa, sb)
    except ValueError:
        raise MatlabError(f"{what} have incompatible sizes for this operation.") from None
    ia = np.broadcast_to(np.arange(_prod(sa)).reshape(sa, order="F"), so).reshape(-1, order="F")
    ib = np.broadcast_to(np.arange(_prod(sb)).reshape(sb, order="F"), so).reshape(-1, order="F")
    return tuple(so), ia, ib


# -- expression nodes ---------------------------------------------------------------------------------

class Node:
    shape: tuple = (1, 1)
    _poly = False

    def ev(self, vals):
        raise NotImplementedError

    def _pl(self):
        return None

    def pl(self):
        if self._poly is False:
            self._poly = self._pl()
        return self._poly

    def variables(self, acc: dict):
        for c in getattr(self, "kids", ()):
            c.variables(acc)

    def txt(self) -> str:
        return "(...)"

    @property
    def m(self):
        return _prod(self.shape)


class NConst(Node):
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)
        if self.a.ndim < 2:
            self.a = self.a.reshape(1, -1) if self.a.ndim == 1 else self.a.reshape(1, 1)
        self.shape = self.a.shape

    def ev(self, vals):
        return self.a

    def _pl(self):
        return Poly(self.a.reshape(-1, order="F"))

    def txt(self):
        return f"{float(self.a.flat[0]):g}" if self.a.size == 1 else f"[{'x'.join(map(str, self.shape))} double]"


class NVar(Node):
    def __init__(self, info: VarInfo, idx=None):
        self.info = info
        self.idx = np.arange(info.size).reshape(info.shape, order="F") if idx is None else np.asarray(idx, dtype=int)
        self.shape = self.idx.shape

    def ev(self, vals):
        if self.info not in vals:
            raise MatlabError(f"No value was given for the variable '{self.info.name}'.")
        return np.asarray(vals[self.info]).reshape(-1, order="F")[self.idx.reshape(-1, order="F")].reshape(
            self.shape, order="F")

    def _pl(self):
        m = self.m
        L = np.zeros((m, self.info.size))
        L[np.arange(m), self.idx.reshape(-1, order="F")] = 1.0
        return Poly(np.zeros(m), {self.info: L})

    def variables(self, acc):
        acc[self.info.id] = self.info

    def txt(self):
        if self.idx.size == self.info.size:
            return self.info.name
        if self.idx.size == 1:
            return self.info.element_name(int(self.idx.flat[0]))
        return f"{self.info.name}(...)"


class NGather(Node):
    """Elements picked from the stacked elements of several children (index, reshape, concat...)."""

    def __init__(self, kids, idx, label="index"):
        self.kids = list(kids)
        self.idx = np.asarray(idx, dtype=int)
        self.shape = self.idx.shape
        self.label = label

    def ev(self, vals):
        flat = np.concatenate([np.asarray(k.ev(vals), dtype=float).reshape(-1, order="F") for k in self.kids])
        return flat[self.idx.reshape(-1, order="F")].reshape(self.shape, order="F")

    def _pl(self):
        polys = [k.pl() for k in self.kids]
        if any(p is None for p in polys):
            return None
        idx = self.idx.reshape(-1, order="F")
        m = idx.size
        offs = np.cumsum([0] + [p.m for p in polys])
        out = Poly(np.zeros(m))
        for p, o in zip(polys, offs[:-1]):
            sel = (idx >= o) & (idx < o + p.m)
            if not sel.any():
                continue
            src = idx[sel] - o
            out.c[sel] = p.c[src]
            for k, v in p.L.items():
                if k not in out.L:
                    out.L[k] = np.zeros((m, v.shape[1]))
                out.L[k][sel] = v[src]
            for k, v in p.Q.items():
                if k not in out.Q:
                    out.Q[k] = np.zeros((m,) + v.shape[1:])
                out.Q[k][sel] = v[src]
        return out

    def txt(self):
        if self.label == "index" and len(self.kids) == 1:
            return f"{self.kids[0].txt()}(...)"
        return "[" + ", ".join(k.txt() for k in self.kids) + "]"


class NReduce(Node):
    """sum, cumsum or diff along an axis (linear maps)."""

    def __init__(self, kid, kind, axis):
        self.kids = [kid]
        self.kind, self.axis = kind, axis
        s = list(kid.shape)
        if kind == "sum":
            s[axis] = 1
        elif kind == "diff":
            s[axis] = max(s[axis] - 1, 0)
        self.shape = tuple(s)

    def _apply(self, a, nd):
        """a has shape kid.shape + trailing."""
        if self.kind == "sum":
            return np.sum(a, axis=self.axis, keepdims=True)
        if self.kind == "cumsum":
            return np.cumsum(a, axis=self.axis)
        return np.diff(a, axis=self.axis)

    def ev(self, vals):
        return self._apply(np.asarray(self.kids[0].ev(vals), dtype=float), 0)

    def _pl(self):
        p = self.kids[0].pl()
        if p is None:
            return None
        ks = self.kids[0].shape

        def fn(v):
            tail = v.shape[1:]
            return self._apply(v.reshape(ks + tail, order="F"), 0).reshape((-1,) + tail, order="F")

        return p.linear_map(fn, self.m)

    def txt(self):
        return f"{self.kind}({self.kids[0].txt()})"


class NAdd(Node):
    def __init__(self, a, b, sign=1.0):
        self.kids = [a, b]
        self.sign = sign
        self.shape, self.ia, self.ib = _bcast(a.shape, b.shape)

    def ev(self, vals):
        return self.kids[0].ev(vals) + self.sign * self.kids[1].ev(vals)

    def _pl(self):
        pa, pb = self.kids[0].pl(), self.kids[1].pl()
        if pa is None or pb is None:
            return None
        return pa.rows(self.ia).plus(pb.rows(self.ib), self.sign)

    def txt(self):
        return f"{self.kids[0].txt()} {'+' if self.sign > 0 else '-'} {self.kids[1].txt()}"


class NMul(Node):
    """Element-wise product."""

    def __init__(self, a, b):
        self.kids = [a, b]
        self.shape, self.ia, self.ib = _bcast(a.shape, b.shape)

    def ev(self, vals):
        return self.kids[0].ev(vals) * self.kids[1].ev(vals)

    def _pl(self):
        pa, pb = self.kids[0].pl(), self.kids[1].pl()
        if pa is None or pb is None:
            return None
        return pa.rows(self.ia).times(pb.rows(self.ib))

    def txt(self):
        return f"({self.kids[0].txt()}) .* ({self.kids[1].txt()})"


class NMatMul(Node):
    def __init__(self, a, b):
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise MatlabError("Incorrect dimensions for matrix multiplication of optimization expressions; "
                              "use .* for element-wise multiplication.")
        self.kids = [a, b]
        self.shape = (a.shape[0], b.shape[1])

    def ev(self, vals):
        return np.asarray(self.kids[0].ev(vals)) @ np.asarray(self.kids[1].ev(vals))

    def _pl(self):
        pa, pb = self.kids[0].pl(), self.kids[1].pl()
        if pa is None or pb is None or pa.degree() + pb.degree() > 2:
            return None
        p, q = self.kids[0].shape
        r = self.kids[1].shape[1]
        m = p * r
        ca, cb = pa.c.reshape((p, q), order="F"), pb.c.reshape((q, r), order="F")
        out = Poly((ca @ cb).reshape(-1, order="F"))

        def a3(v):
            return v.reshape((p, q) + v.shape[1:], order="F")

        def b3(v):
            return v.reshape((q, r) + v.shape[1:], order="F")

        for k, v in pb.L.items():
            out.L[k] = out.L.get(k, 0) + np.einsum("ij,jkn->ikn", ca, b3(v)).reshape((m, -1), order="F")
        for k, v in pa.L.items():
            out.L[k] = out.L.get(k, 0) + np.einsum("ijn,jk->ikn", a3(v), cb).reshape((m, -1), order="F")
        for k, v in pb.Q.items():
            _add_quad(out.Q, k[0], k[1], np.einsum("ij,jkab->ikab", ca, b3(v)).reshape((m,) + v.shape[1:], order="F"))
        for k, v in pa.Q.items():
            _add_quad(out.Q, k[0], k[1], np.einsum("ijab,jk->ikab", a3(v), cb).reshape((m,) + v.shape[1:], order="F"))
        for ka, va in pa.L.items():
            for kb, vb in pb.L.items():
                blk = np.einsum("ija,jkb->ikab", a3(va), b3(vb)).reshape((m, va.shape[1], vb.shape[1]), order="F")
                _add_quad(out.Q, ka, kb, blk)
        return out

    def txt(self):
        return f"({self.kids[0].txt()}) * ({self.kids[1].txt()})"


class NDiv(Node):
    def __init__(self, a, b):
        self.kids = [a, b]
        self.shape, self.ia, self.ib = _bcast(a.shape, b.shape)

    def ev(self, vals):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.kids[0].ev(vals) / self.kids[1].ev(vals)

    def _pl(self):
        pa, pb = self.kids[0].pl(), self.kids[1].pl()
        if pa is None or pb is None or pb.degree() > 0:
            return None
        with np.errstate(divide="ignore"):
            return pa.rows(self.ia).scaled(1.0 / pb.c[self.ib])

    def txt(self):
        return f"({self.kids[0].txt()}) ./ ({self.kids[1].txt()})"


class NPow(Node):
    def __init__(self, a, b):
        self.kids = [a, b]
        self.shape, self.ia, self.ib = _bcast(a.shape, b.shape)

    def ev(self, vals):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self.kids[0].ev(vals), dtype=float) ** np.asarray(self.kids[1].ev(vals), dtype=float)

    def _pl(self):
        pa, pb = self.kids[0].pl(), self.kids[1].pl()
        if pa is None or pb is None or pb.degree() > 0:
            return None
        e = pb.c[self.ib]
        base = pa.rows(self.ia)
        if base.degree() == 0:
            return Poly(base.c ** e)
        if np.all(e == 1):
            return base
        if np.all(e == 0):
            return Poly(np.ones(base.m))
        if np.all(e == 2):
            return base.times(base)
        return None

    def txt(self):
        return f"({self.kids[0].txt()}).^{self.kids[1].txt()}"


_FUNCS = {
    "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin,
    "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh,
    "cosh": np.cosh, "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "abs": np.abs, "sign": np.sign, "sec": lambda v: 1 / np.cos(v), "csc": lambda v: 1 / np.sin(v),
    "cot": lambda v: 1 / np.tan(v), "expm1": np.expm1, "log1p": np.log1p,
}


class NFunc(Node):
    def __init__(self, name, a):
        self.kids = [a]
        self.name = name
        self.shape = a.shape

    def ev(self, vals):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _FUNCS[self.name](np.asarray(self.kids[0].ev(vals), dtype=float))

    def _pl(self):
        p = self.kids[0].pl()
        if p is not None and p.degree() == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                return Poly(_FUNCS[self.name](p.c))
        return None

    def txt(self):
        return f"{self.name}({self.kids[0].txt()})"


class NProd(Node):
    def __init__(self, a, axis):
        self.kids = [a]
        self.axis = axis
        s = list(a.shape)
        s[axis] = 1
        self.shape = tuple(s)

    def ev(self, vals):
        return np.prod(np.asarray(self.kids[0].ev(vals), dtype=float), axis=self.axis, keepdims=True)

    def _pl(self):
        p = self.kids[0].pl()
        if p is not None and p.degree() == 0:
            return Poly(np.prod(p.c.reshape(self.kids[0].shape, order="F"), axis=self.axis,
                                keepdims=True).reshape(-1, order="F"))
        if self.kids[0].shape[self.axis] == 1:
            return p
        return None

    def txt(self):
        return f"prod({self.kids[0].txt()})"


class NCall(Node):
    """A MATLAB function applied to expressions (fcn2optimexpr)."""

    def __init__(self, interp, fun, args, k, nout, shape):
        self.interp, self.fun, self.args, self.k, self.nout = interp, fun, args, k, nout
        self.kids = [a for a in args if isinstance(a, Node)]
        self.shape = tuple(shape)

    def ev(self, vals):
        a = [np.asarray(x.ev(vals), dtype=float) if isinstance(x, Node) else x for x in self.args]
        res = self.interp.call_function_value(self.fun, a, self.nout)
        if len(res) <= self.k:
            raise MatlabError("fcn2optimexpr: the function returned fewer outputs than requested.")
        v = np.real(fnum(res[self.k])).astype(float)
        if v.size != self.m:
            raise MatlabError(f"fcn2optimexpr: the function returned {v.size} values where "
                              f"{'x'.join(map(str, self.shape))} were expected (see 'OutputSize').")
        return v.reshape(self.shape, order="F")

    def txt(self):
        return "fcn(" + ", ".join(a.txt() if isinstance(a, Node) else "data" for a in self.args) + ")"


# -- the MATLAB-level objects ------------------------------------------------------------------------------

def _node(v) -> Node:
    if isinstance(v, OptimExpr):
        return v.node
    if isinstance(v, (OptimConstraint, OptimProblem)):
        raise MatlabError("This operation is not supported for optimization constraints or problems.")
    if isinstance(v, (Struct, Cell)) or is_text(v):
        raise MatlabError("Optimization expressions can only be combined with numeric values.")
    a = fnum(v)
    if np.iscomplexobj(a) and np.any(np.imag(a) != 0):
        raise MatlabError("Optimization expressions must be real.")
    return NConst(np.real(a).astype(float))


def _plain(a):
    if isinstance(a, OptimExpr):
        raise MatlabError("An optimization expression cannot be used as an index.")
    return a


def _poly_text(p: Poly, row: int) -> str:
    """Readable form of one polynomial element."""
    terms = []
    for (va, vb), Qm in sorted(p.Q.items(), key=lambda kv: (kv[0][0].name, kv[0][1].name)):
        blk = Qm[row]
        if va is vb:
            blk = blk + blk.T - np.diag(np.diag(blk))
            blk = np.triu(blk)
        for i, j in zip(*np.nonzero(blk)):
            coef = blk[i, j]
            if va is vb and i == j:
                terms.append((coef, f"{va.element_name(i)}^2"))
            else:
                terms.append((coef, f"{va.element_name(i)}*{vb.element_name(j)}"))
    for var, Lm in sorted(p.L.items(), key=lambda kv: kv[0].name):
        for j in np.nonzero(Lm[row])[0]:
            terms.append((Lm[row, j], var.element_name(j)))
    out = ""
    for coef, name in terms:
        mag = abs(coef)
        piece = name if mag == 1 else f"{mag:g}*{name}"
        out += (("-" if coef < 0 else "") + piece) if not out else (f" {'-' if coef < 0 else '+'} {piece}")
    c = p.c[row]
    if c != 0 or not out:
        out += f"{c:g}" if not out else f" {'-' if c < 0 else '+'} {abs(c):g}"
    return out


def _expr_lines(node: Node, limit=20) -> list:
    p = node.pl()
    if p is None:
        return [node.txt()]
    if node.m == 1:
        return [_poly_text(p, 0)]
    lines = []
    for r in range(min(node.m, limit)):
        sub = np.unravel_index(r, node.shape, order="F")
        lines.append(f"({', '.join(str(i + 1) for i in sub)})  {_poly_text(p, r)}")
    if node.m > limit:
        lines.append("...")
    return lines


class OptimExpr(MObject):
    """An array of optimisation expressions; a variable is the special case of a bare NVar node."""

    def __init__(self, node: Node):
        self.node = node

    @property
    def is_var(self) -> bool:
        return isinstance(self.node, NVar)

    def class_name(self) -> str:
        return "optim.problemdef.OptimizationVariable" if self.is_var else \
            "optim.problemdef.OptimizationExpression"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "optim.problemdef.OptimizationExpression")

    def matlab_shape(self) -> tuple:
        return self.node.shape

    # -- operators ---------------------------------------------------------------------------------
    def binary_op(self, interp, op, left, right):
        if isinstance(left, (OptimConstraint, OptimProblem)) or isinstance(right, (OptimConstraint, OptimProblem)):
            return NotImplemented
        a, b = _node(left), _node(right)
        if op == "+":
            return OptimExpr(NAdd(a, b, 1.0))
        if op == "-":
            return OptimExpr(NAdd(a, b, -1.0))
        if op == ".*":
            return OptimExpr(NMul(a, b))
        if op == "*":
            return OptimExpr(NMul(a, b) if a.m == 1 or b.m == 1 else NMatMul(a, b))
        if op == "./":
            return OptimExpr(NDiv(a, b))
        if op == "/":
            if b.m != 1:
                raise MatlabError("Division of optimization expressions needs a scalar divisor; use ./ for "
                                  "element-wise division.")
            return OptimExpr(NDiv(a, b))
        if op == ".^":
            return OptimExpr(NPow(a, b))
        if op == "^":
            if a.m != 1 or b.m != 1:
                raise MatlabError("The ^ operator needs scalar operands for optimization expressions; "
                                  "use .^ for element-wise powers.")
            return OptimExpr(NPow(a, b))
        if op in ("<=", ">=", "=="):
            return OptimConstraint(NAdd(a, b, -1.0), op)
        if op in ("<", ">", "~=", "!="):
            raise MatlabError(f"The operator '{op}' is not supported for optimization expressions; "
                              "constraints use <=, >= or ==.")
        return NotImplemented

    def unary_op(self, interp, op):
        if op == "-":
            return OptimExpr(NAdd(NConst(0.0), self.node, -1.0))
        if op == "+":
            return self
        if op in ("'", ".'"):
            return self._transpose()
        return NotImplemented

    def _positions(self):
        return np.arange(self.node.m, dtype=float).reshape(self.node.shape, order="F")

    def _gather(self, idx, label="index"):
        idx = np.asarray(idx)
        if self.is_var:
            flat = self.node.idx.reshape(-1, order="F")[idx.astype(int).reshape(-1, order="F")]
            return OptimExpr(NVar(self.node.info, flat.reshape(idx.shape, order="F")))
        return OptimExpr(NGather([self.node], idx.astype(int), label))

    def _transpose(self):
        if len(self.node.shape) != 2:
            raise MatlabError("Transpose is only defined for 2-D optimization expressions.")
        return self._gather(self._positions().T, "transpose")

    # -- indexing --------------------------------------------------------------------------------------
    def _named(self, args):
        """Translate index names ('apple') into positions."""
        out = []
        info = self.node.info if self.is_var else None
        for d, a in enumerate(args):
            if info is not None and (is_text(a) and text(a) != ":" or isinstance(a, Cell)):
                dim = d if len(args) > 1 else next((i for i, n in enumerate(info.shape) if n != 1), 0)
                names = info.index_names[dim] if dim < len(info.index_names) else None
                if not names:
                    raise MatlabError(f"The variable '{info.name}' has no index names for this dimension.")
                wanted = [text(v) for v in a.values_f()] if isinstance(a, Cell) else [text(a)]
                pos = []
                for w in wanted:
                    if w not in names:
                        raise MatlabError(f"'{w}' is not an index name of the variable '{info.name}'.")
                    pos.append(names.index(w) + 1.0)
                out.append(np.array([pos]))
            else:
                out.append(_plain(a))
        return out

    def index_get(self, interp, args):
        return self._gather(ix.get(self._positions(), self._named(args)))

    def index_set(self, interp, args, value):
        if self.is_var:
            raise MatlabError("Elements of an optimization variable cannot be assigned; build an expression "
                              "with optimexpr instead.")
        val = _node(value)
        m = self.node.m
        P = (self._positions() + 1.0)
        V = np.arange(m + 1, m + 1 + val.m, dtype=float).reshape(val.shape, order="F")
        R = ix.set_(P.copy(), [_plain(a) for a in args], V if val.m != 1 else np.array([[m + 1.0]]),
                    lambda: 0.0)
        return OptimExpr(NGather([NConst(0.0), self.node, val], np.asarray(R, dtype=int), "assign"))

    @classmethod
    def concat(cls, items, axis):
        nodes = [_node(i) for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if not nodes:
            return OptimExpr(NConst(np.zeros((0, 0))))
        offs = np.cumsum([0] + [n.m for n in nodes])
        parts = [np.arange(n.m).reshape(n.shape, order="F") + o for n, o in zip(nodes, offs[:-1])]
        nd = max([axis + 1] + [p.ndim for p in parts])
        parts = [p.reshape(p.shape + (1,) * (nd - p.ndim)) for p in parts]
        try:
            idx = np.concatenate(parts, axis=axis)
        except ValueError:
            raise MatlabError("Dimensions of optimization expressions being concatenated are not "
                              "consistent.") from None
        return OptimExpr(NGather(nodes, idx, "concat"))

    # -- properties --------------------------------------------------------------------------------------
    def get_field(self, interp, name):
        if self.is_var:
            info = self.node.info
            idx = self.node.idx
            if name == "Name":
                return char(info.name)
            if name == "Type":
                return char(info.type)
            if name == "LowerBound":
                return info.lb.reshape(-1, order="F")[idx.reshape(-1, order="F")].reshape(idx.shape, order="F")
            if name == "UpperBound":
                return info.ub.reshape(-1, order="F")[idx.reshape(-1, order="F")].reshape(idx.shape, order="F")
        if name == "IndexNames":
            names = self.node.info.index_names if self.is_var else [None] * len(self.node.shape)
            return Cell.from_list([Cell.from_list([char(n) for n in nm]) if nm else Cell.empty(1, 0)
                                   for nm in names])
        if name == "Variables":
            acc: dict = {}
            self.node.variables(acc)
            return Struct.scalar({v.name: OptimExpr(NVar(v)) for v in sorted(acc.values(), key=lambda v: v.name)})
        if self.has_method(name):
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name, value):
        if not self.is_var:
            raise MatlabError(f"Cannot set property '{name}' of an optimization expression.")
        info, idx = self.node.info, self.node.idx.reshape(-1, order="F")
        if name in ("LowerBound", "UpperBound"):
            v = np.real(fnum(value)).astype(float)
            if v.size not in (1, idx.size):
                raise MatlabError(f"{name} must be a scalar or have the same size as the variable.")
            arr = info.lb if name == "LowerBound" else info.ub
            flat = arr.reshape(-1, order="F").copy()
            flat[idx] = v.reshape(-1, order="F") if v.size > 1 else v.flat[0]
            arr[...] = flat.reshape(arr.shape, order="F")
            return self
        if name == "Type":
            t = text(value).lower()
            if t not in ("continuous", "integer"):
                raise MatlabError("Type must be 'continuous' or 'integer'.")
            info.type = t
            return self
        raise MatlabError(f"Cannot set property '{name}' of an optimization variable.")

    # -- methods -------------------------------------------------------------------------------------------
    _METHODS = ("sum", "prod", "dot", "mean", "diff", "cumsum", "transpose", "ctranspose", "reshape", "evaluate",
                "show", "write", "showexpr", "showvar", "showbounds", "writevar", "writebounds", "uminus", "plus",
                "minus", "times", "mtimes", "rdivide", "mrdivide", "power", "mpower", "horzcat", "vertcat",
                "cat", "colon_unused", "norm", "fcn2optimexpr") + tuple(_FUNCS)

    def has_method(self, name):
        return name in self._METHODS and name != "colon_unused"

    def call_method(self, interp, name, args, nargout):
        if name == "fcn2optimexpr":
            return list(_fcn2optimexpr(interp, args, nargout))
        if name in ("plus", "minus", "times", "mtimes", "rdivide", "mrdivide", "power", "mpower"):
            op = {"plus": "+", "minus": "-", "times": ".*", "mtimes": "*", "rdivide": "./", "mrdivide": "/",
                  "power": ".^", "mpower": "^"}[name]
            return [self.binary_op(interp, op, args[0], args[1])]
        if name in ("horzcat", "vertcat"):
            return [OptimExpr.concat(list(args), 1 if name == "horzcat" else 0)]
        if name == "cat":
            return [OptimExpr.concat(list(args[1:]), int(fnum(args[0]).flat[0]) - 1)]
        if name == "dot":
            a, b = _node(args[0]), _node(args[1])
            if a.m != b.m:
                raise MatlabError("dot: the two arguments must have the same number of elements.")
            ea = OptimExpr(NGather([a], np.arange(a.m).reshape(-1, 1), "reshape"))
            eb = OptimExpr(NGather([b], np.arange(b.m).reshape(-1, 1), "reshape"))
            return [OptimExpr(NReduce(NMul(ea.node, eb.node), "sum", 0))]
        if not isinstance(args[0], OptimExpr) or args[0] is not self:
            raise MatlabError(f"{name}: the first argument must be the optimization expression.")
        rest = list(args[1:])
        node = self.node
        if name in _FUNCS:
            return [OptimExpr(NFunc(name, node))]
        if name in ("sum", "prod", "mean", "cumsum", "diff"):
            if rest and is_text(rest[0]) and text(rest[0]) == "all":
                flat = OptimExpr(NGather([node], np.arange(node.m).reshape(-1, 1), "reshape")).node
                axis, node = 0, flat
            elif rest and not is_text(rest[-1]) and name != "diff":
                axis = int(fnum(rest[0]).flat[0]) - 1
            elif name == "diff" and len(rest) > 1:
                axis = int(fnum(rest[1]).flat[0]) - 1
            else:
                axis = next((i for i, d in enumerate(node.shape) if d != 1), 0)
            if axis >= len(node.shape):
                return [self]
            if name == "prod":
                return [OptimExpr(NProd(node, axis))]
            if name == "mean":
                return [OptimExpr(NDiv(NReduce(node, "sum", axis), NConst(float(node.shape[axis]))))]
            if name == "diff":
                order = int(fnum(rest[0]).flat[0]) if rest and not om._is_empty(rest[0]) else 1
                for _ in range(order):
                    node = NReduce(node, "diff", axis)
                return [OptimExpr(node)]
            return [OptimExpr(NReduce(node, name, axis))]
        if name in ("transpose", "ctranspose"):
            return [self._transpose()]
        if name == "uminus":
            return [self.unary_op(interp, "-")]
        if name == "reshape":
            dims = [int(v) for a in rest for v in np.atleast_1d(fnum(a).flatten())]
            if _prod(dims) != node.m:
                raise MatlabError("reshape: the number of elements must not change.")
            if len(dims) == 1:
                dims.append(1)
            return [self._gather(np.arange(node.m).reshape(dims, order="F"), "reshape")]
        if name == "norm":
            flat = OptimExpr(NGather([node], np.arange(node.m).reshape(-1, 1), "reshape")).node
            if rest and float(fnum(rest[0]).flat[0]) != 2:
                raise MatlabError("norm of an optimization expression supports the 2-norm only.")
            return [OptimExpr(NFunc("sqrt", NReduce(NPow(flat, NConst(2.0)), "sum", 0)))]
        if name == "evaluate":
            if not rest:
                raise MatlabError("evaluate needs a structure of variable values, evaluate(expr, sol).")
            return [np.asarray(node.ev(_values_from_struct(node, rest[0])), dtype=float)]
        if name in ("show", "showexpr", "showvar", "showbounds", "write", "writevar", "writebounds"):
            lines = self._show_lines(name)
            if name.startswith("write"):
                fname = text(rest[0]) if rest else ("expression.txt" if not self.is_var else "variable.txt")
                with open(fname, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            else:
                interp.out("\n".join(lines) + "\n")
            return []
        return super().call_method(interp, name, args, nargout)

    def _show_lines(self, name="show"):
        if self.is_var and name in ("show", "showvar", "writevar", "write"):
            info = self.node.info
            lines = [f"    [ {info.name} ]" if info.size == 1 else
                     "    " + "\n    ".join(f"[ {info.element_name(k)} ]" for k in
                                             self.node.idx.reshape(-1, order="F")[:20])]
            return lines
        if name in ("showbounds", "writebounds"):
            return _bound_lines([self.node.info]) or ["    This variable does not have bounds."]
        return ["    " + s for s in _expr_lines(self.node)]

    def display_lines(self):
        shape = "x".join(str(d) for d in self.node.shape)
        if self.is_var:
            info = self.node.info
            kind = "OptimizationVariable" if self.node.m == 1 else f"{shape} OptimizationVariable array"
            lines = [f"  {kind} with properties:", "", f"          Name: '{info.name}'",
                     f"          Type: '{info.type}'", "    IndexNames: {{}  {}}"]
            if self.node.m == 1:
                lines += [f"    LowerBound: {float(self.get_field(None, 'LowerBound').flat[0]):g}",
                          f"    UpperBound: {float(self.get_field(None, 'UpperBound').flat[0]):g}"]
            else:
                lines += [f"    LowerBound: [{shape} double]", f"    UpperBound: [{shape} double]"]
            return lines + ["", "  See variables with show.", "  See bounds with showbounds."]
        p = self.node.pl()
        kind = "Linear" if p is not None and p.degree() <= 1 else ("Quadratic" if p is not None else "Nonlinear")
        head = f"  {kind} OptimizationExpression" if self.node.m == 1 else \
            f"  {shape} {kind} OptimizationExpression array"
        return [head, ""] + ["    " + s for s in _expr_lines(self.node)]


def _bound_lines(infos) -> list:
    lines = []
    for info in infos:
        for k in range(min(info.size, 50)):
            lo, hi = info.lb.reshape(-1, order="F")[k], info.ub.reshape(-1, order="F")[k]
            nm = info.element_name(k)
            if np.isfinite(lo) and np.isfinite(hi):
                lines.append(f"    {lo:g} <= {nm} <= {hi:g}")
            elif np.isfinite(lo):
                lines.append(f"    {lo:g} <= {nm}")
            elif np.isfinite(hi):
                lines.append(f"    {nm} <= {hi:g}")
    return lines


def _values_from_struct(node_or_vars, s) -> dict:
    if not isinstance(s, Struct) or s.numel() != 1:
        raise MatlabError("The variable values must be given as a structure with one field per variable.")
    if isinstance(node_or_vars, Node):
        acc: dict = {}
        node_or_vars.variables(acc)
        infos = list(acc.values())
    else:
        infos = list(node_or_vars)
    data = s.data.flat[0]
    vals = {}
    for info in infos:
        if info.name not in data:
            raise MatlabError(f"The structure has no field for the variable '{info.name}'.")
        v = np.real(fnum(data[info.name])).astype(float)
        if v.size != info.size:
            raise MatlabError(f"The value for '{info.name}' must have {info.size} elements.")
        vals[info] = v.reshape(info.shape, order="F")
    return vals


class OptimConstraint(MObject):
    """An array of constraints  expr (<=, ==, >=) 0."""

    def __init__(self, node: Node, op: str | None, equation=False):
        self.node, self.op, self.equation = node, op, equation

    def class_name(self):
        if self.op == "==":
            return "optim.problemdef.OptimizationEquality"
        if self.op is None:
            return "optim.problemdef.OptimizationConstraint"
        return "optim.problemdef.OptimizationInequality"

    def isa_class(self, name):
        return name in (self.class_name(), "optim.problemdef.OptimizationConstraint")

    def matlab_shape(self):
        return self.node.shape

    def index_get(self, interp, args):
        idx = ix.get(np.arange(self.node.m, dtype=float).reshape(self.node.shape, order="F"),
                     [_plain(a) for a in args])
        return OptimConstraint(NGather([self.node], np.asarray(idx, dtype=int)), self.op)

    def index_set(self, interp, args, value):
        if not isinstance(value, OptimConstraint):
            raise MatlabError("Only constraints (expressions compared with <=, >= or ==) can be assigned "
                              "into a constraint array.")
        if self.op is not None and value.op != self.op:
            raise MatlabError("All constraints in one array must use the same comparison; use separate "
                              "arrays for <=, >= and ==.")
        m = self.node.m
        P = np.arange(1, m + 1, dtype=float).reshape(self.node.shape, order="F")
        V = np.arange(m + 1, m + 1 + value.node.m, dtype=float).reshape(value.node.shape, order="F")
        R = ix.set_(P.copy(), [_plain(a) for a in args], V if value.node.m != 1 else np.array([[m + 1.0]]),
                    lambda: 0.0)
        # unassigned entries are 0 <= 0 and so never bind
        return OptimConstraint(NGather([NConst(0.0), self.node, value.node], np.asarray(R, dtype=int), "assign"),
                               value.op)

    @classmethod
    def concat(cls, items, axis):
        cons = [i for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        if not all(isinstance(c, OptimConstraint) for c in cons):
            raise MatlabError("Constraints can only be concatenated with other constraints.")
        ops = {c.op for c in cons if c.op is not None}
        if len(ops) > 1:
            raise MatlabError("Constraints with different comparisons cannot be concatenated.")
        e = OptimExpr.concat([OptimExpr(c.node) for c in cons], axis)
        return OptimConstraint(e.node, ops.pop() if ops else None)

    def has_method(self, name):
        return name in ("show", "write", "showconstr", "writeconstr", "infeasibility", "evaluate", "issatisfied")

    def call_method(self, interp, name, args, nargout):
        if name in ("infeasibility", "evaluate", "issatisfied"):
            if len(args) < 2:
                raise MatlabError(f"{name} needs a structure of variable values.")
            v = np.asarray(self.node.ev(_values_from_struct(self.node, args[1])), dtype=float)
            if name == "evaluate":
                return [v]
            viol = np.abs(v) if self.op == "==" else (np.maximum(v, 0.0) if self.op == "<=" else np.maximum(-v, 0.0))
            if name == "infeasibility":
                return [viol]
            return [viol <= 1e-6]
        lines = self.lines()
        if name.startswith("write"):
            fname = text(args[1]) if len(args) > 1 else "constraint.txt"
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        else:
            interp.out("\n".join(lines) + "\n")
        return []

    def lines(self, limit=20):
        p = self.node.pl()
        if self.op is None:
            return ["    (empty constraint)"]
        if p is None:
            return [f"    {self.node.txt()} {self.op} 0"]
        out = []
        for r in range(min(self.node.m, limit)):
            if not any(np.any(v[r]) for v in p.L.values()) and not any(np.any(v[r]) for v in p.Q.values()) \
                    and p.c[r] == 0:
                continue
            lhs = Poly(np.zeros(p.m), p.L, p.Q)
            prefix = "" if self.node.m == 1 else \
                f"({', '.join(str(i + 1) for i in np.unravel_index(r, self.node.shape, order='F'))})  "
            out.append(f"    {prefix}{_poly_text(lhs, r)} {self.op} {-p.c[r]:g}")
        if self.node.m > limit:
            out.append("    ...")
        return out or ["    (empty constraint)"]

    def display_lines(self):
        shape = "x".join(str(d) for d in self.node.shape)
        p = self.node.pl()
        kind = "Linear" if p is not None and p.degree() <= 1 else ("Quadratic" if p is not None else "Nonlinear")
        what = "OptimizationEquality" if self.op == "==" else (
            "OptimizationInequality" if self.op else "OptimizationConstraint")
        head = f"  {kind} {what}" if self.node.m == 1 else f"  {shape} {kind} {what} array"
        return [head, ""] + self.lines()


# -- constructors ---------------------------------------------------------------------------------------

def _dims_and_names(args: list, what: str):
    """Sizes (and index names) from optimvar/optimexpr/optimconstr size arguments."""
    dims, names = [], []
    for a in args:
        if isinstance(a, Cell) or isinstance(a, MString) or (is_text(a) and not isinstance(a, np.ndarray)):
            items = [text(v) for v in a.values_f()] if isinstance(a, Cell) else [text(a)]
            dims.append(len(items))
            names.append(items)
        elif hasattr(a, "values_f") and not isinstance(a, Cell):   # a string array of names
            items = [str(v) for v in a.values_f()]
            dims.append(len(items))
            names.append(items)
        else:
            v = np.real(fnum(a)).flatten()
            if np.any(v < 0) or np.any(v != np.round(v)):
                raise MatlabError(f"{what}: sizes must be nonnegative integers.")
            for d in v:
                dims.append(int(d))
                names.append(None)
    if not dims:
        dims, names = [1, 1], [None, None]
    elif len(dims) == 1 and names[0] is not None:
        dims, names = [1, dims[0]], [None, names[0]]      # one list of index names makes a row
    elif len(dims) == 1:
        dims, names = [dims[0], 1], [names[0], None]
    return dims, names


@builtin("optimvar")
def _optimvar(interp, args, nargout):
    nargs(args, 1, None, "optimvar")
    if not is_text(args[0]):
        raise MatlabError("optimvar: the first argument must be the variable name.")
    name = text(args[0])
    if not name.isidentifier():
        raise MatlabError(f"optimvar: '{name}' is not a valid variable name.")
    rest = list(args[1:])
    size_args = []
    while rest and not (is_text(rest[0]) and isinstance(rest[0], np.ndarray)
                        and text(rest[0]).lower() in ("type", "lowerbound", "upperbound")):
        if isinstance(rest[0], MString) and text(rest[0]).lower() in ("type", "lowerbound", "upperbound"):
            break
        size_args.append(rest.pop(0))
    dims, names = _dims_and_names(size_args, "optimvar")
    if len(rest) % 2:
        raise MatlabError("optimvar: options must be name-value pairs.")
    vtype, lb, ub = "continuous", -np.inf, np.inf
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower()
        if key == "type":
            vtype = text(rest[k + 1]).lower()
            if vtype not in ("continuous", "integer"):
                raise MatlabError("optimvar: Type must be 'continuous' or 'integer'.")
        elif key in ("lowerbound", "upperbound"):
            v = np.real(fnum(rest[k + 1])).astype(float)
            if v.size not in (1, _prod(dims)):
                raise MatlabError(f"optimvar: {text(rest[k])} must be a scalar or match the variable size.")
            v = v.flat[0] if v.size == 1 else v.reshape(dims, order="F")
            if key == "lowerbound":
                lb = v
            else:
                ub = v
        else:
            raise MatlabError(f"optimvar: unknown option '{text(rest[k])}'.")
    return OptimExpr(NVar(VarInfo(name, dims, vtype, lb, ub, names)))


@builtin("optimexpr")
def _optimexpr(interp, args, nargout):
    dims, _ = _dims_and_names(list(args), "optimexpr")
    return OptimExpr(NConst(np.zeros(dims)))


def _optimconstr(kind):
    def f(interp, args, nargout):
        dims, _ = _dims_and_names(list(args), kind)
        return OptimConstraint(NConst(np.zeros(dims)), {"optimeq": "==", "optimineq": None}.get(kind))
    return f


for _nm in ("optimconstr", "optimineq", "optimeq"):
    builtin(_nm)(_optimconstr(_nm))


def _fcn2optimexpr(interp, args, nargout):
    if not args or not om.is_fn(args[0]):
        raise MatlabError("fcn2optimexpr: the first argument must be a function handle.")
    fun = args[0]
    rest = list(args[1:])
    out_size = None
    k = 0
    fargs = []
    while k < len(rest):
        a = rest[k]
        if is_text(a) and text(a).lower() in ("outputsize", "reuseevaluation", "analysis", "display") \
                and k + 1 < len(rest):
            key = text(a).lower()
            if key == "outputsize":
                v = rest[k + 1]
                out_size = [tuple(int(x) for x in fnum(c).flatten()) for c in v.values_f()] \
                    if isinstance(v, Cell) else [tuple(int(x) for x in fnum(v).flatten())]
            k += 2
            continue
        fargs.append(a.node if isinstance(a, OptimExpr) else a)
        k += 1
    nout = max(nargout, 1)
    if out_size is None or len(out_size) < nout:
        # find the output sizes by calling the function once at a point inside the bounds
        acc: dict = {}
        for a in fargs:
            if isinstance(a, Node):
                a.variables(acc)
        vals = {}
        for info in acc.values():
            lo = np.where(np.isfinite(info.lb), info.lb, np.where(np.isfinite(info.ub), info.ub - 1.0, 0.0))
            hi = np.where(np.isfinite(info.ub), info.ub, lo + 1.0)
            vals[info] = np.clip(np.ones(info.shape) * 0.5 + 0 * lo, lo, hi) if np.all(lo <= 0.5) else 0.5 * (lo + hi)
        a_now = [np.asarray(x.ev(vals), dtype=float) if isinstance(x, Node) else x for x in fargs]
        res = interp.call_function_value(fun, a_now, nout)
        if len(res) < nout:
            raise MatlabError("fcn2optimexpr: the function returned fewer outputs than requested.")
        out_size = [fnum(r).shape for r in res[:nout]]
    outs_ = []
    for i in range(nout):
        shape = out_size[i] if len(out_size[i]) >= 2 else (out_size[i][0], 1)
        outs_.append(OptimExpr(NCall(interp, fun, fargs, i, nout, shape)))
    return tuple(outs_)


@builtin("fcn2optimexpr")
def _fcn2optimexpr_builtin(interp, args, nargout):
    res = _fcn2optimexpr(interp, args, nargout)
    return res if nargout > 1 else res[0]


# -- problems ---------------------------------------------------------------------------------------------

class OptimProblem(MObject):
    def __init__(self, equation=False):
        self.equation = equation
        self.description = ""
        self.sense = "minimize"
        self.objective = None           # OptimExpr or None
        self.constraints = None         # OptimConstraint, Struct of them, or None

    def class_name(self):
        return "optim.problemdef.EquationProblem" if self.equation else "optim.problemdef.OptimizationProblem"

    def _copy(self):
        new = OptimProblem(self.equation)
        new.__dict__.update(self.__dict__)
        return new

    _CONS = ("Constraints", "Equations")

    def get_field(self, interp, name):
        if name == "Description":
            return char(self.description)
        if name == "ObjectiveSense" and not self.equation:
            return char(self.sense)
        if name == "Objective" and not self.equation:
            return self.objective if self.objective is not None else empty_double()
        if name == ("Equations" if self.equation else "Constraints"):
            return self.constraints if self.constraints is not None else empty_double()
        if name == "Variables":
            return Struct.scalar({v.name: OptimExpr(NVar(v)) for v in self.variables()})
        if self.has_method(name):
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")

    def set_field(self, interp, name, value):
        new = self._copy()
        if name == "Description":
            new.description = text(value)
        elif name == "ObjectiveSense" and not self.equation:
            s = text(value).lower()
            if s not in ("min", "max", "minimize", "maximize"):
                raise MatlabError("ObjectiveSense must be 'minimize' or 'maximize' (or 'min' / 'max').")
            new.sense = "maximize" if s.startswith("max") else "minimize"
        elif name == "Objective" and not self.equation:
            new.objective = _check_objective(value)
        elif name == ("Equations" if self.equation else "Constraints"):
            new.constraints = _check_constraints(value, self.equation)
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.")
        return new

    def named_constraints(self) -> list:
        """[(name, OptimConstraint)]"""
        c = self.constraints
        if c is None:
            return []
        if isinstance(c, OptimConstraint):
            return [("constraint" if not self.equation else "equation", c)]
        return [(k, v) for k, v in c.data.flat[0].items() if isinstance(v, OptimConstraint)]

    def variables(self) -> list:
        acc: dict = {}
        if self.objective is not None:
            self.objective.node.variables(acc)
        for _, c in self.named_constraints():
            c.node.variables(acc)
        return sorted(acc.values(), key=lambda v: (v.name, v.id))

    def has_method(self, name):
        return name in ("solve", "show", "write", "showproblem", "writeproblem", "prob2struct", "varindex",
                        "solvers", "optimoptions")

    def call_method(self, interp, name, args, nargout):
        if args[0] is not self:
            raise MatlabError(f"{name}: the first argument must be the problem.")
        rest = list(args[1:])
        if name == "solve":
            return list(_solve(interp, self, rest, nargout))
        if name in ("show", "showproblem", "write", "writeproblem"):
            lines = self.display_lines()
            if name.startswith("write"):
                fname = text(rest[0]) if rest else "problem.txt"
                with open(fname, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            else:
                interp.out("\n".join(lines) + "\n")
            return []
        if name == "prob2struct":
            return [_prob2struct(interp, self, rest)]
        if name == "varindex":
            infos, offs = _layout(self.variables())
            if rest:
                nm = text(rest[0])
                for v in infos:
                    if v.name == nm:
                        return [np.arange(offs[v] + 1, offs[v] + v.size + 1, dtype=float).reshape(1, -1)]
                raise MatlabError(f"varindex: the problem has no variable named '{nm}'.")
            return [Struct.scalar({v.name: np.arange(offs[v] + 1, offs[v] + v.size + 1, dtype=float).reshape(1, -1)
                                   for v in infos})]
        if name == "solvers":
            kind = _classify(self)
            default = kind["solver"]
            valid = {"linprog": ["linprog", "intlinprog", "quadprog", "fmincon"],
                     "intlinprog": ["intlinprog", "ga"], "quadprog": ["quadprog", "fmincon"],
                     "fmincon": ["fmincon"], "ga": ["ga"], "lsqlin": ["lsqlin", "fsolve", "lsqnonlin"],
                     "fsolve": ["fsolve", "lsqnonlin"]}.get(default, [default])
            return [MString(default), Cell.from_list([MString(v) for v in valid])][:max(nargout, 1)]
        if name == "optimoptions":
            return [om.OptimOptions(_classify(self)["solver"])]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self):
        lines = ["  " + ("EquationProblem" if self.equation else "OptimizationProblem") + " : "
                 + self.description, ""]
        infos = self.variables()
        lines += ["\tSolve for:", "       " + ", ".join(v.name for v in infos), ""]
        if not self.equation:
            lines.append(f"\t{self.sense} :")
            lines += ["       " + s for s in (_expr_lines(self.objective.node) if self.objective is not None
                                            else ["(no objective)"])]
            lines.append("")
        for nm, c in self.named_constraints():
            lines.append(f"\t{'subject to' if not self.equation else 'equation'} {nm}:")
            lines += ["   " + s for s in c.lines()]
            lines.append("")
        b = _bound_lines(infos)
        if b:
            lines += ["\tvariable bounds:"] + ["   " + s for s in b] + [""]
        return lines


def _check_objective(value):
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    if isinstance(value, Struct):
        items = [v for v in value.data.flat[0].values()]
        if len(items) != 1:
            raise MatlabError("Problems with several objectives are not supported.")
        value = items[0]
    e = value if isinstance(value, OptimExpr) else OptimExpr(_node(value))
    if e.node.m != 1:
        raise MatlabError("The objective must be a scalar optimization expression.")
    return e


def _check_constraints(value, equation):
    if isinstance(value, np.ndarray) and value.size == 0:
        return None
    if isinstance(value, OptimConstraint):
        items = [value]
    elif isinstance(value, Struct) and value.numel() == 1:
        items = list(value.data.flat[0].values())
    else:
        raise MatlabError("Constraints must be made by comparing optimization expressions with <=, >= or ==.")
    for c in items:
        if not isinstance(c, OptimConstraint):
            raise MatlabError("Constraints must be made by comparing optimization expressions with <=, >= "
                              "or ==.")
        if equation and c.op not in ("==", None):
            raise MatlabError("An equation problem only accepts equations made with ==.")
    return value


def _problem_ctor(equation: bool):
    what = "eqnproblem" if equation else "optimproblem"

    def f(interp, args, nargout):
        if len(args) % 2:
            raise MatlabError(f"{what}: arguments must be name-value pairs.")
        prob = OptimProblem(equation)
        for k in range(0, len(args), 2):
            prob = prob.set_field(interp, {"description": "Description", "objective": "Objective",
                                           "objectivesense": "ObjectiveSense", "constraints": "Constraints",
                                           "equations": "Equations"}.get(text(args[k]).lower(), text(args[k])),
                                  args[k + 1])
        return prob

    return f


builtin("optimproblem")(_problem_ctor(False))
builtin("eqnproblem")(_problem_ctor(True))


# -- solving -------------------------------------------------------------------------------------------------

def _layout(infos):
    offs, total = {}, 0
    for v in infos:
        offs[v] = total
        total += v.size
    return infos, offs


def _linear_rows(p: Poly, infos, offs, n):
    A = np.zeros((p.m, n))
    for v, Lm in p.L.items():
        A[:, offs[v]:offs[v] + v.size] = Lm
    return A


def _classify(prob: OptimProblem) -> dict:
    infos, offs = _layout(prob.variables())
    n = sum(v.size for v in infos)
    integer = any(v.type == "integer" for v in infos)
    obj_p = prob.objective.node.pl() if prob.objective is not None else Poly(np.zeros(1))
    cons = prob.named_constraints()
    cons_p = [(nm, c, c.node.pl()) for nm, c in cons]
    lin_cons = all(p is not None and p.degree() <= 1 for _, _, p in cons_p)
    if prob.equation:
        solver = "lsqlin" if lin_cons else "fsolve"
    elif obj_p is not None and obj_p.degree() <= 1 and lin_cons:
        solver = "intlinprog" if integer else "linprog"
    elif obj_p is not None and obj_p.degree() == 2 and lin_cons and not integer:
        solver = "quadprog"
    else:
        solver = "ga" if integer else "fmincon"
    return {"infos": infos, "offs": offs, "n": n, "integer": integer, "obj_p": obj_p, "cons_p": cons_p,
            "lin_cons": lin_cons, "solver": solver}


def _matrices(info: dict):
    """Linear constraint matrices and bounds from the classified problem."""
    infos, offs, n = info["infos"], info["offs"], info["n"]
    A, b, Aeq, beq = [], [], [], []
    where = []          # (name, kind, rows) for the multipliers
    for nm, c, p in info["cons_p"]:
        if p is None or p.degree() > 1 or c.op is None:
            continue
        rows = _linear_rows(p, infos, offs, n)
        keep = np.any(rows != 0, axis=1) | (p.c != 0)
        if c.op == "==":
            where.append((nm, "eq", len(beq), c.node.shape, keep))
            Aeq.append(rows[keep])
            beq.append(-p.c[keep])
        else:
            sgn = 1.0 if c.op == "<=" else -1.0
            where.append((nm, "ineq", sum(x.size for x in b), c.node.shape, keep))
            A.append(sgn * rows[keep])
            b.append(-sgn * p.c[keep])
    A = np.vstack(A) if A else np.zeros((0, n))
    b = np.concatenate(b) if b else np.zeros(0)
    Aeq = np.vstack(Aeq) if Aeq else np.zeros((0, n))
    beq = np.concatenate(beq) if beq else np.zeros(0)
    lo = np.concatenate([v.lb.reshape(-1, order="F") for v in infos]) if infos else np.zeros(0)
    hi = np.concatenate([v.ub.reshape(-1, order="F") for v in infos]) if infos else np.zeros(0)
    intcon = np.concatenate([np.arange(offs[v], offs[v] + v.size) for v in infos if v.type == "integer"]
                            or [np.zeros(0, dtype=int)]).astype(int)
    return A, b, Aeq, beq, lo, hi, intcon, where


def _quad_objective(p: Poly, infos, offs, n):
    H = np.zeros((n, n))
    for (va, vb), Qm in p.Q.items():
        H[offs[va]:offs[va] + va.size, offs[vb]:offs[vb] + vb.size] += Qm[0]
    H = H + H.T
    f = _linear_rows(p, infos, offs, n)[0]
    return H, f, float(p.c[0])


def _unpack(x, infos, offs) -> Struct:
    return Struct.scalar({v.name: np.asarray(x[offs[v]:offs[v] + v.size], dtype=float).reshape(v.shape, order="F")
                          for v in infos})


def _mat(M):
    return M if M.shape[0] else empty_double()


def _col(v):
    return v.reshape(-1, 1) if v.size else empty_double()


def _solve(interp, prob: OptimProblem, rest: list, nargout: int):
    x0s = None
    opts_raw = None
    solver_name = None
    k = 0
    if rest and isinstance(rest[0], Struct):
        x0s = rest[0]
        k = 1
    while k < len(rest):
        if not is_text(rest[k]) or k + 1 >= len(rest):
            raise MatlabError("solve: options must be given as name-value pairs ('Solver', 'Options').")
        key = text(rest[k]).lower()
        if key == "options":
            opts_raw = rest[k + 1]
        elif key == "solver":
            solver_name = text(rest[k + 1])
        elif key in ("constraintderivative", "objectivederivative", "equationderivative"):
            pass  # derivatives are always taken by finite differences or from the polynomial form
        else:
            raise MatlabError(f"solve: unknown option '{text(rest[k])}'.")
        k += 2
    info = _classify(prob)
    infos, offs, n = info["infos"], info["offs"], info["n"]
    if n == 0:
        raise MatlabError("solve: the problem has no variables.")
    solver = solver_name or info["solver"]
    if isinstance(opts_raw, om.OptimOptions) and solver_name is None and opts_raw.solver != solver:
        solver = opts_raw.solver
    A, b, Aeq, beq, lo, hi, intcon, where = _matrices(info)
    maximize = prob.sense == "maximize" and not prob.equation
    sgn = -1.0 if maximize else 1.0
    x0 = None
    if x0s is not None:
        vals0 = _values_from_struct(infos, x0s)
        x0 = np.concatenate([vals0[v].reshape(-1, order="F") for v in infos])
    options = opts_raw
    if options is None and solver in om.SOLVER_OPTIONS:
        options = om.OptimOptions(solver)
    lam_struct = Struct.scalar({})
    e = empty_double()
    if not prob.equation and solver in ("linprog", "intlinprog", "quadprog", "lsqlin"):
        p = info["obj_p"]
        if p is None or not info["lin_cons"]:
            raise MatlabError(f"solve: the solver '{solver}' cannot handle this nonlinear problem.")
        H, f, c0 = _quad_objective(p, infos, offs, n)
        interp.out(f"Solving problem using {solver}.\n")
        if solver in ("linprog", "intlinprog"):
            if p.degree() > 1:
                raise MatlabError(f"solve: the objective is not linear, so '{solver}' cannot be used.")
            if solver == "linprog":
                if info["integer"]:
                    raise MatlabError("solve: the problem has integer variables; use intlinprog.")
                res = interp.call_function_value(char("linprog"), [(sgn * f).reshape(-1, 1), _mat(A), _col(b),
                                                                   _mat(Aeq), _col(beq), _col(lo), _col(hi),
                                                                   options], 5)
            else:
                res = interp.call_function_value(char("intlinprog"), [
                    (sgn * f).reshape(-1, 1), (intcon + 1.0).reshape(1, -1) if intcon.size else e, _mat(A), _col(b),
                    _mat(Aeq), _col(beq), _col(lo), _col(hi), e, options], 4)
        else:
            if info["integer"]:
                raise MatlabError("solve: quadratic problems with integer variables are not supported.")
            res = interp.call_function_value(char("quadprog"), [sgn * H, (sgn * f).reshape(-1, 1), _mat(A), _col(b),
                                                                _mat(Aeq), _col(beq), _col(lo), _col(hi),
                                                                e if x0 is None else x0.reshape(-1, 1), options], 5)
        xs = np.real(fnum(res[0])).flatten()
        flag = float(fnum(res[2]).flat[0])
        out = res[3]
        if xs.size != n:
            sol = Struct.scalar({v.name: empty_double() for v in infos})
            fval = empty_double()
        else:
            sol = _unpack(xs, infos, offs)
            fval = float(0.5 * xs @ H @ xs + f @ xs + c0)
            if len(res) > 4 and isinstance(res[4], Struct):
                lam_struct = _lambda_out(res[4], infos, offs, where)
    elif prob.equation:
        sol, fval, flag, out = _solve_equations(interp, prob, info, x0, solver, options)
    else:
        if solver not in ("fmincon", "ga", "fminunc", "patternsearch", "particleswarm", "simulannealbnd",
                          "surrogateopt"):
            raise MatlabError(f"solve: the solver '{solver}' cannot be used for this problem.")
        sol, fval, flag, out, lam_struct = _solve_nonlinear(interp, prob, info, x0, solver, options, sgn,
                                                            (A, b, Aeq, beq, lo, hi, intcon, where))
    if isinstance(out, Struct) and "solver" not in out.fields:
        out = out.copy()
        out.set_field(0, "solver", char(solver))
    if isinstance(fval, (float, int)):
        fval = np.array([[float(fval)]])
    return (sol, fval, np.array([[float(flag)]]), out, lam_struct)[:max(nargout, 1)]


def _lambda_out(lam: Struct, infos, offs, where) -> Struct:
    d = lam.data.flat[0]

    def vec(name, size):
        v = d.get(name)
        a = np.real(fnum(v)).flatten() if v is not None and not om._is_empty(v) else np.zeros(size)
        return a if a.size == size else np.zeros(size)

    n = sum(v.size for v in infos)
    lower, upper = vec("lower", n), vec("upper", n)
    variables = {v.name: Struct.scalar({"Lower": lower[offs[v]:offs[v] + v.size].reshape(v.shape, order="F"),
                                        "Upper": upper[offs[v]:offs[v] + v.size].reshape(v.shape, order="F")})
                 for v in infos}
    n_ineq = sum(int(np.sum(k)) for _, kind, _, _, k in where if kind == "ineq")
    n_eq = sum(int(np.sum(k)) for _, kind, _, _, k in where if kind == "eq")
    ineq, eq = vec("ineqlin", n_ineq), vec("eqlin", n_eq)
    cons = {}
    for nm, kind, start, shape, keep in where:
        full = np.zeros(keep.size)
        src = ineq if kind == "ineq" else eq
        full[keep] = src[start:start + int(np.sum(keep))]
        cons[nm] = full.reshape(shape, order="F")
    return Struct.scalar({"Variables": Struct.scalar(variables), "Constraints": Struct.scalar(cons)})


def _solve_nonlinear(interp, prob, info, x0, solver, options, sgn, mats):
    from ._optim_local import NLP, solve_nlp
    infos, offs, n = info["infos"], info["offs"], info["n"]
    A, b, Aeq, beq, lo, hi, intcon, where = mats
    obj = prob.objective.node if prob.objective is not None else NConst(0.0)
    obj_p = info["obj_p"]
    count = [0]

    def vals_of(x):
        return {v: x[offs[v]:offs[v] + v.size].reshape(v.shape, order="F") for v in infos}

    def f(x):
        count[0] += 1
        v = float(np.asarray(obj.ev(vals_of(x)), dtype=float).flat[0])
        return sgn * v

    if obj_p is not None:
        H, fl, _ = _quad_objective(obj_p, infos, offs, n)

        def g(x):
            return sgn * (H @ x + fl)
    else:
        def g(x):
            f0 = f(x)
            h = np.sqrt(om.EPS) * np.maximum(np.abs(x), 1.0)
            out = np.zeros(n)
            for j in range(n):
                hj = h[j] if x[j] + h[j] <= hi[j] else -h[j]
                xp = x.copy()
                xp[j] += hj
                out[j] = (f(xp) - f0) / hj
            return out
    nonlin = [(nm, c) for nm, c, p in info["cons_p"] if (p is None or p.degree() > 1) and c.op is not None]

    def nl(x):
        vals = vals_of(x)
        cs, es = [], []
        for _, c in nonlin:
            v = np.asarray(c.node.ev(vals), dtype=float).reshape(-1, order="F")
            if c.op == "==":
                es.append(v)
            else:
                cs.append(v if c.op == "<=" else -v)
        return (np.concatenate(cs) if cs else np.zeros(0)), (np.concatenate(es) if es else np.zeros(0))

    opts = om.Opt(options, solver)
    rep = om.Report(interp, opts)
    interp.out(f"Solving problem using {solver}.\n")
    if solver == "fmincon" or solver == "fminunc":
        if x0 is None:
            raise MatlabError("solve: a nonlinear problem needs an initial point, solve(prob, x0), where x0 "
                              "is a structure with one field per variable.")
        if info["integer"]:
            raise MatlabError("solve: fmincon cannot handle integer variables; use the 'ga' solver.")
        P = NLP(n, f, g, A, b, Aeq, beq, lo, hi, nl if nonlin else None)
        alg = opts.word("Algorithm", "interior-point")
        res = solve_nlp(P, x0, "interior-point" if alg == "interior-point" else "sqp",
                        opts.num("OptimalityTolerance", 1e-6), opts.num("ConstraintTolerance", 1e-6),
                        opts.num("StepTolerance", 1e-10), opts.num("MaxIterations", 1000))
        rep.finish(res["flag"], res["message"])
        x = res["x"]
        out = om.output_struct(iterations=res["iterations"], funcCount=count[0], constrviolation=res["viol"],
                               firstorderopt=res["firstorder"], algorithm=alg, message=res["message"])
        lam = res["lam"]
        lin = Struct.scalar({"lower": lam["lower"].reshape(-1, 1), "upper": lam["upper"].reshape(-1, 1),
                             "ineqlin": lam["ineqlin"].reshape(-1, 1), "eqlin": lam["eqlin"].reshape(-1, 1)})
        lam_struct = _lambda_out(lin, infos, offs, where)
        cons = lam_struct.data.flat[0]["Constraints"]
        ki = ke = 0
        for nm, c in nonlin:
            size = c.node.m
            if c.op == "==":
                cons.set_field(0, nm, lam["eqnonlin"][ke:ke + size].reshape(c.node.shape, order="F"))
                ke += size
            else:
                cons.set_field(0, nm, lam["ineqnonlin"][ki:ki + size].reshape(c.node.shape, order="F"))
                ki += size
        flag = res["flag"]
    else:
        # hand the problem to one of the derivative-free solvers through small MATLAB-callable shims
        key = f"__optim_problem_{id(prob)}"
        interp.state.setdefault("_optim_shims", {})[key] = (f, nl, n)
        from ..parser import parse
        from ..values import FunctionHandle

        def handle(which):
            node = parse(f"@(x) optimProblemShim__('{key}', '{which}', x)").body[0].expr
            return FunctionHandle(params=list(node.params), body=node.body, closure={"__functions__": {}},
                                  text="@(x) objective(x)")

        e = empty_double()
        nonl = handle("nl2") if nonlin else e
        try:
            if solver == "ga":
                a = [handle("f"), np.array([[float(n)]]), _mat(A), _col(b), _mat(Aeq), _col(beq), _col(lo).T
                     if lo.size else e, _col(hi).T if hi.size else e, nonl,
                     (intcon + 1.0).reshape(1, -1) if intcon.size else e, options]
                res = interp.call_function_value(char("ga"), a, 4)
            elif solver == "patternsearch":
                if x0 is None:
                    raise MatlabError("solve: patternsearch needs an initial point, solve(prob, x0).")
                a = [handle("f"), x0.reshape(1, -1), _mat(A), _col(b), _mat(Aeq), _col(beq), _col(lo).T, _col(hi).T,
                     nonl, options]
                res = interp.call_function_value(char("patternsearch"), a, 4)
            else:
                if A.shape[0] or Aeq.shape[0] or nonlin:
                    raise MatlabError(f"solve: the solver '{solver}' handles bound constraints only.")
                if solver == "particleswarm":
                    res = interp.call_function_value(char("particleswarm"), [handle("f"), np.array([[float(n)]]),
                                                                             _col(lo).T, _col(hi).T, options], 4)
                elif solver == "surrogateopt":
                    res = interp.call_function_value(char("surrogateopt"), [handle("f"), _col(lo).T, _col(hi).T,
                                                                            options], 4)
                else:
                    if x0 is None:
                        raise MatlabError("solve: simulannealbnd needs an initial point, solve(prob, x0).")
                    res = interp.call_function_value(char("simulannealbnd"), [handle("f"), x0.reshape(1, -1),
                                                                              _col(lo).T, _col(hi).T, options], 4)
        finally:
            interp.state["_optim_shims"].pop(key, None)
        x = np.real(fnum(res[0])).flatten()
        flag = float(fnum(res[2]).flat[0])
        out = res[3]
        lam_struct = Struct.scalar({})
    fval = float(np.asarray(obj.ev(vals_of(x)), dtype=float).flat[0])
    return _unpack(x, infos, offs), fval, flag, out, lam_struct


@builtin("optimProblemShim__")
def _optim_problem_shim(interp, args, nargout):
    f, nl, n = interp.state.get("_optim_shims", {}).get(text(args[0]), (None, None, 0))
    if f is None:
        raise MatlabError("This function handle belongs to a solve call that has finished.")
    x = np.real(fnum(args[2])).astype(float).flatten()
    if text(args[1]) == "f":
        return f(x)
    c, ceq = nl(x)
    return c.reshape(-1, 1), ceq.reshape(-1, 1)


def _solve_equations(interp, prob, info, x0, solver, options):
    infos, offs, n = info["infos"], info["offs"], info["n"]
    A, b, Aeq, beq, lo, hi, _, _ = _matrices(info)
    eqs = prob.named_constraints()
    opts = om.Opt(options, solver if solver in om.SOLVER_OPTIONS else "fsolve")
    rep = om.Report(interp, opts)

    def vals_of(x):
        return {v: x[offs[v]:offs[v] + v.size].reshape(v.shape, order="F") for v in infos}

    def resid(x):
        vals = vals_of(x)
        return np.concatenate([np.asarray(c.node.ev(vals), dtype=float).reshape(-1, order="F") for _, c in eqs])

    bounded = bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))
    interp.out(f"Solving problem using {solver}.\n")
    if info["lin_cons"] and solver in ("lsqlin",):
        if bounded:
            from ._optim_local import solve_qp
            x, _, _, _ = solve_qp(Aeq.T @ Aeq + 1e-14 * np.eye(n), -(Aeq.T @ beq), np.zeros((0, n)), np.zeros(0),
                                  np.zeros((0, n)), np.zeros(0), lo, hi)
        else:
            x = np.linalg.lstsq(Aeq, beq, rcond=None)[0]
        r = resid(x)
        flag = 1 if np.max(np.abs(r), initial=0.0) <= 1e-6 * max(1.0, float(np.max(np.abs(beq), initial=0.0))) else -2
        iters = 1
    else:
        if x0 is None:
            raise MatlabError("solve: a nonlinear equation problem needs an initial point, solve(prob, x0).")
        from scipy.optimize import least_squares
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bounded or resid(x0).size < n:
                res = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
            else:
                res = least_squares(resid, x0, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        x = res.x
        r = resid(x)
        iters = int(res.nfev)
        flag = 1 if float(r @ r) <= np.sqrt(opts.num("FunctionTolerance", 1e-6)) else -2
    msg = "Equation solved." if flag == 1 else "No solution found."
    rep.finish(flag, msg)
    out = om.output_struct(iterations=iters, funcCount=iters, algorithm="levenberg-marquardt", message=msg)
    if len(eqs) == 1 and isinstance(prob.constraints, OptimConstraint):
        fval = r.reshape(eqs[0][1].node.shape, order="F")
    else:
        d, k = {}, 0
        for nm, c in eqs:
            d[nm] = r[k:k + c.node.m].reshape(c.node.shape, order="F")
            k += c.node.m
        fval = Struct.scalar(d)
    return _unpack(x, infos, offs), fval, flag, out


def _prob2struct(interp, prob, rest):
    info = _classify(prob)
    infos, offs, n = info["infos"], info["offs"], info["n"]
    if prob.equation or info["obj_p"] is None or not info["lin_cons"]:
        raise MatlabError("prob2struct supports linear and quadratic problems only.")
    A, b, Aeq, beq, lo, hi, intcon, _ = _matrices(info)
    H, f, c0 = _quad_objective(info["obj_p"], infos, offs, n)
    sgn = -1.0 if prob.sense == "maximize" else 1.0
    solver = info["solver"]
    d = {}
    if solver == "quadprog":
        d["H"] = sgn * H
    d["f"] = (sgn * f).reshape(-1, 1)
    if solver == "intlinprog":
        d["intcon"] = (intcon + 1.0).reshape(1, -1)
    d.update({"Aineq": _mat(A), "bineq": _col(b), "Aeq": _mat(Aeq), "beq": _col(beq), "lb": _col(lo),
              "ub": _col(hi), "x0": empty_double(), "solver": char(solver),
              "options": om.OptimOptions(solver), "f0": np.array([[sgn * c0]])})
    return Struct.scalar(d)
