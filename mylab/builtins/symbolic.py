"""Symbolic math: sym, syms and the sym object, backed by SymPy.

SymPy is an optional dependency. It is imported the first time a symbolic function
runs (:func:`sympy`), so ``import mylab`` stays fast and works without it.

A :class:`SymObject` is a 2-D array of SymPy expressions (a scalar is 1-by-1). A
:class:`SymFun` is a symbolic function ``f(x, y)``: the same array plus its argument
list. Functions for sym values are kept in :data:`SYM`; the interpreter calls them
through the object's ``has_method``/``call_method`` hooks, so ``diff(f, x)`` or
``sin(x)`` reach the symbolic version whenever an argument is a sym.

This module holds the object, the MATLAB-style printer, creation (``syms``, ``sym``,
assumptions), element-wise functions and the bridges to numbers and text. Calculus,
algebra, solvers, transforms and linear algebra are in :mod:`symcalc`.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any, Callable

import numpy as np

from .. import ast as A
from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import (
    Cell, FunctionHandle, MString, StringArray, char, to_matlab, to_str,
)
from ._registry import BUILTINS
from ._util import bool_scalar, is_text, text

TIMES = "×"
_sympy = None
INSTALL_HINT = ("Symbolic math needs the SymPy package, which is not installed. "
                'Install it with:  pip install "mylab-lang[symbolic]"')


def sympy():
    """The sympy module, imported on first use."""
    global _sympy
    if _sympy is None:
        try:
            import sympy as module
        except ImportError:
            raise MatlabError(INSTALL_HINT, "mylab:symbolic:sympyMissing") from None
        _sympy = module
    return _sympy


# -- function table ---------------------------------------------------------------------------

SYM: dict[str, Callable] = {}
# Functions whose result keeps the argument list of a symbolic function input.
KEEP_FUN: set[str] = set()


def _guard(fn: Callable) -> Callable:
    def guarded(interp, args, nargout):
        sympy()
        return fn(interp, args, nargout)
    guarded.__name__ = getattr(fn, "__name__", "symbolic")
    return guarded


def symfn(*names: str, keep: bool = False):
    """Register a function for sym arguments; also as a built-in when the name is free."""

    def deco(fn: Callable) -> Callable:
        for n in names:
            SYM[n] = fn
            if keep:
                KEEP_FUN.add(n)
            if n not in BUILTINS:
                BUILTINS[n] = _guard(fn)
        return fn

    return deco


# -- numbers -> exact values ------------------------------------------------------------------

def num2sym(x: Any):
    """A Python/NumPy number as an exact SymPy value (1/3, pi/2 and sqrt(2) are recognised)."""
    sp = sympy()
    if isinstance(x, (bool, np.bool_)):
        return sp.Integer(int(x))
    if isinstance(x, (int, np.integer)):
        return sp.Integer(int(x))
    if isinstance(x, (complex, np.complexfloating)):
        if x.imag == 0:
            return num2sym(float(x.real))
        return num2sym(float(x.real)) + sp.I * num2sym(float(x.imag))
    x = float(x)
    if x != x:
        return sp.nan
    if math.isinf(x):
        return sp.oo if x > 0 else -sp.oo
    if x == int(x) and abs(x) < 1e15:
        return sp.Integer(int(x))
    fr = Fraction(x).limit_denominator(10 ** 12)
    if float(fr) == x and abs(fr.numerator) * fr.denominator < 2 ** 44:
        return sp.Rational(fr.numerator, fr.denominator)
    fr = Fraction(x / math.pi).limit_denominator(10000)
    if fr != 0 and abs(float(fr) * math.pi - x) <= 4e-16 * abs(x):
        return sp.pi * sp.Rational(fr.numerator, fr.denominator)
    fr = Fraction(x * x).limit_denominator(10000)
    if fr > 0 and abs(math.sqrt(float(fr)) - abs(x)) <= 4e-16 * abs(x):
        root = sp.sqrt(sp.Rational(fr.numerator, fr.denominator))
        return root if x > 0 else -root
    return sp.Rational(x)


# -- the MATLAB-style printer -----------------------------------------------------------------

_FUNC_NAMES = {
    "Abs": "abs", "ceiling": "ceil", "re": "real", "im": "imag", "conjugate": "conj", "arg": "angle",
    "Max": "max", "Min": "min", "Mod": "mod", "binomial": "nchoosek", "loggamma": "gammaln",
    "polygamma": "psi", "digamma": "psi", "Si": "sinint", "Ci": "cosint", "Ei": "ei", "li": "logint",
    "KroneckerDelta": "kroneckerDelta", "uppergamma": "igamma", "Shi": "sinhint", "Chi": "coshint",
    "fresnels": "fresnels", "fresnelc": "fresnelc", "elliptic_k": "ellipticK", "elliptic_e": "ellipticE",
}
_printers: dict = {}


def _make_printers() -> None:
    sp = sympy()
    from sympy.printing.precedence import PRECEDENCE, precedence
    from sympy.printing.str import StrPrinter

    class SymPrinter(StrPrinter):
        MUL, DIV, POW = "*", "/", "^"
        vector = False

        def _neg_exp(self, e) -> bool:
            if e.is_Number:
                return bool(e.is_negative)
            return e.is_Mul and e.args[0].is_Number and bool(e.args[0].is_negative)

        def _print_Mul(self, expr):
            c, rest = expr.as_coeff_Mul()
            sign = ""
            if c.is_real and c.is_negative:
                sign, c = "-", -c
            num, den = [], []
            if c != 1:
                if c.is_Rational and not c.is_Integer:
                    if c.p != 1:
                        num.append(sp.Integer(c.p))
                    den.append(sp.Integer(c.q))
                else:
                    num.append(c)
            imag = False
            for f in (rest.as_ordered_factors() if rest.is_Mul else [rest]):
                if f is sp.I:
                    imag = True
                elif f.is_Pow and f.base is not sp.E and self._neg_exp(f.exp):
                    den.append(f.base if f.exp == -1 else sp.Pow(f.base, -f.exp, evaluate=False))
                elif f != 1:
                    num.append(f)
            level = PRECEDENCE["Mul"]
            parts = [self.parenthesize(f, level, strict=False) for f in num]
            if imag:
                if len(num) == 1 and num[0].is_Integer and not self.vector:
                    parts = [parts[0] + "i"]
                else:
                    parts.append("1i")
            if not den:
                return sign + self.MUL.join(parts)
            top = "1" if not parts else parts[0] if len(parts) == 1 else "(" + self.MUL.join(parts) + ")"
            dparts = [self.parenthesize(f, level, strict=False) for f in den]
            bottom = dparts[0] if len(dparts) == 1 else "(" + self.MUL.join(dparts) + ")"
            return sign + top + self.DIV + bottom

        def _print_Pow(self, expr, rational=False):
            b, e = expr.base, expr.exp
            if b is sp.E:
                return f"exp({self._print(e)})"
            if self._neg_exp(e):
                inv = b if e == -1 else sp.Pow(b, -e, evaluate=False)
                return "1" + self.DIV + self.parenthesize(inv, PRECEDENCE["Mul"], strict=False)
            if self.vector and e == sp.Rational(1, 2):
                return f"sqrt({self._print(b)})"
            bs = self.parenthesize(b, PRECEDENCE["Pow"], strict=True)
            if (e.is_Integer and e.is_nonnegative) or e.is_Symbol:
                es = self._print(e)
            else:
                es = "(" + self._print(e) + ")"
            return bs + self.POW + es

        def _degree(self, term) -> int:
            try:
                return sp.Poly(term).total_degree() if term.free_symbols else 0
            except (sp.PolynomialError, sp.GeneratorsError, ValueError):
                return 0

        def _print_Add(self, expr, order=None):
            parts = self._as_ordered_terms(expr, order=order)
            terms = [self._print(t) if precedence(t) >= PRECEDENCE["Add"] else f"({self._print(t)})"
                     for t in parts]
            if terms[0].startswith("-"):
                # Lead with a positive term (b^2 - 4*a, not -4*a + b^2) unless that would put a
                # lower power first (- x^3/6 + x stays in descending order; constants may lead: 1 - x).
                k = next((i for i, t in enumerate(terms) if not t.startswith("-")), None)
                if k is not None and (parts[k].is_number or self._degree(parts[k]) >= self._degree(parts[0])):
                    terms.insert(0, terms.pop(k))
            out = "- " + terms[0][1:] if terms[0].startswith("-") else terms[0]
            for t in terms[1:]:
                out += " - " + t[1:] if t.startswith("-") else " + " + t
            return out

        def _print_Rational(self, expr):
            return f"{expr.p}/{expr.q}"

        def _print_Float(self, expr):
            if self.vector:
                return repr(float(expr))
            from mpmath.libmp import prec_to_dps, to_str
            dps = prec_to_dps(expr._prec)
            s = to_str(expr._mpf_, dps, strip_zeros=True, min_fixed=-6, max_fixed=dps + 4)
            if s.endswith(".") or ("." not in s and "e" not in s):
                s = s.rstrip(".") + ".0"
            return s

        def _print_ImaginaryUnit(self, expr):
            return "1i"

        def _print_Infinity(self, expr):
            return "Inf"

        def _print_NegativeInfinity(self, expr):
            return "-Inf"

        def _print_ComplexInfinity(self, expr):
            return "Inf"

        def _print_NaN(self, expr):
            return "NaN"

        def _print_Exp1(self, expr):
            return "exp(1)"

        def _print_EulerGamma(self, expr):
            return "eulergamma"

        def _print_BooleanTrue(self, expr):
            return "true" if self.vector else "symtrue"

        def _print_BooleanFalse(self, expr):
            return "false" if self.vector else "symfalse"

        def _print_Relational(self, expr):
            ops = {"==": "==", "!=": "~=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
            boolean = (sp.core.relational.Relational, sp.logic.boolalg.BooleanFunction)
            lhs, rhs = [f"({self._print(a)})" if isinstance(a, boolean) else self._print(a)
                        for a in (expr.lhs, expr.rhs)]
            return f"{lhs} {ops.get(expr.rel_op, expr.rel_op)} {rhs}"

        def _print_And(self, expr):
            return " & ".join(self.parenthesize(a, PRECEDENCE["And"]) for a in expr.args)

        def _print_Or(self, expr):
            return " | ".join(self.parenthesize(a, PRECEDENCE["Or"]) for a in expr.args)

        def _print_Not(self, expr):
            return "~" + self.parenthesize(expr.args[0], PRECEDENCE["Not"])

        def _print_Function(self, expr):
            name = expr.func.__name__
            name = _FUNC_NAMES.get(name, name)
            return f"{name}({', '.join(self._print(a) for a in expr.args)})"

        def _print_Heaviside(self, expr):
            return f"heaviside({self._print(expr.args[0])})"

        def _print_DiracDelta(self, expr):
            if len(expr.args) > 1 and expr.args[1] != 0:
                return f"dirac({self._print(expr.args[1])}, {self._print(expr.args[0])})"
            return f"dirac({self._print(expr.args[0])})"

        def _print_LambertW(self, expr):
            k = expr.args[1] if len(expr.args) > 1 else sp.Integer(0)
            return f"lambertw({self._print(k)}, {self._print(expr.args[0])})"

        def _print_expint(self, expr):
            if expr.args[0] == 1:
                return f"expint({self._print(expr.args[1])})"
            return f"expint({self._print(expr.args[0])}, {self._print(expr.args[1])})"

        def _print_Piecewise(self, expr):
            items = []
            for value, cond in expr.args:
                if cond is sp.true:
                    if value is not sp.nan or self.vector:  # NaN = "no otherwise value given"
                        items.append(self._print(value))
                else:
                    items += [self._print(cond), self._print(value)]
            return f"piecewise({', '.join(items)})"

        def _print_Derivative(self, expr):
            return f"diff({self._print(expr.expr)}, {', '.join(self._print(v) for v in expr.variables)})"

        def _print_Integral(self, expr):
            s = self._print(expr.function)
            for lim in expr.limits:
                s = f"int({s}, {', '.join(self._print(a) for a in lim)})"
            return s

        def _print_Sum(self, expr):
            s = self._print(expr.function)
            for lim in expr.limits:
                s = f"symsum({s}, {', '.join(self._print(a) for a in lim)})"
            return s

        def _print_Product(self, expr):
            s = self._print(expr.function)
            for lim in expr.limits:
                s = f"symprod({s}, {', '.join(self._print(a) for a in lim)})"
            return s

        def _print_Limit(self, expr):
            e, z, z0, direction = expr.args
            side = {"+": ", 'right'", "-": ", 'left'"}.get(str(direction), "")
            return f"limit({self._print(e)}, {self._print(z)}, {self._print(z0)}{side})"

        def _print_Subs(self, expr):
            e, old, new = expr.args
            if len(old) == 1:
                return f"subs({self._print(e)}, {self._print(old[0])}, {self._print(new[0])})"
            return (f"subs({self._print(e)}, [{', '.join(self._print(a) for a in old)}], "
                    f"[{', '.join(self._print(a) for a in new)}])")

        def _print_LaplaceTransform(self, expr):
            return f"laplace({', '.join(self._print(a) for a in expr.args[:3])})"

        def _print_InverseLaplaceTransform(self, expr):
            return f"ilaplace({', '.join(self._print(a) for a in expr.args[:3])})"

        def _print_FourierTransform(self, expr):
            return f"fourier({', '.join(self._print(a) for a in expr.args[:3])})"

        def _print_InverseFourierTransform(self, expr):
            return f"ifourier({', '.join(self._print(a) for a in expr.args[:3])})"

        def _print_ComplexRootOf(self, expr):
            z = sp.Symbol("z")
            return f"root({self._print(expr.expr.subs(expr.poly.gen, z))}, z, {expr.index + 1})"

        def _print_Dummy(self, expr):
            return expr.name

        def _print_Tuple(self, expr):
            return "[" + ", ".join(self._print(a) for a in expr.args) + "]"

    class VectorPrinter(SymPrinter):
        MUL, DIV, POW = ".*", "./", ".^"
        vector = True

        def _print_Rational(self, expr):
            return f"({expr.p}/{expr.q})"

        def _print_Integer(self, expr):
            return str(expr.p)

    _printers["plain"] = SymPrinter({"order": None})
    _printers["vector"] = VectorPrinter({"order": None})


def sym_str(expr: Any, vector: bool = False) -> str:
    """Text of one SymPy expression in MATLAB notation."""
    if not _printers:
        _make_printers()
    return _printers["vector" if vector else "plain"].doprint(expr)


# -- the sym object ---------------------------------------------------------------------------

def _one(e: Any) -> np.ndarray:
    out = np.empty((1, 1), dtype=object)
    out[0, 0] = e
    return out


def _obj_array(items: list, shape: tuple) -> np.ndarray:
    """Object array of ``shape`` filled in column-major order."""
    flat = np.empty(len(items), dtype=object)
    for i, e in enumerate(items):
        flat[i] = e
    return flat.reshape(shape, order="F")


def _size_text(shape: tuple) -> str:
    return TIMES.join(str(d) for d in shape)


class SymObject(MObject):
    """A symbolic scalar or matrix: a 2-D object array of SymPy expressions."""

    is_sym = True

    def __init__(self, data: np.ndarray):
        self.data = data

    # duck-typed array attributes (used by for-loops and generic helpers)
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __getitem__(self, key):
        return SymObject(np.asarray(self.data[key], dtype=object))

    def empty_like(self) -> "SymObject":
        return SymObject(np.empty((0, 0), dtype=object))

    def like(self, data: np.ndarray) -> "SymObject":
        return SymObject(data)

    @property
    def expr(self):
        """The single expression of a scalar sym."""
        if self.data.size != 1:
            raise MatlabError("A scalar symbolic expression is expected here, not a "
                              f"{_size_text(self.data.shape)} symbolic matrix.")
        return self.data.flat[0]

    # -- identity -----------------------------------------------------------------------------
    def class_name(self) -> str:
        return "sym"

    def isa_class(self, name: str) -> bool:
        return name == "sym"

    def matlab_shape(self) -> tuple:
        return tuple(int(d) for d in self.data.shape)

    def object_equals(self, other: Any) -> bool:
        try:
            b = to_data(other)
        except MatlabError:
            return False
        a = self.data
        return a.shape == b.shape and all(x == y for x, y in zip(a.flat, b.flat))

    def to_dense(self) -> np.ndarray:
        """The numeric array of a sym without variables (lets numeric functions accept sym(2))."""
        return to_double(self)

    def truth_value(self) -> bool:
        """Used by if/while: every element must be decidably true (nonzero)."""
        if self.data.size == 0:
            return False
        return all(_truth(e) for e in self.data.flat)

    # -- methods ------------------------------------------------------------------------------
    def has_method(self, name: str) -> bool:
        return name in SYM

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        res = SYM[name](interp, list(args), nargout)
        if res is None:
            return []
        out = list(res) if isinstance(res, tuple) else [res]
        if name in KEEP_FUN:
            fun = next((a for a in args if isinstance(a, SymFun)), None)
            if fun is not None:
                out = [SymFun(r.data, fun.vars) if type(r) is SymObject else r for r in out]
        return [to_matlab(r) for r in out]

    def get_field(self, interp, name: str) -> Any:
        raise MatlabError(f"Dot indexing is not supported for variables of type '{self.class_name()}'.")

    # -- operators ----------------------------------------------------------------------------
    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        res = binop(op, left, right)
        for side in (left, right):
            if isinstance(side, SymFun):
                return SymFun(res.data, side.vars)
        return res

    def unary_op(self, interp, op: str) -> Any:
        sp = sympy()
        if op == "-":
            return self.like(_map1(lambda e: -e, self.data))
        if op == "+":
            return self
        if op in ("~", "!"):
            return self.like(_map1(lambda e: sp.Not(_as_bool(e)), self.data))
        if op == "'":
            return self.like(_map1(sp.conjugate, self.data.T.copy()))
        if op == ".'":
            return self.like(self.data.T.copy())
        return NotImplemented

    # -- indexing -----------------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        return SymObject(ix.get(self.data, [_plain_index(a) for a in args]))

    def index_set(self, interp, args: list, value: Any) -> Any:
        fvars = _function_vars(args)
        if fvars is not None:  # f(x, y) = expression defines a symbolic function
            return SymFun(to_data(value), fvars)
        sp = sympy()
        rhs = to_data(value)
        data = ix.set_(self.data.copy(), [_plain_index(a) for a in args], rhs, lambda: sp.Integer(0))
        if data.ndim > 2:
            raise MatlabError("Symbolic arrays with more than two dimensions are not supported.")
        return SymObject(data)

    def index_delete(self, interp, args: list) -> Any:
        return SymObject(ix.delete(self.data, [_plain_index(a) for a in args]))

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        if axis > 1:
            raise MatlabError("Symbolic arrays with more than two dimensions are not supported.")
        parts = [to_data(i) for i in items if not (isinstance(i, np.ndarray) and i.size == 0)]
        parts = [p for p in parts if p.size]
        if not parts:
            return SymObject(np.empty((0, 0), dtype=object))
        other = 1 - axis
        if len({p.shape[other] for p in parts}) > 1:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch")
        return SymObject(np.concatenate(parts, axis=axis))

    # -- display ------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        s = f"{_size_text(self.matlab_shape())} {self.class_name()}"
        return s if in_cell else f"[{s}]"

    def display_lines(self) -> list[str]:
        d = self.data
        if d.size == 0:
            return [f"Empty sym: {d.shape[0]}-by-{d.shape[1]}"]
        if d.shape == (1, 1):
            return [sym_str(d[0, 0])]
        cells = [[sym_str(e) for e in row] for row in d]
        widths = [max(len(cells[i][j]) for i in range(d.shape[0])) for j in range(d.shape[1])]
        return ["[" + ", ".join(c.rjust(w) for c, w in zip(row, widths)) + "]" for row in cells]

    def display_head(self, name: str) -> str:
        return name

    def format_display(self, name: str) -> str:
        return f"{self.display_head(name)} =\n \n" + "\n".join(self.display_lines()) + "\n \n"


class SymFun(SymObject):
    """A symbolic function ``f(x, y)``: an expression plus its ordered arguments."""

    def __init__(self, data: np.ndarray, fvars: list):
        super().__init__(data)
        self.vars = list(fvars)

    def like(self, data: np.ndarray) -> "SymObject":
        return SymFun(data, self.vars)

    def class_name(self) -> str:
        return "symfun"

    def isa_class(self, name: str) -> bool:
        return name in ("symfun", "sym")

    def index_get(self, interp, args: list) -> Any:
        if len(args) != len(self.vars):
            raise MatlabError(f"This symbolic function takes {len(self.vars)} argument"
                              f"{'s' if len(self.vars) != 1 else ''}, but {len(args)} were given.")
        vals = [to_data(a) for a in args]
        if all(v.size == 1 for v in vals):
            mapping = {v: val.flat[0] for v, val in zip(self.vars, vals)}
            return SymObject(_map1(lambda e: _subs(e, mapping), self.data))
        try:
            bcast = np.broadcast(*vals)
        except ValueError:
            raise MatlabError("The arguments of a symbolic function must have compatible sizes.") from None
        body = self.expr
        out = [_subs(body, dict(zip(self.vars, point))) for point in bcast]
        return SymObject(_obj_array(out, bcast.shape[::-1]).T.copy() if len(bcast.shape) == 2
                         else _obj_array(out, (1, len(out))))

    def index_set(self, interp, args: list, value: Any) -> Any:
        fvars = _function_vars(args)
        if fvars is None:
            raise MatlabError("To redefine a symbolic function, use symbolic variables as the "
                              "arguments, e.g. f(x) = x^2.")
        return SymFun(to_data(value), fvars)

    def index_delete(self, interp, args: list) -> Any:
        raise MatlabError("Cannot delete elements of a symbolic function.")

    def display_head(self, name: str) -> str:
        return f"{name}({', '.join(sym_str(v) for v in self.vars)})"


def _plain_index(a: Any) -> Any:
    if isinstance(a, SymObject):
        return to_double(a)
    return a


def _function_vars(args: list) -> list | None:
    """The argument symbols when every index is a scalar symbolic variable, else None."""
    if not args or not all(isinstance(a, SymObject) and a.data.size == 1 for a in args):
        return None
    syms = [a.data.flat[0] for a in args]
    return syms if all(getattr(s, "is_Symbol", False) for s in syms) else None


def define_function(args: list, value: Any) -> Any:
    """``f(x) = value`` for a new name ``f`` (called by the interpreter); None if not a definition."""
    fvars = _function_vars(args)
    return None if fvars is None else SymFun(to_data(value), fvars)


def is_logic(e: Any) -> bool:
    """True for conditions (equations, inequalities, and/or/not, true/false), not for plain symbols."""
    sp = sympy()
    return isinstance(e, (sp.core.relational.Relational, sp.logic.boolalg.BooleanFunction,
                          sp.logic.boolalg.BooleanAtom))


def _truth(e: Any) -> bool:
    sp = sympy()
    if e is sp.true:
        return True
    if e is sp.false:
        return False
    if isinstance(e, sp.Eq):
        return bool(e.lhs == e.rhs or sp.simplify(e.lhs - e.rhs) == 0)
    if isinstance(e, sp.Ne):
        return not (e.lhs == e.rhs or sp.simplify(e.lhs - e.rhs) == 0)
    if is_logic(e):
        val = sp.simplify(e.doit() if hasattr(e, "doit") else e)
        if val is sp.true or val is sp.false:
            return val is sp.true
        raise MatlabError(f"Unable to prove '{sym_str(e)}' literally. Use isAlways to test the "
                          "statement mathematically.")
    if e.free_symbols:
        raise MatlabError("A symbolic expression with variables cannot be used as a condition. "
                          "Use isAlways, or substitute numbers first with subs.")
    return bool(e != 0)


def _as_bool(e: Any):
    sp = sympy()
    if is_logic(e):
        return e
    return sp.Ne(e, 0)


# -- conversion -------------------------------------------------------------------------------

def wrap(v: Any) -> SymObject:
    """A SymObject from a SymPy expression, SymPy matrix or object array."""
    if isinstance(v, SymObject):
        return v
    if isinstance(v, np.ndarray) and v.dtype == object:
        return SymObject(v if v.ndim == 2 else v.reshape(1, -1))
    sp = sympy()
    if isinstance(v, sp.MatrixBase):
        return SymObject(_obj_array(list(v.T), (v.rows, v.cols)) if v.rows * v.cols
                         else np.empty((v.rows, v.cols), dtype=object))
    return SymObject(_one(sp.sympify(v)))


def column(items: list) -> SymObject:
    return SymObject(_obj_array(list(items), (len(items), 1)))


def row(items: list) -> SymObject:
    return SymObject(_obj_array(list(items), (1, len(items))))


def to_data(v: Any, interp: Any = None) -> np.ndarray:
    """Any value that can become a sym, as a 2-D object array of SymPy expressions."""
    if isinstance(v, SymObject):
        return v.data
    sp = sympy()
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "U":
            return parse_text("".join(v.flatten().tolist()), interp).data
        if v.dtype == object:
            return v
        if v.ndim > 2:
            raise MatlabError("Symbolic arrays with more than two dimensions are not supported.")
        out = np.empty(v.shape, dtype=object)
        for idx in np.ndindex(v.shape):
            out[idx] = num2sym(v[idx].item())
        return out
    if isinstance(v, MString):
        return parse_text(str(v), interp).data
    if isinstance(v, sp.MatrixBase):
        return wrap(v).data
    if isinstance(v, sp.Basic):
        return _one(v)
    if isinstance(v, (bool, int, float, complex, np.number, np.bool_)):
        return _one(num2sym(v))
    if hasattr(v, "to_dense"):
        return to_data(v.to_dense())
    if isinstance(v, Cell):
        parts = [to_data(x, interp) for x in v.data.flat]
        if all(p.size == 1 for p in parts):
            out = np.empty(v.data.shape, dtype=object)
            for i, p in enumerate(parts):
                out.flat[i] = p.flat[0]
            return out
    from ..values import class_name
    raise MatlabError(f"Cannot convert a value of class '{class_name(v)}' to sym.")


def scalar(v: Any, what: str = "argument"):
    """The single SymPy expression of a scalar value."""
    d = to_data(v)
    if d.size != 1:
        raise MatlabError(f"The {what} must be a scalar, not a {_size_text(d.shape)} array.")
    return d.flat[0]


def matrix(v: Any):
    """A SymPy Matrix from any sym-like value."""
    sp = sympy()
    d = to_data(v)
    return sp.Matrix(d.shape[0], d.shape[1], [d[i, j] for i in range(d.shape[0]) for j in range(d.shape[1])])


def flat_list(v: Any) -> list:
    """Elements in column-major order."""
    return list(to_data(v).flatten(order="F"))


def _map1(fn: Callable, a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape, dtype=object)
    for idx in np.ndindex(a.shape):
        out[idx] = fn(a[idx])
    return out


def map_data(fn: Callable, *vals: Any) -> SymObject:
    """Apply a scalar function element by element, with implicit expansion."""
    datas = [to_data(v) for v in vals]
    if len(datas) == 1:
        return SymObject(_map1(fn, datas[0]))
    try:
        shape = np.broadcast_shapes(*[d.shape for d in datas])
    except ValueError:
        raise MatlabError("Arrays have incompatible sizes for this operation.",
                          "MATLAB:sizeDimensionsMustMatch") from None
    bs = [np.broadcast_to(d, shape) for d in datas]
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        out[idx] = fn(*[b[idx] for b in bs])
    return SymObject(out)


def _subs(e: Any, mapping: dict):
    if not mapping:
        return e
    return e.subs(mapping, simultaneous=True) if len(mapping) > 1 else e.subs(mapping)


def to_double(v: Any) -> np.ndarray:
    """Numeric array of a sym without free variables."""
    sp = sympy()
    d = to_data(v)
    out = np.empty(d.shape, dtype=complex)
    for idx in np.ndindex(d.shape):
        e = d[idx]
        if e is sp.true or e is sp.false:
            out[idx] = 1.0 if e is sp.true else 0.0
            continue
        if is_logic(e):
            out[idx] = 1.0 if _truth(e) else 0.0
            continue
        if e.free_symbols:
            raise MatlabError("Unable to convert an expression containing symbolic variables into a "
                              "double array. Use subs to substitute values for the variables first.",
                              "symbolic:double:cantconvert")
        if e is sp.zoo:
            out[idx] = np.inf
            continue
        try:
            val = complex(e.evalf(17))
        except (TypeError, ValueError):
            raise MatlabError(f"Unable to convert '{sym_str(e)}' to a number.") from None
        out[idx] = val
    if not np.any(out.imag != 0):
        return out.real.copy()
    return out


# -- operators --------------------------------------------------------------------------------

def _relation(kind: str) -> Callable:
    def f(a, b):
        sp = sympy()
        cls = {"==": sp.Eq, "~=": sp.Ne, "<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[kind]
        if not getattr(a, "free_symbols", True) and not getattr(b, "free_symbols", True):
            try:
                res = cls(a, b)
                if res is sp.true or res is sp.false:
                    return res
            except TypeError:
                pass
        if kind in ("==", "~="):
            return cls(a, b, evaluate=False)
        try:
            return cls(a, b, evaluate=False)
        except TypeError:
            raise MatlabError("Symbolic inequalities need real-valued operands.") from None
    return f


def _divide(a, b):
    sp = sympy()
    if b == 0:
        if a == 0:
            return sp.nan
        if a.is_number and a.is_real:
            return sp.oo if a > 0 else -sp.oo
        return sp.zoo * a if a.is_number else sp.Mul(a, sp.zoo)
    return a / b


_ELEMENT_OPS: dict[str, Callable] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    ".*": lambda a, b: a * b,
    "./": _divide,
    ".\\": lambda a, b: _divide(b, a),
    ".^": lambda a, b: a ** b,
    "&": lambda a, b: sympy().And(_as_bool(a), _as_bool(b)),
    "|": lambda a, b: sympy().Or(_as_bool(a), _as_bool(b)),
}
for _kind in ("==", "~=", "<", "<=", ">", ">="):
    _ELEMENT_OPS[_kind] = _relation(_kind)
_ELEMENT_OPS["!="] = _ELEMENT_OPS["~="]


def binop(op: str, left: Any, right: Any) -> SymObject:
    sp = sympy()
    a, b = to_data(left), to_data(right)
    fn = _ELEMENT_OPS.get(op)
    if fn is None:
        if op == "*":
            if a.size == 1 or b.size == 1:
                fn = _ELEMENT_OPS[".*"]
            else:
                if a.shape[1] != b.shape[0]:
                    raise MatlabError(
                        "Incorrect dimensions for matrix multiplication. Check that the number of columns "
                        "in the first matrix matches the number of rows in the second matrix. To multiply "
                        "element by element, use '.*'.", "MATLAB:innerdim")
                return wrap(matrix(a) * matrix(b))
        elif op == "/":
            if b.size == 1:
                fn = _ELEMENT_OPS["./"]
            else:
                return wrap(matrix(b).T.solve(matrix(a).T).T) if a.shape[1] == b.shape[1] else _dim_error()
        elif op == "\\":
            if a.size == 1:
                fn = _ELEMENT_OPS[".\\"]
            else:
                if a.shape[0] != b.shape[0]:
                    _dim_error()
                return wrap(solve_linear(matrix(a), matrix(b)))
        elif op == "^":
            if a.size == 1 and b.size == 1:
                fn = _ELEMENT_OPS[".^"]
            elif b.size == 1 and a.shape[0] == a.shape[1]:
                return wrap(matrix(a) ** b.flat[0])
            elif a.size == 1 and b.shape[0] == b.shape[1]:
                return wrap((matrix(b) * sp.log(a.flat[0])).exp())
            else:
                raise MatlabError("For a matrix power the matrix must be square. To raise each element "
                                  "to a power, use '.^'.")
        else:
            raise MatlabError(f"Operator '{op}' is not supported for operands of type 'sym'.")
    return map_data(fn, a, b)


def _dim_error():
    raise MatlabError("Matrix dimensions must agree.", "MATLAB:dimagree")


def solve_linear(a, b):
    """Solution of a*x = b for SymPy matrices (least-squares form when a is not square)."""
    sp = sympy()
    if a.rows == a.cols:
        try:
            return a.LUsolve(b).applyfunc(sp.cancel)
        except (ValueError, sp.matrices.exceptions.NonInvertibleMatrixError):
            pass
    try:
        sol, params = a.gauss_jordan_solve(b)
    except ValueError:
        raise MatlabError("The symbolic linear system has no solution.") from None
    return sol.subs({p: 0 for p in params}) if params else sol


# -- parsing text -----------------------------------------------------------------------------

_CONSTANTS = {"pi": "pi", "Inf": "oo", "inf": "oo", "NaN": "nan", "nan": "nan", "i": "I", "j": "I",
              "eulergamma": "EulerGamma", "catalan": "Catalan"}
_ASSUME_WORDS = {"real", "positive", "integer", "rational", "clear"}


def _registry(interp) -> dict:
    if interp is None:
        return {}
    return interp.state.setdefault("sym_assumptions", {})


def make_symbol(interp, name: str):
    """The symbol called ``name`` with the assumptions currently set for it."""
    sp = sympy()
    entry = _registry(interp).get(name)
    return sp.Symbol(name, **entry["kw"]) if entry else sp.Symbol(name)


def parse_text(source: str, interp: Any = None) -> SymObject:
    """A sym from text such as ``'x^2 + 1/3'``: numbers stay exact, unknown names become symbols."""
    from ..parser import parse
    sp = sympy()
    src = source.strip()
    if re.fullmatch(r"[+-]?\d+", src):
        return wrap(sp.Integer(src))
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)", src):
        return wrap(sp.Rational(src))
    src = re.sub(r"(?<![=<>~!])=(?!=)", "==", src)
    try:
        program = parse(src)
    except MatlabError as e:
        raise MatlabError(f"Cannot turn '{source}' into a symbolic expression: {e}") from None
    if len(program.body) != 1 or not isinstance(program.body[0], A.ExprStmt):
        raise MatlabError(f"Cannot turn '{source}' into a symbolic expression.")
    if interp is None:
        from ..interpreter import current_interpreter
        interp = current_interpreter()
    return wrap(_eval_node(interp, program.body[0].expr))


def _eval_node(interp, node: A.Node) -> Any:
    sp = sympy()
    t = type(node)
    if t is A.Number:
        return wrap(num2sym(node.value))
    if t is A.IntLiteral:
        return wrap(sp.Integer(node.value))
    if t is A.Ident:
        const = _CONSTANTS.get(node.name)
        if const is not None:
            return wrap(getattr(sp, const))
        return wrap(make_symbol(interp, node.name))
    if t is A.BinOp:
        op = {"&&": "&", "||": "|"}.get(node.op, node.op)
        return interp.binop_values(op, _eval_node(interp, node.left), _eval_node(interp, node.right))
    if t is A.UnaryOp:
        return interp.unary_values(node.op, _eval_node(interp, node.operand))
    if t is A.Postfix:
        return _eval_node(interp, node.operand).unary_op(interp, node.op)
    if t is A.Matrix:
        from .. import ops
        return ops.build_matrix([[_eval_node(interp, e) for e in r] for r in node.rows])
    if t is A.Index and isinstance(node.base, A.Ident):
        args = [_eval_node(interp, a) for a in node.args]
        name = node.base.name
        if name in SYM:
            res = args[0].call_method(interp, name, args, 1) if args else []
            if res:
                return res[0]
        return wrap(sp.Function(name)(*[scalar(a) for a in args]))
    raise MatlabError("This text cannot be turned into a symbolic expression.")


# -- creation ---------------------------------------------------------------------------------

def _set_assumption(interp, name: str, kw: dict | None, conds: list | None = None, also: bool = False) -> Any:
    """Set (or clear, kw=None) the assumptions of a symbol and update the workspace variables."""
    reg = _registry(interp)
    if kw is None:
        reg.pop(name, None)
    elif also and name in reg:
        reg[name]["kw"].update(kw)
        reg[name]["conds"] += list(conds or [])
    else:
        reg[name] = {"kw": dict(kw), "conds": list(conds or [])}
    new = make_symbol(interp, name)
    if name in reg:  # conditions are kept with a plain symbol, so that they stay unevaluated
        plain = sympy().Symbol(name)
        reg[name]["conds"] = [c.xreplace({s: plain for s in c.free_symbols if s.name == name})
                              for c in reg[name]["conds"]]

    def fix(e):
        swap = {s: new for s in getattr(e, "free_symbols", ()) if s.name == name and s != new}
        return e.xreplace(swap) if swap else e

    frames = [interp.frame] + ([interp.base] if interp.base is not interp.frame else [])
    for frame in frames:
        for key, val in list(frame.vars.items()):
            if isinstance(val, SymObject):
                data = _map1(fix, val.data)
                if isinstance(val, SymFun):
                    frame.vars[key] = SymFun(data, [fix(v) for v in val.vars])
                else:
                    frame.vars[key] = SymObject(data)
    return new


def _flag_kw(flag: str) -> dict | None:
    return {"real": {"real": True}, "positive": {"positive": True}, "integer": {"integer": True},
            "rational": {"rational": True}, "clear": None}[flag]


def _symbol_array(interp, name: str, dims: tuple, kw: Any = False) -> SymObject:
    r, c = dims
    names = []
    for j in range(c):
        for i in range(r):
            if r == 1 or c == 1:
                names.append(f"{name}{max(i, j) + 1}")
            else:
                names.append(f"{name}{i + 1}_{j + 1}")
    items = []
    for n in names:
        items.append(_set_assumption(interp, n, kw) if kw is not False else make_symbol(interp, n))
    return SymObject(_obj_array(items, (r, c)))


def _dims_of(v: Any) -> tuple:
    d = [int(x) for x in np.asarray(v, dtype=float).flatten()]
    if len(d) == 1:
        return (d[0], d[0])
    if len(d) != 2 or min(d) < 0:
        raise MatlabError("The size of a symbolic array must be [rows cols].")
    return (d[0], d[1])


def _words(args: list) -> list[str]:
    out: list[str] = []
    for a in args:
        if isinstance(a, StringArray):
            out += [str(x) for x in a.data.flat]
        elif isinstance(a, Cell):
            out += [to_str(x) for x in a.data.flat]
        elif is_text(a):
            out.append(text(a))
        elif isinstance(a, np.ndarray):
            out.append("[" + " ".join(str(int(x)) for x in a.flatten()) + "]")
        else:
            raise MatlabError("syms expects variable names, e.g. syms x y.")
    return out


@symfn("syms")
def _syms(interp, args, nargout):
    sp = sympy()
    words = _words(args)
    frame = interp.frame
    if not words:
        names = sorted(k for k, v in frame.vars.items() if isinstance(v, SymObject))
        if nargout:
            return Cell.from_list([char(n) for n in names], column=True)
        if names:
            interp.out("Your symbolic variables are:\n\n" + "  ".join(names) + "\n\n")
        return None
    # Split brackets that were separated by spaces: 'a', '[1', '3]' -> 'a', '[1 3]'.
    merged: list[str] = []
    for w in words:
        if merged and merged[-1].count("[") > merged[-1].count("]") \
                or merged and merged[-1].count("(") > merged[-1].count(")"):
            merged[-1] += " " + w
        else:
            merged.append(w)
    items: list[dict] = []
    kw: Any = False
    dims = None
    for w in merged:
        if w in _ASSUME_WORDS:
            kw = _flag_kw(w)
        elif w.startswith("["):
            dims = _dims_of([float(x) for x in re.split(r"[\s,]+", w.strip("[] ")) if x])
        elif w.isdigit():
            dims = (int(w), int(w))
        else:
            m = re.fullmatch(r"([A-Za-z]\w*)\s*(?:\(([^()]*)\))?", w)
            if not m:
                raise MatlabError(f"'{w}' is not a valid name for a symbolic variable.")
            fargs = [a.strip() for a in m.group(2).split(",")] if m.group(2) is not None else None
            if fargs is not None and not all(re.fullmatch(r"[A-Za-z]\w*", a) for a in fargs):
                raise MatlabError(f"The arguments of the symbolic function in '{w}' must be variable names.")
            items.append({"name": m.group(1), "args": fargs})
    for item in items:
        name = item["name"]
        if item["args"] is not None:
            fvars = []
            for a in item["args"]:
                s = _set_assumption(interp, a, kw) if kw is not False else make_symbol(interp, a)
                frame.set(a, wrap(s))
                fvars.append(s)
            frame.set(name, SymFun(_one(sp.Function(name)(*fvars)), fvars))
        elif dims is not None:
            frame.set(name, _symbol_array(interp, name, dims, kw))
        else:
            # A plain 'syms x' starts from a clean symbol, like MATLAB.
            frame.set(name, wrap(_set_assumption(interp, name, kw if kw is not False else None)))
    return None


@symfn("sym")
def _sym(interp, args, nargout):
    sp = sympy()
    if not args:
        return SymObject(np.empty((0, 0), dtype=object))
    a, rest = args[0], args[1:]
    if isinstance(a, SymObject):
        return a
    if is_text(a):
        s = text(a).strip()
        if re.fullmatch(r"[A-Za-z]\w*", s) and s not in _CONSTANTS:
            kw: Any = False
            dims = None
            for r in rest:
                if is_text(r):
                    flag = text(r)
                    if flag not in _ASSUME_WORDS:
                        raise MatlabError(f"Unknown assumption '{flag}'. Use 'real', 'positive', "
                                          "'integer' or 'rational'.")
                    kw = _flag_kw(flag)
                else:
                    dims = _dims_of(r)
            if dims is not None:
                return _symbol_array(interp, s, dims, kw)
            if kw is not False:
                return wrap(_set_assumption(interp, s, kw))
            return wrap(make_symbol(interp, s))
        return parse_text(s, interp)
    if isinstance(a, FunctionHandle):
        if not a.is_anonymous:
            raise MatlabError("sym expects an anonymous function, e.g. sym(@(x) x^2).")
        params = [wrap(make_symbol(interp, p)) for p in a.params]
        return wrap(to_data(interp.call_handle(a, params, 1)[0]))
    flag = text(rest[0]) if rest and is_text(rest[0]) else "r"
    if isinstance(a, np.ndarray) and a.dtype.kind in "fc" and flag in ("d", "f"):
        digits = get_digits(interp)
        conv = (lambda x: sp.Float(x, digits)) if flag == "d" else (lambda x: sp.Rational(float(x)))
        out = np.empty(a.shape, dtype=object)
        for idx in np.ndindex(a.shape):
            out[idx] = conv(float(np.real(a[idx])))
        return SymObject(out)
    return wrap(to_data(a, interp))


@symfn("str2sym")
def _str2sym(interp, args, nargout):
    a = args[0]
    if isinstance(a, (StringArray, Cell)):
        items = [str(x) if isinstance(a, StringArray) else to_str(x) for x in a.data.flatten(order="F")]
        return SymObject(_obj_array([parse_text(s, interp).expr for s in items], a.data.shape))
    return parse_text(text(a), interp)


@symfn("symfun")
def _symfun(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("symfun expects an expression and its variables: symfun(x^2 + y, [x y]).")
    fvars = flat_list(args[1])
    if not all(getattr(v, "is_Symbol", False) for v in fvars):
        raise MatlabError("The second argument of symfun must be a vector of symbolic variables.")
    return SymFun(to_data(args[0], interp), fvars)


@symfn("argnames")
def _argnames(interp, args, nargout):
    if not isinstance(args[0], SymFun):
        raise MatlabError("argnames expects a symbolic function.")
    return row(args[0].vars)


@symfn("formula")
def _formula(interp, args, nargout):
    return SymObject(to_data(args[0]))


def free_symbols(v: Any) -> list:
    """Free symbols of a value, sorted the way symvar lists them."""
    found = set()
    for e in to_data(v).flat:
        found |= getattr(e, "free_symbols", set())
    return sorted(found, key=lambda s: s.name)


def default_var(v: Any, extra: Any = None):
    """The variable a function works on when none is given: x if present, else the closest to x."""
    if isinstance(v, SymFun) and v.vars:
        return v.vars[0]
    names = free_symbols(v)
    if not names:
        return sympy().Symbol("x")

    def key(s):
        c = s.name[0]
        return (abs(ord(c.lower()) - ord("x")), -ord(c.lower()), c.isupper(), s.name)

    letters = [s for s in names if len(s.name) >= 1]
    return sorted(letters, key=key)[0]


@symfn("symvar")
def _symvar(interp, args, nargout):
    names = free_symbols(args[0])
    if len(args) > 1:
        n = int(to_double(args[1]).flat[0]) if isinstance(args[1], SymObject) else int(np.asarray(args[1]).flat[0])

        def key(s):
            c = s.name[0]
            return (abs(ord(c.lower()) - ord("x")), -ord(c.lower()), c.isupper(), s.name)
        names = sorted(names, key=key)[:n]
    if not names:
        return SymObject(np.empty((1, 0), dtype=object))
    return row(names)


def _assume_from(interp, cond: Any, also: bool) -> None:
    sp = sympy()
    for e in to_data(cond).flat:
        if isinstance(e, sp.And):
            parts = list(e.args)
        else:
            parts = [e]
        for p in parts:
            if not isinstance(p, sp.core.relational.Relational):
                raise MatlabError("assume expects a condition such as x > 0, or a variable and a "
                                  "set name: assume(x, 'real').")
            syms = sorted(p.free_symbols, key=lambda s: s.name)
            if len(syms) != 1:
                raise MatlabError("Each assumption must involve exactly one symbolic variable.")
            s = syms[0]
            kw: dict = {}
            lhs_is_var = p.lhs == s and not p.rhs.free_symbols
            rhs_is_var = p.rhs == s and not p.lhs.free_symbols
            if lhs_is_var or rhs_is_var:
                bound = p.rhs if lhs_is_var else p.lhs
                op = p.rel_op if lhs_is_var else {"<": ">", ">": "<", "<=": ">=", ">=": "<="}.get(p.rel_op, p.rel_op)
                if op in (">", ">=") and bound.is_real:
                    kw = {"positive": True} if (bound > 0 or (bound == 0 and op == ">")) else {"real": True}
                    if bound == 0 and op == ">=":
                        kw = {"nonnegative": True}
                elif op in ("<", "<=") and bound.is_real:
                    kw = {"negative": True} if (bound < 0 or (bound == 0 and op == "<")) else {"real": True}
                    if bound == 0 and op == "<=":
                        kw = {"nonpositive": True}
                elif op == "!=" and bound == 0:
                    kw = {"nonzero": True}
            _set_assumption(interp, s.name, kw, [p], also=True if also else s.name in _fresh)
            _fresh.add(s.name)


_fresh: set = set()


def _assume(interp, args, also: bool) -> None:
    _fresh.clear()
    if not args:
        raise MatlabError("assume expects a condition, e.g. assume(x > 0).")
    if len(args) >= 2 and is_text(args[1]):
        word = text(args[1])
        if word not in _ASSUME_WORDS:
            raise MatlabError(f"Unknown assumption '{word}'. Use 'real', 'positive', 'integer', "
                              "'rational' or 'clear'.")
        for s in flat_list(args[0]):
            if not s.is_Symbol:
                raise MatlabError("assume(x, 'real') expects symbolic variables as the first argument.")
            _set_assumption(interp, s.name, _flag_kw(word), also=also)
        return
    _assume_from(interp, args[0], also)


@symfn("assume")
def _assume_fn(interp, args, nargout):
    _assume(interp, args, False)
    return None


@symfn("assumeAlso")
def _assume_also(interp, args, nargout):
    _assume(interp, args, True)
    return None


@symfn("assumptions")
def _assumptions(interp, args, nargout):
    sp = sympy()
    reg = _registry(interp)
    names = [s.name for s in free_symbols(args[0])] if args else sorted(reg)
    out = []
    member = sp.Function("in")
    for n in names:
        entry = reg.get(n)
        if not entry:
            continue
        s = sp.Symbol(n)
        conds = list(entry["conds"])
        if not conds:
            kw = entry["kw"]
            if kw.get("positive"):
                conds.append(sp.Lt(0, s, evaluate=False))
            for word in ("integer", "rational", "real"):
                if kw.get(word):
                    conds.append(member(s, sp.Symbol(f"'{word}'")))
        out += conds
    if not out:
        return SymObject(np.empty((1, 0), dtype=object))
    return row(out)


# -- sizes, conversion to numbers and text ----------------------------------------------------

def get_digits(interp) -> int:
    return int(interp.state.get("sym_digits", 32)) if interp is not None else 32


@symfn("digits")
def _digits(interp, args, nargout):
    old = get_digits(interp)
    if args:
        d = int(to_double(args[0]).flat[0])
        if d < 1:
            raise MatlabError("The number of digits must be a positive integer.")
        interp.state["sym_digits"] = d
        return float(old) if nargout else None
    if nargout:
        return float(old)
    interp.out(f"\nDigits = {old}\n\n")
    return None


@symfn("double")
def _double(interp, args, nargout):
    return to_double(args[0])


@symfn("single")
def _single(interp, args, nargout):
    return to_double(args[0]).astype(np.float32)


@symfn("logical")
def _logical(interp, args, nargout):
    d = to_data(args[0])
    out = np.zeros(d.shape, dtype=bool)
    for idx in np.ndindex(d.shape):
        out[idx] = _truth(d[idx])
    return out


@symfn("vpa", keep=True)
def _vpa(interp, args, nargout):
    sp = sympy()
    digits = int(to_double(args[1]).flat[0]) if len(args) > 1 else get_digits(interp)

    def f(e):
        if is_logic(e) and not isinstance(e, sp.core.relational.Relational):
            return e
        return e.evalf(digits)
    return map_data(f, to_data(args[0], interp))


def text_of(v: Any) -> str:
    """One-line text of a sym value (matrices as [a, b; c, d])."""
    d = to_data(v)
    if d.shape == (1, 1):
        return sym_str(d[0, 0])
    return "[" + "; ".join(", ".join(sym_str(e) for e in r) for r in d) + "]"


@symfn("char")
def _char(interp, args, nargout):
    return char(text_of(args[0]))


@symfn("string")
def _string(interp, args, nargout):
    d = to_data(args[0])
    if d.shape == (1, 1):
        return MString(sym_str(d[0, 0]))
    from ..values import string_result
    out = np.empty(d.shape, dtype=object)
    for idx in np.ndindex(d.shape):
        out[idx] = sym_str(d[idx])
    return string_result(out)


@symfn("disp")
def _disp(interp, args, nargout):
    interp.out("\n".join(wrap(args[0]).display_lines()) + "\n")
    return None


@symfn("display")
def _display(interp, args, nargout):
    interp.out(wrap(args[0]).format_display("ans"))
    return None


@symfn("latex")
def _latex(interp, args, nargout):
    sp = sympy()
    d = to_data(args[0])
    obj = d[0, 0] if d.shape == (1, 1) else matrix(d)
    return char(sp.latex(obj))


@symfn("pretty")
def _pretty(interp, args, nargout):
    sp = sympy()
    d = to_data(args[0])
    obj = d[0, 0] if d.shape == (1, 1) else matrix(d)
    out = sp.pretty(obj, use_unicode=False)
    if nargout:
        return char(out)
    interp.out(out + "\n \n")
    return None


@symfn("eval")
def _eval(interp, args, nargout):
    return interp.call_named("eval", [char(text_of(args[0]))], max(nargout, 1), interp.frame)[0]


def _format_function(name: str) -> Callable:
    """fprintf/sprintf/num2str with sym arguments: constants print as numbers, the rest as text."""
    def f(interp, args, nargout):
        conv = []
        for a in args:
            if isinstance(a, SymObject):
                try:
                    a = to_double(a)
                except MatlabError:
                    a = char(text_of(a))
            conv.append(a)
        res = interp.call_builtin(name, interp.builtins[name], conv, nargout)
        return tuple(res) if len(res) != 1 else res[0]
    return f


for _name in ("fprintf", "sprintf", "num2str", "mat2str", "int2str"):
    SYM[_name] = _format_function(_name)


@symfn("isequal")
def _isequal(interp, args, nargout):
    first = wrap(to_data(args[0]))
    return bool_scalar(all(first.object_equals(b) for b in args[1:]))


@symfn("isAlways")
def _is_always(interp, args, nargout):
    sp = sympy()
    d = to_data(args[0])
    out = np.zeros(d.shape, dtype=bool)
    for idx in np.ndindex(d.shape):
        e = d[idx]
        try:
            if isinstance(e, (sp.Eq, sp.Ne)):
                same = e.lhs == e.rhs or sp.simplify(e.lhs - e.rhs) == 0
                if not same and isinstance(e, sp.Eq):
                    same = (e.lhs - e.rhs).equals(0) is True
                out[idx] = same if isinstance(e, sp.Eq) else (not same and sp.simplify(e.lhs - e.rhs).is_nonzero
                                                               is True)
            elif isinstance(e, sp.core.relational.Relational):
                diff = sp.simplify(e.lhs - e.rhs)
                test = {"<": diff.is_negative, "<=": diff.is_nonpositive, ">": diff.is_positive,
                        ">=": diff.is_nonnegative}[e.rel_op]
                if test is None:
                    test = sp.simplify(e.func(diff, 0)) is sp.true
                out[idx] = test is True
            elif is_logic(e):
                out[idx] = sp.simplify(e) is sp.true
            else:
                out[idx] = bool(sp.simplify(e) != 0) and not e.free_symbols
        except (TypeError, NotImplementedError):
            out[idx] = False
    return out


@symfn("isSymType")
def _is_sym_type(interp, args, nargout):
    sp = sympy()
    kind = text(args[1]).lower()
    e = scalar(args[0])
    tests = {
        "variable": lambda: e.is_Symbol,
        "number": lambda: e.is_Number,
        "integer": lambda: e.is_Integer,
        "rational": lambda: e.is_Rational,
        "real": lambda: e.is_Number and e.is_real,
        "vpareal": lambda: e.is_Float,
        "constant": lambda: e.is_number and not is_logic(e),
        "expression": lambda: not is_logic(e),
        "equation": lambda: isinstance(e, sp.core.relational.Relational),
        "logicalexpression": lambda: is_logic(e),
        "function": lambda: isinstance(e, sp.Function),
        "symfun": lambda: isinstance(args[0], SymFun),
        "plus": lambda: e.is_Add,
        "times": lambda: e.is_Mul,
        "power": lambda: e.is_Pow,
    }
    if kind not in tests:
        raise MatlabError(f"Unknown symbolic type '{kind}'.")
    return bool_scalar(bool(tests[kind]()))


@symfn("has")
def _has(interp, args, nargout):
    targets = flat_list(args[1])
    d = to_data(args[0])
    out = np.zeros(d.shape, dtype=bool)
    for idx in np.ndindex(d.shape):
        out[idx] = any(d[idx].has(t) for t in targets)
    return out


@symfn("symtrue")
def _symtrue(interp, args, nargout):
    return wrap(sympy().true)


@symfn("symfalse")
def _symfalse(interp, args, nargout):
    return wrap(sympy().false)


def _predicate(test: Callable) -> Callable:
    def f(interp, args, nargout):
        d = to_data(args[0])
        out = np.zeros(d.shape, dtype=bool)
        for idx in np.ndindex(d.shape):
            out[idx] = bool(test(d[idx]))
        return out
    return f


SYM["isnan"] = _predicate(lambda e: e is sympy().nan)
SYM["isinf"] = _predicate(lambda e: e in (sympy().oo, -sympy().oo, sympy().zoo))
SYM["isfinite"] = _predicate(lambda e: e.is_finite is not False and e is not sympy().nan)
SYM["isreal"] = lambda interp, args, nargout: bool_scalar(all(e.is_real is True for e in to_data(args[0]).flat))


# -- shape functions --------------------------------------------------------------------------

@symfn("transpose", keep=True)
def _transpose(interp, args, nargout):
    return wrap(args[0]).unary_op(interp, ".'")


@symfn("ctranspose", keep=True)
def _ctranspose(interp, args, nargout):
    return wrap(args[0]).unary_op(interp, "'")


def _op_function(op: str) -> Callable:
    def f(interp, args, nargout):
        return binop(op, args[0], args[1])
    return f


for _name, _op in {"plus": "+", "minus": "-", "times": ".*", "rdivide": "./", "ldivide": ".\\", "power": ".^",
                   "mtimes": "*", "mrdivide": "/", "mldivide": "\\", "mpower": "^", "eq": "==", "ne": "~=",
                   "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "and": "&", "or": "|"}.items():
    SYM[_name] = _op_function(_op)
    KEEP_FUN.add(_name)
SYM["uminus"] = lambda interp, args, nargout: wrap(args[0]).unary_op(interp, "-")
SYM["uplus"] = lambda interp, args, nargout: wrap(args[0])
SYM["not"] = lambda interp, args, nargout: wrap(args[0]).unary_op(interp, "~")


@symfn("horzcat")
def _horzcat(interp, args, nargout):
    return SymObject.concat(args, 1)


@symfn("vertcat")
def _vertcat(interp, args, nargout):
    return SymObject.concat(args, 0)


@symfn("cat")
def _cat(interp, args, nargout):
    return SymObject.concat(args[1:], int(to_double(args[0]).flat[0]) - 1)


def _int_list(args: list) -> list[int]:
    out: list[int] = []
    for a in args:
        out += [int(x) for x in np.real(to_double(a)).flatten()]
    return out


@symfn("reshape")
def _reshape(interp, args, nargout):
    d = to_data(args[0])
    dims: list = []
    for a in args[1:]:
        if isinstance(a, np.ndarray) and a.size == 0:
            dims.append(None)
        else:
            dims += _int_list([a])
    if None in dims:
        known = int(np.prod([x for x in dims if x is not None]))
        dims[dims.index(None)] = d.size // known if known else 0
    if len(dims) != 2 or dims[0] * dims[1] != d.size:
        raise MatlabError("To reshape, the number of elements must not change (and sym arrays are 2-D).")
    return SymObject(d.reshape(dims, order="F").copy())


@symfn("repmat")
def _repmat(interp, args, nargout):
    reps = _int_list(args[1:])
    if len(reps) == 1:
        reps = [reps[0], reps[0]]
    return SymObject(np.tile(to_data(args[0]), reps[:2]))


@symfn("diag")
def _diag(interp, args, nargout):
    sp = sympy()
    d = to_data(args[0])
    k = _int_list(args[1:2])[0] if len(args) > 1 else 0
    if 1 in d.shape:
        v = d.flatten(order="F")
        n = v.size + abs(k)
        out = np.empty((n, n), dtype=object)
        out[:] = sp.Integer(0)
        for i, e in enumerate(v):
            out[i + max(-k, 0), i + max(k, 0)] = e
        return SymObject(out)
    return column(list(np.diagonal(d, k)))


def _tri(lower: bool) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()
        d = to_data(args[0]).copy()
        k = _int_list(args[1:2])[0] if len(args) > 1 else 0
        for (i, j) in np.ndindex(d.shape):
            if (j - i > k) if lower else (j - i < k):
                d[i, j] = sp.Integer(0)
        return SymObject(d)
    return f


SYM["tril"] = _tri(True)
SYM["triu"] = _tri(False)


def _reduce(op: Callable, start: int, cumulative: bool = False) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()
        d = to_data(args[0])
        if len(args) > 1 and is_text(args[1]):
            d = d.reshape(-1, 1, order="F")
            axis = 0
        elif len(args) > 1:
            axis = _int_list(args[1:2])[0] - 1
        else:
            axis = 0 if d.shape[0] != 1 else 1
        if axis > 1:
            return SymObject(d)
        work = d if axis == 0 else d.T
        cols = []
        for j in range(work.shape[1]):
            acc = sp.Integer(start)
            col = []
            for e in work[:, j]:
                acc = op(acc, e)
                col.append(acc)
            cols.append(col if cumulative else [acc])
        rows = len(cols[0]) if cols else 0
        out = np.empty((rows, len(cols)), dtype=object)
        for j, col in enumerate(cols):
            for i, e in enumerate(col):
                out[i, j] = e
        return SymObject(out if axis == 0 else out.T.copy())
    return f


SYM["sum"] = _reduce(lambda a, b: a + b, 0)
SYM["prod"] = _reduce(lambda a, b: a * b, 1)
SYM["cumsum"] = _reduce(lambda a, b: a + b, 0, True)
SYM["cumprod"] = _reduce(lambda a, b: a * b, 1, True)


@symfn("dot")
def _dot(interp, args, nargout):
    sp = sympy()
    a, b = flat_list(args[0]), flat_list(args[1])
    if len(a) != len(b):
        raise MatlabError("dot expects two vectors of the same length.")
    return wrap(sum((sp.conjugate(x) * y for x, y in zip(a, b)), sp.Integer(0)))


@symfn("cross")
def _cross(interp, args, nargout):
    a, b = flat_list(args[0]), flat_list(args[1])
    if len(a) != 3 or len(b) != 3:
        raise MatlabError("cross expects two vectors with 3 elements.")
    out = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return row(out) if to_data(args[0]).shape[0] == 1 else column(out)


@symfn("kron")
def _kron(interp, args, nargout):
    from sympy.physics.quantum import TensorProduct
    return wrap(TensorProduct(matrix(args[0]), matrix(args[1])))


@symfn("norm")
def _norm(interp, args, nargout):
    sp = sympy()
    m = matrix(args[0])
    kind: Any = 2
    if len(args) > 1:
        kind = text(args[1]) if is_text(args[1]) else scalar(args[1])
    if kind == "fro":
        return wrap(m.norm())
    if kind in ("inf", "Inf"):
        kind = sp.oo
    if 1 in m.shape:
        return wrap(m.norm(kind))
    if kind == 1:
        return wrap(sp.Max(*[sum((sp.Abs(e) for e in m.col(j)), sp.Integer(0)) for j in range(m.cols)]))
    if kind == sp.oo:
        return wrap(sp.Max(*[sum((sp.Abs(e) for e in m.row(i)), sp.Integer(0)) for i in range(m.rows)]))
    return wrap(m.norm(2))


# -- element-wise mathematical functions -------------------------------------------------------

def _numeric_only(args: list) -> bool:
    return all(isinstance(a, np.ndarray) and a.dtype.kind in "fiubc" for a in args)


def _real_if_possible(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.real.copy() if a.dtype.kind == "c" and not np.any(a.imag) else a


def _special(name: str) -> Callable:
    def f(x):
        import scipy.special as sc
        if name == "zeta":
            return sc.zeta(x)
        if name in ("sinint", "cosint"):
            return sc.sici(x)[0 if name == "sinint" else 1]
        if name in ("sinhint", "coshint"):
            return sc.shichi(x)[0 if name == "sinhint" else 1]
        return sc.expi(x)
    return f


# NumPy versions of the elementary functions that are registered here for numeric inputs too.
_NUMERIC: dict[str, Callable] = {
    "cbrt": np.cbrt,
    "coth": lambda x: 1 / np.tanh(x),
    "sech": lambda x: 1 / np.cosh(x),
    "csch": lambda x: 1 / np.sinh(x),
    "acot": lambda x: np.arctan(1 / x),
    "asec": lambda x: np.emath.arccos(1 / x),
    "acsc": lambda x: np.emath.arcsin(1 / x),
    "acoth": lambda x: np.emath.arctanh(1 / x),
    "asech": lambda x: np.emath.arccosh(1 / x),
    "acsch": lambda x: np.arcsinh(1 / x),
    "zeta": _special("zeta"), "sinint": _special("sinint"), "cosint": _special("cosint"),
    "sinhint": _special("sinhint"), "coshint": _special("coshint"), "ei": _special("ei"),
}


def _elementwise(name: str, fn: Callable, nargs: int = 1) -> None:
    def f(interp, args, nargout):
        if len(args) < nargs:
            raise MatlabError(f"Not enough input arguments for {name}.")
        if _numeric_only(args):  # reached only for names that have no other numeric built-in
            numeric = _NUMERIC.get(name)
            if numeric is not None and nargs == 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = args[0] if args[0].dtype.kind == "c" else args[0].astype(float)
                    return _real_if_possible(numeric(x))
            return to_double(map_data(fn, *args[:nargs]))
        return map_data(fn, *args[:nargs])
    f.__name__ = f"_sym_{name}"
    SYM[name] = f
    KEEP_FUN.add(name)
    if name not in BUILTINS:
        BUILTINS[name] = f  # SymPy is imported only when an argument is symbolic


def _register_elementwise() -> None:
    def lazy(attr: str) -> Callable:
        return lambda *a: getattr(sympy(), attr)(*a)

    same = ["sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc",
            "sinh", "cosh", "tanh", "coth", "sech", "csch", "asinh", "acosh", "atanh", "acoth", "asech",
            "acsch", "exp", "log", "sqrt", "sign", "floor", "gamma", "erf", "erfc", "erfinv", "erfcinv",
            "factorial", "zeta"]
    for n in same:
        _elementwise(n, lazy(n))
    renamed = {"abs": "Abs", "real": "re", "imag": "im", "conj": "conjugate", "angle": "arg", "ceil": "ceiling",
               "gammaln": "loggamma", "heaviside": "Heaviside", "sinint": "Si", "cosint": "Ci", "ei": "Ei",
               "sinhint": "Shi", "coshint": "Chi", "cbrt": "cbrt"}
    for n, attr in renamed.items():
        _elementwise(n, lazy(attr))
    sp = sympy
    _elementwise("log2", lambda e: sp().log(e) / sp().log(2))
    _elementwise("log10", lambda e: sp().log(e) / sp().log(10))
    _elementwise("log1p", lambda e: sp().log(1 + e))
    _elementwise("expm1", lambda e: sp().exp(e) - 1)
    _elementwise("sind", lambda e: sp().sin(sp().pi * e / 180))
    _elementwise("cosd", lambda e: sp().cos(sp().pi * e / 180))
    _elementwise("tand", lambda e: sp().tan(sp().pi * e / 180))
    _elementwise("sinc", lambda e: sp().Piecewise((1, sp().Eq(e, 0)), (sp().sin(sp().pi * e) / (sp().pi * e), True))
                 if e.is_number else sp().sin(sp().pi * e) / (sp().pi * e))
    _elementwise("fix", lambda e: sp().sign(e) * sp().floor(sp().Abs(e)))
    _elementwise("round", lambda e: sp().sign(e) * sp().floor(sp().Abs(e) + sp().Rational(1, 2)))
    _elementwise("expint", lambda e: sp().expint(1, e))
    _elementwise("psi", lambda e: sp().digamma(e))
    _elementwise("atan2", lazy("atan2"), 2)
    _elementwise("mod", lambda a, b: sp().Mod(a, b), 2)
    _elementwise("rem", lambda a, b: a - b * sp().sign(a / b) * sp().floor(sp().Abs(a / b)), 2)
    _elementwise("nthroot", lambda a, n: sp().real_root(a, n), 2)
    _elementwise("beta", lazy("beta"), 2)
    _elementwise("nchoosek", lazy("binomial"), 2)
    _elementwise("besselj", lazy("besselj"), 2)
    _elementwise("bessely", lazy("bessely"), 2)
    _elementwise("besseli", lazy("besseli"), 2)
    _elementwise("besselk", lazy("besselk"), 2)
    _elementwise("gcd", lazy("gcd"), 2)
    _elementwise("lcm", lazy("lcm"), 2)


_register_elementwise()


@symfn("kroneckerDelta")
def _kronecker(interp, args, nargout):
    other = args[1] if len(args) > 1 else np.zeros((1, 1))
    if _numeric_only([args[0], other]):
        return (np.asarray(args[0]) == np.asarray(other)).astype(float)
    sp = sympy()
    return map_data(lambda a, b: sp.KroneckerDelta(a, b), args[0], other)


def _lambertw(interp, args, nargout):
    if _numeric_only(args):
        import scipy.special as sc
        x = np.asarray(args[-1], dtype=complex if args[-1].dtype.kind == "c" else float)
        k = int(np.real(args[0].flat[0])) if len(args) > 1 else 0
        return _real_if_possible(sc.lambertw(x, k))
    sp = sympy()
    if len(args) == 1:
        return map_data(lambda e: sp.LambertW(e), args[0])
    return map_data(lambda k, e: sp.LambertW(e, k), args[0], args[1])


SYM["lambertw"] = _lambertw
KEEP_FUN.add("lambertw")
BUILTINS["lambertw"] = _lambertw
BUILTINS["kroneckerDelta"] = _kronecker


def _heaviside(interp, args, nargout):
    if _numeric_only(args):
        x = np.asarray(args[0], dtype=float)
        return np.where(x > 0, 1.0, np.where(x < 0, 0.0, np.where(np.isnan(x), np.nan, 0.5)))
    return map_data(lambda e: sympy().Heaviside(e, sympy().Rational(1, 2)), args[0])


SYM["heaviside"] = _heaviside
KEEP_FUN.add("heaviside")
BUILTINS["heaviside"] = _heaviside  # numeric inputs need no SymPy


def _dirac(interp, args, nargout):
    if _numeric_only(args):
        x = np.asarray(args[-1], dtype=float)
        return np.where(x == 0, np.inf, np.where(np.isnan(x), np.nan, 0.0))
    sp = sympy()
    if len(args) == 2:
        return map_data(lambda n, e: sp.DiracDelta(e, n), args[0], args[1])
    return map_data(lambda e: sp.DiracDelta(e), args[0])


SYM["dirac"] = _dirac
KEEP_FUN.add("dirac")
BUILTINS["dirac"] = _dirac


@symfn("piecewise")
def _piecewise(interp, args, nargout):
    if args and not any(isinstance(a, SymObject) for a in args):  # numbers: pick values by condition
        items = [np.asarray(a) for a in args]
        conds = [c.astype(bool) for c in items[0:len(items) - len(items) % 2:2]]
        values = [v.astype(float) for v in items[1::2]]
        default = items[-1].astype(float) if len(items) % 2 else np.array([[np.nan]])
        shape = np.broadcast_shapes(*[a.shape for a in items])
        out = np.broadcast_to(default, shape).copy()
        for c, v in reversed(list(zip(conds, values))):
            c, v = np.broadcast_to(c, shape), np.broadcast_to(v, shape)
            out[c] = v[c]
        return out
    sp = sympy()
    pairs = []
    items = list(args)
    while len(items) >= 2:
        cond, value = scalar(items.pop(0), "condition"), scalar(items.pop(0), "value")
        pairs.append((value, _as_bool(cond)))
    if items:
        pairs.append((scalar(items[0], "value"), True))
    else:
        pairs.append((sp.nan, True))
    return wrap(sp.Piecewise(*pairs, evaluate=False) if len(pairs) > 1 else pairs[0][0])


def _extreme(which: str) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()
        fn = sp.Max if which == "max" else sp.Min
        if len(args) >= 2 and not (isinstance(args[1], np.ndarray) and args[1].size == 0):
            return map_data(lambda a, b: fn(a, b), args[0], args[1])
        d = to_data(args[0])
        if 1 in d.shape:
            return wrap(fn(*list(d.flat)))
        return row([fn(*list(d[:, j])) for j in range(d.shape[1])])
    return f


SYM["max"] = _extreme("max")
SYM["min"] = _extreme("min")


# -- function handles and plotting ------------------------------------------------------------

def make_handle(interp, exprs: Any, fvars: list, pad: bool = False) -> FunctionHandle:
    """An anonymous function ``@(vars) text`` that evaluates the expression on numeric arrays."""
    from ..parser import parse
    d = to_data(exprs)
    names = [sym_str(v) for v in fvars]

    def code(e):
        s = sym_str(e, vector=True)
        if pad:  # keep the array shape when the expression does not use every variable
            missing = [n for v, n in zip(fvars, names) if v not in e.free_symbols]
            if missing:
                s = f"{s} + 0.*{'.*'.join(missing)}"
        return s

    if d.shape == (1, 1):
        body = code(d[0, 0])
    else:
        body = "[" + "; ".join(", ".join(code(e) for e in r) for r in d) + "]"
    source = f"@({','.join(names)}){body}"
    try:
        node = parse(source).body[0].expr
    except MatlabError as e:
        raise MatlabError(f"Cannot turn the symbolic expression into a function: {e}") from None
    return FunctionHandle(params=list(node.params), body=node.body,
                          closure={"__functions__": {}}, text=source)


@symfn("matlabFunction")
def _matlab_function(interp, args, nargout):
    exprs = []
    opts: dict[str, Any] = {}
    items = list(args)
    while items:
        a = items.pop(0)
        if is_text(a):
            if not items:
                raise MatlabError(f"matlabFunction: the option '{text(a)}' needs a value.")
            opts[text(a).lower()] = items.pop(0)
        else:
            exprs.append(a)
    if not exprs:
        raise MatlabError("matlabFunction expects a symbolic expression.")
    if "file" in opts:
        raise MatlabError("matlabFunction cannot write a file here; use the returned function handle.")
    value = exprs[0] if len(exprs) == 1 else SymObject.concat(exprs, 1)
    if "vars" in opts:
        v = opts["vars"]
        if isinstance(v, Cell):
            fvars = []
            for item in v.data.flatten(order="F"):
                if is_text(item):
                    fvars.append(make_symbol(interp, text(item)))
                else:
                    fvars += flat_list(item)
        elif isinstance(v, StringArray):
            fvars = [make_symbol(interp, str(x)) for x in v.data.flat]
        else:
            fvars = flat_list(v)
    elif isinstance(value, SymFun):
        fvars = list(value.vars)
    else:
        fvars = free_symbols(value)
    unknown = [s for s in free_symbols(value) if s not in fvars]
    if unknown:
        raise MatlabError(f"The variable '{unknown[0].name}' of the expression is missing from 'Vars'.")
    return make_handle(interp, value, fvars)


_PLOT_VARS = {"fplot": 1, "ezplot": 1, "fplot3": 1, "fsurf": 2, "fmesh": 2, "fcontour": 2, "ezsurf": 2,
              "ezcontour": 2, "ezmesh": 2, "fimplicit": 2, "ezpolar": 1}


def _plot_function(name: str, nvars: int) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()
        symbolic = [a for a in args if isinstance(a, SymObject)]
        found: list = []
        for a in symbolic:
            for s in ([*a.vars] if isinstance(a, SymFun) else free_symbols(a)):
                if s not in found:
                    found.append(s)
        want = nvars
        if name in ("fplot", "ezplot") and len(found) == 2 and len(symbolic) == 1:
            want = 2  # an implicit curve f(x, y) = 0
        if not all(isinstance(a, SymFun) for a in symbolic):
            found.sort(key=lambda s: s.name)
        defaults = [sp.Symbol(n) for n in ("x", "y", "z")]
        while len(found) < want:
            found.append(next(s for s in defaults if s not in found))
        if len(found) > want:
            raise MatlabError(f"{name} expects an expression with at most {want} symbolic variable"
                              f"{'s' if want != 1 else ''}; substitute values for the others with subs.")
        target = name
        new_args = []
        for a in args:
            if isinstance(a, SymObject):
                e = a.expr
                if isinstance(e, sp.Eq):
                    e = e.lhs - e.rhs
                new_args.append(make_handle(interp, e, found, pad=True))
            else:
                new_args.append(a)
        if want == 2 and name in ("fplot", "ezplot"):
            target = "fimplicit" if "fimplicit" in interp.builtins else name
        fn = interp.builtins.get(target)
        if fn is None:
            raise MatlabError(f"{name} is not available.")
        res = interp.call_builtin(target, fn, new_args, nargout)
        return tuple(res) if len(res) != 1 else res[0]
    return f


for _name, _n in _PLOT_VARS.items():
    if _name in BUILTINS:
        SYM[_name] = _plot_function(_name, _n)
