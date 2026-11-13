"""Symbolic calculus, algebra, equation solving, transforms and linear algebra.

Everything here works on the :class:`~mylab.builtins.symbolic.SymObject` values created by
``syms``/``sym`` and is computed by SymPy (imported lazily, see :func:`symbolic.sympy`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, Struct
from ._util import is_text, text
from .symbolic import (
    SymFun, SymObject, _registry, column, default_var, flat_list, free_symbols, get_digits,
    is_logic, make_symbol, map_data, matrix, row, scalar, solve_linear, symfn, sympy, to_data,
    to_double, wrap,
)


# -- helpers ----------------------------------------------------------------------------------

def split_options(args: list, names: set[str]) -> tuple[list, dict]:
    """Separate trailing Name, Value pairs (names compared without case) from the other arguments."""
    for i, a in enumerate(args):
        if is_text(a) and text(a).lower() in names:
            opts = {}
            rest = args[i:]
            if len(rest) % 2:
                raise MatlabError(f"The option '{text(rest[-1])}' needs a value.")
            for k in range(0, len(rest), 2):
                if not is_text(rest[k]):
                    raise MatlabError("Option names must be text.")
                opts[text(rest[k]).lower()] = rest[k + 1]
            return list(args[:i]), opts
    return list(args), {}


def _flag(v: Any) -> bool:
    if is_text(v):
        return text(v).lower() in ("true", "on", "1")
    return bool(np.asarray(to_double(v) if isinstance(v, SymObject) else v).flat[0])


def _is_var(a: Any) -> bool:
    return isinstance(a, SymObject) and a.data.size == 1 and getattr(a.data.flat[0], "is_Symbol", False)


def _is_count(a: Any) -> bool:
    if isinstance(a, np.ndarray) and a.dtype.kind in "fiu" and a.size == 1:
        return float(a.flat[0]) == int(a.flat[0])
    return isinstance(a, SymObject) and a.data.size == 1 and getattr(a.data.flat[0], "is_Integer", False)


def _count(a: Any) -> int:
    return int(to_double(a).flat[0]) if isinstance(a, SymObject) else int(a.flat[0])


def on_sides(fn: Callable, e: Any):
    """Apply ``fn`` to an expression, or to both sides of an equation."""
    sp = sympy()
    if isinstance(e, sp.core.relational.Relational):
        return e.func(fn(e.lhs), fn(e.rhs), evaluate=False)
    if is_logic(e):
        return e
    return fn(e)


def _var_or_default(args: list, pos: int, f: Any):
    if len(args) > pos:
        return scalar(args[pos], "variable")
    return default_var(f)


def _as_expr(e: Any):
    """Equation -> left side minus right side."""
    sp = sympy()
    if isinstance(e, sp.Eq):
        return e.lhs - e.rhs
    return e


def _struct(fields: dict) -> Struct:
    s = Struct()
    for k, v in fields.items():
        s.set_field(0, k, v)
    return s


# -- calculus ---------------------------------------------------------------------------------

@symfn("diff", keep=True)
def _diff(interp, args, nargout):
    sp = sympy()
    f = args[0]
    wrt: list = []
    for a in args[1:]:
        if _is_count(a) and not _is_var(a):
            n = _count(a)
            if n < 0:
                raise MatlabError("The order of differentiation must be a nonnegative integer.")
            if not wrt:
                wrt = [default_var(f)] * n
                if n == 0:
                    return wrap(to_data(f))
            else:
                wrt += [wrt[-1]] * (n - 1)
        else:
            wrt += flat_list(a)
    if not wrt:
        wrt = [default_var(f)]
    return map_data(lambda e: on_sides(lambda x: sp.diff(x, *wrt), e), f)


@symfn("int")
def _int(interp, args, nargout):
    sp = sympy()
    args, _opts = split_options(args, {"ignoreanalyticconstraints", "ignorespecialcases", "principalvalue", "hold"})
    f, rest = args[0], args[1:]
    limits = None
    var = None
    if rest and to_data(rest[-1]).size == 2:
        limits = flat_list(rest[-1])
        rest = rest[:-1]
    elif len(rest) >= 2:
        limits = [scalar(rest[-2], "limit"), scalar(rest[-1], "limit")]
        rest = rest[:-2]
    if rest:
        var = scalar(rest[0], "integration variable")
        if not var.is_Symbol:
            raise MatlabError("The integration variable must be a symbolic variable.")
    else:
        var = default_var(f)

    def one(e):
        try:
            if limits is None:
                res = sp.integrate(e, var)
                if isinstance(res, sp.Piecewise) and not any(c.has(var) for _, c in res.args):
                    # Keep the general case, e.g. -cos(a*x)/a rather than a branch for a == 0.
                    general = [v for v, c in res.args if isinstance(c, sp.Ne)]
                    res = general[0] if general else res
                return res
            return sp.integrate(e, (var, limits[0], limits[1]), conds="none")
        except (NotImplementedError, ValueError):
            return sp.Integral(e, var) if limits is None else sp.Integral(e, (var, limits[0], limits[1]))

    res = map_data(one, f)
    if limits is None and isinstance(f, SymFun):
        return SymFun(res.data, f.vars)
    return res


@symfn("limit")
def _limit(interp, args, nargout):
    sp = sympy()
    args = list(args)
    direction = "+-"
    if args and is_text(args[-1]):
        word = text(args.pop()).lower()
        if word not in ("left", "right"):
            raise MatlabError("The direction of a limit must be 'left' or 'right'.")
        direction = "-" if word == "left" else "+"
    f = args[0]
    if len(args) >= 3:
        var, point = scalar(args[1], "variable"), scalar(args[2], "limit point")
    elif len(args) == 2:
        var, point = default_var(f), scalar(args[1], "limit point")
    else:
        var, point = default_var(f), sp.Integer(0)

    def one(e):
        try:
            if point in (sp.oo, -sp.oo):
                return sp.limit(e, var, point)
            return sp.limit(e, var, point, direction)
        except ValueError:
            return sp.nan  # the one-sided limits differ
        except NotImplementedError:
            return sp.Limit(e, var, point, direction if direction != "+-" else "+")

    return map_data(one, f)


@symfn("taylor", keep=True)
def _taylor(interp, args, nargout):
    sp = sympy()
    args, opts = split_options(args, {"expansionpoint", "order", "ordermode"})
    f = args[0]
    var = _var_or_default(args, 1, f)
    point = scalar(args[2], "expansion point") if len(args) > 2 else sp.Integer(0)
    if "expansionpoint" in opts:
        point = scalar(opts["expansionpoint"], "expansion point")
    order = _count(opts["order"]) if "order" in opts else 6

    def one(e):
        total = sp.Integer(0)
        d = e
        try:
            for k in range(order):
                c = d.subs(var, point)
                if c.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    c = sp.limit(d, var, point)
                    if c.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                        raise ValueError
                total += c / sp.factorial(k) * (var - point) ** k
                d = sp.diff(d, var)
            return total
        except (ValueError, NotImplementedError):
            return sp.series(e, var, point, order).removeO()

    return map_data(one, f)


def _sum_like(kind: str) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()
        do = sp.summation if kind == "symsum" else sp.product
        expr = args[0]
        rest = list(args[1:])
        if rest and to_data(rest[-1]).size == 2:
            lo, hi = flat_list(rest.pop())
        elif len(rest) >= 2:
            hi, lo = scalar(rest.pop(), "upper bound"), scalar(rest.pop(), "lower bound")
        else:
            lo = hi = None
        var = scalar(rest[0], "index variable") if rest else default_var(expr)

        def one(e):
            if lo is None:  # indefinite: F(k) with F(k+1) - F(k) = f(k)
                i = sp.Dummy("i", integer=True)
                return sp.simplify(do(e.subs(var, i), (i, 0, var - 1)))
            res = do(e, (var, lo, hi))
            if isinstance(res, sp.Piecewise) and hi in (sp.oo,) and len(free_symbols(e)) <= 1:
                res = res.args[0][0]
            return res

        return map_data(one, expr)
    return f


symfn("symsum")(_sum_like("symsum"))
symfn("symprod")(_sum_like("symprod"))


def _vars_arg(args: list, pos: int, f: Any) -> list:
    if len(args) > pos:
        return flat_list(args[pos])
    return free_symbols(f)


@symfn("jacobian")
def _jacobian(interp, args, nargout):
    sp = sympy()
    fs = flat_list(args[0])
    vs = _vars_arg(args, 1, args[0])
    out = np.empty((len(fs), len(vs)), dtype=object)
    for i, fi in enumerate(fs):
        for j, v in enumerate(vs):
            out[i, j] = sp.diff(fi, v)
    return SymObject(out)


@symfn("gradient", keep=True)
def _gradient(interp, args, nargout):
    sp = sympy()
    f = scalar(args[0], "function")
    vs = args[0].vars if isinstance(args[0], SymFun) and len(args) < 2 else _vars_arg(args, 1, args[0])
    return column([sp.diff(f, v) for v in vs])


@symfn("hessian", keep=True)
def _hessian(interp, args, nargout):
    sp = sympy()
    f = scalar(args[0], "function")
    vs = args[0].vars if isinstance(args[0], SymFun) and len(args) < 2 else _vars_arg(args, 1, args[0])
    out = np.empty((len(vs), len(vs)), dtype=object)
    for i, a in enumerate(vs):
        for j, b in enumerate(vs):
            out[i, j] = sp.diff(f, a, b)
    return SymObject(out)


@symfn("laplacian", keep=True)
def _laplacian(interp, args, nargout):
    sp = sympy()
    f = scalar(args[0], "function")
    vs = _vars_arg(args, 1, args[0])
    return wrap(sum((sp.diff(f, v, 2) for v in vs), sp.Integer(0)))


@symfn("divergence", keep=True)
def _divergence(interp, args, nargout):
    sp = sympy()
    fs = flat_list(args[0])
    vs = _vars_arg(args, 1, args[0])
    if len(fs) != len(vs):
        raise MatlabError("divergence expects a vector field and as many variables as it has components.")
    return wrap(sum((sp.diff(fi, v) for fi, v in zip(fs, vs)), sp.Integer(0)))


@symfn("curl", keep=True)
def _curl(interp, args, nargout):
    sp = sympy()
    fs = flat_list(args[0])
    vs = _vars_arg(args, 1, args[0])
    if len(fs) != 3 or len(vs) != 3:
        raise MatlabError("curl expects a vector field with 3 components and 3 variables.")
    out = [sp.diff(fs[2], vs[1]) - sp.diff(fs[1], vs[2]),
           sp.diff(fs[0], vs[2]) - sp.diff(fs[2], vs[0]),
           sp.diff(fs[1], vs[0]) - sp.diff(fs[0], vs[1])]
    return column(out)


# -- simplification and rewriting -------------------------------------------------------------

@symfn("simplify", keep=True)
def _simplify(interp, args, nargout):
    sp = sympy()
    args, _ = split_options(args, {"steps", "seconds", "ignoreanalyticconstraints", "criterion", "all"})

    def one(e):
        if isinstance(e, sp.core.relational.Relational):
            res = sp.simplify(e)
            return res if res in (sp.true, sp.false) else on_sides(sp.simplify, e)
        return sp.simplify(e)

    return map_data(one, args[0])


@symfn("simplifyFraction", keep=True)
def _simplify_fraction(interp, args, nargout):
    sp = sympy()
    return map_data(lambda e: on_sides(lambda x: sp.cancel(sp.together(x)), e), args[0])


@symfn("expand", keep=True)
def _expand(interp, args, nargout):
    sp = sympy()
    args, _ = split_options(args, {"arithmeticonly", "ignoreanalyticconstraints"})
    return map_data(lambda e: on_sides(lambda x: sp.expand(x, trig=True, log=False), e), args[0])


@symfn("factor")
def _factor(interp, args, nargout):
    sp = sympy()
    args, _ = split_options(args, {"factormode"})
    d = to_data(args[0])
    if d.size != 1:
        return map_data(sp.factor, d)
    e = d.flat[0]
    if e.is_Integer:
        if e in (0, 1, -1):
            return wrap(e)
        items = []
        for p, k in sorted(sp.factorint(abs(e)).items()):
            items += [sp.Integer(p)] * k
        if e < 0:
            items.insert(0, sp.Integer(-1))
        return row(items)
    if e.is_Rational:
        return row([e])
    gens = flat_list(args[1]) if len(args) > 1 else []
    try:
        coeff, factors = sp.factor_list(e, *gens)
    except (sp.PolynomialError, ValueError):
        return wrap(sp.factor(e))
    items = []
    if coeff != 1:
        items.append(coeff)
    for base, k in factors:
        items += [base] * int(k)
    return row(items or [sp.Integer(1)])


@symfn("collect", keep=True)
def _collect(interp, args, nargout):
    sp = sympy()
    f = args[0]
    targets = flat_list(args[1]) if len(args) > 1 else [default_var(f)]
    return map_data(lambda e: on_sides(lambda x: sp.collect(sp.expand(x), targets), e), f)


@symfn("combine", keep=True)
def _combine(interp, args, nargout):
    sp = sympy()
    args, _ = split_options(args, {"ignoreanalyticconstraints"})
    target = text(args[1]).lower() if len(args) > 1 and is_text(args[1]) else ""

    def one(x):
        if target == "log":
            return sp.logcombine(x, force=True)
        if target in ("sincos", "sinhcosh"):
            from sympy.simplify.fu import TR8
            return TR8(x)
        if target == "exp":
            return sp.powsimp(x)
        if target in ("atan", "int", "gamma"):
            return sp.simplify(x)
        return sp.powsimp(sp.logcombine(x), force=True)

    return map_data(lambda e: on_sides(one, e), args[0])


@symfn("rewrite", keep=True)
def _rewrite(interp, args, nargout):
    sp = sympy()
    target = text(args[1]).lower()
    table = {"exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "log": sp.log, "sqrt": sp.sqrt,
             "heaviside": sp.Heaviside, "piecewise": sp.Piecewise, "sinh": sp.sinh, "cosh": sp.cosh,
             "tanh": sp.tanh, "gamma": sp.gamma, "factorial": sp.factorial, "erf": sp.erf, "cot": sp.cot,
             "asin": sp.asin, "acos": sp.acos, "atan": sp.atan}
    if target == "sincos":
        return map_data(lambda e: on_sides(lambda x: x.rewrite(sp.sin).rewrite(sp.cos), e), args[0])
    if target not in table:
        raise MatlabError(f"rewrite does not know the target '{target}'. Try 'exp', 'sin', 'cos', 'tan', "
                          "'log', 'sincos' or 'heaviside'.")
    return map_data(lambda e: on_sides(lambda x: x.rewrite(table[target]), e), args[0])


@symfn("horner", keep=True)
def _horner(interp, args, nargout):
    sp = sympy()
    f = args[0]
    var = _var_or_default(args, 1, f)
    return map_data(lambda e: sp.horner(e, var), f)


@symfn("numden")
def _numden(interp, args, nargout):
    sp = sympy()
    d = to_data(args[0])
    nums = np.empty(d.shape, dtype=object)
    dens = np.empty(d.shape, dtype=object)
    for idx in np.ndindex(d.shape):
        n, q = sp.fraction(sp.cancel(sp.together(d[idx])))
        if q.could_extract_minus_sign():
            n, q = -n, -q
        nums[idx], dens[idx] = n, q
    return SymObject(nums), SymObject(dens)


@symfn("partfrac", keep=True)
def _partfrac(interp, args, nargout):
    sp = sympy()
    args, opts = split_options(args, {"factormode"})
    f = args[0]
    var = _var_or_default(args, 1, f)
    full = text(opts.get("factormode", "rational")).lower() in ("full", "complex") if opts else False

    def one(e):
        try:
            return sp.apart(e, var, full=True).doit() if full else sp.apart(e, var)
        except (sp.PolynomialError, NotImplementedError):
            return e

    return map_data(one, f)


def _pairs(interp, old: Any, new: Any) -> list[tuple]:
    """(old, new) expression pairs from the many forms subs accepts."""
    if isinstance(old, Cell):
        olds = [v for v in old.data.flatten(order="F")]
        news = list(new.data.flatten(order="F")) if isinstance(new, Cell) else None
        if news is None:
            vals = flat_list(new)
            news = vals if len(vals) == len(olds) else [new] * len(olds)
        if len(news) != len(olds):
            raise MatlabError("subs: the lists of old and new values must have the same length.")
        out = []
        for o, n in zip(olds, news):
            out += _pairs(interp, o, n)
        return out
    if isinstance(old, (MString, StringArray)) or is_text(old):
        old = wrap(to_data(old, interp))
    olds = flat_list(old)
    if isinstance(new, Cell):
        news = list(new.data.flatten(order="F"))
        if len(news) != len(olds):
            raise MatlabError("subs: the lists of old and new values must have the same length.")
        return list(zip(olds, news))
    nd = to_data(new, interp)
    if len(olds) == 1:
        return [(olds[0], new if nd.size != 1 else nd.flat[0])]
    if nd.size == 1:
        return [(o, nd.flat[0]) for o in olds]
    if nd.size != len(olds):
        raise MatlabError("subs: the lists of old and new values must have the same length.")
    return list(zip(olds, nd.flatten(order="F")))


@symfn("subs")
def _subs_fn(interp, args, nargout):
    sp = sympy()
    f = args[0]
    data = to_data(f, interp)
    if len(args) == 1:  # use the values of workspace variables with the same names
        pairs = []
        for s in free_symbols(data):
            val = interp.frame.lookup(s.name)
            if val is None:
                continue
            if isinstance(val, SymObject) and val.data.size == 1 and val.data.flat[0] == s:
                continue
            if isinstance(val, (np.ndarray, SymObject)):
                pairs.append((s, val))
    elif len(args) == 2:
        if isinstance(args[1], Struct):
            st = args[1].data.flat[0]
            pairs = [(make_symbol(interp, k), v) for k, v in st.items()]
        else:
            d1 = to_data(args[1], interp)
            eqs = [e for e in d1.flat if isinstance(e, sp.Eq)]
            if eqs and len(eqs) == d1.size:  # subs(f, x == 2)
                pairs = [(e.lhs, e.rhs) for e in eqs]
            else:
                pairs = _pairs(interp, wrap(default_var(f)), args[1])
    else:
        pairs = _pairs(interp, args[1], args[2])
    scalars = {}
    arrays = []
    for old, new in pairs:
        if isinstance(new, (np.ndarray, SymObject, Cell)) or not hasattr(new, "free_symbols"):
            nd = to_data(new, interp)
            if nd.size == 1:
                scalars[old] = nd.flat[0]
            else:
                arrays.append((old, nd))
        else:
            scalars[old] = new

    def apply(e, mapping):
        if not mapping:
            return e
        res = e.subs(mapping, simultaneous=True) if len(mapping) > 1 else e.subs(mapping)
        return res.doit() if isinstance(res, sp.Subs) else res

    if not arrays:
        res = map_data(lambda e: apply(e, scalars), data)
        return SymFun(res.data, f.vars) if isinstance(f, SymFun) and not (set(f.vars) & set(scalars)) else res
    # Array values: each element of the expression expands to an array of that size.
    shape = arrays[0][1].shape
    if any(a.shape != shape for _, a in arrays):
        raise MatlabError("subs: array values must all have the same size.")
    blocks = np.empty(data.shape, dtype=object)
    for idx in np.ndindex(data.shape):
        block = np.empty(shape, dtype=object)
        for pos in np.ndindex(shape):
            mapping = dict(scalars)
            for old, arr in arrays:
                mapping[old] = arr[pos]
            block[pos] = apply(data[idx], mapping)
        blocks[idx] = block
    return SymObject(np.block([[blocks[i, j] for j in range(data.shape[1])] for i in range(data.shape[0])]))


# -- polynomials and expression parts ---------------------------------------------------------

@symfn("coeffs")
def _coeffs(interp, args, nargout):
    sp = sympy()
    args = list(args)
    want_all = False
    if args and is_text(args[-1]) and text(args[-1]).lower() == "all":
        want_all = True
        args.pop()
    p = sp.expand(scalar(args[0], "polynomial"))
    gens = flat_list(args[1]) if len(args) > 1 else free_symbols(args[0])
    if not gens:
        return (wrap(p), wrap(sp.Integer(1))) if nargout > 1 else wrap(p)
    try:
        poly = sp.Poly(p, *gens)
    except sp.PolynomialError:
        raise MatlabError("coeffs expects a polynomial in the given variables.") from None
    if want_all:
        if len(gens) != 1:
            raise MatlabError("The 'All' option of coeffs needs a single variable.")
        cs = poly.all_coeffs()
        ts = [gens[0] ** k for k in range(len(cs) - 1, -1, -1)]
    else:
        terms = poly.terms()
        cs = [c for _, c in terms]
        ts = [sp.Mul(*[g ** k for g, k in zip(gens, mon)]) for mon, _ in terms]
        if nargout <= 1:
            cs = cs[::-1]
    if nargout > 1:
        return row(cs), row(ts)
    return row(cs)


@symfn("sym2poly")
def _sym2poly(interp, args, nargout):
    sp = sympy()
    p = scalar(args[0], "polynomial")
    gens = free_symbols(args[0])
    if not gens:
        return to_double(wrap(p))
    if len(gens) > 1:
        raise MatlabError("sym2poly expects a polynomial in one variable.")
    try:
        cs = sp.Poly(p, gens[0]).all_coeffs()
    except sp.PolynomialError:
        raise MatlabError("sym2poly expects a polynomial.") from None
    return to_double(row(cs))


@symfn("poly2sym")
def _poly2sym(interp, args, nargout):
    sp = sympy()
    cs = flat_list(args[0])
    var = scalar(args[1], "variable") if len(args) > 1 else make_symbol(interp, "x")
    n = len(cs)
    return wrap(sum((c * var ** (n - 1 - k) for k, c in enumerate(cs)), sp.Integer(0)))


@symfn("polynomialDegree")
def _poly_degree(interp, args, nargout):
    sp = sympy()
    p = scalar(args[0], "polynomial")
    gens = flat_list(args[1]) if len(args) > 1 else free_symbols(args[0])
    if not gens:
        return 0.0
    try:
        return float(sp.Poly(p, *gens).total_degree())
    except sp.PolynomialError:
        raise MatlabError("polynomialDegree expects a polynomial.") from None


def _side(which: str) -> Callable:
    def f(interp, args, nargout):
        sp = sympy()

        def one(e):
            if not isinstance(e, sp.core.relational.Relational):
                raise MatlabError(f"{which} expects an equation or inequality, e.g. {which}(x^2 == 4).")
            return e.lhs if which == "lhs" else e.rhs
        return map_data(one, args[0])
    return f


symfn("lhs", keep=True)(_side("lhs"))
symfn("rhs", keep=True)(_side("rhs"))


@symfn("children")
def _children(interp, args, nargout):
    def kids(e):
        items = list(e.args) if e.args else [e]
        if e.is_Add or e.is_Mul:
            items = e.as_ordered_terms() if e.is_Add else e.as_ordered_factors()
        return Cell.from_list([wrap(a) for a in items])

    d = to_data(args[0])
    if d.size == 1:
        return kids(d.flat[0])
    out = np.empty(d.shape, dtype=object)
    for idx in np.ndindex(d.shape):
        out[idx] = kids(d[idx])
    return Cell(out)


@symfn("finverse", keep=True)
def _finverse(interp, args, nargout):
    sp = sympy()
    f = scalar(args[0], "function")
    var = _var_or_default(args, 1, args[0])
    y = sp.Dummy("y")
    sols = sp.solve(sp.Eq(f, y), var)
    if not sols:
        raise MatlabError("finverse could not find an inverse function.")
    return wrap(sols[0].subs(y, var))


@symfn("compose")
def _compose(interp, args, nargout):
    f, g = scalar(args[0], "function"), scalar(args[1], "function")
    x = scalar(args[2], "variable") if len(args) > 2 else default_var(args[0])
    return wrap(f.subs(x, g))


# -- equation solving -------------------------------------------------------------------------

def _split_equations(args: list) -> tuple[list, list]:
    """Equations (as expressions equal to zero) and the unknowns of a solve-style call."""
    sp = sympy()
    groups = [flat_list(a) for a in args]
    cut = len(groups)
    while cut > 1 and groups[cut - 1] and all(getattr(e, "is_Symbol", False) for e in groups[cut - 1]):
        cut -= 1
    eqs = [e for g in groups[:cut] for e in g]
    unknowns = [e for g in groups[cut:] for e in g]
    exprs = []
    for e in eqs:
        if e is sp.true:
            continue
        if e is sp.false:
            exprs.append(sp.Integer(1))
        elif isinstance(e, sp.Eq):
            exprs.append(e.lhs - e.rhs)
        elif isinstance(e, sp.core.relational.Relational):
            exprs.append(e)
        else:
            exprs.append(e)
    return exprs, unknowns


def _pick_unknowns(exprs: list, n: int) -> list:
    found = set()
    for e in exprs:
        found |= e.free_symbols

    def key(s):
        c = s.name[0]
        return (abs(ord(c.lower()) - ord("x")), -ord(c.lower()), c.isupper(), s.name)

    picked = sorted(found, key=key)[:n]
    return sorted(picked, key=lambda s: s.name)


def _solution_outputs(unknowns: list, columns: list, nargout: int, extra: dict | None = None):
    """Shape solve results like MATLAB: a vector, a struct, or one output per unknown."""
    cols = [column(c) if c else SymObject(np.empty((0, 1), dtype=object)) for c in columns]
    extra = extra or {}
    if nargout <= 1:
        if len(unknowns) == 1 and not extra:
            return cols[0]
        fields = {u.name: c for u, c in zip(unknowns, cols)}
        fields.update(extra)
        return _struct(fields)
    outs = cols + list(extra.values())
    if nargout > len(outs):
        raise MatlabError(f"solve returns at most {len(outs)} outputs here.")
    if nargout < len(unknowns):
        raise MatlabError(f"There are {len(unknowns)} unknowns: use one output (a struct) or "
                          f"{len(unknowns)} outputs.")
    return tuple(outs[:nargout])


def _poly_roots(exprs: list, unknowns: list, opts: dict) -> list | None:
    """Roots of one polynomial with numeric coefficients; factors above MaxDegree give root(...) forms."""
    sp = sympy()
    if len(exprs) != 1 or len(unknowns) != 1:
        return None
    u = unknowns[0]
    num = sp.fraction(sp.together(exprs[0]))[0]
    if num.free_symbols != {u} or not num.is_polynomial(u):
        return None
    max_degree = _count(opts["maxdegree"]) if "maxdegree" in opts else 2
    try:
        _, factors = sp.factor_list(num, u)
    except sp.PolynomialError:
        return None
    if all(sp.degree(f, u) <= max_degree for f, _ in factors):
        return None
    sols = []
    for f, _mult in factors:
        deg = sp.degree(f, u)
        if deg <= max_degree:
            sols += [{u: r} for r in sp.solve(f, u)]
        else:
            sols += [{u: sp.CRootOf(f, u, k)} for k in range(deg)]
    return sols


@symfn("solve")
def _solve(interp, args, nargout):
    sp = sympy()
    args, opts = split_options(args, {"returnconditions", "real", "principalvalue", "maxdegree",
                                      "ignoreanalyticconstraints", "ignoreproperties"})
    args = [wrap(to_data(a, interp)) for a in args]
    exprs, unknowns = _split_equations(args)
    if not exprs:
        raise MatlabError("solve expects at least one equation, e.g. solve(x^2 == 4, x).")
    if not unknowns:
        unknowns = _pick_unknowns(exprs, len(exprs)) if len(exprs) > 1 else [default_var(row(exprs))]
    if any(isinstance(e, sp.core.relational.Relational) for e in exprs):
        raise MatlabError("solve does not handle inequalities; solve the equation and test the "
                          "solutions with isAlways or subs.")
    try:
        sols = _poly_roots(exprs, unknowns, opts)
        if sols is None:
            sols = sp.solve(exprs, unknowns, dict=True)
    except NotImplementedError:
        interp.warn("Unable to solve symbolically. Returning a numeric solution using vpasolve.")
        return _vpasolve(interp, args, nargout)
    reg = _registry(interp)
    kept = []
    for sol in sols:
        vals = [sol.get(u, u) for u in unknowns]
        if "real" in opts and _flag(opts["real"]) and any(v.is_real is False for v in vals):
            continue
        ok = True
        for u, v in zip(unknowns, vals):
            for cond in reg.get(u.name, {}).get("conds", []):
                if cond.subs(sp.Symbol(u.name), v) is sp.false:
                    ok = False
        if ok and vals not in kept:
            kept.append(vals)
    if "principalvalue" in opts and _flag(opts["principalvalue"]):
        kept = kept[:1]
    columns = [[vals[i] for vals in kept] for i in range(len(unknowns))]
    extra = None
    if "returnconditions" in opts and _flag(opts["returnconditions"]):
        extra = {"parameters": SymObject(np.empty((1, 0), dtype=object)),
                 "conditions": column([sp.true] * len(kept)) if kept else SymObject(np.empty((0, 1), dtype=object))}
    return _solution_outputs(unknowns, columns, nargout, extra)


def _vpasolve(interp, args, nargout):
    sp = sympy()
    args, opts = split_options(list(args), {"random"})
    digits = get_digits(interp)
    guess = None
    if len(args) >= 2 and isinstance(args[-1], np.ndarray) and args[-1].dtype.kind in "fiuc":
        guess = np.asarray(args[-1], dtype=complex)
        args = args[:-1]
    args = [wrap(to_data(a, interp)) for a in args]
    exprs, unknowns = _split_equations(args)
    if not unknowns:
        unknowns = _pick_unknowns(exprs, len(exprs)) if len(exprs) > 1 else [default_var(row(exprs))]
    left = set().union(*[e.free_symbols for e in exprs]) - set(unknowns)
    if left:
        raise MatlabError("vpasolve needs numeric equations: the variable "
                          f"'{sorted(s.name for s in left)[0]}' has no value.")
    if len(exprs) == 1 and len(unknowns) == 1 and guess is None:
        e, u = sp.together(exprs[0]), unknowns[0]
        num = sp.fraction(e)[0]
        if num.is_polynomial(u):
            roots = sp.Poly(num, u).nroots(n=digits, maxsteps=200)
            roots = [r for r in roots if abs(sp.fraction(e)[1].subs(u, r).evalf()) > 0]
            return _solution_outputs(unknowns, [roots], nargout)
    n = len(unknowns)
    starts: list = []
    interval = None
    if guess is not None:
        if n == 1 and guess.size == 2:
            interval = (float(guess.flat[0].real), float(guess.flat[1].real))
            starts = [(interval[0] + interval[1]) / 2, interval[0], interval[1]]
        elif guess.shape == (n, 2) and n > 1:
            starts = [[float(np.mean(r).real) for r in guess]]
        else:
            g = [complex(x) if x.imag else float(x.real) for x in guess.flatten()]
            starts = [g[0] if n == 1 else g]
    else:
        rng = np.random.default_rng(0)
        base = [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 10.0]
        starts = base if n == 1 else [[b] * n for b in base]
        starts += [float(x) for x in rng.uniform(-10, 10, 8)] if n == 1 else \
            [list(map(float, rng.uniform(-10, 10, n))) for _ in range(8)]
    for start in starts:
        try:
            if n == 1 and interval is not None and start == starts[0]:
                try:
                    sol = sp.nsolve(exprs[0], unknowns[0], interval, solver="bisect", prec=digits)
                except (ValueError, ZeroDivisionError):
                    sol = sp.nsolve(exprs[0], unknowns[0], start, prec=digits)
            elif n == 1:
                sol = sp.nsolve(exprs[0], unknowns[0], start, prec=digits)
            else:
                sol = sp.nsolve(exprs, unknowns, start, prec=digits)
        except (ValueError, ZeroDivisionError, TypeError, ArithmeticError):
            continue
        vals = [sol] if n == 1 else list(sol)
        if interval is not None and not (interval[0] <= float(sp.re(vals[0])) <= interval[1]):
            continue
        vals = [sp.re(v) if abs(sp.im(v)) < sp.Float(10) ** (-digits + 4) else v for v in vals]
        return _solution_outputs(unknowns, [[v] for v in vals], nargout)
    return _solution_outputs(unknowns, [[] for _ in unknowns], nargout)


symfn("vpasolve")(_vpasolve)


@symfn("linsolve")
def _linsolve(interp, args, nargout):
    return wrap(solve_linear(matrix(args[0]), matrix(args[1])))


@symfn("dsolve")
def _dsolve(interp, args, nargout):
    sp = sympy()
    from sympy.core.function import AppliedUndef
    args, _opts = split_options(args, {"ignoreanalyticconstraints", "maxdegree", "implicit", "expansionpoint",
                                       "order"})
    items = []
    for a in args:
        if is_text(a) or isinstance(a, MString):
            raise MatlabError("dsolve expects symbolic equations, e.g.  syms y(t);  "
                              "dsolve(diff(y, t) == -2*y, y(0) == 1).")
        items += flat_list(a)
    eqs, ics = [], {}
    for e in items:
        if e is sp.true:
            continue
        if not isinstance(e, sp.Eq):
            e = sp.Eq(e, 0, evaluate=False)
        applied = e.atoms(AppliedUndef)
        is_condition = bool(applied) and not any(a.free_symbols & e.free_symbols for a in applied)
        if is_condition:
            lhs, rhs = e.lhs, e.rhs
            if not (isinstance(lhs, (AppliedUndef, sp.Subs))) and isinstance(rhs, (AppliedUndef, sp.Subs)):
                lhs, rhs = rhs, lhs
            if not isinstance(lhs, (AppliedUndef, sp.Subs)):
                raise MatlabError("An initial condition must look like y(0) == 1 or Dy(0) == 0.")
            ics[lhs] = rhs
        else:
            eqs.append(sp.Eq(e.lhs, e.rhs))
    if not eqs:
        raise MatlabError("dsolve expects a differential equation written with diff, e.g. "
                          "diff(y, t) == -2*y.")
    funcs = set()
    for e in eqs:
        if e in (sp.true, sp.false):
            raise MatlabError("The differential equation reduces to a constant; check that the unknown "
                              "was declared as a function, e.g. syms y(t).")
        funcs |= {a for a in e.atoms(AppliedUndef) if a.free_symbols}
    funcs = sorted(funcs, key=lambda f: f.func.__name__)
    if not funcs:
        raise MatlabError("dsolve found no unknown function. Declare it with  syms y(t).")
    try:
        if len(eqs) == 1 and len(funcs) == 1:
            sol = sp.dsolve(eqs[0], funcs[0], ics=ics or None)
        else:
            sol = sp.dsolve(eqs, funcs, ics=ics or None)
    except (NotImplementedError, ValueError, TypeError) as err:
        raise MatlabError(f"dsolve could not find an explicit solution ({err}).") from None
    sols = sol if isinstance(sol, (list, tuple)) else [sol]
    by_name = {}
    for s in sols:
        if isinstance(s, sp.Eq) and isinstance(s.lhs, AppliedUndef):
            by_name[s.lhs.func.__name__] = sp.simplify(s.rhs) if ics else sp.expand(s.rhs)
        else:
            by_name.setdefault("implicit", s)
    names = sorted(by_name)
    if len(names) == 1:
        return wrap(by_name[names[0]])
    if nargout <= 1:
        return _struct({n: wrap(by_name[n]) for n in names})
    return tuple(wrap(by_name[n]) for n in names[:nargout])


# -- integral transforms ----------------------------------------------------------------------

def _transform_vars(interp, args: list, default_in: str, default_out: str, alt_out: str):
    """(expression value, input variable, output variable) for laplace-style calls."""
    f = args[0]
    names = {s.name: s for s in free_symbols(f)}
    if len(args) >= 3:
        return f, scalar(args[1], "variable"), scalar(args[2], "transform variable")
    var = names.get(default_in) or (default_var(f) if names else make_symbol(interp, default_in))
    if isinstance(f, SymFun) and default_in not in names:
        var = f.vars[0]
    if len(args) == 2:
        return f, var, scalar(args[1], "transform variable")
    out = make_symbol(interp, default_out if var.name != default_out else alt_out)
    return f, var, out


@symfn("laplace")
def _laplace(interp, args, nargout):
    sp = sympy()
    f, t, s = _transform_vars(interp, args, "t", "s", "z")
    tp = sp.Dummy(t.name, positive=True)

    def one(e):
        try:
            res = sp.laplace_transform(e.subs(t, tp), tp, s, noconds=True)
        except (NotImplementedError, ValueError):
            return sp.Function("laplace")(e, t, s)
        return res.subs(tp, t)

    return map_data(one, f)


@symfn("ilaplace")
def _ilaplace(interp, args, nargout):
    sp = sympy()
    f, s, t = _transform_vars(interp, args, "s", "t", "x")
    tp = sp.Dummy(t.name, positive=True)

    def one(e):
        try:
            res = sp.inverse_laplace_transform(e, s, tp)
        except (NotImplementedError, ValueError):
            return sp.Function("ilaplace")(e, s, t)
        return sp.simplify(res).subs(tp, t)

    return map_data(one, f)


def _fourier_core(e, x, w):
    """Integral of e(x)*exp(-i*w*x) over the real line, or None when it cannot be found."""
    sp = sympy()
    if not e.has(x):
        return 2 * sp.pi * e * sp.DiracDelta(w)
    xr = sp.Dummy(x.name, real=True)
    wr = sp.Dummy(w.name, real=True)
    body = e.subs(x, xr)
    try:
        if body.has(sp.DiracDelta):
            res = sp.integrate(body * sp.exp(-sp.I * wr * xr), (xr, -sp.oo, sp.oo))
        else:
            k = sp.Dummy("k", real=True)
            res = sp.fourier_transform(body, xr, k)
            if res.has(sp.FourierTransform):
                res = sp.integrate(body * sp.exp(-sp.I * wr * xr), (xr, -sp.oo, sp.oo), conds="none")
            else:
                res = res.subs(k, wr / (2 * sp.pi))
    except (NotImplementedError, ValueError, TypeError):
        return None
    if res.has(sp.Integral, sp.FourierTransform, sp.nan, sp.zoo, sp.oo):
        return None
    return sp.simplify(res).subs(wr, w)


@symfn("fourier")
def _fourier(interp, args, nargout):
    sp = sympy()
    f, x, w = _transform_vars(interp, args, "x", "w", "v")

    def one(e):
        res = _fourier_core(sp.expand(e) if e.is_Add else e, x, w)
        return sp.Function("fourier")(e, x, w) if res is None else res

    return map_data(one, f)


@symfn("ifourier")
def _ifourier(interp, args, nargout):
    sp = sympy()
    f, w, x = _transform_vars(interp, args, "w", "x", "t")

    def one(e):
        # (1/2pi) * integral of F(w)*exp(i*w*x) dw  =  fourier(F)(-x) / (2*pi)
        xm = sp.Dummy("xm")
        res = _fourier_core(e, w, xm)
        if res is None:
            return sp.Function("ifourier")(e, w, x)
        return sp.simplify(res.subs(xm, -x) / (2 * sp.pi))

    return map_data(one, f)


@symfn("ztrans")
def _ztrans(interp, args, nargout):
    sp = sympy()
    f, n, z = _transform_vars(interp, args, "n", "z", "w")
    k = sp.Dummy("k", integer=True, nonnegative=True)

    def one(e):
        body = e.subs(n, k)
        if body.has(sp.KroneckerDelta):
            body = body.rewrite(sp.Piecewise)
        try:
            res = sp.summation(body * z ** (-k), (k, 0, sp.oo))
        except (NotImplementedError, ValueError):
            res = None
        if isinstance(res, sp.Piecewise):
            res = res.args[0][0]
        if res is None or res.has(sp.Sum):
            return sp.Function("ztrans")(e, n, z)
        return sp.simplify(res)

    return map_data(one, f)


@symfn("iztrans")
def _iztrans(interp, args, nargout):
    sp = sympy()
    f, z, n = _transform_vars(interp, args, "z", "n", "k")

    def one(e):
        fail = sp.Function("iztrans")(e, z, n)
        try:
            g = sp.apart(sp.cancel(e / z), z)
            if any(sp.degree(sp.fraction(t)[1], z) > 1 and not sp.fraction(sp.factor(t))[1].is_Pow
                   for t in sp.Add.make_args(g)):
                g = sp.apart(sp.cancel(e / z), z, full=True).doit()
        except (sp.PolynomialError, NotImplementedError):
            return fail
        total = sp.Integer(0)
        for term in sp.Add.make_args(g):
            num, den = sp.fraction(sp.factor(term))
            if num.has(z):
                return fail
            coeff, factors = sp.factor_list(den, z)
            num = num / coeff
            if not factors:  # a constant c: F has the term c*z, which is not causal
                return fail
            if len(factors) != 1:
                return fail
            base, m = factors[0]
            poly = sp.Poly(base, z)
            if poly.degree() != 1:
                return fail
            lead, const = poly.all_coeffs()
            a = -const / lead
            num = num / lead ** m
            if a == 0:  # c / z^m  ->  c * delta(n - (m - 1))
                total += num * sp.KroneckerDelta(n, m - 1)
            else:
                total += num * sp.binomial(n, m - 1) * a ** (n - m + 1)
        return sp.simplify(total) if total.has(sp.I) else sp.expand_func(total)

    return map_data(one, f)


# -- linear algebra ---------------------------------------------------------------------------

def _square(args: list, name: str):
    m = matrix(args[0])
    if m.rows != m.cols:
        raise MatlabError(f"{name} expects a square matrix.")
    return m


@symfn("det")
def _det(interp, args, nargout):
    return wrap(_square(args, "det").det())


@symfn("inv")
def _inv(interp, args, nargout):
    sp = sympy()
    m = _square(args, "inv")
    try:
        return wrap(m.inv())
    except (ValueError, sp.matrices.exceptions.NonInvertibleMatrixError):
        raise MatlabError("The symbolic matrix is singular and has no inverse.") from None


@symfn("pinv")
def _pinv(interp, args, nargout):
    return wrap(matrix(args[0]).pinv())


@symfn("rank")
def _rank(interp, args, nargout):
    return wrap(sympy().Integer(matrix(args[0]).rank()))


@symfn("trace")
def _trace(interp, args, nargout):
    return wrap(_square(args, "trace").trace())


@symfn("rref")
def _rref(interp, args, nargout):
    return wrap(matrix(args[0]).rref()[0])


@symfn("adjoint")
def _adjoint(interp, args, nargout):
    return wrap(_square(args, "adjoint").adjugate())


@symfn("expm")
def _expm(interp, args, nargout):
    sp = sympy()
    return wrap(_square(args, "expm").exp().applyfunc(sp.simplify))


def _basis(vectors: list, rows: int) -> SymObject:
    if not vectors:
        return SymObject(np.empty((rows, 0), dtype=object))
    sp = sympy()
    return wrap(sp.Matrix.hstack(*vectors))


@symfn("null")
def _null(interp, args, nargout):
    m = matrix(args[0])
    return _basis(m.nullspace(), m.cols)


@symfn("colspace")
def _colspace(interp, args, nargout):
    m = matrix(args[0])
    return _basis(m.columnspace(), m.rows)


@symfn("eig")
def _eig(interp, args, nargout):
    sp = sympy()
    m = _square(args, "eig")
    if nargout <= 1:
        vals = []
        for val, mult in m.eigenvals().items():
            vals += [val] * int(mult)
        try:
            vals.sort(key=lambda v: (complex(v.evalf()).real, complex(v.evalf()).imag))
        except (TypeError, ValueError):
            pass
        return column(vals)
    vecs, vals = [], []
    triples = m.eigenvects()
    try:
        triples.sort(key=lambda t: (complex(t[0].evalf()).real, complex(t[0].evalf()).imag))
    except (TypeError, ValueError):
        pass
    for val, _mult, basis in triples:
        for v in basis:
            vecs.append(v.applyfunc(sp.simplify))
            vals.append(val)
    return _basis(vecs, m.rows), wrap(sp.diag(*vals) if vals else sp.Matrix(0, 0, []))


@symfn("charpoly")
def _charpoly(interp, args, nargout):
    sp = sympy()
    m = _square(args, "charpoly")
    if len(args) > 1:
        return wrap(m.charpoly(scalar(args[1], "variable")).as_expr())
    return row(m.charpoly(sp.Dummy("lambda")).all_coeffs())

