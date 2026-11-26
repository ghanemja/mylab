"""ODE suite: option structures, initial value solvers (mass matrices, events, dense output,
output functions), fully implicit equations, solution evaluation and extension.

Boundary value problems, delay equations, 1-D parabolic PDEs and the ``ode`` object live in
``_ode_extra``; numerical integration lives in ``_quad``. Both are loaded from here."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, Struct, char, empty_double, simplify
from ._registry import builtin
from ._util import fnum, is_text, nargs, text

# -- option structures --------------------------------------------------------------------------

_ODE_NAMES = ["AbsTol", "BDF", "Events", "InitialStep", "Jacobian", "JConstant", "JPattern", "Mass",
              "MassSingular", "MaxOrder", "MaxStep", "NonNegative", "NormControl", "OutputFcn",
              "OutputSel", "Refine", "RelTol", "Stats", "Vectorized", "MStateDependence", "MvPattern",
              "InitialSlope"]
_ODE_HINTS = {
    "AbsTol": "positive scalar or vector {1e-6}", "BDF": "on | {off}", "Events": "function handle",
    "InitialStep": "positive scalar", "Jacobian": "matrix | function handle", "JConstant": "on | {off}",
    "JPattern": "sparse matrix", "Mass": "matrix | function handle",
    "MassSingular": "yes | no | {maybe}", "MaxOrder": "1 | 2 | 3 | 4 | {5}", "MaxStep": "positive scalar",
    "NonNegative": "vector of integers", "NormControl": "on | {off}", "OutputFcn": "function handle",
    "OutputSel": "vector of integers", "Refine": "positive integer", "RelTol": "positive scalar {1e-3}",
    "Stats": "on | {off}", "Vectorized": "on | {off}", "MStateDependence": "none | {weak} | strong",
    "MvPattern": "sparse matrix", "InitialSlope": "vector",
}
_BVP_NAMES = ["AbsTol", "RelTol", "SingularTerm", "FJacobian", "BCJacobian", "Stats", "NMax", "Vectorized"]
_DDE_NAMES = ["AbsTol", "Events", "InitialStep", "InitialY", "Jumps", "MaxStep", "NormControl", "OutputFcn",
              "OutputSel", "Refine", "RelTol", "Stats"]


def _is_empty(v) -> bool:
    return isinstance(v, np.ndarray) and v.size == 0


def match_option(name: str, names: list, what: str) -> str:
    """Canonical option name for a case-insensitive name or unambiguous prefix."""
    low = name.lower()
    for n in names:
        if n.lower() == low:
            return n
    hits = [n for n in names if n.lower().startswith(low)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise MatlabError(f"Unrecognized property name '{name}'. See {what} for the list of options.")
    raise MatlabError(f"Ambiguous property name '{name}' ({', '.join(hits)}).")


def _empty_options(names: list) -> Struct:
    return Struct.scalar({n: empty_double() for n in names})


def _set_options(args: list, names: list, what: str, nargout: int, interp, hints=None):
    if not args and nargout == 0:
        width = max(len(n) for n in names)
        for n in names:
            interp.out(f"{n:>{width + 4}}: [ {(hints or {}).get(n, '')} ]\n")
        return None
    out = _empty_options(names)
    rest = list(args)
    while rest and (isinstance(rest[0], Struct) or _is_empty(rest[0])):
        s = rest.pop(0)
        if isinstance(s, Struct):
            if s.numel() != 1:
                raise MatlabError(f"{what}: an options structure must be 1-by-1.")
            for k, v in s.data.flat[0].items():
                if not _is_empty(v):
                    out.set_field(0, match_option(k, names, what), v)
    if len(rest) % 2:
        raise MatlabError(f"{what}: arguments must be name-value pairs.")
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]):
            raise MatlabError(f"{what}: expected an option name at argument {k + 1}.")
        out.set_field(0, match_option(text(rest[k]), names, what), rest[k + 1])
    return out


def _get_option(args: list, names: list, what: str):
    nargs(args, 2, 3, what)
    s = args[0]
    default = args[2] if len(args) > 2 else empty_double()
    if _is_empty(s):
        return default
    if not isinstance(s, Struct):
        raise MatlabError(f"{what}: the first argument must be an options structure.")
    name = match_option(text(args[1]), names, what)
    for k, v in s.data.flat[0].items():
        if k.lower() == name.lower() and not _is_empty(v):
            return v
    return default


@builtin("odeset")
def _odeset(interp, args, nargout):
    return _set_options(args, _ODE_NAMES, "odeset", nargout, interp, _ODE_HINTS)


@builtin("odeget")
def _odeget(interp, args, nargout):
    return _get_option(args, _ODE_NAMES, "odeget")


@builtin("bvpset")
def _bvpset(interp, args, nargout):
    return _set_options(args, _BVP_NAMES, "bvpset", nargout, interp)


@builtin("bvpget")
def _bvpget(interp, args, nargout):
    return _get_option(args, _BVP_NAMES, "bvpget")


@builtin("ddeset")
def _ddeset(interp, args, nargout):
    return _set_options(args, _DDE_NAMES, "ddeset", nargout, interp)


@builtin("ddeget")
def _ddeget(interp, args, nargout):
    return _get_option(args, _DDE_NAMES, "ddeget")


class Opts:
    """Reads values out of an options structure (unknown or empty fields are skipped)."""

    def __init__(self, s, names: list, what: str):
        self.vals: dict = {}
        self.what = what
        self.struct = s if isinstance(s, Struct) else None
        if s is None or _is_empty(s):
            return
        if not isinstance(s, Struct) or s.numel() != 1:
            raise MatlabError(f"{what}: the options argument must be a structure made with odeset, "
                              "bvpset or ddeset.")
        lower = {n.lower(): n for n in names}
        for k, v in s.data.flat[0].items():
            name = lower.get(k.lower())
            if name is not None and not _is_empty(v):
                self.vals[name] = v

    def has(self, name) -> bool:
        return name in self.vals

    def raw(self, name, default=None):
        return self.vals.get(name, default)

    def num(self, name, default=None):
        if name not in self.vals:
            return default
        v = self.vals[name]
        if is_text(v) or isinstance(v, (Struct, Cell, FunctionHandle)):
            raise MatlabError(f"{self.what}: option {name} must be a number.")
        a = fnum(v)
        if a.size != 1:
            raise MatlabError(f"{self.what}: option {name} must be a scalar.")
        return float(np.real(a.flat[0]))

    def vec(self, name):
        if name not in self.vals:
            return None
        v = self.vals[name]
        if is_text(v) or isinstance(v, (Struct, Cell, FunctionHandle)):
            raise MatlabError(f"{self.what}: option {name} must be numeric.")
        return fnum(v).flatten(order="F")

    def onoff(self, name, default=False) -> bool:
        if name not in self.vals:
            return default
        v = self.vals[name]
        if is_text(v):
            w = text(v).lower()
            if w in ("on", "yes", "true"):
                return True
            if w in ("off", "no", "false"):
                return False
            raise MatlabError(f"{self.what}: option {name} must be 'on' or 'off'.")
        a = fnum(v)
        if a.size != 1:
            raise MatlabError(f"{self.what}: option {name} must be 'on' or 'off'.")
        return bool(a.flat[0] != 0)

    def word(self, name, default, allowed):
        if name not in self.vals:
            return default
        v = self.vals[name]
        if not is_text(v):
            raise MatlabError(f"{self.what}: option {name} must be one of {', '.join(allowed)}.")
        w = text(v).lower()
        if w not in allowed:
            raise MatlabError(f"{self.what}: option {name} must be one of {', '.join(allowed)}.")
        return w


# -- building a problem from MATLAB arguments ------------------------------------------------------

def _is_fn(v) -> bool:
    return isinstance(v, FunctionHandle) or (is_text(v) and len(text(v)) > 0)


def _scalar(t) -> np.ndarray:
    return np.array([[float(t)]])


def _col(y) -> np.ndarray:
    return y.reshape(-1, 1)


def _real_if_possible(v: np.ndarray) -> np.ndarray:
    if v.dtype.kind == "c" and not np.any(v.imag):
        return v.real
    return v


def check_tspan(tspan_arg, what: str) -> np.ndarray:
    if not isinstance(tspan_arg, np.ndarray) or tspan_arg.dtype.kind not in "biuf":
        raise MatlabError(f"{what}: tspan must be a real numeric vector [t0 tfinal].")
    tspan = fnum(tspan_arg).flatten(order="F").astype(float)
    if tspan.size < 2:
        raise MatlabError(f"{what}: tspan must have at least two elements, [t0 tfinal].")
    if not np.all(np.isfinite(tspan)):
        raise MatlabError(f"{what}: the entries of tspan must be finite.")
    d = np.diff(tspan)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise MatlabError(f"{what}: the entries in tspan must strictly increase or decrease.")
    return tspan


def make_ivp(interp, solver: str, fn, tspan, y0, opts: Opts, extra: list):
    """Translate MATLAB-level arguments into a driver problem."""
    from ._ode_driver import IVP, is_stiff_solver
    n = y0.size
    extra = list(extra)

    def call(f, args, nout=1):
        return interp.call_function_value(f, args, nout)

    def rhs(t, y):
        out = call(fn, [_scalar(t), _col(y), *extra])
        if not out:
            raise MatlabError(f"{solver}: the ODE function must return a value.")
        v = fnum(out[0])
        if v.size != n:
            raise MatlabError(f"{solver}: the ODE function must return a column vector with {n} elements "
                              f"(it returned {v.size}).")
        return _real_if_possible(v.reshape(-1, order="F"))

    def rhs_mat(t, Y):
        out = call(fn, [_scalar(t), Y, *extra])
        v = fnum(out[0])
        if v.shape != Y.shape:
            raise MatlabError(f"{solver}: with Vectorized 'on', the ODE function must return one column "
                              "per column of y.")
        return v

    P = IVP(solver, rhs, tspan, y0)
    P.rhs_mat = rhs_mat
    P.warn = interp.warn
    rtol = opts.num("RelTol", 1e-3)
    if not rtol > 0:
        raise MatlabError(f"{solver}: RelTol must be a positive scalar.")
    if rtol < 100 * np.finfo(float).eps:
        rtol = 100 * np.finfo(float).eps
        interp.warn(f"RelTol has been increased to {rtol:g}.")
    P.rtol = rtol
    atol = opts.vec("AbsTol")
    if atol is None:
        atol = np.array([1e-6])
    atol = np.real(atol).astype(float)
    if np.any(atol <= 0):
        raise MatlabError(f"{solver}: AbsTol must be positive.")
    if atol.size not in (1, n):
        raise MatlabError(f"{solver}: AbsTol must be a scalar or a vector with one entry per component ({n}).")
    P.normcontrol = opts.onoff("NormControl")
    if P.normcontrol and atol.size != 1:
        raise MatlabError(f"{solver}: with NormControl 'on', AbsTol must be a scalar.")
    P.atol = atol if atol.size > 1 else atol.reshape(1)
    hmax = opts.num("MaxStep")
    if hmax is not None:
        if not hmax > 0:
            raise MatlabError(f"{solver}: MaxStep must be positive.")
        P.hmax = hmax
    h0 = opts.num("InitialStep")
    if h0 is not None:
        if not h0 > 0:
            raise MatlabError(f"{solver}: InitialStep must be positive.")
        P.h0 = h0
    refine = opts.num("Refine")
    if refine is not None:
        if refine < 1 or refine != int(refine):
            raise MatlabError(f"{solver}: Refine must be a positive integer.")
        P.refine = int(refine)
    nn = opts.vec("NonNegative")
    if nn is not None:
        idx = np.real(nn).astype(int)
        if np.any(idx < 1) or np.any(idx > n) or np.any(idx != np.real(nn)):
            raise MatlabError(f"{solver}: NonNegative must list component numbers between 1 and {n}.")
        if solver in ("ode23s", "ode15i"):
            raise MatlabError(f"{solver} does not support the NonNegative option.")
        if is_stiff_solver(solver) and opts.has("Mass"):
            interp.warn(f"{solver}: NonNegative is ignored for problems with a mass matrix.")
        else:
            P.nonneg = np.unique(idx - 1)
    # mass matrix
    if opts.has("Mass") and solver != "ode15i":
        Mv = opts.raw("Mass")
        P.mstate = opts.word("MStateDependence", "weak", ("none", "weak", "strong"))
        P.mass_singular = opts.word("MassSingular", "maybe", ("yes", "no", "maybe"))
        if _is_fn(Mv):
            def mass(t, y):
                a = [_scalar(t)] if P.mstate == "none" else [_scalar(t), _col(y)]
                out = call(Mv, a + extra)
                if not out:
                    raise MatlabError(f"{solver}: the mass function must return a matrix.")
                M = _real_if_possible(fnum(out[0]))
                if M.shape != (n, n):
                    raise MatlabError(f"{solver}: the mass matrix must be {n}-by-{n}.")
                return M
            P.mass = ("fun", mass)
        else:
            M = _real_if_possible(fnum(Mv))
            if M.shape != (n, n):
                raise MatlabError(f"{solver}: the mass matrix must be {n}-by-{n}.")
            P.mass = ("const", M)
        if opts.has("MvPattern"):
            P.mvpattern = fnum(opts.raw("MvPattern")) != 0
    # Jacobian
    if opts.has("Jacobian") and solver != "ode15i":
        Jv = opts.raw("Jacobian")
        if _is_fn(Jv):
            def jac(t, y):
                out = call(Jv, [_scalar(t), _col(y), *extra])
                if not out:
                    raise MatlabError(f"{solver}: the Jacobian function must return a matrix.")
                J = _real_if_possible(fnum(out[0]))
                if J.shape != (n, n):
                    raise MatlabError(f"{solver}: the Jacobian must be {n}-by-{n}.")
                return J
            if opts.onoff("JConstant"):
                J0 = jac(float(tspan[0]), y0)
                P.jac = ("const", J0)
            else:
                P.jac = ("fun", jac)
        else:
            J = _real_if_possible(fnum(Jv))
            if J.shape != (n, n):
                raise MatlabError(f"{solver}: the Jacobian must be {n}-by-{n}.")
            P.jac = ("const", J)
    if opts.has("JPattern") and solver != "ode15i":
        P.jpattern = fnum(opts.raw("JPattern")) != 0
    P.vectorized = opts.onoff("Vectorized")
    slope = opts.vec("InitialSlope")
    if slope is not None:
        if slope.size != n:
            raise MatlabError(f"{solver}: InitialSlope must have {n} elements.")
        P.initial_slope = _real_if_possible(slope)
    mo = opts.num("MaxOrder")
    if mo is not None:
        if mo not in (1, 2, 3, 4, 5):
            raise MatlabError(f"{solver}: MaxOrder must be an integer from 1 to 5.")
        P.max_order = int(mo)
    P.bdf = opts.onoff("BDF")
    # events
    if opts.has("Events"):
        ev_fn = opts.raw("Events")
        if not _is_fn(ev_fn):
            raise MatlabError(f"{solver}: Events must be a function handle.")
        P.events = _event_caller(interp, ev_fn, extra, solver == "ode15i", solver)
        P.events_yp = solver == "ode15i"
    # output function
    out_fn = opts.raw("OutputFcn")
    sel = opts.vec("OutputSel")
    if sel is not None:
        sel = np.real(sel).astype(int) - 1
        if np.any(sel < 0) or np.any(sel >= n):
            raise MatlabError(f"{solver}: OutputSel must list component numbers between 1 and {n}.")
    if out_fn is not None:
        if not _is_fn(out_fn):
            raise MatlabError(f"{solver}: OutputFcn must be a function handle.")
        P.outfcn = _output_caller(interp, out_fn, sel, extra)
    return P


def _event_caller(interp, ev_fn, extra, with_slope: bool, what: str):
    state = {"nout": 3}

    def ev(t, y, yp):
        a = [_scalar(t), _col(y)]
        if with_slope:
            a.append(_col(yp))
        out = interp.call_function_value(ev_fn, a + list(extra), 3)
        if len(out) < 3:
            raise MatlabError(f"{what}: the event function must return three outputs: "
                              "[value, isterminal, direction].")
        v = np.real(fnum(out[0])).astype(float).flatten(order="F")
        term = np.real(fnum(out[1])).flatten(order="F") if not _is_empty(out[1]) else np.zeros(v.size)
        dirn = np.real(fnum(out[2])).flatten(order="F") if not _is_empty(out[2]) else np.zeros(v.size)
        if term.size == 1 and v.size != 1:
            term = np.full(v.size, term[0])
        if dirn.size == 1 and v.size != 1:
            dirn = np.full(v.size, dirn[0])
        if term.size != v.size or dirn.size != v.size:
            raise MatlabError(f"{what}: the outputs of the event function must have the same length.")
        return v, term != 0, dirn

    ev.state = state
    return ev


def _output_caller(interp, out_fn, sel, extra):
    def out(t_row, y_cols, flag):
        if flag == "done":
            a = [empty_double(), empty_double(), char("done")]
        else:
            ysel = np.asarray(y_cols if sel is None else y_cols[sel, :])
            a = [np.asarray(t_row, dtype=float).reshape(1, -1), simplify(np.array(ysel)), char(flag)]
        res = interp.call_function_value(out_fn, a + list(extra), 1 if flag == "" else 0)
        if flag == "" and res:
            v = res[0]
            if isinstance(v, np.ndarray) and v.size:
                return bool(np.any(v != 0))
        return False

    return out


# -- running a solver and shaping its outputs ---------------------------------------------------

def _stats_struct(cnt, stiff: bool) -> Struct:
    d = {"nsteps": float(cnt.nsteps), "nfailed": float(cnt.nfailed), "nfevals": float(cnt.nfevals)}
    if stiff:
        d.update({"npds": float(cnt.npds), "ndecomps": float(cnt.ndecomps), "nsolves": float(cnt.nsolves)})
    return Struct.scalar(d)


def _print_stats(interp, cnt, stiff: bool) -> None:
    interp.out(f"{cnt.nsteps} successful steps\n{cnt.nfailed} failed attempts\n"
               f"{cnt.nfevals} function evaluations\n")
    if stiff:
        interp.out(f"{cnt.npds} partial derivatives\n{cnt.ndecomps} LU decompositions\n"
                   f"{cnt.nsolves} solutions of linear systems\n")


def pack_coefs(coefs: list, n: int) -> np.ndarray:
    """Stack per-step polynomial coefficients into an n x (d+1) x nsteps array."""
    if not coefs:
        return np.zeros((n, 0, 0))
    deg = max(c.shape[1] for c in coefs)
    dtype = np.result_type(*[c.dtype for c in coefs])
    out = np.zeros((n, deg, len(coefs)), dtype=dtype)
    for k, c in enumerate(coefs):
        out[:, :c.shape[1], k] = c
    return out


def result_to_sol(res, solver: str, n: int, extdata: Struct, stiff: bool) -> Struct:
    fields = {"solver": char(solver), "extdata": extdata,
              "x": np.array(res.x, dtype=float).reshape(1, -1),
              "y": simplify(np.column_stack(res.y)) if res.y else np.zeros((n, 0))}
    if res.te is not None and res.has_events:
        fields["xe"] = np.array(res.te, dtype=float).reshape(1, -1)
        fields["ye"] = simplify(np.column_stack(res.ye)) if res.ye else np.zeros((n, 0))
        fields["ie"] = np.array(res.ie, dtype=float).reshape(1, -1)
    fields["stats"] = _stats_struct(res.counters, stiff)
    fields["idata"] = Struct.scalar({"kind": char("poly"), "coefs": pack_coefs(res.coefs, n)})
    return Struct.scalar(fields)


def run_ivp(interp, P, opts: Opts, nargout: int, extdata: Struct | None = None):
    from ._ode_driver import integrate, is_stiff_solver
    stiff = is_stiff_solver(P.solver)
    n = P.y0.size
    plot_it = nargout == 0 and P.outfcn is None
    res = integrate(P, want_dense=nargout == 1)
    res.has_events = P.events is not None
    if res.message:
        interp.warn(res.message)
    if opts.onoff("Stats"):
        _print_stats(interp, res.counters, stiff)
    if nargout == 1:
        if extdata is None:
            extdata = Struct.scalar({})
        return result_to_sol(res, P.solver, n, extdata, stiff)
    t = np.array(res.tout, dtype=float).reshape(-1, 1)
    y = simplify(np.vstack([np.asarray(v).reshape(1, -1) for v in res.yout])) if res.yout else np.zeros((0, n))
    if plot_it:
        if "plot" in interp.builtins:
            interp.call_builtin("plot", interp.builtins["plot"], [t, np.real(y), char("-o")], 0)
        return None
    if nargout == 0:
        return None
    outs = [t, y]
    if nargout > 2:
        te = np.array(res.te, dtype=float).reshape(-1, 1) if res.te else np.zeros((0, 1))
        ye = simplify(np.vstack([np.asarray(v).reshape(1, -1) for v in res.ye])) if res.ye else np.zeros((0, n))
        ie = np.array(res.ie, dtype=float).reshape(-1, 1) if res.ie else np.zeros((0, 1))
        outs += [te, ye, ie]
    if nargout > len(outs):
        raise MatlabError(f"{P.solver}: too many output arguments.")
    return tuple(outs[:nargout])


def _ode_front(solver: str):
    def f(interp, args, nargout):
        nargs(args, 3, None, solver)
        fn = args[0]
        if not _is_fn(fn):
            raise MatlabError(f"{solver}: the first argument must be a function handle.")
        tspan = check_tspan(args[1], solver)
        if isinstance(args[2], (Struct, Cell, FunctionHandle)) or is_text(args[2]):
            raise MatlabError(f"{solver}: the initial condition y0 must be a numeric vector.")
        y0 = fnum(args[2]).flatten(order="F")
        if y0.size == 0:
            raise MatlabError(f"{solver}: the initial condition y0 must not be empty.")
        y0 = y0.astype(complex if np.iscomplexobj(y0) and np.any(y0.imag) else float)
        opt_arg = args[3] if len(args) > 3 else None
        opts = Opts(opt_arg, _ODE_NAMES, solver)
        extra = list(args[4:])
        P = make_ivp(interp, solver, fn, tspan, y0, opts, extra)
        extdata = Struct.scalar({"odefun": fn, "options": opt_arg if isinstance(opt_arg, Struct) else empty_double(),
                                 "varargin": Cell.from_list(extra)})
        return run_ivp(interp, P, opts, nargout, extdata)

    return f


for _name in ("ode45", "ode23", "ode113", "ode78", "ode89", "ode15s", "ode23s", "ode23t", "ode23tb"):
    builtin(_name)(_ode_front(_name))


# -- fully implicit equations --------------------------------------------------------------------

def _implicit_pieces(interp, what, fn, opts: Opts, n: int, extra: list):
    def G(t, y, yp):
        out = interp.call_function_value(fn, [_scalar(t), _col(y), _col(yp), *extra], 1)
        if not out:
            raise MatlabError(f"{what}: the function must return the residual f(t, y, yp).")
        v = fnum(out[0])
        if v.size != n:
            raise MatlabError(f"{what}: the residual must have {n} elements (it has {v.size}).")
        return _real_if_possible(v.reshape(-1, order="F"))

    jac = None
    pattern = None
    if opts.has("Jacobian"):
        Jv = opts.raw("Jacobian")
        if isinstance(Jv, Cell):
            mats = Jv.values_f()
            if len(mats) != 2:
                raise MatlabError(f"{what}: a constant Jacobian must be a cell {{dfdy, dfdyp}}.")
            A, B = (None if _is_empty(m) else fnum(m) for m in mats)
        elif _is_fn(Jv):
            A = B = None
        else:
            raise MatlabError(f"{what}: Jacobian must be a function handle or a cell {{dfdy, dfdyp}}.")
        typical = np.full(n, 1e-3)

        def jac(t, y, yp):
            from . import _ode_steppers as st
            if _is_fn(Jv):
                out = interp.call_function_value(Jv, [_scalar(t), _col(y), _col(yp), *extra], 2)
                a = None if not out or _is_empty(out[0]) else fnum(out[0])
                b = None if len(out) < 2 or _is_empty(out[1]) else fnum(out[1])
            else:
                a, b = A, B
            if a is None or b is None:
                g0 = G(t, y, yp)
                if a is None:
                    a, _ = st.fd_jacobian(lambda v: G(t, v, yp), y, g0, typical)
                if b is None:
                    b, _ = st.fd_jacobian(lambda v: G(t, y, v), yp, g0, typical)
            if a.shape != (n, n) or b.shape != (n, n):
                raise MatlabError(f"{what}: the Jacobians must be {n}-by-{n}.")
            return a, b
    if opts.has("JPattern"):
        pv = opts.raw("JPattern")
        if not isinstance(pv, Cell) or len(pv.values_f()) != 2:
            raise MatlabError(f"{what}: JPattern must be a cell {{dfdyPattern, dfdypPattern}}.")
        pats = []
        for m in pv.values_f():
            pats.append(np.ones((n, n), dtype=bool) if _is_empty(m) else fnum(m) != 0)
        pattern = tuple(pats)
    return G, jac, pattern


@builtin("ode15i")
def _ode15i(interp, args, nargout):
    nargs(args, 4, None, "ode15i")
    fn = args[0]
    if not _is_fn(fn):
        raise MatlabError("ode15i: the first argument must be a function handle f(t, y, yp).")
    tspan = check_tspan(args[1], "ode15i")
    y0 = fnum(args[2]).flatten(order="F")
    yp0 = fnum(args[3]).flatten(order="F")
    if y0.size == 0 or yp0.size != y0.size:
        raise MatlabError("ode15i: y0 and yp0 must be vectors of the same length.")
    cplx = np.iscomplexobj(y0) or np.iscomplexobj(yp0)
    y0, yp0 = y0.astype(complex if cplx else float), yp0.astype(complex if cplx else float)
    opt_arg = args[4] if len(args) > 4 else None
    opts = Opts(opt_arg, _ODE_NAMES, "ode15i")
    extra = list(args[5:])
    n = y0.size
    P = make_ivp(interp, "ode15i", fn, tspan, y0, opts, extra)
    P.implicit, P.implicit_jac, P.implicit_pattern = _implicit_pieces(interp, "ode15i", fn, opts, n, extra)
    P.yp0 = yp0
    r0 = P.implicit(float(tspan[0]), y0, yp0)
    scale = max(1.0, float(np.linalg.norm(y0)), float(np.linalg.norm(yp0)))
    if np.linalg.norm(r0) > 1e3 * max(P.rtol, 1e-6) * scale:
        interp.warn("ode15i: the initial conditions y0, yp0 do not satisfy f(t0, y0, yp0) = 0; "
                    "use decic to compute consistent values.")
    extdata = Struct.scalar({"odefun": fn, "options": opt_arg if isinstance(opt_arg, Struct) else empty_double(),
                             "varargin": Cell.from_list(extra)})
    return run_ivp(interp, P, opts, nargout, extdata)


@builtin("decic")
def _decic(interp, args, nargout):
    from . import _ode_steppers as st
    nargs(args, 6, None, "decic")
    fn = args[0]
    t0 = float(np.real(fnum(args[1]).flat[0]))
    y = fnum(args[2]).flatten(order="F").astype(float)
    yp = fnum(args[4]).flatten(order="F").astype(float)
    n = y.size
    if yp.size != n:
        raise MatlabError("decic: y0 and yp0 must have the same length.")

    def fixed(v, what):
        if _is_empty(v):
            return np.zeros(n, dtype=bool)
        a = fnum(v).flatten(order="F")
        if a.size != n:
            raise MatlabError(f"decic: {what} must have one entry per component ({n}) or be [].")
        return a != 0

    fy, fyp = fixed(args[3], "fixed_y0"), fixed(args[5], "fixed_yp0")
    if int(fy.sum() + fyp.sum()) > n:
        raise MatlabError(f"decic: at most {n} components of y0 and yp0 together can be held fixed.")
    opts = Opts(args[6] if len(args) > 6 else None, _ODE_NAMES, "decic")
    extra = list(args[7:])
    G, jac, _ = _implicit_pieces(interp, "decic", fn, opts, n, extra)
    rtol = opts.num("RelTol", 1e-3)
    atol_v = opts.vec("AbsTol")
    atol = float(np.min(atol_v)) if atol_v is not None else 1e-6
    free_y, free_yp = np.nonzero(~fy)[0], np.nonzero(~fyp)[0]
    typical = np.full(n, 1e-3)

    def mats(yv, ypv, g0):
        if jac is not None:
            return jac(t0, yv, ypv)
        a, _ = st.fd_jacobian(lambda v: G(t0, v, ypv), yv, g0, np.maximum(typical, 0))
        b, _ = st.fd_jacobian(lambda v: G(t0, yv, v), ypv, g0, typical)
        return a, b

    g = G(t0, y, yp)
    target = 1e-3 * max(rtol * max(np.linalg.norm(y), np.linalg.norm(yp)), atol)
    best = (float(np.linalg.norm(g)), y.copy(), yp.copy())
    for stage in ("slopes", "all"):
        yv, ypv = y.copy(), yp.copy()
        g = G(t0, yv, ypv)
        prev = float(np.linalg.norm(g))
        for _ in range(60):
            if prev <= min(target, 1e-12 * max(1.0, np.linalg.norm(yv))) or prev == 0:
                break
            A, B = mats(yv, ypv, g)
            if stage == "slopes":
                Jm = B[:, free_yp]
                if Jm.shape[1] == 0:
                    break
                d = np.linalg.lstsq(Jm, -g, rcond=None)[0]
                dy, dyp = np.zeros(n), np.zeros(n)
                dyp[free_yp] = d
            else:
                Jm = np.hstack([A[:, free_y], B[:, free_yp]])
                if Jm.shape[1] == 0:
                    break
                d = np.linalg.lstsq(Jm, -g, rcond=None)[0]
                dy, dyp = np.zeros(n), np.zeros(n)
                dy[free_y] = d[:free_y.size]
                dyp[free_yp] = d[free_y.size:]
            lam = 1.0
            for _ls in range(12):
                gn = G(t0, yv + lam * dy, ypv + lam * dyp)
                nn = float(np.linalg.norm(gn))
                if np.isfinite(nn) and nn < prev:
                    break
                lam *= 0.5
            else:
                break
            yv, ypv, g = yv + lam * dy, ypv + lam * dyp, gn
            if nn > 0.999 * prev and nn <= target:
                prev = nn
                break
            prev = nn
        if prev < best[0]:
            best = (prev, yv, ypv)
        if best[0] <= target:
            break
    resnrm, y_new, yp_new = best
    if resnrm > max(target, 1e-8 * max(1.0, np.linalg.norm(y_new))):
        raise MatlabError("decic: could not find consistent initial conditions; try different guesses "
                          "or free more components.")
    outs = (y_new.reshape(-1, 1), yp_new.reshape(-1, 1), resnrm)
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


# -- evaluating and extending solutions -------------------------------------------------------------

def sol_field(sol: Struct, name: str, default=None):
    if name in sol.fields:
        return sol.data.flat[0][name]
    return default


def eval_solution(sol: Struct, xq: np.ndarray, want_deriv: bool):
    """Values (n x numel(xq)) and optionally derivatives of a solution structure."""
    from . import _ode_steppers as st
    if not isinstance(sol, Struct) or "x" not in sol.fields or "y" not in sol.fields:
        raise MatlabError("deval needs a solution structure returned by an ODE, DDE or BVP solver.")
    x = np.real(fnum(sol_field(sol, "x"))).flatten(order="F").astype(float)
    y = fnum(sol_field(sol, "y"))
    if y.ndim != 2 or y.shape[1] != x.size:
        y = y.reshape(-1, x.size)
    n = y.shape[0]
    xq = np.asarray(xq, dtype=float)
    if x.size < 2:
        if xq.size and np.any(xq != x[0]):
            raise MatlabError("deval: the solution has a single point; it cannot be evaluated elsewhere.")
        return np.repeat(y[:, :1], xq.size, axis=1), np.zeros((n, xq.size))
    lo, hi = min(x[0], x[-1]), max(x[0], x[-1])
    span = hi - lo
    if xq.size and (xq.min() < lo - 1e-12 * max(span, abs(lo), 1e-300)
                    or xq.max() > hi + 1e-12 * max(span, abs(hi), 1e-300)):
        raise MatlabError(f"deval: attempting to evaluate the solution outside the interval "
                          f"[{lo:g}, {hi:g}] where it is defined.")
    if xq.size and not np.all(np.isfinite(xq)):
        raise MatlabError("deval: the evaluation points must be finite.")
    xq = np.clip(xq, lo, hi)
    forward = x[-1] >= x[0]
    xs = x if forward else -x
    qs = xq if forward else -xq
    idx = np.clip(np.searchsorted(xs, qs, side="right") - 1, 0, x.size - 2)
    h = x[idx + 1] - x[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(h != 0, (xq - x[idx]) / np.where(h != 0, h, 1.0), 0.0)
    idata = sol_field(sol, "idata")
    coefs = None
    if isinstance(idata, Struct) and "coefs" in idata.fields:
        c = fnum(idata.data.flat[0]["coefs"])
        if c.size and c.size % (n * (x.size - 1)) == 0:
            coefs = c.reshape(n, -1, x.size - 1)
    Y = np.zeros((n, xq.size), dtype=complex if np.iscomplexobj(y) or (coefs is not None and np.iscomplexobj(coefs))
                 else float)
    dY = np.zeros_like(Y) if want_deriv else None
    if coefs is not None:
        for k in np.unique(idx):
            m = idx == k
            Y[:, m] = st.eval_piece(coefs[:, :, k], s[m])
            if want_deriv:
                dY[:, m] = st.eval_piece_deriv(coefs[:, :, k], s[m]) / (x[k + 1] - x[k])
        exact = xq == x[idx + 1]
        if np.any(exact):
            Y[:, exact] = y[:, idx[exact] + 1]
        return Y, dY
    yp = sol_field(sol, "yp")
    if yp is not None and not _is_empty(yp) and fnum(yp).shape == y.shape:
        yp = fnum(yp)
        y0, y1, d0, d1 = y[:, idx], y[:, idx + 1], yp[:, idx], yp[:, idx + 1]
        dy = y1 - y0
        c2 = 3 * dy - h * (2 * d0 + d1)
        c3 = -2 * dy + h * (d0 + d1)
        Y = y0 + s * (h * d0 + s * (c2 + s * c3))
        if want_deriv:
            with np.errstate(invalid="ignore", divide="ignore"):
                dY = np.where(h != 0, (h * d0 + s * (2 * c2 + 3 * s * c3)) / np.where(h != 0, h, 1.0), d0)
        return Y, dY
    from scipy.interpolate import CubicSpline
    order = np.argsort(x, kind="stable")
    keep = np.concatenate(([True], np.diff(x[order]) != 0))
    cs = CubicSpline(x[order][keep], y[:, order][:, keep], axis=1)
    Y = cs(xq)
    if want_deriv:
        dY = cs(xq, 1)
    return Y, dY


@builtin("deval")
def _deval(interp, args, nargout):
    nargs(args, 2, 3, "deval")
    if isinstance(args[0], Struct):
        sol, xq_arg = args[0], args[1]
    elif isinstance(args[1], Struct):
        xq_arg, sol = args[0], args[1]
    else:
        raise MatlabError("deval needs a solution structure returned by an ODE, DDE or BVP solver.")
    xq = np.real(fnum(xq_arg)).flatten(order="F").astype(float)
    Y, dY = eval_solution(sol, xq, nargout > 1)
    if len(args) > 2 and not _is_empty(args[2]):
        rows = np.real(fnum(args[2])).flatten(order="F")
        if np.any(rows < 1) or np.any(rows > Y.shape[0]) or np.any(rows != np.floor(rows)):
            raise MatlabError(f"deval: the index must select components 1 to {Y.shape[0]}.")
        rows = rows.astype(int) - 1
        Y = Y[rows, :]
        if dY is not None:
            dY = dY[rows, :]
    if nargout > 1:
        return simplify(Y), simplify(dY)
    return simplify(Y)


@builtin("odextend")
def _odextend(interp, args, nargout):
    nargs(args, 3, None, "odextend")
    sol = args[0]
    if not isinstance(sol, Struct) or "solver" not in sol.fields or "x" not in sol.fields:
        raise MatlabError("odextend: the first argument must be a solution structure from an ODE solver.")
    solver = text(sol_field(sol, "solver"))
    if ("ode" + solver.replace("ode", "", 1)) not in ("ode45", "ode23", "ode113", "ode78", "ode89", "ode15s",
                                                    "ode23s", "ode23t", "ode23tb", "ode15i"):
        raise MatlabError(f"odextend: solutions from '{solver}' cannot be extended.")
    ext = sol_field(sol, "extdata")
    ext = ext if isinstance(ext, Struct) and ext.numel() == 1 else Struct.scalar({})
    fn = args[1]
    if _is_empty(fn):
        fn = sol_field(ext, "odefun")
        if fn is None or _is_empty(fn):
            raise MatlabError("odextend: the solution does not record its ODE function; pass it as the "
                              "second argument.")
    tfinal = float(np.real(fnum(args[2]).flat[0]))
    x = fnum(sol_field(sol, "x")).flatten()
    y = fnum(sol_field(sol, "y"))
    y = y.reshape(-1, x.size)
    n = y.shape[0]
    if tfinal == x[-1] or (x.size > 1 and np.sign(tfinal - x[-1]) != np.sign(x[-1] - x[0])):
        raise MatlabError("odextend: tfinal must lie beyond the end of the solution, in the direction "
                          "of integration.")
    old_opts = sol_field(ext, "options")
    old_extra = sol_field(ext, "varargin")
    extra = list(args[5:]) if len(args) > 5 else (old_extra.values_f() if isinstance(old_extra, Cell) else [])
    merged = _set_options([a for a in (old_opts, args[4] if len(args) > 4 else None) if isinstance(a, Struct)],
                          _ODE_NAMES, "odextend", 1, interp)
    opts = Opts(merged, _ODE_NAMES, "odextend")
    tspan = np.array([x[-1], tfinal])
    if solver == "ode15i":
        if len(args) > 3 and not _is_empty(args[3]):
            raise MatlabError("odextend: for ode15i pass the new initial values as a 2-column matrix "
                              "[y0 yp0], or [] to continue from the end of the solution.")
        y0 = y[:, -1].copy()
        _, dY = eval_solution(sol, np.array([x[-1]]), True)
        yp0 = dY[:, 0]
        P = make_ivp(interp, solver, fn, tspan, y0, opts, extra)
        P.implicit, P.implicit_jac, P.implicit_pattern = _implicit_pieces(interp, solver, fn, opts, n, extra)
        P.yp0 = np.array(yp0, dtype=y0.dtype)
    else:
        y0 = y[:, -1].copy() if len(args) < 4 or _is_empty(args[3]) else fnum(args[3]).flatten(order="F")
        if y0.size != n:
            raise MatlabError(f"odextend: y0 must have {n} elements.")
        y0 = y0.astype(complex if np.iscomplexobj(y0) and np.any(np.imag(y0)) else float)
        P = make_ivp(interp, solver, fn, tspan, y0, opts, extra)
    extdata = Struct.scalar({"odefun": fn, "options": merged, "varargin": Cell.from_list(extra)})
    new = run_ivp(interp, P, opts, 1, extdata)
    return join_solutions(sol, new, n)


def join_solutions(old: Struct, new: Struct, n: int) -> Struct:
    x1 = fnum(sol_field(old, "x")).flatten()
    y1 = fnum(sol_field(old, "y")).reshape(n, -1)
    x2 = fnum(sol_field(new, "x")).flatten()
    y2 = fnum(sol_field(new, "y")).reshape(n, -1)
    out = new.copy()
    out.set_field(0, "x", np.concatenate([x1, x2[1:]]).reshape(1, -1))
    out.set_field(0, "y", simplify(np.hstack([y1, y2[:, 1:]])))

    def coefs_of(s, nsteps):
        idata = sol_field(s, "idata")
        if isinstance(idata, Struct) and "coefs" in idata.fields:
            c = fnum(idata.data.flat[0]["coefs"])
            if nsteps and c.size and c.size % (n * nsteps) == 0:
                return c.reshape(n, -1, nsteps)
        return None

    c1, c2 = coefs_of(old, x1.size - 1), coefs_of(new, x2.size - 1)
    if c1 is not None and c2 is not None:
        deg = max(c1.shape[1], c2.shape[1])
        c = np.zeros((n, deg, c1.shape[2] + c2.shape[2]), dtype=np.result_type(c1.dtype, c2.dtype))
        c[:, :c1.shape[1], :c1.shape[2]] = c1
        c[:, :c2.shape[1], c1.shape[2]:] = c2
        out.set_field(0, "idata", Struct.scalar({"kind": char("poly"), "coefs": c}))
    elif c2 is not None and x1.size == 1:
        pass
    else:
        out.set_field(0, "idata", Struct.scalar({"kind": char("none")}))
    for name in ("xe", "ye", "ie"):
        a, b = sol_field(old, name), sol_field(new, name)
        if a is not None and b is not None:
            a, b = fnum(a), fnum(b)
            a = a.reshape(n if name == "ye" else 1, -1)
            b = b.reshape(n if name == "ye" else 1, -1)
            out.set_field(0, name, np.hstack([a, b]))
        elif a is not None:
            out.set_field(0, name, a)
    s1, s2 = sol_field(old, "stats"), sol_field(new, "stats")
    if isinstance(s1, Struct) and isinstance(s2, Struct):
        tot = {}
        for k, v in s2.data.flat[0].items():
            prev = s1.data.flat[0].get(k)
            tot[k] = float(fnum(v).flat[0]) + (float(fnum(prev).flat[0]) if prev is not None else 0.0)
        out.set_field(0, "stats", Struct.scalar(tot))
    return out


# -- simple output functions ----------------------------------------------------------------------

def _plot_output(kind: str):
    def f(interp, args, nargout):
        nargs(args, 3, None, kind)
        flag = text(args[2]) if is_text(args[2]) else ""
        store = interp.state.setdefault("_ode_output", {})
        if flag == "init":
            store[kind] = ([], [])
            y = fnum(args[1]).reshape(-1, 1)
            t = fnum(args[0]).flatten()
            store[kind][0].append(np.array([t[0]]))
            store[kind][1].append(y)
        elif flag == "done":
            ts, ys = store.pop(kind, ([], []))
            if ts and "plot" in interp.builtins:
                t = np.concatenate(ts)
                Y = np.hstack(ys)
                if kind == "odeplot":
                    a = [t.reshape(-1, 1), np.real(Y.T), char("-o")]
                    fn = "plot"
                elif kind == "odephas2":
                    a = [np.real(Y[0]).reshape(-1, 1), np.real(Y[1]).reshape(-1, 1), char("-o")]
                    fn = "plot"
                else:
                    a = [np.real(Y[k]).reshape(-1, 1) for k in range(3)] + [char("-o")]
                    fn = "plot3"
                if fn in interp.builtins:
                    interp.call_builtin(fn, interp.builtins[fn], a, 0)
        else:
            if kind in store:
                store[kind][0].append(fnum(args[0]).flatten())
                store[kind][1].append(fnum(args[1]).reshape(-1, fnum(args[0]).size))
        return np.array([[False]])

    return f


for _name in ("odeplot", "odephas2", "odephas3"):
    builtin(_name)(_plot_output(_name))


@builtin("odeprint")
def _odeprint(interp, args, nargout):
    nargs(args, 3, None, "odeprint")
    flag = text(args[2]) if is_text(args[2]) else ""
    if flag != "done":
        t = fnum(args[0]).flatten()
        y = fnum(args[1])
        if flag == "init":
            t = t[:1]
            y = y.reshape(-1, 1)
        else:
            y = y.reshape(-1, t.size)
        for k in range(t.size):
            interp.out(f"t = {t[k]:g}\n")
            interp.out("y = " + "  ".join(f"{float(np.real(v)):g}" for v in y[:, k]) + "\n")
    return np.array([[False]])


from . import _ode_extra, _quad  # noqa: E402,F401  (register the remaining functions)
