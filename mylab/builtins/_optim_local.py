"""Local optimisation solvers: root finding, 1-D and simplex minimisation, unconstrained and
constrained nonlinear programming, equation solving, least squares, linear, integer and
quadratic programming, minimax and goal attainment."""

from __future__ import annotations

import warnings

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char, empty_double
from ._registry import builtin
from ._util import fnum, nargs
from . import optim_more as om
from .optim_more import EPS, Objective, Opt, OutputHook, Report, outs, output_struct


def _extra_after(args: list, k: int) -> list:
    return list(args[k + 1:]) if len(args) > k + 1 else []


def _scalar_fun(interp, fun, extra, what, opts: Opt):
    check = opts.flag("FunValCheck")
    count = [0]

    def f(x):
        res = interp.call_function_value(fun, [np.array([[float(x)]]), *extra], 1)
        count[0] += 1
        if not res:
            raise MatlabError(f"{what}: the function must return a value.")
        a = fnum(res[0])
        if a.size != 1:
            raise MatlabError(f"{what}: the function must return a scalar.")
        v = a.flat[0]
        if check and (np.imag(v) != 0 or not np.isfinite(v)):
            raise MatlabError(f"{what}: the function returned a complex, NaN or Inf value "
                              "(FunValCheck is 'on').")
        return float(np.real(v))

    return f, count


# -- fzero ------------------------------------------------------------------------------------------------

@builtin("fzero")
def _fzero(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "options"), "fzero")
    nargs(args, 2, None, "fzero")
    fun = args[0]
    if not om.is_fn(fun):
        raise MatlabError("fzero: the first argument must be a function handle.")
    x0 = np.real(fnum(args[1])).flatten().astype(float)
    if x0.size not in (1, 2) or not np.all(np.isfinite(x0)):
        raise MatlabError("fzero: the second argument must be a finite scalar or a finite interval [a b].")
    opts = Opt(om._options_at(args, 2), "fzero")
    extra = _extra_after(args, 2)
    f, count = _scalar_fun(interp, fun, extra, "fzero", opts)
    tol = opts.num("StepTolerance", EPS)
    rep = Report(interp, opts, "notify")
    hook = OutputHook(interp, opts, (1, 1), "fzero")
    header = " Func-count    x          f(x)             Procedure"
    search_iters = 0

    def finish(x, fx, flag, iters, msg):
        rep.finish(flag, msg)
        hook.call(x, "done", iteration=iters, funccount=count[0], fval=fx, procedure="")
        out = output_struct(intervaliterations=search_iters, iterations=iters, funcCount=count[0],
                            algorithm="bisection, interpolation", message=msg)
        return outs(nargout, (x, fx, float(flag), out))

    if x0.size == 2:
        a, b = float(x0[0]), float(x0[1])
        fa, fb = f(a), f(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            raise MatlabError("fzero: the function values at the interval endpoints must be finite and real.")
        if fa * fb > 0:
            raise MatlabError("fzero: the function values at the interval endpoints must differ in sign.")
    else:
        x = float(x0[0])
        fx = f(x)
        if fx == 0:
            return finish(x, fx, 1, 0, "Zero found at the starting point.")
        if not np.isfinite(fx):
            raise MatlabError("fzero: the function value at the starting guess must be finite and real.")
        dx = abs(x) / 50 if x != 0 else 1 / 50
        a, fa, b, fb = x, fx, x, fx
        while fa * fb > 0:
            search_iters += 1
            dx *= np.sqrt(2)
            a = x - dx
            fa = f(a)
            rep.iter_line(header, f"{count[0]:5d}   {a:12.6g} {fa:14.6g}   search")
            if not np.isfinite(fa):
                return finish(np.nan, np.nan, -3, 0,
                              "Exiting fzero: a NaN or Inf function value was met while searching for an "
                              "interval containing a sign change.")
            if fa * fb <= 0:
                break
            b = x + dx
            fb = f(b)
            rep.iter_line(header, f"{count[0]:5d}   {b:12.6g} {fb:14.6g}   search")
            if not np.isfinite(fb):
                return finish(np.nan, np.nan, -3, 0,
                              "Exiting fzero: a NaN or Inf function value was met while searching for an "
                              "interval containing a sign change.")
            if not np.isfinite(dx) or search_iters > 2000:
                return finish(np.nan, np.nan, -6, 0,
                              "Exiting fzero: no sign change was found while searching for an interval; "
                              "the function may not have a root.")
    scale0 = max(abs(fa), abs(fb))
    # Brent's method: bisection safeguarded inverse quadratic interpolation
    c, fc = a, fa
    d = e = b - a
    iters = 0
    hook.call(b, "init", iteration=0, funccount=count[0], fval=fb, procedure="initial")
    while fb != 0 and a != b:
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        m = 0.5 * (c - b)
        toler = 2.0 * tol * max(abs(b), 1.0)
        if abs(m) <= toler or fb == 0:
            break
        procedure = "bisection"
        if abs(e) < toler or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p, q = 2.0 * m * s, 1.0 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2.0 * p < 3.0 * m * q - abs(toler * q) and p < abs(0.5 * e * q):
                e, d = d, p / q
                procedure = "interpolation"
            else:
                d = e = m
        a, fa = b, fb
        b = b + d if abs(d) > toler else b + (toler if m > 0 else -toler)
        fb = f(b)
        iters += 1
        rep.iter_line(header, f"{count[0]:5d}   {b:12.6g} {fb:14.6g}   {procedure}")
        if hook.call(b, "iter", iteration=iters, funccount=count[0], fval=fb, procedure=procedure):
            return finish(b, fb, -1, iters, "Stopped by the output function.")
        if iters > 10000:
            break
    if not np.isfinite(fb) or abs(fb) > scale0:
        return finish(b, fb, -5, iters, "The interval shrank onto a point where the function is not "
                                        "continuous; this is probably a singularity, not a root.")
    lo, hi = min(a, b, c), max(a, b, c)
    return finish(b, fb, 1, iters, f"Zero found in the interval [{lo:g}, {hi:g}].")


# -- fminbnd ------------------------------------------------------------------------------------------------

@builtin("fminbnd")
def _fminbnd(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x1", "x2", "options"), "fminbnd")
    nargs(args, 3, None, "fminbnd")
    fun = args[0]
    if not om.is_fn(fun):
        raise MatlabError("fminbnd: the first argument must be a function handle.")
    a, b = float(np.real(fnum(args[1]).flat[0])), float(np.real(fnum(args[2]).flat[0]))
    opts = Opt(om._options_at(args, 3), "fminbnd")
    f, count = _scalar_fun(interp, fun, _extra_after(args, 3), "fminbnd", opts)
    tol = opts.num("StepTolerance", 1e-4)
    maxit = opts.num("MaxIterations", 500)
    maxfun = opts.num("MaxFunctionEvaluations", 500)
    rep = Report(interp, opts, "notify")
    hook = OutputHook(interp, opts, (1, 1), "fminbnd")
    alg = "golden section search, parabolic interpolation"
    if a > b:
        msg = "Exiting: the bounds are inconsistent because x1 > x2."
        rep.finish(-2, msg)
        return outs(nargout, (empty_double(), empty_double(), -2.0,
                              output_struct(iterations=0, funcCount=0, algorithm=alg, message=msg)))
    header = " Func-count     x          f(x)         Procedure"
    golden = 0.5 * (3.0 - np.sqrt(5.0))
    seps = np.sqrt(EPS)
    v = w = x = a + golden * (b - a)
    fv = fw = fx = f(x)
    d = e = 0.0
    iters = 0
    flag = 1
    hook.call(x, "init", iteration=0, funccount=1, fval=fx, procedure="initial")
    rep.iter_line(header, f"{count[0]:5d}   {x:12.6g} {fx:12.6g}        initial")
    while True:
        xm = 0.5 * (a + b)
        tol1 = seps * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        if iters >= maxit or count[0] >= maxfun:
            flag = 0
            break
        use_golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            r, e = e, d
            if abs(p) < abs(0.5 * q * r) and q * (a - x) < p < q * (b - x):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if xm >= x else -tol1
                use_golden = False
        if use_golden:
            e = (a - x) if x >= xm else (b - x)
            d = golden * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d >= 0 else -tol1))
        fu = f(u)
        iters += 1
        proc = "golden" if use_golden else "parabolic"
        rep.iter_line(header, f"{count[0]:5d}   {u:12.6g} {fu:12.6g}        {proc}")
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
        if hook.call(x, "iter", iteration=iters, funccount=count[0], fval=fx, procedure=proc):
            flag = -1
            break
    msg = {1: f"Optimization terminated: the current x satisfies the termination criteria using "
              f"OPTIONS.TolX of {tol:e}.",
           0: "Exiting: the maximum number of iterations or function evaluations has been exceeded; "
              "increase MaxIter or MaxFunEvals.",
           -1: "Stopped by the output function."}[flag]
    rep.finish(flag, msg)
    hook.call(x, "done", iteration=iters, funccount=count[0], fval=fx, procedure="")
    return outs(nargout, (x, fx, float(flag),
                          output_struct(iterations=iters, funcCount=count[0], algorithm=alg, message=msg)))


# -- fminsearch -----------------------------------------------------------------------------------------------

@builtin("fminsearch")
def _fminsearch(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "options"), "fminsearch")
    nargs(args, 2, None, "fminsearch")
    x0a = np.real(fnum(args[1])).astype(float)
    shape = x0a.shape
    n = x0a.size
    if n == 0:
        raise MatlabError("fminsearch: the starting point must not be empty.")
    opts = Opt(om._options_at(args, 2), "fminsearch")
    obj = Objective(interp, args[0], shape, opts, "fminsearch", _extra_after(args, 2))
    tolx = opts.num("StepTolerance", 1e-4)
    tolf = opts.num("FunctionTolerance", 1e-4)
    maxit = opts.num("MaxIterations", 200 * n)
    maxfun = opts.num("MaxFunctionEvaluations", 200 * n)
    rep = Report(interp, opts, "notify")
    hook = OutputHook(interp, opts, shape, "fminsearch")
    header = " Iteration   Func-count         f(x)         Procedure"
    x0 = x0a.flatten(order="F")
    V = np.empty((n + 1, n))
    V[0] = x0
    for j in range(n):
        y = x0.copy()
        y[j] = y[j] * 1.05 if y[j] != 0 else 0.00025
        V[j + 1] = y
    fv = np.array([obj(v) for v in V])
    order = np.argsort(fv, kind="stable")
    V, fv = V[order], fv[order]
    iters = 0
    flag = 1
    hook.call(V[0], "init", iteration=0, funccount=obj.count, fval=fv[0], procedure="")
    rep.iter_line(header, f"{0:6d} {obj.count:11d} {fv[0]:16.6g}         initial simplex")
    while True:
        if np.max(np.abs(fv[1:] - fv[0])) <= max(tolf, 10 * np.spacing(abs(fv[0]))) and \
                np.max(np.abs(V[1:] - V[0])) <= max(tolx, 10 * np.spacing(np.max(np.abs(V[0])))):
            break
        if iters >= maxit or obj.count >= maxfun:
            flag = 0
            break
        centroid = V[:-1].mean(axis=0)
        xr = 2 * centroid - V[-1]
        fr = obj(xr)
        proc = "reflect"
        if fr < fv[0]:
            xe = 3 * centroid - 2 * V[-1]
            fe = obj(xe)
            if fe < fr:
                V[-1], fv[-1], proc = xe, fe, "expand"
            else:
                V[-1], fv[-1] = xr, fr
        elif fr < fv[-2]:
            V[-1], fv[-1] = xr, fr
        else:
            shrink = False
            if fr < fv[-1]:
                xc = 1.5 * centroid - 0.5 * V[-1]
                fc = obj(xc)
                if fc <= fr:
                    V[-1], fv[-1], proc = xc, fc, "contract outside"
                else:
                    shrink = True
            else:
                xc = 0.5 * centroid + 0.5 * V[-1]
                fc = obj(xc)
                if fc < fv[-1]:
                    V[-1], fv[-1], proc = xc, fc, "contract inside"
                else:
                    shrink = True
            if shrink:
                for j in range(1, n + 1):
                    V[j] = V[0] + 0.5 * (V[j] - V[0])
                    fv[j] = obj(V[j])
                proc = "shrink"
        order = np.argsort(fv, kind="stable")
        V, fv = V[order], fv[order]
        iters += 1
        rep.iter_line(header, f"{iters:6d} {obj.count:11d} {fv[0]:16.6g}         {proc}")
        if hook.call(V[0], "iter", iteration=iters, funccount=obj.count, fval=fv[0], procedure=proc):
            flag = -1
            break
    msg = {1: f"Optimization terminated: the current x satisfies the termination criteria using "
              f"OPTIONS.TolX of {tolx:e} and F(X) satisfies the convergence criteria using OPTIONS.TolFun "
              f"of {tolf:e}.",
           0: "Exiting: the maximum number of iterations or function evaluations has been exceeded; "
              f"increase MaxIter or MaxFunEvals. Current function value: {fv[0]:g}.",
           -1: "Stopped by the output function."}[flag]
    rep.finish(flag, msg)
    hook.call(V[0], "done", iteration=iters, funccount=obj.count, fval=fv[0], procedure="")
    out = output_struct(iterations=iters, funcCount=obj.count, algorithm="Nelder-Mead simplex direct search",
                        message=msg)
    return outs(nargout, (V[0].reshape(shape, order="F"), float(fv[0]), float(flag), out))


# -- nonlinear programming core ---------------------------------------------------------------------------------

class NLP:
    """min f(x) s.t. A x <= b, Aeq x = beq, lo <= x <= hi, c(x) <= 0, ceq(x) = 0."""

    def __init__(self, n, f, g, A=None, b=None, Aeq=None, beq=None, lo=None, hi=None, nl=None, nl_jac=None):
        self.n, self.f, self.g = n, f, g
        self.A = np.zeros((0, n)) if A is None else A
        self.b = np.zeros(0) if b is None else b
        self.Aeq = np.zeros((0, n)) if Aeq is None else Aeq
        self.beq = np.zeros(0) if beq is None else beq
        self.lo = np.full(n, -np.inf) if lo is None else lo
        self.hi = np.full(n, np.inf) if hi is None else hi
        self.nl = nl
        self._nl_jac = nl_jac
        self._cache = (None, None)

    def cons(self, x):
        if self.nl is None:
            return np.zeros(0), np.zeros(0)
        key = x.tobytes()
        if self._cache[0] != key:
            self._cache = (key, self.nl(x))
        return self._cache[1]

    def cons_jac(self, x):
        if self.nl is None:
            return np.zeros((0, self.n)), np.zeros((0, self.n))
        if self._nl_jac is not None:
            return self._nl_jac(x)
        c0, e0 = self.cons(x)
        h = np.sqrt(EPS) * np.maximum(np.abs(x), 1.0)
        Jc, Je = np.zeros((c0.size, self.n)), np.zeros((e0.size, self.n))
        for j in range(self.n):
            hj = h[j] if x[j] + h[j] <= self.hi[j] else -h[j]
            xp = x.copy()
            xp[j] += hj
            c1, e1 = self.nl(xp)
            Jc[:, j] = (c1 - c0) / hj
            Je[:, j] = (e1 - e0) / hj
        return Jc, Je

    def violation(self, x) -> float:
        c, ceq = self.cons(x)
        parts = [0.0]
        if self.A.shape[0]:
            parts.append(float(np.max(self.A @ x - self.b)))
        if self.Aeq.shape[0]:
            parts.append(float(np.max(np.abs(self.Aeq @ x - self.beq))))
        if c.size:
            parts.append(float(np.max(c)))
        if ceq.size:
            parts.append(float(np.max(np.abs(ceq))))
        parts.append(float(np.max(self.lo - x, initial=0.0)))
        parts.append(float(np.max(x - self.hi, initial=0.0)))
        return max(parts)

    def multipliers(self, x, g, act_tol=1e-6):
        """KKT multipliers at x: dict of vectors and the first-order optimality measure."""
        n = self.n
        c, ceq = self.cons(x)
        Jc, Je = self.cons_jac(x)
        normals, is_eq, where = [], [], []
        for i in range(n):
            if np.isfinite(self.lo[i]) and x[i] - self.lo[i] <= act_tol * (1 + abs(self.lo[i])):
                if self.lo[i] == self.hi[i]:
                    continue
                e = np.zeros(n)
                e[i] = -1.0
                normals.append(e)
                is_eq.append(False)
                where.append(("lower", i))
        for i in range(n):
            if np.isfinite(self.hi[i]) and self.hi[i] - x[i] <= act_tol * (1 + abs(self.hi[i])):
                e = np.zeros(n)
                e[i] = 1.0
                normals.append(e)
                is_eq.append(self.lo[i] == self.hi[i])
                where.append(("upper", i))
        for i in range(self.A.shape[0]):
            if self.b[i] - self.A[i] @ x <= act_tol * (1 + abs(self.b[i])):
                normals.append(self.A[i])
                is_eq.append(False)
                where.append(("ineqlin", i))
        for i in range(self.Aeq.shape[0]):
            normals.append(self.Aeq[i])
            is_eq.append(True)
            where.append(("eqlin", i))
        for i in range(c.size):
            if c[i] >= -act_tol * (1 + abs(c[i])):
                normals.append(Jc[i])
                is_eq.append(False)
                where.append(("ineqnonlin", i))
        for i in range(ceq.size):
            normals.append(Je[i])
            is_eq.append(True)
            where.append(("eqnonlin", i))
        lam, resid = om.kkt_multipliers(g, normals, is_eq)
        out = {"lower": np.zeros(n), "upper": np.zeros(n), "ineqlin": np.zeros(self.A.shape[0]),
               "eqlin": np.zeros(self.Aeq.shape[0]), "ineqnonlin": np.zeros(c.size),
               "eqnonlin": np.zeros(ceq.size)}
        for (name, i), v in zip(where, lam):
            out[name][i] = v
        return out, resid


def lambda_struct(lam: dict, names=("eqlin", "eqnonlin", "ineqlin", "lower", "upper", "ineqnonlin")) -> Struct:
    return Struct.scalar({k: lam[k].reshape(-1, 1) if lam[k].size else np.zeros((0, 1)) for k in names})


def solve_nlp(P: NLP, x0, algorithm="interior-point", tol_opt=1e-6, tol_con=1e-6, tol_x=1e-10, maxiter=1000,
              callback=None):
    """Returns dict(x, f, g, flag, iterations, viol, firstorder, lam, message, stopped)."""
    from scipy.optimize import BFGS, Bounds, LinearConstraint, NonlinearConstraint, minimize
    x0 = np.clip(np.asarray(x0, dtype=float), P.lo, P.hi)
    has_bounds = bool(np.any(np.isfinite(P.lo)) or np.any(np.isfinite(P.hi)))
    c0, e0 = P.cons(x0)
    iters = [0]
    stopped = [False]

    class _Stop(Exception):
        pass

    last = [x0]

    def cb(xk, *rest):
        iters[0] += 1
        last[0] = np.array(xk, copy=True)
        if callback is not None and callback(last[0], iters[0]):
            stopped[0] = True
            raise _Stop()

    def run_slsqp(xs, budget):
        cons = []
        if P.A.shape[0]:
            cons.append({"type": "ineq", "fun": lambda x: P.b - P.A @ x, "jac": lambda x: -P.A})
        if P.Aeq.shape[0]:
            cons.append({"type": "eq", "fun": lambda x: P.Aeq @ x - P.beq, "jac": lambda x: P.Aeq})
        if c0.size:
            cons.append({"type": "ineq", "fun": lambda x: -P.cons(x)[0], "jac": lambda x: -P.cons_jac(x)[0]})
        if e0.size:
            cons.append({"type": "eq", "fun": lambda x: P.cons(x)[1], "jac": lambda x: P.cons_jac(x)[1]})
        bnds = None
        if has_bounds:
            bnds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b)
                    for a, b in zip(P.lo, P.hi)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(P.f, xs, jac=P.g, method="SLSQP", bounds=bnds, constraints=cons, callback=cb,
                         options={"maxiter": int(max(budget, 1)), "ftol": min(1e-10, tol_opt * 1e-4)})
        return r, r.status == 9

    def run_ip(xs, budget):
        cons = []
        if P.A.shape[0]:
            cons.append(LinearConstraint(P.A, -np.inf, P.b))
        if P.Aeq.shape[0]:
            cons.append(LinearConstraint(P.Aeq, P.beq, P.beq))
        if c0.size:
            cons.append(NonlinearConstraint(lambda x: P.cons(x)[0], -np.inf, 0.0, jac=lambda x: P.cons_jac(x)[0]))
        if e0.size:
            cons.append(NonlinearConstraint(lambda x: P.cons(x)[1], 0.0, 0.0, jac=lambda x: P.cons_jac(x)[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = minimize(P.f, xs, jac=P.g, hess=BFGS(), method="trust-constr",
                         bounds=Bounds(P.lo, P.hi, keep_feasible=True) if has_bounds else None,
                         constraints=cons, callback=cb,
                         options={"maxiter": int(max(budget, 1)), "gtol": tol_opt, "xtol": max(tol_x, 1e-12),
                                  "barrier_tol": 1e-8, "verbose": 0})
        return r, r.status == 0

    def assess(x):
        f = float(P.f(x))
        g = np.asarray(P.g(x), dtype=float)
        viol = P.violation(x)
        lam, resid = P.multipliers(x, g, act_tol=max(1e-6, 10 * tol_con))
        return {"x": x, "f": f, "g": g, "viol": viol, "lam": lam, "firstorder": resid}

    g0 = np.asarray(P.g(x0), dtype=float)
    scale = max(1.0, float(np.max(np.abs(g0), initial=0.0)))
    order = [run_ip, run_slsqp] if algorithm == "interior-point" else [run_slsqp, run_ip]
    best = None
    hit_max = False
    xs = x0
    for run in order:
        budget = maxiter - iters[0]
        if budget <= 0:
            hit_max = True
            break
        try:
            res, status_max = run(xs, budget)
            xk = np.clip(np.asarray(res.x, dtype=float), P.lo, P.hi)
        except _Stop:
            xk, status_max = last[0], False
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(xk)):
            continue
        cur = assess(xk)
        hit_max = status_max
        if best is None:
            best = cur
        else:
            feas_c, feas_b = cur["viol"] <= tol_con, best["viol"] <= tol_con
            if (feas_c and not feas_b) or (feas_c == feas_b and (
                    (feas_c and cur["f"] < best["f"] - 1e-12 * (1 + abs(best["f"])))
                    or (feas_c and abs(cur["f"] - best["f"]) <= 1e-12 * (1 + abs(best["f"]))
                        and cur["firstorder"] < best["firstorder"])
                    or (not feas_c and cur["viol"] < best["viol"]))):
                best = cur
        if stopped[0]:
            break
        if best["viol"] <= tol_con and best["firstorder"] <= tol_opt * scale:
            break
        xs = best["x"]
    if best is None:
        best = assess(x0)
    feasible = best["viol"] <= tol_con
    if stopped[0]:
        flag, msg = -1, "Stopped by the output function."
    elif feasible and best["firstorder"] <= max(tol_opt * scale, 1e-9):
        flag, msg = 1, "Local minimum found that satisfies the constraints."
    elif hit_max and iters[0] >= maxiter:
        flag, msg = 0, "Solver stopped prematurely: the iteration limit was reached."
    elif feasible:
        flag, msg = 2, "Local minimum possible. Constraints satisfied."
    else:
        flag, msg = -2, "Converged to an infeasible point: no feasible point was found."
    best.update({"flag": flag, "message": msg, "iterations": iters[0], "stopped": stopped[0]})
    return best


def fd_hessian(grad, x):
    """Symmetric finite-difference Hessian from a gradient function."""
    n = x.size
    H = np.zeros((n, n))
    g0 = np.asarray(grad(x), dtype=float)
    h = EPS ** (1 / 3) * np.maximum(np.abs(x), 1.0)
    for j in range(n):
        xp = x.copy()
        xp[j] += h[j]
        H[:, j] = (np.asarray(grad(xp), dtype=float) - g0) / h[j]
    return 0.5 * (H + H.T)


def _nonlcon(interp, nonlcon, shape, n, opts: Opt, what: str, extra=()):
    """Wrap a MATLAB nonlinear constraint function -> (nl, nl_jac or None)."""
    if nonlcon is None or om._is_empty(nonlcon):
        return None, None
    if not om.is_fn(nonlcon):
        raise MatlabError(f"{what}: the nonlinear constraint must be a function handle.")
    user_jac = opts.flag("SpecifyConstraintGradient")
    cache: dict = {}

    def call(x, nout):
        key = (x.tobytes(), nout)
        if key not in cache:
            cache.clear()
            res = interp.call_function_value(nonlcon, [x.reshape(shape, order="F"), *extra], nout)
            if len(res) < 2:
                raise MatlabError(f"{what}: the nonlinear constraint function must return [c, ceq].")
            cache[key] = res
        return cache[key]

    def nl(x):
        res = call(x, 4 if user_jac else 2)
        c = np.real(fnum(res[0])).astype(float).flatten(order="F")
        ceq = np.real(fnum(res[1])).astype(float).flatten(order="F")
        return c, ceq

    nl_jac = None
    if user_jac:
        def nl_jac(x):
            res = call(x, 4)
            if len(res) < 4:
                raise MatlabError(f"{what}: with SpecifyConstraintGradient the constraint function must "
                                  "return [c, ceq, gradc, gradceq].")
            c, ceq = nl(x)
            Gc = np.real(fnum(res[2])).astype(float)
            Ge = np.real(fnum(res[3])).astype(float)
            Jc = Gc.T if Gc.size else np.zeros((0, n))
            Je = Ge.T if Ge.size else np.zeros((0, n))
            if Jc.shape != (c.size, n) or Je.shape != (ceq.size, n):
                raise MatlabError(f"{what}: the constraint gradients must have one column per constraint "
                                  f"and {n} rows.")
            return Jc, Je
    return nl, nl_jac


_FMINCON_IMPL = ("Algorithm", "Display", "MaxIterations", "MaxFunctionEvaluations", "OptimalityTolerance",
                 "StepTolerance", "FunctionTolerance", "ConstraintTolerance", "SpecifyObjectiveGradient",
                 "SpecifyConstraintGradient", "FiniteDifferenceType", "FiniteDifferenceStepSize", "TypicalX",
                 "CheckGradients", "OutputFcn", "PlotFcn", "FunValCheck", "ObjectiveLimit", "UseParallel",
                 "HonorBounds", "ScaleProblem", "HessianApproximation", "AbsoluteMaxObjectiveCount",
                 "EqualityGoalCount")


def _nlp_setup(interp, args, what: str, vector=False):
    """Parse (fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options) for fmincon-like solvers."""
    x0a = np.real(fnum(args[1])).astype(float)
    if x0a.size == 0:
        raise MatlabError(f"{what}: the starting point x0 must not be empty.")
    shape = x0a.shape
    n = x0a.size
    opts = Opt(om._options_at(args, 9), what, _FMINCON_IMPL)
    obj = Objective(interp, args[0], shape, opts, what, vector=vector)
    A, b = om.linear(om._arg(args, 2), om._arg(args, 3), n, what, "A")
    Aeq, beq = om.linear(om._arg(args, 4), om._arg(args, 5), n, what, "Aeq")
    lo, hi = om.bounds(om._arg(args, 6), om._arg(args, 7), n, what)
    obj.lo, obj.hi = lo, hi
    nonl = args[8] if len(args) > 8 else None
    nl, nl_jac = _nonlcon(interp, nonl, shape, n, opts, what)
    return x0a, shape, n, opts, obj, A, b, Aeq, beq, lo, hi, nl, nl_jac


@builtin("fmincon")
def _fmincon(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon",
                                  "options"), "fmincon")
    nargs(args, 2, 10, "fmincon")
    x0a, shape, n, opts, obj, A, b, Aeq, beq, lo, hi, nl, nl_jac = _nlp_setup(interp, args, "fmincon")
    alg = opts.word("Algorithm", "interior-point", om.SOLVER_ALGORITHMS["fmincon"])
    if alg == "trust-region-reflective" and not opts.flag("SpecifyObjectiveGradient"):
        interp.warn("fmincon: the trust-region-reflective algorithm needs the objective gradient "
                    "(SpecifyObjectiveGradient); using interior-point instead.")
        alg = "interior-point"
    tol_opt = opts.num("OptimalityTolerance", 1e-6)
    tol_con = opts.num("ConstraintTolerance", 1e-6)
    tol_x = opts.num("StepTolerance", 1e-10 if alg == "interior-point" else 1e-6)
    maxiter = opts.num("MaxIterations", 1000 if alg == "interior-point" else 400)
    maxfun = opts.num("MaxFunctionEvaluations", 3000 if alg == "interior-point" else 100 * n)
    limit = opts.num("ObjectiveLimit", -1e20)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, "fmincon")
    x0 = x0a.flatten(order="F")
    if opts.flag("CheckGradients"):
        obj.check_gradient(np.clip(x0, lo, hi))
    P = NLP(n, obj, obj.grad, A, b, Aeq, beq, lo, hi, nl, nl_jac)
    header = " Iter  Func-count            Fval   Feasibility"
    state = {"limit": False, "maxfun": False}

    def callback(xk, k):
        fk = obj(xk)
        rep.iter_line(header, f"{k:5d} {obj.count:11d} {fk:15.6e} {P.violation(xk):13.3e}")
        if fk < limit:
            state["limit"] = True
            return True
        if obj.count >= maxfun:
            state["maxfun"] = True
            return True
        return hook.call(xk, "iter", iteration=k, funccount=obj.count, fval=fk,
                         constrviolation=P.violation(xk))

    hook.call(x0, "init", iteration=0, funccount=0, fval=obj(np.clip(x0, lo, hi)))
    res = solve_nlp(P, x0, "interior-point" if alg in ("interior-point", "trust-region-reflective") else "sqp",
                    tol_opt, tol_con, tol_x, maxiter, callback)
    flag, msg = res["flag"], res["message"]
    if state["limit"]:
        flag, msg = -3, "The objective function went below ObjectiveLimit; the problem may be unbounded."
    elif state["maxfun"] and flag != 1:
        flag, msg = 0, "Solver stopped prematurely: the function evaluation limit was reached."
    x = res["x"]
    hook.call(x, "done", iteration=res["iterations"], funccount=obj.count, fval=res["f"])
    rep.finish(flag, msg)
    out = output_struct(iterations=res["iterations"], funcCount=obj.count, constrviolation=res["viol"],
                        stepsize=0.0, algorithm=alg, firstorderopt=res["firstorder"], cgiterations=0,
                        message=msg, bestfeasible=empty_double())
    vals = [x.reshape(shape, order="F"), res["f"], float(flag), out, lambda_struct(res["lam"]),
            res["g"].reshape(-1, 1)]
    if nargout >= 7:
        lam = res["lam"]

        def lag_grad(z):
            g = np.asarray(obj.grad(z), dtype=float).copy()
            Jc, Je = P.cons_jac(z)
            if Jc.shape[0]:
                g += Jc.T @ lam["ineqnonlin"]
            if Je.shape[0]:
                g += Je.T @ lam["eqnonlin"]
            return g

        vals.append(fd_hessian(lag_grad, x))
    return outs(nargout, tuple(vals))


# -- fminunc --------------------------------------------------------------------------------------------------

@builtin("fminunc")
def _fminunc(interp, args, nargout):
    from scipy.optimize import minimize
    args = om.problem_args(args, ("objective", "x0", "options"), "fminunc")
    nargs(args, 2, None, "fminunc")
    x0a = np.real(fnum(args[1])).astype(float)
    shape, n = x0a.shape, x0a.size
    if n == 0:
        raise MatlabError("fminunc: the starting point x0 must not be empty.")
    opts = Opt(om._options_at(args, 2), "fminunc",
               ("Algorithm", "Display", "MaxIterations", "MaxFunctionEvaluations", "OptimalityTolerance",
                "StepTolerance", "FunctionTolerance", "SpecifyObjectiveGradient", "FiniteDifferenceType",
                "FiniteDifferenceStepSize", "TypicalX", "CheckGradients", "OutputFcn", "PlotFcn",
                "FunValCheck", "ObjectiveLimit", "UseParallel", "HessianApproximation", "HessianFcn"))
    obj = Objective(interp, args[0], shape, opts, "fminunc", _extra_after(args, 2))
    alg = opts.word("Algorithm", "quasi-newton", om.SOLVER_ALGORITHMS["fminunc"])
    if alg == "trust-region" and not obj.user_grad:
        interp.warn("fminunc: the trust-region algorithm needs the gradient (SpecifyObjectiveGradient); "
                    "using quasi-newton instead.")
        alg = "quasi-newton"
    hess_user = None
    if opts.has("HessianFcn"):
        hv = opts.get("HessianFcn")
        if om.is_fn(hv) and not isinstance(hv, om.FunctionHandle) and om.text(hv) == "objective":
            hess_user = "objective"
        elif not om._is_empty(hv):
            raise MatlabError("fminunc: HessianFcn must be 'objective' (the Hessian is the third output of "
                              "the objective function) or [].")
    tol_opt = opts.num("OptimalityTolerance", 1e-6)
    tol_x = opts.num("StepTolerance", 1e-6)
    maxiter = int(opts.num("MaxIterations", 400))
    maxfun = opts.num("MaxFunctionEvaluations", 100 * n)
    limit = opts.num("ObjectiveLimit", -1e20)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, "fminunc")
    x0 = x0a.flatten(order="F")
    if opts.flag("CheckGradients"):
        obj.check_gradient(x0)
    f0 = obj(x0)
    g0 = np.asarray(obj.grad(x0), dtype=float)
    scale = 1.0 + float(np.max(np.abs(g0), initial=0.0))
    header = " Iteration  Func-count       f(x)        First-order optimality"
    it = [0]
    state = {"stop": None}
    last = [x0, x0]

    class _Stop(Exception):
        pass

    def cb(xk, *rest):
        it[0] += 1
        last[1], last[0] = last[0], np.array(xk, copy=True)
        fk = obj(xk)
        if rep.level == "iter":
            rep.iter_line(header, f"{it[0]:6d} {obj.count:11d} {fk:14.6g} {np.max(np.abs(obj.grad(xk))):14.3g}")
        if fk < limit:
            state["stop"] = -3
        elif obj.count >= maxfun:
            state["stop"] = 0
        elif hook.call(xk, "iter", iteration=it[0], funccount=obj.count, fval=fk):
            state["stop"] = -1
        if state["stop"] is not None:
            raise _Stop()

    hook.call(x0, "init", iteration=0, funccount=obj.count, fval=f0)

    def hess(x):
        if hess_user == "objective":
            res = interp.call_function_value(obj.fun, [x.reshape(shape, order="F"), *obj.extra], 3)
            if len(res) < 3:
                raise MatlabError("fminunc: with HessianFcn 'objective' the objective must return the "
                                  "Hessian as its third output.")
            return np.real(fnum(res[2])).astype(float)
        return fd_hessian(obj.grad, x)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if alg == "quasi-newton":
                res = minimize(obj, x0, jac=obj.grad, method="BFGS", callback=cb,
                               options={"gtol": tol_opt * 1e-2, "maxiter": maxiter})
            else:
                res = minimize(obj, x0, jac=obj.grad, hess=hess, method="trust-exact" if n <= 200 else "trust-ncg",
                               callback=cb, options={"gtol": tol_opt * 1e-2, "maxiter": maxiter})
        x = np.asarray(res.x, dtype=float)
        maxed = res.status == 1
    except _Stop:
        x, maxed = last[0], False
    f = obj(x)
    g = np.asarray(obj.grad(x), dtype=float)
    first = float(np.max(np.abs(g), initial=0.0))
    step = float(np.linalg.norm(last[0] - last[1])) if it[0] else 0.0
    if state["stop"] == -1:
        flag, msg = -1, "Stopped by the output function."
    elif state["stop"] == -3:
        flag, msg = -3, "The objective function went below ObjectiveLimit; the problem may be unbounded."
    elif first <= tol_opt * scale:
        flag, msg = 1, "Local minimum found."
    elif state["stop"] == 0 or maxed:
        flag, msg = 0, "Solver stopped prematurely: the iteration or function evaluation limit was reached."
    elif step <= tol_x * (1 + float(np.linalg.norm(x))):
        flag, msg = 2, "Local minimum possible: the step size is below the step tolerance."
    else:
        flag, msg = 5, "Local minimum possible: the solver cannot decrease the objective function further."
    hook.call(x, "done", iteration=it[0], funccount=obj.count, fval=f)
    rep.finish(flag, msg)
    out = output_struct(iterations=it[0], funcCount=obj.count, stepsize=step, lssteplength=1.0,
                        firstorderopt=first, algorithm=alg, message=msg)
    vals = [x.reshape(shape, order="F"), f, float(flag), out, g.reshape(-1, 1)]
    if nargout >= 6:
        vals.append(hess(x))
    return outs(nargout, tuple(vals))


# -- nonlinear equations and least squares ---------------------------------------------------------------------

_LSQ_IMPL = ("Algorithm", "Display", "MaxIterations", "MaxFunctionEvaluations", "OptimalityTolerance",
             "StepTolerance", "FunctionTolerance", "ConstraintTolerance", "SpecifyObjectiveGradient",
             "SpecifyConstraintGradient", "FiniteDifferenceType", "FiniteDifferenceStepSize", "TypicalX",
             "CheckGradients", "OutputFcn", "PlotFcn", "FunValCheck", "UseParallel")


def _least_squares(interp, obj: Objective, x0, lo, hi, opts: Opt, what: str, rep: Report, hook: OutputHook,
                   default_alg="trust-region-reflective"):
    """Bound-constrained nonlinear least squares. Returns (x, r, J, flag, message, iterations)."""
    from scipy.optimize import least_squares
    n = x0.size
    alg = opts.word("Algorithm", default_alg, om.SOLVER_ALGORITHMS.get(what, None))
    tol_f = opts.num("FunctionTolerance", 1e-6)
    tol_x = opts.num("StepTolerance", 1e-6)
    tol_g = opts.num("OptimalityTolerance", 1e-6)
    maxiter = opts.num("MaxIterations", 400)
    maxfun = opts.num("MaxFunctionEvaluations", 100 * n)
    bounded = bool(np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))
    r0 = obj(x0)
    m = r0.size
    method = "trf"
    if alg == "levenberg-marquardt" and not bounded and m >= n:
        method = "lm"
    elif alg in ("trust-region-reflective", "trust-region") and m < n:
        method = "trf"   # scipy's trf handles underdetermined systems; MATLAB switches algorithms here
    header = " Iteration  Func-count      Resnorm      First-order optimality"
    it = [0]
    state = {"stop": None}
    last = [x0]

    class _Stop(Exception):
        pass

    def fun(x):
        if obj.count >= maxfun + 1:
            state["stop"] = 0
            raise _Stop()
        return obj(x)

    def cb(intermediate_result=None, **kw):
        it[0] += 1
        xk = np.asarray(getattr(intermediate_result, "x", last[0]), dtype=float)
        last[0] = xk
        rk = obj(xk)
        rep.iter_line(header, f"{it[0]:6d} {obj.count:11d} {float(rk @ rk):14.6g}")
        if hook.call(xk, "iter", iteration=it[0], funccount=obj.count, resnorm=float(rk @ rk)):
            state["stop"] = -1
            raise _Stop()

    kwargs = {}
    import inspect
    if "callback" in inspect.signature(least_squares).parameters:
        kwargs["callback"] = cb
    elif hook:
        raise MatlabError(f"{what}: OutputFcn needs SciPy 1.16 or newer for this solver.")
    hook.call(x0, "init", iteration=0, funccount=obj.count, resnorm=float(r0 @ r0))
    res = None
    # The solver is run tighter than the requested tolerances: MATLAB's criteria are relative to
    # the problem scale and usually leave the answer far more accurate than the tolerance itself.
    tight = dict(xtol=max(min(tol_x, 1e-6) * 1e-4, 1e-15), ftol=max(min(tol_f, 1e-6) * 1e-4, 1e-15),
                 gtol=max(min(tol_g, 1e-6) * 1e-4, 1e-15))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = least_squares(fun, np.clip(x0, lo, hi), jac=obj.grad,
                                bounds=(lo, hi) if method != "lm" else (-np.inf, np.inf), method=method,
                                max_nfev=int(max(maxiter, 1)) if method != "lm" else None, **tight, **kwargs)
        x = np.asarray(res.x, dtype=float)
        status = res.status
        # the reflective method stops a hair inside an active bound: put the point on it
        mask = getattr(res, "active_mask", None)
        if mask is not None and np.size(mask) == n:
            x = np.where(np.asarray(mask) < 0, lo, np.where(np.asarray(mask) > 0, hi, x))
    except _Stop:
        x, status = last[0], None
    with np.errstate(invalid="ignore"):
        near_lo = np.isfinite(lo) & (x - lo <= 1e-9 * np.maximum(1.0, np.abs(np.where(np.isfinite(lo), lo, 0.0))))
        near_hi = np.isfinite(hi) & (hi - x <= 1e-9 * np.maximum(1.0, np.abs(np.where(np.isfinite(hi), hi, 0.0))))
    x = np.where(near_lo, lo, np.where(near_hi, hi, x))
    r = obj(x)
    J = np.atleast_2d(obj.grad(x))
    grad = J.T @ r
    # projected gradient measure
    pg = grad.copy()
    pg[near_lo & (grad > 0)] = 0.0
    pg[near_hi & (grad < 0)] = 0.0
    first = float(np.max(np.abs(pg), initial=0.0))
    if state["stop"] == -1:
        flag, msg = -1, "Stopped by the output function."
    elif state["stop"] == 0 or status == 0:
        flag, msg = 0, "Solver stopped prematurely: the iteration or function evaluation limit was reached."
    elif status == -1:
        flag, msg = -2, "The solver failed to make progress."
    elif status == 1 or first <= tol_g * (1 + float(np.max(np.abs(J.T @ r0), initial=0.0))):
        flag, msg = 1, "Local minimum found."
    elif status == 3:
        flag, msg = 2, "Local minimum possible: the step size is below the step tolerance."
    else:
        flag, msg = 3, "Local minimum possible: the change in the residual is below the function tolerance."
    return x, r, J, flag, msg, it[0] if it[0] else int(getattr(res, "nfev", 0) or 0), first


def _bound_lambda(x, lo, hi, grad) -> Struct:
    flo, fhi = np.isfinite(lo), np.isfinite(hi)
    lo_f, hi_f = np.where(flo, lo, 0.0), np.where(fhi, hi, 0.0)
    lower = np.where(flo & (x <= lo_f + 1e-10 * (1 + np.abs(lo_f))) & (grad > 0), grad, 0.0)
    upper = np.where(fhi & (x >= hi_f - 1e-10 * (1 + np.abs(hi_f))) & (grad < 0), -grad, 0.0)
    return Struct.scalar({"lower": lower.reshape(-1, 1), "upper": upper.reshape(-1, 1)})


def _lsq_common(interp, what, fun_args, x0a, rest, nargout, wrap=None, ydata=None):
    """rest = arguments after the data: lb, ub, [A, b, Aeq, beq, nonlcon,] options."""
    shape, n = x0a.shape, x0a.size
    opt_pos = next((k for k, a in enumerate(rest) if isinstance(a, (Struct, om.OptimOptions))), None)
    opts = Opt(rest[opt_pos] if opt_pos is not None else None, what, _LSQ_IMPL)
    lo, hi = om.bounds(om._arg(rest, 0), om._arg(rest, 1), n, what)
    general = opt_pos is None or opt_pos > 2
    A, b = om.linear(om._arg(rest, 2) if general else None, om._arg(rest, 3) if general else None, n, what, "A")
    Aeq, beq = om.linear(om._arg(rest, 4) if general else None, om._arg(rest, 5) if general else None, n, what,
                         "Aeq")
    nonl = rest[6] if general and len(rest) > 6 and om.is_fn(rest[6]) else None
    obj = Objective(interp, fun_args[0], shape, opts, what, fun_args[1:], vector=True)
    if wrap is not None:
        base_convert = obj._convert

        def convert(v):
            a = np.real(fnum(v)).astype(float)
            if a.size != ydata.size:
                raise MatlabError(f"{what}: the model function must return an array of the same size as "
                                  "ydata.")
            return base_convert((a.flatten(order="F") - ydata.flatten(order="F")).reshape(ydata.shape, order="F"))

        obj._convert = convert
    obj.lo, obj.hi = lo, hi
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, what)
    x0 = x0a.flatten(order="F")
    if opts.flag("CheckGradients"):
        obj.check_gradient(np.clip(x0, lo, hi))
    constrained = A.shape[0] or Aeq.shape[0] or nonl is not None
    if constrained:
        nl, nl_jac = _nonlcon(interp, nonl, shape, n, opts, what)

        def f(x):
            r = obj(x)
            return float(r @ r)

        def g(x):
            return 2.0 * (np.atleast_2d(obj.grad(x)).T @ obj(x))

        P = NLP(n, f, g, A, b, Aeq, beq, lo, hi, nl, nl_jac)
        res = solve_nlp(P, x0, "interior-point", opts.num("OptimalityTolerance", 1e-6),
                        opts.num("ConstraintTolerance", 1e-6), opts.num("StepTolerance", 1e-6),
                        opts.num("MaxIterations", 400))
        x, flag, msg, iters, first = res["x"], res["flag"], res["message"], res["iterations"], res["firstorder"]
        if flag == 1:
            msg = "Local minimum found that satisfies the constraints."
        r = obj(x)
        J = np.atleast_2d(obj.grad(x))
        lam = lambda_struct(res["lam"])
        alg = "interior-point"
    else:
        x, r, J, flag, msg, iters, first = _least_squares(interp, obj, x0, lo, hi, opts, what, rep, hook)
        lam = _bound_lambda(x, lo, hi, J.T @ r)
        alg = opts.word("Algorithm", "trust-region-reflective")
    hook.call(x, "done", iteration=iters, funccount=obj.count, resnorm=float(r @ r))
    rep.finish(flag, msg)
    out = output_struct(firstorderopt=first, iterations=iters, funcCount=obj.count, cgiterations=0,
                        algorithm=alg, stepsize=0.0, message=msg)
    r_out = r.reshape(obj.out_shape, order="F") if obj.out_shape and int(np.prod(obj.out_shape)) == r.size \
        else r.reshape(-1, 1)
    return outs(nargout, (x.reshape(shape, order="F"), float(r @ r), r_out, float(flag), out, lam, J))


@builtin("lsqnonlin")
def _lsqnonlin(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "lb", "ub", ("Aineq", "A"), ("bineq", "b"), "Aeq", "beq",
                                  "nonlcon", "options"), "lsqnonlin")
    nargs(args, 2, 10, "lsqnonlin")
    x0a = np.real(fnum(args[1])).astype(float)
    if x0a.size == 0:
        raise MatlabError("lsqnonlin: the starting point x0 must not be empty.")
    return _lsq_common(interp, "lsqnonlin", [args[0]], x0a, list(args[2:]), nargout)


@builtin("lsqcurvefit")
def _lsqcurvefit(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "xdata", "ydata", "lb", "ub", ("Aineq", "A"),
                                  ("bineq", "b"), "Aeq", "beq", "nonlcon", "options"), "lsqcurvefit")
    nargs(args, 4, 12, "lsqcurvefit")
    x0a = np.real(fnum(args[1])).astype(float)
    if x0a.size == 0:
        raise MatlabError("lsqcurvefit: the starting point x0 must not be empty.")
    ydata = np.real(fnum(args[3])).astype(float)
    return _lsq_common(interp, "lsqcurvefit", [args[0], args[2]], x0a, list(args[4:]), nargout, wrap=True,
                       ydata=ydata)


@builtin("fsolve")
def _fsolve(interp, args, nargout):
    from scipy.optimize import root
    args = om.problem_args(args, ("objective", "x0", "options"), "fsolve")
    nargs(args, 2, None, "fsolve")
    x0a = np.real(fnum(args[1])).astype(float)
    shape, n = x0a.shape, x0a.size
    if n == 0:
        raise MatlabError("fsolve: the starting point x0 must not be empty.")
    opts = Opt(om._options_at(args, 2), "fsolve", _LSQ_IMPL)
    obj = Objective(interp, args[0], shape, opts, "fsolve", _extra_after(args, 2), vector=True)
    alg = opts.word("Algorithm", "trust-region-dogleg", om.SOLVER_ALGORITHMS["fsolve"])
    tol_f = opts.num("FunctionTolerance", 1e-6)
    tol_x = opts.num("StepTolerance", 1e-6)
    maxiter = opts.num("MaxIterations", 400)
    maxfun = opts.num("MaxFunctionEvaluations", 100 * n)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, "fsolve")
    x0 = x0a.flatten(order="F")
    if opts.flag("CheckGradients"):
        obj.check_gradient(x0)
    F0 = obj(x0)
    m = F0.size
    iters = 0
    first = 0.0
    if alg == "trust-region-dogleg" and m != n:
        interp.warn("fsolve: the trust-region-dogleg algorithm needs as many equations as unknowns; "
                    "using levenberg-marquardt instead.")
        alg = "levenberg-marquardt"
    if alg == "trust-region-dogleg" and not hook and rep.level != "iter":
        hook.call(x0, "init", iteration=0, funccount=obj.count, fval=F0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = root(obj, x0, jac=obj.grad, method="hybr",
                       options={"xtol": max(min(tol_x, 1e-6) * 1e-4, 1e-15), "maxfev": int(max(maxfun, 10) * 50)})
        x = np.asarray(res.x, dtype=float)
        F = obj(x)
        J = np.atleast_2d(obj.grad(x))
        iters = int(getattr(res, "nfev", 0))
        first = float(np.max(np.abs(J.T @ F), initial=0.0))
        if float(F @ F) <= np.sqrt(tol_f):
            flag, msg = 1, "Equation solved."
        else:
            # the hybrid method stalled: finish with Levenberg-Marquardt before giving up
            x2, F2, J2, _, _, it2, first2 = _least_squares(interp, obj, x, np.full(n, -np.inf),
                                                           np.full(n, np.inf), opts, "fsolve",
                                                           Report(interp, Opt(None, "fsolve"), "off"), hook,
                                                           "levenberg-marquardt")
            if float(F2 @ F2) < float(F @ F):
                x, F, J, first = x2, F2, J2, first2
            iters += it2
            if float(F @ F) <= np.sqrt(tol_f):
                flag, msg = 1, "Equation solved."
            elif obj.count >= maxfun or iters >= maxiter:
                flag, msg = 0, "Solver stopped prematurely: the iteration or function evaluation limit was reached."
            else:
                flag, msg = -2, "No solution found: the solver converged to a point that is not a root."
    else:
        lsq_alg = "levenberg-marquardt" if alg in ("levenberg-marquardt", "trust-region-dogleg") else \
            "trust-region-reflective"
        o2 = Opt(None, "fsolve")
        o2.user = dict(opts.user)
        o2.user["Algorithm"] = char(lsq_alg)
        x, F, J, lflag, msg, iters, first = _least_squares(interp, obj, x0, np.full(n, -np.inf), np.full(n, np.inf),
                                                           o2, "lsqnonlin", rep, hook, lsq_alg)
        if lflag == -1:
            flag = -1
        elif float(F @ F) <= np.sqrt(tol_f):
            flag, msg = 1, "Equation solved."
        elif lflag == 0:
            flag = 0
        else:
            flag, msg = -2, "No solution found: the solver converged to a point that is not a root."
    hook.call(x, "done", iteration=iters, funccount=obj.count, fval=F)
    rep.finish(flag, msg)
    out = output_struct(iterations=iters, funcCount=obj.count, algorithm=alg, firstorderopt=first, message=msg)
    F_out = F.reshape(obj.out_shape, order="F") if obj.out_shape and int(np.prod(obj.out_shape)) == F.size \
        else F.reshape(-1, 1)
    return outs(nargout, (x.reshape(shape, order="F"), F_out, float(flag), out, J))


# -- linear and integer programming -------------------------------------------------------------------------------

def _lin_problem(args, what, n_first=0):
    """(f, A, b, Aeq, beq, lo, hi) from positional arguments starting at index n_first."""
    f = om._arg(args, 0)
    if f is None:
        raise MatlabError(f"{what}: the objective coefficient vector f must not be empty.")
    f = f.flatten(order="F")
    n = f.size
    k = 1 + n_first
    A, b = om.linear(om._arg(args, k), om._arg(args, k + 1), n, what, "A")
    Aeq, beq = om.linear(om._arg(args, k + 2), om._arg(args, k + 3), n, what, "Aeq")
    lo, hi = om.bounds(om._arg(args, k + 4), om._arg(args, k + 5), n, what)
    return f, n, A, b, Aeq, beq, lo, hi


def _find_options(args, start):
    for a in args[start:]:
        if isinstance(a, (Struct, om.OptimOptions)):
            return a
    return None


def linprog_core(f, A, b, Aeq, beq, lo, hi, method="highs", maxiter=None, tol=None, time_limit=None):
    from scipy.optimize import linprog
    options = {}
    if maxiter is not None:
        options["maxiter"] = int(maxiter)
    if tol is not None:
        options["primal_feasibility_tolerance"] = tol[0]
        options["dual_feasibility_tolerance"] = tol[1]
    if time_limit is not None and np.isfinite(time_limit):
        options["time_limit"] = float(time_limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return linprog(f, A_ub=A if A.shape[0] else None, b_ub=b if A.shape[0] else None,
                       A_eq=Aeq if Aeq.shape[0] else None, b_eq=beq if Aeq.shape[0] else None,
                       bounds=list(zip(lo, hi)), method=method, options=options)


@builtin("linprog")
def _linprog(interp, args, nargout):
    args = om.problem_args(args, ("f", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "options"), "linprog")
    nargs(args, 1, 9, "linprog")
    f, n, A, b, Aeq, beq, lo, hi = _lin_problem(args, "linprog")
    opts = Opt(_find_options(args, 7), "linprog",
               ("Algorithm", "Display", "MaxIterations", "ConstraintTolerance", "OptimalityTolerance", "MaxTime",
                "Preprocess"))
    alg = opts.word("Algorithm", "dual-simplex-highs", om.SOLVER_ALGORITHMS["linprog"])
    method = "highs-ipm" if alg.startswith("interior-point") else "highs-ds"
    tol = None
    if opts.has("ConstraintTolerance") or opts.has("OptimalityTolerance"):
        tol = (min(max(opts.num("ConstraintTolerance", 1e-7), 1e-10), 1e-2),
               min(max(opts.num("OptimalityTolerance", 1e-7), 1e-10), 1e-2))
    rep = Report(interp, opts)
    res = linprog_core(f, A, b, Aeq, beq, lo, hi, method, opts.num("MaxIterations"), tol, opts.num("MaxTime"))
    flag = {0: 1, 1: 0, 2: -2, 3: -3, 4: -4}.get(res.status, -4)
    msg = {1: "Optimal solution found.", 0: "Solver stopped prematurely: the iteration limit was reached.",
           -2: "No feasible solution found.", -3: "Problem is unbounded.",
           -4: "The solver ran into numerical difficulties."}[flag]
    if res.x is None or flag < 0:
        x, fval = np.zeros((0, 0)), np.zeros((0, 0))
        lam = Struct.scalar({k: np.zeros((0, 0)) for k in ("lower", "upper", "eqlin", "ineqlin")})
        viol = first = np.zeros((0, 0))
    else:
        xv = np.asarray(res.x, dtype=float)
        x, fval = xv.reshape(-1, 1), float(f @ xv)

        def marg(name, size, sign):
            part = getattr(res, name, None)
            m = getattr(part, "marginals", None) if part is not None else None
            if m is None or np.size(m) != size:
                return np.zeros((size, 1))
            v = sign * np.asarray(m, dtype=float).reshape(-1, 1)
            return np.where(np.abs(v) < 1e-14, 0.0, v)

        lam = Struct.scalar({"lower": marg("lower", n, 1.0), "upper": marg("upper", n, -1.0),
                             "eqlin": marg("eqlin", Aeq.shape[0], -1.0),
                             "ineqlin": marg("ineqlin", A.shape[0], -1.0)})
        viol = max([0.0] + ([float(np.max(A @ xv - b))] if A.shape[0] else [])
                   + ([float(np.max(np.abs(Aeq @ xv - beq)))] if Aeq.shape[0] else [])
                   + [float(np.max(lo - xv, initial=0.0)), float(np.max(xv - hi, initial=0.0))])
        first = 0.0
    rep.finish(flag, msg)
    out = output_struct(iterations=float(getattr(res, "nit", 0) or 0), algorithm=alg, constrviolation=viol,
                        message=msg, firstorderopt=first)
    return outs(nargout, (x, fval, float(flag), out, lam))


@builtin("intlinprog")
def _intlinprog(interp, args, nargout):
    from scipy.optimize import Bounds, LinearConstraint, milp
    args = om.problem_args(args, ("f", "intcon", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options"),
                           "intlinprog")
    nargs(args, 2, 10, "intlinprog")
    f, n, A, b, Aeq, beq, lo, hi = _lin_problem(args, "intlinprog", 1)
    intcon = om._arg(args, 1)
    integrality = np.zeros(n)
    if intcon is not None:
        ic = intcon.flatten()
        if np.any(ic < 1) or np.any(ic > n) or np.any(ic != np.round(ic)):
            raise MatlabError(f"intlinprog: intcon must list variable numbers between 1 and {n}.")
        integrality[ic.astype(int) - 1] = 1
    opts = Opt(_find_options(args, 8), "intlinprog",
               ("Display", "MaxTime", "MaxNodes", "RelativeGapTolerance", "AbsoluteGapTolerance",
                "IntegerTolerance", "ConstraintTolerance", "Algorithm"))
    rep = Report(interp, opts, "final")
    cons = []
    if A.shape[0]:
        cons.append(LinearConstraint(A, -np.inf, b))
    if Aeq.shape[0]:
        cons.append(LinearConstraint(Aeq, beq, beq))
    mopts = {"disp": False}
    if opts.has("MaxTime") and np.isfinite(opts.num("MaxTime")):
        mopts["time_limit"] = opts.num("MaxTime")
    if opts.has("MaxNodes") and np.isfinite(opts.num("MaxNodes")):
        mopts["node_limit"] = int(opts.num("MaxNodes"))
    if opts.has("RelativeGapTolerance"):
        mopts["mip_rel_gap"] = opts.num("RelativeGapTolerance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = milp(f, integrality=integrality, bounds=Bounds(lo, hi), constraints=cons, options=mopts)
    flag = {0: 1, 1: 0, 2: -2, 3: -3}.get(res.status, -9)
    if res.status == 1 and res.x is not None:
        flag = 2
    msg = {1: "Optimal solution found.", 2: "Solver stopped prematurely; an integer feasible point was found.",
           0: "Solver stopped prematurely; no integer feasible point was found.",
           -2: "No feasible solution found.", -3: "Problem is unbounded.",
           -9: "The solver lost feasibility."}[flag]
    if res.x is None:
        x, fval, viol = np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    else:
        xv = np.asarray(res.x, dtype=float).copy()
        xv[integrality == 1] = np.round(xv[integrality == 1])
        x, fval = xv.reshape(-1, 1), float(f @ xv)
        viol = max([0.0] + ([float(np.max(A @ xv - b))] if A.shape[0] else [])
                   + ([float(np.max(np.abs(Aeq @ xv - beq)))] if Aeq.shape[0] else []))
    rep.finish(flag, msg)
    gap = getattr(res, "mip_gap", 0.0) or 0.0
    out = output_struct(relativegap=float(gap), absolutegap=float(gap) * (abs(fval) if np.isscalar(fval) else 0.0),
                        numfeaspoints=float(res.x is not None), numnodes=float(getattr(res, "mip_node_count", 0) or 0),
                        constrviolation=viol, algorithm="highs", message=msg)
    return outs(nargout, (x, fval, float(flag), out))


# -- quadratic programming -------------------------------------------------------------------------------------------

def solve_qp(H, f, A, b, Aeq, beq, lo, hi, x0=None, tol=1e-8, maxit=200):
    """Convex quadratic programme by a primal-dual interior-point method with an active-set
    refinement. Returns (x, lam dict, flag, iterations)."""
    n = f.size
    m_lin = A.shape[0]
    low_idx = np.nonzero(np.isfinite(lo))[0]
    up_idx = np.nonzero(np.isfinite(hi))[0]
    G = np.vstack([A, -np.eye(n)[low_idx], np.eye(n)[up_idx]])
    h = np.concatenate([b, -lo[low_idx], hi[up_idx]])
    m, p = G.shape[0], Aeq.shape[0]
    lam = {"ineqlin": np.zeros(m_lin), "eqlin": np.zeros(p), "lower": np.zeros(n), "upper": np.zeros(n)}

    def split(z, y):
        lam["ineqlin"] = z[:m_lin].copy()
        lam["lower"][low_idx] = z[m_lin:m_lin + low_idx.size]
        lam["upper"][up_idx] = z[m_lin + low_idx.size:]
        lam["eqlin"] = np.asarray(y, dtype=float).copy()

    def kkt_solve(Hb, rhs1, rhs2):
        if p == 0:
            try:
                return np.linalg.solve(Hb, rhs1), np.zeros(0)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(Hb, rhs1, rcond=None)[0], np.zeros(0)
        K = np.block([[Hb, Aeq.T], [Aeq, np.zeros((p, p))]])
        rhs = np.concatenate([rhs1, rhs2])
        try:
            sol = np.linalg.solve(K, rhs)
            if not np.all(np.isfinite(sol)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return sol[:n], sol[n:]

    hscale = max(1.0, float(np.max(np.abs(H), initial=0.0)), float(np.max(np.abs(f), initial=0.0)))
    if m == 0:
        x, y = kkt_solve(H, -f, beq)
        rd = H @ x + f + (Aeq.T @ y if p else 0.0)
        if p and np.max(np.abs(Aeq @ x - beq), initial=0.0) > 1e-7 * max(1.0, float(np.max(np.abs(beq)))):
            return x, lam, -2, 1
        if np.max(np.abs(rd), initial=0.0) > 1e-7 * hscale * max(1.0, float(np.max(np.abs(x), initial=0.0))):
            return x, lam, -3, 1
        split(np.zeros(0), y)
        return x, lam, 1, 1
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    s = np.maximum(h - G @ x, 1.0)
    z = np.ones(m)
    y = np.zeros(p)
    flag = 0
    bscale = max(1.0, float(np.max(np.abs(h), initial=0.0)), float(np.max(np.abs(beq), initial=0.0)))
    it = 0
    while it < maxit:
        it += 1
        rd = H @ x + f + G.T @ z + (Aeq.T @ y if p else 0.0)
        rp = G @ x + s - h
        re = Aeq @ x - beq if p else np.zeros(0)
        mu = float(s @ z) / m
        if max(np.max(np.abs(rd)), 0.0) <= tol * hscale * max(1.0, np.max(np.abs(x))) and \
                max(np.max(np.abs(rp)), np.max(np.abs(re), initial=0.0)) <= tol * bscale and mu <= tol * hscale:
            flag = 1
            break
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e15:
            flag = -3
            break
        D = z / s
        Hb = H + G.T @ (D[:, None] * G)

        def direction(rc):
            r1 = -rd + G.T @ (rc / s) - G.T @ (D * rp)
            dx, dy = kkt_solve(Hb, r1, -re)
            ds = -rp - G @ dx
            dz = -(rc + z * ds) / s
            return dx, dy, ds, dz

        def step_len(v, dv):
            neg = dv < 0
            return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0

        dx, dy, ds, dz = direction(s * z)
        a_aff = min(step_len(s, ds), step_len(z, dz))
        mu_aff = float((s + a_aff * ds) @ (z + a_aff * dz)) / m
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
        dx, dy, ds, dz = direction(s * z + ds * dz - sigma * mu)
        a = 0.995 * min(step_len(s, ds), step_len(z, dz))
        a = min(1.0, a)
        x, s, z = x + a * dx, s + a * ds, z + a * dz
        if p:
            y = y + a * dy
    else:
        flag = 0
    if flag != 1:
        # is the problem infeasible?
        try:
            chk = linprog_core(np.zeros(n), A, b, Aeq, beq, lo, hi)
            if chk.status == 2:
                return x, lam, -2, it
        except Exception:  # pragma: no cover
            pass
        if flag == -3 or np.max(np.abs(x)) > 1e8 * bscale:
            return x, lam, -3, it
    # refinement: solve the equality-constrained problem on the active set exactly
    active = (s < 1e-6 * (1 + np.abs(h))) & (z > 1e-8 * max(1.0, float(np.max(z))))
    if active.any() or p:
        Ga = G[active]
        K = np.block([[H, Ga.T, Aeq.T], [Ga, np.zeros((Ga.shape[0], Ga.shape[0] + p))],
                      [Aeq, np.zeros((p, Ga.shape[0] + p))]])
        rhs = np.concatenate([-f, h[active], beq])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        xr, zr, yr = sol[:n], sol[n:n + Ga.shape[0]], sol[n + Ga.shape[0]:]
        ok = np.all(G @ xr - h <= 1e-9 * (1 + np.abs(h))) and np.all(zr >= -1e-9 * (1 + np.max(np.abs(zr), initial=0)))
        resid = H @ xr + f + Ga.T @ zr + (Aeq.T @ yr if p else 0.0)
        xr_scale = max(1.0, float(np.max(np.abs(xr), initial=0.0)))
        if ok and np.max(np.abs(resid), initial=0.0) <= 1e-8 * hscale * xr_scale:
            x = xr
            z = np.zeros(m)
            z[active] = np.maximum(zr, 0.0)
            y = yr
            flag = 1 if flag in (0, 1) else flag
    z = np.where(z < 1e-10 * max(1.0, float(np.max(z, initial=0.0))), 0.0, z)
    split(z, y)
    return x, lam, flag, it


@builtin("quadprog")
def _quadprog(interp, args, nargout):
    args = om.problem_args(args, ("H", "f", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options"),
                           "quadprog")
    nargs(args, 1, 11, "quadprog")
    H = om._arg(args, 0)
    f = om._arg(args, 1)
    if H is None and f is None:
        raise MatlabError("quadprog: H and f cannot both be empty.")
    n = H.shape[0] if H is not None else f.size
    H = np.zeros((n, n)) if H is None else H
    if H.shape != (n, n):
        raise MatlabError("quadprog: H must be a square matrix.")
    f = np.zeros(n) if f is None else f.flatten(order="F")
    if f.size != n:
        raise MatlabError("quadprog: f must have one element per variable.")
    if not np.allclose(H, H.T, atol=1e-12 * max(1.0, float(np.max(np.abs(H), initial=0.0)))):
        interp.warn("quadprog: H is not symmetric; using (H + H')/2 instead.")
    H = 0.5 * (H + H.T)
    A, b = om.linear(om._arg(args, 2), om._arg(args, 3), n, "quadprog", "A")
    Aeq, beq = om.linear(om._arg(args, 4), om._arg(args, 5), n, "quadprog", "Aeq")
    lo, hi = om.bounds(om._arg(args, 6), om._arg(args, 7), n, "quadprog")
    x0 = om._arg(args, 8)
    opts = Opt(_find_options(args, 8), "quadprog",
               ("Algorithm", "Display", "MaxIterations", "OptimalityTolerance", "StepTolerance",
                "ConstraintTolerance", "LinearSolver"))
    alg = opts.word("Algorithm", "interior-point-convex", om.SOLVER_ALGORITHMS["quadprog"])
    rep = Report(interp, opts)
    ev = np.linalg.eigvalsh(H) if n else np.zeros(0)
    convex = bool(np.all(ev >= -1e-10 * max(1.0, float(np.max(np.abs(ev), initial=0.0)))))
    empty = (np.zeros((0, 0)), np.zeros((0, 0)))
    iters = 0
    if not convex and alg == "interior-point-convex":
        flag, msg = -6, "The problem is non-convex."
        x, fval, lam = empty[0], empty[1], Struct.scalar({k: np.zeros((0, 0)) for k in
                                                          ("ineqlin", "eqlin", "lower", "upper")})
    else:
        if convex:
            x, lam_d, flag, iters = solve_qp(H, f, A, b, Aeq, beq, lo, hi, None if x0 is None else x0.flatten(),
                                             tol=min(opts.num("OptimalityTolerance", 1e-8), 1e-8),
                                             maxit=int(opts.num("MaxIterations", 200)))
        else:
            P = NLP(n, lambda v: float(0.5 * v @ H @ v + f @ v), lambda v: H @ v + f, A, b, Aeq, beq, lo, hi)
            res = solve_nlp(P, np.zeros(n) if x0 is None else x0.flatten(), "sqp", 1e-8, 1e-8, 1e-12, 1000)
            x, lam_d, flag, iters = res["x"], res["lam"], (1 if res["flag"] in (1, 2) else res["flag"]), \
                res["iterations"]
        msg = {1: "Minimum found that satisfies the constraints.",
               0: "Solver stopped prematurely: the iteration limit was reached.",
               -2: "No feasible solution found: the constraints are inconsistent.",
               -3: "The problem is unbounded."}.get(flag, "The solver failed.")
        if flag in (-2, -3):
            x, fval = empty
            lam = Struct.scalar({k: np.zeros((0, 0)) for k in ("ineqlin", "eqlin", "lower", "upper")})
        else:
            fval = float(0.5 * x @ H @ x + f @ x)
            lam = lambda_struct(lam_d, ("ineqlin", "eqlin", "lower", "upper"))
            x = x.reshape(-1, 1)
    rep.finish(flag, msg)
    out = output_struct(message=msg, algorithm=alg, firstorderopt=0.0, constrviolation=0.0, iterations=iters,
                        linearsolver="dense", cgiterations=empty_double())
    return outs(nargout, (x, fval, float(flag), out, lam))


@builtin("lsqlin")
def _lsqlin(interp, args, nargout):
    args = om.problem_args(args, ("C", "d", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options"),
                           "lsqlin")
    nargs(args, 2, 10, "lsqlin")
    C, d = om._arg(args, 0), om._arg(args, 1)
    if C is None or d is None:
        raise MatlabError("lsqlin: C and d must not be empty.")
    d = d.flatten(order="F")
    if C.shape[0] != d.size:
        raise MatlabError("lsqlin: C and d must have the same number of rows.")
    n = C.shape[1]
    A, b = om.linear(om._arg(args, 2), om._arg(args, 3), n, "lsqlin", "A")
    Aeq, beq = om.linear(om._arg(args, 4), om._arg(args, 5), n, "lsqlin", "Aeq")
    lo, hi = om.bounds(om._arg(args, 6), om._arg(args, 7), n, "lsqlin")
    x0 = om._arg(args, 8)
    opts = Opt(_find_options(args, 8), "lsqlin",
               ("Algorithm", "Display", "MaxIterations", "OptimalityTolerance", "StepTolerance",
                "ConstraintTolerance", "LinearSolver"))
    alg = opts.word("Algorithm", "interior-point", om.SOLVER_ALGORITHMS["lsqlin"])
    rep = Report(interp, opts)
    unconstrained = not (A.shape[0] or Aeq.shape[0] or np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)))
    iters = 0
    lam_d = {"ineqlin": np.zeros(0), "eqlin": np.zeros(0), "lower": np.zeros(n), "upper": np.zeros(n)}
    if unconstrained:
        x = np.linalg.lstsq(C, d, rcond=None)[0]
        flag = 1
    else:
        H = C.T @ C
        # a tiny ridge keeps the method going when C does not have full column rank
        if np.linalg.matrix_rank(C) < n:
            H = H + 1e-12 * max(1.0, float(np.trace(H)) / max(n, 1)) * np.eye(n)
        x, lam_d, flag, iters = solve_qp(H, -(C.T @ d), A, b, Aeq, beq, lo, hi,
                                         None if x0 is None else x0.flatten(), tol=1e-9,
                                         maxit=int(opts.num("MaxIterations", 200)))
    msg = {1: "Minimum found that satisfies the constraints.",
           0: "Solver stopped prematurely: the iteration limit was reached.",
           -2: "No feasible solution found: the constraints are inconsistent.",
           -3: "The problem is unbounded."}.get(flag, "The solver failed.")
    rep.finish(flag, msg)
    out = output_struct(message=msg, algorithm=alg, firstorderopt=0.0, constrviolation=0.0, iterations=iters,
                        linearsolver="dense", cgiterations=empty_double())
    if flag == -2:
        e = np.zeros((0, 0))
        return outs(nargout, (e, e, e, float(flag), out,
                              Struct.scalar({k: e for k in ("ineqlin", "eqlin", "lower", "upper")})))
    resid = C @ x - d
    return outs(nargout, (x.reshape(-1, 1), float(resid @ resid), resid.reshape(-1, 1), float(flag), out,
                          lambda_struct(lam_d, ("ineqlin", "eqlin", "lower", "upper"))))


@builtin("lsqnonneg")
def _lsqnonneg(interp, args, nargout):
    from scipy.optimize import nnls
    args = om.problem_args(args, ("C", "d", "options"), "lsqnonneg")
    nargs(args, 2, 4, "lsqnonneg")
    C = np.real(fnum(args[0])).astype(float)
    d = np.real(fnum(args[1])).astype(float).flatten(order="F")
    if C.shape[0] != d.size:
        raise MatlabError("lsqnonneg: C and d must have the same number of rows.")
    opts = Opt(_find_options(args, 2), "lsqnonneg")
    rep = Report(interp, opts, "notify")
    n = C.shape[1]
    maxit = max(3 * n, 200)
    try:
        x, _ = nnls(C, d, maxiter=maxit)
        flag = 1
    except RuntimeError:
        x, _ = nnls(C, d, maxiter=50 * maxit)
        flag = 0
    resid = d - C @ x
    w = C.T @ resid
    msg = "Optimization terminated." if flag == 1 else \
        "Exiting: the iteration count was exceeded; the result may not be optimal."
    rep.finish(flag, msg)
    out = output_struct(iterations=float(np.count_nonzero(x)), algorithm="active-set", message=msg)
    return outs(nargout, (x.reshape(-1, 1), float(resid @ resid), resid.reshape(-1, 1), float(flag), out,
                          w.reshape(-1, 1)))


# -- minimax, goal attainment, semi-infinite ---------------------------------------------------------------------------

def _multi_objective(interp, what, args, nargout, goal=None, weight=None):
    """Shared driver: minimise gamma subject to F_i(x) - w_i*gamma <= goal_i."""
    x0a, shape, n, opts, obj, A, b, Aeq, beq, lo, hi, nl, nl_jac = _nlp_setup(interp, args, what, vector=True)
    x0 = np.clip(x0a.flatten(order="F"), lo, hi)
    F0 = obj(x0)
    m = F0.size
    if goal is None:
        g_vec, w_vec = np.zeros(m), np.ones(m)
        nabs = int(opts.num("AbsoluteMaxObjectiveCount", 0))
        neq = 0
    else:
        g_vec, w_vec = goal.flatten(order="F"), weight.flatten(order="F")
        if g_vec.size != m or w_vec.size != m:
            raise MatlabError(f"{what}: goal and weight must have one entry per objective ({m}).")
        nabs = 0
        neq = int(opts.num("EqualityGoalCount", 0))
    tol_opt = opts.num("OptimalityTolerance", 1e-6)
    tol_con = opts.num("ConstraintTolerance", 1e-6)
    rep = Report(interp, opts)
    hook = OutputHook(interp, opts, shape, what)

    def cons(z):
        x, gam = z[:n], z[n]
        F = obj(x)
        ineq = [(F - w_vec * gam - g_vec)[neq:]]
        if nabs:
            ineq.append(-F[:nabs] - gam)
        c_user, ceq_user = nl(x) if nl is not None else (np.zeros(0), np.zeros(0))
        eq = [ceq_user]
        if neq:
            eq.append(F[:neq] - g_vec[:neq])      # goals that must be met exactly
        return np.concatenate(ineq + [c_user]), np.concatenate(eq)

    def cons_jac(z):
        x = z[:n]
        J = np.atleast_2d(obj.grad(x))
        rows = [np.hstack([J, -w_vec.reshape(-1, 1)])[neq:]]
        if nabs:
            rows.append(np.hstack([-J[:nabs], -np.ones((nabs, 1))]))
        Jc_u, Je_u = (nl_jac(x) if nl_jac is not None else (None, None))
        if nl is not None and Jc_u is None:
            c0, e0 = nl(x)
            h = np.sqrt(EPS) * np.maximum(np.abs(x), 1.0)
            Jc_u, Je_u = np.zeros((c0.size, n)), np.zeros((e0.size, n))
            for j in range(n):
                xp = x.copy()
                xp[j] += h[j]
                c1, e1 = nl(xp)
                Jc_u[:, j] = (c1 - c0) / h[j]
                Je_u[:, j] = (e1 - e0) / h[j]
        if Jc_u is None:
            Jc_u, Je_u = np.zeros((0, n)), np.zeros((0, n))
        eq_rows = [np.hstack([Je_u, np.zeros((Je_u.shape[0], 1))])]
        if neq:
            eq_rows.append(np.hstack([J[:neq], np.zeros((neq, 1))]))
        rows.append(np.hstack([Jc_u, np.zeros((Jc_u.shape[0], 1))]))
        return np.vstack(rows), np.vstack(eq_rows)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(w_vec != 0, (F0 - g_vec) / np.where(w_vec != 0, w_vec, 1.0), -np.inf)
    gam0 = float(np.max(ratios)) if np.any(np.isfinite(ratios)) else 0.0
    if nabs:
        gam0 = max(gam0, float(np.max(np.abs(F0[:nabs]))))
    z0 = np.concatenate([x0, [gam0]])
    Az = np.hstack([A, np.zeros((A.shape[0], 1))])
    Aeqz = np.hstack([Aeq, np.zeros((Aeq.shape[0], 1))])
    P = NLP(n + 1, lambda z: float(z[n]), lambda z: np.concatenate([np.zeros(n), [1.0]]), Az, b, Aeqz, beq,
            np.concatenate([lo, [-np.inf]]), np.concatenate([hi, [np.inf]]), cons, cons_jac)

    def callback(zk, k):
        return hook.call(zk[:n], "iter", iteration=k, funccount=obj.count, fval=obj(zk[:n]))

    res = solve_nlp(P, z0, "sqp", tol_opt, tol_con, opts.num("StepTolerance", 1e-6),
                    opts.num("MaxIterations", 400), callback if hook else None)
    x = res["x"][:n]
    F = obj(x)
    gam = float(res["x"][n])
    flag = res["flag"]
    msg = res["message"]
    if flag == 1 and what == "fgoalattain":
        msg = "Local minimum found that satisfies the constraints."
    elif flag == 2:
        flag = 4
    rep.finish(flag, msg)
    out = output_struct(iterations=res["iterations"], funcCount=obj.count, lssteplength=1.0, stepsize=0.0,
                        algorithm="active-set", firstorderopt=res["firstorder"], constrviolation=res["viol"],
                        message=msg)
    lam = res["lam"]
    nobj = m - neq + nabs
    lam_out = {"lower": lam["lower"][:n], "upper": lam["upper"][:n], "eqlin": lam["eqlin"],
               "ineqlin": lam["ineqlin"], "eqnonlin": lam["eqnonlin"][:lam["eqnonlin"].size - neq],
               "ineqnonlin": lam["ineqnonlin"][nobj:]}
    F_out = F.reshape(obj.out_shape, order="F") if obj.out_shape and int(np.prod(obj.out_shape)) == F.size \
        else F.reshape(-1, 1)
    return x.reshape(shape, order="F"), F_out, gam, float(flag), out, lambda_struct(lam_out), F, nabs


@builtin("fminimax")
def _fminimax(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "nonlcon",
                                  "options"), "fminimax")
    nargs(args, 2, 10, "fminimax")
    x, F, gam, flag, out, lam, Fv, nabs = _multi_objective(interp, "fminimax", args, nargout)
    worst = Fv.copy()
    if nabs:
        worst[:nabs] = np.abs(worst[:nabs])
    return outs(nargout, (x, F, float(np.max(worst)), flag, out, lam))


@builtin("fgoalattain")
def _fgoalattain(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "goal", "weight", "Aineq", "bineq", "Aeq", "beq", "lb",
                                  "ub", "nonlcon", "options"), "fgoalattain")
    nargs(args, 4, 12, "fgoalattain")
    goal, weight = om._arg(args, 2), om._arg(args, 3)
    if goal is None or weight is None:
        raise MatlabError("fgoalattain: goal and weight must not be empty.")
    rest = [args[0], args[1]] + list(args[4:])
    x, F, gam, flag, out, lam, _, _ = _multi_objective(interp, "fgoalattain", rest, nargout, goal, weight)
    return outs(nargout, (x, F, gam, flag, out, lam))


@builtin("fseminf")
def _fseminf(interp, args, nargout):
    args = om.problem_args(args, ("objective", "x0", "ntheta", "seminfcon", "Aineq", "bineq", "Aeq", "beq",
                                  "lb", "ub", "options"), "fseminf")
    nargs(args, 4, 11, "fseminf")
    ntheta = int(np.real(fnum(args[2]).flat[0]))
    semi = args[3]
    if not om.is_fn(semi):
        raise MatlabError("fseminf: seminfcon must be a function handle.")
    rest = [args[0], args[1]] + list(args[4:10]) + [None] + ([args[10]] if len(args) > 10 else [])
    while len(rest) < 9:
        rest.append(empty_double())
    x0a, shape, n, opts, obj, A, b, Aeq, beq, lo, hi, _, _ = _nlp_setup(interp, rest, "fseminf")
    rep = Report(interp, opts)
    S = [np.full((ntheta, 2), np.nan)]

    def nl(x):
        res = interp.call_function_value(semi, [x.reshape(shape, order="F"), S[0]], 3 + ntheta)
        if len(res) < 3 + ntheta:
            raise MatlabError(f"fseminf: seminfcon must return [c, ceq, K1, ..., K{ntheta}, s].")
        c = np.real(fnum(res[0])).astype(float).flatten(order="F")
        ceq = np.real(fnum(res[1])).astype(float).flatten(order="F")
        ks = [np.real(fnum(k)).astype(float).flatten(order="F") for k in res[2:2 + ntheta]]
        s_new = fnum(res[2 + ntheta])
        if s_new.shape == (ntheta, 2):
            S[0] = np.real(s_new).astype(float)
        return np.concatenate([c] + ks), ceq

    P = NLP(n, obj, obj.grad, A, b, Aeq, beq, lo, hi, nl, None)
    res = solve_nlp(P, x0a.flatten(order="F"), "sqp", opts.num("OptimalityTolerance", 1e-6),
                    opts.num("ConstraintTolerance", 1e-6), opts.num("StepTolerance", 1e-6),
                    opts.num("MaxIterations", 400))
    flag = 4 if res["flag"] == 2 else res["flag"]
    rep.finish(flag, res["message"])
    out = output_struct(iterations=res["iterations"], funcCount=obj.count, lssteplength=1.0, stepsize=0.0,
                        algorithm="active-set", firstorderopt=res["firstorder"], constrviolation=res["viol"],
                        message=res["message"])
    return outs(nargout, (res["x"].reshape(shape, order="F"), res["f"], float(flag), out,
                          lambda_struct(res["lam"])))
