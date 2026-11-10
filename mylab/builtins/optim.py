"""Constrained optimization: linear, quadratic, integer and nonlinear programming, least squares."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char
from ._registry import builtin
from ._util import fnum, is_text, nargs


def _arg(args: list, i: int) -> np.ndarray | None:
    """Numeric argument i, or None when missing or given as []."""
    if len(args) <= i:
        return None
    a = args[i]
    if isinstance(a, (Struct,)) or is_text(a):
        return None
    if isinstance(a, np.ndarray) and a.size == 0:
        return None
    return fnum(a).astype(float)


def _bounds(lb, ub, n):
    lo = np.full(n, -np.inf) if lb is None else np.broadcast_to(lb.flatten(), (n,)).astype(float).copy()
    hi = np.full(n, np.inf) if ub is None else np.broadcast_to(ub.flatten(), (n,)).astype(float).copy()
    if lb is not None and lb.size not in (1, n) or ub is not None and ub.size not in (1, n):
        raise MatlabError("Bounds must have one entry per variable.")
    if np.any(lo > hi):
        raise MatlabError("Lower bounds must not exceed upper bounds.")
    return lo, hi


def _opts_struct(args: list, i: int) -> dict:
    if len(args) > i and isinstance(args[i], Struct):
        s = args[i]
        return {k.lower(): v for k, v in s.data.flat[0].items()}
    return {}


def _opt_num(opts: dict, name: str, default: float) -> float:
    v = opts.get(name.lower())
    if isinstance(v, np.ndarray) and v.size:
        return float(fnum(v).flat[0])
    return default


def _call(interp, f, args, nargout=1):
    vals = interp.call_function_value(f, args, nargout)
    if not vals:
        raise MatlabError("The function must return a value.")
    return vals


def _outs(nargout, *vals):
    return vals[:max(nargout, 1)] if nargout > 1 else vals[0]


def _lin_constraints(A, b, Aeq, beq, n):
    if A is not None:
        A = A.reshape(-1, n) if A.size else None
        if b is None or b.size != A.shape[0]:
            raise MatlabError("A and b must have compatible sizes.")
        b = b.flatten()
    if Aeq is not None:
        Aeq = Aeq.reshape(-1, n)
        if beq is None or beq.size != Aeq.shape[0]:
            raise MatlabError("Aeq and beq must have compatible sizes.")
        beq = beq.flatten()
    return A, b, Aeq, beq


# -- linear programming -----------------------------------------------------------------

_LP_FLAG = {0: 1.0, 1: 0.0, 2: -2.0, 3: -3.0, 4: -4.0}


@builtin("linprog")
def _linprog(interp, args, nargout):
    from scipy.optimize import linprog
    nargs(args, 1, 8, "linprog")
    f = _arg(args, 0)
    if f is None:
        raise MatlabError("linprog: f must be a non-empty vector.")
    f = f.flatten()
    n = f.size
    A, b, Aeq, beq = _lin_constraints(_arg(args, 1), _arg(args, 2), _arg(args, 3), _arg(args, 4), n)
    lo, hi = _bounds(_arg(args, 5), _arg(args, 6), n)
    res = linprog(f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=list(zip(lo, hi)), method="highs")
    flag = _LP_FLAG.get(res.status, -4.0)
    if res.x is None or flag < 0:
        x = np.zeros((0, 1))
        fval = np.zeros((0, 0))
    else:
        x = res.x.reshape(-1, 1)
        fval = float(res.fun)
    output = Struct.scalar({"iterations": np.array([[float(getattr(res, "nit", 0))]]),
                            "message": char(str(res.message))})
    return _outs(nargout, x, fval, flag, output)


@builtin("intlinprog")
def _intlinprog(interp, args, nargout):
    from scipy.optimize import Bounds, LinearConstraint, milp
    nargs(args, 2, 9, "intlinprog")
    f = _arg(args, 0)
    if f is None:
        raise MatlabError("intlinprog: f must be a non-empty vector.")
    f = f.flatten()
    n = f.size
    intcon = _arg(args, 1)
    integrality = np.zeros(n)
    if intcon is not None:
        ic = intcon.flatten().astype(int)
        if np.any(ic < 1) or np.any(ic > n):
            raise MatlabError("intlinprog: intcon entries must be variable indices.")
        integrality[ic - 1] = 1
    A, b, Aeq, beq = _lin_constraints(_arg(args, 2), _arg(args, 3), _arg(args, 4), _arg(args, 5), n)
    lo, hi = _bounds(_arg(args, 6), _arg(args, 7), n)
    cons = []
    if A is not None:
        cons.append(LinearConstraint(A, -np.inf, b))
    if Aeq is not None:
        cons.append(LinearConstraint(Aeq, beq, beq))
    res = milp(f, integrality=integrality, bounds=Bounds(lo, hi), constraints=cons)
    flag = {0: 1.0, 1: 0.0, 2: -2.0, 3: -3.0}.get(res.status, -9.0)
    if res.x is None:
        x = np.zeros((0, 1))
        fval = np.zeros((0, 0))
    else:
        xv = res.x.copy()
        xv[integrality == 1] = np.round(xv[integrality == 1])
        x = xv.reshape(-1, 1)
        fval = float(f @ xv)
    return _outs(nargout, x, fval, flag)


# -- quadratic programming --------------------------------------------------------------

def _slsqp(fun, jac, x0, A, b, Aeq, beq, lo, hi, extra_cons=(), maxiter=1000):
    from scipy.optimize import minimize
    cons = list(extra_cons)
    if A is not None:
        cons.append({"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A})
    if Aeq is not None:
        cons.append({"type": "eq", "fun": lambda x: Aeq @ x - beq, "jac": lambda x: Aeq})
    bounds = None
    if np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)):
        bounds = [(None if not np.isfinite(a) else a, None if not np.isfinite(c) else c) for a, c in zip(lo, hi)]
        x0 = np.clip(x0, np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf))
    return minimize(fun, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
                    options={"maxiter": maxiter, "ftol": 1e-12})


def _feasible(x, A, b, Aeq, beq, lo, hi, tol=1e-6) -> bool:
    ok = np.all(x >= lo - tol) and np.all(x <= hi + tol)
    if A is not None:
        ok = ok and np.all(A @ x <= b + tol)
    if Aeq is not None:
        ok = ok and np.all(np.abs(Aeq @ x - beq) <= tol * max(1.0, np.max(np.abs(beq))))
    return bool(ok)


@builtin("quadprog")
def _quadprog(interp, args, nargout):
    nargs(args, 1, 10, "quadprog")
    H = _arg(args, 0)
    f = _arg(args, 1)
    if H is None:
        raise MatlabError("quadprog: H must be a non-empty matrix.")
    n = H.shape[0]
    if H.shape != (n, n):
        raise MatlabError("quadprog: H must be square.")
    H = (H + H.T) / 2
    f = np.zeros(n) if f is None else f.flatten()
    A, b, Aeq, beq = _lin_constraints(_arg(args, 2), _arg(args, 3), _arg(args, 4), _arg(args, 5), n)
    lb, ub = _arg(args, 6), _arg(args, 7)
    lo, hi = _bounds(lb, ub, n)
    x0 = _arg(args, 8)
    x0 = np.zeros(n) if x0 is None else x0.flatten()

    def obj(x):
        return 0.5 * x @ H @ x + f @ x

    def grad(x):
        return H @ x + f

    unconstrained = A is None and Aeq is None and lb is None and ub is None
    if unconstrained:
        try:
            ev = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            ev = np.array([-1.0])
        if np.all(ev > 0):
            x = np.linalg.solve(H, -f)
            return _outs(nargout, x.reshape(-1, 1), float(obj(x)), 1.0)
        raise MatlabError("quadprog: problem is unbounded (H is not positive definite and there are no constraints).")
    res = _slsqp(obj, grad, x0, A, b, Aeq, beq, lo, hi)
    x = res.x
    flag = 1.0 if res.success and _feasible(x, A, b, Aeq, beq, lo, hi) else (
        -2.0 if not _feasible(x, A, b, Aeq, beq, lo, hi) else 0.0)
    return _outs(nargout, x.reshape(-1, 1), float(obj(x)), flag)


# -- nonlinear programming ----------------------------------------------------------------

@builtin("fmincon")
def _fmincon(interp, args, nargout):
    nargs(args, 2, 10, "fmincon")
    fun = args[0]
    x0a = fnum(args[1]).astype(float)
    shape = x0a.shape
    n = x0a.size
    A, b, Aeq, beq = _lin_constraints(_arg(args, 2), _arg(args, 3), _arg(args, 4), _arg(args, 5), n)
    lo, hi = _bounds(_arg(args, 6), _arg(args, 7), n)
    nonlcon = args[8] if len(args) > 8 and not (isinstance(args[8], np.ndarray) and args[8].size == 0) else None
    opts = _opts_struct(args, 9)
    maxiter = int(_opt_num(opts, "MaxIterations", _opt_num(opts, "MaxIter", 1000)))

    def obj(x):
        v = fnum(_call(interp, fun, [x.reshape(shape)])[0])
        return float(np.real(v.flat[0]))

    extra = []
    if nonlcon is not None:
        cache: dict = {}

        def nl(x):
            key = x.tobytes()
            if key not in cache:
                cache.clear()
                vals = _call(interp, nonlcon, [x.reshape(shape)], 2)
                c = fnum(vals[0]).flatten() if vals and vals[0] is not None else np.zeros(0)
                ceq = fnum(vals[1]).flatten() if len(vals) > 1 else np.zeros(0)
                cache[key] = (c.astype(float), ceq.astype(float))
            return cache[key]

        c0, ceq0 = nl(x0a.flatten())
        if c0.size:
            extra.append({"type": "ineq", "fun": lambda x: -nl(x)[0]})
        if ceq0.size:
            extra.append({"type": "eq", "fun": lambda x: nl(x)[1]})
    res = _slsqp(obj, None, x0a.flatten(), A, b, Aeq, beq, lo, hi, extra, maxiter=maxiter)
    x = res.x
    feasible = _feasible(x, A, b, Aeq, beq, lo, hi)
    if nonlcon is not None and feasible:
        vals = _call(interp, nonlcon, [x.reshape(shape)], 2)
        c = fnum(vals[0]).flatten() if vals else np.zeros(0)
        ceq = fnum(vals[1]).flatten() if len(vals) > 1 else np.zeros(0)
        feasible = bool(np.all(c <= 1e-6) and np.all(np.abs(ceq) <= 1e-6))
    if res.success and feasible:
        flag = 1.0
    elif not feasible:
        flag = -2.0
    else:
        flag = 0.0
    return _outs(nargout, x.reshape(shape), float(res.fun), flag)


# -- least squares ------------------------------------------------------------------------

@builtin("lsqnonneg")
def _lsqnonneg(interp, args, nargout):
    from scipy.optimize import nnls
    nargs(args, 2, 3, "lsqnonneg")
    C = fnum(args[0]).astype(float)
    d = fnum(args[1]).astype(float).flatten()
    if C.shape[0] != d.size:
        raise MatlabError("lsqnonneg: C and d must have the same number of rows.")
    x, _ = nnls(C, d, maxiter=max(50 * C.shape[1], 1000))
    resid = d - C @ x
    return _outs(nargout, x.reshape(-1, 1), float(resid @ resid), resid.reshape(-1, 1), 1.0)


def _least_squares(interp, resid_fn, x0a, lb, ub, opts):
    from scipy.optimize import least_squares
    n = x0a.size
    lo, hi = _bounds(lb, ub, n)
    x0 = np.clip(x0a.flatten(), lo, hi)
    bounded = np.any(np.isfinite(lo)) or np.any(np.isfinite(hi))
    res = least_squares(resid_fn, x0, bounds=(lo, hi), method="trf" if bounded else "lm",
                        xtol=_opt_num(opts, "StepTolerance", _opt_num(opts, "TolX", 1e-10)),
                        ftol=_opt_num(opts, "FunctionTolerance", _opt_num(opts, "TolFun", 1e-10)))
    flag = 1.0 if res.success else 0.0
    return res.x, res.fun, flag


@builtin("lsqcurvefit")
def _lsqcurvefit(interp, args, nargout):
    nargs(args, 4, 7, "lsqcurvefit")
    fun = args[0]
    x0a = fnum(args[1]).astype(float)
    shape = x0a.shape
    xdata = args[2]
    ydata = fnum(args[3]).astype(float)
    opts = _opts_struct(args, 6)

    def resid(x):
        v = fnum(_call(interp, fun, [x.reshape(shape), xdata])[0]).astype(float)
        if v.size != ydata.size:
            raise MatlabError("lsqcurvefit: the model output must have the same size as ydata.")
        return (v - ydata).flatten(order="F")

    x, r, flag = _least_squares(interp, resid, x0a, _arg(args, 4), _arg(args, 5), opts)
    r_out = r.reshape(ydata.shape, order="F")
    return _outs(nargout, x.reshape(shape), float(r @ r), r_out, flag)


@builtin("lsqnonlin")
def _lsqnonlin(interp, args, nargout):
    nargs(args, 2, 5, "lsqnonlin")
    fun = args[0]
    x0a = fnum(args[1]).astype(float)
    shape = x0a.shape
    opts = _opts_struct(args, 4)
    out_shape: list = []

    def resid(x):
        v = fnum(_call(interp, fun, [x.reshape(shape)])[0]).astype(float)
        if not out_shape:
            out_shape.append(v.shape)
        return v.flatten(order="F")

    x, r, flag = _least_squares(interp, resid, x0a, _arg(args, 2), _arg(args, 3), opts)
    r_out = r.reshape(out_shape[0], order="F") if out_shape and np.prod(out_shape[0]) == r.size else r.reshape(-1, 1)
    return _outs(nargout, x.reshape(shape), float(r @ r), r_out, flag)

