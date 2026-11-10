"""Polynomials, interpolation, root finding, optimization, integration, ODE solvers."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..values import Struct, char, simplify, to_int
from ._registry import builtin
from ._util import fnum, is_text, nargs, text


def _call_scalar(interp, f, *xs) -> float:
    vals = interp.call_function_value(f, [np.array([[float(x)]]) for x in xs], 1)
    if not vals:
        raise MatlabError("The function must return a value.")
    a = fnum(vals[0])
    return float(np.real(a.flat[0]))


def _call_vector(interp, f, args) -> np.ndarray:
    vals = interp.call_function_value(f, args, 1)
    if not vals:
        raise MatlabError("The function must return a value.")
    return fnum(vals[0])


# -- polynomials ----------------------------------------------------------------------

@builtin("polyfit")
def _polyfit(interp, args, nargout):
    x, y = fnum(args[0]).flatten(), fnum(args[1]).flatten()
    n = to_int(args[2], "degree")
    p = np.polyfit(x, y, n)
    if nargout > 1:
        s = Struct.scalar({"normr": float(np.linalg.norm(y - np.polyval(p, x))),
                           "df": float(max(0, x.size - (n + 1)))})
        return p.reshape(1, -1), s
    return p.reshape(1, -1)


@builtin("polyval")
def _polyval(interp, args, nargout):
    p = fnum(args[0]).flatten()
    x = fnum(args[1])
    return simplify(np.polyval(p, x))


@builtin("roots")
def _roots(interp, args, nargout):
    r = np.roots(fnum(args[0]).flatten())
    return simplify(r.reshape(-1, 1)) if r.size else np.zeros((0, 1))


@builtin("poly")
def _poly(interp, args, nargout):
    a = fnum(args[0])
    if a.shape[0] == a.shape[1] and a.shape[0] > 1:
        return simplify(np.poly(a).reshape(1, -1))
    return simplify(np.poly(a.flatten()).reshape(1, -1))


@builtin("polyder")
def _polyder(interp, args, nargout):
    return np.polyder(fnum(args[0]).flatten()).reshape(1, -1)


@builtin("polyint")
def _polyint(interp, args, nargout):
    k = float(fnum(args[1]).flat[0]) if len(args) > 1 else 0.0
    return np.polyint(fnum(args[0]).flatten(), k=k).reshape(1, -1)


@builtin("conv")
def _conv(interp, args, nargout):
    a, b = fnum(args[0]), fnum(args[1])
    shape = text(args[2]) if len(args) > 2 else "full"
    r = np.convolve(a.flatten(), b.flatten(), mode="full")
    if shape == "same":
        start = int(np.ceil((b.size - 1) / 2))
        r = r[start:start + a.size]
    elif shape == "valid":
        r = np.convolve(a.flatten(), b.flatten(), mode="valid") if a.size >= b.size else np.zeros(0)
    col = a.shape[1] == 1 and a.shape[0] > 1
    return simplify(r.reshape(-1, 1) if col else r.reshape(1, -1))


@builtin("deconv")
def _deconv(interp, args, nargout):
    q, r = np.polydiv(fnum(args[0]).flatten(), fnum(args[1]).flatten())
    full_r = np.zeros(fnum(args[0]).size)
    full_r[-r.size:] = r
    return q.reshape(1, -1), full_r.reshape(1, -1)


@builtin("filter")
def _filter(interp, args, nargout):
    from scipy.signal import lfilter
    b, a, x = fnum(args[0]).flatten(), fnum(args[1]).flatten(), fnum(args[2])
    axis = 1 if x.shape[0] == 1 else 0
    return lfilter(b, a, x, axis=axis)


# -- interpolation ----------------------------------------------------------------------

@builtin("interp1")
def _interp1(interp, args, nargout):
    nargs(args, 2, 5, "interp1")
    rest = list(args)
    method = "linear"
    extrap = None
    strs = [k for k, a in enumerate(rest) if is_text(a)]
    if strs:
        k = strs[0]
        method = text(rest[k]).lower()
        tail = rest[k + 1:]
        rest = rest[:k]
        if tail:
            extrap = "extrap" if is_text(tail[0]) and text(tail[0]) == "extrap" else float(fnum(tail[0]).flat[0])
    if len(rest) == 2:
        y = fnum(rest[0])
        xq = fnum(rest[1])
        x = np.arange(1, (y.size if y.shape[0] == 1 or y.shape[1] == 1 else y.shape[0]) + 1, dtype=float)
    else:
        x, y, xq = fnum(rest[0]).flatten(), fnum(rest[1]), fnum(rest[2])
    yv = y.flatten() if (y.shape[0] == 1 or y.shape[1] == 1) else y
    from scipy.interpolate import interp1d
    kind = {"linear": "linear", "nearest": "nearest", "previous": "previous", "next": "next",
            "spline": "cubic", "cubic": "cubic", "pchip": "pchip", "makima": "cubic"}.get(method)
    if kind is None:
        raise MatlabError(f"Unknown interpolation method '{method}'.")
    order = np.argsort(x)
    x = x[order]
    yv = yv[order] if yv.ndim == 1 else yv[order, :]
    if kind == "pchip":
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(x, yv, extrapolate=extrap == "extrap" or extrap is None and False)
        out = f(xq.flatten())
        if extrap != "extrap":
            outside = (xq.flatten() < x[0]) | (xq.flatten() > x[-1])
            out[outside] = np.nan if extrap is None else extrap
    else:
        fill = "extrapolate" if (extrap == "extrap" or (extrap is None and method == "spline")) else \
            (np.nan if extrap is None else extrap)
        f = interp1d(x, yv, kind=kind, axis=0, bounds_error=False, fill_value=fill)
        out = f(xq.flatten())
    if yv.ndim == 1:
        return np.asarray(out).reshape(xq.shape)
    return np.asarray(out)


@builtin("spline")
def _spline(interp, args, nargout):
    from scipy.interpolate import CubicSpline
    x, y = fnum(args[0]).flatten(), fnum(args[1]).flatten()
    cs = CubicSpline(x, y, bc_type="not-a-knot")
    if len(args) > 2:
        xq = fnum(args[2])
        return cs(xq.flatten()).reshape(xq.shape)
    raise MatlabError("spline(x, y) without query points (pp form) is not supported yet.")


@builtin("interp2")
def _interp2(interp, args, nargout):
    from scipy.interpolate import RegularGridInterpolator
    X, Y, Z, Xq, Yq = (fnum(a) for a in args[:5])
    method = text(args[5]).lower() if len(args) > 5 else "linear"
    x = X[0, :] if X.shape[0] > 1 else X.flatten()
    y = Y[:, 0] if Y.shape[1] > 1 else Y.flatten()
    f = RegularGridInterpolator((y, x), Z, method="linear" if method == "linear" else "nearest",
                                bounds_error=False, fill_value=np.nan)
    pts = np.stack([Yq.flatten(), Xq.flatten()], axis=-1)
    return f(pts).reshape(Xq.shape)


# -- root finding and optimization ----------------------------------------------------

@builtin("optimset", "optimoptions")
def _optimset(interp, args, nargout):
    s = Struct()
    start = 1 if args and is_text(args[0]) and len(args) % 2 == 1 else 0
    for k in range(start, len(args) - 1, 2):
        s.set_field(0, text(args[k]), args[k + 1])
    return s


def _opt(options, name, default):
    if isinstance(options, Struct) and name in options.fields:
        v = options.data.flat[0][name]
        if isinstance(v, np.ndarray) and v.size:
            return float(np.real(fnum(v).flat[0]))
    return default


@builtin("fzero")
def _fzero(interp, args, nargout):
    from scipy.optimize import brentq
    f = args[0]
    x0 = fnum(args[1]).flatten()
    opts = args[2] if len(args) > 2 else None
    tol = _opt(opts, "TolX", 2.2e-16)
    g = lambda x: _call_scalar(interp, f, x)  # noqa: E731
    if x0.size == 2:
        a, b = float(x0[0]), float(x0[1])
        fa, fb = g(a), g(b)
        if np.sign(fa) == np.sign(fb) and fa != 0 and fb != 0:
            raise MatlabError("The function values at the interval endpoints must differ in sign.")
    else:
        a = b = float(x0[0])
        fa = fb = g(a)
        if fa == 0:
            return (a, 0.0, 1.0) if nargout > 2 else ((a, 0.0) if nargout == 2 else a)
        dx = abs(a) / 50 if a != 0 else 1 / 50
        found = False
        for _ in range(200):
            a, b = float(x0[0]) - dx, float(x0[0]) + dx
            fa, fb = g(a), g(b)
            if np.isfinite(fa) and np.isfinite(fb) and np.sign(fa) != np.sign(fb):
                found = True
                break
            dx *= np.sqrt(2)
        if not found:
            raise MatlabError("fzero could not find an interval with a sign change around the starting point.")
    if fa == 0:
        x = a
    elif fb == 0:
        x = b
    else:
        x = brentq(g, a, b, xtol=max(tol, 1e-300), rtol=4 * np.finfo(float).eps, maxiter=500)
    if nargout > 1:
        return (x, g(x), 1.0)[:max(nargout, 2)] if nargout <= 3 else (x, g(x), 1.0)
    return x


@builtin("fminsearch")
def _fminsearch(interp, args, nargout):
    from scipy.optimize import minimize
    f = args[0]
    x0 = fnum(args[1])
    opts = args[2] if len(args) > 2 else None
    shape = x0.shape

    def g(x):
        return float(np.real(_call_vector(interp, f, [x.reshape(shape)]).flat[0]))

    res = minimize(g, x0.flatten(), method="Nelder-Mead",
                   options={"xatol": _opt(opts, "TolX", 1e-4), "fatol": _opt(opts, "TolFun", 1e-4),
                            "maxiter": int(_opt(opts, "MaxIter", 200 * x0.size)),
                            "maxfev": int(_opt(opts, "MaxFunEvals", 200 * x0.size))})
    x = res.x.reshape(shape)
    outs = (x, float(res.fun), 1.0 if res.success else 0.0)
    return outs[:nargout] if nargout > 1 else x


@builtin("fminbnd")
def _fminbnd(interp, args, nargout):
    from scipy.optimize import minimize_scalar
    f = args[0]
    a, b = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])
    res = minimize_scalar(lambda x: _call_scalar(interp, f, x), bounds=(a, b), method="bounded",
                          options={"xatol": 1e-4})
    outs = (float(res.x), float(res.fun), 1.0 if res.success else 0.0)
    return outs[:nargout] if nargout > 1 else outs[0]


@builtin("fminunc")
def _fminunc(interp, args, nargout):
    from scipy.optimize import minimize
    f = args[0]
    x0 = fnum(args[1])
    shape = x0.shape
    res = minimize(lambda x: float(np.real(_call_vector(interp, f, [x.reshape(shape)]).flat[0])),
                   x0.flatten(), method="BFGS")
    x = res.x.reshape(shape)
    outs = (x, float(res.fun), 1.0 if res.success else 0.0)
    return outs[:nargout] if nargout > 1 else x


@builtin("fsolve")
def _fsolve(interp, args, nargout):
    from scipy.optimize import root
    f = args[0]
    x0 = fnum(args[1])
    shape = x0.shape
    res = root(lambda x: _call_vector(interp, f, [x.reshape(shape)]).flatten(), x0.flatten(), method="hybr")
    x = res.x.reshape(shape)
    fval = _call_vector(interp, f, [x])
    outs = (x, fval, 1.0 if res.success else 0.0)
    return outs[:nargout] if nargout > 1 else x


# -- integration ----------------------------------------------------------------------------

@builtin("integral", "quad", "quadgk", "quadl")
def _integral(interp, args, nargout):
    from scipy.integrate import quad
    f = args[0]
    a, b = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])
    rest = args[3:]
    opts = {}
    for k in range(0, len(rest) - 1, 2):
        if is_text(rest[k]):
            opts[text(rest[k]).lower()] = float(fnum(rest[k + 1]).flat[0])
    val, _err = quad(lambda x: _call_scalar(interp, f, x), a, b,
                     epsabs=opts.get("abstol", 1e-10), epsrel=opts.get("reltol", 1e-6), limit=200)
    return val


@builtin("integral2")
def _integral2(interp, args, nargout):
    from scipy.integrate import dblquad
    f = args[0]
    xmin, xmax = float(fnum(args[1]).flat[0]), float(fnum(args[2]).flat[0])

    def bound(v):
        if isinstance(v, np.ndarray):
            c = float(fnum(v).flat[0])
            return lambda x: c
        return lambda x: _call_scalar(interp, v, x)

    ymin, ymax = bound(args[3]), bound(args[4])
    val, _ = dblquad(lambda y, x: _call_scalar(interp, f, x, y), xmin, xmax, ymin, ymax)
    return val


# -- ODE solvers ------------------------------------------------------------------------------

def _ode(method):
    def f(interp, args, nargout):
        from scipy.integrate import solve_ivp
        nargs(args, 3, 4, "ode solver")
        fn = args[0]
        tspan = fnum(args[1]).flatten()
        y0 = fnum(args[2]).flatten()
        opts = args[3] if len(args) > 3 else None
        rtol = _opt(opts, "RelTol", 1e-3)
        atol = _opt(opts, "AbsTol", 1e-6)
        is_complex = np.iscomplexobj(y0)

        def rhs(t, y):
            out = _call_vector(interp, fn, [np.array([[t]]), y.reshape(-1, 1)])
            return out.flatten()

        t_eval = tspan if tspan.size > 2 else None
        sol = solve_ivp(rhs, (float(tspan[0]), float(tspan[-1])), y0.astype(complex if is_complex else float),
                        method=method, t_eval=t_eval, rtol=rtol, atol=atol,
                        dense_output=t_eval is None)
        if not sol.success:
            raise MatlabError(f"ODE solver failed: {sol.message}")
        t = sol.t
        y = sol.y
        if t_eval is None:
            # Refine output like MATLAB's default (Refine = 4 for ode45).
            refine = 4 if method == "RK45" else 1
            if refine > 1 and t.size > 1:
                fine = [t[0]]
                for k in range(t.size - 1):
                    fine.extend(np.linspace(t[k], t[k + 1], refine + 1)[1:])
                t = np.array(fine)
                y = sol.sol(t)
        T = t.reshape(-1, 1)
        Y = y.T
        if nargout <= 1:
            return Struct.scalar({"x": t.reshape(1, -1), "y": y, "solver": char(method)})
        return T, Y
    return f


builtin("ode45")(_ode("RK45"))
builtin("ode23")(_ode("RK23"))
builtin("ode113")(_ode("DOP853"))
builtin("ode15s", "ode23s", "ode23t", "ode23tb")(_ode("BDF"))


@builtin("odeset")
def _odeset(interp, args, nargout):
    s = Struct()
    for k in range(0, len(args) - 1, 2):
        s.set_field(0, text(args[k]), args[k + 1])
    return s


@builtin("deval")
def _deval(interp, args, nargout):
    sol, xq = args[0], fnum(args[1])
    t = fnum(sol.get_field(0, "x")).flatten()
    y = fnum(sol.get_field(0, "y"))
    out = np.vstack([np.interp(xq.flatten(), t, row) for row in y])
    return out


