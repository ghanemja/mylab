"""Boundary value problems, delay differential equations, 1-D parabolic PDEs and the ode object."""

from __future__ import annotations

import bisect

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, Struct, char, empty_double, simplify
from ._registry import builtin
from ._util import fnum, is_text, nargs, text
from . import ode_more as om

# -- boundary value problems -------------------------------------------------------------------------


@builtin("bvpinit")
def _bvpinit(interp, args, nargout):
    nargs(args, 2, None, "bvpinit")
    first = args[0]
    extra = []
    if isinstance(first, Struct):
        # extend an existing solution to a new interval
        x = fnum(om.sol_field(first, "x")).flatten()
        y = fnum(om.sol_field(first, "y")).reshape(-1, x.size)
        new = fnum(args[1]).flatten()
        if new.size != 2:
            raise MatlabError("bvpinit: the new interval must be [anew bnew].")
        lo, hi = float(new[0]), float(new[1])
        forward = x[-1] > x[0]
        if (forward and (lo > x[0] or hi < x[-1])) or (not forward and (lo < x[0] or hi > x[-1])):
            raise MatlabError("bvpinit: the new interval must contain the old one.")
        if lo != x[0]:
            y = np.hstack([y[:, :1], y])
            x = np.concatenate([[lo], x])
        if hi != x[-1]:
            y = np.hstack([y, y[:, -1:]])
            x = np.concatenate([x, [hi]])
        fields = {"solver": char("bvpinit"), "x": x.reshape(1, -1), "y": y}
        params = args[2] if len(args) > 2 else om.sol_field(first, "parameters")
        if params is not None and not om._is_empty(params):
            fields["parameters"] = fnum(params).reshape(-1, 1)
        return Struct.scalar(fields)
    x = np.real(fnum(first)).flatten(order="F").astype(float)
    if x.size < 2:
        raise MatlabError("bvpinit: the mesh must have at least two points.")
    d = np.diff(x)
    if not (np.all(d >= 0) or np.all(d <= 0)) or x[0] == x[-1]:
        raise MatlabError("bvpinit: the mesh must be ordered, with a ~= b.")
    yinit = args[1]
    params = args[2] if len(args) > 2 else None
    extra = list(args[3:])
    if om._is_fn(yinit):
        cols = []
        for xv in x:
            out = interp.call_function_value(yinit, [om._scalar(xv), *extra], 1)
            cols.append(fnum(out[0]).flatten(order="F"))
        y = np.column_stack(cols)
    else:
        v = fnum(yinit).flatten(order="F")
        if v.size == 0:
            raise MatlabError("bvpinit: the guess for the solution must not be empty.")
        y = np.repeat(v.reshape(-1, 1), x.size, axis=1)
    fields = {"solver": char("bvpinit"), "x": x.reshape(1, -1), "y": simplify(y)}
    if params is not None and not om._is_empty(params):
        fields["parameters"] = fnum(params).reshape(-1, 1)
    fields["yinit"] = yinit
    return Struct.scalar(fields)


def _bvp(solver: str):
    def f(interp, args, nargout):
        from scipy.integrate import solve_bvp
        nargs(args, 3, None, solver)
        odefun, bcfun, init = args[0], args[1], args[2]
        if not isinstance(init, Struct) or "x" not in init.fields or "y" not in init.fields:
            raise MatlabError(f"{solver}: the third argument must be an initial guess made with bvpinit.")
        opts = om.Opts(args[3] if len(args) > 3 else None, om._BVP_NAMES, solver)
        extra = list(args[4:])
        x = np.real(fnum(om.sol_field(init, "x"))).flatten().astype(float)
        y = fnum(om.sol_field(init, "y"))
        y = y.reshape(-1, x.size) if y.size % x.size == 0 else None
        if y is None or x.size < 2:
            raise MatlabError(f"{solver}: the initial guess must have one column of y per mesh point.")
        if np.any(np.diff(x) == 0):
            raise MatlabError(f"{solver}: multipoint problems (repeated mesh points) are not supported.")
        n = y.shape[0]
        pv = om.sol_field(init, "parameters")
        p0 = None if pv is None or om._is_empty(pv) else np.real(fnum(pv)).flatten().astype(float)
        flip = x[-1] < x[0]
        sgn = -1.0 if flip else 1.0
        xs = sgn * x
        vectorized = opts.onoff("Vectorized")
        counts = {"f": 0, "bc": 0}
        cplx = np.iscomplexobj(y)

        def pargs(p):
            return [] if p0 is None else [np.asarray(p, dtype=float).reshape(-1, 1)]

        def fun(xv, Y, p=None):
            m = xv.size
            out = np.empty((n, m), dtype=Y.dtype if cplx else float)
            if vectorized:
                res = interp.call_function_value(odefun, [(sgn * xv).reshape(1, -1), Y, *pargs(p), *extra], 1)
                v = fnum(res[0])
                if v.shape != (n, m):
                    raise MatlabError(f"{solver}: with Vectorized 'on' the function must return one column "
                                      "per mesh point.")
                counts["f"] += 1
                return sgn * (v if cplx else np.real(v))
            for k in range(m):
                res = interp.call_function_value(odefun, [om._scalar(sgn * xv[k]), Y[:, k].reshape(-1, 1),
                                                          *pargs(p), *extra], 1)
                v = fnum(res[0])
                if v.size != n:
                    raise MatlabError(f"{solver}: the ODE function must return a column vector with {n} "
                                      f"elements (it returned {v.size}).")
                out[:, k] = sgn * (v.flatten(order="F") if cplx else np.real(v.flatten(order="F")))
            counts["f"] += m
            return out

        nbc = n + (0 if p0 is None else p0.size)

        def bc(ya, yb, p=None):
            res = interp.call_function_value(bcfun, [ya.reshape(-1, 1), yb.reshape(-1, 1), *pargs(p), *extra], 1)
            v = fnum(res[0]).flatten(order="F")
            if v.size != nbc:
                raise MatlabError(f"{solver}: the boundary condition function must return {nbc} residuals "
                                  f"(it returned {v.size}).")
            counts["bc"] += 1
            return v if cplx else np.real(v)

        fun_jac = bc_jac = None
        if opts.has("FJacobian"):
            fj = opts.raw("FJacobian")

            def fun_jac(xv, Y, p=None):
                m = xv.size
                dy = np.empty((n, n, m))
                dp = None if p0 is None else np.empty((n, p0.size, m))
                for k in range(m):
                    if om._is_fn(fj):
                        res = interp.call_function_value(fj, [om._scalar(sgn * xv[k]), Y[:, k].reshape(-1, 1),
                                                              *pargs(p), *extra], 1 if p0 is None else 2)
                        dy[:, :, k] = sgn * np.real(fnum(res[0]))
                        if dp is not None:
                            dp[:, :, k] = sgn * np.real(fnum(res[1]))
                    elif isinstance(fj, Cell):
                        mats = fj.values_f()
                        dy[:, :, k] = sgn * np.real(fnum(mats[0]))
                        if dp is not None:
                            dp[:, :, k] = sgn * np.real(fnum(mats[1]))
                    else:
                        dy[:, :, k] = sgn * np.real(fnum(fj))
                        if dp is not None:
                            raise MatlabError(f"{solver}: with unknown parameters FJacobian must also give "
                                              "the derivative with respect to them.")
                return dy if dp is None else (dy, dp)
        if opts.has("BCJacobian"):
            bj = opts.raw("BCJacobian")

            def bc_jac(ya, yb, p=None):
                if om._is_fn(bj):
                    res = interp.call_function_value(bj, [ya.reshape(-1, 1), yb.reshape(-1, 1), *pargs(p), *extra],
                                                     2 if p0 is None else 3)
                    mats = [np.real(fnum(r)) for r in res]
                elif isinstance(bj, Cell):
                    mats = [np.real(fnum(r)) for r in bj.values_f()]
                else:
                    raise MatlabError(f"{solver}: BCJacobian must be a function handle or a cell array.")
                if len(mats) < (2 if p0 is None else 3):
                    raise MatlabError(f"{solver}: BCJacobian must give dbc/dya and dbc/dyb"
                                      + ("" if p0 is None else " and dbc/dp") + ".")
                return tuple(mats[:2]) if p0 is None else tuple(mats[:3])
        S = None
        if opts.has("SingularTerm"):
            S = np.real(fnum(opts.raw("SingularTerm")))
            if S.shape != (n, n):
                raise MatlabError(f"{solver}: SingularTerm must be an {n}-by-{n} matrix.")
            if flip or xs[0] != 0:
                raise MatlabError(f"{solver}: SingularTerm needs the interval to start at 0.")
        rtol = opts.num("RelTol", 1e-3)
        atol = opts.vec("AbsTol")
        nmax = int(opts.num("NMax", max(1000, 10000 // n)))
        y0 = y if cplx else np.real(y).astype(float)
        res = solve_bvp(fun, bc, xs, y0, p=p0, S=S, fun_jac=fun_jac, bc_jac=bc_jac, tol=rtol,
                        max_nodes=nmax, bc_tol=rtol if atol is None else max(rtol, float(np.min(atol))))
        if res.status == 2:
            raise MatlabError(f"{solver}: unable to solve the collocation equations: a singular Jacobian "
                              "was encountered. Try a different initial guess.")
        if res.status == 1:
            interp.warn(f"{solver}: unable to meet the tolerance without using more than {nmax} mesh points.")
        elif res.status == 3:
            interp.warn(f"{solver}: the boundary conditions are not satisfied to the requested tolerance.")
        fields = {"solver": char(solver), "x": (sgn * res.x).reshape(1, -1), "y": simplify(np.array(res.y)),
                  "yp": simplify(sgn * np.array(res.yp))}
        if p0 is not None:
            fields["parameters"] = np.array(res.p).reshape(-1, 1)
        maxres = float(np.max(res.rms_residuals)) if res.rms_residuals.size else 0.0
        fields["stats"] = Struct.scalar({"nmeshpoints": float(res.x.size), "maxres": maxres,
                                         "nODEevals": float(counts["f"]), "nBCevals": float(counts["bc"])})
        if opts.onoff("Stats"):
            interp.out(f"The solution was obtained on a mesh of {res.x.size} points.\n"
                       f"The maximum residual is {maxres:.3e}.\n"
                       f"There were {counts['f']} calls to the ODE function.\n"
                       f"There were {counts['bc']} calls to the BC function.\n")
        return Struct.scalar(fields)

    return f


builtin("bvp4c")(_bvp("bvp4c"))
builtin("bvp5c")(_bvp("bvp5c"))


# -- delay differential equations --------------------------------------------------------------------

class _History:
    """The solution so far: the user's history before t0 and cubic Hermite pieces after it.

    ``yl[k]`` and ``yr[k]`` are the slopes at mesh point k from the left and from the right (they
    differ at tracked discontinuities)."""

    def __init__(self, interp, hist, t0, y0, extra, what):
        self.interp, self.hist, self.t0, self.extra, self.what = interp, hist, t0, extra, what
        self.t = [t0]
        self.y = [y0]
        self.yl: list = [None]
        self.yr: list = [None]
        self.n = y0.size

    def before(self, tq):
        h = self.hist
        while isinstance(h, Struct):
            x = fnum(om.sol_field(h, "x")).flatten()
            if tq >= min(x[0], x[-1]):
                Y, _ = om.eval_solution(h, np.array([min(tq, max(x[0], x[-1]))]), False)
                return Y[:, 0]
            h = om.sol_field(h, "history")
            if h is None:
                raise MatlabError(f"{self.what}: the history solution does not reach back to t = {tq:g}.")
        if om._is_fn(h):
            out = self.interp.call_function_value(h, [om._scalar(tq), *self.extra], 1)
            v = fnum(out[0]).flatten(order="F")
        else:
            v = fnum(h).flatten(order="F")
        if v.size != self.n:
            raise MatlabError(f"{self.what}: the history must give {self.n} values (it gave {v.size}).")
        return om._real_if_possible(v)

    def __call__(self, tq, piece=None):
        """y at time tq; ``piece`` = (t0, t1, y0, yp0, y1, yp1) covers the step in progress."""
        if tq < self.t0:
            return self.before(tq)
        if piece is not None and tq > self.t[-1]:
            return _hermite(*piece, tq)
        if len(self.t) == 1:
            return self.y[0] if self.yr[0] is None else self.y[0] + (tq - self.t0) * self.yr[0]
        k = min(max(bisect.bisect_right(self.t, tq) - 1, 0), len(self.t) - 2)
        return _hermite(self.t[k], self.t[k + 1], self.y[k], self.yr[k], self.y[k + 1], self.yl[k + 1], tq)


def _hermite(t0, t1, y0, d0, y1, d1, tq):
    h = t1 - t0
    s = (tq - t0) / h
    dy = y1 - y0
    return y0 + s * (h * d0 + s * ((3 * dy - h * (2 * d0 + d1)) + s * (-2 * dy + h * (d0 + d1))))


def _hermite_d(t0, t1, y0, d0, y1, d1, tq):
    h = t1 - t0
    s = (tq - t0) / h
    dy = y1 - y0
    return (h * d0 + s * (2 * (3 * dy - h * (2 * d0 + d1)) + 3 * s * (-2 * dy + h * (d0 + d1)))) / h


def _dde(solver: str):
    def f(interp, args, nargout):
        from scipy.integrate import RK23
        from . import _ode_steppers as st
        nargs(args, 4, None, solver)
        ddefun, lag_arg, hist_arg = args[0], args[1], args[2]
        tspan = om.check_tspan(args[3], solver)
        opts = om.Opts(args[4] if len(args) > 4 else None, om._DDE_NAMES, solver)
        extra = list(args[5:])
        t0, tf = float(tspan[0]), float(tspan[-1])
        if tf <= t0:
            raise MatlabError(f"{solver}: tspan must be increasing, [t0 tfinal].")
        # constant lags or delay functions
        delay_fns = None
        lags = np.zeros(0)
        if solver == "dde23" or not (om._is_fn(lag_arg) or isinstance(lag_arg, Cell)):
            lags = np.real(fnum(lag_arg)).flatten().astype(float) if not om._is_empty(lag_arg) else np.zeros(0)
            if np.any(lags <= 0) and solver == "dde23":
                raise MatlabError("dde23: the lags must be positive.")
        else:
            delay_fns = lag_arg.values_f() if isinstance(lag_arg, Cell) else [lag_arg]
        # initial value and history
        if isinstance(hist_arg, Struct):
            hx = fnum(om.sol_field(hist_arg, "x")).flatten()
            if abs(hx[-1] - t0) > 1e-12 * max(1.0, abs(t0)):
                raise MatlabError(f"{solver}: the history solution must end at tspan(1).")
            y0 = fnum(om.sol_field(hist_arg, "y")).reshape(-1, hx.size)[:, -1].copy()
        elif om._is_fn(hist_arg):
            out = interp.call_function_value(hist_arg, [om._scalar(t0), *extra], 1)
            y0 = fnum(out[0]).flatten(order="F")
        else:
            y0 = fnum(hist_arg).flatten(order="F")
        if opts.has("InitialY"):
            iy = opts.vec("InitialY")
            if iy.size != y0.size:
                raise MatlabError(f"{solver}: InitialY must have {y0.size} elements.")
            y0 = iy
        y0 = om._real_if_possible(np.asarray(y0)).astype(complex if np.iscomplexobj(y0) and np.any(np.imag(y0))
                                                         else float)
        n = y0.size
        if n == 0:
            raise MatlabError(f"{solver}: the history must not be empty.")
        H = _History(interp, hist_arg, t0, y0, extra, solver)
        nlag = lags.size
        rtol = opts.num("RelTol", 1e-3)
        atol = opts.vec("AbsTol")
        atol = np.array([1e-6]) if atol is None else np.real(atol)
        if atol.size not in (1, n):
            raise MatlabError(f"{solver}: AbsTol must be a scalar or have one entry per component.")
        hmax = opts.num("MaxStep", 0.1 * (tf - t0))
        ctl = st.Control(rtol, atol, opts.onoff("NormControl"), hmax, opts.num("InitialStep"), t0, tf)
        cnt = st.Counters()

        def delayed_times(t, y):
            if delay_fns is None:
                return t - lags
            out = []
            for fn in delay_fns:
                r = interp.call_function_value(fn, [om._scalar(t), om._col(y), *extra], 1)
                out.extend(np.real(fnum(r[0])).flatten(order="F").tolist())
            d = np.array(out, dtype=float)
            return np.minimum(d, t)

        def Zmat(t, y, piece):
            td = delayed_times(t, y)
            Z = np.empty((n, td.size), dtype=y0.dtype)
            for j, tq in enumerate(td):
                Z[:, j] = H(tq, piece)
            return Z

        def rhs(t, y, piece=None):
            Z = Zmat(t, y, piece)
            out = interp.call_function_value(ddefun, [om._scalar(t), om._col(y), Z, *extra], 1)
            v = fnum(out[0])
            if v.size != n:
                raise MatlabError(f"{solver}: the function must return a column vector with {n} elements "
                                  f"(it returned {v.size}).")
            cnt.nfevals += 1
            return om._real_if_possible(v.flatten(order="F"))

        # discontinuities propagated by constant lags (and user jumps)
        disc: list = []
        if delay_fns is None and nlag:
            starts = [t0]
            if opts.has("Jumps"):
                starts += [float(v) for v in np.real(opts.vec("Jumps"))]
            level = sorted(set(starts))
            pts = set(v for v in level if t0 < v < tf)
            for _ in range(4):
                nxt = set()
                for v in level:
                    for lag in lags:
                        w = v + lag
                        if t0 < w < tf:
                            nxt.add(w)
                level = sorted(nxt)
                pts |= nxt
                if len(pts) > 4000:
                    break
            disc = sorted(pts)
            # merge points that coincide up to round-off
            merged = []
            for v in disc:
                if not merged or v - merged[-1] > 1e-10 * max(1.0, abs(v)):
                    merged.append(v)
            disc = merged
        elif opts.has("Jumps"):
            disc = sorted(float(v) for v in np.real(opts.vec("Jumps")) if t0 < v < tf)
        min_lag = float(np.min(lags)) if (delay_fns is None and nlag) else np.inf
        # events
        ev = None
        if opts.has("Events"):
            ev_fn = opts.raw("Events")

            def ev(t, y, piece=None):
                out = interp.call_function_value(ev_fn, [om._scalar(t), om._col(y), Zmat(t, y, piece), *extra], 3)
                if len(out) < 3:
                    raise MatlabError(f"{solver}: the event function must return [value, isterminal, direction].")
                v = np.real(fnum(out[0])).flatten()
                term = np.real(fnum(out[1])).flatten() if not om._is_empty(out[1]) else np.zeros(v.size)
                dirn = np.real(fnum(out[2])).flatten() if not om._is_empty(out[2]) else np.zeros(v.size)
                return v, np.broadcast_to(term, v.shape) != 0, np.broadcast_to(dirn, v.shape)
        out_fn = None
        if opts.has("OutputFcn"):
            sel = opts.vec("OutputSel")
            sel = None if sel is None else np.real(sel).astype(int) - 1
            out_fn = om._output_caller(interp, opts.raw("OutputFcn"), sel, extra)
        refine = int(opts.num("Refine", 1))
        A, B, C, E = (np.asarray(RK23.A), np.asarray(RK23.B), np.asarray(RK23.C), np.asarray(RK23.E))
        t, y = t0, y0
        f0 = rhs(t, y)
        H.yl[0] = H.yr[0] = f0
        absh = ctl.first_step(t, y, f0, 1.0 / 3)
        te, ye, ie = [], [], []
        v_old = ev(t, y)[0] if ev is not None else None
        if out_fn is not None:
            out_fn(np.array([t0, tf]), y.reshape(-1, 1), "init")
        di = 0
        message = None
        stop = False
        implicit_iter = delay_fns is not None or not nlag
        while t < tf and not stop:
            hmin = ctl.hmin(t)
            while di < len(disc) and disc[di] <= t + 1e-12 * max(1.0, abs(t)):
                di += 1
            limit = tf if di >= len(disc) else disc[di]
            absh = min(hmax, max(hmin, absh))
            if np.isfinite(min_lag):
                absh = min(absh, min_lag)
            nofailed = True
            while True:
                h = absh
                hit = False
                if 1.1 * h >= limit - t:
                    h = limit - t
                    hit = True
                tnew = limit if hit else t + h
                piece = None
                passes = 3 if implicit_iter else 1
                for _p in range(passes):
                    K = np.empty((4, n), dtype=np.result_type(y.dtype, f0.dtype))
                    K[0] = f0
                    for s in range(1, 3):
                        K[s] = rhs(t + C[s] * h, y + h * (A[s, :s] @ K[:s]), piece)
                    ynew = y + h * (B @ K[:3])
                    if implicit_iter:
                        # the slope at the new point may need the solution inside this very step
                        guess = (t, tnew, y, f0, ynew, K[2] if piece is None else piece[5])
                        K[3] = rhs(tnew, ynew, guess)
                    else:
                        K[3] = rhs(tnew, ynew, None)
                    piece = (t, tnew, y, f0, ynew, K[3])
                err = h * ctl.err_norm(K.T @ E, y, ynew) if np.all(np.isfinite(ynew)) else np.inf
                if err > rtol:
                    cnt.nfailed += 1
                    if h <= hmin:
                        message = (f"Failure at t={t:.6e}. Unable to meet integration tolerances without "
                                   f"reducing the step size below the smallest value allowed ({hmin:.6e}).")
                        break
                    if nofailed and np.isfinite(err):
                        absh = max(hmin, h * max(0.1, 0.8 * (rtol / err) ** (1.0 / 3)))
                    else:
                        absh = max(hmin, 0.5 * h)
                    nofailed = False
                    continue
                break
            if message:
                break
            cnt.nsteps += 1
            fnew = K[3]
            if ev is not None:
                v_new, terminal, direction = ev(tnew, ynew)
                hits = []
                for k in range(v_new.size):
                    a, b = v_old[k], v_new[k]
                    if a == 0 or np.sign(a) == np.sign(b):
                        continue
                    if (direction[k] > 0 and not b > a) or (direction[k] < 0 and not b < a):
                        continue
                    cache = {0.0: a, 1.0: b}

                    def g(s, k=k, cache=cache):
                        if s not in cache:
                            tq = t + s * (tnew - t)
                            cache[s] = ev(tq, _hermite(t, tnew, y, f0, ynew, fnew, tq), piece)[0][k]
                        return cache[s]

                    from ._ode_driver import _find_root
                    hits.append((_find_root(g, a, b, 128 * st.EPS * max(abs(t), abs(tnew)) / (tnew - t)), k))
                hits.sort()
                for s_e, k in hits:
                    tq = tnew if s_e >= 1 else t + s_e * (tnew - t)
                    yq = ynew if s_e >= 1 else _hermite(t, tnew, y, f0, ynew, fnew, tq)
                    te.append(tq)
                    ye.append(np.array(yq, copy=True))
                    ie.append(k + 1)
                    if terminal[k]:
                        stop = True
                        if s_e < 1:
                            fq = _hermite_d(t, tnew, y, f0, ynew, fnew, tq)
                            tnew, ynew, fnew = tq, np.array(yq, copy=True), fq
                        break
                v_old = v_new
            if out_fn is not None:
                tt = [t + (tnew - t) * i / refine for i in range(1, refine)] + [tnew]
                yy = [_hermite(t, tnew, y, f0, ynew, fnew, q) for q in tt[:-1]] + [ynew]
                if out_fn(np.array(tt), np.column_stack(yy), ""):
                    stop = True
            H.t.append(tnew)
            H.y.append(ynew)
            H.yl.append(fnew)
            at_disc = hit and limit < tf and not stop
            t, y = tnew, ynew
            # at a tracked discontinuity the slope jumps: evaluate it afresh from the right
            f0 = rhs(t, y) if at_disc else fnew
            H.yr.append(f0)
            if nofailed:
                temp = 1.25 * (err / rtol) ** (1.0 / 3)
                absh = h / temp if temp > 0.2 else 5.0 * h
            else:
                absh = h
        if out_fn is not None:
            out_fn(np.zeros(0), np.zeros((n, 0)), "done")
        if message:
            interp.warn(message)
        fields = {"solver": char(solver), "history": hist_arg,
                  "discont": np.array([t0] + disc, dtype=float).reshape(1, -1),
                  "x": np.array(H.t, dtype=float).reshape(1, -1), "y": simplify(np.column_stack(H.y)),
                  "stats": Struct.scalar({"nsteps": float(cnt.nsteps), "nfailed": float(cnt.nfailed),
                                          "nfevals": float(cnt.nfevals)}),
                  "yp": simplify(np.column_stack(H.yl))}
        if ev is not None:
            fields["xe"] = np.array(te, dtype=float).reshape(1, -1)
            fields["ye"] = simplify(np.column_stack(ye)) if ye else np.zeros((n, 0))
            fields["ie"] = np.array(ie, dtype=float).reshape(1, -1)
        if opts.onoff("Stats"):
            interp.out(f"{cnt.nsteps} successful steps\n{cnt.nfailed} failed attempts\n"
                       f"{cnt.nfevals} function evaluations\n")
        return Struct.scalar(fields)

    return f


builtin("dde23")(_dde("dde23"))
builtin("ddesd")(_dde("ddesd"))


# -- 1-D parabolic and elliptic PDEs -------------------------------------------------------------------

@builtin("pdepe")
def _pdepe(interp, args, nargout):
    from ._ode_driver import IVP, integrate
    nargs(args, 6, None, "pdepe")
    m = float(np.real(fnum(args[0]).flat[0]))
    if m not in (0.0, 1.0, 2.0):
        raise MatlabError("pdepe: the symmetry constant m must be 0 (slab), 1 (cylindrical) or 2 (spherical).")
    m = int(m)
    pdefun, icfun, bcfun = args[1], args[2], args[3]
    x = np.real(fnum(args[4])).flatten().astype(float)
    tspan = np.real(fnum(args[5])).flatten().astype(float)
    if x.size < 3:
        raise MatlabError("pdepe: xmesh must have at least 3 points.")
    if np.any(np.diff(x) <= 0):
        raise MatlabError("pdepe: xmesh must be strictly increasing.")
    if m > 0 and x[0] < 0:
        raise MatlabError("pdepe: for m > 0 the mesh must satisfy xmesh(1) >= 0.")
    if tspan.size < 3:
        raise MatlabError("pdepe: tspan must have at least 3 points.")
    if np.any(np.diff(tspan) <= 0):
        raise MatlabError("pdepe: tspan must be strictly increasing.")
    opts = om.Opts(args[6] if len(args) > 6 else None, om._ODE_NAMES, "pdepe")
    if opts.has("Events"):
        raise MatlabError("pdepe: the Events option is not supported.")
    extra = list(args[7:])
    nx = x.size
    # initial values
    cols = []
    for xv in x:
        out = interp.call_function_value(icfun, [om._scalar(xv), *extra], 1)
        cols.append(np.real(fnum(out[0])).flatten(order="F"))
    U0 = np.column_stack(cols)          # npde x nx
    npde = U0.shape[0]
    xi = 0.5 * (x[:-1] + x[1:])
    hx = np.diff(x)
    xm = xi ** m
    wL = (x[1:] ** (m + 1) - xi ** (m + 1)) / (m + 1)    # right half of interval i belongs to node i+1
    wR = (xi ** (m + 1) - x[:-1] ** (m + 1)) / (m + 1)   # left half of interval i belongs to node i
    singular = m > 0 and x[0] == 0

    def coefficients(t, U):
        """c, f, s at the interval midpoints (each npde x (nx-1))."""
        c = np.empty((npde, nx - 1))
        f = np.empty((npde, nx - 1))
        s = np.empty((npde, nx - 1))
        tv = om._scalar(t)
        um = 0.5 * (U[:, :-1] + U[:, 1:])
        ux = (U[:, 1:] - U[:, :-1]) / hx
        for i in range(nx - 1):
            out = interp.call_function_value(pdefun, [om._scalar(xi[i]), tv, um[:, i].reshape(-1, 1),
                                                      ux[:, i].reshape(-1, 1), *extra], 3)
            if len(out) < 3:
                raise MatlabError("pdepe: the PDE function must return [c, f, s].")
            for arr, v in zip((c, f, s), out):
                a = np.real(fnum(v)).flatten(order="F")
                if a.size != npde:
                    raise MatlabError(f"pdepe: c, f and s must each have {npde} elements.")
                arr[:, i] = a
        return c, f, s

    def boundary(t, U):
        out = interp.call_function_value(bcfun, [om._scalar(x[0]), U[:, 0].reshape(-1, 1), om._scalar(x[-1]),
                                                 U[:, -1].reshape(-1, 1), om._scalar(t), *extra], 4)
        if len(out) < 4:
            raise MatlabError("pdepe: the boundary function must return [pl, ql, pr, qr].")
        vals = [np.real(fnum(v)).flatten(order="F") for v in out]
        for v in vals:
            if v.size != npde:
                raise MatlabError(f"pdepe: pl, ql, pr and qr must each have {npde} elements.")
        return vals

    t0 = float(tspan[0])
    c0, _, _ = coefficients(t0, U0)
    _, ql0, _, qr0 = boundary(t0, U0)
    # rows that are differential equations (the others are algebraic conditions)
    diff = np.ones((npde, nx), dtype=bool)
    Cnode0 = np.zeros((npde, nx))
    Cnode0[:, :-1] += c0 * wR
    Cnode0[:, 1:] += c0 * wL
    diff &= Cnode0 != 0
    if not singular:
        diff[:, 0] &= ql0 != 0
    diff[:, -1] &= qr0 != 0
    if not diff.any():
        raise MatlabError("pdepe: there must be at least one parabolic equation.")

    def rhs(t, yv):
        U = yv.reshape(nx, npde).T
        c, f, s = coefficients(t, U)
        pl, ql, pr, qr = boundary(t, U)
        F = np.zeros((npde, nx))
        Cn = np.zeros((npde, nx))
        flux = f * xm
        F[:, :-1] += flux + s * wR
        F[:, 1:] += -flux + s * wL
        Cn[:, :-1] += c * wR
        Cn[:, 1:] += c * wL
        if not singular:
            with np.errstate(divide="ignore", invalid="ignore"):
                fl = np.where(ql != 0, -pl / np.where(ql != 0, ql, 1.0), 0.0)
            F[:, 0] -= (x[0] ** m) * fl
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = np.where(qr != 0, -pr / np.where(qr != 0, qr, 1.0), 0.0)
        F[:, -1] += (x[-1] ** m) * fr
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(diff, F / np.where(Cn != 0, Cn, 1.0), F)
        if not singular:
            out[:, 0] = np.where(ql0 != 0, out[:, 0], pl)
        out[:, -1] = np.where(qr0 != 0, out[:, -1], pr)
        return out.T.reshape(-1)

    y0 = U0.T.reshape(-1).astype(float)
    P = IVP("ode15s", rhs, tspan, y0)
    P.rtol = opts.num("RelTol", 1e-3)
    atol = opts.vec("AbsTol")
    P.atol = np.array([1e-6]) if atol is None else np.real(atol)
    if P.atol.size not in (1, y0.size):
        if P.atol.size == npde:
            P.atol = np.tile(P.atol, nx)
        else:
            raise MatlabError("pdepe: AbsTol must be a scalar or have one entry per PDE.")
    P.normcontrol = opts.onoff("NormControl")
    P.hmax = opts.num("MaxStep")
    P.h0 = opts.num("InitialStep")
    mask = diff.T.reshape(-1)
    if not mask.all():
        P.mass = ("const", np.diag(mask.astype(float)))
        P.mass_singular = "yes"
    # block tridiagonal sparsity pattern
    blocks = np.zeros((nx, nx), dtype=bool)
    idx = np.arange(nx)
    blocks[idx, idx] = True
    blocks[idx[:-1], idx[:-1] + 1] = True
    blocks[idx[1:], idx[1:] - 1] = True
    P.jpattern = np.kron(blocks, np.ones((npde, npde), dtype=bool))
    P.warn = interp.warn
    res = integrate(P, want_dense=False)
    if res.message:
        interp.warn("pdepe: time integration has failed. " + res.message)
    nt = len(res.tout)
    sol = np.full((tspan.size, nx, npde), np.nan)
    for i in range(nt):
        sol[i] = np.real(res.yout[i]).reshape(nx, npde)
    sol = sol[:nt] if nt < tspan.size else sol
    if opts.onoff("Stats"):
        om._print_stats(interp, res.counters, True)
    if npde == 1:
        sol = sol[:, :, 0]
    if nargout > 1:
        return sol, np.array(res.tout).reshape(-1, 1)
    return sol


@builtin("pdeval")
def _pdeval(interp, args, nargout):
    nargs(args, 4, 4, "pdeval")
    x = np.real(fnum(args[1])).flatten().astype(float)
    u = np.real(fnum(args[2])).flatten().astype(float)
    xq_arg = fnum(args[3])
    xq = np.real(xq_arg).flatten().astype(float)
    if u.size != x.size:
        raise MatlabError("pdeval: ui must have one value per mesh point.")
    if xq.size and (xq.min() < x[0] or xq.max() > x[-1]):
        raise MatlabError("pdeval: the evaluation points must lie inside the mesh.")
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    slope = (u[k + 1] - u[k]) / (x[k + 1] - x[k])
    uq = u[k] + slope * (xq - x[k])
    shape = (1, -1)
    if nargout > 1:
        return uq.reshape(shape), slope.reshape(shape)
    return uq.reshape(shape)


# -- the ode object --------------------------------------------------------------------------------------

def _pairs(args: list, what: str) -> list:
    if len(args) % 2:
        raise MatlabError(f"{what}: arguments must be Name=Value pairs.")
    out = []
    for k in range(0, len(args), 2):
        if not is_text(args[k]):
            raise MatlabError(f"{what}: expected a property name at argument {k + 1}.")
        out.append((text(args[k]), args[k + 1]))
    return out


class _Props(MObject):
    """A small value object with a fixed set of named properties."""

    NAME = "object"
    PROPS: dict = {}
    READONLY: tuple = ()

    def __init__(self, pairs=()):
        self.p = {k: (v() if callable(v) else v) for k, v in self.PROPS.items()}
        for name, value in pairs:
            self.p[self._name(name)] = self._check(self._name(name), value)

    def _name(self, name: str) -> str:
        for k in self.PROPS:
            if k.lower() == name.lower():
                return k
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.NAME}'.")

    def _check(self, name, value):
        return value

    def class_name(self) -> str:
        return self.NAME

    def get_field(self, interp, name):
        if self.has_method(name):
            return super().get_field(interp, name)
        return self.p[self._name(name)]

    def set_field(self, interp, name, value):
        key = self._name(name)
        if key in self.READONLY:
            raise MatlabError(f"Cannot set read-only property '{key}' of class '{self.NAME}'.")
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.p = dict(self.p)
        new.p[key] = new._check(key, value)
        return new

    def display_lines(self):
        from ..display import format_display
        lines = [f"  {self.NAME} with properties:", ""]
        width = max(len(k) for k in self.PROPS)
        for k in self.PROPS:
            v = self.p[k]
            if isinstance(v, MObject):
                shown = f"[1x1 {v.class_name()}]"
            else:
                shown = format_display("x", v).split("=", 1)[-1].strip().replace("\n", " ")
                if len(shown) > 60:
                    shown = shown[:57] + "..."
            lines.append(f"    {k:>{width}}: {shown}")
        return lines


def _word(value, allowed, what):
    if not is_text(value):
        raise MatlabError(f"{what} must be one of: {', '.join(allowed)}.")
    w = text(value)
    for a in allowed:
        if a.lower() == w.lower():
            return MString(a)
    raise MatlabError(f"{what} must be one of: {', '.join(allowed)}.")


class OdeEvent(_Props):
    NAME = "odeEvent"
    PROPS = {"EventFcn": empty_double, "Direction": lambda: MString("both"),
             "Response": lambda: MString("proceed"), "CallbackFcn": empty_double}

    def _check(self, name, value):
        if name == "Direction":
            return _word(value, ("both", "ascending", "descending"), "Direction")
        if name == "Response":
            return _word(value, ("proceed", "stop", "callback"), "Response")
        return value


class OdeMassMatrix(_Props):
    NAME = "odeMassMatrix"
    PROPS = {"MassMatrix": empty_double, "Singular": lambda: MString("maybe"),
             "StateDependence": lambda: MString("weak"), "SparsityPattern": empty_double}

    def _check(self, name, value):
        if name == "Singular":
            return _word(value, ("yes", "no", "maybe"), "Singular")
        if name == "StateDependence":
            return _word(value, ("none", "weak", "strong"), "StateDependence")
        return value


class OdeJacobian(_Props):
    NAME = "odeJacobian"
    PROPS = {"Jacobian": empty_double, "SparsityPattern": empty_double}


@builtin("odeEvent")
def _ode_event(interp, args, nargout):
    return OdeEvent(_pairs(args, "odeEvent"))


@builtin("odeMassMatrix")
def _ode_mass(interp, args, nargout):
    return OdeMassMatrix(_pairs(args, "odeMassMatrix"))


@builtin("odeJacobian")
def _ode_jacobian(interp, args, nargout):
    return OdeJacobian(_pairs(args, "odeJacobian"))


_SOLVER_WORDS = ("auto", "stiff", "nonstiff", "ode45", "ode23", "ode113", "ode78", "ode89", "ode15s", "ode23s",
                 "ode23t", "ode23tb")


class OdeResults(_Props):
    NAME = "ODEResults"
    PROPS = {"Time": empty_double, "Solution": empty_double, "EventTime": empty_double,
             "EventSolution": empty_double, "EventIndex": empty_double}
    READONLY = tuple(PROPS)


class OdeObject(_Props):
    NAME = "ode"
    PROPS = {"ODEFcn": empty_double, "Parameters": empty_double, "InitialTime": lambda: np.array([[0.0]]),
             "InitialValue": empty_double, "Jacobian": empty_double, "MassMatrix": empty_double,
             "EquationType": lambda: MString("standard"), "EventDefinition": empty_double,
             "InitialSlope": empty_double, "NonNegativeVariables": empty_double,
             "RelativeTolerance": lambda: np.array([[1e-3]]), "AbsoluteTolerance": lambda: np.array([[1e-6]]),
             "Solver": lambda: MString("auto"), "SelectedSolver": lambda: MString("ode45"),
             "SolverOptions": lambda: Struct.scalar({})}

    def _check(self, name, value):
        if name == "Solver":
            return _word(value, _SOLVER_WORDS, "Solver")
        if name == "EquationType":
            w = _word(value, ("standard", "fullyimplicit"), "EquationType")
            if w == "fullyimplicit":
                raise MatlabError("ode: EquationType 'fullyimplicit' is not supported; call ode15i directly.")
            return w
        if name == "SelectedSolver":
            raise MatlabError("Cannot set read-only property 'SelectedSolver' of class 'ode'.")
        if name in ("RelativeTolerance",):
            v = fnum(value)
            if v.size != 1 or not np.real(v.flat[0]) > 0:
                raise MatlabError("RelativeTolerance must be a positive scalar.")
        if name == "SolverOptions" and not isinstance(value, Struct):
            raise MatlabError("SolverOptions must be a structure of solver options.")
        return value

    def get_field(self, interp, name):
        if not self.has_method(name) and self._name(name) == "SelectedSolver":
            return MString(self.selected())
        return super().get_field(interp, name)

    def display_lines(self):
        self.p["SelectedSolver"] = MString(self.selected())
        return super().display_lines()

    def selected(self) -> str:
        s = str(self.p["Solver"])
        if s in ("auto", "nonstiff", "stiff"):
            stiffish = not om._is_empty(self.p["MassMatrix"]) or not om._is_empty(self.p["Jacobian"])
            if s == "stiff" or (s == "auto" and stiffish):
                return "ode15s"
            return "ode45"
        return s

    _METHODS = ("solve", "solutionFcn")

    def has_method(self, name):
        return name in self._METHODS

    def call_method(self, interp, name, args, nargout):
        objs = [a for a in args if isinstance(a, OdeObject)]
        if not objs or args[0] is not objs[0]:
            raise MatlabError(f"{name}: the first argument must be an ode object.")
        rest = list(args[1:])
        if name == "solve":
            return [self.solve(interp, rest)]
        if name == "solutionFcn":
            fh, S = self.solution_fcn(interp, rest)
            return [fh, S][:max(nargout, 1)]
        return super().call_method(interp, name, args, nargout)

    # -- running the solver ---------------------------------------------------------------------
    def _options(self, interp) -> tuple:
        """(options struct for the functional solvers, callback info)."""
        p = self.p
        d: dict = {"RelTol": p["RelativeTolerance"], "AbsTol": p["AbsoluteTolerance"]}
        so = p["SolverOptions"]
        if isinstance(so, Struct) and so.numel() == 1:
            for k, v in so.data.flat[0].items():
                if not om._is_empty(v):
                    d[om.match_option(k, om._ODE_NAMES, "ode SolverOptions")] = v
        if not om._is_empty(p["NonNegativeVariables"]):
            d["NonNegative"] = p["NonNegativeVariables"]
        if not om._is_empty(p["InitialSlope"]):
            d["InitialSlope"] = p["InitialSlope"]
        mm = p["MassMatrix"]
        if isinstance(mm, OdeMassMatrix):
            d["Mass"] = mm.p["MassMatrix"]
            d["MassSingular"] = char(str(mm.p["Singular"]))
            d["MStateDependence"] = char(str(mm.p["StateDependence"]))
            if not om._is_empty(mm.p["SparsityPattern"]):
                d["MvPattern"] = mm.p["SparsityPattern"]
        elif not om._is_empty(mm):
            d["Mass"] = mm
        jj = p["Jacobian"]
        if isinstance(jj, OdeJacobian):
            if not om._is_empty(jj.p["Jacobian"]):
                d["Jacobian"] = jj.p["Jacobian"]
            if not om._is_empty(jj.p["SparsityPattern"]):
                d["JPattern"] = jj.p["SparsityPattern"]
        elif not om._is_empty(jj):
            d["Jacobian"] = jj
        return d

    def _run(self, interp, tspan: np.ndarray, want_sol: bool, refine=None, y_start=None):
        """Integrate over tspan (restarting after event callbacks); returns (pieces, events, solver, n)."""
        p = self.p
        fn = p["ODEFcn"]
        if not om._is_fn(fn):
            raise MatlabError("ode: set ODEFcn to a function handle before solving.")
        if om._is_empty(p["InitialValue"]):
            raise MatlabError("ode: set InitialValue before solving.")
        y0 = fnum(p["InitialValue"]).flatten(order="F") if y_start is None else np.asarray(y_start)
        y0 = y0.astype(complex if np.iscomplexobj(y0) and np.any(np.imag(y0)) else float)
        solver = self.selected()
        extra = [] if om._is_empty(p["Parameters"]) else [p["Parameters"]]
        d = self._options(interp)
        if refine is not None:
            d["Refine"] = np.array([[float(refine)]])
        evdef = p["EventDefinition"]
        response = "proceed"
        direction = 0.0
        callback = None
        ev_fn = None
        n = y0.size
        if isinstance(evdef, OdeEvent) and om._is_fn(evdef.p["EventFcn"]):
            response = str(evdef.p["Response"])
            direction = {"both": 0.0, "ascending": 1.0, "descending": -1.0}[str(evdef.p["Direction"])]
            callback = evdef.p["CallbackFcn"]
            ev_fn = evdef.p["EventFcn"]
            if response == "callback" and not om._is_fn(callback):
                raise MatlabError("ode: an event with Response 'callback' needs a CallbackFcn.")
        elif not om._is_empty(evdef):
            raise MatlabError("ode: EventDefinition must be an odeEvent object with an EventFcn.")
        from ._ode_driver import integrate
        pieces = []
        events: tuple = ([], [], [])
        span = np.asarray(tspan, dtype=float)
        tdir = np.sign(span[-1] - span[0])
        while True:
            opts = om.Opts(Struct.scalar(d), om._ODE_NAMES, "ode")
            P = om.make_ivp(interp, solver, fn, span, y0, opts, extra)
            if ev_fn is not None:
                def ev(t, y, yp):
                    out = interp.call_function_value(ev_fn, [om._scalar(t), om._col(y), *extra], 1)
                    v = np.real(fnum(out[0])).flatten(order="F").astype(float)
                    return v, np.full(v.size, response != "proceed"), np.full(v.size, direction)
                P.events = ev
            res = integrate(P, want_dense=want_sol)
            res.has_events = ev_fn is not None
            if res.message:
                interp.warn(res.message)
            pieces.append(res)
            for k, lst in enumerate((res.te, res.ye, res.ie)):
                events[k].extend(lst)
            if not res.stopped_by_event or response != "callback":
                break
            te, ye = res.x[-1], res.y[-1]
            out = interp.call_function_value(callback, [om._scalar(te), om._col(ye), *extra], 2)
            stop = bool(np.any(fnum(out[0]) != 0)) if out else False
            if len(out) > 1 and not om._is_empty(out[1]):
                ye = fnum(out[1]).flatten(order="F")
                if ye.size != n:
                    raise MatlabError(f"ode: the event callback must return a state with {n} elements.")
            rest = span[(span - te) * tdir > 0]
            if stop or rest.size == 0:
                break
            span = np.concatenate([[te], rest])
            y0 = np.asarray(ye, dtype=y0.dtype).copy()
        return pieces, events, solver, n

    def _span(self, rest: list, what: str):
        refine = None
        nums = []
        k = 0
        while k < len(rest):
            if is_text(rest[k]):
                key = text(rest[k]).lower()
                if key != "refine" or k + 1 >= len(rest):
                    raise MatlabError(f"{what}: unknown option '{text(rest[k])}'.")
                refine = int(fnum(rest[k + 1]).flat[0])
                k += 2
                continue
            nums.append(rest[k])
            k += 1
        t_init = float(fnum(self.p["InitialTime"]).flat[0])
        if len(nums) == 2:
            a, b = (float(np.real(fnum(v).flat[0])) for v in nums)
            if a == b:
                raise MatlabError(f"{what}: t0 and tf must differ.")
            return a, b, None, t_init, refine
        if len(nums) == 1:
            tv = np.real(fnum(nums[0])).flatten().astype(float)
            if tv.size == 0:
                raise MatlabError(f"{what}: the vector of times must not be empty.")
            return float(tv[0]), float(tv[-1]), tv, t_init, refine
        raise MatlabError(f"{what}: give the interval as (F, t0, tf) or the output times as (F, t).")

    def _steps(self, interp, t_from, t_to, y_start, refine):
        """All solver steps from t_from to t_to: (t, y, events)."""
        pieces, events, _, _ = self._run(interp, np.array([t_from, t_to]), False, refine, y_start)
        t = np.concatenate([np.array(r.tout if i == 0 else r.tout[1:], dtype=float)
                            for i, r in enumerate(pieces)])
        y = np.column_stack([v for i, r in enumerate(pieces) for v in (r.yout if i == 0 else r.yout[1:])])
        return t, y, events

    def _at(self, interp, t_from, times, y_start=None):
        """Solution at the given times (monotone, moving away from t_from): (y, events)."""
        times = np.asarray(times, dtype=float)
        span = np.concatenate([[t_from], times])
        pad = span.size == 2
        if pad:
            span = np.array([t_from, 0.5 * (t_from + times[0]), times[0]])
        pieces, events, _, n = self._run(interp, span, False, None, y_start)
        tt = np.concatenate([np.array(r.tout if i == 0 else r.tout[1:], dtype=float)
                             for i, r in enumerate(pieces)])
        yy = np.column_stack([v for i, r in enumerate(pieces) for v in (r.yout if i == 0 else r.yout[1:])])
        cols = []
        for tq in times:
            hit = np.nonzero(tt == tq)[0]
            if hit.size == 0:
                break  # the integration stopped early (terminal event or failure)
            cols.append(yy[:, hit[-1]])
        y = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=yy.dtype)
        return y, events

    def solve(self, interp, rest: list):
        a, b, tv, t_init, refine = self._span(rest, "solve")
        ev: tuple = ([], [], [])

        def add(events):
            for k in range(3):
                ev[k].extend(events[k])

        if tv is None:
            lo, hi = min(a, b), max(a, b)
            if t_init <= lo:
                y_lo = None
                if t_init < lo:
                    y_lo, e = self._at(interp, t_init, [lo])
                    if y_lo.shape[1] == 0:
                        raise MatlabError("solve: the integration stopped before reaching the interval.")
                    y_lo = y_lo[:, 0]
                t, y, e = self._steps(interp, lo, hi, y_lo, refine)
                add(e)
            elif t_init >= hi:
                y_hi = None
                if t_init > hi:
                    y_hi, e = self._at(interp, t_init, [hi])
                    if y_hi.shape[1] == 0:
                        raise MatlabError("solve: the integration stopped before reaching the interval.")
                    y_hi = y_hi[:, 0]
                t, y, e = self._steps(interp, hi, lo, y_hi, refine)
                add(e)
                t, y = t[::-1], y[:, ::-1]
            else:
                t1, y1, e1 = self._steps(interp, t_init, lo, None, refine)
                t2, y2, e2 = self._steps(interp, t_init, hi, None, refine)
                add(e1)
                add(e2)
                t = np.concatenate([t1[::-1], t2[1:]])
                y = np.hstack([y1[:, ::-1], y2[:, 1:]])
        else:
            d = np.diff(tv)
            if tv.size > 1 and not (np.all(d > 0) or np.all(d < 0)):
                raise MatlabError("solve: the output times must be strictly increasing or decreasing.")
            y0 = fnum(self.p["InitialValue"]).flatten(order="F")
            n = y0.size
            cols: dict = {}
            below = np.sort(tv[tv < t_init])[::-1]
            above = np.sort(tv[tv > t_init])
            for side in (below, above):
                if side.size:
                    ys, e = self._at(interp, t_init, side)
                    add(e)
                    for k in range(ys.shape[1]):
                        cols[float(side[k])] = ys[:, k]
            if np.any(tv == t_init):
                cols[t_init] = y0
            keep = [float(v) for v in tv if float(v) in cols]
            t = np.array(keep)
            y = np.column_stack([cols[v] for v in keep]) if keep else np.zeros((n, 0))
        out = OdeResults()
        out.p["Time"] = np.asarray(t, dtype=float).reshape(1, -1)
        out.p["Solution"] = simplify(np.asarray(y))
        if ev[0]:
            order = np.argsort(np.array(ev[0]), kind="stable")
            out.p["EventTime"] = np.array(ev[0], dtype=float)[order].reshape(1, -1)
            out.p["EventSolution"] = simplify(np.column_stack(ev[1])[:, order])
            out.p["EventIndex"] = np.array(ev[2], dtype=float)[order].reshape(1, -1)
        return out

    def solution_fcn(self, interp, rest: list):
        from ..parser import parse
        a, b, tv, t_init, _ = self._span(rest, "solutionFcn")
        lo, hi = min(a, b, t_init), max(a, b, t_init)
        sols = []
        for end in (lo, hi):
            if end == t_init:
                continue
            pieces, _, solver, n = self._run(interp, np.array([t_init, end]), True)
            sol = None
            for r in pieces:
                piece = om.result_to_sol(r, solver, n, Struct.scalar({}), False)
                sol = piece if sol is None else om.join_solutions(sol, piece, n)
            sols.append(sol)
        if not sols:
            raise MatlabError("solutionFcn: the interval must extend away from InitialTime.")
        node = parse("@(t) odeSolutionValue__(sols__, t)").body[0].expr
        fh = FunctionHandle(params=list(node.params), body=node.body,
                            closure={"__functions__": {}, "sols__": Cell.from_list(sols)},
                            text="@(t) solution(t)")
        S = OdeResults()
        xs = np.concatenate([fnum(om.sol_field(q, "x")).flatten() for q in sols])
        ys = np.hstack([fnum(om.sol_field(q, "y")).reshape(-1, fnum(om.sol_field(q, "x")).size) for q in sols])
        order = np.argsort(xs, kind="stable")
        keep = np.concatenate([[True], np.diff(xs[order]) != 0])
        S.p["Time"] = xs[order][keep].reshape(1, -1)
        S.p["Solution"] = simplify(ys[:, order][:, keep])
        return fh, S


@builtin("ode")
def _ode_ctor(interp, args, nargout):
    return OdeObject(_pairs(args, "ode"))


@builtin("odeSolutionValue__")
def _ode_solution_value(interp, args, nargout):
    sols = args[0].values_f()
    tq_arg = fnum(args[1])
    tq = np.real(tq_arg).flatten(order="F").astype(float)
    out = None
    done = np.zeros(tq.size, dtype=bool)
    for s in sols:
        x = fnum(om.sol_field(s, "x")).flatten()
        lo, hi = min(x[0], x[-1]), max(x[0], x[-1])
        m = (tq >= lo) & (tq <= hi) & ~done
        if not m.any():
            continue
        Y, _ = om.eval_solution(s, tq[m], False)
        if out is None:
            out = np.full((Y.shape[0], tq.size), np.nan, dtype=Y.dtype)
        elif np.iscomplexobj(Y) and not np.iscomplexobj(out):
            out = out.astype(complex)
        out[:, m] = Y
        done |= m
    if out is None or not done.all():
        raise MatlabError("The solution function was evaluated outside the interval it was built for.")
    return simplify(out)
