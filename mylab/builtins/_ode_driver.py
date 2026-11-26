"""Driver for the initial value solvers: builds a stepper for a problem description, then
handles output points, refinement, events, output functions and solution storage.

The problem is described with plain Python callables so that the same driver serves the
ODE front ends, the ``ode`` object, ``odextend``, ``pdepe`` and the delay solvers."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from . import _ode_steppers as st


class IVP:
    """Problem description (all callables take NumPy vectors)."""

    def __init__(self, solver, rhs, tspan, y0):
        self.solver = solver          # 'ode45', 'ode15s', ...
        self.rhs = rhs                # f(t, y) -> vector
        self.rhs_mat = None           # f(t, Y) -> matrix for vectorised calls, or None
        self.tspan = np.asarray(tspan, dtype=float)
        self.y0 = y0
        self.rtol = 1e-3
        self.atol = np.array(1e-6)
        self.normcontrol = False
        self.hmax = None
        self.h0 = None
        self.refine = None
        self.nonneg = None
        self.mass = None              # ('const', M) | ('fun', M(t, y))
        self.mstate = "weak"
        self.mvpattern = None
        self.mass_singular = "maybe"
        self.jac = None               # ('const', J) | ('fun', J(t, y))
        self.jpattern = None
        self.vectorized = False
        self.events = None            # ev(t, y, yp) -> (value, terminal, direction)
        self.events_yp = False        # whether the event function needs the slope
        self.outfcn = None            # out(t_row, y_cols, flag) -> stop?
        self.initial_slope = None
        self.max_order = 5
        self.bdf = False
        # fully implicit problems (ode15i)
        self.implicit = None          # G(t, y, yp)
        self.implicit_jac = None      # (t, y, yp) -> (dG/dy, dG/dyp), or None
        self.implicit_pattern = None
        self.yp0 = None
        self.warn = None              # warn(message)


class Result:
    def __init__(self):
        self.x = []
        self.y = []
        self.coefs = []
        self.tout = []
        self.yout = []
        self.te, self.ye, self.ie = [], [], []
        self.counters = None
        self.message = None
        self.stopped_by_event = False
        self.yp_end = None


_FAMILY = {
    "ode45": ("erk", "dp45", 4), "ode23": ("erk", "bs23", 1), "ode113": ("erk", "dop853", 1),
    "ode78": ("erk", "dop853", 1), "ode89": ("erk", "dop853", 1),
    "ode15s": ("bdf", None, 1), "ode23t": ("one", "tr", 1), "ode23tb": ("one", "trbdf2", 1),
    "ode23s": ("ros", None, 1), "ode15i": ("bdfi", None, 1),
}


def is_stiff_solver(name: str) -> bool:
    return _FAMILY[name][0] in ("bdf", "one", "ros", "bdfi")


def default_refine(name: str) -> int:
    return _FAMILY[name][2]


class _Rhs:
    """Counts evaluations and remembers the last value (the FD Jacobian asks for it again)."""

    def __init__(self, f, n, counters):
        self.f, self.n, self.cnt = f, n, counters
        self.key = None
        self.val = None

    def __call__(self, t, y):
        key = (t, y.tobytes())
        if key == self.key:
            return self.val
        v = self.f(t, y)
        self.cnt.nfevals += 1
        self.key, self.val = key, v
        return v


def _mass_function(P: IVP):
    """(M(t, y) -> matrix or None, constant?)"""
    if P.mass is None:
        return (lambda t, y: None), True
    kind, M = P.mass
    if kind == "const":
        return (lambda t, y: M), True
    return M, False


def _jacobian_function(P: IVP, f, ctl, cnt):
    """J(t, y) = df/dy from the user's Jacobian or by finite differences."""
    if P.jac is not None:
        kind, J = P.jac
        if kind == "const":
            return (lambda t, y: J), True
        return J, False
    groups = None
    if P.jpattern is not None:
        pat = np.asarray(P.jpattern) != 0
        if pat.shape != (P.y0.size, P.y0.size):
            raise MatlabError("JPattern must be a square matrix with one row per equation.")
        groups = st.color_columns(pat)
    typical = np.broadcast_to(ctl.threshold, (P.y0.size,))

    def J(t, y):
        f0 = f(t, y)
        if P.vectorized and P.rhs_mat is not None and groups is None:
            jac, calls = st.fd_jacobian(lambda Y: P.rhs_mat(t, Y), y, f0, typical, None, True)
        else:
            jac, calls = st.fd_jacobian(lambda v: P.rhs(t, v), y, f0, typical, groups, False)
        cnt.nfevals += calls
        return jac

    return J, False


def _dae_init(P: IVP, f, Mat, Jf, ctl):
    """Initial slope, adjusted initial state and null-space projector for M y' = f."""
    t0 = float(P.tspan[0])
    y0 = np.array(P.y0, copy=True)
    M0 = Mat(t0, y0)
    f0 = f(t0, y0)
    n = y0.size
    if M0 is None:
        return y0, np.array(f0, copy=True), None
    M0 = np.asarray(M0)
    if M0.shape != (n, n):
        raise MatlabError(f"The mass matrix must be {n}-by-{n}.")
    if P.mass_singular == "no":
        try:
            return y0, np.linalg.solve(M0, f0), None
        except np.linalg.LinAlgError:
            raise MatlabError("The mass matrix is singular, but MassSingular is 'no'.") from None
    U, s, Vt = np.linalg.svd(M0)
    tol = max(M0.shape) * st.EPS * (s[0] if s.size else 0.0)
    r = int(np.sum(s > tol))
    if r == n:
        return y0, np.linalg.solve(M0, f0), None
    Ua, V0 = U[:, r:], Vt[r:].conj().T
    y = y0
    J = Jf(t0, y)
    A = Ua.conj().T @ J @ V0
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise MatlabError("This DAE appears to be of index greater than 1.")
    scale = max(1.0, float(np.linalg.norm(f0)))
    res = Ua.conj().T @ f0
    nr = float(np.linalg.norm(res))
    if nr > 1e-3 * ctl.rtol * scale:
        for it in range(25):
            y = y + V0 @ np.linalg.lstsq(A, -res, rcond=None)[0]
            f0 = f(t0, y)
            res = Ua.conj().T @ f0
            nr_new = float(np.linalg.norm(res))
            if nr_new <= 1e-10 * scale or (it >= 2 and nr_new > 0.5 * nr and nr_new <= 1e-6 * scale):
                break
            nr = nr_new
            J = Jf(t0, y)
            A = Ua.conj().T @ J @ V0
        else:
            raise MatlabError("Could not find consistent initial conditions for this DAE; "
                              "supply a better initial guess y0.")
    yp = np.linalg.lstsq(M0, f0, rcond=None)[0]
    if P.initial_slope is not None and np.linalg.norm(M0 @ P.initial_slope - f0) <= 1e-6 * scale:
        yp = np.array(P.initial_slope, dtype=yp.dtype, copy=True)
    else:
        # slope of the algebraic variables from the differentiated constraint
        dt = np.sqrt(st.EPS) * max(abs(t0), 1.0) * ctl.tdir
        ft = (P.rhs(t0 + dt, y) - f0) / dt
        try:
            z = np.linalg.solve(A, -(Ua.conj().T @ (ft + J @ yp)))
            yp = yp + V0 @ z
        except np.linalg.LinAlgError:
            pass
    return y, yp, V0 @ V0.conj().T


def _build_stepper(P: IVP, ctl, cnt):
    family, method, _ = _FAMILY[P.solver]
    n = P.y0.size
    t0 = float(P.tspan[0])
    if family == "bdfi":
        G = P.implicit

        def Gc(t, y, yp):
            cnt.nfevals += 1
            return G(t, y, yp)

        if P.implicit_jac is not None:
            lin = P.implicit_jac
        else:
            typical = np.broadcast_to(ctl.threshold, (n,))
            gy = gyp = None
            if P.implicit_pattern is not None:
                gy = st.color_columns(np.asarray(P.implicit_pattern[0]) != 0)
                gyp = st.color_columns(np.asarray(P.implicit_pattern[1]) != 0)

            def lin(t, y, yp):
                g0 = G(t, y, yp)
                A, c1 = st.fd_jacobian(lambda v: G(t, v, yp), y, g0, typical, gy)
                B, c2 = st.fd_jacobian(lambda v: G(t, y, v), yp, g0, np.maximum(typical, 1e-3), gyp)
                cnt.nfevals += c1 + c2 + 1
                return A, B
        system = st.ImplicitSystem(Gc, lin, const=False)
        return st.BDF(system, t0, P.y0, P.yp0, ctl, cnt, max_order=P.max_order, bdf=True), P.y0
    f = _Rhs(P.rhs, n, cnt)
    Mat, mconst = _mass_function(P)
    if family == "erk":
        if P.mass is None:
            feff = f
        elif mconst:
            from scipy.linalg import lu_factor, lu_solve
            M = np.asarray(Mat(t0, P.y0))
            if M.shape != (n, n):
                raise MatlabError(f"The mass matrix must be {n}-by-{n}.")
            if np.linalg.matrix_rank(M) < n:
                raise MatlabError(f"The mass matrix is singular, so this is a differential-algebraic "
                                  f"problem; {P.solver} cannot solve it. Use ode15s or ode23t.")
            lu = lu_factor(M)

            def feff(t, y):
                return lu_solve(lu, f(t, y))
        else:
            def feff(t, y):
                try:
                    return np.linalg.solve(Mat(t, y), f(t, y))
                except np.linalg.LinAlgError:
                    raise MatlabError(f"The mass matrix became singular at t = {t:g}; {P.solver} cannot "
                                      "solve differential-algebraic problems. Use ode15s or ode23t.") from None
        safety = 0.8
        return st.ExplicitRK(feff, t0, P.y0, ctl, method, cnt, nonneg=P.nonneg, safety=safety), P.y0
    Jf, jconst = _jacobian_function(P, f, ctl, cnt)
    if family == "ros":
        if P.mass is not None and not mconst:
            raise MatlabError("ode23s works only with a constant mass matrix; use ode15s, ode23t or ode23tb.")
        M = None if P.mass is None else np.asarray(Mat(t0, P.y0), dtype=float)
        if M is not None and np.linalg.matrix_rank(M) < n:
            raise MatlabError("ode23s cannot solve problems with a singular mass matrix; use ode15s or ode23t.")

        def jac(t, y, f0):
            J = Jf(t, y)
            dt = np.sqrt(st.EPS) * max(abs(t), abs(ctl.tfinal), 1e-8) * ctl.tdir
            ft = (P.rhs(t + dt, y) - f0) / dt
            cnt.nfevals += 1
            return J, ft

        yp0 = f(t0, P.y0) if M is None else np.linalg.solve(M, f(t0, P.y0))
        return st.Rosenbrock23(f, jac, M, t0, P.y0, yp0, ctl, cnt), P.y0
    y0, yp0, null = _dae_init(P, f, Mat, Jf, ctl)
    strong = P.mass is not None and not mconst and P.mstate == "strong"
    eye = np.eye(n)

    def G(t, y, yp):
        M = Mat(t, y)
        return (yp if M is None else M @ yp) - f(t, y)

    def lin(t, y, yp):
        J = Jf(t, y)
        M = Mat(t, y)
        dGdy = -J
        if strong:
            typical = np.broadcast_to(ctl.threshold, (n,))
            m0 = M @ yp
            groups = st.color_columns(np.asarray(P.mvpattern) != 0) if P.mvpattern is not None else None
            extra, _ = st.fd_jacobian(lambda v: Mat(t, v) @ yp, y, m0, typical, groups)
            dGdy = dGdy + extra
        return dGdy, (eye if M is None else np.asarray(M))

    system = st.ImplicitSystem(G, lin, const=(jconst and mconst), null=null)
    if P.initial_slope is not None and null is None and P.mass is None:
        yp0 = yp0  # the slope of an explicit ODE is f(t0, y0); the option is only a guess for DAEs
    if family == "bdf":
        return st.BDF(system, t0, y0, yp0, ctl, cnt, max_order=P.max_order, bdf=P.bdf), y0
    return st.ImplicitOneStep(system, t0, y0, yp0, ctl, cnt, method=method), y0


def _find_root(g, v0, v1, tol):
    """Root of g on [0, 1] given g(0) = v0 and g(1) = v1 of opposite sign."""
    if v1 == 0:
        return 1.0
    from scipy.optimize import brentq
    s = brentq(g, 0.0, 1.0, xtol=max(tol, 4 * st.EPS), rtol=8 * st.EPS, maxiter=200)
    # report the side on which the event has happened (value has the sign it has after the crossing)
    if s < 1.0:
        gs = g(s)
        if gs != 0 and np.sign(gs) == np.sign(v0):
            s2 = min(1.0, s + max(tol, 4 * st.EPS))
            if np.sign(g(s2)) != np.sign(v0):
                s = s2
    return float(s)


def integrate(P: IVP, want_dense: bool = True) -> Result:
    tspan = P.tspan
    t0, tf = float(tspan[0]), float(tspan[-1])
    if P.hmax is None:
        hmax = 0.1 * abs(tf - t0)
    else:
        hmax = abs(float(P.hmax))
    if hmax <= 0:
        raise MatlabError("MaxStep must be positive.")
    ctl = st.Control(P.rtol, P.atol, P.normcontrol, hmax, P.h0, t0, tf)
    cnt = st.Counters()
    res = Result()
    res.counters = cnt
    n = P.y0.size
    stepper, y0 = _build_stepper(P, ctl, cnt)
    tdir = ctl.tdir
    steps_mode = tspan.size == 2
    refine = max(int(P.refine if P.refine is not None else default_refine(P.solver)), 1)
    if not steps_mode:
        refine = 1
    nn = P.nonneg
    res.x.append(t0)
    res.y.append(np.array(y0, copy=True))
    res.tout.append(t0)
    res.yout.append(np.array(y0, copy=True))
    nxt = 1
    ev = P.events
    v_old = None
    if ev is not None:
        v_old, terminal, direction = ev(t0, y0, stepper.yp)
    if P.outfcn is not None:
        P.outfcn(np.array([t0, tf]), np.array(y0, copy=True).reshape(-1, 1), "init")
    stop = False
    while not stop:
        try:
            stepper.step()
        except st.StepFailure as e:
            res.message = str(e)
            break
        t_old, t_new = stepper.t_old, stepper.t
        y_new = stepper.y
        coefs = stepper.coefs
        if nn is not None and not isinstance(stepper, st.ExplicitRK):
            bad = nn[np.real(y_new[nn]) < 0]
            if bad.size:
                y_new = np.array(y_new, copy=True)
                y_new[bad] = 0.0
                stepper.set_state(y_new)
        h = t_new - t_old

        def interp_y(s, coefs=coefs):
            v = st.eval_piece(coefs, s)
            if nn is not None:
                v[nn] = np.maximum(np.real(v[nn]), 0.0)
            return v

        if ev is not None:
            yp_new = None
            if P.events_yp:
                yp_new = stepper.yp if stepper.yp is not None else st.eval_piece_deriv(coefs, 1.0)[:, 0] / h
            v_new, terminal, direction = ev(t_new, y_new, yp_new)
            hits = []
            if v_old is not None and v_new.size == v_old.size:
                for k in range(v_new.size):
                    a, b = v_old[k], v_new[k]
                    if a == 0 or np.sign(a) == np.sign(b):
                        continue
                    if direction[k] > 0 and not b > a:
                        continue
                    if direction[k] < 0 and not b < a:
                        continue
                    cache = {0.0: a, 1.0: b}

                    def g(s, k=k, cache=cache):
                        if s not in cache:
                            ys = interp_y(s)[:, 0]
                            yps = st.eval_piece_deriv(coefs, s)[:, 0] / h if P.events_yp else None
                            cache[s] = ev(t_old + s * h, ys, yps)[0][k]
                        return cache[s]

                    tol_s = 128 * st.EPS * max(abs(t_old), abs(t_new), 1e-300) / abs(h)
                    hits.append((_find_root(g, a, b, tol_s), k))
            hits.sort()
            for s_e, k in hits:
                te = t_new if s_e >= 1.0 else t_old + s_e * h
                ye = y_new if s_e >= 1.0 else interp_y(s_e)[:, 0]
                res.te.append(te)
                res.ye.append(np.array(ye, copy=True))
                res.ie.append(k + 1)
                if terminal[k]:
                    stop = True
                    res.stopped_by_event = True
                    if s_e < 1.0:
                        # shorten the step to end at the event
                        coefs = coefs * (s_e ** np.arange(coefs.shape[1]))
                        t_new, y_new, h = te, np.array(ye, copy=True), te - t_old
                    break
            v_old = v_new
        # output points produced by this step
        new_t, new_y = [], []
        if steps_mode:
            for i in range(1, refine):
                s = i / refine
                new_t.append(t_old + s * h)
                new_y.append(st.eval_piece(coefs, s)[:, 0])
            new_t.append(t_new)
            new_y.append(y_new)
        else:
            while nxt < tspan.size and tdir * (tspan[nxt] - t_new) <= 0:
                tq = float(tspan[nxt])
                if tq == t_new:
                    new_y.append(y_new)
                else:
                    new_y.append(st.eval_piece(coefs, (tq - t_old) / h)[:, 0])
                new_t.append(tq)
                nxt += 1
            if stop and (not new_t or new_t[-1] != t_new):
                new_t.append(t_new)
                new_y.append(y_new)
        if nn is not None:
            for v in new_y[:-1] if steps_mode else new_y:
                v[nn] = np.maximum(np.real(v[nn]), 0.0)
        res.tout.extend(new_t)
        res.yout.extend(np.array(v, copy=True) for v in new_y)
        res.x.append(t_new)
        res.y.append(np.array(y_new, copy=True))
        if want_dense:
            res.coefs.append(coefs)
        if P.outfcn is not None and new_t:
            if P.outfcn(np.array(new_t), np.column_stack(new_y), ""):
                stop = True
        if stepper.done:
            break
    res.yp_end = stepper.yp
    if P.outfcn is not None:
        P.outfcn(np.zeros(0), np.zeros((n, 0)), "done")
    return res
