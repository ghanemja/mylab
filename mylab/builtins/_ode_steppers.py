"""Time steppers behind the ODE suite (pure NumPy; no interpreter types in here).

Every stepper advances one accepted step per :meth:`step` call and afterwards exposes

``t_old, t, y, yp``   the step just taken (``yp`` is the slope at the new point)
``coefs``             an ``n x (d+1)`` array: the continuous extension on the step is
                      ``y(t_old + s*h) = sum_j coefs[:, j] * s**j`` for ``0 <= s <= 1``

so that the driver can produce refined output, locate events and build solution structures
in one uniform way. Step-size control follows the published design of the classic
variable-step suite: a mixed relative/absolute error test in the max norm (or the 2-norm
with ``NormControl``), a safety factor of 0.8 for the explicit pairs, and limits on how fast
the step may grow or shrink.
"""

from __future__ import annotations

import numpy as np

EPS = float(np.finfo(float).eps)


class StepFailure(Exception):
    """The step size underflowed or the nonlinear solver gave up; carries a message."""


class Counters:
    def __init__(self):
        self.nsteps = self.nfailed = self.nfevals = self.npds = self.ndecomps = self.nsolves = 0


class Control:
    """Tolerances and step limits shared by all steppers."""

    def __init__(self, rtol, atol, normcontrol, hmax, h0, t0, tfinal):
        self.rtol = float(rtol)
        self.atol = np.asarray(atol, dtype=float)
        self.threshold = self.atol / self.rtol
        self.normcontrol = bool(normcontrol)
        self.tdir = 1.0 if tfinal >= t0 else -1.0
        self.tfinal = float(tfinal)
        self.hmax = float(hmax)
        self.h0 = h0

    def err_norm(self, e, y, ynew):
        """Error measure to compare with rtol."""
        if e.size == 0:
            return 0.0
        if self.normcontrol:
            scale = max(np.linalg.norm(y), np.linalg.norm(ynew), float(np.max(self.threshold)))
            return float(np.linalg.norm(e)) / scale
        return float(np.max(np.abs(e) / np.maximum(np.maximum(np.abs(y), np.abs(ynew)), self.threshold)))

    def weights(self, y, ynew=None):
        """1 / (rtol * scale) so that a weighted max norm of 1 means 'at the tolerance'."""
        a = np.abs(y) if ynew is None else np.maximum(np.abs(y), np.abs(ynew))
        if self.normcontrol:
            s = max(float(np.linalg.norm(a)), float(np.max(self.threshold)))
            return np.full(a.shape, 1.0 / (self.rtol * s))
        return 1.0 / (self.rtol * np.maximum(a, self.threshold))

    def wnorm(self, v, w):
        if v.size == 0:
            return 0.0
        if self.normcontrol:
            return float(np.linalg.norm(v * w))
        return float(np.max(np.abs(v * w)))

    def hmin(self, t):
        return 16.0 * EPS * max(abs(t), np.finfo(float).tiny / EPS)

    def first_step(self, t0, y0, f0, power, factor=0.8):
        """Initial step from the size of the slope relative to the solution scale."""
        span = abs(self.tfinal - t0)
        absh = min(self.hmax, span)
        if self.h0 is not None:
            return min(self.hmax, max(self.hmin(t0), abs(self.h0)))
        if self.normcontrol:
            rh = (np.linalg.norm(f0) / max(np.linalg.norm(y0), float(np.max(self.threshold)))) \
                / (factor * self.rtol ** power)
        else:
            rh = float(np.max(np.abs(f0) / np.maximum(np.abs(y0), self.threshold), initial=0.0)) \
                / (factor * self.rtol ** power)
        if absh * rh > 1:
            absh = 1.0 / rh
        return max(absh, self.hmin(t0))


# -- explicit Runge-Kutta pairs ---------------------------------------------------------------------

_TABLEAUS: dict = {}


def _tableau(name: str) -> dict:
    """Coefficients of the embedded pairs (taken from SciPy's solver classes)."""
    if name in _TABLEAUS:
        return _TABLEAUS[name]
    from scipy import integrate
    if name in ("dp45", "bs23"):
        cls = integrate.RK45 if name == "dp45" else integrate.RK23
        A = np.asarray(cls.A, dtype=float)
        tab = {"A": A, "B": np.asarray(cls.B, dtype=float), "C": np.asarray(cls.C, dtype=float),
               "E": np.asarray(cls.E, dtype=float), "P": np.asarray(cls.P, dtype=float),
               "stages": A.shape[0], "power": 1.0 / (5 if name == "dp45" else 3)}
    else:
        cls = integrate.DOP853
        ns = cls.n_stages
        # powers of s for the nested 7th-order interpolant y0 + s*(F0 + (1-s)*(F1 + s*(F2 + ...)))
        T = np.zeros((7, 8))
        s = np.poly1d([1.0, 0.0])
        one_minus = np.poly1d([-1.0, 1.0])
        p = np.poly1d([1.0])
        for k in range(7):
            p = p * (s if k % 2 == 0 else one_minus)
            c = p.coeffs[::-1]
            T[k, :c.size] = c
        tab = {"A": np.asarray(cls.A, dtype=float)[:ns, :ns], "B": np.asarray(cls.B, dtype=float),
               "C": np.asarray(cls.C, dtype=float)[:ns], "E3": np.asarray(cls.E3, dtype=float),
               "E5": np.asarray(cls.E5, dtype=float), "D": np.asarray(cls.D, dtype=float),
               "A_EXTRA": np.asarray(cls.A_EXTRA, dtype=float), "C_EXTRA": np.asarray(cls.C_EXTRA, dtype=float),
               "T": T, "stages": ns, "power": 1.0 / 8}
    _TABLEAUS[name] = tab
    return tab


class ExplicitRK:
    """Embedded explicit Runge-Kutta pair with first-same-as-last and a continuous extension."""

    def __init__(self, f, t0, y0, ctl: Control, method: str, counters: Counters, nonneg=None,
                 safety: float = 0.8):
        self.f, self.ctl, self.cnt = f, ctl, counters
        self.tab = _tableau(method)
        self.method = method
        self.power = self.tab["power"]
        self.safety = safety
        self.nonneg = nonneg
        self.t = float(t0)
        self.y = np.array(y0, copy=True)
        self.yp = self._f(self.t, self.y)
        self.absh = ctl.first_step(self.t, self.y, self.yp, self.power, safety)
        self.done = False
        self.t_old = self.t
        self.coefs = None

    def _f(self, t, y):
        self.cnt.nfevals += 1
        v = self.f(t, y)
        if self.nonneg is not None:
            idx = self.nonneg[y[self.nonneg] <= 0]
            if idx.size:
                v = np.array(v, copy=True)
                v[idx] = np.maximum(np.real(v[idx]), 0.0)
        return v

    def step(self):
        ctl, tab = self.ctl, self.tab
        t, y, f0 = self.t, self.y, self.yp
        n = y.size
        ns = tab["stages"]
        A, B, C = tab["A"], tab["B"], tab["C"]
        hmin = ctl.hmin(t)
        absh = min(ctl.hmax, max(hmin, self.absh))
        nofailed = True
        dtype = np.result_type(y.dtype, f0.dtype, float)
        while True:
            h = ctl.tdir * absh
            last = False
            if 1.1 * absh >= abs(ctl.tfinal - t):
                h = ctl.tfinal - t
                absh = abs(h)
                last = True
            K = np.empty((ns + 1, n), dtype=dtype)
            K[0] = f0
            for s in range(1, ns):
                K[s] = self._f(t + C[s] * h, y + h * (A[s, :s] @ K[:s]))
            ynew = y + h * (B @ K[:ns])
            tnew = ctl.tfinal if last else t + h
            h = tnew - t
            K[ns] = self._f(tnew, ynew)
            if self.method == "dop853":
                e5 = K.T @ tab["E5"]
                e3 = K.T @ tab["E3"]
                den = np.sqrt(np.abs(e5) ** 2 + 0.01 * np.abs(e3) ** 2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.where(den > 0, np.abs(e5) ** 2 / den, 0.0)
            else:
                e = K.T @ tab["E"]
            if not (np.all(np.isfinite(ynew)) and np.all(np.isfinite(K[ns]))):
                err = np.inf
            else:
                err = absh * ctl.err_norm(e, y, ynew)
            nn_reject = False
            if self.nonneg is not None and err <= ctl.rtol:
                bad = self.nonneg[np.real(ynew[self.nonneg]) < 0]
                if bad.size:
                    e_nn = ctl.err_norm(np.where(np.isin(np.arange(n), bad), ynew, 0.0), y, ynew)
                    if e_nn > ctl.rtol:
                        err = e_nn
                        nn_reject = True
                    else:
                        ynew = np.array(ynew, copy=True)
                        ynew[bad] = 0.0
                        K[ns] = self._f(tnew, ynew)
            if err > ctl.rtol:
                self.cnt.nfailed += 1
                if absh <= hmin:
                    raise StepFailure(
                        f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                        f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
                if nofailed and np.isfinite(err) and not nn_reject:
                    nofailed = False
                    absh = max(hmin, absh * max(0.1, self.safety * (ctl.rtol / err) ** self.power))
                else:
                    nofailed = False
                    absh = max(hmin, 0.5 * absh)
                continue
            break
        self.cnt.nsteps += 1
        # continuous extension as powers of s = (t - t_old)/h
        if self.method == "dop853":
            Kx = np.empty((ns + 4, n), dtype=dtype)
            Kx[:ns + 1] = K
            for j in range(3):
                s_idx = ns + 1 + j
                a = tab["A_EXTRA"][j]
                Kx[s_idx] = self._f(t + tab["C_EXTRA"][j] * h, y + h * (a[:s_idx] @ Kx[:s_idx]))
            F = np.empty((7, n), dtype=dtype)
            dy = ynew - y
            F[0] = dy
            F[1] = h * K[0] - dy
            F[2] = 2 * dy - h * (K[ns] + K[0])
            F[3:] = h * (tab["D"] @ Kx)
            coefs = np.empty((n, 8), dtype=dtype)
            coefs[:, 0] = y
            coefs[:, 1:] = (tab["T"][:, 1:].T @ F).T
        else:
            P = tab["P"]
            coefs = np.empty((n, P.shape[1] + 1), dtype=dtype)
            coefs[:, 0] = y
            coefs[:, 1:] = h * (K.T @ P)
        self.t_old, self.coefs = t, coefs
        self.t, self.y, self.yp = tnew, ynew, K[ns]
        self.done = last
        if nofailed:
            temp = 1.25 * (err / ctl.rtol) ** self.power
            absh = absh / temp if temp > 0.2 else 5.0 * absh
        self.absh = absh
        return True


# -- implicit methods: shared pieces -----------------------------------------------------------

def fd_jacobian(fun, y, f0, typical, groups=None, vectorized=False):
    """Forward-difference Jacobian of ``fun(y)``; ``groups`` lists columns that can share a call."""
    n = y.size
    m = f0.size
    scale = np.maximum(np.abs(y), typical)
    delta = np.sqrt(EPS) * scale
    delta = np.where(np.real(y) < 0, -delta, delta)
    J = np.zeros((m, n), dtype=np.result_type(f0.dtype, float))
    calls = 0
    if groups is None:
        if vectorized and n:
            Y = np.repeat(y.reshape(-1, 1), n, axis=1).astype(J.dtype)
            Y[np.arange(n), np.arange(n)] += delta
            F = fun(Y)
            calls = 1
            J = (F - f0.reshape(-1, 1)) / delta.reshape(1, -1)
        else:
            for j in range(n):
                yj = np.array(y, dtype=J.dtype, copy=True)
                yj[j] += delta[j]
                J[:, j] = (fun(yj) - f0) / delta[j]
            calls = n
        return J, calls
    cols_list, pattern = groups
    for cols in cols_list:
        yj = np.array(y, dtype=J.dtype, copy=True)
        yj[cols] += delta[cols]
        df = fun(yj) - f0
        calls += 1
        for j in cols:
            rows = pattern[j]
            J[rows, j] = df[rows] / delta[j]
    return J, calls


def color_columns(pattern: np.ndarray):
    """Greedy grouping of structurally orthogonal columns of a sparsity pattern (bool m x n)."""
    m, n = pattern.shape
    rows_of = [np.nonzero(pattern[:, j])[0] for j in range(n)]
    groups: list = []
    used: list = []
    for j in range(n):
        placed = False
        for g, mask in zip(groups, used):
            if not mask[rows_of[j]].any():
                g.append(j)
                mask[rows_of[j]] = True
                placed = True
                break
        if not placed:
            mask = np.zeros(m, dtype=bool)
            mask[rows_of[j]] = True
            groups.append([j])
            used.append(mask)
    return [np.array(g, dtype=int) for g in groups], rows_of


class ImplicitSystem:
    """``G(t, y, yp) = 0`` with its partial derivatives; built by the front end.

    ``G``        residual function
    ``lin``      ``(t, y, yp) -> (dG/dy, dG/dyp)`` as dense arrays
    ``const``    True when the two matrices never change
    ``null``     projector onto the null space of dG/dyp (singular mass), or None
    """

    def __init__(self, G, lin, const=False, null=None):
        self.G, self.lin, self.const, self.null = G, lin, const, null


class _Linear:
    """Caches the linearisation and the LU factors of ``dG/dy + alpha * dG/dyp``."""

    def __init__(self, system: ImplicitSystem, counters: Counters):
        self.sys, self.cnt = system, counters
        self.mats = None
        self.fresh = False
        self.lu = None
        self.alpha = None
        self.age = 0
        self.h_ref = None

    def refresh(self, t, y, yp, h=None):
        if self.mats is None or not self.sys.const:
            self.mats = self.sys.lin(t, y, yp)
            self.cnt.npds += 1
            self.lu = None
            self.age = 0
            if h is not None:
                self.h_ref = abs(h)
        self.fresh = True

    def keep_current(self, t, y, yp, h):
        """Re-linearise when the step size has moved far from the one the Jacobian was formed
        for, or the Jacobian is old. A stale Jacobian can make the simplified Newton iteration
        stall in a way that looks like convergence."""
        self.age += 1
        if self.sys.const or self.fresh:
            return
        ratio = abs(h) / self.h_ref if self.h_ref else 1.0
        if ratio > 5.0 or ratio < 0.2 or self.age > 50:
            self.refresh(t, y, yp, h)

    def factor(self, alpha):
        if self.lu is None or alpha != self.alpha:
            from scipy.linalg import lu_factor
            N = self.mats[0] + alpha * self.mats[1]
            if not np.all(np.isfinite(N)):
                raise StepFailure("The Jacobian contains Inf or NaN.")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.lu = lu_factor(N, check_finite=False)
            self.alpha = alpha
            self.cnt.ndecomps += 1
        return self.lu

    def solve(self, rhs):
        from scipy.linalg import lu_solve
        self.cnt.nsolves += 1
        return lu_solve(self.lu, rhs, check_finite=False)


def _newton(system, lin: _Linear, ctl: Control, t, y_guess, alpha, beta, w, tol, maxit=4):
    """Simplified Newton iteration for ``G(t, y, alpha*y + beta) = 0``.

    Returns (converged, y, iterations)."""
    y = np.array(y_guess, copy=True)
    old = None
    rate = None
    for it in range(maxit):
        g = system.G(t, y, alpha * y + beta)
        if not np.all(np.isfinite(g)):
            return False, y, it + 1
        dy = lin.solve(-g)
        y = y + dy
        nrm = ctl.wnorm(dy, w)
        if old is not None and old > 0:
            rate = nrm / old
            if rate >= 1 or (rate > 0.5 and rate ** (maxit - 1 - it) / (1 - rate) * nrm > tol):
                return False, y, it + 1
        # a small first correction proves nothing when the Jacobian is stale, so (except for an exact
        # zero) convergence is only declared once a contraction rate has been observed
        if nrm == 0 or (rate is not None and rate / (1 - rate) * nrm < tol):
            return True, y, it + 1
        old = nrm
    return False, y, maxit


def _newton_tol(rtol):
    return max(10 * EPS / rtol, min(0.03, np.sqrt(rtol)))


# -- variable-order BDF / NDF -------------------------------------------------------------------

_KAPPA = np.array([0.0, -0.1850, -1.0 / 9, -0.0823, -0.0415, 0.0])


def _binom_poly(j: int, s):
    """s (s+1) ... (s+j-1) / j!  (the backward-difference basis function)."""
    out = np.ones_like(np.asarray(s, dtype=float))
    for m in range(j):
        out = out * (s + m) / (m + 1)
    return out


def _rescale_matrix(order: int, rho: float) -> np.ndarray:
    """Matrix mapping backward differences at spacing h to those at spacing rho*h."""
    from math import comb
    T = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            T[i, j] = sum((-1) ** el * comb(i, el) * float(_binom_poly(j, -el * rho)) for el in range(i + 1))
    return T


_BDF_BASIS: dict = {}


def _bdf_basis(order: int) -> np.ndarray:
    """Rows j = 1..order: powers of s (low to high) of prod_{m<j} (s - 1 + m)/(m + 1)."""
    if order not in _BDF_BASIS:
        Q = np.zeros((order, order + 1))
        for j in range(1, order + 1):
            p = np.poly1d([1.0])
            for m in range(j):
                p = p * np.poly1d([1.0, m - 1.0]) / (m + 1)
            c = p.coeffs[::-1]
            Q[j - 1, :c.size] = c
        _BDF_BASIS[order] = Q
    return _BDF_BASIS[order]


class BDF:
    """Quasi-constant step-size NDF/BDF of orders 1 to ``max_order`` for ``G(t, y, y') = 0``."""

    def __init__(self, system: ImplicitSystem, t0, y0, yp0, ctl: Control, counters: Counters,
                 max_order=5, bdf=False):
        self.sys, self.ctl, self.cnt = system, ctl, counters
        self.max_order = int(max_order)
        kappa = np.zeros(6) if bdf else _KAPPA
        self.gamma = np.hstack((0.0, np.cumsum(1.0 / np.arange(1, 7))))
        self.alpha = (1 - kappa) * self.gamma[:6]
        self.erconst = kappa * self.gamma[:6] + 1.0 / np.arange(1, 7)
        self.t = float(t0)
        self.y = np.array(y0, dtype=np.result_type(y0.dtype, yp0.dtype, float), copy=True)
        self.yp = np.array(yp0, dtype=self.y.dtype, copy=True)
        n = self.y.size
        absh = ctl.first_step(self.t, self.y, self.yp, 0.5, 0.8)
        self.h = ctl.tdir * absh
        self.order = 1
        self.D = np.zeros((self.max_order + 3, n), dtype=self.y.dtype)
        self.D[0] = self.y
        self.D[1] = self.h * self.yp
        self.n_equal = 0
        self.lin = _Linear(system, counters)
        self.lin.refresh(self.t, self.y, self.yp, self.h)
        self.done = False
        self.t_old = self.t
        self.coefs = None

    def _change(self, factor):
        k = self.order
        self.D[:k + 1] = _rescale_matrix(k, factor) @ self.D[:k + 1]
        self.h *= factor
        self.n_equal = 0

    def step(self):
        ctl = self.ctl
        t = self.t
        hmin = ctl.hmin(t)
        if abs(self.h) > ctl.hmax:
            self._change(ctl.hmax / abs(self.h))
        last = False
        if 1.0001 * abs(self.h) >= abs(ctl.tfinal - t):
            if self.h != ctl.tfinal - t:
                self._change(abs(ctl.tfinal - t) / abs(self.h))
            last = True
        tol = _newton_tol(ctl.rtol)
        maxit = 4
        self.lin.keep_current(t, self.y, self.yp, self.h)
        while True:
            h = self.h
            if abs(h) < hmin:
                raise StepFailure(
                    f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                    f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
            k = self.order
            tnew = ctl.tfinal if last else t + h
            D = self.D
            y_pred = D[:k + 1].sum(axis=0)
            psi = (self.gamma[1:k + 1] @ D[1:k + 1]) / self.alpha[k]
            c = h / self.alpha[k]
            w = ctl.weights(y_pred, self.y)
            converged = False
            while True:
                try:
                    self.lin.factor(1.0 / c)
                    converged, ynew, nit = _newton(self.sys, self.lin, ctl, tnew, y_pred, 1.0 / c,
                                                   (psi - y_pred) / c, w, tol, maxit)
                except StepFailure:
                    converged, nit = False, maxit
                if converged or self.lin.fresh or self.sys.const:
                    break
                self.lin.refresh(tnew, y_pred, psi / c, h)
            if not converged:
                self.cnt.nfailed += 1
                self.lin.fresh = False
                last = False
                self._change(0.3)
                continue
            d = ynew - y_pred
            safety = 0.9 * (2 * maxit + 1) / (2 * maxit + nit)
            w = ctl.weights(ynew, self.y)
            err = ctl.wnorm(self.erconst[k] * d, w)
            if err > 1:
                self.cnt.nfailed += 1
                last = False
                self._change(max(0.2, safety * err ** (-1.0 / (k + 1))))
                continue
            break
        self.cnt.nsteps += 1
        self.lin.fresh = False
        self.n_equal += 1
        D[k + 2] = d - D[k + 1]
        D[k + 1] = d
        for i in range(k, -1, -1):
            D[i] += D[i + 1]
        Q = _bdf_basis(k)
        coefs = np.empty((self.y.size, k + 1), dtype=self.y.dtype)
        coefs[:] = (Q.T @ D[1:k + 1]).T
        coefs[:, 0] += D[0]
        self.t_old, self.coefs = t, coefs
        self.t, self.y, self.yp = tnew, ynew, (psi + d) / c
        self.done = last
        if self.n_equal < k + 1 or last:
            return True
        em = ctl.wnorm(self.erconst[k - 1] * D[k], w) if k > 1 else np.inf
        ep = ctl.wnorm(self.erconst[k + 1] * D[k + 2], w) if k < self.max_order else np.inf

        def fac(e, p):
            return 10.0 if e == 0 else (0.0 if not np.isfinite(e) else min(10.0, e ** (-1.0 / p)))

        factors = np.array([fac(em, k) if k > 1 else 0.0, fac(err, k + 1),
                            fac(ep, k + 2) if k < self.max_order else 0.0])
        pick = int(np.argmax(factors))
        self.order = k + pick - 1
        factor = min(10.0, safety * float(factors[pick]))
        if factor >= 1.2 or factor < 0.9 or pick != 1:
            self._change(max(factor, 0.2))
        else:
            self.n_equal = k + 1  # keep the step; look again next time
        return True

    def set_state(self, y):
        """Overwrite the current solution (used by the NonNegative clip)."""
        delta = y - self.y
        self.y = y
        self.D[0] += delta


# -- one-step implicit methods: trapezoidal rule and TR-BDF2 -------------------------------------

class ImplicitOneStep:
    """Trapezoidal rule (``tr``) or the TR-BDF2 pair (``trbdf2``) for ``G(t, y, y') = 0`` where
    G is linear in y' (mass-matrix form). The continuous extension is a cubic Hermite polynomial
    for TR-BDF2 and a parabola through three solution values for the trapezoidal rule."""

    GAMMA = 2.0 - np.sqrt(2.0)

    def __init__(self, system: ImplicitSystem, t0, y0, yp0, ctl: Control, counters: Counters, method="tr"):
        self.sys, self.ctl, self.cnt, self.method = system, ctl, counters, method
        self.t = float(t0)
        self.y = np.array(y0, dtype=np.result_type(y0.dtype, yp0.dtype, float), copy=True)
        self.yp = np.array(yp0, dtype=self.y.dtype, copy=True)
        self.absh = ctl.first_step(self.t, self.y, self.yp, 1.0 / 3, 0.8)
        self.lin = _Linear(system, counters)
        self.lin.refresh(self.t, self.y, self.yp, self.absh)
        self.done = False
        self.t_old = self.t
        self.coefs = None
        self.hist: list = []  # up to two earlier (t, y) points, for the trapezoidal error estimate

    def step(self):
        ctl, sysm = self.ctl, self.sys
        t, y, yp = self.t, self.y, self.yp
        hmin = ctl.hmin(t)
        absh = min(ctl.hmax, max(hmin, self.absh))
        tol = _newton_tol(ctl.rtol)
        g = self.GAMMA
        nofailed = True
        while True:
            h = ctl.tdir * absh
            last = False
            if 1.1 * absh >= abs(ctl.tfinal - t):
                h = ctl.tfinal - t
                absh = abs(h)
                last = True
            tnew = ctl.tfinal if last else t + h
            if absh < hmin:
                raise StepFailure(
                    f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                    f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
            w = ctl.weights(y)
            ok = False
            self.lin.keep_current(t, y, yp, h)
            while True:
                try:
                    if self.method == "tr":
                        alpha = 2.0 / h
                        self.lin.factor(alpha)
                        beta = -alpha * y - yp
                        # predictor from earlier solution values: the slope of the trapezoidal rule
                        # oscillates on stiff components, so y + h*yp would be a poor guess
                        if len(self.hist) == 2:
                            (ta, ya), (tb, yb) = self.hist
                            d1 = (y - yb) / (t - tb)
                            d2 = (d1 - (yb - ya) / (tb - ta)) / (t - ta)
                            guess = y + d1 * h + d2 * h * (tnew - tb)
                        elif len(self.hist) == 1:
                            (tb, yb), = self.hist
                            guess = y + (y - yb) / (t - tb) * h
                        else:
                            guess = y + h * yp
                        ok, ynew, _ = _newton(sysm, self.lin, ctl, tnew, guess, alpha, beta, w, tol)
                        ypnew = alpha * ynew + beta
                    else:
                        alpha = 2.0 / (g * h)
                        self.lin.factor(alpha)
                        beta = -alpha * y - yp
                        ok, yg, _ = _newton(sysm, self.lin, ctl, t + g * h, y + g * h * yp, alpha, beta, w, tol)
                        if ok:
                            ypg = alpha * yg + beta
                            a = 1.0 / (g * (2 - g))
                            b = (1 - g) ** 2 / (g * (2 - g))
                            beta2 = alpha * (-a * yg + b * y)
                            # predictor: the cubic through the first stage, extended to the end of the step
                            s = 1.0 / g
                            dyg = yg - y
                            hg = g * h
                            guess = y + s * hg * yp + s * s * (3 * dyg - hg * (2 * yp + ypg)) \
                                + s ** 3 * (-2 * dyg + hg * (yp + ypg))
                            ok, ynew, _ = _newton(sysm, self.lin, ctl, tnew, guess, alpha, beta2, w, tol)
                            if ok:
                                ypnew = alpha * ynew + beta2
                except StepFailure:
                    ok = False
                if ok or self.lin.fresh or sysm.const:
                    break
                self.lin.refresh(t, y, yp, h)
            if not ok:
                self.cnt.nfailed += 1
                self.lin.fresh = False
                nofailed = False
                absh = max(hmin, 0.3 * absh)
                if absh <= hmin:
                    raise StepFailure(
                        f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                        f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
                continue
            M = self.lin.mats[1]
            if self.method == "tr" and len(self.hist) == 2:
                # local error h^3/12 times the third derivative, from the third divided difference of y
                (ta, ya), (tb, yb) = self.hist
                d1 = (yb - ya) / (tb - ta)
                d2 = (y - yb) / (t - tb)
                d3 = (ynew - y) / (tnew - t)
                dd2a = (d2 - d1) / (t - ta)
                dd2b = (d3 - d2) / (tnew - tb)
                e = (abs(h) ** 3 / 2.0) * (dd2b - dd2a) / (tnew - ta)
                e = alpha * self.lin.solve(M @ e)  # damp the stiff (ringing) part of the estimate
            elif self.method == "tr":
                # start-up: compare with Simpson's rule on the Hermite midpoint, damped for stiff parts
                ymid = 0.5 * (y + ynew) + h / 8.0 * (yp - ypnew)
                fmid = -sysm.G(t + 0.5 * h, ymid, np.zeros_like(yp))
                raw = (h / 3.0) * (2 * fmid - M @ (yp + ypnew))
                e = alpha * self.lin.solve(raw)
            else:
                r2 = np.sqrt(2.0)
                raw = M @ (h * ((1 - r2) / 3.0 * yp + ypg / 3.0 - (2 - r2) / 3.0 * ypnew))
                e = alpha * self.lin.solve(raw)
            if np.all(np.isfinite(ynew)):
                err = ctl.wnorm(e, ctl.weights(y, ynew))
            else:
                err = np.inf
            if err > 1:
                self.cnt.nfailed += 1
                if absh <= hmin:
                    raise StepFailure(
                        f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                        f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
                if nofailed and np.isfinite(err):
                    absh = max(hmin, absh * max(0.1, 0.7 * err ** (-1.0 / 3)))
                else:
                    absh = max(hmin, 0.5 * absh)
                nofailed = False
                continue
            break
        self.cnt.nsteps += 1
        self.lin.fresh = False
        dy = ynew - y
        coefs = np.zeros((y.size, 4), dtype=self.y.dtype)
        coefs[:, 0] = y
        if self.method == "tr" and self.hist:
            # the parabola through the last three solution values (order 2, like the method, and
            # free of the slope oscillation on stiff components)
            tb, yb = self.hist[-1]
            dd = (dy / h - (y - yb) / (t - tb)) / (tnew - tb)
            coefs[:, 1] = dy - dd * h * h
            coefs[:, 2] = dd * h * h
        else:
            coefs[:, 1] = h * yp
            coefs[:, 2] = 3 * dy - h * (2 * yp + ypnew)
            coefs[:, 3] = -2 * dy + h * (yp + ypnew)
        self.hist = (self.hist + [(t, y)])[-2:]
        self.t_old, self.coefs = t, coefs
        self.t, self.y, self.yp = tnew, ynew, ypnew
        self.done = last
        if nofailed:
            temp = 1.25 * err ** (1.0 / 3)
            absh = absh / temp if temp > 0.2 else 5.0 * absh
        self.absh = absh
        return True

    def set_state(self, y):
        self.y = y


# -- linearly implicit Rosenbrock (2,3) pair -----------------------------------------------------

class Rosenbrock23:
    """The modified Rosenbrock triple of order 2 with a 3rd-order error estimate, for
    ``M y' = f(t, y)`` with a constant mass matrix. ``jac(t, y, f0)`` returns (df/dy, df/dt)."""

    D = 1.0 / (2.0 + np.sqrt(2.0))
    E32 = 6.0 + np.sqrt(2.0)

    def __init__(self, f, jac, mass, t0, y0, yp0, ctl: Control, counters: Counters):
        self.f, self.jac, self.M, self.ctl, self.cnt = f, jac, mass, ctl, counters
        self.t = float(t0)
        self.y = np.array(y0, dtype=np.result_type(y0.dtype, float), copy=True)
        self.F = self._f(self.t, self.y)
        self.yp = np.array(yp0, copy=True)
        self.absh = ctl.first_step(self.t, self.y, self.yp, 1.0 / 3, 0.8)
        self.done = False
        self.t_old = self.t
        self.coefs = None

    def _f(self, t, y):
        self.cnt.nfevals += 1
        return self.f(t, y)

    def step(self):
        from scipy.linalg import lu_factor, lu_solve
        ctl = self.ctl
        t, y, F0 = self.t, self.y, self.F
        n = y.size
        d, e32 = self.D, self.E32
        M = np.eye(n) if self.M is None else self.M
        hmin = ctl.hmin(t)
        absh = min(ctl.hmax, max(hmin, self.absh))
        J, T = self.jac(t, y, F0)
        self.cnt.npds += 1
        nofailed = True
        while True:
            h = ctl.tdir * absh
            last = False
            if 1.1 * absh >= abs(ctl.tfinal - t):
                h = ctl.tfinal - t
                absh = abs(h)
                last = True
            tnew = ctl.tfinal if last else t + h
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lu = lu_factor(M - h * d * J, check_finite=False)
            self.cnt.ndecomps += 1
            k1 = lu_solve(lu, F0 + h * d * T, check_finite=False)
            F1 = self._f(t + 0.5 * h, y + 0.5 * h * k1)
            k2 = lu_solve(lu, F1 - M @ k1, check_finite=False) + k1
            ynew = y + h * k2
            F2 = self._f(tnew, ynew)
            k3 = lu_solve(lu, F2 - e32 * (M @ k2 - F1) - 2 * (M @ k1 - F0) + h * d * T, check_finite=False)
            self.cnt.nsolves += 3
            if np.all(np.isfinite(ynew)) and np.all(np.isfinite(k3)):
                err = (absh / 6.0) * ctl.err_norm(k1 - 2 * k2 + k3, y, ynew)
            else:
                err = np.inf
            if err > ctl.rtol:
                self.cnt.nfailed += 1
                if absh <= hmin:
                    raise StepFailure(
                        f"Failure at t={t:.6e}. Unable to meet integration tolerances without reducing "
                        f"the step size below the smallest value allowed ({hmin:.6e}) at time t.")
                if nofailed and np.isfinite(err):
                    absh = max(hmin, absh * max(0.1, 0.8 * (ctl.rtol / err) ** (1.0 / 3)))
                else:
                    absh = max(hmin, 0.5 * absh)
                nofailed = False
                continue
            break
        self.cnt.nsteps += 1
        coefs = np.empty((n, 3), dtype=np.result_type(y.dtype, k1.dtype))
        coefs[:, 0] = y
        coefs[:, 1] = h * (k1 - 2 * d * k2) / (1 - 2 * d)
        coefs[:, 2] = h * (k2 - k1) / (1 - 2 * d)
        self.t_old, self.coefs = t, coefs
        self.t, self.y, self.F = tnew, ynew, F2
        self.yp = k2 if self.M is None else None
        self.done = last
        if nofailed:
            temp = 1.25 * (err / ctl.rtol) ** (1.0 / 3)
            absh = absh / temp if temp > 0.2 else 5.0 * absh
        self.absh = absh
        return True

    def set_state(self, y):
        self.y = y
        self.F = self._f(self.t, y)


# -- evaluation of stored continuous extensions ---------------------------------------------------

def eval_piece(coefs: np.ndarray, s) -> np.ndarray:
    """Values (n x numel(s)) of one step's polynomial at relative positions ``s``."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.zeros((coefs.shape[0], s.size), dtype=coefs.dtype)
    for j in range(coefs.shape[1] - 1, -1, -1):
        out = out * s + coefs[:, j:j + 1]
    return out


def eval_piece_deriv(coefs: np.ndarray, s) -> np.ndarray:
    """d/ds of one step's polynomial at relative positions ``s``."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.zeros((coefs.shape[0], s.size), dtype=coefs.dtype)
    for j in range(coefs.shape[1] - 1, 0, -1):
        out = out * s + j * coefs[:, j:j + 1]
    return out
