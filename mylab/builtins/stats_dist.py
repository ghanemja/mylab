"""Probability distribution families.

One table of families drives the ``<prefix>pdf``, ``<prefix>cdf``, ``<prefix>inv``, ``<prefix>rnd``,
``<prefix>stat``, ``<prefix>fit`` and ``<prefix>like`` functions and the by-name functions ``pdf``,
``cdf``, ``icdf``, ``random`` and ``mle``. Parameter conventions follow the MATLAB language (for
example the exponential distribution is parameterised by its mean and the Weibull distribution by
scale then shape); the numerical work is delegated to SciPy.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import Struct
from ._registry import builtin
from ._stats_util import (
    as_bool, call_num, get_opts, given, is_empty_arg, num_hessian, outs, rng_of, scalar, vec,
)
from ._util import cast_class, dims_from_args, fnum, is_text, text

_BIG = 1e300


def bcast(name: str, *vals) -> tuple[list[np.ndarray], tuple]:
    """Broadcast scalar/array arguments (non-scalars must agree in size)."""
    arrs = [np.asarray(fnum(v), dtype=float) for v in vals]
    shapes = {a.shape for a in arrs if a.size != 1}
    if len(shapes) > 1:
        raise MatlabError(f"{name}: non-scalar arguments must match in size.")
    shape = shapes.pop() if shapes else (1, 1)
    return [np.broadcast_to(a.reshape(1, 1) if a.size == 1 else a, shape) for a in arrs], shape


class Fam:
    """One distribution family: parameter names and the mapping onto a SciPy distribution."""

    def __init__(self, name: str, prefix: str | None, pnames: tuple, sp: str, conv: Callable, valid: Callable,
                 safe: tuple, kinds: tuple, *, defaults: tuple | None = None, discrete: bool = False,
                 aliases: tuple = (), est: tuple | None = None, start: Callable | None = None,
                 closed: Callable | None = None, logci: tuple | None = None, sampler: Callable | None = None,
                 desc: tuple | None = None, display: str | None = None):
        self.name = name                    # e.g. 'Normal', 'GeneralizedExtremeValue'
        self.prefix = prefix                # e.g. 'norm' (None: no <prefix>pdf functions)
        self.pnames = pnames
        self.sp = sp
        self.conv = conv
        self.valid = valid
        self.safe = safe                    # valid parameter values (makedist defaults)
        self.kinds = kinds                  # 'real', 'pos', 'unit' or 'fixed' for each parameter
        self.defaults = defaults if defaults is not None else (None,) * len(pnames)
        self.discrete = discrete
        self.aliases = aliases
        self.est = est if est is not None else tuple(i for i, k in enumerate(kinds) if k != "fixed")
        self.start = start
        self.closed = closed
        self.logci = logci if logci is not None else (False,) * len(pnames)
        self.sampler = sampler
        self.desc = desc or pnames
        self.display = display or name

    # -- evaluation ------------------------------------------------------------------------------
    def ev(self, method: str, x: Any, p: list) -> np.ndarray:
        """Evaluate pdf/cdf/sf/ppf/logpdf/logsf/logcdf elementwise; invalid parameters give NaN."""
        import scipy.stats as st
        d = getattr(st, self.sp)
        x = np.asarray(x, dtype=float)
        p = [np.asarray(q, dtype=float) for q in p]
        ok = np.asarray(self.valid(*p), dtype=bool)
        ps = [np.where(ok, q, s) for q, s in zip(p, self.safe)]
        with np.errstate(all="ignore"):
            if self.discrete:
                args, loc = self.conv(*ps)
                fn = {"pdf": d.pmf, "logpdf": d.logpmf}.get(method) or getattr(d, method)
                r = fn(x, *args, loc=loc)
            else:
                args, loc, scale = self.conv(*ps)
                r = getattr(d, method)(x, *args, loc=loc, scale=scale)
        return np.where(ok & ~np.isnan(x), r, np.nan)

    def stat(self, p: list) -> tuple[np.ndarray, np.ndarray]:
        import scipy.stats as st
        d = getattr(st, self.sp)
        p = [np.asarray(q, dtype=float) for q in p]
        ok = np.asarray(self.valid(*p), dtype=bool)
        ps = [np.where(ok, q, s) for q, s in zip(p, self.safe)]
        with np.errstate(all="ignore"):
            if self.discrete:
                args, loc = self.conv(*ps)
                m, v = d.stats(*args, loc=loc, moments="mv")
            else:
                args, loc, scale = self.conv(*ps)
                m, v = d.stats(*args, loc=loc, scale=scale, moments="mv")
        shape = np.broadcast(*p).shape if p else ()
        m = np.broadcast_to(np.asarray(m, dtype=float), shape)
        v = np.broadcast_to(np.asarray(v, dtype=float), shape)
        return np.where(ok, m, np.nan), np.where(ok, v, np.nan)

    def rnd(self, rng: np.random.Generator, shape: tuple, p: list) -> np.ndarray:
        p = [np.broadcast_to(np.asarray(q, dtype=float), shape) for q in p]
        ok = np.asarray(self.valid(*p), dtype=bool)
        if self.sampler is not None:
            ps = [np.where(ok, q, s) for q, s in zip(p, self.safe)]
            r = np.asarray(self.sampler(rng, shape, *ps), dtype=float)
            return np.where(ok, r, np.nan)
        return self.ev("ppf", rng.random(shape), p)

    # -- likelihood and fitting ---------------------------------------------------------------------
    def nll(self, params, x: np.ndarray, cens: np.ndarray | None = None, freq: np.ndarray | None = None) -> float:
        p = [np.asarray(q, dtype=float) for q in params]
        ll = self.ev("logpdf", x, p)
        if cens is not None and cens.any():
            ll = np.where(cens, self.ev("logsf", x, p), ll)
        if freq is not None:
            ll = ll * freq
        s = -float(np.sum(ll))
        return s if np.isfinite(s) else np.inf

    def fit(self, x: np.ndarray, cens: np.ndarray | None = None, freq: np.ndarray | None = None,
            fixed: dict | None = None, start: Any = None, unbiased: bool = True) -> np.ndarray:
        """Maximum-likelihood parameter estimates (all parameters, fixed ones included)."""
        fixed = dict(fixed or {})
        if x.size == 0:
            raise MatlabError(f"Cannot fit a {self.display} distribution to empty data.")
        has_cens = cens is not None and cens.any()
        if has_cens and self.name == "Exponential" and start is None and not np.all(cens):
            f = np.ones(x.size) if freq is None else freq       # total time on test over the number of failures
            return np.array([float(np.sum(f * x) / np.sum(f * ~cens))])
        if self.closed is not None and not has_cens and start is None:
            xe = x if freq is None else np.repeat(x, np.round(freq).astype(int)) \
                if np.all(freq == np.round(freq)) else None
            if xe is not None:
                return np.asarray(self.closed(xe, fixed, unbiased), dtype=float)
        if start is not None:
            theta0 = np.asarray(start, dtype=float).flatten()
            if theta0.size == len(self.est):
                full = np.array(self.safe, dtype=float)
                full[list(self.est)] = theta0
                theta0 = full
            elif theta0.size != len(self.pnames):
                raise MatlabError(f"Start must have {len(self.est)} values for the {self.display} distribution.")
        elif self.closed is not None:
            theta0 = np.asarray(self.closed(x, fixed, False), dtype=float)
        elif self.start is not None:
            theta0 = np.asarray(self.start(x, fixed), dtype=float)
        else:
            raise MatlabError(f"Fitting the {self.display} distribution is not supported.")
        for k, v in fixed.items():
            theta0[k] = v
        est = [i for i in self.est if i not in fixed]
        return self._optimise(theta0, est, x, cens, freq)

    def _optimise(self, theta0: np.ndarray, est: list, x, cens, freq) -> np.ndarray:
        from scipy.optimize import minimize
        kinds = [self.kinds[i] for i in est]

        def to_u(t):
            return np.array([np.log(max(v, 1e-300)) if k == "pos" else
                             (np.log(min(max(v, 1e-12), 1 - 1e-12) / (1 - min(max(v, 1e-12), 1 - 1e-12)))
                              if k == "unit" else v) for v, k in zip(t, kinds)])

        def from_u(u):
            return np.array([np.exp(min(v, 700.0)) if k == "pos" else (1 / (1 + np.exp(-v)) if k == "unit" else v)
                             for v, k in zip(u, kinds)])

        def obj(u):
            th = theta0.copy()
            th[est] = from_u(u)
            v = self.nll(th, x, cens, freq)
            return v if np.isfinite(v) else _BIG

        u0 = to_u(theta0[est])
        best = minimize(obj, u0, method="Nelder-Mead",
                        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 800 * len(est), "maxfev": 1600 * len(est)})
        try:
            pol = minimize(obj, best.x, method="BFGS", options={"gtol": 1e-8})
            if pol.fun < best.fun:
                best = pol
        except (ValueError, FloatingPointError):  # pragma: no cover - polishing is optional
            pass
        out = theta0.copy()
        out[est] = from_u(best.x)
        return out

    def acov(self, params: np.ndarray, x, cens=None, freq=None, fixed: dict | None = None) -> np.ndarray:
        """Asymptotic covariance of the estimated parameters (zero rows/columns for fixed ones)."""
        est = [i for i in self.est if i not in (fixed or {})]
        k = len(self.pnames)
        C = np.zeros((k, k))
        if not est:
            return C
        base = np.asarray(params, dtype=float)

        def f(t):
            th = base.copy()
            th[est] = t
            v = self.nll(th, x, cens, freq)
            return v if np.isfinite(v) else np.nan

        spread = float(np.std(x)) if x.size > 1 and np.std(x) > 0 else 1.0
        steps = np.array([1e-3 * (max(abs(base[i]), spread) if self.kinds[i] == "real" else abs(base[i]) or 1e-3)
                          for i in est])
        with np.errstate(all="ignore"):
            H = num_hessian(f, base[est], steps=steps)
            try:
                Ci = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                Ci = np.full((len(est), len(est)), np.nan)
        C[np.ix_(est, est)] = Ci
        return C


# -- exact confidence intervals for the families that have them ---------------------------------------

def _ci_normal(params, x, alpha, n):
    import scipy.stats as st
    mu, s = params[0], params[1]
    t = st.t.ppf(1 - alpha / 2, n - 1) if n > 1 else np.nan
    with np.errstate(all="ignore"):
        lo_s = s * np.sqrt((n - 1) / st.chi2.ppf(1 - alpha / 2, n - 1)) if n > 1 else np.nan
        hi_s = s * np.sqrt((n - 1) / st.chi2.ppf(alpha / 2, n - 1)) if n > 1 else np.nan
    return np.array([[mu - t * s / np.sqrt(n), lo_s], [mu + t * s / np.sqrt(n), hi_s]])


def _ci_exp(params, x, alpha, n):
    import scipy.stats as st
    mu = params[0]
    return np.array([[2 * n * mu / st.chi2.ppf(1 - alpha / 2, 2 * n)], [2 * n * mu / st.chi2.ppf(alpha / 2, 2 * n)]])


def _ci_poisson(params, x, alpha, n):
    import scipy.stats as st
    s = params[0] * n
    lo = st.chi2.ppf(alpha / 2, 2 * s) / 2 if s > 0 else 0.0
    return np.array([[lo / n], [st.chi2.ppf(1 - alpha / 2, 2 * (s + 1)) / 2 / n]])


def _ci_rayleigh(params, x, alpha, n):
    import scipy.stats as st
    ss = 2 * n * params[0] ** 2
    return np.array([[np.sqrt(ss / st.chi2.ppf(1 - alpha / 2, 2 * n))], [np.sqrt(ss / st.chi2.ppf(alpha / 2, 2 * n))]])


def _ci_binomial(params, x, alpha, n):
    N, p = params[0], params[1]
    lo, hi = clopper_pearson(np.array([p * N * n]), np.array([N * n]), alpha)
    return np.array([[N, lo[0]], [N, hi[0]]])


def clopper_pearson(x: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    import scipy.stats as st
    with np.errstate(all="ignore"):
        lo = np.where(x > 0, st.beta.ppf(alpha / 2, x, n - x + 1), 0.0)
        hi = np.where(x < n, st.beta.ppf(1 - alpha / 2, x + 1, n - x), 1.0)
    return lo, hi


EXACT_CI = {"normal": _ci_normal, "lognormal": _ci_normal, "exponential": _ci_exp, "poisson": _ci_poisson,
            "rayleigh": _ci_rayleigh, "binomial": _ci_binomial}


def wald_ci(fam: Fam, params: np.ndarray, cov: np.ndarray, alpha: float) -> np.ndarray:
    import scipy.stats as st
    z = st.norm.ppf(1 - alpha / 2)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ci = np.vstack([params - z * se, params + z * se])
    for i, flag in enumerate(fam.logci):
        if flag and params[i] > 0:
            ci[:, i] = params[i] * np.exp(np.array([-z, z]) * se[i] / params[i])
    return ci


def param_ci(fam: Fam, params: np.ndarray, x: np.ndarray, alpha: float, cov: np.ndarray | None,
             n: float | None = None, exact: bool = True, cens=None) -> np.ndarray:
    key = _key(fam.name)
    n = float(x.size if n is None else n)
    if exact and key in EXACT_CI and not (cens is not None and cens.any()):
        return EXACT_CI[key](params, x, alpha, n)
    if key == "weibull" and cov is not None:
        # Work on the extreme-value scale of log(data), where the interval is symmetric.
        import scipy.stats as st
        z = st.norm.ppf(1 - alpha / 2)
        A, B = params
        J = np.array([[1 / A, 0], [0, -1 / B ** 2]])
        ce = J @ cov @ J.T
        mu, sg = np.log(A), 1 / B
        se = np.sqrt(np.clip(np.diag(ce), 0, None))
        mu_ci = np.array([mu - z * se[0], mu + z * se[0]])
        sg_ci = sg * np.exp(np.array([-z, z]) * se[1] / sg)
        return np.array([[np.exp(mu_ci[0]), 1 / sg_ci[1]], [np.exp(mu_ci[1]), 1 / sg_ci[0]]])
    if cov is None:
        return np.full((2, len(fam.pnames)), np.nan)
    return wald_ci(fam, params, cov, alpha)


# -- closed-form estimators and starting values -----------------------------------------------------------

def _sd(x, unbiased):
    return float(np.std(x, ddof=1 if (unbiased and x.size > 1) else 0))


def _sp_fit(name, x, **kw):
    import warnings

    import scipy.stats as st
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return getattr(st, name).fit(x, **kw)


def _need_pos(x, what, strict=True):
    if np.any(x < 0) or (strict and np.any(x <= 0)):
        raise MatlabError(f"The data must be {'positive' if strict else 'non-negative'} to fit a {what} distribution.")


def _start_gamma(x, fixed):
    _need_pos(x, "gamma", strict=False)
    xp = np.where(x > 0, x, np.min(x[x > 0]) * 1e-3 if np.any(x > 0) else 1e-3)
    s = np.log(xp.mean()) - np.mean(np.log(xp))
    a = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s) if s > 0 else 1.0
    return [a, xp.mean() / a]


def _start_weibull(x, fixed):
    _need_pos(x, "Weibull")
    lx = np.log(x)
    sg = max(np.std(lx) * np.sqrt(6) / np.pi, 1e-6)
    return [np.exp(lx.mean() + 0.5772156649 * sg), 1 / sg]


def _start_beta(x, fixed):
    if np.any((x <= 0) | (x >= 1)):
        raise MatlabError("The data must be strictly between 0 and 1 to fit a beta distribution.")
    m, v = x.mean(), max(x.var(), 1e-12)
    c = max(m * (1 - m) / v - 1, 1e-3)
    return [m * c, (1 - m) * c]


def _start_ev(x, fixed):
    sg = max(np.std(x) * np.sqrt(6) / np.pi, 1e-8)
    return [x.mean() + 0.5772156649 * sg, sg]


def _start_gev(x, fixed):
    sg = max(np.std(x) * np.sqrt(6) / np.pi, 1e-8)
    return [0.01, sg, x.mean() - 0.5772156649 * sg]


def _start_gp(x, fixed):
    theta = fixed.get(2, 0.0)
    y = x - theta
    if np.any(y <= 0):
        raise MatlabError("The data must exceed the threshold parameter to fit a generalized Pareto distribution.")
    m, v = y.mean(), max(y.var(), 1e-12)
    k = 0.5 * (1 - m * m / v)
    return [min(max(k, -0.4), 0.9), max(0.5 * m * (m * m / v + 1), 1e-8), theta]


def _start_nbin(x, fixed):
    m, v = x.mean(), x.var()
    if v <= m:
        raise MatlabError("The data are not overdispersed (variance <= mean); the negative binomial "
                          "distribution cannot be fitted.")
    return [m * m / (v - m), m / v]


def _start_logistic(x, fixed):
    return [float(np.median(x)), max(np.std(x) * np.sqrt(3) / np.pi, 1e-8)]


def _start_loglogistic(x, fixed):
    _need_pos(x, "loglogistic")
    return _start_logistic(np.log(x), fixed)


def _start_nakagami(x, fixed):
    _need_pos(x, "Nakagami")
    x2 = x ** 2
    om = x2.mean()
    return [max(om * om / max(x2.var(), 1e-12), 0.5), om]


def _start_rician(x, fixed):
    _need_pos(x, "Rician", strict=False)
    b, _, sc = _sp_fit("rice", x, floc=0)
    return [b * sc, sc]


def _closed_invgauss(x, fixed, unbiased):
    _need_pos(x, "inverse Gaussian")
    mu = x.mean()
    return [mu, x.size / np.sum(1 / x - 1 / mu)]


def _start_bs(x, fixed):
    _need_pos(x, "Birnbaum-Saunders")
    s, r = x.mean(), 1 / np.mean(1 / x)
    return [np.sqrt(s * r), np.sqrt(max(2 * (np.sqrt(s / r) - 1), 1e-8))]


def _start_burr(x, fixed):
    _need_pos(x, "Burr")
    c, k, _, sc = _sp_fit("burr12", x, floc=0)
    return [sc, c, k]


def _start_tls(x, fixed):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) / 0.6745 or max(np.std(x), 1e-8)
    return [med, mad, 5.0]


def _closed_normal(x, fixed, unbiased):
    return [x.mean(), _sd(x, unbiased)]


def _closed_lognormal(x, fixed, unbiased):
    _need_pos(x, "lognormal")
    lx = np.log(x)
    return [lx.mean(), _sd(lx, unbiased)]


def _closed_binomial(x, fixed, unbiased):
    if 0 not in fixed:
        raise MatlabError("Fitting a binomial distribution needs the number of trials ('NTrials').")
    N = float(fixed[0])
    if np.any((x < 0) | (x > N) | (x != np.round(x))):
        raise MatlabError("Binomial data must be integers between 0 and the number of trials.")
    return [N, x.mean() / N]


def _closed_halfnormal(x, fixed, unbiased):
    mu = float(fixed.get(0, 0.0))
    if np.any(x < mu):
        raise MatlabError("The data must not be below the location parameter of the half-normal distribution.")
    return [mu, np.sqrt(np.mean((x - mu) ** 2))]


def _closed_exp(x, fixed, unbiased):
    _need_pos(x, "exponential", strict=False)
    return [x.mean()]


def _closed_rayleigh(x, fixed, unbiased):
    _need_pos(x, "Rayleigh", strict=False)
    return [np.sqrt(np.mean(x ** 2) / 2)]


def _closed_poisson(x, fixed, unbiased):
    if np.any((x < 0) | (x != np.round(x))):
        raise MatlabError("Poisson data must be non-negative integers.")
    return [x.mean()]


def _closed_geometric(x, fixed, unbiased):
    if np.any((x < 0) | (x != np.round(x))):
        raise MatlabError("Geometric data must be non-negative integers.")
    return [1 / (1 + x.mean())]


# -- the family table -----------------------------------------------------------------------------------

def _pos(*a):
    r = np.ones(np.broadcast(*a).shape, dtype=bool)
    for q in a:
        r = r & (q > 0)
    return r


def _is_int(a):
    return a == np.round(a)


_G = 0.5772156649015329


def _families() -> list[Fam]:
    F = Fam
    return [
        F("Normal", "norm", ("mu", "sigma"), "norm", lambda m, s: ((), m, s), lambda m, s: s > 0, (0, 1),
          ("real", "pos"), defaults=(0, 1), aliases=("gaussian",), closed=_closed_normal,
          sampler=lambda g, sh, m, s: m + s * g.standard_normal(sh),
          desc=("location", "scale")),
        F("Lognormal", "logn", ("mu", "sigma"), "lognorm", lambda m, s: ((s,), 0, np.exp(m)), lambda m, s: s > 0,
          (0, 1), ("real", "pos"), defaults=(0, 1), closed=_closed_lognormal,
          sampler=lambda g, sh, m, s: np.exp(m + s * g.standard_normal(sh)), desc=("log location", "log scale")),
        F("Exponential", "exp", ("mu",), "expon", lambda m: ((), 0, m), lambda m: m > 0, (1,), ("pos",),
          defaults=(1,), closed=_closed_exp, sampler=lambda g, sh, m: m * g.standard_exponential(sh),
          desc=("mean",)),
        F("Gamma", "gam", ("a", "b"), "gamma", lambda a, b: ((a,), 0, b), _pos, (1, 1), ("pos", "pos"),
          defaults=(None, 1), start=_start_gamma, logci=(True, True),
          sampler=lambda g, sh, a, b: b * g.standard_gamma(a, sh),
          desc=("shape", "scale")),
        F("Beta", "beta", ("a", "b"), "beta", lambda a, b: ((a, b), 0, 1), _pos, (1, 1), ("pos", "pos"),
          start=_start_beta, logci=(True, True), sampler=lambda g, sh, a, b: g.beta(a, b, sh)),
        F("Weibull", "wbl", ("A", "B"), "weibull_min", lambda a, b: ((b,), 0, a), _pos, (1, 1), ("pos", "pos"),
          defaults=(1, 1), aliases=("wbl", "weib"), start=_start_weibull,
          sampler=lambda g, sh, a, b: a * g.weibull(b, sh), desc=("scale", "shape")),
        F("Uniform", "unif", ("Lower", "Upper"), "uniform", lambda a, b: ((), a, b - a), lambda a, b: a < b,
          (0, 1), ("real", "real"), defaults=(0, 1),
          closed=lambda x, fixed, unbiased: [x.min(), x.max()],
          sampler=lambda g, sh, a, b: a + (b - a) * g.random(sh), desc=("lower limit", "upper limit")),
        F("DiscreteUniform", "unid", ("N",), "randint", lambda n: ((1, n + 1), 0),
          lambda n: (n >= 1) & _is_int(n), (1,), ("fixed",), discrete=True, aliases=("discreteuniform",),
          est=(0,), closed=lambda x, fixed, unbiased: [x.max()],
          sampler=lambda g, sh, n: np.floor(g.random(sh) * n) + 1, display="discrete uniform"),
        F("Binomial", "bino", ("N", "p"), "binom", lambda n, p: ((n, p), 0),
          lambda n, p: (n >= 0) & _is_int(n) & (p >= 0) & (p <= 1), (1, 0.5), ("fixed", "unit"), discrete=True,
          closed=_closed_binomial, sampler=lambda g, sh, n, p: g.binomial(n.astype(np.int64), p, sh),
          desc=("number of trials", "probability of success")),
        F("Poisson", "poiss", ("lambda",), "poisson", lambda lam: ((lam,), 0), lambda lam: lam >= 0, (1,),
          ("pos",), discrete=True, closed=_closed_poisson, sampler=lambda g, sh, lam: g.poisson(lam, sh),
          desc=("mean",)),
        F("Geometric", "geo", ("p",), "geom", lambda p: ((p,), -1), lambda p: (p > 0) & (p <= 1), (0.5,),
          ("unit",), discrete=True, closed=_closed_geometric, sampler=lambda g, sh, p: g.geometric(p, sh) - 1.0),
        F("NegativeBinomial", "nbin", ("R", "P"), "nbinom", lambda r, p: ((r, p), 0),
          lambda r, p: (r > 0) & (p > 0) & (p <= 1), (1, 0.5), ("pos", "unit"), discrete=True,
          aliases=("negativebinomial",), start=_start_nbin, sampler=lambda g, sh, r, p: g.negative_binomial(r, p, sh),
          display="negative binomial"),
        F("Hypergeometric", "hyge", ("M", "K", "N"), "hypergeom", lambda m, k, n: ((m, k, n), 0),
          lambda m, k, n: (m >= 0) & (k >= 0) & (n >= 0) & (k <= m) & (n <= m) & _is_int(m) & _is_int(k) & _is_int(n),
          (1, 1, 1), ("fixed", "fixed", "fixed"), discrete=True,
          sampler=lambda g, sh, m, k, n: g.hypergeometric(k.astype(np.int64), (m - k).astype(np.int64),
                                                          n.astype(np.int64), sh)),
        F("Rayleigh", "rayl", ("B",), "rayleigh", lambda b: ((), 0, b), _pos, (1,), ("pos",), defaults=(1,),
          closed=_closed_rayleigh, sampler=lambda g, sh, b: g.rayleigh(b, sh), desc=("scale",)),
        F("ExtremeValue", "ev", ("mu", "sigma"), "gumbel_l", lambda m, s: ((), m, s), lambda m, s: s > 0, (0, 1),
          ("real", "pos"), defaults=(0, 1), aliases=("extremevalue",), start=_start_ev, logci=(False, True),
          display="extreme value", desc=("location", "scale")),
        F("GeneralizedExtremeValue", "gev", ("k", "sigma", "mu"), "genextreme", lambda k, s, m: ((-k,), m, s),
          lambda k, s, m: s > 0, (0, 1, 0), ("real", "pos", "real"), defaults=(0, 1, 0),
          aliases=("generalizedextremevalue",), start=_start_gev, logci=(False, True, False),
          display="generalized extreme value", desc=("shape", "scale", "location")),
        F("GeneralizedPareto", "gp", ("k", "sigma", "theta"), "genpareto", lambda k, s, t: ((k,), t, s),
          lambda k, s, t: s > 0, (1, 1, 1), ("real", "pos", "fixed"), defaults=(0, 1, 0),
          aliases=("generalizedpareto",), start=_start_gp, logci=(False, True, False),
          display="generalized Pareto", desc=("shape", "scale", "threshold")),
        F("T", "t", ("nu",), "t", lambda v: ((v,), 0, 1), _pos, (1,), ("pos",),
          sampler=lambda g, sh, v: g.standard_t(v, sh), display="Student's t"),
        F("Chisquare", "chi2", ("nu",), "chi2", lambda v: ((v,), 0, 1), _pos, (1,), ("pos",),
          aliases=("chi-square", "chisquare"), sampler=lambda g, sh, v: g.chisquare(v, sh), display="chi-square"),
        F("F", "f", ("nu1", "nu2"), "f", lambda a, b: ((a, b), 0, 1), _pos, (1, 1), ("pos", "pos"),
          sampler=lambda g, sh, a, b: g.f(a, b, sh)),
        F("NoncentralT", "nct", ("nu", "delta"), "nct", lambda v, d: ((v, d), 0, 1), lambda v, d: v > 0, (1, 0),
          ("pos", "real"), aliases=("noncentralt",), display="noncentral t"),
        F("NoncentralF", "ncf", ("nu1", "nu2", "delta"), "ncf", lambda a, b, d: ((a, b, d), 0, 1),
          lambda a, b, d: (a > 0) & (b > 0) & (d >= 0), (1, 1, 0), ("pos", "pos", "pos"),
          aliases=("noncentralf",), display="noncentral F"),
        F("NoncentralChisquare", "ncx2", ("nu", "delta"), "ncx2", lambda v, d: ((v, d), 0, 1),
          lambda v, d: (v > 0) & (d >= 0), (1, 0), ("pos", "pos"),
          aliases=("noncentralchi-square", "noncentralchisquare"), display="noncentral chi-square"),
        F("Logistic", None, ("mu", "sigma"), "logistic", lambda m, s: ((), m, s), lambda m, s: s > 0, (0, 1),
          ("real", "pos"), start=_start_logistic, sampler=lambda g, sh, m, s: g.logistic(m, s, sh),
          desc=("mean", "scale")),
        F("Loglogistic", None, ("mu", "sigma"), "fisk", lambda m, s: ((1 / s,), 0, np.exp(m)), lambda m, s: s > 0,
          (0, 1), ("real", "pos"), start=_start_loglogistic, desc=("log mean", "log scale")),
        F("Nakagami", None, ("mu", "omega"), "nakagami", lambda m, w: ((m,), 0, np.sqrt(w)),
          lambda m, w: (m >= 0.5) & (w > 0), (1, 1), ("pos", "pos"), start=_start_nakagami,
          desc=("shape", "scale")),
        F("Rician", None, ("s", "sigma"), "rice", lambda s, sg: ((s / sg,), 0, sg), lambda s, sg: (s >= 0) & (sg > 0),
          (1, 1), ("pos", "pos"), start=_start_rician, desc=("noncentrality", "scale")),
        F("InverseGaussian", None, ("mu", "lambda"), "invgauss", lambda m, lam: ((m / lam,), 0, lam), _pos, (1, 1),
          ("pos", "pos"), closed=_closed_invgauss, display="inverse Gaussian", desc=("scale", "shape")),
        F("BirnbaumSaunders", None, ("beta", "gamma"), "fatiguelife", lambda b, g: ((g,), 0, b), _pos, (1, 1),
          ("pos", "pos"), start=_start_bs, display="Birnbaum-Saunders", desc=("scale", "shape")),
        F("Burr", None, ("alpha", "c", "k"), "burr12", lambda a, c, k: ((c, k), 0, a), _pos, (1, 1, 1),
          ("pos", "pos", "pos"), start=_start_burr, desc=("scale", "first shape", "second shape")),
        F("Triangular", None, ("A", "B", "C"), "triang",
          lambda a, b, c: (((b - a) / np.where(c > a, c - a, 1),), a, c - a),
          lambda a, b, c: (a <= b) & (b <= c) & (a < c), (0, 0.5, 1), ("fixed", "fixed", "fixed"),
          desc=("lower limit", "peak location", "upper limit")),
        F("tLocationScale", None, ("mu", "sigma", "nu"), "t", lambda m, s, v: ((v,), m, s),
          lambda m, s, v: (s > 0) & (v > 0), (0, 1, 5), ("real", "pos", "pos"), start=_start_tls,
          display="t location-scale", desc=("location", "scale", "degrees of freedom")),
        F("HalfNormal", "hn", ("mu", "sigma"), "halfnorm", lambda m, s: ((), m, s), lambda m, s: s > 0, (0, 1),
          ("fixed", "pos"), aliases=("halfnormal",), closed=_closed_halfnormal, display="half normal",
          desc=("location", "scale")),
        F("Loguniform", None, ("Lower", "Upper"), "loguniform", lambda a, b: ((a, b), 0, 1),
          lambda a, b: (a > 0) & (a < b), (1, 4), ("fixed", "fixed"), desc=("lower limit", "upper limit")),
    ]


FAMILIES: dict[str, Fam] = {}
_NAMES: dict[str, str] = {}


def _key(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _register_families() -> None:
    for fam in _families():
        k = _key(fam.name)
        FAMILIES[k] = fam
        _NAMES[k] = k
        if fam.prefix:
            _NAMES[_key(fam.prefix)] = k
        for a in fam.aliases:
            _NAMES[_key(a)] = k
    _NAMES.update({"exp": "exponential", "bino": "binomial", "wbl": "weibull", "student": "t", "studentst": "t",
                   "inversegaussian": "inversegaussian", "bisa": "birnbaumsaunders", "gaussian": "normal"})


_register_families()
# hn has no <prefix>pdf functions in the function-style interface.
FAMILIES["halfnormal"].prefix = None


def family(name: Any, fname: str = "distribution") -> Fam:
    if not is_text(name):
        raise MatlabError(f"{fname}: expected a distribution name.")
    k = _NAMES.get(_key(text(name)))
    if k is None:
        raise MatlabError(f"{fname}: unknown distribution '{text(name)}'.")
    return FAMILIES[k]


# -- <prefix>pdf / cdf / inv ----------------------------------------------------------------------------

def _eval_args(fam: Fam, fname: str, rest: list) -> tuple[list, tuple]:
    k = len(fam.pnames)
    nreq = sum(1 for d in fam.defaults if d is None)
    if len(rest) < 1 + nreq:
        raise MatlabError(f"Not enough input arguments for {fname}.")
    if len(rest) > 1 + k:
        raise MatlabError(f"Too many input arguments for {fname}.")
    vals = list(rest) + [np.array([[float(d)]]) for d in fam.defaults[len(rest) - 1:]]
    return bcast(fname, *vals)


def _dist_fn(fam: Fam, kind: str) -> Callable:
    fname = f"{fam.prefix}{kind}"

    def f(interp, args, nargout):
        rest = list(args)
        upper = False
        if kind == "cdf" and rest and is_text(rest[-1]):
            if text(rest[-1]).lower() != "upper":
                raise MatlabError(f"{fname}: the only text option is 'upper'.")
            upper = True
            rest.pop()
        if fam.prefix in ("norm", "exp") and kind in ("cdf", "inv") and len(rest) > 1 + len(fam.pnames):
            return _with_bounds(fam, kind, fname, rest, upper, nargout)
        if nargout > 1:
            raise MatlabError(f"{fname}: confidence bounds need the parameter covariance as an extra input.")
        arrs, shape = _eval_args(fam, fname, rest)
        method = {"pdf": "pdf", "cdf": "sf" if upper else "cdf", "inv": "ppf"}[kind]
        return fam.ev(method, arrs[0], arrs[1:]).reshape(shape)
    return f


def _with_bounds(fam: Fam, kind: str, fname: str, rest: list, upper: bool, nargout: int):
    """normcdf/norminv/expcdf/expinv with a parameter covariance: value plus confidence bounds."""
    import scipy.stats as st
    k = len(fam.pnames)
    pcov = np.asarray(fnum(rest[1 + k]), dtype=float)
    alpha = scalar(rest[2 + k], "alpha") if len(rest) > 2 + k else 0.05
    if len(rest) > 3 + k:
        raise MatlabError(f"Too many input arguments for {fname}.")
    arrs, shape = bcast(fname, *rest[:1 + k])
    x, p = arrs[0], arrs[1:]
    q = st.norm.ppf(1 - alpha / 2)
    with np.errstate(all="ignore"):
        if fam.prefix == "norm":
            if pcov.shape != (2, 2):
                raise MatlabError(f"{fname}: the parameter covariance must be 2-by-2.")
            mu, sg = p
            if kind == "cdf":
                z = (x - mu) / sg
                half = q * np.sqrt(pcov[0, 0] + 2 * pcov[0, 1] * z + pcov[1, 1] * z ** 2) / sg
                fn = st.norm.sf if upper else st.norm.cdf
                lo, hi = (z + half, z - half) if upper else (z - half, z + half)
                res = (fn(z), fn(lo), fn(hi))
            else:
                z = st.norm.ppf(x)
                val = mu + sg * z
                half = q * np.sqrt(pcov[0, 0] + 2 * pcov[0, 1] * z + pcov[1, 1] * z ** 2)
                res = (val, val - half, val + half)
        else:
            if pcov.size != 1:
                raise MatlabError(f"{fname}: the parameter variance must be a scalar.")
            mu = p[0]
            half = q * np.sqrt(float(pcov.flat[0])) / mu
            if kind == "cdf":
                lz = np.log(x / mu)
                fn = (lambda t: np.exp(-np.exp(t))) if upper else (lambda t: -np.expm1(-np.exp(t)))
                lo, hi = (lz + half, lz - half) if upper else (lz - half, lz + half)
                res = (fn(lz), fn(lo), fn(hi))
            else:
                val = -mu * np.log1p(-x)
                res = (val, val * np.exp(-half), val * np.exp(half))
    return outs(nargout, *[np.asarray(r, dtype=float).reshape(shape) for r in res])


def _rnd_fn(fam: Fam) -> Callable:
    fname = f"{fam.prefix}rnd"

    def f(interp, args, nargout):
        return _random_core(interp, fam, fname, list(args))
    return f


def _random_core(interp, fam: Fam, fname: str, args: list) -> np.ndarray:
    k = len(fam.pnames)
    if len(args) < k:
        raise MatlabError(f"Not enough input arguments for {fname}.")
    params, pshape = bcast(fname, *args[:k]) if k else ([], (1, 1))
    cls = None
    if len(args) > k:
        shape, cls = dims_from_args(list(args[k:]), fname)
        if pshape != (1, 1) and tuple(pshape) != tuple(shape):
            raise MatlabError(f"{fname}: size arguments must match the size of the parameters.")
    else:
        shape = pshape
    params = [np.broadcast_to(q, shape) if q.shape == tuple(shape) else np.broadcast_to(q.reshape(-1)[:1], shape)
              for q in params]
    r = np.asarray(fam.rnd(rng_of(interp), tuple(shape), params), dtype=float).reshape(shape)
    return cast_class(r, cls) if cls not in (None, "double") else r


def _stat_fn(fam: Fam) -> Callable:
    fname = "poisstat" if fam.prefix == "poiss" else f"{fam.prefix}stat"

    def f(interp, args, nargout):
        k = len(fam.pnames)
        if len(args) < k:
            raise MatlabError(f"Not enough input arguments for {fname}.")
        if len(args) > k:
            raise MatlabError(f"Too many input arguments for {fname}.")
        params, shape = bcast(fname, *args)
        m, v = fam.stat(params)
        return outs(max(nargout, 1), np.asarray(m, dtype=float).reshape(shape), np.asarray(v,
                dtype=float).reshape(shape))
    return f


# -- data preparation for fitting ---------------------------------------------------------------------------

def clean_data(x: Any, cens: Any = None, freq: Any = None, fname: str = "fit"):
    """Vector data with optional censoring flags and frequencies; NaN rows and zero counts are dropped."""
    xv = vec(x)
    c = None
    f = None
    if cens is not None and not is_empty_arg(cens):
        c = vec(cens) != 0
        if c.size != xv.size:
            raise MatlabError(f"{fname}: Censoring must have one value per observation.")
    if freq is not None and not is_empty_arg(freq):
        f = vec(freq)
        if f.size != xv.size:
            raise MatlabError(f"{fname}: Frequency must have one value per observation.")
        if np.any(f < 0):
            raise MatlabError(f"{fname}: Frequency values must not be negative.")
    keep = ~np.isnan(xv)
    if f is not None:
        keep &= ~np.isnan(f) & (f > 0)
    xv = xv[keep]
    c = c[keep] if c is not None else None
    f = f[keep] if f is not None else None
    if c is not None and not c.any():
        c = None
    if f is not None and np.all(f == 1):
        f = None
    return xv, c, f


def _count(x, freq) -> float:
    return float(x.size if freq is None else freq.sum())


def fit_with_ci(fam: Fam, x, cens, freq, alpha: float, fixed: dict | None = None, start=None,
                unbiased: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = fam.fit(x, cens, freq, fixed, start, unbiased)
    cov = fam.acov(params, x, cens, freq, fixed)
    ci = param_ci(fam, params, x, alpha, cov, _count(x, freq), cens=cens)
    return params, ci, cov


def _alpha(args: list, i: int) -> float:
    a = scalar(args[i], "alpha") if given(args, i) else 0.05
    if not 0 < a < 1:
        raise MatlabError("alpha must be between 0 and 1.")
    return a


def _check_options(args: list, i: int, fname: str) -> None:
    for a in args[1:]:
        if is_text(a):  # these functions take no name-value options
            raise MatlabError(f"{fname}: the option '{text(a)}' is not supported.", f"MATLAB:{fname}:UnknownOption")
    if len(args) > i and not is_empty_arg(args[i]) and not isinstance(args[i], Struct):
        raise MatlabError(f"{fname}: the options argument must be a structure created by statset.")


def _fit_fn(fam: Fam, censoring: bool) -> Callable:
    fname = f"{fam.prefix}fit"
    est = list(fam.est)

    def f(interp, args, nargout):
        if not args:
            raise MatlabError(f"Not enough input arguments for {fname}.")
        maxargs = 5 if censoring else 3
        _check_options(args, 4 if censoring else 2, fname)
        if len(args) > maxargs:
            raise MatlabError(f"Too many input arguments for {fname}.")
        alpha = _alpha(args, 1)
        cens = args[2] if censoring and len(args) > 2 else None
        freq = args[3] if censoring and len(args) > 3 else None
        X = np.asarray(fnum(args[0]), dtype=float)
        if fam.prefix in ("exp", "poiss") and X.ndim == 2 and min(X.shape) > 1 and cens is None and freq is None:
            cols = [fit_with_ci(fam, *clean_data(X[:, j], None, None, fname), alpha) for j in range(X.shape[1])]
            return outs(nargout, np.array([[c[0][0] for c in cols]]), np.hstack([c[1] for c in cols]))
        x, c, fr = clean_data(X, cens, freq, fname)
        params, ci, _ = fit_with_ci(fam, x, c, fr, alpha)
        return outs(nargout, params[est].reshape(1, -1), ci[:, est])
    return f


def _like_fn(fam: Fam) -> Callable:
    fname = f"{fam.prefix}like"

    def f(interp, args, nargout):
        if len(args) < 2:
            raise MatlabError(f"Not enough input arguments for {fname}.")
        if len(args) > 4:
            raise MatlabError(f"Too many input arguments for {fname}.")
        p = vec(args[0])
        est = list(fam.est)
        if p.size == len(fam.pnames):
            params = p
        elif p.size == len(est):
            params = np.array(fam.safe, dtype=float)
            if fam.prefix == "gp":
                params[2] = 0.0
            params[est] = p
        else:
            raise MatlabError(f"{fname}: the parameter vector must have {len(est)} elements.")
        x, c, fr = clean_data(args[1], args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else None, fname)
        nll = fam.nll(params, x, c, fr)
        if nargout > 1:
            cov = fam.acov(params, x, c, fr, {i: params[i] for i in range(len(params)) if i not in est})
            return nll, cov[np.ix_(est, est)]
        return nll
    return f


def _normfit(interp, args, nargout):
    if not args or len(args) > 5:
        raise MatlabError("normfit: expected normfit(x), normfit(x, alpha) or normfit(x, alpha, censoring, freq).")
    fam = FAMILIES["normal"]
    _check_options(args, 4, "normfit")
    alpha = _alpha(args, 1)
    X = np.asarray(fnum(args[0]), dtype=float)
    cens = args[2] if len(args) > 2 else None
    freq = args[3] if len(args) > 3 else None
    if X.ndim == 2 and min(X.shape) > 1 and not given(args, 2) and not given(args, 3):
        cols = [fit_with_ci(fam, *clean_data(X[:, j], None, None, "normfit"), alpha) for j in range(X.shape[1])]
        return outs(nargout, np.array([[c[0][0] for c in cols]]), np.array([[c[0][1] for c in cols]]),
                    np.hstack([c[1][:, :1] for c in cols]), np.hstack([c[1][:, 1:] for c in cols]))
    x, c, fr = clean_data(X, cens, freq, "normfit")
    p, ci, _ = fit_with_ci(fam, x, c, fr, alpha)
    return outs(nargout, p[0], p[1], ci[:, :1], ci[:, 1:])


def _unifit(interp, args, nargout):
    if not args or len(args) > 2:
        raise MatlabError("unifit: expected unifit(x) or unifit(x, alpha).")
    alpha = _alpha(args, 1)
    X = np.asarray(fnum(args[0]), dtype=float)
    if X.ndim == 2 and X.shape[0] == 1:
        X = X.reshape(-1, 1)
    a, b = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    n = np.sum(~np.isnan(X), axis=0)
    tmp = (b - a) / alpha ** (1.0 / n)
    return outs(nargout, a.reshape(1, -1), b.reshape(1, -1), np.vstack([b - tmp, a]), np.vstack([b, a + tmp]))


def _binofit(interp, args, nargout):
    if len(args) < 2 or len(args) > 3:
        raise MatlabError("binofit: expected binofit(x, n) or binofit(x, n, alpha).")
    alpha = _alpha(args, 2)
    (x, n), shape = bcast("binofit", args[0], args[1])
    if np.any((x < 0) | (x > n) | (x != np.round(x)) | (n != np.round(n))):
        raise MatlabError("binofit: x must hold integers between 0 and n.")
    xv, nv = x.flatten(order="F"), n.flatten(order="F")
    with np.errstate(all="ignore"):
        phat = xv / nv
    lo, hi = clopper_pearson(xv, nv, alpha)
    return outs(nargout, phat.reshape(shape, order="F"), np.column_stack([lo, hi]))


def _register_functions() -> None:
    censor = {"exp", "gam", "wbl", "logn", "ev"}
    fits = {"exp", "gam", "beta", "wbl", "logn", "ev", "gev", "gp", "nbin", "rayl", "poiss"}
    likes = {"norm", "exp", "gam", "beta", "wbl", "logn", "ev", "gev", "gp", "nbin"}
    for fam in FAMILIES.values():
        if not fam.prefix:
            continue
        for kind in ("pdf", "cdf", "inv"):
            builtin(f"{fam.prefix}{kind}")(_dist_fn(fam, kind))
        builtin(f"{fam.prefix}rnd")(_rnd_fn(fam))
        builtin("poisstat" if fam.prefix == "poiss" else f"{fam.prefix}stat")(_stat_fn(fam))
        if fam.prefix in fits:
            builtin(f"{fam.prefix}fit")(_fit_fn(fam, fam.prefix in censor))
        if fam.prefix in likes:
            builtin(f"{fam.prefix}like")(_like_fn(fam))
    builtin("normfit")(_normfit)
    builtin("unifit")(_unifit)
    builtin("binofit")(_binofit)


_register_functions()


# -- pdf / cdf / icdf / random by distribution name --------------------------------------------------------

def _by_name(kind: str) -> Callable:
    def f(interp, args, nargout):
        if not args:
            raise MatlabError(f"Not enough input arguments for {kind}.")
        if not is_text(args[0]):
            raise MatlabError(f"{kind}: the first input must be a distribution name or a probability "
                              "distribution object.")
        fam = family(args[0], kind)
        rest = list(args[1:])
        upper = False
        if kind == "cdf" and rest and is_text(rest[-1]):
            if text(rest[-1]).lower() != "upper":
                raise MatlabError("cdf: the only text option is 'upper'.")
            upper = True
            rest.pop()
        k = len(fam.pnames)
        if len(rest) < 1 + k:
            raise MatlabError(f"{kind}: the {fam.display} distribution needs {k} parameter(s) after the "
                              "evaluation points.")
        if len(rest) > 1 + k:
            raise MatlabError(f"Too many input arguments for {kind}.")
        arrs, shape = bcast(kind, *rest)
        method = {"pdf": "pdf", "cdf": "sf" if upper else "cdf", "icdf": "ppf"}[kind]
        return fam.ev(method, arrs[0], arrs[1:]).reshape(shape)
    return f


builtin("pdf")(_by_name("pdf"))
builtin("cdf")(_by_name("cdf"))
builtin("icdf")(_by_name("icdf"))


@builtin("random")
def _random(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("random needs a distribution name first, e.g. random('Normal', 0, 1, 3, 2).")
    k = _NAMES.get(_key(text(args[0])))
    if k is None:
        raise MatlabError(f"random: unknown distribution '{text(args[0])}'. Known names: "
                          + ", ".join(sorted(f.name for f in FAMILIES.values())) + ".")
    fam = FAMILIES[k]
    if len(args) - 1 < len(fam.pnames):
        raise MatlabError(f"random: the {text(args[0])} distribution needs {len(fam.pnames)} parameter(s).")
    return _random_core(interp, fam, "random", list(args[1:]))


# -- mle ------------------------------------------------------------------------------------------------------

_MLE_OPTS = ["Distribution", "Alpha", "Censoring", "Frequency", "NTrials", "Start", "pdf", "cdf", "logpdf", "logsf",
             "nloglf", "LowerBound", "UpperBound", "Options", "TruncationBounds", "mu", "theta", "OptimFun",
             "CheckInputs", "Display"]


@builtin("mle")
def _mle(interp, args, nargout):
    pos, opts = get_opts(args, 1, _MLE_OPTS, "mle")
    if len(pos) != 1:
        raise MatlabError("mle: expected mle(data, 'Name', value, ...).")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    x, cens, freq = clean_data(pos[0], opts.get("censoring"), opts.get("frequency"), "mle")
    custom = [k for k in ("pdf", "logpdf", "nloglf") if k in opts]
    if "truncationbounds" in opts:
        raise MatlabError("mle: the option 'TruncationBounds' is not supported.")
    if custom:
        return _mle_custom(interp, x, cens, freq, opts, alpha, nargout)
    fam = family(opts["distribution"], "mle") if "distribution" in opts else FAMILIES["normal"]
    fixed: dict[int, float] = {}
    if _key(fam.name) == "binomial":
        if "ntrials" not in opts:
            raise MatlabError("mle: the binomial distribution needs 'NTrials'.")
        fixed[0] = scalar(opts["ntrials"], "NTrials")
    if _key(fam.name) == "generalizedpareto":
        fixed[2] = scalar(opts["theta"], "theta") if "theta" in opts else 0.0
    if _key(fam.name) == "halfnormal":
        fixed[0] = scalar(opts["mu"], "mu") if "mu" in opts else 0.0
    start = vec(opts["start"]) if "start" in opts and not is_empty_arg(opts["start"]) else None
    params, ci, _ = fit_with_ci(fam, x, cens, freq, alpha, fixed, start, unbiased=False)
    est = [i for i in fam.est]
    if _key(fam.name) in ("generalizedpareto",):
        est = [0, 1]
    if _key(fam.name) == "binomial":
        est = [1]
    return outs(nargout, params[est].reshape(1, -1), ci[:, est])


def _mle_custom(interp, x, cens, freq, opts, alpha, nargout):
    """mle with a user-supplied pdf (and cdf), logpdf (and logsf) or negative log-likelihood."""
    from scipy.optimize import minimize
    if "start" not in opts:
        raise MatlabError("mle: a custom distribution needs starting values ('Start').")
    start = vec(opts["start"])
    lb = vec(opts["lowerbound"]) if "lowerbound" in opts else np.full(start.size, -np.inf)
    ub = vec(opts["upperbound"]) if "upperbound" in opts else np.full(start.size, np.inf)
    if lb.size != start.size or ub.size != start.size:
        raise MatlabError("mle: LowerBound and UpperBound must have one value per parameter.")
    xc = x.reshape(-1, 1)
    f = freq if freq is not None else np.ones(x.size)
    c = cens if cens is not None else np.zeros(x.size, dtype=bool)

    def pargs(th):
        return [np.array([[float(t)]]) for t in th]

    def nll(th):
        if np.any(th < lb) or np.any(th > ub):
            return _BIG
        with np.errstate(all="ignore"):
            if "nloglf" in opts:
                v = float(call_num(interp, opts["nloglf"], np.asarray(th, dtype=float).reshape(1, -1), xc,
                                   c.reshape(-1, 1), f.reshape(-1, 1)).flat[0])
                return v if np.isfinite(v) else _BIG
            if "logpdf" in opts:
                ll = call_num(interp, opts["logpdf"], xc, *pargs(th)).flatten()
                if c.any():
                    if "logsf" not in opts:
                        raise MatlabError("mle: censored data need 'logsf' together with 'logpdf'.")
                    ll = np.where(c, call_num(interp, opts["logsf"], xc, *pargs(th)).flatten(), ll)
            else:
                ll = np.log(call_num(interp, opts["pdf"], xc, *pargs(th)).flatten())
                if c.any():
                    if "cdf" not in opts:
                        raise MatlabError("mle: censored data need 'cdf' together with 'pdf'.")
                    ll = np.where(c, np.log1p(-call_num(interp, opts["cdf"], xc, *pargs(th)).flatten()), ll)
            v = -float(np.sum(f * ll))
        return v if np.isfinite(v) else _BIG

    best = minimize(nll, start, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 1000 * start.size, "maxfev": 2000 * start.size})
    phat = best.x
    if nargout > 1:
        import scipy.stats as st
        with np.errstate(all="ignore"):
            H = num_hessian(nll, phat)
            try:
                se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
            except np.linalg.LinAlgError:
                se = np.full(phat.size, np.nan)
        z = st.norm.ppf(1 - alpha / 2)
        return phat.reshape(1, -1), np.vstack([phat - z * se, phat + z * se])
    return phat.reshape(1, -1)


# -- statset -----------------------------------------------------------------------------------------------------

_STATSET = ["Display", "MaxFunEvals", "MaxIter", "TolBnd", "TolFun", "TolTypeFun", "TolX", "TolTypeX", "GradObj",
            "Jacobian", "DerivStep", "FunValCheck", "Robust", "RobustWgtFun", "WgtFun", "Tune", "UseParallel",
            "UseSubstreams", "Streams", "OutputFcn"]


@builtin("statset")
def _statset(interp, args, nargout):
    from ..values import empty_double
    s = Struct.scalar({n: empty_double() for n in _STATSET})
    rest = list(args)
    if rest and isinstance(rest[0], Struct):
        old = rest.pop(0)
        for n in old.fields:
            if n in _STATSET:
                s.set_field(0, n, old.get_field(0, n))
    elif len(rest) == 1 and is_text(rest[0]):
        return s  # statset('fitname'): the defaults of a fitting function (all left empty here)
    if len(rest) % 2:
        raise MatlabError("statset: name-value arguments must come in pairs.")
    lookup = {n.lower(): n for n in _STATSET}
    for k in range(0, len(rest), 2):
        key = lookup.get(text(rest[k]).lower())
        if key is None:
            raise MatlabError(f"statset: unknown option '{text(rest[k])}'.")
        s.set_field(0, key, rest[k + 1])
    return s


def bool_opt(opts: dict, name: str, default: bool) -> bool:
    return as_bool(opts[name], name) if name in opts else default
