"""Probability distribution objects (makedist, fitdist), multivariate distributions, the empirical
distribution function, kernel density estimates and the distribution plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, char
from ._registry import builtin
from ._stats_util import (
    StatObject, as_bool, builtin_call, cellstr, choice, get_opts, given, group_codes, intval, is_empty_arg,
    num_text, outs, rng_of, scalar, struct, vec,
)
from ._util import dims_from_args, fnum, is_text, text
from .stats_dist import FAMILIES, Fam, _key, clean_data, family, fit_with_ci, param_ci


# -- parametric distribution objects ---------------------------------------------------------------------

class ProbDist(StatObject):
    """A parametric probability distribution, possibly truncated and possibly fitted to data."""

    also_isa = ("prob.ProbabilityDistribution", "prob.ParametricDistribution")

    def __init__(self, fam: Fam, params, trunc: tuple | None = None, data: dict | None = None,
                 cov: np.ndarray | None = None, fixed: dict | None = None):
        super().__init__()
        self.fam = fam
        self.params = np.asarray(params, dtype=float).flatten()
        self.trunc = trunc
        self.data = data
        self.cov = cov
        self.fixed = dict(fixed or {})
        self.matlab_class = f"prob.{fam.name if fam.name != 'T' else 'T'}Distribution"
        if not np.all(self.fam.valid(*self.params)):
            raise MatlabError(f"Invalid parameter values for the {fam.display} distribution.")

    # -- core math ---------------------------------------------------------------------------------
    def _p(self) -> list:
        return [np.float64(v) for v in self.params]

    def _limits(self) -> tuple[float, float, float, float]:
        """Truncation limits and the probability mass below the lower and up to the upper limit."""
        lo, hi = self.trunc
        if self.fam.discrete:
            flo = float(self.fam.ev("cdf", np.ceil(lo) - 1, self._p())) if np.isfinite(lo) else 0.0
        else:
            flo = float(self.fam.ev("cdf", lo, self._p())) if np.isfinite(lo) else 0.0
        fhi = float(self.fam.ev("cdf", hi, self._p())) if np.isfinite(hi) else 1.0
        return lo, hi, flo, fhi

    def pdf(self, x: np.ndarray) -> np.ndarray:
        f = self.fam.ev("pdf", x, self._p())
        if self.trunc is None:
            return f
        lo, hi, flo, fhi = self._limits()
        return np.where(np.isnan(x), np.nan, np.where((x >= lo) & (x <= hi), f / (fhi - flo), 0.0))

    def cdf(self, x: np.ndarray, upper: bool = False) -> np.ndarray:
        if self.trunc is None:
            return self.fam.ev("sf" if upper else "cdf", x, self._p())
        lo, hi, flo, fhi = self._limits()
        c = (self.fam.ev("cdf", x, self._p()) - flo) / (fhi - flo)
        c = np.where(x < lo, 0.0, np.where(x >= hi, 1.0, np.clip(c, 0, 1)))
        c = np.where(np.isnan(x), np.nan, c)
        return 1 - c if upper else c

    def icdf(self, p: np.ndarray) -> np.ndarray:
        if self.trunc is None:
            return self.fam.ev("ppf", p, self._p())
        lo, hi, flo, fhi = self._limits()
        with np.errstate(all="ignore"):
            q = self.fam.ev("ppf", np.clip(flo + p * (fhi - flo), 0, 1), self._p())
        q = np.clip(q, lo, hi)
        return np.where((p < 0) | (p > 1) | np.isnan(p), np.nan, q)

    def moments(self) -> tuple[float, float]:
        if self.trunc is None:
            m, v = self.fam.stat(self._p())
            return float(m), float(v)
        lo, hi, _, _ = self._limits()
        if self.fam.discrete:
            a = np.ceil(lo) if np.isfinite(lo) else float(self.fam.ev("ppf", 1e-13, self._p()))
            b = np.floor(hi) if np.isfinite(hi) else float(self.fam.ev("ppf", 1 - 1e-13, self._p()))
            k = np.arange(a, b + 1)
            w = self.pdf(k)
            m = float(np.sum(k * w))
            return m, float(np.sum((k - m) ** 2 * w))
        from scipy.integrate import quad
        f = lambda t: float(self.pdf(np.float64(t)))  # noqa: E731
        m = quad(lambda t: t * f(t), lo, hi, limit=200)[0]
        v = quad(lambda t: (t - m) ** 2 * f(t), lo, hi, limit=200)[0]
        return float(m), float(v)

    def sample(self, rng, shape: tuple) -> np.ndarray:
        if self.trunc is None:
            return self.fam.rnd(rng, shape, self._p())
        return self.icdf(rng.random(shape))

    # -- properties --------------------------------------------------------------------------------
    def property_names(self) -> list[str]:
        return list(self.fam.pnames) + ["DistributionName", "NumParameters", "ParameterNames",
                                        "ParameterDescription", "ParameterValues", "Truncation", "IsTruncated",
                                        "ParameterCovariance", "ParameterIsFixed", "InputData", "NLogL"]

    def get_field(self, interp, name: str) -> Any:
        fam = self.fam
        if name in fam.pnames:
            return np.array([[self.params[fam.pnames.index(name)]]])
        k = len(fam.pnames)
        if name == "DistributionName":
            return char(fam.name if fam.name != "T" else "t")
        if name == "NumParameters":
            return np.array([[float(k)]])
        if name == "ParameterNames":
            return cellstr(fam.pnames, column=False)
        if name == "ParameterDescription":
            return cellstr(fam.desc, column=False)
        if name == "ParameterValues":
            return self.params.reshape(1, -1).copy()
        if name == "Truncation":
            return np.array([list(self.trunc)]) if self.trunc is not None else np.zeros((0, 0))
        if name == "IsTruncated":
            return np.array([[self.trunc is not None]])
        if name == "ParameterCovariance":
            return self.cov.copy() if self.cov is not None else np.zeros((k, k))
        if name == "ParameterIsFixed":
            if self.data is None:
                return np.ones((1, k), dtype=bool)
            return np.array([[i not in fam.est or i in self.fixed for i in range(k)]])
        if name == "InputData":
            if self.data is None:
                return np.zeros((0, 0))
            d = self.data
            return struct(data=d["x"].reshape(-1, 1),
                          cens=d["cens"].astype(float).reshape(-1, 1) if d["cens"] is not None else np.zeros((0, 0)),
                          freq=d["freq"].reshape(-1, 1) if d["freq"] is not None else np.zeros((0, 0)))
        if name == "NLogL":
            return self._nll()
        return super().get_field(interp, name)

    def set_field(self, interp, name: str, value: Any):
        if name not in self.fam.pnames:
            raise MatlabError(f"Cannot set property '{name}' of a {self.fam.display} distribution.")
        params = self.params.copy()
        params[self.fam.pnames.index(name)] = scalar(value, name)
        return ProbDist(self.fam, params, self.trunc)

    def _nll(self):
        if self.data is None:
            return np.zeros((0, 0))
        d = self.data
        return np.array([[self.fam.nll(self.params, d["x"], d["cens"], d["freq"])]])

    # -- methods -----------------------------------------------------------------------------------
    def m_pdf(self, interp, args, nargout):
        _nargs(args, 1, 1, "pdf")
        x = np.asarray(fnum(args[0]), dtype=float)
        return self.pdf(x).reshape(x.shape)

    def m_cdf(self, interp, args, nargout):
        _nargs(args, 1, 2, "cdf")
        upper = False
        if len(args) == 2:
            if not is_text(args[1]) or text(args[1]).lower() != "upper":
                raise MatlabError("cdf: the only option is 'upper'.")
            upper = True
        x = np.asarray(fnum(args[0]), dtype=float)
        return self.cdf(x, upper).reshape(x.shape)

    def m_icdf(self, interp, args, nargout):
        _nargs(args, 1, 1, "icdf")
        p = np.asarray(fnum(args[0]), dtype=float)
        return np.asarray(self.icdf(p), dtype=float).reshape(p.shape)

    def m_random(self, interp, args, nargout):
        shape = dims_from_args(list(args), "random")[0] if args else (1, 1)
        return np.asarray(self.sample(rng_of(interp), tuple(shape)), dtype=float).reshape(shape)

    def m_mean(self, interp, args, nargout):
        _nargs(args, 0, 0, "mean")
        return self.moments()[0]

    def m_var(self, interp, args, nargout):
        _nargs(args, 0, 0, "var")
        return self.moments()[1]

    def m_std(self, interp, args, nargout):
        _nargs(args, 0, 0, "std")
        return float(np.sqrt(self.moments()[1]))

    def m_median(self, interp, args, nargout):
        _nargs(args, 0, 0, "median")
        return float(self.icdf(np.float64(0.5)))

    def m_iqr(self, interp, args, nargout):
        _nargs(args, 0, 0, "iqr")
        return float(self.icdf(np.float64(0.75)) - self.icdf(np.float64(0.25)))

    def m_truncate(self, interp, args, nargout):
        _nargs(args, 2, 2, "truncate")
        lo, hi = scalar(args[0], "lower limit"), scalar(args[1], "upper limit")
        if not lo < hi:
            raise MatlabError("truncate: the lower limit must be less than the upper limit.")
        t = ProbDist(self.fam, self.params, (lo, hi), self.data, self.cov, self.fixed)
        _, _, flo, fhi = t._limits()
        if not fhi - flo > 0:
            raise MatlabError("truncate: the distribution has no probability between the limits.")
        return t

    def m_paramci(self, interp, args, nargout):
        _, opts = get_opts(args, 0, ["Alpha", "Parameter", "Type", "LogFlag"], "paramci")
        alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
        kind = choice(opts["type"], ["exact", "Wald", "lr"], "Type") if "type" in opts else "exact"
        if kind == "lr":
            raise MatlabError("paramci: likelihood-ratio intervals ('Type', 'lr') are not supported.")
        k = len(self.fam.pnames)
        if self.data is None:
            ci = np.vstack([self.params, self.params])
        else:
            fam = self.fam
            if "logflag" in opts:
                flags = vec(opts["logflag"]) != 0
                if flags.size != k:
                    raise MatlabError(f"paramci: LogFlag must have {k} elements.")
                import copy
                fam = copy.copy(fam)
                fam.logci = tuple(bool(f) for f in flags)
            d = self.data
            n = float(d["x"].size if d["freq"] is None else d["freq"].sum())
            ci = param_ci(fam, self.params, d["x"], alpha, self.cov, n, exact=(kind == "exact"), cens=d["cens"])
            for i in range(k):
                if i not in self.fam.est or i in self.fixed:
                    ci[:, i] = self.params[i]
        if "parameter" in opts:
            v = opts["parameter"]
            if is_text(v) or isinstance(v, Cell):
                names = [text(v)] if is_text(v) else [text(c) for c in v.values_f()]
                for nme in names:
                    if nme not in self.fam.pnames:
                        raise MatlabError(f"paramci: unknown parameter '{nme}'.")
                idx = [self.fam.pnames.index(nme) for nme in names]
            else:
                idx = [int(i) - 1 for i in vec(v)]
            ci = ci[:, idx]
        return ci

    def m_negloglik(self, interp, args, nargout):
        _nargs(args, 0, 0, "negloglik")
        return self._nll()

    def m_plot(self, interp, args, nargout):
        return _plot_dist(interp, self, args, nargout)

    # -- display -----------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        fam = self.fam
        lines = [f"  {self.matlab_class.split('.')[-1]}", "",
                f"  {fam.display[0].upper() + fam.display[1:]} distribution"]
        width = max(len(n) for n in fam.pnames)
        ci = None
        if self.data is not None and self.cov is not None:
            d = self.data
            n = float(d["x"].size if d["freq"] is None else d["freq"].sum())
            try:
                ci = param_ci(fam, self.params, d["x"], 0.05, self.cov, n, cens=d["cens"])
            except Exception:  # pragma: no cover - display must not fail
                ci = None
        vals = [num_text(v) if v == int(v) else f"{v:.6g}" for v in self.params]
        vw = max(len(v) for v in vals)
        for i, n in enumerate(fam.pnames):
            line = f"    {n.rjust(width)} = {vals[i].rjust(vw)}"
            if ci is not None and i in fam.est and i not in self.fixed and np.all(np.isfinite(ci[:, i])):
                line += f"   [{ci[0, i]:.6g}, {ci[1, i]:.6g}]"
            lines.append(line)
        if self.trunc is not None:
            lines.append(f"  Truncated to the interval [{num_text(self.trunc[0])}, {num_text(self.trunc[1])}]")
        return lines


def _nargs(args: list, lo: int, hi: int, name: str) -> None:
    if len(args) < lo:
        raise MatlabError(f"Not enough input arguments for {name}.")
    if len(args) > hi:
        raise MatlabError(f"Too many input arguments for {name}.")


# -- kernel distribution objects ---------------------------------------------------------------------------

_KERNELS = ["normal", "box", "triangle", "epanechnikov"]
_KW = {"box": np.sqrt(3.0), "triangle": np.sqrt(6.0), "epanechnikov": np.sqrt(5.0)}


def kernel_pdf(z: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "normal":
        return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    a = _KW[kernel]
    if kernel == "box":
        return (np.abs(z) <= a) / (2 * a)
    if kernel == "triangle":
        return np.clip(1 - np.abs(z) / a, 0, None) / a
    return np.clip(1 - (z / a) ** 2, 0, None) * 0.75 / a


def kernel_cdf(z: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "normal":
        from scipy.special import ndtr
        return ndtr(z)
    a = _KW[kernel]
    t = np.clip(z / a, -1, 1)
    if kernel == "box":
        return (t + 1) / 2
    if kernel == "triangle":
        return np.where(t < 0, (t + 1) ** 2 / 2, 1 - (1 - t) ** 2 / 2)
    return 0.5 + 0.75 * t - 0.25 * t ** 3


def default_bandwidth(x: np.ndarray, w: np.ndarray | None = None, d: int = 1) -> float:
    """Normal-reference bandwidth with a robust (median absolute deviation) scale estimate."""
    med = np.median(x)
    sig = np.median(np.abs(x - med)) / 0.6745
    if sig <= 0:
        sig = float(np.max(x) - np.min(x))
    if sig <= 0:
        sig = 1.0
    n = x.size
    return float(sig * (4.0 / ((d + 2.0) * n)) ** (1.0 / (d + 4.0)))


class _Support:
    """Maps a bounded support onto the real line (log for positive, logit for an interval)."""

    def __init__(self, spec: Any, x: np.ndarray, fname: str):
        self.lo, self.hi = -np.inf, np.inf
        if spec is None or (is_text(spec) and text(spec).lower() == "unbounded"):
            self.kind = "unbounded"
        elif is_text(spec):
            if text(spec).lower() != "positive":
                raise MatlabError(f"{fname}: Support must be 'unbounded', 'positive' or a two-element vector.")
            self.kind, self.lo = "positive", 0.0
        else:
            v = vec(spec)
            if v.size != 2 or not v[0] < v[1]:
                raise MatlabError(f"{fname}: Support must be 'unbounded', 'positive' or [lower upper].")
            self.lo, self.hi = float(v[0]), float(v[1])
            self.kind = "unbounded" if np.isinf(v[0]) and np.isinf(v[1]) else \
                ("positive" if v[0] == 0 and np.isinf(v[1]) else "interval")
            if self.kind == "interval" and (np.isinf(v[0]) or np.isinf(v[1])):
                self.kind = "half"
        if x.size and (np.any(x <= self.lo) or np.any(x >= self.hi)):
            raise MatlabError(f"{fname}: all data values must lie strictly inside the support.")

    def fwd(self, x):
        with np.errstate(all="ignore"):
            if self.kind == "unbounded":
                return np.asarray(x, dtype=float)
            if self.kind == "positive":
                return np.log(x)
            if self.kind == "half":
                return np.log(x - self.lo) if np.isfinite(self.lo) else -np.log(self.hi - x)
            return np.log((x - self.lo) / (self.hi - x))

    def inv(self, y):
        with np.errstate(all="ignore"):
            if self.kind == "unbounded":
                return np.asarray(y, dtype=float)
            if self.kind == "positive":
                return np.exp(y)
            if self.kind == "half":
                return self.lo + np.exp(y) if np.isfinite(self.lo) else self.hi - np.exp(-y)
            e = np.exp(-y)
            return (self.lo * e + self.hi) / (1 + e)

    def jac(self, x):
        """dy/dx of the transform (multiplies the density on the transformed scale)."""
        with np.errstate(all="ignore"):
            if self.kind == "unbounded":
                return np.ones_like(np.asarray(x, dtype=float))
            if self.kind == "positive":
                return 1.0 / x
            if self.kind == "half":
                return 1.0 / (x - self.lo) if np.isfinite(self.lo) else 1.0 / (self.hi - x)
            return (self.hi - self.lo) / ((x - self.lo) * (self.hi - x))

    def inside(self, x):
        return (x > self.lo) & (x < self.hi)


class KDE:
    """A univariate kernel density estimate."""

    def __init__(self, x, w, kernel: str, bw: float | None, support: _Support, reflect: bool = False):
        self.x = x
        self.w = w if w is not None else np.full(x.size, 1.0 / x.size)
        self.kernel = kernel
        self.support = support
        self.reflect = reflect and support.kind != "unbounded"
        self.y = x if self.reflect else support.fwd(x)
        self.bw = float(bw) if bw is not None else default_bandwidth(self.y)
        if not self.bw > 0:
            raise MatlabError("The bandwidth must be positive.")

    def _sum(self, fn, pts):
        out = np.empty(pts.size)
        step = max(1, 2_000_000 // max(self.y.size, 1))
        for s in range(0, pts.size, step):
            z = (pts[s:s + step, None] - self.y[None, :]) / self.bw
            out[s:s + step] = fn(z, self.kernel) @ self.w
        return out

    def pdf(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).flatten()
        sp = self.support
        if self.reflect:
            f = self._sum(kernel_pdf, xi)
            if np.isfinite(sp.lo):
                f = f + self._sum(kernel_pdf, 2 * sp.lo - xi)
            if np.isfinite(sp.hi):
                f = f + self._sum(kernel_pdf, 2 * sp.hi - xi)
            return np.where((xi >= sp.lo) & (xi <= sp.hi), f / self.bw, 0.0)
        ok = sp.inside(xi)
        xs = np.where(ok, xi, self.x[0] if self.x.size else 0.0)
        f = self._sum(kernel_pdf, sp.fwd(xs)) / self.bw * sp.jac(xs)
        return np.where(ok, f, 0.0)

    def cdf(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).flatten()
        sp = self.support
        if self.reflect:
            xc = np.clip(xi, sp.lo, sp.hi)
            f = self._sum(kernel_cdf, xc)
            if np.isfinite(sp.lo) and np.isfinite(sp.hi):
                f = (f - self._sum(kernel_cdf, 2 * sp.lo - xc)
                     + self._sum(kernel_cdf, np.full(xc.size, 2 * sp.hi - sp.lo)) - self._sum(kernel_cdf, 2 * sp.hi
                             - xc))
            elif np.isfinite(sp.lo):
                f = f - self._sum(kernel_cdf, 2 * sp.lo - xc)
            else:
                f = f + 1 - self._sum(kernel_cdf, 2 * sp.hi - xc)
            return np.clip(f, 0, 1)
        ok = sp.inside(xi)
        xs = np.where(ok, xi, self.x[0] if self.x.size else 0.0)
        f = self._sum(kernel_cdf, sp.fwd(xs))
        return np.where(ok, f, np.where(xi <= sp.lo, 0.0, 1.0))

    def icdf(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).flatten()
        span = 40 * self.bw + (self.y.max() - self.y.min())
        lo = np.full(p.size, self.y.min() - span)
        hi = np.full(p.size, self.y.max() + span)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            c = self._sum(kernel_cdf, mid) if not self.reflect else self.cdf(mid)
            up = c < p
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
            if np.max(hi - lo) < 1e-13 * max(1.0, span):
                break
        q = 0.5 * (lo + hi)
        q = q if self.reflect else self.support.inv(q)
        q = np.where(p <= 0, self.support.lo if self.support.kind != "unbounded" else -np.inf, q)
        q = np.where(p >= 1, self.support.hi if self.support.kind != "unbounded" else np.inf, q)
        return np.where((p < 0) | (p > 1) | np.isnan(p), np.nan, q)

    def grid(self, m: int) -> np.ndarray:
        if self.reflect:
            lo, hi = max(self.x.min() - 3 * self.bw, self.support.lo), min(self.x.max() + 3 * self.bw, self.support.hi)
            return np.linspace(lo, hi, m)
        return self.support.inv(np.linspace(self.y.min() - 3 * self.bw, self.y.max() + 3 * self.bw, m))

    def sample(self, rng, shape: tuple) -> np.ndarray:
        n = int(np.prod(shape))
        idx = rng.choice(self.y.size, size=n, p=self.w / self.w.sum())
        if self.kernel == "normal":
            e = rng.standard_normal(n)
        else:
            u = rng.random(n)
            a = _KW[self.kernel]
            if self.kernel == "box":
                e = a * (2 * u - 1)
            elif self.kernel == "triangle":
                e = a * (rng.random(n) - u)
            else:  # median of three uniforms on [-1, 1] follows the Epanechnikov density
                v = np.sort(np.column_stack([2 * u - 1, 2 * rng.random(n) - 1, 2 * rng.random(n) - 1]), axis=1)
                e = a * v[:, 1]
        y = self.y[idx] + self.bw * e
        if self.reflect:
            sp = self.support
            for _ in range(8):
                y = np.where(y < sp.lo, 2 * sp.lo - y, y)
                y = np.where(y > sp.hi, 2 * sp.hi - y, y)
            return y.reshape(shape)
        return self.support.inv(y).reshape(shape)


class KernelDist(StatObject):
    matlab_class = "prob.KernelDistribution"
    also_isa = ("prob.ProbabilityDistribution",)

    def __init__(self, kde: KDE, freq=None, trunc: tuple | None = None):
        super().__init__()
        self.kde = kde
        self.freq = freq
        self.trunc = trunc

    def _lim(self):
        lo, hi = self.trunc
        flo = float(self.kde.cdf(np.array([lo]))[0]) if np.isfinite(lo) else 0.0
        fhi = float(self.kde.cdf(np.array([hi]))[0]) if np.isfinite(hi) else 1.0
        return lo, hi, flo, fhi

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        f = self.kde.pdf(x.flatten()).reshape(x.shape)
        if self.trunc is None:
            return f
        lo, hi, flo, fhi = self._lim()
        return np.where((x >= lo) & (x <= hi), f / (fhi - flo), 0.0)

    def cdf(self, x, upper=False):
        x = np.asarray(x, dtype=float)
        c = self.kde.cdf(x.flatten()).reshape(x.shape)
        if self.trunc is not None:
            lo, hi, flo, fhi = self._lim()
            c = np.where(x < lo, 0.0, np.where(x >= hi, 1.0, np.clip((c - flo) / (fhi - flo), 0, 1)))
        return 1 - c if upper else c

    def icdf(self, p):
        p = np.asarray(p, dtype=float)
        q = p
        if self.trunc is not None:
            lo, hi, flo, fhi = self._lim()
            q = flo + p * (fhi - flo)
        r = self.kde.icdf(q.flatten()).reshape(p.shape)
        if self.trunc is not None:
            r = np.clip(r, self.trunc[0], self.trunc[1])
        return np.where((p < 0) | (p > 1), np.nan, r)

    def moments(self):
        from scipy.integrate import quad
        k = self.kde
        if self.trunc is None and k.support.kind == "unbounded" and not k.reflect:
            w = k.w / k.w.sum()
            m = float(np.sum(w * k.x))
            return m, float(np.sum(w * (k.x - m) ** 2) + k.bw ** 2)
        lo = self.trunc[0] if self.trunc else k.support.lo
        hi = self.trunc[1] if self.trunc else k.support.hi
        f = lambda t: float(self.pdf(np.array([t]))[0])  # noqa: E731
        pts = np.sort(k.x)[:: max(1, k.x.size // 40)]
        pts = pts[(pts > lo) & (pts < hi)]
        m = quad(lambda t: t * f(t), lo, hi, limit=400, points=pts if np.isfinite(lo) and np.isfinite(hi) else None)[0]
        v = quad(lambda t: (t - m) ** 2 * f(t), lo, hi, limit=400,
                 points=pts if np.isfinite(lo) and np.isfinite(hi) else None)[0]
        return float(m), float(v)

    def property_names(self) -> list[str]:
        return ["Kernel", "Bandwidth", "Support", "DistributionName", "InputData", "Truncation", "IsTruncated"]

    def get_field(self, interp, name: str) -> Any:
        k = self.kde
        if name == "Kernel":
            return char(k.kernel)
        if name in ("Bandwidth", "BandWidth"):
            return np.array([[k.bw]])
        if name == "DistributionName":
            return char("Kernel")
        if name == "Support":
            sp = k.support
            return struct(range=char(sp.kind) if sp.kind in ("unbounded", "positive") else np.array([[sp.lo, sp.hi]]),
                          closedbound=np.zeros((1, 2), dtype=bool), iscontinuous=True)
        if name == "InputData":
            return struct(data=k.x.reshape(-1, 1), cens=np.zeros((0, 0)),
                          freq=self.freq.reshape(-1, 1) if self.freq is not None else np.zeros((0, 0)))
        if name == "Truncation":
            return np.array([list(self.trunc)]) if self.trunc is not None else np.zeros((0, 0))
        if name == "IsTruncated":
            return np.array([[self.trunc is not None]])
        if name in ("NumParameters",):
            return np.array([[0.0]])
        return super().get_field(interp, name)

    m_pdf = ProbDist.m_pdf
    m_cdf = ProbDist.m_cdf
    m_icdf = ProbDist.m_icdf
    m_mean = ProbDist.m_mean
    m_var = ProbDist.m_var
    m_std = ProbDist.m_std
    m_median = ProbDist.m_median
    m_iqr = ProbDist.m_iqr

    def m_random(self, interp, args, nargout):
        shape = dims_from_args(list(args), "random")[0] if args else (1, 1)
        if self.trunc is not None:
            return self.icdf(rng_of(interp).random(shape)).reshape(shape)
        return self.kde.sample(rng_of(interp), tuple(shape))

    def m_truncate(self, interp, args, nargout):
        _nargs(args, 2, 2, "truncate")
        lo, hi = scalar(args[0], "lower limit"), scalar(args[1], "upper limit")
        if not lo < hi:
            raise MatlabError("truncate: the lower limit must be less than the upper limit.")
        return KernelDist(self.kde, self.freq, (lo, hi))

    def m_negloglik(self, interp, args, nargout):
        k = self.kde
        f = self.pdf(k.x)
        with np.errstate(all="ignore"):
            return -float(np.sum((self.freq if self.freq is not None else 1.0) * np.log(f)))

    def m_plot(self, interp, args, nargout):
        return _plot_dist(interp, self, args, nargout)

    def display_lines(self) -> list[str]:
        k = self.kde
        sp = k.support.kind if k.support.kind in ("unbounded",
                "positive") else f"[{num_text(k.support.lo)}, {num_text(k.support.hi)}]"
        lines = ["  KernelDistribution", "", f"    Kernel = {k.kernel}", f"    Bandwidth = {k.bw:.6g}",
                 f"    Support = {sp}"]
        if self.trunc is not None:
            lines.append(f"  Truncated to the interval [{num_text(self.trunc[0])}, {num_text(self.trunc[1])}]")
        return lines


def _plot_dist(interp, pd, args, nargout):
    _, opts = get_opts(args, 0, ["PlotType", "Discrete", "Parent"], "plot")
    kind = choice(opts["plottype"], ["pdf", "cdf", "probability"], "PlotType") if "plottype" in opts else "pdf"
    if kind == "probability" or "parent" in opts or "discrete" in opts:
        raise MatlabError("plot: only 'PlotType' 'pdf' and 'cdf' are supported for distribution objects.")
    lo, hi = float(pd.icdf(np.float64(0.0005))), float(pd.icdf(np.float64(0.9995)))
    discrete = isinstance(pd, ProbDist) and pd.fam.discrete
    x = np.arange(np.floor(lo), np.ceil(hi) + 1) if discrete else np.linspace(lo, hi, 300)
    y = pd.pdf(x) if kind == "pdf" else pd.cdf(x)
    name = "stem" if discrete and kind == "pdf" else ("stairs" if discrete else "plot")
    h = builtin_call(interp, name, [x.reshape(1, -1), y.reshape(1, -1)], 1)
    builtin_call(interp, "ylabel", [char("PDF" if kind == "pdf" else "CDF")], 0)
    return h[0] if nargout else None


# -- makedist / fitdist --------------------------------------------------------------------------------------

@builtin("makedist")
def _makedist(interp, args, nargout):
    if not args:
        names = sorted(f.name for f in FAMILIES.values() if f.name not in ("T", "Chisquare", "F", "NoncentralT",
                                                                           "NoncentralF", "NoncentralChisquare",
                                                                           "Hypergeometric", "DiscreteUniform",
                                                                           "Geometric"))
        if nargout:
            return cellstr(names)
        interp.out("\n".join("    " + n for n in names) + "\n")
        return None
    fam = family(args[0], "makedist")
    rest = list(args[1:])
    if len(rest) % 2:
        raise MatlabError("makedist: parameters are given as name-value pairs, e.g. "
                          "makedist('Normal', 'mu', 0, 'sigma', 1).")
    params = np.array(fam.safe, dtype=float)
    lookup = {n.lower(): i for i, n in enumerate(fam.pnames)}
    for k in range(0, len(rest), 2):
        if not is_text(rest[k]) or text(rest[k]).lower() not in lookup:
            shown = text(rest[k]) if is_text(rest[k]) else "?"
            raise MatlabError(f"makedist: '{shown}' is not a parameter of the {fam.display} distribution "
                              f"(parameters: {', '.join(fam.pnames)}).")
        params[lookup[text(rest[k]).lower()]] = scalar(rest[k + 1], text(rest[k]))
    return ProbDist(fam, params)


_FITDIST_OPTS = ["By", "Censoring", "Frequency", "Options", "NTrials", "Theta", "mu", "Kernel", "Support", "Width",
                 "Bandwidth"]


def fit_object(name: Any, x, cens, freq, opts: dict, fname: str = "fitdist"):
    if is_text(name) and text(name).lower() == "kernel":
        if cens is not None:
            raise MatlabError(f"{fname}: censoring is not supported for kernel distributions.")
        kernel = choice(opts["kernel"], _KERNELS, "Kernel") if "kernel" in opts else "normal"
        bw = opts.get("width", opts.get("bandwidth"))
        bw = scalar(bw, "Width") if bw is not None and not is_empty_arg(bw) else None
        w = freq / freq.sum() if freq is not None else None
        if x.size == 0:
            raise MatlabError(f"{fname}: cannot fit a kernel distribution to empty data.")
        return KernelDist(KDE(x, w, kernel, bw, _Support(opts.get("support"), x, fname)), freq)
    fam = family(name, fname)
    for o in ("kernel", "support", "width", "bandwidth"):
        if o in opts:
            raise MatlabError(f"{fname}: the option '{o}' applies to kernel distributions only.")
    key = _key(fam.name)
    fixed: dict[int, float] = {}
    if key == "binomial":
        fixed[0] = scalar(opts["ntrials"], "NTrials") if "ntrials" in opts else 1.0
    elif "ntrials" in opts:
        raise MatlabError(f"{fname}: 'NTrials' applies to the binomial distribution only.")
    if key == "generalizedpareto":
        fixed[2] = scalar(opts["theta"], "Theta") if "theta" in opts else 0.0
    elif "theta" in opts:
        raise MatlabError(f"{fname}: 'Theta' applies to the generalized Pareto distribution only.")
    if key == "halfnormal":
        fixed[0] = scalar(opts["mu"], "mu") if "mu" in opts else 0.0
    elif "mu" in opts:
        raise MatlabError(f"{fname}: 'mu' applies to the half-normal distribution only.")
    if fam.closed is None and fam.start is None:
        raise MatlabError(f"{fname}: fitting the {fam.display} distribution is not supported.")
    params, _, cov = fit_with_ci(fam, x, cens, freq, 0.05, fixed)
    return ProbDist(fam, params, None, {"x": x, "cens": cens, "freq": freq}, cov, fixed)


@builtin("fitdist")
def _fitdist(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("fitdist: expected fitdist(x, distname, ...).")
    pos, opts = get_opts(args, 2, _FITDIST_OPTS, "fitdist")
    if len(pos) != 2:
        raise MatlabError("fitdist: expected fitdist(x, distname, 'Name', value, ...).")
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2 or min(X.shape) > 1:
        raise MatlabError("fitdist: the data must be a vector.")
    if "options" in opts and not isinstance(opts["options"], Struct) and not is_empty_arg(opts["options"]):
        raise MatlabError("fitdist: Options must be a structure created by statset.")
    if "by" in opts and not is_empty_arg(opts["by"]):
        codes, names, levels = group_codes(opts["by"])
        xv = X.flatten(order="F")
        if codes.size != xv.size:
            raise MatlabError("fitdist: the grouping variable must have one value per observation.")
        cens = vec(opts["censoring"]) if "censoring" in opts else None
        freq = vec(opts["frequency"]) if "frequency" in opts else None
        pds = []
        for g in range(len(names)):
            sel = codes == g
            x, c, f = clean_data(xv[sel], cens[sel] if cens is not None else None,
                                 freq[sel] if freq is not None else None, "fitdist")
            pds.append(fit_object(pos[1], x, c, f, opts))
        gl = levels if isinstance(levels, Cell) else Cell.from_list([np.array([[v]])
                for v in np.asarray(levels).flatten()],
                                                                     column=True)
        return outs(nargout, Cell.from_list(pds), cellstr(names), gl)
    x, c, f = clean_data(X, opts.get("censoring"), opts.get("frequency"), "fitdist")
    return fit_object(pos[1], x, c, f, opts)


# -- multivariate distributions ------------------------------------------------------------------------------

def _mv_args(X, mu, sigma, fname):
    X = np.asarray(fnum(X), dtype=float)
    if X.ndim != 2:
        raise MatlabError(f"{fname}: X must be a matrix with one point per row.")
    n, d = X.shape
    mu = np.zeros((1, d)) if mu is None or is_empty_arg(mu) else np.asarray(fnum(mu), dtype=float)
    if mu.size == 1:
        mu = np.full((1, d), float(mu.flat[0]))
    if mu.shape == (d, 1) and d > 1:
        mu = mu.T
    if mu.shape[1] != d or mu.shape[0] not in (1, n):
        raise MatlabError(f"{fname}: mu must be a row vector with one entry per column of X.")
    S = np.eye(d) if sigma is None or is_empty_arg(sigma) else np.asarray(fnum(sigma), dtype=float)
    if S.shape == (1, d) and d > 1:
        S = np.diag(S.flatten())
    if S.shape != (d, d):
        raise MatlabError(f"{fname}: Sigma must be a {d}-by-{d} matrix.")
    return X, mu, S


@builtin("mvncdf")
def _mvncdf(interp, args, nargout):
    import scipy.stats as st
    if not args or len(args) > 5:
        raise MatlabError("mvncdf: expected mvncdf(X), mvncdf(X, mu, Sigma) or mvncdf(xl, xu, mu, Sigma).")
    if len(args) == 5 and not isinstance(args[4], Struct):
        raise MatlabError("mvncdf: the fifth input must be an options structure.")
    rect = len(args) >= 4
    hi_arg = args[1] if rect else args[0]
    mu = args[2 if rect else 1] if len(args) > (2 if rect else 1) else None
    sg = args[3 if rect else 2] if len(args) > (3 if rect else 2) else None
    XU, mu, S = _mv_args(hi_arg, mu, sg, "mvncdf")
    n, d = XU.shape
    XL = np.asarray(fnum(args[0]), dtype=float) if rect else None
    if rect and XL.shape != XU.shape:
        raise MatlabError("mvncdf: the lower and upper limits must have the same size.")
    p = np.empty((n, 1))
    err = np.zeros((n, 1))
    for i in range(n):
        m = mu[i if mu.shape[0] > 1 else 0]
        if d == 1:
            s = np.sqrt(S[0, 0])
            hi = st.norm.cdf((XU[i, 0] - m[0]) / s)
            p[i, 0] = hi - (st.norm.cdf((XL[i, 0] - m[0]) / s) if rect else 0.0)
            continue
        dist = st.multivariate_normal(mean=m, cov=S, allow_singular=True)
        if rect:
            p[i, 0] = dist.cdf(XU[i], lower_limit=XL[i])
        else:
            p[i, 0] = dist.cdf(XU[i])
        err[i, 0] = 1e-8 if d <= 3 else 1e-4
    return outs(nargout, np.clip(p, 0, 1), err)


@builtin("mvtpdf")
def _mvtpdf(interp, args, nargout):
    from scipy.special import gammaln
    if len(args) != 3:
        raise MatlabError("mvtpdf: expected mvtpdf(X, C, df).")
    X, _, C = _mv_args(args[0], None, args[1], "mvtpdf")
    n, d = X.shape
    df = vec(args[2])
    if df.size not in (1, n) or np.any(df <= 0):
        raise MatlabError("mvtpdf: df must be a positive scalar or one value per row of X.")
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise MatlabError("mvtpdf: C must be a positive definite correlation matrix.") from None
    z = np.linalg.solve(L, (X / sd).T)
    q = np.sum(z ** 2, axis=0)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    logp = (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi) - 0.5 * logdet
            - (df + d) / 2 * np.log1p(q / df))
    return np.exp(logp).reshape(-1, 1)


@builtin("mvtcdf")
def _mvtcdf(interp, args, nargout):
    import scipy.stats as st
    if len(args) not in (3, 4):
        raise MatlabError("mvtcdf: expected mvtcdf(X, C, df) or mvtcdf(xl, xu, C, df).")
    rect = len(args) == 4
    XU, _, C = _mv_args(args[1] if rect else args[0], None, args[2] if rect else args[1], "mvtcdf")
    df = scalar(args[-1], "df")
    XL = np.asarray(fnum(args[0]), dtype=float) if rect else None
    n, d = XU.shape
    p = np.empty((n, 1))
    for i in range(n):
        if d == 1:
            p[i, 0] = st.t.cdf(XU[i, 0], df) - (st.t.cdf(XL[i, 0], df) if rect else 0.0)
            continue
        dist = st.multivariate_t(loc=np.zeros(d), shape=C, df=df, allow_singular=True)
        p[i, 0] = dist.cdf(XU[i], lower_limit=XL[i]) if rect else dist.cdf(XU[i])
    return np.clip(p, 0, 1)


def _psd_root(S: np.ndarray, fname: str) -> np.ndarray:
    if not np.allclose(S, S.T, atol=1e-10 * max(1.0, np.abs(S).max(initial=0.0))):
        raise MatlabError(f"{fname}: the matrix must be symmetric.")
    w, V = np.linalg.eigh((S + S.T) / 2)
    if w.size and w.min() < -1e-8 * max(1.0, w.max()):
        raise MatlabError(f"{fname}: the matrix must be positive semi-definite.")
    return V * np.sqrt(np.clip(w, 0, None))


def _corr_of(C: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(C))
    return C / np.outer(sd, sd)


@builtin("mvtrnd")
def _mvtrnd(interp, args, nargout):
    if len(args) not in (2, 3):
        raise MatlabError("mvtrnd: expected mvtrnd(C, df) or mvtrnd(C, df, n).")
    C = np.asarray(fnum(args[0]), dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise MatlabError("mvtrnd: C must be a square correlation matrix.")
    df = vec(args[1])
    n = intval(args[2], "number of samples") if len(args) > 2 else df.size
    if df.size not in (1, n) or np.any(df <= 0):
        raise MatlabError("mvtrnd: df must be a positive scalar or have one value per sample.")
    rng = rng_of(interp)
    root = _psd_root(_corr_of(C), "mvtrnd")
    z = rng.standard_normal((n, C.shape[0])) @ root.T
    g = rng.chisquare(np.broadcast_to(df, (n,))) / np.broadcast_to(df, (n,))
    return z / np.sqrt(g)[:, None]


@builtin("wishrnd")
def _wishrnd(interp, args, nargout):
    if len(args) not in (2, 3):
        raise MatlabError("wishrnd: expected wishrnd(Sigma, df) or wishrnd(Sigma, df, D).")
    S = np.asarray(fnum(args[0]), dtype=float)
    df = scalar(args[1], "df")
    d = S.shape[0]
    if S.ndim != 2 or S.shape[1] != d:
        raise MatlabError("wishrnd: Sigma must be a square matrix.")
    if given(args, 2):
        D = np.asarray(fnum(args[2]), dtype=float)
    else:
        try:
            D = np.linalg.cholesky(S).T
        except np.linalg.LinAlgError:
            raise MatlabError("wishrnd: Sigma must be symmetric and positive definite.") from None
    rng = rng_of(interp)
    if df == int(df) and df <= 81 + d:
        Z = rng.standard_normal((int(df), d)) @ D
        W = Z.T @ Z
    else:
        if df <= d - 1:
            raise MatlabError("wishrnd: df must exceed the dimension minus one.")
        A = np.zeros((d, d))  # Bartlett decomposition
        for i in range(d):
            A[i, i] = np.sqrt(rng.chisquare(df - i))
            A[i, i + 1:] = rng.standard_normal(d - i - 1)
        Z = A @ D
        W = Z.T @ Z
    return outs(nargout, (W + W.T) / 2, D)


@builtin("iwishrnd")
def _iwishrnd(interp, args, nargout):
    if len(args) not in (2, 3):
        raise MatlabError("iwishrnd: expected iwishrnd(Tau, df) or iwishrnd(Tau, df, DI).")
    T = np.asarray(fnum(args[0]), dtype=float)
    if given(args, 2):
        DI = np.asarray(fnum(args[2]), dtype=float)
    else:
        try:
            DI = np.linalg.cholesky(np.linalg.inv(T)).T
        except np.linalg.LinAlgError:
            raise MatlabError("iwishrnd: Tau must be symmetric and positive definite.") from None
    W = _wishrnd(interp, [DI.T @ DI, args[1], DI], 1)
    return outs(nargout, np.linalg.inv(W), DI)


@builtin("copularnd")
def _copularnd(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 3 or not is_text(args[0]):
        raise MatlabError("copularnd: expected copularnd('Gaussian', rho, n), copularnd('t', rho, nu, n) or "
                          "copularnd(family, alpha, n).")
    fam = choice(args[0], ["Gaussian", "t", "Clayton", "Frank", "Gumbel"], "The copula family")
    rng = rng_of(interp)

    def corr(v):
        R = np.asarray(fnum(v), dtype=float)
        if R.size == 1:
            r = float(R.flat[0])
            R = np.array([[1.0, r], [r, 1.0]])
        return R

    if fam == "gaussian":
        if len(args) != 3:
            raise MatlabError("copularnd: expected copularnd('Gaussian', rho, n).")
        R = corr(args[1])
        n = intval(args[2], "n")
        return st.norm.cdf(rng.standard_normal((n, R.shape[0])) @ _psd_root(R, "copularnd").T)
    if fam == "t":
        if len(args) != 4:
            raise MatlabError("copularnd: expected copularnd('t', rho, nu, n).")
        R = corr(args[1])
        nu = scalar(args[2], "nu")
        n = intval(args[3], "n")
        z = rng.standard_normal((n, R.shape[0])) @ _psd_root(R, "copularnd").T
        return st.t.cdf(z / np.sqrt(rng.chisquare(nu, n) / nu)[:, None], nu)
    if len(args) != 3:
        raise MatlabError("copularnd: expected copularnd(family, alpha, n).")
    alpha = scalar(args[1], "alpha")
    n = intval(args[2], "n")
    u, p = rng.random(n), rng.random(n)
    with np.errstate(all="ignore"):
        if fam == "clayton":
            if alpha < 0:
                raise MatlabError("copularnd: alpha must not be negative for the Clayton copula.")
            v = u if alpha == 0 else u * (p ** (-alpha / (1 + alpha)) - 1 + u ** alpha) ** (-1 / alpha)
            v = p if alpha == 0 else v
        elif fam == "frank":
            v = p if alpha == 0 else -np.log1p(p * np.expm1(-alpha) / (np.exp(-alpha * u) * (1 - p) + p)) / alpha
        else:
            if alpha < 1:
                raise MatlabError("copularnd: alpha must be at least 1 for the Gumbel copula.")
            # Marshall-Olkin: a positive stable frailty with index 1/alpha.
            th = (rng.random(n) - 0.5) * np.pi
            w = rng.standard_exponential(n)
            a = 1 / alpha
            s = (np.sin(a * (th + np.pi / 2)) / np.cos(th) ** (1 / a)
                 * (np.cos(th - a * (th + np.pi / 2)) / w) ** ((1 - a) / a)) if alpha > 1 else np.ones(n)
            e = rng.standard_exponential((n, 2))
            uv = np.exp(-(e / s[:, None]) ** a)
            return uv
    return np.column_stack([u, v])


# -- empirical cdf ---------------------------------------------------------------------------------------------

def ecdf_core(y, cens, freq, alpha: float, func: str):
    """Kaplan-Meier estimate. Returns (f, x, lower, upper) as 1-D arrays; the first x repeats min(y)."""
    import scipy.stats as st
    f = freq if freq is not None else np.ones(y.size)
    c = cens if cens is not None else np.zeros(y.size, dtype=bool)
    order = np.argsort(y, kind="mergesort")
    y, f, c = y[order], f[order], c[order]
    ux, inv = np.unique(y, return_inverse=True)
    deaths = np.bincount(inv, weights=f * ~c, minlength=ux.size)
    total = np.bincount(inv, weights=f, minlength=ux.size)
    n = total.sum()
    at_risk = n - np.concatenate([[0.0], np.cumsum(total)[:-1]])
    keep = deaths > 0
    x, d, r = ux[keep], deaths[keep], at_risk[keep]
    if x.size == 0:
        raise MatlabError("ecdf: there are no uncensored observations.")
    with np.errstate(all="ignore"):
        S = np.cumprod(1 - d / r)
        gw = np.cumsum(d / (r * (r - d)))
        se = S * np.sqrt(gw)
        H = np.cumsum(d / r)
        seH = np.sqrt(np.cumsum(d / r ** 2))
    z = st.norm.ppf(1 - alpha / 2)
    if func == "cdf":
        val, lo, hi = 1 - S, 1 - S - z * se, 1 - S + z * se
        first = 0.0
    elif func == "survivor":
        val, lo, hi = S, S - z * se, S + z * se
        first = 1.0
    else:
        val, lo, hi = H, H - z * seH, H + z * seH
        first = 0.0
    if func != "cumulative hazard":
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        bad = ~np.isfinite(se) | (S <= 0)
        lo[bad], hi[bad] = np.nan, np.nan
    else:
        lo = np.clip(lo, 0, None)
        bad = ~np.isfinite(seH) | (r <= d)
        lo[bad], hi[bad] = np.nan, np.nan
    return (np.concatenate([[first], val]), np.concatenate([[x[0]], x]),
            np.concatenate([[np.nan], lo]), np.concatenate([[np.nan], hi]))


@builtin("ecdf")
def _ecdf(interp, args, nargout):
    pos, opts = get_opts(args, 1, ["Censoring", "Frequency", "Alpha", "Function", "Bounds"], "ecdf")
    if len(pos) != 1:
        raise MatlabError("ecdf: expected ecdf(y, 'Name', value, ...).")
    y, c, f = clean_data(pos[0], opts.get("censoring"), opts.get("frequency"), "ecdf")
    if y.size == 0:
        raise MatlabError("ecdf: the sample is empty.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    func = choice(opts["function"], ["cdf", "survivor", "cumulative hazard", "cumhazard"], "Function") \
        if "function" in opts else "cdf"
    func = "cumulative hazard" if func == "cumhazard" else func
    bounds = as_bool(opts["bounds"], "Bounds") if "bounds" in opts else False
    fv, x, lo, hi = ecdf_core(y, c, f, alpha, func)
    if nargout == 0:
        builtin_call(interp, "stairs", [x.reshape(1, -1), fv.reshape(1, -1)], 0)
        if bounds:
            builtin_call(interp, "hold", [char("on")], 0)
            builtin_call(interp, "stairs", [x.reshape(1, -1), lo.reshape(1, -1), char("r:")], 0)
            builtin_call(interp, "stairs", [x.reshape(1, -1), hi.reshape(1, -1), char("r:")], 0)
            builtin_call(interp, "hold", [char("off")], 0)
        builtin_call(interp, "xlabel", [char("x")], 0)
        builtin_call(interp, "ylabel", [char({"cdf": "F(x)", "survivor": "S(x)"}.get(func, "H(x)"))], 0)
        return None
    return outs(nargout, fv.reshape(-1, 1), x.reshape(-1, 1), lo.reshape(-1, 1), hi.reshape(-1, 1))


@builtin("cdfplot")
def _cdfplot(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("cdfplot: expected cdfplot(x).")
    x = vec(args[0])
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise MatlabError("cdfplot: the sample is empty.")
    fv, xs, _, _ = ecdf_core(x, None, None, 0.05, "cdf")
    span = (xs[-1] - xs[0]) or 1.0
    px = np.concatenate([[xs[0] - 0.05 * span], xs[1:], [xs[-1] + 0.05 * span]])
    py = np.concatenate([[0.0], fv[1:], [1.0]])
    h = builtin_call(interp, "stairs", [px.reshape(1, -1), py.reshape(1, -1)], 1)
    builtin_call(interp, "xlabel", [char("x")], 0)
    builtin_call(interp, "ylabel", [char("F(x)")], 0)
    builtin_call(interp, "title", [char("Empirical CDF")], 0)
    builtin_call(interp, "grid", [char("on")], 0)
    stats = struct(min=float(x.min()), max=float(x.max()), mean=float(x.mean()), median=float(np.median(x)),
                   std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0)
    return outs(nargout, h[0], stats) if nargout else None


# -- ksdensity -----------------------------------------------------------------------------------------------------

_KS_OPTS = ["Bandwidth", "Width", "Kernel", "Function", "NumPoints", "NPoints", "Support", "Weights",
            "BoundaryCorrection", "Censoring", "PlotFcn"]


@builtin("ksdensity")
def _ksdensity(interp, args, nargout):
    pos, opts = get_opts(args, 1, _KS_OPTS, "ksdensity")
    if not 1 <= len(pos) <= 2:
        raise MatlabError("ksdensity: expected ksdensity(x) or ksdensity(x, pts, 'Name', value, ...).")
    for o in ("censoring", "plotfcn"):
        if o in opts:
            raise MatlabError(f"ksdensity: the option '{o}' is not supported.")
    X = np.asarray(fnum(pos[0]), dtype=float)
    kernel = choice(opts["kernel"], _KERNELS + ["gaussian"], "Kernel") if "kernel" in opts else "normal"
    kernel = "normal" if kernel == "gaussian" else kernel
    func = choice(opts["function"], ["pdf", "cdf", "icdf", "survivor", "cumhazard"], "Function") \
        if "function" in opts else "pdf"
    bw_arg = opts.get("bandwidth", opts.get("width"))
    bw_given = bw_arg is not None and not is_text(bw_arg) and not is_empty_arg(bw_arg)
    npts = intval(opts.get("numpoints", opts.get("npoints", np.array([[100.0]]))), "NumPoints")
    if X.ndim == 2 and X.shape[1] == 2 and X.shape[0] > 2:
        return _ksdensity2(X, pos, opts, kernel, func, bw_arg if bw_given else None, nargout)
    x = X.flatten(order="F")
    w = None
    if "weights" in opts:
        w = vec(opts["weights"])
        if w.size != x.size or np.any(w < 0):
            raise MatlabError("ksdensity: Weights must have one non-negative entry per sample value.")
    keep = ~np.isnan(x)
    x, w = x[keep], (w[keep] if w is not None else None)
    if x.size == 0:
        raise MatlabError("ksdensity: the sample is empty.")
    if w is not None:
        w = w / w.sum()
    reflect = "boundarycorrection" in opts and choice(opts["boundarycorrection"], ["log", "reflection"],
                                                      "BoundaryCorrection") == "reflection"
    kde = KDE(x, w, kernel, scalar(bw_arg, "Bandwidth") if bw_given else None,
              _Support(opts.get("support"), x, "ksdensity"), reflect)
    if len(pos) > 1 and not is_empty_arg(pos[1]):
        xi = np.asarray(fnum(pos[1]), dtype=float)
        shape = xi.shape
        xi = xi.flatten(order="F")
    else:
        xi = (np.arange(1, npts + 1) - 0.5) / npts if func == "icdf" else kde.grid(npts)
        shape = (1, npts)
    with np.errstate(all="ignore"):
        if func == "pdf":
            f = kde.pdf(xi)
        elif func == "icdf":
            f = kde.icdf(xi)
        else:
            c = kde.cdf(xi)
            f = c if func == "cdf" else (1 - c if func == "survivor" else -np.log(1 - c))
    if nargout == 0:
        builtin_call(interp, "plot", [xi.reshape(1, -1), f.reshape(1, -1)], 0)
        return None
    return outs(nargout, f.reshape(shape, order="F"), xi.reshape(shape, order="F"), np.array([[kde.bw]]))


def _ksdensity2(X, pos, opts, kernel, func, bw_arg, nargout):
    """Bivariate kernel density (product kernel) on a 30-by-30 grid or at given points."""
    if func not in ("pdf", "cdf"):
        raise MatlabError("ksdensity: bivariate data support 'Function' 'pdf' and 'cdf' only.")
    for o in ("support", "boundarycorrection"):
        if o in opts:
            raise MatlabError(f"ksdensity: the option '{o}' is not supported for bivariate data.")
    X = X[~np.isnan(X).any(axis=1)]
    n = X.shape[0]
    w = np.full(n, 1.0 / n)
    if "weights" in opts:
        w = vec(opts["weights"])
        w = w / w.sum()
    if bw_arg is not None:
        bw = vec(bw_arg)
        bw = np.repeat(bw, 2) if bw.size == 1 else bw
    else:
        bw = np.array([default_bandwidth(X[:, j], d=2) for j in range(2)])
    if len(pos) > 1 and not is_empty_arg(pos[1]):
        pts = np.asarray(fnum(pos[1]), dtype=float)
        if pts.shape[1] != 2:
            raise MatlabError("ksdensity: the evaluation points must have two columns.")
    else:
        g1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
        g2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
        G1, G2 = np.meshgrid(g1, g2)
        pts = np.column_stack([G1.flatten(order="F"), G2.flatten(order="F")])
    fn = kernel_pdf if func == "pdf" else kernel_cdf
    k1 = fn((pts[:, None, 0] - X[None, :, 0]) / bw[0], kernel)
    k2 = fn((pts[:, None, 1] - X[None, :, 1]) / bw[1], kernel)
    f = (k1 * k2) @ w
    if func == "pdf":
        f = f / (bw[0] * bw[1])
    return outs(nargout, f.reshape(-1, 1), pts, bw.reshape(1, -1))


# -- distribution plots ----------------------------------------------------------------------------------------------

def _handles(vals: list) -> list:
    out = []
    for v in vals:
        items = getattr(v, "items", None)
        out.extend(items if items is not None else [v])
    return out


def _harray(items: list):
    from .plotting import GraphicsArray
    return items[0] if len(items) == 1 else GraphicsArray(items)


def _quartile_line(x: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the line through the first and third quartiles of (q, x)."""
    x1, x3 = np.quantile(x, [0.25, 0.75], method="hazen")
    q1, q3 = np.quantile(q, [0.25, 0.75], method="hazen")
    slope = (x3 - x1) / (q3 - q1) if q3 != q1 else 0.0
    return slope, x1 - slope * q1


def _plot_positions(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) - 0.5) / n


_PROB_TICKS = np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995,
                        0.999, 0.9995, 0.9999])


def _prob_axis(interp, quantile_fn, pmin: float, pmax: float) -> None:
    ticks = _PROB_TICKS[(_PROB_TICKS >= pmin * 0.999) & (_PROB_TICKS <= 1 - (1 - pmax) * 0.999)]
    if ticks.size < 2:
        ticks = _PROB_TICKS[4:13]
    gca = builtin_call(interp, "gca", [], 1)[0]
    builtin_call(interp, "set", [gca, char("YTick"), quantile_fn(ticks).reshape(1, -1), char("YTickLabel"),
                                 cellstr([f"{t:g}" for t in ticks], column=False)], 0)
    builtin_call(interp, "ylabel", [char("Probability")], 0)
    builtin_call(interp, "grid", [char("on")], 0)


@builtin("normplot")
def _normplot(interp, args, nargout):
    import scipy.stats as st
    if len(args) != 1:
        raise MatlabError("normplot: expected normplot(x).")
    X = np.asarray(fnum(args[0]), dtype=float)
    if X.ndim == 2 and X.shape[0] == 1:
        X = X.reshape(-1, 1)
    hs = []
    pmin, pmax = 0.5, 0.5
    for j in range(X.shape[1]):
        x = np.sort(X[:, j][~np.isnan(X[:, j])])
        if x.size == 0:
            continue
        p = _plot_positions(x.size)
        q = st.norm.ppf(p)
        pmin, pmax = min(pmin, p[0]), max(pmax, p[-1])
        hs += _handles(builtin_call(interp, "plot", [x.reshape(1, -1), q.reshape(1, -1), char("+")], 1))
        builtin_call(interp, "hold", [char("on")], 0)
        slope, icpt = _quartile_line(x, q)
        if slope > 0:
            ends = np.array([x[0], x[-1]])
            hs += _handles(builtin_call(interp, "plot", [ends.reshape(1, -1), ((ends - icpt) / slope).reshape(1, -1),
                                                         char("r-.")], 1))
    builtin_call(interp, "hold", [char("off")], 0)
    _prob_axis(interp, st.norm.ppf, pmin, pmax)
    builtin_call(interp, "xlabel", [char("Data")], 0)
    builtin_call(interp, "title", [char("Normal Probability Plot")], 0)
    return _harray(hs) if nargout and hs else None


_PROBPLOT = {"normal": False, "exponential": False, "extremevalue": False, "lognormal": True, "rayleigh": False,
             "weibull": True, "logistic": False, "loglogistic": True, "halfnormal": False}


@builtin("probplot")
def _probplot(interp, args, nargout):
    if not args:
        raise MatlabError("probplot: expected probplot(y) or probplot(distname, y).")
    rest = list(args)
    name = "normal"
    pd = None
    if is_text(rest[0]):
        name = _key(text(rest.pop(0)))
    elif isinstance(rest[0], (ProbDist,)):
        pd = rest.pop(0)
        name = _key(pd.fam.name)
    if name not in _PROBPLOT:
        raise MatlabError("probplot: supported distributions are normal, exponential, extreme value, lognormal, "
                          "rayleigh, weibull, logistic, loglogistic and half normal.")
    if len(rest) < 1 or len(rest) > 3:
        raise MatlabError("probplot: expected probplot(distname, y, cens, freq).")
    if is_text(rest[-1]):
        raise MatlabError(f"probplot: the option '{text(rest[-1])}' is not supported.")
    y, cens, freq = clean_data(rest[0], rest[1] if len(rest) > 1 else None, rest[2] if len(rest) > 2 else None,
                               "probplot")
    fam = FAMILIES[name]
    std = [np.float64(v) for v in fam.safe]
    logx = _PROBPLOT[name]
    if logx:  # the reference distribution lives on the log scale
        fam = FAMILIES["normal" if name == "lognormal" else ("extremevalue" if name == "weibull" else "logistic")]
        std = [np.float64(0), np.float64(1)]
    if cens is not None or freq is not None:
        fv, xs, _, _ = ecdf_core(y, cens, freq, 0.05, "cdf")
        x = xs[1:]
        p = (fv[:-1] + fv[1:]) / 2
    else:
        x = np.sort(y)
        p = _plot_positions(x.size)
    if x.size == 0:
        raise MatlabError("probplot: there are no data to plot.")
    q = fam.ev("ppf", p, std)
    xp = np.log(x) if logx else x
    h = _handles(builtin_call(interp, "plot", [x.reshape(1, -1), q.reshape(1, -1), char("+")], 1))
    builtin_call(interp, "hold", [char("on")], 0)
    if pd is not None:
        qq = fam.ev("ppf", pd.cdf(x), std)
        h += _handles(builtin_call(interp, "plot", [x.reshape(1, -1), qq.reshape(1, -1), char("r-")], 1))
    else:
        slope, icpt = _quartile_line(xp, q)
        if slope > 0:
            ends = np.array([xp[0], xp[-1]])
            ex = np.exp(ends) if logx else ends
            h += _handles(builtin_call(interp, "plot", [ex.reshape(1, -1), ((ends - icpt) / slope).reshape(1, -1),
                                                        char("r-.")], 1))
    builtin_call(interp, "hold", [char("off")], 0)
    if logx:
        builtin_call(interp, "set", [builtin_call(interp, "gca", [], 1)[0], char("XScale"), char("log")], 0)
    _prob_axis(interp, lambda t: fam.ev("ppf", t, std), float(p[0]), float(p[-1]))
    builtin_call(interp, "xlabel", [char("Data")], 0)
    builtin_call(interp, "title", [char(f"Probability plot for {FAMILIES[name].display} distribution")], 0)
    return _harray(h) if nargout else None


@builtin("qqplot")
def _qqplot(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("qqplot: expected qqplot(x), qqplot(x, y), qqplot(x, pd) or qqplot(x, y, pvec).")
    x = vec(args[0])
    x = np.sort(x[~np.isnan(x)])
    if x.size == 0:
        raise MatlabError("qqplot: the sample is empty.")
    xlabel = "Standard Normal Quantiles"
    ylabel = "Quantiles of Input Sample"
    if len(args) == 1 or isinstance(args[1], (ProbDist, KernelDist)):
        p = _plot_positions(x.size)
        if len(args) > 1:
            q = np.asarray(args[1].icdf(p), dtype=float)
            xlabel = "Quantiles of the distribution"
        else:
            q = st.norm.ppf(p)
        qx, qy = q, x
    else:
        y = vec(args[1])
        y = np.sort(y[~np.isnan(y)])
        if len(args) > 2:
            pv = vec(args[2]) / 100.0
        else:
            n = min(x.size, y.size)
            pv = _plot_positions(n)
        qx = np.quantile(x, pv, method="hazen")
        qy = np.quantile(y, pv, method="hazen")
        xlabel, ylabel = "X Quantiles", "Y Quantiles"
    h = _handles(builtin_call(interp, "plot", [qx.reshape(1, -1), qy.reshape(1, -1), char("+")], 1))
    slope, icpt = _quartile_line(qy, qx)
    builtin_call(interp, "hold", [char("on")], 0)
    ends = np.array([qx.min(), qx.max()])
    h += _handles(builtin_call(interp, "plot", [ends.reshape(1, -1), (icpt + slope * ends).reshape(1, -1),
                                                char("r-.")], 1))
    builtin_call(interp, "hold", [char("off")], 0)
    builtin_call(interp, "xlabel", [char(xlabel)], 0)
    builtin_call(interp, "ylabel", [char(ylabel)], 0)
    builtin_call(interp, "title", [char("QQ Plot of Sample Data versus " +
                                        ("Standard Normal" if len(args) == 1 else "Distribution"))], 0)
    return _harray(h) if nargout else None


@builtin("histfit")
def _histfit(interp, args, nargout):
    if not 1 <= len(args) <= 3:
        raise MatlabError("histfit: expected histfit(data), histfit(data, nbins) or histfit(data, nbins, dist).")
    x = vec(args[0])
    x = x[~np.isnan(x)]
    if x.size < 2:
        raise MatlabError("histfit: the data must have at least two values.")
    nbins = intval(args[1], "nbins") if given(args, 1) else int(np.ceil(np.sqrt(x.size)))
    dist = args[2] if len(args) > 2 else char("normal")
    pd = fit_object(dist, x, None, None, {}, "histfit")
    counts, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    hb = _handles(builtin_call(interp, "bar", [centers.reshape(1, -1), counts.astype(float).reshape(1, -1),
                                               np.array([[1.0]])], 1))
    builtin_call(interp, "hold", [char("on")], 0)
    span = edges[-1] - edges[0]
    xs = np.linspace(edges[0] - 0.1 * span, edges[-1] + 0.1 * span, 200) \
        if isinstance(pd, KernelDist) or not pd.fam.discrete else np.arange(np.floor(edges[0]), np.ceil(edges[-1]) + 1)
    ys = pd.pdf(xs) * x.size * width
    hl = _handles(builtin_call(interp, "plot", [xs.reshape(1, -1), ys.reshape(1, -1), char("r-"),
                                                char("LineWidth"), np.array([[2.0]])], 1))
    builtin_call(interp, "hold", [char("off")], 0)
    return _harray(hb + hl) if nargout else None
