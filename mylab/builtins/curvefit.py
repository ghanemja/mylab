"""Curve Fitting Toolbox core: fittype, fitoptions, fit, and the cfit and sfit objects.

``ft = fittype('exp1')`` names a library model (polynomials, exponentials, power, Gaussian, sum of sines,
Fourier series, rational, Weibull, interpolants and smoothing splines); ``fittype('a*exp(-b*x)+c')``,
``fittype(@(a,b,x) a*x+b)`` and ``fittype({'x','sin(x)','1'})`` make custom models. ``[f, gof, output] =
fit(x, y, ft, ...)`` fits a model: linear models by (weighted, optionally bounded or robust) linear least
squares, nonlinear ones with SciPy's ``least_squares`` (trust-region-reflective with bounds, or
Levenberg-Marquardt) from data-driven starting points. ``f`` is a cfit (curves) or sfit (surfaces) object:
``f(x)``, ``confint(f)``, ``predint(f, x)``, ``differentiate``, ``integrate``, ``plot(f, x, y)`` and so on.
Implemented from the documented behaviour of the MATLAB functions.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char, class_name
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, match_option, parse_options, text, unknown_option_error

_EPS = np.finfo(float).eps


def _g(v: float) -> str:
    """A number as MATLAB's %.4g prints it."""
    s = f"{v:.4g}"
    return s.replace("nan", "NaN").replace("inf", "Inf")


def _row(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(1, -1)


def _col(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _cellstr(names, column: bool = True) -> Cell:
    return Cell.from_list([char(n) for n in names], column=column)


def _text_list(v: Any, what: str) -> list[str]:
    if is_text(v):
        return [text(v)]
    if isinstance(v, Cell):
        return [text(x, what) for x in v.data.flatten(order="F")]
    from ..values import StringArray
    if isinstance(v, StringArray):
        return [str(x) for x in v.data.flatten(order="F")]
    raise MatlabError(f"The {what} must be text or a cell array of text.")


# ==========================================================================================================
# fittype
# ==========================================================================================================

class FitType(MObject):
    """A model: library, custom (equation, anonymous function or linear terms), interpolant or spline."""

    def __init__(self, *, name: str, label: str, category: str, coeffs: list[str], expr: str,
                 fn: Callable | None = None, basis: Callable | None = None, probs: list[str] | None = None,
                 indep: list[str] | None = None, dep: str | None = None, start: Callable | None = None,
                 lower: list[float] | None = None, upper: list[float] | None = None, nonneg_x: bool = False):
        self.name = name                    # type(): 'poly2', 'exp1', 'customnonlinear', 'linearinterp', ...
        self.label = label                  # display header: 'Linear model Poly2', 'General model', ...
        self.category = category            # 'library', 'custom', 'interpolant', 'spline', 'lowess'
        self.coeffs = list(coeffs)
        self.probs = list(probs or [])
        self.indep = list(indep or ["x"])
        self.dep = dep or ("z" if len(self.indep) == 2 else "y")
        self.expr = expr
        self.basis = basis                  # linear models: (X, P) -> n-by-p design matrix
        self.fn = fn                        # (c, X, P) -> values
        self.start = start                  # (X, y, w) -> list of candidate start points
        self.lower = lower
        self.upper = upper
        self.nonneg_x = nonneg_x            # power and Weibull models need x > 0
        self.options: FitOptions | None = None

    @property
    def linear(self) -> bool:
        return self.basis is not None

    @property
    def parametric(self) -> bool:
        return self.category in ("library", "custom")

    def method(self) -> str:
        if self.category == "interpolant":
            from ._surffit import METHODS
            return {**METHODS, "pchipinterp": "PchipInterpolant"}[self.name]
        if self.category == "spline":
            return "SmoothingSpline"
        if self.category == "lowess":
            return "LowessFit"
        return "LinearLeastSquares" if self.linear else "NonlinearLeastSquares"

    def evaluate(self, c: np.ndarray, X: list[np.ndarray], P: list[float]) -> np.ndarray:
        if self.basis is not None:
            A = self.basis(X, P)
            return A @ np.asarray(c, dtype=float)
        return self.fn(np.asarray(c, dtype=float), X, P)

    # -- identity and members -----------------------------------------------------------------------------
    def class_name(self) -> str:
        return "fittype"

    def property_names(self) -> list[str]:
        return []

    def method_names(self) -> list[str]:
        return sorted(_FT_METHODS, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _FT_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        return _call(self, _FT_METHODS, interp, name, args, nargout)

    def argnames(self) -> list[str]:
        return self.coeffs + self.probs + self.indep

    def formula_text(self) -> str:
        if self.category == "interpolant" and len(self.indep) == 2:
            return {"biharmonicinterp": "biharmonic surface computed from p",
                    "thinplateinterp": "thin-plate spline surface computed from p"}.get(
                        self.name, "piecewise polynomial surface computed from p")
        if self.category == "interpolant" or self.category == "spline":
            return "piecewise polynomial computed from p"
        if self.category == "lowess":
            return ("lowess (linear)" if self.name == "lowess" else "loess (quadratic)") + \
                " smoothing regression computed from p"
        return self.expr

    def header(self) -> str:
        return self.label + ":"

    def index_get(self, interp, args: list) -> Any:
        return _ft_feval(self, interp, list(args), 1)

    def display_lines(self, name: str = "f") -> list[str]:
        return [f"     {self.header()}", f"     {name}({','.join(self.argnames())}) = {self.formula_text()}"]

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.display_lines(name)) + "\n\n"


def _call(obj, table: dict, interp, name: str, args: list, nargout: int) -> list:
    from ..values import to_matlab
    fn = table.get(name)
    if fn is None:
        raise MatlabError(f"No method '{name}' for class '{obj.class_name()}'.")
    target = next((a for a in args if a is obj), None)
    if target is None:
        target = next((a for a in args if isinstance(a, type(obj))), obj)
    rest = list(args)
    rest.remove(target)
    res = fn(target, interp, rest, nargout)
    if res is None:
        return []
    return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]


# -- library models ----------------------------------------------------------------------------------------

def _poly_text(deg: int, var: str = "x", prefix: str = "p") -> str:
    terms = []
    for k in range(deg + 1):
        p = deg - k
        t = f"{prefix}{k + 1}"
        if p == 1:
            t += f"*{var}"
        elif p > 1:
            t += f"*{var}^{p}"
        terms.append(t)
    return " + ".join(terms)


def _poly(deg: int) -> FitType:
    return FitType(name=f"poly{deg}", label=f"Linear model Poly{deg}", category="library",
                   coeffs=[f"p{k + 1}" for k in range(deg + 1)], expr=_poly_text(deg),
                   basis=lambda X, P: np.column_stack([X[0] ** (deg - k) for k in range(deg + 1)]))


def _surface_terms(i: int, j: int) -> list[tuple[int, int]]:
    out = []
    for total in range(max(i, j) + 1):
        for a in range(min(total, i), -1, -1):
            b = total - a
            if b <= j:
                out.append((a, b))
    return out


def _poly_surface(i: int, j: int) -> FitType:
    terms = _surface_terms(i, j)

    def term_text(a, b):
        t = f"p{a}{b}"
        if a:
            t += "*x" + (f"^{a}" if a > 1 else "")
        if b:
            t += "*y" + (f"^{b}" if b > 1 else "")
        return t
    return FitType(name=f"poly{i}{j}", label=f"Linear model Poly{i}{j}", category="library",
                   coeffs=[f"p{a}{b}" for a, b in terms], expr=" + ".join(term_text(a, b) for a, b in terms),
                   indep=["x", "y"], dep="z",
                   basis=lambda X, P: np.column_stack([X[0] ** a * X[1] ** b for a, b in terms]))


def _lin_sse(A: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    if not np.all(np.isfinite(A)):
        return np.inf, np.zeros(A.shape[1])
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    r = (A @ coef - y) * sw
    return float(r @ r), coef


def _rate_grid(x: np.ndarray, n: int = 24) -> np.ndarray:
    span = float(np.ptp(x)) or 1.0
    g = np.logspace(-2, 1.5, n) / span
    return np.concatenate([-g[::-1], [0.0], g])


def _best(cands: list[tuple[float, list]], k: int) -> list[list]:
    cands = [c for c in cands if np.isfinite(c[0])]
    cands.sort(key=lambda c: c[0])
    return [c[1] for c in cands[:k]]


def _start_exp1(X, y, w):
    x = X[0]
    rates = list(_rate_grid(x))
    pos = y > 0 if np.sum(y > 0) >= np.sum(y < 0) else y < 0
    if np.sum(pos) >= 2 and np.ptp(x[pos]) > 0:
        rates.append(np.polyfit(x[pos], np.log(np.abs(y[pos])), 1)[0])
    cands = []
    with np.errstate(all="ignore"):
        for b in rates:
            sse, a = _lin_sse(np.exp(b * x)[:, None], y, w)
            cands.append((sse, [a[0], b]))
    return _best(cands, 3)


def _start_exp2(X, y, w):
    x = X[0]
    rates = _rate_grid(x, 14)
    cands = []
    with np.errstate(all="ignore"):
        for i, b in enumerate(rates):
            for d in rates[i + 1:]:
                sse, ac = _lin_sse(np.column_stack([np.exp(b * x), np.exp(d * x)]), y, w)
                cands.append((sse, [ac[0], b, ac[1], d]))
        for s in _start_exp1(X, y, w)[:1]:
            a, b = s
            cands.append((0.0, [a, b, a * 0.01, b / 2 if b else -0.1]))
    return _best(cands, 6)


def _start_power(X, y, w, with_c: bool):
    x = X[0]
    bs = list(np.linspace(-4, 4, 81))
    pos = y > 0
    if np.sum(pos) >= 2 and np.ptp(np.log(x[pos])) > 0:
        bs.append(np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)[0])
    cands = []
    with np.errstate(all="ignore"):
        for b in bs:
            A = np.column_stack([x ** b, np.ones_like(x)]) if with_c else (x ** b)[:, None]
            sse, coef = _lin_sse(A, y, w)
            cands.append((sse, [coef[0], b] + ([coef[1]] if with_c else [])))
    return _best(cands, 3)


def _gauss_fn(N: int):
    def fn(c, X, P):
        x = X[0]
        out = np.zeros(np.shape(x))
        for k in range(N):
            a, b, cc = c[3 * k:3 * k + 3]
            with np.errstate(all="ignore"):
                out = out + a * np.exp(-((x - b) / cc) ** 2)
        return out
    return fn


def _greedy_start(N: int, X, y, w, per_term: int, term_cands: Callable, full_fn: Callable,
                  lower_fn: Callable) -> list[list]:
    """Add one term at a time (best candidate against the residual), refining all terms after each."""
    from scipy.optimize import least_squares
    params: list[float] = []
    sw = np.sqrt(w)
    for k in range(N):
        r = y - (full_fn(np.array(params), X, [], k) if params else 0.0)
        best = term_cands(X, r, w)
        if not best:
            best = [[0.0] * per_term]
        params = params + list(best[0])
        p0 = np.array(params, dtype=float)
        lb = lower_fn(k + 1)
        p0 = np.maximum(p0, lb + 1e-12 * (np.abs(lb) + 1))
        try:
            res = least_squares(lambda c: (full_fn(c, X, [], k + 1) - y) * sw, p0, bounds=(lb, np.inf),
                                method="trf", x_scale="jac", max_nfev=200 * (k + 1))
            if np.all(np.isfinite(res.x)):
                params = list(res.x)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return [params]


def _gauss(N: int) -> FitType:
    coeffs = [f"{v}{k}" for k in range(1, N + 1) for v in "abc"]
    expr = " + ".join(f"a{k}*exp(-((x-b{k})/c{k})^2)" for k in range(1, N + 1))
    lower = [-np.inf, -np.inf, 0.0] * N
    gfull = _gauss_fn(N)

    def full(c, X, P, k=N):
        return _gauss_fn(k)(c, X, P)

    def term_cands(X, r, w):
        x = X[0]
        span = float(np.ptp(x)) or 1.0
        order = np.argsort(-np.abs(r))
        centers = list(x[order[:5]]) + list(np.quantile(x, np.linspace(0, 1, 11)))
        widths = span * np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0])
        cands = []
        with np.errstate(all="ignore"):
            for b in centers:
                for c in widths:
                    sse, a = _lin_sse(np.exp(-((x - b) / c) ** 2)[:, None], r, w)
                    cands.append((sse, [a[0], b, c]))
        return _best(cands, 1)

    def start(X, y, w):
        return _greedy_start(N, X, y, w, 3, term_cands, full, lambda k: np.array(lower[:3 * k]))
    return FitType(name=f"gauss{N}", label=f"General model Gauss{N}", category="library", coeffs=coeffs,
                   expr=expr, fn=gfull, start=start, lower=lower)


def _freq_grid(x: np.ndarray, harmonics: int = 1) -> np.ndarray:
    span = float(np.ptp(x)) or 1.0
    n = x.size
    top = max((n - 1) / 2.0 / harmonics, 1.0)
    m = int(min(1500, max(200, 20 * n)))
    return 2 * np.pi / span * np.linspace(0.05, top, m)


def _sin_fn(N: int):
    def fn(c, X, P):
        x = X[0]
        out = np.zeros(np.shape(x))
        for k in range(N):
            a, b, cc = c[3 * k:3 * k + 3]
            out = out + a * np.sin(b * x + cc)
        return out
    return fn


def _sin(N: int) -> FitType:
    coeffs = [f"{v}{k}" for k in range(1, N + 1) for v in "abc"]
    expr = " + ".join(f"a{k}*sin(b{k}*x+c{k})" for k in range(1, N + 1))
    lower = [-np.inf, 0.0, -np.inf] * N

    def full(c, X, P, k=N):
        return _sin_fn(k)(c, X, P)

    def term_cands(X, r, w):
        x = X[0]
        grid = _freq_grid(x)
        cands = []
        for b in grid:
            sse, AB = _lin_sse(np.column_stack([np.sin(b * x), np.cos(b * x)]), r, w)
            cands.append((sse, b, AB))
        cands.sort(key=lambda c: c[0])
        sse, b, AB = cands[0]
        step = grid[1] - grid[0]
        for bb in np.linspace(b - step, b + step, 41):  # refine the frequency
            s2, ab2 = _lin_sse(np.column_stack([np.sin(bb * x), np.cos(bb * x)]), r, w)
            if s2 < sse and bb > 0:
                sse, b, AB = s2, bb, ab2
        A, B = AB
        return [[float(np.hypot(A, B)), float(b), float(np.arctan2(B, A))]]

    def start(X, y, w):
        return _greedy_start(N, X, y, w, 3, term_cands, full, lambda k: np.array(lower[:3 * k]))
    return FitType(name=f"sin{N}", label=f"General model Sin{N}", category="library", coeffs=coeffs,
                   expr=expr, fn=_sin_fn(N), start=start, lower=lower)


def _fourier_design(x: np.ndarray, w: float, N: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for k in range(1, N + 1):
        cols += [np.cos(k * w * x), np.sin(k * w * x)]
    return np.column_stack(cols)


def _fourier(N: int) -> FitType:
    coeffs = ["a0"] + [f"{v}{k}" for k in range(1, N + 1) for v in "ab"] + ["w"]
    terms = ["a0"]
    for k in range(1, N + 1):
        m = "" if k == 1 else f"{k}*"
        terms += [f"a{k}*cos({m}x*w)", f"b{k}*sin({m}x*w)"]

    def fn(c, X, P):
        return _fourier_design(np.asarray(X[0], dtype=float).reshape(-1), c[-1], N).dot(c[:-1]).reshape(
            np.shape(X[0]))

    def start(X, y, wts):
        x = X[0]
        grid = _freq_grid(x, N)
        cands = []
        for om in grid:
            sse, coef = _lin_sse(_fourier_design(x, om, N), y, wts)
            cands.append((sse, list(coef) + [om]))
        best = _best(cands, 3)
        step = grid[1] - grid[0]
        out = []
        for b in best:
            om0 = b[-1]
            ref = [(_lin_sse(_fourier_design(x, om, N), y, wts), om) for om in np.linspace(om0 - step, om0 + step, 41)
                   if om > 0]
            (sse, coef), om = min(ref, key=lambda t: t[0][0])
            out.append(list(coef) + [om])
        return out
    return FitType(name=f"fourier{N}", label=f"General model Fourier{N}", category="library", coeffs=coeffs,
                   expr=" + ".join(terms), fn=fn, start=start)


def _rat(n: int, m: int) -> FitType:
    coeffs = [f"p{k + 1}" for k in range(n + 1)] + [f"q{k + 1}" for k in range(m)]
    num = _poly_text(n)
    den_terms = ["x" + (f"^{m}" if m > 1 else "")]
    for k in range(1, m + 1):
        p = m - k
        den_terms.append(f"q{k}" + ("" if p == 0 else "*x" if p == 1 else f"*x^{p}"))
    expr = f"({num}) / ({' + '.join(den_terms)})"

    def fn(c, X, P):
        x = X[0]
        with np.errstate(all="ignore"):
            return np.polyval(c[:n + 1], x) / np.polyval(np.concatenate([[1.0], c[n + 1:]]), x)

    def start(X, y, w):
        x = X[0]
        A = np.column_stack([x ** (n - k) for k in range(n + 1)] + [-y * x ** (m - k) for k in range(1, m + 1)])
        sse, coef = _lin_sse(A, y * x ** m, w)
        return [list(coef), [1.0] * (n + 1 + m)]
    return FitType(name=f"rat{n}{m}", label=f"General model Rat{n}{m}", category="library", coeffs=coeffs,
                   expr=expr, fn=fn, start=start)


def _exp(N: int) -> FitType:
    if N == 1:
        return FitType(name="exp1", label="General model Exp1", category="library", coeffs=["a", "b"],
                       expr="a*exp(b*x)", fn=lambda c, X, P: c[0] * np.exp(c[1] * X[0]), start=_start_exp1)
    return FitType(name="exp2", label="General model Exp2", category="library", coeffs=["a", "b", "c", "d"],
                   expr="a*exp(b*x) + c*exp(d*x)",
                   fn=lambda c, X, P: c[0] * np.exp(c[1] * X[0]) + c[2] * np.exp(c[3] * X[0]), start=_start_exp2)


def _power(N: int) -> FitType:
    if N == 1:
        return FitType(name="power1", label="General model Power1", category="library", coeffs=["a", "b"],
                       expr="a*x^b", fn=lambda c, X, P: c[0] * X[0] ** c[1],
                       start=lambda X, y, w: _start_power(X, y, w, False), nonneg_x=True)
    return FitType(name="power2", label="General model Power2", category="library", coeffs=["a", "b", "c"],
                   expr="a*x^b+c", fn=lambda c, X, P: c[0] * X[0] ** c[1] + c[2],
                   start=lambda X, y, w: _start_power(X, y, w, True), nonneg_x=True)


def _weibull() -> FitType:
    def fn(c, X, P):
        a, b = c
        x = X[0]
        with np.errstate(all="ignore"):
            return a * b * x ** (b - 1) * np.exp(-a * x ** b)

    def start(X, y, w):
        cands = []
        sw = np.sqrt(w)
        for a in np.logspace(-4, 4, 33):
            for b in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
                r = (fn([a, b], X, []) - y) * sw
                cands.append((float(r @ r), [a, b]))
        return _best(cands, 3)
    return FitType(name="weibull", label="General model Weibull", category="library", coeffs=["a", "b"],
                   expr="a*b*x^(b-1)*exp(-a*x^b)", fn=fn, start=start, nonneg_x=True)


_INTERP_LABELS = {"linearinterp": "Linear interpolant", "nearestinterp": "Nearest neighbor interpolant",
                  "cubicinterp": "Cubic interpolant", "pchipinterp": "Shape-preserving (pchip) interpolant"}


def _surface_interp(n: str) -> FitType:
    from ._surffit import SURFACE_INTERPS
    return FitType(name=n, label=SURFACE_INTERPS[n], category="interpolant", coeffs=["p"], expr="",
                   indep=["x", "y"], dep="z")


def _library(name: str) -> FitType | None:
    n = name.lower()
    if n in ("biharmonicinterp", "thinplateinterp"):  # surfaces only
        return _surface_interp(n)
    if n in _INTERP_LABELS:
        return FitType(name=n, label=_INTERP_LABELS[n], category="interpolant", coeffs=["p"], expr="")
    if n == "smoothingspline":
        return FitType(name=n, label="Smoothing spline", category="spline", coeffs=["p"], expr="")
    if n in ("lowess", "loess"):
        return FitType(name=n, label="Locally weighted smoothing " + ("linear" if n == "lowess" else "quadratic")
                       + " regression", category="lowess", coeffs=["p"], expr="", indep=["x", "y"], dep="z")
    if n == "weibull":
        return _weibull()
    mt = re.fullmatch(r"(poly|exp|power|gauss|sin|fourier|rat)(\d+)", n)
    if not mt:
        return None
    kind, digits = mt.group(1), mt.group(2)
    if kind == "poly":
        if len(digits) == 1 and 1 <= int(digits) <= 9:
            return _poly(int(digits))
        if len(digits) == 2 and all(1 <= int(d) <= 5 for d in digits) or digits in ("10", "01"):
            return _poly_surface(int(digits[0]), int(digits[1]))
        return None
    if kind == "rat":
        if len(digits) == 2 and 0 <= int(digits[0]) <= 5 and 1 <= int(digits[1]) <= 5:
            return _rat(int(digits[0]), int(digits[1]))
        return None
    N = int(digits)
    if kind == "exp" and N in (1, 2):
        return _exp(N)
    if kind == "power" and N in (1, 2):
        return _power(N)
    if kind in ("gauss", "sin", "fourier") and 1 <= N <= 8:
        return {"gauss": _gauss, "sin": _sin, "fourier": _fourier}[kind](N)
    return None


# -- custom models -----------------------------------------------------------------------------------------

_KNOWN_NAMES = {"pi", "eps", "Inf", "inf", "NaN", "nan", "i", "j", "true", "false", "e"}


def _vectorize(expr: str) -> str:
    return re.sub(r"(?<![.])([*/^\\])", r".\1", expr)


def _identifiers(expr: str) -> list[str]:
    """Variable names in an equation: identifiers not followed by '(' and not constants, in order of first use."""
    out = []
    stripped = re.sub(r"'[^']*'", "", expr)
    for mt in re.finditer(r"(?<![\w.])([A-Za-z]\w*)(?!\w)(\s*\()?", stripped):
        nm = mt.group(1)
        if mt.group(2) or nm in _KNOWN_NAMES:
            continue
        if re.match(r"\d", stripped[max(mt.start() - 1, 0):mt.start()] or "x") and mt.start() > 0:
            continue
        if nm not in out:
            out.append(nm)
    return out


def _make_handle(interp, params: list[str], body: str) -> FunctionHandle:
    try:
        fh = interp.eval_expression(f"@({','.join(params)}) {body}")
    except MatlabError as e:
        raise MatlabError(f"fittype: the expression '{body}' is not a valid MATLAB expression: {e}",
                          "curvefit:fittype:invalidExpression") from None
    return fh


def _call_handle(interp, fh, args: list, n: int) -> np.ndarray:
    res = interp.call_handle(fh, [np.asarray(a, dtype=float) if not isinstance(a, np.ndarray) else a for a in args],
                             1)
    v = fnum(res[0]).astype(float) if res else np.zeros(0)
    if v.size == 1 and n != 1:
        v = np.full(n, float(v.flat[0]))
    if v.size != n:
        raise MatlabError("The model function must return one value per data point (use element-wise "
                          "operators such as .* and .^).", "curvefit:fittype:BadEvaluation")
    return v.reshape(-1)


def _ft_options(args: list, names: list[str], fname: str) -> dict:
    opts = parse_options(args, names, fname)
    return opts


def _custom_equation(interp, expr: str, opts: dict) -> FitType:
    indep = _text_list(opts["independent"], "independent variables") if "independent" in opts else None
    dep = _text_list(opts["dependent"], "dependent variable")[0] if "dependent" in opts else "y"
    probs = _text_list(opts["problem"], "problem parameters") if "problem" in opts else []
    names = _identifiers(expr)
    if indep is None:
        indep = ["x"]
        if "x" not in names and len(names) > 0 and "coefficients" not in opts:
            pass
    if "coefficients" in opts:
        coeffs = _text_list(opts["coefficients"], "coefficient names")
        extra = [n for n in names if n not in coeffs and n not in indep and n not in probs and n != dep]
        if extra:
            raise MatlabError(f"fittype: the expression uses '{extra[0]}', which is not a coefficient, a problem "
                              "parameter or an independent variable.", "curvefit:fittype:unknownVariable")
    else:
        coeffs = sorted(n for n in names if n not in indep and n not in probs and n != dep)
    if not coeffs:
        raise MatlabError("fittype: the expression has no coefficients to fit.", "curvefit:fittype:noCoeffs")
    for v in indep:
        if v not in names:
            raise MatlabError(f"fittype: the independent variable '{v}' does not appear in the expression "
                              f"'{expr}'.", "curvefit:fittype:noIndependentVariable")
    params = coeffs + probs + indep
    fh = _make_handle(interp, params, _vectorize(expr))
    p = len(coeffs)

    def fn(c, X, P):
        n = int(np.size(X[0]))
        return _call_handle(interp, fh, [np.array([[v]]) for v in c] + [np.array([[v]]) for v in P]
                            + [np.asarray(x, dtype=float).reshape(-1, 1) for x in X], n).reshape(np.shape(X[0]))
    ft = FitType(name="customnonlinear", label="General model", category="custom", coeffs=coeffs, expr=expr,
                 fn=fn, probs=probs, indep=indep, dep=dep)
    ft.nparams = p
    return ft


def _custom_handle(interp, fh: FunctionHandle, opts: dict) -> FitType:
    if not fh.is_anonymous:
        raise MatlabError("fittype: give the model as an anonymous function such as @(a,b,x) a*x+b.",
                          "curvefit:fittype:notAnonymous")
    params = list(fh.params)
    probs = _text_list(opts["problem"], "problem parameters") if "problem" in opts else []
    indep = _text_list(opts["independent"], "independent variables") if "independent" in opts else [params[-1]]
    dep = _text_list(opts["dependent"], "dependent variable")[0] if "dependent" in opts else \
        ("z" if len(indep) == 2 else "y")
    if "coefficients" in opts:
        coeffs = _text_list(opts["coefficients"], "coefficient names")
    else:
        coeffs = [p for p in params if p not in indep and p not in probs]
    for nm in coeffs + probs + indep:
        if nm not in params:
            raise MatlabError(f"fittype: '{nm}' is not an input of the anonymous function.",
                              "curvefit:fittype:unknownVariable")
    m = re.match(r"@\s*\([^)]*\)\s*(.*)$", fh.text or "", re.S)
    if m:
        body = m.group(1).strip()
    else:
        from ..display import unparse
        body = unparse(fh.body)

    def fn(c, X, P):
        vals = dict(zip(coeffs, [np.array([[v]]) for v in c]))
        vals.update(zip(probs, [np.array([[v]]) for v in P]))
        vals.update(zip(indep, [np.asarray(x, dtype=float).reshape(-1, 1) for x in X]))
        return _call_handle(interp, fh, [vals.get(p, np.zeros((0, 0))) for p in params],
                            int(np.size(X[0]))).reshape(np.shape(X[0]))
    return FitType(name="customnonlinear", label="General model", category="custom", coeffs=coeffs, expr=body,
                   fn=fn, probs=probs, indep=indep, dep=dep)


def _custom_linear(interp, terms: list[str], opts: dict) -> FitType:
    indep = _text_list(opts["independent"], "independent variables") if "independent" in opts else ["x"]
    dep = _text_list(opts["dependent"], "dependent variable")[0] if "dependent" in opts else "y"
    if "coefficients" in opts:
        coeffs = _text_list(opts["coefficients"], "coefficient names")
        if len(coeffs) != len(terms):
            raise MatlabError("fittype: give one coefficient name per term of the linear model.",
                              "curvefit:fittype:coeffNumMismatch")
    else:
        if len(terms) > 26:
            raise MatlabError("fittype: a linear model with more than 26 terms needs 'coefficients' names.")
        coeffs = [chr(ord("a") + k) for k in range(len(terms))]
    handles = [_make_handle(interp, indep, _vectorize(t)) for t in terms]

    def basis(X, P):
        n = int(np.size(X[0]))
        cols = [_call_handle(interp, fh, [np.asarray(x, dtype=float).reshape(-1, 1) for x in X], n)
                for fh in handles]
        return np.column_stack(cols)
    parts = []
    for c, t in zip(coeffs, terms):
        ts = t.strip()
        if ts == "1":
            parts.append(c)
        elif re.fullmatch(r"[\w.^()]+", ts) and "+" not in ts and "-" not in ts:
            parts.append(f"{c}*{ts}")
        else:
            parts.append(f"{c}*({ts})")
    return FitType(name="customlinear", label="Linear model", category="custom", coeffs=coeffs,
                   expr=" + ".join(parts), basis=basis, indep=indep, dep=dep)


_FT_OPTS = ["independent", "dependent", "coefficients", "problem", "options"]


def _make_fittype(interp, spec: Any, opt_args: list, fname: str = "fittype") -> FitType:
    opts = _ft_options(opt_args, _FT_OPTS, fname)
    if isinstance(spec, FitType):
        if opts:
            raise MatlabError(f"{fname}: options cannot be given with an existing fittype object.")
        return spec
    if isinstance(spec, FunctionHandle):
        ft = _custom_handle(interp, spec, opts)
    elif isinstance(spec, Cell):
        ft = _custom_linear(interp, _text_list(spec, "linear model terms"), opts)
    elif is_text(spec):
        s = text(spec).strip()
        ft = _library(s)
        if ft is None:
            if re.fullmatch(r"[A-Za-z]\w*", s):
                raise MatlabError(f"{fname}: '{s}' is not a library model and not an equation. Library models "
                                  "include poly1-poly9, exp1, exp2, power1, power2, gauss1-gauss8, sin1-sin8, "
                                  "fourier1-fourier8, rat01-rat55, weibull, linearinterp, nearestinterp, "
                                  "cubicinterp, pchipinterp, smoothingspline and poly11-poly55.",
                                  "curvefit:fittype:unknownLibraryModel")
            ft = _custom_equation(interp, s, opts)
        elif any(k in opts for k in ("independent", "dependent", "coefficients", "problem")):
            raise MatlabError(f"{fname}: the options 'independent', 'dependent', 'coefficients' and 'problem' "
                              "apply to custom models only.", "curvefit:fittype:optionsForLibrary")
    else:
        raise MatlabError(f"{fname}: the model must be a library model name, an equation, an anonymous function "
                          "or a cell array of linear model terms.", "curvefit:fittype:invalidModel")
    if "options" in opts:
        if not isinstance(opts["options"], FitOptions):
            raise MatlabError(f"{fname}: 'options' must be a fitoptions object.")
        ft.options = opts["options"]
    return ft


@builtin("fittype")
def _fittype(interp, args, nargout):
    if not args:
        raise MatlabError("fittype needs a model: a library model name, an equation, an anonymous function or "
                          "a cell array of linear terms.", "MATLAB:narginchk:notEnoughInputs")
    return _make_fittype(interp, args[0], list(args[1:]))


# ==========================================================================================================
# fitoptions
# ==========================================================================================================

_COMMON = ["Normalize", "Exclude", "Weights", "Method"]
_METHOD_OPTS = {
    "None": [],
    "LinearLeastSquares": ["Robust", "Lower", "Upper"],
    "NonlinearLeastSquares": ["StartPoint", "Lower", "Upper", "Algorithm", "DiffMinChange", "DiffMaxChange",
                              "Display", "MaxFunEvals", "MaxIter", "TolFun", "TolX", "Robust"],
    "LinearInterpolant": [], "NearestInterpolant": [], "CubicSplineInterpolant": [], "PchipInterpolant": [],
    "BiharmonicInterpolant": [], "ThinPlateInterpolant": [],
    "SmoothingSpline": ["SmoothingParam"],
    "LowessFit": ["Span", "Robust"],
}
_ALL_OPTS = list(dict.fromkeys(_COMMON + sum(_METHOD_OPTS.values(), [])))
_OPT_CLASS = {"None": "curvefit.basefitoptions", "LinearLeastSquares": "curvefit.llsqoptions",
              "NonlinearLeastSquares": "curvefit.nlsqoptions", "SmoothingSpline": "curvefit.smoothingsplineoptions",
              "LowessFit": "curvefit.lowessoptions"}
_DEFAULTS = {"Normalize": "off", "Exclude": None, "Weights": None, "StartPoint": None, "Lower": None,
             "Upper": None, "Algorithm": "Trust-Region", "DiffMinChange": 1e-8, "DiffMaxChange": 0.1,
             "Display": "Notify", "MaxFunEvals": 600.0, "MaxIter": 400.0, "TolFun": 1e-6, "TolX": 1e-6,
             "Robust": "Off", "SmoothingParam": None, "Span": 0.25}
_DISPLAY_ORDER = ["StartPoint", "Lower", "Upper", "Algorithm", "DiffMinChange", "DiffMaxChange", "Display",
                  "MaxFunEvals", "MaxIter", "TolFun", "TolX", "Robust", "SmoothingParam", "Span"]


def _canon_method(v: Any) -> str:
    s = text(v, "fit method") if is_text(v) else ""
    for k in _METHOD_OPTS:
        if k.lower() == s.lower():
            return k
    raise MatlabError(f"fitoptions: unknown fitting method '{s}'. Use one of: " + ", ".join(_METHOD_OPTS) + ".",
                      "curvefit:fitoptions:invalidMethod")


def _choice(v: Any, choices: tuple, what: str) -> str:
    s = text(v, what) if is_text(v) else None
    if s is not None:
        for c in choices:
            if c.lower() == s.lower():
                return c
    raise MatlabError(f"The {what} must be one of: " + ", ".join(f"'{c}'" for c in choices) + ".",
                      "curvefit:fitoptions:invalidValue")


def _norm_value(name: str, v: Any) -> Any:
    if name == "Normalize":
        return _choice(v, ("on", "off"), "Normalize option")
    if name == "Robust":
        r = _choice(v, ("Off", "On", "LAR", "Bisquare"), "Robust option")
        return r
    if name == "Algorithm":
        return _choice(v, ("Trust-Region", "Levenberg-Marquardt"), "Algorithm")
    if name == "Display":
        return _choice(v, ("Notify", "Off", "Iter", "Final"), "Display option")
    if name == "Method":
        return _canon_method(v)
    if isinstance(v, np.ndarray) and v.size == 0:
        return None
    if name == "Exclude":
        a = np.asarray(v)
        if a.dtype == bool:
            return a.copy()
        return fnum(v).astype(float)
    a = fnum(v).astype(float)
    if name in ("DiffMinChange", "DiffMaxChange", "MaxFunEvals", "MaxIter", "TolFun", "TolX", "SmoothingParam",
                "Span"):
        if a.size != 1:
            raise MatlabError(f"The {name} option must be a scalar.", "curvefit:fitoptions:invalidValue")
        val = float(a.flat[0])
        if name == "SmoothingParam" and not 0 <= val <= 1:
            raise MatlabError("SmoothingParam must be between 0 and 1.", "curvefit:fitoptions:invalidValue")
        return val
    return a.reshape(-1)


class FitOptions(MObject):
    """fitoptions object for one fitting method."""

    def __init__(self, method: str = "None", values: dict | None = None):
        self.method = method
        self.values = {k: _DEFAULTS.get(k) for k in self.names()}
        if values:
            for k, v in values.items():
                if k in self.values:
                    self.values[k] = v
        self.values["Method"] = method

    def names(self) -> list[str]:
        return _COMMON + _METHOD_OPTS[self.method]

    def copy(self) -> "FitOptions":
        new = FitOptions(self.method)
        new.values = dict(self.values)
        return new

    def class_name(self) -> str:
        return _OPT_CLASS.get(self.method, "curvefit.interpoptions")

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "curvefit.basefitoptions")

    def property_names(self) -> list[str]:
        return [n for n in _DISPLAY_ORDER if n in self.values] + _COMMON

    def method_names(self) -> list[str]:
        return ["fitoptions"]

    def _key(self, name: str) -> str | None:
        return next((k for k in self.values if k.lower() == name.lower()), None)

    def get_field(self, interp, name: str) -> Any:
        key = self._key(name)
        if key is None:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.",
                              "MATLAB:noSuchMethodOrField")
        return _opt_matlab(key, self.values[key])

    def set_field(self, interp, name: str, value: Any) -> "FitOptions":
        key = self._key(name)
        if key is None:
            raise MatlabError(f"'{name}' is not an option of fitting method {self.method}.",
                              "curvefit:fitoptions:invalidProperty")
        if key == "Method":
            return FitOptions(_canon_method(value), self.values)
        new = self.copy()
        new.values[key] = _norm_value(key, value)
        return new

    def display_lines(self) -> list[str]:
        from ..display import summary
        names = self.property_names()
        w = max(len(n) for n in names)
        lines = [f"  {self.class_name().split('.')[-1]} with properties:", ""]
        for n in names:
            lines.append(f"    {n:>{w}}: {summary(_opt_matlab(n, self.values[n]), False)}")
        return lines


def _opt_matlab(name: str, v: Any) -> Any:
    if v is None:
        return np.zeros((0, 0))
    if isinstance(v, str):
        return char(v)
    if isinstance(v, float):
        return np.array([[v]])
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return v.reshape(1, -1) if v.ndim < 2 else v
    return _row(v)


def _apply_opts(o: FitOptions, pairs: dict, fname: str) -> FitOptions:
    o = o.copy()
    if "Method" in pairs:
        o = FitOptions(_canon_method(pairs["Method"]), o.values)
    for k, v in pairs.items():
        if k == "Method":
            continue
        if k not in o.values:
            raise MatlabError(f"{fname}: '{k}' is not an option of fitting method {o.method}.",
                              "curvefit:fitoptions:invalidProperty")
        o.values[k] = _norm_value(k, v)
    return o


def _default_options(ft: FitType) -> FitOptions:
    if ft.options is not None:
        return ft.options.copy()
    o = FitOptions(ft.method())
    if ft.lower is not None and "Lower" in o.values:
        o.values["Lower"] = np.asarray(ft.lower, dtype=float)
    if ft.upper is not None and "Upper" in o.values:
        o.values["Upper"] = np.asarray(ft.upper, dtype=float)
    return o


@builtin("fitoptions")
def _fitoptions(interp, args, nargout):
    args = list(args)
    if args and isinstance(args[0], FitOptions):
        return _apply_opts(args[0], parse_options(args[1:], _ALL_OPTS, "fitoptions"), "fitoptions")
    if args and isinstance(args[0], FitType):
        base = _default_options(args[0])
        return _apply_opts(base, parse_options(args[1:], _ALL_OPTS, "fitoptions"), "fitoptions")
    if args and is_text(args[0]) and match_option(text(args[0]), _ALL_OPTS) is None:
        ft = _library(text(args[0]))
        if ft is None:
            raise MatlabError(f"fitoptions: '{text(args[0])}' is not a library model or an option name.",
                              "curvefit:fitoptions:unknownLibraryModel")
        return _apply_opts(_default_options(ft), parse_options(args[1:], _ALL_OPTS, "fitoptions"), "fitoptions")
    pairs = parse_options(args, _ALL_OPTS, "fitoptions")
    return _apply_opts(FitOptions("None"), pairs, "fitoptions")


# ==========================================================================================================
# fitting
# ==========================================================================================================

def _robust_weights(r: np.ndarray, h: np.ndarray, kind: str, p: int) -> np.ndarray:
    radj = r / np.sqrt(np.clip(1 - h, 1e-4, None))
    if kind == "LAR":
        return 1.0 / np.maximum(np.abs(radj), 1e-6 * max(np.max(np.abs(radj)), 1e-300))
    dev = np.abs(radj - np.median(radj))
    s = np.median(dev) / 0.6745
    if s <= 0:
        s = max(np.mean(dev) / 0.6745, 1e-300)
    u = radj / (4.685 * s)
    return np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)


def _leverage(J: np.ndarray) -> np.ndarray:
    if J.size == 0:
        return np.zeros(J.shape[0])
    Q, _ = np.linalg.qr(J)
    return np.sum(Q ** 2, axis=1)


def _cov_unscaled(Jw: np.ndarray) -> np.ndarray:
    """inv(Jw'*Jw) from a QR factorization of the column-scaled Jacobian (stable for ill-conditioned fits)."""
    p = Jw.shape[1]
    d = np.linalg.norm(Jw, axis=0)
    d = np.where(d > 0, d, 1.0)
    try:
        R = np.linalg.qr(Jw / d, mode="r")
        if R.shape[0] < p or np.any(np.abs(np.diag(R)) < 1e-14 * np.max(np.abs(np.diag(R)))):
            raise np.linalg.LinAlgError
        from scipy.linalg import solve_triangular
        Ri = solve_triangular(R[:p, :p], np.eye(p))
        return (Ri @ Ri.T) / np.outer(d, d)
    except np.linalg.LinAlgError:
        return np.linalg.pinv((Jw / d).T @ (Jw / d), rcond=1e-15) / np.outer(d, d)


def _num_jac(fun: Callable, c: np.ndarray, f0: np.ndarray | None = None) -> np.ndarray:
    """Central-difference Jacobian of a vector function."""
    c = np.asarray(c, dtype=float)
    cols = []
    for j in range(c.size):
        h = _EPS ** (1 / 3) * max(abs(c[j]), 1.0)
        cp, cm = c.copy(), c.copy()
        cp[j] += h
        cm[j] -= h
        cols.append((fun(cp) - fun(cm)) / (2 * h))
    return np.column_stack(cols) if cols else np.zeros((0 if f0 is None else f0.size, 0))


class _Data:
    """Fit data after exclusion and normalization."""

    def __init__(self, X: list[np.ndarray], y: np.ndarray, w: np.ndarray, mu: np.ndarray | None,
                 sd: np.ndarray | None):
        self.X, self.y, self.w, self.mu, self.sd = X, y, w, mu, sd


def _lsq_linear_fit(A, y, w, lb, ub):
    sw = np.sqrt(w)
    Aw, yw = A * sw[:, None], y * sw
    if lb is not None and np.any(np.isfinite(lb)) or ub is not None and np.any(np.isfinite(ub)):
        from scipy.optimize import lsq_linear
        lo = -np.inf * np.ones(A.shape[1]) if lb is None else lb
        hi = np.inf * np.ones(A.shape[1]) if ub is None else ub
        return lsq_linear(Aw, yw, bounds=(lo, hi), lsmr_tol="auto", method="bvls").x
    scale = np.linalg.norm(Aw, axis=0)
    scale = np.where(scale > 0, scale, 1.0)  # column scaling keeps badly scaled polynomials accurate
    return np.linalg.lstsq(Aw / scale, yw, rcond=None)[0] / scale


def _bounds(o: FitOptions, p: int) -> tuple[np.ndarray, np.ndarray]:
    lb = o.values.get("Lower")
    ub = o.values.get("Upper")
    lb = -np.inf * np.ones(p) if lb is None else np.asarray(lb, dtype=float)
    ub = np.inf * np.ones(p) if ub is None else np.asarray(ub, dtype=float)
    if lb.size != p or ub.size != p:
        raise MatlabError(f"fit: the Lower and Upper bounds must have one element per coefficient ({p}).",
                          "curvefit:fit:invalidBounds")
    if np.any(lb > ub):
        raise MatlabError("fit: the Lower bounds must not exceed the Upper bounds.", "curvefit:fit:invalidBounds")
    return lb, ub


def _random_start(interp, p: int) -> np.ndarray:
    try:
        return fnum(BUILTINS["rand"](interp, [np.array([[1.0]]), np.array([[float(p)]])], 1)).reshape(-1)
    except MatlabError:
        return np.random.default_rng().random(p)


def _nls(interp, ft: FitType, d: _Data, o: FitOptions, P: list[float], w: np.ndarray):
    """Nonlinear least squares from the best of several starting points."""
    from scipy.optimize import least_squares
    p = len(ft.coeffs)
    lb, ub = _bounds(o, p)
    alg = o.values.get("Algorithm", "Trust-Region")
    if alg == "Levenberg-Marquardt" and (np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))):
        dlb = -np.inf * np.ones(p) if ft.lower is None else np.asarray(ft.lower, dtype=float)
        dub = np.inf * np.ones(p) if ft.upper is None else np.asarray(ft.upper, dtype=float)
        if not (np.array_equal(lb, dlb) and np.array_equal(ub, dub)):  # bounds the user asked for
            raise MatlabError("fit: the Levenberg-Marquardt algorithm does not handle bound constraints; use the "
                              "Trust-Region algorithm.", "curvefit:fit:LMWithBounds")
        lb, ub = -np.inf * np.ones(p), np.inf * np.ones(p)  # library default bounds are dropped for LM
    sw = np.sqrt(w)

    def resid(c):
        with np.errstate(all="ignore"):
            r = (ft.evaluate(c, d.X, P) - d.y) * sw
        return np.where(np.isfinite(r), r, 1e100)
    sp = o.values.get("StartPoint")
    if sp is not None:
        sp = np.asarray(sp, dtype=float)
        if sp.size != p:
            raise MatlabError(f"fit: the StartPoint must have one element per coefficient ({p}).",
                              "curvefit:fit:invalidStartPoint")
        starts = [sp]
    elif ft.start is not None:
        try:
            with np.errstate(all="ignore"):
                starts = [np.asarray(s, dtype=float) for s in ft.start(d.X, d.y, w)]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            starts = []
        starts = [s for s in starts if s.size == p and np.all(np.isfinite(s))] or [np.ones(p)]
    else:
        interp.warn("Start point not provided, choosing random start point.", "curvefit:fit:usingRandomStartPoint")
        starts = [_random_start(interp, p)]
    tol_f = o.values.get("TolFun", 1e-6)
    tol_x = o.values.get("TolX", 1e-6)
    ftol = 1e-12 if tol_f == 1e-6 else max(tol_f, 1e-15)
    xtol = 1e-12 if tol_x == 1e-6 else max(tol_x, 1e-15)
    max_fev = int(max(o.values.get("MaxFunEvals", 600.0), 1))
    max_iter = int(max(o.values.get("MaxIter", 400.0), 1))
    method = "lm" if alg == "Levenberg-Marquardt" else "trf"
    if method == "lm" and d.y.size < p:
        method = "trf"
    best = None
    for s in starts:
        s = np.clip(s, lb, ub)
        try:
            kw = dict(ftol=ftol, xtol=xtol, gtol=1e-12 if tol_f == 1e-6 else ftol,
                      max_nfev=max_fev if method == "trf" else max_fev * (p + 1))
            if method == "trf":
                res = least_squares(resid, s, bounds=(lb, ub), method="trf", x_scale="jac", **kw)
            else:
                res = least_squares(resid, s, method="lm", x_scale="jac", **kw)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise MatlabError(f"fit: the nonlinear fit failed: {e}", "curvefit:fit:nonlinearFitFailed") from None
        if best is None or res.cost < best.cost:
            best = res
    res = best
    if np.any(np.abs(resid(res.x)) >= 1e99):
        raise MatlabError("fit: the model computed Inf or NaN values at the solution; check the model, the data and "
                          "the starting point.", "curvefit:fit:infComputed")
    status = int(res.status)
    exitflag = {1: 1, 2: 3, 3: 2, 4: 3}.get(status, 0)
    iters = int(res.njev) if res.njev is not None else int(res.nfev)
    if exitflag == 0 or iters > max_iter:
        exitflag = 0
        if o.values.get("Display", "Notify") != "Off":
            interp.warn("Fit computation did not converge: the maximum number of function evaluations or "
                        "iterations was reached. Increasing MaxFunEvals or MaxIter (in fitoptions) may allow for "
                        "a better fit, or the current equation may not be a good model for the data.",
                        "curvefit:fit:noConvergence")
    info = {"exitflag": exitflag, "iterations": iters, "funcCount": int(res.nfev) + iters * p,
            "firstorderopt": float(res.optimality),
            "algorithm": "levenberg-marquardt" if method == "lm" else "trust-region-reflective",
            "message": res.message}
    return res.x, info


def _fit_parametric(interp, ft: FitType, d: _Data, o: FitOptions, P: list[float]):
    p = len(ft.coeffs)
    robust = o.values.get("Robust", "Off")
    robust = "Bisquare" if robust == "On" else robust
    w_user = d.w
    w = w_user.copy()
    if ft.linear:
        A = ft.basis(d.X, P)
        lb = o.values.get("Lower")
        ub = o.values.get("Upper")
        lb = None if lb is None else np.asarray(lb, dtype=float)
        ub = None if ub is None else np.asarray(ub, dtype=float)
        for b in (lb, ub):
            if b is not None and b.size != p:
                raise MatlabError(f"fit: the Lower and Upper bounds must have one element per coefficient ({p}).",
                                  "curvefit:fit:invalidBounds")

        def solve(wt):
            return _lsq_linear_fit(A, d.y, wt, lb, ub)
        c = solve(w)
        info = {"exitflag": 1, "algorithm": "QR factorization and solve"}
        if robust != "Off":
            for _ in range(50):
                r = d.y - A @ c
                h = _leverage(A * np.sqrt(w_user)[:, None])
                wr = _robust_weights(r * np.sqrt(w_user), h, robust, p)
                w = w_user * wr
                c_new = solve(w)
                done = np.max(np.abs(c_new - c)) <= 1e-6 * max(np.max(np.abs(c)), 1e-12)
                c = c_new
                if done:
                    break
        J = A
    else:
        c, info = _nls(interp, ft, d, o, P, w)
        if robust != "Off":
            o2 = o.copy()
            o2.values["StartPoint"] = c
            for _ in range(30):
                r = d.y - ft.evaluate(c, d.X, P)
                Jw = _num_jac(lambda cc: ft.evaluate(cc, d.X, P), c) * np.sqrt(w_user)[:, None]
                wr = _robust_weights(r * np.sqrt(w_user), _leverage(Jw), robust, p)
                w = w_user * wr
                o2.values["StartPoint"] = c
                c_new, info = _nls(interp, ft, d, o2, P, w)
                done = np.max(np.abs(c_new - c)) <= 1e-6 * max(np.max(np.abs(c)), 1e-12)
                c = c_new
                if done:
                    break
        J = _num_jac(lambda cc: ft.evaluate(cc, d.X, P), c)
    return np.asarray(c, dtype=float), J, w, info


def _check_xy(x: Any, y: Any, fname: str = "fit") -> tuple[np.ndarray, np.ndarray]:
    X = fnum(x)
    Y = fnum(y)
    if np.iscomplexobj(X) or np.iscomplexobj(Y):
        raise MatlabError(f"{fname}: the data must be real.", "curvefit:fit:complexData")
    X = X.astype(float)
    Y = Y.astype(float)
    if X.ndim != 2 or X.shape[1] not in (1, 2):
        raise MatlabError("X must be a matrix with one or two columns.", "curvefit:fit:xDataMustBeColumnVector")
    if Y.ndim != 2 or Y.shape[1] != 1:
        raise MatlabError("Y must be a column vector.", "curvefit:fit:yDataMustBeColumnVector")
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("X and Y must have the same number of rows.", "curvefit:fit:xyMismatch")
    return X, Y.reshape(-1)


def _exclusion(o: FitOptions, n: int) -> np.ndarray:
    ex = o.values.get("Exclude")
    mask = np.zeros(n, dtype=bool)
    if ex is None:
        return mask
    a = np.asarray(ex)
    if a.dtype == bool:
        if a.size != n:
            raise MatlabError("fit: a logical Exclude vector must have one element per data point.",
                              "curvefit:fit:invalidExclude")
        return a.reshape(-1).copy()
    idx = a.reshape(-1)
    if np.any(idx != np.round(idx)) or np.any(idx < 1) or np.any(idx > n):
        raise MatlabError(f"fit: Exclude indices must be integers from 1 to {n}.", "curvefit:fit:invalidExclude")
    mask[idx.astype(np.int64) - 1] = True
    return mask


def _fit_pp(ft: FitType, x: np.ndarray, y: np.ndarray, w: np.ndarray, o: FitOptions, interp):
    """Interpolants and smoothing splines: a scipy piecewise polynomial in x."""
    from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    ux, inv = np.unique(x, return_inverse=True)
    extra = {}
    if ux.size < x.size:
        ws = np.bincount(inv, weights=w)
        y = np.bincount(inv, weights=w * y) / np.where(ws > 0, ws, 1)
        w = ws
        x = ux
    n = x.size
    if n < 2:
        raise MatlabError("fit: interpolants and smoothing splines need at least two distinct data points.",
                          "curvefit:fit:tooFewPoints")
    if ft.name == "linearinterp" or (ft.name in ("cubicinterp", "pchipinterp") and n < 3):
        pp = PPoly(np.vstack([np.diff(y) / np.diff(x), y[:-1]]), x)
    elif ft.name == "nearestinterp":
        mids = (x[:-1] + x[1:]) / 2
        brk = np.concatenate([[x[0]], mids, [x[-1]]])
        pp = PPoly(y.reshape(1, -1), brk)
    elif ft.name == "cubicinterp":
        pp = CubicSpline(x, y, bc_type="not-a-knot" if n > 3 else "natural")
    elif ft.name == "pchipinterp":
        pp = PchipInterpolator(x, y)
    else:  # smoothing spline
        p = o.values.get("SmoothingParam")
        if p is None:
            h = float(np.mean(np.diff(x)))
            p = 1.0 / (1.0 + h ** 3 / 6.0)
        extra["p"] = p
        pp = _smoothing_spline(x, y, w, p)
    return pp, extra


def _smoothing_spline(x, y, w, p):
    from scipy.interpolate import CubicSpline, PPoly, make_smoothing_spline
    if p >= 1:
        return CubicSpline(x, y, bc_type="natural") if x.size > 2 else \
            PPoly(np.vstack([np.diff(y) / np.diff(x), y[:-1]]), x)
    if p <= 0 or x.size < 5:
        c = np.polyfit(x, y, 1, w=np.sqrt(w)) if x.size >= 2 else [0.0, y[0]]
        return PPoly(np.array([[c[0]] * (x.size - 1), list(c[0] * x[:-1] + c[1])]), x)
    spl = make_smoothing_spline(x, y, w=w, lam=(1 - p) / p)
    return PPoly.from_spline(spl)


def _parse_fit_args(interp, args: list):
    if len(args) < 3:
        raise MatlabError("fit needs data and a model: fit(x, y, fitType).", "MATLAB:narginchk:notEnoughInputs")
    X, y = _check_xy(args[0], args[1])
    spec = args[2]
    rest = list(args[3:])
    user_opts = None
    if rest and isinstance(rest[0], FitOptions):
        user_opts = rest.pop(0)
    pairs = parse_options(rest, _ALL_OPTS + ["problem"], "fit")
    ft = _make_fittype(interp, spec, [], "fit")
    if ft.category == "interpolant" and X.shape[1] == 2 and len(ft.indep) == 1 and ft.name != "pchipinterp":
        ft = _surface_interp(ft.name)  # linear, nearest and cubic interpolants also fit surfaces
    if len(ft.indep) != X.shape[1]:
        if X.shape[1] == 2:
            raise MatlabError(f"fit: the model {ft.name} is a curve model (one independent variable), but X has "
                              "two columns.", "curvefit:fit:surfaceDataForCurve")
        raise MatlabError(f"fit: the model {ft.name} is a surface model; give X as a matrix with two columns, "
                          "[x, y].", "curvefit:fit:curveDataForSurface")
    o = _default_options(ft)
    if user_opts is not None:
        for k, v in user_opts.values.items():
            if k in o.values and k != "Method" and v is not None:
                o.values[k] = v
    probs = pairs.pop("problem", None)
    applicable = set(o.values)
    for k in pairs:
        if k not in applicable:
            raise MatlabError(f"fit: the option '{k}' does not apply to the {ft.name} model (fitting method "
                              f"{o.method}).", "curvefit:fit:invalidOption")
    o = _apply_opts(o, pairs, "fit")
    P: list[float] = []
    if ft.probs:
        if probs is None:
            raise MatlabError("fit: the model has problem parameters (" + ", ".join(ft.probs) + "); give their "
                              "values with the 'problem' option.", "curvefit:fit:missingProblemParameters")
        vals = list(probs.data.flatten(order="F")) if isinstance(probs, Cell) else list(fnum(probs).reshape(-1))
        if len(vals) != len(ft.probs):
            raise MatlabError(f"fit: give {len(ft.probs)} problem parameter value(s).",
                              "curvefit:fit:missingProblemParameters")
        P = [float(fnum(v).reshape(-1)[0]) if isinstance(v, np.ndarray) else float(v) for v in vals]
    elif probs is not None:
        raise MatlabError("fit: the model has no problem parameters.", "curvefit:fit:noProblemParameters")
    return ft, X, y, o, P


@builtin("fit")
def _fit(interp, args, nargout):
    ft, X, y, o, P = _parse_fit_args(interp, list(args))
    n = y.size
    excl = _exclusion(o, n)
    w = o.values.get("Weights")
    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.size != n:
            raise MatlabError("fit: the Weights vector must have one element per data point.",
                              "curvefit:fit:invalidWeights")
        if np.any(w < 0):
            raise MatlabError("fit: the weights must not be negative.", "curvefit:fit:invalidWeights")
    keep = ~excl
    bad = keep & ~(np.all(np.isfinite(X), axis=1) & np.isfinite(y))
    if bad.any():
        raise MatlabError("fit: the data contain NaN or Inf values. Remove them first (prepareCurveData or "
                          "prepareSurfaceData do that) or exclude them with the 'Exclude' option.",
                          "curvefit:fit:nanInfData")
    Xk = X[keep]
    yk = y[keep]
    wk = w[keep]
    if ft.nonneg_x and np.any(Xk[:, 0] <= 0):
        raise MatlabError(f"fit: the {ft.name} model needs positive x values (it uses a power of x).",
                          "curvefit:fit:nonPositiveXData")
    mu = sd = None
    if o.values.get("Normalize") == "on":
        mu = Xk.mean(axis=0)
        sd = Xk.std(axis=0, ddof=1) if Xk.shape[0] > 1 else np.ones(Xk.shape[1])
        sd = np.where(sd > 0, sd, 1.0)
        Xn = (Xk - mu) / sd
    else:
        Xn = Xk
    cols = [Xn[:, j] for j in range(Xn.shape[1])]
    d = _Data(cols, yk, wk, mu, sd)
    nobs = yk.size
    if ft.parametric:
        p = len(ft.coeffs)
        if nobs < p:
            raise MatlabError(f"fit: the model has {p} coefficients but only {nobs} data points; it needs at least "
                              "as many data points as coefficients.", "curvefit:fit:tooFewDataPoints")
        c, J, wfin, info = _fit_parametric(interp, ft, d, o, P)
        yhat = ft.evaluate(c, cols, P)
        cov = _cov_unscaled(J * np.sqrt(wfin)[:, None])
        fitobj = CFit(ft, c, P, mu, sd, cov=cov, dfe=nobs - p, sse=float(np.sum(wfin * (yk - yhat) ** 2)),
                      xrange=_ranges(Xk))
        numparam = p
    else:
        from . import _surffit
        if ft.category == "lowess":
            robust = o.values.get("Robust") or "Off"
            model = _surffit.Lowess(ft.name, cols[0], cols[1], yk, wk, float(o.values.get("Span") or 0.25),
                                    "Bisquare" if robust == "On" else robust)
            fitobj = CFit(ft, np.zeros(0), [], mu, sd, pp=model, xrange=_ranges(Xk))
            yhat = fitobj.eval_norm(cols)
            numparam = model.dof()
        elif len(ft.indep) == 2:
            model = _surffit.Interpolant(ft.name, cols[0], cols[1], yk)
            fitobj = CFit(ft, np.zeros(0), [], mu, sd, pp=model, xrange=_ranges(Xk))
            yhat = fitobj.eval_norm(cols)
            numparam = nobs
        else:
            pp, extra = _fit_pp(ft, cols[0], yk, wk, o, interp)
            fitobj = CFit(ft, np.zeros(0), [], mu, sd, pp=pp, xrange=_ranges(Xk), extra=extra)
            yhat = fitobj.eval_norm([cols[0]])
            numparam = nobs if ft.category == "interpolant" else _spline_dof(fitobj, cols[0], yk, wk)
        info = {"exitflag": 1, "algorithm": ""}
        J = np.zeros((nobs, 0))
        wfin = wk
    r = yk - yhat
    sse = float(np.sum(wk * r ** 2)) if ft.category != "interpolant" else 0.0
    ybar = float(np.sum(wk * yk) / np.sum(wk)) if np.sum(wk) > 0 else float(np.mean(yk))
    sst = float(np.sum(wk * (yk - ybar) ** 2))
    dfe = nobs - numparam
    rsq = 1 - sse / sst if sst > 0 else (1.0 if sse == 0 else -np.inf)
    adj = 1 - (1 - rsq) * (nobs - 1) / dfe if dfe > 0 else np.nan
    rmse = np.sqrt(sse / dfe) if dfe > 0 else np.nan
    if ft.parametric:
        fitobj.sse, fitobj.dfe = sse if np.all(wfin == wk) else fitobj.sse, dfe
    gof = Struct.scalar({"sse": np.array([[sse]]), "rsquare": np.array([[rsq]]), "dfe": np.array([[float(dfe)]]),
                         "adjrsquare": np.array([[adj]]), "rmse": np.array([[rmse]])})
    if nargout < 3:
        return (fitobj, gof) if nargout == 2 else fitobj
    out = {"numobs": float(nobs), "numparam": float(numparam), "residuals": _col(r),
           "Jacobian": J * np.sqrt(wk)[:, None], "exitflag": float(info["exitflag"])}
    if "iterations" in info:
        out.update({"firstorderopt": info["firstorderopt"], "iterations": float(info["iterations"]),
                    "funcCount": float(info["funcCount"])})
    if info.get("algorithm"):
        out["algorithm"] = char(info["algorithm"])
    if "message" in info:
        out["message"] = char(str(info["message"]))
    if ft.category == "spline":
        out["p"] = float(fitobj.extra["p"])
    return fitobj, gof, Struct.scalar({k: (np.array([[v]]) if isinstance(v, float) else v) for k, v in out.items()})


def _ranges(X: np.ndarray) -> list[tuple[float, float]]:
    return [(float(X[:, j].min()), float(X[:, j].max())) if X.shape[0] else (0.0, 1.0) for j in range(X.shape[1])]


def _spline_dof(f: "CFit", x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Effective number of parameters of a smoothing spline: the trace of its smoother matrix."""
    n = x.size
    xs = np.unique(x)
    if n > 400 or xs.size != n:
        return float(min(n, 4))
    p = f.extra["p"]
    order = np.argsort(x)
    tr = 0.0
    for i in range(n):
        e = np.zeros(n)
        e[order[i]] = 1.0
        pp = _smoothing_spline(x[order], e[order], w[order], p)
        tr += float(pp(x[order[i]]))
    return tr


# ==========================================================================================================
# cfit and sfit
# ==========================================================================================================

class CFit(MObject):
    """A fitted model: cfit (one independent variable) or sfit (two)."""

    def __init__(self, ft: FitType, coef: np.ndarray, P: list[float], mu=None, sd=None, *, cov=None,
                 dfe: int = 0, sse: float = np.nan, pp=None, xrange=None, extra: dict | None = None):
        self.ft = ft
        self.coef = np.asarray(coef, dtype=float)
        self.P = list(P)
        self.mu = mu
        self.sd = sd
        self.cov = cov            # inv(J'WJ), unscaled
        self.dfe = dfe
        self.sse = sse
        self.pp = pp
        self.xrange = xrange
        self.extra = extra or {}

    @property
    def surface(self) -> bool:
        return len(self.ft.indep) == 2

    def class_name(self) -> str:
        return "sfit" if self.surface else "cfit"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "fittype")

    def property_names(self) -> list[str]:
        return list(self.ft.coeffs) + list(self.ft.probs)

    def method_names(self) -> list[str]:
        return sorted(set(_FT_METHODS) | set(_CF_METHODS), key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _CF_METHODS or name in _FT_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        table = _CF_METHODS if name in _CF_METHODS else _FT_METHODS
        return _call(self, table, interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        if name in self.ft.coeffs:
            if self.pp is not None:
                return _p_value(self)
            return np.array([[self.coef[self.ft.coeffs.index(name)]]])
        if name in self.ft.probs:
            return np.array([[self.P[self.ft.probs.index(name)]]])
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "CFit":
        new = CFit(self.ft, self.coef.copy(), list(self.P), self.mu, self.sd, cov=self.cov, dfe=self.dfe,
                   sse=self.sse, pp=self.pp, xrange=self.xrange, extra=dict(self.extra))
        v = fnum(value).astype(float)
        if v.size != 1:
            raise MatlabError(f"The value of '{name}' must be a scalar.")
        if name in self.ft.coeffs and self.pp is None:
            new.coef[self.ft.coeffs.index(name)] = float(v.flat[0])
            new.cov = None
            return new
        if name in self.ft.probs:
            new.P[self.ft.probs.index(name)] = float(v.flat[0])
            return new
        raise MatlabError(f"Unable to set '{name}' of class '{self.class_name()}'.", "MATLAB:class:SetProhibited")

    # -- evaluation ------------------------------------------------------------------------------------------
    def normalize(self, X: list[np.ndarray]) -> list[np.ndarray]:
        if self.mu is None:
            return X
        return [(x - self.mu[j]) / self.sd[j] for j, x in enumerate(X)]

    def eval_norm(self, Xn: list[np.ndarray]) -> np.ndarray:
        if self.pp is not None:
            if self.surface:
                return np.asarray(self.pp(Xn[0], Xn[1]), dtype=float)
            return np.asarray(self.pp(Xn[0]), dtype=float)
        with np.errstate(all="ignore"):
            return np.asarray(self.ft.evaluate(self.coef, Xn, self.P), dtype=float)

    def values_at(self, X: list[np.ndarray]) -> np.ndarray:
        shape = X[0].shape
        flat = [np.asarray(x, dtype=float).reshape(-1, order="F") for x in X]
        return self.eval_norm(self.normalize(flat)).reshape(shape, order="F")

    def query(self, args: list, fname: str) -> list[np.ndarray]:
        if self.surface:
            if len(args) == 1:
                Q = fnum(args[0]).astype(float)
                if Q.ndim != 2 or Q.shape[1] != 2:
                    raise MatlabError(f"{fname}: give the surface query points as x and y arrays or as a "
                                      "two-column matrix [x, y].", "curvefit:sfit:invalidQuery")
                return [Q[:, :1], Q[:, 1:2]]
            if len(args) == 2:
                try:
                    return [a.copy() for a in np.broadcast_arrays(fnum(args[0]).astype(float),
                                                                  fnum(args[1]).astype(float))]
                except ValueError:
                    raise MatlabError(f"{fname}: x and y must be the same size.") from None
            raise MatlabError(f"{fname}: an sfit takes x and y query arrays.", "MATLAB:TooManyInputs")
        if len(args) != 1:
            raise MatlabError(f"{fname}: a cfit takes one array of x values.",
                              "MATLAB:TooManyInputs" if args else "MATLAB:minrhs")
        return [fnum(args[0]).astype(float)]

    def index_get(self, interp, args: list) -> Any:
        return self.values_at(self.query(args, "feval"))

    # -- statistics -----------------------------------------------------------------------------------------
    def mse(self) -> float:
        return self.sse / self.dfe if self.dfe > 0 else np.nan

    def jac_at(self, Xn: list[np.ndarray]) -> np.ndarray:
        if self.ft.basis is not None:
            return self.ft.basis(Xn, self.P)
        return _num_jac(lambda c: self.ft.evaluate(c, Xn, self.P), self.coef)

    # -- display ----------------------------------------------------------------------------------------------
    def lines(self, name: str) -> list[str]:
        ft = self.ft
        args = ",".join(ft.indep)
        out = [f"     {ft.header()}", f"     {name}({args}) = {ft.formula_text()}"]
        if self.mu is not None:
            parts = [f"{v} is normalized by mean {_g(self.mu[j])} and std {_g(self.sd[j])}"
                     for j, v in enumerate(ft.indep)]
            out.append("       where " + "\n       and ".join(parts))
        if self.pp is not None:
            out += ["     Coefficients:", "       p = coefficient structure"]
            return out
        ci = _confint(self, 0.95) if self.cov is not None else None
        out.append("     Coefficients (with 95% confidence bounds):" if ci is not None else "     Coefficients:")
        w = max(len(c) for c in ft.coeffs)
        for k, c in enumerate(ft.coeffs):
            ln = f"       {c:>{w}} = {_g(self.coef[k]):>11}"
            if ci is not None:
                ln += f"  ({_g(ci[0, k])}, {_g(ci[1, k])})"
            out.append(ln)
        if ft.probs:
            out.append("     Problem parameters:")
            w = max(len(c) for c in ft.probs)
            for k, c in enumerate(ft.probs):
                out.append(f"       {c:>{w}} = {_g(self.P[k]):>11}")
        return out

    def display_lines(self) -> list[str]:
        return self.lines("f")

    def format_display(self, name: str) -> str:
        return f"{name} = \n\n" + "\n".join(self.lines(name)) + "\n\n"


def _p_value(f: CFit) -> Any:
    """The 'p' coefficient of a nonparametric fit: a piecewise polynomial, or a struct for a surface."""
    if f.surface:
        def conv(v):
            if isinstance(v, str):
                return char(v)
            a = np.asarray(v, dtype=float)
            return a.reshape(1, 1) if a.ndim == 0 else (a.reshape(-1, 1) if a.ndim == 1 else a)
        return Struct.scalar({k: conv(v) for k, v in f.pp.describe().items()})
    from .numerics import _pp_from_ppoly
    return _pp_from_ppoly(f.pp, 1)


def _confint(f: CFit, level: float) -> np.ndarray:
    from scipy.stats import t
    p = f.coef.size
    if f.cov is None or f.dfe <= 0:
        return np.full((2, p), np.nan)
    se = np.sqrt(np.clip(np.diag(f.cov) * f.mse(), 0, None))
    tc = t.ppf((1 + level) / 2, f.dfe)
    return np.vstack([f.coef - tc * se, f.coef + tc * se])


# -- methods ------------------------------------------------------------------------------------------------

_FT_METHODS: dict = {}
_CF_METHODS: dict = {}


def _m(name: str, table: dict):
    def deco(fn):
        table[name] = fn
        return fn
    return deco


def _ft(obj) -> FitType:
    return obj.ft if isinstance(obj, CFit) else obj


def _no_args(args, name):
    if args:
        raise MatlabError(f"Too many input arguments for {name}.", "MATLAB:TooManyInputs")


@_m("coeffnames", _FT_METHODS)
def _coeffnames(obj, interp, args, nargout):
    _no_args(args, "coeffnames")
    return _cellstr(_ft(obj).coeffs)


@_m("probnames", _FT_METHODS)
def _probnames(obj, interp, args, nargout):
    _no_args(args, "probnames")
    probs = _ft(obj).probs
    return _cellstr(probs) if probs else Cell.from_list([], column=True)


@_m("argnames", _FT_METHODS)
def _argnames(obj, interp, args, nargout):
    _no_args(args, "argnames")
    return _cellstr(_ft(obj).argnames())


@_m("indepnames", _FT_METHODS)
def _indepnames(obj, interp, args, nargout):
    _no_args(args, "indepnames")
    return _cellstr(_ft(obj).indep)


@_m("dependnames", _FT_METHODS)
def _dependnames(obj, interp, args, nargout):
    _no_args(args, "dependnames")
    return _cellstr([_ft(obj).dep])


@_m("numcoeffs", _FT_METHODS)
def _numcoeffs(obj, interp, args, nargout):
    _no_args(args, "numcoeffs")
    return float(len(_ft(obj).coeffs))


@_m("numargs", _FT_METHODS)
def _numargs(obj, interp, args, nargout):
    _no_args(args, "numargs")
    return float(len(_ft(obj).argnames()))


@_m("formula", _FT_METHODS)
def _formula(obj, interp, args, nargout):
    _no_args(args, "formula")
    return char(_ft(obj).formula_text())


@_m("type", _FT_METHODS)
def _type(obj, interp, args, nargout):
    _no_args(args, "type")
    return char(_ft(obj).name)


@_m("category", _FT_METHODS)
def _category(obj, interp, args, nargout):
    _no_args(args, "category")
    return char(_ft(obj).category)


@_m("islinear", _FT_METHODS)
def _islinear(obj, interp, args, nargout):
    _no_args(args, "islinear")
    return np.array([[_ft(obj).linear]])


@_m("fitoptions", _FT_METHODS)
def _ft_fitoptions(obj, interp, args, nargout):
    return _apply_opts(_default_options(_ft(obj)), parse_options(args, _ALL_OPTS, "fitoptions"), "fitoptions")


@_m("setoptions", _FT_METHODS)
def _setoptions(obj, interp, args, nargout):
    if len(args) != 1 or not isinstance(args[0], FitOptions):
        raise MatlabError("setoptions expects setoptions(ft, options) with a fitoptions object.")
    if isinstance(obj, CFit):
        raise MatlabError("setoptions applies to fittype objects.")
    import copy
    new = copy.copy(obj)
    new.options = args[0].copy()
    return new


@_m("feval", _FT_METHODS)
def _ft_feval(obj, interp, args, nargout):
    if isinstance(obj, CFit):
        return obj.values_at(obj.query(args, "feval"))
    ft = obj
    need = len(ft.coeffs) + len(ft.probs) + len(ft.indep)
    if len(args) != need:
        raise MatlabError(f"feval: this fittype takes {need} inputs: {', '.join(ft.argnames())}.",
                          "curvefit:fittype:feval:wrongNumberOfInputs")
    if ft.category not in ("library", "custom"):
        raise MatlabError("feval: interpolant and smoothing spline fittypes cannot be evaluated without data.")
    vals = [fnum(a).astype(float) for a in args]
    p, q = len(ft.coeffs), len(ft.probs)
    c = np.array([float(v.reshape(-1)[0]) for v in vals[:p]])
    P = [float(v.reshape(-1)[0]) for v in vals[p:p + q]]
    X = list(np.broadcast_arrays(*vals[p + q:]))
    shape = X[0].shape
    flat = [x.reshape(-1, order="F") for x in X]
    with np.errstate(all="ignore"):
        return np.asarray(ft.evaluate(c, flat, P), dtype=float).reshape(shape, order="F")


def _need_cfit(obj, name: str) -> CFit:
    if not isinstance(obj, CFit):
        raise MatlabError(f"{name} needs a fit object (cfit or sfit) made by fit.", "MATLAB:UndefinedFunction")
    return obj


def _need_parametric(f: CFit, name: str) -> None:
    if f.pp is not None:
        raise MatlabError(f"{name} is not available for interpolants and smoothing splines (they have no "
                          "coefficient covariance).", f"curvefit:{name}:notParametric")


@_m("coeffvalues", _CF_METHODS)
def _coeffvalues(obj, interp, args, nargout):
    f = _need_cfit(obj, "coeffvalues")
    _no_args(args, "coeffvalues")
    if f.pp is not None:
        return Cell.from_list([_p_value(f)])
    return _row(f.coef)


@_m("probvalues", _CF_METHODS)
def _probvalues(obj, interp, args, nargout):
    f = _need_cfit(obj, "probvalues")
    _no_args(args, "probvalues")
    return _row(f.P) if f.P else np.zeros((1, 0))


def _level(args: list, pos: int, name: str) -> float:
    if len(args) > pos and not is_text(args[pos]):
        a = fnum(args[pos])
        if a.size == 0:
            return 0.95
        lv = float(a.reshape(-1)[0])
        if not 0 < lv < 1:
            raise MatlabError(f"{name}: the confidence level must be between 0 and 1.",
                              f"curvefit:{name}:invalidLevel")
        return lv
    return 0.95


@_m("confint", _CF_METHODS)
def _m_confint(obj, interp, args, nargout):
    f = _need_cfit(obj, "confint")
    _need_parametric(f, "confint")
    if len(args) > 1:
        raise MatlabError("confint expects confint(fitobj) or confint(fitobj, level).", "MATLAB:TooManyInputs")
    return _confint(f, _level(args, 0, "confint"))


@_m("predint", _CF_METHODS)
def _m_predint(obj, interp, args, nargout):
    from scipy.stats import f as fdist
    from scipy.stats import t
    f = _need_cfit(obj, "predint")
    _need_parametric(f, "predint")
    if not args:
        raise MatlabError("predint expects predint(fitobj, x, ...).", "MATLAB:minrhs")
    nq = 2 if f.surface and len(args) >= 2 and not is_text(args[1]) and fnum(args[1]).shape == \
        fnum(args[0]).shape and fnum(args[0]).ndim == 2 and fnum(args[0]).shape[1] != 2 else 1
    X = f.query(list(args[:nq]), "predint")
    rest = list(args[nq:])
    level = _level(rest, 0, "predint")
    words = [text(a).lower() for a in rest if is_text(a)]
    for wd in words:
        if wd not in ("observation", "functional", "on", "off"):
            raise unknown_option_error("predint", wd, ["observation", "functional", "on", "off"], None)
    intopt = "functional" if "functional" in words else "observation"
    simopt = "on" if "on" in words else "off"
    shape = X[0].shape
    flat = [x.reshape(-1, order="F") for x in X]
    Xn = f.normalize(flat)
    yhat = f.eval_norm(Xn)
    p = f.coef.size
    if f.cov is None or f.dfe <= 0:
        lohi = np.full((yhat.size, 2), np.nan)
    else:
        Jq = f.jac_at(Xn)
        var = np.einsum("ij,jk,ik->i", Jq, f.cov, Jq) * f.mse()
        if intopt == "observation":
            var = var + f.mse()
        if simopt == "off":
            crit = t.ppf((1 + level) / 2, f.dfe)
        else:
            k = p + 1 if intopt == "observation" else p
            crit = np.sqrt(k * fdist.ppf(level, k, f.dfe))
        half = crit * np.sqrt(np.clip(var, 0, None))
        lohi = np.column_stack([yhat - half, yhat + half])
    if nargout >= 2:
        return lohi, yhat.reshape(shape, order="F") if len(shape) == 2 and shape[1] == 1 else _col(yhat)
    return lohi


def _numdiff(fn, x: np.ndarray, order: int) -> np.ndarray:
    if order == 1:
        h = _EPS ** (1 / 3) * np.maximum(np.abs(x), 1.0)
        return (fn(x + h) - fn(x - h)) / (2 * h)
    h = _EPS ** (1 / 4) * np.maximum(np.abs(x), 1.0)
    return (fn(x + h) - 2 * fn(x) + fn(x - h)) / h ** 2


@_m("differentiate", _CF_METHODS)
def _m_differentiate(obj, interp, args, nargout):
    f = _need_cfit(obj, "differentiate")
    X = f.query(args, "differentiate")
    shape = X[0].shape
    flat = [x.reshape(-1, order="F") for x in X]
    if f.surface:
        x, y = flat

        def fx(dx, dy):
            return f.eval_norm(f.normalize([x + dx, y + dy]))
        hx = _EPS ** (1 / 3) * np.maximum(np.abs(x), 1.0)
        hy = _EPS ** (1 / 3) * np.maximum(np.abs(y), 1.0)
        out = [(fx(hx, 0) - fx(-hx, 0)) / (2 * hx), (fx(0, hy) - fx(0, -hy)) / (2 * hy)]
        if nargout > 2:
            hx2 = _EPS ** (1 / 4) * np.maximum(np.abs(x), 1.0)
            hy2 = _EPS ** (1 / 4) * np.maximum(np.abs(y), 1.0)
            f0 = fx(0, 0)
            out += [(fx(hx2, 0) - 2 * f0 + fx(-hx2, 0)) / hx2 ** 2,
                    (fx(hx2, hy2) - fx(hx2, -hy2) - fx(-hx2, hy2) + fx(-hx2, -hy2)) / (4 * hx2 * hy2),
                    (fx(0, hy2) - 2 * f0 + fx(0, -hy2)) / hy2 ** 2]
        return tuple(o.reshape(shape, order="F") for o in out[:max(nargout, 1)])
    x = flat[0]
    s = 1.0 if f.sd is None else float(f.sd[0])
    xn = f.normalize([x])[0]
    if f.pp is not None:
        d1 = f.pp.derivative(1)(xn) / s
        d2 = f.pp.derivative(2)(xn) / s ** 2 if f.pp.c.shape[0] > 2 else np.zeros_like(xn)
    elif f.ft.name.startswith("poly") and f.ft.linear and f.ft.category == "library":
        c = f.coef
        d1 = np.polyval(np.polyder(c), xn) / s if c.size > 1 else np.zeros_like(xn)
        d2 = np.polyval(np.polyder(c, 2), xn) / s ** 2 if c.size > 2 else np.zeros_like(xn)
    else:
        def fn(z):
            return f.eval_norm(f.normalize([z]))
        d1 = _numdiff(fn, x, 1)
        d2 = _numdiff(fn, x, 2) if nargout > 1 else None
    if nargout > 1:
        return d1.reshape(shape, order="F"), d2.reshape(shape, order="F")
    return d1.reshape(shape, order="F")


@_m("integrate", _CF_METHODS)
def _m_integrate(obj, interp, args, nargout):
    f = _need_cfit(obj, "integrate")
    if f.surface:
        raise MatlabError("integrate is not available for surface fits.", "curvefit:sfit:integrate:notSupported")
    if len(args) != 2:
        raise MatlabError("integrate expects integrate(fitobj, x, x0).",
                          "MATLAB:minrhs" if len(args) < 2 else "MATLAB:TooManyInputs")
    xq = fnum(args[0]).astype(float)
    x0 = float(fnum(args[1]).reshape(-1)[0])
    flat = xq.reshape(-1, order="F")
    s = 1.0 if f.sd is None else float(f.sd[0])
    if f.pp is not None or (f.ft.name.startswith("poly") and f.ft.category == "library" and f.ft.linear):
        if f.pp is not None:
            F = f.pp.antiderivative()
        else:
            ci = np.polyint(f.coef)

            def F(z):
                return np.polyval(ci, z)
        a, b = f.normalize([np.array([x0])])[0], f.normalize([flat])[0]
        out = (F(b) - F(a)) * s
    else:
        from scipy.integrate import quad

        def g(z):
            return float(f.eval_norm(f.normalize([np.array([z])]))[0])
        out = np.array([quad(g, x0, z, limit=200)[0] if np.isfinite(z) else np.nan for z in flat])
    return np.asarray(out, dtype=float).reshape(xq.shape, order="F")


# -- plotting ------------------------------------------------------------------------------------------------

_PTYPES = ("fit", "predfunc", "predobs", "residuals", "stresiduals", "deriv1", "deriv2", "integral")


def _plot_call(interp, name: str, args: list, nargout: int = 0) -> list:
    return interp.call_builtin(name, BUILTINS[name], args, nargout)


@_m("plot", _CF_METHODS)
def _m_plot(obj, interp, args, nargout):
    f = _need_cfit(obj, "plot")
    rest = list(args)
    from .plotting import GraphicsHandle
    ax = []
    if rest and isinstance(rest[0], GraphicsHandle):
        ax = [rest.pop(0)]
    nums, specs, ptypes = [], [], []
    fit_spec = None
    for a in rest:
        if is_text(a):
            s = text(a)
            if s.lower() in _PTYPES:
                ptypes.append(s.lower())
            elif not nums and fit_spec is None and not specs:
                fit_spec = s
            else:
                specs.append(s)
        else:
            nums.append(a)
    data_spec = specs[0] if specs else "b."
    fit_spec = fit_spec or "r-"
    ptype = ptypes[0] if ptypes else "fit"
    if f.surface:
        return _plot_surface(interp, f, ax, nums, nargout)
    xd = yd = None
    if len(nums) >= 2:
        xd = fnum(nums[0]).astype(float).reshape(-1)
        yd = fnum(nums[1]).astype(float).reshape(-1)
        if xd.size != yd.size:
            raise MatlabError("plot: the x and y data must have the same number of elements.")
    elif len(nums) == 1:
        raise MatlabError("plot: give both x and y data, plot(fitobj, x, y).")
    if xd is not None and xd.size:
        lo, hi = float(np.min(xd)), float(np.max(xd))
    else:
        lo, hi = f.xrange[0] if f.xrange else (0.0, 1.0)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    xs = np.linspace(lo, hi, 1001)
    was_hold = bool(np.asarray(_plot_call(interp, "ishold", ax, 1)[0]).reshape(-1)[0])
    handles = []
    labels = []
    if ptype in ("residuals", "stresiduals"):
        if xd is None:
            raise MatlabError("plot: residual plots need the data, plot(fitobj, x, y, 'residuals').")
        r = yd - f.values_at([xd])
        if ptype == "stresiduals":
            r = r / np.sqrt(f.mse()) if np.isfinite(f.mse()) and f.mse() > 0 else r
        handles += _plot_call(interp, "plot", ax + [_col(xd), _col(r), char(data_spec)], 1)
        labels.append("data")
        _plot_call(interp, "hold", ax + [char("on")])
        handles += _plot_call(interp, "plot", ax + [_row([lo, hi]), _row([0.0, 0.0]), char("k-")], 1)
        labels.append("zero line")
    else:
        if xd is not None:
            handles += _plot_call(interp, "plot", ax + [_col(xd), _col(yd), char(data_spec)], 1)
            labels.append("data")
            _plot_call(interp, "hold", ax + [char("on")])
        if ptype in ("fit", "predfunc", "predobs"):
            ys = f.values_at([xs])
            label = "fitted curve"
        elif ptype in ("deriv1", "deriv2"):
            ys = _m_differentiate(f, interp, [xs], 2)[0 if ptype == "deriv1" else 1]
            label = "first derivative" if ptype == "deriv1" else "second derivative"
        else:
            ys = _m_integrate(f, interp, [xs, np.array([[lo]])], 1)
            label = "integral"
        handles += _plot_call(interp, "plot", ax + [_col(xs), _col(ys), char(fit_spec)], 1)
        labels.append(label)
        if ptype in ("predfunc", "predobs"):
            _plot_call(interp, "hold", ax + [char("on")])
            b = _m_predint(f, interp, [xs, np.array([[0.95]]),
                                       char("functional" if ptype == "predfunc" else "observation")], 1)
            handles += _plot_call(interp, "plot", ax + [_col(xs), b[:, :1].copy(), char("r:")], 1)
            handles += _plot_call(interp, "plot", ax + [_col(xs), b[:, 1:].copy(), char("r:")], 1)
            labels += ["prediction bounds", ""]
    if not was_hold:
        _plot_call(interp, "hold", ax + [char("off")])
    if len(labels) > 1:
        _plot_call(interp, "legend", ax + [char(lab) for lab in labels if lab])
    _plot_call(interp, "xlabel", ax + [char(f.ft.indep[0])])
    _plot_call(interp, "ylabel", ax + [char(f.ft.dep)])
    if nargout:
        return handles[0] if len(handles) == 1 else _hcat(handles)
    return None


def _hcat(handles: list) -> Any:
    try:
        return BUILTINS["vertcat"](None, handles, 1)
    except (MatlabError, TypeError, ValueError):
        return handles[0]


def _plot_surface(interp, f: CFit, ax: list, nums: list, nargout):
    if len(nums) >= 2:
        XY = fnum(nums[0]).astype(float)
        z = fnum(nums[1]).astype(float).reshape(-1)
        if XY.ndim != 2 or XY.shape[1] != 2:
            raise MatlabError("plot: give the surface data as plot(sfit, [x, y], z).")
        rx = (float(XY[:, 0].min()), float(XY[:, 0].max()))
        ry = (float(XY[:, 1].min()), float(XY[:, 1].max()))
    else:
        XY = z = None
        rx, ry = f.xrange[0], f.xrange[1]
    gx, gy = np.meshgrid(np.linspace(*rx, 49), np.linspace(*ry, 49))
    Z = f.values_at([gx, gy])
    was_hold = bool(np.asarray(_plot_call(interp, "ishold", ax, 1)[0]).reshape(-1)[0])
    h = _plot_call(interp, "surf", ax + [gx, gy, Z], 1)
    if XY is not None:
        _plot_call(interp, "hold", ax + [char("on")])
        _plot_call(interp, "plot3", ax + [XY[:, :1].copy(), XY[:, 1:].copy(), _col(z), char("ko"),
                                          char("MarkerFaceColor"), char("k")])
        if not was_hold:
            _plot_call(interp, "hold", ax + [char("off")])
    for nm, lab in (("xlabel", f.ft.indep[0]), ("ylabel", f.ft.indep[1]), ("zlabel", f.ft.dep)):
        _plot_call(interp, nm, ax + [char(lab)])
    return h[0] if nargout and h else None


# -- constructors and helpers --------------------------------------------------------------------------------

def _from_coeffs(interp, args: list, kind: str) -> CFit:
    if not args:
        raise MatlabError(f"{kind} expects {kind}(fittype, coef1, coef2, ...).", "MATLAB:minrhs")
    ft = args[0] if isinstance(args[0], FitType) else _make_fittype(interp, args[0], [], kind)
    if not ft.parametric:
        raise MatlabError(f"{kind}: interpolant and smoothing-spline models cannot be made from coefficients.")
    if (kind == "sfit") != (len(ft.indep) == 2):
        raise MatlabError(f"{kind}: the model has {len(ft.indep)} independent variable(s); use "
                          f"{'sfit' if len(ft.indep) == 2 else 'cfit'}.")
    for a in args[1:]:
        if is_text(a):
            raise unknown_option_error(kind, text(a), [], None)
    vals = [float(fnum(a).reshape(-1)[0]) for a in args[1:]]
    p, q = len(ft.coeffs), len(ft.probs)
    if len(vals) != p + q:
        raise MatlabError(f"{kind}: give {p} coefficient value(s)" + (f" and {q} problem parameter value(s)"
                                                                       if q else "") + ".")
    return CFit(ft, np.array(vals[:p]), vals[p:], cov=None)


@builtin("cfit")
def _cfit(interp, args, nargout):
    return _from_coeffs(interp, list(args), "cfit")


@builtin("sfit")
def _sfit(interp, args, nargout):
    return _from_coeffs(interp, list(args), "sfit")


def _prepare(args: list, nvars: int, fname: str) -> tuple:
    if len(args) not in (nvars, nvars + 1):
        raise MatlabError(f"{fname} expects {nvars} or {nvars + 1} data arrays.",
                          "MATLAB:minrhs" if len(args) < nvars else "MATLAB:TooManyInputs")
    arrs = [fnum(a).astype(float).reshape(-1, 1, order="F") for a in args]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise MatlabError(f"{fname}: the data arrays must have the same number of elements.",
                          f"curvefit:{fname}:unequalNumel")
    keep = np.all([np.isfinite(a[:, 0]) for a in arrs], axis=0) if n else np.zeros(0, dtype=bool)
    return tuple(a[keep] for a in arrs), int(np.sum(~keep))


@builtin("prepareCurveData")
def _prepare_curve(interp, args, nargout):
    out, dropped = _prepare(list(args), 2, "prepareCurveData")
    if dropped:
        interp.warn("Removing NaN and Inf from data", "curvefit:prepareFittingData:removingNaNAndInf")
    return out


@builtin("prepareSurfaceData")
def _prepare_surface(interp, args, nargout):
    if len(args) >= 3:
        x, y, z = (fnum(a).astype(float) for a in args[:3])
        if x.ndim == 2 and y.ndim == 2 and min(x.shape) == 1 and min(y.shape) == 1 and z.shape == (y.size, x.size) \
                and x.size * y.size == z.size and x.size != z.size:
            gx, gy = np.meshgrid(x.reshape(-1), y.reshape(-1))
            args = [gx, gy, z] + list(args[3:])
    out, dropped = _prepare(list(args), 3, "prepareSurfaceData")
    if dropped:
        interp.warn("Removing NaN and Inf from data", "curvefit:prepareFittingData:removingNaNAndInf")
    return out


# -- function forms of the methods: coeffvalues(f), confint(f), ... ------------------------------------------

def _dispatcher(name: str):
    def f(interp, args, nargout):
        for a in args:
            if isinstance(a, (FitType, CFit)) and a.has_method(name):
                return tuple(a.call_method(interp, name, list(args), nargout))
        kind = class_name(args[0]) if args else "double"
        raise MatlabError(f"Undefined function '{name}' for input arguments of type '{kind}'. It needs a fit "
                          "object (cfit or sfit) or a fittype.", "MATLAB:UndefinedFunction")
    f.__name__ = f"_cf_{name}"
    return f


# Names another module owns (the symbolic toolbox registers argnames and formula when it is imported, which
# is after this module); fit objects still answer them as methods, since an object's methods come first.
_OWNED_ELSEWHERE = {"argnames", "formula", "type", "feval", "plot", "fitoptions"}

for _name in sorted(set(_FT_METHODS) | set(_CF_METHODS)):
    if _name not in BUILTINS and _name not in _OWNED_ELSEWHERE:
        builtin(_name)(_dispatcher(_name))
