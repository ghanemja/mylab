"""Nonparametric surface fits for fit([x y], z, ...): lowess/loess smoothing and scattered-data interpolants.

* lowess / loess: locally weighted regression. At each point the model is a weighted least-squares fit, linear
  (lowess) or quadratic (loess) in x and y, to the nearest ``span * n`` data points, with tricube weights
  ``(1 - (d / dmax)^3)^3`` of the distance d relative to the largest of those distances. Robust fitting
  repeats the fit with bisquare weights of the residuals (or weights that approximate least absolute
  residuals, 'LAR').
* linearinterp, nearestinterp, cubicinterp: interpolation over a Delaunay triangulation of the data (scipy's
  griddata methods); NaN outside the convex hull of the data.
* thinplateinterp: thin-plate spline with a linear polynomial part (scipy's RBFInterpolator).
* biharmonicinterp: the biharmonic Green's-function surface, ``z(p) = sum_j w_j g(|p - p_j|)`` with
  ``g(r) = r^2 (ln r - 1)``, whose weights make it pass through every data point.
"""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError

SURFACE_INTERPS = {"linearinterp": "Linear interpolant", "nearestinterp": "Nearest neighbor interpolant",
                   "cubicinterp": "Cubic interpolant", "biharmonicinterp": "Biharmonic interpolant",
                   "thinplateinterp": "Thin-plate spline interpolant"}
SURFACE_ONLY = ("biharmonicinterp", "thinplateinterp")
METHODS = {"linearinterp": "LinearInterpolant", "nearestinterp": "NearestInterpolant",
           "cubicinterp": "CubicSplineInterpolant", "biharmonicinterp": "BiharmonicInterpolant",
           "thinplateinterp": "ThinPlateInterpolant"}


def _unique_points(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Data with repeated (x, y) points replaced by one point at the mean of their z values."""
    pts = np.column_stack([x, y])
    u, inv = np.unique(pts, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    if u.shape[0] == pts.shape[0]:
        return x, y, z
    zs = np.bincount(inv, weights=z) / np.bincount(inv)
    return u[:, 0], u[:, 1], zs


class SurfaceModel:
    """A callable surface z = f(x, y) built from data (the 'p' of a nonparametric sfit)."""

    kind = "surface"

    def __call__(self, x, y) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"form": self.kind}


class Interpolant(SurfaceModel):
    def __init__(self, name: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        x, y, z = _unique_points(x, y, z)
        need = 1 if name == "nearestinterp" else 3
        if x.size < need:
            raise MatlabError(f"fit: the {name} surface needs at least {need} distinct data points.",
                              "curvefit:fit:tooFewPoints")
        self.kind = name
        self.pts = np.column_stack([x, y])
        self.z = z
        self._tri = None
        if name in ("linearinterp", "nearestinterp", "cubicinterp"):
            from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
            from scipy.spatial import Delaunay, QhullError
            try:
                self._tri = Delaunay(self.pts)
            except QhullError:
                raise MatlabError("fit: the data points of an interpolant surface must not all lie on one line.",
                                  "curvefit:fit:collinearData") from None
            if name == "linearinterp":
                self._f = LinearNDInterpolator(self._tri, z)
            elif name == "cubicinterp":
                self._f = CloughTocher2DInterpolator(self._tri, z)
            else:
                self._f = NearestNDInterpolator(self.pts, z)
        elif name == "thinplateinterp":
            from scipy.interpolate import RBFInterpolator
            self._f = RBFInterpolator(self.pts, z, kernel="thin_plate_spline", degree=1)
        else:  # biharmonic
            G = _green(np.hypot(self.pts[:, :1] - self.pts[:, 0], self.pts[:, 1:] - self.pts[:, 1]))
            self.weights = np.linalg.lstsq(G, z, rcond=None)[0]

    def __call__(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        q = np.column_stack([x, y])
        out = np.full(x.size, np.nan)
        ok = np.all(np.isfinite(q), axis=1)
        if not ok.any():
            return out
        qq = q[ok]
        if self.kind == "biharmonicinterp":
            vals = np.empty(qq.shape[0])
            for s in range(0, qq.shape[0], 2000):
                part = qq[s:s + 2000]
                r = np.hypot(part[:, :1] - self.pts[:, 0], part[:, 1:] - self.pts[:, 1])
                vals[s:s + 2000] = _green(r) @ self.weights
        else:
            vals = np.asarray(self._f(qq), dtype=float).reshape(-1)
            if self.kind == "nearestinterp":  # like the other triangulation interpolants: NaN outside the hull
                vals = np.where(self._tri.find_simplex(qq) >= 0, vals, np.nan)
        out[ok] = vals
        return out

    def describe(self) -> dict:
        return {"form": self.kind, "breaks": self.pts, "values": self.z}


def _green(r: np.ndarray) -> np.ndarray:
    """The biharmonic Green's function in the plane, r^2 (ln r - 1), with value 0 at r = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        g = r ** 2 * (np.log(r) - 1.0)
    return np.where(r > 0, g, 0.0)


def _design(dx: np.ndarray, dy: np.ndarray, quadratic: bool) -> np.ndarray:
    cols = [np.ones_like(dx), dx, dy]
    if quadratic:
        cols += [dx * dx, dx * dy, dy * dy]
    return np.stack(cols, axis=-1)


class Lowess(SurfaceModel):
    """Locally weighted linear (lowess) or quadratic (loess) regression surface."""

    def __init__(self, name: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray, span: float,
                 robust: str):
        self.kind = name
        self.quadratic = name == "loess"
        self.pts = np.column_stack([x, y])
        self.z = np.asarray(z, dtype=float)
        self.w = np.asarray(w, dtype=float)
        n = self.z.size
        need = 6 if self.quadratic else 3
        if n < need:
            raise MatlabError(f"fit: a {name} surface needs at least {need} data points.",
                              "curvefit:fit:tooFewPoints")
        if not 0 < span <= 1:
            raise MatlabError("fit: Span must be a fraction of the data points, greater than 0 and at most 1.",
                              "curvefit:fit:invalidSpan")
        self.span = span
        # neighbours per local fit: the span's share of the points, but at least enough for the local model
        self.k = int(min(n, max(int(np.ceil(span * n)), need + 1)))
        self.rw = np.ones(n)
        if robust != "Off":
            for _ in range(5):
                r = self.z - self._smooth(self.pts)[0]
                if robust == "LAR":
                    self.rw = 1.0 / np.maximum(np.abs(r), 1e-6 * max(np.max(np.abs(r)), 1e-300))
                else:  # bisquare weights of the residuals scaled by 6 median absolute deviations
                    mad = np.median(np.abs(r - np.median(r)))
                    if mad <= 0:
                        break
                    u = r / (6 * mad)
                    self.rw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

    def _smooth(self, q: np.ndarray, hat: bool = False):
        """Values at the points q (and, with ``hat``, the diagonal of the smoother matrix for q = the data)."""
        out = np.full(q.shape[0], np.nan)
        lev = np.zeros(q.shape[0]) if hat else None
        base_w = self.w * self.rw
        for i, p in enumerate(q):
            d = np.hypot(self.pts[:, 0] - p[0], self.pts[:, 1] - p[1])
            near = np.argpartition(d, self.k - 1)[:self.k] if self.k < d.size else np.arange(d.size)
            dmax = d[near].max()
            if dmax > 0:
                u = d[near] / dmax
                tw = (1 - u ** 3) ** 3
            else:
                tw = np.ones(near.size)
            wt = tw * base_w[near]
            A = _design(self.pts[near, 0] - p[0], self.pts[near, 1] - p[1], self.quadratic)
            sw = np.sqrt(wt)
            coef, _, rank, _ = np.linalg.lstsq(A * sw[:, None], self.z[near] * sw, rcond=None)
            out[i] = coef[0]  # the local model is centred on p: its constant term is the value there
            if hat:
                pinv = np.linalg.pinv(A * sw[:, None])
                j = np.nonzero(near == i)[0]
                if j.size:
                    lev[i] = pinv[0, j[0]] * sw[j[0]]
        return out, lev

    def __call__(self, x, y) -> np.ndarray:
        q = np.column_stack([np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)])
        out = np.full(q.shape[0], np.nan)
        ok = np.all(np.isfinite(q), axis=1)
        if ok.any():
            out[ok] = self._smooth(q[ok])[0]
        return out

    def dof(self) -> float:
        """Effective number of parameters: the trace of the smoother matrix at the data."""
        return float(np.sum(self._smooth(self.pts, hat=True)[1]))

    def describe(self) -> dict:
        return {"form": self.kind, "span": self.span, "data": self.pts, "values": self.z}
