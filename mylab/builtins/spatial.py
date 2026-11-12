"""Computational geometry and scattered-data functions.

pdist2, knnsearch, dsearchn, inpolygon, polyarea, rectint, convhull, convhulln,
delaunay, delaunayn, griddata, scatteredInterpolant, sphere, cylinder, ellipsoid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import char
from ._registry import builtin
from ._util import fnum, fscalar, is_text, nargs, text, to_int


def _real(v: Any, what: str = "input") -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind not in "biuf":
        if isinstance(v, np.ndarray) and v.dtype.kind == "c":
            raise MatlabError(f"The {what} must be real.")
        raise MatlabError(f"The {what} must be a numeric array.")
    return v.astype(float)


def _options(args: list, names: set[str], what: str) -> tuple[list, dict]:
    """Split trailing Name,Value pairs (names matched without regard to case)."""
    rest = list(args)
    opts: dict = {}
    k = 0
    out = []
    while k < len(rest):
        a = rest[k]
        if is_text(a) and text(a).lower() in names:
            if k + 1 >= len(rest):
                raise MatlabError(f"{what}: the option '{text(a)}' needs a value.")
            opts[text(a).lower()] = rest[k + 1]
            k += 2
        else:
            out.append(a)
            k += 1
    return out, opts


# -- distances and nearest neighbours ----------------------------------------------------------

def _cdist(X: np.ndarray, Y: np.ndarray, metric: str, extra: list, what: str) -> np.ndarray:
    from scipy.spatial.distance import cdist

    from .stats import _METRICS
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise MatlabError(f"{what}: X and Y must have the same number of columns (one row per point); "
                          f"they have {X.shape[1] if X.ndim == 2 else '?'} and "
                          f"{Y.shape[1] if Y.ndim == 2 else '?'}.")
    metric = metric.lower()
    if metric not in _METRICS and metric != "mahalanobis":
        raise MatlabError(f"{what}: unsupported distance '{metric}'. Use one of: "
                          + ", ".join(sorted(_METRICS)) + ", mahalanobis.")
    if X.size == 0 or Y.size == 0:
        return np.zeros((X.shape[0], Y.shape[0]))
    kw: dict = {}
    if metric == "minkowski":
        kw["p"] = float(fnum(extra[0]).flat[0]) if extra else 2.0
    elif metric == "seuclidean":
        kw["V"] = fnum(extra[0]).flatten() ** 2 if extra else np.var(X, axis=0, ddof=1)
    elif metric == "mahalanobis":
        C = fnum(extra[0]) if extra else np.cov(X, rowvar=False)
        kw["VI"] = np.linalg.inv(np.atleast_2d(C))
        return cdist(X, Y, "mahalanobis", **kw)
    elif metric == "spearman":
        from scipy.stats import rankdata
        return cdist(np.apply_along_axis(rankdata, 1, X), np.apply_along_axis(rankdata, 1, Y), "correlation")
    return cdist(X, Y, _METRICS[metric], **kw)


@builtin("pdist2")
def _pdist2(interp, args, nargout):
    rest, opts = _options(args, {"smallest", "largest"}, "pdist2")
    nargs(rest, 2, 4, "pdist2")
    X, Y = _real(rest[0], "first input of pdist2"), _real(rest[1], "second input of pdist2")
    metric = "euclidean"
    extra = rest[3:]
    if len(rest) > 2:
        if not is_text(rest[2]):
            raise MatlabError("pdist2: the distance must be a name such as 'euclidean' or 'cityblock'; "
                              "function-handle distances are not supported.")
        metric = text(rest[2])
    D = _cdist(X, Y, metric, extra, "pdist2")
    if not opts:
        if nargout > 1:
            raise MatlabError("pdist2 returns indices only with the 'Smallest' or 'Largest' option.")
        return D
    key = "smallest" if "smallest" in opts else "largest"
    k = min(to_int(opts[key], "number of distances"), D.shape[0])
    order = np.argsort(D if key == "smallest" else -D, axis=0, kind="stable")[:k, :]
    return np.take_along_axis(D, order, axis=0), order + 1.0


@builtin("knnsearch")
def _knnsearch(interp, args, nargout):
    rest, opts = _options(args, {"k", "distance", "p", "nsmethod", "includeties", "cov", "scale",
                                 "bucketsize", "sortindices"}, "knnsearch")
    nargs(rest, 2, 2, "knnsearch")
    X, Y = _real(rest[0], "first input of knnsearch"), _real(rest[1], "second input of knnsearch")
    k = to_int(opts["k"], "K") if "k" in opts else 1
    if k < 1:
        raise MatlabError("knnsearch: K must be a positive integer.")
    metric = text(opts["distance"]) if "distance" in opts else "euclidean"
    extra = [opts[n] for n in ("p", "scale", "cov") if n in opts]
    D = _cdist(X, Y, metric, extra, "knnsearch").T  # one row per query point
    k = min(k, X.shape[0])
    order = np.argsort(D, axis=1, kind="stable")[:, :k]
    return order + 1.0, np.take_along_axis(D, order, axis=1)


@builtin("dsearchn")
def _dsearchn(interp, args, nargout):
    nargs(args, 2, 3, "dsearchn")
    X = _real(args[0], "points")
    Q = _real(args[-1], "query points")
    if X.shape[1] != Q.shape[1]:
        raise MatlabError("dsearchn: the points and the query points must have the same number of columns.")
    from scipy.spatial import cKDTree
    d, i = cKDTree(X).query(Q)
    return (i + 1.0).reshape(-1, 1), d.reshape(-1, 1)


# -- polygons ------------------------------------------------------------------------------------

def _loops(xv: np.ndarray, yv: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split polygon vertices at NaN into closed loops."""
    loops = []
    start = 0
    bad = np.isnan(xv) | np.isnan(yv)
    for k in list(np.flatnonzero(bad)) + [xv.size]:
        if k - start >= 1:
            x, y = xv[start:k], yv[start:k]
            if x.size > 1 and x[0] == x[-1] and y[0] == y[-1]:
                x, y = x[:-1], y[:-1]
            loops.append((x, y))
        start = k + 1
    return loops


@builtin("inpolygon")
def _inpolygon(interp, args, nargout):
    nargs(args, 4, 4, "inpolygon")
    xq, yq = _real(args[0], "query x"), _real(args[1], "query y")
    xv, yv = _real(args[2], "polygon x").flatten(), _real(args[3], "polygon y").flatten()
    if xq.shape != yq.shape:
        raise MatlabError("inpolygon: xq and yq must be the same size.")
    if xv.size != yv.size:
        raise MatlabError("inpolygon: xv and yv must have the same number of elements.")
    px, py = xq.flatten(order="F"), yq.flatten(order="F")
    inside = np.zeros(px.size, dtype=bool)
    on = np.zeros(px.size, dtype=bool)
    for x, y in _loops(xv, yv):
        n = x.size
        crossings = np.zeros(px.size, dtype=bool)
        scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1.0) if n else 1.0
        for i in range(n):
            x1, y1, x2, y2 = x[i], y[i], x[(i + 1) % n], y[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            cross = dx * (py - y1) - dy * (px - x1)
            within = (np.minimum(x1, x2) <= px) & (px <= np.maximum(x1, x2)) & \
                     (np.minimum(y1, y2) <= py) & (py <= np.maximum(y1, y2))
            on |= within & (np.abs(cross) <= 4 * np.finfo(float).eps * scale * max(abs(dx), abs(dy), 1.0))
            straddle = (y1 > py) != (y2 > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = x1 + (py - y1) * dx / dy
            crossings ^= straddle & (px < xi)
        inside ^= crossings
    inside |= on
    shape = xq.shape
    return inside.reshape(shape, order="F"), on.reshape(shape, order="F")


@builtin("polyarea")
def _polyarea(interp, args, nargout):
    nargs(args, 2, 3, "polyarea")
    x, y = _real(args[0], "x"), _real(args[1], "y")
    if x.shape != y.shape:
        raise MatlabError("polyarea: x and y must be the same size.")
    if len(args) > 2:
        axis = to_int(args[2], "dimension") - 1
    else:
        axis = next((d for d, n in enumerate(x.shape) if n != 1), 0)
    if axis >= x.ndim:
        return np.zeros(x.shape)
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=axis) - y * np.roll(x, -1, axis=axis), axis=axis))
    return np.expand_dims(area, axis)


@builtin("rectint")
def _rectint(interp, args, nargout):
    nargs(args, 2, 2, "rectint")
    A, B = _real(args[0], "A"), _real(args[1], "B")
    if A.shape[1] != 4 or B.shape[1] != 4:
        raise MatlabError("rectint: each rectangle is a row [x y width height].")
    w = np.minimum(A[:, None, 0] + A[:, None, 2], B[None, :, 0] + B[None, :, 2]) - \
        np.maximum(A[:, None, 0], B[None, :, 0])
    h = np.minimum(A[:, None, 1] + A[:, None, 3], B[None, :, 1] + B[None, :, 3]) - \
        np.maximum(A[:, None, 1], B[None, :, 1])
    return np.maximum(w, 0) * np.maximum(h, 0)


# -- hulls and triangulations ---------------------------------------------------------------------

def _points(args: list, what: str) -> tuple[np.ndarray, list]:
    """Points from (P) or (x, y) or (x, y, z); returns the n-by-d array and the remaining arguments."""
    nums = []
    rest = list(args)
    while rest and not is_text(rest[0]) and isinstance(rest[0], np.ndarray):
        nums.append(_real(rest.pop(0), f"input of {what}"))
    if not nums:
        raise MatlabError(f"{what} needs point coordinates.")
    if len(nums) == 1:
        P = nums[0]
        if P.ndim != 2 or P.shape[1] < 2:
            raise MatlabError(f"{what}: give the points as an n-by-2 or n-by-3 matrix, or as separate "
                              "x, y (and z) vectors.")
    else:
        if len({a.size for a in nums}) != 1:
            raise MatlabError(f"{what}: the coordinate vectors must have the same number of elements.")
        P = np.column_stack([a.flatten(order="F") for a in nums])
    if np.isnan(P).any() or np.isinf(P).any():
        raise MatlabError(f"{what}: the points must not contain NaN or Inf.")
    return P, rest


def _qhull(fn, P: np.ndarray, what: str):
    from scipy.spatial import QhullError
    if P.shape[0] <= P.shape[1]:
        raise MatlabError(f"{what} needs at least {P.shape[1] + 1} points that are not all on one "
                          f"{'line' if P.shape[1] == 2 else 'plane'}.")
    try:
        return fn(P)
    except QhullError:
        raise MatlabError(f"{what}: the points are degenerate (all on one "
                          f"{'line' if P.shape[1] == 2 else 'plane'}, or repeated), so there is nothing "
                          "to compute.") from None


@builtin("convhull")
def _convhull(interp, args, nargout):
    from scipy.spatial import ConvexHull
    P, _ = _points(args, "convhull")
    hull = _qhull(ConvexHull, P, "convhull")
    if P.shape[1] == 2:
        v = hull.vertices  # counterclockwise
        start = int(np.lexsort((P[v, 1], P[v, 0]))[0])  # begin at the leftmost (then lowest) point
        v = np.roll(v, -start)
        k = np.append(v, v[0]) + 1.0
        return k.reshape(-1, 1), fscalar(hull.volume)
    return _outward(P, hull) + 1.0, fscalar(hull.volume)


def _outward(P: np.ndarray, hull) -> np.ndarray:
    """Hull facets with vertices ordered so that normals point outwards (3-D)."""
    faces = hull.simplices.copy()
    if P.shape[1] == 3:
        a, b, c = P[faces[:, 0]], P[faces[:, 1]], P[faces[:, 2]]
        flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces.astype(float)


@builtin("convhulln")
def _convhulln(interp, args, nargout):
    from scipy.spatial import ConvexHull
    P = _real(args[0], "points") if args else None
    if P is None or P.ndim != 2:
        raise MatlabError("convhulln needs an n-by-d matrix of points.")
    hull = _qhull(ConvexHull, P, "convhulln")
    return _outward(P, hull) + 1.0, fscalar(hull.volume)


def _delaunay_tri(P: np.ndarray, what: str) -> np.ndarray:
    from scipy.spatial import Delaunay
    tri = _qhull(Delaunay, P, what).simplices.copy()
    if P.shape[1] == 2:  # counterclockwise triangles
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        cw = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
        tri[cw] = tri[cw][:, [0, 2, 1]]
    return tri


@builtin("delaunay")
def _delaunay(interp, args, nargout):
    P, _ = _points(args, "delaunay")
    return _delaunay_tri(P, "delaunay") + 1.0


@builtin("delaunayn")
def _delaunayn(interp, args, nargout):
    P = _real(args[0], "points") if args else None
    if P is None or P.ndim != 2:
        raise MatlabError("delaunayn needs an n-by-d matrix of points.")
    return _delaunay_tri(P, "delaunayn") + 1.0


# -- scattered data interpolation ---------------------------------------------------------------------

_METHODS = {"linear", "nearest", "natural", "cubic", "v4"}


def _v4(P: np.ndarray, v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Biharmonic spline interpolation in two dimensions."""
    def green(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = r ** 2 * (np.log(r) - 1.0)
        g[r == 0] = 0.0
        return g
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    w = np.linalg.lstsq(green(d), v, rcond=None)[0]
    dq = np.hypot(Q[:, None, 0] - P[None, :, 0], Q[:, None, 1] - P[None, :, 1])
    return green(dq) @ w


def _scatter_eval(P: np.ndarray, v: np.ndarray, Q: np.ndarray, method: str) -> np.ndarray:
    from scipy.interpolate import griddata
    if P.shape[0] == 0:
        raise MatlabError("There are no sample points to interpolate.")
    if method == "v4":
        if P.shape[1] != 2:
            raise MatlabError("The 'v4' method works for two-dimensional data only.")
        return _v4(P, v, Q)
    if method == "cubic" and P.shape[1] != 2:
        method = "linear"
    kind = {"natural": "linear"}.get(method, method)
    from scipy.spatial import QhullError
    try:
        if np.iscomplexobj(v):
            return griddata(P, v.real, Q, method=kind) + 1j * griddata(P, v.imag, Q, method=kind)
        return griddata(P, v, Q, method=kind)
    except QhullError:
        raise MatlabError("The sample points are degenerate (for example all on one line), so they "
                          "cannot be triangulated.") from None


def _dedupe(P: np.ndarray, v: np.ndarray, interp, what: str) -> tuple[np.ndarray, np.ndarray]:
    uniq, inverse = np.unique(P, axis=0, return_inverse=True)
    if uniq.shape[0] == P.shape[0]:
        return P, v
    interp.warn(f"{what}: duplicate sample points were found; their values were averaged.",
                "MATLAB:griddata:DuplicateDataPoints")
    inverse = inverse.flatten()
    sums = np.zeros(uniq.shape[0], dtype=v.dtype)
    np.add.at(sums, inverse, v)
    return uniq, sums / np.bincount(inverse)


@builtin("griddata")
def _griddata(interp, args, nargout):
    rest = list(args)
    method = "linear"
    if rest and is_text(rest[-1]):
        method = text(rest.pop()).lower()
        if method not in _METHODS:
            raise MatlabError(f"griddata: unknown method '{method}'. Use 'linear', 'nearest', 'natural', "
                              "'cubic' or 'v4'.")
    if len(rest) == 5:
        d = 2
    elif len(rest) == 7:
        d = 3
    else:
        raise MatlabError("griddata expects griddata(x, y, v, xq, yq) or griddata(x, y, z, v, xq, yq, zq), "
                          "optionally followed by a method name.")
    coords = [fnum(a).astype(float).flatten(order="F") for a in rest[:d]]
    v = fnum(rest[d]).flatten(order="F")
    if any(c.size != v.size for c in coords):
        raise MatlabError("griddata: the sample coordinates and the values must have the same number of "
                          "elements.")
    q = [fnum(a).astype(float) for a in rest[d + 1:]]
    if d == 2 and q[0].shape != q[1].shape:
        if q[0].shape[0] == 1 and q[1].shape[1] == 1:  # row xq and column yq describe a grid
            q[0], q[1] = np.meshgrid(q[0].flatten(), q[1].flatten())
        else:
            raise MatlabError("griddata: xq and yq must be the same size (use meshgrid), or a row vector "
                              "and a column vector.")
    if len({a.shape for a in q}) != 1:
        raise MatlabError("griddata: the query coordinate arrays must be the same size.")
    P = np.column_stack(coords)
    keep = ~(np.isnan(P).any(axis=1) | np.isnan(v))
    P, v = _dedupe(P[keep], v[keep], interp, "griddata")
    Q = np.column_stack([a.flatten(order="F") for a in q])
    out = _scatter_eval(P, v, Q, method).reshape(q[0].shape, order="F")
    if nargout >= 3 and d == 2:
        return q[0], q[1], out
    return out


class ScatteredInterpolant(MObject):
    """F = scatteredInterpolant(x, y, v); vq = F(xq, yq)."""

    def __init__(self, P: np.ndarray, v: np.ndarray, method: str = "linear", extrap: str = "linear",
                 kind: str = "scatteredInterpolant"):
        self.P, self.v, self.method, self.extrap, self.kind = P, v, method, extrap, kind

    def class_name(self) -> str:
        return self.kind

    def copy(self) -> "ScatteredInterpolant":
        return ScatteredInterpolant(self.P, self.v, self.method, self.extrap, self.kind)

    def get_field(self, interp, name: str) -> Any:
        if name in ("Points", "X"):
            return self.P.copy()
        if name in ("Values", "V"):
            return self.v.reshape(-1, 1).copy()
        if name == "Method":
            return char(self.method)
        if name == "ExtrapolationMethod":
            return char(self.extrap)
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")

    def set_field(self, interp, name: str, value: Any) -> "ScatteredInterpolant":
        new = self.copy()
        if name in ("Values", "V"):
            v = fnum(value).flatten(order="F")
            if v.size != self.P.shape[0]:
                raise MatlabError("The number of values must equal the number of sample points.")
            new.v = v
        elif name in ("Points", "X"):
            P = _real(value, "points")
            if P.shape[0] != self.v.size:
                raise MatlabError("The number of points must equal the number of values.")
            new.P = P
        elif name == "Method":
            new.method = _method_name(value)
        elif name == "ExtrapolationMethod":
            new.extrap = _extrap_name(value)
        else:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.kind}'.")
        return new

    def _extrapolate(self, Q: np.ndarray) -> np.ndarray:
        """Linear extrapolation: extend the plane of the triangle nearest to each query point."""
        from scipy.spatial import Delaunay, cKDTree
        tri = Delaunay(self.P)
        _, nearest = cKDTree(self.P).query(Q)
        simplex_of_vertex = np.full(self.P.shape[0], -1)
        for s, verts in enumerate(tri.simplices):
            simplex_of_vertex[verts] = s
        s = simplex_of_vertex[nearest]
        d = self.P.shape[1]
        T = tri.transform[s]
        bary = np.einsum("ijk,ik->ij", T[:, :d, :], Q - T[:, d, :])
        bary = np.column_stack([bary, 1 - bary.sum(axis=1)])
        return np.einsum("ij,ij->i", bary, self.v[tri.simplices[s]])

    def evaluate(self, Q: np.ndarray) -> np.ndarray:
        out = _scatter_eval(self.P, self.v, Q, self.method)
        missing = np.isnan(out) & ~np.isnan(Q).any(axis=1)
        if missing.any() and self.extrap != "none":
            if self.extrap == "nearest" or self.method == "nearest":
                out[missing] = _scatter_eval(self.P, self.v, Q[missing], "nearest")
            else:
                out[missing] = self._extrapolate(Q[missing])
        return out

    def index_get(self, interp, args: list) -> Any:
        d = self.P.shape[1]
        q = [fnum(a).astype(float) for a in args]
        if len(q) == 1 and q[0].ndim == 2 and q[0].shape[1] == d:
            return self.evaluate(q[0]).reshape(-1, 1)
        if len(q) != d:
            raise MatlabError(f"This interpolant has {d}-D sample points: call it with {d} coordinate "
                              f"arrays, F({', '.join('xyz'[k] + 'q' for k in range(d))}), or one "
                              f"n-by-{d} matrix of query points.")
        try:
            q = list(np.broadcast_arrays(*q))
        except ValueError:
            raise MatlabError("The query coordinate arrays must be the same size.") from None
        Q = np.column_stack([a.flatten(order="F") for a in q])
        return self.evaluate(Q).reshape(q[0].shape, order="F")

    def display_lines(self) -> list[str]:
        n, d = self.P.shape
        return [f"  {self.kind} with properties:", "",
                f"                 Points: [{n}×{d} double]",
                f"                 Values: [{n}×1 double]",
                f"                 Method: '{self.method}'",
                f"    ExtrapolationMethod: '{self.extrap}'"]


def _method_name(v: Any) -> str:
    m = text(v).lower()
    if m not in ("linear", "nearest", "natural"):
        raise MatlabError(f"Unknown interpolation method '{m}'. Use 'linear', 'nearest' or 'natural'.")
    return m


def _extrap_name(v: Any) -> str:
    m = text(v).lower()
    if m not in ("linear", "nearest", "none"):
        raise MatlabError(f"Unknown extrapolation method '{m}'. Use 'linear', 'nearest' or 'none'.")
    return m


def _make_interpolant(interp, args: list, kind: str, default_extrap: str) -> ScatteredInterpolant:
    rest = list(args)
    words = []
    while rest and is_text(rest[-1]):
        words.insert(0, rest.pop())
    if len(rest) < 2:
        raise MatlabError(f"{kind} needs sample points and values, e.g. {kind}(x, y, v).")
    v = fnum(rest[-1]).flatten(order="F")
    coords = [_real(a, "sample points") for a in rest[:-1]]
    P = coords[0] if len(coords) == 1 else np.column_stack([c.flatten(order="F") for c in coords])
    if P.ndim != 2 or P.shape[1] not in (2, 3):
        raise MatlabError(f"{kind} works with 2-D or 3-D sample points.")
    if P.shape[0] != v.size:
        raise MatlabError(f"{kind}: there are {P.shape[0]} sample points but {v.size} values.")
    P, v = _dedupe(P, v, interp, kind)
    method = _method_name(words[0]) if words else "linear"
    extrap = _extrap_name(words[1]) if len(words) > 1 else default_extrap
    return ScatteredInterpolant(P, v, method, extrap, kind)


@builtin("scatteredInterpolant")
def _scattered_interpolant(interp, args, nargout):
    return _make_interpolant(interp, args, "scatteredInterpolant", "linear")


@builtin("TriScatteredInterp")
def _tri_scattered_interp(interp, args, nargout):
    return _make_interpolant(interp, args, "TriScatteredInterp", "none")


# -- sphere, cylinder, ellipsoid ----------------------------------------------------------------------

def _snap(a: np.ndarray) -> np.ndarray:
    """Round-off such as cos(pi/2) = 6e-17 becomes an exact zero, so poles and seams close up."""
    a[np.abs(a) < 4 * np.finfo(float).eps] = 0.0
    return a


def _unit_sphere(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longitude runs from -pi to pi along the columns, latitude from the south pole up the rows."""
    lon = np.linspace(-np.pi, np.pi, n + 1).reshape(1, -1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n + 1).reshape(-1, 1)
    ring = _snap(np.cos(lat))
    x = ring * _snap(np.cos(lon))
    y = ring * _snap(np.sin(lon))
    z = np.sin(lat) * np.ones((1, n + 1))
    return x, y, z


def _show_surface(interp, X, Y, Z) -> None:
    interp.call_builtin("surf", interp.builtins["surf"], [X, Y, Z], 0)


def _count(args: list, default: int, what: str) -> int:
    if not args:
        return default
    n = to_int(args[0], "number of faces")
    if n < 1:
        raise MatlabError(f"{what}: the number of faces must be a positive integer.")
    return n


def _drop_axes(args: list) -> list:
    from .plotting import GraphicsHandle
    return [a for a in args if not isinstance(a, GraphicsHandle)]


@builtin("sphere")
def _sphere(interp, args, nargout):
    x, y, z = _unit_sphere(_count(_drop_axes(args), 20, "sphere"))
    if nargout == 0:
        _show_surface(interp, x, y, z)
        return None
    return x, y, z


@builtin("ellipsoid")
def _ellipsoid(interp, args, nargout):
    args = _drop_axes(args)
    nargs(args, 6, 7, "ellipsoid")
    c = [float(fnum(a).flat[0]) for a in args[:6]]
    x, y, z = _unit_sphere(_count(args[6:], 20, "ellipsoid"))
    x, y, z = c[0] + c[3] * x, c[1] + c[4] * y, c[2] + c[5] * z
    if nargout == 0:
        _show_surface(interp, x, y, z)
        return None
    return x, y, z


@builtin("cylinder")
def _cylinder(interp, args, nargout):
    args = _drop_axes(args)
    r = fnum(args[0]).astype(float).flatten() if args else np.array([1.0, 1.0])
    n = _count(args[1:], 20, "cylinder")
    if r.size == 1:
        r = np.array([r[0], r[0]])
    angle = np.linspace(0.0, 2 * np.pi, n + 1).reshape(1, -1)
    x = r.reshape(-1, 1) * _snap(np.cos(angle))
    y = r.reshape(-1, 1) * _snap(np.sin(angle))
    z = (np.arange(r.size) / (r.size - 1)).reshape(-1, 1) * np.ones((1, n + 1))
    if nargout == 0:
        _show_surface(interp, x, y, z)
        return None
    return x, y, z
