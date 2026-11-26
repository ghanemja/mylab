"""griddedInterpolant: interpolation on a full grid in any number of dimensions.

``F = griddedInterpolant(x, v)``, ``F = griddedInterpolant(X1, X2, ..., V)`` (ndgrid-format full grids),
``F = griddedInterpolant({x1, x2, ...}, V)`` or ``F = griddedInterpolant(V)``, each optionally followed by
the interpolation method and the extrapolation method. ``F(xq)``, ``F(Xq1, Xq2, ...)``, ``F(Q)`` (one query
point per row) and ``F({xq1, xq2, ...})`` (a grid of query points) evaluate it. The interpolation is the
tensor product of 1-D interpolations along each dimension (SciPy for the piecewise polynomials). Values with
one more dimension than the grid hold several values per grid point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..display import summary
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, char
from ._registry import builtin
from ._util import fnum, is_text, text

METHODS = ("linear", "nearest", "next", "previous", "pchip", "cubic", "spline", "makima")
_EXTRAP = METHODS + ("none",)
_PPOLY = ("spline", "pchip", "makima")


def _method(v: Any, extrap: bool = False) -> str:
    if not is_text(v):
        raise MatlabError("griddedInterpolant: the interpolation method must be given as text.",
                          "MATLAB:griddedInterpolant:BadMethodErrId")
    m = text(v).lower()
    allowed = _EXTRAP if extrap else METHODS
    if m not in allowed:
        what = "Extrapolation" if extrap else "Interpolation"
        raise MatlabError(f"{what} method '{text(v)}' is not supported. Use one of: "
                          + ", ".join(f"'{a}'" for a in allowed) + ".",
                          "MATLAB:griddedInterpolant:BadExtrapMethodErrId" if extrap
                          else "MATLAB:griddedInterpolant:BadMethodErrId")
    return m


def _vector(v: Any, what: str) -> np.ndarray:
    a = fnum(v)
    if np.iscomplexobj(a):
        raise MatlabError(f"griddedInterpolant: the {what} must be real.", "MATLAB:griddedInterpolant:CmplxGridErrId")
    if a.ndim > 2 or (a.ndim == 2 and min(a.shape) > 1):
        raise MatlabError(f"griddedInterpolant: the {what} must be a vector.",
                          "MATLAB:griddedInterpolant:GridVectorNotVectorErrId")
    return a.astype(float).reshape(-1)


def _check_grid(g: np.ndarray, k: int) -> None:
    if g.size < 2:
        raise MatlabError("Interpolation requires at least two sample points in each dimension.",
                          "MATLAB:griddedInterpolant:NotEnoughPtsErrId")
    if not np.all(np.isfinite(g)):
        raise MatlabError("The grid vectors must contain finite values (no NaN or Inf).",
                          "MATLAB:griddedInterpolant:NonFiniteGridErrId")
    d = np.diff(g)
    if np.any(d == 0):
        raise MatlabError(f"The grid vectors must contain unique points (grid vector {k} has repeated values).",
                          "MATLAB:griddedInterpolant:NonUniqueCompVecsErrId")
    if np.any(d < 0):
        raise MatlabError(f"The grid vectors must be strictly monotonically increasing (grid vector {k} is not).",
                          "MATLAB:griddedInterpolant:NonMonotonicCompVecsErrId")


def _core(grid: list[np.ndarray], V: np.ndarray) -> tuple[np.ndarray, tuple]:
    """The values as an (n1, ..., nd, k) array, and the shape of the values at one grid point."""
    d = len(grid)
    sizes = tuple(g.size for g in grid)
    if d == 1 and V.ndim == 2 and 1 in V.shape and V.size == sizes[0]:
        return V.reshape(sizes[0], 1), ()
    shape = V.shape
    while len(shape) < d:
        shape = shape + (1,)
    if shape[:d] != sizes or len(shape) > d + 1:
        got = "×".join(str(s) for s in V.shape)
        want = "×".join(str(s) for s in sizes)
        raise MatlabError(f"The size of the values array ({got}) does not match the grid ({want}); the values "
                          "need one element per grid point, with at most one extra trailing dimension for "
                          "several values per point.", "MATLAB:griddedInterpolant:BadValsSizeErrId")
    extra = tuple(shape[d:])
    if extra == (1,):
        extra = ()
    k = int(np.prod(extra)) if extra else 1
    return V.reshape(sizes + (k,), order="F"), extra


# -- 1-D building blocks ------------------------------------------------------------------------------

def _uniform(g: np.ndarray) -> bool:
    h = np.diff(g)
    return bool(np.all(np.abs(h - h[0]) <= 1e-10 * max(abs(h[0]), np.max(np.abs(g)))))


def _kind(method: str, grid: list[np.ndarray]) -> str:
    """'cubic' is cubic convolution on a uniform grid (3 or more points per dimension), else a spline."""
    if method == "cubic":
        if all(g.size >= 3 and _uniform(g) for g in grid):
            return "cubic"
        return "spline"
    return method


def _stencil(method: str, g: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local interpolation stencil along one axis: indices (m, s) into the (padded) axis and weights."""
    n = g.size
    if method == "linear":
        i = np.clip(np.searchsorted(g, q, side="right") - 1, 0, n - 2)
        t = (q - g[i]) / (g[i + 1] - g[i])
        return np.stack([i, i + 1], axis=1), np.stack([1 - t, t], axis=1)
    if method == "nearest":
        i = np.clip(np.searchsorted(g, q, side="right") - 1, 0, n - 2)
        up = (q - g[i]) >= (g[i + 1] - q)  # a tie goes to the larger neighbour
        j = np.where(up, i + 1, i)
        j[q < g[0]] = 0
        j[q > g[-1]] = n - 1
        return j[:, None], np.ones((q.size, 1))
    if method == "next":
        j = np.searchsorted(g, q, side="left")
        w = np.where(j >= n, np.nan, 1.0)
        return np.minimum(j, n - 1)[:, None], w[:, None]
    if method == "previous":
        j = np.searchsorted(g, q, side="right") - 1
        w = np.where(j < 0, np.nan, 1.0)
        return np.maximum(j, 0)[:, None], w[:, None]
    # cubic convolution (Keys, a = -1/2) on a uniform grid padded with one point at each end
    h = g[1] - g[0]
    i = np.clip(np.floor((q - g[0]) / h).astype(np.int64), 0, n - 2)
    i = np.where(np.isfinite(q), i, 0)
    t = (q - g[i]) / h
    t2, t3 = t * t, t * t * t
    w = np.stack([(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2],
                 axis=1)
    return np.stack([i, i + 1, i + 2, i + 3], axis=1), w  # indices into the padded axis


def _pad_cubic(C: np.ndarray, d: int) -> np.ndarray:
    """Pad each grid axis with the virtual end points of cubic convolution."""
    for ax in range(d):
        A = np.moveaxis(C, ax, 0)
        lo = 3 * A[0] - 3 * A[1] + A[2]
        hi = 3 * A[-1] - 3 * A[-2] + A[-3]
        C = np.moveaxis(np.concatenate([lo[None], A, hi[None]], axis=0), 0, ax)
    return C


def _ppoly(method: str, g: np.ndarray, A: np.ndarray, axis: int):
    from scipy.interpolate import CubicSpline, PchipInterpolator
    if method == "spline":
        return CubicSpline(g, A, axis=axis, bc_type="not-a-knot", extrapolate=True)
    if method == "pchip":
        return PchipInterpolator(g, A, axis=axis, extrapolate=True)
    from .._compat import makima
    if g.size < 3:  # too few points for Akima slopes: the line through them
        from scipy.interpolate import make_interp_spline
        from scipy.interpolate import PPoly
        return PPoly.from_spline(make_interp_spline(g, A, k=1, axis=axis), extrapolate=True)
    return makima(g, A, axis=axis, extrapolate=True)


def _rowwise(P, q: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise polynomial built along axis 1 of an (m, n, ...) array at q[i] for row i."""
    x, c = P.x, P.c  # c: (order, pieces, m, ...)
    idx = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    idx = np.where(np.isfinite(q), idx, 0)
    s = (q - x[idx]).reshape((-1,) + (1,) * (c.ndim - 3))
    cc = c[:, idx, np.arange(q.size)]  # (order, m, ...)
    out = cc[0]
    for j in range(1, cc.shape[0]):
        out = out * s + cc[j]
    return out


# -- evaluation -------------------------------------------------------------------------------------------

def _eval_points(grid, C, Q: np.ndarray, method: str) -> np.ndarray:
    """Values at the query points Q (m-by-d): an (m, k) array."""
    d = len(grid)
    m = Q.shape[0]
    kind = _kind(method, grid)
    if kind in _PPOLY:
        rest = int(np.prod(C.shape[1:]))
        step = max(1, int(2_000_000 // max(rest, 1)))
        out = np.empty((m, C.shape[-1]))
        P0 = _ppoly(kind, grid[0], C, 0)
        for s in range(0, m, step):
            q = Q[s:s + step]
            A = P0(q[:, 0])  # (chunk, n2, ..., nd, k)
            for ax in range(1, d):
                A = _rowwise(_ppoly(kind, grid[ax], A, 1), q[:, ax])
            out[s:s + step] = A
        return out
    if kind == "cubic":
        C = _pad_cubic(C, d)
    stencils = [_stencil(kind, grid[ax], Q[:, ax]) for ax in range(d)]
    out = np.zeros((m, C.shape[-1]))
    for combo in np.ndindex(*(s[0].shape[1] for s in stencils)):
        w = np.ones(m)
        idx = []
        for ax, c in enumerate(combo):
            w = w * stencils[ax][1][:, c]
            idx.append(stencils[ax][0][:, c])
        out += w[:, None] * C[tuple(idx)]
    return out


def _eval_grid(grid, C, qs: list[np.ndarray], method: str) -> np.ndarray:
    """Values on the grid of query vectors qs: an (len1, ..., lend, k) array."""
    d = len(grid)
    kind = _kind(method, grid)
    A = C
    if kind in _PPOLY:
        for ax in range(d):
            A = _ppoly(kind, grid[ax], A, ax)(qs[ax])
        return A
    if kind == "cubic":
        A = _pad_cubic(A, d)
    for ax in range(d):
        idx, w = _stencil(kind, grid[ax], qs[ax])
        W = np.zeros((qs[ax].size, A.shape[ax]))
        rows = np.repeat(np.arange(qs[ax].size), idx.shape[1])
        np.add.at(W, (rows, idx.reshape(-1)), np.nan_to_num(w.reshape(-1), nan=0.0))
        bad = np.isnan(w).any(axis=1)
        A = np.moveaxis(np.tensordot(W, A, axes=([1], [ax])), 0, ax)
        if bad.any():
            sl = [slice(None)] * A.ndim
            sl[ax] = bad
            A[tuple(sl)] = np.nan
    return A


class GriddedInterpolant(MObject):
    """F = griddedInterpolant(...); Vq = F(Xq)."""

    def __init__(self, grid: list[np.ndarray], V: np.ndarray, method: str = "linear", extrap: str | None = None):
        for k, g in enumerate(grid, 1):
            _check_grid(g, k)
        self.grid = grid
        self.V = V
        self.C, self.extra = _core(grid, V)
        self.method = method
        self.extrap = extrap or method

    def class_name(self) -> str:
        return "griddedInterpolant"

    def copy(self) -> "GriddedInterpolant":
        return GriddedInterpolant(list(self.grid), self.V, self.method, self.extrap)

    def property_names(self) -> list[str]:
        return ["GridVectors", "Values", "Method", "ExtrapolationMethod"]

    def method_names(self) -> list[str]:
        return ["griddedInterpolant"]

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, GriddedInterpolant) and other.method == self.method and \
            other.extrap == self.extrap and len(other.grid) == len(self.grid) and \
            all(np.array_equal(a, b) for a, b in zip(other.grid, self.grid)) and \
            other.V.shape == self.V.shape and np.array_equal(other.V, self.V, equal_nan=True)

    def _grid_cell(self) -> Cell:
        return Cell.from_list([g.reshape(1, -1).copy() for g in self.grid])

    def get_field(self, interp, name: str) -> Any:
        if name == "GridVectors":
            return self._grid_cell()
        if name == "Values":
            return self.V.copy()
        if name == "Method":
            return char(self.method)
        if name == "ExtrapolationMethod":
            return char(self.extrap)
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class 'griddedInterpolant'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "GriddedInterpolant":
        if name == "GridVectors":
            if not isinstance(value, Cell):
                raise MatlabError("GridVectors must be a cell array of grid vectors, one per dimension.",
                                  "MATLAB:griddedInterpolant:GridVectorsNotCellErrId")
            grid = [_vector(g, "grid vectors") for g in value.data.flatten(order="F")]
            return GriddedInterpolant(grid, self.V, self.method, self.extrap)
        if name == "Values":
            V = fnum(value)
            if np.iscomplexobj(V):
                V = V.astype(complex)
            else:
                V = V.astype(float)
            return GriddedInterpolant(self.grid, V, self.method, self.extrap)
        if name == "Method":
            new = self.copy()
            new.method = _method(value)
            return new
        if name == "ExtrapolationMethod":
            new = self.copy()
            new.extrap = _method(value, extrap=True)
            return new
        raise MatlabError(f"Unrecognized property '{name}' for class 'griddedInterpolant'.",
                          "MATLAB:noPublicFieldForClass")

    # -- evaluation ---------------------------------------------------------------------------------------
    def _values(self, fn, *a) -> np.ndarray:
        """Real and imaginary parts are interpolated separately (the methods act on real data)."""
        if np.iscomplexobj(self.C):
            return fn(self.grid, self.C.real, *a) + 1j * fn(self.grid, self.C.imag, *a)
        return fn(self.grid, self.C, *a)

    def points(self, Q: np.ndarray) -> np.ndarray:
        out = self._values(_eval_points, Q, self.method)
        if self.extrap != self.method:
            lo = np.array([g[0] for g in self.grid])
            hi = np.array([g[-1] for g in self.grid])
            outside = np.any((Q < lo) | (Q > hi), axis=1)
            if outside.any():
                if self.extrap == "none":
                    out[outside] = np.nan
                else:
                    out[outside] = self._values(_eval_points, Q[outside], self.extrap)
        out[np.isnan(Q).any(axis=1)] = np.nan
        return out

    def on_grid(self, qs: list[np.ndarray]) -> np.ndarray:
        out = self._values(_eval_grid, qs, self.method)
        masks = [(q < g[0]) | (q > g[-1]) for g, q in zip(self.grid, qs)]
        if self.extrap != self.method and any(mk.any() for mk in masks):
            outside = np.zeros(tuple(q.size for q in qs), dtype=bool)
            for ax, mk in enumerate(masks):
                shape = [1] * len(qs)
                shape[ax] = mk.size
                outside = outside | mk.reshape(shape)
            other = np.nan if self.extrap == "none" else self._values(_eval_grid, qs, self.extrap)
            out = np.where(outside[..., None], other, out)
        for ax, q in enumerate(qs):
            if np.isnan(q).any():
                sl = [slice(None)] * out.ndim
                sl[ax] = np.isnan(q)
                out[tuple(sl)] = np.nan
        return out

    def _shape_out(self, out: np.ndarray, shape: tuple) -> np.ndarray:
        """Results for queries of the given shape: that shape, then the shape of the values at a point."""
        if not self.extra:
            return out.reshape(shape, order="F")
        vec = len(shape) == 2 and 1 in shape
        lead = (int(np.prod(shape)),) if vec else shape
        return out.reshape(lead + self.extra, order="F")

    def index_get(self, interp, args: list) -> Any:
        d = len(self.grid)
        if len(args) == 1 and isinstance(args[0], Cell):
            qs = [_vector(q, "query grid vectors") for q in args[0].data.flatten(order="F")]
            if len(qs) != d:
                raise MatlabError(f"The number of query grid vectors ({len(qs)}) must match the number of "
                                  f"dimensions of the grid ({d}).", "MATLAB:griddedInterpolant:BadGridQueryErrId")
            out = self.on_grid(qs)
            if d == 1:
                q0 = fnum(args[0].data.flat[0])
                return self._shape_out(out, q0.shape if q0.size else (0, 1))
            lead = tuple(q.size for q in qs)
            return out.reshape(lead + self.extra, order="F") if self.extra else out.reshape(lead, order="F")
        qa = [fnum(a) for a in args]
        for a in qa:
            if np.iscomplexobj(a):
                raise MatlabError("The query points must be real.", "MATLAB:griddedInterpolant:CmplxQueryErrId")
        if len(qa) == 1 and d == 1:
            q = qa[0].astype(float)
            return self._shape_out(self.points(q.reshape(-1, 1, order="F")), q.shape)
        if len(qa) == 1:
            Q = qa[0].astype(float)
            if Q.ndim != 2 or Q.shape[1] != d:
                raise MatlabError(f"The query points must be a matrix with {d} columns (one point per row), or "
                                  f"give {d} coordinate arrays or a cell array of {d} grid vectors.",
                                  "MATLAB:griddedInterpolant:BadQueryDimsErrId")
            return self._shape_out(self.points(Q), (Q.shape[0], 1))
        if len(qa) != d:
            raise MatlabError(f"The interpolant has {d} dimensions, so it needs {d} query coordinate arrays "
                              f"(got {len(qa)}).", "MATLAB:griddedInterpolant:BadQueryDimsErrId")
        try:
            qa = list(np.broadcast_arrays(*[a.astype(float) for a in qa]))
        except ValueError:
            raise MatlabError("The query coordinate arrays must be the same size.",
                              "MATLAB:griddedInterpolant:QueryArraysSizeErrId") from None
        Q = np.column_stack([a.reshape(-1, order="F") for a in qa])
        return self._shape_out(self.points(Q), qa[0].shape)

    def display_lines(self) -> list[str]:
        gv = "  ".join(summary(g.reshape(1, -1), False) for g in self.grid)
        gv = "{" + gv + "}" if len(gv) < 70 else f"{{1×{len(self.grid)} cell}}"
        return ["  griddedInterpolant with properties:", "",
                f"            GridVectors: {gv}",
                f"                 Values: {summary(self.V, False)}",
                f"                 Method: '{self.method}'",
                f"    ExtrapolationMethod: '{self.extrap}'"]


def _full_grids(arrs: list[np.ndarray]) -> list[np.ndarray]:
    """Grid vectors from ndgrid-format full grids X1, ..., Xd."""
    d = len(arrs)
    shape = arrs[0].shape
    if any(a.shape != shape for a in arrs):
        raise MatlabError("The full grid arrays must all be the same size.",
                          "MATLAB:griddedInterpolant:InputSizeMismatchErrId")
    grid = []
    for k, X in enumerate(arrs):
        Xs = X.reshape(X.shape + (1,) * max(0, d - X.ndim))
        sl = [0] * Xs.ndim
        sl[k] = slice(None)
        g = Xs[tuple(sl)].astype(float).reshape(-1)
        if not np.array_equal(Xs, np.broadcast_to(g.reshape([-1 if j == k else 1 for j in range(Xs.ndim)]),
                                                  Xs.shape)):
            if d == 2:
                raise MatlabError("The grid is in meshgrid format, but griddedInterpolant needs ndgrid format. "
                                  "Make the grid with ndgrid, or transpose X1, X2 and V (permute with [2 1]).",
                                  "MATLAB:griddedInterpolant:NdgridNotMeshgrid2DErrId")
            raise MatlabError("The full grid arrays must be in ndgrid format (array k varies along dimension k "
                              "only).", "MATLAB:griddedInterpolant:NdgridNotMeshgrid3DErrId")
        grid.append(g)
    return grid


@builtin("griddedInterpolant")
def _gridded_interpolant(interp, args, nargout):
    rest = list(args)
    words = []
    while rest and is_text(rest[-1]):
        words.insert(0, rest.pop())
    if len(words) > 2:
        raise MatlabError("griddedInterpolant takes at most two text arguments: the interpolation method and "
                          "the extrapolation method.", "MATLAB:griddedInterpolant:BadMethodErrId")
    method = _method(words[0]) if words else "linear"
    extrap = _method(words[1], extrap=True) if len(words) > 1 else None
    if not rest:
        raise MatlabError("griddedInterpolant needs the sample values, e.g. griddedInterpolant(x, v).",
                          "MATLAB:narginchk:notEnoughInputs")
    V = fnum(rest[-1])
    V = V.astype(complex) if np.iscomplexobj(V) else V.astype(float)
    if V.size == 0:
        raise MatlabError("The values array must not be empty.", "MATLAB:griddedInterpolant:EmptyValsErrId")
    if len(rest) == 1:
        if V.ndim == 2 and 1 in V.shape:
            grid = [np.arange(1.0, V.size + 1)]
        else:
            grid = [np.arange(1.0, n + 1) for n in V.shape]
    elif len(rest) == 2 and isinstance(rest[0], Cell):
        grid = [_vector(g, "grid vectors") for g in rest[0].data.flatten(order="F")]
    elif len(rest) == 2:
        x = fnum(rest[0])
        if x.ndim == 2 and 1 in x.shape:
            grid = [_vector(x, "sample points")]
        else:
            raise MatlabError("For 1-D interpolation the sample points must be a vector; for N-D data give "
                              "the full grids X1, ..., Xn or a cell array of grid vectors.",
                              "MATLAB:griddedInterpolant:BadGridErrId")
    else:
        grid = _full_grids([fnum(a) for a in rest[:-1]])
    return GriddedInterpolant(grid, V, method, extrap)
