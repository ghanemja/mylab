"""triangulation and delaunayTriangulation objects.

``TR = triangulation(T, P)`` wraps a connectivity list and its points; ``DT = delaunayTriangulation(P)``
computes the Delaunay triangulation (scipy / Qhull), optionally with 2-D edge constraints (recovered by
edge flips after the unconstrained triangulation). Both answer the topological and geometric queries of
MATLAB's classes: edges, freeBoundary, neighbors, vertexAttachments, pointLocation, circumcenter and so on.
Methods are called as ``TR.edges()`` or ``edges(TR)``. Indices are 1-based outside, 0-based inside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, class_name, to_matlab
from ._registry import BUILTINS, builtin
from .spatial import _delaunay_tri, _real

_EMPTY_ROW = np.zeros((1, 0))


# -- argument helpers ---------------------------------------------------------------------------------------

def _ids(v: Any, n: int, what: str, fname: str) -> np.ndarray:
    """0-based indices from 1-based IDs (any shape, read in column order)."""
    a = _real(v, what).flatten(order="F")
    if a.size and (np.any(a != np.round(a)) or a.min() < 1 or a.max() > n):
        raise MatlabError(f"{fname}: the {what} must be integers in the range 1 to {n}.",
                          f"MATLAB:triangulation:{fname}:InvalidIDs")
    return a.astype(np.int64) - 1


def _query_points(args: list, d: int, fname: str) -> np.ndarray:
    """Query points from (QP) with d columns, or from d separate coordinate arrays."""
    if len(args) == 1:
        Q = _real(args[0], "query points")
        if Q.ndim != 2 or Q.shape[1] != d:
            raise MatlabError(f"{fname}: the query points must be a matrix with {d} columns, one point per row.",
                              f"MATLAB:triangulation:{fname}:InvalidQueryPoints")
        return Q
    if len(args) == d:
        cols = [_real(a, "query coordinates").flatten(order="F") for a in args]
        if len({c.size for c in cols}) != 1:
            raise MatlabError(f"{fname}: the query coordinate arrays must have the same number of elements.")
        return np.column_stack(cols)
    raise MatlabError(f"{fname}: give the query points as one matrix with {d} columns or as {d} coordinate "
                      "vectors.", "MATLAB:narginchk:notEnoughInputs" if not args else
                      "MATLAB:narginchk:tooManyInputs")


def _edge_args(args: list, n: int, fname: str) -> tuple[np.ndarray, np.ndarray]:
    """Edges from (E) as an m-by-2 matrix or from (V1, V2); 0-based start and end vertices."""
    if len(args) == 1:
        E = _real(args[0], "edges")
        if E.ndim != 2 or E.shape[1] != 2:
            raise MatlabError(f"{fname}: the edges must be a matrix with two columns, one edge per row.")
        a = _ids(E[:, 0], n, "vertex IDs", fname)
        b = _ids(E[:, 1], n, "vertex IDs", fname)
        return a, b
    if len(args) == 2:
        a = _ids(args[0], n, "vertex IDs", fname)
        b = _ids(args[1], n, "vertex IDs", fname)
        if a.size != b.size:
            raise MatlabError(f"{fname}: V1 and V2 must have the same number of elements.")
        return a, b
    raise MatlabError(f"{fname} expects {fname}(TR, V1, V2) or {fname}(TR, E).")


def _nargs(args: list, hi: int, fname: str) -> None:
    if len(args) > hi:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")


def _row(ids) -> np.ndarray:
    a = np.asarray(ids, dtype=float).reshape(1, -1) + 1.0
    return a if a.size else _EMPTY_ROW.copy()


# -- the object ------------------------------------------------------------------------------------------

class Triangulation(MObject):
    """A triangulation (or a delaunayTriangulation, when ``delaunay`` is set)."""

    def __init__(self, T: np.ndarray, P: np.ndarray, delaunay: bool = False, constraints: np.ndarray | None = None):
        self.T = np.asarray(T, dtype=np.int64)
        self.P = np.asarray(P, dtype=float)
        self.delaunay = delaunay
        self.C = constraints if constraints is not None and constraints.size else None

    # identity ------------------------------------------------------------------------------------------
    def class_name(self) -> str:
        return "delaunayTriangulation" if self.delaunay else "triangulation"

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), "triangulation")

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, Triangulation) and other.delaunay == self.delaunay and \
            np.array_equal(other.T, self.T) and np.array_equal(other.P, self.P) and \
            np.array_equal(_conn(other.C), _conn(self.C))

    @property
    def dim(self) -> int:
        return self.P.shape[1] if self.P.ndim == 2 else 2

    def is_surface(self) -> bool:
        return self.T.shape[1] == 3 and self.dim == 3

    # members ---------------------------------------------------------------------------------------------
    def property_names(self) -> list[str]:
        return ["Points", "ConnectivityList"] + (["Constraints"] if self.delaunay else [])

    def method_names(self) -> list[str]:
        return sorted(set(_METHODS) | (set(_DT_METHODS) if self.delaunay else set()), key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _METHODS or (self.delaunay and name in _DT_METHODS)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = _METHODS.get(name) or (_DT_METHODS.get(name) if self.delaunay else None)
        if fn is None:
            return super().call_method(interp, name, args, nargout)
        if not args or not isinstance(args[0], Triangulation):
            raise MatlabError(f"Undefined function '{name}' for input arguments of type "
                              f"'{class_name(args[0]) if args else 'double'}'.", "MATLAB:UndefinedFunction")
        res = fn(args[0], interp, list(args[1:]), nargout)
        return [to_matlab(r) for r in res] if isinstance(res, (tuple, list)) else [to_matlab(res)]

    def connectivity(self) -> np.ndarray:
        return self.T.astype(float) + 1.0

    def get_field(self, interp, name: str) -> Any:
        if name == "Points":
            return self.P.copy()
        if name == "ConnectivityList":
            return self.connectivity()
        if name == "Constraints" and self.delaunay:
            return np.zeros((0, 0)) if self.C is None else self.C.astype(float) + 1.0
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> "Triangulation":
        if name in self.property_names() and not self.delaunay or name == "ConnectivityList":
            raise MatlabError(f"Unable to set the '{name}' property of class '{self.class_name()}' because it "
                              "is read-only.", "MATLAB:class:SetProhibited")
        if name == "Points":
            P = _real(value, "points") if not (isinstance(value, np.ndarray) and value.size == 0) \
                else np.zeros((0, self.dim))
            if P.ndim != 2 or P.shape[1] not in (2, 3):
                raise MatlabError("The points of a delaunayTriangulation must be an n-by-2 or n-by-3 matrix.")
            C = self.C
            if C is not None and (P.shape[1] != 2 or C.max() >= P.shape[0]):
                C = None  # the constraints no longer refer to existing points
            return _make_dt(interp, P, C)
        if name == "Constraints":
            return _make_dt(interp, self.P, _constraint_ids(value, self.P.shape[0], self.dim))
        raise MatlabError(f"Unrecognized property '{name}' for class '{self.class_name()}'.",
                          "MATLAB:noPublicFieldForClass")

    # indexing: TR(i, j) is TR.ConnectivityList(i, j) ------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        return ix.get(self.connectivity(), args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        raise MatlabError(f"Unable to modify the connectivity of a {self.class_name()} by indexing; its "
                          "ConnectivityList is read-only.", "MATLAB:class:SetProhibited")

    def display_lines(self) -> list[str]:
        n, d = self.P.shape
        m, k = self.T.shape
        lines = [f"  {self.class_name()} with properties:", "",
                 f"              Points: [{n}×{d} double]",
                 f"    ConnectivityList: [{m}×{k} double]"]
        if self.delaunay:
            c = "[]" if self.C is None else f"[{self.C.shape[0]}×2 double]"
            lines.append(f"         Constraints: {c}")
        return lines


def _conn(C: np.ndarray | None) -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int64) if C is None else C


# -- construction -------------------------------------------------------------------------------------------

def _coords(nums: list, fname: str) -> np.ndarray:
    cols = [_real(a, "point coordinates") for a in nums]
    if len({c.size for c in cols}) != 1:
        raise MatlabError(f"{fname}: the coordinate vectors must have the same number of elements.",
                          f"MATLAB:{fname}:InconsistentCoordinates")
    return np.column_stack([c.flatten(order="F") for c in cols]) if cols[0].size else np.zeros((0, len(cols)))


@builtin("triangulation")
def _triangulation(interp, args, nargout):
    if len(args) not in (2, 3, 4):
        raise MatlabError("triangulation expects triangulation(T, P), triangulation(T, x, y) or "
                          "triangulation(T, x, y, z).",
                          "MATLAB:narginchk:notEnoughInputs" if len(args) < 2 else "MATLAB:narginchk:tooManyInputs")
    T = _real(args[0], "connectivity list")
    if len(args) == 2:
        P = _real(args[1], "points")
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise MatlabError("triangulation: the points must be a matrix with 2 or 3 columns, one point per row.",
                              "MATLAB:triangulation:InvalidPoints")
    else:
        P = _coords(list(args[1:]), "triangulation")
    if T.size == 0:
        T = np.zeros((0, 3 if P.shape[1] == 2 else 3))
    if T.ndim != 2 or T.shape[1] not in (3, 4):
        raise MatlabError("triangulation: the connectivity list must be a matrix with 3 columns (triangles) or "
                          "4 columns (tetrahedra), one per row.", "MATLAB:triangulation:InvalidConnectivity")
    if T.shape[1] == 4 and P.shape[1] != 3:
        raise MatlabError("triangulation: a tetrahedral connectivity list (4 columns) needs 3-D points.",
                          "MATLAB:triangulation:InvalidConnectivity")
    n = P.shape[0]
    if T.size and (np.any(T != np.round(T)) or T.min() < 1 or T.max() > n):
        raise MatlabError(f"triangulation: the connectivity list must contain integer vertex IDs in the range "
                          f"1 to {n} (the number of points).", "MATLAB:triangulation:InvalidConnectivity")
    if np.isnan(P).any() or np.isinf(P).any():
        raise MatlabError("triangulation: the points must not contain NaN or Inf.",
                          "MATLAB:triangulation:InvalidPoints")
    return Triangulation(T.astype(np.int64) - 1, P)


def _constraint_ids(v: Any, n: int, d: int) -> np.ndarray | None:
    C = _real(v, "constraints")
    if C.size == 0:
        return None
    if d != 2:
        raise MatlabError("delaunayTriangulation: edge constraints are supported for 2-D triangulations only.",
                          "MATLAB:delaunayTriangulation:ConsSupport2DOnly")
    if C.ndim != 2 or C.shape[1] != 2:
        raise MatlabError("delaunayTriangulation: the constraints must be a matrix with two columns, one "
                          "constrained edge (a pair of point indices) per row.",
                          "MATLAB:delaunayTriangulation:InvalidConstraints")
    if np.any(C != np.round(C)) or C.min() < 1 or C.max() > n:
        raise MatlabError(f"delaunayTriangulation: the constraints must contain integer point indices in the "
                          f"range 1 to {n}.", "MATLAB:delaunayTriangulation:InvalidConstraints")
    return C.astype(np.int64) - 1


@builtin("delaunayTriangulation")
def _delaunay_triangulation(interp, args, nargout):
    if not args:
        return Triangulation(np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2)), delaunay=True)
    if len(args) > 3:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    first = _real(args[0], "points")
    C = None
    if first.ndim == 2 and first.shape[1] in (2, 3) and first.shape[0] != 1 and len(args) <= 2:
        P = first  # (P) or (P, C)
        if len(args) == 2:
            C = args[1]
    else:
        if len(args) < 2:
            raise MatlabError("delaunayTriangulation: give the points as an n-by-2 or n-by-3 matrix, or as "
                              "separate x, y (and z) vectors.", "MATLAB:delaunayTriangulation:InvalidPoints")
        nums = list(args[:2])
        if len(args) == 3:
            third = _real(args[2], "input")
            if third.size == first.size and (third.ndim < 2 or min(third.shape) == 1 or third.shape[1] != 2):
                nums.append(args[2])  # (x, y, z)
            else:
                C = args[2]  # (x, y, C)
        P = _coords(nums, "delaunayTriangulation")
    if np.isnan(P).any() or np.isinf(P).any():
        raise MatlabError("delaunayTriangulation: the points must not contain NaN or Inf.",
                          "MATLAB:delaunayTriangulation:InvalidPoints")
    return _make_dt(interp, P, None if C is None else _constraint_ids(C, P.shape[0], P.shape[1]))


def _make_dt(interp, P: np.ndarray, C: np.ndarray | None) -> Triangulation:
    d = P.shape[1]
    if P.shape[0] > 1:
        uniq, first, inv = np.unique(P, axis=0, return_index=True, return_inverse=True)
        if uniq.shape[0] < P.shape[0]:
            interp.warn("Duplicate data points have been detected and removed - their indices have been "
                        "updated.", "MATLAB:delaunayTriangulation:DupPtsWarnId")
            order = np.argsort(first)  # keep the first occurrence of each point, in the original order
            new_pos = np.empty(order.size, dtype=np.int64)
            new_pos[order] = np.arange(order.size)
            old_to_new = new_pos[np.asarray(inv).reshape(-1)]
            P = P[np.sort(first)]
            if C is not None:
                C = old_to_new[C]
    try:
        T = _delaunay_tri(P, "delaunayTriangulation") if P.shape[0] else np.zeros((0, d + 1), dtype=np.int64)
    except MatlabError:  # too few points, or all on one line (plane): nothing to triangulate
        T = np.zeros((0, d + 1), dtype=np.int64)
    if d == 3 and T.size:  # positively oriented tetrahedra
        a, b, c, e = (P[T[:, k]] for k in range(4))
        neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), e - a) < 0
        T[neg] = T[neg][:, [0, 2, 1, 3]]
    if C is not None:
        C = C[C[:, 0] != C[:, 1]]
        C = _split_constraints(P, C)
        _check_crossings(P, C)
        if T.size:
            T = _recover_constraints(P, T, C)
    return Triangulation(T.astype(np.int64), P, delaunay=True, constraints=C)


# -- constrained Delaunay ----------------------------------------------------------------------------------------

def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Any:
    """Twice the signed area of (a, b, c): positive when counterclockwise."""
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _split_constraints(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Split constrained edges at points lying on them; drop repeated edges."""
    scale = max(float(np.max(np.abs(P))) if P.size else 1.0, 1.0)
    out = []
    for a, b in C:
        pa, pb = P[a], P[b]
        ab = pb - pa
        L2 = float(ab @ ab)
        t = (P - pa) @ ab / L2
        dist = np.abs(_orient(pa, pb, P)) / np.sqrt(L2)
        on = (dist <= 1e-12 * scale) & (t > 1e-12) & (t < 1 - 1e-12)
        on[[a, b]] = False
        chain = [a] + [int(i) for i in np.flatnonzero(on)[np.argsort(t[on])]] + [b]
        out.extend(zip(chain[:-1], chain[1:]))
    seen = set()
    keep = []
    for a, b in out:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            keep.append((a, b))
    return np.array(keep, dtype=np.int64).reshape(-1, 2)


def _check_crossings(P: np.ndarray, C: np.ndarray) -> None:
    if C.shape[0] < 2 or C.shape[0] > 4000:
        return
    a, b = P[C[:, 0]], P[C[:, 1]]
    A, B = a[:, None, :], b[:, None, :]
    c, d = a[None, :, :], b[None, :, :]
    o1, o2 = _orient(A, B, c), _orient(A, B, d)
    o3, o4 = _orient(c, d, A), _orient(c, d, B)
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    if cross.any():
        i, j = np.argwhere(cross)[0]
        raise MatlabError(f"delaunayTriangulation: the constrained edges [{C[i, 0] + 1} {C[i, 1] + 1}] and "
                          f"[{C[j, 0] + 1} {C[j, 1] + 1}] intersect; intersecting constraints are not supported.",
                          "MATLAB:delaunayTriangulation:IntersectingConstraints")


class _Mesh:
    """Mutable 2-D triangle mesh with an edge -> triangles map, for edge flips."""

    def __init__(self, P: np.ndarray, T: np.ndarray):
        self.P = P
        self.T = [list(map(int, t)) for t in T]
        self.edges: dict = {}
        for i, t in enumerate(self.T):
            for k in range(3):
                self.edges.setdefault(_key(t[k], t[(k + 1) % 3]), []).append(i)

    def quad(self, u: int, v: int):
        """For the interior edge (u, v): triangles t1 = (u, v, p) and t2 = (v, u, q), counterclockwise."""
        ts = self.edges.get(_key(u, v), [])
        if len(ts) != 2:
            return None
        t1, t2 = ts
        if not _has_directed(self.T[t1], u, v):
            t1, t2 = t2, t1
        p = next(x for x in self.T[t1] if x not in (u, v))
        q = next(x for x in self.T[t2] if x not in (u, v))
        return t1, t2, p, q

    def convex(self, u: int, v: int, p: int, q: int) -> bool:
        P = self.P
        return _orient(P[p], P[q], P[u]) * _orient(P[p], P[q], P[v]) < 0

    def flip(self, u: int, v: int, t1: int, t2: int, p: int, q: int) -> None:
        self.T[t1] = [u, q, p]
        self.T[t2] = [q, v, p]
        del self.edges[_key(u, v)]
        self.edges[_key(p, q)] = [t1, t2]
        uq = self.edges[_key(u, q)]
        uq[uq.index(t2)] = t1
        vp = self.edges[_key(v, p)]
        vp[vp.index(t1)] = t2

    def crossing_edges(self, a: int, b: int) -> list:
        P = self.P
        keys = np.array(list(self.edges.keys()), dtype=np.int64)
        u, v = P[keys[:, 0]], P[keys[:, 1]]
        pa, pb = P[a], P[b]
        hit = (_orient(pa, pb, u) * _orient(pa, pb, v) < 0) & (_orient(u, v, pa) * _orient(u, v, pb) < 0)
        return [tuple(k) for k in keys[hit].tolist()]


def _key(a: int, b: int) -> tuple:
    return (a, b) if a < b else (b, a)


def _has_directed(t: list, u: int, v: int) -> bool:
    i = t.index(u)
    return t[(i + 1) % 3] == v


def _incircle(P: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Positive when d is inside the circumcircle of the counterclockwise triangle (a, b, c)."""
    m = np.array([P[a] - P[d], P[b] - P[d], P[c] - P[d]])
    m = np.column_stack([m, (m ** 2).sum(axis=1)])
    return float(np.linalg.det(m))


def _recover_constraints(P: np.ndarray, T: np.ndarray, C: np.ndarray) -> np.ndarray:
    mesh = _Mesh(P, T)
    fixed = {_key(int(a), int(b)) for a, b in C}
    for a, b in C:
        a, b = int(a), int(b)
        if _key(a, b) in mesh.edges:
            continue
        todo = mesh.crossing_edges(a, b)
        guard = 0
        limit = 50 * (len(todo) + 1) ** 2 + 1000
        while todo:
            guard += 1
            if guard > limit:
                raise MatlabError(f"delaunayTriangulation: the constrained edge [{a + 1} {b + 1}] could not be "
                                  "inserted.", "MATLAB:delaunayTriangulation:ConstraintRecoveryFailed")
            u, v = todo.pop(0)
            quad = mesh.quad(u, v)
            if quad is None:
                continue
            t1, t2, p, q = quad
            if not mesh.convex(u, v, p, q):
                todo.append((u, v))
                continue
            mesh.flip(u, v, t1, t2, p, q)
            pa, pb = P[a], P[b]
            if p not in (a, b) and q not in (a, b) and \
                    _orient(pa, pb, P[p]) * _orient(pa, pb, P[q]) < 0:
                todo.append(_key(p, q))
    # restore the Delaunay property away from the constraints (Lawson flips)
    scale = max(float(np.ptp(P, axis=0).max()) if P.size else 1.0, 1e-300)
    tol = 1e-10 * scale ** 4
    stack = [k for k, ts in mesh.edges.items() if len(ts) == 2 and k not in fixed]
    count = 0
    limit = 100 * (len(stack) + 10) ** 1.5
    while stack and count < limit:
        count += 1
        u, v = stack.pop()
        if _key(u, v) in fixed:
            continue
        quad = mesh.quad(u, v)
        if quad is None:
            continue
        t1, t2, p, q = quad
        if _incircle(P, u, v, p, q) > tol and mesh.convex(u, v, p, q):
            mesh.flip(u, v, t1, t2, p, q)
            stack.extend([_key(u, q), _key(q, v), _key(v, p), _key(p, u)])
    return np.array(mesh.T, dtype=np.int64)


# -- shared geometry ------------------------------------------------------------------------------------------

def _sel(tr: Triangulation, args: list, fname: str, what: str = "triangle IDs") -> np.ndarray:
    """The selected simplices (0-based), all of them when no ID argument is given."""
    if not args:
        return np.arange(tr.T.shape[0])
    return _ids(args[0], tr.T.shape[0], what, fname)


def _corners(tr: Triangulation, ids: np.ndarray) -> list[np.ndarray]:
    return [tr.P[tr.T[ids, k]] for k in range(tr.T.shape[1])]


def _cross3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.shape[1] == 2:
        z = np.zeros((u.shape[0], 1))
        u, v = np.hstack([u, z]), np.hstack([v, z])
    return np.cross(u, v)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _attachments(tr: Triangulation) -> list[np.ndarray]:
    """For every point, the (ascending, 0-based) simplices that use it."""
    m, k = tr.T.shape
    flat = tr.T.reshape(-1)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=tr.P.shape[0])
    return np.split(order // k, np.cumsum(counts)[:-1]) if tr.P.shape[0] else []


def _unique_edges(tr: Triangulation) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique undirected edges, and for each edge occurrence (simplex-major) its unique row."""
    k = tr.T.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    E = tr.T[:, pairs].reshape(-1, 2) if tr.T.size else np.zeros((0, 2), dtype=np.int64)
    E = np.sort(E, axis=1)
    if not E.size:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    uniq, inv = np.unique(E, axis=0, return_inverse=True)
    return uniq, np.asarray(inv).reshape(-1)


def _need_triangles(tr: Triangulation, fname: str) -> None:
    if tr.T.shape[1] != 3:
        raise MatlabError(f"{fname} is only supported for triangulations made of triangles, not tetrahedra.",
                          f"MATLAB:triangulation:{fname}:NotTriangles")


def _need_surface(tr: Triangulation, fname: str) -> None:
    if not tr.is_surface():
        raise MatlabError(f"{fname} is only supported for 3-D surface triangulations (triangles with 3-D points).",
                          f"MATLAB:triangulation:{fname}:NotSurface")


# -- methods ---------------------------------------------------------------------------------------------------

_METHODS: dict = {}
_DT_METHODS: dict = {}


def _method(name: str, table: dict | None = None):
    def deco(fn):
        (_METHODS if table is None else table)[name] = fn
        return fn
    return deco


@_method("size")
def _m_size(tr, interp, args, nargout):
    res = BUILTINS["size"](interp, [tr.connectivity(), *args], max(nargout, 1))
    return res


@_method("edges")
def _m_edges(tr, interp, args, nargout):
    _nargs(args, 0, "edges")
    return _unique_edges(tr)[0].astype(float) + 1.0


def _chain(E: np.ndarray) -> np.ndarray:
    """Order directed boundary edges into loops (the end of each edge is the start of the next)."""
    if not E.size:
        return E
    out_of: dict = {}
    for i in np.lexsort((E[:, 1], E[:, 0])):
        out_of.setdefault(int(E[i, 0]), []).append(int(i))
    used = np.zeros(E.shape[0], dtype=bool)
    order = []
    for i in np.lexsort((E[:, 1], E[:, 0])):
        cur = int(i)
        while cur is not None and not used[cur]:
            used[cur] = True
            order.append(cur)
            cur = next((j for j in out_of.get(int(E[cur, 1]), []) if not used[j]), None)
    return E[order]


@_method("freeBoundary")
def _m_free_boundary(tr, interp, args, nargout):
    _nargs(args, 0, "freeBoundary")
    m, k = tr.T.shape
    if k == 3:
        F = tr.T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        opp = None
    else:
        faces = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        F = tr.T[:, sum(faces, [])].reshape(-1, 3)
        opp = tr.T.reshape(-1)
    if F.size:
        _, inv, counts = np.unique(np.sort(F, axis=1), axis=0, return_inverse=True, return_counts=True)
        free = counts[np.asarray(inv).reshape(-1)] == 1
        F = F[free]
        if k == 3:
            F = _chain(F)
        else:
            a, b, c = tr.P[F[:, 0]], tr.P[F[:, 1]], tr.P[F[:, 2]]
            inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), tr.P[opp[free]] - a) > 0
            F[inward] = F[inward][:, [0, 2, 1]]
    F = F.reshape(-1, k - 1)
    if nargout < 2:
        return F.astype(float) + 1.0
    used, inv = np.unique(F, return_inverse=True)
    return np.asarray(inv).reshape(F.shape).astype(float) + 1.0, tr.P[used].reshape(-1, tr.dim)


@_method("neighbors")
def _m_neighbors(tr, interp, args, nargout):
    _nargs(args, 1, "neighbors")
    ids = _sel(tr, args, "neighbors")
    m, k = tr.T.shape
    N = np.full((m, k), np.nan)
    if m:
        others = [[i for i in range(k) if i != j] for j in range(k)]
        F = np.sort(np.concatenate([tr.T[:, o] for o in others]), axis=1)  # row r: simplex r % m, vertex r // m
        _, inv = np.unique(F, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        order = np.argsort(inv, kind="stable")
        same = inv[order[1:]] == inv[order[:-1]]
        r1, r2 = order[:-1][same], order[1:][same]
        N[r1 % m, r1 // m] = r2 % m + 1
        N[r2 % m, r2 // m] = r1 % m + 1
    return N[ids]


@_method("vertexAttachments")
def _m_vertex_attachments(tr, interp, args, nargout):
    _nargs(args, 1, "vertexAttachments")
    att = _attachments(tr)
    vids = np.arange(tr.P.shape[0]) if not args else _ids(args[0], tr.P.shape[0], "vertex IDs", "vertexAttachments")
    return Cell.from_list([_row(att[v]) for v in vids], column=True)


@_method("edgeAttachments")
def _m_edge_attachments(tr, interp, args, nargout):
    a, b = _edge_args(args, tr.P.shape[0], "edgeAttachments")
    att = _attachments(tr)
    return Cell.from_list([_row(np.intersect1d(att[i], att[j])) for i, j in zip(a, b)], column=True)


@_method("isConnected")
def _m_is_connected(tr, interp, args, nargout):
    a, b = _edge_args(args, tr.P.shape[0], "isConnected")
    E = {tuple(e) for e in _unique_edges(tr)[0].tolist()}
    return np.array([[_key(int(i), int(j)) in E] for i, j in zip(a, b)], dtype=bool).reshape(-1, 1)


@_method("faceNormal")
def _m_face_normal(tr, interp, args, nargout):
    _nargs(args, 1, "faceNormal")
    _need_surface(tr, "faceNormal")
    a, b, c = _corners(tr, _sel(tr, args, "faceNormal"))
    return _unit(np.cross(b - a, c - a)).reshape(-1, 3)


@_method("vertexNormal")
def _m_vertex_normal(tr, interp, args, nargout):
    _nargs(args, 1, "vertexNormal")
    _need_surface(tr, "vertexNormal")
    a, b, c = _corners(tr, np.arange(tr.T.shape[0]))
    fn = np.cross(b - a, c - a)  # length = twice the area: an area-weighted average
    acc = np.zeros((tr.P.shape[0], 3))
    for k in range(3):
        np.add.at(acc, tr.T[:, k], fn)
    vids = np.arange(tr.P.shape[0]) if not args else _ids(args[0], tr.P.shape[0], "vertex IDs", "vertexNormal")
    return _unit(acc[vids]).reshape(-1, 3)


def _tri_incenter(a, b, c):
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (b - c, c - a, a - b))
    s = la + lb + lc
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (la[:, None] * a + lb[:, None] * b + lc[:, None] * c) / s[:, None]
        r = np.linalg.norm(_cross3(b - a, c - a), axis=1) / s
    return C, r


@_method("incenter")
def _m_incenter(tr, interp, args, nargout):
    _nargs(args, 1, "incenter")
    pts = _corners(tr, _sel(tr, args, "incenter"))
    if len(pts) == 3:
        C, r = _tri_incenter(*pts)
    else:
        a, b, c, d = pts
        areas = [np.linalg.norm(np.cross(y - x, z - x), axis=1) / 2 for x, y, z in ((b, c, d), (a, c, d), (a, b, d),
                                                                                    (a, b, c))]
        s = sum(areas)
        vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6
        with np.errstate(invalid="ignore", divide="ignore"):
            C = sum(w[:, None] * p for w, p in zip(areas, pts)) / s[:, None]
            r = 3 * vol / s
    return C.reshape(-1, tr.dim), r.reshape(-1, 1)


@_method("circumcenter")
def _m_circumcenter(tr, interp, args, nargout):
    _nargs(args, 1, "circumcenter")
    pts = _corners(tr, _sel(tr, args, "circumcenter"))
    C, r = _circumcenters(pts)
    return C.reshape(-1, tr.dim), r.reshape(-1, 1)


def _circumcenters(pts: list) -> tuple[np.ndarray, np.ndarray]:
    a = pts[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        if len(pts) == 3 and a.shape[1] == 2:
            b, c = pts[1] - a, pts[2] - a
            d = 2 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
            bb, cc = (b ** 2).sum(axis=1), (c ** 2).sum(axis=1)
            u = np.column_stack([(c[:, 1] * bb - b[:, 1] * cc) / d, (b[:, 0] * cc - c[:, 0] * bb) / d])
        elif len(pts) == 3:
            b, c = pts[1] - a, pts[2] - a
            n = np.cross(b, c)
            nn = (n ** 2).sum(axis=1)
            u = ((c ** 2).sum(axis=1)[:, None] * np.cross(n, b) + (b ** 2).sum(axis=1)[:, None] * np.cross(c, n)) \
                / (2 * nn[:, None])
        else:
            M = np.stack([p - a for p in pts[1:]], axis=1)  # m-by-3-by-3
            rhs = (M ** 2).sum(axis=2) / 2
            u = np.full(a.shape, np.nan)
            ok = np.abs(np.linalg.det(M)) > 0 if M.size else np.zeros(0, dtype=bool)
            if ok.any():
                u[ok] = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
    return a + u, np.linalg.norm(u, axis=1)


@_method("barycentricToCartesian")
def _m_bary_to_cart(tr, interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("barycentricToCartesian expects barycentricToCartesian(TR, ID, B).",
                          "MATLAB:narginchk:notEnoughInputs" if len(args) < 2 else "MATLAB:TooManyInputs")
    ids = _ids(args[0], tr.T.shape[0], "triangle IDs", "barycentricToCartesian")
    B = _real(args[1], "barycentric coordinates")
    k = tr.T.shape[1]
    if B.ndim != 2 or B.shape != (ids.size, k):
        raise MatlabError(f"barycentricToCartesian: B must have one row per ID and {k} columns.",
                          "MATLAB:triangulation:barycentricToCartesian:InvalidB")
    V = np.stack(_corners(tr, ids), axis=1)  # q-by-k-by-d
    return np.einsum("qk,qkd->qd", B, V).reshape(-1, tr.dim)


@_method("cartesianToBarycentric")
def _m_cart_to_bary(tr, interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("cartesianToBarycentric expects cartesianToBarycentric(TR, ID, C).",
                          "MATLAB:narginchk:notEnoughInputs" if len(args) < 2 else "MATLAB:TooManyInputs")
    ids = _ids(args[0], tr.T.shape[0], "triangle IDs", "cartesianToBarycentric")
    C = _real(args[1], "Cartesian coordinates")
    d, k = tr.dim, tr.T.shape[1]
    if C.ndim != 2 or C.shape != (ids.size, d):
        raise MatlabError(f"cartesianToBarycentric: C must have one row per ID and {d} columns.",
                          "MATLAB:triangulation:cartesianToBarycentric:InvalidC")
    V = np.stack(_corners(tr, ids), axis=2)  # q-by-d-by-k
    M = np.concatenate([V, np.ones((ids.size, 1, k))], axis=1)  # q-by-(d+1)-by-k
    rhs = np.concatenate([C, np.ones((ids.size, 1))], axis=1)[..., None]
    if not ids.size:
        return np.zeros((0, k))
    Minv = np.linalg.inv(M) if d + 1 == k else np.linalg.pinv(M)
    return (Minv @ rhs)[..., 0]


@_method("nearestNeighbor")
def _m_nearest_neighbor(tr, interp, args, nargout):
    from scipy.spatial import cKDTree
    Q = _query_points(args, tr.dim, "nearestNeighbor")
    verts = np.unique(tr.T) if tr.T.size else np.arange(tr.P.shape[0])
    if not verts.size:
        return np.full((Q.shape[0], 1), np.nan), np.full((Q.shape[0], 1), np.nan)
    d, i = cKDTree(tr.P[verts]).query(Q)
    return (verts[i] + 1.0).reshape(-1, 1), d.reshape(-1, 1)


@_method("pointLocation")
def _m_point_location(tr, interp, args, nargout):
    if tr.is_surface():
        raise MatlabError("pointLocation is only supported for 2-D triangulations and 3-D tetrahedral "
                          "triangulations.", "MATLAB:triangulation:pointLocation:NotSupported")
    Q = _query_points(args, tr.dim, "pointLocation")
    m, k = tr.T.shape
    q = Q.shape[0]
    ID = np.full(q, np.nan)
    B = np.full((q, k), np.nan)
    if m and q:
        V = np.stack(_corners(tr, np.arange(m)), axis=2)  # m-by-d-by-k
        M = np.concatenate([V, np.ones((m, 1, k))], axis=1)
        ok = np.abs(np.linalg.det(M)) > 1e-300
        Minv = np.full(M.shape, np.nan)
        if ok.any():
            Minv[ok] = np.linalg.inv(M[ok])
        R = np.concatenate([Q, np.ones((q, 1))], axis=1)
        scale = max(float(np.abs(tr.P).max()), 1.0)
        tol = -1e-12 * scale
        step = max(1, int(4e6 // (m * k)))
        for s in range(0, q, step):
            bary = np.einsum("mkj,qj->qmk", Minv, R[s:s + step])  # chunk-by-m-by-k
            inside = np.all(bary >= tol, axis=2)
            found = inside.any(axis=1)
            first = inside.argmax(axis=1)
            rows = np.arange(s, min(s + step, q))
            ID[rows[found]] = first[found] + 1.0
            B[rows[found]] = bary[np.flatnonzero(found), first[found]]
    if nargout >= 2:
        return ID.reshape(-1, 1), B
    return ID.reshape(-1, 1)


@_method("featureEdges")
def _m_feature_edges(tr, interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("featureEdges expects featureEdges(TR, filterangle).",
                          "MATLAB:narginchk:notEnoughInputs" if not args else "MATLAB:TooManyInputs")
    _need_triangles(tr, "featureEdges")
    angle = _real(args[0], "filter angle")
    if angle.size != 1:
        raise MatlabError("featureEdges: the filter angle must be a scalar (in radians).")
    angle = float(angle.flat[0])
    uniq, inv = _unique_edges(tr)
    if not uniq.size:
        return np.zeros((0, 2))
    m = tr.T.shape[0]
    tri = np.repeat(np.arange(m), 3)  # the triangle of each edge occurrence (simplex-major)
    a, b, c = _corners(tr, np.arange(m))
    fn = _unit(_cross3(b - a, c - a))
    counts = np.bincount(inv, minlength=uniq.shape[0])
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    feature = counts != 2
    two = np.flatnonzero(counts == 2)
    t1, t2 = tri[order[starts[two]]], tri[order[starts[two] + 1]]
    cosang = np.clip(np.einsum("ij,ij->i", fn[t1], fn[t2]), -1.0, 1.0)
    feature[two] = np.arccos(cosang) > angle
    return uniq[feature].astype(float) + 1.0


# -- delaunayTriangulation only ----------------------------------------------------------------------------------

@_method("convexHull", _DT_METHODS)
def _m_convex_hull(tr, interp, args, nargout):
    _nargs(args, 0, "convexHull")
    if not tr.T.size:
        raise MatlabError("convexHull: the triangulation is empty, so it has no convex hull.",
                          "MATLAB:delaunayTriangulation:convexHull:EmptyTri")
    res = BUILTINS["convhull"](interp, [tr.P], 2)
    return res if nargout >= 2 else res[0]


@_method("isInterior", _DT_METHODS)
def _m_is_interior(tr, interp, args, nargout):
    _nargs(args, 0, "isInterior")
    if tr.dim != 2 or tr.C is None:
        raise MatlabError("isInterior is only supported for 2-D delaunayTriangulations with edge constraints "
                          "that bound a region.", "MATLAB:delaunayTriangulation:isInterior:NoConstraints")
    m = tr.T.shape[0]
    cen = sum(_corners(tr, np.arange(m))) / 3.0
    x1, y1 = tr.P[tr.C[:, 0], 0], tr.P[tr.C[:, 0], 1]
    x2, y2 = tr.P[tr.C[:, 1], 0], tr.P[tr.C[:, 1], 1]
    inside = np.zeros(m, dtype=bool)
    step = max(1, int(4e6 // max(tr.C.shape[0], 1)))
    for s in range(0, m, step):
        px, py = cen[s:s + step, 0:1], cen[s:s + step, 1:2]
        straddle = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside[s:s + step] = np.count_nonzero(straddle & (px < xi), axis=1) % 2 == 1
    return inside.reshape(-1, 1)


@_method("voronoiDiagram", _DT_METHODS)
def _m_voronoi_diagram(tr, interp, args, nargout):
    _nargs(args, 0, "voronoiDiagram")
    if tr.C is not None:
        raise MatlabError("voronoiDiagram is not supported for a delaunayTriangulation with edge constraints.",
                          "MATLAB:delaunayTriangulation:voronoiDiagram:Constrained")
    m, k = tr.T.shape
    d = tr.dim
    cc, _ = _circumcenters(_corners(tr, np.arange(m)))
    V = np.vstack([np.full((1, d), np.inf), cc.reshape(-1, d)])
    att = _attachments(tr)
    F = _m_free_boundary(tr, interp, [], 1)
    on_hull = np.zeros(tr.P.shape[0], dtype=bool)
    on_hull[(F.astype(np.int64) - 1).reshape(-1)] = True
    cen = sum(_corners(tr, np.arange(m))) / k if m else np.zeros((0, d))
    regions = []
    for v in range(tr.P.shape[0]):
        ts = att[v]
        if not ts.size:
            regions.append(_EMPTY_ROW.copy())
            continue
        if d == 2:
            ang = np.arctan2(cen[ts, 1] - tr.P[v, 1], cen[ts, 0] - tr.P[v, 0])
            o = np.argsort(ang)
            ts, ang = ts[o], ang[o]
            if on_hull[v] and ts.size > 1:
                gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
                ts = np.roll(ts, -(int(np.argmax(gaps)) + 1))
        ids = ts + 2.0
        if on_hull[v]:
            ids = np.concatenate([[1.0], ids])
        regions.append(ids.reshape(1, -1))
    return V, Cell.from_list(regions, column=True)


# -- function forms for the method names: edges(TR), freeBoundary(TR), ... -----------------------------------------

def _dispatcher(name: str):
    def f(interp, args, nargout):
        for a in args:
            if isinstance(a, Triangulation) and a.has_method(name):
                return tuple(a.call_method(interp, name, list(args), nargout))
        kind = class_name(args[0]) if args else "double"
        need = "delaunayTriangulation" if name in _DT_METHODS else "triangulation or delaunayTriangulation"
        raise MatlabError(f"Undefined function '{name}' for input arguments of type '{kind}'. It needs a {need} "
                          "object.", "MATLAB:UndefinedFunction")
    f.__name__ = f"_tri_{name}"
    return f


for _name in sorted(set(_METHODS) | set(_DT_METHODS)):
    if _name not in BUILTINS:
        builtin(_name)(_dispatcher(_name))


def plot_args(args: list) -> list:
    """triplot(TR), trisurf(TR), trimesh(TR): replace a triangulation by its T, x, y (, z) arguments."""
    out = []
    for a in args:
        if isinstance(a, Triangulation):
            out.append(a.connectivity())
            out.extend(a.P[:, j].reshape(-1, 1).copy() for j in range(a.dim))
        else:
            out.append(a)
    return out
