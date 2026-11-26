"""polyshape objects (2-D polygons with holes and several regions) and nsidedpoly.

A polyshape value is an array of shapes (usually 1-by-1). Each shape is a list of boundaries
(closed vertex loops, stored without repeating the first vertex) and a hole flag per boundary.
Simplified shapes follow the convention that solid boundaries run clockwise and holes
counterclockwise. With the default 'SolidBoundaryOrientation' 'auto', simplification decides what is
solid by nesting: a point is inside when it is enclosed by an odd number of input boundaries; with
'cw' or 'ccw' the boundaries of that orientation are solid and the others cut holes.

Boolean operations (union, intersect, subtract, xor, the simplification itself and polybuffer) use
the edge-arrangement algorithm in :mod:`._polyclip`. Methods are called as ``area(p)`` or
``p.area()``. Indices are 1-based outside, 0-based inside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, class_name, to_matlab
from . import _polyclip as pc
from ._registry import BUILTINS, builtin
from ._util import is_text, parse_options, text

_SIMPLIFY_ID = "MATLAB:polyshape:repairedBySimplify"
_FEW_ID = "MATLAB:polyshape:boundary3Points"


# -- one shape ---------------------------------------------------------------------------------------------------

class _Shape:
    """One polygon: boundaries (n-by-2, not closed) and whether each is a hole."""

    __slots__ = ("bounds", "hole", "simplified")

    def __init__(self, bounds: list[np.ndarray] | None = None, hole: list[bool] | None = None,
                 simplified: bool = True):
        self.bounds = [np.asarray(b, dtype=float).reshape(-1, 2) for b in (bounds or [])]
        self.hole = list(hole) if hole is not None else [pc.signed_area(b) > 0 for b in self.bounds]
        self.simplified = simplified

    def copy(self) -> "_Shape":
        return _Shape([b.copy() for b in self.bounds], list(self.hole), self.simplified)

    def vertices(self) -> np.ndarray:
        if not self.bounds:
            return np.zeros((0, 2))
        parts = []
        for k, b in enumerate(self.bounds):
            if k:
                parts.append(np.full((1, 2), np.nan))
            parts.append(b)
        return np.vstack(parts)

    def role_oriented(self) -> list[np.ndarray]:
        """Boundaries with solid ones counterclockwise and holes clockwise (the engine's convention)."""
        out = []
        for b, h in zip(self.bounds, self.hole):
            out.append(b[::-1] if (pc.signed_area(b) > 0) == h else b)
        return out

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        bs = self.role_oriented()
        if not bs:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.vstack(bs), np.vstack([np.roll(b, -1, axis=0) for b in bs])

    def area(self) -> float:
        return float(sum(-abs(pc.signed_area(b)) if h else abs(pc.signed_area(b))
                         for b, h in zip(self.bounds, self.hole)))

    def regions(self) -> list[tuple[int, list[int]]]:
        """(solid boundary, its holes) for each region."""
        solids = [k for k, h in enumerate(self.hole) if not h]
        holes = [k for k, h in enumerate(self.hole) if h]
        owner: dict[int, list[int]] = {k: [] for k in solids}
        areas = {k: abs(pc.signed_area(self.bounds[k])) for k in solids}
        for h in holes:
            pt = _inner_point(self.bounds[h])
            best = None
            for s in solids:
                b = self.bounds[s]
                if pc.winding(pt[None], b, np.roll(b, -1, axis=0))[0] != 0 and \
                        (best is None or areas[s] < areas[best]):
                    best = s
            if best is not None:
                owner[best].append(h)
        return [(s, owner[s]) for s in solids]


def _inner_point(b: np.ndarray) -> np.ndarray:
    """The midpoint of the longest edge of a boundary (a point of it away from its vertices)."""
    nxt = np.roll(b, -1, axis=0)
    k = int(np.argmax(np.sum((nxt - b) ** 2, axis=1)))
    return 0.5 * (b[k] + nxt[k])


def _scale(shapes: list[_Shape]) -> float:
    pts = [b for s in shapes for b in s.bounds]
    return pc.scale_of(np.vstack(pts)) if pts else 1.0


def _cw(b: np.ndarray) -> np.ndarray:
    """The boundary clockwise, keeping its first vertex first."""
    if pc.signed_area(b) > 0:
        return np.vstack([b[:1], b[:0:-1]])
    return b


def _ccw(b: np.ndarray) -> np.ndarray:
    if pc.signed_area(b) < 0:
        return np.vstack([b[:1], b[:0:-1]])
    return b


def _same_loop(a: np.ndarray, b: np.ndarray) -> bool:
    """True when a and b are the same closed loop (any start, either direction)."""
    if a.shape != b.shape or not a.size:
        return False
    hits = np.flatnonzero(np.all(a == b[0], axis=1))
    for k in hits:
        r = np.roll(a, -k, axis=0)
        if np.array_equal(r, b) or np.array_equal(np.vstack([r[:1], r[:0:-1]]), b):
            return True
    return False


def _from_loops(loops: list[np.ndarray], keep_collinear: bool, inputs: list[np.ndarray] | None = None
                ) -> tuple[_Shape, bool]:
    """A simplified shape from engine loops (inside on the left). Returns (shape, all inputs kept)."""
    items = []
    for L in loops:
        if not keep_collinear:
            L = pc.remove_collinear(L)
        if L.shape[0] < 3 or pc.signed_area(L) == 0:
            continue
        items.append((L, pc.signed_area(L) < 0))
    if inputs is not None:
        refs = [pc.remove_collinear(b) if not keep_collinear else b for b in inputs]
        used = [False] * len(refs)
        keyed = []
        all_kept = True
        for L, is_hole in items:
            match = None
            for k, r in enumerate(refs):
                if not used[k] and _same_loop(L, r):
                    match = k
                    break
            if match is None:
                all_kept = False
                keyed.append((len(refs) + len(keyed), L, is_hole))
            else:
                used[match] = True
                keyed.append((match, refs[match], is_hole))
        all_kept = all_kept and all(used)
        keyed.sort(key=lambda t: t[0])
        bounds = [(_ccw(b) if h else _cw(b)) for _, b, h in keyed]
        return _Shape(bounds, [h for _, _, h in keyed], True), all_kept
    # regions in the order found, each solid boundary followed by its holes
    shape = _Shape([(_ccw(L) if h else _cw(L)) for L, h in items], [h for _, h in items], True)
    order = []
    for s, hs in shape.regions():
        order.append(s)
        order.extend(hs)
    order += [k for k in range(len(shape.bounds)) if k not in order]
    return _Shape([shape.bounds[k] for k in order], [shape.hole[k] for k in order], True), True


def _odd(w: np.ndarray) -> np.ndarray:
    return np.mod(w[:, 0], 2) != 0


def _build(interp, bounds: list[np.ndarray], simplify: bool, keep_collinear: bool, orientation: str = "auto",
           warn: bool = True) -> _Shape:
    """A shape from raw boundaries (NaN-free, possibly with repeated vertices)."""
    modified = False
    clean = []
    for b in bounds:
        if b.shape[0] >= 2 and np.all(b[-1] == b[0]):
            b = b[:-1]
        c = pc.dedupe(b)
        if c.shape[0] != b.shape[0]:
            modified = True
        clean.append(c)
    if not simplify:
        kept = [b for b in clean if b.shape[0] >= 1]
        return _Shape(kept, _roles(kept, orientation), simplified=False)
    few = [b for b in clean if b.shape[0] < 3]
    clean = [b for b in clean if b.shape[0] >= 3]
    if few and warn and interp is not None:
        interp.warn("A boundary with fewer than 3 distinct vertices was removed from the polyshape.", _FEW_ID)
    if not clean:
        return _Shape()
    if orientation == "auto":
        loops = pc.overlay([clean], _odd)
    else:
        # solid boundaries add one to the winding number and holes take one away
        oriented = clean if orientation == "ccw" else [b[::-1] for b in clean]
        loops = pc.overlay([oriented], lambda w: w[:, 0] > 0)
    shape, kept = _from_loops(loops, keep_collinear, clean)
    if (modified or not kept) and warn and interp is not None:
        interp.warn("The polyshape had repeated vertices, crossing or overlapping boundaries, or boundaries "
                    "that do not bound anything; its boundaries were repaired to make a well-defined polygon.",
                    _SIMPLIFY_ID)
    return shape


def _roles(bounds: list[np.ndarray], orientation: str) -> list[bool]:
    """Hole flags of unsimplified boundaries."""
    if orientation == "cw":
        return [pc.signed_area(b) > 0 for b in bounds]
    if orientation == "ccw":
        return [pc.signed_area(b) < 0 for b in bounds]
    out = []
    for k, b in enumerate(bounds):  # nesting depth: odd depth is a hole
        if b.shape[0] < 3:
            out.append(False)
            continue
        pt = _inner_point(b)
        depth = 0
        for j, c in enumerate(bounds):
            if j != k and c.shape[0] >= 3 and pc.winding(pt[None], c, np.roll(c, -1, axis=0))[0] != 0:
                depth += 1
        out.append(depth % 2 == 1)
    return out


def _is_simplified(shape: _Shape) -> bool:
    if shape.simplified:
        return True
    bounds = [b for b in shape.bounds]
    if any(b.shape[0] < 3 or np.any(np.all(b == np.roll(b, -1, axis=0), axis=1)) for b in bounds):
        return False
    loops = pc.overlay([shape.role_oriented()], lambda w: w[:, 0] > 0)
    s2, kept = _from_loops(loops, True, bounds)
    if not kept:
        return False
    return all(np.array_equal(a, b) and h1 == h2 for a, b, h1, h2 in zip(s2.bounds, bounds, s2.hole, shape.hole))


# -- the object ------------------------------------------------------------------------------------------------

class PolyShape(MObject):
    """An array of polyshapes (1-by-1 for a single polygon)."""

    def __init__(self, data: np.ndarray | _Shape | None = None):
        if data is None:
            data = _Shape()
        if isinstance(data, _Shape):
            arr = np.empty((1, 1), dtype=object)
            arr[0, 0] = data
            data = arr
        self.data = data

    # identity ---------------------------------------------------------------------------------------------
    def class_name(self) -> str:
        return "polyshape"

    def isa_class(self, name: str) -> bool:
        return name == "polyshape"

    def matlab_shape(self) -> tuple:
        return tuple(self.data.shape)

    def shapes(self) -> list[_Shape]:
        return list(self.data.flatten(order="F"))

    def is_scalar(self) -> bool:
        return self.data.size == 1

    def one(self, fname: str) -> _Shape:
        if self.data.size != 1:
            raise MatlabError(f"{fname}: the polyshape must be a scalar (1-by-1), not an array of "
                              f"{self.data.size}.", "MATLAB:polyshape:scalarPolyshapeError")
        return self.data.flat[0]

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, PolyShape) and other.data.shape == self.data.shape and all(
            _shape_equal(a, b) for a, b in zip(self.shapes(), other.shapes()))

    def empty_like(self) -> "PolyShape":
        return PolyShape(np.empty((0, 0), dtype=object))

    # members ------------------------------------------------------------------------------------------
    def property_names(self) -> list[str]:
        return ["Vertices", "NumRegions", "NumHoles"]

    def method_names(self) -> list[str]:
        return sorted(_METHODS, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = _METHODS.get(name)
        if fn is None:
            return super().call_method(interp, name, args, nargout)
        res = fn(interp, list(args), nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, (tuple, list)) else [to_matlab(res)]

    def _prop(self, shape: _Shape, name: str) -> Any:
        if name == "Vertices":
            return shape.vertices()
        if name == "NumRegions":
            return np.array([[float(sum(1 for h in shape.hole if not h))]])
        if name == "NumHoles":
            return np.array([[float(sum(1 for h in shape.hole if h))]])
        raise KeyError(name)

    def field_list(self, interp, name: str) -> list:
        if name in self.property_names():
            return [self._prop(s, name) for s in self.shapes()]
        if self.has_method(name):
            return self.call_method(interp, name, [self], 1)
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class 'polyshape'.",
                          "MATLAB:noSuchMethodOrField")

    def get_field(self, interp, name: str) -> Any:
        vals = self.field_list(interp, name)
        if len(vals) != 1:
            raise MatlabError(f"Reading '{name}' from an array of {self.data.size} polyshapes gives "
                              f"{len(vals)} values; index one element first, e.g. p(1).{name}.")
        return vals[0]

    def set_field(self, interp, name: str, value: Any) -> "PolyShape":
        if name in ("NumRegions", "NumHoles"):
            raise MatlabError(f"Unable to set the '{name}' property of class 'polyshape' because it is "
                              "read-only.", "MATLAB:class:SetProhibited")
        if name != "Vertices":
            raise MatlabError(f"Unrecognized property '{name}' for class 'polyshape'.",
                              "MATLAB:noPublicFieldForClass")
        self.one("Vertices")
        V = _real(value, "vertices")
        if V.size == 0:
            return PolyShape(_Shape())
        if V.ndim != 2 or V.shape[1] != 2:
            raise MatlabError("The vertices of a polyshape must be an N-by-2 matrix.",
                              "MATLAB:polyshape:vertexMatrixError")
        return PolyShape(_build(interp, _split_nan(V[:, 0], V[:, 1]), True, False))

    # indexing: an array of polyshapes ---------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        return PolyShape(ix.get(self.data, list(args)))

    def index_set(self, interp, args: list, value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.size == 0:
            return self.index_delete(interp, args)
        if not isinstance(value, PolyShape):
            raise MatlabError(f"Unable to store a value of class '{class_name(value)}' in a polyshape array.",
                              "MATLAB:invalidConversion")
        return PolyShape(ix.set_(self.data, list(args), value.data, _Shape))

    def index_delete(self, interp, args: list) -> Any:
        return PolyShape(ix.delete(self.data, list(args)))

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        parts = []
        for it in items:
            if isinstance(it, PolyShape):
                parts.append(it.data)
            elif isinstance(it, np.ndarray) and it.size == 0:
                continue
            else:
                raise MatlabError(f"A polyshape cannot be concatenated with a value of class '{class_name(it)}'.",
                                  "MATLAB:polyshape:concatenation")
        if not parts:
            return PolyShape(np.empty((0, 0), dtype=object))
        parts = [p for p in parts if p.size] or parts[:1]
        try:
            return PolyShape(np.concatenate(parts, axis=axis))
        except ValueError:
            raise MatlabError("Dimensions of arrays being concatenated are not consistent.",
                              "MATLAB:catenate:dimensionMismatch") from None

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return PolyShape(self.data.T.copy())
        return NotImplemented

    # display ------------------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        return "×".join(str(d) for d in self.data.shape) + " polyshape"

    def display_lines(self) -> list[str]:
        if self.data.size == 1:
            s = self.data.flat[0]
            n = s.vertices().shape[0]
            return ["  polyshape with properties:", "",
                    f"      Vertices: [{n}×2 double]",
                    f"    NumRegions: {sum(1 for h in s.hole if not h)}",
                    f"      NumHoles: {sum(1 for h in s.hole if h)}"]
        dims = "×".join(str(d) for d in self.data.shape)
        if self.data.size == 0:
            return [f"  {dims} empty polyshape array with properties:", "", "    Vertices", "    NumRegions",
                    "    NumHoles"]
        return [f"  {dims} polyshape array with properties:", "", "    Vertices", "    NumRegions", "    NumHoles"]


def _shape_equal(a: _Shape, b: _Shape) -> bool:
    if len(a.bounds) != len(b.bounds) or a.hole != b.hole:
        return False
    return all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a.bounds, b.bounds))


# -- argument helpers ---------------------------------------------------------------------------------------------

def _real(v: Any, what: str) -> np.ndarray:
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return v.astype(float)
    raise MatlabError(f"The {what} must be numeric.", "MATLAB:polyshape:xyValuesError")


def _split_nan(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    x = x.reshape(-1)
    y = y.reshape(-1)
    if np.isinf(x).any() or np.isinf(y).any():
        raise MatlabError("The vertices of a polyshape must be finite (NaN may only separate boundaries).",
                          "MATLAB:polyshape:xyValuesError")
    gap = np.isnan(x) | np.isnan(y)
    if np.any(np.isnan(x) != np.isnan(y)):
        raise MatlabError("The NaN separators in x and y must be in the same places.",
                          "MATLAB:polyshape:xyNaNMismatch")
    out = []
    idx = np.flatnonzero(gap)
    starts = np.concatenate([[0], idx + 1])
    ends = np.concatenate([idx, [x.size]])
    for s, e in zip(starts, ends):
        if e > s:
            out.append(np.column_stack([x[s:e], y[s:e]]))
    return out


def _xy_bounds(args: list, fname: str) -> tuple[list[np.ndarray], list]:
    """Boundaries from (P), (x, y) or ({x1, x2}, {y1, y2}); returns them and the remaining arguments."""
    if not args:
        return [], []
    a = args[0]
    if isinstance(a, Cell):
        if len(args) < 2 or not isinstance(args[1], Cell):
            raise MatlabError(f"{fname}: give the x and the y coordinates as two cell arrays of the same size.",
                              "MATLAB:polyshape:cellInputError")
        xs, ys = a.values_f(), args[1].values_f()
        if len(xs) != len(ys):
            raise MatlabError(f"{fname}: the cell arrays of x and y coordinates must have the same number of "
                              "elements.", "MATLAB:polyshape:cellInputError")
        bounds = []
        for x, y in zip(xs, ys):
            x, y = _real(x, "x coordinates"), _real(y, "y coordinates")
            if x.size != y.size:
                raise MatlabError(f"{fname}: each x vector must have as many elements as the matching y vector.",
                                  "MATLAB:polyshape:xyMismatch")
            bounds += _split_nan(x, y)
        return bounds, list(args[2:])
    P = _real(a, "vertices")
    if len(args) >= 2 and not is_text(args[1]):
        y = _real(args[1], "y coordinates")
        if P.size != y.size:
            raise MatlabError(f"{fname}: x and y must have the same number of elements.",
                              "MATLAB:polyshape:xyMismatch")
        return _split_nan(P, y), list(args[2:])
    if P.size == 0:
        return [], list(args[1:])
    if P.ndim != 2 or P.shape[1] != 2:
        raise MatlabError(f"{fname}: a matrix of vertices must have two columns (x and y).",
                          "MATLAB:polyshape:vertexMatrixError")
    return _split_nan(P[:, 0], P[:, 1]), list(args[1:])


def _flag(v: Any, name: str) -> bool:
    if is_text(v):
        s = text(v).lower()
        if s in ("on", "true"):
            return True
        if s in ("off", "false"):
            return False
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "biuf":
        return bool(v.flat[0])
    raise MatlabError(f"The value of '{name}' must be true or false.", "MATLAB:polyshape:logicalValueError")


def _build_opts(rest: list, fname: str, orientation: bool = True) -> tuple[bool, bool, str]:
    names = ["Simplify", "KeepCollinearPoints"] + (["SolidBoundaryOrientation"] if orientation else [])
    opts = parse_options(rest, names, fname)
    simplify = _flag(opts["Simplify"], "Simplify") if "Simplify" in opts else True
    keep = _flag(opts["KeepCollinearPoints"], "KeepCollinearPoints") if "KeepCollinearPoints" in opts else False
    orient = "auto"
    if "SolidBoundaryOrientation" in opts:
        orient = text(opts["SolidBoundaryOrientation"]).lower()
        if orient not in ("auto", "cw", "ccw"):
            raise MatlabError("SolidBoundaryOrientation must be 'auto', 'cw' or 'ccw'.",
                              "MATLAB:polyshape:orientationError")
    return simplify, keep, orient


def _keep_opt(rest: list, fname: str) -> bool:
    opts = parse_options(rest, ["KeepCollinearPoints"], fname)
    return _flag(opts["KeepCollinearPoints"], "KeepCollinearPoints") if opts else False


# -- construction --------------------------------------------------------------------------------------------------

@builtin("polyshape")
def _polyshape(interp, args, nargout):
    if not args:
        return PolyShape(_Shape())
    if isinstance(args[0], PolyShape) and len(args) == 1:
        return args[0]
    bounds, rest = _xy_bounds(list(args), "polyshape")
    simplify, keep, orient = _build_opts(rest, "polyshape")
    return PolyShape(_build(interp, bounds, simplify, keep, orient))


@builtin("nsidedpoly")
def _nsidedpoly(interp, args, nargout):
    if not args:
        raise MatlabError("nsidedpoly needs the number of sides.", "MATLAB:narginchk:notEnoughInputs")
    n = _real(args[0], "number of sides")
    if n.size != 1 or n.flat[0] != round(float(n.flat[0])) or n.flat[0] < 3:
        raise MatlabError("nsidedpoly: the number of sides must be an integer of at least 3.",
                          "MATLAB:nsidedpoly:invalidN")
    n = int(n.flat[0])
    opts = parse_options(list(args[1:]), ["Center", "SideLength", "Radius"], "nsidedpoly")
    if "SideLength" in opts and "Radius" in opts:
        raise MatlabError("nsidedpoly: give either 'SideLength' or 'Radius', not both.",
                          "MATLAB:nsidedpoly:sideLengthAndRadius")
    c = _real(opts.get("Center", np.zeros((1, 2))), "center").reshape(-1)
    if c.size != 2:
        raise MatlabError("nsidedpoly: the center must be a two-element vector [x y].")
    r = 1.0
    if "Radius" in opts:
        r = float(_real(opts["Radius"], "radius").reshape(-1)[0])
    elif "SideLength" in opts:
        r = float(_real(opts["SideLength"], "side length").reshape(-1)[0]) / (2 * np.sin(np.pi / n))
    if not r > 0:
        raise MatlabError("nsidedpoly: the radius or side length must be positive.")
    # one side flat at the bottom; vertices clockwise starting at its left end
    k = np.arange(n)
    ang = -np.pi / 2 - np.pi / n - 2 * np.pi * k / n
    V = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
    V[np.abs(V) < 1e-15 * r] = 0.0
    return PolyShape(_Shape([V], [False], True))


# -- methods ------------------------------------------------------------------------------------------------------

_METHODS: dict = {}


def _method(name: str):
    def deco(fn):
        _METHODS[name] = fn
        return fn
    return deco


def _ps(v: Any, fname: str) -> PolyShape:
    if not isinstance(v, PolyShape):
        raise MatlabError(f"{fname}: expected a polyshape, but the input is of class '{class_name(v)}'.",
                          "MATLAB:polyshape:polyshapeInput")
    return v


def _nargs(args: list, lo: int, hi: int, fname: str) -> None:
    if len(args) < lo:
        raise MatlabError("Not enough input arguments.", "MATLAB:narginchk:notEnoughInputs")
    if len(args) > hi:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")


def _bound_ids(v: Any, shape: _Shape, fname: str) -> np.ndarray:
    a = _real(v, "boundary indices").reshape(-1)
    n = len(shape.bounds)
    if a.size and (np.any(a != np.round(a)) or a.min() < 1 or a.max() > n):
        raise MatlabError(f"{fname}: the boundary index must be an integer from 1 to {n} (the number of "
                          "boundaries).", "MATLAB:polyshape:indexOutOfRange")
    return a.astype(np.int64) - 1


def _per(p: PolyShape, f) -> np.ndarray:
    return np.array([f(s) for s in p.shapes()], dtype=float).reshape(p.data.shape, order="F") \
        if p.data.size else np.zeros(p.data.shape)


def _map(p: PolyShape, f) -> PolyShape:
    out = np.empty(p.data.shape, dtype=object)
    for idx in np.ndindex(p.data.shape):
        out[idx] = f(p.data[idx])
    return PolyShape(out)


def _perimeter_of(b: np.ndarray) -> float:
    return float(np.sum(np.hypot(*(np.roll(b, -1, axis=0) - b).T))) if b.shape[0] > 1 else 0.0


@_method("area")
def _m_area(interp, args, nargout):
    _nargs(args, 1, 2, "area")
    p = _ps(args[0], "area")
    if len(args) == 2:
        s = p.one("area")
        ids = _bound_ids(args[1], s, "area")
        return np.array([abs(pc.signed_area(s.bounds[k])) for k in ids]).reshape(-1, 1) if ids.size != 1 \
            else np.array([[abs(pc.signed_area(s.bounds[ids[0]]))]])
    return _per(p, lambda s: s.area())


@_method("perimeter")
def _m_perimeter(interp, args, nargout):
    _nargs(args, 1, 2, "perimeter")
    p = _ps(args[0], "perimeter")
    if len(args) == 2:
        s = p.one("perimeter")
        ids = _bound_ids(args[1], s, "perimeter")
        return np.array([_perimeter_of(s.bounds[k]) for k in ids]).reshape(-1, 1)
    return _per(p, lambda s: sum(_perimeter_of(b) for b in s.bounds))


def _centroid_of(bounds: list[np.ndarray], signs: list[float]) -> tuple[float, float]:
    A = cx = cy = 0.0
    for b, sg in zip(bounds, signs):
        if b.shape[0] < 3:
            continue
        x0, y0 = b[:, 0].mean(), b[:, 1].mean()
        x, y = b[:, 0] - x0, b[:, 1] - y0
        xs, ys = np.roll(x, -1), np.roll(y, -1)
        c = x * ys - xs * y
        a = 0.5 * c.sum()
        if a == 0:
            continue
        w = sg * (1 if a > 0 else -1)
        A += w * a
        cx += w * (np.sum((x + xs) * c) / 6 + x0 * a)
        cy += w * (np.sum((y + ys) * c) / 6 + y0 * a)
    if A == 0:
        return np.nan, np.nan
    return cx / A, cy / A


@_method("centroid")
def _m_centroid(interp, args, nargout):
    _nargs(args, 1, 2, "centroid")
    p = _ps(args[0], "centroid")
    if len(args) == 2:
        s = p.one("centroid")
        ids = _bound_ids(args[1], s, "centroid")
        c = np.array([_centroid_of([s.bounds[k]], [1.0]) for k in ids]).reshape(-1, 2)
        return c[:, :1], c[:, 1:]
    xs, ys = [], []
    for s in p.shapes():
        x, y = _centroid_of(s.bounds, [-1.0 if h else 1.0 for h in s.hole])
        xs.append(x)
        ys.append(y)
    shp = p.data.shape
    return np.array(xs).reshape(shp, order="F"), np.array(ys).reshape(shp, order="F")


@_method("boundingbox")
def _m_boundingbox(interp, args, nargout):
    _nargs(args, 1, 2, "boundingbox")
    p = _ps(args[0], "boundingbox")
    if len(args) == 2:
        s = p.one("boundingbox")
        pts = [s.bounds[k] for k in _bound_ids(args[1], s, "boundingbox")]
    else:
        pts = [b for s in p.shapes() for b in s.bounds]
    if not pts:
        return np.zeros((0, 0)), np.zeros((0, 0))
    P = np.vstack(pts)
    return np.array([[P[:, 0].min(), P[:, 0].max()]]), np.array([[P[:, 1].min(), P[:, 1].max()]])


def _query(args: list, fname: str) -> np.ndarray:
    if len(args) == 1:
        Q = _real(args[0], "query points")
        if Q.ndim != 2 or Q.shape[1] != 2:
            raise MatlabError(f"{fname}: the query points must be an N-by-2 matrix.",
                              "MATLAB:polyshape:queryPointsError")
        return Q
    if len(args) == 2:
        x, y = _real(args[0], "x coordinates"), _real(args[1], "y coordinates")
        if x.size != y.size:
            raise MatlabError(f"{fname}: x and y must have the same number of elements.",
                              "MATLAB:polyshape:xyMismatch")
        return np.column_stack([x.reshape(-1, order="F"), y.reshape(-1, order="F")])
    raise MatlabError(f"{fname} expects {fname}(p, x, y) or {fname}(p, [x y]).",
                      "MATLAB:narginchk:notEnoughInputs" if not args else "MATLAB:TooManyInputs")


def _inside(s: _Shape, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A, B = s.edges()
    tol = 1e-10 * _scale([s]) if s.bounds else 0.0
    on = pc.point_segment_distance(Q, A, B) <= tol
    inn = (pc.winding(Q, A, B) > 0) | on
    return inn, on


@_method("isinterior")
def _m_isinterior(interp, args, nargout):
    p = _ps(args[0], "isinterior")
    s = p.one("isinterior")
    Q = _query(list(args[1:]), "isinterior")
    inn, on = _inside(s, Q)
    return inn.reshape(-1, 1), on.reshape(-1, 1)


@_method("numboundaries")
def _m_numboundaries(interp, args, nargout):
    _nargs(args, 1, 1, "numboundaries")
    return _per(_ps(args[0], "numboundaries"), lambda s: len(s.bounds))


@_method("numsides")
def _m_numsides(interp, args, nargout):
    _nargs(args, 1, 2, "numsides")
    p = _ps(args[0], "numsides")
    if len(args) == 2:
        s = p.one("numsides")
        return np.array([float(s.bounds[k].shape[0]) for k in _bound_ids(args[1], s, "numsides")]).reshape(-1, 1)
    return _per(p, lambda s: sum(b.shape[0] for b in s.bounds))


@_method("ishole")
def _m_ishole(interp, args, nargout):
    _nargs(args, 1, 2, "ishole")
    s = _ps(args[0], "ishole").one("ishole")
    ids = _bound_ids(args[1], s, "ishole") if len(args) == 2 else np.arange(len(s.bounds))
    return np.array([s.hole[k] for k in ids], dtype=bool).reshape(-1, 1)


@_method("issimplified")
def _m_issimplified(interp, args, nargout):
    _nargs(args, 1, 1, "issimplified")
    p = _ps(args[0], "issimplified")
    return np.array([_is_simplified(s) for s in p.shapes()], dtype=bool).reshape(p.data.shape, order="F")


@_method("nearestvertex")
def _m_nearestvertex(interp, args, nargout):
    s = _ps(args[0], "nearestvertex").one("nearestvertex")
    Q = _query(list(args[1:]), "nearestvertex")
    n = Q.shape[0]
    if not s.bounds:
        e = np.full((n, 1), np.nan)
        return e, e.copy(), e.copy()
    V = s.vertices()
    row = np.flatnonzero(~np.isnan(V[:, 0]))
    bid = np.concatenate([np.full(b.shape[0], k + 1.0) for k, b in enumerate(s.bounds)])
    vix = np.concatenate([np.arange(1.0, b.shape[0] + 1) for b in s.bounds])
    from scipy.spatial import cKDTree
    _, i = cKDTree(V[row]).query(Q)
    return (row[i] + 1.0).reshape(-1, 1), bid[i].reshape(-1, 1), vix[i].reshape(-1, 1)


def _boundaries_touch(a: _Shape, b: _Shape, tol: float) -> bool:
    A1, B1 = a.edges()
    A2, B2 = b.edges()
    if not A1.size or not A2.size:
        return False
    if np.any(pc.point_segment_distance(A1, A2, B2) <= tol) or np.any(pc.point_segment_distance(A2, A1, B1) <= tol):
        return True
    seg, _, _ = pc._splits(np.vstack([A1, A2]), np.vstack([B1, B2]), tol)
    return bool(np.any(seg < A1.shape[0])) and bool(np.any(seg >= A1.shape[0]))


def _overlap(a: _Shape, b: _Shape) -> bool:
    if not a.bounds or not b.bounds:
        return False
    tol = 1e-10 * _scale([a, b])
    if _boolean([a, b], "intersect").area() > 0:
        return True
    return _boundaries_touch(a, b, tol)


@_method("overlaps")
def _m_overlaps(interp, args, nargout):
    _nargs(args, 1, 2, "overlaps")
    if len(args) == 1:
        sh = _ps(args[0], "overlaps").shapes()
        n = len(sh)
        out = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i, n):
                out[i, j] = out[j, i] = _overlap(sh[i], sh[j]) if i != j else bool(sh[i].bounds)
        return out
    a, b = _ps(args[0], "overlaps").shapes(), _ps(args[1], "overlaps").shapes()
    out = np.array([[_overlap(x, y) for y in b] for x in a], dtype=bool)
    return out.reshape(len(a), len(b))


@_method("boundary")
def _m_boundary(interp, args, nargout):
    _nargs(args, 1, 2, "boundary")
    s = _ps(args[0], "boundary").one("boundary")
    ids = _bound_ids(args[1], s, "boundary") if len(args) == 2 else np.arange(len(s.bounds))
    parts = []
    for j, k in enumerate(ids):
        if j:
            parts.append(np.full((1, 2), np.nan))
        b = s.bounds[k]
        parts.append(np.vstack([b, b[:1]]))
    P = np.vstack(parts) if parts else np.zeros((0, 2))
    return P[:, :1], P[:, 1:]


@_method("regions")
def _m_regions(interp, args, nargout):
    _nargs(args, 1, 1, "regions")
    s = _ps(args[0], "regions").one("regions")
    out = [_Shape([s.bounds[r]] + [s.bounds[h] for h in hs], [False] + [True] * len(hs), s.simplified)
           for r, hs in s.regions()]
    return _column(out)


def _column(shapes: list[_Shape]) -> PolyShape:
    data = np.empty((len(shapes), 1), dtype=object)
    for k, sh in enumerate(shapes):
        data[k, 0] = sh
    return PolyShape(data)


@_method("holes")
def _m_holes(interp, args, nargout):
    _nargs(args, 1, 1, "holes")
    s = _ps(args[0], "holes").one("holes")
    return _column([_Shape([_cw(b)], [False], True) for b, h in zip(s.bounds, s.hole) if h])


@_method("rmholes")
def _m_rmholes(interp, args, nargout):
    _nargs(args, 1, 1, "rmholes")
    return _map(_ps(args[0], "rmholes"), lambda s: _Shape([b for b, h in zip(s.bounds, s.hole) if not h],
                                                           [False] * sum(1 for h in s.hole if not h), s.simplified))


@_method("addboundary")
def _m_addboundary(interp, args, nargout):
    p = _ps(args[0], "addboundary")
    s = p.one("addboundary")
    new, rest = _xy_bounds(list(args[1:]), "addboundary")
    simplify, keep, orient = _build_opts(rest, "addboundary")
    return PolyShape(_build(interp, [b.copy() for b in s.bounds] + new, simplify, keep, orient))


@_method("rmboundary")
def _m_rmboundary(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("rmboundary expects rmboundary(p, I).", "MATLAB:narginchk:notEnoughInputs")
    s = _ps(args[0], "rmboundary").one("rmboundary")
    ids = set(_bound_ids(args[1], s, "rmboundary").tolist())
    simplify, keep, orient = _build_opts(list(args[2:]), "rmboundary", orientation=False)
    rest = [b for k, b in enumerate(s.bounds) if k not in ids]
    if not simplify:
        return PolyShape(_Shape(rest, [h for k, h in enumerate(s.hole) if k not in ids], False))
    return PolyShape(_resimplify(interp, rest, [h for k, h in enumerate(s.hole) if k not in ids], keep))


def _resimplify(interp, bounds: list[np.ndarray], hole: list[bool], keep: bool) -> _Shape:
    """Simplify boundaries whose roles are known (solid or hole)."""
    if not bounds:
        return _Shape()
    s = _Shape(bounds, hole, False)
    loops = pc.overlay([s.role_oriented()], lambda w: w[:, 0] > 0)
    return _from_loops(loops, keep, [b for b in bounds if b.shape[0] >= 3])[0]


@_method("rmslivers")
def _m_rmslivers(interp, args, nargout):
    _nargs(args, 2, 2, "rmslivers")
    p = _ps(args[0], "rmslivers")
    tol = _real(args[1], "tolerance")
    if tol.size != 1 or not tol.flat[0] >= 0:
        raise MatlabError("rmslivers: the tolerance must be a nonnegative scalar.")
    t = float(tol.flat[0])

    def fix(s: _Shape) -> _Shape:
        out, holes = [], []
        for b, h in zip(s.bounds, s.hole):
            b = b.copy()
            changed = True
            while changed and b.shape[0] > 3:
                changed = False
                prev, nxt = np.roll(b, 1, axis=0), np.roll(b, -1, axis=0)
                d = nxt - prev
                L = np.hypot(d[:, 0], d[:, 1])
                L[L == 0] = 1.0
                dist = np.abs(d[:, 0] * (b - prev)[:, 1] - d[:, 1] * (b - prev)[:, 0]) / L
                k = int(np.argmin(dist))
                if dist[k] <= t:
                    b = np.delete(b, k, axis=0)
                    changed = True
            if b.shape[0] >= 3 and abs(pc.signed_area(b)) > t * _perimeter_of(b) / 2:
                out.append(b)
                holes.append(h)
        return _resimplify(interp, out, holes, False)
    return _map(p, fix)


@_method("sortboundaries")
def _m_sortboundaries(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("sortboundaries expects sortboundaries(p, criterion, direction).",
                          "MATLAB:narginchk:notEnoughInputs")
    s = _ps(args[0], "sortboundaries").one("sortboundaries")
    crit, direction = text(args[1]).lower(), text(args[2]).lower()
    opts = parse_options(list(args[3:]), ["ReferencePoint"], "sortboundaries")
    if direction not in ("ascend", "descend"):
        raise MatlabError("sortboundaries: the direction must be 'ascend' or 'descend'.")
    if crit == "area":
        key = [abs(pc.signed_area(b)) for b in s.bounds]
    elif crit == "perimeter":
        key = [_perimeter_of(b) for b in s.bounds]
    elif crit == "numsides":
        key = [float(b.shape[0]) for b in s.bounds]
    elif crit == "centroid":
        ref = _real(opts.get("ReferencePoint", np.zeros((1, 2))), "reference point").reshape(-1)
        key = [float(np.hypot(*(np.array(_centroid_of([b], [1.0])) - ref))) for b in s.bounds]
    else:
        raise MatlabError("sortboundaries: the criterion must be 'area', 'perimeter', 'numsides' or 'centroid'.")
    order = sorted(range(len(key)), key=lambda k: key[k], reverse=direction == "descend")
    out = _Shape([s.bounds[k] for k in order], [s.hole[k] for k in order], s.simplified)
    return PolyShape(out), (np.array(order, dtype=float) + 1).reshape(-1, 1)


@_method("simplify")
def _m_simplify(interp, args, nargout):
    p = _ps(args[0], "simplify")
    keep = _keep_opt(list(args[1:]), "simplify")

    def fix(s: _Shape) -> _Shape:
        if s.simplified and not keep:
            return _Shape([pc.remove_collinear(b) for b in s.bounds], list(s.hole), True)
        if s.simplified:
            return s.copy()
        return _build(interp, [b.copy() for b in s.bounds], True, keep, warn=False)
    return _map(p, fix)


# -- transforms -----------------------------------------------------------------------------------------------------

def _vec2(args: list, fname: str, what: str) -> np.ndarray:
    if len(args) == 1:
        v = _real(args[0], what).reshape(-1)
        if v.size != 2:
            raise MatlabError(f"{fname}: the {what} must be a two-element vector.")
        return v
    if len(args) == 2:
        return np.array([float(_real(args[0], what).reshape(-1)[0]), float(_real(args[1], what).reshape(-1)[0])])
    raise MatlabError(f"{fname}: wrong number of input arguments.", "MATLAB:narginchk:notEnoughInputs")


def _affine(p: PolyShape, f, flips: bool) -> PolyShape:
    def g(s: _Shape) -> _Shape:
        bs = [f(b) for b in s.bounds]
        if flips:  # a reflection reverses the orientation; restore clockwise solids
            bs = [np.vstack([b[:1], b[:0:-1]]) for b in bs]
        return _Shape(bs, list(s.hole), s.simplified)
    return _map(p, g)


@_method("translate")
def _m_translate(interp, args, nargout):
    p = _ps(args[0], "translate")
    v = _vec2(list(args[1:]), "translate", "translation")
    return _affine(p, lambda b: b + v, False)


@_method("rotate")
def _m_rotate(interp, args, nargout):
    _nargs(args, 2, 3, "rotate")
    p = _ps(args[0], "rotate")
    th = np.deg2rad(float(_real(args[1], "angle").reshape(-1)[0]))
    ref = _vec2([args[2]], "rotate", "reference point") if len(args) == 3 else np.zeros(2)
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_], [s_, c]])
    return _affine(p, lambda b: (b - ref) @ R.T + ref, False)


@_method("scale")
def _m_scale(interp, args, nargout):
    _nargs(args, 2, 3, "scale")
    p = _ps(args[0], "scale")
    f = _real(args[1], "scale factor").reshape(-1)
    if f.size == 1:
        f = np.array([f[0], f[0]])
    if f.size != 2 or np.any(f == 0):
        raise MatlabError("scale: the scale factor must be a nonzero scalar or a two-element vector.")
    ref = _vec2([args[2]], "scale", "reference point") if len(args) == 3 else np.zeros(2)
    return _affine(p, lambda b: (b - ref) * f + ref, f[0] * f[1] < 0)


# -- set operations -------------------------------------------------------------------------------------------------

_OPS = {
    "union": lambda a, b: a | b,
    "intersect": lambda a, b: a & b,
    "subtract": lambda a, b: a & ~b,
    "xor": lambda a, b: a ^ b,
}


def _boolean(shapes: list[_Shape], op: str, keep: bool = False) -> _Shape:
    groups = [s.role_oriented() for s in shapes]
    if len(groups) == 1:
        loops = pc.overlay(groups, lambda w: w[:, 0] > 0)
    elif op in ("union", "intersect"):
        f = np.any if op == "union" else np.all
        loops = pc.overlay(groups, lambda w: f(w > 0, axis=1))
    else:
        g = _OPS[op]
        loops = pc.overlay(groups, lambda w: g(w[:, 0] > 0, w[:, 1] > 0))
    return _from_loops(loops, keep)[0]


def _ids_out(result: _Shape, inputs: list[_Shape]) -> tuple[np.ndarray, np.ndarray]:
    """For each row of result.Vertices: which input it came from (0 = new) and its row there."""
    V = result.vertices()
    sid = np.full((V.shape[0], 1), np.nan)
    vid = np.full((V.shape[0], 1), np.nan)
    lookups = []
    for s in inputs:
        W = s.vertices()
        lookups.append({(float(x), float(y)): r for r, (x, y) in enumerate(W) if not np.isnan(x)})
    for r, (x, y) in enumerate(V):
        if np.isnan(x):
            continue
        sid[r, 0], vid[r, 0] = 0, 0
        for k, lk in enumerate(lookups):
            j = lk.get((float(x), float(y)))
            if j is not None:
                sid[r, 0], vid[r, 0] = k + 1, j + 1
                break
    return sid, vid


def _setop(op: str):
    def fn(interp, args, nargout):
        polys = [a for a in args if isinstance(a, PolyShape)]
        rest = [a for a in args if not isinstance(a, PolyShape)]
        if op == "intersect" and len(polys) == 1 and rest and not is_text(rest[0]):
            return _intersect_lines(polys[0].one("intersect"), rest)
        if not polys or (args and not isinstance(args[0], PolyShape)):
            raise MatlabError(f"{op}: the inputs must be polyshape objects.", "MATLAB:polyshape:polyshapeInput")
        keep = _keep_opt(rest, op)
        if len(polys) == 1:
            if op in ("subtract", "xor"):
                raise MatlabError(f"{op} needs two polyshapes.", "MATLAB:narginchk:notEnoughInputs")
            shapes = polys[0].shapes()
        elif len(polys) == 2:
            if op in ("subtract", "xor"):
                shapes = [polys[0].one(op), polys[1].one(op)]
            else:
                shapes = polys[0].shapes() + polys[1].shapes()
        else:
            raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
        if not shapes:
            return PolyShape(_Shape())
        if op == "intersect" and len(shapes) > 2:
            res = shapes[0]
            for s in shapes[1:]:
                res = _boolean([res, s], "intersect", keep)
        else:
            res = _boolean(shapes, op, keep)
        if nargout >= 2:
            sid, vid = _ids_out(res, shapes)
            return PolyShape(res), sid, vid
        return PolyShape(res)
    return fn


for _op in _OPS:
    _METHODS[_op] = _setop(_op)


def _intersect_lines(s: _Shape, rest: list):
    """[in, out] = intersect(p, lineseg): the parts of a polyline inside and outside the polygon."""
    L = _real(rest[0], "line segments")
    if L.ndim != 2 or L.shape[1] != 2:
        raise MatlabError("intersect: the line segments must be an N-by-2 matrix of points.",
                          "MATLAB:polyshape:lineInputError")
    if len(rest) > 1:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    pieces_in, pieces_out = [], []
    tol = 1e-10 * max(_scale([s]), pc.scale_of(L[~np.isnan(L).any(axis=1)]) if L.size else 1.0)
    for line in _split_nan(L[:, 0], L[:, 1]) if L.size else []:
        pts = pc.crossings_with_polyline(s.role_oriented(), line, tol)
        if pts.shape[0] < 2:
            continue
        mid = 0.5 * (pts[:-1] + pts[1:])
        inside = _inside(s, mid)[0] if s.bounds else np.zeros(mid.shape[0], dtype=bool)
        start = 0
        for k in range(1, inside.size + 1):
            if k == inside.size or inside[k] != inside[start]:
                (pieces_in if inside[start] else pieces_out).append(pts[start:k + 1])
                start = k

    def join(parts):
        if not parts:
            return np.zeros((0, 2))
        out = []
        for k, q in enumerate(parts):
            if k:
                out.append(np.full((1, 2), np.nan))
            out.append(q)
        return np.vstack(out)
    return join(pieces_in), join(pieces_out)


@_method("convhull")
def _m_convhull(interp, args, nargout):
    p = _ps(args[0], "convhull")
    keep = _keep_opt(list(args[1:]), "convhull")

    def hull(s: _Shape) -> _Shape:
        if not s.bounds:
            return _Shape()
        P = np.unique(np.vstack(s.bounds), axis=0)
        if P.shape[0] < 3:
            return _Shape()
        from scipy.spatial import ConvexHull, QhullError
        try:
            h = ConvexHull(P)
        except QhullError:
            return _Shape()
        H = P[h.vertices]  # counterclockwise
        if keep:
            on = pc.point_segment_distance(P, H, np.roll(H, -1, axis=0)) <= 1e-10 * pc.scale_of(P)
            loops = pc.overlay([[H]], lambda w: w[:, 0] > 0) if not on.any() else [H]
            H = loops[0] if loops else H
        return _Shape([_cw(H)], [False], True)
    return _map(p, hull)


def _circle(c: np.ndarray, r: float, n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def _stroke(segs: list[tuple[np.ndarray, np.ndarray]], verts: np.ndarray, d: float, joint: str, miter: float,
            corners: list, n: int = 64) -> list[np.ndarray]:
    """Pieces whose union is the band of width d around the segments (each counterclockwise)."""
    parts = []
    for a, b in segs:
        t = b - a
        L = np.hypot(*t)
        if L == 0:
            continue
        nrm = np.array([-t[1], t[0]]) / L * d
        parts.append(np.array([a - nrm, b - nrm, b + nrm, a + nrm]))
    if joint == "round":
        parts += [_circle(v, d, n) for v in verts]
    else:
        parts += [c for c in (_corner_piece(v, u1, u2, n1, n2, d, joint, miter)
                              for v, u1, u2, n1, n2 in corners) if c is not None]
    return [q if pc.signed_area(q) > 0 else q[::-1] for q in parts]


def _corner_piece(v, u1, u2, n1, n2, d, joint, miter):
    """The piece that fills the outer side of a corner with a square or mitered joint."""
    bis = n1 + n2
    nb = np.hypot(*bis)
    if nb < 1e-12:
        return None
    bis = bis / nb
    cos_half = float(np.dot(bis, n1))
    if cos_half <= 1e-12:
        return None
    t1, t2 = v + d * n1, v + d * n2
    if joint == "miter" and 1.0 / cos_half <= miter:
        return np.array([v, t1, v + d * bis / cos_half, t2])
    far = v + d * bis  # square joint: cut across the bisector at distance d from the corner
    k1, k2 = float(np.dot(u1, bis)), float(np.dot(u2, bis))
    if abs(k1) < 1e-12 or abs(k2) < 1e-12:
        return None
    q1 = t1 + u1 * float(np.dot(far - t1, bis)) / k1
    q2 = t2 + u2 * float(np.dot(far - t2, bis)) / k2
    return np.array([v, t1, q1, q2, t2])


def _corners(lines: list[np.ndarray], closed: bool) -> list:
    """(vertex, incoming and outgoing unit directions, their normals on the convex side) for each corner."""
    out = []
    for b in lines:
        n = b.shape[0]
        ks = range(n) if closed else range(1, n - 1)
        for k in ks:
            p0, v, p1 = b[k - 1], b[k], b[(k + 1) % n]
            u1, u2 = _unit(v - p0), _unit(p1 - v)
            turn = u1[0] * u2[1] - u1[1] * u2[0]
            if abs(turn) < 1e-12:
                continue
            n1, n2 = np.array([u1[1], -u1[0]]), np.array([u2[1], -u2[0]])  # right-hand normals
            if turn < 0:  # a right turn: the convex side is on the left
                n1, n2 = -n1, -n2
            out.append((v, u1, u2, n1, n2))
    return out


def _unit(t: np.ndarray) -> np.ndarray:
    L = np.hypot(*t)
    return t / L if L else np.zeros(2)


@_method("polybuffer")
def _m_polybuffer(interp, args, nargout):
    return _polybuffer(interp, args[0], list(args[1:]))


def _buffer_opts(rest: list) -> tuple[str, float]:
    opts = parse_options(rest, ["JointType", "MiterLimit"], "polybuffer")
    joint = text(opts["JointType"]).lower() if "JointType" in opts else "round"
    if joint not in ("round", "square", "miter"):
        raise MatlabError("polybuffer: JointType must be 'round', 'square' or 'miter'.",
                          "MATLAB:polyshape:jointTypeError")
    miter = float(_real(opts["MiterLimit"], "miter limit").reshape(-1)[0]) if "MiterLimit" in opts else 3.0
    if miter < 2:
        raise MatlabError("polybuffer: MiterLimit must be at least 2.", "MATLAB:polyshape:miterLimitError")
    return joint, miter


def _polybuffer(interp, p, rest: list):
    if isinstance(p, PolyShape):
        if not rest:
            raise MatlabError("polybuffer expects polybuffer(p, d).", "MATLAB:narginchk:notEnoughInputs")
        d = float(_real(rest[0], "buffer distance").reshape(-1)[0])
        joint, miter = _buffer_opts(rest[1:])

        def buf(s: _Shape) -> _Shape:
            if not s.bounds or d == 0:
                return s.copy()
            bounds = s.role_oriented()
            segs = [(b[k], b[(k + 1) % b.shape[0]]) for b in bounds for k in range(b.shape[0])]
            pieces = _stroke(segs, np.vstack(bounds), abs(d), joint, miter, _corners(bounds, True))
            band = _boolean([_Shape(pieces, [False] * len(pieces), False)], "union")
            return _boolean([s, band], "union" if d > 0 else "subtract")
        return _map(p, buf)
    P = _real(p, "points")
    if len(rest) < 2 or not is_text(rest[0]):
        raise MatlabError("polybuffer expects polybuffer(P, 'lines', d) or polybuffer(P, 'points', d) for a "
                          "matrix of points.", "MATLAB:narginchk:notEnoughInputs")
    kind = text(rest[0]).lower()
    d = float(_real(rest[1], "buffer distance").reshape(-1)[0])
    joint, miter = _buffer_opts(rest[2:])
    if P.ndim != 2 or P.shape[1] != 2:
        raise MatlabError("polybuffer: the points must be an N-by-2 matrix.")
    if not d > 0:
        raise MatlabError("polybuffer: the buffer distance for lines and points must be positive.")
    if kind == "points":
        disks = [_circle(v, d, 64) for v in P[~np.isnan(P).any(axis=1)]]
        return PolyShape(_boolean([_Shape(disks, [False] * len(disks), False)], "union"))
    if kind != "lines":
        raise MatlabError(f"polybuffer: unknown buffer type '{kind}'; use 'lines' or 'points'.")
    lines = [pc.dedupe_open(b) for b in _split_nan(P[:, 0], P[:, 1])]
    segs = [(b[k], b[k + 1]) for b in lines for k in range(b.shape[0] - 1)]
    verts = np.vstack(lines) if lines else np.zeros((0, 2))
    pieces = _stroke(segs, verts, d, joint, miter, _corners(lines, False))
    if joint != "round":  # square ends, reaching d beyond the end points
        for b in lines:
            if b.shape[0] < 2:
                continue
            for a, c in ((b[0], b[1]), (b[-1], b[-2])):
                t = _unit(a - c)
                nrm = np.array([-t[1], t[0]]) * d
                cap = np.array([a - nrm, a - nrm + d * t, a + nrm + d * t, a + nrm])
                pieces.append(cap if pc.signed_area(cap) > 0 else cap[::-1])
    return PolyShape(_boolean([_Shape(pieces, [False] * len(pieces), False)], "union"))


@_method("triangulation")
def _m_triangulation(interp, args, nargout):
    _nargs(args, 1, 1, "triangulation")
    s = _ps(args[0], "triangulation").one("triangulation")
    from .triangulation import Triangulation, _make_dt
    if not s.bounds:
        return Triangulation(np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2)))
    V = np.vstack(s.bounds)
    P, inv = np.unique(V, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    first = np.full(P.shape[0], V.shape[0])
    np.minimum.at(first, inv, np.arange(V.shape[0]))
    order = np.argsort(first)  # keep the vertices in the order they appear
    renum = np.empty_like(order)
    renum[order] = np.arange(order.size)
    P = P[order]
    ids = renum[inv]
    C, start = [], 0
    for b in s.bounds:
        n = b.shape[0]
        k = ids[start:start + n]
        C += [(k[i], k[(i + 1) % n]) for i in range(n) if k[i] != k[(i + 1) % n]]
        start += n
    dt = _make_dt(interp, P, np.array(C, dtype=np.int64).reshape(-1, 2))
    T = dt.T
    if T.size:
        cen = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3
        T = T[_inside(s, cen)[0]]
    return Triangulation(T, P)


@_method("turningdist")
def _m_turningdist(interp, args, nargout):
    _nargs(args, 2, 2, "turningdist")
    a = _ps(args[0], "turningdist").shapes()
    b = _ps(args[1], "turningdist").shapes()
    out = np.array([[_turning_distance(x, y) for y in b] for x in a]).reshape(len(a), len(b))
    return out


def _turning(b: np.ndarray):
    """Turning function of a closed boundary: breakpoints (normalized arc length) and angles."""
    b = _ccw(pc.remove_collinear(b))
    e = np.roll(b, -1, axis=0) - b
    L = np.hypot(e[:, 0], e[:, 1])
    ang = np.unwrap(np.arctan2(e[:, 1], e[:, 0]))
    s = np.concatenate([[0], np.cumsum(L)]) / L.sum()
    return s, ang


def _turning_distance(x: _Shape, y: _Shape) -> float:
    """L2 distance of turning functions (Arkin et al.), minimized over the starting point of the second."""
    if len(x.bounds) != 1 or len(y.bounds) != 1:
        raise MatlabError("turningdist: each polyshape must have exactly one boundary.",
                          "MATLAB:polyshape:turningdistBoundaries")
    s1, a1 = _turning(x.bounds[0])
    s2, a2 = _turning(y.bounds[0])
    grid = np.linspace(0, 1, 2049)[:-1] + 0.5 / 2048

    def f(s, a, t):
        k = np.searchsorted(s, np.mod(t, 1.0), side="right") - 1
        return a[np.clip(k, 0, a.size - 1)] + 2 * np.pi * np.floor(t)

    best = np.inf
    g1 = f(s1, a1, grid)
    for k in range(s2.size - 1):
        g2 = f(s2, a2, grid + s2[k]) - a2[k]
        diff = g1 - a1[0] - g2
        theta = -diff.mean()
        best = min(best, float(np.sqrt(np.mean((diff + theta) ** 2))))
    return best


@_method("isequal")
def _m_isequal(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    first = args[0]
    return np.array([[all(isinstance(first, PolyShape) and first.object_equals(b) for b in args[1:])]])


@_method("plot")
def _m_plot(interp, args, nargout):
    from .plotting import _with_axes_arg
    return _with_axes_arg(_plot_core)(interp, list(args), nargout)


def _plot_core(interp, args, nargout):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    from . import plotting as P
    shapes = [a for a in args if isinstance(a, PolyShape)]
    if len(shapes) != 1 or not isinstance(args[0], PolyShape):
        raise MatlabError("plot: give one polyshape (or array of polyshapes) followed by Name,Value options.",
                          "MATLAB:polyshape:plotInput")
    p = shapes[0]
    rest, opts = P._split_name_values(list(args[1:]), {"facecolor", "edgecolor", "facealpha", "linewidth",
                                                       "linestyle", "displayname", "edgealpha"}, "patch")
    for a in rest:
        if is_text(a):
            raise MatlabError(f"Unrecognized property '{text(a)}' for class 'Polygon'.",
                              "MATLAB:class:InvalidProperty")
        raise MatlabError("plot: expected Name,Value options after the polyshape.", "MATLAB:polyshape:plotInput")
    ax = P._axes(interp, projection="rectilinear")
    handles = []
    for s in p.shapes():
        verts, codes = [], []
        for b in s.role_oriented():
            verts += b.tolist() + [b[0].tolist()]
            codes += [Path.MOVETO] + [Path.LINETO] * (b.shape[0] - 1) + [Path.CLOSEPOLY]
        if not verts:
            verts, codes = [[np.nan, np.nan]], [Path.MOVETO]
        color = _next_color(ax)
        kw: dict = {"facecolor": color, "edgecolor": color, "alpha": None, "linewidth": 0.5}
        patch = PathPatch(Path(np.array(verts, dtype=float), codes), **kw)
        from matplotlib.colors import to_rgba
        patch.set_facecolor(to_rgba(color, 0.35))
        patch.set_edgecolor(to_rgba(color, 1.0))
        ax.add_patch(patch)
        h = P._wrap(interp, "polygon", patch)
        h.extra.update({"axes": ax, "figure": ax.figure, "Shape": p.__class__(s.copy())})
        for key, name in (("facecolor", "FaceColor"), ("edgecolor", "EdgeColor"), ("facealpha", "FaceAlpha"),
                          ("linewidth", "LineWidth"), ("linestyle", "LineStyle"), ("displayname", "DisplayName"),
                          ("edgealpha", "EdgeAlpha")):
            if key in opts:
                P._set_prop(h, name, opts[key])
        P._apply_extra(interp, [h], opts, "polygon")
        handles.append(h)
    ax.autoscale_view()
    P._redraw(interp)
    if nargout <= 0:
        return None
    if len(handles) == 1:
        return handles[0]
    return P.GraphicsArray(handles, row=p.data.shape[0] == 1)


def _next_color(ax) -> Any:
    try:
        return ax._get_lines.get_next_color()
    except Exception:  # noqa: BLE001 - older matplotlib
        return "#0072BD"


def _polygon_alpha(h):
    from matplotlib.colors import to_rgba
    return np.array([[to_rgba(h.obj.get_facecolor())[3]]])


def _polygon_alpha_set(h, v):
    from matplotlib.colors import to_rgba
    from . import plotting as P
    a = P._scalar(v)
    h.obj.set_facecolor((*to_rgba(h.obj.get_facecolor())[:3], a))


def _polygon_color_set(which: str):
    def f(h, v):
        from matplotlib.colors import to_rgba
        from . import plotting as P
        if is_text(v) and text(v).lower() == "none":
            getattr(h.obj, f"set_{which}color")("none")
            return
        c = P._color(v)
        alpha = to_rgba(getattr(h.obj, f"get_{which}color")())[3]
        getattr(h.obj, f"set_{which}color")((*to_rgba(c)[:3], alpha if alpha > 0 else (0.35 if which == "face"
                                                                                        else 1.0)))
    return f


def _polygon_color_get(which: str):
    def f(h):
        from matplotlib.colors import to_rgba
        c = getattr(h.obj, f"get_{which}color")()
        rgba = to_rgba(c)
        return np.array([rgba[:3]])
    return f


def _register_graphics() -> None:
    from . import plotting as P
    P._CLASS["polygon"] = "matlab.graphics.primitive.Polygon"
    P._TYPE["polygon"] = "polygon"
    props = dict(P._PATCH_PROPS)
    props.update({
        "FaceColor": (_polygon_color_get("face"), _polygon_color_set("face")),
        "EdgeColor": (_polygon_color_get("edge"), _polygon_color_set("edge")),
        "FaceAlpha": (_polygon_alpha, _polygon_alpha_set),
        "Shape": (lambda h: h.extra["Shape"], None),
    })
    P._PROPS["polygon"] = props


_register_graphics()


# -- function forms for the method names that are not built-ins of their own ------------------------------------------

def _dispatcher(name: str):
    def f(interp, args, nargout):
        for a in args:
            if isinstance(a, PolyShape):
                return tuple(a.call_method(interp, name, list(args), nargout))
        kind = class_name(args[0]) if args else "double"
        raise MatlabError(f"Undefined function '{name}' for input arguments of type '{kind}'. It needs a polyshape.",
                          "MATLAB:UndefinedFunction")
    f.__name__ = f"_polyshape_{name}"
    return f


@builtin("polybuffer")
def _polybuffer_builtin(interp, args, nargout):
    if not args:
        raise MatlabError("polybuffer expects polybuffer(p, d) or polybuffer(P, 'lines' | 'points', d).",
                          "MATLAB:narginchk:notEnoughInputs")
    return _polybuffer(interp, args[0], list(args[1:]))


# simplify (symbolic) and triangulation have built-ins of their own in modules imported after this one
for _name in sorted(set(_METHODS) - {"simplify", "triangulation"}):
    if _name not in BUILTINS:
        builtin(_name)(_dispatcher(_name))
