"""Feature points: detectors, point objects, descriptors, matching and match display.

The detectors run OpenCV (optional dependency) except detectSURFFeatures, which uses a Hessian-
determinant blob detector and SURF-style descriptor written here in NumPy/SciPy (OpenCV's own SURF
lives in its non-free module). Point objects store 1-based [x y] locations; OpenCV keypoints are
0-based and are converted at the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import indexing as ix
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, char, class_name, to_matlab
from ._cv import (
    col, cv2mod, feature2d, fmat, fscalar, fvec, gray_float, gray_u8, image_arg, opt_text, options,
    rgb_u8, scalar,
)
from ._registry import builtin
from ._util import is_text, text

# -- point objects ----------------------------------------------------------------------------------------

_POINT_CLASSES = ("cornerPoints", "ORBPoints", "SIFTPoints", "SURFPoints", "BRISKPoints", "KAZEPoints")
_HAS_SCALE = {"ORBPoints", "SIFTPoints", "SURFPoints", "BRISKPoints", "KAZEPoints"}
_DEFAULT_SCALE = {"ORBPoints": 1.0, "SIFTPoints": 1.6, "SURFPoints": 1.6, "BRISKPoints": 12.0, "KAZEPoints": 1.6}


class FeaturePoints(MObject):
    """An M-by-1 array of feature points of one detector type (cornerPoints, SURFPoints, ...)."""

    def __init__(self, kind: str, loc, metric=None, scale=None, orientation=None, extra=None):
        self.kind = kind
        self.loc = np.asarray(loc, dtype=float).reshape(-1, 2)
        n = self.loc.shape[0]

        def fill(v, default):
            if v is None:
                return np.full(n, default, dtype=float)
            a = np.asarray(v, dtype=float).reshape(-1)
            return np.full(n, a[0]) if a.size == 1 and n != 1 else a

        self.metric = fill(metric, 0.0)
        self.scale = fill(scale, _DEFAULT_SCALE.get(kind, 1.0))
        self.orientation = fill(orientation, 0.0)
        # hidden data: sign of the Laplacian (SURF), OpenCV size and octave (to rebuild keypoints)
        self.extra = {k: np.asarray(v).reshape(-1) for k, v in (extra or {}).items()}
        if kind == "SURFPoints" and "laplacian" not in self.extra:
            self.extra["laplacian"] = np.zeros(n, dtype=np.int8)
        for name, a in (("Metric", self.metric), ("Scale", self.scale), ("Orientation", self.orientation)):
            if a.size != n:
                raise MatlabError(f"{kind}: {name} must have one value per point (or be a scalar).",
                                  f"mylab:{kind}:sizeMismatch")

    # identity ------------------------------------------------------------------------------------------
    def class_name(self) -> str:
        return self.kind

    def isa_class(self, name: str) -> bool:
        return name == self.kind

    def matlab_shape(self) -> tuple:
        return (self.loc.shape[0], 1)

    @property
    def count(self) -> int:
        return self.loc.shape[0]

    def point_xy(self) -> np.ndarray:
        return self.loc.copy()

    def subset(self, idx) -> "FeaturePoints":
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return FeaturePoints(self.kind, self.loc[idx], self.metric[idx], self.scale[idx], self.orientation[idx],
                             {k: v[idx] for k, v in self.extra.items()})

    def object_equals(self, other: Any) -> bool:
        return isinstance(other, FeaturePoints) and other.kind == self.kind and \
            np.array_equal(other.loc, self.loc) and np.array_equal(other.metric, self.metric)

    # members ---------------------------------------------------------------------------------------------
    def property_names(self) -> list[str]:
        names = ["Location", "Metric", "Count"]
        if self.kind in _HAS_SCALE:
            names += ["Scale", "Orientation"]
        if self.kind == "SURFPoints":
            names.append("SignOfLaplacian")
        return names

    def method_names(self) -> list[str]:
        return sorted(_POINT_METHODS, key=str.lower)

    def has_method(self, name: str) -> bool:
        return name in _POINT_METHODS

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        fn = _POINT_METHODS[name]
        me = next((i for i, a in enumerate(args) if a is self), 0)
        rest = [a for i, a in enumerate(args) if i != me]
        res = fn(self, interp, rest, nargout)
        if res is None:
            return []
        return [to_matlab(r) for r in res] if isinstance(res, tuple) else [to_matlab(res)]

    def get_field(self, interp, name: str) -> Any:
        if name == "Location":
            return self.loc.astype(np.float32)
        if name == "Metric":
            return col(self.metric).astype(np.float32)
        if name == "Count":
            return scalar(self.count)
        if name == "Scale" and self.kind in _HAS_SCALE:
            return col(self.scale).astype(np.float32)
        if name == "Orientation" and self.kind in _HAS_SCALE:
            return col(self.orientation).astype(np.float32)
        if name == "SignOfLaplacian" and self.kind == "SURFPoints":
            return self.extra["laplacian"].astype(np.int8).reshape(-1, 1)
        if name in _POINT_METHODS:
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class '{self.kind}'.",
                          "MATLAB:noSuchMethodOrField")

    def set_field(self, interp, name: str, value: Any) -> Any:
        if name not in self.property_names() or name == "Count":
            raise MatlabError(f"Unable to set the '{name}' property of class '{self.kind}'.",
                              "MATLAB:class:SetProhibited")
        out = self.subset(np.arange(self.count))
        if name == "Location":
            loc = fmat(value, self.kind, "Location", 2)
            if loc.shape[0] != self.count:
                raise MatlabError(f"{self.kind}: the new Location must have {self.count} rows.")
            out.loc = loc
            return out
        v = fvec(value, self.kind, name)
        if v.size not in (1, self.count):
            raise MatlabError(f"{self.kind}: {name} must have one value per point (or be a scalar).")
        v = np.full(self.count, v[0]) if v.size == 1 else v
        if name == "Metric":
            out.metric = v
        elif name == "Scale":
            out.scale = v
        elif name == "Orientation":
            out.orientation = v
        else:
            out.extra["laplacian"] = v.astype(np.int8)
        return out

    # indexing: points(idx) ---------------------------------------------------------------------------
    def index_get(self, interp, args: list) -> Any:
        pos = np.arange(self.count, dtype=float).reshape(-1, 1)
        if len(args) == 2:
            ix.get(np.zeros((self.count, 1)), list(args))  # checks the column index
            args = [args[0]]
        sel = ix.get(pos, list(args))
        return self.subset(np.asarray(sel).reshape(-1).astype(np.int64))

    def index_delete(self, interp, args: list) -> Any:
        pos = np.arange(self.count, dtype=float).reshape(-1, 1)
        keep = ix.delete(pos, list(args[:1]))
        return self.subset(np.asarray(keep).reshape(-1).astype(np.int64))

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        parts = [it for it in items if not (isinstance(it, np.ndarray) and it.size == 0)]
        if not parts:
            return np.zeros((0, 0))
        kinds = {getattr(p, "kind", class_name(p)) for p in parts}
        if len(kinds) != 1 or not all(isinstance(p, FeaturePoints) for p in parts):
            raise MatlabError("Only point objects of the same class can be concatenated.",
                              "MATLAB:catenate:dimensionMismatch")
        if axis != 0 and any(p.count != 1 for p in parts):
            raise MatlabError("Point objects are column arrays; concatenate them vertically, e.g. [p1; p2].",
                              "MATLAB:catenate:dimensionMismatch")
        keys = set.intersection(*(set(p.extra) for p in parts))
        return FeaturePoints(parts[0].kind, np.vstack([p.loc for p in parts]),
                             np.concatenate([p.metric for p in parts]), np.concatenate([p.scale for p in parts]),
                             np.concatenate([p.orientation for p in parts]),
                             {k: np.concatenate([p.extra[k] for p in parts]) for k in keys})

    # display ---------------------------------------------------------------------------------------------
    def summary_text(self, in_cell: bool) -> str:
        t = f"{self.count}×1 {self.kind}"
        return t if in_cell else f"[{t}]"

    def display_lines(self) -> list[str]:
        n = self.count
        lines = [f"  {n}×1 {self.kind} array with properties:", ""]
        rows = [("Location", f"[{n}×2 single]"), ("Metric", f"[{n}×1 single]"), ("Count", str(n))]
        if self.kind in _HAS_SCALE:
            rows += [("Scale", f"[{n}×1 single]"), ("Orientation", f"[{n}×1 single]")]
        if self.kind == "SURFPoints":
            rows.append(("SignOfLaplacian", f"[{n}×1 int8]"))
        w = max(len(r[0]) for r in rows)
        return lines + [f"    {k.rjust(w)}: {v}" for k, v in rows]


_POINT_METHODS: dict = {}


def _pmethod(name: str):
    def deco(fn):
        _POINT_METHODS[name] = fn
        return fn
    return deco


@_pmethod("selectStrongest")
def _m_select_strongest(self: FeaturePoints, interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("selectStrongest: expected selectStrongest(points, N).")
    n = int(fscalar(args[0], "selectStrongest", "N"))
    if n < 0:
        raise MatlabError("selectStrongest: N must be a non-negative integer.")
    order = np.argsort(-self.metric, kind="stable")[:n]
    return self.subset(order)


def select_uniform(loc: np.ndarray, metric: np.ndarray, n: int, size: tuple[int, int]) -> np.ndarray:
    """Indices of up to n strong points spread over the image: greedy picking with a shrinking radius."""
    m = loc.shape[0]
    if n >= m:
        return np.argsort(-metric, kind="stable")
    order = np.argsort(-metric, kind="stable")
    r = np.sqrt(size[0] * size[1] / (np.pi * max(n, 1)))
    chosen: list[int] = []
    taken = np.zeros(m, dtype=bool)
    while len(chosen) < n and r >= 0.5:
        for i in order:
            if taken[i]:
                continue
            if chosen:
                d2 = np.sum((loc[chosen] - loc[i]) ** 2, axis=1)
                if np.any(d2 < r * r):
                    continue
            chosen.append(int(i))
            taken[i] = True
            if len(chosen) == n:
                break
        r *= 0.7
    if len(chosen) < n:
        chosen += [int(i) for i in order if not taken[i]][: n - len(chosen)]
    chosen = np.array(chosen, dtype=np.int64)
    return chosen[np.argsort(-metric[chosen], kind="stable")]


@_pmethod("selectUniform")
def _m_select_uniform(self: FeaturePoints, interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("selectUniform: expected selectUniform(points, N, imageSize).")
    n = int(fscalar(args[0], "selectUniform", "N"))
    sz = fvec(args[1], "selectUniform", "imageSize")
    if sz.size < 2:
        raise MatlabError("selectUniform: imageSize must be [rows cols] (or size(I)).")
    return self.subset(select_uniform(self.loc, self.metric, n, (int(sz[0]), int(sz[1]))))


@_pmethod("isempty")
def _m_isempty(self, interp, args, nargout):
    return np.array([[self.count == 0]])


@_pmethod("length")
def _m_length(self, interp, args, nargout):
    return scalar(self.count)


@_pmethod("gather")
def _m_gather(self, interp, args, nargout):
    return self


@_pmethod("plot")
def _m_plot(self: FeaturePoints, interp, args, nargout):
    ax = []
    if args and hasattr(args[0], "isa_class") and not isinstance(args[0], FeaturePoints):
        ax, args = [args[0]], args[1:]
    _, opts = options(list(args), ("showScale", "showOrientation", "ShowPixelList", "ShowEllipses", "Parent"),
                      "plot", 0)
    if "Parent" in opts:
        ax = [opts["Parent"]]
    show_scale = _flag_opt(opts, "showScale", self.kind in _HAS_SCALE)
    show_orient = _flag_opt(opts, "showOrientation", False)
    plot_points(interp, ax, self.loc, self.scale if show_scale and self.kind in _HAS_SCALE else None,
                self.orientation if show_orient else None, _radius_factor(self.kind))
    return None


def _radius_factor(kind: str) -> float:
    return {"SURFPoints": 6.0, "SIFTPoints": 6.0, "KAZEPoints": 6.0, "ORBPoints": 15.5, "BRISKPoints": 1.0}.get(kind, 0)


def _flag_opt(opts: dict, key: str, default: bool) -> bool:
    if key not in opts:
        return default
    v = opts[key]
    return text(v).lower() in ("true", "on") if is_text(v) else bool(np.ravel(v)[0])


def plot_points(interp, ax: list, loc: np.ndarray, scale, orientation, factor: float) -> None:
    """Markers at the points; optional circles (radius factor*scale) and orientation spokes."""
    frame = interp.frame
    held = bool(np.all(np.asarray(interp.call_named("ishold", list(ax), 1, frame)[0]) != 0))
    if loc.shape[0] == 0:
        return
    interp.call_named("plot", [*ax, loc[:, 0:1].T, loc[:, 1:2].T, char("g+")], 0, frame)
    interp.call_named("hold", [*ax, char("on")], 0, frame)
    try:
        if scale is not None and factor > 0:
            t = np.linspace(0, 2 * np.pi, 33)
            r = (scale * factor).reshape(-1, 1)
            xs = (loc[:, 0:1] + r * np.cos(t)).T
            ys = (loc[:, 1:2] + r * np.sin(t)).T
            interp.call_named("plot", [*ax, xs, ys, char("g-")], 0, frame)
        if orientation is not None:
            r = (scale * factor if scale is not None and factor > 0 else np.full(loc.shape[0], 5.0))
            x2 = loc[:, 0] + r * np.cos(orientation)
            y2 = loc[:, 1] + r * np.sin(orientation)
            xs = np.vstack([loc[:, 0], x2])
            ys = np.vstack([loc[:, 1], y2])
            interp.call_named("plot", [*ax, xs, ys, char("g-")], 0, frame)
    finally:
        if not held:
            interp.call_named("hold", [*ax, char("off")], 0, frame)


# -- constructors ------------------------------------------------------------------------------------------

def _point_ctor(kind: str):
    names = ("Metric",) + (("Scale", "Orientation") if kind in _HAS_SCALE else ()) + \
        (("SignOfLaplacian",) if kind == "SURFPoints" else ())

    def fn(interp, args, nargout):
        pos, opts = options(list(args), names, kind, 1)
        loc = fmat(pos[0], kind, "locations", 2) if pos else np.zeros((0, 2))
        get = (lambda k: fvec(opts[k], kind, k) if k in opts else None)
        extra = {}
        if kind == "SURFPoints" and "SignOfLaplacian" in opts:
            s = fvec(opts["SignOfLaplacian"], kind, "SignOfLaplacian")
            extra["laplacian"] = (np.full(loc.shape[0], s[0]) if s.size == 1 else s).astype(np.int8)
        return FeaturePoints(kind, loc, get("Metric"), get("Scale"), get("Orientation"), extra)
    builtin(kind)(fn)


for _k in _POINT_CLASSES:
    _point_ctor(_k)


# -- MSER regions ------------------------------------------------------------------------------------------

class MSERRegions(MObject):
    """Maximally stable extremal regions: pixel lists plus ellipse fits."""

    def __init__(self, pixel_lists: list[np.ndarray]):
        self.pix = [np.asarray(p, dtype=float).reshape(-1, 2) for p in pixel_lists]
        n = len(self.pix)
        self.loc = np.zeros((n, 2))
        self.axes = np.zeros((n, 2))
        self.orient = np.zeros(n)
        for i, p in enumerate(self.pix):
            c = p.mean(axis=0)
            self.loc[i] = c
            d = p - c
            cov = d.T @ d / max(p.shape[0], 1) + np.eye(2) / 12.0
            w, v = np.linalg.eigh(cov)
            self.axes[i] = 4.0 * np.sqrt(np.maximum(w[::-1], 0))  # full lengths (2 * two standard deviations)
            major = v[:, 1]
            self.orient[i] = float(np.arctan2(major[1], major[0]))

    def class_name(self) -> str:
        return "MSERRegions"

    def isa_class(self, name: str) -> bool:
        return name == "MSERRegions"

    def matlab_shape(self) -> tuple:
        return (len(self.pix), 1)

    @property
    def count(self) -> int:
        return len(self.pix)

    def point_xy(self) -> np.ndarray:
        return self.loc.copy()

    def subset(self, idx) -> "MSERRegions":
        return MSERRegions([self.pix[int(i)] for i in np.asarray(idx).reshape(-1)])

    def property_names(self) -> list[str]:
        return ["Count", "Location", "Axes", "Orientation", "PixelList"]

    def has_method(self, name: str) -> bool:
        return name in ("plot", "isempty", "length")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "isempty":
            return [np.array([[self.count == 0]])]
        if name == "length":
            return [scalar(self.count)]
        _, opts = options(rest, ("showPixelList", "showEllipses", "Parent"), "plot", 0)
        ax = [opts["Parent"]] if "Parent" in opts else []
        frame = interp.frame
        held = bool(np.all(np.asarray(interp.call_named("ishold", list(ax), 1, frame)[0]) != 0))
        interp.call_named("hold", [*ax, char("on")], 0, frame)
        try:
            if _flag_opt(opts, "showPixelList", False):
                for p in self.pix:
                    interp.call_named("plot", [*ax, p[:, 0:1].T, p[:, 1:2].T, char(".")], 0, frame)
            if _flag_opt(opts, "showEllipses", True) and self.count:
                t = np.linspace(0, 2 * np.pi, 41)
                xs, ys = [], []
                for (cx, cy), (a, b), th in zip(self.loc, self.axes, self.orient):
                    x = a / 2 * np.cos(t)
                    y = b / 2 * np.sin(t)
                    xs.append(cx + x * np.cos(th) - y * np.sin(th))
                    ys.append(cy + x * np.sin(th) + y * np.cos(th))
                interp.call_named("plot", [*ax, np.array(xs).T, np.array(ys).T, char("g-")], 0, frame)
        finally:
            if not held:
                interp.call_named("hold", [*ax, char("off")], 0, frame)
        return []

    def get_field(self, interp, name: str) -> Any:
        if name == "Count":
            return scalar(self.count)
        if name == "Location":
            return self.loc.astype(np.float32)
        if name == "Axes":
            return self.axes.astype(np.float32)
        if name == "Orientation":
            return col(self.orient).astype(np.float32)
        if name == "PixelList":
            data = np.empty((self.count, 1), dtype=object)
            for i, p in enumerate(self.pix):
                data[i, 0] = p.astype(np.int32)
            return Cell(data)
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class 'MSERRegions'.",
                          "MATLAB:noSuchMethodOrField")

    def index_get(self, interp, args: list) -> Any:
        pos = np.arange(self.count, dtype=float).reshape(-1, 1)
        sel = ix.get(pos, list(args[:1]))
        return self.subset(np.asarray(sel).reshape(-1).astype(np.int64))

    def summary_text(self, in_cell: bool) -> str:
        t = f"{self.count}×1 MSERRegions"
        return t if in_cell else f"[{t}]"

    def display_lines(self) -> list[str]:
        n = self.count
        return [f"  {n}×1 MSERRegions array with properties:", "",
                f"          Count: {n}", f"       Location: [{n}×2 single]", f"           Axes: [{n}×2 single]",
                f"    Orientation: [{n}×1 single]", f"      PixelList: {{{n}×1 cell}}"]


@builtin("MSERRegions")
def _mserregions(interp, args, nargout):
    if not args:
        return MSERRegions([])
    v = args[0]
    if not isinstance(v, Cell):
        raise MatlabError("MSERRegions: the pixel lists must be a cell array of M-by-2 [x y] matrices.")
    return MSERRegions([fmat(p, "MSERRegions", "pixel list", 2) for p in v.data.flatten(order="F")])


# -- detectors ---------------------------------------------------------------------------------------------

def _roi(opts: dict, shape: tuple, fname: str) -> tuple[int, int, int, int]:
    """ROI [x y w h] (1-based) as 0-based (x0, y0, x1, y1) exclusive bounds."""
    H, W = shape[:2]
    if "ROI" not in opts:
        return 0, 0, W, H
    r = fvec(opts["ROI"], fname, "ROI", 4)
    x0, y0 = int(round(r[0])) - 1, int(round(r[1])) - 1
    x1, y1 = x0 + int(round(r[2])), y0 + int(round(r[3]))
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or r[2] <= 0 or r[3] <= 0:
        raise MatlabError(f"{fname}: the ROI [x y width height] must lie inside the image.",
                          f"mylab:{fname}:invalidROI")
    return x0, y0, x1, y1


def _crop(g: np.ndarray, roi) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return np.ascontiguousarray(g[y0:y1, x0:x1])


def _kp_points(kind: str, kps, roi, scale_of, extra_fn=None) -> FeaturePoints:
    n = len(kps)
    loc = np.array([kp.pt for kp in kps], dtype=float).reshape(n, 2) + np.array([roi[0] + 1, roi[1] + 1])
    metric = np.array([kp.response for kp in kps], dtype=float)
    size = np.array([kp.size for kp in kps], dtype=float)
    octave = np.array([kp.octave for kp in kps], dtype=np.int64)
    angle = np.array([kp.angle for kp in kps], dtype=float)
    orient = np.where(angle >= 0, np.deg2rad(angle), 0.0)
    extra = {"size": size, "octave": octave, "class_id": np.array([kp.class_id for kp in kps], dtype=np.int64)}
    if extra_fn:
        extra.update(extra_fn(kps))
    pts = FeaturePoints(kind, loc, metric, scale_of(size, octave), orient, extra)
    order = np.argsort(-pts.metric, kind="stable")
    return pts.subset(order)


def _corner_detect(fname: str, harris: bool, args):
    pos, opts = options(list(args), ("MinQuality", "FilterSize", "ROI"), fname, 1)
    if not pos:
        raise MatlabError("Not enough input arguments.", "MATLAB:minrhs")
    cv2 = cv2mod(fname)
    img = image_arg(pos[0], fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_float(img, fname).astype(np.float32)
    q = fscalar(opts["MinQuality"], fname, "MinQuality") if "MinQuality" in opts else 0.01
    fs = int(fscalar(opts["FilterSize"], fname, "FilterSize")) if "FilterSize" in opts else 5
    if fs < 3 or fs % 2 == 0:
        raise MatlabError(f"{fname}: FilterSize must be an odd integer of at least 3.",
                          f"mylab:{fname}:invalidFilterSize")
    if not 0 <= q <= 1:
        raise MatlabError(f"{fname}: MinQuality must be between 0 and 1.", f"mylab:{fname}:invalidMinQuality")
    roi = _roi(opts, g.shape, fname)
    sub = _crop(g, roi)
    # smooth with a Gaussian of the requested filter size, then a 3x3 structure tensor window
    sigma = fs / 3.0
    sub = cv2.GaussianBlur(sub, (fs, fs), sigma)
    if harris:
        resp = cv2.cornerHarris(sub, 3, 3, 0.04)
    else:
        resp = cv2.cornerMinEigenVal(sub, 3, 3)
    resp = resp.astype(float)
    if resp.size == 0 or resp.max() <= 0:
        return FeaturePoints("cornerPoints", np.zeros((0, 2)))
    dil = cv2.dilate(resp.astype(np.float32), np.ones((3, 3), np.uint8)).astype(float)
    peak = (resp >= dil) & (resp > q * resp.max()) & (resp > 0)
    peak[[0, -1], :] = False
    peak[:, [0, -1]] = False
    ys, xs = np.nonzero(peak)
    # sub-pixel refinement by fitting a parabola through the neighbours
    def refine(c, m, p):
        den = m - 2 * c + p
        return np.where(np.abs(den) > 1e-12, 0.5 * (m - p) / den, 0.0).clip(-0.5, 0.5)
    c = resp[ys, xs]
    dx = refine(c, resp[ys, xs - 1], resp[ys, xs + 1])
    dy = refine(c, resp[ys - 1, xs], resp[ys + 1, xs])
    loc = np.column_stack([xs + dx + roi[0] + 1, ys + dy + roi[1] + 1])
    pts = FeaturePoints("cornerPoints", loc, c)
    return pts.subset(np.argsort(-pts.metric, kind="stable"))


@builtin("detectHarrisFeatures")
def _detect_harris(interp, args, nargout):
    return _corner_detect("detectHarrisFeatures", True, args)


@builtin("detectMinEigenFeatures")
def _detect_mineig(interp, args, nargout):
    return _corner_detect("detectMinEigenFeatures", False, args)


@builtin("detectFASTFeatures")
def _detect_fast(interp, args, nargout):
    fname = "detectFASTFeatures"
    pos, opts = options(list(args), ("MinQuality", "MinContrast", "ROI"), fname, 1)
    cv2 = cv2mod(fname)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    q = fscalar(opts["MinQuality"], fname, "MinQuality") if "MinQuality" in opts else 0.1
    c = fscalar(opts["MinContrast"], fname, "MinContrast") if "MinContrast" in opts else 0.2
    if not (0 <= q <= 1 and 0 < c < 1):
        raise MatlabError(f"{fname}: MinQuality must be in [0, 1] and MinContrast in (0, 1).")
    roi = _roi(opts, g.shape, fname)
    det = cv2.FastFeatureDetector_create(threshold=max(1, int(round(c * 255))), nonmaxSuppression=True)
    kps = det.detect(_crop(g, roi), None)
    pts = _kp_points("cornerPoints", kps, roi, lambda s, o: None)
    if pts.count:
        pts = pts.subset(np.nonzero(pts.metric >= q * pts.metric.max())[0])
    return FeaturePoints("cornerPoints", pts.loc, pts.metric)


@builtin("detectORBFeatures")
def _detect_orb(interp, args, nargout):
    fname = "detectORBFeatures"
    pos, opts = options(list(args), ("ScaleFactor", "NumLevels", "ROI"), fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    sf = fscalar(opts["ScaleFactor"], fname, "ScaleFactor") if "ScaleFactor" in opts else 1.2
    nl = int(fscalar(opts["NumLevels"], fname, "NumLevels")) if "NumLevels" in opts else 8
    if sf <= 1 or nl < 1:
        raise MatlabError(f"{fname}: ScaleFactor must be greater than 1 and NumLevels at least 1.")
    roi = _roi(opts, g.shape, fname)
    cv2 = cv2mod(fname)
    orb = cv2.ORB_create(nfeatures=100000, scaleFactor=sf, nlevels=nl, edgeThreshold=31, patchSize=31,
                         fastThreshold=20)
    kps = orb.detect(_crop(g, roi), None)
    return _kp_points("ORBPoints", kps, roi, lambda s, o: sf ** o.astype(float))


@builtin("detectSIFTFeatures")
def _detect_sift(interp, args, nargout):
    fname = "detectSIFTFeatures"
    pos, opts = options(list(args), ("ContrastThreshold", "EdgeThreshold", "NumLayersInOctave", "Sigma", "ROI"),
                        fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    layers = int(fscalar(opts["NumLayersInOctave"], fname, "NumLayersInOctave")) if "NumLayersInOctave" in opts else 3
    ct = fscalar(opts["ContrastThreshold"], fname, "ContrastThreshold") if "ContrastThreshold" in opts else 0.0133
    et = fscalar(opts["EdgeThreshold"], fname, "EdgeThreshold") if "EdgeThreshold" in opts else 10.0
    sigma = fscalar(opts["Sigma"], fname, "Sigma") if "Sigma" in opts else 1.6
    roi = _roi(opts, g.shape, fname)
    cv2 = cv2mod(fname)
    # OpenCV divides its contrast threshold by the number of layers
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=layers, contrastThreshold=ct * layers, edgeThreshold=et,
                           sigma=sigma)
    kps = sift.detect(_crop(g, roi), None)
    return _kp_points("SIFTPoints", kps, roi, lambda s, o: s / 2.0)


@builtin("detectBRISKFeatures")
def _detect_brisk(interp, args, nargout):
    fname = "detectBRISKFeatures"
    pos, opts = options(list(args), ("MinContrast", "MinQuality", "NumOctaves", "ROI"), fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    c = fscalar(opts["MinContrast"], fname, "MinContrast") if "MinContrast" in opts else 0.2
    q = fscalar(opts["MinQuality"], fname, "MinQuality") if "MinQuality" in opts else 0.1
    octaves = int(fscalar(opts["NumOctaves"], fname, "NumOctaves")) if "NumOctaves" in opts else 4
    roi = _roi(opts, g.shape, fname)
    det = feature2d(fname, "BRISK", thresh=max(1, int(round(c * 255))), octaves=octaves)
    kps = det.detect(_crop(g, roi), None)
    pts = _kp_points("BRISKPoints", kps, roi, lambda s, o: s)
    if pts.count:
        pts = pts.subset(np.nonzero(pts.metric >= q * pts.metric.max())[0])
    return pts


@builtin("detectKAZEFeatures")
def _detect_kaze(interp, args, nargout):
    fname = "detectKAZEFeatures"
    pos, opts = options(list(args), ("Diffusion", "Threshold", "NumOctaves", "NumScaleLevels", "ROI"), fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    diff = opt_text(opts, "Diffusion", "region", ("region", "sharpedge", "edge"), fname)
    th = fscalar(opts["Threshold"], fname, "Threshold") if "Threshold" in opts else 0.0001
    no = int(fscalar(opts["NumOctaves"], fname, "NumOctaves")) if "NumOctaves" in opts else 3
    ns = int(fscalar(opts["NumScaleLevels"], fname, "NumScaleLevels")) if "NumScaleLevels" in opts else 4
    roi = _roi(opts, g.shape, fname)
    cv2 = cv2mod(fname)
    dmap = {"region": cv2.KAZE_DIFF_PM_G2 if hasattr(cv2, "KAZE_DIFF_PM_G2") else 1,
            "sharpedge": cv2.KAZE_DIFF_PM_G1 if hasattr(cv2, "KAZE_DIFF_PM_G1") else 0,
            "edge": cv2.KAZE_DIFF_WEICKERT if hasattr(cv2, "KAZE_DIFF_WEICKERT") else 2}
    det = feature2d(fname, "KAZE", threshold=th, nOctaves=no, nOctaveLayers=ns, diffusivity=dmap[diff])
    kps = det.detect(_crop(g, roi), None)
    return _kp_points("KAZEPoints", kps, roi, lambda s, o: s / 2.0)


@builtin("detectMSERFeatures")
def _detect_mser(interp, args, nargout):
    fname = "detectMSERFeatures"
    pos, opts = options(list(args), ("ThresholdDelta", "RegionAreaRange", "MaxAreaVariation", "ROI"), fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_u8(img, fname)
    delta = fscalar(opts["ThresholdDelta"], fname, "ThresholdDelta") if "ThresholdDelta" in opts else 2.0
    rng = fvec(opts["RegionAreaRange"], fname, "RegionAreaRange", 2) if "RegionAreaRange" in opts \
        else np.array([30.0, 14000.0])
    var = fscalar(opts["MaxAreaVariation"], fname, "MaxAreaVariation") if "MaxAreaVariation" in opts else 0.25
    roi = _roi(opts, g.shape, fname)
    cv2 = cv2mod(fname)
    # ThresholdDelta is a percentage of the intensity range; OpenCV's delta is in grey levels
    mser = cv2.MSER_create(delta=max(1, int(round(delta * 255 / 100))), min_area=int(rng[0]),
                           max_area=int(rng[1]), max_variation=var)
    regions, _ = mser.detectRegions(_crop(g, roi))
    off = np.array([roi[0] + 1, roi[1] + 1])
    lists = [np.asarray(r, dtype=float).reshape(-1, 2) + off for r in regions]
    out = MSERRegions(lists)
    if nargout > 1:
        from ..values import Struct
        H, W = g.shape
        idx = np.empty((1, len(lists)), dtype=object)
        for i, p in enumerate(lists):
            lin = (p[:, 0] - 1) * H + p[:, 1]
            idx[0, i] = np.sort(lin).reshape(-1, 1)
        cc = Struct.scalar({"Connectivity": scalar(8), "ImageSize": np.array([[float(H), float(W)]]),
                     "NumObjects": scalar(len(lists)), "PixelIdxList": Cell(idx)})
        return out, cc
    return out


# -- SURF (NumPy implementation) ---------------------------------------------------------------------------

def _gauss_derivs(img: np.ndarray, sigma: float):
    from scipy import ndimage
    dxx = ndimage.gaussian_filter(img, sigma, order=(0, 2), mode="nearest")
    dyy = ndimage.gaussian_filter(img, sigma, order=(2, 0), mode="nearest")
    dxy = ndimage.gaussian_filter(img, sigma, order=(1, 1), mode="nearest")
    return dxx, dyy, dxy


def surf_detect(g: np.ndarray, threshold: float, n_octaves: int, n_levels: int):
    """Hessian-determinant blobs: (x, y) 0-based, scale (sigma), response, sign of Laplacian."""
    from scipy import ndimage
    img = g * 255.0
    found = []
    for o in range(n_octaves):
        step = 2 ** o
        sub = img[::step, ::step] if step > 1 else img
        if min(sub.shape) < 8:
            break
        sub = ndimage.gaussian_filter(sub, 0.5 * (step > 1))
        # filter sizes 9, 15, 21, 27 in the first octave; the size increment doubles each octave
        sizes = [3 * (2 ** (o + 1)) * (k + 1) + 3 for k in range(n_levels)]
        sigmas = [1.2 * L / 9.0 for L in sizes]
        stack, lap = [], []
        for s in sigmas:
            sp = s / step
            dxx, dyy, dxy = _gauss_derivs(sub, sp)
            stack.append((dxx * dyy - (0.9 * dxy) ** 2) * sp ** 4)
            lap.append(dxx + dyy)
        R = np.stack(stack)
        L = np.stack(lap)
        mx = ndimage.maximum_filter(R, size=(3, 3, 3), mode="constant", cval=-np.inf)
        peak = (R == mx) & (R > threshold)
        peak[0] = peak[-1] = False
        margin = 2
        peak[:, :margin, :] = peak[:, -margin:, :] = False
        peak[:, :, :margin] = peak[:, :, -margin:] = False
        for k, y, x in zip(*np.nonzero(peak)):
            c = R[k, y, x]

            def off(m, p):
                den = m - 2 * c + p
                return float(np.clip(0.5 * (m - p) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
            ox = off(R[k, y, x - 1], R[k, y, x + 1])
            oy = off(R[k, y - 1, x], R[k, y + 1, x])
            ok = off(R[k - 1, y, x], R[k + 1, y, x])
            s = sigmas[k] + ok * (sigmas[k + 1] - sigmas[k] if ok > 0 else sigmas[k] - sigmas[k - 1])
            found.append(((x + ox) * step, (y + oy) * step, s, c, np.sign(L[k, y, x])))
    if not found:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int8)
    a = np.array(found, dtype=float)
    return a[:, :2], a[:, 2], a[:, 3], a[:, 4].astype(np.int8)


class _GradPyramid:
    """Gaussian-smoothed gradients at a few scales, sampled by bilinear interpolation."""

    def __init__(self, g: np.ndarray):
        self.g = g
        self.cache: dict = {}

    def grads(self, s: float):
        from scipy import ndimage
        key = round(float(np.log2(max(s, 0.5)) * 4)) / 4
        if key not in self.cache:
            sig = 2.0 ** key
            gx = ndimage.gaussian_filter(self.g, sig, order=(0, 1), mode="nearest") * sig
            gy = ndimage.gaussian_filter(self.g, sig, order=(1, 0), mode="nearest") * sig
            self.cache[key] = (gx, gy)
        return self.cache[key]

    @staticmethod
    def sample(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from scipy import ndimage
        return ndimage.map_coordinates(a, [y.ravel(), x.ravel()], order=1, mode="constant", cval=0.0).reshape(x.shape)


def surf_orientation(pyr: _GradPyramid, xy: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.zeros(xy.shape[0])
    gi, gj = np.mgrid[-6:7, -6:7]
    inside = gi ** 2 + gj ** 2 <= 36
    gi, gj = gi[inside].astype(float), gj[inside].astype(float)
    w = np.exp(-(gi ** 2 + gj ** 2) / (2 * 2.5 ** 2))
    for i, ((x, y), s) in enumerate(zip(xy, scale)):
        gx, gy = pyr.grads(s)
        dx = pyr.sample(gx, x + gj * s, y + gi * s) * w
        dy = pyr.sample(gy, x + gj * s, y + gi * s) * w
        ang = np.arctan2(dy, dx)
        best, best_ang = -1.0, 0.0
        for a0 in np.arange(0, 2 * np.pi, np.pi / 18):
            d = np.mod(ang - a0, 2 * np.pi)
            m = d < np.pi / 3
            sx, sy = dx[m].sum(), dy[m].sum()
            mag = sx * sx + sy * sy
            if mag > best:
                best, best_ang = mag, np.arctan2(sy, sx)
        out[i] = np.mod(best_ang, 2 * np.pi)
    return out


def surf_describe(pyr: _GradPyramid, xy: np.ndarray, scale: np.ndarray, orient: np.ndarray,
                  extended: bool) -> np.ndarray:
    n = xy.shape[0]
    D = 128 if extended else 64
    out = np.zeros((n, D), dtype=np.float32)
    u = (np.arange(20) - 9.5)  # 20x20 samples across a 20s window
    U, V = np.meshgrid(u, u)
    w = np.exp(-(U ** 2 + V ** 2) / (2 * 3.3 ** 2))
    for i in range(n):
        x, y = xy[i]
        s, th = scale[i], orient[i]
        c, si = np.cos(th), np.sin(th)
        X = x + s * (c * U - si * V)
        Y = y + s * (si * U + c * V)
        gx, gy = pyr.grads(s)
        ax = pyr.sample(gx, X, Y)
        ay = pyr.sample(gy, X, Y)
        du = (c * ax + si * ay) * w   # gradient along the rotated axes
        dv = (-si * ax + c * ay) * w
        feats = []
        for bi in range(4):
            for bj in range(4):
                a = du[bi * 5:(bi + 1) * 5, bj * 5:(bj + 1) * 5].ravel()
                b = dv[bi * 5:(bi + 1) * 5, bj * 5:(bj + 1) * 5].ravel()
                if extended:
                    pos_b, neg_b = b >= 0, b < 0
                    pos_a, neg_a = a >= 0, a < 0
                    feats += [a[pos_b].sum(), np.abs(a[pos_b]).sum(), a[neg_b].sum(), np.abs(a[neg_b]).sum(),
                              b[pos_a].sum(), np.abs(b[pos_a]).sum(), b[neg_a].sum(), np.abs(b[neg_a]).sum()]
                else:
                    feats += [a.sum(), b.sum(), np.abs(a).sum(), np.abs(b).sum()]
        f = np.array(feats)
        nrm = np.linalg.norm(f)
        out[i] = f / nrm if nrm > 0 else f
    return out


@builtin("detectSURFFeatures")
def _detect_surf(interp, args, nargout):
    fname = "detectSURFFeatures"
    pos, opts = options(list(args), ("MetricThreshold", "NumOctaves", "NumScaleLevels", "ROI"), fname, 1)
    img = image_arg(pos[0] if pos else None, fname)
    if img.ndim != 2:
        raise MatlabError(f"{fname}: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    g = gray_float(img, fname)
    th = fscalar(opts["MetricThreshold"], fname, "MetricThreshold") if "MetricThreshold" in opts else 1000.0
    no = int(fscalar(opts["NumOctaves"], fname, "NumOctaves")) if "NumOctaves" in opts else 3
    ns = int(fscalar(opts["NumScaleLevels"], fname, "NumScaleLevels")) if "NumScaleLevels" in opts else 4
    if no < 1 or ns < 3:
        raise MatlabError(f"{fname}: NumOctaves must be at least 1 and NumScaleLevels at least 3.")
    roi = _roi(opts, g.shape, fname)
    xy, s, r, lap = surf_detect(_crop(g, roi), th, no, ns)
    loc = xy + np.array([roi[0] + 1, roi[1] + 1])
    pts = FeaturePoints("SURFPoints", loc, r, s, np.zeros(len(s)), {"laplacian": lap})
    return pts.subset(np.argsort(-pts.metric, kind="stable"))


# -- binary feature vectors -------------------------------------------------------------------------------

class BinaryFeatures(MObject):
    """binaryFeatures: one row of packed uint8 bits per feature."""

    def __init__(self, data: np.ndarray):
        self.data = np.ascontiguousarray(np.asarray(data, dtype=np.uint8).reshape(data.shape[0], -1))

    def class_name(self) -> str:
        return "binaryFeatures"

    def isa_class(self, name: str) -> bool:
        return name == "binaryFeatures"

    def matlab_shape(self) -> tuple:
        return (self.data.shape[0], self.data.shape[1])

    def property_names(self) -> list[str]:
        return ["Features", "NumBits", "NumFeatures"]

    def get_field(self, interp, name: str) -> Any:
        if name == "Features":
            return self.data.copy()
        if name == "NumBits":
            return scalar(self.data.shape[1] * 8)
        if name == "NumFeatures":
            return scalar(self.data.shape[0])
        raise MatlabError(f"Unrecognized method, property, or field '{name}' for class 'binaryFeatures'.",
                          "MATLAB:noSuchMethodOrField")

    def index_get(self, interp, args: list) -> Any:
        pos = np.arange(self.data.shape[0], dtype=float).reshape(-1, 1)
        sel = np.asarray(ix.get(pos, list(args[:1]))).reshape(-1).astype(np.int64)
        return BinaryFeatures(self.data[sel])

    @classmethod
    def concat(cls, items: list, axis: int) -> Any:
        parts = [it for it in items if not (isinstance(it, np.ndarray) and it.size == 0)]
        if not all(isinstance(p, BinaryFeatures) for p in parts) or axis != 0:
            raise MatlabError("binaryFeatures objects can only be concatenated vertically with each other.")
        return BinaryFeatures(np.vstack([p.data for p in parts]))

    def summary_text(self, in_cell: bool) -> str:
        t = f"{self.data.shape[0]}×{self.data.shape[1]} binaryFeatures"
        return t if in_cell else f"[{t}]"

    def display_lines(self) -> list[str]:
        n, b = self.data.shape
        return ["  binaryFeatures with properties:", "", f"       Features: [{n}×{b} uint8]",
                f"        NumBits: {b * 8}", f"    NumFeatures: {n}"]


@builtin("binaryFeatures")
def _binary_features(interp, args, nargout):
    if len(args) != 1 or not isinstance(args[0], np.ndarray) or args[0].dtype != np.uint8 or args[0].ndim != 2:
        raise MatlabError("binaryFeatures: the input must be an M-by-N uint8 matrix (one feature per row).",
                          "mylab:binaryFeatures:invalidInput")
    return BinaryFeatures(args[0])


# -- extractFeatures ---------------------------------------------------------------------------------------

def _keypoints(cv2, pts: FeaturePoints, method: str):
    """OpenCV keypoints for the points, with class_id = row number (to find the survivors)."""
    kps = []
    size = pts.extra.get("size")
    octave = pts.extra.get("octave")
    for i in range(pts.count):
        x, y = pts.loc[i] - 1.0
        s = float(pts.scale[i])
        if size is not None and pts.kind in ("ORBPoints", "SIFTPoints", "BRISKPoints", "KAZEPoints"):
            sz = float(size[i])
        elif method == "ORB":
            sz = 31.0 * (s if pts.kind == "ORBPoints" else 1.0)
        elif method == "SIFT":
            sz = 2.0 * (s if pts.kind in _HAS_SCALE else 1.6)
        elif method == "KAZE":
            sz = 2.0 * (s if pts.kind in _HAS_SCALE else 1.6)
        else:
            sz = 12.0 if pts.kind not in _HAS_SCALE else max(12.0 * s / 1.6, 7.0)
        oc = int(octave[i]) if octave is not None and pts.kind != "cornerPoints" else _default_octave(method, pts, i)
        ang = float(np.rad2deg(pts.orientation[i])) if pts.kind in _HAS_SCALE else -1.0
        cid = i
        if method == "KAZE":  # KAZE keeps the scale-space level of each point in class_id
            stored = pts.extra.get("class_id")
            if stored is not None and pts.kind == "KAZEPoints":
                cid = int(np.clip(stored[i], 0, 15))
            else:
                cid = int(np.clip(round(2 * np.log2(max(sz / 2.0, 1.6) / 1.6)), 0, 15))
        kps.append(cv2.KeyPoint(float(x), float(y), sz, ang, float(pts.metric[i]), oc, cid))
    return kps


def _default_octave(method: str, pts: FeaturePoints, i: int) -> int:
    if method == "ORB" and pts.kind == "ORBPoints":
        return int(round(np.log(max(pts.scale[i], 1.0)) / np.log(1.2)))
    if method == "SIFT":
        # OpenCV packs octave (low byte, signed) and layer (next byte); sigma = 1.6 * 2^(octave + layer/3)
        sigma = pts.scale[i] if pts.kind in _HAS_SCALE else 1.6
        v = np.log2(max(sigma, 0.2) / 1.6)
        octv = int(np.floor(v))
        layer = int(np.clip(round((v - octv) * 3), 0, 2)) + 1
        return (octv & 255) | (layer << 8)
    return 0


def _survivors(kps_out, pts: FeaturePoints, by_location: bool = False) -> np.ndarray:
    ids = np.array([kp.class_id for kp in kps_out], dtype=np.int64)
    if ids.size and (by_location or ids.min() < 0 or ids.max() >= pts.count):
        # fall back to the nearest original point
        xy = np.array([kp.pt for kp in kps_out]) + 1.0
        d = ((xy[:, None, :] - pts.loc[None]) ** 2).sum(-1)
        ids = d.argmin(axis=1)
    return ids


_EXTRACT_METHODS = ("Auto", "BRISK", "FREAK", "SURF", "ORB", "KAZE", "SIFT", "Block")


def _auto_method(pts) -> str:
    if isinstance(pts, MSERRegions):
        return "SURF"
    if not isinstance(pts, FeaturePoints):
        return "BRISK"
    return {"cornerPoints": "BRISK", "BRISKPoints": "BRISK", "SURFPoints": "SURF", "ORBPoints": "ORB",
            "KAZEPoints": "KAZE", "SIFTPoints": "SIFT"}[pts.kind]


@builtin("extractFeatures")
def _extract_features(interp, args, nargout):
    fname = "extractFeatures"
    pos, opts = options(list(args), ("Method", "BlockSize", "Upright", "FeatureSize"), fname, 2)
    if len(pos) < 2:
        raise MatlabError("extractFeatures: expected extractFeatures(I, points).", "MATLAB:minrhs")
    img = image_arg(pos[0], fname)
    if img.ndim != 2:
        raise MatlabError("extractFeatures: the image must be a 2-D grayscale image.", f"mylab:{fname}:invalidImage")
    raw = pos[1]
    matrix_in = isinstance(raw, np.ndarray)
    if matrix_in:
        pts = FeaturePoints("cornerPoints", fmat(raw, fname, "points", 2))
    elif isinstance(raw, MSERRegions):
        pts = FeaturePoints("SURFPoints", raw.loc, np.zeros(raw.count),
                            np.clip(np.sqrt(raw.axes[:, 0] * raw.axes[:, 1]) / 4.0, 1.6, None), raw.orient)
    elif isinstance(raw, FeaturePoints):
        pts = raw
    else:
        raise MatlabError(f"extractFeatures: points of class '{class_name(raw)}' are not supported.")
    method = opt_text(opts, "Method", "Auto", _EXTRACT_METHODS, fname)
    if method == "Auto":
        method = _auto_method(raw)
    upright = bool(np.ravel(opts["Upright"])[0]) if "Upright" in opts and not is_text(opts["Upright"]) else False
    if method == "Block":
        return _extract_block(img, pts, raw, matrix_in, opts, nargout)
    if method == "SURF":
        fs = int(fscalar(opts["FeatureSize"], fname, "FeatureSize")) if "FeatureSize" in opts else 64
        if fs not in (64, 128):
            raise MatlabError("extractFeatures: FeatureSize must be 64 or 128.")
        g = gray_float(img, fname)
        H, W = g.shape
        scale = pts.scale if pts.kind in _HAS_SCALE else np.full(pts.count, 1.6)
        xy = pts.loc - 1.0
        r = 10 * scale * np.sqrt(2)
        keep = np.nonzero((xy[:, 0] - r >= -0.5) & (xy[:, 1] - r >= -0.5) & (xy[:, 0] + r <= W - 0.5)
                          & (xy[:, 1] + r <= H - 0.5))[0]
        pyr = _GradPyramid(g)
        if upright:
            orient = np.zeros(keep.size)
        else:
            orient = surf_orientation(pyr, xy[keep], scale[keep])
        feats = surf_describe(pyr, xy[keep], scale[keep], orient, fs == 128)
        valid = pts.subset(keep)
        if valid.kind == "SURFPoints" or isinstance(raw, MSERRegions):
            valid.orientation = orient
        out_valid = _valid_out(raw, valid, keep, matrix_in)
        return (feats, out_valid) if nargout > 1 else feats
    cv2 = cv2mod(fname)
    g = gray_u8(img, fname)
    if method == "ORB":
        ext = cv2.ORB_create(edgeThreshold=31, patchSize=31)
    elif method == "SIFT":
        ext = cv2.SIFT_create()
    elif method == "BRISK":
        ext = feature2d(fname, "BRISK")
    elif method == "KAZE":
        ext = feature2d(fname, "KAZE", upright=upright)
    else:  # FREAK
        x2 = getattr(cv2, "xfeatures2d", None)
        if x2 is None or not hasattr(x2, "FREAK_create"):
            raise MatlabError("extractFeatures: the FREAK descriptor needs an OpenCV build with the extra "
                              "(contrib) modules. Use 'Method', 'BRISK' or 'ORB' instead.",
                              "mylab:extractFeatures:noFREAK")
        ext = x2.FREAK_create()
    kps = _keypoints(cv2, pts, method)
    if upright and method in ("SIFT", "BRISK", "ORB"):
        for kp in kps:
            kp.angle = 0.0
    if method in ("BRISK", "FREAK") and pts.kind != "BRISKPoints" and not upright:
        for kp in kps:
            kp.angle = -1.0
    kps_out, desc = ext.compute(g, kps)
    kps_out = list(kps_out or [])
    ids = _survivors(kps_out, pts, method == "KAZE")
    if desc is None:
        desc = np.zeros((0, 32 if method in ("ORB",) else 64), dtype=np.uint8 if method != "SIFT" else np.float32)
    valid = pts.subset(ids)
    if pts.kind in _HAS_SCALE and method in ("ORB", "BRISK", "KAZE", "SIFT") and not upright and kps_out:
        ang = np.array([kp.angle for kp in kps_out])
        valid.orientation = np.where(ang >= 0, np.deg2rad(ang), valid.orientation)
    out_valid = _valid_out(raw, valid, ids, matrix_in)
    if desc.dtype == np.uint8 and method in ("ORB", "BRISK", "FREAK"):
        feats = BinaryFeatures(desc)
    else:
        feats = desc.astype(np.float32)
    return (feats, out_valid) if nargout > 1 else feats


def _valid_out(raw, valid: FeaturePoints, ids, matrix_in: bool):
    if matrix_in:
        return valid.loc.astype(raw.dtype if raw.dtype.kind == "f" else float)
    if isinstance(raw, MSERRegions):
        return raw.subset(ids)
    return valid


def _extract_block(img, pts: FeaturePoints, raw, matrix_in, opts, nargout):
    bs = int(fscalar(opts["BlockSize"], "extractFeatures", "BlockSize")) if "BlockSize" in opts else 11
    if bs < 1 or bs % 2 == 0:
        raise MatlabError("extractFeatures: BlockSize must be an odd positive integer.")
    g = gray_float(img, "extractFeatures")
    H, W = g.shape
    h = bs // 2
    c = np.round(pts.loc).astype(int) - 1
    keep = np.nonzero((c[:, 0] >= h) & (c[:, 1] >= h) & (c[:, 0] < W - h) & (c[:, 1] < H - h))[0]
    feats = np.zeros((keep.size, bs * bs), dtype=np.float32)
    for r, i in enumerate(keep):
        x, y = c[i]
        feats[r] = g[y - h:y + h + 1, x - h:x + h + 1].ravel(order="F")
    valid = pts.subset(keep)
    out_valid = _valid_out(raw, valid, keep, matrix_in)
    return (feats, out_valid) if nargout > 1 else feats


# -- matchFeatures -----------------------------------------------------------------------------------------

def _popcount_table() -> np.ndarray:
    return np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


_POP = _popcount_table()


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for k in range(a.shape[1]):
        out += _POP[np.bitwise_xor(a[:, k:k + 1], b[None, :, k])]
    return out


def match_distances(f1, f2, metric: str):
    """Distance matrix, the largest possible distance and whether the features are binary."""
    if isinstance(f1, BinaryFeatures) != isinstance(f2, BinaryFeatures):
        raise MatlabError("matchFeatures: both feature sets must be binaryFeatures, or both numeric matrices.",
                          "mylab:matchFeatures:mixedTypes")
    if isinstance(f1, BinaryFeatures):
        if f1.data.shape[1] != f2.data.shape[1]:
            raise MatlabError("matchFeatures: the two feature sets must have the same number of bits.")
        return _hamming(f1.data, f2.data), f1.data.shape[1] * 8.0, True
    a = fmat(f1, "matchFeatures", "first feature set")
    b = fmat(f2, "matchFeatures", "second feature set")
    if a.shape[1] != b.shape[1] and a.size and b.size:
        raise MatlabError("matchFeatures: the two feature sets must have the same number of columns.",
                          "mylab:matchFeatures:sizeMismatch")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    if metric == "SAD":
        d = np.abs(a[:, None, :] - b[None, :, :]).sum(-1) if a.size and b.size else np.zeros((a.shape[0], b.shape[0]))
        return d, 2.0 * np.sqrt(a.shape[1]) if a.shape[1] else 1.0, False
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.maximum(d, 0.0), 4.0, False


@builtin("matchFeatures")
def _match_features(interp, args, nargout):
    fname = "matchFeatures"
    pos, opts = options(list(args), ("Method", "MatchThreshold", "MaxRatio", "Metric", "Unique"), fname, 2)
    if len(pos) != 2:
        raise MatlabError("matchFeatures: expected matchFeatures(features1, features2, ...).", "MATLAB:minrhs")
    opt_text(opts, "Method", "Exhaustive", ("Exhaustive", "Approximate"), fname)
    metric = opt_text(opts, "Metric", "SSD", ("SSD", "SAD"), fname)
    binary = isinstance(pos[0], BinaryFeatures)
    th = fscalar(opts["MatchThreshold"], fname, "MatchThreshold") if "MatchThreshold" in opts \
        else (10.0 if binary else 1.0)
    ratio = fscalar(opts["MaxRatio"], fname, "MaxRatio") if "MaxRatio" in opts else 0.6
    unique = bool(np.ravel(opts["Unique"])[0]) if "Unique" in opts else False
    if not 0 < th <= 100 or not 0 < ratio <= 1:
        raise MatlabError("matchFeatures: MatchThreshold must be in (0, 100] and MaxRatio in (0, 1].")
    d, dmax, binary = match_distances(pos[0], pos[1], metric)
    n1, n2 = d.shape
    if n1 == 0 or n2 == 0:
        return (np.zeros((0, 2), dtype=np.uint32), np.zeros((0, 1), dtype=np.float32)) if nargout > 1 \
            else np.zeros((0, 2), dtype=np.uint32)
    best = d.argmin(axis=1)
    bestd = d[np.arange(n1), best]
    ok = bestd <= th / 100.0 * dmax
    if n2 > 1 and ratio < 1:
        part = np.partition(d, 1, axis=1)
        second = part[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(second > 0, bestd / second, np.where(bestd > 0, np.inf, 1.0))
        ok &= r <= ratio
    if unique:  # keep only mutual best matches
        back = d.argmin(axis=0)
        ok &= back[best] == np.arange(n1)
    i1 = np.nonzero(ok)[0]
    pairs = np.column_stack([i1 + 1, best[i1] + 1]).astype(np.uint32)
    mm = bestd[i1]
    if binary:
        mm = mm / dmax * 100.0  # Hamming distance as a percentage of the bit count
    return (pairs, mm.reshape(-1, 1).astype(np.float32)) if nargout > 1 else pairs


# -- showMatchedFeatures -----------------------------------------------------------------------------------

@builtin("showMatchedFeatures")
def _show_matched(interp, args, nargout):
    fname = "showMatchedFeatures"
    i = 4
    method = "falsecolor"
    if len(args) > 4 and is_text(args[4]) and text(args[4]).lower() in ("falsecolor", "blend", "montage",
                                                                         "montag"):
        method = text(args[4]).lower()
        i = 5
    pos, opts = options(list(args[:4]) + list(args[i:]), ("PlotOptions", "Parent"), fname, 4)
    if len(pos) != 4:
        raise MatlabError("showMatchedFeatures: expected showMatchedFeatures(I1, I2, points1, points2).",
                          "MATLAB:minrhs")
    a, b = rgb_u8(pos[0], fname), rgb_u8(pos[1], fname)
    p1 = fmat(pos[2], fname, "points1", 2)
    p2 = fmat(pos[3], fname, "points2", 2)
    if p1.shape[0] != p2.shape[0]:
        raise MatlabError("showMatchedFeatures: points1 and points2 must have the same number of points.",
                          "mylab:showMatchedFeatures:sizeMismatch")
    H = max(a.shape[0], b.shape[0])
    if method.startswith("montag"):
        W1 = a.shape[1]
        canvas = np.zeros((H, W1 + b.shape[1], 3), dtype=np.uint8)
        canvas[:a.shape[0], :W1] = a
        canvas[:b.shape[0], W1:] = b
        q2 = p2 + np.array([W1, 0])
    else:
        W = max(a.shape[1], b.shape[1])
        ga = np.zeros((H, W))
        gb = np.zeros((H, W))
        ga[:a.shape[0], :a.shape[1]] = a.mean(axis=2)
        gb[:b.shape[0], :b.shape[1]] = b.mean(axis=2)
        if method == "blend":
            canvas = np.repeat(((ga + gb) / 2)[:, :, None], 3, axis=2).astype(np.uint8)
        else:
            canvas = np.stack([ga, gb, gb], axis=2).astype(np.uint8)
        q2 = p2
    ax = [opts["Parent"]] if "Parent" in opts else []
    frame = interp.frame
    styles = ["ro", "g+", "y-"]
    if "PlotOptions" in opts:
        po = opts["PlotOptions"]
        if isinstance(po, Cell) and po.data.size == 3:
            styles = [text(x) for x in po.data.flatten(order="F")]
    interp.call_named("imshow", [*ax, canvas], 0, frame)
    held = bool(np.all(np.asarray(interp.call_named("ishold", list(ax), 1, frame)[0]) != 0))
    interp.call_named("hold", [*ax, char("on")], 0, frame)
    try:
        if p1.shape[0]:
            interp.call_named("plot", [*ax, p1[:, 0:1].T, p1[:, 1:2].T, char(styles[0])], 0, frame)
            interp.call_named("plot", [*ax, q2[:, 0:1].T, q2[:, 1:2].T, char(styles[1])], 0, frame)
            interp.call_named("plot", [*ax, np.vstack([p1[:, 0], q2[:, 0]]), np.vstack([p1[:, 1], q2[:, 1]]),
                                       char(styles[2])], 0, frame)
    finally:
        if not held:
            interp.call_named("hold", [*ax, char("off")], 0, frame)
    if nargout:
        return interp.call_named("gca", [], 1, frame)[0]
    return None
