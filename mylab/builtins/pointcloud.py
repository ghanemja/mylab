"""Point clouds: the pointCloud object, PLY and PCD files, display, downsampling, denoising, normals,
plane fitting, ICP registration, merging and transformation.

Everything here is NumPy/SciPy (k-d trees from scipy.spatial); OpenCV is not needed. Points are
M-by-3 [x y z]; an organized cloud (from a depth sensor or reconstructScene) keeps its M-by-N-by-3
layout in Location and becomes unorganized when points are selected or removed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import char
from ._cv import VisionObject, col, flag, fscalar, fvec, method, opt_text, options, row, scalar
from ._registry import builtin
from ._util import text
from .vision_geometry import GeoTform3, tform3_arg


def _tree(xyz: np.ndarray):
    from scipy.spatial import cKDTree
    return cKDTree(xyz)


class PointCloud(VisionObject):
    CLASS = "pointCloud"
    PROPS = ("Location", "Count", "XLimits", "YLimits", "ZLimits", "Color", "Normal", "Intensity")
    SETTABLE = frozenset(("Color", "Normal", "Intensity"))

    def __init__(self, xyz: np.ndarray, color=None, normal=None, intensity=None, shape=None, dtype=None):
        super().__init__()
        self.xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        n = self.xyz.shape[0]
        self.shape = shape  # (rows, cols) for an organized cloud
        self.dtype = dtype or np.float64
        self.color = None if color is None or np.size(color) == 0 else np.asarray(color).reshape(n, 3)
        self.normal = None if normal is None or np.size(normal) == 0 else np.asarray(normal, dtype=float).reshape(n, 3)
        self.intensity = None if intensity is None or np.size(intensity) == 0 else np.asarray(intensity).reshape(n)

    @property
    def n(self) -> int:
        return self.xyz.shape[0]

    def finite(self) -> np.ndarray:
        return np.all(np.isfinite(self.xyz), axis=1)

    def _layout(self, a: np.ndarray, k: int) -> np.ndarray:
        if self.shape is not None:
            return a.reshape(self.shape[0], self.shape[1], k) if k > 1 else a.reshape(self.shape)
        return a.reshape(-1, k) if k > 1 else a.reshape(-1, 1)

    def prop(self, name: str) -> Any:
        if name == "Location":
            return self._layout(self.xyz, 3).astype(self.dtype)
        if name == "Count":
            return scalar(self.n)
        if name in ("XLimits", "YLimits", "ZLimits"):
            k = "XYZ".index(name[0])
            v = self.xyz[self.finite(), k]
            return row([v.min(), v.max()]).astype(self.dtype) if v.size else np.zeros((0, 2))
        if name == "Color":
            return np.zeros((0, 0), dtype=np.uint8) if self.color is None else self._layout(self.color, 3)
        if name == "Normal":
            return np.zeros((0, 0)) if self.normal is None else self._layout(self.normal, 3).astype(self.dtype)
        if name == "Intensity":
            return np.zeros((0, 0)) if self.intensity is None else self._layout(self.intensity, 1)
        raise KeyError(name)

    def check_set(self, name: str, value: Any) -> Any:
        a = np.asarray(value)
        if a.size == 0:
            setattr(self, name.lower(), None)
            return value
        k = 1 if name == "Intensity" else 3
        if a.size == k and self.n != 1:  # one colour for every point
            a = np.tile(a.reshape(1, k), (self.n, 1))
        if a.size != self.n * k:
            raise MatlabError(f"pointCloud: {name} must have one {'value' if k == 1 else 'row'} per point.",
                              "mylab:pointCloud:sizeMismatch")
        a = a.reshape(self.n, k) if k > 1 else a.reshape(self.n)
        if name == "Color":
            self.color = a
        elif name == "Normal":
            self.normal = a.astype(float)
        else:
            self.intensity = a
        return value

    def subset(self, idx: np.ndarray, keep_shape: bool = False) -> "PointCloud":
        idx = np.asarray(idx)
        return PointCloud(self.xyz[idx], None if self.color is None else self.color[idx],
                          None if self.normal is None else self.normal[idx],
                          None if self.intensity is None else self.intensity[idx],
                          self.shape if keep_shape else None, self.dtype)

    def summary_text(self, in_cell: bool) -> str:
        return "1×1 pointCloud" if in_cell else "[1×1 pointCloud]"


def pc_arg(v: Any, fname: str) -> PointCloud:
    if isinstance(v, PointCloud):
        return v
    raise MatlabError(f"{fname}: expected a pointCloud object.", f"mylab:{fname}:invalidInput")


@builtin("pointCloud")
def _point_cloud(interp, args, nargout):
    fname = "pointCloud"
    pos, opts = options(list(args), ("Color", "Normal", "Intensity"), fname, 1)
    if not pos:
        raise MatlabError("pointCloud: expected pointCloud(xyzPoints, ...).", "MATLAB:minrhs")
    a = pos[0]
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "iuf":
        raise MatlabError("pointCloud: the points must be a numeric M-by-3 or M-by-N-by-3 array.")
    shape = None
    if a.ndim == 3 and a.shape[2] == 3:
        shape = a.shape[:2]
        xyz = a.reshape(-1, 3)
    elif a.ndim == 2 and (a.shape[1] == 3 or a.size == 0):
        xyz = a.reshape(-1, 3)
    else:
        raise MatlabError("pointCloud: the points must be an M-by-3 or M-by-N-by-3 array.",
                          "mylab:pointCloud:invalidPoints")
    pc = PointCloud(xyz, shape=shape, dtype=np.float32 if a.dtype == np.float32 else np.float64)
    for k in ("Color", "Normal", "Intensity"):
        if k in opts:
            pc.check_set(k, opts[k])
    return pc


# -- methods ---------------------------------------------------------------------------------------------

def _query_point(v, fname: str) -> np.ndarray:
    return fvec(v, fname, "the query point", 3)


@method(PointCloud, "findNearestNeighbors")
def _pc_knn(self: PointCloud, interp, args, nargout):
    pos, _ = options(list(args), ("Sort", "MaxLeafChecks"), "findNearestNeighbors", 2)
    if len(pos) != 2:
        raise MatlabError("findNearestNeighbors: expected findNearestNeighbors(ptCloud, point, K).", "MATLAB:minrhs")
    q = _query_point(pos[0], "findNearestNeighbors")
    k = int(fscalar(pos[1], "findNearestNeighbors", "K"))
    ok = np.nonzero(self.finite())[0]
    k = min(k, ok.size)
    if k <= 0:
        return np.zeros((0, 1)), np.zeros((0, 1))
    d, i = _tree(self.xyz[ok]).query(q, k=k)
    d, i = np.atleast_1d(d), np.atleast_1d(i)
    return col(ok[i] + 1), col(d)


@method(PointCloud, "findNeighborsInRadius")
def _pc_radius(self: PointCloud, interp, args, nargout):
    pos, opts = options(list(args), ("Sort", "MaxLeafChecks"), "findNeighborsInRadius", 2)
    q = _query_point(pos[0], "findNeighborsInRadius")
    r = fscalar(pos[1], "findNeighborsInRadius", "radius")
    ok = np.nonzero(self.finite())[0]
    d = np.sqrt(((self.xyz[ok] - q) ** 2).sum(1))
    sel = np.nonzero(d <= r)[0]
    if "Sort" in opts and flag(opts["Sort"]):
        sel = sel[np.argsort(d[sel], kind="stable")]
    return col(ok[sel] + 1), col(d[sel])


@method(PointCloud, "findPointsInROI")
def _pc_roi(self: PointCloud, interp, args, nargout):
    roi = fvec(args[0], "findPointsInROI", "roi", 6)
    x = self.xyz
    inside = (x[:, 0] >= roi[0]) & (x[:, 0] <= roi[1]) & (x[:, 1] >= roi[2]) & (x[:, 1] <= roi[3]) & \
        (x[:, 2] >= roi[4]) & (x[:, 2] <= roi[5])
    return col(np.nonzero(inside)[0] + 1)


@method(PointCloud, "select")
def _pc_select(self: PointCloud, interp, args, nargout):
    pos, opts = options(list(args), ("OutputSize",), "select", 2)
    if len(pos) == 2 and self.shape is not None:
        r = fvec(pos[0], "select", "row indices").astype(int) - 1
        c = fvec(pos[1], "select", "column indices").astype(int) - 1
        idx = np.ravel_multi_index((r, c), self.shape)
    else:
        a = np.asarray(pos[0])
        if a.dtype == bool:
            idx = np.nonzero(a.ravel(order="F") if self.shape is None else a.reshape(-1))[0]
        else:
            idx = a.astype(np.int64).ravel() - 1
    if idx.size and (idx.min() < 0 or idx.max() >= self.n):
        raise MatlabError(f"select: indices must be between 1 and {self.n}.", "MATLAB:badsubscript")
    full = "OutputSize" in opts and text(opts["OutputSize"]).lower() == "full"
    if full:
        out = PointCloud(np.full_like(self.xyz, np.nan), self.color, self.normal, self.intensity, self.shape,
                         self.dtype)
        out.xyz[idx] = self.xyz[idx]
        return out
    return self.subset(idx)


@method(PointCloud, "removeInvalidPoints")
def _pc_remove_invalid(self: PointCloud, interp, args, nargout):
    ok = self.finite()
    idx = np.nonzero(ok)[0]
    return self.subset(idx), col(idx + 1)


@method(PointCloud, "copy")
def _pc_copy(self: PointCloud, interp, args, nargout):
    return self.subset(np.arange(self.n), keep_shape=True)


# -- files -------------------------------------------------------------------------------------------------

_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h", "ushort": "H",
              "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I", "float": "f", "float32": "f",
              "double": "d", "float64": "d"}


def read_ply(path: str) -> PointCloud:
    with open(path, "rb") as f:
        data = f.read()
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise MatlabError(f"pcread: '{path}' is not a PLY file.", "mylab:pcread:badFile")
    nl = data.find(b"\n", end)
    header = data[:nl].decode("ascii", "replace").splitlines()
    body = data[nl + 1:]
    fmt, elements, cur = "ascii", [], None
    for line in header:
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            cur = {"name": t[1], "count": int(t[2]), "props": []}
            elements.append(cur)
        elif t[0] == "property" and cur is not None:
            if t[1] == "list":
                cur["props"].append((t[4], "list", t[2], t[3]))
            else:
                cur["props"].append((t[2], t[1]))
    endian = "<" if fmt == "binary_little_endian" else ">"
    verts: dict[str, np.ndarray] = {}
    if fmt == "ascii":
        lines = body.decode("ascii", "replace").split("\n")
        li = 0
        for el in elements:
            rows = []
            for _ in range(el["count"]):
                while li < len(lines) and not lines[li].strip():
                    li += 1
                rows.append(lines[li].split())
                li += 1
            if el["name"] == "vertex":
                names = [p[0] for p in el["props"] if p[1] != "list"]
                arr = np.array(rows, dtype=float).reshape(el["count"], -1)[:, :len(names)]
                verts = {nm: arr[:, k] for k, nm in enumerate(names)}
    else:
        off = 0
        for el in elements:
            if any(p[1] == "list" for p in el["props"]):
                if el["name"] == "vertex":
                    raise MatlabError("pcread: vertex elements with list properties are not supported.")
                # skip faces and similar elements after the vertices
                break
            dt = np.dtype([(p[0], endian + _PLY_TYPES[p[1]]) for p in el["props"]])
            arr = np.frombuffer(body, dtype=dt, count=el["count"], offset=off)
            off += dt.itemsize * el["count"]
            if el["name"] == "vertex":
                verts = {nm: arr[nm].astype(float) for nm in arr.dtype.names}
    if not all(k in verts for k in ("x", "y", "z")):
        raise MatlabError(f"pcread: '{path}' has no x, y, z vertex properties.", "mylab:pcread:badFile")
    xyz = np.column_stack([verts["x"], verts["y"], verts["z"]])
    color = None
    for names in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if all(k in verts for k in names):
            color = np.column_stack([verts[k] for k in names]).astype(np.uint8)
            break
    normal = np.column_stack([verts[k] for k in ("nx", "ny", "nz")]) if all(k in verts for k in ("nx", "ny", "nz")) \
        else None
    inten = verts.get("intensity", verts.get("scalar_intensity"))
    return PointCloud(xyz, color, normal, inten)


def write_ply(pc: PointCloud, path: str, binary: bool) -> None:
    props = [("x", "float"), ("y", "float"), ("z", "float")]
    cols = [pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]]
    if pc.normal is not None:
        props += [("nx", "float"), ("ny", "float"), ("nz", "float")]
        cols += [pc.normal[:, 0], pc.normal[:, 1], pc.normal[:, 2]]
    if pc.color is not None:
        c = _color_u8(pc.color)
        props += [("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]
        cols += [c[:, 0], c[:, 1], c[:, 2]]
    if pc.intensity is not None:
        props.append(("intensity", "float"))
        cols.append(pc.intensity.astype(float))
    head = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0", "comment written by mylab",
            f"element vertex {pc.n}"] + [f"property {t} {n}" for n, t in props] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if binary:
            dt = np.dtype([(n, "<" + _PLY_TYPES[t]) for n, t in props])
            rec = np.empty(pc.n, dtype=dt)
            for (n, _), c in zip(props, cols):
                rec[n] = c
            f.write(rec.tobytes())
        else:
            lines = []
            for i in range(pc.n):
                parts = [f"{int(c[i])}" if t == "uchar" else f"{c[i]:.9g}" for (n, t), c in zip(props, cols)]
                lines.append(" ".join(parts))
            f.write(("\n".join(lines) + ("\n" if lines else "")).encode("ascii"))


def _color_u8(c: np.ndarray) -> np.ndarray:
    if c.dtype == np.uint8:
        return c
    if c.dtype.kind == "f":
        return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)
    return np.clip(c, 0, 255).astype(np.uint8)


def read_pcd(path: str) -> PointCloud:
    with open(path, "rb") as f:
        data = f.read()
    pos, hdr = 0, {}
    while True:
        nl = data.find(b"\n", pos)
        if nl < 0:
            raise MatlabError(f"pcread: '{path}' is not a PCD file.", "mylab:pcread:badFile")
        line = data[pos:nl].decode("ascii", "replace").strip()
        pos = nl + 1
        if not line or line.startswith("#"):
            continue
        t = line.split()
        hdr[t[0].upper()] = t[1:]
        if t[0].upper() == "DATA":
            break
    fields = hdr.get("FIELDS", [])
    sizes = [int(s) for s in hdr.get("SIZE", [])]
    types = hdr.get("TYPE", [])
    counts = [int(c) for c in hdr.get("COUNT", ["1"] * len(fields))]
    n = int(hdr.get("POINTS", ["0"])[0])
    width = int(hdr.get("WIDTH", [str(n)])[0])
    height = int(hdr.get("HEIGHT", ["1"])[0])
    mode = hdr["DATA"][0].lower()
    body = data[pos:]
    kinds = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("I", 1): "i1",
             ("I", 2): "i2", ("I", 4): "i4", ("U", 8): "u8", ("I", 8): "i8"}
    dt = np.dtype([(f, "<" + kinds[(t.upper(), s)], (c,)) if c > 1 else (f, "<" + kinds[(t.upper(), s)])
                   for f, s, t, c in zip(fields, sizes, types, counts)])
    if mode == "ascii":
        vals = np.array(body.decode("ascii", "replace").split(), dtype=float)
        ncol = sum(counts)
        vals = vals[: n * ncol].reshape(n, ncol)
        rec = {}
        k = 0
        for f, c in zip(fields, counts):
            rec[f] = vals[:, k] if c == 1 else vals[:, k:k + c]
            k += c
        # rgb packed in a float: reinterpret the bits
        if "rgb" in rec and dt["rgb"].kind == "f":
            rec["rgb"] = np.asarray(rec["rgb"], dtype=np.float32).view(np.uint32)
        elif "rgb" in rec:
            rec["rgb"] = rec["rgb"].astype(np.uint32)
    elif mode == "binary":
        arr = np.frombuffer(body, dtype=dt, count=n)
        rec = {f: arr[f] for f in fields}
        if "rgb" in rec:
            rec["rgb"] = np.ascontiguousarray(rec["rgb"]).view(np.uint32) if rec["rgb"].dtype.itemsize == 4 \
                else rec["rgb"].astype(np.uint32)
    else:
        raise MatlabError("pcread: compressed PCD files (DATA binary_compressed) are not supported.",
                          "mylab:pcread:unsupported")
    xyz = np.column_stack([np.asarray(rec[k], dtype=float) for k in ("x", "y", "z")])
    color = None
    if "rgb" in rec or "rgba" in rec:
        v = np.asarray(rec.get("rgb", rec.get("rgba"))).astype(np.uint32)
        color = np.column_stack([(v >> 16) & 255, (v >> 8) & 255, v & 255]).astype(np.uint8)
    normal = np.column_stack([np.asarray(rec[k], float) for k in ("normal_x", "normal_y", "normal_z")]) \
        if all(k in rec for k in ("normal_x", "normal_y", "normal_z")) else None
    inten = np.asarray(rec["intensity"], dtype=float) if "intensity" in rec else None
    shape = (height, width) if height > 1 else None
    return PointCloud(xyz, color, normal, inten, shape)


def write_pcd(pc: PointCloud, path: str, binary: bool) -> None:
    fields, types, cols = ["x", "y", "z"], ["F", "F", "F"], [pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]]
    if pc.normal is not None:
        fields += ["normal_x", "normal_y", "normal_z"]
        types += ["F", "F", "F"]
        cols += [pc.normal[:, 0], pc.normal[:, 1], pc.normal[:, 2]]
    if pc.color is not None:
        c = _color_u8(pc.color).astype(np.uint32)
        packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        fields.append("rgb")
        types.append("U")
        cols.append(packed)
    if pc.intensity is not None:
        fields.append("intensity")
        types.append("F")
        cols.append(pc.intensity.astype(float))
    h, w = (pc.shape if pc.shape is not None else (1, pc.n))
    head = ["# .PCD v0.7 - written by mylab", "VERSION 0.7", "FIELDS " + " ".join(fields),
            "SIZE " + " ".join("4" for _ in fields), "TYPE " + " ".join(types),
            "COUNT " + " ".join("1" for _ in fields),
            f"WIDTH {w}", f"HEIGHT {h}", "VIEWPOINT 0 0 0 1 0 0 0", f"POINTS {pc.n}",
            f"DATA {'binary' if binary else 'ascii'}"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if binary:
            dt = np.dtype([(fn, "<f4" if t == "F" else "<u4") for fn, t in zip(fields, types)])
            rec = np.empty(pc.n, dtype=dt)
            for fn, c in zip(fields, cols):
                rec[fn] = c
            f.write(rec.tobytes())
        else:
            lines = [" ".join(f"{int(c[i])}" if t == "U" else f"{c[i]:.9g}" for c, t in zip(cols, types))
                     for i in range(pc.n)]
            f.write(("\n".join(lines) + ("\n" if lines else "")).encode("ascii"))


@builtin("pcread")
def _pcread(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("pcread: expected pcread(filename).", "MATLAB:minrhs")
    path = text(args[0])
    if not os.path.isfile(path):
        raise MatlabError(f"pcread: cannot find the file '{path}'.", "mylab:pcread:fileNotFound")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return read_ply(path)
    if ext == ".pcd":
        return read_pcd(path)
    raise MatlabError("pcread: only PLY and PCD files are supported.", "mylab:pcread:unsupported")


@builtin("pcwrite")
def _pcwrite(interp, args, nargout):
    pos, opts = options(list(args), ("Encoding", "PLYFormat"), "pcwrite", 2, lead=2)
    if len(pos) != 2:
        raise MatlabError("pcwrite: expected pcwrite(ptCloud, filename).", "MATLAB:minrhs")
    pc = pc_arg(pos[0], "pcwrite")
    path = text(pos[1])
    ext = os.path.splitext(path)[1].lower()
    enc_v = opts.get("Encoding", opts.get("PLYFormat"))
    enc = text(enc_v).lower() if enc_v is not None else "ascii"
    if enc not in ("ascii", "binary", "compressed"):
        raise MatlabError("pcwrite: Encoding must be 'ascii', 'binary' or 'compressed'.")
    if enc == "compressed":
        raise MatlabError("pcwrite: compressed PCD output is not supported; use 'binary'.", "mylab:pcwrite:unsupported")
    if ext == ".ply":
        write_ply(pc, path, enc == "binary")
    elif ext == ".pcd":
        write_pcd(pc, path, enc == "binary")
    else:
        raise MatlabError("pcwrite: the file name must end in .ply or .pcd.", "mylab:pcwrite:unsupported")
    return None


# -- display -----------------------------------------------------------------------------------------------

@builtin("pcshow")
def _pcshow(interp, args, nargout):
    pos, opts = options(list(args), ("MarkerSize", "VerticalAxis", "VerticalAxisDir", "Parent", "BackgroundColor",
                                     "ViewPlane", "ColorSource", "AxesVisibility", "Projection"), "pcshow", 2)
    if not pos:
        raise MatlabError("pcshow: expected pcshow(ptCloud) or pcshow(xyzPoints).", "MATLAB:minrhs")
    if isinstance(pos[0], PointCloud):
        pc = pos[0]
        xyz, color = pc.xyz, pc.color
    else:
        a = np.asarray(pos[0], dtype=float)
        xyz = a.reshape(-1, 3)
        color = None
    if len(pos) > 1:
        color = np.asarray(pos[1])
    ok = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[ok]
    ms = fscalar(opts["MarkerSize"], "pcshow", "MarkerSize") if "MarkerSize" in opts else 6.0
    ax = [opts["Parent"]] if "Parent" in opts else []
    frame = interp.frame
    if color is not None and np.size(color) == np.size(ok) * 3:
        c = np.asarray(color).reshape(-1, 3)[ok]
        c = c.astype(float) / 255.0 if c.dtype == np.uint8 else c.astype(float)
    elif color is not None and np.size(color) == 3:
        c = np.asarray(color, dtype=float).reshape(1, 3)
        c = c / 255.0 if c.max() > 1 else c
    else:
        c = xyz[:, 2:3].copy()  # colour by height
    args_ = [*ax, xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], scalar(ms), c, char("filled")]
    h = interp.call_named("scatter3", args_, 1, frame)[0]
    interp.call_named("axis", [*ax, char("equal")], 0, frame)
    return h if nargout else None


@builtin("pcshowpair")
def _pcshowpair(interp, args, nargout):
    pos, opts = options(list(args), ("MarkerSize", "Parent", "BlendFactor", "VerticalAxis", "VerticalAxisDir"),
                        "pcshowpair", 2)
    a, b = pc_arg(pos[0], "pcshowpair"), pc_arg(pos[1], "pcshowpair")
    frame = interp.frame
    held = bool(np.all(np.asarray(interp.call_named("ishold", [], 1, frame)[0]) != 0))
    _pcshow(interp, [a, np.array([[255.0, 0.0, 255.0]])], 0)
    interp.call_named("hold", [char("on")], 0, frame)
    _pcshow(interp, [b, np.array([[0.0, 255.0, 0.0]])], 0)
    if not held:
        interp.call_named("hold", [char("off")], 0, frame)
    return None


# -- processing --------------------------------------------------------------------------------------------

def _grid_average(pc: PointCloud, step: float) -> PointCloud:
    ok = np.nonzero(pc.finite())[0]
    x = pc.xyz[ok]
    if x.shape[0] == 0:
        return pc.subset(ok)
    key = np.floor((x - x.min(0)) / step).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m = counts.size

    def avg(a):
        out = np.zeros((m, a.shape[1]))
        np.add.at(out, inv, a)
        return out / counts[:, None]
    xyz = avg(x)
    color = None if pc.color is None else np.round(avg(pc.color[ok].astype(float))).astype(pc.color.dtype)
    normal = None
    if pc.normal is not None:
        normal = avg(pc.normal[ok])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    inten = None if pc.intensity is None else avg(pc.intensity[ok].reshape(-1, 1).astype(float)).ravel()
    return PointCloud(xyz, color, normal, inten, None, pc.dtype)


@builtin("pcdownsample")
def _pcdownsample(interp, args, nargout):
    fname = "pcdownsample"
    if len(args) < 3:
        raise MatlabError("pcdownsample: expected pcdownsample(ptCloud, 'random', percentage) or "
                          "pcdownsample(ptCloud, 'gridAverage', gridStep).", "MATLAB:minrhs")
    pc = pc_arg(args[0], fname)
    meth = text(args[1]).lower()
    if meth == "random":
        p = fscalar(args[2], fname, "the percentage")
        if not 0 <= p <= 1:
            raise MatlabError("pcdownsample: the percentage must be between 0 and 1.")
        from .core import _rng
        ok = np.nonzero(pc.finite())[0]
        k = int(round(p * ok.size))
        idx = np.sort(ok[np.argsort(_rng(interp).random(ok.size), kind="stable")[:k]])
        return pc.subset(idx)
    if meth == "gridaverage":
        step = fscalar(args[2], fname, "gridStep")
        if step <= 0:
            raise MatlabError("pcdownsample: gridStep must be positive.")
        return _grid_average(pc, step)
    if meth == "nonuniformgridsample":
        mx = int(fscalar(args[2], fname, "maxNumPoints"))
        if mx < 6:
            raise MatlabError("pcdownsample: maxNumPoints must be at least 6.")
        ok = np.nonzero(pc.finite())[0]
        keep = []
        stack = [ok]
        while stack:
            idx = stack.pop()
            if idx.size <= mx:
                if idx.size:
                    keep.append(idx[len(idx) // 2])
                continue
            x = pc.xyz[idx]
            d = int(np.argmax(x.max(0) - x.min(0)))
            order = idx[np.argsort(x[:, d], kind="stable")]
            h = order.size // 2
            stack += [order[:h], order[h:]]
        return pc.subset(np.sort(np.array(keep, dtype=np.int64)))
    raise MatlabError("pcdownsample: the method must be 'random', 'gridAverage' or 'nonuniformGridSample'.")


@builtin("pcdenoise")
def _pcdenoise(interp, args, nargout):
    fname = "pcdenoise"
    pos, opts = options(list(args), ("NumNeighbors", "Threshold", "PreserveStructure"), fname, 1)
    pc = pc_arg(pos[0] if pos else None, fname)
    k = int(fscalar(opts["NumNeighbors"], fname, "NumNeighbors")) if "NumNeighbors" in opts else 4
    th = fscalar(opts["Threshold"], fname, "Threshold") if "Threshold" in opts else 1.0
    ok = np.nonzero(pc.finite())[0]
    x = pc.xyz[ok]
    if x.shape[0] <= k:
        return pc.subset(ok), col(ok + 1), np.zeros((0, 1))
    d, _ = _tree(x).query(x, k=k + 1)
    mean_d = d[:, 1:].mean(1)
    limit = mean_d.mean() + th * mean_d.std()
    good = mean_d <= limit
    inl, out = ok[good], ok[~good]
    keep_struct = flag(opts["PreserveStructure"]) if "PreserveStructure" in opts else False
    if keep_struct:
        res = pc.subset(np.arange(pc.n), keep_shape=True)
        res.xyz[out] = np.nan
    else:
        res = pc.subset(inl)
    return res, col(inl + 1), col(out + 1)


def normals(x: np.ndarray, k: int, sensor=None) -> np.ndarray:
    n = np.full((x.shape[0], 3), np.nan)
    ok = np.nonzero(np.all(np.isfinite(x), axis=1))[0]
    if ok.size < 3:
        return n
    xs = x[ok]
    _, nb = _tree(xs).query(xs, k=min(k, ok.size))
    P = xs[nb]  # (m, k, 3)
    P = P - P.mean(axis=1, keepdims=True)
    C = np.einsum("mki,mkj->mij", P, P)
    _, v = np.linalg.eigh(C)
    nn = v[:, :, 0]
    # orient towards the sensor (the origin by default), as for a camera-centred cloud
    view = (np.zeros(3) if sensor is None else sensor) - xs
    flip = np.sum(nn * view, axis=1) < 0
    nn[flip] *= -1
    n[ok] = nn
    return n


@builtin("pcnormals")
def _pcnormals(interp, args, nargout):
    if not args:
        raise MatlabError("pcnormals: expected pcnormals(ptCloud) or pcnormals(ptCloud, K).", "MATLAB:minrhs")
    pc = pc_arg(args[0], "pcnormals")
    k = int(fscalar(args[1], "pcnormals", "K")) if len(args) > 1 else 6
    if k < 3:
        raise MatlabError("pcnormals: K must be at least 3.")
    nrm = normals(pc.xyz, k)
    return pc._layout(nrm, 3)


# -- plane fitting -----------------------------------------------------------------------------------------

class PlaneModel(VisionObject):
    CLASS = "planeModel"
    PROPS = ("Parameters", "Normal")

    def __init__(self, params: np.ndarray):
        super().__init__()
        self.params = np.asarray(params, dtype=float).reshape(4)

    def prop(self, name: str) -> Any:
        if name == "Parameters":
            return row(self.params)
        return row(self.params[:3])


@method(PlaneModel, "plot")
def _plane_plot(self, interp, args, nargout):
    a, b, c, d = self.params
    ax = interp.call_named("gca", [], 1, interp.frame)[0]
    xl = np.asarray(interp.call_named("xlim", [ax], 1, interp.frame)[0]).ravel()
    yl = np.asarray(interp.call_named("ylim", [ax], 1, interp.frame)[0]).ravel()
    X, Y = np.meshgrid(np.linspace(xl[0], xl[1], 2), np.linspace(yl[0], yl[1], 2))
    if abs(c) < 1e-12:
        raise MatlabError("plot: vertical planes cannot be drawn over x-y limits.")
    Z = -(a * X + b * Y + d) / c
    interp.call_named("surf", [X, Y, Z, char("FaceAlpha"), scalar(0.3), char("EdgeColor"), char("none")], 0,
                      interp.frame)
    return None


@builtin("planeModel")
def _plane_model(interp, args, nargout):
    p = fvec(args[0], "planeModel", "parameters", 4)
    return PlaneModel(p / max(np.linalg.norm(p[:3]), 1e-300))


def _plane_from(p3: np.ndarray):
    nrm = np.cross(p3[1] - p3[0], p3[2] - p3[0])
    s = np.linalg.norm(nrm)
    if s < 1e-12:
        return None
    nrm = nrm / s
    return np.append(nrm, -nrm @ p3[0])


@builtin("pcfitplane")
def _pcfitplane(interp, args, nargout):
    fname = "pcfitplane"
    pos, opts = options(list(args), ("SampleIndices", "MaxNumTrials", "Confidence"), fname, 4)
    if len(pos) < 2:
        raise MatlabError("pcfitplane: expected pcfitplane(ptCloud, maxDistance).", "MATLAB:minrhs")
    pc = pc_arg(pos[0], fname)
    maxd = fscalar(pos[1], fname, "maxDistance")
    ref = fvec(pos[2], fname, "referenceVector", 3) if len(pos) > 2 else None
    maxang = np.deg2rad(fscalar(pos[3], fname, "maxAngularDistance")) if len(pos) > 3 else np.deg2rad(5.0)
    trials = int(fscalar(opts["MaxNumTrials"], fname, "MaxNumTrials")) if "MaxNumTrials" in opts else 1000
    conf = fscalar(opts["Confidence"], fname, "Confidence") if "Confidence" in opts else 99.0
    cand = np.nonzero(pc.finite())[0]
    if "SampleIndices" in opts:
        cand = np.intersect1d(cand, fvec(opts["SampleIndices"], fname, "SampleIndices").astype(int) - 1)
    x = pc.xyz[cand]
    if x.shape[0] < 3:
        raise MatlabError("pcfitplane: at least three valid points are needed.", "mylab:pcfitplane:notEnoughPts")
    if ref is not None:
        ref = ref / np.linalg.norm(ref)
    from .core import _rng
    rng = _rng(interp)
    best, best_cost = None, np.inf
    needed, it = trials, 0
    while it < min(needed, trials):
        it += 1
        pl = _plane_from(x[rng.choice(x.shape[0], 3, replace=False)])
        if pl is None:
            continue
        if ref is not None:
            ang = np.arccos(np.clip(abs(pl[:3] @ ref), 0, 1))
            if ang > maxang:
                continue
        d = np.abs(x @ pl[:3] + pl[3])
        cost = np.minimum(d, maxd).sum()
        if cost < best_cost:
            best_cost, best = cost, pl
            ratio = (d < maxd).mean()
            den = np.log(max(1 - ratio ** 3, 1e-12))
            if den < 0:
                needed = int(np.ceil(np.log(max(1 - conf / 100, 1e-12)) / den))
    if best is None:
        raise MatlabError("pcfitplane: no plane was found; try a larger maxDistance.", "mylab:pcfitplane:noPlane")
    inl = np.abs(x @ best[:3] + best[3]) < maxd
    if inl.sum() >= 3:  # least-squares refit on the inliers
        P = x[inl]
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c)
        nrm = Vt[-1]
        if nrm @ best[:3] < 0:
            nrm = -nrm
        ok_ref = ref is None or np.arccos(np.clip(abs(nrm @ ref), 0, 1)) <= maxang
        if ok_ref:
            best = np.append(nrm, -nrm @ c)
            inl = np.abs(x @ best[:3] + best[3]) < maxd
    dist = np.abs(x @ best[:3] + best[3])
    inliers = cand[inl]
    outliers = np.setdiff1d(np.arange(pc.n), inliers)
    mean_err = float(dist[inl].mean()) if inl.any() else 0.0
    return PlaneModel(best), col(inliers + 1), col(outliers + 1), scalar(mean_err)


# -- registration ------------------------------------------------------------------------------------------

def _best_rigid(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rigid transform (4x4) minimising the distances from the transformed a to b."""
    w = np.ones(len(a)) if w is None else w
    ws = w.sum()
    ma = (a * w[:, None]).sum(0) / ws
    mb = (b * w[:, None]).sum(0) / ws
    H = ((a - ma) * w[:, None]).T @ (b - mb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mb - R @ ma
    return T


def _point_to_plane(a: np.ndarray, b: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """One linearised point-to-plane step (small-angle approximation)."""
    A = np.column_stack([np.cross(a, nb), nb])
    rhs = np.sum((b - a) * nb, axis=1)
    x, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    from .vision_geometry import rodrigues
    T = np.eye(4)
    T[:3, :3] = rodrigues(x[:3])
    T[:3, 3] = x[3:]
    return T


@builtin("pcregistericp")
def _pcregistericp(interp, args, nargout):
    fname = "pcregistericp"
    pos, opts = options(list(args), ("Metric", "MaxIterations", "Tolerance", "InitialTransform", "InlierRatio",
                                     "InlierDistance", "Extrapolate", "Verbose", "UseAllMatches"), fname, 2)
    if len(pos) != 2:
        raise MatlabError("pcregistericp: expected pcregistericp(moving, fixed, ...).", "MATLAB:minrhs")
    mov, fix = pc_arg(pos[0], fname), pc_arg(pos[1], fname)
    metric = opt_text(opts, "Metric", "pointToPoint", ("pointToPoint", "pointToPlane", "planeToPlane"), fname)
    iters = int(fscalar(opts["MaxIterations"], fname, "MaxIterations")) if "MaxIterations" in opts else 20
    tol = fvec(opts["Tolerance"], fname, "Tolerance", 2) if "Tolerance" in opts else np.array([0.01, 0.5])
    ratio = fscalar(opts["InlierRatio"], fname, "InlierRatio") if "InlierRatio" in opts else 1.0
    in_dist = fscalar(opts["InlierDistance"], fname, "InlierDistance") if "InlierDistance" in opts else None
    T = tform3_arg(opts["InitialTransform"], fname).A.copy() if "InitialTransform" in opts else np.eye(4)
    a0 = mov.xyz[mov.finite()]
    fx = fix.xyz[fix.finite()]
    if a0.shape[0] < 3 or fx.shape[0] < 3:
        raise MatlabError("pcregistericp: both point clouds need at least three valid points.")
    tree = _tree(fx)
    fn = None
    if metric != "pointToPoint":
        fn = fix.normal[fix.finite()] if fix.normal is not None else normals(fx, 6)
        fn = np.nan_to_num(fn)
    rmse = np.inf
    for _ in range(iters):
        a = a0 @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(a)
        keep = np.ones(len(d), dtype=bool)
        if in_dist is not None:
            keep &= d <= in_dist
        elif ratio < 1:
            keep &= d <= np.quantile(d, ratio)
        if keep.sum() < 3:
            break
        if metric == "pointToPoint":
            dT = _best_rigid(a[keep], fx[j[keep]])
        else:
            dT = _point_to_plane(a[keep], fx[j[keep]], fn[j[keep]])
        T = dT @ T
        rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
        ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(dT[:3, 3]) < tol[0] and ang < tol[1]:
            break
    a = a0 @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(a)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    tf = GeoTform3("rigidtform3d", T)
    if nargout > 1:
        return tf, _transform_pc(mov, T), scalar(rmse)
    return tf


def _transform_pc(pc: PointCloud, A: np.ndarray) -> PointCloud:
    h = np.column_stack([pc.xyz, np.ones(pc.n)]) @ A.T
    xyz = h[:, :3] / h[:, 3:4]
    nrm = None
    if pc.normal is not None:
        L = A[:3, :3]
        nrm = pc.normal @ np.linalg.inv(L)  # normals transform with the inverse transpose
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    return PointCloud(xyz, pc.color, nrm, pc.intensity, pc.shape, pc.dtype)


@builtin("pctransform")
def _pctransform(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("pctransform: expected pctransform(ptCloud, tform) or pctransform(ptCloud, D).",
                          "MATLAB:minrhs")
    pc = pc_arg(args[0], "pctransform")
    if isinstance(args[1], GeoTform3):
        return _transform_pc(pc, args[1].A)
    D = np.asarray(args[1], dtype=float)
    if D.size != pc.n * 3:
        raise MatlabError("pctransform: the displacement field must have the size of the point locations.")
    out = pc.subset(np.arange(pc.n), keep_shape=True)
    out.xyz = pc.xyz + D.reshape(-1, 3)
    return out


@builtin("pcmerge")
def _pcmerge(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("pcmerge: expected pcmerge(ptCloudA, ptCloudB, gridStep).", "MATLAB:minrhs")
    a, b = pc_arg(args[0], "pcmerge"), pc_arg(args[1], "pcmerge")
    step = fscalar(args[2], "pcmerge", "gridStep")

    def both(x, y, k):
        if x is None or y is None:
            return None
        return np.concatenate([x.reshape(-1, k) if k > 1 else x, y.reshape(-1, k) if k > 1 else y])
    merged = PointCloud(np.vstack([a.xyz, b.xyz]), both(a.color, b.color, 3), both(a.normal, b.normal, 3),
                        both(a.intensity, b.intensity, 1))
    return _grid_average(merged, step)


@builtin("pcsegdist")
def _pcsegdist(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("pcsegdist: expected pcsegdist(ptCloud, minDistance).", "MATLAB:minrhs")
    pc = pc_arg(args[0], "pcsegdist")
    dmin = fscalar(args[1], "pcsegdist", "minDistance")
    from scipy.sparse.csgraph import connected_components
    ok = np.nonzero(pc.finite())[0]
    labels = np.zeros(pc.n)
    if ok.size:
        pairs = _tree(pc.xyz[ok]).query_pairs(dmin, output_type="ndarray")
        from scipy.sparse import coo_matrix
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(ok.size, ok.size))
        ncomp, lab = connected_components(g, directed=False)
        labels[ok] = lab + 1
    else:
        ncomp = 0
    return pc._layout(labels, 1).astype(np.uint32), scalar(ncomp)

