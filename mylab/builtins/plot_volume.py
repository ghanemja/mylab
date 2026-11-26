"""Volume and vector-field visualisation, lighting, the camera, and object groups.

* scalar volumes: ``isosurface``, ``isocaps``, ``isonormals``, ``slice``, ``contourslice``, ``smooth3``,
  ``reducepatch``, ``surf2patch``, ``tetramesh``, ``contourc``, ``contour3``, ``fimplicit3``;
* vector fields: ``stream2``, ``stream3``, ``streamline``, ``streamslice``, ``streamtube``, ``coneplot``;
* lights (``light``, ``camlight``, ``lightangle``, ``lighting``, ``material``): faces of surfaces and 3-D
  patches are shaded with one colour per face from the lights of their axes;
* the camera (``campos``, ``camtarget``, ``camva``, ``camorbit``, ...): mapped to the azimuth, elevation,
  roll and zoom of matplotlib's 3-D axes, which always look at the centre of the plot box;
* groups (``hggroup``, ``hgtransform`` with ``makehgtform``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, char
from . import plotting as P
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, text

_EMPTY = np.zeros((0, 0))


def _real(v) -> np.ndarray:
    return np.real(fnum(v)).astype(float)


def _vec(v) -> np.ndarray:
    return _real(v).flatten(order="F")


def _lead_axes(interp, args: list, projection: str | None = "3d", fresh: bool = True):
    """(axes, remaining args): a leading axes handle or a 'Parent' option selects the axes."""
    args = list(args)
    st = P._state(interp)
    target = None
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in P._AXES_KINDS:
        target = args.pop(0).obj
    k = 0
    while k < len(args) - 1:
        if is_text(args[k]) and text(args[k]).lower() == "parent" and isinstance(args[k + 1], P.GraphicsHandle) \
                and args[k + 1].kind in P._AXES_KINDS:
            target = args[k + 1].obj
            del args[k:k + 2]
            continue
        k += 1
    prev = st.override_ax
    if target is not None:
        st.override_ax = target
    try:
        ax = P._axes(interp, fresh=fresh, projection=projection)
    finally:
        st.override_ax = prev
    return ax, args


def _current_axes(interp, args: list):
    args = list(args)
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in P._AXES_KINDS:
        return args.pop(0).obj, args
    return P._axes(interp, fresh=False), args


def _cmap(interp):
    import matplotlib as mpl
    c = P._current_cmap(interp)
    return mpl.colormaps[c] if isinstance(c, str) else c


# -- volume grids --------------------------------------------------------------------------------------------

def _volume_args(args: list, name: str, extra: int):
    """(x, y, z axis vectors, V, the ``extra`` arguments after the volume) for (V, ...) or (X, Y, Z, V, ...)."""
    nums = list(args)
    if len(nums) >= 4 + extra or (len(nums) >= 4 and _real(nums[3]).ndim == 3 and _real(nums[0]).size > 1
                                  and _real(nums[0]).ndim >= 1 and len(nums) > extra + 3):
        X, Y, Z, V = (_real(a) for a in nums[:4])
        rest = nums[4:]
        if V.ndim != 3:
            raise MatlabError(f"{name}: the volume data must be a 3-D array.")
        ny, nx, nz = V.shape
        x = X[0, :, 0] if X.ndim == 3 else X.flatten()
        y = Y[:, 0, 0] if Y.ndim == 3 else Y.flatten()
        z = Z[0, 0, :] if Z.ndim == 3 else Z.flatten()
        if x.size != nx or y.size != ny or z.size != nz:
            raise MatlabError(f"{name}: X, Y and Z must match the size of the volume.")
    else:
        if not nums:
            raise MatlabError(f"{name} needs volume data.")
        V = _real(nums[0])
        rest = nums[1:]
        if V.ndim != 3:
            raise MatlabError(f"{name}: the volume data must be a 3-D array.")
        ny, nx, nz = V.shape
        x, y, z = np.arange(1.0, nx + 1), np.arange(1.0, ny + 1), np.arange(1.0, nz + 1)
    return x, y, z, V, rest


def _interp3(x, y, z, V, px, py, pz, method: str = "linear") -> np.ndarray:
    """Values of the volume at points (NaN outside it). V is indexed (y, x, z) like meshgrid arrays."""
    from scipy.interpolate import RegularGridInterpolator

    def increasing(a, axis, vol):
        if a.size > 1 and a[0] > a[-1]:
            return a[::-1], np.flip(vol, axis=axis)
        return a, vol
    y2, V = increasing(y, 0, V)
    x2, V = increasing(x, 1, V)
    z2, V = increasing(z, 2, V)
    f = RegularGridInterpolator((y2, x2, z2), V, method="nearest" if method == "nearest" else "linear",
                                bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([np.ravel(py), np.ravel(px), np.ravel(pz)])
    return f(pts).reshape(np.shape(px))


# -- isosurfaces ----------------------------------------------------------------------------------------------------

# The six tetrahedra that fill a cube; the corners are numbered by (dy, dx, dz) offsets below.
_CORNERS = np.array([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)])
_TETS = np.array([(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)])


def marching_tetrahedra(x, y, z, V, iso: float):
    """Triangles of the surface V == iso: (vertices n-by-3 as [x y z], faces m-by-3, 0-based)."""
    ny, nx, nz = V.shape
    if min(ny, nx, nz) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    iy, ix, iz = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), np.arange(nz - 1), indexing="ij")
    iy, ix, iz = iy.ravel(), ix.ravel(), iz.ravel()
    # corner values and coordinates of every cube: arrays (ncubes, 8)
    cv = np.empty((iy.size, 8))
    cx = np.empty((iy.size, 8))
    cy = np.empty((iy.size, 8))
    cz = np.empty((iy.size, 8))
    for k, (dy, dx, dz) in enumerate(_CORNERS):
        cv[:, k] = V[iy + dy, ix + dx, iz + dz]
        cx[:, k] = x[ix + dx]
        cy[:, k] = y[iy + dy]
        cz[:, k] = z[iz + dz]
    tris = []
    for tet in _TETS:
        v = cv[:, tet]
        ok = np.all(np.isfinite(v), axis=1)
        inside = v > iso
        count = inside.sum(axis=1)
        px, py, pz = cx[:, tet], cy[:, tet], cz[:, tet]

        def cut(rows, a, b):
            """Point on edge a-b of each selected tetrahedron where the value crosses iso."""
            va, vb = v[rows, a], v[rows, b]
            t = (iso - va) / np.where(vb != va, vb - va, 1.0)
            t = np.clip(t, 0.0, 1.0)
            return np.column_stack([px[rows, a] + t * (px[rows, b] - px[rows, a]),
                                    py[rows, a] + t * (py[rows, b] - py[rows, a]),
                                    pz[rows, a] + t * (pz[rows, b] - pz[rows, a])])
        for lone_inside in (True, False):
            # one corner on its own side of the surface: one triangle
            for a in range(4):
                others = [b for b in range(4) if b != a]
                if lone_inside:
                    sel = ok & (count == 1) & inside[:, a]
                else:
                    sel = ok & (count == 3) & ~inside[:, a]
                rows = np.flatnonzero(sel)
                if rows.size:
                    p0, p1, p2 = (cut(rows, a, b) for b in others)
                    tris.append(np.stack([p0, p1, p2] if lone_inside else [p0, p2, p1], axis=1))
        # two corners on each side: a quadrilateral, drawn as two triangles
        for a, b in ((0, 1), (0, 2), (0, 3)):
            c, d = [k for k in range(4) if k not in (a, b)]
            for flip in (False, True):
                sel = ok & (count == 2) & (inside[:, a] == (not flip)) & (inside[:, b] == (not flip))
                rows = np.flatnonzero(sel)
                if rows.size:
                    q0, q1, q2, q3 = cut(rows, a, c), cut(rows, a, d), cut(rows, b, d), cut(rows, b, c)
                    tris.append(np.stack([q0, q1, q2], axis=1))
                    tris.append(np.stack([q0, q2, q3], axis=1))
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    T = np.concatenate(tris, axis=0)            # (ntri, 3, 3)
    pts = T.reshape(-1, 3)
    scale = max(float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z)), 1e-300)
    key = np.round(pts / scale * 1e9).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    verts = pts[first]
    faces = np.asarray(inverse).reshape(-1, 3)
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[good]
    # orient every triangle so that its normal points toward smaller values of V
    if faces.size:
        gy, gx, gz = np.gradient(np.nan_to_num(V), y, x, z)
        c = verts[faces].mean(axis=1)
        g = np.column_stack([_interp3(x, y, z, gx, c[:, 0], c[:, 1], c[:, 2]),
                             _interp3(x, y, z, gy, c[:, 0], c[:, 1], c[:, 2]),
                             _interp3(x, y, z, gz, c[:, 0], c[:, 1], c[:, 2])])
        nrm = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
        wrong = np.einsum("ij,ij->i", nrm, np.nan_to_num(g)) > 0
        faces[wrong] = faces[wrong][:, ::-1]
    return verts, faces


def _fv_struct(verts, faces, colors=None) -> Struct:
    d = {"vertices": np.asarray(verts, dtype=float).reshape(-1, 3),
         "faces": np.asarray(faces, dtype=float).reshape(-1, 3) + 1.0}
    if colors is not None:
        d["facevertexcdata"] = np.asarray(colors, dtype=float).reshape(-1, 1)
    return Struct.scalar(d)


def _draw_fv(interp, ax, verts, faces, colors=None, face_color=None, edge="k", nargout: int = 0,
             alpha: float | None = None):
    """Add a triangle (or polygon) mesh to 3-D axes as one patch object; returns its handle."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys = [verts[np.asarray(f, dtype=int)] for f in faces]
    if not polys:
        return None
    kw: dict = {"edgecolor": edge, "linewidth": 0.3}
    if alpha is not None:
        kw["alpha"] = alpha
    coll = Poly3DCollection(polys, **kw)
    if colors is not None:
        vals = np.array([float(np.mean(np.asarray(colors, dtype=float).flatten()[np.asarray(f, dtype=int)]))
                         for f in faces])
        coll.set_cmap(P._current_cmap(interp))
        coll.set_array(vals)
    else:
        coll.set_facecolor(face_color if face_color is not None else (0.0, 0.447, 0.741))
    had = ax.has_data()
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2], had)
    h = P._wrap(interp, "patch", coll)
    h.extra.update({"axes": ax, "figure": ax.figure, "_polys": [np.asarray(q, dtype=float) for q in polys],
                    "Vertices": np.asarray(verts, dtype=float),
                    "Faces": np.asarray(faces, dtype=float) + 1.0 if np.asarray(faces).ndim == 2 else None})
    if h.extra["Faces"] is None:
        del h.extra["Faces"]
    relight(interp, ax)
    P._redraw(interp)
    return h


@builtin("isosurface")
def _isosurface(interp, args, nargout):
    """isosurface(V, isovalue) or isosurface(X, Y, Z, V, isovalue): the surface where V equals isovalue."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    flags = [text(a).lower() for a in args if is_text(a)]
    for f in flags:
        if f not in ("verbose", "noshare"):
            raise MatlabError(f"isosurface: unknown option '{f}'.")
    nums = [a for a in args if not is_text(a)]
    if len(nums) in (1, 4):  # no isovalue: use the middle of the data range
        x, y, z, V, rest = _volume_args(nums, "isosurface", 0)
        f = V[np.isfinite(V)]
        rest = [np.array([[float((f.min() + f.max()) / 2)]])] if f.size else [np.zeros((1, 1))]
    else:
        x, y, z, V, rest = _volume_args(nums, "isosurface", 1 if len(nums) in (2, 5) else 2)
    if not rest:
        raise MatlabError("isosurface needs an isovalue.")
    iso = float(_vec(rest[0])[0])
    color_vol = _real(rest[1]) if len(rest) > 1 else None
    if color_vol is not None and color_vol.shape != V.shape:
        raise MatlabError("isosurface: the colour data must be the same size as the volume.")
    verts, faces = marching_tetrahedra(x, y, z, V, iso)
    colors = None
    if color_vol is not None and verts.size:
        colors = _interp3(x, y, z, color_vol, verts[:, 0], verts[:, 1], verts[:, 2])
    if nargout == 0:
        ax, _ = _lead_axes(interp, lead, projection="3d", fresh=False)
        if verts.size:
            _draw_fv(interp, ax, verts, faces, colors, face_color=(0.85, 0.2, 0.2) if colors is None else None,
                     edge="none")
        return None
    if nargout == 1:
        return _fv_struct(verts, faces, colors)
    out = (faces.astype(float) + 1.0, verts, colors.reshape(-1, 1) if colors is not None else np.zeros((0, 1)))
    return out[:nargout]


def _clip_triangles(P3, vals, iso: float, above: bool):
    """Clip triangles (n, 3, 3) with corner values (n, 3) to the part where the value is above/below iso."""
    out_p, out_v = [], []
    keep_fn = (lambda v: v >= iso) if above else (lambda v: v <= iso)
    for tri, v in zip(P3, vals):
        if not np.all(np.isfinite(v)):
            continue
        keep = keep_fn(v)
        if keep.all():
            out_p.append(tri)
            out_v.append(v)
            continue
        if not keep.any():
            continue
        pts, pv = [], []
        for i in range(3):
            j = (i + 1) % 3
            if keep[i]:
                pts.append(tri[i])
                pv.append(v[i])
            if keep[i] != keep[j]:
                t = (iso - v[i]) / (v[j] - v[i])
                pts.append(tri[i] + t * (tri[j] - tri[i]))
                pv.append(iso)
        for k in range(1, len(pts) - 1):
            out_p.append(np.array([pts[0], pts[k], pts[k + 1]]))
            out_v.append(np.array([pv[0], pv[k], pv[k + 1]]))
    return out_p, out_v


@builtin("isocaps")
def _isocaps(interp, args, nargout):
    """isocaps(V, isovalue): the parts of the volume's outer faces where the data lies above the isovalue."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    words = [text(a).lower() for a in args if is_text(a)]
    nums = [a for a in args if not is_text(a)]
    enclose, which = "above", "all"
    for w in words:
        if w in ("above", "below"):
            enclose = w
        elif w in ("all", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            which = w
        elif w != "verbose":
            raise MatlabError(f"isocaps: unknown option '{w}'.")
    x, y, z, V, rest = _volume_args(nums, "isocaps", 1)
    if not rest:
        raise MatlabError("isocaps needs an isovalue.")
    iso = float(_vec(rest[0])[0])
    tris, vals = [], []
    planes = [("xmin", 1, 0), ("xmax", 1, -1), ("ymin", 0, 0), ("ymax", 0, -1), ("zmin", 2, 0), ("zmax", 2, -1)]
    for name, axis, idx in planes:
        if which not in ("all", name):
            continue
        S = np.take(V, idx, axis=axis)
        if axis == 0:      # S is (x, z)
            A, B = np.meshgrid(x, z, indexing="ij")
            coords = lambda a, b, c=y[idx]: np.stack([a, np.full_like(a, c), b], axis=-1)  # noqa: E731
        elif axis == 1:    # S is (y, z)
            A, B = np.meshgrid(y, z, indexing="ij")
            coords = lambda a, b, c=x[idx]: np.stack([np.full_like(a, c), a, b], axis=-1)  # noqa: E731
        else:              # S is (y, x)
            A, B = np.meshgrid(y, x, indexing="ij")
            coords = lambda a, b, c=z[idx]: np.stack([b, a, np.full_like(a, c)], axis=-1)  # noqa: E731
        Pc = coords(A, B)
        n0, n1 = S.shape
        for i in range(n0 - 1):
            for j in range(n1 - 1):
                quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                for a, b, c in ((0, 1, 2), (0, 2, 3)):
                    idxs = [quad[a], quad[b], quad[c]]
                    tris.append(np.array([Pc[p] for p in idxs]))
                    vals.append(np.array([S[p] for p in idxs]))
    clipped_p, clipped_v = _clip_triangles(tris, vals, iso, enclose == "above")
    if clipped_p:
        verts = np.concatenate(clipped_p, axis=0)
        cdata = np.concatenate(clipped_v)
        faces = np.arange(verts.shape[0]).reshape(-1, 3)
    else:
        verts, cdata, faces = np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), dtype=int)
    if nargout == 0:
        ax, _ = _lead_axes(interp, lead, projection="3d", fresh=False)
        if verts.size:
            _draw_fv(interp, ax, verts, faces, cdata, edge="none")
        return None
    if nargout == 1:
        return _fv_struct(verts, faces, cdata)
    return (faces.astype(float) + 1.0, verts, cdata.reshape(-1, 1))[:nargout]


@builtin("isonormals")
def _isonormals(interp, args, nargout):
    """n = isonormals(V, vertices): normals of an isosurface from the gradient of the volume data."""
    args = list(args)
    negate = any(is_text(a) and text(a).lower() == "negate" for a in args)
    for a in args:
        if is_text(a) and text(a).lower() != "negate":
            raise MatlabError(f"isonormals: unknown option '{text(a)}'.")
    nums = [a for a in args if not is_text(a)]
    patch_handle = None
    if nums and isinstance(nums[-1], P.GraphicsHandle):
        patch_handle = nums.pop()
        if patch_handle.kind != "patch":
            raise MatlabError("isonormals expects vertices or a patch handle.")
        vertices = np.asarray(P._get_prop(patch_handle, "Vertices"), dtype=float)
    else:
        if not nums:
            raise MatlabError("isonormals needs the volume and the vertices.")
        vertices = _real(nums.pop())
    x, y, z, V, _ = _volume_args(nums, "isonormals", 0)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise MatlabError("isonormals: the vertices must be an n-by-3 matrix.")
    gy, gx, gz = np.gradient(V, y, x, z)
    n = np.column_stack([_interp3(x, y, z, g, vertices[:, 0], vertices[:, 1], vertices[:, 2])
                         for g in (gx, gy, gz)])
    n = -n if not negate else n
    if patch_handle is not None:
        patch_handle.extra["VertexNormals"] = n
        relight(interp, P._obj_axes(patch_handle))
        return n if nargout > 0 else None
    return n


P._PATCH_PROPS["VertexNormals"] = (lambda h: h.extra.get("VertexNormals", _EMPTY),
                                   lambda h, v: h.extra.__setitem__("VertexNormals", _real(v)))


def _fv_input(args: list, name: str):
    """(faces 0-based n-by-k, vertices, patch handle or None, remaining args) from fv forms."""
    args = list(args)
    if args and isinstance(args[0], P.GraphicsHandle):
        h = args.pop(0)
        F = np.asarray(P._get_prop(h, "Faces"), dtype=float)
        V = np.asarray(P._get_prop(h, "Vertices"), dtype=float)
        return F - 1, V, h, args
    if args and isinstance(args[0], Struct):
        s = args.pop(0)
        low = {f.lower(): f for f in s.fields}
        if "faces" not in low or "vertices" not in low:
            raise MatlabError(f"{name}: the structure needs the fields faces and vertices.")
        return _real(s.get_field(0, low["faces"])) - 1, _real(s.get_field(0, low["vertices"])), None, args
    if len(args) >= 2:
        return _real(args[0]) - 1, _real(args[1]), None, args[2:]
    raise MatlabError(f"{name} needs faces and vertices, a structure with those fields, or a patch.")


@builtin("reducepatch")
def _reducepatch(interp, args, nargout):
    """nfv = reducepatch(fv, r): a mesh with fewer faces (a fraction r of them, or about r faces if r > 1)."""
    flags = [text(a).lower() for a in args if is_text(a)]
    for f in flags:
        if f not in ("fast", "verbose"):
            raise MatlabError(f"reducepatch: unknown option '{f}'.")
    F, V, handle, rest = _fv_input([a for a in args if not is_text(a)], "reducepatch")
    r = float(_vec(rest[0])[0]) if rest else 0.5
    if r <= 0:
        raise MatlabError("reducepatch: the reduction factor must be positive.")
    F = F.astype(float)
    tri = []
    for row in F:  # polygons become triangle fans
        idx = row[np.isfinite(row)].astype(int)
        for k in range(1, idx.size - 1):
            tri.append((idx[0], idx[k], idx[k + 1]))
    T = np.array(tri, dtype=int).reshape(-1, 3)
    target = int(round(r * T.shape[0])) if r <= 1 else int(round(r))
    target = max(target, 1)

    def cluster(n: int):
        lo, hi = V.min(axis=0), V.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        cell = np.minimum(np.floor((V - lo) / span * n).astype(np.int64), n - 1)
        key = cell[:, 0] * n * n + cell[:, 1] * n + cell[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        inv = np.asarray(inv).ravel()
        newV = np.zeros((uniq.size, 3))
        np.add.at(newV, inv, V)
        newV /= np.bincount(inv, minlength=uniq.size).reshape(-1, 1)
        nf = inv[T]
        good = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
        nf = nf[good]
        nf = nf[np.unique(np.sort(nf, axis=1), axis=0, return_index=True)[1]] if nf.size else nf
        return newV, nf

    if T.shape[0] <= target or V.shape[0] < 4:
        newV, newF = V.copy(), T.copy()
    else:
        lo_n, hi_n = 1, 2
        while cluster(hi_n)[1].shape[0] < target and hi_n < 4096:
            lo_n, hi_n = hi_n, hi_n * 2
        while hi_n - lo_n > 1:  # largest grid whose mesh still has at most `target` faces
            mid = (lo_n + hi_n) // 2
            if cluster(mid)[1].shape[0] <= target:
                lo_n = mid
            else:
                hi_n = mid
        newV, newF = cluster(lo_n)
        if newF.shape[0] == 0:
            newV, newF = cluster(hi_n)
        used = np.unique(newF)
        remap = np.full(newV.shape[0], -1, dtype=int)
        remap[used] = np.arange(used.size)
        newV, newF = newV[used], remap[newF]
    if handle is not None and nargout == 0:
        P._set_prop(handle, "Vertices", newV)
        P._set_prop(handle, "Faces", newF.astype(float) + 1.0)
        return None
    if nargout >= 2:
        return newF.astype(float) + 1.0, newV
    return _fv_struct(newV, newF)


@builtin("smooth3")
def _smooth3(interp, args, nargout):
    """W = smooth3(V, 'box' | 'gaussian', size, sd): a smoothed copy of 3-D data."""
    from scipy import ndimage
    if not args:
        raise MatlabError("smooth3 needs 3-D data.")
    V = _real(args[0])
    if V.ndim != 3:
        raise MatlabError("smooth3: the data must be a 3-D array.")
    kind = "box"
    size: Any = 3
    sd = 0.65
    rest = list(args[1:])
    if rest and is_text(rest[0]):
        kind = text(rest.pop(0)).lower()
        if kind not in ("box", "gaussian"):
            raise MatlabError("smooth3: the filter must be 'box' or 'gaussian'.")
    if rest:
        size = _vec(rest.pop(0))
    if rest:
        sd = float(_vec(rest.pop(0))[0])
    if rest:
        raise MatlabError("smooth3: too many inputs.")
    sz = np.atleast_1d(np.asarray(size, dtype=float)).astype(int)
    if sz.size == 1:
        sz = np.repeat(sz, 3)
    if sz.size != 3 or np.any(sz < 1) or np.any(sz % 2 == 0):
        raise MatlabError("smooth3: the filter size must be odd (one number or three).")
    if kind == "box":
        kernel = np.ones(tuple(sz)) / float(np.prod(sz))
    else:
        if sd <= 0:
            raise MatlabError("smooth3: the standard deviation must be positive.")
        ax = [np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) for n in sz]
        g = np.exp(-(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2)
                   / (2 * sd * sd))
        kernel = g / g.sum()
    return ndimage.correlate(V, kernel, mode="nearest")


@builtin("surf2patch")
def _surf2patch(interp, args, nargout):
    """fvc = surf2patch(X, Y, Z, C): the faces, vertices and colours of a surface, for patch."""
    triangles = any(is_text(a) and text(a).lower() == "triangles" for a in args)
    for a in args:
        if is_text(a) and text(a).lower() != "triangles":
            raise MatlabError(f"surf2patch: unknown option '{text(a)}'.")
    nums = [a for a in args if not is_text(a)]
    if not nums:
        raise MatlabError("surf2patch needs surface data or a surface handle.")
    if isinstance(nums[0], P.GraphicsHandle):
        h = nums[0]
        if h.kind != "surface":
            raise MatlabError("surf2patch expects a surface handle.")
        X, Y, Z = (np.asarray(h.extra[k], dtype=float) for k in ("XData", "YData", "ZData"))
        C = np.asarray(h.extra.get("CData", Z), dtype=float)
    elif len(nums) == 1:
        Z = _real(nums[0])
        X, Y = np.meshgrid(np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1))
        C = Z
    elif len(nums) == 2:
        Z, C = _real(nums[0]), _real(nums[1])
        X, Y = np.meshgrid(np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1))
    else:
        X, Y, Z = (_real(a) for a in nums[:3])
        C = _real(nums[3]) if len(nums) > 3 else Z
        if X.shape != Z.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    m, n = Z.shape
    if X.shape != Z.shape or Y.shape != Z.shape:
        raise MatlabError("surf2patch: X, Y and Z must have the same size.")
    verts = np.column_stack([X.flatten(order="F"), Y.flatten(order="F"), Z.flatten(order="F")])
    idx = np.arange(1, m * n + 1).reshape(m, n, order="F")
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    quads = np.column_stack([a.flatten(order="F"), b.flatten(order="F"), c.flatten(order="F"),
                             d.flatten(order="F")]).astype(float)
    if triangles:
        faces = np.vstack([np.column_stack([quads[:, 0], quads[:, 1], quads[:, 2]]),
                           np.column_stack([quads[:, 0], quads[:, 2], quads[:, 3]])])
        order = np.argsort(np.concatenate([np.arange(quads.shape[0]) * 2, np.arange(quads.shape[0]) * 2 + 1]))
        faces = faces[order]
    else:
        faces = quads
    cdata = C.reshape(-1, 1, order="F") if C.shape == Z.shape else (
        C.reshape(-1, 3, order="F") if C.ndim == 3 and C.shape[:2] == Z.shape else None)
    if cdata is None:
        raise MatlabError("surf2patch: the colour data must match the size of Z.")
    if nargout <= 1:
        return Struct.scalar({"faces": faces, "vertices": verts, "facevertexcdata": cdata})
    return (faces, verts, cdata)[:nargout]


@builtin("tetramesh")
def _tetramesh(interp, args, nargout):
    """tetramesh(T, X, c): the tetrahedra with corner numbers T (m-by-4) and points X (n-by-3)."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, {"facecolor", "edgecolor", "facealpha", "linewidth"})
    nums = [a for a in args if not is_text(a)]
    if len(nums) < 2:
        raise MatlabError("tetramesh needs the tetrahedra and the points, tetramesh(T, X).")
    T, X = _real(nums[0]).astype(int), _real(nums[1])
    if T.ndim != 2 or T.shape[1] != 4 or X.ndim != 2 or X.shape[1] != 3:
        raise MatlabError("tetramesh: T must have four columns and X three.")
    if T.size and (T.min() < 1 or T.max() > X.shape[0]):
        raise MatlabError("tetramesh: T contains a point number outside the rows of X.")
    cvals = _vec(nums[2]) if len(nums) > 2 else np.arange(1.0, T.shape[0] + 1)
    if cvals.size != T.shape[0]:
        raise MatlabError("tetramesh: give one colour value for each tetrahedron.")
    ax, _ = _lead_axes(interp, lead, projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = _cmap(interp)
    lo, hi = (float(cvals.min()), float(cvals.max())) if cvals.size else (0.0, 1.0)
    hs = []
    alpha = P._scalar(opts["facealpha"]) if "facealpha" in opts else 0.9
    for k, tet in enumerate(T - 1):
        polys = [X[[tet[a], tet[b], tet[c]]] for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
        color = P._color(opts["facecolor"]) if "facecolor" in opts else cmap((cvals[k] - lo) / (hi - lo)
                                                                             if hi > lo else 0.5)
        coll = Poly3DCollection(polys, facecolor=color, alpha=alpha, linewidth=P._scalar(opts["linewidth"])
                                if "linewidth" in opts else 0.4,
                                edgecolor=P._color(opts["edgecolor"]) if "edgecolor" in opts else "k")
        ax.add_collection3d(coll)
        h = P._wrap(interp, "patch", coll)
        h.extra.update({"axes": ax, "figure": ax.figure, "_polys": polys})
        hs.append(h)
    if X.size:
        ax.auto_scale_xyz(X[:, 0], X[:, 1], X[:, 2])
    P._redraw(interp)
    return P._handle_list(hs) if nargout > 0 else None


# -- contours: matrices, 3-D contour lines, slices ----------------------------------------------------------------

def _contour_levels(Z: np.ndarray, spec) -> np.ndarray:
    f = Z[np.isfinite(Z)]
    if f.size == 0:
        return np.zeros(0)
    lo, hi = float(f.min()), float(f.max())
    if spec is None or (np.size(spec) == 1 and float(np.ravel(spec)[0]) == int(np.ravel(spec)[0])
                        and np.size(spec) == 1 and not isinstance(spec, tuple)):
        from matplotlib.ticker import MaxNLocator
        n = int(np.ravel(spec)[0]) if spec is not None else 7
        if spec is not None:
            if n < 1:
                raise MatlabError("The number of contour levels must be positive.")
            step = (hi - lo) / (n + 1)
            return lo + step * np.arange(1, n + 1)
        lv = MaxNLocator(nbins=n + 1, min_n_ticks=1).tick_values(lo, hi)
        return lv[(lv > lo) & (lv < hi)]
    lv = np.asarray(spec, dtype=float).flatten()
    return np.unique(lv)


def _contour_lines(x, y, Z, levels):
    """[(level, [n-by-2 arrays])]: the contour lines of a gridded function."""
    import contourpy
    gen = contourpy.contour_generator(x=x, y=y, z=np.ma.masked_invalid(Z))
    return [(float(lv), [np.asarray(seg, dtype=float) for seg in gen.lines(float(lv))]) for lv in levels]


def _matrix_from_lines(lines) -> np.ndarray:
    cols = []
    for level, segs in lines:
        for seg in segs:
            if seg.shape[0] == 0:
                continue
            cols.append(np.array([[level], [float(seg.shape[0])]]))
            cols.append(seg.T)
    return np.hstack(cols) if cols else np.zeros((2, 0))


def _xyz_grid(nums: list, name: str):
    if len(nums) >= 3 and _real(nums[2]).size > 2 and _real(nums[2]).ndim == 2 and min(_real(nums[2]).shape) > 1:
        x, y, Z = _real(nums[0]), _real(nums[1]), _real(nums[2])
        rest = nums[3:]
        xv = x[0, :] if x.ndim == 2 and min(x.shape) > 1 else x.flatten()
        yv = y[:, 0] if y.ndim == 2 and min(y.shape) > 1 else y.flatten()
        if xv.size != Z.shape[1] or yv.size != Z.shape[0]:
            raise MatlabError(f"{name}: X and Y must match the size of Z.")
    else:
        Z = _real(nums[0])
        rest = nums[1:]
        xv, yv = np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1)
    if Z.ndim != 2 or min(Z.shape) < 2:
        raise MatlabError(f"{name}: Z must be a matrix with at least two rows and columns.")
    return xv, yv, Z, rest


@builtin("contourc")
def _contourc(interp, args, nargout):
    """C = contourc(Z, levels) or contourc(x, y, Z, levels): the contour matrix, without drawing."""
    nums = [a for a in args if not is_text(a)]
    if len(nums) != len(args) or not nums:
        raise MatlabError("contourc expects contourc(Z), contourc(Z, n), contourc(Z, v) or contourc(x, y, Z, ...).")
    x, y, Z, rest = _xyz_grid(nums, "contourc")
    spec = None
    if rest:
        v = _vec(rest[0])
        spec = v if v.size > 1 else v[0]
        if v.size == 2 and v[0] == v[1]:
            spec = (float(v[0]),)
    levels = np.array(spec) if isinstance(spec, tuple) else _contour_levels(Z, spec)
    return _matrix_from_lines(_contour_lines(x, y, Z, levels))


@builtin("contour3")
def _contour3(interp, args, nargout):
    """contour3(Z) or contour3(X, Y, Z, levels): contour lines drawn at their heights in 3-D axes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, {"linewidth", "linecolor", "linestyle", "levellist", "showtext"})
    spec_text = [text(a) for a in args if is_text(a)]
    for sp in spec_text:
        if not P._is_linespec(sp):
            raise MatlabError(f"contour3: unknown option '{sp}'.")
    nums = [a for a in args if not is_text(a)]
    if not nums:
        raise MatlabError("contour3 needs data, e.g. contour3(peaks).")
    x, y, Z, rest = _xyz_grid(nums, "contour3")
    spec = None
    if rest:
        v = _vec(rest[0])
        spec = v if v.size > 1 else v[0]
        if v.size == 2 and v[0] == v[1]:
            spec = (float(v[0]),)
    if "levellist" in opts:
        spec = tuple(float(t) for t in _vec(opts["levellist"]))
    levels = np.array(spec) if isinstance(spec, tuple) else _contour_levels(Z, spec)
    lines = _contour_lines(x, y, Z, levels)
    ax, _ = _lead_axes(interp, lead, projection="3d")
    cmap = _cmap(interp)
    lo, hi = (float(levels.min()), float(levels.max())) if levels.size else (0.0, 1.0)
    import re
    fixed = None
    if spec_text:
        m = re.search(r"[rgbcmykw]", spec_text[0])
        fixed = m.group(0) if m else None
    if "linecolor" in opts and not (is_text(opts["linecolor"]) and text(opts["linecolor"]).lower() == "flat"):
        fixed = P._color(opts["linecolor"])
    lw = P._scalar(opts["linewidth"]) if "linewidth" in opts else 0.8
    out = []
    for level, segs in lines:
        color = fixed if fixed is not None else cmap((level - lo) / (hi - lo) if hi > lo else 0.5)
        for seg in segs:
            out.extend(ax.plot(seg[:, 0], seg[:, 1], np.full(seg.shape[0], level), color=color, linewidth=lw))
    P._redraw(interp)
    C = _matrix_from_lines(lines)
    hs = [P._wrap(interp, "line", ln) for ln in out]
    if nargout >= 2:
        return C, P._handle_list(hs)
    return C if nargout == 1 else None


def _slice_positions(v) -> np.ndarray:
    return _vec(v) if isinstance(v, np.ndarray) and v.size else np.zeros(0)


def _plane_grids(x, y, z, axis: str, value: float):
    if axis == "x":
        B, C = np.meshgrid(y, z, indexing="ij")
        return np.full(B.shape, value), B, C
    if axis == "y":
        A, C = np.meshgrid(x, z, indexing="ij")
        return A, np.full(A.shape, value), C
    A, B = np.meshgrid(x, y, indexing="xy")
    return A, B, np.full(A.shape, value)


@builtin("slice")
def _slice(interp, args, nargout):
    """slice(V, xslice, yslice, zslice) or slice(X, Y, Z, V, ...): coloured planes through volume data."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    method = "linear"
    nums = []
    for a in args:
        if is_text(a):
            method = text(a).lower()
            if method not in ("linear", "nearest", "cubic"):
                raise MatlabError("slice: the method must be 'linear', 'cubic' or 'nearest'.")
        else:
            nums.append(a)
    x, y, z, V, rest = _volume_args(nums, "slice", 3)
    if len(rest) != 3:
        raise MatlabError("slice needs the slice positions: slice(V, xslice, yslice, zslice).")
    f = V[np.isfinite(V)]
    lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
    ax, _ = _lead_axes(interp, lead, projection="3d")
    cmap = _cmap(interp)
    hs = []
    surfaces = []
    if all(isinstance(r, np.ndarray) and _real(r).ndim == 2 and min(_real(r).shape) > 1 for r in rest):
        surfaces.append(tuple(_real(r) for r in rest))  # a surface to slice along
    else:
        for axis, pos in zip("xyz", rest):
            for value in _slice_positions(pos):
                surfaces.append(_plane_grids(x, y, z, axis, float(value)))
    for SX, SY, SZ in surfaces:
        vals = _interp3(x, y, z, V, SX, SY, SZ, "nearest" if method == "nearest" else "linear")
        frac = (vals - lo) / (hi - lo) if hi > lo else np.full(vals.shape, 0.5)
        colors = cmap(np.nan_to_num(frac))
        colors[~np.isfinite(vals)] = (1, 1, 1, 0)
        coll = ax.plot_surface(SX, SY, SZ, facecolors=colors, shade=False, linewidth=0.2, edgecolor="k",
                               rcount=min(SX.shape[0], 120), ccount=min(SX.shape[1], 120))
        coll._mylab_surface = True
        h = P._wrap(interp, "surface", coll)
        h.extra.update({"XData": SX, "YData": SY, "ZData": SZ, "CData": vals})
        hs.append(h)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.min()), float(z.max()))
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    sm = cm.ScalarMappable(norm=Normalize(lo, hi if hi > lo else lo + 1), cmap=cmap)
    sm.set_array(np.array([lo, hi]))
    ax._mylab_mappable = sm
    P._redraw(interp)
    return P._handle_list(hs) if nargout > 0 else None


@builtin("contourslice")
def _contourslice(interp, args, nargout):
    """contourslice(V, xslice, yslice, zslice, n): contour lines of volume data in slice planes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    nums = [a for a in args if not is_text(a)]
    for a in args:
        if is_text(a) and text(a).lower() not in ("linear", "nearest", "cubic"):
            raise MatlabError(f"contourslice: unknown option '{text(a)}'.")
    extra = 4 if len(nums) in (5, 8) else 3
    x, y, z, V, rest = _volume_args(nums, "contourslice", extra)
    if len(rest) < 3:
        raise MatlabError("contourslice needs the slice positions: contourslice(V, xslice, yslice, zslice).")
    spec = None
    if len(rest) > 3:
        v = _vec(rest[3])
        spec = v if v.size > 1 else v[0]
        if v.size == 2 and v[0] == v[1]:
            spec = (float(v[0]),)
    levels = np.array(spec) if isinstance(spec, tuple) else _contour_levels(V, spec)
    ax, _ = _lead_axes(interp, lead, projection="3d")
    cmap = _cmap(interp)
    lo, hi = (float(levels.min()), float(levels.max())) if levels.size else (0.0, 1.0)
    out = []
    for axis, pos in zip("xyz", rest[:3]):
        for value in _slice_positions(pos):
            SX, SY, SZ = _plane_grids(x, y, z, axis, float(value))
            vals = _interp3(x, y, z, V, SX, SY, SZ)
            if axis == "x":
                u, w, S = y, z, vals.T      # rows follow z, columns follow y
            elif axis == "y":
                u, w, S = x, z, vals.T
            else:
                u, w, S = x, y, vals
            if not np.isfinite(S).any():
                continue
            for level, segs in _contour_lines(u, w, S, levels):
                color = cmap((level - lo) / (hi - lo) if hi > lo else 0.5)
                for seg in segs:
                    a, b = seg[:, 0], seg[:, 1]
                    c = np.full(seg.shape[0], float(value))
                    xyz = (c, a, b) if axis == "x" else (a, c, b) if axis == "y" else (a, b, c)
                    out.extend(ax.plot(*xyz, color=color, linewidth=0.8))
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.min()), float(z.max()))
    P._redraw(interp)
    return P._ret(interp, "line", out, nargout)


def _with_base_contour(name: str, kind: str):
    def f(interp, args, nargout):
        h = P._surface(interp, list(args), kind, 1)
        ax = P._obj_axes(h)
        X, Y, Z = (np.asarray(h.extra[k], dtype=float) for k in ("XData", "YData", "ZData"))
        zmin = float(np.nanmin(Z))
        zmax = float(np.nanmax(Z))
        base = zmin - 0.25 * (zmax - zmin if zmax > zmin else 1.0) if name.endswith("c") else zmin
        cs = ax.contour(X, Y, Z, zdir="z", offset=base, cmap=P._current_cmap(interp))
        cs._mylab_internal = True
        ax.set_zlim(base, zmax if zmax > base else base + 1)
        ch = P._wrap(interp, "contour", cs)
        ch.extra["_contour"] = None
        P._redraw(interp)
        if nargout <= 0:
            return None
        return P.GraphicsArray([h, ch])
    return f


builtin("surfc")(P._with_axes_arg(_with_base_contour("surfc", "surf")))
builtin("meshc")(P._with_axes_arg(_with_base_contour("meshc", "mesh")))


@builtin("meshz")
@P._with_axes_arg
def _meshz(interp, args, nargout):
    """meshz(X, Y, Z): a mesh with a curtain around it down to the lowest value."""
    nums = [a for a in args if not is_text(a)]
    words = [a for a in args if is_text(a)]
    if not nums:
        raise MatlabError("meshz needs data, e.g. meshz(peaks).")
    if len(nums) >= 3:
        X, Y, Z = (_real(a) for a in nums[:3])
        if X.shape != Z.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    else:
        Z = _real(nums[0])
        X, Y = np.meshgrid(np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1))
    base = float(np.nanmin(Z))
    Zp = np.pad(Z, 1, mode="constant", constant_values=base)
    Xp = np.pad(X, 1, mode="edge")
    Yp = np.pad(Y, 1, mode="edge")
    return P._surface(interp, [Xp, Yp, Zp] + words, "mesh", nargout)


@builtin("hidden")
def _hidden(interp, args, nargout):
    """hidden on/off: whether a mesh hides what lies behind it (opaque faces) or is see-through."""
    ax, args = _current_axes(interp, args)
    meshes = [c for c in ax.collections if getattr(c, "_mylab_mesh", False)]
    if not args:
        state = interp.state.get("mesh_hidden", True)
        if nargout > 0:
            return char("on" if state else "off")
        on = not state
    else:
        word = text(args[0]).lower()
        if word not in ("on", "off"):
            raise MatlabError("hidden must be 'on' or 'off'.")
        on = word == "on"
    interp.state["mesh_hidden"] = on
    for c in meshes:
        c.set_facecolor(ax.get_facecolor() if on else "none")
    P._redraw(interp)
    return None


@builtin("fplot3")
def _fplot3(interp, args, nargout):
    """fplot3(fx, fy, fz, [tmin tmax]): the space curve x = fx(t), y = fy(t), z = fz(t)."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, P._LINE_OPTS | {"meshdensity"})
    funcs = []
    while args and (isinstance(args[0], P.FunctionHandle) or (is_text(args[0]) and len(funcs) < 3
                                                             and not P._is_linespec(text(args[0])))):
        funcs.append(P._as_fn(interp, args.pop(0))[0])
    if len(funcs) != 3:
        raise MatlabError("fplot3 needs three functions of one variable, e.g. fplot3(@sin, @cos, @(t) t).")
    lims = np.array([-5.0, 5.0])
    if args and not is_text(args[0]):
        lims = _vec(args.pop(0))
        if lims.size != 2 or not lims[0] < lims[1]:
            raise MatlabError("fplot3: the interval must be [tmin tmax] with tmin < tmax.")
    spec = text(args.pop(0)) if args and is_text(args[0]) and P._is_linespec(text(args[0])) else ""
    if args:
        raise MatlabError("fplot3: unexpected input.")
    n = int(P._scalar(opts.pop("meshdensity"))) * 8 if "meshdensity" in opts else 400
    t = np.linspace(lims[0], lims[1], max(n, 2)).reshape(1, -1)
    xs, ys, zs = (P._eval_fn(interp, f, [t]).flatten() for f in funcs)
    ax, _ = _lead_axes(interp, lead, projection="3d")
    kw = P._mpl_kwargs(opts)
    lines = ax.plot(xs, ys, zs, spec, **kw) if spec else ax.plot(xs, ys, zs, **kw)
    P._tag_lines(interp, lines, opts)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


builtin("ezplot3")(lambda interp, args, nargout: _fplot3(
    interp, list(args) if len([a for a in args if not isinstance(a, P.GraphicsHandle)]) > 3
    else list(args) + [np.array([[0.0, 2 * math.pi]])], nargout))


@builtin("fimplicit3")
def _fimplicit3(interp, args, nargout):
    """fimplicit3(f, limits): the surface f(x, y, z) = 0."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, {"meshdensity", "facecolor", "edgecolor", "facealpha"})
    if not args:
        raise MatlabError("fimplicit3 needs a function of three variables, e.g. @(x, y, z) x.^2 + y.^2 - z.")
    f = P._as_fn(interp, args[0])[0]
    lims = np.array([-5.0, 5.0] * 3)
    if len(args) > 1:
        v = _vec(args[1])
        if v.size == 2:
            lims = np.tile(v, 3)
        elif v.size == 6:
            lims = v
        else:
            raise MatlabError("fimplicit3: the limits must be [min max] or [xmin xmax ymin ymax zmin zmax].")
    n = int(P._scalar(opts["meshdensity"])) if "meshdensity" in opts else 35
    x = np.linspace(lims[0], lims[1], n)
    y = np.linspace(lims[2], lims[3], n)
    z = np.linspace(lims[4], lims[5], n)
    X, Y, Z = np.meshgrid(x, y, z)
    V = P._eval_fn(interp, f, [X, Y, Z])
    verts, faces = marching_tetrahedra(x, y, z, V, 0.0)
    ax, _ = _lead_axes(interp, lead, projection="3d")
    h = None
    if verts.size:
        color = P._color(opts["facecolor"]) if "facecolor" in opts else None
        h = _draw_fv(interp, ax, verts, faces, verts[:, 2] if color is None else None, face_color=color,
                     edge=P._color(opts["edgecolor"]) if "edgecolor" in opts else "none",
                     alpha=P._scalar(opts["facealpha"]) if "facealpha" in opts else None)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_zlim(lims[4], lims[5])
    P._redraw(interp)
    return h if nargout > 0 and h is not None else (np.zeros((0, 0)) if nargout > 0 else None)


# -- streamlines --------------------------------------------------------------------------------------------------

def _field_args(nums: list, ndim: int, name: str):
    """(axis vectors, field components, remaining args) for (U, V, ...) or (X, Y, U, V, ...) and 3-D forms."""
    if len(nums) < ndim:
        raise MatlabError(f"{name} needs the vector field.")
    first = _real(nums[0])
    # (X, Y[, Z], U, V[, W], ...): the components come after ndim coordinate arrays
    with_grid = len(nums) >= 2 * ndim and all(_real(a).ndim == ndim and _real(a).shape == _real(nums[ndim]).shape
                                              for a in nums[ndim:2 * ndim]) and (
        first.shape == _real(nums[ndim]).shape or first.size in _real(nums[ndim]).shape)
    if with_grid and len(nums) < 2 * ndim + ndim and first.shape == _real(nums[ndim]).shape:
        # ambiguous only for (U, V, startx, starty) with start points shaped like the field: prefer the grid
        with_grid = True
    if with_grid:
        coords = [_real(a) for a in nums[:ndim]]
        comps = [_real(a) for a in nums[ndim:2 * ndim]]
        rest = nums[2 * ndim:]
        shape = comps[0].shape
        if ndim == 2:
            x = coords[0][0, :] if coords[0].shape == shape else coords[0].flatten()
            y = coords[1][:, 0] if coords[1].shape == shape else coords[1].flatten()
            axes = [x, y]
        else:
            x = coords[0][0, :, 0] if coords[0].shape == shape else coords[0].flatten()
            y = coords[1][:, 0, 0] if coords[1].shape == shape else coords[1].flatten()
            z = coords[2][0, 0, :] if coords[2].shape == shape else coords[2].flatten()
            axes = [x, y, z]
    else:
        comps = [_real(a) for a in nums[:ndim]]
        rest = nums[ndim:]
        shape = comps[0].shape
        axes = [np.arange(1.0, shape[1] + 1), np.arange(1.0, shape[0] + 1)]
        if ndim == 3:
            axes.append(np.arange(1.0, shape[2] + 1))
    if any(c.shape != shape for c in comps) or len(shape) != ndim:
        raise MatlabError(f"{name}: the components of the vector field must be {ndim}-D arrays of one size.")
    want = (shape[1], shape[0]) + ((shape[2],) if ndim == 3 else ())
    if tuple(a.size for a in axes) != want:
        raise MatlabError(f"{name}: the coordinates do not match the size of the vector field.")
    return axes, comps, rest


def _trace(axes: list, comps: list, start: np.ndarray, step: float, maxvert: int) -> np.ndarray:
    """One streamline from ``start`` (data coordinates), stepping ``step`` cells at a time."""
    nd = len(axes)
    sizes = [a.size for a in axes]
    # work in index coordinates (0-based, fractional); axis k of `axes` is x, y[, z]
    def to_index(p):
        return np.array([np.interp(p[k], axes[k] if axes[k][0] < axes[k][-1] else axes[k][::-1],
                                   np.arange(sizes[k]) if axes[k][0] < axes[k][-1]
                                   else np.arange(sizes[k])[::-1]) for k in range(nd)])

    def to_data(q):
        return np.array([np.interp(q[k], np.arange(sizes[k]), axes[k]) for k in range(nd)])

    spacing = [np.gradient(a) if a.size > 1 else np.ones(1) for a in axes]

    def velocity(q):
        """Velocity in index units per unit time at fractional index q, or None outside the grid."""
        if any(q[k] < 0 or q[k] > sizes[k] - 1 for k in range(nd)):
            return None
        i0 = [min(int(math.floor(q[k])), sizes[k] - 2) if sizes[k] > 1 else 0 for k in range(nd)]
        f = [q[k] - i0[k] for k in range(nd)]
        out = np.zeros(nd)
        for corner in range(2 ** nd):
            w = 1.0
            idx = []
            for k in range(nd):
                bit = (corner >> k) & 1
                w *= f[k] if bit else 1 - f[k]
                idx.append(min(i0[k] + bit, sizes[k] - 1))
            if w == 0:
                continue
            sel = (idx[1], idx[0]) + ((idx[2],) if nd == 3 else ())
            for c in range(nd):
                val = comps[c][sel]
                if not np.isfinite(val):
                    return None
                out[c] += w * val
        for k in range(nd):  # data units -> index units
            h = spacing[k][min(max(int(round(q[k])), 0), sizes[k] - 1)]
            out[k] = out[k] / h if h != 0 else 0.0
        return out

    for k in range(nd):
        lo, hi = min(axes[k][0], axes[k][-1]), max(axes[k][0], axes[k][-1])
        if not (lo <= start[k] <= hi) or not np.isfinite(start[k]):
            return np.zeros((0, nd))
    q = to_index(start)
    pts = [to_data(q)]
    for _ in range(maxvert - 1):
        v1 = velocity(q)
        if v1 is None:
            break
        s1 = float(np.linalg.norm(v1))
        if s1 == 0:
            break
        mid = q + 0.5 * step * v1 / s1
        v2 = velocity(mid)
        if v2 is None:
            v2, s2 = v1, s1
        else:
            s2 = float(np.linalg.norm(v2))
            if s2 == 0:
                break
        q = q + step * v2 / s2
        if any(q[k] < 0 or q[k] > sizes[k] - 1 for k in range(nd)):
            break
        pts.append(to_data(q))
    return np.array(pts)


def _stream_core(nums: list, ndim: int, name: str):
    axes, comps, rest = _field_args(nums, ndim, name)
    if len(rest) < ndim:
        raise MatlabError(f"{name} needs the starting points of the streamlines.")
    starts = [_vec(a) for a in rest[:ndim]]
    if any(s.size != starts[0].size for s in starts):
        raise MatlabError(f"{name}: the starting coordinates must have the same number of elements.")
    step, maxvert = 0.1, 10000
    if len(rest) > ndim:
        o = _vec(rest[ndim])
        if o.size >= 1:
            step = float(o[0])
        if o.size >= 2:
            maxvert = int(o[1])
        if step <= 0 or maxvert < 1:
            raise MatlabError(f"{name}: the options are [stepsize maxvertices], both positive.")
    out = []
    for k in range(starts[0].size):
        out.append(_trace(axes, comps, np.array([s[k] for s in starts]), step, maxvert))
    return out


@builtin("stream2")
def _stream2(interp, args, nargout):
    """XY = stream2(X, Y, U, V, startx, starty): streamlines of a 2-D vector field, a cell array of [x y]."""
    lines = _stream_core([a for a in args], 2, "stream2")
    return Cell.from_list(lines)


@builtin("stream3")
def _stream3(interp, args, nargout):
    """XYZ = stream3(X, Y, Z, U, V, W, startx, starty, startz): streamlines of a 3-D vector field."""
    lines = _stream_core([a for a in args], 3, "stream3")
    return Cell.from_list(lines)


@builtin("streamline")
def _streamline(interp, args, nargout):
    """streamline(X, Y, U, V, startx, starty): draw streamlines of a vector field from starting points."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, P._LINE_OPTS)
    for a in args:
        if is_text(a):
            raise MatlabError(f"streamline: unknown option '{text(a)}'.")
    if args and isinstance(args[0], Cell):
        lines = [_real(v) for v in args[0].values_f()]
        ndim = lines[0].shape[1] if lines and lines[0].ndim == 2 else 2
    else:
        first = _real(args[0]) if args else np.zeros((0, 0))
        ndim = 3 if first.ndim == 3 else 2
        lines = _stream_core(args, ndim, "streamline")
    ax, _ = _lead_axes(interp, lead, projection="3d" if ndim == 3 else "rectilinear", fresh=False)
    kw = P._mpl_kwargs(opts)
    kw.setdefault("color", ax._get_lines.get_next_color())
    out = []
    for ln in lines:
        if ln.ndim != 2 or ln.shape[0] == 0:
            continue
        out.extend(ax.plot(*[ln[:, k] for k in range(ln.shape[1])], **kw))
    P._tag_lines(interp, out, opts)
    P._redraw(interp)
    return P._ret(interp, "line", out, nargout)


def _even_streamlines(x, y, U, V, density: float):
    """Evenly spread streamlines of a 2-D field and arrow positions, from matplotlib's stream plotter."""
    plt = P._plt()
    tmp = plt.figure(figsize=(4, 3), dpi=50)
    try:
        ax = tmp.add_subplot(1, 1, 1)
        if x[0] > x[-1]:
            x, U, V = x[::-1], U[:, ::-1], V[:, ::-1]
        if y[0] > y[-1]:
            y, U, V = y[::-1], U[::-1, :], V[::-1, :]
        xi = np.linspace(x[0], x[-1], max(x.size, 2))
        yi = np.linspace(y[0], y[-1], max(y.size, 2))
        if not (np.allclose(xi, x) and np.allclose(yi, y)):  # the plotter needs an evenly spaced grid
            from scipy.interpolate import RegularGridInterpolator
            XI, YI = np.meshgrid(xi, yi)
            pts = np.column_stack([YI.ravel(), XI.ravel()])
            U = RegularGridInterpolator((y, x), U)(pts).reshape(XI.shape)
            V = RegularGridInterpolator((y, x), V)(pts).reshape(XI.shape)
        res = ax.streamplot(xi, yi, np.nan_to_num(U), np.nan_to_num(V), density=density)
        segs = [np.asarray(sg, dtype=float) for sg in res.lines.get_segments()]
        lines = []
        cur: list = []
        for sg in segs:  # pieces that share an end point form one streamline
            if sg.shape[0] < 2:
                continue
            if cur and np.allclose(cur[-1], sg[0]):
                cur.extend(sg[1:])
            else:
                if cur:
                    lines.append(np.array(cur))
                cur = list(sg)
        if cur:
            lines.append(np.array(cur))
        arrows = []
        for line in lines:
            if line.shape[0] >= 2:
                k = line.shape[0] // 2
                arrows.append((line[max(k - 1, 0)], line[k]))
        return lines, arrows
    finally:
        plt.close(tmp)


@builtin("streamslice")
def _streamslice(interp, args, nargout):
    """streamslice(X, Y, U, V, density): well-spaced streamlines with direction arrows."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    arrows_on = True
    nums = []
    for a in args:
        if is_text(a):
            w = text(a).lower()
            if w == "noarrows":
                arrows_on = False
            elif w not in ("arrows", "linear", "nearest", "cubic"):
                raise MatlabError(f"streamslice: unknown option '{text(a)}'.")
        else:
            nums.append(a)
    if not nums:
        raise MatlabError("streamslice needs a vector field.")
    volume = _real(nums[0]).ndim == 3
    segs3: list = []   # (list of n-by-3 lines, list of arrow (p0, p1) pairs)
    if not volume:
        axes, comps, rest = _field_args(nums, 2, "streamslice")
        density = float(_vec(rest[0])[0]) if rest else 1.0
        if density <= 0:
            raise MatlabError("streamslice: the density must be positive.")
        lines, arrows = _even_streamlines(axes[0], axes[1], comps[0], comps[1], density)
        ax, _ = _lead_axes(interp, lead, projection="rectilinear")
        color = ax._get_lines.get_next_color()
        out = []
        for ln in lines:
            out.extend(ax.plot(ln[:, 0], ln[:, 1], color=color, linewidth=0.8))
        if arrows_on:
            for p0, p1 in arrows:
                note = ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8})
                note._mylab_internal = True
        ax.set_xlim(float(min(axes[0])), float(max(axes[0])))
        ax.set_ylim(float(min(axes[1])), float(max(axes[1])))
        P._redraw(interp)
        return P._ret(interp, "line", out, nargout)
    axes, comps, rest = _field_args(nums, 3, "streamslice")
    if len(rest) < 3:
        raise MatlabError("streamslice needs the slice positions: streamslice(X, Y, Z, U, V, W, sx, sy, sz).")
    density = float(_vec(rest[3])[0]) if len(rest) > 3 else 1.0
    x, y, z = axes
    U, V, W = comps
    for axis, pos in zip("xyz", rest[:3]):
        for value in _slice_positions(pos):
            if axis == "z":
                k = int(np.argmin(np.abs(z - value)))
                lines, arrows = _even_streamlines(x, y, U[:, :, k], V[:, :, k], density)
                lift = lambda p, c=float(value): np.column_stack([p[:, 0], p[:, 1], np.full(len(p), c)])  # noqa: E731
            elif axis == "x":
                k = int(np.argmin(np.abs(x - value)))
                lines, arrows = _even_streamlines(y, z, V[:, k, :].T, W[:, k, :].T, density)
                lift = lambda p, c=float(value): np.column_stack([np.full(len(p), c), p[:, 0], p[:, 1]])  # noqa: E731
            else:
                k = int(np.argmin(np.abs(y - value)))
                lines, arrows = _even_streamlines(x, z, U[k, :, :].T, W[k, :, :].T, density)
                lift = lambda p, c=float(value): np.column_stack([p[:, 0], np.full(len(p), c), p[:, 1]])  # noqa: E731
            segs3.append(([lift(ln) for ln in lines], [lift(np.array(a)) for a in arrows]))
    ax, _ = _lead_axes(interp, lead, projection="3d")
    color = ax._get_lines.get_next_color()
    out = []
    for lines, arrows in segs3:
        for ln in lines:
            out.extend(ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], color=color, linewidth=0.8))
        if arrows_on:
            for a in arrows:
                d = a[1] - a[0]
                q = ax.quiver(a[0][0], a[0][1], a[0][2], d[0], d[1], d[2], color=color, length=1.0,
                              arrow_length_ratio=0.9, linewidth=0.8)
                q._mylab_internal = True
    P._redraw(interp)
    return P._ret(interp, "line", out, nargout)


def _tube(line: np.ndarray, radius: np.ndarray, n: int = 12):
    """Surface grids (X, Y, Z) of a tube of varying radius around a 3-D polyline."""
    m = line.shape[0]
    tang = np.gradient(line, axis=0)
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(norm > 0, norm, 1.0)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(tang[0], ref)
    a /= np.linalg.norm(a)
    frames = np.zeros((m, 2, 3))
    for k in range(m):
        a = a - np.dot(a, tang[k]) * tang[k]  # carry the frame along the curve without twisting it
        ln = np.linalg.norm(a)
        a = a / ln if ln > 1e-12 else np.cross(tang[k], ref)
        frames[k, 0] = a
        frames[k, 1] = np.cross(tang[k], a)
    ang = np.linspace(0, 2 * math.pi, n + 1)
    ring = (np.cos(ang)[None, :, None] * frames[:, None, 0, :] + np.sin(ang)[None, :, None] * frames[:, None, 1, :])
    pts = line[:, None, :] + radius[:, None, None] * ring
    return pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]


@builtin("streamtube")
def _streamtube(interp, args, nargout):
    """streamtube(X, Y, Z, U, V, W, sx, sy, sz): streamlines drawn as tubes; narrow where the flow converges."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    for a in args:
        if is_text(a):
            raise MatlabError(f"streamtube: unknown option '{text(a)}'.")
    scale, nside = 1.0, 20
    field = None
    if args and isinstance(args[0], Cell):
        lines = [_real(v) for v in args[0].values_f()]
        rest = args[1:]
        widths = None
        if rest and isinstance(rest[0], np.ndarray) and _real(rest[0]).size == 1:
            widths = float(_vec(rest[0])[0])
            rest = rest[1:]
    else:
        axes, comps, rest_f = _field_args(args, 3, "streamtube")
        if len(rest_f) < 3:
            raise MatlabError("streamtube needs the starting points of the streamlines.")
        lines = _stream_core(args[:9] if len(args) >= 9 else args, 3, "streamtube")
        rest = rest_f[3:]
        widths = None
        field = (axes, comps)
    if rest:
        o = _vec(rest[0])
        scale = float(o[0])
        if o.size > 1:
            nside = int(o[1])
    ax, _ = _lead_axes(interp, lead, projection="3d")
    cmap = _cmap(interp)
    hs = []
    span = None
    for ln in lines:
        if ln.ndim != 2 or ln.shape[0] < 2 or ln.shape[1] != 3:
            continue
        if span is None:
            span = float(np.max(np.ptp(np.vstack([q for q in lines if q.ndim == 2 and q.shape[0]]), axis=0))) or 1.0
        base = 0.02 * span * scale
        if widths is not None:
            radius = np.full(ln.shape[0], widths / 2)
        elif field is not None:
            (x, y, z), (U, V, W) = field
            dvx = np.gradient(U, x, axis=1)
            dvy = np.gradient(V, y, axis=0)
            dvz = np.gradient(W, z, axis=2)
            div = np.nan_to_num(_interp3(x, y, z, dvx + dvy + dvz, ln[:, 0], ln[:, 1], ln[:, 2]))
            speed = np.sqrt(sum(np.nan_to_num(_interp3(x, y, z, c, ln[:, 0], ln[:, 1], ln[:, 2])) ** 2
                                for c in (U, V, W)))
            ds = np.concatenate([[0.0], np.linalg.norm(np.diff(ln, axis=0), axis=1)])
            growth = np.exp(0.5 * np.cumsum(div / np.where(speed > 0, speed, 1.0) * ds))
            radius = base * np.clip(growth, 0.2, 5.0)
        else:
            radius = np.full(ln.shape[0], base)
        X, Y, Z = _tube(ln, radius, nside)
        coll = ax.plot_surface(X, Y, Z, color=cmap(0.6), linewidth=0, edgecolor="none", shade=True,
                               rcount=min(X.shape[0], 200), ccount=X.shape[1])
        coll._mylab_surface = True
        h = P._wrap(interp, "surface", coll)
        h.extra.update({"XData": X, "YData": Y, "ZData": Z})
        hs.append(h)
    P._redraw(interp)
    return P._handle_list(hs) if nargout > 0 else None


@builtin("coneplot")
def _coneplot(interp, args, nargout):
    """coneplot(X, Y, Z, U, V, W, Cx, Cy, Cz, s): velocity vectors drawn as cones at the given points."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    style = "cones"
    nums = []
    for a in args:
        if is_text(a):
            w = text(a).lower()
            if w in ("quiver", "nointerp"):
                style = "quiver" if w == "quiver" else style
            elif w not in ("linear", "nearest", "cubic"):
                raise MatlabError(f"coneplot: unknown option '{text(a)}'.")
        else:
            nums.append(a)
    axes, comps, rest = _field_args(nums, 3, "coneplot")
    if len(rest) < 3:
        raise MatlabError("coneplot needs the cone positions: coneplot(X, Y, Z, U, V, W, Cx, Cy, Cz).")
    cx, cy, cz = (_vec(a) for a in rest[:3])
    if not cx.size == cy.size == cz.size:
        raise MatlabError("coneplot: the cone positions must have the same number of elements.")
    s = float(_vec(rest[3])[0]) if len(rest) > 3 and _real(rest[3]).size == 1 else 1.0
    x, y, z = axes
    vec = np.column_stack([np.nan_to_num(_interp3(x, y, z, c, cx, cy, cz)) for c in comps])
    speed = np.linalg.norm(vec, axis=1)
    top = float(speed.max()) if speed.size else 0.0
    if top == 0:
        raise MatlabError("coneplot: the vector field is zero at every cone position.")
    cell = min(float(np.min(np.abs(np.diff(a)))) if a.size > 1 else 1.0 for a in axes)
    length = s * 1.5 * cell * speed / top if s > 0 else speed
    ax, _ = _lead_axes(interp, lead, projection="3d")
    cmap = _cmap(interp)
    if style == "quiver":
        q = ax.quiver(cx, cy, cz, vec[:, 0], vec[:, 1], vec[:, 2], length=float(1.5 * cell / top) * (s or 1.0),
                      color=cmap(0.3))
        h = P._wrap(interp, "quiver", q)
        P._redraw(interp)
        return h if nargout > 0 else None
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys, vals = [], []
    nside = 10
    ang = np.linspace(0, 2 * math.pi, nside + 1)
    for k in range(cx.size):
        if speed[k] == 0:
            continue
        d = vec[k] / speed[k]
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        a = np.cross(d, ref)
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        base_c = np.array([cx[k], cy[k], cz[k]]) - 0.5 * length[k] * d
        apex = base_c + length[k] * d
        r = 0.3 * length[k]
        ring = base_c + r * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)
        for i in range(nside):
            polys.append(np.array([ring[i], ring[i + 1], apex]))
            vals.append(speed[k])
        polys.append(ring[:-1])
        vals.append(speed[k])
    coll = Poly3DCollection(polys, linewidth=0, edgecolor="none")
    coll.set_cmap(P._current_cmap(interp))
    coll.set_array(np.array(vals))
    had = ax.has_data()
    ax.add_collection3d(coll)
    allp = np.vstack(polys) if polys else np.zeros((0, 3))
    if allp.size:
        ax.auto_scale_xyz(allp[:, 0], allp[:, 1], allp[:, 2], had)
    h = P._wrap(interp, "patch", coll)
    h.extra.update({"axes": ax, "figure": ax.figure, "_polys": polys})
    relight(interp, ax)
    P._redraw(interp)
    return h if nargout > 0 else None


# -- lights ---------------------------------------------------------------------------------------------------------

class LightSource:
    """One light of an axes."""

    def __init__(self, ax, position, style: str = "infinite", color=(1.0, 1.0, 1.0)):
        self.axes = ax
        self.position = np.asarray(position, dtype=float)
        self.style = style
        self.color = tuple(float(c) for c in color)
        self.visible = True

    @property
    def figure(self):
        return self.axes.figure

    def remove(self) -> None:
        lights = getattr(self.axes, "_mylab_lights", [])
        if self in lights:
            lights.remove(self)

    def get_visible(self) -> bool:
        return self.visible

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)


def _lights(ax) -> list:
    return ax.__dict__.setdefault("_mylab_lights", [])


def _camera_direction(ax) -> np.ndarray:
    """Unit vector from the centre of the plot box toward the camera."""
    az, el = getattr(ax, "_mylab_view", None) or (ax.azim + 90, ax.elev)
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


def _material_of(h) -> tuple:
    return tuple(float(np.asarray(h.extra.get(k, d)).flat[0]) for k, d in
                 (("AmbientStrength", 0.3), ("DiffuseStrength", 0.6), ("SpecularStrength", 0.9),
                  ("SpecularExponent", 10.0)))


def _lit_colors(base: np.ndarray, normals: np.ndarray, centers: np.ndarray, ax, material: tuple) -> np.ndarray:
    """Face colours after lighting: ambient + diffuse + specular terms summed over the lights of ``ax``."""
    ka, kd, ks, shin = material
    view = _camera_direction(ax)
    n = normals / np.where(np.linalg.norm(normals, axis=1, keepdims=True) > 0,
                           np.linalg.norm(normals, axis=1, keepdims=True), 1.0)
    # the side of each face that is seen is the one that is lit
    flip = np.einsum("ij,j->i", n, view) < 0
    n[flip] *= -1
    rgb = base[:, :3]
    out = ka * rgb
    box = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    box = np.where(box > 0, box, 1.0)
    for light in _lights(ax):
        if not light.visible:
            continue
        if light.style == "local":
            L = (light.position - centers) / box
        else:
            L = np.tile(light.position / box, (centers.shape[0], 1))
        L = L / np.where(np.linalg.norm(L, axis=1, keepdims=True) > 0, np.linalg.norm(L, axis=1, keepdims=True), 1.0)
        nn = n * box  # normals of the box-scaled surface
        nn = nn / np.where(np.linalg.norm(nn, axis=1, keepdims=True) > 0,
                           np.linalg.norm(nn, axis=1, keepdims=True), 1.0)
        diff = np.clip(np.einsum("ij,ij->i", nn, L), 0, 1)
        half = L + view / np.linalg.norm(view)
        half = half / np.where(np.linalg.norm(half, axis=1, keepdims=True) > 0,
                               np.linalg.norm(half, axis=1, keepdims=True), 1.0)
        spec = np.clip(np.einsum("ij,ij->i", nn, half), 0, 1) ** shin
        lc = np.array(light.color)
        out = out + kd * diff[:, None] * rgb * lc + ks * spec[:, None] * lc * (diff[:, None] > 0)
    res = base.copy()
    res[:, :3] = np.clip(out, 0, 1)
    return res


def _newell(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    return np.array([np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
                     np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
                     np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1]))])


def relight(interp, ax) -> None:
    """Colour the faces of the surfaces and 3-D patches of an axes from its lights and lighting mode."""
    if ax is None or not hasattr(ax, "get_zlim"):
        return
    st = P._state(interp)
    lights = [lt for lt in _lights(ax) if lt.visible]
    for h in list(st.handles.values()):
        if h.kind not in ("surface", "patch") or h.extra.get("deleted"):
            continue
        coll = h.obj
        if getattr(coll, "axes", None) is not ax or not hasattr(coll, "set_facecolor"):
            continue
        if getattr(coll, "_mylab_mesh", False) or coll.__class__.__name__ != "Poly3DCollection":
            continue
        mode = str(h.extra.get("FaceLighting", "flat")).lower()
        polys = h.extra.get("_polys")
        if polys is None and h.kind == "surface":
            X, Y, Z = (np.asarray(h.extra.get(k, _EMPTY), dtype=float) for k in ("XData", "YData", "ZData"))
            if Z.size == 0 or X.shape != Z.shape:
                continue
            rc, cc = min(Z.shape[0], 200), min(Z.shape[1], 200)
            if rc != Z.shape[0] or cc != Z.shape[1] or np.isnan(Z).any():
                continue
            P00 = np.dstack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]])
            P10 = np.dstack([X[1:, :-1], Y[1:, :-1], Z[1:, :-1]])
            P11 = np.dstack([X[1:, 1:], Y[1:, 1:], Z[1:, 1:]])
            P01 = np.dstack([X[:-1, 1:], Y[:-1, 1:], Z[:-1, 1:]])
            normals = np.cross(P11 - P00, P01 - P10).reshape(-1, 3)
            centers = ((P00 + P10 + P11 + P01) / 4).reshape(-1, 3)
        elif polys is not None:
            normals = np.array([_newell(q) for q in polys])
            centers = np.array([np.asarray(q, dtype=float).mean(axis=0) for q in polys])
        else:
            continue
        base = h.extra.get("_unlit")
        if base is None or base.shape[0] != normals.shape[0]:
            try:
                fc = P.face_colors(coll)
            except (ValueError, TypeError):
                continue
            if fc.ndim != 2 or fc.shape[0] not in (1, normals.shape[0]):
                continue
            base = np.repeat(fc, normals.shape[0], axis=0) if fc.shape[0] == 1 else fc
            h.extra["_unlit"] = base.copy()
        if not lights or mode == "none":
            coll.set_facecolor(base)
            continue
        if coll.get_array() is not None:
            coll.set_array(None)  # the face colours are now explicit
        coll.set_facecolor(_lit_colors(base, normals, centers, ax, _material_of(h)))


def _relight_hook(interp, ax) -> None:
    if getattr(ax, "_mylab_lights", None):
        relight(interp, ax)


def _light_pos_set(h, v):
    p = _vec(v)
    if p.size != 3:
        raise MatlabError("Position must be [x y z].")
    h.obj.position = p
    relight(h.interp, h.obj.axes)


def _light_style_set(h, v):
    s = text(v).lower()
    if s not in ("infinite", "local"):
        raise MatlabError("Style must be 'infinite' or 'local'.")
    h.obj.style = s
    relight(h.interp, h.obj.axes)


def _light_color_set(h, v):
    from matplotlib.colors import to_rgb
    h.obj.color = tuple(float(c) for c in to_rgb(P._color(v)))
    relight(h.interp, h.obj.axes)


P._CLASS["light"] = "matlab.graphics.primitive.Light"
P._TYPE["light"] = "light"
P._PROPS["light"] = {
    "Position": (lambda h: P._row(h.obj.position), _light_pos_set),
    "Style": (lambda h: char(h.obj.style), _light_style_set),
    "Color": (lambda h: P._row(h.obj.color), _light_color_set),
}
P._DISPLAY["light"] = ["Color", "Style", "Position", "Visible"]


def _new_light(interp, ax, position, style="infinite", color=(1.0, 1.0, 1.0)):
    lt = LightSource(ax, position, style, color)
    _lights(ax).append(lt)
    h = P._wrap(interp, "light", lt)
    h.extra["_child_of"] = ax
    h.extra["_artists"] = []
    interp.state["relight_hook"] = _relight_hook
    return h


@builtin("light")
def _light(interp, args, nargout):
    """light('Position', [x y z], 'Style', 'infinite' | 'local', 'Color', c): add a light to the axes."""
    args = list(args)
    ax = None
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in P._AXES_KINDS:
        ax = args.pop(0).obj
    pairs = P._name_values(args, "light")
    for i, (name, value) in enumerate(list(pairs)):
        if name.lower() == "parent":
            hs = P._handles_of(value)
            if not hs or hs[0].kind not in P._AXES_KINDS:
                raise MatlabError("The parent of a light must be an axes.")
            ax = hs[0].obj
            pairs.pop(i)
            break
    if ax is None:
        ax = P._axes(interp, fresh=False)
    h = _new_light(interp, ax, [1.0, 0.0, 1.0])
    for name, value in pairs:
        P._set_prop(h, name, value)
    relight(interp, ax)
    P._redraw(interp)
    return h if nargout > 0 else None


def _sph_to_dir(az: float, el: float) -> np.ndarray:
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


@builtin("lightangle")
def _lightangle(interp, args, nargout):
    """lightangle(az, el) adds a light shining from that direction; lightangle(h, az, el) moves one."""
    args = list(args)
    target = None
    ax = None
    if args and isinstance(args[0], P.GraphicsHandle):
        if args[0].kind == "light":
            target = args.pop(0)
        elif args[0].kind in P._AXES_KINDS:
            ax = args.pop(0).obj
        else:
            raise MatlabError("lightangle expects a light or an axes.")
    if target is not None and not args:
        p = target.obj.position
        r = float(np.linalg.norm(p)) or 1.0
        az = math.degrees(math.atan2(p[0], -p[1]))
        el = math.degrees(math.asin(max(min(p[2] / r, 1.0), -1.0)))
        return (P._num_val(az), P._num_val(el)) if nargout > 1 else P._row([az, el])
    if len(args) != 2:
        raise MatlabError("lightangle needs an azimuth and an elevation in degrees.")
    az, el = P._scalar(args[0]), P._scalar(args[1])
    if target is None:
        if ax is None:
            ax = P._axes(interp, fresh=False)
        target = _new_light(interp, ax, _sph_to_dir(az, el))
    else:
        r = float(np.linalg.norm(target.obj.position)) or 1.0
        target.obj.position = r * _sph_to_dir(az, el)
    relight(interp, target.obj.axes)
    P._redraw(interp)
    return target if nargout > 0 else None


@builtin("camlight")
def _camlight(interp, args, nargout):
    """camlight, camlight('headlight' | 'right' | 'left'), camlight(az, el): a light placed from the camera."""
    args = list(args)
    target = None
    ax = None
    if args and isinstance(args[0], P.GraphicsHandle):
        if args[0].kind == "light":
            target = args.pop(0)
            ax = target.obj.axes
        elif args[0].kind in P._AXES_KINDS:
            ax = args.pop(0).obj
        else:
            raise MatlabError("camlight expects a light or an axes.")
    style = "local"
    where = "right"
    az = el = None
    nums = []
    for a in args:
        if is_text(a):
            w = text(a).lower()
            if w in ("headlight", "right", "left"):
                where = w
            elif w in ("local", "infinite"):
                style = w
            else:
                raise MatlabError(f"camlight: unknown option '{text(a)}'.")
        else:
            nums.append(P._scalar(a))
    if len(nums) == 2:
        az, el = nums
    elif nums:
        raise MatlabError("camlight(az, el) needs both angles.")
    if ax is None:
        ax = P._axes(interp, fresh=False)
    if not hasattr(ax, "get_zlim"):
        raise MatlabError("camlight needs 3-D axes; draw a surface first.")
    if az is None:
        az, el = {"headlight": (0.0, 0.0), "right": (30.0, 30.0), "left": (-30.0, 30.0)}[where]
    view_az, view_el = getattr(ax, "_mylab_view", None) or (ax.azim + 90, ax.elev)
    d = _sph_to_dir(view_az + az, min(max(view_el + el, -89.0), 89.0))
    center = np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), np.mean(ax.get_zlim())])
    box = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    pos = center + 2.5 * box * d if style == "local" else box * d
    if target is None:
        target = _new_light(interp, ax, pos, style)
    else:
        target.obj.position = pos
        target.obj.style = style
    relight(interp, ax)
    P._redraw(interp)
    return target if nargout > 0 else None


@builtin("lighting")
def _lighting(interp, args, nargout):
    """lighting flat | gouraud | none: how the lights of the axes shade its surfaces and patches."""
    ax, args = _current_axes(interp, args)
    if not args or not is_text(args[0]):
        raise MatlabError("lighting needs a mode: 'flat', 'gouraud' or 'none'.")
    mode = text(args[0]).lower()
    if mode == "phong":
        mode = "gouraud"
    if mode not in ("flat", "gouraud", "none"):
        raise MatlabError("lighting must be 'flat', 'gouraud' or 'none'.")
    st = P._state(interp)
    for h in list(st.handles.values()):
        if h.kind in ("surface", "patch") and not h.extra.get("deleted") and getattr(h.obj, "axes", None) is ax:
            h.extra["FaceLighting"] = mode
    interp.state["relight_hook"] = _relight_hook
    relight(interp, ax)
    P._redraw(interp)
    return None


_MATERIALS = {"shiny": (0.3, 0.6, 0.9, 20.0, 1.0), "dull": (0.3, 0.8, 0.0, 10.0, 1.0),
              "metal": (0.3, 0.3, 1.0, 25.0, 0.5), "default": (0.3, 0.6, 0.9, 10.0, 1.0)}


@builtin("material")
def _material(interp, args, nargout):
    """material shiny | dull | metal | default, or material([ka kd ks n sc]): how surfaces reflect light."""
    args = list(args)
    targets = None
    if args and isinstance(args[0], (P.GraphicsHandle, P.GraphicsArray)) and not (
            isinstance(args[0], P.GraphicsHandle) and args[0].kind in P._AXES_KINDS):
        targets = P._handles_of(args.pop(0))
    ax, args = _current_axes(interp, args)
    if not args:
        raise MatlabError("material needs 'shiny', 'dull', 'metal', 'default' or [ka kd ks n sc].")
    if is_text(args[0]):
        name = text(args[0]).lower()
        if name not in _MATERIALS:
            raise MatlabError("material must be 'shiny', 'dull', 'metal' or 'default'.")
        vals = list(_MATERIALS[name])
    else:
        v = _vec(args[0])
        if v.size not in (3, 4, 5):
            raise MatlabError("material expects [ka kd ks], [ka kd ks n] or [ka kd ks n sc].")
        vals = list(v) + [10.0, 1.0][v.size - 3:]
    st = P._state(interp)
    if targets is None:
        targets = [h for h in st.handles.values() if h.kind in ("surface", "patch") and not h.extra.get("deleted")
                   and getattr(h.obj, "axes", None) is ax]
    axes_seen = []
    for h in targets:
        if h.kind not in ("surface", "patch"):
            raise MatlabError("material applies to surfaces and patches.")
        for key, val in zip(("AmbientStrength", "DiffuseStrength", "SpecularStrength", "SpecularExponent",
                             "SpecularColorReflectance"), vals):
            h.extra[key] = np.array([[float(val)]])
        a = P._obj_axes(h)
        if a is not None and a not in axes_seen:
            axes_seen.append(a)
    for a in axes_seen or [ax]:
        relight(interp, a)
    P._redraw(interp)
    return None


# -- camera ---------------------------------------------------------------------------------------------------------

_DEFAULT_VA = 10.0


def _camera(ax) -> dict:
    return ax.__dict__.setdefault("_mylab_camera", {"zoom": 1.0, "roll": 0.0, "target": None, "pos": None})


def _need_3d(ax, name: str) -> None:
    if not hasattr(ax, "view_init"):
        raise MatlabError(f"{name} works on 3-D axes; draw with a 3-D function (plot3, surf, ...) first.")


def _box_center(ax) -> np.ndarray:
    return np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), np.mean(ax.get_zlim())])


def _box_size(ax) -> np.ndarray:
    s = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())], dtype=float)
    return np.where(s > 0, s, 1.0)


def _apply_view(interp, ax, az: float, el: float) -> None:
    cam = _camera(ax)
    ax.view_init(elev=el, azim=az - 90, roll=cam["roll"])
    ax._mylab_view = (az, el)
    relight(interp, ax)


def _view_angles(ax) -> tuple:
    return getattr(ax, "_mylab_view", None) or (ax.azim + 90, ax.elev)


def _cam_distance(ax) -> float:
    """Distance from the target to the camera for the current view angle (in box-diagonal units)."""
    va = _DEFAULT_VA / _camera(ax)["zoom"]
    return float(np.linalg.norm(_box_size(ax))) / (2 * math.tan(math.radians(va) / 2))


@builtin("campos")
def _campos(interp, args, nargout):
    """campos([x y z]) places the camera; campos returns where it is."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "campos")
    cam = _camera(ax)
    if not args:
        if cam["pos"] is not None:
            return P._row(cam["pos"])
        az, el = _view_angles(ax)
        target = cam["target"] if cam["target"] is not None else _box_center(ax)
        return P._row(target + _cam_distance(ax) * _sph_to_dir(az, el) * _box_size(ax) / np.linalg.norm(_box_size(ax))
                      * math.sqrt(3))
    if is_text(args[0]):
        mode = text(args[0]).lower()
        if mode == "auto":
            cam["pos"] = None
        elif mode == "mode":
            return char("manual" if cam["pos"] is not None else "auto")
        elif mode != "manual":
            raise MatlabError("campos expects [x y z], 'auto', 'manual' or 'mode'.")
        return None
    p = _vec(args[0])
    if p.size != 3:
        raise MatlabError("campos expects [x y z].")
    target = cam["target"] if cam["target"] is not None else _box_center(ax)
    d = (p - target) / _box_size(ax)  # direction in the proportions of the plot box
    if not np.any(d):
        raise MatlabError("campos: the camera cannot sit on its target.")
    az = math.degrees(math.atan2(d[0], -d[1]))
    el = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    cam["pos"] = p
    _apply_view(interp, ax, az, el)
    P._redraw(interp)
    return None


@builtin("camtarget")
def _camtarget(interp, args, nargout):
    """camtarget([x y z]) sets the point the camera looks at; camtarget returns it."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camtarget")
    cam = _camera(ax)
    if not args:
        return P._row(cam["target"] if cam["target"] is not None else _box_center(ax))
    if is_text(args[0]):
        mode = text(args[0]).lower()
        if mode == "auto":
            cam["target"] = None
        elif mode == "mode":
            return char("manual" if cam["target"] is not None else "auto")
        elif mode != "manual":
            raise MatlabError("camtarget expects [x y z], 'auto', 'manual' or 'mode'.")
        return None
    t = _vec(args[0])
    if t.size != 3:
        raise MatlabError("camtarget expects [x y z].")
    pos = _vec(_campos(interp, [P._wrap(interp, "axes", ax)], 1))
    cam["target"] = t
    # the view always passes through the middle of the plot box: move the limits so that the target is there
    size = _box_size(ax)
    ax.set_xlim(t[0] - size[0] / 2, t[0] + size[0] / 2)
    ax.set_ylim(t[1] - size[1] / 2, t[1] + size[1] / 2)
    ax.set_zlim(t[2] - size[2] / 2, t[2] + size[2] / 2)
    d = (pos - t) / size
    if np.any(d):
        _apply_view(interp, ax, math.degrees(math.atan2(d[0], -d[1])),
                    math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))))
    P._redraw(interp)
    return None


def _set_zoom(ax, zoom: float) -> None:
    cam = _camera(ax)
    cam["zoom"] = float(zoom)
    try:
        ax.set_box_aspect(ax.get_box_aspect(), zoom=float(zoom))
    except (TypeError, ValueError, AttributeError):
        ax._dist = 10.0 / float(zoom)  # older matplotlib: the camera distance


@builtin("camva")
def _camva(interp, args, nargout):
    """camva(angle) sets the camera view angle in degrees (smaller zooms in); camva returns it."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camva")
    cam = _camera(ax)
    if not args:
        return P._num_val(_DEFAULT_VA / cam["zoom"])
    if is_text(args[0]):
        mode = text(args[0]).lower()
        if mode == "auto":
            _set_zoom(ax, 1.0)
        elif mode == "mode":
            return char("auto" if cam["zoom"] == 1.0 else "manual")
        elif mode != "manual":
            raise MatlabError("camva expects an angle, 'auto', 'manual' or 'mode'.")
        P._redraw(interp)
        return None
    va = P._scalar(args[0])
    if not 0 < va < 180:
        raise MatlabError("camva: the view angle must be between 0 and 180 degrees.")
    _set_zoom(ax, _DEFAULT_VA / va)
    P._redraw(interp)
    return None


@builtin("camzoom")
def _camzoom(interp, args, nargout):
    """camzoom(f): zoom in (f > 1) or out (f < 1) by changing the camera view angle."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camzoom")
    if not args:
        raise MatlabError("camzoom needs a zoom factor.")
    f = P._scalar(args[0])
    if f <= 0:
        raise MatlabError("camzoom: the factor must be positive.")
    _set_zoom(ax, _camera(ax)["zoom"] * f)
    P._redraw(interp)
    return None


@builtin("camorbit")
def _camorbit(interp, args, nargout):
    """camorbit(dtheta, dphi): move the camera around its target, horizontally and vertically, in degrees."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camorbit")
    nums = [a for a in args if not is_text(a)]
    for a in args:
        if is_text(a) and text(a).lower() not in ("data", "camera"):
            raise MatlabError(f"camorbit: unknown option '{text(a)}'.")
    if len(nums) < 2:
        raise MatlabError("camorbit needs the horizontal and vertical rotation in degrees.")
    az, el = _view_angles(ax)
    el2 = el + P._scalar(nums[1])
    az2 = az + P._scalar(nums[0])
    if el2 > 90 or el2 < -90:  # passing over the pole turns the camera around
        el2 = math.copysign(180, el2) - el2
        az2 += 180
    _camera(ax)["pos"] = None
    _apply_view(interp, ax, ((az2 + 180) % 360) - 180, el2)
    P._redraw(interp)
    return None


@builtin("camroll")
def _camroll(interp, args, nargout):
    """camroll(dtheta): turn the camera about its viewing axis by dtheta degrees."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camroll")
    if not args:
        raise MatlabError("camroll needs an angle in degrees.")
    cam = _camera(ax)
    cam["roll"] = (cam["roll"] + P._scalar(args[0])) % 360
    az, el = _view_angles(ax)
    _apply_view(interp, ax, az, el)
    P._redraw(interp)
    return None


@builtin("camup")
def _camup(interp, args, nargout):
    """camup([x y z]) sets the direction that points up in the picture; camup returns it."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camup")
    cam = _camera(ax)
    az, el = _view_angles(ax)
    view = _sph_to_dir(az, el)
    right = np.cross(np.array([0.0, 0.0, 1.0]), view)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up0 = np.cross(view, right)
    if not args:
        r = math.radians(cam["roll"])
        if abs(cam["roll"]) < 1e-12:
            return P._row([0.0, 0.0, 1.0])
        return P._row(math.cos(r) * up0 + math.sin(r) * right)
    if is_text(args[0]):
        mode = text(args[0]).lower()
        if mode == "auto":
            cam["roll"] = 0.0
            _apply_view(interp, ax, az, el)
        elif mode == "mode":
            return char("auto" if cam["roll"] == 0 else "manual")
        elif mode != "manual":
            raise MatlabError("camup expects [x y z], 'auto', 'manual' or 'mode'.")
        return None
    u = _vec(args[0])
    if u.size != 3 or not np.any(u):
        raise MatlabError("camup expects a direction [x y z].")
    proj = u - np.dot(u, view) * view  # the part of the direction that is seen
    if np.linalg.norm(proj) < 1e-9:
        raise MatlabError("camup: the up direction cannot point along the line of sight.")
    cam["roll"] = math.degrees(math.atan2(np.dot(proj, right), np.dot(proj, up0))) % 360
    _apply_view(interp, ax, az, el)
    P._redraw(interp)
    return None


@builtin("camproj")
def _camproj(interp, args, nargout):
    """camproj('orthographic' | 'perspective') sets the projection; camproj returns it."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camproj")
    if not args:
        return char(getattr(ax, "_mylab_proj", "orthographic"))
    mode = text(args[0]).lower()
    if mode not in ("orthographic", "perspective"):
        raise MatlabError("camproj must be 'orthographic' or 'perspective'.")
    ax.set_proj_type("ortho" if mode == "orthographic" else "persp")
    ax._mylab_proj = mode
    h = P._wrap(interp, "axes", ax)
    h.extra["Projection"] = mode
    P._redraw(interp)
    return None


@builtin("camdolly")
def _camdolly(interp, args, nargout):
    """camdolly(dx, dy, dz): move the camera and its target sideways, up and toward the scene."""
    ax, args = _current_axes(interp, args)
    _need_3d(ax, "camdolly")
    nums = [a for a in args if not is_text(a)]
    if len(nums) < 3:
        raise MatlabError("camdolly needs dx, dy and dz.")
    dx, dy, dz = (P._scalar(a) for a in nums[:3])
    az, el = _view_angles(ax)
    view = _sph_to_dir(az, el)
    right = np.cross(np.array([0.0, 0.0, 1.0]), view)
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(view, right)
    size = _box_size(ax)
    shift = (dx * right + dy * up - dz * view) * size * 0.5
    ax.set_xlim(*(np.array(ax.get_xlim()) + shift[0]))
    ax.set_ylim(*(np.array(ax.get_ylim()) + shift[1]))
    ax.set_zlim(*(np.array(ax.get_zlim()) + shift[2]))
    P._redraw(interp)
    return None


@builtin("camlookat")
def _camlookat(interp, args, nargout):
    """camlookat(h): fit the view to the objects h (or to everything in the axes)."""
    args = list(args)
    hs = []
    ax = None
    if args:
        hs = P._handles_of(args[0])
        if hs and hs[0].kind in P._AXES_KINDS:
            ax = hs[0].obj
            hs = []
    if ax is None:
        ax = P._obj_axes(hs[0]) if hs else P._axes(interp, fresh=False)
    _need_3d(ax, "camlookat")
    if not hs:
        hs = P._axes_children(interp, ax)
    pts = []
    for h in hs:
        for key in (("XData", "YData", "ZData"),):
            try:
                xyz = [np.asarray(P._get_prop(h, k), dtype=float).flatten() for k in key]
            except (MatlabError, ValueError, TypeError):
                continue
            if xyz[0].size and xyz[0].size == xyz[1].size == xyz[2].size:
                pts.append(np.column_stack(xyz))
        polys = h.extra.get("_polys")
        if polys:
            pts.append(np.vstack([q for q in polys if np.asarray(q).shape[1] == 3]))
    if not pts:
        raise MatlabError("camlookat: the objects have no 3-D data to look at.")
    allp = np.vstack(pts)
    allp = allp[np.all(np.isfinite(allp), axis=1)]
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = np.where(hi > lo, (hi - lo) * 0.05, 0.5)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    P._redraw(interp)
    return None


def _camera_prop(fn_name: str):
    def g(h):
        if not hasattr(h.obj, "view_init"):
            return {"campos": P._row([0, 0, 1]), "camtarget": P._row([0.5, 0.5, 0]), "camup": P._row([0, 1, 0]),
                    "camva": P._num_val(_DEFAULT_VA)}[fn_name]
        return BUILTINS[fn_name](h.interp, [h], 1)

    def s(h, v):
        BUILTINS[fn_name](h.interp, [h, v], 0)
    return g, s


for _prop, _fn in (("CameraPosition", "campos"), ("CameraTarget", "camtarget"), ("CameraUpVector", "camup"),
                   ("CameraViewAngle", "camva")):
    P._AXES_PROPS[_prop] = _camera_prop(_fn)
    P._AXES_PROPS[_prop + "Mode"] = (lambda h, _f=_fn: BUILTINS[_f](h.interp, [h, char("mode")], 1)
                                     if hasattr(h.obj, "view_init") else char("auto"),
                                     lambda h, v, _f=_fn: BUILTINS[_f](h.interp, [h, v], 0))


# -- groups and transforms -----------------------------------------------------------------------------------------

class Group:
    """An hggroup or hgtransform: a set of objects handled as one, optionally moved by a 4-by-4 matrix."""

    def __init__(self, interp, ax, transform: bool):
        self.interp = interp
        self.axes = ax
        self.children: list = []
        self.matrix = np.eye(4)
        self.transform = transform
        self.visible = True

    @property
    def figure(self):
        return self.axes.figure

    def adopt(self, h) -> None:
        if h.kind in P._GROUP_KINDS:
            if h.obj is self or self in h.obj.ancestors():
                raise MatlabError("A group cannot contain itself.")
        old = h.extra.get("_part_of")
        if old is not None and old.obj is not self:
            old.obj.release(h)
        if h not in self.children:
            self.children.append(h)
        h.extra["_part_of"] = P._wrap(self.interp, "hgtransform" if self.transform else "hggroup", self)
        self.apply(h)

    def release(self, h) -> None:
        if h in self.children:
            self.children.remove(h)
        h.extra.pop("_part_of", None)
        _move(h, np.eye(4))

    def ancestors(self) -> list:
        out = []
        h = P._wrap(self.interp, "hgtransform" if self.transform else "hggroup", self)
        parent = h.extra.get("_part_of")
        while parent is not None:
            out.append(parent.obj)
            parent = parent.extra.get("_part_of")
        return out

    def total_matrix(self) -> np.ndarray:
        M = self.matrix
        for g in self.ancestors():
            M = g.matrix @ M
        return M

    def apply(self, h=None) -> None:
        M = self.total_matrix()
        for c in ([h] if h is not None else list(self.children)):
            if c.kind in P._GROUP_KINDS:
                c.obj.apply()
            else:
                _move(c, M)

    def remove(self) -> None:
        for c in list(self.children):
            P._delete_handle(self.interp, c)
        self.children = []

    def get_visible(self) -> bool:
        return self.visible

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        for c in self.children:
            P._set_prop(c, "Visible", char("on" if on else "off"))


def _move(h, M: np.ndarray) -> None:
    """Draw a plot object as transformed by the 4-by-4 matrix M (its own data stay as they are)."""
    identity = np.allclose(M, np.eye(4))
    arts = P._artists(h.obj)
    ax = P._obj_axes(h)
    if ax is None:
        return
    if not hasattr(ax, "get_zlim"):
        from matplotlib.transforms import Affine2D
        A = np.array([[M[0, 0], M[0, 1], M[0, 3]], [M[1, 0], M[1, 1], M[1, 3]], [0.0, 0.0, 1.0]])
        tr = Affine2D(A) + ax.transData
        for a in arts:
            if hasattr(a, "set_transform"):
                a.set_transform(ax.transData if identity else tr)
        ax.relim()
        if not identity:  # the data limits must cover the moved objects
            pts = _object_points(h)
            if pts is not None and pts.size:
                moved = (np.column_stack([pts[:, :2], np.zeros(len(pts)), np.ones(len(pts))]) @ M.T)[:, :2]
                ax.update_datalim(moved)
        ax.autoscale_view()
        return
    # 3-D objects: the coordinates themselves are recomputed from the untransformed data
    obj = h.obj
    if h.kind in ("line", "animatedline", "stair") and hasattr(obj, "get_data_3d"):
        base = h.extra.get("_base_xyz")
        if base is None:
            base = h.extra["_base_xyz"] = np.column_stack([np.asarray(v, dtype=float) for v in obj.get_data_3d()])
        moved = (np.column_stack([base, np.ones(len(base))]) @ M.T)[:, :3]
        obj.set_data_3d(moved[:, 0], moved[:, 1], moved[:, 2])
        ax.auto_scale_xyz(moved[:, 0], moved[:, 1], moved[:, 2], True)
    elif h.kind == "surface" and "ZData" in h.extra:
        base = h.extra.get("_base_grid")
        if base is None:
            base = h.extra["_base_grid"] = tuple(np.asarray(h.extra[k], dtype=float) for k in ("XData", "YData",
                                                                                                "ZData"))
        X, Y, Z = base
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.size)]) @ M.T
        saved = {k: h.extra[k] for k in ("XData", "YData", "ZData")}
        h.extra["XData"], h.extra["YData"], h.extra["ZData"] = (pts[:, k].reshape(X.shape) for k in range(3))
        from .plot_objects import redraw_surface
        redraw_surface(h)
        h.extra.update(saved)
    elif h.kind == "patch" and h.extra.get("_polys") is not None and hasattr(obj, "set_verts"):
        base = h.extra.get("_base_polys")
        if base is None:
            base = h.extra["_base_polys"] = [np.asarray(q, dtype=float) for q in h.extra["_polys"]]
        moved = [(np.column_stack([q, np.ones(len(q))]) @ M.T)[:, :3] for q in base if q.shape[1] == 3]
        if moved:
            obj.set_verts(moved)
            allp = np.vstack(moved)
            ax.auto_scale_xyz(allp[:, 0], allp[:, 1], allp[:, 2], True)
    elif not identity:
        raise MatlabError(f"A {P._short(h.kind)} object in 3-D axes cannot be moved by an hgtransform.")


def _object_points(h):
    try:
        x = np.asarray(P._get_prop(h, "XData"), dtype=float).flatten()
        y = np.asarray(P._get_prop(h, "YData"), dtype=float).flatten()
        if x.size == y.size and x.size:
            return np.column_stack([x, y])
    except (MatlabError, ValueError, TypeError):
        pass
    polys = h.extra.get("_polys")
    if polys:
        return np.vstack([np.asarray(q, dtype=float)[:, :2] for q in polys])
    if h.kind == "rectangle":
        p = h.extra["Position"]
        return np.array([[p[0], p[1]], [p[0] + p[2], p[1] + p[3]], [p[0], p[1] + p[3]], [p[0] + p[2], p[1]]])
    return None


def _matrix_set(h, v):
    M = _real(v)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        raise MatlabError("Matrix must be a 4-by-4 transform, e.g. from makehgtform.")
    if not np.allclose(M[3], [0, 0, 0, 1]):
        raise MatlabError("Matrix: the last row of a transform must be [0 0 0 1].")
    h.obj.matrix = M
    h.obj.apply()


def _group_children(h):
    kids = [c for c in h.obj.children if P._alive(c)]
    kids = kids[::-1]
    if not kids:
        return np.zeros((0, 1))
    return kids[0] if len(kids) == 1 else P.GraphicsArray(kids)


for _kind, _cls in (("hggroup", "matlab.graphics.primitive.Group"),
                    ("hgtransform", "matlab.graphics.primitive.Transform")):
    P._CLASS[_kind] = _cls
    P._TYPE[_kind] = _kind
    P._PROPS[_kind] = {"Children": (_group_children, None),
                       "DisplayName": (lambda h: char(h.extra.get("DisplayName", "")),
                                       lambda h, v: h.extra.__setitem__("DisplayName", text(v)))}
    P._DISPLAY[_kind] = ["Children", "Visible"]
P._PROPS["hgtransform"]["Matrix"] = (lambda h: h.obj.matrix.copy(), _matrix_set)
P._DISPLAY["hgtransform"] = ["Children", "Matrix", "Visible"]


def _make_group(interp, args, nargout, transform: bool):
    name = "hgtransform" if transform else "hggroup"
    args = list(args)
    parent = None
    if args and isinstance(args[0], P.GraphicsHandle):
        parent = args.pop(0)
    pairs = P._name_values(args, name)
    for i, (nm, value) in enumerate(list(pairs)):
        if nm.lower() == "parent":
            hs = P._handles_of(value)
            if not hs:
                raise MatlabError("Parent must be an axes or a group.")
            parent = hs[0]
            pairs.pop(i)
            break
    if parent is not None and parent.kind not in P._AXES_KINDS + P._GROUP_KINDS:
        raise MatlabError(f"The parent of an {name} must be an axes or another group.")
    if parent is None:
        ax = P._axes(interp, fresh=False)
    else:
        ax = parent.obj if parent.kind in P._AXES_KINDS else parent.obj.axes
    g = Group(interp, ax, transform)
    h = P._wrap(interp, name, g)
    h.extra["_child_of"] = ax
    h.extra["_artists"] = []
    h.extra["axes"] = ax
    h.extra["figure"] = ax.figure
    if parent is not None and parent.kind in P._GROUP_KINDS:
        parent.obj.adopt(h)
    for nm, value in pairs:
        P._set_prop(h, nm, value)
    return h


builtin("hggroup")(lambda interp, args, nargout: _make_group(interp, args, nargout, False))
builtin("hgtransform")(lambda interp, args, nargout: _make_group(interp, args, nargout, True))


@builtin("makehgtform")
def _makehgtform(interp, args, nargout):
    """M = makehgtform('translate', [tx ty tz], 'scale', s, 'xrotate', t, ...): a 4-by-4 transform matrix."""
    M = np.eye(4)
    args = list(args)
    k = 0

    def rot(axis: np.ndarray, t: float) -> np.ndarray:
        u = axis / np.linalg.norm(axis)
        c, s = math.cos(t), math.sin(t)
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        R = np.eye(4)
        R[:3, :3] = c * np.eye(3) + s * K + (1 - c) * np.outer(u, u)
        return R

    while k < len(args):
        if not is_text(args[k]):
            raise MatlabError("makehgtform expects transform names such as 'translate', 'scale' or 'zrotate'.")
        op = text(args[k]).lower()
        need = {"axisrotate": 2}.get(op, 1)
        vals = args[k + 1:k + 1 + need]
        if len(vals) < need:
            raise MatlabError(f"makehgtform: '{op}' needs {'two values' if need == 2 else 'a value'}.")
        T = np.eye(4)
        if op == "translate":
            t = _vec(vals[0])
            if t.size == 1 and k + 3 < len(args) and not is_text(args[k + 2]) and not is_text(args[k + 3]):
                t = np.array([t[0], P._scalar(args[k + 2]), P._scalar(args[k + 3])])
                need = 3
            if t.size != 3:
                raise MatlabError("makehgtform: 'translate' needs [tx ty tz].")
            T[:3, 3] = t
        elif op == "scale":
            s = _vec(vals[0])
            if s.size == 1:
                s = np.repeat(s, 3)
            if s.size != 3 or np.any(s == 0):
                raise MatlabError("makehgtform: 'scale' needs one non-zero factor or [sx sy sz].")
            T[0, 0], T[1, 1], T[2, 2] = s
        elif op in ("xrotate", "yrotate", "zrotate"):
            axis = {"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}[op[0]]
            T = rot(np.array(axis), P._scalar(vals[0]))
        elif op == "axisrotate":
            axis = _vec(vals[0])
            if axis.size != 3 or not np.any(axis):
                raise MatlabError("makehgtform: 'axisrotate' needs an axis [ax ay az] and an angle.")
            T = rot(axis, P._scalar(vals[1]))
        else:
            raise MatlabError(f"makehgtform: unknown transform '{op}'.")
        M = M @ T
        k += 1 + need
    return M
