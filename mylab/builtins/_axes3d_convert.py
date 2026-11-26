"""Turning a 2-D axes into a 3-D one and back (view(3) on a plot, view(2) after it).

Matplotlib keeps 2-D and 3-D axes apart, while every MATLAB axes can be seen from any direction. When a
2-D axes is viewed in 3-D, a 3-D axes takes its place and the objects drawn in it move across: lines,
patches, collections and text become their 3-D counterparts at z = 0 *as the same Python objects*, so
graphics handles to them stay valid. Images are shown as a flat textured surface while the image object
itself stays the child the handle refers to. Title, axis labels, legend, grid and limits move along.
"""

from __future__ import annotations

import numpy as np

# Attributes of an axes that belong to its kind (2-D or 3-D) and are not carried across.
_NOT_COPIED = frozenset({"_mylab_view", "_mylab_padded", "_mylab_restore2d", "_mylab_from2d",
                         "_mylab_lims2d"})


def _points(artist) -> np.ndarray:
    """The (n, 3) data points of a converted artist, for the limits of the 3-D axes."""
    for name in ("_verts3d", "_offsets3d"):
        v = getattr(artist, name, None)
        if v is not None:
            cols = np.broadcast_arrays(*[np.ma.filled(np.asarray(c, dtype=float), np.nan).ravel() for c in v])
            return np.column_stack(cols)
    seg = getattr(artist, "_segment3d", None)
    if seg is not None:
        return np.asarray(seg, dtype=float).reshape(-1, 3)
    segs = getattr(artist, "_segments3d", None)
    if segs is not None:
        parts = [np.asarray(s, dtype=float).reshape(-1, 3) for s in segs if len(s)]
        return np.vstack(parts) if parts else np.zeros((0, 3))
    vec = getattr(artist, "_vec", None)
    if vec is not None:
        return np.asarray(vec[:3], dtype=float).T
    return np.zeros((0, 3))


class _FlatImage:
    """An image in a 3-D axes: a flat surface draws it, and the image object stays the child."""

    def draw(self, renderer):  # the surface draws the picture
        return None

    def remove(self):
        surf = self.__dict__.pop("_mylab_flat3d", None)
        if surf is not None:
            _detach(surf)
        super().remove()


def _image_surface(new, img):
    """A z = 0 surface coloured like the image (rows and columns thinned to at most 300)."""
    arr = img.get_array()
    if arr is None:
        return None
    rgba = img.to_rgba(arr, norm=True) if np.ndim(arr) == 2 else np.asarray(arr, dtype=float)
    if rgba.dtype.kind in "iu":
        rgba = rgba / 255.0
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.ones(rgba.shape[:2] + (1,))], axis=-1)
    rows, cols = rgba.shape[:2]
    sr, sc = max(1, rows // 300), max(1, cols // 300)
    rgba = rgba[::sr, ::sc]
    left, right, bottom, top = img.get_extent()
    # row 0 of the image is at the top with origin 'upper', at the bottom with origin 'lower'
    ys = np.linspace(top, bottom, rgba.shape[0] + 1) if img.origin == "upper" \
        else np.linspace(bottom, top, rgba.shape[0] + 1)
    xs = np.linspace(left, right, rgba.shape[1] + 1)
    X, Y = np.meshgrid(xs, ys)
    surf = new.plot_surface(X, Y, np.zeros_like(X), facecolors=rgba, shade=False, rstride=1, cstride=1,
                            linewidth=0, antialiased=False)
    surf._mylab_internal = True
    surf._mylab_image_of = img
    return surf


def _retarget(artist, old, new) -> bool:
    """Point the transforms and clipping of ``artist`` at ``new`` instead of ``old``; True when the artist
    is placed in data units."""
    from matplotlib.artist import Artist
    base = Artist.get_transform(artist)
    in_data = base is old.transData
    if in_data:
        artist.set_transform(new.transData)
    elif base is old.transAxes:
        artist.set_transform(new.transAxes)
    if hasattr(artist, "get_offset_transform") and artist.get_offset_transform() is old.transData:
        artist.set_offset_transform(new.transData)
    if artist.get_clip_path() is not None or artist.get_clip_box() is not None:
        artist.set_clip_path(new.patch)
    return in_data


def _detach(artist) -> None:
    try:
        artist.remove()
    except (NotImplementedError, ValueError):
        pass


def _attach(new, artist) -> None:
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    if isinstance(artist, Line2D):
        new.add_line(artist)
    elif isinstance(artist, Collection):
        new.add_collection(artist, autolim=False)
    elif isinstance(artist, AxesImage):
        new.add_image(artist)
    elif isinstance(artist, Patch):
        from mpl_toolkits.mplot3d import art3d
        if isinstance(artist, art3d.Patch3D) and not hasattr(artist, "_path2d"):
            # add_patch reads the path; Matplotlib before 3.9 has it only after the patch is first projected
            artist.axes = new
            new.M = new.get_proj()
            artist.do_3d_projection()
        new.add_patch(artist)
    elif isinstance(artist, Text):
        new._add_text(artist)


def _to3d(new, old, artist, restore: list) -> None:
    """Move ``artist`` from the 2-D axes ``old`` into the 3-D axes ``new`` at z = 0."""
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d import art3d
    cls = artist.__class__
    _detach(artist)
    in_data = _retarget(artist, old, new)
    if isinstance(artist, Text) and not in_data:  # text placed in axes units stays put on the new axes
        new._add_text(artist)
        restore.append((artist, cls, None))
        return
    saved = None
    if isinstance(artist, Line2D):
        art3d.line_2d_to_3d(artist, zs=0)
    elif isinstance(artist, Text):
        art3d.text_2d_to_3d(artist, z=0)
    elif isinstance(artist, PathPatch):
        saved = artist.get_path()
        art3d.pathpatch_2d_to_3d(artist, z=0)
    elif isinstance(artist, Patch):
        art3d.patch_2d_to_3d(artist, z=0)
    elif isinstance(artist, PathCollection):
        saved = np.array(artist.get_offsets(), dtype=float)
        art3d.patch_collection_2d_to_3d(artist, zs=np.zeros(len(saved)))
    elif isinstance(artist, LineCollection):
        saved = [np.array(s, dtype=float) for s in artist.get_segments()]
        art3d.line_collection_2d_to_3d(artist, zs=0)
    elif isinstance(artist, PolyCollection):
        saved = [np.array(p.vertices, dtype=float) for p in artist.get_paths()]
        art3d.poly_collection_2d_to_3d(artist, zs=0)
    elif isinstance(artist, AxesImage):
        surf = _image_surface(new, artist)
        artist._mylab_flat3d = surf
        artist.__class__ = type("AxesImage", (_FlatImage, cls), {})
        new.add_image(artist)
        restore.append((artist, cls, None))
        return
    else:  # something without a 3-D form: left out of the 3-D view
        restore.append((artist, cls, "dropped"))
        return
    _attach(new, artist)
    restore.append((artist, cls, saved))


def _to2d(old, new, artist, cls, saved) -> None:
    """Undo :func:`_to3d` for one artist, putting it into the 2-D axes ``new``."""
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    _detach(artist)
    _retarget(artist, old, new)
    surf = artist.__dict__.pop("_mylab_flat3d", None)
    if surf is not None:
        _detach(surf)
    if isinstance(saved, str):
        _attach(new, artist)
        return
    v3 = getattr(artist, "_verts3d", None)
    artist.__class__ = cls
    if issubclass(cls, Line2D) and v3 is not None:
        artist.set_data(np.asarray(v3[0]), np.asarray(v3[1]))
    elif issubclass(cls, PathPatch) and saved is not None:
        artist.set_path(saved)
    elif issubclass(cls, PathCollection) and saved is not None:
        artist.set_offsets(saved)
    elif issubclass(cls, LineCollection) and saved is not None:
        artist.set_segments(saved)
    elif issubclass(cls, PolyCollection) and saved is not None:
        artist.set_verts(saved)
    _attach(new, artist)


def _move_legend(old, new) -> None:
    leg = old.get_legend()
    if leg is None:
        return
    old.legend_ = None
    leg.parent = new
    leg._axes = new  # the Legend object moves (its handle stays valid)
    leg._remove_method = new._remove_legend
    leg.set_bbox_to_anchor(None)
    new.legend_ = leg


def _move_containers(old, new) -> None:
    """Bar, error-bar and stem containers stay listed with their axes."""
    for c in list(old.containers):
        new.containers.append(c)
        c._remove_method = new.containers.remove
    old.containers.clear()


def _copy_text(src, dst) -> None:
    dst.update_from(src)
    dst.set_text(src.get_text())
    dst.set_visible(src.get_visible())


def _texts(ax) -> list:
    out = [ax.title, ax.xaxis.label, ax.yaxis.label]
    if hasattr(ax, "zaxis"):
        out.append(ax.zaxis.label)
    return out


def _grid_on(ax) -> bool:
    if hasattr(ax, "_draw_grid"):  # a 3-D axes keeps one switch for its grid
        return bool(ax._draw_grid)
    on = getattr(ax.xaxis, "_major_tick_kw", {}).get("gridOn")
    if on is not None:
        return bool(on)
    try:
        return any(gl.get_visible() for gl in ax.xaxis.get_gridlines())
    except (AttributeError, IndexError):
        return bool(getattr(ax.xaxis, "_major_tick_kw", {}).get("gridOn", False))


def children_of(ax) -> list:
    return list(ax.lines) + list(ax.collections) + list(ax.images) + list(ax.patches) + list(ax.texts)


def detach_2d(old) -> dict:
    """Everything to carry from a 2-D axes before it is replaced by a 3-D one."""
    return {"children": children_of(old), "grid": _grid_on(old),
            "xlim": old.get_xlim(), "ylim": old.get_ylim(),
            "auto": (old.get_autoscalex_on(), old.get_autoscaley_on()),
            "scales": (old.get_xscale(), old.get_yscale()),
            "attrs": {k: v for k, v in old.__dict__.items() if k.startswith("_mylab_") and k not in _NOT_COPIED}}


def attach_3d(interp, old, new, info: dict) -> None:
    """Fill the new 3-D axes ``new`` with what ``old`` held (see :func:`detach_2d`)."""
    from .plotting import _state
    restore: list = []
    pts = []
    for a in info["children"]:
        _to3d(new, old, a, restore)
        if not isinstance(a, _FlatImage):
            pts.append(_points(a))
        elif getattr(a, "_mylab_flat3d", None) is not None:
            x0, x1, y0, y1 = a.get_extent()
            pts.append(np.array([[x0, y0, 0.0], [x1, y1, 0.0]]))
    for k, v in info["attrs"].items():
        setattr(new, k, v)
    new._mylab_restore2d = restore
    new._mylab_from2d = True
    _move_legend(old, new)
    _move_containers(old, new)
    for src, dst in zip(_texts(old), _texts(new)):
        _copy_text(src, dst)
    _remap(_state(interp), _texts(old), _texts(new))
    new.grid(info["grid"])
    allp = np.vstack([p for p in pts if p.size]) if any(p.size for p in pts) else np.zeros((0, 3))
    allp = allp[np.all(np.isfinite(allp), axis=1)] if allp.size else allp
    xlim, ylim = info["xlim"], info["ylim"]
    new.set_xlim(*xlim)  # the 2-D limits carry over; z spans [-1 1] around flat data, as in MATLAB
    new.set_ylim(*ylim)
    new.set_autoscalex_on(info["auto"][0])
    new.set_autoscaley_on(info["auto"][1])
    new._mylab_lims2d = (tuple(new.get_xlim()), tuple(new.get_ylim()), info["xlim"], info["ylim"], info["auto"])
    if new.get_autoscalez_on():
        zs = allp[:, 2] if allp.size else np.zeros(0)
        lo, hi = (float(zs.min()), float(zs.max())) if zs.size else (0.0, 0.0)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        new.set_zlim(lo, hi)
        new.set_autoscalez_on(True)
    if allp.size:
        new.xy_dataLim.update_from_data_xy(allp[:, :2], ignore=True)
        new.zz_dataLim.update_from_data_xy(np.column_stack([allp[:, 2], allp[:, 2]]), ignore=True)


def can_return_to_2d(ax) -> bool:
    """A 3-D axes made from a 2-D one by view(3) that has had nothing 3-D added since."""
    restore = getattr(ax, "_mylab_restore2d", None)
    if restore is None:
        return False
    converted = {id(a) for a, _, _ in restore}
    for a in children_of(ax):
        if id(a) in converted or getattr(a, "_mylab_internal", False) and getattr(a, "_mylab_image_of", None):
            continue
        return False
    for a, _, _ in restore:
        v = getattr(a, "_verts3d", None)
        if v is not None and np.any(np.asarray(v[2], dtype=float) != 0):
            return False
    return True


def detach_3d(old) -> dict:
    xlim, ylim = tuple(old.get_xlim()), tuple(old.get_ylim())
    auto = (old.get_autoscalex_on(), old.get_autoscaley_on())
    saved = getattr(old, "_mylab_lims2d", None)
    if saved is not None and saved[0] == xlim and saved[1] == ylim:  # untouched in 3-D: the 2-D limits return
        xlim, ylim, auto = saved[2], saved[3], saved[4]
    return {"restore": [r for r in old._mylab_restore2d if r[0].axes is old or isinstance(r[2], str)],
            "grid": _grid_on(old), "xlim": xlim, "ylim": ylim, "auto": auto,
            "attrs": {k: v for k, v in old.__dict__.items() if k.startswith("_mylab_") and k not in _NOT_COPIED}}


def attach_2d(interp, old, new, info: dict) -> None:
    from .plotting import _state
    for a, cls, saved in info["restore"]:
        _to2d(old, new, a, cls, saved)
    for k, v in info["attrs"].items():
        setattr(new, k, v)
    _move_legend(old, new)
    _move_containers(old, new)
    for src, dst in zip(_texts(old)[:3], _texts(new)):
        _copy_text(src, dst)
    _remap(_state(interp), _texts(old)[:3], _texts(new))
    new.grid(info["grid"])
    new.set_xlim(*info["xlim"])
    new.set_ylim(*info["ylim"])
    new.set_autoscalex_on(info["auto"][0])
    new.set_autoscaley_on(info["auto"][1])
    new.relim()
    if any(info["auto"]):
        new.autoscale_view()


def _remap(st, olds: list, news: list) -> None:
    """Handles to the title and axis labels follow them to the new axes."""
    for o, n in zip(olds, news):
        h = st.handles.pop(id(o), None)
        if h is not None:
            h.obj = n
            st.handles[id(n)] = h
