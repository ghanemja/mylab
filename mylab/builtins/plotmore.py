"""More plot types: quiver3, surface, pcolor, surfl, trisurf, trimesh, triplot, stem3,
xline, yline, alpha, boxplot, gobjects."""

from __future__ import annotations

import re

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, is_char, to_str
from ._registry import builtin
from ._util import fnum, is_text, text
from .plotting import (
    GraphicsArray, GraphicsHandle, _axes, _color, _current_cmap, _is_linespec, _onoff, _redraw, _ret,
    _scalar, _split_name_values, _state, _with_axes_arg, _wrap,
)


def _reals(args: list) -> list[np.ndarray]:
    return [np.real(fnum(a)).astype(float) for a in args]


def _spec_color(spec: str):
    m = re.search(r"[rgbcmykw]", spec or "")
    return m.group(0) if m else None


# -- 3-D arrows --------------------------------------------------------------------------------------

@builtin("quiver3")
@_with_axes_arg
def _quiver3(interp, args, nargout):
    args, opts = _split_name_values(args, {"color", "linewidth", "autoscale", "autoscalefactor",
                                           "displayname", "maxheadsize", "showarrowhead"})
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    nums = _reals([a for a in args if not is_text(a)])
    scale = 1.0
    if len(nums) in (5, 7):
        scale = float(nums.pop().flat[0])
    if len(nums) == 4:
        Z, U, V, W = nums
        X, Y = np.meshgrid(np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1))
    elif len(nums) == 6:
        X, Y, Z, U, V, W = nums
        if X.shape != Z.shape and X.size * Y.size == Z.size:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    else:
        raise MatlabError("quiver3 expects quiver3(Z, U, V, W) or quiver3(X, Y, Z, U, V, W).")
    sizes = {a.size for a in (X, Y, Z, U, V, W)}
    if len(sizes) != 1:
        raise MatlabError("quiver3: the position and direction arrays must all be the same size.")
    if "autoscalefactor" in opts:
        scale = _scalar(opts["autoscalefactor"])
    autoscale = not ("autoscale" in opts and not _onoff(opts["autoscale"])) and scale != 0
    x, y, z, u, v, w = (a.flatten() for a in (X, Y, Z, U, V, W))
    k = 1.0
    if autoscale and x.size:
        longest = float(np.nanmax(np.sqrt(u ** 2 + v ** 2 + w ** 2)))
        extent = max(np.ptp(x), np.ptp(y), np.ptp(z))
        spacing = extent / max(round(x.size ** (1 / 2)), 1) if extent > 0 else 1.0
        k = 0.9 * scale * spacing / longest if longest > 0 else 1.0
    color = _spec_color(spec)
    if "color" in opts:
        color = _color(opts["color"])
    kw: dict = {"color": color if color is not None else "C0", "arrow_length_ratio": 0.25}
    if "showarrowhead" in opts and not _onoff(opts["showarrowhead"]):
        kw["arrow_length_ratio"] = 0.0
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    ax = _axes(interp, projection="3d")
    q = ax.quiver(x, y, z, u * k, v * k, w * k, **kw)
    ends = np.column_stack([np.concatenate([x, x + u * k]), np.concatenate([y, y + v * k]),
                            np.concatenate([z, z + w * k])])
    if ends.size:
        ax.auto_scale_xyz(ends[:, 0], ends[:, 1], ends[:, 2])
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "quiver", q)
    h.extra.update({"XData": X, "YData": Y, "ZData": Z, "UData": U, "VData": V, "WData": W})
    return h


# -- surfaces -----------------------------------------------------------------------------------------

@builtin("surface")
@_with_axes_arg
def _surface_fn(interp, args, nargout):
    """Low-level surf: adds a surface to the current axes without clearing them."""
    data = list(args)
    # Name-value only form: surface('XData', X, 'YData', Y, 'ZData', Z, ...)
    pairs = {}
    k = 0
    while k + 1 < len(data):
        if is_text(data[k]) and text(data[k]).lower() in ("xdata", "ydata", "zdata", "cdata"):
            pairs[text(data[k]).lower()] = data[k + 1]
            del data[k:k + 2]
        else:
            k += 1
    if "zdata" in pairs:
        lead = [pairs[n] for n in ("xdata", "ydata") if n in pairs]
        data = (lead if len(lead) == 2 else []) + [pairs["zdata"]] + data
    if not any(not is_text(a) for a in data):
        raise MatlabError("surface needs data: surface(Z) or surface(X, Y, Z).")
    st = _state(interp)
    ax = _axes(interp, fresh=False)
    was_held = st.hold.get(id(ax), False)
    st.hold[id(ax)] = True
    try:
        res = interp.call_builtin("surf", interp.builtins["surf"], data, nargout)
    finally:
        now = _axes(interp, fresh=False)
        if now is not ax:
            st.hold.pop(id(ax), None)
        st.hold[id(now)] = was_held
    return res[0] if res else None


@builtin("pcolor")
@_with_axes_arg
def _pcolor(interp, args, nargout):
    args, opts = _split_name_values(args, {"edgecolor", "facecolor", "facealpha", "linestyle", "linewidth"})
    nums = _reals([a for a in args if not is_text(a)])
    if len(nums) == 1:
        C = nums[0]
        X, Y = np.meshgrid(np.arange(1.0, C.shape[1] + 1), np.arange(1.0, C.shape[0] + 1))
    elif len(nums) == 3:
        X, Y, C = nums
        if X.shape != C.shape:
            if X.size != C.shape[1] or Y.size != C.shape[0]:
                raise MatlabError("pcolor: X and Y must match the size of C (one value per column and row).")
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    else:
        raise MatlabError("pcolor expects pcolor(C) or pcolor(X, Y, C).")
    if C.ndim != 2 or min(C.shape) < 2:
        raise MatlabError("pcolor: C must be a matrix with at least two rows and two columns.")
    shade = interp.state.get("shading", "faceted")
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"cmap": _current_cmap(interp)}
    if shade == "interp":
        mesh = ax.pcolormesh(X, Y, C, shading="gouraud", **kw)
    else:
        # Each cell is coloured by its lower-left corner; the last row and column are not drawn.
        kw["edgecolors"] = "k" if shade == "faceted" else "none"
        kw["linewidth"] = 0.3 if shade == "faceted" else 0
        mesh = ax.pcolormesh(X, Y, C[:-1, :-1], shading="flat", **kw)
    if "edgecolor" in opts:
        ec = opts["edgecolor"]
        mesh.set_edgecolor("none" if is_text(ec) and text(ec).lower() in ("none", "flat", "interp") else _color(ec))
    if "facealpha" in opts:
        mesh.set_alpha(_scalar(opts["facealpha"]))
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "surface", mesh)
    h.extra.update({"XData": X, "YData": Y, "ZData": np.zeros_like(C), "CData": C})
    return h


@builtin("surfl")
def _surfl(interp, args, nargout):
    """A surface with lighting-like shading (drawn as a shaded surf)."""
    data = [a for a in args if not is_text(a)]
    ax_arg = [data.pop(0)] if data and isinstance(data[0], GraphicsHandle) else []
    if len(data) in (2, 4, 5):  # trailing light direction / reflectance arguments are ignored
        data = data[:1] if len(data) == 2 else data[:3]
    res = interp.call_builtin("surf", interp.builtins["surf"], ax_arg + data, max(nargout, 1))
    try:
        from matplotlib.colors import LightSource
        h = res[0]
        Z = h.extra.get("ZData")
        if Z is not None and Z.ndim == 2 and min(Z.shape) > 1:
            import matplotlib
            cmap = matplotlib.colormaps[_current_cmap(interp)] if isinstance(_current_cmap(interp), str) \
                else _current_cmap(interp)
            rgb = LightSource(azdeg=315, altdeg=45).shade(Z, cmap=cmap, blend_mode="soft")
            h.obj.set_facecolors(rgb[:-1, :-1].reshape(-1, 4)) if rgb[:-1, :-1].reshape(-1, 4).shape[0] == \
                len(h.obj.get_facecolors()) else None
    except Exception:  # noqa: BLE001 - the shading is cosmetic; the surface itself is already drawn
        pass
    _redraw(interp)
    return res[0] if nargout > 0 else None


def _triangles(tri, npoints: int, what: str) -> np.ndarray:
    t = np.real(fnum(tri))
    if t.ndim != 2 or t.shape[1] != 3:
        raise MatlabError(f"{what}: the triangulation must have three columns (one triangle per row), "
                          "as returned by delaunay.")
    t = t.astype(int) - 1
    if t.size and (t.min() < 0 or t.max() >= npoints):
        raise MatlabError(f"{what}: the triangulation refers to points that do not exist.")
    return t


@builtin("trimesh")
def _trimesh(interp, args, nargout):
    from ..values import char
    return _trisurf(interp, list(args) + [char("FaceColor"), char("none"), char("EdgeColor"), char("C0")],
                    nargout)


@builtin("trisurf")
@_with_axes_arg
def _trisurf(interp, args, nargout):
    args, opts = _split_name_values(args, {"edgecolor", "facecolor", "facealpha", "linewidth"})
    nums = [a for a in args if not is_text(a)]
    if len(nums) < 4:
        raise MatlabError("trisurf expects trisurf(T, x, y, z) with T from delaunay(x, y).")
    x, y, z = (a.flatten() for a in _reals(nums[1:4]))
    if not (x.size == y.size == z.size):
        raise MatlabError("trisurf: x, y and z must have the same number of elements.")
    t = _triangles(nums[0], x.size, "trisurf")
    ax = _axes(interp, projection="3d")
    kw: dict = {"cmap": _current_cmap(interp), "edgecolor": "k", "linewidth": 0.3}
    if "edgecolor" in opts:
        ec = opts["edgecolor"]
        kw["edgecolor"] = "none" if is_text(ec) and text(ec).lower() == "none" else _color(ec)
    if "facealpha" in opts:
        kw["alpha"] = _scalar(opts["facealpha"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    coll = ax.plot_trisurf(x, y, z, triangles=t, **kw)
    if "facecolor" in opts and not (is_text(opts["facecolor"]) and
                                    text(opts["facecolor"]).lower() in ("flat", "interp")):
        fc = opts["facecolor"]
        coll.set_facecolor("none" if is_text(fc) and text(fc).lower() == "none" else _color(fc))
    _redraw(interp)
    return _ret(interp, "patch", [coll], nargout)


@builtin("triplot")
@_with_axes_arg
def _triplot(interp, args, nargout):
    args, opts = _split_name_values(args, {"color", "linewidth"})
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    nums = [a for a in args if not is_text(a)]
    if len(nums) < 3:
        raise MatlabError("triplot expects triplot(T, x, y) with T from delaunay(x, y).")
    x, y = (a.flatten() for a in _reals(nums[1:3]))
    t = _triangles(nums[0], x.size, "triplot")
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"linewidth": _scalar(opts["linewidth"]) if "linewidth" in opts else 0.8,
                "color": _spec_color(spec) or "C0", "linestyle": "-"}
    style = re.sub(r"[rgbcmykw]", "", spec)
    for dash in ("--", "-.", ":", "-"):
        if dash in style:
            kw["linestyle"] = dash
            style = style.replace(dash, "")
            break
    if style:
        kw["marker"] = {"p": "p", "h": "h", "d": "d", "s": "s"}.get(style[0], style[0])
    if "color" in opts:
        kw["color"] = _color(opts["color"])
    lines = ax.triplot(x, y, t, **kw)
    _redraw(interp)
    return _ret(interp, "line", [lines[0]], nargout)


@builtin("stem3")
@_with_axes_arg
def _stem3(interp, args, nargout):
    nums = _reals([a for a in args if not is_text(a)])
    filled = any(is_text(a) and text(a).lower() == "filled" for a in args)
    if len(nums) == 1:
        Z = nums[0]
        if 1 in Z.shape:
            X, Y = np.arange(1.0, Z.size + 1).reshape(Z.shape), np.ones(Z.shape)
            if Z.shape[1] == 1:
                X, Y = Y, X
        else:
            X, Y = np.meshgrid(np.arange(1.0, Z.shape[1] + 1), np.arange(1.0, Z.shape[0] + 1))
    elif len(nums) == 3:
        X, Y, Z = nums
    else:
        raise MatlabError("stem3 expects stem3(Z) or stem3(X, Y, Z).")
    ax = _axes(interp, projection="3d")
    x, y, z = X.flatten(), Y.flatten(), Z.flatten()
    cont = ax.stem(x, y, z, basefmt=" ")
    if not filled:
        cont.markerline.set_markerfacecolor("none")
    _redraw(interp)
    return _ret(interp, "stem", [cont], nargout)


# -- reference lines ------------------------------------------------------------------------------

def _ref_line(interp, args, nargout, vertical: bool):
    name = "xline" if vertical else "yline"
    args, opts = _split_name_values(args, {"color", "linewidth", "linestyle", "label", "displayname",
                                           "labelhorizontalalignment", "labelverticalalignment",
                                           "labelorientation", "alpha", "handlevisibility"})
    if not args or is_text(args[0]):
        raise MatlabError(f"{name} needs a position, e.g. {name}(2.5).")
    values = np.real(fnum(args[0])).flatten()
    rest = list(args[1:])
    spec = ""
    labels = None
    for a in rest:
        if is_text(a) and _is_linespec(text(a)) and not spec:
            spec = text(a)
        elif is_text(a) or isinstance(a, (Cell, StringArray)):
            labels = a
    if "label" in opts:
        labels = opts["label"]
    if labels is None:
        texts: list[str] = []
    elif isinstance(labels, Cell):
        texts = [to_str(x) for x in labels.data.flatten(order="F")]
    elif isinstance(labels, StringArray):
        texts = [x or "" for x in labels.data.flatten(order="F")]
    elif isinstance(labels, MString) or is_char(labels):
        texts = [to_str(labels)]
    else:
        texts = []
    ax = _axes(interp, fresh=False, projection=None)
    kw: dict = {"color": _spec_color(spec) or "k", "linewidth": 0.8}
    style = re.sub(r"[rgbcmykw]", "", spec)
    if style in ("-", "--", ":", "-."):
        kw["linestyle"] = style
    if "color" in opts:
        kw["color"] = _color(opts["color"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "linestyle" in opts:
        kw["linestyle"] = text(opts["linestyle"])
    if "alpha" in opts:
        kw["alpha"] = _scalar(opts["alpha"])
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    lines = []
    for k, value in enumerate(values):
        ln = ax.axvline(value, **kw) if vertical else ax.axhline(value, **kw)
        lines.append(ln)
        label = texts[k] if k < len(texts) else (texts[0] if len(texts) == 1 else "")
        if label:
            if vertical:
                ax.annotate(label, xy=(value, 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                            textcoords="offset points", rotation=90, va="top", ha="left", fontsize=9)
            else:
                ax.annotate(label, xy=(1, value), xycoords=("axes fraction", "data"), xytext=(-3, 3),
                            textcoords="offset points", va="bottom", ha="right", fontsize=9)
    _redraw(interp)
    if nargout <= 0:
        return None
    hs = [_wrap(interp, "line", ln) for ln in lines]
    return hs[0] if len(hs) == 1 else GraphicsArray(hs)


builtin("xline")(_with_axes_arg(lambda interp, args, nargout: _ref_line(interp, args, nargout, True)))
builtin("yline")(_with_axes_arg(lambda interp, args, nargout: _ref_line(interp, args, nargout, False)))


# -- transparency -----------------------------------------------------------------------------------

@builtin("alpha")
def _alpha(interp, args, nargout):
    if not args:
        raise MatlabError("alpha needs a transparency value between 0 and 1, e.g. alpha(0.5).")
    targets = None
    rest = list(args)
    if isinstance(rest[0], (GraphicsHandle, GraphicsArray)):
        first = rest.pop(0)
        targets = first.items if isinstance(first, GraphicsArray) else [first]
    if not rest:
        raise MatlabError("alpha needs a transparency value between 0 and 1.")
    v = rest[0]
    if is_text(v):
        word = text(v).lower()
        value = {"clear": 0.0, "opaque": 1.0}.get(word)
        if value is None:
            return None  # 'flat', 'interp', 'scaled' ... : per-vertex transparency is not drawn
    else:
        value = _scalar(v)
        if not 0 <= value <= 1:
            raise MatlabError("alpha: the transparency must be between 0 (clear) and 1 (opaque).")
    artists = []
    if targets is None or any(t.kind in ("axes", "figure") for t in targets):
        ax = _axes(interp, fresh=False)
        artists = list(ax.collections) + list(ax.patches) + list(ax.images)
    else:
        for t in targets:
            obj = t.obj
            artists += list(obj) if isinstance(obj, (list, tuple)) else [obj]
    for a in artists:
        if hasattr(a, "set_alpha"):
            a.set_alpha(value)
    _redraw(interp)
    return None


# -- box plots -------------------------------------------------------------------------------------

@builtin("boxplot", "boxchart")
@_with_axes_arg
def _boxplot(interp, args, nargout):
    args, opts = _split_name_values(args, {"labels", "notch", "orientation", "whisker", "widths", "colors",
                                           "symbol", "positions", "boxstyle", "outliersize", "plotstyle"})
    if not args:
        raise MatlabError("boxplot needs data: boxplot(x) or boxplot(x, groups).")
    x = np.real(fnum(args[0])).astype(float)
    groups = args[1] if len(args) > 1 else None
    labels: list[str] | None = None
    if groups is not None:
        from .ndmath import group_index
        idx, names, _ = group_index(groups)
        flat = x.flatten(order="F")
        if idx.size != flat.size:
            raise MatlabError("boxplot: the grouping variable must have one entry per data value.")
        series = [flat[idx == k + 1] for k in range(len(names))]
        labels = names
    elif x.ndim == 2 and min(x.shape) > 1:
        series = [x[:, j] for j in range(x.shape[1])]
    else:
        series = [x.flatten()]
    series = [s[~np.isnan(s)] for s in series]
    if "labels" in opts:
        lab = opts["labels"]
        if isinstance(lab, Cell):
            labels = [to_str(v) if (is_char(v) or isinstance(v, MString)) else f"{float(np.asarray(v).flat[0]):g}"
                      for v in lab.data.flatten(order="F")]
        elif isinstance(lab, StringArray):
            labels = [v or "" for v in lab.data.flatten(order="F")]
        elif is_char(lab) and lab.ndim == 2 and lab.shape[0] > 1:
            labels = ["".join(r).strip() for r in lab]
        elif is_text(lab):
            labels = [text(lab)]
        else:
            labels = [f"{v:g}" for v in np.real(fnum(lab)).flatten()]
        if len(labels) != len(series):
            raise MatlabError(f"boxplot: there are {len(series)} boxes but {len(labels)} labels.")
    kw: dict = {}
    if "notch" in opts:
        kw["notch"] = _onoff(opts["notch"]) if not (is_text(opts["notch"]) and
                                                    text(opts["notch"]).lower() == "marker") else False
    if "whisker" in opts:
        kw["whis"] = _scalar(opts["whisker"])
    if "widths" in opts:
        kw["widths"] = _scalar(opts["widths"])
    if "positions" in opts:
        kw["positions"] = list(np.real(fnum(opts["positions"])).flatten())
    if "orientation" in opts and text(opts["orientation"]).lower() == "horizontal":
        kw["vert"] = False
    ax = _axes(interp, projection="rectilinear")
    res = ax.boxplot(series, **kw)
    ticks = kw.get("positions", list(range(1, len(series) + 1)))
    names = labels if labels is not None else [str(k) for k in range(1, len(series) + 1)]
    if kw.get("vert", True):
        ax.set_xticks(ticks)
        ax.set_xticklabels(names)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(names)
    _redraw(interp)
    return _ret(interp, "line", list(res["boxes"]), nargout)


# -- placeholders for graphics handles -----------------------------------------------------------------

@builtin("gobjects")
def _gobjects(interp, args, nargout):
    """An array to hold graphics handles; mylab pre-allocates it as an empty cell-like list."""
    dims = [int(np.real(fnum(a)).flat[0]) for a in args if not is_text(a)]
    if len(args) == 1 and not is_text(args[0]) and np.asarray(args[0]).size > 1:
        dims = [int(v) for v in np.real(fnum(args[0])).flatten()]
    if not dims:
        dims = [1, 1]
    if len(dims) == 1:
        dims = [dims[0], dims[0]]
    n = int(np.prod(dims))
    if n == 0:
        return GraphicsArray([], row=dims[0] <= 1)
    return GraphicsArray([GraphicsHandle(interp, "placeholder", None) for _ in range(n)], row=dims[0] == 1)
