"""Figure files: savefig / openfig / hgsave / hgload, and copyobj.

A figure is saved as a *description*: a tree of nodes ``{type, handle, properties, children, special}``
for the figure, its axes and everything drawn in them, with MATLAB's property names. ``openfig`` reads
the tree and draws the figure again by calling the ordinary plotting functions, so a saved figure comes
back as live objects (lines, axes, labels, legends, subplots) that can be changed and saved again.
``copyobj`` uses the same descriptions to duplicate objects.

On disk a ``.fig`` file is a MAT-file (version 5) holding the tree in the variable ``hgS_070000``, the
layout MATLAB itself uses for figure files. Details MATLAB has no property for (date rulers, exact bar
geometry) are kept in a second variable, ``mylab_fig``. Reading accepts files written by mylab and plain
MATLAB figure files; objects it has no description for are skipped with a warning.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, Struct, char, is_char, to_str
from . import plotting as P
from ._registry import BUILTINS
from ._util import fnum, is_text, text

_STYLE = {
    "line": ["Color", "LineStyle", "LineWidth", "Marker", "MarkerSize", "MarkerFaceColor", "MarkerEdgeColor"],
    "stair": ["Color", "LineStyle", "LineWidth", "Marker", "MarkerSize"],
    "animatedline": ["Color", "LineStyle", "LineWidth", "Marker", "MarkerSize"],
    "scatter": ["Marker", "LineWidth"],
    "bar": ["EdgeColor", "LineWidth", "FaceAlpha"],
    "histogram": ["FaceColor", "EdgeColor", "FaceAlpha", "LineWidth"],
    "surface": ["FaceAlpha", "LineWidth"],
    "patch": ["FaceAlpha", "LineWidth"],
    "area": ["FaceAlpha", "LineWidth"],
    "image": [],
    "text": ["FontSize", "FontWeight", "FontAngle", "Color", "Rotation", "HorizontalAlignment",
             "VerticalAlignment"],
    "errorbar": ["Color", "LineWidth", "LineStyle", "Marker", "MarkerSize", "CapSize"],
    "stem": ["Color", "LineWidth", "Marker", "MarkerSize"],
    "quiver": ["Color", "LineWidth"],
    "rectangle": ["FaceColor", "EdgeColor", "LineWidth", "FaceAlpha"],
    "contour": [],
}
_COMMON = ["DisplayName", "Tag", "Visible", "HandleVisibility"]

# MATLAB's names for the same kinds of object in figure files written by different releases
_TYPE_ALIASES = {
    "graph2d.lineseries": "line", "lineseries": "line", "line": "line", "primitiveline": "line",
    "specgraph.barseries": "bar", "barseries": "bar", "bar": "bar",
    "specgraph.scattergroup": "scatter", "scattergroup": "scatter", "scatter": "scatter",
    "specgraph.stemseries": "stem", "stemseries": "stem", "stem": "stem",
    "specgraph.stairseries": "stair", "stairseries": "stair", "stair": "stair",
    "specgraph.errorbarseries": "errorbar", "errorbarseries": "errorbar", "errorbar": "errorbar",
    "specgraph.areaseries": "area", "areaseries": "area", "area": "area",
    "specgraph.contourgroup": "contour", "contourgroup": "contour", "contour": "contour",
    "specgraph.quivergroup": "quiver", "quivergroup": "quiver", "quiver": "quiver",
    "graph3d.surfaceplot": "surface", "surfaceplot": "surface", "surface": "surface",
    "scribe.legend": "legend", "legend": "legend", "scribe.colorbar": "colorbar", "colorbar": "colorbar",
    "histogram": "histogram", "image": "image", "patch": "patch", "text": "text", "rectangle": "rectangle",
    "figure": "figure", "axes": "axes", "polaraxes": "polaraxes", "hggroup": "group", "hgtransform": "group",
    "animatedline": "line", "constantline": "constantline",
}


def _node(kind: str, props: dict, children: list | None = None, extra: dict | None = None) -> dict:
    return {"type": kind, "properties": props, "children": children or [], "special": [], "extra": extra or {}}


def _get(h, name: str):
    try:
        return P._get_prop(h, name)
    except (MatlabError, ValueError, TypeError, AttributeError, IndexError, KeyError, NotImplementedError):
        return None


def _props(h, names: list) -> dict:
    out = {}
    for n in names:
        v = _get(h, n)
        if v is not None and not isinstance(v, (P.GraphicsHandle, P.GraphicsArray)):
            out[n] = v
    return out


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


# -- describing ---------------------------------------------------------------------------------------

def describe_object(interp, h) -> dict | None:
    """The description of one plotted object, or None if this kind of object cannot be described."""
    kind = h.kind
    obj = h.obj
    props = _props(h, _STYLE.get(kind, []) + _COMMON)
    extra: dict = {}
    if kind in ("line", "stair", "animatedline"):
        if getattr(obj, "_mylab_constant", None) is not None:
            axis, value = obj._mylab_constant
            props.update(Value=np.array([[float(value)]]), InterceptAxis=char(axis))
            if getattr(obj, "_mylab_affect_limits", False):
                props["AffectAutoLimits"] = char("on")
            return _node("constantline", props)
        props["XData"], props["YData"] = _f(obj.get_xdata()).reshape(1, -1), _f(obj.get_ydata()).reshape(1, -1)
        if hasattr(obj, "get_data_3d"):
            props["ZData"] = _f(obj.get_data_3d()[2]).reshape(1, -1)
        return _node("stair" if kind == "stair" else "line", props)
    if kind == "scatter":
        off = P._sc_offsets(h)
        props["XData"], props["YData"] = off[:, 0].reshape(1, -1), off[:, 1].reshape(1, -1)
        z3 = getattr(obj, "_offsets3d", None)
        if z3 is not None:
            props["XData"], props["YData"], props["ZData"] = (_f(v).reshape(1, -1) for v in z3)
        props["SizeData"] = _f(obj.get_sizes()).reshape(1, -1)
        arr = obj.get_array()
        if arr is not None:
            props["CData"] = _f(arr).reshape(1, -1)
        extra["facecolors"] = _f(obj.get_facecolor())
        extra["edgecolors"] = _f(obj.get_edgecolor())
        extra["marker"] = str(h.extra.get("Marker", "o"))
        return _node("scatter", props, extra=extra)
    if kind == "bar":
        from matplotlib.colors import to_rgba
        rects = [p for p in P._patches(h) if hasattr(p, "get_height")]
        props.update(XData=_f(h.extra.get("XData", [])).reshape(1, -1),
                     YData=_f(h.extra.get("YData", [])).reshape(1, -1),
                     BarWidth=np.array([[float(h.extra.get("BarWidth", 0.8))]]),
                     BarLayout=char(str(h.extra.get("BarLayout", "grouped"))),
                     Horizontal=char("on" if h.extra.get("Horizontal") else "off"))
        if rects:
            props["FaceColor"] = P._rgb(rects[0].get_facecolor())
        extra["rects"] = np.array([[p.get_x(), p.get_y(), p.get_width(), p.get_height()] for p in rects]).reshape(-1, 4)
        extra["colors"] = np.array([to_rgba(p.get_facecolor()) for p in rects]).reshape(-1, 4)
        return _node("bar", props, extra=extra)
    if kind == "histogram":
        props.update(Data=_f(obj.data).reshape(1, -1), BinEdges=_f(obj.edges).reshape(1, -1),
                     Normalization=char(obj.normalization), Orientation=char(obj.orientation),
                     DisplayStyle=char(getattr(obj, "display_style", "bar")))
        if obj.counts is not None:
            extra["counts"] = _f(obj.counts).reshape(1, -1)
        return _node("histogram", props, extra=extra)
    if kind == "surface":
        X, Y, Z = (h.extra.get(k) for k in ("XData", "YData", "ZData"))
        if X is None or Z is None:
            return None
        props.update(XData=_f(X), YData=_f(Y), ZData=_f(Z))
        if h.extra.get("CData") is not None:
            props["CData"] = _f(h.extra["CData"])
        name = obj.__class__.__name__
        extra["style"] = "pcolor" if name == "QuadMesh" else "mesh" if (
            name == "Line3DCollection" or getattr(obj, "_mylab_mesh", False)) else "surf"
        if hasattr(obj, "get_edgecolor") and name != "QuadMesh":
            ec = _f(obj.get_edgecolor())
            extra["edge"] = ec[0] if ec.ndim == 2 and len(ec) else np.zeros(0)
        return _node("surface", props, extra=extra)
    if kind in ("patch", "pie"):
        polys = h.extra.get("_polys")
        if polys is None:
            arts = P._artists(obj)
            polys = [_f(a.get_xy())[:-1] for a in arts if hasattr(a, "get_xy")]
            if not polys and hasattr(obj, "get_paths"):
                polys = [_f(pth.vertices) for pth in obj.get_paths()]
        if not polys:
            return None
        width = max(len(q) for q in polys)
        verts = np.vstack(polys)
        faces = np.full((len(polys), width), np.nan)
        k = 1
        for i, q in enumerate(polys):
            faces[i, :len(q)] = np.arange(k, k + len(q))
            k += len(q)
        props.update(Vertices=verts, Faces=faces)
        arts = P._artists(obj)
        fc = P.face_colors(arts[0]) if arts else np.zeros((0, 4))
        ec = _f(arts[0].get_edgecolor()) if arts else np.zeros((0, 4))
        extra["facecolors"] = np.atleast_2d(fc) if len(arts) == 1 else np.array(
            [np.atleast_2d(_f(a.get_facecolor()))[0] for a in arts])
        extra["edgecolors"] = np.atleast_2d(ec)
        arr = arts[0].get_array() if arts and hasattr(arts[0], "get_array") else None
        if arr is not None:
            props["FaceVertexCData"] = _f(arr).reshape(-1, 1)
        return _node("patch", props, extra=extra)
    if kind == "area":
        props.update(XData=_f(h.extra.get("XData", [])).reshape(1, -1),
                     YData=_f(h.extra.get("YData", [])).reshape(1, -1),
                     BaseValue=_f(h.extra.get("BaseValue", 0.0)).reshape(1, 1))
        props["FaceColor"] = P._rgb(obj.get_facecolor())
        props["EdgeColor"] = P._rgb(obj.get_edgecolor())
        if obj.get_paths():
            extra["outline"] = _f(obj.get_paths()[0].vertices)
        return _node("area", props, extra=extra)
    if kind == "image":
        arr = np.asarray(obj.get_array())
        props.update(CData=_f(arr), XData=_get(h, "XData"), YData=_get(h, "YData"),
                     CDataMapping=char(str(h.extra.get("CDataMapping", "scaled"))))
        extra["clim"] = _f(obj.get_clim()).reshape(1, 2) if arr.ndim == 2 else np.zeros((1, 0))
        extra["aspect"] = str(obj.axes.get_aspect()) if obj.axes is not None else "auto"
        if arr.ndim == 2:
            cm = obj.get_cmap()
            extra["cmap"] = _f(cm(np.linspace(0, 1, min(getattr(cm, "N", 256), 256))))[:, :3]
        return _node("image", props, extra=extra)
    if kind == "text":
        raw = getattr(obj, "_mylab_raw", None)
        props["String"] = char(raw if raw is not None else obj.get_text())
        props["Position"] = _get(h, "Position")
        props["Interpreter"] = _get(h, "Interpreter")
        units = str(h.extra.get("Units", "data"))
        if units != "data":
            props["Units"] = char(units)
        return _node("text", props)
    if kind == "contour":
        info = h.extra.get("_contour")
        if info is None:
            return None
        props.update(XData=_f(info["X"]), YData=_f(info["Y"]), ZData=_f(info["Z"]),
                     LevelList=_f(obj.levels).reshape(1, -1), Fill=char("on" if info.get("fill") else "off"),
                     ShowText=char("on" if info.get("showtext") else "off"))
        colors = info.get("kw", {}).get("colors")
        if colors:
            props["LineColor"] = P._rgb(colors[0])
        return _node("contour", props)
    if kind == "errorbar":
        for name in ("XData", "YData", "YNegativeDelta", "YPositiveDelta", "XNegativeDelta", "XPositiveDelta"):
            props[name] = _f(h.extra.get(name, [])).reshape(1, -1)
        return _node("errorbar", props)
    if kind == "stem":
        ln = obj.markerline
        props.update(XData=_f(ln.get_xdata()).reshape(1, -1), YData=_f(ln.get_ydata()).reshape(1, -1))
        extra["filled"] = str(ln.get_markerfacecolor()) != "none"
        return _node("stem", props, extra=extra)
    if kind == "quiver":
        for name in ("XData", "YData", "UData", "VData"):
            if h.extra.get(name) is None:
                return None
            props[name] = _f(h.extra[name])
        return _node("quiver", props)
    if kind == "rectangle":
        props.update(Position=_f(h.extra["Position"]).reshape(1, 4), Curvature=_f(h.extra["Curvature"]).reshape(1, 2))
        return _node("rectangle", props)
    return None


def _text_node(interp, artist) -> dict:
    h = P._wrap(interp, "text", artist)
    raw = getattr(artist, "_mylab_raw", None)
    props = _props(h, ["FontSize", "FontWeight", "Color", "Interpreter"])
    props["String"] = char(raw if raw is not None else artist.get_text())
    return _node("text", props)


def describe_axes(interp, ax) -> dict:
    """The description of an axes with its labels, the objects drawn in it, its legend and colorbar."""
    h = P._wrap(interp, P._axes_kind(ax), ax)
    three_d = hasattr(ax, "get_zlim")
    polar = getattr(ax, "name", "") == "polar"
    st = P._state(interp)
    props: dict = {"Units": char("normalized"), "Position": _f(ax.get_position().bounds).reshape(1, 4),
                   "Visible": P._oo(getattr(ax, "_axis3don", ax.axison)), "FontSize": P._fontsize_get(h),
                   "NextPlot": char("add" if st.hold.get(id(ax), False) else "replace")}
    extra: dict = {"kind": "polar" if polar else "3d" if three_d else "2d"}
    for which in ("x", "y") + (("z",) if three_d else ()):
        up = which.upper()
        lo, hi = getattr(ax, f"get_{which}lim")()
        props[f"{up}Lim"] = np.array([[min(lo, hi), max(lo, hi)]], dtype=float)
        if not polar:
            props[f"{up}Scale"] = char("log" if getattr(ax, f"get_{which}scale")() == "log" else "linear")
            props[f"{up}Dir"] = char("reverse" if getattr(ax, f"{which}axis_inverted")() else "normal")
        fixed = type(getattr(ax, f"{which}axis").get_major_locator()).__name__ == "FixedLocator"
        if (getattr(ax, f"_mylab_{which}ticks_fixed", False) or fixed) and not polar:
            props[f"{up}Tick"] = _f(getattr(ax, f"get_{which}ticks")()).reshape(1, -1)
        labels = getattr(ax, f"_mylab_{which}ticklabels", None)
        if labels is not None:
            props[f"{up}TickLabel"] = Cell.from_list([char(s) for s in labels])
        r = P._ruler_of(ax, which)
        if r is not None:
            extra[f"ruler_{which}"] = {"kind": r.kind, "zone": r.zone, "fmt": r.fmt or "", "unit": float(r.unit),
                                       "dfmt": r.dfmt, "classic": bool(r.classic),
                                       "categories": Cell.from_list([char(c) for c in r.categories])}
        fmt = getattr(ax, "_mylab_tickfmt", {}).get(which)
        if fmt:
            extra[f"tickfmt_{which}"] = fmt
    if not polar and not three_d:
        props["XGrid"], props["YGrid"] = _get(h, "XGrid"), _get(h, "YGrid")
        props["Box"] = _get(h, "Box")
        props["XAxisLocation"], props["YAxisLocation"] = _get(h, "XAxisLocation"), _get(h, "YAxisLocation")
        aspect = ax.get_aspect()
        extra["aspect"] = "auto" if aspect == "auto" else float(aspect)
        extra["adjustable"] = str(ax.get_adjustable())
        if ax.get_box_aspect() is not None:
            extra["box_aspect"] = float(ax.get_box_aspect())
    if three_d:
        props["View"] = _get(h, "View")
    props["Color"] = P._rgb(ax.get_facecolor())
    ms = P._mappables(ax)
    if ms:
        props["CLim"] = _f(ms[-1].get_clim()).reshape(1, 2)
    side = getattr(ax, "_mylab_yyside", None)
    if side is not None:
        extra["yyside"] = side
    children: list = []
    for g in reversed(P._axes_children(interp, ax)):  # creation order
        node = describe_object(interp, g)
        if node is None:
            skipped = interp.state.setdefault("fig_skipped", set())
            skipped.add(P._short(g.kind))
            continue
        children.append(node)
    node = _node("polaraxes" if polar else "axes", props, children, extra)
    labels = [ax.title, ax.xaxis.label, ax.yaxis.label] + ([ax.zaxis.label] if three_d else [])
    for t in labels:
        node["children"].append(_text_node(interp, t))
    n = len(node["children"])
    node["special"] = list(range(n - len(labels) + 1, n + 1))  # 1-based: title, xlabel, ylabel[, zlabel]
    sub = getattr(ax, "_mylab_subtitle", None)
    if sub is not None and sub.get_text():
        node["extra"]["subtitle"] = getattr(sub, "_mylab_raw", sub.get_text())
    leg = ax.get_legend()
    if leg is None and getattr(ax, "_mylab_yyright", None) is not None:
        leg = ax._mylab_yyright.get_legend()
    if leg is not None:
        lh = P._wrap(interp, "legend", leg)
        lp = {"String": P._legend_strings(lh), "Location": char(getattr(leg, "_mylab_loc", "northeast")),
              "Orientation": char(getattr(leg, "_mylab_orientation", "vertical")), "Box": P._oo(leg.get_frame_on()),
              "Visible": P._oo(leg.get_visible())}
        if leg.get_texts():
            lp["FontSize"] = P._num_val(leg.get_texts()[0].get_fontsize())
        if leg.get_title().get_text():
            lp["Title"] = char(getattr(leg, "_mylab_title_raw", None) or leg.get_title().get_text())
        node["extra"]["legend"] = lp
        handles = getattr(leg, "_mylab_handles", None)
        if handles is not None:
            order = [g.obj for g in reversed(P._axes_children(interp, ax))]
            idx = []
            for a in handles:
                hit = [i for i, o in enumerate(order) if o is a or a in P._artists(o)]
                idx.append(hit[0] + 1 if hit else 0)
            node["extra"]["legend_items"] = np.array([idx], dtype=float)
    cb = getattr(ax, "_mylab_colorbar", None)
    if cb is not None:
        ch = P._wrap(interp, "colorbar", cb)
        lab = P._cb_label(ch).obj
        node["extra"]["colorbar"] = {"Location": char(str(ch.extra.get("Location", "eastoutside"))),
                                     "Label": char(getattr(lab, "_mylab_raw", None) or lab.get_text()),
                                     "Position": _f(cb.ax.get_position().bounds).reshape(1, 4)}
    return node


def describe_figure(interp, fh) -> dict:
    fig = fh.obj
    try:
        fig.canvas.draw()  # layouts place the axes when the figure is drawn
    except (ValueError, RuntimeError):
        pass
    props = {"Name": char(getattr(fig, "_mylab_name", "")), "Color": P._rgb(fig.get_facecolor()),
             "Position": _get(fh, "Position"), "Visible": char("on"),
             "NumberTitle": char(str(fh.extra.get("NumberTitle", "on"))), "Tag": char(str(fh.extra.get("Tag", "")))}
    extra: dict = {}
    for attr, key in (("_suptitle", "suptitle"), ("_supxlabel", "supxlabel"), ("_supylabel", "supylabel")):
        t = getattr(fig, attr, None)
        if t is not None and t.get_text():
            extra[key] = getattr(t, "_mylab_raw", None) or t.get_text()
    cmap = interp.state.get("colormap")
    if cmap is not None:
        props["Colormap"] = P._cmap_query(interp)
    children = []
    for ax in fig.axes:
        if P._is_colorbar_axes(ax) or getattr(ax, "_mylab_internal", False):
            continue
        children.append(describe_axes(interp, ax))
    return _node("figure", props, children, extra)


# -- building -----------------------------------------------------------------------------------------

def _apply(h, props: dict, names: list) -> None:
    for n in names:
        if n in props and props[n] is not None:
            v = props[n]
            if isinstance(v, np.ndarray) and v.size == 0 and not is_char(v):
                continue
            try:
                P._set_prop(h, n, v)
            except (MatlabError, ValueError, TypeError, AttributeError):
                continue


def _row(v) -> np.ndarray:
    return np.asarray(fnum(v), dtype=float).reshape(1, -1)


def _has(props: dict, name: str) -> bool:
    v = props.get(name)
    return v is not None and not (isinstance(v, np.ndarray) and v.size == 0)


def _txt(v, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, MString) or is_char(v):
        return to_str(v)
    if isinstance(v, (Cell, StringArray)):
        return "\n".join(P._text_list(v))
    return default


def build_object(interp, ax, node: dict):
    """Draw one described object in ``ax``; returns its handle (None if the kind is not known)."""
    kind = _TYPE_ALIASES.get(str(node["type"]).lower())
    if kind is None:
        kind = _TYPE_ALIASES.get(str(node["type"]).lower().rsplit(".", 1)[-1])
    p = node["properties"]
    extra = node.get("extra", {})
    axh = P._wrap(interp, P._axes_kind(ax), ax)
    call = lambda name, *args: BUILTINS[name](interp, [axh, *args], 1)  # noqa: E731
    h = None
    if kind == "group":
        hs = [build_object(interp, ax, c) for c in node["children"]]
        hs = [x for x in hs if x is not None]
        return hs[0] if hs else None
    if kind == "constantline":
        which = _txt(p.get("InterceptAxis"), "x")
        more = [char("AffectAutoLimits"), char("on")] if _txt(p.get("AffectAutoLimits"), "off") == "on" else []
        h = call("xline" if which == "x" else "yline", p["Value"], *more)
    elif kind in ("line", "stair"):
        if not (_has(p, "XData") and _has(p, "YData")):
            return None
        x, y = _row(p["XData"]), _row(p["YData"])
        if kind == "stair":
            h = call("stairs", x, y)
        elif _has(p, "ZData") and hasattr(ax, "get_zlim"):
            h = call("plot3", x, y, _row(p["ZData"]))
        else:
            h = call("line", x, y)
    elif kind == "scatter":
        x, y = _row(p["XData"]), _row(p["YData"])
        size = _row(p["SizeData"]) if _has(p, "SizeData") else np.array([[36.0]])
        args = [x, y] + ([_row(p["ZData"])] if _has(p, "ZData") and hasattr(ax, "get_zlim") else [])
        args.append(size)
        if _has(p, "CData") and _row(p["CData"]).size == x.size and x.size != 3:
            args.append(_row(p["CData"]))
        h = call("scatter3" if len(args) > 3 and _has(p, "ZData") and hasattr(ax, "get_zlim") else "scatter",
                 *args, char("filled"))
        if "facecolors" in extra and h is not None:
            fc, ec = _f(extra["facecolors"]), _f(extra["edgecolors"])
            if h.obj.get_array() is None:
                h.obj.set_facecolor(fc.reshape(-1, 4) if fc.size else "none")
            h.obj.set_edgecolor(ec.reshape(-1, 4) if ec.size else "face")
        elif h is not None:
            _apply(h, p, ["MarkerFaceColor", "MarkerEdgeColor"])
        if h is not None and extra.get("marker") not in (None, "o"):
            P._set_prop(h, "Marker", char(str(extra["marker"])))
    elif kind == "bar":
        horizontal = _txt(p.get("Horizontal"), "off") == "on"
        if "rects" in extra and _f(extra["rects"]).size:
            rects = _f(extra["rects"]).reshape(-1, 4)
            colors = _f(extra["colors"]).reshape(-1, 4)
            if horizontal:
                cont = ax.barh(rects[:, 1], rects[:, 2], height=rects[:, 3], left=rects[:, 0], align="edge",
                               color=colors, edgecolor="black", linewidth=0.5)
            else:
                cont = ax.bar(rects[:, 0], rects[:, 3], width=rects[:, 2], bottom=rects[:, 1], align="edge",
                              color=colors, edgecolor="black", linewidth=0.5)
            h = P._wrap(interp, "bar", cont)
            h.extra.update({"XData": _row(p.get("XData", [])).flatten(), "YData": _row(p.get("YData", [])).flatten(),
                            "BarWidth": float(_row(p.get("BarWidth", 0.8)).flat[0]), "axes": ax, "figure": ax.figure,
                            "Horizontal": horizontal, "BarLayout": _txt(p.get("BarLayout"), "grouped")})
        else:
            args = [_row(p["XData"]), _row(p["YData"])] if _has(p, "XData") else [_row(p["YData"])]
            if _has(p, "BarWidth"):
                args.append(_row(p["BarWidth"]))
            h = call("barh" if horizontal else "bar", *args)
            _apply(h, p, ["FaceColor"])
    elif kind == "histogram":
        args = [_row(p["Data"]), _row(p["BinEdges"])]
        for name in ("Normalization", "Orientation", "DisplayStyle"):
            if _has(p, name):
                args += [char(name), p[name] if not isinstance(p[name], str) else char(p[name])]
        if "counts" in extra:
            args = [char("BinEdges"), _row(p["BinEdges"]), char("BinCounts"), _row(extra["counts"])] + args[2:]
        h = call("histogram", *args)
    elif kind == "surface":
        X, Y, Z = (_f(fnum(p[k])) for k in ("XData", "YData", "ZData"))
        style = str(extra.get("style", "surf"))
        if style == "pcolor":
            h = call("pcolor", X, Y, _f(fnum(p["CData"])))
        else:
            h = call("mesh" if style == "mesh" else "surf", X, Y, Z)
            edge = extra.get("edge")
            if edge is not None and style == "surf" and h is not None:
                e = _f(edge).flatten()
                h.obj.set_edgecolor(tuple(e) if e.size == 4 and e[3] > 0 else "none")
    elif kind == "patch":
        if _has(p, "Faces") and _has(p, "Vertices"):
            F, V = _f(fnum(p["Faces"])), _f(fnum(p["Vertices"]))
            F = F.reshape(1, -1) if F.ndim == 1 else F
            args = [char("Faces"), F, char("Vertices"), V]
        elif _has(p, "XData") and _has(p, "YData"):
            args = [char("XData"), _f(fnum(p["XData"])), char("YData"), _f(fnum(p["YData"]))]
            if _has(p, "ZData"):
                args += [char("ZData"), _f(fnum(p["ZData"]))]
        else:
            return None
        if _has(p, "FaceVertexCData"):
            args += [char("FaceVertexCData"), _f(fnum(p["FaceVertexCData"]))]
        h = call("patch", *args)
        if h is not None and "facecolors" in extra:
            fc, ec = _f(extra["facecolors"]), _f(extra["edgecolors"])
            if not _has(p, "FaceVertexCData"):
                h.obj.set_facecolor(fc.reshape(-1, 4) if fc.size else "none")
            h.obj.set_edgecolor(ec.reshape(-1, 4) if ec.size else "none")
        elif h is not None:
            _apply(h, p, ["FaceColor", "EdgeColor"])
    elif kind == "area":
        x, y = _row(p["XData"]).flatten(), _row(p["YData"]).flatten()
        if "outline" in extra and _f(extra["outline"]).size:
            from matplotlib.collections import PolyCollection
            coll = PolyCollection([_f(extra["outline"]).reshape(-1, 2)], edgecolor="black", linewidth=0.5)
            ax.add_collection(coll)
            ax.update_datalim(_f(extra["outline"]).reshape(-1, 2))
            ax.autoscale_view()
            h = P._wrap(interp, "area", coll)
            h.extra.update({"XData": x.reshape(1, -1), "YData": y.reshape(1, -1), "axes": ax, "figure": ax.figure,
                            "BaseValue": _row(p.get("BaseValue", 0.0))})
        else:
            h = call("area", x.reshape(1, -1), y.reshape(1, -1))
        _apply(h, p, ["FaceColor", "EdgeColor"])
    elif kind == "image":
        C = np.asarray(fnum(p["CData"]), dtype=float)
        args = [char("CData"), C, char("CDataMapping"), char(_txt(p.get("CDataMapping"), "scaled"))]
        for name in ("XData", "YData"):
            if _has(p, name):
                args += [char(name), _row(p[name])]
        hold = P._state(interp).hold.get(id(ax), False)
        h = call("image", *args)
        P._state(interp).hold[id(ax)] = hold
        if h is not None and "cmap" in extra and _f(extra["cmap"]).size and C.ndim == 2:
            from matplotlib.colors import ListedColormap
            h.obj.set_cmap(ListedColormap(np.clip(_f(extra["cmap"]).reshape(-1, 3), 0, 1)))
        if h is not None and "clim" in extra and _f(extra["clim"]).size == 2:
            lo, hi = _f(extra["clim"]).flatten()
            h.obj.set_clim(lo, hi)
    elif kind == "text":
        pos = _row(p.get("Position", [0, 0, 0])).flatten()
        if pos.size < 2:
            return None
        args = [P._num_val(pos[0]), P._num_val(pos[1])]
        if hasattr(ax, "get_zlim") and pos.size > 2:
            args.append(P._num_val(pos[2]))
        args.append(char(_txt(p.get("String"))))
        if _has(p, "Units") and _txt(p["Units"]) != "data":
            args += [char("Units"), char(_txt(p["Units"]))]
        if _has(p, "Interpreter"):
            args += [char("Interpreter"), char(_txt(p["Interpreter"]))]
        h = call("text", *args)
    elif kind == "contour":
        fill = _txt(p.get("Fill"), "off") == "on"
        args = [_f(fnum(p["XData"])), _f(fnum(p["YData"])), _f(fnum(p["ZData"]))]
        if _has(p, "LevelList"):
            lv = _row(p["LevelList"])
            args.append(np.repeat(lv, 2, axis=1) if lv.size == 1 else lv)
        if _has(p, "LineColor") and not is_text(p["LineColor"]):
            args += [char("LineColor"), p["LineColor"]]
        if _txt(p.get("ShowText"), "off") == "on":
            args += [char("ShowText"), char("on")]
        res = BUILTINS["contourf" if fill else "contour"](interp, [axh, *args], 2)
        h = res[1]
    elif kind == "errorbar":
        x, y = _row(p["XData"]), _row(p["YData"])

        def part(name):
            return _row(p[name]) if _has(p, name) and _row(p[name]).size == x.size else np.zeros((0, 0))
        yn, yp, xn, xp = part("YNegativeDelta"), part("YPositiveDelta"), part("XNegativeDelta"), part("XPositiveDelta")
        if not yn.size and not yp.size and not xn.size and not xp.size:
            yn = yp = np.zeros_like(x)
        h = call("errorbar", x, y, yn, yp, xn, xp)
    elif kind == "stem":
        args = [_row(p["XData"]), _row(p["YData"])]
        if extra.get("filled"):
            args.append(char("filled"))
        h = call("stem", *args)
    elif kind == "quiver":
        h = call("quiver", *(_f(fnum(p[k])) for k in ("XData", "YData", "UData", "VData")))
    elif kind == "rectangle":
        args = [char("Position"), _row(p["Position"])]
        if _has(p, "Curvature"):
            args += [char("Curvature"), _row(p["Curvature"])]
        h = BUILTINS["rectangle"](interp, [axh, *args], 1)
    else:
        return None
    if h is None:
        return None
    for one in P._handles_of(h):
        k = one.kind
        _apply(one, p, _STYLE.get(k, []) + _COMMON)
    return h


def _ruler_from(info: dict):
    cats = info.get("categories")
    names = P._text_list(cats) if isinstance(cats, (Cell, StringArray)) else (
        [str(c) for c in np.atleast_1d(cats)] if cats is not None and np.size(cats) else [])
    r = P.Ruler(_txt(info.get("kind"), "datetime"), zone=_txt(info.get("zone")),
                fmt=_txt(info.get("fmt")) or None, unit=float(np.asarray(fnum(info.get("unit", 1.0))).flat[0]),
                categories=names, dfmt=_txt(info.get("dfmt"), "s"))
    r.classic = bool(np.asarray(fnum(info.get("classic", 0))).flat[0]) if info.get("classic") is not None else False
    return r


def build_axes(interp, fig, node: dict, position=None):
    """Create the described axes (with everything in it) in the matplotlib figure ``fig``."""
    st = P._state(interp)
    p = node["properties"]
    extra = node.get("extra", {})
    kind = str(extra.get("kind", "") or "")
    kids = node["children"]
    special = [int(i) - 1 for i in np.atleast_1d(np.asarray(node.get("special", []), dtype=float)).flatten()
               if 0 < int(i) <= len(kids)]
    plain = [c for i, c in enumerate(kids) if i not in special]
    if not kind:  # a MATLAB file: decide from what the axes holds
        kind = "polar" if str(node["type"]).lower().endswith("polaraxes") else "2d"
        view = _row(p["View"]).flatten() if _has(p, "View") else np.array([0.0, 90.0])
        if view.size == 2 and not (abs(view[0]) < 1e-9 and abs(view[1] - 90) < 1e-9):
            kind = "3d"
    rect = [float(v) for v in (_row(position if position is not None else p.get("Position", [0.13, 0.11, 0.775, 0.815]))
                               ).flatten()[:4]]
    if _txt(p.get("Units"), "normalized") == "pixels" and position is None:
        w, hh = fig.get_size_inches() * P._fig_dpi(fig)
        rect = [rect[0] / w, rect[1] / hh, rect[2] / w, rect[3] / hh]
    st.counter = getattr(st, "counter", 0) + 1
    kw = {"projection": "polar"} if kind == "polar" else {"projection": "3d"} if kind == "3d" else {}
    ax = fig.add_axes(rect, label=f"mylab-loaded-{st.counter}", **kw)
    n = P._fig_number(interp, fig)
    if n is not None:
        st.axes[n] = ax
    P._tick_aligned(ax)
    h = P._wrap(interp, P._axes_kind(ax), ax)
    for which in "xyz":
        info = extra.get(f"ruler_{which}")
        if isinstance(info, dict) and hasattr(ax, f"{which}axis"):
            P.install_ruler(ax, which, _ruler_from(info))
    st.hold[id(ax)] = True
    built = []
    try:
        for child in plain:
            built.append(build_object(interp, ax, child))
    finally:
        st.hold[id(ax)] = _txt(p.get("NextPlot"), "replace") == "add"
    for which in "xyz":
        info = extra.get(f"ruler_{which}")
        if isinstance(info, dict) and hasattr(ax, f"{which}axis") and P._ruler_of(ax, which) is None:
            P.install_ruler(ax, which, _ruler_from(info))
        fmt = extra.get(f"tickfmt_{which}")
        if fmt is not None and hasattr(ax, f"{which}axis") and _txt(fmt):
            from .plot_rulers import tick_format_set
            tick_format_set(ax, which, _txt(fmt))
    if kind != "polar":
        _apply(h, p, ["XScale", "YScale", "XDir", "YDir", "ZDir"])
    if kind == "2d" and "aspect" in extra:
        a = extra["aspect"]
        a = "auto" if (isinstance(a, str) or is_char(a) or isinstance(a, MString)) and _txt(a, "auto") == "auto" \
            else float(np.asarray(fnum(a)).flat[0])
        ax.set_aspect(a, adjustable=_txt(extra.get("adjustable"), "box") or "box")
        if extra.get("box_aspect") is not None:
            ax.set_box_aspect(float(np.asarray(fnum(extra["box_aspect"])).flat[0]))
    for which in "xyz":
        name = f"{which.upper()}Lim"
        if _has(p, name) and hasattr(ax, f"set_{which}lim"):
            lim = _row(p[name]).flatten()
            if lim.size == 2 and lim[0] < lim[1] and np.all(np.isfinite(lim)):
                inverted = getattr(ax, f"{which}axis_inverted", lambda: False)()
                getattr(ax, f"set_{which}lim")((lim[1], lim[0]) if inverted else (lim[0], lim[1]))
        tick = f"{which.upper()}Tick"
        if _has(p, tick) and hasattr(ax, f"set_{which}ticks"):
            P._set_tick_values(ax, which, _row(p[tick]).flatten())
        lab = f"{which.upper()}TickLabel"
        if p.get(lab) is not None and hasattr(ax, f"set_{which}ticklabels") and which != "z":
            try:
                P._set_ticklabels(ax, which, p[lab] if not isinstance(p[lab], str) else char(p[lab]))
            except MatlabError:
                pass
    _apply(h, p, ["XGrid", "YGrid", "Box", "Color", "FontSize", "View", "XAxisLocation", "YAxisLocation", "CLim"])
    names = ["Title", "XLabel", "YLabel", "ZLabel"]
    for name, idx in zip(names, special):
        tnode = kids[idx]
        s = _txt(tnode["properties"].get("String"))
        target = {"Title": ax.title, "XLabel": ax.xaxis.label, "YLabel": ax.yaxis.label,
                  "ZLabel": getattr(getattr(ax, "zaxis", None), "label", None)}[name]
        if target is None:
            continue
        interp_name = _txt(tnode["properties"].get("Interpreter")) or None
        if name == "Title" and s:
            ax.set_title("", fontweight="bold")
        P.set_text(interp, target, s, interp_name)
        th = P._wrap(interp, "text", target)
        _apply(th, tnode["properties"], ["FontSize", "FontWeight", "Color"])
    if extra.get("subtitle") is not None and _txt(extra["subtitle"]):
        t = P._make_subtitle(ax, _txt(extra["subtitle"]))
        P.set_text(interp, t, _txt(extra["subtitle"]))
        P._layout_subtitle(ax)
    leg = extra.get("legend")
    if isinstance(leg, dict):
        labels = leg.get("String")
        labels = P._text_list(labels) if isinstance(labels, (Cell, StringArray)) else (
            [_txt(labels)] if labels is not None and _txt(labels) else [])
        handles = None
        items = extra.get("legend_items")
        if items is not None and np.size(items):
            order = [b for b in built]
            handles = []
            for i in np.asarray(fnum(items), dtype=float).flatten():
                b = order[int(i) - 1] if 0 < int(i) <= len(order) else None
                if b is not None:
                    o = P._handles_of(b)[0].obj
                    handles.append(o.container if isinstance(o, P._Hist) else (o[0] if isinstance(o, list) else o))
        lg = P._make_legend(interp, ax, labels=labels or None, handles=handles or None,
                            location=_txt(leg.get("Location"), "northeast"),
                            orientation=_txt(leg.get("Orientation"), "vertical"),
                            fontsize=float(np.asarray(fnum(leg["FontSize"])).flat[0]) if leg.get("FontSize") is not None
                            else None,
                            box=_txt(leg.get("Box"), "on") == "on",
                            title=_txt(leg.get("Title")) or None)
        lg.set_visible(_txt(leg.get("Visible"), "on") == "on")
    cb = extra.get("colorbar")
    if isinstance(cb, dict):
        ch = BUILTINS["colorbar"](interp, [h, char(_txt(cb.get("Location"), "eastoutside"))], 1)
        if _txt(cb.get("Label")):
            P.set_text(interp, P._cb_label(ch).obj, _txt(cb.get("Label")))
        ax.set_position(rect)
        if cb.get("Position") is not None and np.size(cb["Position"]) == 4 and not ch.obj.ax.get_label() == "inset":
            cpos = [float(v) for v in np.asarray(fnum(cb["Position"])).flatten()]
            if _txt(cb.get("Location"), "eastoutside") not in ("east", "west", "north", "south"):
                ch.obj.ax.set_position(cpos)
    if _txt(p.get("Visible"), "on") == "off":
        ax.set_axis_off()
    return ax


def build_figure(interp, node: dict, visible: bool = True, reuse: bool = False):
    """Create a new figure from its description; returns the figure handle."""
    st = P._state(interp)
    p = node["properties"]
    extra = node.get("extra", {})
    fh = BUILTINS["figure"](interp, [], 1)
    fig = fh.obj
    _apply(fh, {k: v for k, v in p.items() if k in ("Name", "Color", "NumberTitle", "Tag")},
           ["Name", "Color", "NumberTitle", "Tag"])
    if _has(p, "Position"):
        pos = _row(p["Position"]).flatten()
        if pos.size == 4 and pos[2] > 0 and pos[3] > 0:
            dpi = P._fig_dpi(fig)
            fig.set_size_inches(pos[2] / dpi, pos[3] / dpi)
            fig._mylab_origin = (float(pos[0]), float(pos[1]))
    if _has(p, "Colormap"):
        cm = np.asarray(fnum(p["Colormap"]), dtype=float)
        if cm.ndim == 2 and cm.shape[1] == 3:
            interp.state["colormap"] = P._to_cmap(np.clip(cm, 0, 1))
    fh.extra["Visible"] = bool(visible)
    pairs: dict = {}
    for child in node["children"]:
        t = _TYPE_ALIASES.get(str(child["type"]).lower()) or str(child["type"]).lower()
        if t in ("axes", "polaraxes"):
            ax = build_axes(interp, fig, child)
            side = child.get("extra", {}).get("yyside")
            if side is not None:
                pairs.setdefault(tuple(np.round(ax.get_position().bounds, 6)), {})[_txt(side)] = ax
    for group in pairs.values():  # two axes saved from one yyaxis chart belong together again
        left, right = group.get("left"), group.get("right")
        if left is not None and right is not None:
            left._mylab_yyright, right._mylab_yybase = right, left
            left._mylab_yyside, right._mylab_yyside = "left", "right"
            right.patch.set_visible(False)
            right.yaxis.tick_right()
            right.yaxis.set_label_position("right")
            right.xaxis.set_visible(False)
            P.yy_right_frame(right)
            st.axes[P._fig_number(interp, fig)] = left
    for key, fn in (("suptitle", fig.suptitle), ("supxlabel", fig.supxlabel), ("supylabel", fig.supylabel)):
        if extra.get(key) is not None and _txt(extra[key]):
            t = fn(_txt(extra[key]))
            P.set_text(interp, t, _txt(extra[key]))
    P._redraw(interp)
    return fh


# -- MAT-file form ---------------------------------------------------------------------------------------

def _to_file(v: Any) -> Any:
    """A mylab value as something scipy.io.savemat stores the way MATLAB expects it."""
    if isinstance(v, MString):
        return to_str(v)
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        if v.ndim == 2 and v.shape[0] > 1:
            return np.array(["".join(r) for r in v], dtype=object).reshape(-1, 1)
        return "".join(v.flatten())
    if isinstance(v, (Cell, StringArray)):
        data = v.data
        out = np.empty(data.shape, dtype=object)
        for idx in np.ndindex(data.shape):
            out[idx] = _to_file(data[idx]) if data[idx] is not None else ""
        return out
    if isinstance(v, dict):
        return {k: _to_file(x) for k, x in v.items()} or {"empty": 1.0}
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, int, float)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.astype(float) if v.dtype.kind in "biu" else v
    if isinstance(v, Struct):
        return {f: _to_file(v.get_field(0, f)) for f in v.fields} or {"empty": 1.0}
    return np.zeros((0, 0))


_FIELDS = ["type", "handle", "properties", "children", "special"]


def _pack(nodes: list, counter: list, extras: list) -> np.ndarray:
    arr = np.zeros((len(nodes), 1), dtype=[(f, object) for f in _FIELDS])
    for i, node in enumerate(nodes):
        counter[0] += 1
        ident = float(counter[0])
        if node.get("extra"):
            extras.append((ident, node["extra"]))
        arr[i, 0]["type"] = str(node["type"])
        arr[i, 0]["handle"] = ident
        arr[i, 0]["properties"] = {k: _to_file(v) for k, v in node["properties"].items() if v is not None} \
            or {"Visible": "on"}
        arr[i, 0]["children"] = _pack(node["children"], counter, extras) if node["children"] else np.zeros((0, 1))
        arr[i, 0]["special"] = np.asarray(node.get("special", []), dtype=float).reshape(-1, 1)
    return arr


def write_fig_file(fname: str, nodes: list) -> None:
    from scipy.io import savemat
    extras: list = []
    tree = _pack(nodes, [0], extras)
    info = {"version": 1.0, "handles": np.array([e[0] for e in extras], dtype=float).reshape(1, -1),
            "extras": np.array([_to_file(e[1]) for e in extras] + [None], dtype=object)[:-1].reshape(1, -1)}
    try:
        with open(fname, "wb") as f:
            savemat(f, {"hgS_070000": tree, "mylab_fig": info}, format="5", oned_as="row")
    except OSError as e:
        raise MatlabError(f"Cannot write '{fname}': {e.strerror or e}") from e


def _from_file(v: Any) -> Any:
    """A value read by scipy.io.loadmat as a mylab value (structs become dictionaries)."""
    if hasattr(v, "_fieldnames"):
        return {f: _from_file(getattr(v, f)) for f in v._fieldnames}
    if isinstance(v, str):
        return char(v)
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            items = [_from_file(x) for x in v.flatten(order="F")]
            return Cell.from_list(items)
        if v.dtype.kind in "US":
            if v.size == 0:
                return char("")
            if v.ndim == 0:
                return char(str(v))
            return Cell.from_list([char(str(x)) for x in v.flatten()])
        a = v.astype(float) if v.dtype.kind in "biuf" else v
        if a.ndim == 0:
            return a.reshape(1, 1)
        if a.ndim == 1:
            return a.reshape(1, -1)
        return a
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.array([[float(v)]])
    return v


def _unpack(s, extras: dict) -> dict:
    kids = getattr(s, "children", np.zeros(0))
    if hasattr(kids, "_fieldnames"):
        kids = [kids]
    elif isinstance(kids, np.ndarray):
        kids = [k for k in kids.flatten() if hasattr(k, "_fieldnames")]
    else:
        kids = []
    props = getattr(s, "properties", None)
    ident = getattr(s, "handle", None)
    try:
        key = float(np.asarray(ident, dtype=float).flat[0])
    except (TypeError, ValueError, IndexError):
        key = None
    special = getattr(s, "special", [])
    return {"type": str(getattr(s, "type", "")),
            "properties": _from_file(props) if hasattr(props, "_fieldnames") else {},
            "children": [_unpack(k, extras) for k in kids],
            "special": [float(x) for x in np.atleast_1d(np.asarray(special, dtype=float)).flatten()]
            if np.size(special) else [],
            "extra": extras.get(key, {})}


def read_fig_file(fname: str) -> list:
    """The object descriptions stored in a figure file (one per saved figure or object)."""
    from scipy.io import loadmat
    try:
        with open(fname, "rb") as f:
            head = f.read(8)
    except OSError as e:
        raise MatlabError(f"Cannot open '{fname}': {e.strerror or e}") from e
    if head.startswith(b"\x89HDF"):
        raise MatlabError(f"'{fname}' is a version 7.3 (HDF5) figure file, which cannot be read. In MATLAB, save "
                          "the figure again with savefig (not the 'compact' or '-v7.3' form).")
    try:
        raw = loadmat(fname, struct_as_record=False, squeeze_me=True, appendmat=False)
    except (ValueError, OSError, TypeError, NotImplementedError) as e:
        raise MatlabError(f"'{fname}' is not a figure file that can be read: {e}") from e
    key = next((k for k in sorted(raw) if k.startswith("hgS_")), None)
    if key is None:
        if any(k.startswith("hgM_") for k in raw):
            raise MatlabError(f"'{fname}' was saved in MATLAB's compact figure format, which holds only objects "
                              "that MATLAB can read. Save it again in MATLAB with savefig(fig, file) without "
                              "'compact'.")
        raise MatlabError(f"'{fname}' is not a figure file: it has no figure description (hgS_...).")
    extras: dict = {}
    info = raw.get("mylab_fig")
    if info is not None and hasattr(info, "_fieldnames"):
        ids = np.atleast_1d(np.asarray(getattr(info, "handles", []), dtype=float)).flatten()
        vals = getattr(info, "extras", [])
        vals = [vals] if hasattr(vals, "_fieldnames") else list(np.atleast_1d(vals).flatten())
        for ident, v in zip(ids, vals):
            if hasattr(v, "_fieldnames"):
                d = _from_file(v)
                d.pop("empty", None)
                extras[float(ident)] = _plain_extra(d)
    top = raw[key]
    tops = [top] if hasattr(top, "_fieldnames") else [t for t in np.atleast_1d(top).flatten()
                                                       if hasattr(t, "_fieldnames")]
    return [_unpack(t, extras) for t in tops]


def _plain_extra(d: dict) -> dict:
    """Extras come back as mylab values; unwrap the scalars and texts the builders test directly."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v.pop("empty", None)
            out[k] = v if k in ("legend", "colorbar") else _plain_extra(v) if not k.startswith("ruler_") else v
        elif is_char(v):
            out[k] = to_str(v)
        else:
            out[k] = v
    return out


# -- built-in functions ------------------------------------------------------------------------------------

def save_fig(interp, handles: list, fname: str) -> None:
    """Write the given figure (or axes) handles to a .fig file."""
    interp.state.pop("fig_skipped", None)
    nodes = []
    for h in handles:
        if not P._alive(h):
            raise MatlabError("Invalid or deleted object.")
        if h.kind == "figure":
            nodes.append(describe_figure(interp, h))
        elif h.kind in ("axes", "polaraxes"):
            nodes.append(describe_axes(interp, h.obj))
        else:
            node = describe_object(interp, h)
            if node is None:
                raise MatlabError(f"A {P._short(h.kind)} object cannot be saved in a figure file.")
            nodes.append(node)
    write_fig_file(fname, nodes)
    skipped = interp.state.pop("fig_skipped", None)
    if skipped:
        interp.warn("These objects could not be saved in the figure file and were left out: "
                    + ", ".join(sorted(skipped)) + ".")


def savefig_builtin(interp, args: list, nargout: int):
    handles = None
    fname = None
    for a in args:
        if isinstance(a, (P.GraphicsHandle, P.GraphicsArray)):
            handles = P._handles_of(a)
        elif is_text(a):
            s = text(a)
            if s.lower() in ("compact", "-v6", "-v7", "-v7.3", "all"):
                if s.lower() == "-v7.3":
                    raise MatlabError("savefig: the -v7.3 file format is not supported.")
                continue  # 'compact' and '-v6' ask for other layouts of the same content
            if fname is not None:
                raise MatlabError(f"savefig: unknown option '{s}'.")
            fname = s
        elif isinstance(a, np.ndarray) and a.size >= 1:
            handles = P._resolve(interp, a)
        else:
            raise MatlabError("savefig expects figure handles and a file name.")
    if handles is None:
        st = P._state(interp)
        if st.current is None or st.current not in st.figures:
            raise MatlabError("There is no figure to save.")
        handles = [P._wrap(interp, "figure", st.figures[st.current])]
    if fname is None:
        fname = "Untitled.fig"
    if not os.path.splitext(fname)[1]:
        fname += ".fig"
    save_fig(interp, handles, fname)
    return None


def _find_fig(interp, fname: str) -> str:
    cands = [fname] if os.path.splitext(fname)[1] else [fname + ".fig", fname]
    for c in cands:
        if os.path.isfile(c):
            return c
    for folder in getattr(interp, "path", []) or []:
        for c in cands:
            full = os.path.join(str(folder), c)
            if os.path.isfile(full):
                return full
    raise MatlabError(f"Cannot find the figure file '{fname}'.")


def openfig_builtin(interp, args: list, nargout: int):
    if not args or not is_text(args[0]):
        raise MatlabError("openfig needs a file name, e.g. openfig('plot.fig').")
    fname = _find_fig(interp, text(args[0]))
    visible, reuse = True, False
    overrides: list = []
    for a in args[1:]:
        if isinstance(a, Struct):  # hgload(file, struct of property values)
            overrides.extend((f, a.get_field(0, f)) for f in a.fields)
            continue
        if not is_text(a):
            raise MatlabError("openfig options are 'new', 'reuse', 'visible' and 'invisible'.")
        word = text(a).lower()
        if word == "invisible":
            visible = False
        elif word == "visible":
            visible = True
        elif word == "reuse":
            reuse = True
        elif word not in ("new", "all"):
            raise MatlabError(f"openfig: unknown option '{text(a)}'.")
    st = P._state(interp)
    key = os.path.abspath(fname)
    if reuse:
        for n, fig in st.figures.items():
            if getattr(fig, "_mylab_file", None) == key:
                BUILTINS["figure"](interp, [np.array([[float(n)]])], 0)
                return P._wrap(interp, "figure", fig) if nargout > 0 else None
    nodes = read_fig_file(fname)
    out = []
    unknown: set = set()
    for node in nodes:
        t = _TYPE_ALIASES.get(str(node["type"]).lower()) or str(node["type"]).lower()
        if t == "figure":
            _collect_unknown(node, unknown)
            fh = build_figure(interp, node, visible)
            fh.obj._mylab_file = key
            fh.extra["FileName"] = key
            out.append(fh)
        elif t in ("axes", "polaraxes"):
            _collect_unknown(node, unknown)
            fig = P._figure(interp)
            ax = build_axes(interp, fig, node)
            out.append(P._wrap(interp, P._axes_kind(ax), ax))
        else:
            ax = P._axes(interp, fresh=False)
            h = build_object(interp, ax, node)
            if h is None:
                unknown.add(str(node["type"]))
            else:
                out.extend(P._handles_of(h))
    for h in out:
        for name, value in overrides:
            P._set_prop(h, name, value)
    if unknown:
        interp.warn("The figure file holds objects that mylab cannot draw; they were left out: "
                    + ", ".join(sorted(unknown)) + ".")
    P._redraw(interp)
    if nargout <= 0:
        return None
    return P._handle_list(out)


_SILENT_TYPES = ("uimenu", "uitoolbar", "uipushtool", "uitoggletool", "uicontextmenu", "annotationpane",
                 "scribe.scribeaxes", "light", "uitogglesplittool", "uipushsplittool")


def _collect_unknown(node: dict, unknown: set) -> None:
    for child in node["children"]:
        t = str(child["type"]).lower()
        if _TYPE_ALIASES.get(t) is None and _TYPE_ALIASES.get(t.rsplit(".", 1)[-1]) is None \
                and t not in _SILENT_TYPES:
            unknown.add(str(child["type"]))
        _collect_unknown(child, unknown)


def copyobj_builtin(interp, args: list, nargout: int):
    if len(args) < 2:
        raise MatlabError("copyobj needs the objects to copy and the new parent.")
    sources = P._handles_of(args[0])
    if not sources:
        raise MatlabError("copyobj expects graphics handles.")
    parents = P._handles_of(args[1]) if isinstance(args[1], (P.GraphicsHandle, P.GraphicsArray)) \
        else P._resolve(interp, args[1])
    if len(parents) not in (1, len(sources)):
        raise MatlabError("copyobj: give one parent, or one parent for each object.")
    if len(parents) == 1 and len(sources) > 1:
        parents = parents * len(sources)
    if len(args) > 2 and not (is_text(args[2]) and text(args[2]).lower() == "legacy"):
        raise MatlabError("copyobj: the only option is 'legacy'.")
    st = P._state(interp)
    out = []
    current = (st.current, st.axes.get(st.current) if st.current is not None else None)
    for h, parent in zip(sources, parents):
        if not P._alive(h) or not P._alive(parent):
            raise MatlabError("Invalid or deleted object.")
        if h.kind == "figure":
            if parent.kind != "root":
                raise MatlabError("copyobj: a figure can only be copied to the root (groot).")
            out.append(build_figure(interp, describe_figure(interp, h)))
        elif h.kind in ("axes", "polaraxes"):
            if parent.kind != "figure":
                raise MatlabError("copyobj: an axes can only be copied into a figure.")
            h.obj.figure.canvas.draw()
            ax = build_axes(interp, parent.obj, describe_axes(interp, h.obj))
            out.append(P._wrap(interp, P._axes_kind(ax), ax))
        elif h.kind in ("legend", "colorbar"):
            if not any(s.kind in ("axes", "polaraxes") and s.obj is P._obj_axes(h) for s in sources):
                raise MatlabError(f"copyobj: copy a {h.kind} together with its axes, e.g. copyobj([ax lgd], fig).")
        else:
            if parent.kind not in ("axes", "polaraxes"):
                raise MatlabError(f"copyobj: a {P._short(h.kind)} object can only be copied into an axes.")
            node = describe_object(interp, h)
            if node is None:
                raise MatlabError(f"copyobj: copying {P._short(h.kind)} objects is not supported.")
            ax = parent.obj
            hold = st.hold.get(id(ax), False)
            st.hold[id(ax)] = True
            try:
                new = build_object(interp, ax, node)
            finally:
                st.hold[id(ax)] = hold
            for one in P._handles_of(new):
                for key in ("UserData",):
                    if key in h.extra:
                        one.extra[key] = h.extra[key]
                out.append(one)
    if current[0] is not None and current[0] in st.figures:  # copying does not change the current figure
        st.current = current[0]
        if current[1] is not None and current[1] in st.figures[current[0]].axes:
            st.axes[current[0]] = current[1]
    P._redraw(interp)
    return P._handle_list(out) if nargout > 0 else None
