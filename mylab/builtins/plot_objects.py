"""The graphics object model beyond the basics: more properties, defaults, callbacks, object utilities.

``mylab.builtins.plotting`` defines the handle objects and a first set of properties for each kind of
object. This module adds the rest of what everyday scripts read and set (units and positions, grid and
tick styling, colour order, limit modes, callbacks, legend and colorbar details ...), default property
values on the root (``set(groot, 'DefaultLineLineWidth', 2)``), and functions that work on handles
(``linkprop``, ``colororder``, ``newplot``, ``hgtransform``, ``uistack``, ``fontsize`` ...).

A property is one of three things here:

* *live*: read from and written to the matplotlib object (most of them);
* *derived*: computed when read (``TightInset``, ``XEndPoints``, ``Extent``);
* *state*: a value the object simply keeps, because it describes interaction or bookkeeping that has no
  picture to change in an off-screen renderer (``ButtonDownFcn``, ``HitTest``, ``BusyAction``, ``Tag``).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, StringArray, Struct, char, is_char
from . import plotting as P
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, text

_EMPTY = np.zeros((0, 0))


# -- property helpers --------------------------------------------------------------------------------

def _state(name: str, default: Any = _EMPTY, check=None, apply=None) -> tuple:
    """A property kept on the handle; ``apply(h, value)`` gives it its visible effect, if it has one."""
    def g(h):
        v = h.extra.get(name, default)
        if callable(v) and not isinstance(v, FunctionHandle):
            v = v(h)
        return char(v) if isinstance(v, str) else v

    def s(h, v):
        val = check(v) if check is not None else v
        if apply is not None:
            apply(h, val)
        h.extra[name] = val
    return g, s


def _choice(name: str, options: tuple, default: str, apply=None) -> tuple:
    """A property that takes one of a few words."""
    def check(v):
        if isinstance(v, np.ndarray) and v.dtype == bool and v.size == 1 and {"on", "off"} <= set(options):
            return "on" if bool(v.flat[0]) else "off"
        if isinstance(v, np.ndarray) and not is_char(v) and v.size == 1 and {"on", "off"} <= set(options):
            return "on" if float(np.real(v.flat[0])) else "off"
        if not is_text(v):
            raise MatlabError(f"{name} must be one of: {', '.join(repr(o) for o in options)}.")
        s = text(v).lower()
        for o in options:
            if o.lower() == s:
                return o
        raise MatlabError(f"'{text(v)}' is not a valid value for {name}. Use one of: "
                          f"{', '.join(repr(o) for o in options)}.")
    return _state(name, default, check, apply)


def _onoff_state(name: str, default: str = "on", apply=None) -> tuple:
    return _choice(name, ("on", "off"), default, apply)


def _callback(name: str, default: Any = "") -> tuple:
    def check(v):
        if isinstance(v, (FunctionHandle, Cell)) or is_text(v) or (isinstance(v, np.ndarray) and v.size == 0):
            return v
        if getattr(v, "is_handle", False) or callable(getattr(v, "call_method", None)):
            return v
        raise MatlabError(f"{name} must be a function handle, a cell array {{@fn, args}} or text.")
    return _state(name, default, check)


def _number(name: str, default: float, lo: float = -math.inf, hi: float = math.inf, apply=None) -> tuple:
    def check(v):
        x = np.real(fnum(v))
        if x.size != 1 or not (lo <= float(x.flat[0]) <= hi):
            rng = f" between {lo:g} and {hi:g}" if math.isfinite(lo) and math.isfinite(hi) else ""
            raise MatlabError(f"{name} must be a number{rng}.")
        return P._num_val(float(x.flat[0]))
    return _state(name, P._num_val(default), check, apply)


def _vector(name: str, default, n: int | None = None, apply=None) -> tuple:
    def check(v):
        x = np.real(fnum(v)).reshape(1, -1)
        if n is not None and x.size != n:
            raise MatlabError(f"{name} must be a vector with {n} elements.")
        return x
    return _state(name, np.asarray(default, dtype=float).reshape(1, -1), check, apply)


def _on_axes(h, fn, *args):
    """Run a plotting helper with ``h``'s axes as the current axes."""
    st = P._state(h.interp)
    prev = st.override_ax
    st.override_ax = h.obj
    try:
        return fn(*args)
    finally:
        st.override_ax = prev


def fire(interp, h, name: str, event_name: str = "") -> bool:
    """Run the callback stored in property ``name`` of handle ``h``; True if there was one."""
    cb = h.extra.get(name)
    if cb is None or (isinstance(cb, np.ndarray) and cb.size == 0) or (isinstance(cb, str) and not cb):
        return False
    from .. import ui as U
    event = Struct.scalar({"Source": h, "EventName": char(event_name or name)})
    U.invoke(interp, h, cb, event)
    return True


def _delete_hook(interp, h) -> None:
    fire(interp, h, "DeleteFcn", "ObjectBeingDestroyed")
    for link in h.extra.get("_links", ()):
        if h in link.handles:
            link.handles.remove(h)


P._DELETE_HOOK = _delete_hook


# -- properties every object has ------------------------------------------------------------------------

def _clip_apply(h, v):
    for a in P._artists(h.obj) if h.kind not in ("axes", "polaraxes", "figure", "root") else []:
        if hasattr(a, "set_clip_on"):
            a.set_clip_on(v == "on")


def _handle_visibility_apply(h, v):
    # Objects hidden from the handle hierarchy are also left out of legends, as in MATLAB.
    if h.kind in ("figure", "axes", "polaraxes", "root", "legend", "colorbar", "text"):
        return
    for a in P._artists(h.obj):
        if not hasattr(a, "get_label"):
            continue
        lab = a.get_label()
        if v == "off" and not str(lab).startswith("_"):
            a._mylab_label = lab
            a.set_label("_hidden")
        elif v != "off" and str(lab) == "_hidden":
            a.set_label(getattr(a, "_mylab_label", ""))
    ax = P._obj_axes(h)
    if ax is not None and ax.get_legend() is not None:
        P._refresh_legend(h.interp, ax)


P._COMMON_PROPS.update({
    "HandleVisibility": _choice("HandleVisibility", ("on", "off", "callback"), "on", _handle_visibility_apply),
    "ButtonDownFcn": _callback("ButtonDownFcn"),
    "CreateFcn": _callback("CreateFcn"),
    "DeleteFcn": _callback("DeleteFcn"),
    "BusyAction": _choice("BusyAction", ("queue", "cancel"), "queue"),
    "Interruptible": _onoff_state("Interruptible"),
    "HitTest": _onoff_state("HitTest"),
    "PickableParts": _choice("PickableParts", ("visible", "all", "none"), "visible"),
    "Selected": _onoff_state("Selected", "off"),
    "SelectionHighlight": _onoff_state("SelectionHighlight"),
    "ContextMenu": _state("ContextMenu"),
    "UIContextMenu": _state("ContextMenu"),
    "Clipping": _onoff_state("Clipping", "on", _clip_apply),
    "BeingDeleted": (lambda h: P._oo(bool(h.extra.get("_deleting"))), None),
})


# -- legend entries: h.Annotation.LegendInformation.IconDisplayStyle ------------------------------------------

class _Sub:
    """A small helper object hanging off a graphics handle (annotation, legend information...)."""

    def __init__(self, owner, role: str):
        self.owner = owner
        self.role = role

    @property
    def figure(self):
        return P._obj_figure(self.owner)


def _sub(h, role: str, kind: str):
    subs = h.extra.setdefault("_subs", {})
    obj = subs.get(role)
    if obj is None:
        obj = subs[role] = _Sub(h, role)
    return P._wrap(h.interp, kind, obj)


def _icon_style_set(h, v):
    style = text(v).lower()
    if style not in ("on", "off", "children"):
        raise MatlabError("IconDisplayStyle must be 'on', 'off' or 'children'.")
    owner = h.obj.owner
    owner.extra["IconDisplayStyle"] = style
    _handle_visibility_apply(owner, "off" if style == "off" else "on")


P._CLASS.update({"annotationinfo": "matlab.graphics.eventdata.Annotation",
                 "legendentry": "matlab.graphics.eventdata.LegendEntry"})
P._PROPS["annotationinfo"] = {"LegendInformation": (lambda h: _sub(h.obj.owner, "legendinfo", "legendentry"), None)}
P._PROPS["legendentry"] = {"IconDisplayStyle": (lambda h: char(h.obj.owner.extra.get("IconDisplayStyle", "on")),
                                                _icon_style_set)}
_PLOT_KINDS = ("line", "stair", "scatter", "bar", "histogram", "surface", "contour", "quiver", "stem", "errorbar",
               "area", "image", "patch", "pie", "animatedline", "rectangle")
for _k in _PLOT_KINDS:
    P._PROPS.setdefault(_k, {})["Annotation"] = (lambda h: _sub(h, "annotation", "annotationinfo"), None)


# -- lines ---------------------------------------------------------------------------------------------------

def _marker_indices_get(h):
    me = h.obj.get_markevery()
    n = len(np.atleast_1d(h.obj.get_xdata()))
    if me is None:
        return P._row(np.arange(1, n + 1))
    return P._row(np.asarray(me, dtype=float).flatten() + 1)


def _marker_indices_set(h, v):
    idx = np.real(fnum(v)).flatten()
    n = len(np.atleast_1d(h.obj.get_xdata()))
    if idx.size and (idx.min() < 1 or idx.max() > n or np.any(idx != np.round(idx))):
        raise MatlabError("MarkerIndices must be whole numbers between 1 and the number of data points.")
    h.obj.set_markevery([int(i) - 1 for i in idx])


def _series_index_set(h, v):
    from matplotlib.colors import to_rgb
    k = int(P._scalar(v))
    if k < 1:
        raise MatlabError("SeriesIndex must be a positive whole number.")
    ax = P._obj_axes(h)
    colors = color_order(ax) if ax is not None else default_colors()
    c = tuple(colors[(k - 1) % len(colors)])
    for a in P._artists(h.obj):
        if hasattr(a, "set_color") and not hasattr(a, "set_facecolor"):
            a.set_color(c)
        elif hasattr(a, "set_facecolor"):
            a.set_facecolor(to_rgb(c))
    h.extra["SeriesIndex"] = P._num_val(k)


def _series_index_get(h):
    if "SeriesIndex" in h.extra:
        return h.extra["SeriesIndex"]
    ax = P._obj_axes(h)
    if ax is not None and h.obj in list(ax.lines):
        return P._num_val(list(ax.lines).index(h.obj) + 1)
    return P._num_val(1)


_JOIN = {"round": "round", "miter": "miter", "chamfer": "bevel"}

_LINE_EXTRA = {
    "LineJoin": _choice("LineJoin", ("round", "miter", "chamfer"), "round",
                        lambda h, v: (h.obj.set_solid_joinstyle(_JOIN[v]), h.obj.set_dash_joinstyle(_JOIN[v]))),
    "AlignVertexCenters": _onoff_state("AlignVertexCenters", "off"),
    "MarkerIndices": (_marker_indices_get, _marker_indices_set),
    "XDataSource": _state("XDataSource", ""),
    "YDataSource": _state("YDataSource", ""),
    "ZDataSource": _state("ZDataSource", ""),
    "ColorMode": _choice("ColorMode", ("auto", "manual"), "auto"),
    "LineStyleMode": _choice("LineStyleMode", ("auto", "manual"), "auto"),
    "MarkerMode": _choice("MarkerMode", ("auto", "manual"), "auto"),
    "SeriesIndex": (_series_index_get, _series_index_set),
}
P._LINE_PROPS.update(_LINE_EXTRA)
P._PROPS["animatedline"].update(_LINE_EXTRA)


# -- scatter ---------------------------------------------------------------------------------------------------

def _scatter_marker_set(h, v):
    from matplotlib.markers import MarkerStyle
    m = P._mpl_marker(text(v))
    try:
        ms = MarkerStyle(m)
    except ValueError:
        raise MatlabError(f"'{text(v)}' is not a marker.") from None
    h.obj.set_paths([ms.get_path().transformed(ms.get_transform())])
    h.extra["Marker"] = m


def _scatter_z(h):
    off = getattr(h.obj, "_offsets3d", None)
    return P._row(off[2]) if off is not None else np.zeros((1, 0))


def _scatter_edge_alpha_set(h, v):
    a = P._scalar(v)
    ec = np.array(h.obj.get_edgecolor(), dtype=float)
    if ec.size:
        ec[:, 3] = a
        h.obj.set_edgecolor(ec)
    h.extra["MarkerEdgeAlpha"] = P._num_val(a)


P._SCATTER_PROPS.update({
    "Marker": (lambda h: char(h.extra.get("Marker", "o")), _scatter_marker_set),
    "ZData": (_scatter_z, None),
    "MarkerEdgeAlpha": (lambda h: h.extra.get("MarkerEdgeAlpha", P._num_val(1)), _scatter_edge_alpha_set),
    "XDataSource": _state("XDataSource", ""),
    "YDataSource": _state("YDataSource", ""),
    "SeriesIndex": (_series_index_get, _series_index_set),
})


# -- bars -------------------------------------------------------------------------------------------------------

def _bar_rects(h) -> list:
    return [p for p in P._patches(h) if hasattr(p, "get_height")]


def _bar_horizontal(h) -> bool:
    return bool(h.extra.get("Horizontal", False))


def _bar_set_y(h, v):
    y = np.real(fnum(v)).flatten()
    rects = _bar_rects(h)
    if y.size != len(rects):
        raise MatlabError("YData must have one value for each bar; draw the bars again to change their number.")
    base = h.extra.get("Bottom")
    for k, (p, val) in enumerate(zip(rects, y)):
        b = float(base[k]) if base is not None else float(P._scalar(h.extra.get("BaseValue", 0.0)))
        if _bar_horizontal(h):
            p.set_x(b)
            p.set_width(float(val) - b if base is None else float(val))
        else:
            p.set_y(b)
            p.set_height(float(val) - b if base is None else float(val))
    h.extra["YData"] = y
    _rescale(h)


def _bar_set_x(h, v):
    x = np.real(fnum(v)).flatten()
    rects = _bar_rects(h)
    if x.size != len(rects):
        raise MatlabError("XData must have one value for each bar.")
    old = np.asarray(h.extra.get("XData", x), dtype=float).flatten()
    for p, new, o in zip(rects, x, old):
        if _bar_horizontal(h):
            p.set_y(p.get_y() + float(new - o))
        else:
            p.set_x(p.get_x() + float(new - o))
    h.extra["XData"] = x
    _rescale(h)


def _rescale(h):
    ax = P._obj_axes(h)
    if ax is not None:
        ax.relim()
        ax.autoscale_view()


def _bar_endpoints(which: str):
    def g(h):
        rects = _bar_rects(h)
        if _bar_horizontal(h):
            pos = [p.get_y() + p.get_height() / 2 for p in rects]
            end = [p.get_x() + p.get_width() for p in rects]
        else:
            pos = [p.get_x() + p.get_width() / 2 for p in rects]
            end = [p.get_y() + p.get_height() for p in rects]
        return P._row(pos if which == "x" else end)
    return g


def _bar_cdata_get(h):
    from matplotlib.colors import to_rgb
    return np.array([to_rgb(p.get_facecolor()) for p in _bar_rects(h)]).reshape(-1, 3)


def _bar_cdata_set(h, v):
    c = np.real(fnum(v))
    rects = _bar_rects(h)
    if c.ndim == 2 and c.shape == (len(rects), 3):
        for p, row in zip(rects, c):
            p.set_facecolor(tuple(float(x) for x in np.clip(row, 0, 1)))
    elif c.size == len(rects):
        import matplotlib as mpl
        cmap = P._current_cmap(h.interp)
        cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        vals = c.flatten()
        lo, hi = float(vals.min()), float(vals.max())
        for p, val in zip(rects, vals):
            p.set_facecolor(cmap((val - lo) / (hi - lo) if hi > lo else 0.5))
    else:
        raise MatlabError("CData must have one row [r g b] (or one number) for each bar.")
    h.extra["FaceColorMode"] = "flat"


def _bar_base_set(h, v):
    b = P._scalar(v)
    for p in _bar_rects(h):
        if _bar_horizontal(h):
            end = p.get_x() + p.get_width()
            p.set_x(b)
            p.set_width(end - b)
        else:
            end = p.get_y() + p.get_height()
            p.set_y(b)
            p.set_height(end - b)
    _rescale(h)


def _bar_facecolor_set(h, v):
    if is_text(v) and text(v).lower() == "flat":
        h.extra["FaceColorMode"] = "flat"
        return
    h.extra.pop("FaceColorMode", None)
    P._set_patch_color(h, "face", v)


def _bar_facecolor_get(h):
    if h.extra.get("FaceColorMode") == "flat":
        return char("flat")
    return P._patch_color(h, "face")


def _patch_linestyle(h):
    ps = P._patches(h)
    ls = ps[0].get_linestyle() if ps else "-"
    return char({"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}.get(ls, ls) if isinstance(ls, str)
                else "-")


def _patch_linestyle_set(h, v):
    s = text(v)
    for p in P._patches(h):
        p.set_linestyle("None" if s.lower() == "none" else s)


def _edge_alpha_set(h, v):
    from matplotlib.colors import to_rgba
    a = P._scalar(v)
    for p in P._patches(h):
        ec = np.atleast_2d(np.array(p.get_edgecolor(), dtype=float))
        if ec.size:
            p.set_edgecolor([(*to_rgba(tuple(c))[:3], a) for c in ec] if ec.shape[0] > 1 else (*ec[0][:3], a))
    h.extra["EdgeAlpha"] = P._num_val(a)


_FACE_COMMON = {
    "LineStyle": (_patch_linestyle, _patch_linestyle_set),
    "EdgeAlpha": (lambda h: h.extra.get("EdgeAlpha", P._num_val(1)), _edge_alpha_set),
    "SeriesIndex": (_series_index_get, _series_index_set),
}

P._BAR_PROPS.update({
    "YData": (lambda h: P._row(h.extra.get("YData", [])), _bar_set_y),
    "XData": (lambda h: P._row(h.extra.get("XData", [])), _bar_set_x),
    "XEndPoints": (_bar_endpoints("x"), None),
    "YEndPoints": (_bar_endpoints("y"), None),
    "CData": (_bar_cdata_get, _bar_cdata_set),
    "FaceColor": (_bar_facecolor_get, _bar_facecolor_set),
    "BaseValue": _number("BaseValue", 0.0, apply=_bar_base_set),
    "BarLayout": (lambda h: char(h.extra.get("BarLayout", "grouped")), None),
    "Horizontal": (lambda h: P._oo(_bar_horizontal(h)), None),
    "ShowBaseLine": _onoff_state("ShowBaseLine"),
    **_FACE_COMMON,
})
P._PATCH_PROPS.update(_FACE_COMMON)
for _k in ("area", "pie", "rectangle"):
    if P._PROPS.get(_k) is not P._PATCH_PROPS:
        P._PROPS[_k].update(_FACE_COMMON)


# -- histograms ---------------------------------------------------------------------------------------------------

def _hist_style(h, key: str, default):
    return h.obj.style.get(key, default)


def _hist_display_style_set(h, v):
    s = text(v).lower()
    if s not in ("bar", "stairs"):
        raise MatlabError("DisplayStyle must be 'bar' or 'stairs'.")
    h.obj.display_style = s
    P._hist_redraw(h)


def _hist_bin_method_set(h, v):
    m = text(v).lower()
    h.obj.bin_method = m
    P._hist_redraw(h, edges=P._hist_edges(h.obj.data, method=m))


P._HIST_PROPS.update({
    "DisplayStyle": (lambda h: char(getattr(h.obj, "display_style", "bar")), _hist_display_style_set),
    "BinMethod": (lambda h: char(getattr(h.obj, "bin_method", "auto")), _hist_bin_method_set),
    "BinLimitsMode": _choice("BinLimitsMode", ("auto", "manual"), "auto"),
    "BinCountsMode": _choice("BinCountsMode", ("auto", "manual"), "auto"),
    "LineStyle": (lambda h: char(_hist_style(h, "linestyle", "-")),
                  lambda h, v: (h.obj.style.__setitem__("linestyle", text(v)), P._hist_redraw(h))),
    "EdgeAlpha": (lambda h: P._num_val(_hist_style(h, "edgealpha", 1.0)),
                  lambda h, v: (h.obj.style.__setitem__("edgealpha", P._scalar(v)), P._hist_redraw(h))),
    "SeriesIndex": _state("SeriesIndex", P._num_val(1)),
})


# -- colour order ------------------------------------------------------------------------------------------------

def default_colors() -> np.ndarray:
    import matplotlib as mpl
    from matplotlib.colors import to_rgb
    return np.array([to_rgb(c) for c in mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])])


def color_order(ax) -> np.ndarray:
    c = getattr(ax, "_mylab_colororder", None)
    return np.asarray(c, dtype=float) if c is not None else default_colors()


def _style_order(ax) -> list:
    return list(getattr(ax, "_mylab_styleorder", ["-"]))


def _split_style(s: str) -> dict:
    marker = "".join(ch for ch in s if ch in "o+*.xsd^v<>ph")
    line = s.replace(marker, "") if marker else s
    out = {"linestyle": line if line in ("-", "--", ":", "-.") else ("None" if marker else "-")}
    out["marker"] = P._mpl_marker(marker) if marker else "None"
    return out


def apply_cycle(ax, start: int = 0) -> None:
    """Install the colour and line-style cycle of an axes; the next line gets entry ``start``."""
    from cycler import cycler
    colors = [tuple(c) for c in color_order(ax)]
    styles = [_split_style(s) for s in _style_order(ax)]
    combos = [(c, s) for s in styles for c in colors]
    k = start % len(combos)
    combos = combos[k:] + combos[:k]
    cyc = cycler(color=[c for c, _ in combos])
    if len(styles) > 1 or styles[0] != {"linestyle": "-", "marker": "None"}:
        cyc = cyc + cycler(linestyle=[s["linestyle"] for _, s in combos]) \
            + cycler(marker=[s["marker"] for _, s in combos])
    ax.set_prop_cycle(cyc)
    ax._mylab_cycle_start = k


def to_color_matrix(v) -> np.ndarray:
    """An n-by-3 colour matrix from a matrix, colour names, hex codes or a palette name."""
    from matplotlib.colors import to_rgb
    if isinstance(v, (Cell, StringArray)) or is_text(v):
        names = P._text_list(v)
        if len(names) == 1 and names[0].lower() in _PALETTES:
            return np.array(_PALETTES[names[0].lower()], dtype=float)
        out = []
        for n in names:
            c = P._COLOR_NAMES.get(n.lower(), n)
            try:
                out.append(to_rgb(c))
            except ValueError:
                raise MatlabError(f"'{n}' is not a colour name, a hexadecimal colour code or a palette.") from None
        return np.array(out, dtype=float).reshape(-1, 3)
    m = np.real(fnum(v))
    if m.ndim != 2 or m.shape[1] != 3 or m.shape[0] < 1 or m.min() < 0 or m.max() > 1:
        raise MatlabError("Colours must be an m-by-3 matrix of RGB values between 0 and 1.")
    return m.astype(float)


_PALETTES = {
    "gem": [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
            [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840]],
    "gem12": [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
              [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840], [1, 0.8390, 0.0390],
              [0.3960, 0.5090, 0.9920], [1, 0.2700, 0.2270], [0, 0.6390, 0.6390], [0.7960, 0.5170, 0.3640]],
    "glow": [[0.1490, 0.5490, 0.8660], [0.9600, 0.4660, 0.1600], [0.9290, 0.6940, 0.1250], [0.6000, 0.3200, 0.7600],
             [0.3960, 0.7450, 0.2470], [0.3010, 0.7450, 0.9330], [0.8940, 0.2200, 0.2630]],
    "sail": [[0.0900, 0.3500, 0.5800], [0.2000, 0.5200, 0.7400], [0.4300, 0.7000, 0.8500], [0.7200, 0.8500, 0.9200],
             [0.9500, 0.6500, 0.2500], [0.8500, 0.3500, 0.2000]],
    "reef": [[0.9500, 0.4500, 0.4000], [0.9800, 0.7000, 0.4000], [0.4000, 0.7800, 0.7200], [0.2000, 0.6000, 0.7000],
             [0.5500, 0.4200, 0.7000], [0.9500, 0.8500, 0.5000]],
    "meadow": [[0.2200, 0.5500, 0.2400], [0.5500, 0.7500, 0.2900], [0.9600, 0.7900, 0.2700], [0.9000, 0.4900, 0.1300],
               [0.4700, 0.3300, 0.2800], [0.1500, 0.6500, 0.6000], [0.7500, 0.2300, 0.2700]],
    "dye": [[0.7300, 0.1500, 0.3600], [0.1200, 0.4700, 0.7100], [0.9900, 0.6500, 0.0400], [0.3500, 0.6900, 0.4100],
            [0.5400, 0.3300, 0.6900], [0.9300, 0.4300, 0.1300], [0.2500, 0.7300, 0.8300]],
    "earth": [[0.4700, 0.3100, 0.1700], [0.7300, 0.5300, 0.2700], [0.3300, 0.4700, 0.2700], [0.6400, 0.3000, 0.2000],
              [0.8600, 0.7400, 0.5000], [0.2700, 0.3900, 0.4700], [0.5800, 0.6100, 0.4000]],
}


def set_color_order(interp, ax, colors: np.ndarray, recolor: bool = True) -> None:
    """Give an axes a new colour order; lines that still have their automatic colour follow it."""
    from matplotlib.colors import to_rgb
    old = color_order(ax)
    ax._mylab_colororder = np.asarray(colors, dtype=float)
    apply_cycle(ax, len(ax.lines))
    if not recolor:
        return
    for k, ln in enumerate(ax.lines):
        try:
            auto = np.allclose(to_rgb(ln.get_color()), old[k % len(old)], atol=1e-3)
        except (ValueError, TypeError):
            auto = False
        if auto:
            ln.set_color(tuple(colors[k % len(colors)]))
    for k, cont in enumerate(c for c in ax.containers if c.__class__.__name__ == "BarContainer"):
        for p in cont.patches:
            try:
                if np.allclose(to_rgb(p.get_facecolor()), old[k % len(old)], atol=1e-3):
                    p.set_facecolor(tuple(colors[k % len(colors)]))
            except (ValueError, TypeError):
                pass
    if ax.get_legend() is not None:
        P._refresh_legend(interp, ax)


def _color_order_index_get(h):
    n = len(color_order(h.obj))
    return P._num_val((getattr(h.obj, "_mylab_cycle_start", 0) + _lines_since_cycle(h.obj)) % n + 1)


def _lines_since_cycle(ax) -> int:
    return len(ax.lines) - getattr(ax, "_mylab_cycle_lines", 0)


def _color_order_index_set(h, v):
    k = int(P._scalar(v))
    if k < 1:
        raise MatlabError("ColorOrderIndex must be a positive whole number.")
    apply_cycle(h.obj, k - 1)
    h.obj._mylab_cycle_lines = len(h.obj.lines)


def _style_order_set(h, v):
    styles = P._text_list(v)
    if not styles:
        raise MatlabError("LineStyleOrder needs at least one line style.")
    for s in styles:
        if not P._is_linespec(s) or re.search(r"[rgbcmykw]", s):
            raise MatlabError(f"'{s}' is not a line style or marker.")
    h.obj._mylab_styleorder = styles
    apply_cycle(h.obj, len(h.obj.lines))
    h.obj._mylab_cycle_lines = len(h.obj.lines)


def _style_order_get(h):
    styles = _style_order(h.obj)
    return char(styles[0]) if len(styles) == 1 else P._cell_column([char(s) for s in styles])


def _color_order_set(h, v):
    set_color_order(h.interp, h.obj, to_color_matrix(v), recolor=False)
    h.obj._mylab_cycle_lines = len(h.obj.lines)


# -- axes: units and positions ---------------------------------------------------------------------------------

_UNITS = ("normalized", "pixels", "inches", "centimeters", "points", "characters")
_CHAR_PX = (5.0, 13.0)  # nominal size of one character cell in pixels


def _unit_scale(units: str, ref_w: float, ref_h: float, dpi: float) -> tuple[float, float]:
    """Pixels per unit, horizontally and vertically."""
    if units == "normalized":
        return ref_w, ref_h
    if units == "inches":
        return dpi, dpi
    if units == "centimeters":
        return dpi / 2.54, dpi / 2.54
    if units == "points":
        return dpi / 72.0, dpi / 72.0
    if units == "characters":
        return _CHAR_PX
    return 1.0, 1.0


def _fig_pixels(fig) -> tuple[float, float]:
    w, hh = fig.get_size_inches() * P._fig_dpi(fig)
    return float(w), float(hh)


def _axes_units(h) -> str:
    return str(h.extra.get("Units", "normalized"))


def _frac_to_units(h, frac) -> np.ndarray:
    fig = h.obj.figure if h.kind != "figure" else h.obj
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(_axes_units(h), w, hh, P._fig_dpi(fig))
    f = np.asarray(frac, dtype=float)
    return P._row([f[0] * w / sx, f[1] * hh / sy, f[2] * w / sx, f[3] * hh / sy])


def _units_to_frac(h, v) -> list:
    p = np.real(fnum(v)).flatten()
    if p.size != 4 or p[2] <= 0 or p[3] <= 0:
        raise MatlabError("A position must be [left bottom width height] with a positive width and height.")
    fig = h.obj.figure
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(_axes_units(h), w, hh, P._fig_dpi(fig))
    return [float(p[0] * sx / w), float(p[1] * sy / hh), float(p[2] * sx / w), float(p[3] * sy / hh)]


def _axes_position_get(h):
    return _frac_to_units(h, h.obj.get_position().bounds)


def _axes_position_set(h, v):
    h.obj.set_position(_units_to_frac(h, v))
    h.extra["PositionConstraint"] = "innerposition"


def _tight_frac(ax):
    """The box around the axes with its tick labels, axis labels and title, in figure fractions."""
    fig = ax.figure
    try:
        fig.canvas.draw()
        bb = ax.get_tightbbox(fig.canvas.get_renderer())
    except (ValueError, RuntimeError, AttributeError):
        bb = None
    if bb is None:
        return ax.get_position().bounds
    inv = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _axes_outer_get(h):
    return _frac_to_units(h, _tight_frac(h.obj))


def _axes_outer_set(h, v):
    want = _units_to_frac(h, v)
    inner = h.obj.get_position().bounds
    outer = _tight_frac(h.obj)
    left, bottom = inner[0] - outer[0], inner[1] - outer[1]
    right = (outer[0] + outer[2]) - (inner[0] + inner[2])
    top = (outer[1] + outer[3]) - (inner[1] + inner[3])
    w, hh = want[2] - left - right, want[3] - bottom - top
    if w <= 0 or hh <= 0:
        raise MatlabError("OuterPosition is too small to hold the axes with its labels.")
    h.obj.set_position([want[0] + left, want[1] + bottom, w, hh])
    h.extra["PositionConstraint"] = "outerposition"


def _axes_tight_inset(h):
    inner = h.obj.get_position().bounds
    outer = _tight_frac(h.obj)
    frac = [max(inner[0] - outer[0], 0.0), max(inner[1] - outer[1], 0.0),
            max(outer[0] + outer[2] - inner[0] - inner[2], 0.0), max(outer[1] + outer[3] - inner[1] - inner[3], 0.0)]
    return _frac_to_units(h, frac)


def _units_set(h, v):
    u = text(v).lower()
    if u not in _UNITS:
        raise MatlabError(f"Units must be one of: {', '.join(_UNITS)}.")
    h.extra["Units"] = u


# -- axes: grid, ticks, fonts -------------------------------------------------------------------------------------

_LS = ("-", "--", ":", "-.", "none")


def _grid_style(which: str, key: str):
    def apply(h, v):
        val = v
        if key == "grid_alpha":
            val = float(P._scalar(v))
        elif key == "grid_color":
            val = P._color(v)
        elif key == "grid_linewidth":
            val = float(P._scalar(v))
        elif key == "grid_linestyle":
            val = "None" if v == "none" else v
        h.obj.tick_params(which=which, **{key: val})
    return apply


def _tick_length_apply(h, v):
    bbox = h.obj.get_window_extent()
    longest = max(bbox.width, bbox.height) * 72.0 / h.obj.figure.dpi
    h.obj.tick_params(which="major", length=float(v.flat[0]) * longest)
    h.obj.tick_params(which="minor", length=float(v.flat[0]) * longest * 0.5)


def _tickdir_set(h, v):
    d = text(v).lower()
    if d not in ("in", "out", "both", "none"):
        raise MatlabError("TickDir must be 'in', 'out', 'both' or 'none'.")
    if d == "none":
        h.obj.tick_params(length=0)
    else:
        h.obj.tick_params(direction={"both": "inout"}.get(d, d))
        if h.extra.get("TickDir") == "none":
            h.obj.tick_params(length=3.5)
    h.extra["TickDir"] = d


def _axes_labels(ax) -> list:
    out = [ax.xaxis.label, ax.yaxis.label]
    if hasattr(ax, "zaxis"):
        out.append(ax.zaxis.label)
    return out


def _tick_labels(ax) -> list:
    out = ax.get_xticklabels() + ax.get_yticklabels()
    if hasattr(ax, "get_zticklabels"):
        out += ax.get_zticklabels()
    return out


def _font_weight_apply(h, v):
    for t in _tick_labels(h.obj) + _axes_labels(h.obj):
        t.set_fontweight(v)
    h.obj.tick_params(labelfontfamily=None)
    for axis in (h.obj.xaxis, h.obj.yaxis):
        for tick in axis.get_major_ticks():
            tick.label1.set_fontweight(v)


def _font_angle_apply(h, v):
    for t in _tick_labels(h.obj) + _axes_labels(h.obj) + [h.obj.title]:
        t.set_fontstyle("italic" if v == "italic" else "normal")


def _base_font(h) -> float:
    return float(P._fontsize_get(h).flat[0])


def _title_mult_apply(h, v):
    h.obj.title.set_fontsize(_base_font(h) * float(v.flat[0]))
    P._layout_subtitle(h.obj)


def _label_mult_apply(h, v):
    for t in _axes_labels(h.obj):
        t.set_fontsize(_base_font(h) * float(v.flat[0]))


def _axes_fontsize_set(h, v):
    fs = P._scalar(v)
    if fs <= 0:
        raise MatlabError("FontSize must be a positive number.")
    ax = h.obj
    h.extra["FontSize"] = fs
    ax.tick_params(labelsize=fs)
    ax.title.set_fontsize(fs * float(np.asarray(h.extra.get("TitleFontSizeMultiplier", 1.1)).flat[0]))
    for t in _axes_labels(ax):
        t.set_fontsize(fs * float(np.asarray(h.extra.get("LabelFontSizeMultiplier", 1.1)).flat[0]))
    sub = getattr(ax, "_mylab_subtitle", None)
    if sub is not None:
        sub.set_fontsize(fs * 1.1)
        P._layout_subtitle(ax)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(fs * 0.9)


def _limit_method(which: str):
    def apply(h, v):
        set_limit_method(h.obj, which, v)
    return _choice(f"{which.upper()}LimitMethod", ("tickaligned", "tight", "padded"), "tickaligned", apply)


def set_limit_method(ax, which: str, method: str) -> None:
    """How automatic limits are chosen: at tick marks, at the data, or with a margin around the data."""
    from matplotlib.ticker import MaxNLocator
    axis = getattr(ax, f"{which}axis", None)
    if axis is None or getattr(ax, "name", "") != "rectilinear":
        return
    if P._ruler_of(ax, which) is None and getattr(ax, f"get_{which}scale")() == "linear" and \
            not getattr(ax, f"_mylab_{which}ticks_fixed", False):
        if method == "tickaligned":
            axis.set_major_locator(P._aligned_locators()["lin"]())
        else:
            axis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
    getattr(ax, f"set_{which}margin")(0.07 if method == "padded" else 0.0)
    if method == "padded":
        ax._mylab_padded = True
    getattr(ax, f"set_autoscale{which}_on")(True)
    ax.autoscale_view(scalex=which == "x", scaley=which == "y")


def _lim_mode(which: str):
    def g(h):
        auto = getattr(h.obj, f"get_autoscale{which}_on", lambda: True)()
        return char("auto" if auto else "manual")

    def s(h, v):
        mode = text(v).lower()
        if mode not in ("auto", "manual"):
            raise MatlabError(f"{which.upper()}LimMode must be 'auto' or 'manual'.")
        f = getattr(h.obj, f"set_autoscale{which}_on", None)
        if f is not None:
            f(mode == "auto")
            if mode == "auto":
                h.obj.autoscale_view()
    return g, s


def _tick_mode(which: str):
    def g(h):
        return char("manual" if getattr(h.obj, f"_mylab_{which}ticks_fixed", False) else "auto")

    def s(h, v):
        mode = text(v).lower()
        if mode == "auto":
            P._set_ticks_auto(h.obj, which)
        elif mode == "manual":
            P._set_tick_values(h.obj, which, np.asarray(getattr(h.obj, f"get_{which}ticks")(), float))
        else:
            raise MatlabError(f"{which.upper()}TickMode must be 'auto' or 'manual'.")
    return g, s


def _ticklabel_mode(which: str):
    def g(h):
        return char("manual" if getattr(h.obj, f"_mylab_{which}ticklabels", None) is not None else "auto")

    def s(h, v):
        mode = text(v).lower()
        if mode == "auto":
            fixed = getattr(h.obj, f"_mylab_{which}ticks_fixed", False)
            ticks = np.asarray(getattr(h.obj, f"get_{which}ticks")(), float)
            P._set_ticks_auto(h.obj, which)
            if fixed:
                P._set_tick_values(h.obj, which, ticks)
        elif mode != "manual":
            raise MatlabError(f"{which.upper()}TickLabelMode must be 'auto' or 'manual'.")
    return g, s


def _minor_tick_get(which: str):
    def g(h):
        axis = getattr(h.obj, f"{which}axis", None)
        if axis is None:
            return P._oo(False)
        return P._oo(type(axis.get_minor_locator()).__name__ != "NullLocator")
    return g


def _z_only(fn):
    def s(h, v):
        if not hasattr(h.obj, "zaxis"):
            raise MatlabError("These axes are two-dimensional and have no z-axis.")
        fn(h, v)
    return s


def _zdir_get(h):
    return char("reverse" if hasattr(h.obj, "zaxis_inverted") and h.obj.zaxis_inverted() else "normal")


def _zdir_set(h, v):
    want = text(v).lower() == "reverse"
    if h.obj.zaxis_inverted() != want:
        h.obj.invert_zaxis()


def _zcolor_set(h, v):
    c = P._color(v)
    h.obj.tick_params(axis="z", colors=c)
    h.obj.zaxis.label.set_color(c)
    h.obj.zaxis.line.set_color(c)


def _zticklabel_get(h):
    if not hasattr(h.obj, "get_zticks"):
        return Cell.from_list([])
    fixed = getattr(h.obj, "_mylab_zticklabels", None)
    labels = fixed if fixed is not None else [f"{t:g}" for t in h.obj.get_zticks()]
    return P._cell_column([char(s) for s in labels])


def _zticklabel_set(h, v):
    labels = P._text_list(v)
    ticks = list(h.obj.get_zticks())
    full = (labels + [""] * len(ticks))[:len(ticks)]
    h.obj.set_zticks(ticks)
    h.obj.set_zticklabels(full)
    h.obj._mylab_zticklabels = full


def _axisloc_set(which: str):
    def s(h, v):
        loc = text(v).lower()
        ax = h.obj
        axis = getattr(ax, f"{which}axis")
        ok = ("bottom", "top", "origin") if which == "x" else ("left", "right", "origin")
        if loc not in ok:
            raise MatlabError(f"{which.upper()}AxisLocation must be one of: {', '.join(ok)}.")
        main, other = ok[0], ok[1]
        if loc == "origin":
            ax.spines[main].set_position("zero")
            ax.spines[other].set_visible(False)
            (axis.tick_bottom if which == "x" else axis.tick_left)()
            axis.set_label_position(main)
        else:
            for name in (main, other):
                ax.spines[name].set_position(("outward", 0))
            ax.spines[other].set_visible(ax.spines["top" if which == "y" else "right"].get_visible()
                                         or loc == other)
            getattr(axis, f"tick_{loc}")()
            axis.set_label_position(loc)
        h.extra[f"{which.upper()}AxisLocation"] = loc
    return s


def _axes_visible_get(h):
    return P._oo(h.obj.axison and h.obj.get_visible())


def _axes_visible_set(h, v):
    # Hiding an axes hides its box, ticks and labels; what is plotted in it stays visible.
    if P._onoff(v):
        h.obj.set_visible(True)
        h.obj.set_axis_on()
    else:
        h.obj.set_axis_off()


def _layer_apply(h, v):
    h.obj.set_axisbelow(v == "bottom")


def _projection_apply(h, v):
    if hasattr(h.obj, "set_proj_type"):
        h.obj.set_proj_type("ortho" if v == "orthographic" else "persp")


def _color_scale_apply(h, v):
    from matplotlib.colors import LogNorm, Normalize
    for m in P._mappables(h.obj):
        lo, hi = m.get_clim()
        if v == "log":
            arr = np.asarray(m.get_array(), dtype=float)
            pos = arr[np.isfinite(arr) & (arr > 0)]
            lo = lo if lo > 0 else (float(pos.min()) if pos.size else 1e-3)
            m.set_norm(LogNorm(vmin=lo, vmax=max(hi, lo * 10)))
        else:
            m.set_norm(Normalize(vmin=lo, vmax=hi))


def _clim_mode_get(h):
    return char(h.extra.get("CLimMode", "auto"))


def _clim_mode_set(h, v):
    mode = text(v).lower()
    if mode not in ("auto", "manual"):
        raise MatlabError("CLimMode must be 'auto' or 'manual'.")
    if mode == "auto":
        for m in P._mappables(h.obj):
            m.autoscale()
    h.extra["CLimMode"] = mode


def _aspect_prop(which: str):
    def g(h):
        return _on_axes(h, P._aspect, h.interp, [], 1, which)

    def s(h, v):
        _on_axes(h, P._aspect, h.interp, [v], 0, which)
    return g, s


def _aspect_mode(which: str):
    def g(h):
        return char("manual" if getattr(h.obj, f"_mylab_{which}", None) is not None else "auto")

    def s(h, v):
        mode = text(v).lower()
        if mode not in ("auto", "manual"):
            raise MatlabError("The mode must be 'auto' or 'manual'.")
        _on_axes(h, P._aspect, h.interp, [char(mode)], 0, which)
    return g, s


def _title_align_apply(h, v):
    x = {"left": 0.0, "center": 0.5, "right": 1.0}[v]
    h.obj.title.set_ha(v)
    h.obj.title.set_x(x)
    sub = getattr(h.obj, "_mylab_subtitle", None)
    if sub is not None:
        sub.set_ha(v)
        sub.set_x(x)


def _zgrid_get(h):
    axis = getattr(h.obj, "zaxis", None)
    return P._oo(bool(axis is not None and getattr(h.obj, "_draw_grid", False)))


def _current_point(h):
    return np.zeros((2, 3))


def _layout_options(h):
    lay = P._layout_of_axes(h.interp, h.obj)
    if lay is None:
        return _EMPTY
    return _sub(h, "layout", "layoutoptions")


P._AXES_PROPS.update({
    "Units": (lambda h: char(_axes_units(h)), _units_set),
    "Position": (_axes_position_get, _axes_position_set),
    "InnerPosition": (_axes_position_get, _axes_position_set),
    "OuterPosition": (_axes_outer_get, _axes_outer_set),
    "TightInset": (_axes_tight_inset, None),
    "PositionConstraint": _choice("PositionConstraint", ("outerposition", "innerposition"), "outerposition"),
    "ActivePositionProperty": _choice("PositionConstraint", ("outerposition", "position", "innerposition"),
                                      "outerposition"),
    "Visible": (_axes_visible_get, _axes_visible_set),
    "FontSize": (P._fontsize_get, _axes_fontsize_set),
    "FontWeight": _choice("FontWeight", ("normal", "bold"), "normal", _font_weight_apply),
    "FontAngle": _choice("FontAngle", ("normal", "italic"), "normal", _font_angle_apply),
    "FontUnits": _choice("FontUnits", ("points", "inches", "centimeters", "normalized", "pixels"), "points"),
    "FontSmoothing": _onoff_state("FontSmoothing"),
    "TitleFontSizeMultiplier": _number("TitleFontSizeMultiplier", 1.1, 0.01, 100, _title_mult_apply),
    "LabelFontSizeMultiplier": _number("LabelFontSizeMultiplier", 1.1, 0.01, 100, _label_mult_apply),
    "TitleFontWeight": _choice("TitleFontWeight", ("bold", "normal"), "bold",
                               lambda h, v: h.obj.title.set_fontweight(v)),
    "TitleHorizontalAlignment": _choice("TitleHorizontalAlignment", ("center", "left", "right"), "center",
                                        _title_align_apply),
    "TickLabelInterpreter": _choice("TickLabelInterpreter", ("tex", "latex", "none"), "tex"),
    "TickDir": (lambda h: char(h.extra.get("TickDir", "in")), _tickdir_set),
    "TickLength": _vector("TickLength", [0.01, 0.025], 2, _tick_length_apply),
    "GridAlpha": _number("GridAlpha", 0.15, 0, 1, _grid_style("major", "grid_alpha")),
    "GridColor": _state("GridColor", P._row([0.15, 0.15, 0.15]), lambda v: P._rgb(P._color(v)),
                        _grid_style("major", "grid_color")),
    "GridLineStyle": _choice("GridLineStyle", _LS, "-", _grid_style("major", "grid_linestyle")),
    "GridLineWidth": _number("GridLineWidth", 0.5, 0, 100, _grid_style("major", "grid_linewidth")),
    "MinorGridAlpha": _number("MinorGridAlpha", 0.25, 0, 1, _grid_style("minor", "grid_alpha")),
    "MinorGridColor": _state("MinorGridColor", P._row([0.1, 0.1, 0.1]), lambda v: P._rgb(P._color(v)),
                             _grid_style("minor", "grid_color")),
    "MinorGridLineStyle": _choice("MinorGridLineStyle", _LS, ":", _grid_style("minor", "grid_linestyle")),
    "MinorGridLineWidth": _number("MinorGridLineWidth", 0.5, 0, 100, _grid_style("minor", "grid_linewidth")),
    "ColorOrder": (lambda h: color_order(h.obj), _color_order_set),
    "ColorOrderIndex": (_color_order_index_get, _color_order_index_set),
    "LineStyleOrder": (_style_order_get, _style_order_set),
    "LineStyleOrderIndex": _number("LineStyleOrderIndex", 1, 1),
    "LineStyleCyclingMethod": _choice("LineStyleCyclingMethod", ("aftercolor", "beforecolor", "withcolor"),
                                      "aftercolor"),
    "NextSeriesIndex": (lambda h: P._num_val(len(h.obj.lines) + len(h.obj.containers) + 1), None),
    "XLimMode": _lim_mode("x"), "YLimMode": _lim_mode("y"), "ZLimMode": _lim_mode("z"),
    "XLimitMethod": _limit_method("x"), "YLimitMethod": _limit_method("y"),
    "ZLimitMethod": _choice("ZLimitMethod", ("tickaligned", "tight", "padded"), "tickaligned"),
    "XTickMode": _tick_mode("x"), "YTickMode": _tick_mode("y"), "ZTickMode": _tick_mode("z"),
    "XTickLabelMode": _ticklabel_mode("x"), "YTickLabelMode": _ticklabel_mode("y"),
    "ZTickLabelMode": _ticklabel_mode("z"),
    "XMinorTick": (_minor_tick_get("x"), P._minortick_set("x")),
    "YMinorTick": (_minor_tick_get("y"), P._minortick_set("y")),
    "ZMinorTick": (_minor_tick_get("z"), _z_only(P._minortick_set("z"))),
    "ZMinorGrid": _onoff_state("ZMinorGrid", "off"),
    "ZGrid": (_zgrid_get, lambda h, v: h.obj.grid(P._onoff(v)) if hasattr(h.obj, "zaxis")
              else h.extra.__setitem__("ZGrid", "on" if P._onoff(v) else "off")),
    "ZDir": (_zdir_get, _z_only(_zdir_set)),
    "ZColor": (lambda h: P._rgb(h.obj.zaxis.label.get_color()) if hasattr(h.obj, "zaxis")
               else P._row([0.15, 0.15, 0.15]), _z_only(_zcolor_set)),
    "ZTickLabel": (_zticklabel_get, _z_only(_zticklabel_set)),
    "ZTickLabelRotation": (lambda h: P._num_val(0), _z_only(lambda h, v: h.obj.tick_params(
        axis="z", labelrotation=P._scalar(v)))),
    "XAxisLocation": (lambda h: char(h.extra.get("XAxisLocation", "bottom")), _axisloc_set("x")),
    "YAxisLocation": (lambda h: char(h.extra.get("YAxisLocation", "left")), _axisloc_set("y")),
    "Layer": _choice("Layer", ("bottom", "top"), "bottom", _layer_apply),
    "BoxStyle": _choice("BoxStyle", ("back", "full"), "back"),
    "Projection": _choice("Projection", ("orthographic", "perspective"), "orthographic", _projection_apply),
    "ColorScale": _choice("ColorScale", ("linear", "log"), "linear", _color_scale_apply),
    "AlphaScale": _choice("AlphaScale", ("linear", "log"), "linear"),
    "CLimMode": (_clim_mode_get, _clim_mode_set),
    "DataAspectRatio": _aspect_prop("daspect"),
    "PlotBoxAspectRatio": _aspect_prop("pbaspect"),
    "DataAspectRatioMode": _aspect_mode("daspect"),
    "PlotBoxAspectRatioMode": _aspect_mode("pbaspect"),
    "SortMethod": _choice("SortMethod", ("depth", "childorder"), "childorder"),
    "ClippingStyle": _choice("ClippingStyle", ("3dbox", "rectangle"), "3dbox"),
    "AmbientLightColor": _state("AmbientLightColor", P._row([1, 1, 1]), lambda v: P._rgb(P._color(v))),
    "Toolbar": _state("Toolbar"),
    "Interactions": _state("Interactions"),
    "CurrentPoint": (_current_point, None),
    "Alphamap": _state("Alphamap", np.linspace(0, 1, 64).reshape(1, -1)),
    "Layout": (_layout_options, None),
})
for _name in ("TickLabelInterpreter", "Layer", "TickLength", "GridAlpha", "MinorGridAlpha", "GridColor",
              "GridLineStyle", "MinorGridLineStyle", "FontWeight", "Units", "ActivePositionProperty",
              "PositionConstraint", "LabelFontSizeMultiplier", "TitleFontSizeMultiplier", "TitleFontWeight",
              "DataAspectRatio", "PlotBoxAspectRatio", "ColorOrder", "LineStyleOrder", "SortMethod",
              "OuterPosition", "Clipping", "ColorOrderIndex", "LineStyleOrderIndex", "ZMinorTick", "ZMinorGrid",
              "XLimMode", "YLimMode", "ZLimMode", "TickLabelFormat", "FontAngle", "FontSmoothing", "XTickMode",
              "YTickMode", "XTickLabelMode", "YTickLabelMode", "ZDir", "ZColor", "ZTickLabel", "CLimMode",
              "DataAspectRatioMode", "PlotBoxAspectRatioMode", "Projection", "AmbientLightColor", "HitTest",
              "PickableParts", "ButtonDownFcn", "Interruptible", "BusyAction", "HandleVisibility", "Toolbar",
              "Interactions", "AlphaScale", "ColorScale"):
    P._AXES_IGNORED.discard(_name)


# tiled layout options of an axes: ax.Layout.Tile, ax.Layout.TileSpan

def _tile_of(h):
    ax = h.obj.owner.obj
    lay = P._layout_of_axes(h.interp, ax)
    ss = ax.get_subplotspec()
    rows, cols = lay.grid_size()
    return lay, ax, ss, rows, cols


def _layout_tile_get(h):
    lay, ax, ss, rows, cols = _tile_of(h)
    return P._num_val(lay.tile_number(ss.rowspan.start, ss.colspan.start))


def _layout_span_get(h):
    lay, ax, ss, rows, cols = _tile_of(h)
    return P._row([len(ss.rowspan), len(ss.colspan)])


def _layout_tile_set(h, v):
    lay, ax, ss, rows, cols = _tile_of(h)
    lay.move(ax, int(P._scalar(v)), (len(ss.rowspan), len(ss.colspan)))


def _layout_span_set(h, v):
    lay, ax, ss, rows, cols = _tile_of(h)
    span = np.real(fnum(v)).flatten()
    if span.size != 2 or np.any(span < 1):
        raise MatlabError("TileSpan must be [rows columns] with positive whole numbers.")
    lay.move(ax, lay.tile_number(ss.rowspan.start, ss.colspan.start), (int(span[0]), int(span[1])))


P._CLASS["layoutoptions"] = "matlab.graphics.layout.TiledChartLayoutOptions"
P._PROPS["layoutoptions"] = {"Tile": (_layout_tile_get, _layout_tile_set),
                             "TileSpan": (_layout_span_get, _layout_span_set)}


# -- figures ----------------------------------------------------------------------------------------------------

_SCREEN = (1920.0, 1080.0)


def _fig_units(h) -> str:
    return str(h.extra.get("Units", "pixels"))


def _fig_scale(h) -> tuple[float, float]:
    return _unit_scale(_fig_units(h), _SCREEN[0], _SCREEN[1], P._fig_dpi(h.obj))


def _fig_pos_get(h):
    w, hh = _fig_pixels(h.obj)
    left, bottom = getattr(h.obj, "_mylab_origin", (100.0, 100.0))
    sx, sy = _fig_scale(h)
    if _fig_units(h) == "pixels":
        return P._row([left, bottom, round(w), round(hh)])
    return P._row([left / sx, bottom / sy, w / sx, hh / sy])


def _fig_pos_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size != 4:
        raise MatlabError("Position must be a 4-element vector [left bottom width height].")
    sx, sy = _fig_scale(h)
    w, hh = p[2] * sx, p[3] * sy
    if w <= 0 or hh <= 0:
        raise MatlabError("Figure width and height must be positive.")
    old = _fig_pixels(h.obj)
    h.obj._mylab_origin = (float(p[0] * sx), float(p[1] * sy))
    dpi = P._fig_dpi(h.obj)
    h.obj.set_size_inches(w / dpi, hh / dpi, forward=True)
    if (round(w), round(hh)) != (round(old[0]), round(old[1])):
        for name in ("SizeChangedFcn", "ResizeFcn"):
            if fire(h.interp, h, name, "SizeChanged"):
                break


def _fig_units_set(h, v):
    u = text(v).lower()
    if u not in _UNITS:
        raise MatlabError(f"Units must be one of: {', '.join(_UNITS)}.")
    h.extra["Units"] = u


def _fig_current_axes_set(h, v):
    hs = P._handles_of(v)
    if len(hs) != 1 or hs[0].kind not in ("axes", "polaraxes") or hs[0].obj.figure is not h.obj:
        raise MatlabError("CurrentAxes must be an axes in this figure.")
    n = P._fig_number(h.interp, h.obj)
    P._state(h.interp).axes[n] = hs[0].obj


def _paper_units(h) -> str:
    return str(h.extra.get("PaperUnits", "inches"))


def _paper_scale(h) -> float:
    """Inches per paper unit."""
    return {"inches": 1.0, "centimeters": 1 / 2.54, "points": 1 / 72.0}.get(_paper_units(h), 1.0)


_PAPER_SIZES = {"usletter": (8.5, 11.0), "uslegal": (8.5, 14.0), "tabloid": (11.0, 17.0), "a0": (33.11, 46.81),
                "a1": (23.39, 33.11), "a2": (16.54, 23.39), "a3": (11.69, 16.54), "a4": (8.27, 11.69),
                "a5": (5.83, 8.27), "b4": (9.84, 13.90), "b5": (6.93, 9.84)}


def _paper_size_in(h) -> tuple[float, float]:
    size = h.extra.get("_PaperSizeIn")
    if size is None:
        size = _PAPER_SIZES[str(h.extra.get("PaperType", "usletter")).lower()]
    w, hh = float(size[0]), float(size[1])
    if str(h.extra.get("PaperOrientation", "portrait")) == "landscape" and w < hh:
        w, hh = hh, w
    return w, hh


def paper_position_inches(h):
    """[left bottom width height] in inches that a manual PaperPosition asks for, or None (auto)."""
    if str(h.extra.get("PaperPositionMode", "auto")) != "manual":
        return None
    p = h.extra.get("_PaperPositionIn")
    return None if p is None else tuple(float(x) for x in p)


def _paper_pos_get(h):
    p = h.extra.get("_PaperPositionIn")
    if p is None or str(h.extra.get("PaperPositionMode", "auto")) == "auto":
        w, hh = (float(x) for x in h.obj.get_size_inches())
        pw, ph = _paper_size_in(h)
        p = ((pw - w) / 2, (ph - hh) / 2, w, hh)
    k = _paper_scale(h)
    if _paper_units(h) == "normalized":
        pw, ph = _paper_size_in(h)
        return P._row([p[0] / pw, p[1] / ph, p[2] / pw, p[3] / ph])
    return P._row([x / k for x in p])


def _paper_pos_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size != 4 or p[2] <= 0 or p[3] <= 0:
        raise MatlabError("PaperPosition must be [left bottom width height] with a positive width and height.")
    k = _paper_scale(h)
    if _paper_units(h) == "normalized":
        pw, ph = _paper_size_in(h)
        h.extra["_PaperPositionIn"] = (p[0] * pw, p[1] * ph, p[2] * pw, p[3] * ph)
    else:
        h.extra["_PaperPositionIn"] = tuple(float(x) * k for x in p)
    h.extra["PaperPositionMode"] = "manual"


def _paper_size_get(h):
    w, hh = _paper_size_in(h)
    k = _paper_scale(h)
    return P._row([w / k, hh / k])


def _paper_size_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size != 2 or np.any(p <= 0):
        raise MatlabError("PaperSize must be [width height] with positive numbers.")
    k = _paper_scale(h)
    h.extra["_PaperSizeIn"] = (float(p[0]) * k, float(p[1]) * k)
    h.extra["PaperType"] = "<custom>"


def _paper_type_set(h, v):
    t = text(v).lower()
    if t not in _PAPER_SIZES:
        raise MatlabError(f"Unknown PaperType '{text(v)}'. Use one of: {', '.join(sorted(_PAPER_SIZES))}.")
    h.extra["PaperType"] = t
    h.extra.pop("_PaperSizeIn", None)


def _fig_alpha_get(h):
    return np.linspace(0, 1, 64).reshape(1, -1)


def _fig_visible_get(h):
    return P._oo(h.extra.get("Visible", True) not in (False, "off"))


P._FIGURE_PROPS.update({
    "Units": (lambda h: char(_fig_units(h)), _fig_units_set),
    "Position": (_fig_pos_get, _fig_pos_set),
    "OuterPosition": (_fig_pos_get, _fig_pos_set),
    "InnerPosition": (_fig_pos_get, _fig_pos_set),
    "CurrentAxes": (P._fig_current_axes, _fig_current_axes_set),
    "CurrentObject": _state("CurrentObject"),
    "CurrentPoint": _vector("CurrentPoint", [0, 0], 2),
    "CurrentCharacter": _state("CurrentCharacter", ""),
    "SelectionType": _choice("SelectionType", ("normal", "extend", "alt", "open"), "normal"),
    "FileName": _state("FileName", ""),
    "NumberTitle": _onoff_state("NumberTitle"),
    "IntegerHandle": _onoff_state("IntegerHandle"),
    "WindowState": _choice("WindowState", ("normal", "minimized", "maximized", "fullscreen"), "normal"),
    "WindowStyle": _choice("WindowStyle", ("normal", "modal", "docked", "alwaysontop"), "normal"),
    "MenuBar": _choice("MenuBar", ("figure", "none"), "figure"),
    "ToolBar": _choice("ToolBar", ("auto", "figure", "none"), "auto"),
    "DockControls": _onoff_state("DockControls"),
    "Resize": _onoff_state("Resize"),
    "NextPlot": _choice("NextPlot", ("add", "new", "replace", "replacechildren"), "add"),
    "Pointer": _state("Pointer", "arrow"),
    "Renderer": _choice("Renderer", ("opengl", "painters"), "opengl"),
    "RendererMode": _choice("RendererMode", ("auto", "manual"), "auto"),
    "GraphicsSmoothing": _onoff_state("GraphicsSmoothing"),
    "InvertHardcopy": _onoff_state("InvertHardcopy"),
    "Theme": _state("Theme"),
    "Icon": _state("Icon", ""),
    "Scrollable": _onoff_state("Scrollable", "off"),
    "AutoResizeChildren": _onoff_state("AutoResizeChildren"),
    "Alphamap": (_fig_alpha_get, None),
    "PaperUnits": _choice("PaperUnits", ("inches", "centimeters", "points", "normalized"), "inches"),
    "PaperPosition": (_paper_pos_get, _paper_pos_set),
    "PaperPositionMode": _choice("PaperPositionMode", ("auto", "manual"), "auto"),
    "PaperSize": (_paper_size_get, _paper_size_set),
    "PaperType": (lambda h: char(h.extra.get("PaperType", "usletter")), _paper_type_set),
    "PaperOrientation": _choice("PaperOrientation", ("portrait", "landscape"), "portrait"),
    "CloseRequestFcn": _callback("CloseRequestFcn", "closereq"),
    "SizeChangedFcn": _callback("SizeChangedFcn"),
    "ResizeFcn": _callback("SizeChangedFcn"),
    "KeyPressFcn": _callback("KeyPressFcn"),
    "KeyReleaseFcn": _callback("KeyReleaseFcn"),
    "WindowButtonDownFcn": _callback("WindowButtonDownFcn"),
    "WindowButtonMotionFcn": _callback("WindowButtonMotionFcn"),
    "WindowButtonUpFcn": _callback("WindowButtonUpFcn"),
    "WindowKeyPressFcn": _callback("WindowKeyPressFcn"),
    "WindowKeyReleaseFcn": _callback("WindowKeyReleaseFcn"),
    "WindowScrollWheelFcn": _callback("WindowScrollWheelFcn"),
})
P._FIGURE_IGNORED.clear()


# -- text -----------------------------------------------------------------------------------------------------------

def _text_bbox(h) -> dict:
    patch = h.obj.get_bbox_patch()
    if patch is None:
        return {"facecolor": "none", "edgecolor": "none", "linewidth": 0.5, "linestyle": "-", "pad": 3.0}
    return {"facecolor": patch.get_facecolor(), "edgecolor": patch.get_edgecolor(),
            "linewidth": patch.get_linewidth(), "linestyle": patch.get_linestyle(),
            "pad": float(h.extra.get("_pad", 3.0))}


def _text_box_set(key: str):
    def s(h, v):
        box = _text_bbox(h)
        if key in ("facecolor", "edgecolor"):
            box[key] = P._color(v)
        elif key == "linestyle":
            ls = text(v)
            box[key] = "None" if ls.lower() == "none" else ls
        else:
            box[key] = P._scalar(v)
        pad = box.pop("pad")
        if key == "pad":
            h.extra["_pad"] = pad = P._scalar(v)
        h.obj.set_bbox({**box, "boxstyle": f"square,pad={pad / max(h.obj.get_fontsize(), 1):.3f}"})
    return s


def _text_box_get(key: str):
    def g(h):
        box = _text_bbox(h)
        if key in ("facecolor", "edgecolor"):
            return P._rgb(box[key])
        if key == "linestyle":
            ls = box[key]
            return char({"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}.get(ls, ls)
                        if isinstance(ls, str) else "-")
        return P._num_val(box[key])
    return g


def _text_units_set(h, v):
    u = text(v).lower()
    if u not in ("data", "normalized", "pixels", "inches", "centimeters", "points", "characters"):
        raise MatlabError("Units of a text object must be 'data', 'normalized', 'pixels', 'inches', "
                          "'centimeters', 'points' or 'characters'.")
    ax = h.obj.axes
    if ax is None:
        raise MatlabError("This text is not inside an axes; its position is in figure units.")
    old = str(h.extra.get("Units", "data"))
    if u == old:
        return
    x, y = h.obj.get_position()
    disp = _text_transform(h, old).transform((x, y))
    tr = _text_transform(h, u)
    nx, ny = tr.inverted().transform(disp)
    h.obj.set_transform(tr)
    h.obj.set_position((float(nx), float(ny)))
    h.extra["Units"] = u


def _text_transform(h, units: str):
    from matplotlib.transforms import Affine2D
    ax = h.obj.axes
    if units == "data":
        return ax.transData
    if units == "normalized":
        return ax.transAxes
    fig = ax.figure
    k = {"pixels": 1.0, "inches": fig.dpi, "centimeters": fig.dpi / 2.54, "points": fig.dpi / 72.0}.get(units)
    sx, sy = (k, k) if k is not None else _CHAR_PX
    bb = ax.get_window_extent()
    return Affine2D().scale(sx, sy).translate(bb.x0, bb.y0)


def _text_extent(h):
    t = h.obj
    fig = t.figure
    try:
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
    except (ValueError, RuntimeError, AttributeError):
        return P._row([0, 0, 0, 0])
    tr = _text_transform(h, str(h.extra.get("Units", "data"))) if t.axes is not None else fig.transFigure
    (x0, y0), (x1, y1) = tr.inverted().transform((bb.x0, bb.y0)), tr.inverted().transform((bb.x1, bb.y1))
    return P._row([min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)])


def _text_position_get(h):
    pos = h.obj.get_position()
    z = getattr(h.obj, "_z", 0.0)
    return P._row([pos[0], pos[1], float(z) if np.isscalar(z) else 0.0])


def _text_position_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size < 2:
        raise MatlabError("Position must be [x y] or [x y z].")
    h.obj.set_position((float(p[0]), float(p[1])))
    if p.size > 2 and hasattr(h.obj, "set_z"):
        h.obj.set_z(float(p[2]))


def _interpreter_set(h, v):
    mode = text(v).lower()
    if mode not in ("tex", "latex", "none"):
        raise MatlabError("Interpreter must be 'tex', 'latex' or 'none'.")
    raw = getattr(h.obj, "_mylab_raw", None)
    if raw is None:
        raw = h.obj.get_text()
    P.set_text(h.interp, h.obj, raw, mode)


P._TEXT_PROPS.update({
    "Position": (_text_position_get, _text_position_set),
    "Units": (lambda h: char(h.extra.get("Units", "data")), _text_units_set),
    "Extent": (_text_extent, None),
    "BackgroundColor": (_text_box_get("facecolor"), _text_box_set("facecolor")),
    "EdgeColor": (_text_box_get("edgecolor"), _text_box_set("edgecolor")),
    "LineWidth": (_text_box_get("linewidth"), _text_box_set("linewidth")),
    "LineStyle": (_text_box_get("linestyle"), _text_box_set("linestyle")),
    "Margin": (_text_box_get("pad"), _text_box_set("pad")),
    "FontUnits": _choice("FontUnits", ("points", "inches", "centimeters", "normalized", "pixels"), "points"),
    "FontSmoothing": _onoff_state("FontSmoothing"),
    "Editing": _onoff_state("Editing", "off"),
})


# -- TeX markup in labels ---------------------------------------------------------------------------------------------

_TEX_NAMES = set("""alpha beta gamma delta epsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma
varsigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega forall exists ni
cong approx Re Im otimes oplus oslash cap cup supseteq supset subseteq subset in o int rfloor lceil nabla lfloor cdot
ldots perp neg prime wedge times vee varpi rceil surd mid copyright langle rangle sim leq infty clubsuit diamondsuit
heartsuit spadesuit leftrightarrow leftarrow Leftarrow uparrow rightarrow Rightarrow downarrow circ pm geq propto
partial bullet div neq equiv aleph wp oplus angle ast star""".split())
_TEX_ALIAS = {"0": r"\emptyset", "o": r"\omicron"}


def _mathtext_ok(s: str) -> bool:
    try:
        from matplotlib.mathtext import MathTextParser
        MathTextParser("path").parse(s)
        return True
    except Exception:  # noqa: BLE001  (any parse failure means: show the text as typed)
        return False


def tex_to_mathtext(s: str) -> str:
    """Translate MATLAB's TeX-style markup (\\alpha, x^2, a_{ij}, \\bf) to matplotlib's math text.

    Plain parts stay plain text; each piece of markup becomes a small ``$...$`` group. Text that does not
    parse as markup is returned unchanged.
    """
    if not any(ch in s for ch in "\\^_{}"):
        return s.replace("$", r"\$")
    out: list[str] = []
    i, n = 0, len(s)
    style = ""  # active font command: '', 'bf', 'it', 'rm'

    def plain(t: str) -> str:
        t = t.replace("$", r"\$")
        if not t or not style or style == "rm":
            return t
        body = t.replace("\\", r"\backslash ").replace(" ", r"\ ").replace("#", r"\#").replace("%", r"\%")
        return "$\\math" + style + "{" + body + "}$"

    def group(j: int) -> tuple[str, int]:
        """The argument starting at s[j]: a {braced group} or one character/command."""
        if j < n and s[j] == "{":
            depth, k = 1, j + 1
            while k < n and depth:
                depth += s[k] == "{"
                depth -= s[k] == "}"
                k += 1
            return s[j + 1:k - 1], k
        if j < n and s[j] == "\\":
            m = re.match(r"\\[A-Za-z]+", s[j:])
            if m:
                return m.group(0), j + m.end()
        return (s[j], j + 1) if j < n else ("", j)

    def inner(t: str) -> str:
        """Markup inside a sub/superscript: commands stay, letters are upright."""
        parts = re.split(r"(\\[A-Za-z]+|[\^_{}])", t)
        res = []
        for p in parts:
            if not p:
                continue
            if p.startswith("\\") or p in "^_{}":
                name = p[1:] if p.startswith("\\") else ""
                res.append(p if not name or name in _TEX_NAMES else "")
            else:
                res.append("\\mathrm{" + p.replace(" ", r"\ ").replace("#", r"\#").replace("%", r"\%") + "}")
        return "".join(res)

    buf = ""
    while i < n:
        ch = s[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                name = m.group(1)
                if name in ("bf", "it", "rm", "sl"):
                    out.append(plain(buf))
                    buf = ""
                    style = {"sl": "it"}.get(name, name)
                    i += m.end()
                    if i < n and s[i] == " ":
                        i += 1
                    continue
                if name in ("fontname", "fontsize", "color"):
                    j = i + m.end()
                    if j < n and s[j] == "[":  # \color[rgb]{...}
                        j = s.find("]", j) + 1 or j
                    _, j = group(j)
                    i = j
                    continue
                if name in _TEX_NAMES or name in _TEX_ALIAS:
                    out.append(plain(buf))
                    buf = ""
                    out.append("$" + _TEX_ALIAS.get(name, "\\" + name) + "$")
                    i += m.end()
                    continue
                buf += s[i:i + m.end()]
                i += m.end()
                continue
            if i + 1 < n and s[i + 1] in "\\{}^_":
                buf += s[i + 1]
                i += 2
                continue
            buf += ch
            i += 1
        elif ch in "^_":
            arg, j = group(i + 1)
            if not arg:
                buf += ch
                i += 1
                continue
            out.append(plain(buf))
            buf = ""
            out.append("$" + ch + "{" + inner(arg) + "}$")
            i = j
        elif ch == "{":
            arg, j = group(i)
            saved = style
            out.append(plain(buf))
            buf = ""
            out.append(tex_to_mathtext_keep(arg, style))
            style = saved
            i = j
        elif ch == "}":
            i += 1
        else:
            buf += ch
            i += 1
    out.append(plain(buf))
    res = "".join(out).replace("$$", "")
    return res if _mathtext_ok(res) else s.replace("$", r"\$")


def tex_to_mathtext_keep(arg: str, style: str) -> str:
    """A braced group: the font command of the surrounding text applies inside, changes stay inside."""
    if style and style != "rm" and not re.match(r"\s*\\(bf|it|rm|sl)\b", arg):
        arg = "\\" + style + " " + arg
    res = tex_to_mathtext(arg)
    return res


def render_text(s: str, interpreter: str = "tex") -> str:
    """The string matplotlib should draw for label text ``s`` under a MATLAB text interpreter."""
    mode = (interpreter or "tex").lower()
    if mode == "none":
        return s.replace("$", r"\$")
    if mode == "latex":
        if "$" in s and _mathtext_ok(s):
            return s
        return s.replace("$", r"\$")
    return "\n".join(tex_to_mathtext(line) for line in s.split("\n"))


def default_interpreter(interp, kind: str = "text") -> str:
    d = interp.state.get("gfx_defaults", {})
    key = {"text": "defaulttextinterpreter", "legend": "defaultlegendinterpreter",
           "ticks": "defaultaxesticklabelinterpreter", "colorbar": "defaultcolorbarticklabelinterpreter"}[kind]
    v = d.get(key)
    return text(v).lower() if v is not None and is_text(v) else "tex"


def _render_hook(interp, s: str, interpreter: str | None, kind: str) -> str:
    return render_text(s, interpreter or default_interpreter(interp, kind))


def _string_set(h, v):
    P.set_text(h.interp, h.obj, P._label_text(v))
    sub = getattr(h.obj, "_mylab_is_subtitle_of", None)
    if sub is not None:
        P._layout_subtitle(sub)


def _string_get(h):
    raw = getattr(h.obj, "_mylab_raw", None)
    if raw is None:
        raw = h.obj.get_text()
    if "\n" in raw:
        return P._cell_column([char(x) for x in raw.split("\n")])
    return char(raw)


def _interpreter_get(h):
    return char(getattr(h.obj, "_mylab_interp", None) or default_interpreter(h.interp, "text"))


P._TEXT_RENDER = _render_hook
P._TEXT_PROPS["String"] = (_string_get, _string_set)
P._TEXT_PROPS["Interpreter"] = (_interpreter_get, _interpreter_set)


# -- legends -------------------------------------------------------------------------------------------------------

def _legend_frame_set(method: str):
    def s(h, v):
        getattr(h.obj.get_frame(), method)(P._color(v))
    return s


def _legend_position_get(h):
    fig = h.obj.figure
    try:
        fig.canvas.draw()
        bb = h.obj.get_window_extent(fig.canvas.get_renderer())
    except (ValueError, RuntimeError, AttributeError):
        return P._row([0, 0, 0, 0])
    inv = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    frac = [x0, y0, x1 - x0, y1 - y0]
    units = str(h.extra.get("Units", "normalized"))
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(units, w, hh, P._fig_dpi(fig))
    return P._row([frac[0] * w / sx, frac[1] * hh / sy, frac[2] * w / sx, frac[3] * hh / sy])


def _legend_position_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size != 4:
        raise MatlabError("Position must be [left bottom width height].")
    fig = h.obj.figure
    units = str(h.extra.get("Units", "normalized"))
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(units, w, hh, P._fig_dpi(fig))
    h.obj.set_bbox_to_anchor((p[0] * sx / w, p[1] * sy / hh), transform=fig.transFigure)
    h.obj.set_loc("lower left")
    h.obj._mylab_loc = "none"


def _legend_font(method: str, conv=lambda v: text(v)):
    def s(h, v):
        for t in h.obj.get_texts():
            getattr(t, method)(conv(v))
    return s


def _legend_direction_apply(h, v):
    leg = h.obj
    P._make_legend(h.interp, leg.axes, reuse=leg, reverse=(v == "reverse"))


def _legend_interpreter_apply(h, v):
    leg = h.obj
    leg._mylab_interpreter = v
    P._make_legend(h.interp, leg.axes, reuse=leg)


def _legend_alpha_apply(h, v):
    h.obj.get_frame().set_alpha(float(v.flat[0]))


def _legend_token_apply(h, v):
    leg = h.obj
    leg._mylab_token = float(v.flat[0])
    P._make_legend(h.interp, leg.axes, reuse=leg)


P._LEGEND_PROPS.update({
    "EdgeColor": (lambda h: P._rgb(h.obj.get_frame().get_edgecolor()), _legend_frame_set("set_edgecolor")),
    "Color": (lambda h: P._rgb(h.obj.get_frame().get_facecolor()), _legend_frame_set("set_facecolor")),
    "LineWidth": (lambda h: P._num_val(h.obj.get_frame().get_linewidth()),
                  lambda h, v: h.obj.get_frame().set_linewidth(P._scalar(v))),
    "FontName": (lambda h: char(h.obj.get_texts()[0].get_fontname() if h.obj.get_texts() else "Helvetica"),
                 _legend_font("set_fontfamily")),
    "FontWeight": (lambda h: char("bold" if h.obj.get_texts() and str(h.obj.get_texts()[0].get_fontweight())
                                  in ("bold", "700") else "normal"),
                   _legend_font("set_fontweight", lambda v: text(v).lower())),
    "FontAngle": (lambda h: char("italic" if h.obj.get_texts() and h.obj.get_texts()[0].get_fontstyle() == "italic"
                                 else "normal"),
                  _legend_font("set_fontstyle", lambda v: "italic" if text(v).lower() == "italic" else "normal")),
    "Position": (_legend_position_get, _legend_position_set),
    "Units": _choice("Units", ("normalized", "pixels", "inches", "centimeters", "points", "characters"),
                     "normalized"),
    "Interpreter": _choice("Interpreter", ("tex", "latex", "none"), "tex", _legend_interpreter_apply),
    "AutoUpdate": _onoff_state("AutoUpdate"),
    "Direction": _choice("Direction", ("normal", "reverse"), "normal", _legend_direction_apply),
    "BackgroundAlpha": _number("BackgroundAlpha", 1.0, 0, 1, _legend_alpha_apply),
    "ItemTokenSize": _vector("ItemTokenSize", [30, 18], 2, _legend_token_apply),
    "IconColumnWidth": _number("IconColumnWidth", 30, 0, 1000, _legend_token_apply),
    "ItemHitFcn": _callback("ItemHitFcn"),
    "NumColumnsMode": _choice("NumColumnsMode", ("auto", "manual"), "auto"),
})
P._LEGEND_IGNORED.clear()


# -- colorbars -----------------------------------------------------------------------------------------------------

def _cb_axis(h):
    cb = h.obj
    return cb.ax.xaxis if cb.orientation == "horizontal" else cb.ax.yaxis


def _cb_location_set(h, v):
    loc = text(v).lower()
    valid = ("north", "south", "east", "west", "northoutside", "southoutside", "eastoutside", "westoutside",
             "manual", "layout")
    if loc not in valid:
        raise MatlabError(f"Location must be one of: {', '.join(valid)}.")
    if loc in ("manual", "layout"):
        h.extra["Location"] = loc
        return
    ax = getattr(h.obj, "_mylab_parent_ax", None)
    if ax is None:
        raise MatlabError("This colorbar is not attached to an axes.")
    label = _cb_axis(h).label.get_text()
    new = BUILTINS["colorbar"](h.interp, [P._wrap(h.interp, P._axes_kind(ax), ax), char(loc)], 1)
    st = P._state(h.interp)
    st.handles.pop(id(new.obj), None)
    h.obj = new.obj
    st.handles[id(h.obj)] = h
    h.extra["Location"] = loc
    if label:
        h.obj.set_label(label)


def _cb_direction_get(h):
    axis = _cb_axis(h)
    return char("reverse" if axis.get_inverted() else "normal")


def _cb_direction_set(h, v):
    want = text(v).lower() == "reverse"
    axis = _cb_axis(h)
    if axis.get_inverted() != want:
        axis.set_inverted(want)


def _cb_position_get(h):
    return _frac_to_units_fig(h, h.obj.ax.get_position().bounds)


def _frac_to_units_fig(h, frac) -> np.ndarray:
    fig = P._obj_figure(h)
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(str(h.extra.get("Units", "normalized")), w, hh, P._fig_dpi(fig))
    return P._row([frac[0] * w / sx, frac[1] * hh / sy, frac[2] * w / sx, frac[3] * hh / sy])


def _cb_position_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size != 4 or p[2] <= 0 or p[3] <= 0:
        raise MatlabError("Position must be [left bottom width height] with a positive width and height.")
    fig = P._obj_figure(h)
    w, hh = _fig_pixels(fig)
    sx, sy = _unit_scale(str(h.extra.get("Units", "normalized")), w, hh, P._fig_dpi(fig))
    try:
        fig.set_layout_engine("none")
    except (AttributeError, ValueError, RuntimeError):
        pass
    h.obj.ax.set_position([p[0] * sx / w, p[1] * sy / hh, p[2] * sx / w, p[3] * sy / hh])
    h.extra["Location"] = "manual"


def _cb_color_set(h, v):
    c = P._color(v)
    h.obj.ax.tick_params(colors=c)
    h.obj.outline.set_edgecolor(c)
    _cb_axis(h).label.set_color(c)
    h.extra["Color"] = P._rgb(c)


def _cb_tickdir_apply(h, v):
    h.obj.ax.tick_params(direction={"both": "inout", "none": "in"}.get(v, v), length=0 if v == "none" else 3.5)


def _cb_axisloc_set(h, v):
    loc = text(v).lower()
    if loc not in ("in", "out"):
        raise MatlabError("AxisLocation must be 'in' or 'out'.")
    cb = h.obj
    side = str(getattr(cb, "ticklocation", "right"))
    flip = {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}
    base = h.extra.setdefault("_side", side)
    target = base if loc == "out" else flip.get(base, base)
    axis = _cb_axis(h)
    axis.set_ticks_position(target)
    axis.set_label_position(target)
    h.extra["AxisLocation"] = loc


def _cb_font(method: str):
    def s(h, v):
        val = text(v) if method != "set_fontweight" else text(v).lower()
        axis = _cb_axis(h)
        for t in axis.get_ticklabels() + [axis.label]:
            getattr(t, method)(val)
        h.extra[{"set_fontfamily": "FontName", "set_fontweight": "FontWeight"}[method]] = val
    return s


def _cb_limits_get(h):
    axis_lim = h.obj.ax.get_ylim() if h.obj.orientation == "vertical" else h.obj.ax.get_xlim()
    return P._row(sorted(axis_lim))


def _cb_limits_set(h, v):
    lim = np.real(fnum(v)).flatten()
    if lim.size != 2 or not lim[0] < lim[1]:
        raise MatlabError("Limits must be [min max] with min < max.")
    inverted = _cb_axis(h).get_inverted()
    if h.obj.orientation == "vertical":
        h.obj.ax.set_ylim(lim[0], lim[1])
    else:
        h.obj.ax.set_xlim(lim[0], lim[1])
    if inverted:
        _cb_axis(h).set_inverted(True)
    h.extra["LimitsMode"] = "manual"


def _cb_title(h):
    return P._wrap(h.interp, "text", h.obj.ax.title)


P._COLORBAR_PROPS.update({
    "Location": (lambda h: char(h.extra.get("Location", "eastoutside")), _cb_location_set),
    "Orientation": (lambda h: char(h.obj.orientation), None),
    "Direction": (_cb_direction_get, _cb_direction_set),
    "Limits": (_cb_limits_get, _cb_limits_set),
    "LimitsMode": _choice("LimitsMode", ("auto", "manual"), "auto"),
    "TicksMode": _choice("TicksMode", ("auto", "manual"), "auto"),
    "TickLabelsMode": _choice("TickLabelsMode", ("auto", "manual"), "auto"),
    "Position": (_cb_position_get, _cb_position_set),
    "Units": _choice("Units", _UNITS, "normalized"),
    "Color": (lambda h: h.extra.get("Color", P._row([0.15, 0.15, 0.15])), _cb_color_set),
    "Box": (lambda h: P._oo(h.obj.outline.get_visible()), lambda h, v: h.obj.outline.set_visible(P._onoff(v))),
    "LineWidth": (lambda h: P._num_val(h.obj.outline.get_linewidth()),
                  lambda h, v: (h.obj.outline.set_linewidth(P._scalar(v)),
                                h.obj.ax.tick_params(width=P._scalar(v)))),
    "TickDirection": _choice("TickDirection", ("in", "out", "both", "none"), "in", _cb_tickdir_apply),
    "TickLength": _number("TickLength", 0.01, 0, 1,
                          lambda h, v: h.obj.ax.tick_params(length=float(v.flat[0]) * 350)),
    "AxisLocation": (lambda h: char(h.extra.get("AxisLocation", "out")), _cb_axisloc_set),
    "FontName": (lambda h: char(h.extra.get("FontName", "Helvetica")), _cb_font("set_fontfamily")),
    "FontWeight": (lambda h: char(h.extra.get("FontWeight", "normal")), _cb_font("set_fontweight")),
    "TickLabelInterpreter": _choice("TickLabelInterpreter", ("tex", "latex", "none"), "tex"),
    "Title": (_cb_title, None),
})


# -- images, surfaces, contours, quivers, stems, error bars, areas ----------------------------------------------------

def _image_extent(h):
    ext = h.obj.get_extent()
    arr = np.asarray(h.obj.get_array())
    nr, nc = arr.shape[:2]
    x0, x1 = sorted(ext[:2])
    y0, y1 = sorted(ext[2:])
    dx = (x1 - x0) / (2 * nc) if nc else 0.5
    dy = (y1 - y0) / (2 * nr) if nr else 0.5
    return x0 + dx, x1 - dx, y0 + dy, y1 - dy, nr, nc


def _image_xy_get(which: str):
    def g(h):
        x0, x1, y0, y1, nr, nc = _image_extent(h)
        return P._row([x0, x1] if which == "x" else [y0, y1])
    return g


def _image_xy_set(which: str):
    def s(h, v):
        d = np.real(fnum(v)).flatten()
        if d.size < 1:
            raise MatlabError(f"{which.upper()}Data must have at least one value.")
        x0, x1, y0, y1, nr, nc = _image_extent(h)
        lo, hi = float(d[0]), float(d[-1])
        if which == "x":
            x0, x1 = (lo, hi) if d.size > 1 else (lo, lo + nc - 1)
        else:
            y0, y1 = (lo, hi) if d.size > 1 else (lo, lo + nr - 1)
        dx = (x1 - x0) / max(nc - 1, 1) / 2 if nc > 1 else 0.5
        dy = (y1 - y0) / max(nr - 1, 1) / 2 if nr > 1 else 0.5
        flipped = h.obj.origin == "upper"
        h.obj.set_extent((x0 - dx, x1 + dx, y1 + dy, y0 - dy) if flipped else (x0 - dx, x1 + dx, y0 - dy, y1 + dy))
    return s


def _image_alpha_get(h):
    a = h.obj.get_alpha()
    if a is None:
        return P._num_val(1)
    return np.asarray(a, dtype=float) if np.ndim(a) else P._num_val(a)


def _image_alpha_set(h, v):
    a = np.real(fnum(v))
    if a.size == 1:
        h.obj.set_alpha(float(a.flat[0]))
        return
    if a.shape != np.asarray(h.obj.get_array()).shape[:2]:
        raise MatlabError("AlphaData must be a number or a matrix the size of the image.")
    h.obj.set_alpha(np.clip(a, 0, 1))


def _image_interp_apply(h, v):
    h.obj.set_interpolation(v)


def _image_mapping_apply(h, v):
    arr = np.asarray(h.obj.get_array(), dtype=float)
    if arr.ndim != 2:
        return
    if v == "direct":
        cmap = h.obj.get_cmap()
        h.obj.set_clim(1, getattr(cmap, "N", 256))
    else:
        h.obj.autoscale()


P._IMAGE_PROPS.update({
    "XData": (_image_xy_get("x"), _image_xy_set("x")),
    "YData": (_image_xy_get("y"), _image_xy_set("y")),
    "AlphaData": (_image_alpha_get, _image_alpha_set),
    "AlphaDataMapping": _choice("AlphaDataMapping", ("none", "scaled", "direct"), "none"),
    "CDataMapping": _choice("CDataMapping", ("scaled", "direct"), "scaled", _image_mapping_apply),
    "Interpolation": _choice("Interpolation", ("nearest", "bilinear"), "nearest", _image_interp_apply),
})

_LIGHTING = {
    "FaceLighting": _choice("FaceLighting", ("flat", "gouraud", "none"), "flat"),
    "EdgeLighting": _choice("EdgeLighting", ("none", "flat", "gouraud"), "none"),
    "BackFaceLighting": _choice("BackFaceLighting", ("reverselit", "unlit", "lit"), "reverselit"),
    "AmbientStrength": _number("AmbientStrength", 0.3, 0, 1),
    "DiffuseStrength": _number("DiffuseStrength", 0.6, 0, 1),
    "SpecularStrength": _number("SpecularStrength", 0.9, 0, 1),
    "SpecularExponent": _number("SpecularExponent", 10, 0.01, 1e6),
    "SpecularColorReflectance": _number("SpecularColorReflectance", 1, 0, 1),
}


def _coll_linestyle_get(h):
    return char(h.extra.get("LineStyle", "-"))


def _coll_linestyle_set(h, v):
    s = text(v)
    for a in P._artists(h.obj):
        if hasattr(a, "set_linestyle"):
            a.set_linestyle("None" if s.lower() == "none" else s)
    h.extra["LineStyle"] = s


def _surface_data_set(which: str):
    def s(h, v):
        data = np.real(fnum(v))
        ref = h.extra.get("ZData")
        if which in ("XData", "YData") and ref is not None and data.shape != np.asarray(ref).shape:
            if data.size == np.asarray(ref).shape[1 if which == "XData" else 0]:
                Xg, Yg = np.meshgrid(data.flatten() if which == "XData" else np.asarray(h.extra["XData"])[0, :],
                                     np.asarray(h.extra["YData"])[:, 0] if which == "XData" else data.flatten())
                data = Xg if which == "XData" else Yg
            else:
                raise MatlabError(f"{which} must match the size of ZData.")
        if which == "ZData" and ref is not None and data.shape != np.asarray(ref).shape:
            raise MatlabError("ZData must keep its size; draw the surface again to change it.")
        h.extra[which] = data
        redraw_surface(h)
    return s


def redraw_surface(h) -> None:
    """Rebuild a surface or mesh from the data kept on its handle (after XData/YData/ZData/CData changed)."""
    old = h.obj
    ax = P._obj_axes(h)
    if ax is None or not hasattr(ax, "plot_surface"):
        raise MatlabError("This surface cannot be changed after it is drawn; draw it again with new data.")
    X, Y, Z = (np.asarray(h.extra[k], dtype=float) for k in ("XData", "YData", "ZData"))
    C = h.extra.get("CData")
    wire = old.__class__.__name__ == "Line3DCollection"
    if wire:
        new = ax.plot_wireframe(X, Y, Z, linewidth=np.asarray(old.get_linewidth()).flat[0],
                                colors=old.get_edgecolor())
    else:
        kw = {"edgecolor": old.get_edgecolor() if len(old.get_edgecolor()) else "none",
              "linewidth": np.asarray(old.get_linewidth()).flat[0] if len(np.atleast_1d(old.get_linewidth())) else 0,
              "rcount": min(Z.shape[0], 200), "ccount": min(Z.shape[1], 200), "alpha": old.get_alpha()}
        cmap = old.get_cmap()
        if C is not None and np.asarray(C).shape == Z.shape:
            import matplotlib as mpl
            Cn = np.asarray(C, dtype=float)
            lo, hi = float(np.nanmin(Cn)), float(np.nanmax(Cn))
            cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
            kw["facecolors"] = cm((Cn - lo) / (hi - lo) if hi > lo else np.zeros_like(Cn))
            kw["shade"] = False
        else:
            kw["cmap"] = cmap
        new = ax.plot_surface(X, Y, Z, **kw)
        new._mylab_surface = True
    try:
        old.remove()
    except (ValueError, NotImplementedError):
        pass
    st = P._state(h.interp)
    st.handles.pop(id(old), None)
    h.obj = new
    st.handles[id(new)] = h


def _surface_normals(h):
    X, Y, Z = (np.asarray(h.extra.get(k, _EMPTY), dtype=float) for k in ("XData", "YData", "ZData"))
    if Z.size == 0 or X.shape != Z.shape:
        return _EMPTY
    return surface_normals(X, Y, Z)


def surface_normals(X, Y, Z) -> np.ndarray:
    """Unit normals of a gridded surface, an m-by-n-by-3 array (pointing to +z for a flat sheet)."""
    def grad(A):
        g0, g1 = (np.gradient(A, axis=0) if A.shape[0] > 1 else np.zeros_like(A)), \
                 (np.gradient(A, axis=1) if A.shape[1] > 1 else np.zeros_like(A))
        return g0, g1
    xr, xc = grad(X)
    yr, yc = grad(Y)
    zr, zc = grad(Z)
    n = np.cross(np.dstack([xc, yc, zc]), np.dstack([xr, yr, zr]))
    length = np.linalg.norm(n, axis=2, keepdims=True)
    length[length == 0] = 1.0
    return n / length


def _mesh_style_apply(h, v):
    if h.obj.__class__.__name__ != "Line3DCollection":
        return
    X, Y, Z = (np.asarray(h.extra[k], dtype=float) for k in ("XData", "YData", "ZData"))
    ax = P._obj_axes(h)
    old = h.obj
    kw = {"linewidth": np.asarray(old.get_linewidth()).flat[0], "colors": old.get_edgecolor()}
    if v == "row":
        kw["cstride"] = 0
    elif v == "column":
        kw["rstride"] = 0
    new = ax.plot_wireframe(X, Y, Z, **kw)
    old.remove()
    st = P._state(h.interp)
    st.handles.pop(id(old), None)
    h.obj = new
    st.handles[id(new)] = h


P._SURFACE_PROPS.update({
    "XData": (lambda h: h.extra.get("XData", _EMPTY), _surface_data_set("XData")),
    "YData": (lambda h: h.extra.get("YData", _EMPTY), _surface_data_set("YData")),
    "ZData": (lambda h: h.extra.get("ZData", _EMPTY), _surface_data_set("ZData")),
    "CData": (lambda h: h.extra.get("CData", h.extra.get("ZData", _EMPTY)), _surface_data_set("CData")),
    "CDataMapping": _choice("CDataMapping", ("scaled", "direct"), "scaled"),
    "LineStyle": (_coll_linestyle_get, _coll_linestyle_set),
    "EdgeAlpha": (lambda h: h.extra.get("EdgeAlpha", P._num_val(1)), _edge_alpha_set),
    "MeshStyle": _choice("MeshStyle", ("both", "row", "column"), "both", _mesh_style_apply),
    "VertexNormals": (_surface_normals, None),
    "FaceNormals": (_surface_normals, None),
    "AlphaData": _state("AlphaData", P._num_val(1)),
    "DisplayName": (lambda h: P._get_label(h.obj), P._set_label),
    **_LIGHTING,
})


def _patch_geom(key: str):
    def g(h):
        v = h.extra.get(key)
        if v is not None:
            return v
        polys = h.extra.get("_polys")
        if polys is None:
            return _EMPTY
        if key == "Vertices":
            return np.vstack(polys)
        if key == "Faces":
            n = max(len(p) for p in polys)
            out = np.full((len(polys), n), np.nan)
            k = 1
            for i, p in enumerate(polys):
                out[i, :len(p)] = np.arange(k, k + len(p))
                k += len(p)
            return out
        col = {"XData": 0, "YData": 1, "ZData": 2}[key]
        if polys[0].shape[1] <= col:
            return _EMPTY
        n = max(len(p) for p in polys)
        out = np.full((n, len(polys)), np.nan)
        for j, p in enumerate(polys):
            out[:len(p), j] = p[:, col]
            out[len(p):, j] = p[-1, col]
        return out
    return g


def _patch_geom_set(key: str):
    def s(h, v):
        coll = h.obj
        if not hasattr(coll, "set_verts"):
            raise MatlabError(f"{key} of this object cannot be changed; draw it again with new data.")
        data = np.real(fnum(v))
        polys = h.extra.get("_polys")
        if key in ("Vertices", "Faces"):
            V = data if key == "Vertices" else np.asarray(P._get_prop(h, "Vertices"), dtype=float)
            F = data if key == "Faces" else np.asarray(P._get_prop(h, "Faces"), dtype=float)
            new = []
            for row in np.atleast_2d(F):
                idx = row[np.isfinite(row)].astype(int) - 1
                if idx.size and (idx.min() < 0 or idx.max() >= V.shape[0]):
                    raise MatlabError("Faces contains an index outside the rows of Vertices.")
                if idx.size:
                    new.append(V[idx])
            h.extra["Vertices"], h.extra["Faces"] = V, F
        else:
            col = {"XData": 0, "YData": 1, "ZData": 2}[key]
            if polys is None:
                raise MatlabError(f"{key} of this object cannot be changed; draw it again with new data.")
            data = data.reshape(-1, 1) if data.ndim == 2 and 1 in data.shape and len(polys) == 1 else data
            if data.shape[1] != len(polys):
                raise MatlabError(f"{key} must keep one column for each face.")
            new = []
            for j, p in enumerate(polys):
                q = np.array(p[:data.shape[0]], dtype=float) if len(p) >= data.shape[0] else \
                    np.vstack([p, np.repeat(p[-1:], data.shape[0] - len(p), axis=0)])
                q[:, col] = data[:, j]
                new.append(q)
            h.extra.pop("Vertices", None)
            h.extra.pop("Faces", None)
        h.extra["_polys"] = new
        coll.set_verts([np.asarray(p) for p in new])
        ax = P._obj_axes(h)
        if ax is not None:
            pts = np.vstack(new)
            if pts.shape[1] == 3 and hasattr(ax, "auto_scale_xyz"):
                ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])
            elif pts.shape[1] >= 2:
                ax.update_datalim(pts[:, :2])
                ax.autoscale_view()
    return s


def _patch_cdata_get(h):
    arr = h.obj.get_array() if hasattr(h.obj, "get_array") else None
    if arr is None:
        return _EMPTY
    return np.asarray(arr, dtype=float).reshape(-1, 1)


def _patch_cdata_set(h, v):
    c = np.real(fnum(v))
    coll = h.obj
    if not hasattr(coll, "set_array"):
        raise MatlabError("The colour data of this object cannot be changed.")
    n = len(h.extra.get("_polys", [])) or c.size
    if c.ndim == 2 and c.shape[1] == 3 and c.shape[0] == n:
        coll.set_array(None)
        coll.set_facecolor(np.clip(c, 0, 1))
    elif c.size == n:
        coll.set_array(c.flatten())
        coll.set_cmap(P._current_cmap(h.interp))
    else:
        raise MatlabError("The colour data must have one value (or one RGB row) for each face.")


_PATCH_EXTRA = {
    "Vertices": (_patch_geom("Vertices"), _patch_geom_set("Vertices")),
    "Faces": (_patch_geom("Faces"), _patch_geom_set("Faces")),
    "XData": (_patch_geom("XData"), _patch_geom_set("XData")),
    "YData": (_patch_geom("YData"), _patch_geom_set("YData")),
    "ZData": (_patch_geom("ZData"), _patch_geom_set("ZData")),
    "FaceVertexCData": (_patch_cdata_get, _patch_cdata_set),
    "CData": (_patch_cdata_get, _patch_cdata_set),
    "CDataMapping": _choice("CDataMapping", ("scaled", "direct"), "scaled"),
    **_LIGHTING,
}
P._PATCH_PROPS.update(_PATCH_EXTRA)


def _contour_levels_set(h, v):
    lv = np.real(fnum(v)).flatten()
    if lv.size == 0:
        raise MatlabError("LevelList needs at least one level.")
    redraw_contour(h, levels=np.sort(lv))


def redraw_contour(h, **changes) -> None:
    """Draw a contour plot again from the data on its handle, with some settings changed."""
    info = h.extra.get("_contour")
    if info is None:
        raise MatlabError("This contour plot cannot be changed after it is drawn; draw it again.")
    info.update(changes)
    old = h.obj
    ax = old.axes
    kw = dict(info.get("kw", {}))
    if info.get("levels") is not None:
        kw["levels"] = info["levels"]
    fn = ax.contourf if info.get("fill") else ax.contour
    new = fn(info["X"], info["Y"], info["Z"], **kw)
    try:
        old.remove()
    except (ValueError, AttributeError, NotImplementedError):
        for c in getattr(old, "collections", []):
            c.remove()
    if info.get("showtext"):
        try:
            ax.clabel(new, inline=True, fontsize=8)
        except (ValueError, IndexError):
            pass
    ax._mylab_contour = new
    st = P._state(h.interp)
    st.handles.pop(id(old), None)
    h.obj = new
    st.handles[id(new)] = h


def _contour_prop(key: str, conv):
    def s(h, v):
        info = h.extra.get("_contour")
        if info is None:
            raise MatlabError("This contour plot cannot be changed after it is drawn; draw it again.")
        kw = dict(info.get("kw", {}))
        val = conv(v)
        if key == "colors":
            kw.pop("cmap", None)
            if val is None:
                kw.pop("colors", None)
                kw["cmap"] = P._current_cmap(h.interp)
            else:
                kw["colors"] = [val]
        else:
            kw[key] = val
        redraw_contour(h, kw=kw)
    return s


def _contour_color_conv(v):
    if is_text(v) and text(v).lower() in ("flat", "auto"):
        return None
    return P._color(v)


def _contour_get(key: str, default):
    def g(h):
        info = h.extra.get("_contour") or {}
        v = info.get(key, default)
        return char(v) if isinstance(v, str) else v
    return g


def _contour_linecolor_get(h):
    kw = (h.extra.get("_contour") or {}).get("kw", {})
    return P._rgb(kw["colors"][0]) if kw.get("colors") else char("flat")


def _contour_levelstep_set(h, v):
    step = P._scalar(v)
    if step <= 0:
        raise MatlabError("LevelStep must be positive.")
    info = h.extra.get("_contour")
    if info is None:
        raise MatlabError("This contour plot cannot be changed after it is drawn; draw it again.")
    Z = np.asarray(info["Z"], dtype=float)
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    redraw_contour(h, levels=np.arange(math.ceil(lo / step) * step, hi + step * 1e-9, step))


def _contour_levelstep_get(h):
    lv = np.asarray(h.obj.levels, dtype=float)
    return P._num_val(float(np.median(np.diff(lv))) if lv.size > 1 else 0)


P._PROPS["contour"].update({
    "LevelList": (lambda h: P._row(h.obj.levels), _contour_levels_set),
    "LevelStep": (_contour_levelstep_get, _contour_levelstep_set),
    "LevelListMode": _choice("LevelListMode", ("auto", "manual"), "auto"),
    "LineColor": (_contour_linecolor_get, _contour_prop("colors", _contour_color_conv)),
    "LineWidth": (lambda h: P._num_val((h.extra.get("_contour") or {}).get("kw", {}).get("linewidths", 0.5)),
                  _contour_prop("linewidths", P._scalar)),
    "LineStyle": (lambda h: char((h.extra.get("_contour") or {}).get("kw", {}).get("linestyles", "-")),
                  _contour_prop("linestyles", lambda v: text(v))),
    "ShowText": (lambda h: P._oo(bool((h.extra.get("_contour") or {}).get("showtext"))),
                 lambda h, v: redraw_contour(h, showtext=P._onoff(v))),
    "Fill": (lambda h: P._oo(bool((h.extra.get("_contour") or {}).get("fill"))),
             lambda h, v: redraw_contour(h, fill=P._onoff(v))),
    "XData": (_contour_get("X", _EMPTY), None),
    "YData": (_contour_get("Y", _EMPTY), None),
    "ZData": (_contour_get("Z", _EMPTY), lambda h, v: redraw_contour(h, Z=np.real(fnum(v)))),
    "ContourMatrix": (lambda h: P._contour_matrix(h.obj), None),
    "FaceAlpha": (lambda h: P._num_val(1 if h.obj.get_alpha() is None else h.obj.get_alpha()),
                  lambda h, v: h.obj.set_alpha(P._scalar(v))),
    "DisplayName": (lambda h: P._get_label(h.obj), P._set_label),
})


def _container_line(h):
    o = h.obj
    return o.markerline if h.kind == "stem" else o.lines[0]


def _stem_xy(which: str):
    def g(h):
        ln = h.obj.markerline
        return P._row(ln.get_xdata() if which == "x" else ln.get_ydata())
    return g


def _stem_set_xy(which: str):
    def s(h, v):
        data = np.real(fnum(v)).flatten()
        cont = h.obj
        x = np.asarray(cont.markerline.get_xdata(), dtype=float)
        y = np.asarray(cont.markerline.get_ydata(), dtype=float)
        if data.size != x.size:
            raise MatlabError("The new data must have the same number of points.")
        if which == "x":
            x = data
        else:
            y = data
        base = float(np.asarray(cont.baseline.get_ydata()).flat[0]) if cont.baseline is not None else 0.0
        cont.markerline.set_data(x, y)
        cont.stemlines.set_segments([[(xi, base), (xi, yi)] for xi, yi in zip(x, y)])
        if cont.baseline is not None:
            cont.baseline.set_xdata([x.min(), x.max()] if x.size else [0, 1])
        _rescale(h)
    return s


def _stem_base_set(h, v):
    b = P._scalar(v)
    cont = h.obj
    x = np.asarray(cont.markerline.get_xdata(), dtype=float)
    y = np.asarray(cont.markerline.get_ydata(), dtype=float)
    cont.stemlines.set_segments([[(xi, b), (xi, yi)] for xi, yi in zip(x, y)])
    if cont.baseline is not None:
        cont.baseline.set_ydata([b, b])
    _rescale(h)


def _stem_linestyle_set(h, v):
    s = text(v)
    h.obj.stemlines.set_linestyle("None" if s.lower() == "none" else s)
    h.extra["LineStyle"] = s


def _baseline_handle(h):
    base = h.obj.baseline
    return P._wrap(h.interp, "line", base) if base is not None else _EMPTY


P._STEM_PROPS.update({
    "XData": (_stem_xy("x"), _stem_set_xy("x")),
    "YData": (_stem_xy("y"), _stem_set_xy("y")),
    "BaseValue": _number("BaseValue", 0.0, apply=_stem_base_set),
    "BaseLine": (_baseline_handle, None),
    "ShowBaseLine": (lambda h: P._oo(h.obj.baseline is not None and h.obj.baseline.get_visible()),
                     lambda h, v: h.obj.baseline.set_visible(P._onoff(v)) if h.obj.baseline is not None else None),
    "LineStyle": (lambda h: char(h.extra.get("LineStyle", "-")), _stem_linestyle_set),
    "MarkerEdgeColor": (lambda h: P._rgb(h.obj.markerline.get_markeredgecolor()),
                        lambda h, v: h.obj.markerline.set_markeredgecolor(P._color(v))),
    "SeriesIndex": (_series_index_get, _series_index_set),
})


def _eb_data(key: str):
    def g(h):
        return P._row(h.extra.get(key, []))
    return g


def _eb_line_prop(getter: str, setter: str, conv, wrap):
    return (lambda h: wrap(getattr(h.obj.lines[0], getter)()) if h.obj.lines[0] is not None else wrap(None),
            lambda h, v: getattr(h.obj.lines[0], setter)(conv(v)) if h.obj.lines[0] is not None else None)


def _eb_cap_set(h, v):
    size = P._scalar(v)
    for cap in h.obj.lines[1]:
        cap.set_markersize(size)
    h.extra["CapSize"] = size


def _eb_linestyle_get(h):
    ln = h.obj.lines[0]
    s = ln.get_linestyle() if ln is not None else "None"
    return char("none" if s in ("None", "", " ") else s)


P._ERRORBAR_PROPS.update({
    "XData": (_eb_data("XData"), None), "YData": (_eb_data("YData"), None),
    "YNegativeDelta": (_eb_data("YNegativeDelta"), None), "YPositiveDelta": (_eb_data("YPositiveDelta"), None),
    "XNegativeDelta": (_eb_data("XNegativeDelta"), None), "XPositiveDelta": (_eb_data("XPositiveDelta"), None),
    "CapSize": (lambda h: P._num_val(h.extra.get("CapSize", 6)), _eb_cap_set),
    "LineStyle": (_eb_linestyle_get,
                  lambda h, v: h.obj.lines[0].set_linestyle(P._mpl_linestyle(text(v)))
                  if h.obj.lines[0] is not None else None),
    "Marker": _eb_line_prop("get_marker", "set_marker", lambda v: P._mpl_marker(text(v)),
                            lambda m: char({"None": "none", "": "none", "none": "none"}.get(str(m), str(m)))),
    "MarkerSize": _eb_line_prop("get_markersize", "set_markersize", P._scalar,
                                lambda v: P._num_val(6 if v is None else v)),
    "MarkerFaceColor": _eb_line_prop("get_markerfacecolor", "set_markerfacecolor", P._color,
                                     lambda c: P._rgb("none" if c is None else c)),
    "MarkerEdgeColor": _eb_line_prop("get_markeredgecolor", "set_markeredgecolor", P._color,
                                     lambda c: P._rgb("none" if c is None else c)),
    "SeriesIndex": (_series_index_get, _series_index_set),
})

P._QUIVER_PROPS.update({
    "AutoScale": (lambda h: P._oo(h.extra.get("AutoScale", True)), None),
    "AutoScaleFactor": (lambda h: P._num_val(h.extra.get("AutoScaleFactor", 0.9)), None),
    "MaxHeadSize": _number("MaxHeadSize", 0.2, 0, 10),
    "ShowArrowHead": _onoff_state("ShowArrowHead"),
    "WData": (lambda h: h.extra.get("WData", _EMPTY), None),
    "ZData": (lambda h: h.extra.get("ZData", _EMPTY), None),
    "LineStyle": (_coll_linestyle_get, _coll_linestyle_set),
    "Alignment": _choice("Alignment", ("tail", "center", "head"), "tail",
                         lambda h, v: h.obj.set(pivot={"tail": "tail", "center": "mid", "head": "tip"}[v])
                         if hasattr(h.obj, "pivot") else None),
})


# -- tiled layouts ------------------------------------------------------------------------------------------------

def _layout_position_get(h):
    lay = h.obj
    axes = [a for a in lay.order if a in lay.fig.axes]
    if not axes:
        return P._row([0.13, 0.11, 0.775, 0.815])
    boxes = np.array([a.get_position().bounds for a in axes])
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
    return P._row([x0, y0, x1 - x0, y1 - y0])


def _layout_indexing_set(h, v):
    mode = text(v).lower()
    if mode not in ("rowmajor", "columnmajor"):
        raise MatlabError("TileIndexing must be 'rowmajor' or 'columnmajor'.")
    if h.obj.order:
        raise MatlabError("TileIndexing can only be changed before the layout has axes.")
    h.obj.indexing = mode


def _layout_grid_set(h, v):
    g = np.real(fnum(v)).flatten()
    if g.size != 2 or np.any(g < 1) or np.any(g != np.round(g)):
        raise MatlabError("GridSize must be [rows columns] with positive whole numbers.")
    h.obj.resize(int(g[0]), int(g[1]))


P._LAYOUT_PROPS.update({
    "GridSize": (lambda h: P._row(h.obj.grid_size()), _layout_grid_set),
    "TileArrangement": (lambda h: char(h.obj.mode), None),
    "TileIndexing": (lambda h: char(getattr(h.obj, "indexing", "rowmajor")), _layout_indexing_set),
    "Subtitle": (lambda h: P._layout_title(h, "Subtitle"), None),
    "Position": (_layout_position_get, None),
    "InnerPosition": (_layout_position_get, None),
    "OuterPosition": (lambda h: P._row([0, 0, 1, 1]), None),
    "Units": _choice("Units", _UNITS, "normalized"),
    "PositionConstraint": _choice("PositionConstraint", ("outerposition", "innerposition"), "outerposition"),
})


# -- root: screen properties and default values --------------------------------------------------------------------

# Default properties that matplotlib applies itself when an object is created.
_RC_DEFAULTS: dict[str, tuple] = {
    "defaultlinelinewidth": (("lines.linewidth",), P._scalar),
    "defaultlinemarkersize": (("lines.markersize",), P._scalar),
    "defaultlinelinestyle": (("lines.linestyle",), lambda v: P._mpl_linestyle(text(v))),
    "defaultlinemarker": (("lines.marker",), lambda v: P._mpl_marker(text(v))),
    "defaultlinemarkerfacecolor": (("lines.markerfacecolor",), lambda v: P._color(v)),
    "defaultlinemarkeredgecolor": (("lines.markeredgecolor",), lambda v: P._color(v)),
    "defaultaxeslinewidth": (("axes.linewidth",), P._scalar),
    "defaultaxesfontsize": (("xtick.labelsize", "ytick.labelsize"), P._scalar),
    "defaulttextfontsize": (("font.size",), P._scalar),
    "defaultaxesfontname": (("font.family",), lambda v: [text(v)]),
    "defaulttextfontname": (("font.family",), lambda v: [text(v)]),
    "defaultaxesfontweight": (("font.weight",), lambda v: text(v).lower()),
    "defaulttextcolor": (("text.color",), lambda v: P._color(v)),
    "defaultaxescolor": (("axes.facecolor",), lambda v: P._color(v)),
    "defaultfigurecolor": (("figure.facecolor", "savefig.facecolor"), lambda v: P._color(v)),
    "defaultaxesxcolor": (("xtick.color", "axes.edgecolor"), lambda v: P._color(v)),
    "defaultaxesycolor": (("ytick.color",), lambda v: P._color(v)),
    "defaultaxesxgrid": (("axes.grid",), lambda v: P._onoff(v)),
    "defaultaxesygrid": (("axes.grid",), lambda v: P._onoff(v)),
    "defaultaxesgridalpha": (("grid.alpha",), P._scalar),
    "defaultaxesgridcolor": (("grid.color",), lambda v: P._color(v)),
    "defaultaxesgridlinestyle": (("grid.linestyle",), lambda v: text(v)),
    "defaultaxestickdir": (("xtick.direction", "ytick.direction"),
                           lambda v: {"both": "inout"}.get(text(v).lower(), text(v).lower())),
    "defaultaxesbox": (("axes.spines.top", "axes.spines.right"), lambda v: P._onoff(v)),
    "defaultaxestitlefontweight": (("axes.titleweight",), lambda v: text(v).lower()),
    "defaultlegendfontsize": (("legend.fontsize",), P._scalar),
    "defaultlegendbox": (("legend.frameon",), lambda v: P._onoff(v)),
    "defaultpatchlinewidth": (("patch.linewidth",), P._scalar),
    "defaultpatchedgecolor": (("patch.edgecolor",), lambda v: P._color(v)),
    "defaulterrorbarcapsize": (("errorbar.capsize",), P._scalar),
    "defaultscattermarker": (("scatter.marker",), lambda v: P._mpl_marker(text(v))),
}

# Classes whose properties can have a default, and the handle kind that checks the property name.
_DEFAULT_CLASSES = {"figure": "figure", "axes": "axes", "line": "line", "text": "text", "legend": "legend",
                    "patch": "patch", "surface": "surface", "image": "image", "scatter": "scatter",
                    "bar": "bar", "histogram": "histogram", "colorbar": "colorbar", "errorbar": "errorbar",
                    "stem": "stem", "stair": "stair", "area": "area", "quiver": "quiver", "contour": "contour",
                    "polaraxes": "polaraxes", "rectangle": "rectangle", "tiledlayout": "tiledlayout",
                    "uicontrol": None, "uipanel": None, "uitable": None, "uimenu": None, "uibuttongroup": None,
                    "animatedline": "animatedline", "light": None, "hggroup": None, "hgtransform": None,
                    "constantline": "line", "functionline": "line", "polaraxesruler": None}


def _split_default(name: str):
    """('line', 'LineWidth') from 'DefaultLineLineWidth'; None if it is not a default property name."""
    low = name.lower()
    if not low.startswith("default") or len(low) <= 7:
        return None
    rest = name[7:]
    for cls in sorted(_DEFAULT_CLASSES, key=len, reverse=True):
        if rest.lower().startswith(cls) and len(rest) > len(cls):
            return cls, rest[len(cls):]
    return None


def _canonical_prop(kind: str | None, prop: str) -> str | None:
    if kind is None:
        return prop
    low = prop.lower()
    for tbl in (P._PROPS.get(kind, {}), P._COMMON_PROPS, P._IGNORED.get(kind, ())):
        for k in tbl:
            if k.lower() == low:
                return k
    return None


def _defaults(interp) -> dict:
    return interp.state.setdefault("gfx_defaults", {})


def _rebuild_rc(interp) -> None:
    rc: dict = {}
    d = _defaults(interp)
    for key, value in d.items():
        spec = _RC_DEFAULTS.get(key)
        if spec is None:
            continue
        names, conv = spec
        try:
            val = conv(value)
        except (MatlabError, ValueError, TypeError):
            continue
        for n in names:
            rc[n] = val
    if "defaultaxesfontsize" in d:
        fs = P._scalar(d["defaultaxesfontsize"])
        rc.update({"axes.labelsize": fs * 1.1, "axes.titlesize": fs * 1.1, "legend.fontsize": fs * 0.9})
    order = d.get("defaultaxescolororder")
    if order is not None:
        from cycler import cycler
        rc["axes.prop_cycle"] = cycler(color=[tuple(c) for c in to_color_matrix(order)])
    interp.state["gfx_rc"] = rc
    interp.state["axes_defaults_hook"] = _apply_axes_defaults if any(
        k.startswith(("defaultaxes", "defaultpolaraxes")) for k in d) else None
    interp.state["figure_defaults_hook"] = (lambda it, h: apply_object_defaults(it, h, "figure")) if any(
        k.startswith("defaultfigure") for k in d) else None


_AXES_DEFAULT_SKIP = {"defaultaxesfontsize", "defaultaxeslinewidth", "defaultaxescolor", "defaultaxesxcolor",
                      "defaultaxesycolor", "defaultaxesgridalpha", "defaultaxesgridcolor", "defaultaxesbox",
                      "defaultaxesgridlinestyle", "defaultaxestickdir", "defaultaxestitlefontweight",
                      "defaultaxescolororder", "defaultaxesfontname", "defaultaxesfontweight",
                      "defaultaxesxgrid", "defaultaxesygrid"}


def _apply_axes_defaults(interp, ax) -> None:
    """Give a new or cleared axes the default axes properties that matplotlib does not apply itself."""
    d = _defaults(interp)
    h = None
    for key, value in d.items():
        if not key.startswith("defaultaxes") or key in _AXES_DEFAULT_SKIP:
            continue
        prop = key[len("defaultaxes"):]
        if h is None:
            h = P._wrap(interp, P._axes_kind(ax), ax)
        try:
            P._set_prop(h, prop, value)
        except (MatlabError, ValueError, TypeError, AttributeError):
            continue
    order = d.get("defaultaxeslinestyleorder")
    if order is not None and h is None:
        ax._mylab_styleorder = P._text_list(order)
        apply_cycle(ax, 0)


def apply_object_defaults(interp, h, cls: str) -> None:
    """Apply stored defaults of class ``cls`` (e.g. 'figure', 'text') to a new object."""
    prefix = "default" + cls
    for key, value in list(_defaults(interp).items()):
        if key.startswith(prefix) and key not in _RC_DEFAULTS:
            try:
                P._set_prop(h, key[len(prefix):], value)
            except (MatlabError, ValueError, TypeError, AttributeError):
                continue


def set_default(interp, name: str, value) -> None:
    parts = _split_default(name)
    if parts is None:
        raise MatlabError(f"'{name}' is not a default property name. Write Default, then the object type, "
                          "then the property, e.g. 'DefaultLineLineWidth'.")
    cls, prop = parts
    kind = _DEFAULT_CLASSES[cls]
    canon = _canonical_prop(kind, prop)
    if canon is None:
        raise MatlabError(f"There is no {cls} property named '{prop}'.")
    key = ("default" + cls + canon).lower()
    d = _defaults(interp)
    if is_text(value) and text(value).lower() in ("remove", "factory", "default"):
        d.pop(key, None)
    else:
        spec = _RC_DEFAULTS.get(key)
        if spec is not None:
            spec[1](value)  # validates the value now, not when the next object is drawn
        if key == "defaultaxescolororder":
            to_color_matrix(value)
        d[key] = value
    _rebuild_rc(interp)


_FACTORY = {"defaultlinelinewidth": 0.5, "defaultlinemarkersize": 6.0, "defaultaxesfontsize": 10.0,
            "defaulttextfontsize": 10.0, "defaultaxeslinewidth": 0.5, "defaultlinelinestyle": "-",
            "defaultlinemarker": "none", "defaulttextinterpreter": "tex", "defaultlegendinterpreter": "tex",
            "defaultaxesticklabelinterpreter": "tex", "defaultfigurecolor": [0.94, 0.94, 0.94],
            "defaultaxescolor": [1.0, 1.0, 1.0], "defaultaxesbox": "off", "defaultaxesxgrid": "off",
            "defaultaxesygrid": "off", "defaultaxestickdir": "in", "defaultfigureposition": [100, 100, 640, 480],
            "defaultfigurevisible": "on", "defaultaxesfontname": "Helvetica", "defaulttextfontname": "Helvetica",
            "defaultlegendbox": "on", "defaultlegendlocation": "northeast", "defaultaxesunits": "normalized",
            "defaultfigureunits": "pixels", "defaultfigurepaperunits": "inches",
            "defaultfigurepaperpositionmode": "auto", "defaultaxesnextplot": "replace",
            "defaultfigurenextplot": "add", "defaultfigurewindowstyle": "normal",
            "defaultfigurecolormap": None, "defaultaxescolororder": None, "defaultaxeslinestyleorder": "-"}


def _to_value(v):
    if v is None:
        return _EMPTY
    if isinstance(v, str):
        return char(v)
    if isinstance(v, (int, float)):
        return P._num_val(v)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float).reshape(1, -1)
    return v


def get_default(interp, name: str):
    low = name.lower()
    if low.startswith("factory"):
        key = "default" + low[7:]
        if key == "defaultaxescolororder":
            return default_colors()
        if key == "defaultfigurecolormap":
            return P._cmap_matrix("viridis", 256)
        if key not in _FACTORY:
            raise MatlabError(f"There is no factory value named '{name}'.")
        return _to_value(_FACTORY[key])
    parts = _split_default(name)
    if parts is None:
        raise MatlabError(f"'{name}' is not a default property name.")
    cls, prop = parts
    canon = _canonical_prop(_DEFAULT_CLASSES[cls], prop)
    if canon is None:
        raise MatlabError(f"There is no {cls} property named '{prop}'.")
    key = ("default" + cls + canon).lower()
    d = _defaults(interp)
    if key in d:
        return d[key]
    if key == "defaultaxescolororder":
        return default_colors()
    if key == "defaultfigurecolormap":
        return P._cmap_query(interp)
    if key in _FACTORY:
        return _to_value(_FACTORY[key])
    return _EMPTY


def all_defaults(interp, factory: bool = False) -> Struct:
    if factory:
        out = {"factory" + _pretty_default(k)[7:]: _to_value(v) for k, v in _FACTORY.items() if v is not None}
        out["factoryAxesColorOrder"] = default_colors()
        return Struct.scalar(out)
    return Struct.scalar({_pretty_default(k): v for k, v in _defaults(interp).items()})


def _pretty_default(key: str) -> str:
    parts = _split_default(key)
    if parts is None:
        return key
    cls, prop = parts
    canon = _canonical_prop(_DEFAULT_CLASSES[cls], prop) or prop
    names = {"polaraxes": "PolarAxes", "errorbar": "ErrorBar", "tiledlayout": "TiledLayout",
             "animatedline": "AnimatedLine", "uicontrol": "Uicontrol", "uipanel": "Uipanel",
             "uitable": "Uitable", "uimenu": "Uimenu", "colorbar": "Colorbar"}
    return "default" + names.get(cls, cls.capitalize()) + canon


def _wrap_with_defaults(name: str) -> None:
    """Run a plotting built-in with the interpreter's default properties in force."""
    fn = BUILTINS[name]

    def wrapper(interp, args, nargout, _fn=fn):
        rc = interp.state.get("gfx_rc")
        if not rc:
            return _fn(interp, args, nargout)
        import matplotlib as mpl
        with mpl.rc_context(rc):
            return _fn(interp, args, nargout)

    wrapper.__name__ = getattr(fn, "__name__", name)
    wrapper.__doc__ = fn.__doc__
    BUILTINS[name] = wrapper


def install_default_wrappers() -> None:
    """Wrap every graphics built-in defined so far (called by the last graphics module to load)."""
    modules = ("plotting", "plotmore", "plot_charts", "plot_volume", "plot_objects", "plot_rulers", "figfiles")
    for name, fn in list(BUILTINS.items()):
        mod = getattr(fn, "__module__", "") or ""
        if mod.rsplit(".", 1)[-1] in modules and not getattr(fn, "_gfx_defaults", False):
            _wrap_with_defaults(name)
            BUILTINS[name]._gfx_defaults = True


def _root_get_hook(h, name: str):
    return get_default(h.interp, name)


def _root_set_hook(h, name: str, value) -> None:
    set_default(h.interp, name, value)


P._ROOT_HOOKS = (_root_get_hook, _root_set_hook)
P._ROOT_PROPS.update({
    "Units": _choice("Units", ("pixels", "normalized", "inches", "centimeters", "points", "characters"), "pixels"),
    "PointerLocation": _vector("PointerLocation", [0, 0], 2),
    "ScreenDepth": (lambda h: P._num_val(24), None),
    "FixedWidthFontName": _state("FixedWidthFontName", "Courier New"),
    "ShowHiddenHandles": _onoff_state("ShowHiddenHandles", "off"),
    "CallbackObject": (lambda h: BUILTINS["gcbo"](h.interp, [], 1) if "gcbo" in BUILTINS else _EMPTY, None),
    "CurrentFigure": (P._root_current_figure,
                      lambda h, v: BUILTINS["figure"](h.interp, [v], 0)),
})


# -- built-ins ---------------------------------------------------------------------------------------------------

def _axes_arg(interp, args: list):
    """(axes object, remaining args) for functions that take an optional leading axes handle."""
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in ("axes", "polaraxes", "geoaxes"):
        return args[0].obj, list(args[1:])
    return None, list(args)


@builtin("colororder")
def _colororder(interp, args, nargout):
    """Query or change the colours that successive plots in an axes or figure get."""
    args = list(args)
    target = None
    if args and isinstance(args[0], P.GraphicsHandle):
        target = args.pop(0)
    if target is None:
        axes = [a for a in P._figure(interp).axes if not P._is_colorbar_axes(a)] or [P._axes(interp, fresh=False)]
    elif target.kind == "figure":
        axes = [a for a in target.obj.axes if not P._is_colorbar_axes(a)]
    elif target.kind in ("axes", "polaraxes"):
        axes = [target.obj]
    elif target.kind == "tiledlayout":
        axes = [a for a in target.obj.order if a in target.obj.fig.axes]
    else:
        raise MatlabError("colororder expects a figure, an axes or a tiled layout.")
    if not args:
        return color_order(axes[0]) if axes else default_colors()
    if is_text(args[0]) and text(args[0]).lower() == "default":
        colors = default_colors()
    else:
        colors = to_color_matrix(args[0])
    for ax in axes:
        set_color_order(interp, ax, colors)
    P._redraw(interp)
    return colors if nargout > 0 else None


@builtin("newplot")
def _newplot(interp, args, nargout):
    """Prepare the current axes for a new plot, following its hold state; returns the axes."""
    target, rest = _axes_arg(interp, args)
    st = P._state(interp)
    prev = st.override_ax
    if target is not None:
        st.override_ax = target
    try:
        fig = P._figure(interp)
        fh = P._wrap(interp, "figure", fig)
        mode = str(fh.extra.get("NextPlot", "add")).lower()
        if mode == "new" and target is None:
            BUILTINS["figure"](interp, [], 0)
        elif mode in ("replace", "replacechildren") and target is None:
            BUILTINS["clf"](interp, [char("reset")] if mode == "replace" else [], 0)
        ax = P._axes(interp, fresh=True)
    finally:
        st.override_ax = prev
    return P._wrap(interp, P._axes_kind(ax), ax) if nargout > 0 else None


class _PropLink:
    """Keeps a set of properties equal on several objects (see linkprop)."""

    def __init__(self, handles: list, props: list):
        self.handles = list(handles)
        self.props = list(props)
        self.enabled = True
        self.figure = P._obj_figure(handles[0]) if handles else None

    def sync(self, source, name: str, value) -> None:
        if not self.enabled:
            return
        self.enabled = False
        try:
            for h in self.handles:
                if h is not source and P._alive(h):
                    P._set_prop(h, name, value)
        finally:
            self.enabled = True


def _links_for(h) -> list:
    return h.extra.setdefault("_links", [])


def propagate_links(h, name: str, value) -> None:
    """Called after a property of ``h`` is set: copy it to the objects linked with linkprop."""
    links = h.extra.get("_links")
    if not links:
        return
    low = name.lower()
    for link in links:
        if any(p.lower() == low for p in link.props):
            link.sync(h, name, value)


P._LINK_HOOK = propagate_links


def _linkprop_targets(h):
    hs = [x for x in h.obj.handles if P._alive(x)]
    return P._handle_list(hs)


def _link_enabled_set(h, v):
    h.obj.enabled = P._onoff(v)


P._CLASS["proplink"] = "matlab.graphics.internal.LinkProp"
P._PROPS["proplink"] = {
    "Targets": (_linkprop_targets, None),
    "PropertyNames": (lambda h: Cell.from_list([char(p) for p in h.obj.props]), None),
    "Enabled": (lambda h: P._oo(h.obj.enabled), _link_enabled_set),
}


@builtin("linkprop")
def _linkprop(interp, args, nargout):
    """Keep the named properties of several graphics objects equal."""
    if len(args) < 2:
        raise MatlabError("linkprop needs the objects and the property name(s), e.g. linkprop([ax1 ax2], 'XLim').")
    hs = P._handles_of(args[0])
    if len(hs) < 1:
        raise MatlabError("linkprop expects an array of graphics handles.")
    props = P._text_list(args[1])
    for h in hs:
        for p in props:
            if P._lookup(h, p)[0] is None:
                raise MatlabError(f"'{p}' is not a property of a {P._short(h.kind)} object.")
    link = _PropLink(hs, props)
    for h in hs:
        _links_for(h).append(link)
    for p in props:  # start from the first object's values
        link.sync(hs[0], p, P._get_prop(hs[0], p))
    P._redraw(interp)
    return P._wrap(interp, "proplink", link)


@builtin("addtarget")
def _addtarget(interp, args, nargout):
    if len(args) != 2 or not isinstance(args[0], P.GraphicsHandle) or args[0].kind != "proplink":
        raise MatlabError("addtarget needs a link object from linkprop and a graphics handle.")
    link = args[0].obj
    for h in P._handles_of(args[1]):
        if h not in link.handles:
            link.handles.append(h)
            _links_for(h).append(link)
            for p in link.props:
                P._set_prop(h, p, P._get_prop(link.handles[0], p))
    return None


@builtin("removetarget")
def _removetarget(interp, args, nargout):
    if len(args) != 2 or not isinstance(args[0], P.GraphicsHandle) or args[0].kind != "proplink":
        raise MatlabError("removetarget needs a link object from linkprop and a graphics handle.")
    link = args[0].obj
    for h in P._handles_of(args[1]):
        if h in link.handles:
            link.handles.remove(h)
            _links_for(h).remove(link)
    return None


@builtin("refreshdata")
def _refreshdata(interp, args, nargout):
    """Re-read XDataSource/YDataSource/ZDataSource of plot objects and update their data."""
    args = list(args)
    where = "base"
    if args and is_text(args[-1]) and text(args[-1]).lower() in ("base", "caller"):
        where = text(args.pop()).lower()
    if args:
        roots = []
        for a in args:
            roots.extend(P._handles_of(a) or P._resolve(interp, a))
    else:
        roots = [P._wrap(interp, "figure", P._figure(interp))]
    seen: list = []

    def visit(h):
        if h in seen:
            return
        seen.append(h)
        for c in P._child_handles(interp, h):
            visit(c)

    for r in roots:
        visit(r)
    for h in seen:
        for axis in "XYZ":
            src = h.extra.get(f"{axis}DataSource")
            if src is None or (isinstance(src, str) and not src) or (not isinstance(src, str) and not is_text(src)):
                continue
            expr = src if isinstance(src, str) else text(src)
            if not expr:
                continue
            value = _eval_in(interp, expr, where)
            P._set_prop(h, f"{axis}Data", value)
    P._redraw(interp)
    return None


def _eval_in(interp, expr: str, where: str):
    if where == "caller" and len(interp.frames) > 1:
        res = BUILTINS["evalin"](interp, [char("caller"), char(expr)], 1)
    else:
        res = BUILTINS["evalin"](interp, [char("base"), char(expr)], 1)
    return res[0] if isinstance(res, tuple) else res


@builtin("uistack")
def _uistack(interp, args, nargout):
    """Move graphics objects up or down in the drawing order."""
    if not args:
        raise MatlabError("uistack needs a graphics handle.")
    hs = P._handles_of(args[0])
    mode = text(args[1]).lower() if len(args) > 1 else "up"
    step = int(P._scalar(args[2])) if len(args) > 2 else 1
    if mode not in ("up", "down", "top", "bottom"):
        raise MatlabError("uistack: the position must be 'up', 'down', 'top' or 'bottom'.")
    for h in hs:
        ax = P._obj_axes(h)
        arts = P._artists(h.obj)
        if ax is None or not arts:
            continue
        others = [a.get_zorder() for a in ax.get_children() if hasattr(a, "get_zorder") and a not in arts]
        z = arts[0].get_zorder()
        if mode == "top":
            z = max(others, default=z) + 1
        elif mode == "bottom":
            z = min([o for o in others if o > 0] or [z]) - 1
        else:
            z = z + (step if mode == "up" else -step)
        for a in arts:
            a.set_zorder(z)
    P._redraw(interp)
    return None


def _text_objects(interp, h) -> list:
    """Text artists reached from a figure, axes, legend, colorbar or text handle."""
    if h.kind == "text":
        return [h.obj]
    if h.kind == "figure":
        out = []
        for ax in h.obj.axes:
            out.extend(_text_objects(interp, P._wrap(interp, P._axes_kind(ax), ax)))
        for name in ("_suptitle", "_supxlabel", "_supylabel"):
            t = getattr(h.obj, name, None)
            if t is not None:
                out.append(t)
        return out
    if h.kind in ("axes", "polaraxes"):
        ax = h.obj
        out = [ax.title] + _axes_labels(ax) + _tick_labels(ax) + list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            out.extend(leg.get_texts())
        return out
    if h.kind == "legend":
        return list(h.obj.get_texts())
    if h.kind == "colorbar":
        axis = _cb_axis(h)
        return axis.get_ticklabels() + [axis.label]
    if h.kind == "tiledlayout":
        out = []
        for ax in h.obj.order:
            out.extend(_text_objects(interp, P._wrap(interp, P._axes_kind(ax), ax)))
        return out
    return []


@builtin("fontsize")
def _fontsize(interp, args, nargout):
    """Set or scale the font size of all text in a figure, axes or other object."""
    args = list(args)
    targets = []
    while args and isinstance(args[0], (P.GraphicsHandle, P.GraphicsArray)):
        targets.extend(P._handles_of(args.pop(0)))
    if not targets:
        targets = [P._wrap(interp, "figure", P._figure(interp))]
    if not args:
        raise MatlabError("fontsize needs a size, e.g. fontsize(14, 'points'), or 'increase', 'decrease', "
                          "'default' or fontsize(scale=1.2).")
    scale = None
    size = None
    first = args[0]
    if is_text(first):
        word = text(first).lower()
        if word == "increase":
            scale = 1.1
        elif word == "decrease":
            scale = 1 / 1.1
        elif word == "default":
            size = 10.0
        elif word == "scale" and len(args) > 1:
            scale = P._scalar(args[1])
        else:
            raise MatlabError(f"fontsize: unknown option '{text(first)}'.")
    else:
        size = P._scalar(first)
        units = text(args[1]).lower() if len(args) > 1 else "points"
        fig = P._obj_figure(targets[0])
        dpi = P._fig_dpi(fig) if fig is not None else 100.0
        factor = {"points": 1.0, "pixels": 72.0 / dpi, "inches": 72.0, "centimeters": 72.0 / 2.54}.get(units)
        if factor is None:
            raise MatlabError("fontsize units must be 'points', 'pixels', 'inches' or 'centimeters'.")
        size *= factor
    if (size is not None and size <= 0) or (scale is not None and scale <= 0):
        raise MatlabError("The font size must be positive.")
    for h in targets:
        if size is not None and h.kind in ("axes", "polaraxes"):
            _axes_fontsize_set(h, np.array([[size]]))
            continue
        if size is not None and h.kind == "figure":
            for ax in h.obj.axes:
                if not P._is_colorbar_axes(ax):
                    _axes_fontsize_set(P._wrap(interp, P._axes_kind(ax), ax), np.array([[size]]))
                else:
                    ax.tick_params(labelsize=size)
            continue
        if h.kind in ("axes", "polaraxes"):
            h.obj.tick_params(labelsize=_base_font(h) * scale)
            h.extra["FontSize"] = _base_font(h) * scale
            for t in [h.obj.title] + _axes_labels(h.obj) + list(h.obj.texts):
                t.set_fontsize(t.get_fontsize() * scale)
            continue
        for t in _text_objects(interp, h):
            t.set_fontsize(size if size is not None else t.get_fontsize() * scale)
    P._redraw(interp)
    return None


@builtin("fontname")
def _fontname(interp, args, nargout):
    """Set the font of all text in a figure, axes or other object."""
    args = list(args)
    targets = []
    while args and isinstance(args[0], (P.GraphicsHandle, P.GraphicsArray)):
        targets.extend(P._handles_of(args.pop(0)))
    if not targets:
        targets = [P._wrap(interp, "figure", P._figure(interp))]
    if not args or not is_text(args[0]):
        raise MatlabError("fontname needs a font name, e.g. fontname('Arial').")
    name = text(args[0])
    for h in targets:
        for t in _text_objects(interp, h):
            t.set_fontfamily(name)
        if h.kind in ("axes", "polaraxes"):
            h.extra["FontName"] = name
    P._redraw(interp)
    return None


@builtin("validatecolor")
def _validatecolor(interp, args, nargout):
    """RGB triplets (one per row) for colour names, short names, hex codes or RGB values."""
    if not args:
        raise MatlabError("validatecolor needs a colour.")
    multiple = len(args) > 1 and is_text(args[1]) and text(args[1]).lower() == "multiple"
    v = args[0]
    if isinstance(v, np.ndarray) and not is_char(v) and v.dtype.kind in "iu" and v.dtype != np.bool_:
        v = v.astype(float) / float(np.iinfo(v.dtype).max)
    m = to_color_matrix(v)
    if m.shape[0] > 1 and not multiple:
        raise MatlabError("Specify one colour, or add 'multiple' to validate several.")
    return m


@builtin("brighten")
def _brighten(interp, args, nargout):
    """Brighten (beta > 0) or darken (beta < 0) a colormap."""
    args = list(args)
    target = None
    if args and isinstance(args[0], P.GraphicsHandle):
        target = args.pop(0)
    if not args:
        raise MatlabError("brighten needs beta, a number between -1 and 1.")
    cmap = None
    if len(args) >= 2:
        cmap, beta = np.real(fnum(args[0])), P._scalar(args[1])
    else:
        beta = P._scalar(args[0])
    if not -1 <= beta <= 1:
        raise MatlabError("brighten: beta must be between -1 and 1.")
    gamma = 1 - beta if beta > 0 else 1 / (1 + beta) if beta > -1 else math.inf
    src = cmap if cmap is not None else P._cmap_query(interp)
    out = np.clip(src, 0, 1) ** gamma if math.isfinite(gamma) else np.zeros_like(src)
    if cmap is not None or nargout > 0:
        return out
    BUILTINS["colormap"](interp, ([target] if target is not None else []) + [out], 0)
    return None


@builtin("contrast")
def _contrast(interp, args, nargout):
    """A gray colormap that spreads the intensities of an image evenly."""
    if not args:
        raise MatlabError("contrast needs an image.")
    x = np.real(fnum(args[0])).flatten()
    m = int(P._scalar(args[1])) if len(args) > 1 else 256
    x = x[np.isfinite(x)]
    if x.size == 0 or m < 1:
        return np.zeros((0, 3))
    lo, hi = float(x.min()), float(x.max())
    counts, _ = np.histogram(x, bins=m, range=(lo, hi if hi > lo else lo + 1))
    cdf = np.cumsum(counts) / max(counts.sum(), 1)
    g = np.concatenate([[0.0], cdf[:-1]]) if m > 1 else np.array([0.0])
    g = g / g.max() if g.max() > 0 else g
    return np.repeat(g.reshape(-1, 1), 3, axis=1)


@builtin("rgbplot")
def _rgbplot(interp, args, nargout):
    """Plot the red, green and blue columns of a colormap."""
    if not args:
        raise MatlabError("rgbplot needs a colormap.")
    m = np.real(fnum(args[0]))
    if m.ndim != 2 or m.shape[1] != 3:
        raise MatlabError("rgbplot expects an m-by-3 colormap.")
    ax = P._axes(interp, projection="rectilinear")
    x = np.arange(1, m.shape[0] + 1)
    lines = [ax.plot(x, m[:, k], color=c)[0] for k, c in enumerate("rgb")]
    ax.set_xlim(0, m.shape[0] + 1)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


@builtin("getpixelposition")
def _getpixelposition(interp, args, nargout):
    """Position of a figure or axes in pixels, [left bottom width height]."""
    if not args:
        raise MatlabError("getpixelposition needs a graphics handle.")
    h = P._resolve(interp, args[0])[0]
    if h.kind == "figure":
        w, hh = _fig_pixels(h.obj)
        left, bottom = getattr(h.obj, "_mylab_origin", (100.0, 100.0))
        return P._row([left, bottom, round(w), round(hh)])
    if h.kind in ("axes", "polaraxes"):
        b = h.obj.get_position().bounds
        w, hh = _fig_pixels(h.obj.figure)
        return P._row([b[0] * w + 1, b[1] * hh + 1, b[2] * w, b[3] * hh])
    raise MatlabError("getpixelposition works on figures and axes.")


@builtin("setpixelposition")
def _setpixelposition(interp, args, nargout):
    """Set the position of a figure or axes in pixels."""
    if len(args) < 2:
        raise MatlabError("setpixelposition needs a handle and [left bottom width height].")
    h = P._resolve(interp, args[0])[0]
    p = np.real(fnum(args[1])).flatten()
    if p.size != 4 or p[2] <= 0 or p[3] <= 0:
        raise MatlabError("The position must be [left bottom width height] with a positive width and height.")
    if h.kind == "figure":
        units = h.extra.get("Units", "pixels")
        h.extra["Units"] = "pixels"
        try:
            _fig_pos_set(h, p)
        finally:
            h.extra["Units"] = units
    elif h.kind in ("axes", "polaraxes"):
        w, hh = _fig_pixels(h.obj.figure)
        h.obj.set_position([(p[0] - 1) / w, (p[1] - 1) / hh, p[2] / w, p[3] / hh])
    else:
        raise MatlabError("setpixelposition works on figures and axes.")
    P._redraw(interp)
    return None


@builtin("orient")
def _orient(interp, args, nargout):
    """Paper orientation used when a figure is printed: 'portrait', 'landscape' or 'tall'."""
    args = list(args)
    fig = None
    if args and not is_text(args[0]):
        fig = P._resolve(interp, args.pop(0))[0]
    if fig is None:
        fig = P._wrap(interp, "figure", P._figure(interp))
    if fig.kind != "figure":
        raise MatlabError("orient expects a figure.")
    if not args:
        return char(str(fig.extra.get("PaperOrientation", "portrait")))
    mode = text(args[0]).lower()
    if mode not in ("portrait", "landscape", "tall"):
        raise MatlabError("orient must be 'portrait', 'landscape' or 'tall'.")
    fig.extra["PaperOrientation"] = "landscape" if mode == "landscape" else "portrait"
    pw, ph = _paper_size_in(fig)
    if mode in ("landscape", "tall"):  # fill the page, leaving a quarter-inch margin
        fig.extra["_PaperPositionIn"] = (0.25, 0.25, pw - 0.5, ph - 0.5)
        fig.extra["PaperPositionMode"] = "manual"
    else:
        fig.extra["PaperPositionMode"] = "auto"
    return None


def _polar_only(interp, args, name: str):
    target, rest = _axes_arg(interp, args)
    ax = target if target is not None else P._axes(interp, fresh=False)
    if getattr(ax, "name", "") != "polar":
        raise MatlabError(f"{name} works on polar axes; draw with polarplot first.")
    return ax, rest


@builtin("polaraxes")
def _polaraxes(interp, args, nargout):
    """Create polar axes in the current figure (or make polar axes current)."""
    args = list(args)
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind == "polaraxes" and len(args) == 1:
        return BUILTINS["axes"](interp, args, nargout)
    parent = None
    if args and isinstance(args[0], P.GraphicsHandle):
        parent = args.pop(0)
    pairs = P._name_values(args, "polaraxes")
    st = P._state(interp)
    if parent is not None and parent.kind == "figure":
        fig = parent.obj
        P._figure(interp, P._fig_number(interp, fig))
    else:
        fig = P._figure(interp)
    cur = st.axes.get(st.current)
    if cur is not None and cur in fig.axes and not cur.has_data() and getattr(cur, "name", "") != "polar":
        ax = P._replace_axes(interp, cur, "polar")
    elif not [a for a in fig.axes if not P._is_colorbar_axes(a)]:
        ax = fig.add_subplot(1, 1, 1, projection="polar")
        st.axes[st.current] = ax
    else:
        st.counter = getattr(st, "counter", 0) + 1
        ax = fig.add_axes([0.13, 0.11, 0.775, 0.815], projection="polar", label=f"mylab-polar-{st.counter}")
        st.axes[st.current] = ax
    h = P._wrap(interp, "polaraxes", ax)
    P._apply_props(h, pairs)
    P._redraw(interp)
    return h if nargout > 0 else None


@builtin("rlim")
def _rlim(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "rlim")
    if not rest:
        return P._row([ax.get_rmin(), ax.get_rmax()])
    if is_text(rest[0]):
        mode = text(rest[0]).lower()
        if mode == "auto":
            ax.set_autoscaley_on(True)
            ax.autoscale_view()
        elif mode == "mode":
            return char("auto" if ax.get_autoscaley_on() else "manual")
        elif mode != "manual":
            raise MatlabError("rlim expects [rmin rmax], 'auto' or 'manual'.")
        return None
    lim = np.real(fnum(rest[0])).flatten()
    if lim.size != 2 or not lim[0] < lim[1]:
        raise MatlabError("rlim expects [rmin rmax] with rmin < rmax.")
    ax.set_rlim(float(lim[0]), float(lim[1]))
    P._redraw(interp)
    return None


@builtin("thetalim")
def _thetalim(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "thetalim")
    if not rest:
        return P._row([np.degrees(ax.get_thetamin() * np.pi / 180) if False else ax.get_thetamin(),
                       ax.get_thetamax()])
    if is_text(rest[0]):
        mode = text(rest[0]).lower()
        if mode == "auto":
            ax.set_thetalim(0, 2 * np.pi)
        elif mode not in ("manual",):
            raise MatlabError("thetalim expects [thetamin thetamax] in degrees, 'auto' or 'manual'.")
        return None
    lim = np.real(fnum(rest[0])).flatten()
    if lim.size != 2 or not lim[0] < lim[1]:
        raise MatlabError("thetalim expects [thetamin thetamax] in degrees.")
    ax.set_thetamin(float(lim[0]))
    ax.set_thetamax(float(lim[1]))
    P._redraw(interp)
    return None


@builtin("rticks")
def _rticks(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "rticks")
    if not rest:
        t = np.asarray(ax.get_yticks(), dtype=float)
        return P._row(t[(t >= ax.get_rmin() - 1e-12) & (t <= ax.get_rmax() + 1e-12)])
    if is_text(rest[0]):
        if text(rest[0]).lower() == "auto":
            from matplotlib.ticker import AutoLocator
            ax.yaxis.set_major_locator(AutoLocator())
        return None
    ax.set_rticks(np.real(fnum(rest[0])).flatten())
    P._redraw(interp)
    return None


@builtin("thetaticks")
def _thetaticks(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "thetaticks")
    if not rest:
        return P._row(np.degrees(np.asarray(ax.get_xticks(), dtype=float)))
    if is_text(rest[0]):
        if text(rest[0]).lower() == "auto":
            ax.set_thetagrids(np.arange(0, 360, 30))
        return None
    ax.set_thetagrids(np.real(fnum(rest[0])).flatten())
    P._redraw(interp)
    return None


@builtin("rticklabels")
def _rticklabels(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "rticklabels")
    if not rest:
        fixed = getattr(ax, "_mylab_yticklabels", None)
        labels = fixed if fixed is not None else [f"{t:g}" for t in ax.get_yticks()]
        return P._cell_column([char(s) for s in labels])
    P._set_ticklabels(ax, "y", rest[0])
    P._redraw(interp)
    return None


@builtin("thetaticklabels")
def _thetaticklabels(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "thetaticklabels")
    if not rest:
        fixed = getattr(ax, "_mylab_xticklabels", None)
        labels = fixed if fixed is not None else [f"{np.degrees(t):g}" for t in ax.get_xticks()]
        return P._cell_column([char(s) for s in labels])
    if is_text(rest[0]) and text(rest[0]).lower() == "auto":
        ax.set_thetagrids(np.degrees(ax.get_xticks()))
        ax._mylab_xticklabels = None
    else:
        labels = P._text_list(rest[0])
        ticks = list(ax.get_xticks())
        full = (labels + [""] * len(ticks))[:len(ticks)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(full)
        ax._mylab_xticklabels = full
    P._redraw(interp)
    return None


@builtin("rtickangle")
def _rtickangle(interp, args, nargout):
    ax, rest = _polar_only(interp, args, "rtickangle")
    if not rest:
        return P._num_val(ax.get_rlabel_position())
    ax.set_rlabel_position(P._scalar(rest[0]))
    P._redraw(interp)
    return None


@builtin("rtickformat", "thetatickformat")
def _polar_tickformat(interp, args, nargout):
    from .plot_rulers import format_number
    from matplotlib.ticker import FuncFormatter
    ax, rest = _polar_only(interp, args, "rtickformat")
    if not rest:
        return char(getattr(ax, "_mylab_rfmt", "%g"))
    pattern = text(rest[0])
    ax._mylab_rfmt = pattern
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_number(pattern, float(x))))
    P._redraw(interp)
    return None


@builtin("zticklabels")
def _zticklabels(interp, args, nargout):
    target, rest = _axes_arg(interp, args)
    ax = target if target is not None else P._axes(interp, fresh=False)
    h = P._wrap(interp, P._axes_kind(ax), ax)
    if not rest:
        return _zticklabel_get(h)
    _z_only(_zticklabel_set)(h, rest[0])
    P._redraw(interp)
    return None


@builtin("ztickangle")
def _ztickangle(interp, args, nargout):
    target, rest = _axes_arg(interp, args)
    ax = target if target is not None else P._axes(interp, fresh=False)
    if not hasattr(ax, "zaxis"):
        raise MatlabError("These axes are two-dimensional and have no z-axis.")
    if not rest:
        labels = ax.get_zticklabels()
        return P._num_val(labels[0].get_rotation() if labels else 0)
    ax.tick_params(axis="z", labelrotation=P._scalar(rest[0]))
    P._redraw(interp)
    return None
