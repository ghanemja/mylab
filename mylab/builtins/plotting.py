"""Plotting, implemented on top of matplotlib.

Figures are numbered like MATLAB (``figure(2)``), ``hold on`` keeps the
current plot, and ``saveas``/``print`` write image files. When no display is
available (e.g. on a server) plots are rendered off-screen and can be saved.

Graphics functions return *handle objects* (:class:`GraphicsHandle`) when an
output is requested. A handle wraps the matplotlib figure, axes or artist and
exposes a set of MATLAB-style properties through dot syntax and ``get``/``set``.
Several handles at once (e.g. ``plot`` drawing three lines) are returned as a
:class:`GraphicsArray`.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, to_int, to_str
from ._registry import builtin
from ._util import fnum, is_text, num, text

_MPL = None


def _plt():
    global _MPL
    if _MPL is None:
        import matplotlib
        if not os.environ.get("MPLBACKEND"):
            headless = os.name == "posix" and not os.environ.get("DISPLAY") and \
                not os.environ.get("WAYLAND_DISPLAY") and os.uname().sysname != "Darwin"
            if headless:
                matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _MPL = plt
    return _MPL


class PlotState:
    def __init__(self):
        self.figures: dict[int, object] = {}
        self.current: int | None = None
        self.axes: dict[int, object] = {}  # figure number -> current axes
        self.hold: dict[int, bool] = {}     # id(axes) -> hold state
        self.handles: dict[int, GraphicsHandle] = {}  # id(matplotlib object) -> handle
        self.layouts: dict[int, _Layout] = {}  # figure number -> tiled layout
        self.override_ax = None             # axes given as first argument, e.g. plot(ax, ...)


def _state(interp) -> PlotState:
    st = interp.state.get("plot")
    if st is None:
        st = PlotState()
        interp.state["plot"] = st
    return st


def _figure(interp, n: int | None = None):
    plt = _plt()
    st = _state(interp)
    if n is None:
        if st.override_ax is not None:
            return st.override_ax.figure
        if st.current is not None and st.current in st.figures:
            return st.figures[st.current]
        n = 1
        while n in st.figures:
            n += 1
    if n not in st.figures:
        # Another interpreter may own matplotlib's figure n; never reuse its contents.
        fresh = not plt.fignum_exists(n)
        fig = plt.figure(n if fresh else None)
        if not fresh and fig.canvas.manager is not None:
            try:
                fig.canvas.manager.set_window_title(f"Figure {n}")
            except (AttributeError, NotImplementedError):
                pass
        st.figures[n] = fig
    st.current = n
    plt.figure(st.figures[n].number)
    return st.figures[n]


def _fig_number(interp, fig) -> int | None:
    for n, f in _state(interp).figures.items():
        if f is fig:
            return n
    return None


def _is_colorbar_axes(ax) -> bool:
    return hasattr(ax, "_colorbar")


def _yy_restyle(ax) -> None:
    """Re-apply the left/right styling of yyaxis axes (cla resets it)."""
    side = getattr(ax, "_mylab_yyside", None)
    if side is None:
        return
    from cycler import cycler
    import matplotlib as mpl
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    color = colors[0] if side == "left" else colors[1 % len(colors)]
    ax.set_prop_cycle(cycler(linestyle=["-", "--", ":", "-."]) + cycler(color=[color] * 4))
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.xaxis.set_visible(False)
        ax.patch.set_visible(False)
    else:
        ax.yaxis.tick_left()
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)


def _replace_axes(interp, ax, projection: str):
    """Replace ``ax`` by a new axes with another projection in the same place."""
    st = _state(interp)
    fig = ax.figure
    hold = st.hold.get(id(ax), False)
    spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
    pos = ax.get_position()
    layout = st.layouts.get(_fig_number(interp, fig) or -1)
    ax.remove()
    kw = {} if projection == "rectilinear" else {"projection": projection}
    new = fig.add_subplot(spec, **kw) if spec is not None else fig.add_axes(pos, **kw)
    if layout is not None:
        layout.replace(ax, new)
    st.hold[id(new)] = hold
    if st.override_ax is ax:
        st.override_ax = new
    else:
        st.axes[st.current] = new
    return new


def _axes(interp, fresh: bool = True, projection: str | None = None):
    """Current axes; cleared first unless ``hold on`` (when ``fresh``).

    ``projection`` ('rectilinear', '3d', 'polar') replaces the current axes by a
    new one of that kind when it differs (and hold is off, or the kinds cannot mix).
    """
    st = _state(interp)
    if st.override_ax is not None:
        ax = st.override_ax
    else:
        fig = _figure(interp)
        ax = st.axes.get(st.current)
        if ax is None or ax.figure is not fig or ax not in fig.axes:
            candidates = [a for a in fig.axes if not _is_colorbar_axes(a)]
            if candidates:
                ax = candidates[-1]
            else:
                kw = {"projection": projection} if projection not in (None, "rectilinear") else {}
                ax = fig.add_subplot(1, 1, 1, **kw)
            st.axes[st.current] = ax
    if projection is not None and getattr(ax, "name", "rectilinear") != projection:
        return _replace_axes(interp, ax, projection)
    if fresh and not st.hold.get(id(ax), False):
        cb = getattr(ax, "_mylab_colorbar", None)
        if cb is not None:
            try:
                cb.remove()
            except (ValueError, AttributeError, KeyError):
                pass
            ax._mylab_colorbar = None
        ax.cla()
        _yy_restyle(ax)
    return ax


def close_all(interp) -> None:
    st = interp.state.get("plot")
    if st is None:
        return
    plt = _plt()
    for fig in st.figures.values():
        plt.close(fig)
    st.figures.clear()
    st.axes.clear()
    st.handles.clear()
    st.layouts.clear()
    st.current = None


def _with_axes_arg(fn: Callable) -> Callable:
    """Let a builtin take an axes handle as its first argument, e.g. plot(ax, x, y)."""
    def wrapper(interp, args, nargout):
        if args and isinstance(args[0], GraphicsHandle) and args[0].kind in ("axes", "polaraxes"):
            st = _state(interp)
            prev = st.override_ax
            st.override_ax = args[0].obj
            try:
                return fn(interp, list(args[1:]), nargout)
            finally:
                st.override_ax = prev
        return fn(interp, args, nargout)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# -- argument helpers ------------------------------------------------------------------------

def _split_name_values(args, known: set[str]):
    """Separate Name,Value pairs (e.g. 'LineWidth', 2) whose names are in ``known``."""
    opts = {}
    rest = list(args)
    k = 0
    out = []
    while k < len(rest):
        a = rest[k]
        if is_text(a) and text(a).lower() in known and k + 1 < len(rest):
            opts[text(a).lower()] = rest[k + 1]
            k += 2
            continue
        out.append(a)
        k += 1
    return out, opts


_LINE_OPTS = {"linewidth", "color", "markersize", "displayname", "marker", "linestyle",
              "markerfacecolor", "markeredgecolor", "visible"}

_MARKER_NAMES = {"circle": "o", "square": "s", "diamond": "d", "pentagram": "p", "hexagram": "h",
                 "none": "None", "plus": "+", "asterisk": "*", "point": ".", "cross": "x",
                 "x": "x", "^": "^", "v": "v", "<": "<", ">": ">"}


def _mpl_marker(s: str) -> str:
    return _MARKER_NAMES.get(s.lower(), s)


def _mpl_linestyle(s: str) -> str:
    return "None" if s.lower() == "none" else s


def _scalar(v) -> float:
    return float(np.real(fnum(v)).flat[0])


def _mpl_kwargs(opts) -> dict:
    kw = {}
    for k, v in opts.items():
        if k == "linewidth":
            kw["linewidth"] = _scalar(v)
        elif k == "markersize":
            kw["markersize"] = _scalar(v)
        elif k == "color":
            kw["color"] = _color(v)
        elif k == "displayname":
            kw["label"] = text(v)
        elif k == "marker":
            kw["marker"] = _mpl_marker(text(v))
        elif k == "linestyle":
            kw["linestyle"] = _mpl_linestyle(text(v))
        elif k == "markerfacecolor":
            kw["markerfacecolor"] = _color(v)
        elif k == "markeredgecolor":
            kw["markeredgecolor"] = _color(v)
        elif k == "visible":
            kw["visible"] = _onoff(v)
    return kw


_COLOR_NAMES = {"red": "r", "green": "g", "blue": "b", "cyan": "c", "magenta": "m",
                "yellow": "y", "black": "k", "white": "w", "none": "none"}


def _color(v):
    if is_text(v):
        s = text(v)
        return _COLOR_NAMES.get(s.lower(), s)
    c = np.real(fnum(v)).flatten()
    if c.size not in (3, 4):
        raise MatlabError("Color must be a color name or an RGB triplet such as [1 0 0].")
    if c.max(initial=0) > 1:
        c = c / 255.0
    return tuple(float(x) for x in c)


def _onoff(v) -> bool:
    if is_text(v):
        s = text(v).lower()
        if s in ("on", "true", "yes"):
            return True
        if s in ("off", "false", "no"):
            return False
        raise MatlabError(f"Expected 'on' or 'off', not '{text(v)}'.")
    return bool(np.real(fnum(v)).flat[0])


def _oo(b) -> np.ndarray:
    return char("on" if b else "off")


def _rgb(c) -> Any:
    from matplotlib.colors import to_rgba
    if isinstance(c, str) and c.lower() == "none":
        return char("none")
    arr = np.asarray(c, dtype=object)
    if arr.ndim == 2:  # Nx4 array of colours: take the first
        if arr.shape[0] == 0:
            return char("none")
        c = tuple(float(x) for x in arr[0])
    try:
        rgba = to_rgba(c)
    except (ValueError, TypeError):
        return char(str(c))
    if rgba[3] == 0:
        return char("none")
    return np.array([[round(rgba[0], 4), round(rgba[1], 4), round(rgba[2], 4)]])


def _row(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(1, -1)


def _text_list(v) -> list[str]:
    if isinstance(v, Cell):
        return [to_str(x) if not isinstance(x, np.ndarray) or is_char(x) else _fmt_num(x) for x in v.values_f()]
    if isinstance(v, StringArray):
        return ["" if x is None else str(x) for x in v.data.flatten(order="F")]
    if isinstance(v, MString):
        return [to_str(v)]
    if is_char(v):
        if v.shape[0] > 1:
            return ["".join(r).rstrip() for r in v]
        return [text(v)] if v.size else []
    if isinstance(v, np.ndarray):
        return [_fmt_num(x) for x in np.real(v).flatten()]
    raise MatlabError("Labels must be a cell array of character vectors or a string array.")


def _fmt_num(x) -> str:
    x = float(np.real(np.asarray(x)).flat[0])
    return f"{x:g}"


def _label_text(v) -> str:
    if isinstance(v, Cell):
        return "\n".join(to_str(x) for x in v.values_f())
    if isinstance(v, StringArray):
        return "\n".join(_text_list(v))
    if is_char(v) and v.shape[0] > 1:
        return "\n".join("".join(r).rstrip() for r in v)
    if isinstance(v, np.ndarray) and not is_char(v):
        return " ".join(_fmt_num(x) for x in v.flatten())
    return text(v)


def _is_linespec(s: str) -> bool:
    """True for a line specification such as 'r--o', 'o-', ':' or 'k*'."""
    if not s:
        return False
    seen = set()
    k = 0
    while k < len(s):
        two = s[k:k + 2]
        if two in ("--", "-."):
            cat, k = "style", k + 2
        elif s[k] in "-:":
            cat, k = "style", k + 1
        elif s[k] in "rgbcmykw":
            cat, k = "color", k + 1
        elif s[k] in "o+*.xsd^v<>ph_|":
            cat, k = "marker", k + 1
        else:
            return False
        if cat in seen:
            return False
        seen.add(cat)
    return True


def _col(a) -> np.ndarray:
    return np.arange(1, a + 1, dtype=float).reshape(-1, 1)


def _xy_groups(args):
    """Parse plot(x1,y1,spec1,x2,y2,...) into (x, y, spec) triples (2-D arrays)."""
    groups = []
    k = 0
    while k < len(args):
        a = args[k]
        if is_text(a):
            raise MatlabError(f"Invalid plot argument '{text(a)}'.")
        if isinstance(a, (Cell, Struct)) or isinstance(a, MObject):
            raise MatlabError("Data inputs to plot must be numeric.")
        if k + 1 < len(args) and not is_text(args[k + 1]):
            x, y = fnum(args[k]), fnum(args[k + 1])
            if np.iscomplexobj(x):
                x = np.real(x)
            if np.iscomplexobj(y):
                y = np.real(y)
            k += 2
        else:
            y = fnum(a)
            if y.ndim > 2:
                y = y.reshape(y.shape[0], -1, order="F")
            if np.iscomplexobj(y):
                x, y = np.real(y), np.imag(y)
                if x.ndim == 2 and x.shape[0] == 1:
                    x, y = x.T, y.T
            elif y.shape[0] == 1 or y.shape[1] == 1:
                x = _col(y.size)
                y = y.reshape(-1, 1)
            else:
                x = _col(y.shape[0])
            k += 1
        spec = ""
        if k < len(args) and is_text(args[k]) and _is_linespec(text(args[k])):
            spec = text(args[k])
            k += 1
        groups.append((x, y, spec))
    return groups


def _prep_xy(x: np.ndarray, y: np.ndarray):
    x, y = np.atleast_2d(np.real(x)), np.atleast_2d(np.real(y))
    xvec = x.shape[0] == 1 or x.shape[1] == 1
    yvec = y.shape[0] == 1 or y.shape[1] == 1
    if xvec and yvec:
        if x.size == y.size:
            return x.flatten(), y.flatten()
        if x.size == 1:
            return np.full(y.size, x.flat[0]), y.flatten()
        if y.size == 1:
            return x.flatten(), np.full(x.size, y.flat[0])
        raise MatlabError("Vectors must be the same length.")
    if xvec:
        xv = x.flatten()
        if y.shape[0] == xv.size:
            return xv, y
        if y.shape[1] == xv.size:
            return xv, y.T
        raise MatlabError("Vectors must be the same length.")
    if yvec:
        yv = y.flatten()
        if x.shape[0] == yv.size:
            return x, yv
        if x.shape[1] == yv.size:
            return x.T, yv
        raise MatlabError("Vectors must be the same length.")
    if x.shape != y.shape:
        raise MatlabError("Matrix dimensions must agree.")
    return x, y


def _redraw(interp):
    plt = _plt()
    if plt.isinteractive():
        plt.pause(0.001)


def _ret(interp, kind: str, objs: list, nargout: int):
    """Return handle(s) for ``objs`` when an output was requested."""
    if nargout <= 0:
        return None
    hs = [_wrap(interp, kind, o) for o in objs]
    if len(hs) == 1:
        return hs[0]
    return GraphicsArray(hs)


# -- graphics handle objects ------------------------------------------------------------------

_CLASS = {
    "root": "matlab.ui.Root", "figure": "matlab.ui.Figure", "axes": "matlab.graphics.axis.Axes",
    "polaraxes": "matlab.graphics.axis.PolarAxes",
    "line": "matlab.graphics.chart.primitive.Line", "stair": "matlab.graphics.chart.primitive.Stair",
    "scatter": "matlab.graphics.chart.primitive.Scatter", "bar": "matlab.graphics.chart.primitive.Bar",
    "histogram": "matlab.graphics.chart.primitive.Histogram",
    "surface": "matlab.graphics.chart.primitive.Surface", "contour": "matlab.graphics.chart.primitive.Contour",
    "quiver": "matlab.graphics.chart.primitive.Quiver", "stem": "matlab.graphics.chart.primitive.Stem",
    "errorbar": "matlab.graphics.chart.primitive.ErrorBar", "area": "matlab.graphics.chart.primitive.Area",
    "image": "matlab.graphics.primitive.Image", "patch": "matlab.graphics.primitive.Patch",
    "text": "matlab.graphics.primitive.Text", "legend": "matlab.graphics.illustration.Legend",
    "colorbar": "matlab.graphics.illustration.ColorBar", "pie": "matlab.graphics.primitive.Patch",
    "tiledlayout": "matlab.graphics.layout.TiledChartLayout",
}

_TYPE = {"root": "root", "figure": "figure", "axes": "axes", "polaraxes": "polaraxes", "line": "line",
         "stair": "stair", "scatter": "scatter", "bar": "bar", "histogram": "histogram",
         "surface": "surface", "contour": "contour", "quiver": "quiver", "stem": "stem",
         "errorbar": "errorbar", "area": "area", "image": "image", "patch": "patch", "text": "text",
         "legend": "legend", "colorbar": "colorbar", "pie": "patch", "tiledlayout": "tiledlayout"}


class _Root:
    """Stand-in object for the graphics root (groot, handle 0)."""


_ROOT = _Root()


def _wrap(interp, kind: str, obj) -> "GraphicsHandle":
    st = _state(interp)
    h = st.handles.get(id(obj))
    if h is not None and h.obj is obj:
        return h
    h = GraphicsHandle(interp, kind, obj)
    st.handles[id(obj)] = h
    return h


def _short(kind: str) -> str:
    return _CLASS.get(kind, kind).rsplit(".", 1)[-1]


class _GraphicsBase(MObject):
    is_handle = True

    @staticmethod
    def concat(items: list, axis: int):
        hs: list[GraphicsHandle] = []
        for it in items:
            if isinstance(it, GraphicsHandle):
                hs.append(it)
            elif isinstance(it, GraphicsArray):
                hs.extend(it.items)
            elif isinstance(it, np.ndarray) and it.size == 0:
                continue
            else:
                raise MatlabError("Graphics handles can only be concatenated with other graphics handles.")
        if len(hs) == 1:
            return hs[0]
        return GraphicsArray(hs, row=(axis == 1))


class GraphicsHandle(_GraphicsBase):
    """A reference to one graphics object (figure, axes, line, ...)."""

    def __init__(self, interp, kind: str, obj):
        self.interp = interp
        self.kind = kind
        self.obj = obj
        self.extra: dict[str, Any] = {}

    def class_name(self) -> str:
        return _CLASS.get(self.kind, "matlab.graphics.Graphics")

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), _short(self.kind), "handle", "matlab.graphics.Graphics")

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete":
            for a in args:
                for h in _handles_of(a):
                    _delete_handle(interp, h)
            return []
        if name == "isvalid":
            return [np.array([[_alive(self)]])]
        return super().call_method(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        return _get_prop(self, name)

    def set_field(self, interp, name: str, value: Any) -> "GraphicsHandle":
        _set_prop(self, name, value)
        _redraw(interp)
        return self

    def index_get(self, interp, args: list) -> Any:
        if len(args) == 1 and isinstance(args[0], np.ndarray) and is_char(args[0]) and text(args[0]) == ":":
            return self
        return super().index_get(interp, args)

    def index_set(self, interp, args: list, value: Any) -> Any:
        idx = _single_index(args)
        if idx == 1:
            if isinstance(value, GraphicsHandle):
                return value
        elif idx == 2 and isinstance(value, GraphicsHandle):
            return GraphicsArray([self, value], row=_row_shaped(args))
        raise MatlabError("Index assignment into graphics handles must replace or append one handle.")

    def display_lines(self) -> list[str]:
        if not _alive(self):
            return [f"  handle to deleted {_short(self.kind)}"]
        names = _DISPLAY.get(self.kind, ["Type", "Visible"])
        lines = [f"  {_short(self.kind)} with properties:", ""]
        values = []
        for n in names:
            try:
                values.append((n, _fmt_prop(_get_prop(self, n))))
            except (MatlabError, ValueError, TypeError, AttributeError, IndexError):
                continue
        width = max((len(n) for n, _ in values), default=0)
        lines += [f"    {n.rjust(width)}: {v}" for n, v in values]
        return lines


class GraphicsArray(_GraphicsBase):
    """Several graphics handles, e.g. the lines drawn by one plot call (a column vector)."""

    def __init__(self, items: list[GraphicsHandle], row: bool = False):
        self.items = list(items)
        self.row = row

    def class_name(self) -> str:
        kinds = {h.kind for h in self.items}
        return _CLASS.get(kinds.pop(), "matlab.graphics.Graphics") if len(kinds) == 1 \
            else "matlab.graphics.Graphics"

    def matlab_shape(self) -> tuple:
        return (1, len(self.items)) if self.row else (len(self.items), 1)

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete":
            for h in self.items:
                _delete_handle(interp, h)
            return []
        if name == "isvalid":
            flags = np.array([_alive(h) for h in self.items], dtype=bool)
            return [flags.reshape(self.matlab_shape())]
        return super().call_method(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        return _cell_column([_get_prop(h, name) for h in self.items])

    def set_field(self, interp, name: str, value: Any) -> "GraphicsArray":
        for h in self.items:
            _set_prop(h, name, value)
        _redraw(interp)
        return self

    def _indices(self, args: list) -> list[int]:
        if len(args) == 2:  # h(1, k) / h(k, 1)
            a, b = args
            args = [b] if self.row else [a]
        if len(args) != 1:
            raise MatlabError("Use one index into an array of graphics handles.")
        a = args[0]
        if is_char(a) and text(a) == ":":
            return list(range(len(self.items)))
        v = np.asarray(a)
        if v.dtype == bool:
            return [int(i) for i in np.flatnonzero(v.flatten(order="F"))]
        idx = [int(round(float(x))) - 1 for x in np.real(fnum(a)).flatten(order="F")]
        for i in idx:
            if i < 0 or i >= len(self.items):
                raise MatlabError(f"Index exceeds the number of array elements. Index must not exceed "
                                  f"{len(self.items)}.")
        return idx

    def index_get(self, interp, args: list) -> Any:
        idx = self._indices(args)
        if len(idx) == 1:
            return self.items[idx[0]]
        return GraphicsArray([self.items[i] for i in idx], self.row)

    def index_set(self, interp, args: list, value: Any) -> Any:
        k = _single_index(args)
        if k is None or not isinstance(value, GraphicsHandle):
            raise MatlabError("Index assignment into graphics handles must replace or append one handle.")
        if k <= len(self.items):
            self.items[k - 1] = value
        elif k == len(self.items) + 1:
            self.items.append(value)
        else:
            raise MatlabError("Graphics handle arrays can only grow by one element at a time.")
        return self

    def display_lines(self) -> list[str]:
        r, c = self.matlab_shape()
        lines = [f"  {r}\u00d7{c} {_short(self.items[0].kind) if self.items else 'graphics'} array:", ""]
        for h in self.items:
            lines.append(f"    {_short(h.kind)}" + ("" if _alive(h) else " (deleted)"))
        return lines


def _single_index(args: list) -> int | None:
    vals = []
    for a in args:
        if not isinstance(a, np.ndarray) or a.size != 1 or is_char(a):
            return None
        vals.append(int(round(float(np.real(a.flat[0])))))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2 and 1 in vals:
        return vals[1] if vals[0] == 1 else vals[0]
    return None


def _row_shaped(args: list) -> bool:
    return len(args) == 2 and _single_index(args[:1]) == 1


def _cell_column(vals: list) -> Cell:
    c = Cell.from_list(vals)
    if c.data.shape[0] == 1 and len(vals) > 1:
        c = Cell(c.data.reshape(-1, 1))
    return c


def _handles_of(v) -> list[GraphicsHandle]:
    if isinstance(v, GraphicsHandle):
        return [v]
    if isinstance(v, GraphicsArray):
        return list(v.items)
    return []


def _alive(h: GraphicsHandle) -> bool:
    st = _state(h.interp)
    if h.kind == "root":
        return True
    if h.kind == "figure":
        return any(f is h.obj for f in st.figures.values())
    fig = _obj_figure(h)
    if fig is None or not any(f is fig for f in st.figures.values()):
        return False
    if h.kind in ("axes", "polaraxes"):
        return h.obj in fig.axes
    if h.extra.get("deleted"):
        return False
    return True


def _obj_figure(h: GraphicsHandle):
    o = h.obj
    if h.kind == "histogram":
        return o.ax.figure
    if h.kind == "tiledlayout":
        return o.fig
    if h.kind in ("bar", "errorbar", "stem"):
        for a in _artists(o):
            if getattr(a, "figure", None) is not None:
                return a.figure
        return h.extra.get("figure")
    if h.kind == "colorbar":
        return o.ax.figure
    if h.kind == "legend":
        return o.figure
    return getattr(o, "figure", None) or h.extra.get("figure")


def _obj_axes(h: GraphicsHandle):
    o = h.obj
    if h.kind in ("axes", "polaraxes"):
        return o
    if h.kind == "histogram":
        return o.ax
    if h.kind in ("bar", "errorbar", "stem", "pie"):
        for a in _artists(o):
            if getattr(a, "axes", None) is not None:
                return a.axes
        return h.extra.get("axes")
    if h.kind == "legend":
        return o.axes
    if h.kind == "colorbar":
        return getattr(o, "_mylab_parent_ax", None)
    return getattr(o, "axes", None) or h.extra.get("axes")


def _artists(o) -> list:
    """Flatten a matplotlib container (or list) into its artists."""
    from matplotlib.container import Container
    if isinstance(o, _Hist):
        return _artists(o.container) if o.container is not None else []
    if isinstance(o, (list, tuple, Container)):
        out = []
        for x in o:
            out.extend(_artists(x))
        return out
    return [] if o is None else [o]


def _delete_handle(interp, h: GraphicsHandle) -> None:
    st = _state(interp)
    if h.kind == "figure":
        n = _fig_number(interp, h.obj)
        if n is not None:
            _close_number(interp, n)
        return
    if h.kind in ("axes", "polaraxes"):
        h.obj.remove()
        return
    if h.kind == "tiledlayout":
        h.obj.fig.clf()
        st.layouts.pop(_fig_number(interp, h.obj.fig) or -1, None)
        return
    if h.kind == "colorbar":
        h.obj.remove()
    elif h.kind == "histogram":
        h.obj.remove()
    else:
        for a in _artists(h.obj):
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
    h.extra["deleted"] = True


def _fmt_prop(v) -> str:
    if isinstance(v, GraphicsHandle):
        return f"[1x1 {_short(v.kind)}]"
    if isinstance(v, GraphicsArray):
        r, c = v.matlab_shape()
        return f"[{r}x{c} graphics array]"
    if isinstance(v, Cell):
        vals = v.values_f()
        if len(vals) <= 4 and all(is_char(x) or isinstance(x, MString) for x in vals):
            return "{" + " ".join(f"'{to_str(x)}'" for x in vals) + "}"
        r, c = v.data.shape
        return f"{{{r}x{c} cell}}"
    if is_char(v):
        return f"'{text(v)}'" if v.size else "''"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "[]"
        if v.size <= 6:
            s = " ".join(f"{float(x):.4g}" for x in np.real(v).flatten())
            return s if v.size == 1 else f"[{s}]"
        r, c = v.shape[:2]
        return f"[{r}x{c} double]"
    return str(v)


# -- properties ------------------------------------------------------------------------------

def _str_val(s: str) -> np.ndarray:
    return char(s)


def _num_val(x) -> np.ndarray:
    return np.array([[float(x)]])


def _line_xdata(h):
    return _row(h.obj.get_xdata())


def _set_line_data(h, which: str, v):
    line = h.obj
    data = np.real(fnum(v)).flatten()
    if which == "z":
        if hasattr(line, "set_data_3d"):
            x, y, _ = line.get_data_3d()
            line.set_data_3d(x, y, data)
        return
    if which == "x":
        line.set_xdata(data)
    else:
        line.set_ydata(data)
    ax = line.axes
    if ax is not None and getattr(ax, "name", "") == "rectilinear":
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        if x.size == y.size and x.size:
            ax.update_datalim(np.column_stack([x, y]))
            ax.autoscale_view()


def _line_zdata(h):
    if hasattr(h.obj, "get_data_3d"):
        return _row(h.obj.get_data_3d()[2])
    return np.zeros((1, 0))


def _get_label(o) -> np.ndarray:
    s = o.get_label() if hasattr(o, "get_label") else ""
    return char("" if s is None or s.startswith("_") else s)


def _set_label(h, v):
    s = text(v)
    o = h.obj
    if isinstance(o, _Hist):
        o.label = s
        if o.container is not None:
            o.container.set_label(s)
    else:
        o.set_label(s)
    ax = _obj_axes(h)
    if ax is not None and ax.get_legend() is not None:
        _refresh_legend(h.interp, ax)


def _marker_get(h):
    m = h.obj.get_marker()
    inv = {"None": "none", "": "none", " ": "none"}
    return char(inv.get(str(m), str(m)))


def _ls_get(h):
    s = h.obj.get_linestyle()
    return char("none" if s in ("None", "", " ") else s)


def _mfc_get(h):
    c = h.obj.get_markerfacecolor()
    return _rgb(c)


def _visible_get(h):
    arts = _artists(h.obj) if h.kind not in ("axes", "polaraxes", "figure") else [h.obj]
    if h.kind == "colorbar":
        arts = [h.obj.ax]
    if h.kind == "tiledlayout":
        return _oo(True)
    return _oo(all(a.get_visible() for a in arts) if arts else True)


def _visible_set(h, v):
    b = _onoff(v)
    if h.kind == "colorbar":
        h.obj.ax.set_visible(b)
        return
    if h.kind == "figure":
        h.extra["Visible"] = b
        return
    arts = _artists(h.obj) if h.kind not in ("axes", "polaraxes") else [h.obj]
    for a in arts:
        a.set_visible(b)


def _parent_get(h):
    interp = h.interp
    if h.kind == "figure":
        return _wrap(interp, "root", _ROOT)
    if h.kind in ("axes", "polaraxes"):
        lay = _layout_of_axes(interp, h.obj)
        if lay is not None:
            return _wrap(interp, "tiledlayout", lay)
        return _wrap(interp, "figure", h.obj.figure)
    if h.kind == "tiledlayout":
        return _wrap(interp, "figure", h.obj.fig)
    if h.kind == "root":
        return np.zeros((0, 0))
    if h.kind == "legend" or h.kind == "colorbar":
        return _wrap(interp, "figure", _obj_figure(h))
    ax = _obj_axes(h)
    if ax is not None:
        return _wrap(interp, _axes_kind(ax), ax)
    fig = _obj_figure(h)
    return _wrap(interp, "figure", fig) if fig is not None else np.zeros((0, 0))


def _axes_kind(ax) -> str:
    return "polaraxes" if getattr(ax, "name", "") == "polar" else "axes"


def _children_get(h):
    interp = h.interp
    out: list[GraphicsHandle] = []
    if h.kind == "figure":
        out = [_wrap(interp, _axes_kind(a), a) for a in h.obj.axes if not _is_colorbar_axes(a)]
    elif h.kind in ("axes", "polaraxes"):
        ax = h.obj
        for a in ax.lines:
            out.append(_wrap(interp, "line", a))
        for c in ax.collections:
            out.append(_wrap(interp, "scatter" if c.__class__.__name__ == "PathCollection" else "patch", c))
        for im in ax.images:
            out.append(_wrap(interp, "image", im))
        for t in ax.texts:
            out.append(_wrap(interp, "text", t))
        out.reverse()  # newest first, as MATLAB lists them
    elif h.kind == "root":
        st = _state(interp)
        out = [_wrap(interp, "figure", st.figures[n]) for n in sorted(st.figures, reverse=True)]
    if not out:
        return np.zeros((0, 1))
    return out[0] if len(out) == 1 else GraphicsArray(out)


# line-like objects (Line2D)
_LINE_PROPS: dict[str, tuple] = {
    "Color": (lambda h: _rgb(h.obj.get_color()), lambda h, v: h.obj.set_color(_color(v))),
    "LineWidth": (lambda h: _num_val(h.obj.get_linewidth()), lambda h, v: h.obj.set_linewidth(_scalar(v))),
    "LineStyle": (_ls_get, lambda h, v: h.obj.set_linestyle(_mpl_linestyle(text(v)))),
    "Marker": (_marker_get, lambda h, v: h.obj.set_marker(_mpl_marker(text(v)))),
    "MarkerSize": (lambda h: _num_val(h.obj.get_markersize()), lambda h, v: h.obj.set_markersize(_scalar(v))),
    "MarkerFaceColor": (_mfc_get, lambda h, v: h.obj.set_markerfacecolor(_color(v))),
    "MarkerEdgeColor": (lambda h: _rgb(h.obj.get_markeredgecolor()),
                        lambda h, v: h.obj.set_markeredgecolor(_color(v))),
    "XData": (_line_xdata, lambda h, v: _set_line_data(h, "x", v)),
    "YData": (lambda h: _row(h.obj.get_ydata()), lambda h, v: _set_line_data(h, "y", v)),
    "ZData": (_line_zdata, lambda h, v: _set_line_data(h, "z", v)),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}


def _sc_offsets(h):
    return np.asarray(h.obj.get_offsets(), dtype=float).reshape(-1, 2)


def _sc_set_offsets(h, which, v):
    off = _sc_offsets(h)
    data = np.real(fnum(v)).flatten()
    if data.size != off.shape[0]:
        off = np.zeros((data.size, 2)) if off.shape[0] == 0 else np.resize(off, (data.size, 2))
    off[:, 0 if which == "x" else 1] = data
    h.obj.set_offsets(off)
    ax = h.obj.axes
    if ax is not None:
        ax.update_datalim(off)
        ax.autoscale_view()


def _sc_cdata(h):
    arr = h.obj.get_array()
    if arr is not None:
        return _row(arr)
    fc = h.obj.get_facecolor()
    return _rgb(fc if len(fc) else h.obj.get_edgecolor())


def _sc_set_cdata(h, v):
    if is_text(v) or (isinstance(v, np.ndarray) and v.size in (3, 4) and h.obj.get_array() is None):
        c = _color(v)
        h.obj.set_edgecolor(c)
        if len(h.obj.get_facecolor()):
            h.obj.set_facecolor(c)
    else:
        h.obj.set_array(np.real(fnum(v)).flatten())


_SCATTER_PROPS: dict[str, tuple] = {
    "XData": (lambda h: _row(_sc_offsets(h)[:, 0]), lambda h, v: _sc_set_offsets(h, "x", v)),
    "YData": (lambda h: _row(_sc_offsets(h)[:, 1]), lambda h, v: _sc_set_offsets(h, "y", v)),
    "SizeData": (lambda h: _row(h.obj.get_sizes()), lambda h, v: h.obj.set_sizes(np.real(fnum(v)).flatten())),
    "CData": (_sc_cdata, _sc_set_cdata),
    "MarkerFaceColor": (lambda h: _rgb(h.obj.get_facecolor()) if len(h.obj.get_facecolor()) else char("none"),
                        lambda h, v: h.obj.set_facecolor(_color(v))),
    "MarkerEdgeColor": (lambda h: _rgb(h.obj.get_edgecolor()), lambda h, v: h.obj.set_edgecolor(_color(v))),
    "LineWidth": (lambda h: _num_val(np.asarray(h.obj.get_linewidths()).flat[0]),
                  lambda h, v: h.obj.set_linewidths(_scalar(v))),
    "Marker": (lambda h: char(h.extra.get("Marker", "o")), None),
    "MarkerFaceAlpha": (lambda h: _num_val(1 if h.obj.get_alpha() is None else h.obj.get_alpha()),
                        lambda h, v: h.obj.set_alpha(_scalar(v))),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}


def _patches(h) -> list:
    return [a for a in _artists(h.obj) if hasattr(a, "set_facecolor")]


def _patch_color(h, which: str):
    ps = _patches(h)
    if not ps:
        return char("none")
    return _rgb(getattr(ps[0], f"get_{which}color")())


def _set_patch_color(h, which: str, v):
    c = _color(v) if not (is_text(v) and text(v).lower() in ("flat", "auto")) else None
    if c is None:
        return
    for p in _patches(h):
        getattr(p, f"set_{which}color")(c)
    if isinstance(h.obj, _Hist):
        h.obj.style[f"{which}color"] = c


def _bar_width_set(h, v):
    w = _scalar(v)
    old = h.extra.get("BarWidth", 0.8)
    for p in _patches(h):
        if hasattr(p, "get_width"):
            cx = p.get_x() + p.get_width() / 2
            nw = p.get_width() * w / old
            p.set_width(nw)
            p.set_x(cx - nw / 2)
    h.extra["BarWidth"] = w


_BAR_PROPS: dict[str, tuple] = {
    "FaceColor": (lambda h: _patch_color(h, "face"), lambda h, v: _set_patch_color(h, "face", v)),
    "EdgeColor": (lambda h: _patch_color(h, "edge"), lambda h, v: _set_patch_color(h, "edge", v)),
    "BarWidth": (lambda h: _num_val(h.extra.get("BarWidth", 0.8)), _bar_width_set),
    "YData": (lambda h: _row(h.extra.get("YData", [])), None),
    "XData": (lambda h: _row(h.extra.get("XData", [])), None),
    "FaceAlpha": (lambda h: _num_val(_patches(h)[0].get_alpha() or 1) if _patches(h) else _num_val(1),
                  lambda h, v: [p.set_alpha(_scalar(v)) for p in _patches(h)]),
    "LineWidth": (lambda h: _num_val(_patches(h)[0].get_linewidth()) if _patches(h) else _num_val(0.5),
                  lambda h, v: [p.set_linewidth(_scalar(v)) for p in _patches(h)]),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}


class _Hist:
    """State of a histogram plot: data, bin edges and the bars drawing it."""

    def __init__(self, ax, data, edges, normalization="count", orientation="vertical", style=None,
                 label=None):
        self.ax = ax
        self.data = data
        self.edges = np.asarray(edges, dtype=float)
        self.normalization = normalization
        self.orientation = orientation
        self.style = dict(style or {})
        self.label = label
        self.container = None
        self.alpha = self.style.pop("alpha", 0.6)

    def values(self) -> np.ndarray:
        counts, _ = np.histogram(self.data, self.edges)
        counts = counts.astype(float)
        n = max(self.data.size, 1)
        widths = np.diff(self.edges)
        norm = self.normalization.lower()
        if norm == "probability":
            return counts / n
        if norm == "percentage":
            return 100 * counts / n
        if norm == "countdensity":
            return counts / widths
        if norm == "pdf":
            return counts / (n * widths)
        if norm == "cumcount":
            return np.cumsum(counts)
        if norm == "cdf":
            return np.cumsum(counts) / n
        return counts

    def remove(self):
        if self.container is not None:
            for p in self.container:
                try:
                    p.remove()
                except (ValueError, NotImplementedError):
                    pass
            if self.container in self.ax.containers:
                self.ax.containers.remove(self.container)
            self.container = None

    def draw(self):
        from matplotlib.colors import to_rgba
        self.remove()
        vals = self.values()
        widths = np.diff(self.edges)
        kw = {"edgecolor": self.style.get("edgecolor", "black"), "linewidth": self.style.get("linewidth", 0.5)}
        if "facecolor" in self.style:
            kw["color"] = self.style["facecolor"]
        if self.label:
            kw["label"] = self.label
        if self.orientation == "horizontal":
            cont = self.ax.barh(self.edges[:-1], vals, height=widths, align="edge", **kw)
        else:
            cont = self.ax.bar(self.edges[:-1], vals, width=widths, align="edge", **kw)
        if cont.patches:
            fc = to_rgba(cont.patches[0].get_facecolor())
            self.style.setdefault("facecolor", fc[:3])
            for p in cont.patches:
                p.set_facecolor((*to_rgba(self.style["facecolor"])[:3], self.alpha))
        self.container = cont


def _hist_redraw(h, **changes):
    hs: _Hist = h.obj
    for k, v in changes.items():
        setattr(hs, k, v)
    hs.draw()
    hs.ax.relim()
    hs.ax.autoscale_view()


def _hist_set_nbins(h, v):
    hs: _Hist = h.obj
    _hist_redraw(h, edges=_hist_edges(hs.data, nbins=to_int(v, "NumBins")))


def _hist_set_width(h, v):
    hs: _Hist = h.obj
    _hist_redraw(h, edges=_hist_edges(hs.data, width=_scalar(v)))


def _hist_set_alpha(h, v):
    h.obj.alpha = _scalar(v)
    _hist_redraw(h)


_HIST_PROPS: dict[str, tuple] = {
    "Data": (lambda h: h.extra.get("DataShape", lambda: _row(h.obj.data))(),
             lambda h, v: _hist_redraw(h, data=_finite(v))),
    "Values": (lambda h: _row(h.obj.values()), None),
    "BinCounts": (lambda h: _row(np.histogram(h.obj.data, h.obj.edges)[0].astype(float)), None),
    "BinEdges": (lambda h: _row(h.obj.edges), lambda h, v: _hist_redraw(h, edges=np.real(fnum(v)).flatten())),
    "BinWidth": (lambda h: _num_val(np.diff(h.obj.edges)[0]) if np.allclose(np.diff(h.obj.edges),
                                                                             np.diff(h.obj.edges)[0])
                 else char("nonuniform"), _hist_set_width),
    "NumBins": (lambda h: _num_val(len(h.obj.edges) - 1), _hist_set_nbins),
    "BinLimits": (lambda h: _row([h.obj.edges[0], h.obj.edges[-1]]),
                  lambda h, v: _hist_redraw(h, edges=_hist_edges(h.obj.data, nbins=len(h.obj.edges) - 1,
                                                                 limits=np.real(fnum(v)).flatten()))),
    "Normalization": (lambda h: char(h.obj.normalization),
                      lambda h, v: _hist_redraw(h, normalization=_check_norm(text(v)))),
    "Orientation": (lambda h: char(h.obj.orientation), lambda h, v: _hist_redraw(h, orientation=text(v).lower())),
    "FaceColor": (lambda h: _rgb(h.obj.style.get("facecolor", "C0")),
                  lambda h, v: (h.obj.style.__setitem__("facecolor", _color(v)), _hist_redraw(h))),
    "EdgeColor": (lambda h: _rgb(h.obj.style.get("edgecolor", "black")),
                  lambda h, v: (h.obj.style.__setitem__("edgecolor", _color(v)), _hist_redraw(h))),
    "FaceAlpha": (lambda h: _num_val(h.obj.alpha), _hist_set_alpha),
    "LineWidth": (lambda h: _num_val(h.obj.style.get("linewidth", 0.5)),
                  lambda h, v: (h.obj.style.__setitem__("linewidth", _scalar(v)), _hist_redraw(h))),
    "DisplayName": (lambda h: char(h.obj.label or ""), _set_label),
}


_NORMS = ("count", "probability", "percentage", "countdensity", "pdf", "cumcount", "cdf")


def _check_norm(s: str) -> str:
    if s.lower() not in _NORMS:
        raise MatlabError(f"Normalization must be one of: {', '.join(_NORMS)}.")
    return s.lower()


def _finite(v) -> np.ndarray:
    x = np.real(fnum(v)).flatten(order="F")
    return x[np.isfinite(x)]


def _nice_width(raw: float) -> float:
    if raw <= 0 or not np.isfinite(raw):
        return 1.0
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= raw:
            return m * p
    return 10 * p


def _hist_edges(x, nbins=None, edges=None, width=None, limits=None, method=None) -> np.ndarray:
    if edges is not None:
        return np.asarray(edges, dtype=float).flatten()
    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    elif x.size:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        lo, hi = 0.0, 1.0
    if method == "integers":
        return np.arange(math.floor(lo) - 0.5, math.ceil(hi) + 1.0, 1.0)
    if width is not None:
        if width <= 0:
            raise MatlabError("BinWidth must be positive.")
        if limits is not None:
            n = max(int(math.ceil((hi - lo) / width - 1e-9)), 1)
            return lo + width * np.arange(n + 1)
        start = math.floor(lo / width) * width
        n = max(int(math.ceil((hi - start) / width - 1e-9)), 1)
        e = start + width * np.arange(n + 1)
        if e[-1] <= hi and x.size:
            e = np.append(e, e[-1] + width)
        return e
    if nbins is not None:
        if nbins < 1:
            raise MatlabError("The number of bins must be a positive integer.")
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5
        return np.linspace(lo, hi, nbins + 1)
    if x.size == 0:
        return np.array([0.0, 1.0])
    if hi == lo:
        return np.array([lo - 0.5, lo + 0.5])
    # automatic: Scott-like bin width rounded to a "nice" number
    n = x.size
    sd = float(np.std(x))
    raw = 3.5 * sd / n ** (1 / 3) if sd > 0 else (hi - lo) / 10
    raw = max(raw, (hi - lo) / 50)
    w = _nice_width(raw)
    if np.all(x == np.round(x)) and hi - lo <= 50:  # integer data: unit-width bins centred on integers
        w = max(1.0, round(w))
        return np.arange(math.floor(lo) - 0.5, math.ceil(hi) + w, w)
    start = math.floor(lo / w) * w
    stop = math.ceil(hi / w) * w
    if stop <= hi:
        stop += w
    k = int(round((stop - start) / w))
    return start + w * np.arange(k + 1)


# axes ---------------------------------------------------------------------------------------

def _lim_get(which):
    def g(h):
        f = getattr(h.obj, f"get_{which}lim", None)
        if f is None:
            return np.zeros((1, 0))
        lo, hi = f()
        return _row(sorted([lo, hi]))
    return g


def _lim_set(which):
    def s(h, v):
        lim = np.real(fnum(v)).flatten()
        _set_lim(h.obj, which, lim[0], lim[1])
    return s


def _set_lim(ax, which: str, lo: float, hi: float) -> None:
    """Set axis limits; an infinite end keeps that side automatic."""
    setter = getattr(ax, f"set_{which}lim", None)
    getter = getattr(ax, f"get_{which}lim", None)
    if setter is None or getter is None:
        return
    if not (np.isfinite(lo) and np.isfinite(hi)):
        if which in ("x", "y"):
            getattr(ax, f"set_autoscale{which}_on")(True)
            ax.autoscale_view(scalex=(which == "x"), scaley=(which == "y"))
        auto = sorted(getter())
        if np.isfinite(lo) and lo >= auto[1]:
            auto[1] = lo + 1
        if np.isfinite(hi) and hi <= auto[0]:
            auto[0] = hi - 1
        lo = auto[0] if not np.isfinite(lo) else lo
        hi = auto[1] if not np.isfinite(hi) else hi
    if lo >= hi:
        raise MatlabError("Limits must be a 2-element vector of increasing numeric values.")
    inverted = getattr(ax, f"{which}axis_inverted", lambda: False)()
    if inverted:
        setter(hi, lo)
    else:
        setter(lo, hi)


def _title_text(h, which: str):
    ax = h.obj
    if which == "Title":
        return _wrap(h.interp, "text", ax.title)
    if which == "Subtitle":
        t = getattr(ax, "_mylab_subtitle", None)
        if t is None:
            t = _make_subtitle(ax, "")
        return _wrap(h.interp, "text", t)
    axis = {"XLabel": "xaxis", "YLabel": "yaxis", "ZLabel": "zaxis"}[which]
    a = getattr(ax, axis, None)
    if a is None:
        raise MatlabError(f"These axes have no {which}.")
    return _wrap(h.interp, "text", a.label)


def _title_set(which: str):
    def s(h, v):
        if isinstance(v, GraphicsHandle):
            return
        _get_prop(_title_text(h, which), "String")  # make sure it exists
        _set_prop(_title_text(h, which), "String", v)
    return s


def _fontsize_set(h, v):
    fs = _scalar(v)
    ax = h.obj
    h.extra["FontSize"] = fs
    ax.tick_params(labelsize=fs)
    ax.title.set_fontsize(fs * 1.1)
    ax.xaxis.label.set_fontsize(fs * 1.1)
    ax.yaxis.label.set_fontsize(fs * 1.1)
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_fontsize(fs * 1.1)


def _fontsize_get(h):
    import matplotlib as mpl
    if "FontSize" in h.extra:
        return _num_val(h.extra["FontSize"])
    labels = h.obj.xaxis.get_ticklabels()
    if labels:
        return _num_val(labels[0].get_fontsize())
    return _num_val(mpl.rcParams["font.size"])


def _box_get(h):
    sp = h.obj.spines
    return _oo(sp["top"].get_visible()) if "top" in sp else _oo(True)


def _box_set(h, v):
    b = _onoff(v)
    for name in ("top", "right"):
        if name in h.obj.spines:
            h.obj.spines[name].set_visible(b)


def _grid_get(axis_name):
    def g(h):
        axis = getattr(h.obj, axis_name)
        lines = axis.get_gridlines()
        return _oo(any(gl.get_visible() for gl in lines))
    return g


def _grid_set(axis_name, minor=False):
    def s(h, v):
        b = _onoff(v)
        axis = getattr(h.obj, axis_name)
        if minor:
            if b:
                axis.set_minor_locator(_auto_minor())
            axis.grid(b, which="minor", alpha=0.3)
        else:
            axis.grid(b, which="major")
    return s


def _minorgrid_set(which):
    def s(h, v):
        _grid_set(f"{which}axis", True)(h, v)
        h.extra[f"{which.upper()}MinorGrid"] = "on" if _onoff(v) else "off"
    return s


def _auto_minor():
    from matplotlib.ticker import AutoMinorLocator
    return AutoMinorLocator()


def _ticks_get(which):
    def g(h):
        ax = h.obj
        ticks = np.asarray(getattr(ax, f"get_{which}ticks")(), float)
        lo, hi = sorted(getattr(ax, f"get_{which}lim")())
        ticks = ticks[(ticks >= lo - 1e-12 * max(1, abs(lo))) & (ticks <= hi + 1e-12 * max(1, abs(hi)))]
        return _row(ticks)
    return g


def _ticks_set(which):
    def s(h, v):
        if is_text(v):
            _set_ticks_auto(h.obj, which)
            return
        getattr(h.obj, f"set_{which}ticks")(np.real(fnum(v)).flatten())
    return s


def _set_ticks_auto(ax, which):
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    axis = getattr(ax, f"{which}axis")
    axis.set_major_locator(AutoLocator())
    axis.set_major_formatter(ScalarFormatter())


def _ticklabels_get(which):
    def g(h):
        ax = h.obj
        ticks = _ticks_get(which)(h).flatten()
        fixed = getattr(ax, f"_mylab_{which}ticklabels", None)
        if fixed is not None:
            return _cell_column([char(s) for s in fixed])
        labels = [f"{t:g}" for t in ticks]
        return _cell_column([char(s) for s in labels]) if labels else Cell.from_list([])
    return g


def _ticklabels_set(which):
    def s(h, v):
        _set_ticklabels(h.obj, which, v)
    return s


def _set_ticklabels(ax, which: str, v) -> None:
    if is_text(v) and text(v).lower() in ("auto", "mode"):
        _set_ticks_auto(ax, which)
        setattr(ax, f"_mylab_{which}ticklabels", None)
        return
    labels = _text_list(v)
    ticks = list(_ticks_get(which)(_FakeAxesHandle(ax)).flatten())
    if len(ticks) < len(labels) and not getattr(ax, f"_mylab_{which}ticks_fixed", False):
        labels = labels[:len(ticks)]
    getattr(ax, f"set_{which}ticks")(ticks)
    full = (labels + [""] * len(ticks))[:len(ticks)]
    getattr(ax, f"set_{which}ticklabels")(full)
    setattr(ax, f"_mylab_{which}ticklabels", full)


class _FakeAxesHandle:
    def __init__(self, ax):
        self.obj = ax
        self.extra: dict = {}


def _dir_get(which):
    def g(h):
        return char("reverse" if getattr(h.obj, f"{which}axis_inverted")() else "normal")
    return g


def _dir_set(which):
    def s(h, v):
        want = text(v).lower() == "reverse"
        if getattr(h.obj, f"{which}axis_inverted")() != want:
            getattr(h.obj, f"invert_{which}axis")()
    return s


def _scale_set(which):
    def s(h, v):
        getattr(h.obj, f"set_{which}scale")(text(v).lower())
    return s


def _scale_get(which):
    def g(h):
        sc = getattr(h.obj, f"get_{which}scale")()
        return char("log" if sc == "log" else "linear")
    return g


def _nextplot_get(h):
    st = _state(h.interp)
    return char("add" if st.hold.get(id(h.obj), False) else "replace")


def _nextplot_set(h, v):
    _state(h.interp).hold[id(h.obj)] = text(v).lower() == "add"


def _legend_prop_get(h):
    leg = h.obj.get_legend()
    return _wrap(h.interp, "legend", leg) if leg is not None else np.zeros((0, 0))


def _cmap_prop_get(h):
    return _cmap_query(h.interp)


def _cmap_prop_set(h, v):
    _apply_colormap(h.interp, v, axes=[h.obj])


def _mappables(ax) -> list:
    return [m for m in list(ax.collections) + list(ax.images) if getattr(m, "get_array", lambda: None)()
            is not None]


def _clim_get(h):
    ms = _mappables(h.obj)
    if ms:
        return _row(ms[-1].get_clim())
    return _row([0, 1])


def _clim_set(h, v):
    lim = np.real(fnum(v)).flatten()
    for m in _mappables(h.obj):
        m.set_clim(lim[0], lim[1])


def _view_get(h):
    ax = h.obj
    if hasattr(ax, "azim"):
        return _row([ax.azim, ax.elev])
    return _row([0, 90])


def _view_set(h, v):
    vals = np.real(fnum(v)).flatten()
    if hasattr(h.obj, "view_init"):
        if vals.size == 1 and vals[0] == 2:
            h.obj.view_init(elev=90, azim=-90)
        elif vals.size == 1 and vals[0] == 3:
            h.obj.view_init(elev=30, azim=-60)
        elif vals.size >= 2:
            h.obj.view_init(elev=float(vals[1]), azim=float(vals[0]))


def _axcolor_set(which):
    def s(h, v):
        c = _color(v)
        ax = h.obj
        axis = getattr(ax, f"{which}axis")
        ax.tick_params(axis=which, colors=c)
        axis.label.set_color(c)
        for name in (("bottom", "top") if which == "x" else ("left", "right")):
            if name in ax.spines:
                ax.spines[name].set_color(c)
    return s


def _axcolor_get(which):
    def g(h):
        return _rgb(getattr(h.obj, f"{which}axis").label.get_color())
    return g


def _axisloc_set(which):
    def s(h, v):
        loc = text(v).lower()
        axis = getattr(h.obj, f"{which}axis")
        if which == "x":
            axis.tick_top() if loc == "top" else axis.tick_bottom()
            axis.set_label_position("top" if loc == "top" else "bottom")
        else:
            axis.tick_right() if loc == "right" else axis.tick_left()
            axis.set_label_position("right" if loc == "right" else "left")
        h.extra[f"{which.upper()}AxisLocation"] = loc
    return s


def _minortick_set(which):
    def s(h, v):
        axis = getattr(h.obj, f"{which}axis")
        if _onoff(v):
            axis.set_minor_locator(_auto_minor())
        else:
            from matplotlib.ticker import NullLocator
            axis.set_minor_locator(NullLocator())
    return s


def _position_get(h):
    b = h.obj.get_position().bounds
    return _row(b)


def _position_set(h, v):
    p = np.real(fnum(v)).flatten()
    h.obj.set_position([float(x) for x in p[:4]])


def _children_axes_lw_set(h, v):
    for sp in h.obj.spines.values():
        sp.set_linewidth(_scalar(v))
    h.obj.tick_params(width=_scalar(v))


def _tickdir_set(h, v):
    d = text(v).lower()
    h.obj.tick_params(direction={"both": "inout"}.get(d, d))
    h.extra["TickDir"] = d


def _fontname_set(h, v):
    name = text(v)
    ax = h.obj
    for lab in ax.get_xticklabels() + ax.get_yticklabels() + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        lab.set_fontfamily(name)
    h.extra["FontName"] = name


def _rotation_get(which):
    def g(h):
        labels = getattr(h.obj, f"get_{which}ticklabels")()
        return _num_val(labels[0].get_rotation() if labels else 0)
    return g


def _rotation_set(which):
    def s(h, v):
        h.obj.tick_params(axis=which, labelrotation=_scalar(v))
    return s


_AXES_PROPS: dict[str, tuple] = {
    "XLim": (_lim_get("x"), _lim_set("x")),
    "YLim": (_lim_get("y"), _lim_set("y")),
    "ZLim": (_lim_get("z"), _lim_set("z")),
    "Title": (lambda h: _title_text(h, "Title"), _title_set("Title")),
    "Subtitle": (lambda h: _title_text(h, "Subtitle"), _title_set("Subtitle")),
    "XLabel": (lambda h: _title_text(h, "XLabel"), _title_set("XLabel")),
    "YLabel": (lambda h: _title_text(h, "YLabel"), _title_set("YLabel")),
    "ZLabel": (lambda h: _title_text(h, "ZLabel"), _title_set("ZLabel")),
    "FontSize": (_fontsize_get, _fontsize_set),
    "FontName": (lambda h: char(h.extra.get("FontName", "Helvetica")), _fontname_set),
    "XScale": (_scale_get("x"), _scale_set("x")),
    "YScale": (_scale_get("y"), _scale_set("y")),
    "ZScale": (lambda h: char(h.obj.get_zscale() if hasattr(h.obj, "get_zscale") else "linear"), None),
    "Box": (_box_get, _box_set),
    "XGrid": (_grid_get("xaxis"), _grid_set("xaxis")),
    "YGrid": (_grid_get("yaxis"), _grid_set("yaxis")),
    "ZGrid": (lambda h: _oo(False), lambda h, v: h.obj.grid(_onoff(v))),
    "XMinorGrid": (lambda h: char(h.extra.get("XMinorGrid", "off")), _minorgrid_set("x")),
    "YMinorGrid": (lambda h: char(h.extra.get("YMinorGrid", "off")), _minorgrid_set("y")),
    "XMinorTick": (lambda h: _oo(False), _minortick_set("x")),
    "YMinorTick": (lambda h: _oo(False), _minortick_set("y")),
    "XTick": (_ticks_get("x"), _ticks_set("x")),
    "YTick": (_ticks_get("y"), _ticks_set("y")),
    "ZTick": (lambda h: _row(h.obj.get_zticks()) if hasattr(h.obj, "get_zticks") else np.zeros((1, 0)),
              lambda h, v: h.obj.set_zticks(np.real(fnum(v)).flatten()) if hasattr(h.obj, "set_zticks") else None),
    "XTickLabel": (_ticklabels_get("x"), _ticklabels_set("x")),
    "YTickLabel": (_ticklabels_get("y"), _ticklabels_set("y")),
    "XTickLabelRotation": (_rotation_get("x"), _rotation_set("x")),
    "YTickLabelRotation": (_rotation_get("y"), _rotation_set("y")),
    "XDir": (_dir_get("x"), _dir_set("x")),
    "YDir": (_dir_get("y"), _dir_set("y")),
    "XColor": (_axcolor_get("x"), _axcolor_set("x")),
    "YColor": (_axcolor_get("y"), _axcolor_set("y")),
    "XAxisLocation": (lambda h: char(h.extra.get("XAxisLocation", "bottom")), _axisloc_set("x")),
    "YAxisLocation": (lambda h: char(h.extra.get("YAxisLocation", "left")), _axisloc_set("y")),
    "Color": (lambda h: _rgb(h.obj.get_facecolor()), lambda h, v: h.obj.set_facecolor(_color(v))),
    "LineWidth": (lambda h: _num_val(next(iter(h.obj.spines.values())).get_linewidth()), _children_axes_lw_set),
    "TickDir": (lambda h: char(h.extra.get("TickDir", "in")), _tickdir_set),
    "NextPlot": (_nextplot_get, _nextplot_set),
    "Legend": (_legend_prop_get, None),
    "Colormap": (_cmap_prop_get, _cmap_prop_set),
    "CLim": (_clim_get, _clim_set),
    "View": (_view_get, _view_set),
    "Position": (_position_get, _position_set),
    "InnerPosition": (_position_get, _position_set),
    "Children": (_children_get, None),
}

# Accepted and remembered, but without a visible effect in mylab.
_AXES_IGNORED = {"TickLabelInterpreter", "Layer", "TickLength", "GridAlpha", "MinorGridAlpha", "GridColor",
                 "GridLineStyle", "MinorGridLineStyle", "FontWeight", "Units", "ActivePositionProperty",
                 "PositionConstraint", "LabelFontSizeMultiplier", "TitleFontSizeMultiplier", "TitleFontWeight",
                 "DataAspectRatio", "PlotBoxAspectRatio", "ColorOrder", "LineStyleOrder", "SortMethod",
                 "OuterPosition", "Clipping", "ColorOrderIndex", "LineStyleOrderIndex", "ZMinorTick",
                 "ZMinorGrid", "XLimMode", "YLimMode", "ZLimMode", "TickLabelFormat", "FontAngle",
                 "FontSmoothing", "XTickMode", "YTickMode", "XTickLabelMode", "YTickLabelMode", "ZDir",
                 "ZColor", "ZTickLabel", "CLimMode", "DataAspectRatioMode", "PlotBoxAspectRatioMode",
                 "Projection", "AmbientLightColor", "HitTest", "PickableParts", "ButtonDownFcn", "Interruptible",
                 "BusyAction", "HandleVisibility", "Toolbar", "Interactions", "AlphaScale", "ColorScale"}


# figures ------------------------------------------------------------------------------------

def _fig_name_get(h):
    return char(getattr(h.obj, "_mylab_name", ""))


def _fig_name_set(h, v):
    name = text(v)
    h.obj._mylab_name = name
    mgr = h.obj.canvas.manager
    if mgr is not None:
        try:
            mgr.set_window_title(name)
        except (AttributeError, NotImplementedError):
            pass


def _fig_pos_get(h):
    w, hh = h.obj.get_size_inches() * h.obj.dpi
    left, bottom = getattr(h.obj, "_mylab_origin", (100.0, 100.0))
    return _row([left, bottom, round(w), round(hh)])


def _fig_pos_set(h, v):
    p = np.real(fnum(v)).flatten()
    if p.size < 4:
        raise MatlabError("Position must be a 4-element vector [left bottom width height].")
    units = h.extra.get("Units", "pixels")
    if units == "normalized":
        w, hh = p[2] * 1920, p[3] * 1080
    elif units == "inches":
        w, hh = p[2] * h.obj.dpi, p[3] * h.obj.dpi
    elif units == "centimeters":
        w, hh = p[2] / 2.54 * h.obj.dpi, p[3] / 2.54 * h.obj.dpi
    else:
        w, hh = p[2], p[3]
    if w <= 0 or hh <= 0:
        raise MatlabError("Figure width and height must be positive.")
    h.obj._mylab_origin = (float(p[0]), float(p[1]))
    h.obj.set_size_inches(w / h.obj.dpi, hh / h.obj.dpi, forward=True)


def _fig_current_axes(h):
    st = _state(h.interp)
    n = _fig_number(h.interp, h.obj)
    ax = st.axes.get(n)
    if ax is None or ax not in h.obj.axes:
        return np.zeros((0, 0))
    return _wrap(h.interp, _axes_kind(ax), ax)


_FIGURE_PROPS: dict[str, tuple] = {
    "Number": (lambda h: _num_val(_fig_number(h.interp, h.obj) or 0), None),
    "Name": (_fig_name_get, _fig_name_set),
    "Color": (lambda h: _rgb(h.obj.get_facecolor()), lambda h, v: h.obj.set_facecolor(_color(v))),
    "Position": (_fig_pos_get, _fig_pos_set),
    "OuterPosition": (_fig_pos_get, _fig_pos_set),
    "InnerPosition": (_fig_pos_get, _fig_pos_set),
    "CurrentAxes": (_fig_current_axes, None),
    "Children": (_children_get, None),
    "Colormap": (lambda h: _cmap_query(h.interp),
                 lambda h, v: _apply_colormap(h.interp, v, axes=list(h.obj.axes))),
}

_FIGURE_IGNORED = {"Units", "WindowState", "MenuBar", "ToolBar", "NumberTitle", "PaperPositionMode",
                   "PaperUnits", "PaperPosition", "PaperSize", "PaperOrientation", "Resize", "InvertHardcopy",
                   "Renderer", "RendererMode", "DockControls", "WindowStyle", "IntegerHandle",
                   "HandleVisibility", "CloseRequestFcn", "SizeChangedFcn", "KeyPressFcn", "Pointer",
                   "GraphicsSmoothing", "Theme"}


# text ---------------------------------------------------------------------------------------

def _text_string_get(h):
    s = h.obj.get_text()
    if "\n" in s:
        return _cell_column([char(x) for x in s.split("\n")])
    return char(s)


def _text_string_set(h, v):
    h.obj.set_text(_label_text(v))
    sub = getattr(h.obj, "_mylab_is_subtitle_of", None)
    if sub is not None:
        _layout_subtitle(sub)


_TEXT_PROPS: dict[str, tuple] = {
    "String": (_text_string_get, _text_string_set),
    "FontSize": (lambda h: _num_val(h.obj.get_fontsize()), lambda h, v: h.obj.set_fontsize(_scalar(v))),
    "FontWeight": (lambda h: char("bold" if str(h.obj.get_fontweight()) in ("bold", "700") else "normal"),
                   lambda h, v: h.obj.set_fontweight(text(v).lower())),
    "FontAngle": (lambda h: char("italic" if h.obj.get_fontstyle() == "italic" else "normal"),
                  lambda h, v: h.obj.set_fontstyle("italic" if text(v).lower() == "italic" else "normal")),
    "FontName": (lambda h: char(h.obj.get_fontname()), lambda h, v: h.obj.set_fontfamily(text(v))),
    "Color": (lambda h: _rgb(h.obj.get_color()), lambda h, v: h.obj.set_color(_color(v))),
    "Position": (lambda h: _row(list(h.obj.get_position()) + [0]),
                 lambda h, v: h.obj.set_position(tuple(float(x) for x in np.real(fnum(v)).flatten()[:2]))),
    "Rotation": (lambda h: _num_val(h.obj.get_rotation()), lambda h, v: h.obj.set_rotation(_scalar(v))),
    "HorizontalAlignment": (lambda h: char(h.obj.get_horizontalalignment()),
                            lambda h, v: h.obj.set_horizontalalignment(text(v).lower())),
    "VerticalAlignment": (lambda h: char({"center": "middle", "center_baseline": "middle"}.get(
                              h.obj.get_verticalalignment(), h.obj.get_verticalalignment())),
                          lambda h, v: h.obj.set_verticalalignment(
                              {"middle": "center", "cap": "top"}.get(text(v).lower(), text(v).lower()))),
    "Interpreter": (lambda h: char(h.extra.get("Interpreter", "tex")),
                    lambda h, v: h.extra.__setitem__("Interpreter", text(v))),
    "BackgroundColor": (lambda h: char("none"), lambda h, v: h.obj.set_backgroundcolor(_color(v))),
    "EdgeColor": (lambda h: char("none"),
                  lambda h, v: h.obj.set_bbox({"facecolor": "none", "edgecolor": _color(v)})),
}


# legend -------------------------------------------------------------------------------------

def _legend_strings(h):
    return Cell.from_list([char(t.get_text()) for t in h.obj.get_texts()])


def _legend_strings_set(h, v):
    labels = _text_list(v)
    for t, s in zip(h.obj.get_texts(), labels):
        t.set_text(s)


def _legend_loc_set(h, v):
    ax = h.obj.axes
    _make_legend(h.interp, ax, location=text(v), reuse=h.obj)


def _legend_fs_set(h, v):
    for t in h.obj.get_texts():
        t.set_fontsize(_scalar(v))


_LEGEND_PROPS: dict[str, tuple] = {
    "String": (_legend_strings, _legend_strings_set),
    "Location": (lambda h: char(getattr(h.obj, "_mylab_loc", "northeast")), _legend_loc_set),
    "Box": (lambda h: _oo(h.obj.get_frame_on()), lambda h, v: h.obj.set_frame_on(_onoff(v))),
    "FontSize": (lambda h: _num_val(h.obj.get_texts()[0].get_fontsize()) if h.obj.get_texts() else _num_val(9),
                 _legend_fs_set),
    "Orientation": (lambda h: char(getattr(h.obj, "_mylab_orientation", "vertical")),
                    lambda h, v: _make_legend(h.interp, h.obj.axes, orientation=text(v), reuse=h.obj)),
    "NumColumns": (lambda h: _num_val(getattr(h.obj, "_ncols", 1)),
                   lambda h, v: _make_legend(h.interp, h.obj.axes, ncols=to_int(v, "NumColumns"), reuse=h.obj)),
    "TextColor": (lambda h: _rgb(h.obj.get_texts()[0].get_color()) if h.obj.get_texts() else _rgb("k"),
                  lambda h, v: [t.set_color(_color(v)) for t in h.obj.get_texts()]),
    "Title": (lambda h: _wrap(h.interp, "text", h.obj.get_title()), None),
}

_LEGEND_IGNORED = {"Interpreter", "AutoUpdate", "EdgeColor", "Color", "ItemHitFcn", "FontName", "FontWeight",
                   "Units", "Position", "ItemTokenSize", "IconColumnWidth"}


# colorbar -----------------------------------------------------------------------------------

def _cb_label(h):
    cb = h.obj
    axis = cb.ax.xaxis if cb.orientation == "horizontal" else cb.ax.yaxis
    return _wrap(h.interp, "text", axis.label)


def _cb_limits_set(h, v):
    lim = np.real(fnum(v)).flatten()
    h.obj.mappable.set_clim(lim[0], lim[1])


_COLORBAR_PROPS: dict[str, tuple] = {
    "Label": (_cb_label, lambda h, v: None if isinstance(v, GraphicsHandle) else h.obj.set_label(_label_text(v))),
    "Limits": (lambda h: _row(h.obj.mappable.get_clim()), _cb_limits_set),
    "Ticks": (lambda h: _row(h.obj.get_ticks()), lambda h, v: h.obj.set_ticks(np.real(fnum(v)).flatten())),
    "TickLabels": (lambda h: _cell_column([char(t.get_text()) for t in
                                           (h.obj.ax.get_yticklabels() if h.obj.orientation == "vertical"
                                            else h.obj.ax.get_xticklabels())]),
                   lambda h, v: h.obj.set_ticklabels(_text_list(v))),
    "Location": (lambda h: char(h.extra.get("Location", "eastoutside")), None),
    "FontSize": (lambda h: _num_val(h.extra.get("FontSize", 10)),
                 lambda h, v: (h.obj.ax.tick_params(labelsize=_scalar(v)),
                               h.extra.__setitem__("FontSize", _scalar(v)))),
    "Colormap": (lambda h: _cmap_matrix(h.obj.cmap, 256), lambda h, v: h.obj.mappable.set_cmap(_to_cmap(v))),
}


# surfaces and other collections -------------------------------------------------------------

def _surf_edge_set(h, v):
    if is_text(v) and text(v).lower() in ("flat", "interp"):
        h.obj.set_edgecolor("none")
    else:
        h.obj.set_edgecolor(_color(v))
    h.extra["EdgeColor"] = v


def _surf_edge_get(h):
    v = h.extra.get("EdgeColor")
    if v is not None:
        return v if is_text(v) else _rgb(_color(v))
    return _rgb(h.obj.get_edgecolor())


_SURFACE_PROPS: dict[str, tuple] = {
    "EdgeColor": (_surf_edge_get, _surf_edge_set),
    "FaceColor": (lambda h: char(h.extra.get("FaceColor", "flat")),
                  lambda h, v: (h.extra.__setitem__("FaceColor", text(v) if is_text(v) else "rgb"),
                                None if is_text(v) and text(v).lower() in ("flat", "interp")
                                else h.obj.set_facecolor(_color(v)))),
    "FaceAlpha": (lambda h: _num_val(1 if h.obj.get_alpha() is None else h.obj.get_alpha()),
                  lambda h, v: h.obj.set_alpha(_scalar(v))),
    "LineWidth": (lambda h: _num_val(np.asarray(h.obj.get_linewidth()).flat[0]),
                  lambda h, v: h.obj.set_linewidth(_scalar(v))),
    "XData": (lambda h: h.extra.get("XData", np.zeros((0, 0))), None),
    "YData": (lambda h: h.extra.get("YData", np.zeros((0, 0))), None),
    "ZData": (lambda h: h.extra.get("ZData", np.zeros((0, 0))), None),
    "CData": (lambda h: h.extra.get("ZData", np.zeros((0, 0))), None),
}


_PATCH_PROPS: dict[str, tuple] = {
    "FaceColor": (lambda h: _patch_color(h, "face"), lambda h, v: _set_patch_color(h, "face", v)),
    "EdgeColor": (lambda h: _patch_color(h, "edge"), lambda h, v: _set_patch_color(h, "edge", v)),
    "FaceAlpha": (lambda h: _num_val(_patches(h)[0].get_alpha() or 1) if _patches(h) else _num_val(1),
                  lambda h, v: [p.set_alpha(_scalar(v)) for p in _patches(h)]),
    "LineWidth": (lambda h: _num_val(np.asarray(_patches(h)[0].get_linewidth()).flat[0]) if _patches(h)
                  else _num_val(0.5),
                  lambda h, v: [p.set_linewidth(_scalar(v)) for p in _patches(h)]),
    "DisplayName": (lambda h: _get_label(_artists(h.obj)[0]) if _artists(h.obj) else char(""),
                    lambda h, v: _artists(h.obj)[0].set_label(text(v)) if _artists(h.obj) else None),
}


def _image_cdata(h):
    return np.asarray(h.obj.get_array(), dtype=float)


_IMAGE_PROPS: dict[str, tuple] = {
    "CData": (_image_cdata, lambda h, v: h.obj.set_data(np.real(fnum(v)))),
    "AlphaData": (lambda h: _num_val(1 if h.obj.get_alpha() is None else h.obj.get_alpha()),
                  lambda h, v: h.obj.set_alpha(_scalar(v))),
}

_QUIVER_PROPS: dict[str, tuple] = {
    "Color": (lambda h: _rgb(h.obj.get_facecolor()), lambda h, v: h.obj.set_color(_color(v))),
    "LineWidth": (lambda h: _num_val(h.extra.get("LineWidth", 0.5)),
                  lambda h, v: (h.obj.set_linewidth(_scalar(v)), h.extra.__setitem__("LineWidth", _scalar(v)))),
    "UData": (lambda h: h.extra.get("UData", np.zeros((0, 0))), None),
    "VData": (lambda h: h.extra.get("VData", np.zeros((0, 0))), None),
    "XData": (lambda h: h.extra.get("XData", np.zeros((0, 0))), None),
    "YData": (lambda h: h.extra.get("YData", np.zeros((0, 0))), None),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}


def _stem_line(h):
    return h.obj.markerline


_STEM_PROPS: dict[str, tuple] = {
    "Color": (lambda h: _rgb(h.obj.markerline.get_color()),
              lambda h, v: (h.obj.markerline.set_color(_color(v)), h.obj.stemlines.set_color(_color(v)))),
    "LineWidth": (lambda h: _num_val(np.asarray(h.obj.stemlines.get_linewidth()).flat[0]),
                  lambda h, v: h.obj.stemlines.set_linewidth(_scalar(v))),
    "Marker": (lambda h: char(str(h.obj.markerline.get_marker())),
               lambda h, v: h.obj.markerline.set_marker(_mpl_marker(text(v)))),
    "MarkerSize": (lambda h: _num_val(h.obj.markerline.get_markersize()),
                   lambda h, v: h.obj.markerline.set_markersize(_scalar(v))),
    "MarkerFaceColor": (lambda h: _rgb(h.obj.markerline.get_markerfacecolor()),
                        lambda h, v: h.obj.markerline.set_markerfacecolor(_color(v))),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}

_ERRORBAR_PROPS: dict[str, tuple] = {
    "Color": (lambda h: _rgb(h.obj.lines[0].get_color()),
              lambda h, v: [a.set_color(_color(v)) for a in _artists(h.obj)]),
    "LineWidth": (lambda h: _num_val(h.obj.lines[0].get_linewidth()),
                  lambda h, v: [a.set_linewidth(_scalar(v)) for a in _artists(h.obj)]),
    "DisplayName": (lambda h: _get_label(h.obj), _set_label),
}


# tiled layouts and root -----------------------------------------------------------------------

def _layout_title(h, which: str):
    fig = h.obj.fig
    if which == "Title":
        t = getattr(fig, "_suptitle", None) or fig.suptitle("")
    elif which == "XLabel":
        t = getattr(fig, "_supxlabel", None) or fig.supxlabel("")
    else:
        t = getattr(fig, "_supylabel", None) or fig.supylabel("")
    return _wrap(h.interp, "text", t)


def _layout_spacing_set(which: str):
    def s(h, v):
        setattr(h.obj, which, text(v).lower())
        h.obj.apply_spacing()
    return s


_LAYOUT_PROPS: dict[str, tuple] = {
    "GridSize": (lambda h: _row(h.obj.grid_size()), None),
    "TileSpacing": (lambda h: char(h.obj.spacing), _layout_spacing_set("spacing")),
    "Padding": (lambda h: char(h.obj.padding), _layout_spacing_set("padding")),
    "Title": (lambda h: _layout_title(h, "Title"), None),
    "XLabel": (lambda h: _layout_title(h, "XLabel"), None),
    "YLabel": (lambda h: _layout_title(h, "YLabel"), None),
    "Children": (lambda h: GraphicsArray([_wrap(h.interp, _axes_kind(a), a) for a in h.obj.order][::-1])
                 if len(h.obj.order) > 1 else (_wrap(h.interp, _axes_kind(h.obj.order[0]), h.obj.order[0])
                                               if h.obj.order else np.zeros((0, 1))), None),
}


def _root_current_figure(h):
    st = _state(h.interp)
    if st.current is None or st.current not in st.figures:
        return np.zeros((0, 0))
    return _wrap(h.interp, "figure", st.figures[st.current])


_ROOT_DEFAULTS = {"defaultlinelinewidth": "lines.linewidth", "defaultlinemarkersize": "lines.markersize",
                  "defaultaxesfontsize": "font.size", "defaulttextfontsize": "font.size",
                  "defaultaxeslinewidth": "axes.linewidth"}

_ROOT_PROPS: dict[str, tuple] = {
    "ScreenSize": (lambda h: _row([1, 1, 1920, 1080]), None),
    "MonitorPositions": (lambda h: _row([1, 1, 1920, 1080]), None),
    "ScreenPixelsPerInch": (lambda h: _num_val(96), None),
    "CurrentFigure": (_root_current_figure, None),
    "Children": (_children_get, None),
}


# property dispatch ---------------------------------------------------------------------------

_PROPS: dict[str, dict] = {
    "line": _LINE_PROPS, "stair": _LINE_PROPS, "scatter": _SCATTER_PROPS, "bar": _BAR_PROPS,
    "histogram": _HIST_PROPS, "axes": _AXES_PROPS, "polaraxes": _AXES_PROPS, "figure": _FIGURE_PROPS,
    "text": _TEXT_PROPS, "legend": _LEGEND_PROPS, "colorbar": _COLORBAR_PROPS, "surface": _SURFACE_PROPS,
    "patch": _PATCH_PROPS, "area": _PATCH_PROPS, "pie": _PATCH_PROPS, "image": _IMAGE_PROPS,
    "quiver": _QUIVER_PROPS, "stem": _STEM_PROPS, "errorbar": _ERRORBAR_PROPS, "tiledlayout": _LAYOUT_PROPS,
    "root": _ROOT_PROPS, "contour": {},
}

_IGNORED: dict[str, set] = {"axes": _AXES_IGNORED, "polaraxes": _AXES_IGNORED, "figure": _FIGURE_IGNORED,
                            "legend": _LEGEND_IGNORED}

_COMMON_PROPS: dict[str, tuple] = {
    "Type": (lambda h: char(_TYPE.get(h.kind, h.kind)), None),
    "Visible": (_visible_get, _visible_set),
    "Parent": (_parent_get, None),
    "Tag": (lambda h: char(h.extra.get("Tag", "")), lambda h, v: h.extra.__setitem__("Tag", text(v))),
    "UserData": (lambda h: h.extra.get("UserData", np.zeros((0, 0))),
                 lambda h, v: h.extra.__setitem__("UserData", v)),
}

_DISPLAY = {
    "line": ["Color", "LineStyle", "LineWidth", "Marker", "MarkerSize", "MarkerFaceColor", "XData", "YData",
             "DisplayName"],
    "stair": ["Color", "LineStyle", "LineWidth", "Marker", "XData", "YData"],
    "scatter": ["XData", "YData", "SizeData", "CData", "Marker", "MarkerFaceColor", "DisplayName"],
    "bar": ["BarWidth", "FaceColor", "EdgeColor", "XData", "YData", "DisplayName"],
    "histogram": ["Data", "Values", "NumBins", "BinEdges", "BinWidth", "Normalization", "FaceColor",
                  "EdgeColor"],
    "axes": ["XLim", "YLim", "XScale", "YScale", "FontSize", "Box", "Position"],
    "polaraxes": ["Position"],
    "figure": ["Number", "Name", "Color", "Position"],
    "text": ["String", "FontSize", "FontWeight", "Color", "HorizontalAlignment", "Position"],
    "legend": ["String", "Location", "Orientation", "FontSize", "Box"],
    "colorbar": ["Location", "Limits", "Ticks"],
    "surface": ["EdgeColor", "FaceColor", "FaceAlpha"],
    "tiledlayout": ["GridSize", "TileSpacing", "Padding"],
    "root": ["CurrentFigure", "ScreenSize"],
}


def _lookup(h: GraphicsHandle, name: str):
    table = _PROPS.get(h.kind, {})
    low = name.lower()
    for tbl in (table, _COMMON_PROPS):
        for k, v in tbl.items():
            if k.lower() == low:
                return k, v
    for k in _IGNORED.get(h.kind, ()):
        if k.lower() == low:
            return k, None
    return None, None


def _get_prop(h: GraphicsHandle, name: str) -> Any:
    key, entry = _lookup(h, name)
    if key is None:
        if h.kind == "root" and name.lower().startswith("default"):
            import matplotlib as mpl
            rc = _ROOT_DEFAULTS.get(name.lower())
            if rc:
                return _num_val(mpl.rcParams[rc])
            return h.extra.get(name.lower(), np.zeros((0, 0)))
        raise MatlabError(f"Unrecognized property '{name}' for class '{_short(h.kind)}'.")
    if not _alive(h) and key not in ("Type", "Tag", "UserData"):
        raise MatlabError("Invalid or deleted object.")
    if entry is None:  # accepted-and-ignored property
        return h.extra.get(key, np.zeros((0, 0)))
    if key == "Visible" and h.kind == "figure":
        return _oo(h.extra.get("Visible", True))
    return entry[0](h)


def _set_prop(h: GraphicsHandle, name: str, value: Any) -> None:
    if h.kind == "root":
        low = name.lower()
        if low.startswith("default"):
            import matplotlib as mpl
            rc = _ROOT_DEFAULTS.get(low)
            if rc:
                mpl.rcParams[rc] = _scalar(value)
            h.extra[low] = value
            return
    key, entry = _lookup(h, name)
    if key is None:
        raise MatlabError(f"Unrecognized property '{name}' for class '{_short(h.kind)}'.")
    if not _alive(h):
        raise MatlabError("Invalid or deleted object.")
    if isinstance(value, GraphicsHandle) and entry is not None:
        # Nested assignment such as h.Parent.Title.String = 'x' writes the (same) handle back.
        try:
            if entry[0](h) is value:
                return
        except (MatlabError, AttributeError, ValueError, TypeError):
            pass
    if entry is None:
        h.extra[key] = value
        if key == "Units" and is_text(value):
            h.extra["Units"] = text(value).lower()
        return
    setter = entry[1]
    if setter is None:
        raise MatlabError(f"Unable to set the '{key}' property of class '{_short(h.kind)}' because it is "
                          "read-only.")
    setter(h, value)


def _resolve(interp, v) -> list[GraphicsHandle]:
    """Handles from a handle, handle array, figure number or 0 (the root)."""
    if isinstance(v, (GraphicsHandle, GraphicsArray)):
        return _handles_of(v)
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size >= 1 and not is_char(v):
        st = _state(interp)
        out = []
        for x in np.real(v).flatten():
            n = int(round(float(x)))
            if n == 0:
                out.append(_wrap(interp, "root", _ROOT))
            elif n in st.figures:
                out.append(_wrap(interp, "figure", st.figures[n]))
            else:
                raise MatlabError("Invalid or deleted object.")
        return out
    raise MatlabError("Expected a graphics handle.")


# -- builtins: get / set and handle utilities ---------------------------------------------------

@builtin("set")
def _set_b(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for set.")
    hs = _resolve(interp, args[0])
    rest = list(args[1:])
    if len(rest) == 1 and isinstance(rest[0], Struct):
        s = rest[0]
        pairs = [(f, s.data.flat[0][f]) for f in s.fields]
    elif len(rest) == 2 and isinstance(rest[0], Cell) and isinstance(rest[1], Cell):
        pairs = list(zip([to_str(x) for x in rest[0].values_f()], rest[1].values_f()))
    else:
        if len(rest) % 2:
            raise MatlabError("set expects property name/value pairs.")
        pairs = [(text(rest[i], "property name"), rest[i + 1]) for i in range(0, len(rest), 2)]
    for h in hs:
        for name, value in pairs:
            _set_prop(h, name, value)
    _redraw(interp)
    return None


def _all_props(h: GraphicsHandle) -> Struct:
    mapping = {}
    for name in list(_PROPS.get(h.kind, {})) + list(_COMMON_PROPS):
        try:
            mapping[name] = _get_prop(h, name)
        except (MatlabError, ValueError, TypeError, AttributeError, IndexError, KeyError):
            continue
    return Struct.scalar(mapping)


@builtin("get")
def _get_b(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for get.")
    hs = _resolve(interp, args[0])
    if len(args) == 1:
        if len(hs) == 1:
            if nargout == 0:
                interp.out("\n".join(hs[0].display_lines()[2:]) + "\n")
                return None
            return _all_props(hs[0])
        raise MatlabError("get with several handles needs a property name.")
    prop = args[1]
    if isinstance(prop, Cell) or isinstance(prop, StringArray):
        names = _text_list(prop)
        vals = np.empty((len(hs), len(names)), dtype=object)
        for i, h in enumerate(hs):
            for j, n in enumerate(names):
                vals[i, j] = _get_prop(h, n)
        return Cell(vals)
    name = text(prop, "property name")
    if len(hs) == 1:
        return _get_prop(hs[0], name)
    return _cell_column([_get_prop(h, name) for h in hs])


@builtin("groot")
def _groot(interp, args, nargout):
    return _wrap(interp, "root", _ROOT)


@builtin("isgraphics", "ishghandle", "ishandle")
def _isgraphics(interp, args, nargout):
    v = args[0]
    want = text(args[1]).lower() if len(args) > 1 else None
    if isinstance(v, GraphicsHandle):
        ok = _alive(v) and (want is None or _TYPE.get(v.kind) == want)
        return np.array([[ok]])
    if isinstance(v, GraphicsArray):
        flags = [_alive(h) and (want is None or _TYPE.get(h.kind) == want) for h in v.items]
        return np.array(flags, dtype=bool).reshape(v.matlab_shape())
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and not is_char(v):
        st = _state(interp)
        flags = []
        for x in np.real(v).flatten(order="F"):
            n = float(x)
            ok = (n == 0 and want in (None, "root")) or (n == int(n) and int(n) in st.figures
                                                          and want in (None, "figure"))
            flags.append(ok)
        return np.array(flags, dtype=bool).reshape(v.shape, order="F")
    return np.array([[False]])


# -- line plots -----------------------------------------------------------------------------

def _do_plot(interp, args, method: str, nargout: int):
    args, opts = _split_name_values(args, _LINE_OPTS)
    if not args:
        raise MatlabError("Not enough input arguments.")
    ax = _axes(interp, projection="rectilinear")
    kw = _mpl_kwargs(opts)
    lines = []
    for x, y, spec in _xy_groups(args):
        x, y = _prep_xy(x, y)
        fn = getattr(ax, method)
        lines.extend(fn(x, y, spec, **kw) if spec else fn(x, y, **kw))
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("plot")
@_with_axes_arg
def _plot(interp, args, nargout):
    return _do_plot(interp, args, "plot", nargout)


@builtin("semilogx")
@_with_axes_arg
def _semilogx(interp, args, nargout):
    return _do_plot(interp, args, "semilogx", nargout)


@builtin("semilogy")
@_with_axes_arg
def _semilogy(interp, args, nargout):
    return _do_plot(interp, args, "semilogy", nargout)


@builtin("loglog")
@_with_axes_arg
def _loglog(interp, args, nargout):
    return _do_plot(interp, args, "loglog", nargout)


@builtin("plot3")
@_with_axes_arg
def _plot3(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    ax = _axes(interp, projection="3d")
    kw = _mpl_kwargs(opts)
    lines = []
    k = 0
    while k + 2 < len(args):
        x, y, z = (np.real(fnum(a)) for a in args[k:k + 3])
        k += 3
        spec = ""
        if k < len(args) and is_text(args[k]) and _is_linespec(text(args[k])):
            spec = text(args[k])
            k += 1
        if x.shape[0] > 1 and x.shape[1] > 1:
            cols = [(x[:, j], y[:, j], z[:, j]) for j in range(x.shape[1])]
        else:
            cols = [(x.flatten(), y.flatten(), z.flatten())]
        for xs, ys, zs in cols:
            lines.extend(ax.plot(xs, ys, zs, spec, **kw) if spec else ax.plot(xs, ys, zs, **kw))
    if not lines:
        raise MatlabError("plot3 needs x, y and z data.")
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("line")
@_with_axes_arg
def _line(interp, args, nargout):
    known = _LINE_OPTS | {"xdata", "ydata", "zdata"}
    nums = []
    k = 0
    while k < len(args) and not is_text(args[k]):
        nums.append(args[k])
        k += 1
    rest, opts = _split_name_values(args[k:], known)
    if rest:
        raise MatlabError(f"Invalid argument to line: '{text(rest[0]) if is_text(rest[0]) else rest[0]}'.")
    x = opts.pop("xdata", nums[0] if nums else None)
    y = opts.pop("ydata", nums[1] if len(nums) > 1 else None)
    z = opts.pop("zdata", nums[2] if len(nums) > 2 else None)
    if x is None or y is None:
        x, y = np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]])
    ax = _axes(interp, fresh=False)
    kw = _mpl_kwargs(opts)
    X, Y = _prep_xy(fnum(x), fnum(y))
    if z is not None and hasattr(ax, "get_zlim"):
        Z = np.real(fnum(z))
        lines = ax.plot(np.asarray(X).flatten(), np.asarray(Y).flatten(), Z.flatten(), **kw)
    else:
        lines = ax.plot(X, Y, **kw)
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("fplot")
@_with_axes_arg
def _fplot(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS | {"meshdensity"})
    funcs = []
    while args and isinstance(args[0], FunctionHandle):
        funcs.append(args.pop(0))
    if not funcs:
        raise MatlabError("fplot expects a function handle, e.g. fplot(@(x) sin(x)).")
    lims = np.array([-5.0, 5.0])
    if args and not is_text(args[0]):
        lims = np.real(fnum(args.pop(0))).flatten()
        if lims.size != 2 or not lims[0] < lims[1]:
            raise MatlabError("The fplot interval must be a 2-element vector [a b] with a < b.")
    spec = ""
    if args and is_text(args[0]) and _is_linespec(text(args[0])):
        spec = text(args.pop(0))
    n = int(_scalar(opts.pop("meshdensity"))) * 8 if "meshdensity" in opts else 401
    t = np.linspace(lims[0], lims[1], max(n, 2))

    def evaluate(f):
        try:
            res = interp.call_function_value(f, [t.reshape(1, -1)], 1)
            vals = np.real(fnum(res[0])).flatten() if res else None
            if vals is not None and vals.size == t.size:
                return vals
            if vals is not None and vals.size == 1:
                return np.full(t.size, vals[0])
        except MatlabError:
            pass
        out = np.empty(t.size)
        for i, ti in enumerate(t):
            res = interp.call_function_value(f, [np.array([[ti]])], 1)
            out[i] = float(np.real(fnum(res[0])).flat[0]) if res else np.nan
        return out

    ax = _axes(interp, projection="rectilinear")
    kw = _mpl_kwargs(opts)
    if len(funcs) == 2:
        xs, ys = evaluate(funcs[0]), evaluate(funcs[1])
    else:
        xs, ys = t, evaluate(funcs[0])
    ys = np.where(np.isfinite(ys), ys, np.nan)
    lines = ax.plot(xs, ys, spec, **kw) if spec else ax.plot(xs, ys, **kw)
    if len(funcs) == 1:
        ax.set_xlim(lims[0], lims[1])
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("polarplot")
@_with_axes_arg
def _polarplot(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    nums = [a for a in args if not is_text(a)]
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    if not nums:
        raise MatlabError("polarplot needs data, e.g. polarplot(theta, rho).")
    if len(nums) == 1:
        rho = np.real(fnum(nums[0])).flatten()
        theta = np.linspace(0, 2 * np.pi, rho.size, endpoint=False)
    else:
        theta, rho = np.real(fnum(nums[0])).flatten(), np.real(fnum(nums[1])).flatten()
        if theta.size == 1 and rho.size > 1:
            theta = np.full(rho.size, theta[0])
        if rho.size == 1 and theta.size > 1:
            rho = np.full(theta.size, rho[0])
    ax = _axes(interp, projection="polar")
    kw = _mpl_kwargs(opts)
    lines = ax.plot(theta, rho, spec, **kw) if spec else ax.plot(theta, rho, **kw)
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("scatter")
@_with_axes_arg
def _scatter(interp, args, nargout):
    args, opts = _split_name_values(args, {"markerfacecolor", "markeredgecolor", "displayname", "linewidth",
                                           "markerfacealpha", "markeredgealpha", "sizedata"})
    filled = any(is_text(a) and text(a) == "filled" for a in args)
    args = [a for a in args if not (is_text(a) and text(a) == "filled")]
    ax = _axes(interp, projection="rectilinear")
    x, y = np.real(fnum(args[0])).flatten(), np.real(fnum(args[1])).flatten()
    s = 36.0
    if len(args) > 2 and isinstance(args[2], np.ndarray) and not is_char(args[2]) and args[2].size:
        sv = np.real(fnum(args[2])).flatten()
        s = float(sv[0]) if sv.size == 1 else sv
    kw: dict = {}
    if len(args) > 3 and not (is_text(args[3]) and not _is_color_text(args[3])):
        c = args[3]
        kw["c"] = _color(c) if is_text(c) else (np.real(fnum(c)).flatten() if fnum(c).size == x.size
                                                 else _color(c))
        if isinstance(kw["c"], np.ndarray):
            kw["cmap"] = _current_cmap(interp)
    marker = next((_mpl_marker(text(a)) for a in args[3:] if is_text(a) and not _is_color_text(a)), "o")
    if "c" not in kw:
        import matplotlib as mpl
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        kw["c"] = colors[len(ax.collections) % len(colors)]
    if not filled:
        if not isinstance(kw["c"], np.ndarray):
            kw["edgecolors"] = kw.pop("c")
            kw["facecolors"] = "none"
        else:
            kw["facecolors"] = "none"
    if "markerfacecolor" in opts:
        kw.pop("facecolors", None)
        fc = opts["markerfacecolor"]
        if is_text(fc) and text(fc).lower() == "flat":
            pass
        else:
            kw["facecolors"] = _color(fc)
            if "c" in kw and not isinstance(kw["c"], np.ndarray):
                kw["edgecolors"] = kw.pop("c")
    if "markeredgecolor" in opts:
        ec = opts["markeredgecolor"]
        if not (is_text(ec) and text(ec).lower() == "flat"):
            kw["edgecolors"] = _color(ec)
    if "linewidth" in opts:
        kw["linewidths"] = _scalar(opts["linewidth"])
    if "markerfacealpha" in opts:
        kw["alpha"] = _scalar(opts["markerfacealpha"])
    if "sizedata" in opts:
        s = np.real(fnum(opts["sizedata"])).flatten()
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    if "c" in kw and not isinstance(kw["c"], np.ndarray):
        kw["color"] = kw.pop("c")  # one colour for all points
    coll = ax.scatter(x, y, s=s, marker=marker, **kw)
    _redraw(interp)
    if nargout > 0:
        h = _wrap(interp, "scatter", coll)
        h.extra["Marker"] = marker
        return h
    return None


def _is_color_text(a) -> bool:
    s = text(a)
    return s.lower() in _COLOR_NAMES or (len(s) == 1 and s in "rgbcmykw") or s.startswith("#")


@builtin("scatter3")
@_with_axes_arg
def _scatter3(interp, args, nargout):
    filled = any(is_text(a) and text(a) == "filled" for a in args)
    nums = [a for a in args if not is_text(a)]
    ax = _axes(interp, projection="3d")
    x, y, z = (np.real(fnum(a)).flatten() for a in nums[:3])
    s = np.real(fnum(nums[3])).flatten() if len(nums) > 3 and fnum(nums[3]).size else 36
    kw = {}
    if len(nums) > 4:
        c = np.real(fnum(nums[4])).flatten()
        kw["c"] = c if c.size == x.size else _color(nums[4])
    coll = ax.scatter(x, y, z, s=s, depthshade=False, **kw) if filled else \
        ax.scatter(x, y, z, s=s, facecolors="none", edgecolors=kw.pop("c", "C0"), depthshade=False)
    _redraw(interp)
    return _ret(interp, "scatter", [coll], nargout)


@builtin("bar")
@_with_axes_arg
def _bar(interp, args, nargout):
    return _bar_core(interp, args, nargout, horizontal=False)


@builtin("barh")
@_with_axes_arg
def _barh(interp, args, nargout):
    return _bar_core(interp, args, nargout, horizontal=True)


def _bar_core(interp, args, nargout, horizontal: bool):
    args, opts = _split_name_values(args, {"facecolor", "edgecolor", "displayname", "linewidth", "barwidth",
                                           "facealpha"})
    stacked = any(is_text(a) and text(a).lower() == "stacked" for a in args)
    color = next((_color(a) for a in args if is_text(a) and _is_color_text(a)), None)
    nums = [a for a in args if not is_text(a)]
    if not nums:
        raise MatlabError("bar needs data, e.g. bar([1 2 3]).")
    width = 0.8
    if len(nums) >= 2 and fnum(nums[-1]).size == 1 and (len(nums) == 3 or fnum(nums[0]).size > 1):
        width = _scalar(nums.pop())
    if "barwidth" in opts:
        width = _scalar(opts["barwidth"])
    if len(nums) == 1:
        y = np.real(fnum(nums[0]))
        x = np.arange(1, (y.size if y.shape[0] == 1 or y.shape[1] == 1 else y.shape[0]) + 1, dtype=float)
    else:
        x, y = np.real(fnum(nums[0])).flatten(), np.real(fnum(nums[1]))
    if y.shape[0] == 1 or y.shape[1] == 1:
        y = y.reshape(-1, 1)
    elif y.shape[0] != x.size and y.shape[1] == x.size:
        y = y.T
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {}
    if "edgecolor" in opts:
        kw["edgecolor"] = _color(opts["edgecolor"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "facealpha" in opts:
        kw["alpha"] = _scalar(opts["facealpha"])
    face = _color(opts["facecolor"]) if "facecolor" in opts and not (
        is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "flat") else color
    n = y.shape[1]
    conts = []
    spacing = float(np.min(np.diff(np.sort(x)))) if x.size > 1 else 1.0
    bottom = np.zeros(y.shape[0])
    for k in range(n):
        if stacked or n == 1:
            pos, w = x, width * spacing
        else:
            w = width * spacing / n
            pos = x - width * spacing / 2 + w * (k + 0.5)
        ckw = dict(kw)
        if face is not None:
            ckw["color"] = face
        if "displayname" in opts and n == 1:
            ckw["label"] = text(opts["displayname"])
        if horizontal:
            cont = ax.barh(pos, y[:, k], height=w, left=bottom if stacked else None, **ckw)
        else:
            cont = ax.bar(pos, y[:, k], width=w, bottom=bottom if stacked else None, **ckw)
        if stacked:
            bottom = bottom + y[:, k]
        conts.append(cont)
    _redraw(interp)
    if nargout <= 0:
        return None
    hs = []
    for k, cont in enumerate(conts):
        h = _wrap(interp, "bar", cont)
        h.extra.update({"BarWidth": width, "XData": x, "YData": y[:, k], "axes": ax, "figure": ax.figure})
        hs.append(h)
    return hs[0] if len(hs) == 1 else GraphicsArray(hs, row=True)


@builtin("histogram")
@_with_axes_arg
def _histogram(interp, args, nargout):
    args, opts = _split_name_values(args, {"numbins", "binedges", "binwidth", "binlimits", "normalization",
                                           "facecolor", "edgecolor", "facealpha", "displayname", "displaystyle",
                                           "orientation", "linewidth", "binmethod"})
    if not args:
        raise MatlabError("histogram needs data, e.g. histogram(randn(100, 1)).")
    raw = args[0]
    data = _finite(raw)
    nbins = edges = None
    if len(args) > 1 and not is_text(args[1]):
        b = np.real(fnum(args[1])).flatten()
        if b.size == 1:
            nbins = int(b[0])
        else:
            edges = b
    if "numbins" in opts:
        nbins = to_int(opts["numbins"], "NumBins")
    if "binedges" in opts:
        edges = np.real(fnum(opts["binedges"])).flatten()
    width = _scalar(opts["binwidth"]) if "binwidth" in opts else None
    limits = np.real(fnum(opts["binlimits"])).flatten() if "binlimits" in opts else None
    method = text(opts["binmethod"]).lower() if "binmethod" in opts else None
    e = _hist_edges(data, nbins=nbins, edges=edges, width=width, limits=limits, method=method)
    norm = _check_norm(text(opts["normalization"])) if "normalization" in opts else "count"
    style: dict = {}
    if "facecolor" in opts and not (is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "auto"):
        style["facecolor"] = _color(opts["facecolor"])
    if "edgecolor" in opts:
        style["edgecolor"] = _color(opts["edgecolor"])
    if "linewidth" in opts:
        style["linewidth"] = _scalar(opts["linewidth"])
    if "facealpha" in opts:
        style["alpha"] = _scalar(opts["facealpha"])
    if "displaystyle" in opts and text(opts["displaystyle"]).lower() == "stairs":
        style["alpha"] = 0.0
        style.setdefault("edgecolor", "C0")
        style["linewidth"] = style.get("linewidth", 1.2)
    orientation = text(opts["orientation"]).lower() if "orientation" in opts else "vertical"
    ax = _axes(interp, projection="rectilinear")
    label = text(opts["displayname"]) if "displayname" in opts else None
    hs = _Hist(ax, data, e, norm, orientation, style, label)
    hs.draw()
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "histogram", hs)
    shape_src = np.real(fnum(raw))
    h.extra["DataShape"] = lambda: shape_src
    return h


@builtin("hist")
def _hist(interp, args, nargout):
    x = _finite(args[0])
    nb = 10
    centers = None
    if len(args) > 1 and not is_text(args[1]):
        b = np.real(fnum(args[1])).flatten()
        if b.size == 1:
            nb = int(b[0])
        else:
            centers = b
    if centers is not None:
        mids = (centers[:-1] + centers[1:]) / 2
        edges = np.concatenate([[-np.inf], mids, [np.inf]])
        counts = np.histogram(x, edges)[0]
        plot_edges = np.concatenate([[centers[0] - (mids[0] - centers[0])], mids,
                                     [centers[-1] + (centers[-1] - mids[-1])]]) if centers.size > 1 \
            else np.array([centers[0] - 0.5, centers[0] + 0.5])
    else:
        lo, hi = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, nb + 1)
        counts = np.histogram(x, edges)[0]
        centers = (edges[:-1] + edges[1:]) / 2
        plot_edges = edges
    if nargout > 0:
        c = counts.astype(float).reshape(1, -1)
        return (c, centers.reshape(1, -1)) if nargout > 1 else c
    ax = _axes(interp, projection="rectilinear")
    ax.bar(plot_edges[:-1], counts, width=np.diff(plot_edges), align="edge", edgecolor="black", linewidth=0.5)
    _redraw(interp)
    return None


def _xy_simple(args):
    nums = [a for a in args if not is_text(a)]
    y = np.real(fnum(nums[-1] if len(nums) < 2 else nums[1])).flatten()
    x = np.real(fnum(nums[0])).flatten() if len(nums) > 1 else np.arange(1, y.size + 1, dtype=float)
    return x, y


@builtin("stem")
@_with_axes_arg
def _stem(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    filled = any(is_text(a) and text(a).lower() == "filled" for a in args)
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    ax = _axes(interp, projection="rectilinear")
    x, y = _xy_simple([a for a in args if not is_text(a)])
    cont = ax.stem(x, y, linefmt=spec or None)
    if not filled:
        cont.markerline.set_markerfacecolor("none")
    for k, v in _mpl_kwargs(opts).items():
        if k == "label":
            cont.set_label(v)
        elif k == "color":
            cont.markerline.set_color(v)
            cont.stemlines.set_color(v)
        elif k == "linewidth":
            cont.stemlines.set_linewidth(v)
        elif k in ("markersize", "marker", "markerfacecolor", "markeredgecolor"):
            getattr(cont.markerline, f"set_{k}")(v)
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "stem", cont)
    h.extra.update({"axes": ax, "figure": ax.figure})
    return h


@builtin("stairs")
@_with_axes_arg
def _stairs(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    ax = _axes(interp, projection="rectilinear")
    x, y = _xy_simple(args)
    kw = _mpl_kwargs(opts)
    lines = ax.step(x, y, spec, where="post", **kw) if spec else ax.step(x, y, where="post", **kw)
    _redraw(interp)
    return _ret(interp, "stair", lines, nargout)


@builtin("area")
@_with_axes_arg
def _area(interp, args, nargout):
    args, opts = _split_name_values(args, {"facecolor", "edgecolor", "facealpha", "displayname", "linewidth"})
    ax = _axes(interp, projection="rectilinear")
    x, y = _xy_simple(args)
    kw: dict = {"alpha": _scalar(opts["facealpha"]) if "facealpha" in opts else 0.6}
    if "facecolor" in opts:
        kw["facecolor"] = _color(opts["facecolor"])
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    coll = ax.fill_between(x, y, **kw)
    ec = _color(opts["edgecolor"]) if "edgecolor" in opts else None
    lw = _scalar(opts["linewidth"]) if "linewidth" in opts else None
    line_kw = {k: v for k, v in (("color", ec), ("linewidth", lw)) if v is not None}
    if "color" not in line_kw:
        from matplotlib.colors import to_rgba
        line_kw["color"] = to_rgba(coll.get_facecolor()[0])[:3]
    ax.plot(x, y, **line_kw)
    _redraw(interp)
    return _ret(interp, "area", [coll], nargout)


@builtin("fill")
@_with_axes_arg
def _fill(interp, args, nargout):
    return _fill_core(interp, args, nargout, patch=False)


@builtin("patch")
@_with_axes_arg
def _patch(interp, args, nargout):
    return _fill_core(interp, args, nargout, patch=True)


def _fill_core(interp, args, nargout, patch: bool):
    args, opts = _split_name_values(args, {"facecolor", "edgecolor", "facealpha", "displayname", "linewidth"})
    ax = _axes(interp, fresh=False) if patch else _axes(interp, projection="rectilinear")
    x, y = np.real(fnum(args[0])), np.real(fnum(args[1]))
    kw: dict = {}
    if len(args) > 2 and (is_text(args[2]) or (isinstance(args[2], np.ndarray) and args[2].size in (3, 4))):
        kw["color"] = _color(args[2])
    if "facecolor" in opts:
        kw["facecolor"] = _color(opts["facecolor"])
    if "edgecolor" in opts:
        kw["edgecolor"] = _color(opts["edgecolor"])
    if "facealpha" in opts:
        kw["alpha"] = _scalar(opts["facealpha"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    if x.shape[0] > 1 and x.shape[1] > 1:
        polys = []
        for j in range(x.shape[1]):
            polys.extend(ax.fill(x[:, j], y[:, j], **kw))
    else:
        polys = ax.fill(x.flatten(), y.flatten(), **kw)
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "patch", polys)
    h.extra.update({"axes": ax, "figure": ax.figure})
    return h


@builtin("errorbar")
@_with_axes_arg
def _errorbar(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS | {"capsize"})
    ax = _axes(interp, projection="rectilinear")
    nums = [a for a in args if not is_text(a)]
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    if len(nums) == 2:
        y, e = fnum(nums[0]).flatten(), fnum(nums[1]).flatten()
        x = np.arange(1, y.size + 1, dtype=float)
        yerr = e
    elif len(nums) == 3:
        x, y, e = (fnum(a).flatten() for a in nums[:3])
        yerr = e
    else:
        x, y, lo, hi = (fnum(a).flatten() for a in nums[:4])
        yerr = np.vstack([lo, hi])
    kw = _mpl_kwargs(opts)
    cap = _scalar(opts["capsize"]) if "capsize" in opts else 3
    cont = ax.errorbar(x, y, yerr=yerr, fmt=spec or "-", capsize=cap, **kw)
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "errorbar", cont)
    h.extra.update({"axes": ax, "figure": ax.figure})
    return h


@builtin("pie")
@_with_axes_arg
def _pie(interp, args, nargout):
    ax = _axes(interp, projection="rectilinear")
    x = fnum(args[0]).flatten()
    labels = _text_list(args[-1]) if len(args) > 1 and isinstance(args[-1], (Cell, StringArray)) else None
    res = ax.pie(x, labels=labels, autopct="%1.0f%%" if labels is None else None)
    ax.axis("equal")
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "pie", list(res[0]))
    h.extra.update({"axes": ax, "figure": ax.figure})
    return h


@builtin("quiver")
@_with_axes_arg
def _quiver(interp, args, nargout):
    args, opts = _split_name_values(args, {"color", "linewidth", "autoscale", "autoscalefactor",
                                           "displayname", "maxheadsize"})
    spec = next((text(a) for a in args if is_text(a) and _is_linespec(text(a))), "")
    nums = [a for a in args if not is_text(a)]
    scale = 1.0
    if len(nums) in (3, 5):
        scale = _scalar(nums.pop())
    if len(nums) == 2:
        U, V = np.real(fnum(nums[0])), np.real(fnum(nums[1]))
        X, Y = np.meshgrid(np.arange(1, U.shape[1] + 1), np.arange(1, U.shape[0] + 1))
    elif len(nums) == 4:
        X, Y, U, V = (np.real(fnum(a)) for a in nums)
        if X.shape != U.shape and X.size * Y.size == U.size:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    else:
        raise MatlabError("quiver expects quiver(U, V) or quiver(X, Y, U, V).")
    if "autoscalefactor" in opts:
        scale = _scalar(opts["autoscalefactor"])
    autoscale = not ("autoscale" in opts and not _onoff(opts["autoscale"])) and scale != 0
    x, y, u, v = (np.asarray(a, float).flatten() for a in (X, Y, U, V))
    if autoscale:
        mag = np.hypot(u, v)
        maxlen = float(np.nanmax(mag)) if mag.size else 0.0

        def spacing(vals):
            d = np.diff(np.unique(vals))
            return float(np.min(d)) if d.size else 0.0
        sp = [s for s in (spacing(x), spacing(y)) if s > 0]
        if not sp:
            rng = max(np.ptp(x) if x.size else 0, np.ptp(y) if y.size else 0)
            sp = [rng / max(math.sqrt(x.size), 1) or 1.0]
        k = 0.9 * scale * min(sp) / maxlen if maxlen > 0 else 1.0
    else:
        k = 1.0
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {}
    color = None
    if spec:
        m = re.search(r"[rgbcmykw]", spec)
        color = m.group(0) if m else None
    if "color" in opts:
        color = _color(opts["color"])
    kw["color"] = color if color is not None else "C0"
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    q = ax.quiver(x, y, u * k, v * k, angles="xy", scale_units="xy", scale=1, width=0.004, **kw)
    if "linewidth" in opts:
        q.set_linewidth(_scalar(opts["linewidth"]))
    ax.update_datalim(np.column_stack([np.concatenate([x, x + u * k]), np.concatenate([y, y + v * k])]))
    ax.autoscale_view()
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "quiver", q)
    h.extra.update({"UData": np.asarray(U, float), "VData": np.asarray(V, float), "XData": np.asarray(X, float),
                    "YData": np.asarray(Y, float)})
    return h


# -- surfaces and images ----------------------------------------------------------------------

def _shading(interp) -> str:
    return interp.state.get("shading", "faceted")


def _surface(interp, args, kind: str, nargout: int):
    args, opts = _split_name_values(args, {"edgecolor", "facecolor", "facealpha", "linestyle", "linewidth",
                                           "levellist", "showtext", "linecolor"})
    nums = [a for a in args if not is_text(a)]
    if not nums:
        raise MatlabError(f"{kind} needs data, e.g. {kind}(peaks).")
    if len(nums) <= 2:
        Z = np.real(fnum(nums[0]))
        X, Y = np.meshgrid(np.arange(1, Z.shape[1] + 1), np.arange(1, Z.shape[0] + 1))
        extra = nums[1:]
    else:
        X, Y, Z = (np.real(fnum(a)) for a in nums[:3])
        if X.shape != Z.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
        extra = nums[3:]
    if kind in ("contour", "contourf"):
        ax = _axes(interp, projection="rectilinear")
        levels = None
        if extra:
            lv = np.real(fnum(extra[0])).flatten()
            levels = int(lv[0]) if lv.size == 1 else (np.array([lv[0], lv[0]]) if lv.size == 2 and lv[0] == lv[1]
                                                     else lv)
            if isinstance(levels, np.ndarray) and levels.size == 2 and levels[0] == levels[1]:
                levels = levels[:1]
        if "levellist" in opts:
            levels = np.real(fnum(opts["levellist"])).flatten()
        kw: dict = {"cmap": _current_cmap(interp)}
        if "linecolor" in opts:
            kw = {"colors": [_color(opts["linecolor"])]}
        if "linewidth" in opts:
            kw["linewidths"] = _scalar(opts["linewidth"])
        cs = getattr(ax, kind)(X, Y, Z, levels=levels, **kw) if levels is not None else \
            getattr(ax, kind)(X, Y, Z, **kw)
        if "showtext" in opts and _onoff(opts["showtext"]):
            ax.clabel(cs, inline=True, fontsize=8)
        _redraw(interp)
        return _ret(interp, "contour", [cs], nargout)
    ax = _axes(interp, projection="3d")
    rc = min(Z.shape[0], 200)
    cc = min(Z.shape[1], 200)
    shade = _shading(interp)
    if kind == "surf":
        edge = "none" if shade in ("flat", "interp") else "k"
        kw = {"cmap": _current_cmap(interp), "edgecolor": edge, "linewidth": 0.2 if edge != "none" else 0,
              "rcount": rc, "ccount": cc, "antialiased": shade == "interp"}
        if "edgecolor" in opts:
            ec = opts["edgecolor"]
            kw["edgecolor"] = "none" if is_text(ec) and text(ec).lower() in ("none", "flat", "interp") \
                else _color(ec)
            kw["linewidth"] = 0.3
        if "facealpha" in opts:
            kw["alpha"] = _scalar(opts["facealpha"])
        coll = ax.plot_surface(X, Y, Z, **kw)
        if "facecolor" in opts and not (is_text(opts["facecolor"]) and
                                        text(opts["facecolor"]).lower() in ("flat", "interp")):
            coll.set_facecolor(_color(opts["facecolor"]))
        coll._mylab_surface = True
    else:
        coll = ax.plot_wireframe(X, Y, Z, linewidth=0.5, rcount=rc, ccount=cc)
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "surface", coll)
    h.extra.update({"XData": X, "YData": Y, "ZData": Z})
    return h


builtin("surf", "surfc")(_with_axes_arg(lambda interp, args, nargout: _surface(interp, args, "surf", nargout)))
builtin("mesh", "meshc")(_with_axes_arg(lambda interp, args, nargout: _surface(interp, args, "mesh", nargout)))
builtin("contour")(_with_axes_arg(lambda interp, args, nargout: _surface(interp, args, "contour", nargout)))
builtin("contourf")(_with_axes_arg(lambda interp, args, nargout: _surface(interp, args, "contourf", nargout)))


@builtin("shading")
@_with_axes_arg
def _shading_b(interp, args, nargout):
    mode = text(args[-1]).lower() if args else "faceted"
    if mode not in ("flat", "interp", "faceted"):
        raise MatlabError("shading must be 'flat', 'interp' or 'faceted'.")
    interp.state["shading"] = mode
    st = _state(interp)
    if st.override_ax is None and (st.current is None or st.current not in st.figures):
        return None
    ax = _axes(interp, fresh=False)
    for c in ax.collections:
        if getattr(c, "_mylab_surface", False):
            if mode == "faceted":
                c.set_edgecolor("k")
                c.set_linewidth(0.2)
            else:
                c.set_edgecolor("none")
                c.set_linewidth(0)
        elif c.__class__.__name__ == "QuadMesh":
            c.set_edgecolor("k" if mode == "faceted" else "face")
    _redraw(interp)
    return None


@builtin("peaks")
def _peaks(interp, args, nargout):
    if not args:
        v = np.linspace(-3, 3, 49)
        X, Y = np.meshgrid(v, v)
    elif len(args) == 1:
        a = np.real(fnum(args[0]))
        if a.size == 1:
            n = int(a.flat[0])
            if n < 1:
                raise MatlabError("peaks(n) needs a positive n.")
            v = np.linspace(-3, 3, n) if n > 1 else np.array([3.0])
        else:
            v = a.flatten()
        X, Y = np.meshgrid(v, v)
    else:
        X, Y = np.real(fnum(args[0])), np.real(fnum(args[1]))
        if X.shape != Y.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    Z = 3 * (1 - X) ** 2 * np.exp(-(X ** 2) - (Y + 1) ** 2) \
        - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2) \
        - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2)
    if nargout == 0:
        _surface(interp, [X, Y, Z], "surf", 0)
        _axes(interp, fresh=False).set_title("Peaks")
        return None
    if nargout >= 2:
        return X, Y, Z
    return Z


@builtin("imagesc", "image", "imshow")
@_with_axes_arg
def _imagesc(interp, args, nargout):
    nums = [a for a in args if not is_text(a)]
    clim = None
    x = y = None
    if len(nums) == 2 and fnum(nums[1]).size == 2 and fnum(nums[0]).size > 2:
        C, clim = np.real(fnum(nums[0])), np.real(fnum(nums[1])).flatten()
    elif len(nums) >= 3:
        x, y, C = np.real(fnum(nums[0])).flatten(), np.real(fnum(nums[1])).flatten(), np.real(fnum(nums[2]))
        if len(nums) > 3:
            clim = np.real(fnum(nums[3])).flatten()
    else:
        C = np.real(fnum(nums[0])) if nums else np.zeros((1, 1))
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"aspect": "auto", "cmap": _current_cmap(interp), "origin": "upper"}
    if C.ndim == 3:
        kw.pop("cmap")
    if x is not None and y is not None and C.ndim >= 2:
        nr, nc = C.shape[:2]
        x0, x1 = (x[0], x[-1]) if x.size > 1 else (x[0], x[0] + nc - 1)
        y0, y1 = (y[0], y[-1]) if y.size > 1 else (y[0], y[0] + nr - 1)
        dx = (x1 - x0) / max(nc - 1, 1) / 2 if nc > 1 else 0.5
        dy = (y1 - y0) / max(nr - 1, 1) / 2 if nr > 1 else 0.5
        kw["extent"] = (x0 - dx, x1 + dx, y1 + dy, y0 - dy)
    if clim is not None:
        kw["vmin"], kw["vmax"] = float(clim[0]), float(clim[1])
    im = ax.imshow(C, **kw)
    _redraw(interp)
    return _ret(interp, "image", [im], nargout)


@builtin("colorbar")
@_with_axes_arg
def _colorbar(interp, args, nargout):
    args, opts = _split_name_values(args, {"location", "fontsize", "ticks", "ticklabels", "limits"})
    ax = _axes(interp, fresh=False)
    if args and is_text(args[0]) and text(args[0]).lower() in ("off", "delete", "hide"):
        cb = getattr(ax, "_mylab_colorbar", None)
        if cb is not None:
            cb.remove()
            ax._mylab_colorbar = None
        return None
    loc = text(opts["location"]).lower() if "location" in opts else \
        next((text(a).lower() for a in args if is_text(a)), "eastoutside")
    location = {"eastoutside": "right", "east": "right", "westoutside": "left", "west": "left",
                "northoutside": "top", "north": "top", "southoutside": "bottom", "south": "bottom"}.get(loc, "right")
    old = getattr(ax, "_mylab_colorbar", None)
    if old is not None:
        try:
            old.remove()
        except (ValueError, AttributeError, KeyError):
            pass
    ms = _mappables(ax)
    if ms:
        mappable = ms[-1]
    else:
        import matplotlib.cm as cm
        mappable = cm.ScalarMappable(cmap=_current_cmap(interp))
    cb = ax.figure.colorbar(mappable, ax=ax, location=location)
    cb._mylab_parent_ax = ax
    ax._mylab_colorbar = cb
    h = _wrap(interp, "colorbar", cb)
    h.extra["Location"] = loc
    for name, key in (("fontsize", "FontSize"), ("ticks", "Ticks"), ("ticklabels", "TickLabels"),
                      ("limits", "Limits")):
        if name in opts:
            _set_prop(h, key, opts[name])
    _redraw(interp)
    return h if nargout > 0 else None


# colormaps ------------------------------------------------------------------------------------

_CMAP_ALIASES = {"parula": "viridis", "default": "viridis", "lines": "tab10", "colorcube": "nipy_spectral",
                 "prism": "prism", "flag": "flag", "white": "Greys", "sky": "Blues", "abyss": "cividis"}

_CMAP_FUNCS = ("parula", "jet", "gray", "hot", "cool", "hsv", "bone", "copper", "spring", "summer", "autumn",
               "winter", "turbo", "pink")


def _cmap_name(name: str) -> str:
    import matplotlib as mpl
    n = _CMAP_ALIASES.get(name.lower(), name)
    if n not in mpl.colormaps:
        low = n.lower()
        match = next((k for k in mpl.colormaps if k.lower() == low), None)
        if match is None:
            raise MatlabError(f"Unknown colormap '{name}'.")
        n = match
    return n


def _current_cmap(interp):
    return interp.state.get("colormap", "viridis")


def _to_cmap(v):
    """A matplotlib colormap (or its name) from a name or an n-by-3 matrix."""
    if is_text(v):
        return _cmap_name(text(v))
    m = np.real(fnum(v))
    if m.ndim != 2 or m.shape[1] != 3 or m.shape[0] < 1:
        raise MatlabError("A colormap must be an m-by-3 matrix of RGB values between 0 and 1.")
    if m.min() < 0 or m.max() > 1:
        raise MatlabError("Colormap values must be between 0 and 1.")
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(m)
    cmap._mylab_user = True
    return cmap


def _cmap_matrix(cmap, n: int) -> np.ndarray:
    import matplotlib as mpl
    if isinstance(cmap, str):
        cmap = mpl.colormaps[_cmap_name(cmap)]
    if n <= 0:
        return np.zeros((0, 3))
    if cmap.name == "tab10" or getattr(cmap, "N", 256) < 32 and cmap.__class__.__name__ == "ListedColormap" \
            and cmap.name in ("tab10",):
        cols = np.array([cmap(i % cmap.N)[:3] for i in range(n)])
        return np.round(cols, 4)
    return np.round(np.asarray(cmap(np.linspace(0, 1, n)))[:, :3], 4)


def _apply_colormap(interp, v, axes=None) -> None:
    cmap = _to_cmap(v)
    interp.state["colormap"] = cmap
    if axes is None:
        st = _state(interp)
        fig = st.figures.get(st.current) if st.current is not None else None
        axes = list(fig.axes) if fig is not None else []
    for ax in axes:
        for m in _mappables(ax):
            try:
                m.set_cmap(cmap)
            except (AttributeError, ValueError):
                pass


@builtin("colormap")
def _colormap(interp, args, nargout):
    if args and isinstance(args[0], GraphicsHandle):
        target = args[0]
        args = list(args[1:])
        axes = [target.obj] if target.kind in ("axes", "polaraxes") else \
            (list(target.obj.axes) if target.kind == "figure" else None)
    else:
        axes = None
    if args:
        _apply_colormap(interp, args[0], axes)
    if nargout > 0:
        return _cmap_query(interp)
    return None


def _cmap_query(interp) -> np.ndarray:
    cmap = _current_cmap(interp)
    if getattr(cmap, "_mylab_user", False):
        return np.asarray(cmap.colors, dtype=float)[:, :3]
    return _cmap_matrix(cmap, 256)


def _make_cmap_fn(name: str):
    def f(interp, args, nargout):
        n = to_int(args[0], "number of colors") if args else 256
        return _cmap_matrix(name if name != "lines" else "tab10", n)
    return f


for _nm in _CMAP_FUNCS:
    builtin(_nm)(_make_cmap_fn(_nm))


@builtin("lines")
def _lines_cmap(interp, args, nargout):
    import matplotlib as mpl
    from matplotlib.colors import to_rgb
    n = to_int(args[0], "number of colors") if args else 7
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return np.round(np.array([to_rgb(colors[i % len(colors)]) for i in range(n)]).reshape(n, 3), 4)


@builtin("view")
@_with_axes_arg
def _view(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    if hasattr(ax, "view_init"):
        if len(args) == 2:
            ax.view_init(elev=_scalar(args[1]), azim=_scalar(args[0]))
        elif args:
            v = np.real(fnum(args[0])).flatten()
            if v.size == 1 and int(v[0]) == 2:
                ax.view_init(elev=90, azim=-90)
            elif v.size == 1 and int(v[0]) == 3:
                ax.view_init(elev=30, azim=-60)
            elif v.size >= 2:
                ax.view_init(elev=float(v[1]), azim=float(v[0]))
    if nargout > 0:
        return (_row([ax.azim, ax.elev]) if hasattr(ax, "azim") else _row([0, 90]))
    return None


# -- figure / axes management -----------------------------------------------------------

def _fig_num_arg(interp, v) -> int:
    if isinstance(v, GraphicsHandle):
        fig = v.obj if v.kind == "figure" else _obj_figure(v)
        n = _fig_number(interp, fig) if fig is not None else None
        if n is None:
            raise MatlabError("Invalid or deleted figure handle.")
        return n
    if isinstance(v, GraphicsArray) and v.items:
        return _fig_num_arg(interp, v.items[0])
    return to_int(v, "figure number")


@builtin("figure")
def _figure_b(interp, args, nargout):
    st = _state(interp)
    if args and not is_text(args[0]):
        n = _fig_num_arg(interp, args[0])
        if n < 1:
            raise MatlabError("Figure number must be a positive integer.")
        fig = _figure(interp, n)
        rest = list(args[1:])
    else:
        n = 1
        while n in st.figures:
            n += 1
        fig = _figure(interp, n)
        rest = list(args)
    h = _wrap(interp, "figure", fig)
    if len(rest) % 2:
        raise MatlabError("figure options must be name/value pairs.")
    for i in range(0, len(rest), 2):
        name = text(rest[i], "property name")
        key, _ = _lookup(h, name)
        if key is None:
            h.extra[name] = rest[i + 1]  # unknown figure options are accepted and ignored
            continue
        _set_prop(h, name, rest[i + 1])
    return h if nargout > 0 else None


@builtin("gcf")
def _gcf(interp, args, nargout):
    return _wrap(interp, "figure", _figure(interp))


@builtin("gca")
def _gca(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    return _wrap(interp, _axes_kind(ax), ax)


@builtin("clf")
def _clf(interp, args, nargout):
    if args and isinstance(args[0], GraphicsHandle):
        n = _fig_num_arg(interp, args[0])
        fig = _state(interp).figures[n]
    else:
        fig = _figure(interp)
        n = _state(interp).current
    fig.clf()
    st = _state(interp)
    st.axes.pop(n, None)
    st.layouts.pop(n, None)
    return None


@builtin("cla")
@_with_axes_arg
def _cla(interp, args, nargout):
    st = _state(interp)
    ax = _axes(interp, fresh=False)
    ax.cla()
    _yy_restyle(ax)
    if args and is_text(args[0]) and text(args[0]).lower() == "reset":
        st.hold[id(ax)] = False
    _redraw(interp)
    return None


def _close_number(interp, n: int) -> None:
    st = _state(interp)
    plt = _plt()
    if n in st.figures:
        fig = st.figures.pop(n)
        plt.close(fig)
        st.axes.pop(n, None)
        st.layouts.pop(n, None)
        if st.current == n:
            st.current = max(st.figures) if st.figures else None


@builtin("close")
def _close(interp, args, nargout):
    st = _state(interp)
    if args and is_text(args[0]):
        s = text(args[0]).lower()
        if s in ("all", "all hidden", "all force"):
            close_all(interp)
            return None
        if s in ("force", "hidden"):
            args = []
        else:
            for n, f in list(st.figures.items()):
                if getattr(f, "_mylab_name", None) == text(args[0]):
                    _close_number(interp, n)
                    return None
            raise MatlabError(f"Specified window '{text(args[0])}' does not exist.")
    if not args:
        if st.current is not None:
            _close_number(interp, st.current)
        return None
    targets = args[0]
    if isinstance(targets, (GraphicsHandle, GraphicsArray)):
        for h in _handles_of(targets):
            if h.kind != "figure":
                raise MatlabError("close only accepts figure handles.")
            n = _fig_number(interp, h.obj)
            if n is not None:
                _close_number(interp, n)
        return None
    for x in np.real(num(targets)).flatten():
        _close_number(interp, int(round(float(x))))
    return None


@builtin("hold")
@_with_axes_arg
def _hold(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    st = _state(interp)
    if not args:
        st.hold[id(ax)] = not st.hold.get(id(ax), False)
    else:
        mode = text(args[0]).lower()
        if mode not in ("on", "off", "all"):
            raise MatlabError("hold must be 'on' or 'off'.")
        st.hold[id(ax)] = mode in ("on", "all")
    pair = _yy_pair(ax)
    if pair is not None:
        for a in pair:
            st.hold[id(a)] = st.hold[id(ax)]
    return None


@builtin("ishold")
@_with_axes_arg
def _ishold(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    return bool(_state(interp).hold.get(id(ax), False))


@builtin("subplot")
def _subplot(interp, args, nargout):
    st = _state(interp)
    if len(args) == 1 and isinstance(args[0], GraphicsHandle):
        h = args[0]
        if h.kind not in ("axes", "polaraxes"):
            raise MatlabError("subplot(ax) needs an axes handle.")
        n = _fig_num_arg(interp, h)
        _figure(interp, n)
        st.axes[n] = h.obj
        return h if nargout > 0 else None
    fig = _figure(interp)
    args, opts = _split_name_values(args, {"position"})
    if "position" in opts and not args:
        pos = [float(x) for x in np.real(fnum(opts["position"])).flatten()[:4]]
        ax = fig.add_axes(pos)
        st.axes[st.current] = ax
        return _wrap(interp, "axes", ax) if nargout > 0 else None
    if len(args) == 1:
        code = to_int(args[0], "subplot")
        m, n, p = code // 100, (code // 10) % 10, code % 10
        pspec = [p]
    else:
        m, n = to_int(args[0], "m"), to_int(args[1], "n")
        pspec = [int(x) for x in num(args[2]).flatten()]
    if m < 1 or n < 1 or any(q < 1 or q > m * n for q in pspec):
        raise MatlabError("Index exceeds number of subplots.")
    rows = [(q - 1) // n for q in pspec]
    cols = [(q - 1) % n for q in pspec]
    r0, r1, c0, c1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
    # Reuse an existing axes occupying exactly the same grid cells (MATLAB does).
    ax = None
    for a in fig.axes:
        ss = a.get_subplotspec() if hasattr(a, "get_subplotspec") else None
        if ss is None or _is_colorbar_axes(a):
            continue
        g = ss.get_gridspec()
        if g.get_geometry() == (m, n) and ss.rowspan == range(r0, r1) and ss.colspan == range(c0, c1):
            ax = a
            break
    if ax is None:
        if len(pspec) == 1:
            ax = fig.add_subplot(m, n, pspec[0])
        else:
            gs = fig.add_gridspec(m, n)
            ax = fig.add_subplot(gs[r0:r1, c0:c1])
    st.axes[st.current] = ax
    return _wrap(interp, "axes", ax) if nargout > 0 else None


# tiled layouts ---------------------------------------------------------------------------------

class _Layout:
    """A tiledlayout: a grid of axes with a fixed size or one that grows ('flow')."""

    def __init__(self, fig, m: int, n: int, mode: str = "fixed"):
        self.fig = fig
        self.m, self.n = m, n
        self.mode = mode  # 'fixed', 'flow', 'vertical', 'horizontal'
        self.tiles: dict[int, Any] = {}   # tile index -> axes (fixed grid)
        self.order: list = []              # axes in creation order
        self.next = 1
        self.spacing = "loose"
        self.padding = "loose"

    def grid_size(self) -> tuple[int, int]:
        if self.mode == "fixed":
            return self.m, self.n
        k = max(len(self.order), 1)
        if self.mode == "vertical":
            return k, 1
        if self.mode == "horizontal":
            return 1, k
        cols = math.ceil(math.sqrt(k))
        return math.ceil(k / cols), cols

    def replace(self, old, new):
        self.order = [new if a is old else a for a in self.order]
        for k, a in list(self.tiles.items()):
            if a is old:
                self.tiles[k] = new

    def apply_spacing(self):
        pads = {"loose": 0.06, "compact": 0.02, "tight": 0.0, "none": 0.0}
        try:
            self.fig.set_layout_engine("constrained")
            eng = self.fig.get_layout_engine()
            p = pads.get(self.padding, 0.04)
            s = pads.get(self.spacing, 0.04)
            eng.set(w_pad=p, h_pad=p, wspace=s, hspace=s)
        except (AttributeError, ValueError, RuntimeError, TypeError):
            pass

    def relayout(self):
        rows, cols = self.grid_size()
        gs = self.fig.add_gridspec(rows, cols)
        for i, ax in enumerate(self.order):
            if ax in self.fig.axes:
                ax.set_subplotspec(gs[i // cols, i % cols])
        return gs

    def new_tile(self, index: int | None = None, span=(1, 1)):
        if self.mode != "fixed":
            if index is not None and index <= len(self.order):
                return self.order[index - 1]
            self.order.append(None)
            gs = self.relayout()
            rows, cols = self.grid_size()
            i = len(self.order) - 1
            ax = self.fig.add_subplot(gs[i // cols, i % cols])
            self.order[-1] = ax
            return ax
        total = self.m * self.n
        if index is None:
            k = self.next
            while k in self.tiles and self.tiles[k] in self.fig.axes:
                k += 1
            index = k
        if index < 1 or index > total:
            raise MatlabError(f"Tile index must be between 1 and {total}.")
        existing = self.tiles.get(index)
        if existing is not None and existing in self.fig.axes and span == (1, 1):
            return existing
        r, c = (index - 1) // self.n, (index - 1) % self.n
        rs, cs = span
        if r + rs > self.m or c + cs > self.n:
            raise MatlabError("The tile span does not fit in the layout.")
        gs = self.fig.add_gridspec(self.m, self.n)
        ax = self.fig.add_subplot(gs[r:r + rs, c:c + cs])
        for dr in range(rs):
            for dc in range(cs):
                self.tiles[(r + dr) * self.n + (c + dc) + 1] = ax
        self.order.append(ax)
        self.next = index + 1
        return ax


def _layout_of_axes(interp, ax):
    for lay in _state(interp).layouts.values():
        if ax in lay.order:
            return lay
    return None


@builtin("tiledlayout")
def _tiledlayout(interp, args, nargout):
    args, opts = _split_name_values(args, {"tilespacing", "padding", "tileindexing"})
    st = _state(interp)
    fig = _figure(interp)
    fig.clf()
    st.axes.pop(st.current, None)
    if args and is_text(args[0]):
        mode = text(args[0]).lower()
        if mode not in ("flow", "vertical", "horizontal"):
            raise MatlabError("tiledlayout arrangement must be 'flow', 'vertical' or 'horizontal'.")
        lay = _Layout(fig, 0, 0, mode)
    elif len(args) >= 2:
        m, n = to_int(args[0], "rows"), to_int(args[1], "columns")
        if m < 1 or n < 1:
            raise MatlabError("tiledlayout needs a positive number of rows and columns.")
        lay = _Layout(fig, m, n)
    elif len(args) == 1:
        sz = np.real(fnum(args[0])).flatten()
        lay = _Layout(fig, int(sz[0]), int(sz[1]))
    else:
        lay = _Layout(fig, 0, 0, "flow")
    if "tilespacing" in opts:
        lay.spacing = text(opts["tilespacing"]).lower()
    if "padding" in opts:
        lay.padding = text(opts["padding"]).lower()
    lay.apply_spacing()
    st.layouts[st.current] = lay
    return _wrap(interp, "tiledlayout", lay) if nargout > 0 else None


@builtin("nexttile")
def _nexttile(interp, args, nargout):
    st = _state(interp)
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind == "tiledlayout":
        lay = args[0].obj
        _figure(interp, _fig_number(interp, lay.fig))
        args = list(args[1:])
    else:
        fig = _figure(interp)
        lay = st.layouts.get(st.current)
        if lay is None or lay.fig is not fig:
            fig.clf()
            lay = _Layout(fig, 0, 0, "flow")
            lay.apply_spacing()
            st.layouts[st.current] = lay
    index = None
    span = (1, 1)
    nums = [a for a in args if not is_text(a)]
    for a in nums:
        v = np.real(fnum(a)).flatten()
        if v.size == 1:
            index = int(v[0])
        elif v.size == 2:
            span = (int(v[0]), int(v[1]))
    ax = lay.new_tile(index, span)
    st.axes[st.current] = ax
    return _wrap(interp, "axes", ax) if nargout > 0 else None


# yyaxis ----------------------------------------------------------------------------------------

def _yy_pair(ax):
    base = getattr(ax, "_mylab_yybase", None) or ax
    right = getattr(base, "_mylab_yyright", None)
    return (base, right) if right is not None else None


@builtin("yyaxis")
@_with_axes_arg
def _yyaxis(interp, args, nargout):
    if not args or not is_text(args[-1]) or text(args[-1]).lower() not in ("left", "right"):
        raise MatlabError("yyaxis needs 'left' or 'right'.")
    side = text(args[-1]).lower()
    st = _state(interp)
    ax = _axes(interp, fresh=False, projection="rectilinear")
    base = getattr(ax, "_mylab_yybase", None) or ax
    right = getattr(base, "_mylab_yyright", None)
    if right is None or right not in base.figure.axes:
        right = base.twinx()
        base._mylab_yyright = right
        right._mylab_yybase = base
        base._mylab_yyside = "left"
        right._mylab_yyside = "right"
        st.hold[id(right)] = st.hold.get(id(base), False)
        # restyle the left side without clearing what is already plotted
        _yy_restyle(base)
        _yy_restyle(right)
        for line in base.lines:
            line.set_color(base.yaxis.label.get_color())
    target = right if side == "right" else base
    if st.override_ax is not None:
        st.override_ax = target
    st.axes[_fig_number(interp, base.figure)] = target
    return None


# -- labels, titles and legends ------------------------------------------------------------

def _label_target(interp, args):
    """Handle an optional axes/layout handle before a label call; returns (target, args)."""
    if args and isinstance(args[0], GraphicsHandle):
        h = args[0]
        if h.kind == "tiledlayout":
            return h.obj, list(args[1:])
        if h.kind in ("axes", "polaraxes"):
            return h.obj, list(args[1:])
    return None, list(args)


_TEXT_OPTS = {"fontsize", "interpreter", "fontweight", "color", "fontname", "rotation", "horizontalalignment",
              "verticalalignment", "fontangle", "backgroundcolor", "edgecolor", "units"}


def _text_kwargs(opts) -> dict:
    kw = {}
    for k, v in opts.items():
        if k == "fontsize":
            kw["fontsize"] = _scalar(v)
        elif k == "fontweight":
            kw["fontweight"] = text(v).lower()
        elif k == "color":
            kw["color"] = _color(v)
        elif k == "fontname":
            kw["fontfamily"] = text(v)
        elif k == "rotation":
            kw["rotation"] = _scalar(v)
        elif k == "horizontalalignment":
            kw["horizontalalignment"] = text(v).lower()
        elif k == "verticalalignment":
            kw["verticalalignment"] = {"middle": "center", "cap": "top"}.get(text(v).lower(), text(v).lower())
        elif k == "fontangle":
            kw["fontstyle"] = "italic" if text(v).lower() == "italic" else "normal"
        elif k == "backgroundcolor":
            kw["backgroundcolor"] = _color(v)
    return kw


def _make_subtitle(ax, s: str, **kw):
    t = getattr(ax, "_mylab_subtitle", None)
    if t is None:
        fs = kw.pop("fontsize", ax.title.get_fontsize() * 0.85)
        t = ax.text(0.5, 1.0, s, transform=ax.transAxes, ha="center", va="bottom", fontsize=fs, **kw)
        t._mylab_is_subtitle_of = ax
        ax._mylab_subtitle = t
    else:
        t.set_text(s)
        t.update(kw)
    _layout_subtitle(ax)
    return t


def _layout_subtitle(ax):
    from matplotlib.transforms import offset_copy
    t = getattr(ax, "_mylab_subtitle", None)
    if t is None:
        return
    has = bool(t.get_text())
    t.set_transform(offset_copy(ax.transAxes, fig=ax.figure, y=6, units="points"))
    lines = max(t.get_text().count("\n") + 1, 1)
    ax.set_title(ax.get_title(), pad=(6 + lines * t.get_fontsize() * 1.3) if has else 6.0)


def _label(method):
    def f(interp, args, nargout):
        target, args = _label_target(interp, args)
        if not args:
            raise MatlabError("Not enough input arguments.")
        sub = None
        if method == "set_title" and len(args) > 1:
            a = args[1]
            if isinstance(a, (Cell, StringArray)) or (is_text(a) and text(a).lower() not in _TEXT_OPTS):
                sub = a
                args = [args[0]] + list(args[2:])
        s = _label_text(args[0])
        _, opts = _split_name_values(args[1:], _TEXT_OPTS)
        kw = _text_kwargs(opts)
        if isinstance(target, _Layout):
            fig = target.fig
            fn = {"set_title": fig.suptitle, "suptitle": fig.suptitle, "set_xlabel": fig.supxlabel,
                  "set_ylabel": fig.supylabel}.get(method)
            if fn is None:
                raise MatlabError("This label cannot be applied to a tiled layout.")
            t = fn(s, **kw)
            _redraw(interp)
            return _wrap(interp, "text", t) if nargout > 0 else None
        if method == "suptitle":
            fig = target.figure if target is not None else _figure(interp)
            t = fig.suptitle(s, **kw)
            _redraw(interp)
            return _wrap(interp, "text", t) if nargout > 0 else None
        st = _state(interp)
        prev = st.override_ax
        if target is not None:
            st.override_ax = target
        try:
            ax = _axes(interp, fresh=False)
        finally:
            st.override_ax = prev
        if method == "set_zlabel" and not hasattr(ax, "set_zlabel"):
            t = ax.text(0, 0, "")  # 2-D axes: nothing to label, return a dummy text
            t.set_visible(False)
        else:
            if method == "set_title" and "fontweight" not in kw:
                kw["fontweight"] = "bold"
            getattr(ax, method)(s, **kw)
            t = {"set_title": ax.title, "set_xlabel": ax.xaxis.label,
                 "set_ylabel": ax.yaxis.label}.get(method) or getattr(getattr(ax, "zaxis", None), "label", None)
            if method == "set_title":
                _layout_subtitle(ax)
        th = _wrap(interp, "text", t)
        if "interpreter" in opts:
            th.extra["Interpreter"] = text(opts["interpreter"])
        if sub is not None:
            st_t = _make_subtitle(ax, _label_text(sub))
            _redraw(interp)
            if nargout > 1:
                return th, _wrap(interp, "text", st_t)
        _redraw(interp)
        return th if nargout > 0 else None
    return f


builtin("xlabel")(_label("set_xlabel"))
builtin("ylabel")(_label("set_ylabel"))
builtin("zlabel")(_label("set_zlabel"))
builtin("title")(_label("set_title"))
builtin("sgtitle", "suptitle")(_label("suptitle"))


@builtin("subtitle")
def _subtitle(interp, args, nargout):
    target, args = _label_target(interp, args)
    if not args:
        raise MatlabError("Not enough input arguments.")
    st = _state(interp)
    prev = st.override_ax
    if target is not None and not isinstance(target, _Layout):
        st.override_ax = target
    try:
        ax = _axes(interp, fresh=False)
    finally:
        st.override_ax = prev
    _, opts = _split_name_values(args[1:], _TEXT_OPTS)
    t = _make_subtitle(ax, _label_text(args[0]), **_text_kwargs(opts))
    _redraw(interp)
    return _wrap(interp, "text", t) if nargout > 0 else None


_LOC = {"northeast": ("upper right", None), "northwest": ("upper left", None),
        "southeast": ("lower right", None), "southwest": ("lower left", None),
        "north": ("upper center", None), "south": ("lower center", None),
        "east": ("center right", None), "west": ("center left", None), "best": ("best", None),
        "northeastoutside": ("upper left", (1.02, 1.0)), "eastoutside": ("center left", (1.02, 0.5)),
        "southeastoutside": ("lower left", (1.02, 0.0)), "northwestoutside": ("upper right", (-0.12, 1.0)),
        "westoutside": ("center right", (-0.12, 0.5)), "southwestoutside": ("lower right", (-0.12, 0.0)),
        "northoutside": ("lower center", (0.5, 1.02)), "southoutside": ("upper center", (0.5, -0.12)),
        "bestoutside": ("upper left", (1.02, 1.0)), "none": ("best", None)}


def _legend_handles(ax) -> list:
    pair = _yy_pair(ax)
    axes = list(pair) if pair is not None else [ax]
    handles = []
    try:
        from matplotlib.legend import _get_legend_handles
        handles = list(_get_legend_handles(axes))
    except (ImportError, AttributeError, TypeError):
        for a in axes:
            handles.extend(a.lines)
    # Each histogram/bar container counts once; its patches are not separate entries.
    from matplotlib.container import Container
    inside = set()
    for h in handles:
        if isinstance(h, Container):
            inside.update(id(p) for p in _artists(h))
    return [h for h in handles if id(h) not in inside]


def _refresh_legend(interp, ax) -> None:
    leg = ax.get_legend()
    if leg is None:
        return
    _make_legend(interp, ax, reuse=leg)


def _make_legend(interp, ax, labels=None, handles=None, location=None, orientation=None, ncols=None,
                 reuse=None, fontsize=None, box=None, title=None):
    pair = _yy_pair(ax)
    host = pair[1] if pair is not None else ax
    if reuse is not None:
        location = location or getattr(reuse, "_mylab_loc", None)
        orientation = orientation or getattr(reuse, "_mylab_orientation", None)
        if box is None:
            box = reuse.get_frame_on()
        if fontsize is None and reuse.get_texts():
            fontsize = reuse.get_texts()[0].get_fontsize()
        if labels is None:
            labels = getattr(reuse, "_mylab_labels", None)
        if handles is None:
            handles = getattr(reuse, "_mylab_handles", None)
    if handles is None:
        handles = _legend_handles(ax)
    if labels is None:
        labels = []
        count = 0
        for h in handles:
            lab = h.get_label() if hasattr(h, "get_label") else ""
            count += 1
            labels.append(lab if lab and not lab.startswith("_") else f"data{count}")
        keep_labels = None
    else:
        keep_labels = labels
        handles = handles[:len(labels)]
        labels = labels[:len(handles)]
    for a in ([pair[0], pair[1]] if pair is not None else [ax]):
        old = a.get_legend()
        if old is not None:
            old.remove()
    kw: dict = {}
    loc_name = (location or "northeast").lower()
    loc, anchor = _LOC.get(loc_name, ("best", None))
    kw["loc"] = loc
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    orient = (orientation or "vertical").lower()
    if ncols is not None:
        kw["ncols"] = ncols
    elif orient == "horizontal":
        kw["ncols"] = max(len(handles), 1)
    if fontsize is not None:
        kw["fontsize"] = fontsize
    if title is not None:
        kw["title"] = title
    leg = host.legend(handles, labels, **kw)
    if box is not None:
        leg.set_frame_on(box)
    leg._mylab_loc = loc_name
    if reuse is not None and reuse is not leg:  # keep existing handles pointing at the legend
        st = _state(interp)
        h = st.handles.pop(id(reuse), None)
        if h is not None:
            h.obj = leg
            st.handles[id(leg)] = h
    leg._mylab_orientation = orient
    leg._mylab_labels = keep_labels
    leg._mylab_handles = handles if keep_labels is not None else None
    return leg


@builtin("legend")
def _legend(interp, args, nargout):
    st = _state(interp)
    target = None
    handles = None
    args = list(args)
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind in ("axes", "polaraxes"):
        target = args.pop(0).obj
    if args and isinstance(args[0], (GraphicsHandle, GraphicsArray)):
        hs = _handles_of(args.pop(0))
        handles = []
        for h in hs:
            o = h.obj
            if isinstance(o, _Hist):
                o = o.container
            elif isinstance(o, list):
                o = o[0] if o else None
            if o is not None:
                handles.append(o)
        if target is None and hs:
            target = _obj_axes(hs[0])
    args, opts = _split_name_values(args, {"location", "fontsize", "orientation", "box", "numcolumns",
                                           "textcolor", "interpreter", "autoupdate", "title", "edgecolor",
                                           "color"})
    prev = st.override_ax
    if target is not None:
        st.override_ax = target
    try:
        ax = _axes(interp, fresh=False)
    finally:
        st.override_ax = prev
    pair = _yy_pair(ax)
    if pair is not None:
        ax = pair[0]
    existing = (pair[1].get_legend() or pair[0].get_legend()) if pair is not None else ax.get_legend()
    cmds = [text(a).lower() for a in args if is_text(a) and text(a).lower() in
            ("off", "hide", "show", "boxoff", "boxon", "toggle")]
    labels = []
    for a in args:
        if isinstance(a, (Cell, StringArray)):
            labels.extend(_text_list(a))
        elif is_text(a) and text(a).lower() not in ("off", "hide", "show", "boxoff", "boxon", "toggle"):
            labels.append(_label_text(a))
    if "off" in cmds or "hide" in cmds or ("toggle" in cmds and existing is not None and existing.get_visible()):
        if existing is not None:
            if "hide" in cmds or "toggle" in cmds:
                existing.set_visible(False)
            else:
                existing.remove()
        return None
    if cmds and all(c in ("boxoff", "boxon") for c in cmds) and existing is not None and not labels \
            and not opts:
        existing.set_frame_on(cmds[-1] == "boxon")
        _redraw(interp)
        return _wrap(interp, "legend", existing) if nargout > 0 else None
    box = None
    if "box" in opts:
        box = _onoff(opts["box"])
    if "boxoff" in cmds:
        box = False
    if "boxon" in cmds:
        box = True
    leg = _make_legend(interp, ax, labels=labels or None, handles=handles,
                       location=text(opts["location"]) if "location" in opts else None,
                       orientation=text(opts["orientation"]) if "orientation" in opts else None,
                       ncols=to_int(opts["numcolumns"], "NumColumns") if "numcolumns" in opts else None,
                       reuse=existing if (not labels and handles is None and existing is not None) else None,
                       fontsize=_scalar(opts["fontsize"]) if "fontsize" in opts else None,
                       box=box, title=_label_text(opts["title"]) if "title" in opts else None)
    if "textcolor" in opts:
        for t in leg.get_texts():
            t.set_color(_color(opts["textcolor"]))
    _redraw(interp)
    return _wrap(interp, "legend", leg) if nargout > 0 else None


@builtin("grid")
@_with_axes_arg
def _grid(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    mode = text(args[0]).lower() if args else "toggle"
    if mode == "minor":
        on = not getattr(ax, "_mylab_minor_grid", False)
        if on:
            ax.minorticks_on()
        ax.grid(on, which="minor", alpha=0.3)
        ax._mylab_minor_grid = on
    elif mode == "toggle":
        current = any(gl.get_visible() for gl in ax.xaxis.get_gridlines())
        ax.grid(not current)
    elif mode in ("on", "off"):
        ax.grid(mode == "on")
        if mode == "off":
            ax.grid(False, which="minor")
            ax._mylab_minor_grid = False
    else:
        raise MatlabError("grid must be 'on', 'off' or 'minor'.")
    _redraw(interp)
    return None


@builtin("box")
@_with_axes_arg
def _box(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    if args:
        mode = text(args[0]).lower()
        if mode not in ("on", "off"):
            raise MatlabError("box must be 'on' or 'off'.")
        on = mode == "on"
    else:
        on = not (ax.spines["top"].get_visible() if "top" in ax.spines else True)
    for name in ("top", "right"):
        if name in ax.spines:
            ax.spines[name].set_visible(on)
    _redraw(interp)
    return None


@builtin("axis")
@_with_axes_arg
def _axis(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    for a in args:
        if is_text(a):
            mode = text(a).lower()
            if mode == "equal":
                try:
                    ax.set_aspect("equal", adjustable="datalim")
                except NotImplementedError:
                    ax.set_box_aspect(None)
            elif mode == "image":
                ax.set_aspect("equal", adjustable="box")
                ax.autoscale(enable=True, tight=True)
            elif mode == "square":
                ax.set_box_aspect(1)
            elif mode == "tight":
                ax.autoscale(enable=True, tight=True)
            elif mode == "padded":
                ax.autoscale(enable=True, tight=False)
            elif mode == "off":
                ax.set_axis_off()
            elif mode == "on":
                ax.set_axis_on()
            elif mode in ("auto", "normal"):
                ax.autoscale(True)
                try:
                    ax.set_aspect("auto")
                except NotImplementedError:
                    pass
                if mode == "normal":
                    ax.set_box_aspect(None)
            elif mode == "ij":
                if not ax.yaxis_inverted():
                    ax.invert_yaxis()
            elif mode == "xy":
                if ax.yaxis_inverted():
                    ax.invert_yaxis()
            elif mode == "manual":
                ax.set_autoscale_on(False)
            elif mode in ("vis3d", "fill"):
                pass
            else:
                raise MatlabError(f"Unknown axis option '{text(a)}'.")
        else:
            lim = np.real(fnum(a)).flatten()
            if lim.size not in (4, 6):
                raise MatlabError("axis limits must be [xmin xmax ymin ymax] or with zmin zmax as well.")
            _set_lim(ax, "x", lim[0], lim[1])
            _set_lim(ax, "y", lim[2], lim[3])
            if lim.size >= 6 and hasattr(ax, "set_zlim"):
                _set_lim(ax, "z", lim[4], lim[5])
    if not args and nargout:
        vals = [*sorted(ax.get_xlim()), *sorted(ax.get_ylim())]
        if hasattr(ax, "get_zlim"):
            vals += list(ax.get_zlim())
        return np.array([vals])
    _redraw(interp)
    return None


def _lim(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        getter = getattr(ax, f"get_{which}lim", None)
        if not args:
            return _row(sorted(getter())) if getter else None
        if is_text(args[0]):
            mode = text(args[0]).lower()
            if mode == "auto":
                if which == "z":
                    if hasattr(ax, "set_autoscalez_on"):
                        ax.set_autoscalez_on(True)
                        ax.autoscale_view()
                else:
                    getattr(ax, f"set_autoscale{which}_on")(True)
                    ax.autoscale_view(scalex=which == "x", scaley=which == "y")
            elif mode == "manual":
                if which != "z":
                    getattr(ax, f"set_autoscale{which}_on")(False)
            elif mode == "mode":
                auto = getattr(ax, f"get_autoscale{which}_on", lambda: True)()
                return char("auto" if auto else "manual")
            else:
                raise MatlabError(f"Invalid {which}lim option '{text(args[0])}'.")
            _redraw(interp)
            return None
        lim = np.real(fnum(args[0])).flatten()
        if lim.size != 2:
            raise MatlabError("Limits must be a 2-element vector.")
        _set_lim(ax, which, float(lim[0]), float(lim[1]))
        _redraw(interp)
        return None
    return _with_axes_arg(f)


builtin("xlim")(_lim("x"))
builtin("ylim")(_lim("y"))
builtin("zlim")(_lim("z"))


def _ticks(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        if not hasattr(ax, f"set_{which}ticks"):
            return None
        if not args:
            return _ticks_get(which)(_FakeAxesHandle(ax))
        if is_text(args[0]):
            if text(args[0]).lower() == "auto":
                _set_ticks_auto(ax, which)
            return None
        getattr(ax, f"set_{which}ticks")(np.real(fnum(args[0])).flatten())
        setattr(ax, f"_mylab_{which}ticks_fixed", True)
        _redraw(interp)
        return None
    return _with_axes_arg(f)


builtin("xticks")(_ticks("x"))
builtin("yticks")(_ticks("y"))
builtin("zticks")(_ticks("z"))


def _ticklabels_b(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        if not args:
            return _ticklabels_get(which)(_FakeAxesHandle(ax))
        _set_ticklabels(ax, which, args[0])
        _redraw(interp)
        return None
    return _with_axes_arg(f)


builtin("xticklabels")(_ticklabels_b("x"))
builtin("yticklabels")(_ticklabels_b("y"))


def _tickangle(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        if not args:
            return _rotation_get(which)(_FakeAxesHandle(ax))
        angle = _scalar(args[0])
        ax.tick_params(axis=which, labelrotation=angle)
        if which == "x" and angle:
            for lab in ax.get_xticklabels():
                lab.set_horizontalalignment("right" if 0 < angle < 90 else "center")
        _redraw(interp)
        return None
    return _with_axes_arg(f)


builtin("xtickangle")(_tickangle("x"))
builtin("ytickangle")(_tickangle("y"))


@builtin("text")
@_with_axes_arg
def _text(interp, args, nargout):
    args = list(args)
    nums = []
    while args and not is_text(args[0]) and not isinstance(args[0], (Cell, StringArray)):
        nums.append(args.pop(0))
    if len(nums) < 2 or not args:
        raise MatlabError("text expects text(x, y, txt).")
    s_arg = args.pop(0)
    _, opts = _split_name_values(args, _TEXT_OPTS)
    ax = _axes(interp, fresh=False)
    kw = {"verticalalignment": "center"}
    kw.update(_text_kwargs(opts))
    xs = np.real(fnum(nums[0])).flatten()
    ys = np.real(fnum(nums[1])).flatten()
    labels = _text_list(s_arg) if isinstance(s_arg, (Cell, StringArray)) and xs.size > 1 else [_label_text(s_arg)]
    texts = []
    for i in range(xs.size):
        lab = labels[i] if i < len(labels) else labels[-1]
        if len(nums) >= 3 and hasattr(ax, "text2D"):
            zs = np.real(fnum(nums[2])).flatten()
            texts.append(ax.text(float(xs[i]), float(ys[i]), float(zs[min(i, zs.size - 1)]), lab, **kw))
        else:
            texts.append(ax.text(float(xs[i]), float(ys[min(i, ys.size - 1)]), lab, **kw))
    _redraw(interp)
    if nargout <= 0:
        return None
    hs = [_wrap(interp, "text", t) for t in texts]
    for h in hs:
        if "interpreter" in opts:
            h.extra["Interpreter"] = text(opts["interpreter"])
    return hs[0] if len(hs) == 1 else GraphicsArray(hs)


@builtin("annotation")
def _annotation(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("annotation needs a type such as 'textbox', 'arrow' or 'line'.")
    kind = text(args[0]).lower()
    fig = _figure(interp)
    rest = list(args[1:])
    nums = []
    while rest and not is_text(rest[0]):
        nums.append(np.real(fnum(rest.pop(0))).flatten())
    _, opts = _split_name_values(rest, {"string", "fontsize", "color", "linewidth", "edgecolor",
                                        "backgroundcolor", "fitboxtotext", "horizontalalignment", "linestyle",
                                        "facecolor"})
    color = _color(opts["color"]) if "color" in opts else "k"
    lw = _scalar(opts["linewidth"]) if "linewidth" in opts else 1.0
    if kind in ("line", "arrow", "doublearrow", "textarrow"):
        if len(nums) < 2:
            raise MatlabError(f"annotation('{kind}', x, y) needs x and y positions.")
        x, y = nums[0], nums[1]
        style = {"line": "-", "arrow": "->", "doublearrow": "<->", "textarrow": "->"}[kind]
        s = _label_text(opts["string"]) if "string" in opts else ""
        art = fig.add_artist(_annotation_arrow(fig, x, y, style, color, lw, s,
                                               _scalar(opts["fontsize"]) if "fontsize" in opts else 10))
    elif kind in ("textbox", "rectangle", "ellipse"):
        pos = nums[0] if nums else np.array([0.1, 0.1, 0.2, 0.1])
        if kind == "textbox":
            s = _label_text(opts["string"]) if "string" in opts else ""
            bbox = {"facecolor": _color(opts["backgroundcolor"]) if "backgroundcolor" in opts else "none",
                    "edgecolor": _color(opts["edgecolor"]) if "edgecolor" in opts else "k"}
            if is_text(opts.get("edgecolor", char(""))) and text(opts.get("edgecolor", char("x"))) == "none":
                bbox["edgecolor"] = "none"
            art = fig.text(float(pos[0]), float(pos[1] + (pos[3] if pos.size > 3 else 0)), s,
                           transform=fig.transFigure, va="top", ha="left", color=color, bbox=bbox,
                           fontsize=_scalar(opts["fontsize"]) if "fontsize" in opts else 10)
            _redraw(interp)
            return _wrap(interp, "text", art) if nargout > 0 else None
        from matplotlib.patches import Ellipse, Rectangle
        fc = _color(opts["facecolor"]) if "facecolor" in opts else "none"
        if kind == "rectangle":
            patch = Rectangle((pos[0], pos[1]), pos[2], pos[3], transform=fig.transFigure, fill=fc != "none",
                              facecolor=fc, edgecolor=color, linewidth=lw)
        else:
            patch = Ellipse((pos[0] + pos[2] / 2, pos[1] + pos[3] / 2), pos[2], pos[3], transform=fig.transFigure,
                            facecolor=fc, edgecolor=color, linewidth=lw)
        art = fig.add_artist(patch)
    else:
        raise MatlabError(f"Unknown annotation type '{kind}'.")
    _redraw(interp)
    if nargout <= 0:
        return None
    h = _wrap(interp, "patch", art)
    h.extra["figure"] = fig
    return h


def _annotation_arrow(fig, x, y, style, color, lw, s, fontsize):
    from matplotlib.text import Annotation
    return Annotation(s, xy=(float(x[1]), float(y[1])), xytext=(float(x[0]), float(y[0])),
                      xycoords=fig.transFigure, textcoords=fig.transFigure, fontsize=fontsize,
                      arrowprops={"arrowstyle": style, "color": color, "lw": lw})


@builtin("drawnow", "shg", "refresh")
def _drawnow(interp, args, nargout):
    _redraw(interp)
    return None


# -- saving -------------------------------------------------------------------------------

_FORMATS = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "pdf": "pdf", "svg": "svg", "eps": "eps", "epsc": "eps",
            "epsc2": "eps", "eps2": "eps", "tiff": "tif", "tiffn": "tif", "tif": "tif", "bmp": "png",
            "meta": "png", "ps": "ps", "psc": "ps"}

_FIG_ERROR = "The .fig format is not supported; use .png, .pdf or .svg."


def _save(interp, n, fname: str, dpi: float, fmt: str | None = None) -> None:
    st = _state(interp)
    if n is None or n not in st.figures:
        raise MatlabError("No figure to save.")
    ext = os.path.splitext(fname)[1].lower().lstrip(".")
    if ext == "fig" or (fmt or "").lower() == "fig":
        raise MatlabError(_FIG_ERROR)
    if not ext:
        fname += "." + (fmt or "png")
    elif ext == "jpeg":
        pass
    try:
        st.figures[n].savefig(fname, dpi=dpi, bbox_inches="tight")
    except ValueError as e:
        raise MatlabError(f"Cannot save the figure as '{fname}': {e}") from e
    except OSError as e:
        raise MatlabError(f"Cannot write '{fname}': {e.strerror or e}") from e


@builtin("saveas")
def _saveas(interp, args, nargout):
    st = _state(interp)
    if len(args) >= 2 and not is_text(args[0]):
        n = _fig_num_arg(interp, args[0])
        fname = text(args[1])
        rest = args[2:]
    else:
        n = st.current
        fname = text(args[-1]) if args else ""
        rest = []
    if not fname:
        raise MatlabError("saveas needs a file name.")
    fmt = None
    if rest:
        f = text(rest[0]).lower().lstrip("-")
        if f.startswith("d") and f[1:] in _FORMATS:
            f = f[1:]
        if f == "fig":
            raise MatlabError(_FIG_ERROR)
        fmt = _FORMATS.get(f, f)
    _save(interp, n, fname, 150, fmt)
    return None


@builtin("print")
def _print(interp, args, nargout):
    st = _state(interp)
    fmt = None
    fname = None
    dpi = 150.0
    n = st.current
    for a in args:
        if isinstance(a, (GraphicsHandle, GraphicsArray)):
            n = _fig_num_arg(interp, a)
        elif is_text(a):
            s = text(a)
            if s.startswith("-d"):
                f = s[2:].lower()
                if f == "fig":
                    raise MatlabError(_FIG_ERROR)
                fmt = _FORMATS.get(f, f) or "png"
            elif s.startswith("-r"):
                try:
                    r = float(s[2:]) if s[2:] else 150.0
                except ValueError:
                    raise MatlabError(f"Invalid resolution option '{s}'.") from None
                dpi = r if r > 0 else 150.0
            elif s.startswith("-f") and s[2:].isdigit():
                n = int(s[2:])
            elif s.startswith("-"):
                continue  # renderer and other options (-painters, -opengl, -bestfit, ...)
            else:
                fname = s
        elif isinstance(a, np.ndarray) and a.size == 1:
            n = to_int(a, "figure")
    if fname is None:
        raise MatlabError("print: specify a file name, e.g. print('-dpng', 'figure.png').")
    _save(interp, n, fname, dpi, fmt)
    return None


@builtin("exportgraphics")
def _exportgraphics(interp, args, nargout):
    args, opts = _split_name_values(args, {"resolution", "contenttype", "backgroundcolor", "append",
                                           "colorspace"})
    st = _state(interp)
    if len(args) >= 2 and not is_text(args[0]):
        n = _fig_num_arg(interp, args[0])
        fname = text(args[1])
    else:
        n = st.current
        fname = text(args[-1])
    dpi = _scalar(opts["resolution"]) if "resolution" in opts else 150.0
    _save(interp, n, fname, dpi)
    return None


def show_figures(interp, block: bool = True) -> None:
    """Show open figures in a window (used by the command-line runner)."""
    st = interp.state.get("plot")
    if not st or not st.figures:
        return
    plt = _plt()
    import matplotlib
    if matplotlib.get_backend().lower() == "agg":
        return
    plt.show(block=block)
