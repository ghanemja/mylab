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

import io
import math
import os
import re
import sys
import time
import zlib
from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, to_int, to_str
from ._registry import builtin
from ._util import fnum, is_text, num, text, unknown_option_error

_MPL = None

# A new figure is 560 x 420 pixels (MATLAB's default) at 100 pixels per inch, whatever matplotlib's own
# defaults are.
FIG_DPI = 100.0
FIG_SIZE_INCHES = (5.6, 4.2)


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
        plt.rcParams["figure.max_open_warning"] = 0  # a script may keep any number of figures open
        # MATLAB's colour order for new axes (and for matplotlib's 'C0', 'C1', ... in mylab's own code).
        from cycler import cycler
        plt.rcParams["axes.prop_cycle"] = cycler(color=list(MATLAB_COLOR_ORDER))
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
        # Size and resolution are fixed here: a user or system matplotlibrc (figure.dpi, figure.figsize)
        # must not change how many pixels a figure, a captured frame or a video has.
        fig = plt.figure(n if fresh else None, figsize=FIG_SIZE_INCHES, dpi=FIG_DPI)
        if not fresh and fig.canvas.manager is not None:
            try:
                fig.canvas.manager.set_window_title(f"Figure {n}")
            except (AttributeError, NotImplementedError):
                pass
        st.figures[n] = fig
    st.current = n
    plt.figure(st.figures[n].number)
    return st.figures[n]


def subplot_grid(fig, m: int, n: int):
    """An m-by-n grid of axes positions with MATLAB's margins and gaps (one grid per size and figure)."""
    grids = fig.__dict__.setdefault("_mylab_grids", {})
    gs = grids.get((m, n))
    if gs is None:
        top = 0.87 if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text() else 0.925
        gs = grids[(m, n)] = fig.add_gridspec(m, n, left=0.13, right=0.905, bottom=0.11, top=top,
                                              wspace=0.3155, hspace=0.389)
    return gs


def _make_room_for_suptitle(fig) -> None:
    """Move subplot grids down a little so that a figure title does not sit on the top row's titles."""
    if getattr(fig, "get_layout_engine", lambda: None)() is not None:
        return
    for gs in fig.__dict__.get("_mylab_grids", {}).values():
        gs.update(top=0.87)


def _fig_dpi(fig) -> float:
    """Pixels per inch of a figure, without the scaling a high-resolution screen adds to it."""
    return float(getattr(fig, "_original_dpi", None) or fig.dpi)


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


def _replace_axes(interp, ax, projection: str, carry: bool = False):
    """Replace ``ax`` by a new axes with another projection in the same place.

    ``carry``: the objects drawn in a 2-D axes move into the new 3-D one (and back to 2-D, for an axes
    that view(3) turned into 3-D with nothing 3-D added since), keeping their handles.
    """
    from . import _axes3d_convert as conv
    st = _state(interp)
    info = None
    kind = getattr(ax, "name", "")
    if carry and kind == "rectilinear" and projection == "3d" and not _is_colorbar_axes(ax):
        info = conv.detach_2d(ax)
    elif carry and kind == "3d" and projection == "rectilinear" and conv.can_return_to_2d(ax):
        info = conv.detach_3d(ax)
    fig = ax.figure
    hold = st.hold.get(id(ax), False)
    spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
    pos = ax.get_position()
    layout = st.layouts.get(_fig_number(interp, fig) or -1)
    keep = {}  # limits set by hand survive the change between 2-D and 3-D, as in MATLAB
    if getattr(ax, "name", "") in ("rectilinear", "3d") and projection in ("rectilinear", "3d"):
        for axis_name in ("x", "y", "z"):
            if axis_name == "z" and not hasattr(ax, "get_zlim"):
                if getattr(ax, "_mylab_zlim", None) is not None:
                    keep["z"] = ax._mylab_zlim
            elif not getattr(ax, f"get_autoscale{axis_name}_on")():
                keep[axis_name] = getattr(ax, f"get_{axis_name}lim")()
    ax.remove()
    kw = {} if projection == "rectilinear" else {"projection": projection}
    new = fig.add_subplot(spec, **kw) if spec is not None else fig.add_axes(pos, **kw)
    if info is not None:
        (conv.attach_3d if projection == "3d" else conv.attach_2d)(interp, ax, new, info)
    for axis_name, lims in keep.items():
        if axis_name == "z" and not hasattr(new, "set_zlim"):
            new._mylab_zlim = tuple(lims)
        else:
            getattr(new, f"set_{axis_name}lim")(*lims)
    if layout is not None:
        layout.replace(ax, new)
    known = st.handles.pop(id(ax), None)
    if known is not None:  # a handle to the axes stays valid when the axes changes between 2-D, 3-D and polar
        known.obj = new
        known.kind = _axes_kind(new)
        st.handles[id(new)] = known
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
                ax = fig.add_subplot(subplot_grid(fig, 1, 1)[0, 0], **kw)
            st.axes[st.current] = ax
    if projection is not None and getattr(ax, "name", "rectilinear") != projection:
        # with hold on (or no clearing), what the axes holds carries over between 2-D and 3-D
        ax = _replace_axes(interp, ax, projection, carry=not fresh or st.hold.get(id(ax), False))
        _axes_created(interp, ax)
        _tick_aligned(ax)
        return ax
    if fresh and not st.hold.get(id(ax), False) and getattr(ax, "_mylab_nextplot", "") == "replacechildren":
        _clear_children(interp, ax)
    elif fresh and not st.hold.get(id(ax), False):
        _forget_children(interp, ax)
        cb = getattr(ax, "_mylab_colorbar", None)
        if cb is not None:
            try:
                cb.remove()
            except (ValueError, AttributeError, KeyError):
                pass
            ax._mylab_colorbar = None
        ax.cla()
        _clear_rulers(ax)
        ax.__dict__.pop("_mylab_padded", None)
        known = st.handles.get(id(ax))
        if known is not None and known.kind == "geoaxes":
            known.kind = "axes"
        _yy_restyle(ax)
        _axes_created(interp, ax)
        if getattr(ax, "_mylab_linked", None):
            _relink(ax)
    _tick_aligned(ax)
    return ax


def _axes_created(interp, ax) -> None:
    """Hook run for new or cleared axes: applies the default axes properties set on the root."""
    hook = interp.state.get("axes_defaults_hook")
    if hook is not None:
        hook(interp, ax)


_ALIGNED: dict = {}


def aligned_limits(dmin: float, dmax: float, intervals: int = 10) -> tuple[float, float]:
    """Limits around [dmin, dmax] on multiples of a 1-2-5 step, with at most ``intervals`` steps."""
    span = dmax - dmin
    if span <= 0 or not math.isfinite(span):
        return dmin, dmax
    mag = 10.0 ** math.floor(math.log10(span / intervals))
    lo, hi = dmin, dmax
    for m in (1, 2, 5, 10, 20, 50):
        step = m * mag
        lo = math.floor(dmin / step + 1e-9) * step
        hi = math.ceil(dmax / step - 1e-9) * step
        if round((hi - lo) / step) <= intervals:
            break
    digits = max(0, 12 - int(math.floor(math.log10(max(abs(lo), abs(hi), 1e-300)))))
    return round(lo, digits), round(hi, digits)


def _aligned_locators() -> dict:
    """Tick locators whose automatic axis limits end on tick marks (built on first use)."""
    if _ALIGNED:
        return _ALIGNED
    from matplotlib.ticker import LogLocator, MaxNLocator

    class AlignedLocator(MaxNLocator):
        """Linear axis: limits are the tick marks around the data; up to ten intervals."""

        def __init__(self):
            super().__init__(nbins=10, steps=[1, 2, 5, 10])

        def tick_values(self, vmin, vmax):
            space = self.axis.get_tick_space() if self.axis is not None else 10
            self.set_params(nbins=int(min(max(space, 2), 10)))
            return super().tick_values(vmin, vmax)

        def nonsingular(self, v0, v1):
            if not (math.isfinite(v0) and math.isfinite(v1)):
                return 0.0, 1.0  # nothing plotted yet: the default limits
            if v0 == v1:
                return v0 - 1.0, v1 + 1.0  # a single value sits in the middle of an interval of 2
            return super().nonsingular(v0, v1)

        def view_limits(self, dmin, dmax):
            dmin, dmax = (float(dmin), float(dmax)) if dmin <= dmax else (float(dmax), float(dmin))
            if not (math.isfinite(dmin) and math.isfinite(dmax)):
                return 0.0, 1.0
            if dmin == dmax:
                return dmin - 1.0, dmax + 1.0
            return aligned_limits(dmin, dmax)

    class AlignedLogLocator(LogLocator):
        """Logarithmic axis: limits are the powers of the base around the data."""

        def view_limits(self, vmin, vmax):
            vmin, vmax = self.nonsingular(vmin, vmax)
            b = float(self._base)
            if vmin > 0 and vmax > 0:
                lo = b ** math.floor(math.log(vmin, b) + 1e-12)
                hi = b ** math.ceil(math.log(vmax, b) - 1e-12)
                if lo == hi:
                    lo, hi = lo / b, hi * b
                return lo, hi
            return vmin, vmax

    _ALIGNED.update(lin=AlignedLocator, log=AlignedLogLocator)
    return _ALIGNED


def _tick_aligned(ax) -> None:
    """Give a 2-D axes automatic limits without padding that end on tick marks, as MATLAB draws them."""
    if getattr(ax, "name", "") == "3d" and getattr(ax, "_mylab_view", None) is None:
        ax.view_init(elev=30, azim=-127.5)  # MATLAB's default 3-D view: azimuth -37.5, elevation 30
        ax._mylab_view = (-37.5, 30.0)
    if getattr(ax, "name", "") != "rectilinear":
        return
    changed = False
    for which in "xy":
        axis = getattr(ax, f"{which}axis")
        name = type(axis.get_major_locator()).__name__
        if name == "AutoLocator":
            axis.set_major_locator(_aligned_locators()["lin"]())
            changed = True
        elif name == "LogLocator":
            axis.set_major_locator(_aligned_locators()["log"](base=axis.get_major_locator()._base))
            changed = True
    if changed and not getattr(ax, "_mylab_padded", False):
        if ax._xmargin != 0:
            ax.set_xmargin(0)
        if ax._ymargin != 0:
            ax.set_ymargin(0)


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


_AXES_KINDS = ("axes", "polaraxes", "geoaxes")
_GROUP_KINDS = ("hggroup", "hgtransform")


def _with_axes_arg(fn: Callable) -> Callable:
    """Let a builtin take an axes handle as its first argument, e.g. plot(ax, x, y), or as 'Parent'.

    The parent may also be a group (hggroup, hgtransform): the new objects are drawn in the group's axes,
    without clearing it, and become children of the group.
    """
    def wrapper(interp, args, nargout):
        st = _state(interp)
        target = None
        if args and isinstance(args[0], GraphicsHandle) and args[0].kind in _AXES_KINDS + _GROUP_KINDS:
            target, args = args[0], list(args[1:])
        else:
            for k in range(len(args) - 1):
                if isinstance(args[k + 1], GraphicsHandle) and is_text(args[k]) \
                        and text(args[k]).lower() == "parent":
                    if args[k + 1].kind not in _AXES_KINDS + _GROUP_KINDS:
                        raise MatlabError("The parent of a plot object must be an axes or a group "
                                          "(hggroup, hgtransform).")
                    target, args = args[k + 1], list(args[:k]) + list(args[k + 2:])
                    break
        if target is None:
            return fn(interp, args, nargout)
        if not _alive(target):
            raise MatlabError("Invalid or deleted object.")
        group = target if target.kind in _GROUP_KINDS else None
        ax = target.obj.axes if group is not None else target.obj
        prev = st.override_ax
        st.override_ax = ax
        held = st.hold.get(id(ax), False)
        if group is not None:
            st.hold[id(ax)] = True
        try:
            res = fn(interp, args, max(nargout, 1) if group is not None else nargout)
        finally:
            st.override_ax = prev
            if group is not None:
                st.hold[id(ax)] = held
        if group is not None:
            first = res[0] if isinstance(res, tuple) else res
            for h in _handles_of(first):
                group.obj.adopt(h)
            return res if nargout > 0 else None
        return res
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# -- argument helpers ------------------------------------------------------------------------

_PROPS_KEY = "\0props"  # options that are plain properties of the new object, applied after it exists


def _split_name_values(args, known: set[str], kind: str | None = None):
    """Separate Name,Value pairs (e.g. 'LineWidth', 2) whose names are in ``known``.

    With ``kind`` (a handle kind such as 'line'), every other settable property of that kind of object
    is accepted too; those pairs are collected under ``_PROPS_KEY`` for :func:`_apply_extra`.
    """
    if kind is None:
        if _LINE_OPTS <= known:
            kind = "line"
        elif _TEXT_OPTS <= known:
            kind = "text"
    opts: dict = {}
    rest = []
    for a in args:  # a structure stands for its fields as Name,Value pairs (p.Unmatched, set(h, s))
        if isinstance(a, Struct) and a.data.size == 1:
            for f in a.fields:
                rest += [char(f), a.get_field(0, f)]
        else:
            rest.append(a)
    k = 0
    out = []
    while k < len(rest):
        a = rest[k]
        if is_text(a) and k + 1 < len(rest):
            low = text(a).lower()
            if low in known:
                opts[low] = rest[k + 1]
                k += 2
                continue
            if kind is not None and len(low) > 2 and _is_settable(kind, low):
                opts.setdefault(_PROPS_KEY, []).append((text(a), rest[k + 1]))
                k += 2
                continue
        out.append(a)
        k += 1
    return out, opts


def _reject_text(args, kind: str, flags=()) -> None:
    """Raise MATLAB's error for a text argument a plotting call did not use (an unknown property).

    Line specifications such as 'r--' and the text ``flags`` the call accepts are allowed.
    """
    allowed = {f.lower() for f in flags}
    for a in args:
        if is_text(a):
            s = text(a)
            if s.lower() in allowed or _is_linespec(s):
                continue
            raise MatlabError(f"Unrecognized property '{s}' for class '{_short(kind)}'.",
                              "MATLAB:class:InvalidProperty")


def _is_settable(kind: str, low: str) -> bool:
    for tbl in (_PROPS.get(kind, {}), _COMMON_PROPS):
        for name, entry in tbl.items():
            if name.lower() == low:
                return entry is None or entry[1] is not None
    return False


def _apply_extra(interp, objs: list, opts: dict, kind: str) -> None:
    """Set the plain properties given as options of a plotting call on the objects it created."""
    pairs = opts.get(_PROPS_KEY)
    if not pairs:
        return
    for o in objs:
        h = o if isinstance(o, GraphicsHandle) else _wrap(interp, kind, o)
        for name, value in pairs:
            _set_prop(h, name, value)


_LINE_OPTS = {"linewidth", "color", "markersize", "displayname", "marker", "linestyle",
              "markerfacecolor", "markeredgecolor", "visible", "tag", "userdata"}

_MARKER_NAMES = {"circle": "o", "square": "s", "diamond": "d", "pentagram": "p", "hexagram": "h",
                 "none": "None", "plus": "+", "asterisk": "*", "point": ".", "cross": "x",
                 "x": "x", "^": "^", "v": "v", "<": "<", ">": ">"}


def _mpl_marker(s: str) -> str:
    return _MARKER_NAMES.get(s.lower(), s)


# A marker as MATLAB reports it: the letters s, d, p and h read back as their names, as in MATLAB.
_MARKER_OUT = {"None": "none", "": "none", " ": "none", "none": "none", "s": "square", "d": "diamond",
               "D": "diamond", "p": "pentagram", "h": "hexagram", "H": "hexagram"}


def marker_name(m) -> np.ndarray:
    return char(_MARKER_OUT.get(str(m), str(m)))


def _is_marker_text(a) -> bool:
    s = text(a)
    return (len(s) == 1 and s in "o+*.x_|sd^v<>ph") or s.lower() in _MARKER_NAMES


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

# MATLAB's colour letters. matplotlib's own g, c, m and y are other colours ([0 0.5 0], [0 0.75 0.75], ...) and
# matplotlib looks a letter up only when it draws, so a letter is turned into its triplet before it gets there.
LETTER_RGB = {"r": (1.0, 0.0, 0.0), "g": (0.0, 1.0, 0.0), "b": (0.0, 0.0, 1.0), "c": (0.0, 1.0, 1.0),
              "m": (1.0, 0.0, 1.0), "y": (1.0, 1.0, 0.0), "k": (0.0, 0.0, 0.0), "w": (1.0, 1.0, 1.0)}

# MATLAB's default colour order (the 'gem' palette).
MATLAB_COLOR_ORDER = ((0.0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250),
                      (0.4940, 0.1840, 0.5560), (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330),
                      (0.6350, 0.0780, 0.1840))


def spec_color(spec: str | None):
    """The colour of a line specification ('r--o') as an RGB triplet, or None."""
    m = re.search(r"[rgbcmykw]", spec or "")
    return LETTER_RGB[m.group(0)] if m else None


def split_spec(spec: str, kw: dict) -> tuple[str, dict]:
    """A line specification without its colour letter, and the plot options with that colour added (a 'Color'
    option given as well wins, as in MATLAB)."""
    m = re.search(r"[rgbcmykw]", spec or "")
    if not m:
        return spec, kw
    kw = dict(kw)
    kw.setdefault("color", LETTER_RGB[m.group(0)])
    return spec[:m.start()] + spec[m.end():], kw


def _color(v):
    if is_text(v):
        s = text(v)
        letter = _COLOR_NAMES.get(s.lower(), s)
        return LETTER_RGB.get(letter, letter)
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


_SPEC_COLOR_NAMES = {"black": "k", "white": "w", "red": "r", "green": "g", "blue": "b", "cyan": "c",
                     "magenta": "m", "yellow": "y"}


def _norm_spec(s: str) -> str:
    """A line specification with a colour name ('black', '--red') written as its letter ('k', '--r')."""
    low = s.lower()
    for name, letter in _SPEC_COLOR_NAMES.items():
        i = low.find(name)
        if i >= 0:
            return s[:i] + letter + s[i + len(name):]
    return s


def _is_linespec(s: str) -> bool:
    """True for a line specification such as 'r--o', 'o-', ':', 'k*' or 'black'."""
    if not s:
        return False
    s = _norm_spec(s)
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
            spec = _norm_spec(text(args[k]))
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


def _tag_lines(interp, lines: list, opts: dict, kind: str = "line") -> None:
    """Apply the 'Tag' and 'UserData' options of a plotting call, and any other property given."""
    for key, name in (("tag", "Tag"), ("userdata", "UserData")):
        if key in opts:
            for ln in lines:
                _wrap(interp, kind, ln).extra[name] = text(opts[key]) if key == "tag" else opts[key]
    _apply_extra(interp, lines, opts, kind)


def _redraw(interp):
    plt = _plt()
    if plt.isinteractive():
        plt.pause(0.001)


def _ret(interp, kind: str, objs: list, nargout: int):
    """Register handle(s) for ``objs``; they are returned when an output was requested."""
    hs = [_wrap(interp, kind, o) for o in objs]
    if nargout <= 0:
        return None
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
    "animatedline": "matlab.graphics.animation.AnimatedLine",
    "rectangle": "matlab.graphics.primitive.Rectangle",
}

_TYPE = {"root": "root", "figure": "figure", "axes": "axes", "polaraxes": "polaraxes", "line": "line",
         "stair": "stair", "scatter": "scatter", "bar": "bar", "histogram": "histogram",
         "surface": "surface", "contour": "contour", "quiver": "quiver", "stem": "stem",
         "errorbar": "errorbar", "area": "area", "image": "image", "patch": "patch", "text": "text",
         "legend": "legend", "colorbar": "colorbar", "pie": "patch", "tiledlayout": "tiledlayout",
         "animatedline": "animatedline", "rectangle": "rectangle"}


# Hooks filled in by mylab.builtins.plot_objects (which imports this module).
_ROOT_HOOKS = None    # (get, set) for default properties on the root: set(groot, 'DefaultLineLineWidth', 2)
_LINK_HOOK = None     # called after a property is set, to keep objects joined by linkprop in step
_TEXT_RENDER = None   # (text, interpreter) -> the string matplotlib draws (TeX markup translated)
_DELETE_HOOK = None   # called before an object is deleted (runs its DeleteFcn)


def _tex(interp, s: str, interpreter: str | None = None, kind: str = "text") -> str:
    """Label text as matplotlib should draw it under the given text interpreter."""
    if _TEXT_RENDER is None:
        return s
    return _TEXT_RENDER(interp, s, interpreter, kind)


def set_text(interp, artist, raw: str, interpreter: str | None = None, kind: str = "text") -> None:
    """Give a text artist its string, remembering what the user typed."""
    if interpreter is None:
        interpreter = getattr(artist, "_mylab_interp", None)
    artist._mylab_raw = raw
    if interpreter is not None:
        artist._mylab_interp = interpreter
    artist.set_text(_tex(interp, raw, interpreter, kind))


class _Root:
    """Stand-in object for the graphics root (groot, handle 0)."""


_ROOT = _Root()


def _wrap(interp, kind: str, obj) -> "GraphicsHandle":
    st = _state(interp)
    h = st.handles.get(id(obj))
    if h is not None and h.obj is obj:
        return h
    if kind == "text" and getattr(obj, "axes", 1) is None:  # a title or axis label: link it to its axes
        from .plot_objects import _owning_axes
        ax = _owning_axes(obj)
        if ax is not None:
            obj.axes = ax
    h = GraphicsHandle(interp, kind, obj)
    st.handles[id(obj)] = h
    return h


# Numeric handles (double(h), or a handle stored into a double array as in h = zeros(1, 3);
# h(1) = plot(...)): figures are their numbers and the root is 0; every other object gets a number
# that is never an integer, so it can never be taken for a figure number.
import weakref  # noqa: E402

_NUMBERED: "weakref.WeakValueDictionary[float, GraphicsHandle]" = weakref.WeakValueDictionary()


def handle_number(interp, h: "GraphicsHandle") -> float:
    if h.kind == "figure":
        return float(_fig_number(interp, h.obj) or 0)
    if h.kind == "root":
        return 0.0
    n = h.extra.get("_number")
    if n is None:
        n = h.extra["_number"] = h.serial + 0.0001 * (1 + h.serial % 9999)
        _NUMBERED[n] = h
    return n


def numbered_handle(x: float) -> "GraphicsHandle | None":
    """The graphics object a non-integer numeric handle stands for (None if there is none)."""
    return _NUMBERED.get(float(x))


def _short(kind: str) -> str:
    return _CLASS.get(kind, kind).rsplit(".", 1)[-1]


class _GraphicsBase(MObject):
    is_handle = True

    @staticmethod
    def concat(items: list, axis: int):
        grids = []
        for it in items:
            if isinstance(it, (GraphicsHandle, GraphicsArray)):
                grids.append(_handle_grid(it))
            elif isinstance(it, np.ndarray) and it.size == 0:
                continue
            else:
                raise MatlabError("Graphics handles can only be concatenated with other graphics handles.")
        if not grids:
            return np.zeros((0, 0))
        try:
            joined = np.concatenate(grids, axis=min(axis, 1))
        except ValueError:
            if not all(1 in g.shape for g in grids):
                raise MatlabError("Dimensions of arrays being concatenated are not consistent.") from None
            # vectors of different orientations: joined end to end (MATLAB asks for matching shapes)
            flat = [x for g in grids for x in g.flatten(order="F")]
            return GraphicsArray(flat, row=(axis == 1))
        return _from_handle_grid(joined)


def _handle_grid(v) -> np.ndarray:
    """Graphics handles as a 2-D numpy object array in MATLAB's shape."""
    if isinstance(v, GraphicsHandle):
        g = np.empty((1, 1), dtype=object)
        g[0, 0] = v
        return g
    g = np.empty(len(v.items), dtype=object)
    for k, h in enumerate(v.items):
        g[k] = h
    return g.reshape(v.matlab_shape(), order="F")


def _from_handle_grid(g: np.ndarray):
    """The value for a numpy object array of handles: one handle, or an array of them."""
    if g.ndim > 2:
        if any(d != 1 for d in g.shape[2:]):
            raise MatlabError("Arrays of graphics objects have two dimensions here; use reshape to keep them 2-D.")
        g = g.reshape(g.shape[:2], order="F")
    if g.size == 1:
        return g.flat[0]
    items = list(g.flatten(order="F"))
    return GraphicsArray(items, row=g.shape[0] == 1 and g.shape[1] != 1, shape=g.shape)


class GraphicsHandle(_GraphicsBase):
    """A reference to one graphics object (figure, axes, line, ...)."""

    _counter = 0

    def __init__(self, interp, kind: str, obj):
        self.interp = interp
        self.kind = kind
        self.obj = obj
        self.extra: dict[str, Any] = {}
        GraphicsHandle._counter += 1
        self.serial = GraphicsHandle._counter  # creation order (children are listed newest first)

    def class_name(self) -> str:
        return _CLASS.get(self.kind, "matlab.graphics.Graphics")

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), _short(self.kind), "handle", "matlab.graphics.Graphics")

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid", "isprop", "properties", "double", "eq", "ne", "addprop")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete":
            for a in args:
                for h in _handles_of(a):
                    _delete_handle(interp, h)
            return []
        if name == "isvalid":
            return [np.array([[_alive(self)]])]
        if name == "isprop":
            ok = len(args) > 1 and is_text(args[1]) and (_lookup(self, text(args[1]))[0] is not None
                                                         or text(args[1]) in self.extra.get("_dynamic", ()))
            return [np.array([[bool(ok)]])]
        if name == "properties":
            names = property_names(self)
            if nargout:
                return [_cell_column([char(n) for n in names])]
            interp.out(f"\nProperties for class {self.class_name()}:\n\n" + "".join(f"    {n}\n" for n in names)
                       + "\n")
            return []
        if name == "double":
            return [_num_val(handle_number(interp, self))]
        if name in ("eq", "ne"):
            a, b = (args + [None, None])[:2]
            if isinstance(a, np.ndarray) and a.size == 1 and isinstance(b, GraphicsHandle):
                a, b = b, a
            if isinstance(b, GraphicsArray):
                flags = np.array([x is a or (x.obj is a.obj and x.kind == a.kind) for x in b.items], dtype=bool)
                flags = flags.reshape(b.matlab_shape())
                return [flags if name == "eq" else ~flags]
            if isinstance(b, GraphicsHandle):
                same = a is b or (a.obj is b.obj and a.kind == b.kind)
            elif isinstance(b, np.ndarray) and b.size == 1 and not is_char(b) and a.kind in ("figure", "root"):
                n = 0 if a.kind == "root" else (_fig_number(interp, a.obj) or -1)
                same = float(np.real(b.flat[0])) == n
            else:
                same = False
            return [np.array([[same if name == "eq" else not same]])]
        if name == "addprop":
            if len(args) < 2 or not is_text(args[1]):
                raise MatlabError("addprop needs a property name.")
            self.extra.setdefault("_dynamic", {}).setdefault(text(args[1]), np.zeros((0, 0)))
            return []
        return super().call_method(interp, name, args, nargout)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("==", "~=") and isinstance(left, (GraphicsHandle, GraphicsArray, np.ndarray)) \
                and isinstance(right, (GraphicsHandle, GraphicsArray, np.ndarray)):
            a, b = (left, right) if isinstance(left, GraphicsHandle) else (right, left)
            if isinstance(a, GraphicsHandle):
                return a.call_method(interp, "eq" if op == "==" else "ne", [a, b], 1)[0]
        return NotImplemented

    def get_field(self, interp, name: str) -> Any:
        dyn = self.extra.get("_dynamic")
        if dyn and name in dyn:
            return dyn[name]
        return _get_prop(self, name)

    def set_field(self, interp, name: str, value: Any) -> "GraphicsHandle":
        dyn = self.extra.get("_dynamic")
        if dyn and name in dyn:
            dyn[name] = value
            return self
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

    def __init__(self, items: list[GraphicsHandle], row: bool = False, shape: tuple | None = None):
        self.items = list(items)   # column-major order
        self.row = row
        self.shape2 = tuple(int(d) for d in shape) if shape is not None and len(shape) == 2 \
            and int(np.prod(shape)) == len(self.items) and 1 not in shape else None

    def class_name(self) -> str:
        kinds = {h.kind for h in self.items}
        return self.items[0].class_name() if len(kinds) == 1 else "matlab.graphics.Graphics"

    def isa_class(self, name: str) -> bool:
        return name in ("handle", "matlab.graphics.Graphics", self.class_name()) or (
            bool(self.items) and all(h.isa_class(name) for h in self.items))

    def matlab_shape(self) -> tuple:
        if self.shape2 is not None:
            return self.shape2
        return (1, len(self.items)) if self.row else (len(self.items), 1)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("==", "~="):
            other = right if left is self else left
            if isinstance(other, GraphicsHandle):
                return other.call_method(interp, "eq" if op == "==" else "ne", [other, self], 1)[0]
            if isinstance(other, GraphicsArray) and len(other.items) == len(self.items):
                flags = np.array([a is b or (a.obj is b.obj and a.kind == b.kind)
                                  for a, b in zip(self.items, other.items)], dtype=bool)
                flags = flags.reshape(self.matlab_shape(), order="F")
                return flags if op == "==" else ~flags
        return NotImplemented

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid", "double") or name in _GRID_FUNCS

    def unary_op(self, interp, op: str) -> Any:
        if op in ("'", ".'"):
            return _from_handle_grid(_handle_grid(self).T)
        return NotImplemented

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name in _GRID_FUNCS:
            return [_grid_call(interp, name, args, nargout)]
        if name == "delete":
            for h in self.items:
                _delete_handle(interp, h)
            return []
        if name == "double":  # the numeric handles, as stored by h(k) = plot(...) into a double array
            nums = np.array([handle_number(interp, h) for h in self.items], dtype=float)
            return [nums.reshape(self.matlab_shape(), order="F") if nums.size else np.zeros(self.matlab_shape())]
        if name == "isvalid":
            flags = np.array([_alive(h) for h in self.items], dtype=bool)
            return [flags.reshape(self.matlab_shape())]
        return super().call_method(interp, name, args, nargout)

    def get_field(self, interp, name: str) -> Any:
        return _cell_column([_get_prop(h, name) for h in self.items])

    def field_list(self, interp, name: str) -> list:
        """h.Name on several objects: one value per object, a comma-separated list as in MATLAB
        ({h.Marker} is a cell of the markers, [h.Color] their colours side by side)."""
        return [_get_prop(h, name) for h in self.items]

    def set_field(self, interp, name: str, value: Any) -> "GraphicsArray":
        for h in self.items:
            _set_prop(h, name, value)
        _redraw(interp)
        return self

    def index_get(self, interp, args: list) -> Any:
        from .. import indexing
        args = [indexing.COLON if (isinstance(a, np.ndarray) and is_char(a) and text(a) == ":") else a
                for a in args]
        try:
            picked = indexing.get(_handle_grid(self), args)
        except MatlabError as e:
            if "exceed" in str(e):
                raise MatlabError(f"Index exceeds the number of array elements. Index must not exceed "
                                  f"{len(self.items)}.") from None
            raise
        return _from_handle_grid(picked)

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


_GRID_FUNCS = frozenset({"transpose", "ctranspose", "reshape", "flip", "fliplr", "flipud", "permute",
                         "rot90", "circshift", "squeeze", "repmat", "horzcat", "vertcat", "cat"})


def _grid_call(interp, name: str, args: list, nargout: int) -> Any:
    """Shape functions (reshape, flip, permute, ...) on arrays of graphics handles: the built-in
    function rearranges the positions 1..n, which then pick the handles."""
    from ._registry import BUILTINS
    if name in ("horzcat", "vertcat", "cat"):
        if name == "cat":
            dim = to_int(args[0], "dim")
            return _GraphicsBase.concat(list(args[1:]), 0 if dim == 1 else dim - 1)
        return _GraphicsBase.concat(list(args), 1 if name == "horzcat" else 0)
    if name == "repmat":
        reps = [to_int(x, "count") for a in args[1:] for x in np.asarray(fnum(a)).flatten()]
        if len(reps) == 1:
            reps *= 2
        return _from_handle_grid(np.tile(_handle_grid(args[0]), reps))
    src = next(a for a in args if isinstance(a, (GraphicsHandle, GraphicsArray)))
    grid = _handle_grid(src)
    positions = np.arange(1.0, grid.size + 1).reshape(grid.shape, order="F")
    rest = [positions if a is src else a for a in args]
    moved = np.asarray(BUILTINS[name](interp, rest, 1))
    picked = grid.flatten(order="F")[moved.astype(np.int64) - 1] if moved.size else np.empty(moved.shape, object)
    return _from_handle_grid(picked.reshape(moved.shape))


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
    if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.size and not is_char(v):
        found = [numbered_handle(x) for x in np.real(v).flatten(order="F")]
        if all(h is not None for h in found):
            return found
    return []


def _alive(h: GraphicsHandle) -> bool:
    st = _state(h.interp)
    if h.kind == "root":
        return True
    ui_alive = getattr(h, "ui_alive", None)
    if ui_alive is not None:  # an axes inside an app window (mylab.ui)
        return ui_alive()
    if h.kind == "figure":
        return any(f is h.obj for f in st.figures.values())
    fig = _obj_figure(h)
    if fig is not None and getattr(fig, "_mylab_ui", False):  # drawn in the axes of an app window
        return not h.extra.get("deleted")
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
    if h.kind == "placeholder" or not _alive(h) or h.extra.get("_deleting"):
        return
    if _DELETE_HOOK is not None:
        h.extra["_deleting"] = True
        try:
            _DELETE_HOOK(interp, h)
        finally:
            h.extra.pop("_deleting", None)
    custom = h.extra.get("_delete")
    if custom is not None:  # chart objects made of several artists remove themselves
        custom(h)
        h.extra["deleted"] = True
        return
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
        ax = getattr(h.obj, "_mylab_parent_ax", None)
        if ax is not None and getattr(ax, "_mylab_colorbar", None) is h.obj:
            ax._mylab_colorbar = None
    elif h.kind == "histogram":
        h.obj.remove()
    elif h.kind == "legend":
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
    if isinstance(v, FunctionHandle):
        return v.text if getattr(v, "text", None) else "@" + str(getattr(v, "name", "function"))
    if isinstance(v, MString):
        return f'"{to_str(v)}"'
    if isinstance(v, MObject):
        shape = "x".join(str(n) for n in v.matlab_shape())
        return f"[{shape} {v.class_name().rsplit('.', 1)[-1]}]"
    if isinstance(v, Struct):
        return "[1x1 struct]"
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
    return marker_name(h.obj.get_marker())


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
    if h.extra.get("_part_of") is not None and _alive(h.extra["_part_of"]):
        return h.extra["_part_of"]
    ax = _obj_axes(h)
    if ax is not None:
        return _wrap(interp, _axes_kind(ax), ax)
    fig = _obj_figure(h)
    return _wrap(interp, "figure", fig) if fig is not None else np.zeros((0, 0))


def _parent_set(h, v):
    hs = _handles_of(v)
    if len(hs) != 1:
        raise MatlabError("Parent must be one graphics object.")
    new = hs[0]
    if h.kind in ("figure", "root", "legend", "colorbar") + _AXES_KINDS:
        raise MatlabError(f"The parent of a {_short(h.kind)} cannot be changed.")
    old = h.extra.get("_part_of")
    if new.kind in _GROUP_KINDS:
        if _obj_axes(h) is not new.obj.axes:
            raise MatlabError("An object can only join a group in the same axes.")
        new.obj.adopt(h)
        return
    if new.kind in _AXES_KINDS:
        if _obj_axes(h) is not new.obj:
            raise MatlabError("Moving an object to another axes is not supported; use copyobj(h, ax) and "
                              "delete the original.")
        if old is not None:
            old.obj.release(h)
        return
    raise MatlabError("The parent of a plot object must be an axes or a group (hggroup, hgtransform).")


def _axes_kind(ax) -> str:
    if getattr(ax, "_mylab_geo", False):
        return "geoaxes"
    return "polaraxes" if getattr(ax, "name", "") == "polar" else "axes"


_NOT_CHILDREN = ("root", "figure", "axes", "polaraxes", "geoaxes", "tiledlayout", "legend", "colorbar", "ruler",
                 "placeholder", "annotationinfo", "legendentry", "layoutoptions", "proplink")


def _guess_kind(artist) -> str:
    name = artist.__class__.__name__
    if name in ("Line2D", "Line3D"):
        return "line"
    if name in ("PathCollection", "Path3DCollection"):
        return "scatter"
    if name in ("AxesImage",):
        return "image"
    if name in ("Text", "Annotation", "Text3D"):
        return "text"
    if getattr(artist, "_mylab_surface", False) or getattr(artist, "_mylab_mesh", False) \
            or name == "Line3DCollection":
        return "surface"
    if name in ("QuadContourSet", "ContourSet"):
        return "contour"
    if name in ("Quiver",):
        return "quiver"
    return "patch"


def _axes_children(interp, ax, hidden: bool = True) -> list:
    """Handles of the objects drawn in an axes, newest first (the order MATLAB lists them in)."""
    st = _state(interp)
    present: dict[int, Any] = {}
    for a in list(ax.lines) + list(ax.collections) + list(ax.images) + list(ax.texts) + list(ax.patches):
        if not getattr(a, "_mylab_internal", False) and getattr(a, "_mylab_is_subtitle_of", None) is None:
            present[id(a)] = a
    for c in ax.containers:
        present[id(c)] = c
    groups = []
    owned: set[int] = set()
    for g in list(st.handles.values()):
        if g.kind in _NOT_CHILDREN or g.extra.get("deleted"):
            continue
        if g.extra.get("_child_of") is ax:
            arts = _artists(g.obj) if not g.extra.get("_artists") else list(g.extra["_artists"])
        else:
            arts = _artists(g.obj)
            if id(g.obj) not in present and not any(id(a) in present for a in arts):
                continue
        owned.update(id(a) for a in arts)
        owned.add(id(g.obj))
        if g.extra.get("_part_of") is not None:  # a child of a group: listed under the group, not here
            continue
        groups.append(g)
        for extra_art in g.extra.get("_artists", ()):
            owned.add(id(extra_art))
    for key, a in present.items():
        if key in owned or a.__class__.__name__.endswith("Container"):
            continue
        groups.append(_wrap(interp, _guess_kind(a), a))
    groups.sort(key=lambda g: g.serial, reverse=True)
    if not hidden:
        groups = [g for g in groups if str(g.extra.get("HandleVisibility", "on")) != "off"]
    return groups


def _children_get(h):
    interp = h.interp
    out: list[GraphicsHandle] = []
    show_hidden = str(_wrap(interp, "root", _ROOT).extra.get("ShowHiddenHandles", "off")) == "on"
    if h.kind == "figure":
        out = [_wrap(interp, _axes_kind(a), a) for a in h.obj.axes if not _is_colorbar_axes(a)
               and not getattr(a, "_mylab_internal", False)]
        out.reverse()
    elif h.kind in ("axes", "polaraxes", "geoaxes"):
        out = _axes_children(interp, h.obj, hidden=show_hidden)
    elif h.kind == "root":
        st = _state(interp)
        out = [_wrap(interp, "figure", st.figures[n]) for n in sorted(st.figures, reverse=True)]
    elif h.extra.get("_children") is not None:
        out = [c for c in h.extra["_children"] if _alive(c)]
    if not show_hidden and h.kind in ("figure", "root"):
        out = [c for c in out if str(c.extra.get("HandleVisibility", "on")) != "off"]
    if not out:
        return np.zeros((0, 1))
    return out[0] if len(out) == 1 else GraphicsArray(out)


def _forget_children(interp, ax) -> None:
    """The objects of an axes are about to be cleared: mark their handles deleted and drop them."""
    st = _state(interp)
    keep = {id(ax.xaxis.label), id(ax.yaxis.label)}
    if hasattr(ax, "zaxis"):
        keep.add(id(ax.zaxis.label))
    for key, g in list(st.handles.items()):
        if g.kind in _NOT_CHILDREN or key in keep:
            continue
        try:
            owner = _obj_axes(g) if g.extra.get("_child_of") is None else g.extra["_child_of"]
        except (AttributeError, TypeError, IndexError):
            continue
        if owner is ax:
            g.extra["deleted"] = True
            del st.handles[key]


def _clear_children(interp, ax) -> None:
    """Remove what is plotted in an axes but keep the axes itself: its labels, limits, scale and grid."""
    _forget_children(interp, ax)
    for a in list(ax.lines) + list(ax.collections) + list(ax.images) + list(ax.patches) + \
            [t for t in ax.texts if getattr(t, "_mylab_is_subtitle_of", None) is None]:
        try:
            a.remove()
        except (ValueError, NotImplementedError, AttributeError):
            pass
    del ax.containers[:]
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    cb = getattr(ax, "_mylab_colorbar", None)
    if cb is not None:
        try:
            cb.remove()
        except (ValueError, AttributeError, KeyError):
            pass
        ax._mylab_colorbar = None
    try:
        ax.set_prop_cycle(None)
        hook = interp.state.get("cycle_hook")
        if hook is not None:
            hook(ax)
    except (ValueError, AttributeError):
        pass
    ax.relim()
    ax.autoscale_view()


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


def _polar_data(which: str, name: str):
    """ThetaData and RData, which a line has only in polar axes (they are its x and y data there)."""
    def check(h):
        ax = getattr(h.obj, "axes", None)
        if ax is None or getattr(ax, "name", "") != "polar":
            raise MatlabError(f"Unrecognized property '{name}' for class 'Line'.", "MATLAB:class:InvalidProperty")

    def get(h):
        check(h)
        return _row(h.obj.get_xdata() if which == "x" else h.obj.get_ydata())

    def set_(h, v):
        check(h)
        _set_line_data(h, which, v)
    return get, set_


_LINE_PROPS["ThetaData"] = _polar_data("x", "ThetaData")
_LINE_PROPS["RData"] = _polar_data("y", "RData")


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
    """State of a histogram plot: data, bin edges and the bars (or stairs) drawing it."""

    def __init__(self, ax, data, edges, normalization="count", orientation="vertical", style=None,
                 label=None, counts=None):
        self.ax = ax
        self.data = data
        self.edges = np.asarray(edges, dtype=float)
        self.normalization = normalization
        self.orientation = orientation
        self.style = dict(style or {})
        self.label = label
        self.container = None
        self.alpha = self.style.pop("alpha", 0.6)
        self.display_style = self.style.pop("display_style", "bar")
        self.bin_method = self.style.pop("bin_method", "auto")
        self.counts = None if counts is None else np.asarray(counts, dtype=float)  # given, not counted

    def bin_counts(self) -> np.ndarray:
        if self.counts is not None and self.counts.size == self.edges.size - 1:
            return self.counts.astype(float)
        counts, _ = np.histogram(self.data, self.edges)
        return counts.astype(float)

    def values(self) -> np.ndarray:
        counts = self.bin_counts()
        n = max(self.data.size if self.counts is None else float(counts.sum()), 1)
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
        ls = self.style.get("linestyle", "-")
        ls = "None" if str(ls).lower() == "none" else ls
        if self.display_style == "stairs":
            color = self.style.get("edgecolor_set") or self.style.get("facecolor") or \
                self.ax._get_lines.get_next_color()
            self.style.setdefault("facecolor", to_rgba(color)[:3])
            sp = self.ax.stairs(vals, self.edges, fill=False, color=color, linestyle=ls,
                                orientation=self.orientation, linewidth=self.style.get("linewidth", 0.5),
                                label=self.label or None)
            self.container = [sp]
            return
        ec = to_rgba(self.style.get("edgecolor", "black"), self.style.get("edgealpha", 1.0))
        kw = {"edgecolor": ec, "linewidth": self.style.get("linewidth", 0.5), "linestyle": ls}
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
                p.set_edgecolor(ec)
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
    if method in ("sturges", "sqrt", "scott", "fd") and width is None and nbins is None and x.size and hi > lo:
        n = x.size
        if method == "sturges":
            raw = (hi - lo) / math.ceil(1 + math.log2(n))
        elif method == "sqrt":
            raw = (hi - lo) / math.ceil(math.sqrt(n))
        elif method == "scott":
            sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
            raw = 3.5 * sd / n ** (1 / 3) if sd > 0 else (hi - lo) / 10
        else:
            q75, q25 = np.percentile(x, [75, 25])
            raw = 2 * float(q75 - q25) / n ** (1 / 3) if q75 > q25 else (hi - lo) / 10
        w = _nice_width(max(raw, (hi - lo) / 2000))
        start = math.floor(lo / w) * w
        stop = math.ceil(hi / w) * w
        if stop <= hi:
            stop += w
        return start + w * np.arange(int(round((stop - start) / w)) + 1)
    if method not in (None, "auto", "integers", "sturges", "sqrt", "scott", "fd"):
        raise MatlabError("BinMethod must be 'auto', 'scott', 'fd', 'integers', 'sturges' or 'sqrt'.")
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
        return np.arange(math.floor(lo) - 0.5, math.ceil(hi) + 1.0, 1.0)
    start = math.floor(lo / w) * w
    stop = math.ceil(hi / w) * w
    if stop <= hi:
        stop += w
    k = int(round((stop - start) / w))
    return start + w * np.arange(k + 1)


# -- rulers: datetime, duration and categorical axes ------------------------------------------------

class Ruler:
    """What one axis measures when it is not plain numbers.

    ``kind`` is 'datetime' (axis numbers are days since 1970-01-01), 'datenum' (days since year 0, used
    by datetick), 'duration' (axis numbers are multiples of ``unit`` seconds) or 'categorical' (axis
    numbers are positions 1..n in ``categories``).
    """

    def __init__(self, kind: str, zone: str = "", fmt: str | None = None, unit: float = 1.0,
                 categories: list | None = None, dfmt: str = "s"):
        self.kind = kind
        self.zone = zone
        self.fmt = fmt              # tick label format chosen by the user (None = automatic)
        self.unit = unit            # seconds per axis unit (duration)
        self.dfmt = dfmt            # display format of a duration ruler
        self.categories = list(categories or [])
        self.step = None            # tick spacing chosen by the locator: ('s' | 'mo' | 'y', size)
        self.secondary = None       # text of the secondary label set by the user (None = automatic)
        self.classic = False        # fmt uses datestr letters (mm = month, MM = minute), as datetick does

    def seconds(self, nums) -> np.ndarray:
        nums = np.asarray(nums, dtype=float)
        if self.kind == "datetime":
            return nums * 86400.0
        if self.kind == "datenum":
            return (nums - 719529.0) * 86400.0
        return nums * self.unit

    def numbers(self, secs) -> np.ndarray:
        secs = np.asarray(secs, dtype=float)
        if self.kind == "datetime":
            return secs / 86400.0
        if self.kind == "datenum":
            return secs / 86400.0 + 719529.0
        return secs / self.unit


def _is_time_like(v) -> bool:
    return isinstance(v, MObject) and getattr(v, "kind", None) in ("datetime", "duration", "categorical")


def _ruler_of(ax, which: str):
    return getattr(ax, "_mylab_rulers", {}).get(which)


def _duration_unit(secs: np.ndarray, fmt: str) -> float:
    from .datetimes import _UNIT_SECONDS
    if fmt in _UNIT_SECONDS:
        return float(_UNIT_SECONDS[fmt])
    finite = np.abs(secs[np.isfinite(secs)]) if secs.size else secs
    top = float(finite.max()) if finite.size else 0.0
    for unit in (86400.0, 3600.0, 60.0):
        if top >= 2 * unit:
            return unit
    return 1.0


def new_ruler_for(v, existing=None):
    """A ruler suited to the datetime, duration or categorical value ``v`` (reusing ``existing``)."""
    kind = v.kind
    if existing is not None and existing.kind == kind:
        if kind == "categorical":
            for c in v.categories:
                if c not in existing.categories:
                    existing.categories.append(c)
        return existing
    if existing is not None and existing.kind == "datenum" and kind == "datetime":
        return existing
    if kind == "datetime":
        return Ruler("datetime", zone=v.zone)
    if kind == "duration":
        return Ruler("duration", unit=_duration_unit(np.asarray(v.data, float), v.format), dfmt=v.format)
    return Ruler("categorical", categories=list(v.categories))


def ruler_numbers(r: Ruler, v) -> np.ndarray:
    """Axis numbers (same shape as ``v``) of a datetime, duration or categorical array on ruler ``r``."""
    if v.kind == "categorical":
        codes = np.asarray(v.data, dtype=float)
        out = np.full(codes.shape, np.nan)
        for idx in np.ndindex(codes.shape):
            c = codes[idx]
            if np.isfinite(c):
                out[idx] = r.categories.index(v.categories[int(c) - 1]) + 1
        return out
    return r.numbers(np.asarray(v.data, dtype=float))


def ruler_values(ax, which: str, nums):
    """Axis numbers as a row of the values the ruler shows (datetime, duration, categorical, numbers)."""
    r = _ruler_of(ax, which)
    nums = np.asarray(nums, dtype=float).reshape(1, -1)
    if r is None or r.kind == "datenum":
        return nums
    if r.kind == "datetime":
        from .datetimes import DatetimeArray
        return DatetimeArray(r.seconds(nums), None, r.zone)
    if r.kind == "duration":
        from .datetimes import DurationArray
        return DurationArray(r.seconds(nums), r.dfmt)
    from .categorical import CategoricalArray
    n = max(len(r.categories), 1)
    if nums.size == 2 and nums.flat[0] != np.floor(nums.flat[0]):  # limits: first and last category inside
        codes = np.array([[max(math.ceil(nums.flat[0]), 1), min(math.floor(nums.flat[1]), n)]], float)
    else:
        codes = np.clip(np.round(nums), 1, n)
    return CategoricalArray(codes, list(r.categories))


def axis_numbers(ax, which: str, v) -> np.ndarray:
    """Numbers for a limit or tick value given to an axis: converts datetime/duration/categories."""
    r = _ruler_of(ax, which)
    if _is_time_like(v):
        if r is None or (r.kind != v.kind and not (r.kind == "datenum" and v.kind == "datetime")):
            r = new_ruler_for(v)
            install_ruler(ax, which, r)
        elif v.kind == "categorical":
            new_ruler_for(v, r)
        return ruler_numbers(r, v)
    if r is not None and r.kind == "categorical" and (isinstance(v, (Cell, StringArray, MString)) or is_char(v)):
        names = _text_list(v)
        missing = [n for n in names if n not in r.categories]
        if missing:
            raise MatlabError(f"'{missing[0]}' is not one of the categories of this axis.")
        return np.array([r.categories.index(n) + 1 for n in names], dtype=float)
    return np.real(fnum(v))


_RULER_CLASSES: dict = {}


def _date_tick_classes():
    """Locator and formatter classes for date axes (built on first use: they subclass matplotlib's)."""
    if _RULER_CLASSES:
        return _RULER_CLASSES["loc"], _RULER_CLASSES["fmt"]
    from matplotlib.ticker import Formatter, Locator

    from .datetimes import add_months, fields_from_seconds, format_classic, seconds_from_fields
    from .datetimes import format_datetime as _format_modern

    def format_datetime(secs, fmt, zone, classic=False):
        return format_classic(secs, fmt) if classic else _format_modern(secs, fmt, zone)

    ladder = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200,
              86400, 2 * 86400, 7 * 86400, 14 * 86400]

    class DateLocator(Locator):
        def __init__(self, ruler, target: int = 7):
            self.ruler = ruler
            self.target = target

        def __call__(self):
            lo, hi = self.axis.get_view_interval()
            return self.tick_values(lo, hi)

        def tick_values(self, vmin, vmax):
            r = self.ruler
            lo, hi = sorted(float(v) for v in r.seconds([vmin, vmax]))
            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                return np.array([vmin])
            n = self.target
            month = 30.4375 * 86400
            if span / (14 * 86400) <= n:
                step = next((s for s in ladder if span / s <= n), ladder[-1])
                r.step = ("s", step)
                first = math.ceil(lo / step - 1e-9) * step
                secs = first + step * np.arange(0, int(math.floor((hi - first) / step + 1e-9)) + 1)
            elif span / (6 * month) <= n:
                k = next(m for m in (1, 2, 3, 4, 6) if span / (m * month) <= n)
                r.step = ("mo", k)
                y, mo = (float(np.asarray(v).flat[0]) for v in fields_from_seconds(lo)[:2])
                start = float(np.asarray(seconds_from_fields(y, 1 + k * math.floor((mo - 1) / k), 1)).flat[0])
                found = []
                for i in range(0, 4 * n + 4):
                    t = float(np.asarray(add_months(np.array([start]), np.array([float(i * k)]))).flat[0])
                    if t > hi + 1e-6:
                        break
                    if t >= lo - 1e-6:
                        found.append(t)
                secs = np.array(found)
            else:
                years = span / (365.2425 * 86400)
                k = 1
                for mag in (1, 10, 100, 1000, 10000):
                    k = next((m * mag for m in (1, 2, 5) if years / (m * mag) <= n), None)
                    if k is not None:
                        break
                k = k or 50000
                r.step = ("y", k)
                y0 = float(np.asarray(fields_from_seconds(lo)[0]).flat[0])
                first = math.ceil(y0 / k) * k
                cand = [float(np.asarray(seconds_from_fields(float(first + (i - 1) * k), 1, 1)).flat[0])
                        for i in range(0, 3 * n + 4)]
                secs = np.array([t for t in cand if lo - 1e-6 <= t <= hi + 1e-6])
            return r.numbers(secs)

    class DateFormatter(Formatter):
        def __init__(self, ruler):
            self.ruler = ruler
            self._offset = ""

        def _format(self):
            r = self.ruler
            if r.fmt:
                return r.fmt, ""
            kind, size = r.step or ("s", 86400)
            if kind == "y":
                return "yyyy", ""
            if kind == "mo":
                return "MMM yyyy", ""
            if size >= 86400:
                return "MMM dd", "yyyy"
            if size >= 60:
                return "HH:mm", "MMM dd, yyyy"
            return "HH:mm:ss", "MMM dd, yyyy"

        def __call__(self, x, pos=None):
            fmt, _ = self._format()
            r = self.ruler
            return format_datetime(float(r.seconds(x)), fmt, r.zone, bool(r.fmt) and r.classic)

        def format_ticks(self, values):
            fmt, second = self._format()
            r = self.ruler
            secs = np.atleast_1d(r.seconds(np.asarray(values, dtype=float)))
            self._offset = ""
            if r.secondary is not None:
                self._offset = r.secondary
            elif second and secs.size:
                a = format_datetime(float(np.min(secs)), second, r.zone)
                b = format_datetime(float(np.max(secs)), second, r.zone)
                self._offset = a if a == b else f"{a} - {b}"
            return [format_datetime(float(t), fmt, r.zone, bool(r.fmt) and r.classic) for t in secs]

        def get_offset(self):
            return self._offset

    _RULER_CLASSES.update(loc=DateLocator, fmt=DateFormatter)
    return DateLocator, DateFormatter


def install_ruler(ax, which: str, r) -> None:
    """Make axis ``which`` of ``ax`` a datetime, duration or categorical ruler (None: plain numbers)."""
    rulers = ax.__dict__.setdefault("_mylab_rulers", {})
    axis = getattr(ax, f"{which}axis", None)
    if axis is None:
        return
    if r is None:
        if rulers.pop(which, None) is not None:
            _set_ticks_auto(ax, which)
        return
    rulers[which] = r
    setattr(ax, f"_mylab_{which}ticklabels", None)
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
    if r.kind in ("datetime", "datenum"):
        loc, fmt = _date_tick_classes()
        axis.set_major_locator(loc(r, 7 if which == "x" else 8))
        axis.set_major_formatter(fmt(r))
    elif r.kind == "duration":
        from .datetimes import format_duration
        axis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 3, 5, 6, 10]))
        axis.set_major_formatter(FuncFormatter(
            lambda x, pos, r=r: format_duration(float(np.round(x * r.unit, 9)), r.fmt or r.dfmt)))
    else:
        n = len(r.categories)
        axis.set_major_locator(FixedLocator(list(range(1, n + 1))))
        axis.set_major_formatter(FuncFormatter(
            lambda x, pos, r=r: r.categories[int(round(x)) - 1]
            if abs(x - round(x)) < 1e-9 and 1 <= round(x) <= len(r.categories) else ""))


def refresh_categorical_ruler(ax, which: str) -> None:
    """After plotting on a categorical axis: one tick per category, and limits that show them all."""
    r = _ruler_of(ax, which)
    if r is not None and r.kind == "categorical":
        install_ruler(ax, which, r)
        inverted = getattr(ax, f"{which}axis_inverted", lambda: False)()
        lo, hi = sorted(getattr(ax, f"get_{which}lim")())
        n = len(r.categories)
        if lo > 0.5 or hi < n + 0.5:
            lo, hi = min(lo, 0.5), max(hi, n + 0.5)
            getattr(ax, f"set_{which}lim")((hi, lo) if inverted else (lo, hi))


_AXES_STATE = ("_mylab_rulers", "_mylab_tickfmt", "_mylab_geo", "_mylab_geo_lat", "_mylab_geo_lon",
               "_mylab_geo_manual", "_mylab_basemap", "_mylab_bubbles", "_mylab_chart_host", "_mylab_contour",
               "_mylab_xticklabels", "_mylab_yticklabels", "_mylab_zticklabels", "_mylab_xticks_fixed",
               "_mylab_yticks_fixed", "_mylab_zticks_fixed", "_mylab_daspect", "_mylab_pbaspect",
               "_mylab_subtitle", "_mylab_minor_grid", "_mylab_colororder", "_mylab_styleorder",
               "_mylab_cycle_start", "_mylab_cycle_lines", "_mylab_zoom_reset", "_mylab_secondary",
               "_mylab_bubblelegend", "_mylab_tickinterp", "_mylab_lights", "_mylab_camera",
               "_mylab_pos_before_legend", "_mylab_mappable")


def _clear_rulers(ax) -> None:
    """Forget what mylab remembers about an axes that has just been cleared."""
    for name in _AXES_STATE:
        ax.__dict__.pop(name, None)
    fig = ax.figure
    if fig is not None:
        for a in list(fig.axes):  # small axes that a chart placed over this one
            if getattr(a, "_mylab_matrix_of", None) is ax:
                a.remove()


# axes ---------------------------------------------------------------------------------------

def _lim_get(which):
    def g(h):
        f = getattr(h.obj, f"get_{which}lim", None)
        if f is None and which == "z" and getattr(h.obj, "name", "") == "rectilinear":
            # 2-D axes still have a z range: [-1 1], or what axis([... zmin zmax]) / zlim set
            return _row(list(getattr(h.obj, "_mylab_zlim", None) or (-1.0, 1.0)))
        if f is None:
            return np.zeros((1, 0))
        lo, hi = f()
        return ruler_values(h.obj, which, sorted([lo, hi]))
    return g


def _lim_set(which):
    def s(h, v):
        lim = np.real(axis_numbers(h.obj, which, v)).flatten()
        if lim.size != 2:
            raise MatlabError("Limits must be a 2-element vector of increasing values.")
        if which == "z" and not hasattr(h.obj, "set_zlim") and getattr(h.obj, "name", "") == "rectilinear":
            h.obj._mylab_zlim = (float(lim[0]), float(lim[1]))  # kept for when the axes turn 3-D
            return
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
    if which in "xy" and getattr(ax, "name", "") == "rectilinear" and ax.get_aspect() != "auto" \
            and ax.get_adjustable() == "datalim":
        ax.set_adjustable("box")


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
    if a is None and which == "ZLabel":
        return _wrap(h.interp, "text", _zlabel_2d(ax))
    if a is None:
        raise MatlabError(f"These axes have no {which}.")
    return _wrap(h.interp, "text", a.label)


def _zlabel_2d(ax):
    """The ZLabel of 2-D axes: a hidden text object, as MATLAB axes always have one."""
    t = getattr(ax, "_mylab_zlabel", None)
    if t is None:
        t = ax.text(0, 0, "")
        t.set_visible(False)
        ax._mylab_zlabel = t
    return t


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
        if isinstance(h, _FakeAxesHandle) and not h.rulers:
            return _row(ticks)
        return ruler_values(ax, which, ticks)
    return g


def _ticks_set(which):
    def s(h, v):
        if is_text(v) and text(v).lower() in ("auto", "manual") and _ruler_of(h.obj, which) is None:
            if text(v).lower() == "auto":
                _set_ticks_auto(h.obj, which)
            return
        _set_tick_values(h.obj, which, np.real(axis_numbers(h.obj, which, v)).flatten())
    return s


def _set_tick_values(ax, which: str, vals: np.ndarray) -> None:
    """Fix the tick positions of one axis, keeping the label format of its ruler."""
    from matplotlib.ticker import FixedLocator
    r = _ruler_of(ax, which)
    if r is None and getattr(ax, "_mylab_tickfmt", {}).get(which) is None:
        getattr(ax, f"set_{which}ticks")(vals)
    else:
        getattr(ax, f"{which}axis").set_major_locator(FixedLocator(list(vals)))
        if r is not None and r.kind in ("datetime", "datenum") and vals.size > 1:
            d = float(np.min(np.diff(np.sort(r.seconds(vals)))))
            r.step = ("y", 1) if d >= 360 * 86400 else ("mo", 1) if d >= 28 * 86400 else ("s", d)
    setattr(ax, f"_mylab_{which}ticks_fixed", True)


def _set_ticks_auto(ax, which):
    from matplotlib.ticker import AutoLocator, LogFormatterSciNotation, LogLocator, ScalarFormatter
    axis = getattr(ax, f"{which}axis")
    r = _ruler_of(ax, which)
    setattr(ax, f"_mylab_{which}ticks_fixed", False)
    setattr(ax, f"_mylab_{which}ticklabels", None)
    if r is not None:
        install_ruler(ax, which, r)
        return
    if getattr(ax, f"get_{which}scale", lambda: "linear")() == "log":
        axis.set_major_locator(LogLocator())
        axis.set_major_formatter(LogFormatterSciNotation())
        return
    axis.set_major_locator(AutoLocator())
    axis.set_major_formatter(ScalarFormatter())


def _ticklabels_get(which):
    def g(h):
        ax = h.obj
        ticks = np.asarray(_ticks_get(which)(_FakeAxesHandle(ax)), dtype=float).flatten()
        fixed = getattr(ax, f"_mylab_{which}ticklabels", None)
        if fixed is not None:
            return _cell_column([char(s) for s in fixed])
        if _ruler_of(ax, which) is not None or getattr(ax, "_mylab_tickfmt", {}).get(which) is not None:
            labels = [str(s) for s in getattr(ax, f"{which}axis").get_major_formatter().format_ticks(list(ticks))]
        else:
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
    mode = getattr(ax, "_mylab_tickinterp", None)
    getattr(ax, f"set_{which}ticklabels")([_tex(None, lab, mode or "tex", "ticks") for lab in full])
    setattr(ax, f"_mylab_{which}ticklabels", full)
    setattr(ax, f"_mylab_{which}ticks_fixed", True)


class _FakeAxesHandle:
    def __init__(self, ax, rulers: bool = False):
        self.obj = ax
        self.extra: dict = {}
        self.rulers = rulers  # True: tick values come back as datetime/duration/categorical


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
        scale = text(v).lower()
        if scale not in ("linear", "log"):
            raise MatlabError(f"{which.upper()}Scale must be 'linear' or 'log'.")
        setter = getattr(h.obj, f"set_{which}scale", None)
        if setter is None:
            raise MatlabError(f"These axes have no {which}-axis.")
        if getattr(h.obj, f"get_{which}scale")() == scale:
            return
        setter(scale)  # matplotlib installs its own tick locator and formatter for the new scale
        r = _ruler_of(h.obj, which)
        if r is not None:
            install_ruler(h.obj, which, r)
        _tick_aligned(h.obj)
    return s


def _scale_get(which):
    def g(h):
        sc = getattr(h.obj, f"get_{which}scale")()
        return char("log" if sc == "log" else "linear")
    return g


def _nextplot_get(h):
    st = _state(h.interp)
    if st.hold.get(id(h.obj), False):
        return char("add")
    return char(getattr(h.obj, "_mylab_nextplot", "") or "replace")


def _nextplot_set(h, v):
    mode = text(v).lower()
    if mode not in ("add", "replace", "replacechildren", "replaceall"):
        raise MatlabError("NextPlot must be 'add', 'replace', 'replacechildren' or 'replaceall'.")
    _state(h.interp).hold[id(h.obj)] = mode == "add"
    h.obj._mylab_nextplot = "replacechildren" if mode == "replacechildren" else ""


def _legend_prop_get(h):
    leg = h.obj.get_legend()
    return _wrap(h.interp, "legend", leg) if leg is not None else np.zeros((0, 0))


def _cmap_prop_get(h):
    return _cmap_query(h.interp)


def _cmap_prop_set(h, v):
    _apply_colormap(h.interp, v, axes=[h.obj])


def _mappables(ax) -> list:
    out = [m for m in list(ax.collections) + list(ax.images) if getattr(m, "get_array", lambda: None)()
           is not None]
    extra = getattr(ax, "_mylab_mappable", None)  # colour scale of objects drawn with explicit face colours
    if extra is not None and not out:
        out.append(extra)
    return out


def face_colors(coll) -> np.ndarray:
    """RGBA face colours of a collection in the order of its faces.

    A 3-D collection reports its colours sorted by depth once it has been drawn, so they are taken
    from the colour data (or the colours it was given) instead.
    """
    arr = coll.get_array() if hasattr(coll, "get_array") else None
    if arr is not None and np.size(arr):
        return np.atleast_2d(np.asarray(coll.to_rgba(np.asarray(arr, dtype=float)), dtype=float))
    fc = getattr(coll, "_facecolor3d", None)
    if fc is None:
        fc = coll.get_facecolor()
    return np.atleast_2d(np.asarray(fc, dtype=float))


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
        az, el = getattr(ax, "_mylab_view", None) or (ax.azim + 90, ax.elev)
        return _row([az, el])
    return _row([0, 90])


def _view_set(h, v):
    vals = np.real(fnum(v)).flatten()
    if vals.size != 2:
        raise MatlabError("View must be [azimuth elevation] in degrees.")
    _view_apply(h.interp, h.obj, (float(vals[0]), float(vals[1])))


def _view_apply(interp, ax, want: tuple):
    """Look at ``ax`` from azimuth/elevation ``want`` (degrees, MATLAB's convention); returns the axes.

    A 2-D axes seen from anywhere but straight above becomes a 3-D axes holding the same objects, and an
    axes made 3-D that way returns to 2-D when it is seen from above again (view(2))."""
    from . import _axes3d_convert as conv
    az, el = float(want[0]), float(want[1])
    flat = abs(el - 90) < 1e-9 and abs(((az + 180) % 360) - 180) < 1e-9
    if not hasattr(ax, "view_init"):
        if flat or getattr(ax, "name", "") != "rectilinear":
            return ax
        ax = _replace_axes(interp, ax, "3d", carry=True)
    elif flat and conv.can_return_to_2d(ax):
        ax = _replace_axes(interp, ax, "rectilinear", carry=True)
        _tick_aligned(ax)
        return ax
    # matplotlib measures azimuth from the x-axis; MATLAB from the negative y-axis
    ax.view_init(elev=el, azim=az - 90)
    ax._mylab_view = (az, el)
    return ax


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
    w, hh = h.obj.get_size_inches() * _fig_dpi(h.obj)
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
        w, hh = p[2] * _fig_dpi(h.obj), p[3] * _fig_dpi(h.obj)
    elif units == "centimeters":
        w, hh = p[2] / 2.54 * _fig_dpi(h.obj), p[3] / 2.54 * _fig_dpi(h.obj)
    else:
        w, hh = p[2], p[3]
    if w <= 0 or hh <= 0:
        raise MatlabError("Figure width and height must be positive.")
    h.obj._mylab_origin = (float(p[0]), float(p[1]))
    h.obj.set_size_inches(w / _fig_dpi(h.obj), hh / _fig_dpi(h.obj), forward=True)


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
                   "GraphicsSmoothing", "Theme", "ResizeFcn", "WindowButtonDownFcn", "WindowButtonMotionFcn",
                   "WindowButtonUpFcn", "WindowKeyPressFcn", "WindowKeyReleaseFcn", "WindowScrollWheelFcn",
                   "KeyReleaseFcn", "CreateFcn", "DeleteFcn", "ButtonDownFcn", "Interruptible", "BusyAction",
                   "AutoResizeChildren", "Scrollable", "Icon"}


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
    raw = getattr(h.obj, "_mylab_raw", None)
    if raw is None:
        raw = [t.get_text() for t in h.obj.get_texts()]
    return Cell.from_list([char(s) for s in raw])


def _legend_strings_set(h, v):
    labels = _text_list(v)
    leg = h.obj
    raw = list(getattr(leg, "_mylab_raw", None) or [t.get_text() for t in leg.get_texts()])
    interpreter = getattr(leg, "_mylab_interpreter", None)
    for i, (t, lab) in enumerate(zip(leg.get_texts(), labels)):
        t.set_text(_tex(h.interp, lab, interpreter, "legend"))
        raw[i] = lab
    leg._mylab_raw = raw
    leg._mylab_labels = raw


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
    "NumColumns": (lambda h: _num_val(getattr(h.obj, "_ncols", 1) or 1),
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
    "Label": (_cb_label, lambda h, v: None if isinstance(v, GraphicsHandle)
              else set_text(h.interp, _cb_label(h).obj, _label_text(v))),
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


def keep_cdata(im, value) -> None:
    """Remember the CData an image was given (uint8, logical, ...): matplotlib draws a double copy of it."""
    im._mylab_cdata = (value, im.get_array())


def _image_cdata(h):
    kept = getattr(h.obj, "_mylab_cdata", None)
    if kept is not None and kept[1] is h.obj.get_array():
        return kept[0]
    return np.asarray(h.obj.get_array(), dtype=float)


def _image_cdata_set(h, v):
    c = np.real(fnum(v))
    if c.ndim == 3 and isinstance(v, np.ndarray) and v.dtype.kind in "iu":  # truecolour integers: 0..intmax
        c = c / float(np.iinfo(v.dtype).max)
    h.obj.set_data(c)
    if isinstance(v, np.ndarray):
        keep_cdata(h.obj, v)


_IMAGE_PROPS: dict[str, tuple] = {
    "CData": (_image_cdata, _image_cdata_set),
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
    "Marker": (lambda h: marker_name(h.obj.markerline.get_marker()),
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

def _layout_subtitle_text(fig, create: bool = True):
    t = getattr(fig, "_mylab_layout_subtitle", None)
    if t is None and create:
        t = fig.text(0.5, 0.93, "", ha="center", va="top", fontsize=10)
        fig._mylab_layout_subtitle = t
    return t


def _layout_title(h, which: str):
    fig = h.obj.fig
    if which == "Subtitle":
        return _wrap(h.interp, "text", _layout_subtitle_text(fig))
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
    "animatedline": {**_LINE_PROPS,
                     "MaximumNumPoints": (lambda h: _num_val(h.extra.get("MaximumNumPoints", np.inf)),
                                          lambda h, v: h.extra.__setitem__("MaximumNumPoints", _scalar(v)))},
    "rectangle": {**_PATCH_PROPS,
                  "Position": (lambda h: _row(h.extra["Position"]), lambda h, v: _rect_update(h, position=v)),
                  "Curvature": (lambda h: _row(h.extra["Curvature"]), lambda h, v: _rect_update(h, curvature=v))},
}

_IGNORED: dict[str, set] = {"axes": _AXES_IGNORED, "polaraxes": _AXES_IGNORED, "figure": _FIGURE_IGNORED,
                            "legend": _LEGEND_IGNORED}

_COMMON_PROPS: dict[str, tuple] = {
    "Type": (lambda h: char(_TYPE.get(h.kind, h.kind)), None),
    "Visible": (_visible_get, _visible_set),
    "Parent": (_parent_get, _parent_set),
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
        low = name.lower()
        if h.kind in ("root", "figure", "axes") and low.startswith(("default", "factory")) and _ROOT_HOOKS:
            return _ROOT_HOOKS[0](h, name)
        if h.kind == "placeholder":
            raise MatlabError("This element of the graphics array is an empty placeholder (see gobjects).")
        raise MatlabError(f"Unrecognized property '{name}' for class '{_short(h.kind)}'.")
    if not _alive(h) and key not in ("Type", "Tag", "UserData"):
        raise MatlabError("Invalid or deleted object.")
    if entry is None:  # accepted-and-ignored property
        return h.extra.get(key, np.zeros((0, 0)))
    if key == "Visible" and h.kind == "figure":
        return _oo(h.extra.get("Visible", True))
    return entry[0](h)


def _set_prop(h: GraphicsHandle, name: str, value: Any) -> None:
    if h.kind in ("root", "figure", "axes") and name.lower().startswith("default") and _ROOT_HOOKS:
        _ROOT_HOOKS[1](h, name, value)
        return
    key, entry = _lookup(h, name)
    if key is None:
        if h.kind == "placeholder":
            raise MatlabError("This element of the graphics array is an empty placeholder (see gobjects).")
        raise MatlabError(f"Unrecognized property '{name}' for class '{_short(h.kind)}'.")
    if not _alive(h):
        raise MatlabError("Invalid or deleted object.")
    if isinstance(value, (GraphicsHandle, GraphicsArray)) and entry is not None:
        # Nested assignment such as h.Parent.Title.String = 'x' or ax.YAxis(2).Color = 'r' writes the (same)
        # handle or handles back.
        try:
            cur = entry[0](h)
            if cur is value or (isinstance(cur, GraphicsArray) and isinstance(value, GraphicsArray)
                                and len(cur.items) == len(value.items)
                                and all(a is b for a, b in zip(cur.items, value.items))):
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
    if _LINK_HOOK is not None and h.extra.get("_links"):
        _LINK_HOOK(h, key, value)


def _resolve(interp, v) -> list[GraphicsHandle]:
    """Handles from a handle, handle array, figure number or 0 (the root)."""
    if isinstance(v, (GraphicsHandle, GraphicsArray)):
        return _handles_of(v)
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.size >= 1 and not is_char(v):
        st = _state(interp)
        out = []
        for x in np.real(v).flatten(order="F"):
            x = float(x)
            if x != int(x):  # the number of a line, axes, ...
                h = numbered_handle(x)
                if h is None or not _alive(h):
                    raise MatlabError("Invalid or deleted object.")
                out.append(h)
                continue
            n = int(x)
            if n == 0:
                out.append(_wrap(interp, "root", _ROOT))
            elif n in st.figures:
                out.append(_wrap(interp, "figure", st.figures[n]))
            else:
                raise MatlabError("Invalid or deleted object.")
        return out
    if isinstance(v, MObject) and hasattr(v, "graphics_handles"):
        return list(v.graphics_handles())
    raise MatlabError("Expected a graphics handle.")


# -- builtins: get / set and handle utilities ---------------------------------------------------

@builtin("set")
def _set_b(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for set.")
    hs = _resolve(interp, args[0])
    rest = list(args[1:])
    if not rest or (len(rest) == 1 and is_text(rest[0])):  # set(h) / set(h, 'Prop'): what can be set
        if len(hs) != 1:
            raise MatlabError("set(h) and set(h, name) need a single graphics object.")
        info = _settable_props(hs[0])
        if rest:
            key, entry = _lookup(hs[0], text(rest[0]))
            if key is None:
                raise MatlabError(f"Unrecognized property '{text(rest[0])}' for class '{_short(hs[0].kind)}'.")
            if key not in info.fields:
                raise MatlabError(f"The '{key}' property is read-only.")
            val = info.get_field(0, key)
            if nargout:
                return val
            interp.out("".join(f"    '{to_str(c)}'\n" for c in val.values_f())
                       if val.data.size else f"    The '{key}' property does not have a fixed set of values.\n")
            return None
        if nargout:
            return info
        lines = []
        for f in info.fields:
            c = info.get_field(0, f)
            lines.append(f"    {f}: " + ("{" + " | ".join(f"'{to_str(x)}'" for x in c.values_f()) + "}"
                                         if c.data.size else "{}"))
        interp.out("\n".join(lines) + "\n")
        return None
    pairs: list = []
    per_handle = None
    if len(rest) == 2 and isinstance(rest[0], Cell) and isinstance(rest[1], Cell):
        names = [to_str(x) for x in rest[0].values_f()]
        vals = rest[1]
        if vals.data.ndim == 2 and vals.data.shape == (len(hs), len(names)) and len(hs) > 1:
            per_handle = [[(n, vals.data[i, j]) for j, n in enumerate(names)] for i in range(len(hs))]
        else:
            pairs = list(zip(names, vals.values_f()))
    else:
        k = 0
        while k < len(rest):
            a = rest[k]
            if isinstance(a, Struct):
                pairs.extend((f, a.data.flat[0][f]) for f in a.fields)
                k += 1
                continue
            if k + 1 >= len(rest):
                raise MatlabError("set expects property name/value pairs.")
            pairs.append((text(a, "property name"), rest[k + 1]))
            k += 2
    for i, h in enumerate(hs):
        for name, value in (per_handle[i] if per_handle is not None else pairs):
            _set_prop(h, name, value)
    _redraw(interp)
    return None


def property_names(h: GraphicsHandle) -> list[str]:
    """Every property of a handle, in alphabetical order."""
    names = set(_PROPS.get(h.kind, {})) | set(_COMMON_PROPS) | set(_IGNORED.get(h.kind, ()))
    if h.kind in ("root", "figure"):
        names.discard("Annotation")
    return sorted(names, key=str.lower)


def _all_props(h: GraphicsHandle) -> Struct:
    mapping = {}
    for name in property_names(h):
        try:
            mapping[name] = _get_prop(h, name)
        except (MatlabError, ValueError, TypeError, AttributeError, IndexError, KeyError, NotImplementedError):
            continue
    return Struct.scalar(mapping)


def _settable_props(h: GraphicsHandle) -> Struct:
    """set(h): the properties that can be set, each with the list of words it accepts (if it is a choice)."""
    mapping = {}
    for name in property_names(h):
        key, entry = _lookup(h, name)
        if entry is not None and entry[1] is None:
            continue
        choices = _PROP_CHOICES.get(name)
        mapping[name] = _cell_column([char(c) for c in choices]) if choices else Cell(np.empty((0, 0), dtype=object))
    return Struct.scalar(mapping)


_PROP_CHOICES = {
    "Visible": ("on", "off"), "Box": ("on", "off"), "XGrid": ("on", "off"), "YGrid": ("on", "off"),
    "ZGrid": ("on", "off"), "XMinorGrid": ("on", "off"), "YMinorGrid": ("on", "off"), "XMinorTick": ("on", "off"),
    "YMinorTick": ("on", "off"), "XScale": ("linear", "log"), "YScale": ("linear", "log"),
    "ZScale": ("linear", "log"), "XDir": ("normal", "reverse"), "YDir": ("normal", "reverse"),
    "ZDir": ("normal", "reverse"), "NextPlot": ("add", "replace", "replacechildren"),
    "LineStyle": ("-", "--", ":", "-.", "none"),
    "Marker": ("o", "+", "*", ".", "x", "_", "|", "square", "diamond", "^", "v", ">", "<", "pentagram",
               "hexagram", "none"),
    "FontWeight": ("normal", "bold"), "FontAngle": ("normal", "italic"),
    "HorizontalAlignment": ("left", "center", "right"),
    "VerticalAlignment": ("middle", "top", "bottom", "baseline", "cap"),
    "Interpreter": ("tex", "latex", "none"), "HandleVisibility": ("on", "off", "callback"),
    "Units": ("normalized", "pixels", "inches", "centimeters", "points", "characters"),
    "TickDir": ("in", "out", "both", "none"), "XAxisLocation": ("bottom", "top", "origin"),
    "YAxisLocation": ("left", "right", "origin"), "Orientation": ("vertical", "horizontal"),
    "Normalization": ("count", "probability", "percentage", "countdensity", "cumcount", "pdf", "cdf"),
    "Layer": ("bottom", "top"), "XLimMode": ("auto", "manual"), "YLimMode": ("auto", "manual"),
    "ZLimMode": ("auto", "manual"), "Selected": ("on", "off"), "HitTest": ("on", "off"),
    "Clipping": ("on", "off"), "BusyAction": ("queue", "cancel"), "Interruptible": ("on", "off"),
}


@builtin("get")
def _get_b(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for get.")
    hs = _resolve(interp, args[0])
    if len(args) == 1:
        if len(hs) == 1:
            s = _all_props(hs[0])
            if nargout == 0:
                width = max((len(f) for f in s.fields), default=0)
                interp.out("".join(f"    {f.rjust(width)}: {_fmt_prop(s.get_field(0, f))}\n" for f in s.fields))
                return None
            return s
        raise MatlabError("get with several handles needs a property name.")
    if is_text(args[1]) and text(args[1]).lower() in ("default", "factory") and len(hs) == 1 and _ROOT_HOOKS:
        from . import plot_objects
        return plot_objects.all_defaults(interp, factory=text(args[1]).lower() == "factory")
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


@builtin("handle")
def _handle_fn(interp, args, nargout):
    """handle(h): the graphics object(s) numeric handles stand for; objects come back unchanged."""
    if len(args) != 1:
        raise MatlabError("handle takes one argument.")
    v = args[0]
    if isinstance(v, MObject):
        return v
    if isinstance(v, np.ndarray) and not is_char(v) and v.dtype.kind in "biuf":
        hs = _resolve(interp, v) if v.size else []
        if len(hs) == 1:
            return hs[0]
        return GraphicsArray(hs, row=v.ndim == 2 and v.shape[0] == 1 and v.size > 1, shape=v.shape)
    raise MatlabError("handle: the argument must be a graphics handle or a handle object.")


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
            if not ok and n != int(n):
                h = numbered_handle(n)
                ok = h is not None and _alive(h) and (want is None or _TYPE.get(h.kind) == want)
            flags.append(ok)
        return np.array(flags, dtype=bool).reshape(v.shape, order="F")
    if isinstance(v, MObject) and v.is_handle and want is None:  # waitbar, VideoWriter, user handles
        return np.array([[not getattr(v, "_deleted", False)]])
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
        sp, kws = split_spec(spec, kw)
        lines.extend(fn(x, y, sp, **kws) if sp else fn(x, y, **kws))
    _tag_lines(interp, lines, opts)
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
            spec = _norm_spec(text(args[k]))
            k += 1
        if x.shape[0] > 1 and x.shape[1] > 1:
            cols = [(x[:, j], y[:, j], z[:, j]) for j in range(x.shape[1])]
        else:
            cols = [(x.flatten(), y.flatten(), z.flatten())]
        sp, kws = split_spec(spec, kw)
        for xs, ys, zs in cols:
            lines.extend(ax.plot(xs, ys, zs, sp, **kws) if sp else ax.plot(xs, ys, zs, **kws))
    if not lines:
        raise MatlabError("plot3 needs x, y and z data.")
    _tag_lines(interp, lines, opts)
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
    _tag_lines(interp, lines, opts)
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
        spec = _norm_spec(text(args.pop(0)))
    _reject_text(args, "line")
    if args:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
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
    spec, kw = split_spec(spec, kw)
    lines = ax.plot(xs, ys, spec, **kw) if spec else ax.plot(xs, ys, **kw)
    if len(funcs) == 1:
        ax.set_xlim(lims[0], lims[1])
    _tag_lines(interp, lines, opts)
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("polarplot")
@_with_axes_arg
def _polarplot(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    nums = [a for a in args if not is_text(a)]
    spec = next((_norm_spec(text(a)) for a in args if is_text(a) and _is_linespec(text(a))), "")
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
    spec, kw = split_spec(spec, _mpl_kwargs(opts))
    lines = ax.plot(theta, rho, spec, **kw) if spec else ax.plot(theta, rho, **kw)
    _tag_lines(interp, lines, opts)
    _redraw(interp)
    return _ret(interp, "line", lines, nargout)


@builtin("scatter")
@_with_axes_arg
def _scatter(interp, args, nargout):
    return _scatter_core(interp, args, nargout, three_d=False)


@builtin("scatter3")
@_with_axes_arg
def _scatter3(interp, args, nargout):
    return _scatter_core(interp, args, nargout, three_d=True)


def _scatter_core(interp, args, nargout, three_d: bool):
    """scatter(x, y, sz, c, marker, 'filled', Name, Value) and scatter3 with a z before sz."""
    name = "scatter3" if three_d else "scatter"
    known = {"markerfacecolor", "markeredgecolor", "displayname", "linewidth", "markerfacealpha",
             "markeredgealpha", "sizedata"}
    args, opts = _split_name_values(args, known, "scatter")
    filled = any(is_text(a) and text(a) == "filled" for a in args)
    args = [a for a in args if not (is_text(a) and text(a) == "filled")]
    k = 3 if three_d else 2
    for a in args[k:]:
        if is_text(a) and not _is_color_text(a) and not _is_marker_text(a):
            raise unknown_option_error(name, text(a), known, None)
    if len(args) < k or any(is_text(a) for a in args[:k]):
        raise MatlabError(f"{name}: needs {'x, y and z' if three_d else 'x and y'} coordinates.")
    ax = _axes(interp, projection="3d" if three_d else "rectilinear")
    coords = [np.real(fnum(a)).flatten() for a in args[:k]]
    x = coords[0]
    s = 36.0
    if len(args) > k and isinstance(args[k], np.ndarray) and not is_char(args[k]) and args[k].size:
        sv = np.real(fnum(args[k])).flatten()
        s = float(sv[0]) if sv.size == 1 else sv
    kw: dict = {"depthshade": False} if three_d else {}
    if len(args) > k + 1 and not (is_text(args[k + 1]) and not _is_color_text(args[k + 1])):
        c = args[k + 1]
        kw["c"] = _color(c) if is_text(c) else (np.real(fnum(c)).flatten() if fnum(c).size == x.size
                                                 else _color(c))
        if isinstance(kw["c"], np.ndarray):
            kw["cmap"] = _current_cmap(interp)
    marker = next((_mpl_marker(text(a)) for a in args[k:] if is_text(a) and not _is_color_text(a)), "o")
    if "c" not in kw:
        import matplotlib as mpl
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        kw["c"] = colors[len(ax.collections) % len(colors)]
    hollow_mapped = False  # open markers whose edges take their colours from CData and the colormap
    if not filled:
        if not isinstance(kw["c"], np.ndarray):
            kw["edgecolors"] = kw.pop("c")
            kw["facecolors"] = "none"
        else:
            hollow_mapped = True
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
    coll = ax.scatter(*coords, s=s, marker=marker, **kw)
    if hollow_mapped and "facecolors" not in kw:
        # (passing facecolors='none' with c would fill the markers: c takes precedence)
        coll.set_facecolor("none")
        if "edgecolors" not in kw:
            coll.set_edgecolor(None)  # edges follow the mapped colours
    h = _wrap(interp, "scatter", coll)
    h.extra["Marker"] = marker
    if "markeredgealpha" in opts:
        _set_prop(h, "MarkerEdgeAlpha", opts["markeredgealpha"])
    _apply_extra(interp, [h], opts, "scatter")
    _redraw(interp)
    return h if nargout > 0 else None


def _is_color_text(a) -> bool:
    s = text(a)
    return s.lower() in _COLOR_NAMES or (len(s) == 1 and s in "rgbcmykw") or s.startswith("#")


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
                                           "facealpha", "basevalue", "horizontal", "barlayout", "edgealpha",
                                           "linestyle", "cdata", "tag", "userdata", "visible", "showbaseline",
                                           "handlevisibility"})
    layout = "grouped"
    color = None
    nums = []
    for a in args:
        if is_text(a):
            low = text(a).lower()
            if low in ("stacked", "grouped", "hist", "histc"):
                layout = low
            elif _is_color_text(a):
                color = _color(a)
            else:
                raise MatlabError(f"bar: unknown option '{text(a)}'.")
        else:
            nums.append(a)
    if "barlayout" in opts:
        layout = text(opts["barlayout"]).lower()
        if layout not in ("stacked", "grouped"):
            raise MatlabError("BarLayout must be 'grouped' or 'stacked'.")
    if "horizontal" in opts:
        horizontal = _onoff(opts["horizontal"])
    if not nums:
        raise MatlabError("bar needs data, e.g. bar([1 2 3]).")
    width = 0.8
    if len(nums) >= 2 and fnum(nums[-1]).size == 1 and (len(nums) == 3 or fnum(nums[0]).size > 1):
        width = _scalar(nums.pop())
    if "barwidth" in opts:
        width = _scalar(opts["barwidth"])
    if layout in ("hist", "histc"):
        width = 1.0
    if len(nums) == 1:
        y = np.real(fnum(nums[0]))
        x = np.arange(1, (y.size if y.shape[0] == 1 or y.shape[1] == 1 else y.shape[0]) + 1, dtype=float)
    else:
        x, y = np.real(fnum(nums[0])).flatten(), np.real(fnum(nums[1]))
    if y.shape[0] == 1 or y.shape[1] == 1:
        y = y.reshape(-1, 1) if x.size != 1 or y.size == 1 else y.reshape(1, -1)
    elif y.shape[0] != x.size and y.shape[1] == x.size:
        y = y.T
    if y.shape[0] != x.size:
        raise MatlabError("bar: X must have one value for each row of Y.")
    base = _scalar(opts["basevalue"]) if "basevalue" in opts else 0.0
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"edgecolor": "black", "linewidth": 0.5}
    if "edgecolor" in opts:
        kw["edgecolor"] = _color(opts["edgecolor"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "facealpha" in opts:
        kw["alpha"] = _scalar(opts["facealpha"])
    flat = "facecolor" in opts and is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "flat"
    face = _color(opts["facecolor"]) if "facecolor" in opts and not flat else color
    n = y.shape[1]
    conts, bottoms = [], []
    spacing = float(np.min(np.diff(np.unique(x)))) if np.unique(x).size > 1 else 1.0
    pos_top = np.full(y.shape[0], base)
    neg_top = np.full(y.shape[0], base)
    stacked = layout == "stacked"
    for k in range(n):
        vals = np.where(np.isfinite(y[:, k]), y[:, k], 0.0)
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
        if stacked:
            bottom = np.where(vals >= 0, pos_top, neg_top)
            pos_top = np.where(vals >= 0, pos_top + vals, pos_top)
            neg_top = np.where(vals < 0, neg_top + vals, neg_top)
            length = vals
        else:
            bottom = np.full(vals.shape, base)
            length = vals - base
        if horizontal:
            cont = ax.barh(pos, length, height=w, left=bottom, **ckw)
        else:
            cont = ax.bar(pos, length, width=w, bottom=bottom, **ckw)
        conts.append(cont)
        bottoms.append(bottom if stacked else None)
    ticks_ok = x.size <= 20 and not _state(interp).hold.get(id(ax), False) and \
        not getattr(ax, f"_mylab_{'y' if horizontal else 'x'}ticks_fixed", False)
    if ticks_ok:
        (ax.set_yticks if horizontal else ax.set_xticks)(np.unique(x))
    _redraw(interp)
    hs = []
    for k, cont in enumerate(conts):
        h = _wrap(interp, "bar", cont)
        h.extra.update({"BarWidth": width, "XData": x, "YData": y[:, k], "axes": ax, "figure": ax.figure,
                        "Horizontal": horizontal, "BarLayout": "stacked" if stacked else "grouped",
                        "Bottom": bottoms[k], "BaseValue": np.array([[base]])})
        if flat:
            h.extra["FaceColorMode"] = "flat"
        for key, name in (("cdata", "CData"), ("edgealpha", "EdgeAlpha"), ("linestyle", "LineStyle"),
                          ("visible", "Visible"), ("tag", "Tag"), ("userdata", "UserData"),
                          ("showbaseline", "ShowBaseLine"), ("handlevisibility", "HandleVisibility")):
            if key in opts:
                _set_prop(h, name, opts[key])
        hs.append(h)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else GraphicsArray(hs, row=True)


@builtin("histogram")
@_with_axes_arg
def _histogram(interp, args, nargout):
    args, opts = _split_name_values(args, {"numbins", "binedges", "binwidth", "binlimits", "normalization",
                                           "facecolor", "edgecolor", "facealpha", "displayname", "displaystyle",
                                           "orientation", "linewidth", "binmethod", "bincounts", "edgealpha",
                                           "linestyle", "tag", "visible", "userdata", "handlevisibility"})
    if not args and "bincounts" not in opts:
        raise MatlabError("histogram needs data, e.g. histogram(randn(100, 1)).")
    counts = None
    if "bincounts" in opts:  # histogram('BinEdges', edges, 'BinCounts', counts): bins counted beforehand
        if "binedges" not in opts:
            raise MatlabError("histogram: 'BinCounts' needs 'BinEdges' as well.")
        counts = np.real(fnum(opts["bincounts"])).flatten()
        edges_c = np.real(fnum(opts["binedges"])).flatten()
        if counts.size != edges_c.size - 1:
            raise MatlabError("histogram: BinCounts must have one value fewer than BinEdges.")
        raw = np.repeat((edges_c[:-1] + edges_c[1:]) / 2, np.maximum(counts, 0).astype(int)).reshape(1, -1)
    else:
        raw = args[0]
    data = _finite(raw)
    nbins = edges = None
    if len(args) > 1 and not is_text(args[1]):
        b = np.real(fnum(args[1])).flatten()
        if b.size == 1:
            nbins = int(b[0])
        else:
            edges = b
    elif len(args) > 1:
        raise MatlabError(f"histogram: unknown option '{text(args[1])}'.")
    if "numbins" in opts:
        nbins = to_int(opts["numbins"], "NumBins")
    if "binedges" in opts:
        edges = np.real(fnum(opts["binedges"])).flatten()
    if edges is not None and (edges.size < 2 or np.any(np.diff(edges) <= 0)):
        raise MatlabError("BinEdges must be a vector of increasing values.")
    width = _scalar(opts["binwidth"]) if "binwidth" in opts else None
    limits = np.real(fnum(opts["binlimits"])).flatten() if "binlimits" in opts else None
    method = text(opts["binmethod"]).lower() if "binmethod" in opts else None
    if limits is not None:
        if limits.size != 2 or not limits[0] < limits[1]:
            raise MatlabError("BinLimits must be [min max] with min < max.")
        data = data[(data >= limits[0]) & (data <= limits[1])]
        if nbins is None and width is None and edges is None:
            ref = _hist_edges(data, method=method) if data.size else np.array([0.0, 1.0])
            nbins = max(len(ref) - 1, 1)
    e = _hist_edges(data, nbins=nbins, edges=edges, width=width, limits=limits, method=method)
    norm = _check_norm(text(opts["normalization"])) if "normalization" in opts else "count"
    style: dict = {"bin_method": method or "auto"}
    if "facecolor" in opts and not (is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "auto"):
        style["facecolor"] = _color(opts["facecolor"])
    if "edgecolor" in opts and not (is_text(opts["edgecolor"]) and text(opts["edgecolor"]).lower() == "auto"):
        style["edgecolor"] = style["edgecolor_set"] = _color(opts["edgecolor"])
    if "linewidth" in opts:
        style["linewidth"] = _scalar(opts["linewidth"])
    if "facealpha" in opts:
        style["alpha"] = _scalar(opts["facealpha"])
    if "edgealpha" in opts:
        style["edgealpha"] = _scalar(opts["edgealpha"])
    if "linestyle" in opts:
        style["linestyle"] = text(opts["linestyle"])
    if "displaystyle" in opts:
        ds = text(opts["displaystyle"]).lower()
        if ds not in ("bar", "stairs"):
            raise MatlabError("DisplayStyle must be 'bar' or 'stairs'.")
        style["display_style"] = ds
    orientation = text(opts["orientation"]).lower() if "orientation" in opts else "vertical"
    if orientation not in ("vertical", "horizontal"):
        raise MatlabError("Orientation must be 'vertical' or 'horizontal'.")
    ax = _axes(interp, projection="rectilinear")
    label = text(opts["displayname"]) if "displayname" in opts else None
    hs = _Hist(ax, data, e, norm, orientation, style, label, counts=counts)
    hs.draw()
    _redraw(interp)
    h = _wrap(interp, "histogram", hs)
    shape_src = np.real(fnum(raw))
    h.extra["DataShape"] = lambda: shape_src
    if limits is not None:
        h.extra["BinLimitsMode"] = "manual"
    for key, name in (("tag", "Tag"), ("visible", "Visible"), ("userdata", "UserData"),
                      ("handlevisibility", "HandleVisibility")):
        if key in opts:
            _set_prop(h, name, opts[key])
    return h if nargout > 0 else None


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
    spec = next((_norm_spec(text(a)) for a in args if is_text(a) and _is_linespec(text(a))), "")
    ax = _axes(interp, projection="rectilinear")
    x, y = _xy_simple([a for a in args if not is_text(a)])
    spec, scol = split_spec(spec, {})
    cont = ax.stem(x, y, linefmt=spec or None)
    if "color" in scol:
        cont.markerline.set_color(scol["color"])
        cont.stemlines.set_color(scol["color"])
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
    h = _wrap(interp, "stem", cont)
    h.extra.update({"axes": ax, "figure": ax.figure})
    _tag_lines(interp, [h], opts, "stem")
    return (h) if nargout > 0 else None


@builtin("stairs")
@_with_axes_arg
def _stairs(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    spec = next((_norm_spec(text(a)) for a in args if is_text(a) and _is_linespec(text(a))), "")
    ax = _axes(interp, projection="rectilinear")
    x, y = _xy_simple(args)
    kw = _mpl_kwargs(opts)
    lines = ax.step(x, y, spec, where="post", **kw) if spec else ax.step(x, y, where="post", **kw)
    _tag_lines(interp, lines, opts, "stair")
    _redraw(interp)
    return _ret(interp, "stair", lines, nargout)


@builtin("area")
@_with_axes_arg
def _area(interp, args, nargout):
    args, opts = _split_name_values(args, {"facecolor", "edgecolor", "facealpha", "displayname", "linewidth",
                                           "linestyle", "basevalue", "edgealpha", "tag", "visible"})
    for a in args:
        if is_text(a):
            raise MatlabError(f"area: unknown option '{text(a)}'.")
    nums = [np.real(fnum(a)) for a in args]
    if not nums:
        raise MatlabError("area needs data, e.g. area([1 3 2]).")
    base = 0.0
    if len(nums) == 3 or (len(nums) == 2 and nums[1].size == 1 and nums[0].size > 1):
        base = float(nums.pop().flat[0])
    if "basevalue" in opts:
        base = _scalar(opts["basevalue"])
    Y = nums[-1]
    if Y.shape[0] == 1 or Y.shape[1] == 1:
        Y = Y.reshape(-1, 1)
    if len(nums) > 1:
        x = nums[0].flatten() if 1 in nums[0].shape else nums[0][:, 0]
        if x.size != Y.shape[0] and x.size == Y.shape[1]:
            Y = Y.T
    else:
        x = np.arange(1, Y.shape[0] + 1, dtype=float)
    if x.size != Y.shape[0]:
        raise MatlabError("area: X must have one value for each row of Y.")
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"alpha": _scalar(opts["facealpha"]) if "facealpha" in opts else 1.0,
                "edgecolor": "black", "linewidth": 0.5}
    if "edgecolor" in opts:
        kw["edgecolor"] = _color(opts["edgecolor"])
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "linestyle" in opts:
        kw["linestyle"] = _mpl_linestyle(text(opts["linestyle"]))
    if "facecolor" in opts and not (is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "flat"):
        kw["facecolor"] = _color(opts["facecolor"])
    lower = np.full(Y.shape[0], base)
    total = np.zeros(Y.shape[0])
    hs = []
    for k in range(Y.shape[1]):
        vals = np.where(np.isfinite(Y[:, k]), Y[:, k], 0.0)
        total = total + vals
        upper = total.copy()
        ckw = dict(kw)
        if "displayname" in opts and Y.shape[1] == 1:
            ckw["label"] = text(opts["displayname"])
        coll = ax.fill_between(x, lower, upper, **ckw)
        h = _wrap(interp, "area", coll)
        h.extra.update({"XData": x.reshape(1, -1), "YData": Y[:, k].reshape(1, -1), "BaseValue": np.array([[base]]),
                        "axes": ax, "figure": ax.figure})
        for key, name in (("tag", "Tag"), ("visible", "Visible"), ("edgealpha", "EdgeAlpha")):
            if key in opts:
                _set_prop(h, name, opts[key])
        hs.append(h)
        lower = upper
    _redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else GraphicsArray(hs, row=True)


@builtin("fill")
@_with_axes_arg
def _fill(interp, args, nargout):
    return _fill_core(interp, args, nargout, patch=False)


@builtin("patch")
@_with_axes_arg
def _patch(interp, args, nargout):
    lead = []
    for a in args:
        if is_text(a):
            break
        lead.append(a)
    if lead and isinstance(lead[0], Struct):  # patch(fv): a structure with faces and vertices
        fv = lead[0]
        low = {f.lower(): f for f in fv.fields}
        if "faces" not in low or "vertices" not in low:
            raise MatlabError("patch(S): the structure needs the fields faces and vertices.")
        named = [char("Faces"), fv.get_field(0, low["faces"]), char("Vertices"), fv.get_field(0, low["vertices"])]
        if "facevertexcdata" in low and np.asarray(fv.get_field(0, low["facevertexcdata"])).size:
            named += [char("FaceVertexCData"), fv.get_field(0, low["facevertexcdata"]),
                      char("FaceColor"), char("flat")]
        return _patch_named(interp, named + list(args[1:]), nargout)
    names = {text(a).lower() for a in args[len(lead)::2] if is_text(a)}
    if "faces" in names or "vertices" in names or "xdata" in names:
        return _patch_named(interp, args[len(lead):], nargout)
    if len(lead) >= 4 or (len(lead) == 3 and len(args) > 3 and _is_color_text(args[3])
                          and np.asarray(lead[2]).size == np.asarray(lead[0]).size):
        return _fill3_core(interp, args, nargout, fresh=False)
    return _fill_core(interp, args, nargout, patch=True)


def _fill_core(interp, args, nargout, patch: bool):
    args, opts = _split_name_values(args, {"facecolor", "edgecolor", "facealpha", "displayname", "linewidth"},
                                    "patch")
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
    h = _wrap(interp, "patch", polys)
    h.extra.update({"axes": ax, "figure": ax.figure,
                    "_polys": [np.asarray(q.get_xy(), dtype=float)[:-1] for q in polys]})
    _apply_extra(interp, [h], opts, "patch")
    return (h) if nargout > 0 else None


@builtin("errorbar")
@_with_axes_arg
def _errorbar(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS | {"capsize"})
    orient, spec, nums = "vertical", "", []
    for a in args:
        if is_text(a):
            low = text(a).lower()
            if low in ("horizontal", "vertical", "both"):
                orient = low
            elif _is_linespec(text(a)) and not spec:
                spec = _norm_spec(text(a))
            else:
                raise MatlabError(f"errorbar: unknown option '{text(a)}'.")
        else:
            nums.append(np.real(fnum(a)))
    n = len(nums)
    empty = np.zeros((0, 0))
    if n == 2:
        y, neg = nums
        x, pos = None, neg
        xneg = xpos = empty
    elif n == 3:
        x, y, neg = nums
        pos = neg
        xneg = xpos = empty
    elif n == 4:
        x, y, neg, pos = nums
        xneg = xpos = empty
    elif n == 6:
        x, y, neg, pos, xneg, xpos = nums
        orient = "explicit"
    else:
        raise MatlabError("errorbar expects errorbar(y, err), errorbar(x, y, err), errorbar(x, y, neg, pos) or "
                          "errorbar(x, y, yneg, ypos, xneg, xpos).")
    if y.shape[0] == 1 or y.shape[1] == 1:
        y = y.reshape(-1, 1)
    m, ns = y.shape

    def cols(v, what):
        """One column per series from a vector (shared) or a matrix; None for an empty input."""
        if v is None or v.size == 0:
            return None
        if v.size == 1:
            return np.full((m, ns), float(v.flat[0]))
        if v.shape[0] == 1 or v.shape[1] == 1:
            if v.size != m:
                raise MatlabError(f"errorbar: {what} must have the same number of points as y.")
            return np.repeat(v.reshape(-1, 1), ns, axis=1)
        if v.shape != (m, ns):
            raise MatlabError(f"errorbar: {what} must be the same size as y.")
        return v

    X = cols(x, "x") if x is not None else np.repeat(np.arange(1.0, m + 1).reshape(-1, 1), ns, axis=1)
    neg, pos, xneg, xpos = cols(neg, "the errors"), cols(pos, "the errors"), cols(xneg, "the x errors"), \
        cols(xpos, "the x errors")
    if orient == "horizontal":
        xneg, xpos, neg, pos = neg, pos, None, None
    elif orient == "both":
        xneg, xpos = neg, pos
    ax = _axes(interp, projection="rectilinear")
    spec, kw = split_spec(spec, _mpl_kwargs(opts))
    cap = _scalar(opts["capsize"]) if "capsize" in opts else 6.0
    fmt = spec or "-"
    hs = []
    for k in range(ns):
        def pair(lo, hi):
            if lo is None and hi is None:
                return None
            lo_k = np.abs(lo[:, k]) if lo is not None else np.zeros(m)
            hi_k = np.abs(hi[:, k]) if hi is not None else np.zeros(m)
            return np.vstack([lo_k, hi_k])
        cont = ax.errorbar(X[:, k], y[:, k], yerr=pair(neg, pos), xerr=pair(xneg, xpos), fmt=fmt,
                           capsize=cap / 2.0, **kw)
        h = _wrap(interp, "errorbar", cont)
        zero = np.zeros((0, 0))
        h.extra.update({
            "axes": ax, "figure": ax.figure, "CapSize": cap, "XData": X[:, k], "YData": y[:, k],
            "YNegativeDelta": neg[:, k] if neg is not None else zero,
            "YPositiveDelta": pos[:, k] if pos is not None else zero,
            "XNegativeDelta": xneg[:, k] if xneg is not None else zero,
            "XPositiveDelta": xpos[:, k] if xpos is not None else zero})
        hs.append(h)
    _tag_lines(interp, [h.obj for h in hs], opts, "errorbar")
    _redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else GraphicsArray(hs, row=True)


@builtin("pie")
@_with_axes_arg
def _pie(interp, args, nargout):
    """pie(X), pie(X, explode), pie(X, labels), pie(X, explode, labels).

    As in MATLAB: values summing to more than 1 are shared out as fractions of the whole, otherwise each value is
    the fraction of a slice itself (a partial pie when they sum to less than 1). Slices start at 12 o'clock and go
    anticlockwise, take their colours from the colormap, and are labelled with their percentages unless labels are
    given. The output is the patches and texts, one pair per slice: [patch1 text1 patch2 text2 ...].
    """
    if not args:
        raise MatlabError("Not enough input arguments.", "MATLAB:narginchk:notEnoughInputs")
    x = np.real(fnum(args[0])).flatten().astype(float)
    rest = list(args[1:])
    labels = None
    if rest and (isinstance(rest[-1], (Cell, StringArray)) or is_text(rest[-1])):
        labels = _text_list(rest.pop())
    explode = None
    if rest:
        explode = np.real(fnum(rest.pop(0))).flatten()
        if explode.size != x.size:
            raise MatlabError("Explode must be the same size as X.")
    if rest:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
    if x.size == 0 or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise MatlabError("Data for a pie chart must be finite, nonnegative numbers.")
    if labels is not None and len(labels) != x.size:
        raise MatlabError("Cell array of strings must be the same length as X.")
    total = float(x.sum())
    frac = x / total if total > 1 else x
    if labels is None:
        labels = []
        for f in frac:
            pct = 100.0 * f
            labels.append("<1%" if 0 < pct < 0.5 else f"{round(pct):d}%")
    ax = _axes(interp, projection="rectilinear")
    import matplotlib as mpl
    cmap = _current_cmap(interp)
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    n = x.size
    colors = [cmap(k / (n - 1) if n > 1 else 0.0) for k in range(n)]
    wedges, texts = ax.pie(frac, explode=None if explode is None else np.where(explode != 0, 0.1, 0.0),
                           labels=labels, labeldistance=1.15, startangle=90, counterclock=True,
                           normalize=False, colors=colors, wedgeprops={"edgecolor": "k", "linewidth": 0.5})
    ax.set_aspect("equal")
    ax.set_axis_off()
    _redraw(interp)
    if nargout <= 0:
        return None
    hs = []
    for w, t in zip(wedges, texts):
        hs.extend([_wrap(interp, "patch", w), _wrap(interp, "text", t)])
    return GraphicsArray(hs, row=True)


@builtin("quiver")
@_with_axes_arg
def _quiver(interp, args, nargout):
    args, opts = _split_name_values(args, {"color", "linewidth", "autoscale", "autoscalefactor",
                                           "displayname", "maxheadsize"}, kind="quiver")
    _reject_text(args, "quiver", flags=("filled",))
    spec = next((_norm_spec(text(a)) for a in args if is_text(a) and _is_linespec(text(a))), "")
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
        color = spec_color(spec)
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
    h = _wrap(interp, "quiver", q)
    h.extra.update({"UData": np.asarray(U, float), "VData": np.asarray(V, float), "XData": np.asarray(X, float),
                    "YData": np.asarray(Y, float)})
    _apply_extra(interp, [h], opts, "quiver")
    return (h) if nargout > 0 else None


# -- surfaces and images ----------------------------------------------------------------------

def _shading(interp) -> str:
    return interp.state.get("shading", "faceted")


def _surface(interp, args, kind: str, nargout: int, handle_first: bool = False):
    args, opts = _split_name_values(args, {"edgecolor", "facecolor", "facealpha", "linestyle", "linewidth",
                                           "levellist", "showtext", "linecolor", "levelstep", "labelspacing",
                                           "fill", "displayname", "meshstyle", "tag", "visible"})
    specs = [text(a) for a in args if is_text(a)]
    for sp in specs:
        if not _is_linespec(sp):
            raise MatlabError(f"{kind}: unknown option '{sp}'.")
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
        if "levelstep" in opts and levels is None:
            step = _scalar(opts["levelstep"])
            if step <= 0:
                raise MatlabError("LevelStep must be positive.")
            zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            levels = np.arange(math.ceil(zmin / step) * step, zmax + step * 1e-9, step)
        fill = kind == "contourf"
        if "fill" in opts:
            fill = _onoff(opts["fill"])
        kw: dict = {"cmap": _current_cmap(interp)}
        line_color = opts.get("linecolor", opts.get("edgecolor"))
        spec = specs[0] if specs else ""
        m = re.search(r"[rgbcmykw]", spec)
        if m and line_color is None:
            line_color = char(m.group(0))
        style = re.sub(r"[rgbcmykw]", "", spec)
        no_lines = False
        if "linestyle" in opts:
            style = text(opts["linestyle"])
        if style.lower() == "none":
            no_lines, style = True, ""
        if line_color is not None and is_text(line_color) and text(line_color).lower() == "none":
            no_lines, line_color = True, None
        elif line_color is not None and is_text(line_color) and text(line_color).lower() in ("flat", "auto"):
            line_color = None
        line_kw: dict = {}
        if style in ("-", "--", ":", "-."):
            line_kw["linestyles"] = style
        if "linewidth" in opts:
            line_kw["linewidths"] = _scalar(opts["linewidth"])
        if fill:
            if "facealpha" in opts:
                kw["alpha"] = _scalar(opts["facealpha"])
        else:
            kw.update(line_kw)
            if line_color is not None:
                kw.pop("cmap")
                kw["colors"] = [_color(line_color)]
            if no_lines:
                kw["linewidths"] = 0
        fn = ax.contourf if fill else ax.contour
        cs = fn(X, Y, Z, levels=levels, **kw) if levels is not None else fn(X, Y, Z, **kw)
        overlay = None
        if fill and not no_lines and len(cs.levels) > 1:  # filled contours are outlined in black
            okw = {"colors": [_color(line_color)] if line_color is not None else ["k"],
                   "linewidths": line_kw.get("linewidths", 0.5)}
            if "linestyles" in line_kw:
                okw["linestyles"] = line_kw["linestyles"]
            overlay = ax.contour(X, Y, Z, levels=cs.levels, **okw)
            overlay._mylab_internal = True
        show = "showtext" in opts and _onoff(opts["showtext"])
        if show:
            ax.clabel(overlay if overlay is not None else cs, inline=True, fontsize=8)
        if "displayname" in opts:
            cs.set_label(text(opts["displayname"]))
        _redraw(interp)
        ax._mylab_contour = cs
        h = _wrap(interp, "contour", cs)
        h.extra["_contour"] = {"X": X, "Y": Y, "Z": Z, "levels": None if levels is None or np.isscalar(levels)
                               or isinstance(levels, int) else np.asarray(levels, dtype=float),
                               "kw": kw, "fill": fill, "showtext": show}
        if overlay is not None:
            h.extra["_artists"] = [overlay]
        for key, name in (("tag", "Tag"), ("visible", "Visible")):
            if key in opts:
                _set_prop(h, name, opts[key])
        if nargout >= 2:  # [C, h] = contour(...)
            return _contour_matrix(overlay if overlay is not None else cs), h
        if nargout == 1 and not handle_first:  # C = contour(Z): the contour matrix comes first
            return _contour_matrix(overlay if overlay is not None else cs)
        return h if nargout > 0 else None
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
        if "linestyle" in opts and text(opts["linestyle"]).lower() == "none":
            coll.set_edgecolor("none")
    else:
        coll = draw_mesh(interp, ax, X, Y, Z)
        if "edgecolor" in opts and not (is_text(opts["edgecolor"])
                                        and text(opts["edgecolor"]).lower() in ("flat", "interp")):
            coll.set_edgecolor(_color(opts["edgecolor"]))
        if "facecolor" in opts and not (is_text(opts["facecolor"])
                                        and text(opts["facecolor"]).lower() in ("flat", "interp")):
            coll.set_facecolor(_color(opts["facecolor"]))
        if "facealpha" in opts:
            coll.set_alpha(_scalar(opts["facealpha"]))
        if "linewidth" in opts:
            coll.set_linewidth(_scalar(opts["linewidth"]))
    _redraw(interp)
    h = _wrap(interp, "surface", coll)
    h.extra.update({"XData": X, "YData": Y, "ZData": Z})
    for key, name in (("tag", "Tag"), ("visible", "Visible"), ("displayname", "DisplayName")):
        if key in opts:
            _set_prop(h, name, opts[key])
    hook = interp.state.get("relight_hook")
    if hook is not None:
        hook(interp, ax)
    return (h) if nargout > 0 else None


def face_values(Z: np.ndarray, rcount: int, ccount: int):
    """Mean height of each face that plot_surface draws for a grid, in the order it draws them."""
    rows, cols = Z.shape
    rstride = max(int(np.ceil(rows / rcount)), 1)
    cstride = max(int(np.ceil(cols / ccount)), 1)
    ri = list(range(0, rows - 1, rstride)) + [rows - 1]
    ci = list(range(0, cols - 1, cstride)) + [cols - 1]
    vals = []
    for r0, r1 in zip(ri[:-1], ri[1:]):
        for c0, c1 in zip(ci[:-1], ci[1:]):
            block = Z[r0:r1 + 1, c0:c1 + 1]
            if np.isnan(block).any():
                continue
            vals.append((Z[r0, c0] + Z[r1, c0] + Z[r0, c1] + Z[r1, c1]) / 4.0)
    return np.array(vals, dtype=float)


def draw_mesh(interp, ax, X, Y, Z, hidden: bool | None = None):
    """A wireframe surface the way MATLAB draws a mesh: lines coloured by height over opaque faces."""
    import matplotlib as mpl
    rc, cc = min(Z.shape[0], 200), min(Z.shape[1], 200)
    if hidden is None:
        hidden = interp.state.get("mesh_hidden", True)
    face = ax.get_facecolor() if hidden else "none"
    coll = ax.plot_surface(X, Y, Z, rcount=rc, ccount=cc, color="w", shade=False, linewidth=0.5,
                           edgecolor="k")
    vals = face_values(np.asarray(Z, dtype=float), rc, cc)
    cmap = _current_cmap(interp)
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    try:
        n_faces = len(coll.get_facecolor()) if len(coll.get_facecolor()) > 1 else vals.size
        if vals.size and vals.size == n_faces:
            lo, hi = float(vals.min()), float(vals.max())
            coll.set_edgecolor(cmap((vals - lo) / (hi - lo) if hi > lo else np.full(vals.shape, 0.5)))
    except (ValueError, TypeError):
        pass
    coll.set_facecolor(face)
    coll._mylab_surface = False
    coll._mylab_mesh = True
    return coll


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
    for a in args:
        if is_text(a):
            raise MatlabError(f"peaks: the option '{text(a)}' is not supported; peaks takes peaks(n), "
                              "peaks(v) or peaks(X, Y).", "MATLAB:peaks:UnknownOption")
    if len(args) > 2:
        raise MatlabError("Too many input arguments.", "MATLAB:TooManyInputs")
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


def _image_core(interp, args, nargout, scaled: bool):
    args, opts = _split_name_values(args, {"cdata", "xdata", "ydata", "cdatamapping", "alphadata", "parent",
                                           "interpolation"}, "image")
    for a in args:
        if is_text(a):
            raise MatlabError(f"{'imagesc' if scaled else 'image'}: unknown option '{text(a)}'.")
    nums = list(args)
    clim = None
    x = y = None
    raw = None
    if len(nums) == 2 and fnum(nums[1]).size == 2 and fnum(nums[0]).size > 2:
        raw, clim = nums[0], np.real(fnum(nums[1])).flatten()
    elif len(nums) >= 3:
        x, y, raw = np.real(fnum(nums[0])).flatten(), np.real(fnum(nums[1])).flatten(), nums[2]
        if len(nums) > 3:
            clim = np.real(fnum(nums[3])).flatten()
    elif nums:
        raw = nums[0]
    if "cdata" in opts:
        raw = opts["cdata"]
    if raw is None:
        raw = np.zeros((1, 1))
    if "xdata" in opts:
        x = np.real(fnum(opts["xdata"])).flatten()
    if "ydata" in opts:
        y = np.real(fnum(opts["ydata"])).flatten()
    if "cdatamapping" in opts:
        mapping = text(opts["cdatamapping"]).lower()
        if mapping not in ("scaled", "direct"):
            raise MatlabError("CDataMapping must be 'scaled' or 'direct'.")
        scaled = mapping == "scaled"
    integer = isinstance(raw, np.ndarray) and raw.dtype.kind in "iu"
    C = np.real(fnum(raw))
    parent = opts.get("parent")
    st = _state(interp)
    prev = st.override_ax
    if parent is not None:
        hs = _handles_of(parent)
        if not hs or hs[0].kind not in ("axes", "polaraxes"):
            raise MatlabError("Parent must be an axes.")
        st.override_ax = hs[0].obj
    try:
        ax = _axes(interp, projection="rectilinear")
    finally:
        st.override_ax = prev
    kw: dict = {"aspect": "auto", "cmap": _current_cmap(interp), "origin": "upper",
                "interpolation": "nearest"}
    if C.ndim == 3:
        kw.pop("cmap")
        if integer:
            C = C / float(np.iinfo(raw.dtype).max)
        C = np.clip(C, 0, 1)
    if x is None:
        x = np.array([1.0, C.shape[1]])
    if y is None:
        y = np.array([1.0, C.shape[0]])
    nr, nc = C.shape[:2]
    x0, x1 = (x[0], x[-1]) if x.size > 1 else (x[0], x[0] + nc - 1)
    y0, y1 = (y[0], y[-1]) if y.size > 1 else (y[0], y[0] + nr - 1)
    dx = (x1 - x0) / max(nc - 1, 1) / 2 if nc > 1 else 0.5
    dy = (y1 - y0) / max(nr - 1, 1) / 2 if nr > 1 else 0.5
    kw["extent"] = (x0 - dx, x1 + dx, y1 + dy, y0 - dy)
    if clim is not None:
        if clim.size != 2 or not clim[0] < clim[1]:
            raise MatlabError("The colour limits must be [cmin cmax] with cmin < cmax.")
        kw["vmin"], kw["vmax"] = float(clim[0]), float(clim[1])
    elif not scaled and C.ndim == 2:  # the values are positions in the colormap
        n = _cmap_matrix(_current_cmap(interp), 256).shape[0] if isinstance(_current_cmap(interp), str) \
            else getattr(_current_cmap(interp), "N", 256)
        kw["vmin"], kw["vmax"] = (0.0, float(n - 1)) if integer else (1.0, float(n))
    im = ax.imshow(C, **kw)
    if isinstance(raw, np.ndarray):
        keep_cdata(im, raw)
    h = _wrap(interp, "image", im)
    h.extra["CDataMapping"] = "scaled" if scaled else "direct"
    if "alphadata" in opts:
        _set_prop(h, "AlphaData", opts["alphadata"])
    if "interpolation" in opts:
        _set_prop(h, "Interpolation", opts["interpolation"])
    _apply_extra(interp, [h], opts, "image")
    _redraw(interp)
    return h if nargout > 0 else None


builtin("imagesc")(_with_axes_arg(lambda interp, args, nargout: _image_core(interp, args, nargout, True)))
builtin("image")(_with_axes_arg(lambda interp, args, nargout: _image_core(interp, args, nargout, False)))


_CB_LOCATIONS = {"eastoutside": "right", "east": "right", "westoutside": "left", "west": "left",
                 "northoutside": "top", "north": "top", "southoutside": "bottom", "south": "bottom"}
_CB_INSIDE = {"east": [0.90, 0.06, 0.04, 0.88], "west": [0.06, 0.06, 0.04, 0.88],
              "north": [0.06, 0.90, 0.88, 0.04], "south": [0.06, 0.06, 0.88, 0.04]}


@builtin("colorbar")
@_with_axes_arg
def _colorbar(interp, args, nargout):
    args = list(args)
    st = _state(interp)
    target = None
    k = 0
    while k < len(args) - 1:  # colorbar('peer', ax) and colorbar(..., 'Parent'/'Target', ax)
        if is_text(args[k]) and text(args[k]).lower() in ("peer", "target") and \
                isinstance(args[k + 1], GraphicsHandle):
            target = args[k + 1].obj
            del args[k:k + 2]
            continue
        k += 1
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind == "tiledlayout":
        lay = args.pop(0).obj
        target = lay.order[-1] if lay.order else None
    prev = st.override_ax
    if target is not None:
        st.override_ax = target
    try:
        ax = _axes(interp, fresh=False)
    finally:
        st.override_ax = prev
    if args and is_text(args[0]) and text(args[0]).lower() in ("off", "delete", "hide"):
        cb = getattr(ax, "_mylab_colorbar", None)
        if cb is not None:
            st.handles.pop(id(cb), None)
            cb.remove()
            ax._mylab_colorbar = None
        _redraw(interp)
        return None
    loc = "eastoutside"
    if args and is_text(args[0]) and text(args[0]).lower() in _CB_LOCATIONS:
        loc = text(args.pop(0)).lower()
    if len(args) % 2 or not all(is_text(a) for a in args[::2]):
        bad = next((a for a in args if is_text(a)), None)
        raise MatlabError(f"colorbar: unknown option '{text(bad)}'." if bad is not None
                          else "colorbar options must be name/value pairs.")
    pairs = [(text(args[i]), args[i + 1]) for i in range(0, len(args), 2)]
    for name, value in pairs:
        if name.lower() == "location":
            loc = text(value).lower()
            if loc not in _CB_LOCATIONS and loc != "manual":
                raise MatlabError(f"'{text(value)}' is not a colorbar location.")
    pairs = [(n, v) for n, v in pairs if n.lower() != "location"]
    old = getattr(ax, "_mylab_colorbar", None)
    old_handle = st.handles.pop(id(old), None) if old is not None else None
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
        mappable.set_clim(0, 1)
    if loc in _CB_INSIDE:  # drawn inside the axes
        cax = ax.inset_axes(_CB_INSIDE[loc])
        cax._mylab_internal = True
        cb = ax.figure.colorbar(mappable, cax=cax, orientation="vertical" if loc in ("east", "west")
                                else "horizontal")
        if loc == "west":
            cax.yaxis.set_ticks_position("right")
        if loc == "north":
            cax.xaxis.set_ticks_position("bottom")
    else:
        cb = ax.figure.colorbar(mappable, ax=ax, location=_CB_LOCATIONS.get(loc, "right"))
    cb._mylab_parent_ax = ax
    ax._mylab_colorbar = cb
    if old_handle is not None:  # an existing handle keeps pointing at the colorbar
        old_handle.obj = cb
        old_handle.extra.pop("deleted", None)
        st.handles[id(cb)] = old_handle
    h = _wrap(interp, "colorbar", cb)
    h.extra["Location"] = loc
    for name, value in pairs:
        _set_prop(h, name, value)
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
        spec = args[0]
        if is_text(spec) and "(" in text(spec):
            # colormap('gray(256)'), or colormap gray(256): MATLAB evaluates the text
            name = text(spec).split("(", 1)[0].strip()
            if name.isidentifier() and (name in interp.builtins or interp.find_file_function(name) is not None):
                spec = interp.eval_expression(text(spec))
            else:
                raise MatlabError(f"Unknown colormap '{text(spec)}'.")
        _apply_colormap(interp, spec, axes)
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
    want = None
    if len(args) == 2:
        want = (_scalar(args[0]), _scalar(args[1]))
    elif len(args) == 1:
        v = np.real(fnum(args[0])).flatten()
        if v.size == 1 and int(v[0]) == 2:
            want = (0.0, 90.0)
        elif v.size == 1 and int(v[0]) == 3:
            want = (-37.5, 30.0)
        elif v.size == 2:
            want = (float(v[0]), float(v[1]))
        elif v.size == 3:  # a direction: the camera looks at the origin from the point [x y z]
            xx, yy, zz = (float(t) for t in v)
            if xx == 0 and yy == 0 and zz == 0:
                raise MatlabError("view: the viewing direction cannot be [0 0 0].")
            want = (math.degrees(math.atan2(xx, -yy)), math.degrees(math.atan2(zz, math.hypot(xx, yy))))
        else:
            raise MatlabError("view expects view(az, el), view([az el]), view([x y z]), view(2) or view(3).")
    elif len(args) > 2:
        raise MatlabError("view expects view(az, el), view([az el]), view([x y z]), view(2) or view(3).")
    if want is not None:
        ax = _view_apply(interp, ax, want)
        _redraw(interp)
    if nargout > 0:
        if hasattr(ax, "azim"):
            az, el = getattr(ax, "_mylab_view", None) or (ax.azim + 90, ax.elev)
        else:
            az, el = 0.0, 90.0
        return (_num_val(az), _num_val(el)) if nargout > 1 else _row([az, el])
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
        created = n not in st.figures
        fig = _figure(interp, n)
        rest = list(args[1:])
    else:
        n = 1
        while n in st.figures:
            n += 1
        created = True
        fig = _figure(interp, n)
        rest = list(args)
    h = _wrap(interp, "figure", fig)
    if created:
        hook = interp.state.get("figure_defaults_hook")
        if hook is not None:
            hook(interp, h)
    if len(rest) == 1 and isinstance(rest[0], Struct):
        rest = [x for f in rest[0].fields for x in (char(f), rest[0].data.flat[0][f])]
    if len(rest) % 2:
        raise MatlabError("figure options must be name/value pairs.")
    for i in range(0, len(rest), 2):
        name = text(rest[i], "property name")
        key, _ = _lookup(h, name)
        if key is None and not name.lower().startswith("default"):
            raise MatlabError(f"Unrecognized property '{name}' for class 'Figure'.")
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
    args = list(args)
    st = _state(interp)
    target = None
    if args and not is_text(args[0]):
        target = args.pop(0)
    reset = bool(args) and is_text(args[0]) and text(args[0]).lower() == "reset"
    if args and not reset:
        raise MatlabError("clf expects a figure and, optionally, 'reset'.")
    if target is not None:
        n = _fig_num_arg(interp, target)
        if n not in st.figures:
            raise MatlabError("Invalid or deleted figure handle.")
        fig = st.figures[n]
    else:
        fig = _figure(interp)
        n = st.current
    for ax in list(fig.axes):
        _forget_children(interp, ax)
    fig.clf()
    fig.__dict__.pop("_mylab_grids", None)
    try:
        fig.set_layout_engine("none")
    except (AttributeError, ValueError, RuntimeError):
        pass
    st.axes.pop(n, None)
    st.layouts.pop(n, None)
    h = _wrap(interp, "figure", fig)
    if reset:  # back to the default figure properties, keeping only where and how big the window is
        import matplotlib as mpl
        keep = {k: v for k, v in h.extra.items() if k in ("Units", "_PaperPositionIn", "PaperUnits",
                                                          "PaperPositionMode", "_appdata")}
        h.extra.clear()
        h.extra.update(keep)
        fig.set_facecolor(mpl.rcParams["figure.facecolor"])
        fig._mylab_name = ""
    _redraw(interp)
    return h if nargout > 0 else None


@builtin("cla")
@_with_axes_arg
def _cla(interp, args, nargout):
    st = _state(interp)
    ax = _axes(interp, fresh=False)
    reset = bool(args) and is_text(args[0]) and text(args[0]).lower() == "reset"
    if args and not reset:
        raise MatlabError("cla expects an axes and, optionally, 'reset'.")
    if reset:
        _forget_children(interp, ax)
        ax.cla()
        _clear_rulers(ax)
        ax.__dict__.pop("_mylab_padded", None)
        ax.__dict__.pop("_mylab_nextplot", None)
        h = st.handles.get(id(ax))
        if h is not None:
            keep = {k: v for k, v in h.extra.items() if k in ("Units", "_appdata")}
            h.extra.clear()
            h.extra.update(keep)
        st.hold[id(ax)] = False
        _yy_restyle(ax)
        _axes_created(interp, ax)
        _tick_aligned(ax)
    else:
        _clear_children(interp, ax)
    _redraw(interp)
    return None


def _close_number(interp, n: int) -> None:
    st = _state(interp)
    plt = _plt()
    if n in st.figures:
        fig = st.figures.pop(n)
        for key, g in list(st.handles.items()):  # the figure's objects are gone: forget their handles
            if g.kind in ("figure", "root") or getattr(g, "ui_alive", None) is not None:
                continue
            try:
                owner = _obj_figure(g)
            except (AttributeError, TypeError, IndexError):
                continue
            if owner is fig:
                g.extra["deleted"] = True
                del st.handles[key]
        plt.close(fig)
        st.axes.pop(n, None)
        st.layouts.pop(n, None)
        if st.current == n:
            st.current = max(st.figures) if st.figures else None


def _request_close(interp, n: int, force: bool) -> bool:
    """Close figure ``n`` the way close does: through its CloseRequestFcn unless forced. True if closed."""
    st = _state(interp)
    fig = st.figures.get(n)
    if fig is None:
        return False
    h = _wrap(interp, "figure", fig)
    cb = h.extra.get("CloseRequestFcn")
    custom = cb is not None and not (isinstance(cb, np.ndarray) and cb.size == 0) and not (
        (isinstance(cb, str) and cb in ("", "closereq")) or (is_text(cb) and text(cb) in ("", "closereq")))
    if custom and not force:
        from .. import ui as U
        prev = st.current
        U.invoke(interp, h, cb, Struct.scalar({"Source": h, "EventName": char("Close")}))
        if prev in st.figures and st.current not in st.figures:
            st.current = prev
        return n not in st.figures
    _delete_handle(interp, h)
    return True


@builtin("close")
def _close(interp, args, nargout):
    st = _state(interp)
    args = list(args)
    words = [text(a).lower() for a in args if is_text(a)]
    force = "force" in words or any(w.endswith(" force") for w in words)
    targets: list[int] = []
    named = [a for a in args if is_text(a)]
    rest = [a for a in args if not is_text(a)]
    phrase = " ".join(words)
    if "all" in phrase.split():
        hidden = "hidden" in phrase.split() or force
        for n, f in list(st.figures.items()):
            vis = str(_wrap(interp, "figure", f).extra.get("HandleVisibility", "on"))
            if hidden or vis != "off":
                targets.append(n)
    elif rest:
        for a in rest:
            if isinstance(a, (GraphicsHandle, GraphicsArray)):
                for h in _handles_of(a):
                    if h.kind != "figure":
                        raise MatlabError("close only accepts figure handles.")
                    n = _fig_number(interp, h.obj)
                    if n is not None:
                        targets.append(n)
            elif isinstance(a, np.ndarray) and a.size == 0:
                continue
            else:
                targets.extend(int(round(float(x))) for x in np.real(num(a)).flatten())
    elif named and words[0] not in ("force", "hidden"):
        name = text(named[0])
        hits = [n for n, f in st.figures.items() if getattr(f, "_mylab_name", None) == name]
        if not hits and name.isdigit():
            hits = [int(name)]
        if not hits:
            raise MatlabError(f"Specified window '{name}' does not exist.")
        targets = hits
    elif st.current is not None:
        targets = [st.current]
    ok = True
    for n in targets:
        if n in st.figures:
            ok = _request_close(interp, n, force) and ok
    return np.array([[ok]]) if nargout > 0 else None


@builtin("hold")
@_with_axes_arg
def _hold(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    st = _state(interp)
    if not args:
        st.hold[id(ax)] = not st.hold.get(id(ax), False)
    else:
        mode = "on" if isinstance(args[0], np.ndarray) and not is_char(args[0]) and args[0].size == 1 \
            and bool(args[0].flat[0]) else ("off" if not is_text(args[0]) else text(args[0]).lower())
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
    flags = [text(a) for a in args if is_text(a)]
    args = [a for a in args if not is_text(a)]
    for f in flags:
        if f.lower() not in ("replace", "align"):
            raise MatlabError(f"subplot: unknown option '{f}'.", "MATLAB:subplot:UnknownOption")
    if "position" in opts and not args:
        pos = [float(x) for x in np.real(fnum(opts["position"])).flatten()[:4]]
        if len(pos) != 4 or pos[2] <= 0 or pos[3] <= 0:
            raise MatlabError("subplot: Position must be [left bottom width height].")
        _remove_overlapping(interp, fig, pos, None)
        st.counter = getattr(st, "counter", 0) + 1
        ax = fig.add_axes(pos, label=f"mylab-subplot-{st.counter}")
        st.axes[st.current] = ax
        return _wrap(interp, "axes", ax) if nargout > 0 else None
    if not args:
        raise MatlabError("subplot needs the grid size and a position, e.g. subplot(2, 1, 1).")
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
    if ax is not None and "replace" in flags:
        _forget_children(interp, ax)
        ax.remove()
        ax = None
    if ax is None:
        gs = subplot_grid(fig, m, n)
        box = gs[r0:r1, c0:c1].get_position(fig)
        _remove_overlapping(interp, fig, list(box.bounds), None)
        ax = fig.add_subplot(gs[r0:r1, c0:c1])
    st.axes[st.current] = ax
    return _wrap(interp, "axes", ax) if nargout > 0 else None


def _remove_overlapping(interp, fig, bounds: list, keep) -> None:
    """Delete the axes that a new subplot at ``bounds`` would cover (as subplot does)."""
    x0, y0, w, hh = bounds
    for a in list(fig.axes):
        if a is keep or _is_colorbar_axes(a) or getattr(a, "_mylab_internal", False):
            continue
        if getattr(a, "_mylab_yybase", None) is not None:
            continue
        b = a.get_position()
        ox = min(x0 + w, b.x1) - max(x0, b.x0)
        oy = min(y0 + hh, b.y1) - max(y0, b.y0)
        if ox > 1e-6 and oy > 1e-6:
            _forget_children(interp, a)
            right = getattr(a, "_mylab_yyright", None)
            if right is not None and right in fig.axes:
                right.remove()
            cb = getattr(a, "_mylab_colorbar", None)
            if cb is not None:
                try:
                    cb.remove()
                except (ValueError, AttributeError, KeyError):
                    pass
            a.remove()


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
        self.indexing = "rowmajor"

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

    def tile_rc(self, index: int) -> tuple[int, int]:
        """0-based (row, column) of tile number ``index``."""
        rows, cols = self.grid_size()
        if self.indexing == "columnmajor":
            return (index - 1) % rows, (index - 1) // rows
        return (index - 1) // cols, (index - 1) % cols

    def tile_number(self, r: int, c: int) -> int:
        rows, cols = self.grid_size()
        return c * rows + r + 1 if self.indexing == "columnmajor" else r * cols + c + 1

    def replace(self, old, new):
        self.order = [new if a is old else a for a in self.order]
        for k, a in list(self.tiles.items()):
            if a is old:
                self.tiles[k] = new

    def apply_spacing(self):
        pads = {"loose": 0.06, "normal": 0.06, "compact": 0.02, "tight": 0.0, "none": 0.0}
        try:
            self.fig.set_layout_engine("constrained")
            eng = self.fig.get_layout_engine()
            p = pads.get(self.padding, 0.04)
            sp = pads.get(self.spacing, 0.04)
            eng.set(w_pad=p, h_pad=p, wspace=sp, hspace=sp)
        except (AttributeError, ValueError, RuntimeError, TypeError):
            pass

    def relayout(self):
        rows, cols = self.grid_size()
        gs = self.fig.add_gridspec(rows, cols)
        for i, ax in enumerate(self.order):
            if ax is not None and ax in self.fig.axes:
                r, c = self.tile_rc(i + 1)
                ax.set_subplotspec(gs[r, c])
        return gs

    def resize(self, m: int, n: int) -> None:
        """Change the grid of a fixed layout, keeping every axes at its row and column."""
        placed = []
        for ax in self.order:
            if ax in self.fig.axes:
                ss = ax.get_subplotspec()
                placed.append((ax, ss.rowspan, ss.colspan))
        if any(rs.stop > m or cs.stop > n for _, rs, cs in placed):
            raise MatlabError("The new GridSize is too small for the axes already in the layout.")
        self.mode, self.m, self.n = "fixed", m, n
        gs = self.fig.add_gridspec(m, n)
        self.tiles = {}
        for ax, rs, cs in placed:
            ax.set_subplotspec(gs[rs.start:rs.stop, cs.start:cs.stop])
            for r in rs:
                for c in cs:
                    self.tiles[self.tile_number(r, c)] = ax

    def move(self, ax, index: int, span: tuple) -> None:
        """Put an existing axes at tile ``index`` covering ``span`` = (rows, columns)."""
        if self.mode != "fixed":
            rows, cols = self.grid_size()
            self.mode, self.m, self.n = "fixed", rows, cols
            self.tiles = {i + 1: a for i, a in enumerate(self.order)}
        total = self.m * self.n
        if index < 1 or index > total:
            raise MatlabError(f"Tile index must be between 1 and {total}.")
        r, c = self.tile_rc(index)
        rs, cs = span
        if r + rs > self.m or c + cs > self.n:
            raise MatlabError("The tile span does not fit in the layout.")
        for k, a in list(self.tiles.items()):
            if a is ax:
                del self.tiles[k]
        gs = self.fig.add_gridspec(self.m, self.n)
        ax.set_subplotspec(gs[r:r + rs, c:c + cs])
        for dr in range(rs):
            for dc in range(cs):
                self.tiles[self.tile_number(r + dr, c + dc)] = ax

    def new_tile(self, index: int | None = None, span=(1, 1)):
        if self.mode != "fixed" and span == (1, 1):
            if index is not None and index <= len(self.order):
                return self.order[index - 1]
            self.order.append(None)
            gs = self.relayout()
            r, c = self.tile_rc(len(self.order))
            ax = self.fig.add_subplot(gs[r, c])
            self.order[-1] = ax
            return ax
        if self.mode != "fixed":  # a spanning tile fixes the grid at a size that holds it
            rows, cols = self.grid_size()
            if self.mode == "vertical":
                rows = max(rows, len(self.order) + span[0])
            elif self.mode == "horizontal":
                cols = max(cols, len(self.order) + span[1])
            else:
                cols = max(cols, span[1])
                rows = max(rows, span[0], math.ceil((len(self.order) + span[0] * span[1]) / cols))
            self.resize(max(rows, span[0]), max(cols, span[1]))
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
        r, c = self.tile_rc(index)
        rs, cs = span
        if rs < 1 or cs < 1 or r + rs > self.m or c + cs > self.n:
            raise MatlabError("The tile span does not fit in the layout.")
        covered = [self.tile_number(r + dr, c + dc) for dr in range(rs) for dc in range(cs)]
        for k in covered:  # axes that the new tile would overlap are replaced by it
            old = self.tiles.get(k)
            if old is not None and old in self.fig.axes:
                old.remove()
                self.order = [a for a in self.order if a is not old]
                for kk, a in list(self.tiles.items()):
                    if a is old:
                        del self.tiles[kk]
        gs = self.fig.add_gridspec(self.m, self.n)
        ax = self.fig.add_subplot(gs[r:r + rs, c:c + cs])
        for k in covered:
            self.tiles[k] = ax
        self.order.append(ax)
        self.next = index + 1
        return ax


def _layout_of_axes(interp, ax):
    for lay in _state(interp).layouts.values():
        if ax in lay.order:
            return lay
    return None


_SPACINGS = ("loose", "compact", "tight", "none", "normal")


@builtin("tiledlayout")
def _tiledlayout(interp, args, nargout):
    args, opts = _split_name_values(args, {"tilespacing", "padding", "tileindexing", "parent", "units",
                                           "position", "outerposition", "innerposition"})
    st = _state(interp)
    args = list(args)
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind == "figure":
        opts["parent"] = args.pop(0)
    if "parent" in opts:
        hs = _handles_of(opts["parent"])
        if not hs or hs[0].kind != "figure":
            raise MatlabError("tiledlayout: the parent must be a figure.")
        _figure(interp, _fig_number(interp, hs[0].obj))
    fig = _figure(interp)
    for ax in list(fig.axes):
        _forget_children(interp, ax)
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
        if sz.size != 2 or np.any(sz < 1):
            raise MatlabError("tiledlayout needs a positive number of rows and columns.")
        lay = _Layout(fig, int(sz[0]), int(sz[1]))
    else:
        lay = _Layout(fig, 0, 0, "flow")
    for key, attr in (("tilespacing", "spacing"), ("padding", "padding")):
        if key in opts:
            v = text(opts[key]).lower()
            if v not in _SPACINGS:
                raise MatlabError(f"{'TileSpacing' if key == 'tilespacing' else 'Padding'} must be 'loose', "
                                  "'compact', 'tight' or 'none'.")
            setattr(lay, attr, "loose" if v == "normal" else v)
    if "tileindexing" in opts:
        v = text(opts["tileindexing"]).lower()
        if v not in ("rowmajor", "columnmajor"):
            raise MatlabError("TileIndexing must be 'rowmajor' or 'columnmajor'.")
        lay.indexing = v
    lay.apply_spacing()
    st.layouts[st.current] = lay
    h = _wrap(interp, "tiledlayout", lay)
    for key in ("units", "position", "outerposition", "innerposition"):
        if key in opts:
            h.extra[key] = opts[key]
    return h if nargout > 0 else None


@builtin("nexttile")
def _nexttile(interp, args, nargout):
    st = _state(interp)
    args = list(args)
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind == "tiledlayout":
        lay = args[0].obj
        _figure(interp, _fig_number(interp, lay.fig))
        args = args[1:]
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
    for a in args:
        if is_text(a):
            raise MatlabError(f"nexttile: unknown option '{text(a)}'. Edge tiles ('north', 'south', 'east', "
                              "'west') are not supported.")
        v = np.real(fnum(a)).flatten()
        if v.size == 1:
            index = int(v[0])
        elif v.size == 2:
            span = (int(v[0]), int(v[1]))
        else:
            raise MatlabError("nexttile expects a tile number and, optionally, a span [rows columns].")
    ax = lay.new_tile(index, span)
    st.axes[st.current] = ax
    return _wrap(interp, "axes", ax) if nargout > 0 else None


@builtin("tilenum")
def _tilenum(interp, args, nargout):
    """Tile number of an axes in its tiled layout, or of a row and column of a layout."""
    if not args or not isinstance(args[0], GraphicsHandle):
        raise MatlabError("tilenum needs an axes in a tiled layout, or a layout with a row and a column.")
    h = args[0]
    if h.kind == "tiledlayout":
        if len(args) != 3:
            raise MatlabError("tilenum(t, row, col) needs a row and a column.")
        lay = h.obj
        rows, cols = lay.grid_size()
        r, c = np.real(fnum(args[1])), np.real(fnum(args[2]))
        if np.any(r < 1) or np.any(r > rows) or np.any(c < 1) or np.any(c > cols):
            raise MatlabError("tilenum: the row or column is outside the layout.")
        out = np.vectorize(lambda a, b: lay.tile_number(int(a) - 1, int(b) - 1))(*np.broadcast_arrays(r, c))
        return np.asarray(out, dtype=float)
    lay = _layout_of_axes(interp, h.obj) if h.kind in ("axes", "polaraxes") else None
    if lay is None:
        raise MatlabError("tilenum: the axes is not in a tiled layout.")
    ss = h.obj.get_subplotspec()
    return _num_val(lay.tile_number(ss.rowspan.start, ss.colspan.start))


@builtin("tilerowcol")
def _tilerowcol(interp, args, nargout):
    """Row and column of an axes (or of tile numbers) in a tiled layout."""
    if not args or not isinstance(args[0], GraphicsHandle):
        raise MatlabError("tilerowcol needs an axes in a tiled layout, or a layout with tile numbers.")
    h = args[0]
    if h.kind == "tiledlayout":
        if len(args) != 2:
            raise MatlabError("tilerowcol(t, tilenum) needs tile numbers.")
        nums = np.real(fnum(args[1]))
        rc = [h.obj.tile_rc(int(k)) for k in nums.flatten(order="F")]
        r = np.array([a + 1.0 for a, _ in rc]).reshape(nums.shape, order="F")
        c = np.array([b + 1.0 for _, b in rc]).reshape(nums.shape, order="F")
        return (r, c) if nargout > 1 else r
    lay = _layout_of_axes(interp, h.obj) if h.kind in ("axes", "polaraxes") else None
    if lay is None:
        raise MatlabError("tilerowcol: the axes is not in a tiled layout.")
    ss = h.obj.get_subplotspec()
    r, c = _num_val(ss.rowspan.start + 1), _num_val(ss.colspan.start + 1)
    return (r, c) if nargout > 1 else r


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
            if isinstance(a, (Cell, StringArray)) or (is_text(a) and text(a).lower() not in _TEXT_OPTS
                                                      and not _is_settable("text", text(a).lower())):
                sub = a
                args = [args[0]] + list(args[2:])
        s = _label_text(args[0])
        left, opts = _split_name_values(args[1:], _TEXT_OPTS)
        if left:
            bad = left[0]
            raise MatlabError(f"Unrecognized text property '{text(bad)}'." if is_text(bad)
                              else "Label options must be name/value pairs.")
        kw = _text_kwargs(opts)
        interp_name = text(opts["interpreter"]).lower() if "interpreter" in opts else None
        if isinstance(target, _Layout):
            fig = target.fig
            fn = {"set_title": fig.suptitle, "suptitle": fig.suptitle, "set_xlabel": fig.supxlabel,
                  "set_ylabel": fig.supylabel}.get(method)
            if fn is None:
                raise MatlabError("This label cannot be applied to a tiled layout.")
            t = fn(s, **kw)
            set_text(interp, t, s, interp_name)
            _apply_extra(interp, [t], opts, "text")
            _redraw(interp)
            return _wrap(interp, "text", t) if nargout > 0 else None
        if method == "suptitle":
            fig = target.figure if target is not None else _figure(interp)
            t = fig.suptitle(s, **kw)
            set_text(interp, t, s, interp_name)
            _make_room_for_suptitle(fig)
            _apply_extra(interp, [t], opts, "text")
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
            t = _zlabel_2d(ax)  # 2-D axes: the label is kept, and shows nowhere until the axes are 3-D
            set_text(interp, t, s, interp_name)
        else:
            if method == "set_title" and "fontweight" not in kw:
                kw["fontweight"] = "bold"
            getattr(ax, method)(s, **kw)
            t = {"set_title": ax.title, "set_xlabel": ax.xaxis.label,
                 "set_ylabel": ax.yaxis.label}.get(method) or getattr(getattr(ax, "zaxis", None), "label", None)
            set_text(interp, t, s, interp_name)
            if method == "set_title":
                _layout_subtitle(ax)
        th = _wrap(interp, "text", t)
        _apply_extra(interp, [th], opts, "text")
        if sub is not None:
            st_t = _make_subtitle(ax, _label_text(sub))
            set_text(interp, st_t, _label_text(sub), interp_name)
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
    left, opts = _split_name_values(args[1:], _TEXT_OPTS)
    if left:
        raise MatlabError("subtitle options must be name/value pairs of text properties.")
    if isinstance(target, _Layout):
        t = _layout_subtitle_text(target.fig)
        t.update(_text_kwargs(opts))
    else:
        t = _make_subtitle(ax, _label_text(args[0]), **_text_kwargs(opts))
    set_text(interp, t, _label_text(args[0]),
             text(opts["interpreter"]).lower() if "interpreter" in opts else None)
    if not isinstance(target, _Layout):
        _layout_subtitle(ax)
    _apply_extra(interp, [t], opts, "text")
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


def _room_for_legend(ax, loc_name: str) -> None:
    """Shrink an axes so that a legend placed outside it stays inside the figure (undone when it moves in)."""
    fig = ax.figure
    before = getattr(ax, "_mylab_pos_before_legend", None)
    if before is not None:
        ax.set_position(before)
        ax._mylab_pos_before_legend = None
    engine = getattr(fig, "get_layout_engine", lambda: None)()
    if not loc_name.endswith("outside") or engine is not None or hasattr(ax, "get_zlim"):
        return
    b = ax.get_position()
    ax._mylab_pos_before_legend = [b.x0, b.y0, b.width, b.height]
    if "east" in loc_name:
        ax.set_position([b.x0, b.y0, b.width * 0.80, b.height])
    elif "west" in loc_name:
        ax.set_position([b.x0 + b.width * 0.20, b.y0, b.width * 0.80, b.height])
    elif loc_name.startswith("north"):
        ax.set_position([b.x0, b.y0, b.width, b.height * 0.86])
    else:
        ax.set_position([b.x0, b.y0 + b.height * 0.16, b.width, b.height * 0.84])


def _legend_look(leg) -> dict:
    """The styling of a legend that must survive when it is built again."""
    frame = leg.get_frame()
    texts = leg.get_texts()
    look = {"edgecolor": frame.get_edgecolor(), "facecolor": frame.get_facecolor(),
            "linewidth": frame.get_linewidth(), "alpha": frame.get_alpha(), "visible": leg.get_visible(),
            "title": leg.get_title().get_text(), "anchor": getattr(leg, "_mylab_anchor", None)}
    if texts:
        t = texts[0]
        look.update(color=t.get_color(), family=t.get_fontfamily(), weight=t.get_fontweight(),
                    style=t.get_fontstyle())
    return look


def _apply_legend_look(leg, look: dict) -> None:
    frame = leg.get_frame()
    frame.set_edgecolor(look["edgecolor"])
    frame.set_facecolor(look["facecolor"])
    frame.set_linewidth(look["linewidth"])
    if look.get("alpha") is not None:
        frame.set_alpha(look["alpha"])
    leg.set_visible(look["visible"])
    for t in leg.get_texts():
        if "color" in look:
            t.set_color(look["color"])
            t.set_fontfamily(look["family"])
            t.set_fontweight(look["weight"])
            t.set_fontstyle(look["style"])
    if look.get("anchor") is not None:
        leg.set_bbox_to_anchor(look["anchor"], transform=leg.figure.transFigure)
        leg.set_loc("lower left")
        leg._mylab_anchor = look["anchor"]


def _make_legend(interp, ax, labels=None, handles=None, location=None, orientation=None, ncols=None,
                 reuse=None, fontsize=None, box=None, title=None, reverse=None, interpreter=None):
    pair = _yy_pair(ax)
    host = pair[1] if pair is not None else ax
    look = None
    token = None
    if reuse is not None:
        look = _legend_look(reuse)
        location = location or getattr(reuse, "_mylab_loc", None)
        orientation = orientation or getattr(reuse, "_mylab_orientation", None)
        if ncols is None:
            ncols = getattr(reuse, "_mylab_ncols", None)
        if box is None:
            box = reuse.get_frame_on()
        if fontsize is None and reuse.get_texts():
            fontsize = reuse.get_texts()[0].get_fontsize()
        if labels is None:
            labels = getattr(reuse, "_mylab_labels", None)
        if handles is None:
            handles = getattr(reuse, "_mylab_handles", None)
        if reverse is None:
            reverse = getattr(reuse, "_mylab_reverse", False)
        if interpreter is None:
            interpreter = getattr(reuse, "_mylab_interpreter", None)
        if title is None and look["title"]:
            title = getattr(reuse, "_mylab_title_raw", look["title"])
        token = getattr(reuse, "_mylab_token", None)
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
    if loc_name not in _LOC:
        raise MatlabError(f"'{location}' is not a legend location. Use for example 'northeast', 'best', "
                          "'southoutside' or 'eastoutside'.")
    loc, anchor = _LOC[loc_name]
    kw["loc"] = loc
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    orient = (orientation or "vertical").lower()
    if orient not in ("vertical", "horizontal"):
        raise MatlabError("Orientation must be 'vertical' or 'horizontal'.")
    if ncols is not None:
        kw["ncols"] = ncols
    elif orient == "horizontal":
        kw["ncols"] = max(len(handles), 1)
    if fontsize is not None:
        kw["fontsize"] = fontsize
    if title is not None:
        kw["title"] = _tex(interp, title, interpreter, "legend")
    if token is not None:
        kw["handlelength"] = max(token / 15.0, 0.1)
    shown = [_tex(interp, lab, interpreter, "legend") for lab in labels]
    order = list(range(len(handles)))
    if reverse:
        order.reverse()
    _room_for_legend(ax, loc_name)
    if pair is not None:
        pair[1].set_position(pair[0].get_position())
    leg = host.legend([handles[i] for i in order], [shown[i] for i in order], **kw)
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
    leg._mylab_ncols = ncols
    leg._mylab_labels = keep_labels
    leg._mylab_raw = [labels[i] for i in order]
    leg._mylab_handles = handles if keep_labels is not None else None
    leg._mylab_reverse = bool(reverse)
    leg._mylab_interpreter = interpreter
    leg._mylab_token = token
    leg._mylab_title_raw = title
    if look is not None:
        _apply_legend_look(leg, look)
    return leg


_LEGEND_WORDS = ("off", "hide", "show", "boxoff", "boxon", "toggle")
_LEGEND_OPTS = {"location", "fontsize", "orientation", "box", "numcolumns", "textcolor", "interpreter", "title",
                "direction"}


@builtin("legend")
def _legend(interp, args, nargout):
    st = _state(interp)
    target = None
    handles = None
    hs: list = []
    args = list(args)
    if args and isinstance(args[0], GraphicsHandle) and args[0].kind in ("axes", "polaraxes", "geoaxes"):
        target = args.pop(0).obj
    if args and (isinstance(args[0], (GraphicsHandle, GraphicsArray)) or _handles_of(args[0])):
        hs = _handles_of(args.pop(0))  # (numeric handles too: h = zeros(1, 2); h(1) = plot(...))
        handles = []
        for h in hs:
            o = h.obj
            if isinstance(o, _Hist):
                o = o.container[0] if isinstance(o.container, list) else o.container
            elif isinstance(o, list):
                o = o[0] if o else None
            if o is not None:
                handles.append(o)
        if target is None and hs:
            target = _obj_axes(hs[0])
    # Name,Value options start at the first text that names a legend property and has a value after it.
    start = len(args)
    for k, a in enumerate(args):
        if is_text(a) and k + 1 < len(args):
            low = text(a).lower()
            if low in _LEGEND_OPTS or (len(low) > 2 and _is_settable("legend", low)):
                if (len(args) - k) % 2 == 0:
                    start = k
                    break
    pairs = [(text(args[i]), args[i + 1]) for i in range(start, len(args), 2)]
    args = args[:start]
    opts = {name.lower(): value for name, value in pairs if name.lower() in _LEGEND_OPTS}
    later = [(name, value) for name, value in pairs if name.lower() not in _LEGEND_OPTS]
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
    cmds = [text(a).lower() for a in args if is_text(a) and text(a).lower() in _LEGEND_WORDS]
    labels = []
    for a in args:
        if isinstance(a, (Cell, StringArray)):
            labels.extend(_text_list(a))
        elif is_text(a) and text(a).lower() not in _LEGEND_WORDS:
            labels.append(_label_text(a))
        elif not is_text(a):
            raise MatlabError("legend expects labels as text, a cell array or a string array.")
    if "off" in cmds or "hide" in cmds or ("toggle" in cmds and existing is not None and existing.get_visible()):
        if existing is not None:
            if "hide" in cmds or "toggle" in cmds:
                existing.set_visible(False)
            else:
                st.handles.pop(id(existing), None)
                existing.remove()
        return None
    if existing is not None and ("show" in cmds or "toggle" in cmds) and not labels and not pairs:
        existing.set_visible(True)
        _redraw(interp)
        return _wrap(interp, "legend", existing) if nargout > 0 else None
    if cmds and all(c in ("boxoff", "boxon") for c in cmds) and existing is not None and not labels \
            and not pairs:
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
    reverse = None
    if "direction" in opts:
        d = text(opts["direction"]).lower()
        if d not in ("normal", "reverse"):
            raise MatlabError("Direction must be 'normal' or 'reverse'.")
        reverse = d == "reverse"
    interpreter = None
    if "interpreter" in opts:
        interpreter = text(opts["interpreter"]).lower()
        if interpreter not in ("tex", "latex", "none"):
            raise MatlabError("Interpreter must be 'tex', 'latex' or 'none'.")
    leg = _make_legend(interp, ax, labels=labels or None, handles=handles,
                       location=text(opts["location"]) if "location" in opts else None,
                       orientation=text(opts["orientation"]) if "orientation" in opts else None,
                       ncols=to_int(opts["numcolumns"], "NumColumns") if "numcolumns" in opts else None,
                       reuse=existing if (not labels and handles is None and existing is not None) else None,
                       fontsize=_scalar(opts["fontsize"]) if "fontsize" in opts else None,
                       box=box, title=_label_text(opts["title"]) if "title" in opts else None, reverse=reverse,
                       interpreter=interpreter)
    leg.set_visible(True)
    if "textcolor" in opts:
        for t in leg.get_texts():
            t.set_color(_color(opts["textcolor"]))
    lh = _wrap(interp, "legend", leg)
    for name, value in later:
        _set_prop(lh, name, value)
    _redraw(interp)
    if nargout > 1:  # [lgd, icons, plots, txt] = legend(...)
        shown = _handle_list(hs) if hs else _handle_list(
            [g for g in _axes_children(interp, ax) if any(a in (leg._mylab_handles or _legend_handles(ax))
                                                          for a in ([g.obj] + _artists(g.obj)))][::-1])
        return (lh, np.zeros((0, 1)), shown,
                Cell.from_list([char(x) for x in leg._mylab_raw]))[:nargout]
    return lh if nargout > 0 else None


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


def _limit_method(interp, ax, method: str, which: str = "xyz") -> None:
    from .plot_objects import set_limit_method
    if hasattr(ax, "get_zlim"):
        ax.autoscale(enable=True, tight=method == "tight")
        return
    for w in which:
        if w in "xy":
            set_limit_method(ax, w, method)


@builtin("axis")
@_with_axes_arg
def _axis(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    args = list(args)
    words: list = []
    for a in args:
        if is_text(a):
            parts = text(a).lower().split()
            if len(parts) == 2 and parts[0] == "auto" and set(parts[1]) <= set("xyz"):
                words.append(("auto", parts[1]))
            elif words and words[-1] == ("auto", "xyz") and len(parts) == 1 and set(parts[0]) <= set("xyz") \
                    and parts[0]:
                words[-1] = ("auto", parts[0])  # command form: axis auto x
            else:
                words.extend((w, "xyz") for w in parts)
        else:
            words.append((a, None))
    three_d = hasattr(ax, "get_zlim")
    for mode, dims in words:
        if dims is None:
            lim = np.real(fnum(mode)).flatten()
            if lim.size not in (4, 6, 8):
                raise MatlabError("axis limits must be [xmin xmax ymin ymax], with zmin zmax and cmin cmax "
                                  "optionally added.")
            _set_lim(ax, "x", lim[0], lim[1])
            _set_lim(ax, "y", lim[2], lim[3])
            if lim.size >= 6 and three_d:
                _set_lim(ax, "z", lim[4], lim[5])
            elif lim.size >= 6:
                ax._mylab_zlim = (float(lim[4]), float(lim[5]))  # used once the axes turn 3-D (view(3))
            if lim.size == 8:
                for m in _mappables(ax):
                    m.set_clim(lim[6], lim[7])
        elif mode == "equal":
            try:
                ax.set_aspect("equal", adjustable="datalim")
            except (NotImplementedError, ValueError):
                ax.set_aspect("equal")
            ax._mylab_daspect = np.array([1.0, 1.0, 1.0])
        elif mode == "image":
            ax.set_aspect("equal", adjustable="box")
            _limit_method(interp, ax, "tight")
            ax._mylab_daspect = np.array([1.0, 1.0, 1.0])
        elif mode == "square":
            if three_d:
                ax.set_box_aspect((1, 1, 1))
            else:
                ax.set_box_aspect(1)
            ax._mylab_pbaspect = np.array([1.0, 1.0, 1.0])
        elif mode == "tight":
            _limit_method(interp, ax, "tight")
        elif mode == "padded":
            _limit_method(interp, ax, "padded")
        elif mode == "tickaligned":
            _limit_method(interp, ax, "tickaligned")
        elif mode == "off":
            ax.set_axis_off()
        elif mode == "on":
            ax.set_axis_on()
        elif mode in ("auto", "normal", "fill"):
            if mode == "auto":
                for w in dims:
                    f = getattr(ax, f"set_autoscale{w}_on", None)
                    if f is not None:
                        f(True)
                ax.autoscale_view()
            else:
                try:
                    ax.set_aspect("auto")
                except NotImplementedError:
                    pass
                if not three_d:
                    ax.set_box_aspect(None)
                ax._mylab_daspect = ax._mylab_pbaspect = None
                if mode == "normal":
                    ax.autoscale(True)
        elif mode == "ij":
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
        elif mode == "xy":
            if ax.yaxis_inverted():
                ax.invert_yaxis()
        elif mode == "manual":
            lims = (ax.get_xlim(), ax.get_ylim())
            ax.set_autoscale_on(False)
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        elif mode == "vis3d":  # keep the current proportions of the plot box when the view changes
            if three_d:
                ax.set_box_aspect(tuple(float(v) for v in ax.get_box_aspect()))
            ax._mylab_vis3d = True
        elif mode == "state":
            auto = ax.get_autoscalex_on() and ax.get_autoscaley_on()
            out = (char("auto" if auto else "manual"), char("on" if ax.axison else "off"),
                   char("ij" if ax.yaxis_inverted() else "xy"))
            return out[:max(nargout, 1)] if nargout > 1 else out[0]
        else:
            raise MatlabError(f"Unknown axis option '{mode}'.")
    if not words or nargout > 0:
        vals = [*sorted(ax.get_xlim()), *sorted(ax.get_ylim())]
        if three_d:
            vals += sorted(ax.get_zlim())
        if words:
            _redraw(interp)
        return np.array([vals], dtype=float)
    _redraw(interp)
    return None


def _lim(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        getter = getattr(ax, f"get_{which}lim", None)
        if not args:
            return ruler_values(ax, which, sorted(getter())) if getter else None
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
        lim = np.real(axis_numbers(ax, which, args[0])).flatten()
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
            return _ticks_get(which)(_FakeAxesHandle(ax, rulers=True))
        if is_text(args[0]) and text(args[0]).lower() in ("auto", "manual", "mode"):
            mode = text(args[0]).lower()
            if mode == "auto":
                _set_ticks_auto(ax, which)
            elif mode == "mode":
                return char("manual" if getattr(ax, f"_mylab_{which}ticks_fixed", False) else "auto")
            else:
                _set_tick_values(ax, which, np.asarray(getattr(ax, f"get_{which}ticks")(), float))
            _redraw(interp)
            return None
        _set_tick_values(ax, which, np.real(axis_numbers(ax, which, args[0])).flatten())
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
    left, opts = _split_name_values(args, _TEXT_OPTS)
    if left:
        bad = left[0]
        raise MatlabError(f"Unrecognized text property '{text(bad)}'." if is_text(bad)
                          else "text options must be name/value pairs.")
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
    interp_name = text(opts["interpreter"]).lower() if "interpreter" in opts else None
    for t in texts:
        set_text(interp, t, t.get_text(), interp_name)
    hs = [_wrap(interp, "text", t) for t in texts]
    if "units" in opts:  # the coordinates given are in these units
        from .plot_objects import set_text_units
        for h in hs:
            set_text_units(h, opts["units"], convert=False)
    if "edgecolor" in opts:
        for h in hs:
            _set_prop(h, "EdgeColor", opts["edgecolor"])
    _apply_extra(interp, hs, opts, "text")
    _redraw(interp)
    return (hs[0] if len(hs) == 1 else GraphicsArray(hs)) if nargout > 0 else None


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
    left, opts = _split_name_values(rest, {"string", "fontsize", "color", "linewidth", "edgecolor",
                                           "backgroundcolor", "fitboxtotext", "horizontalalignment",
                                           "linestyle", "facecolor"})
    if left:
        cls = {"line": "Line", "arrow": "Arrow", "doublearrow": "DoubleEndArrow", "textarrow": "TextArrow",
               "textbox": "TextBox", "rectangle": "Rectangle", "ellipse": "Ellipse"}.get(kind, kind)
        bad = text(left[0]) if is_text(left[0]) else None
        if bad is None:
            raise MatlabError("annotation options must be name/value pairs.")
        raise MatlabError(f"Unrecognized property '{bad}' for class '{cls}'.", "MATLAB:class:InvalidProperty")
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
    h = _wrap(interp, "patch", art)
    h.extra["figure"] = fig
    return (h) if nargout > 0 else None


def _annotation_arrow(fig, x, y, style, color, lw, s, fontsize):
    from matplotlib.text import Annotation
    return Annotation(s, xy=(float(x[1]), float(y[1])), xytext=(float(x[0]), float(y[0])),
                      xycoords=fig.transFigure, textcoords=fig.transFigure, fontsize=fontsize,
                      arrowprops={"arrowstyle": style, "color": color, "lw": lw})


@builtin("drawnow", "shg", "refresh")
def _drawnow(interp, args, nargout):
    # Options such as 'limitrate', 'nocallbacks', 'update' and 'expose' are accepted.
    _redraw(interp)
    _capture_frame(interp)
    return None


# -- saving -------------------------------------------------------------------------------

# print devices and saveas format names -> the kind of file that is written
_FORMATS = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "pdf": "pdf", "svg": "svg", "eps": "eps", "epsc": "eps",
            "epsc2": "eps", "eps2": "eps", "tiff": "tif", "tiffn": "tif", "tif": "tif", "bmp": "bmp",
            "bmpmono": "bmp", "bmp16m": "bmp", "bmp256": "bmp", "gif": "gif", "ps": "ps", "psc": "ps",
            "ps2": "ps", "psc2": "ps", "fig": "fig"}
_RASTER = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif")
_VECTOR = ("pdf", "eps", "svg", "ps")
_NOT_AVAILABLE = {"meta": "Windows metafiles (.emf) cannot be written", "emf": "Windows metafiles (.emf) cannot "
                  "be written", "win": "there is no printer", "winc": "there is no printer",
                  "hdf": "HDF image files cannot be written", "pcx": "PCX files cannot be written",
                  "pbm": "PBM files cannot be written", "ppm": "PPM files cannot be written"}


def _file_kind(fname: str, fmt: str | None, what: str) -> tuple[str, str]:
    """(file name with an extension, kind of file) for a save request."""
    ext = os.path.splitext(fname)[1].lower().lstrip(".")
    kind = (fmt or "").lower() or {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext) or "png"
    kind = {"jpeg": "jpg", "tiff": "tif"}.get(kind, kind)
    if kind in _NOT_AVAILABLE:
        raise MatlabError(f"{what}: {_NOT_AVAILABLE[kind]} by mylab. Use png, jpg, tif, bmp, gif, pdf, eps, "
                          "svg or ps.")
    if kind not in _RASTER + _VECTOR + ("fig",):
        raise MatlabError(f"{what}: the file type '{kind}' is not supported. Use png, jpg, tif, bmp, gif, "
                          "pdf, eps, svg, ps or fig.")
    if not ext:
        fname += "." + kind
    return fname, kind


def _paper_layout(interp, fig, page: bool, fit: str | None = None):
    """(figure size in inches to draw at, page box in inches or None) for print and saveas.

    The figure is drawn at its own size unless PaperPositionMode is 'manual'. For page formats (pdf, ps)
    the output is a sheet of PaperSize with the figure placed on it, centred when the position is automatic.
    """
    from . import plot_objects
    h = _wrap(interp, "figure", fig)
    w, hh = (float(v) for v in fig.get_size_inches())
    manual = plot_objects.paper_position_inches(h)
    pw, ph = plot_objects._paper_size_in(h)
    left, bottom = (pw - w) / 2, (ph - hh) / 2
    if manual is not None:
        left, bottom, w, hh = manual
    if fit == "fillpage":
        left, bottom, w, hh = 0.25, 0.25, pw - 0.5, ph - 0.5
    elif fit == "bestfit":
        k = min((pw - 0.5) / w, (ph - 0.5) / hh)
        w, hh = w * k, hh * k
        left, bottom = (pw - w) / 2, (ph - hh) / 2
    if w <= 0 or hh <= 0:
        raise MatlabError("The paper is too small for the figure; change PaperSize or PaperPosition.")
    box = (-left, -bottom, pw, ph) if page else None
    return (w, hh), box


def _write_figure(fig, fname: str, kind: str, dpi: float, size=None, box=None, tight: bool = False,
                  facecolor=None, transparent: bool = False, gray: bool = False, clip=None) -> None:
    """Write a matplotlib figure to a file of the given kind."""
    from matplotlib.transforms import Bbox
    old_size = tuple(float(v) for v in fig.get_size_inches())
    kw: dict = {"dpi": dpi}
    if facecolor is not None:
        kw["facecolor"] = facecolor
    if transparent:
        kw["transparent"] = True
    if clip is not None:
        kw["bbox_inches"] = clip
    elif box is not None:
        kw["bbox_inches"] = Bbox.from_bounds(*box)
    elif tight:
        kw["bbox_inches"] = "tight"
        kw["pad_inches"] = 0.05
    try:
        if size is not None and tuple(size) != old_size:
            fig.set_size_inches(*size)
        if kind in ("bmp", "gif") or (gray and kind in _RASTER):
            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", **kw)
            buf.seek(0)
            im = Image.open(buf)
            if kind in ("bmp", "gif", "jpg") or gray:
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[3])
                im = bg
            if gray:
                im = im.convert("L")
            im.save(fname, format={"jpg": "JPEG", "tif": "TIFF"}.get(kind, kind.upper()), dpi=(dpi, dpi))
        else:
            if gray:
                raise MatlabError("Colorspace 'gray' applies to image files (png, jpg, tif, bmp, gif).")
            fig.savefig(fname, format={"jpg": "jpeg", "tif": "tiff"}.get(kind, kind), **kw)
    except ValueError as e:
        raise MatlabError(f"Cannot save the figure as '{fname}': {e}") from e
    except OSError as e:
        raise MatlabError(f"Cannot write '{fname}': {e.strerror or e}") from e
    finally:
        if tuple(float(v) for v in fig.get_size_inches()) != old_size:
            fig.set_size_inches(*old_size)


def _save(interp, n, fname: str, dpi: float, fmt: str | None = None, what: str = "saveas",
          fit: str | None = None) -> None:
    """Save figure ``n`` whole, the way print and saveas do (no cropping)."""
    st = _state(interp)
    if n is None or n not in st.figures:
        raise MatlabError("No figure to save.")
    fig = st.figures[n]
    fname, kind = _file_kind(fname, fmt, what)
    if kind == "fig":
        from . import figfiles
        figfiles.save_fig(interp, [_wrap(interp, "figure", fig)], fname)
        return
    size, box = _paper_layout(interp, fig, page=kind in ("pdf", "ps"), fit=fit)
    _write_figure(fig, fname, kind, dpi, size=size, box=box)


@builtin("saveas")
def _saveas(interp, args, nargout):
    st = _state(interp)
    if len(args) >= 2 and not is_text(args[0]):
        n = _fig_num_arg(interp, args[0])
        fname = text(args[1])
        rest = args[2:]
    else:
        n = st.current
        fname = text(args[0]) if args else ""
        rest = args[1:]
    if not fname:
        raise MatlabError("saveas needs a file name.")
    fmt = None
    if rest:
        f = text(rest[0]).lower().lstrip("-")
        if f.startswith("d") and f[1:] in _FORMATS:
            f = f[1:]
        if f in ("m", "mfig", "mmat"):
            raise MatlabError("saveas: generating MATLAB code for a figure is not supported. Save a .fig file "
                              "with savefig instead.")
        if f not in _FORMATS and f not in _NOT_AVAILABLE:
            raise MatlabError(f"saveas: unknown format '{text(rest[0])}'.")
        fmt = _FORMATS.get(f, f)
    _save(interp, n, fname, 150, fmt, "saveas")
    return None


@builtin("print")
def _print(interp, args, nargout):
    st = _state(interp)
    fmt = None
    fname = None
    dpi = 150.0
    n = st.current
    fit = None
    rgb = False
    for a in args:
        if isinstance(a, (GraphicsHandle, GraphicsArray)):
            n = _fig_num_arg(interp, a)
        elif is_text(a):
            s = text(a)
            low = s.lower()
            if low == "-rgbimage":
                rgb = True
            elif low in ("-clipboard",) or low.startswith("-dclipboard"):
                raise MatlabError("print: there is no clipboard to copy the figure to; save it to a file.")
            elif low.startswith("-d"):
                f = low[2:]
                if f in _NOT_AVAILABLE:
                    raise MatlabError(f"print: {_NOT_AVAILABLE[f]} by mylab.")
                if f not in _FORMATS:
                    raise MatlabError(f"print: unknown device '{s}'. Use for example -dpng, -djpeg, -dtiff, "
                                      "-dbmp, -dpdf, -depsc, -dsvg or -dps.")
                fmt = _FORMATS[f]
            elif low.startswith("-r"):
                try:
                    r = float(s[2:]) if s[2:] else 150.0
                except ValueError:
                    raise MatlabError(f"Invalid resolution option '{s}'.") from None
                if r < 0:
                    raise MatlabError("print: the resolution must not be negative.")
                dpi = r if r > 0 else FIG_DPI  # -r0: the resolution of the screen
            elif low.startswith("-f") and s[2:].isdigit():
                n = int(s[2:])
            elif low in ("-bestfit", "-fillpage"):
                fit = low[1:]
            elif low in ("-painters", "-opengl", "-vector", "-image", "-zbuffer", "-noui", "-loose", "-tight",
                         "-cmyk", "-rgb", "-append", "-tiff"):
                if low == "-append":
                    raise MatlabError("print: -append (adding pages to a PostScript file) is not supported. "
                                      "Use exportgraphics(..., 'Append', true) with a PDF file.")
                if low == "-cmyk":
                    raise MatlabError("print: CMYK output is not supported.")
                # -painters/-opengl/-vector/-image: vector files are always vector, images always images.
                # -noui: figures have no user-interface controls to leave out. -loose/-tight: the bounding
                # box of an EPS file is the figure in both cases. -tiff (a preview image inside an EPS file):
                # the file is written without the preview.
            elif low.startswith("-p") or low.startswith("-v") or low.startswith("-s"):
                raise MatlabError(f"print: option '{s}' needs a printer or a Simulink model, which mylab does "
                                  "not have. Print to a file instead.")
            elif s.startswith("-"):
                raise MatlabError(f"print: unknown option '{s}'.")
            else:
                fname = s
        elif isinstance(a, np.ndarray) and a.size == 1:
            n = to_int(a, "figure")
    if n is None or n not in st.figures:
        raise MatlabError("No figure to print.")
    if rgb:
        size, _ = _paper_layout(interp, st.figures[n], page=False)
        fig = st.figures[n]
        old = tuple(float(v) for v in fig.get_size_inches())
        try:
            if tuple(size) != old:
                fig.set_size_inches(*size)
            return _render_rgb(fig, dpi)
        finally:
            if tuple(float(v) for v in fig.get_size_inches()) != old:
                fig.set_size_inches(*old)
    if fname is None:
        raise MatlabError("print: there is no printer; give a file name, e.g. print('-dpng', 'figure.png').")
    _save(interp, n, fname, dpi, fmt, "print", fit)
    return None


def _export_target(interp, v):
    """(figure, axes list or None) for the object given to exportgraphics."""
    hs = _resolve(interp, v)
    if not hs:
        raise MatlabError("exportgraphics needs a figure, an axes, a tiled layout or a chart.")
    h = hs[0]
    if not _alive(h):
        raise MatlabError("Invalid or deleted object.")
    if h.kind == "figure":
        return h.obj, None
    if h.kind == "tiledlayout":
        return h.obj.fig, [a for a in h.obj.order if a in h.obj.fig.axes]
    if h.kind in ("axes", "polaraxes", "geoaxes"):
        return h.obj.figure, [h.obj]
    ax = h.extra.get("_child_of") or _obj_axes(h)
    if ax is None:
        raise MatlabError("exportgraphics needs a figure, an axes, a tiled layout or a chart.")
    return ax.figure, [ax]


def _axes_clip(fig, axes: list):
    """Bounding box in inches of the given axes with their labels, titles, legends and colorbars."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in axes:
        bb = ax.get_tightbbox(r)
        if bb is not None:
            boxes.append(bb)
        cb = getattr(ax, "_mylab_colorbar", None)
        if cb is not None:
            cbb = cb.ax.get_tightbbox(r)
            if cbb is not None:
                boxes.append(cbb)
        right = getattr(ax, "_mylab_yyright", None)
        if right is not None and right in fig.axes:
            rb = right.get_tightbbox(r)
            if rb is not None:
                boxes.append(rb)
    if not boxes:
        return None
    return Bbox.union(boxes).padded(3).transformed(fig.dpi_scale_trans.inverted())


@builtin("exportgraphics")
def _exportgraphics(interp, args, nargout):
    args, opts = _split_name_values(args, {"resolution", "contenttype", "backgroundcolor", "append",
                                           "colorspace", "padding", "width", "height", "units",
                                           "preserveaspectratio"})
    st = _state(interp)
    if len(args) >= 2 and not is_text(args[0]):
        fig, axes = _export_target(interp, args[0])
        fname = text(args[1])
        extra = args[2:]
    elif len(args) == 1 and is_text(args[0]):
        if st.current is None or st.current not in st.figures:
            raise MatlabError("No figure to export.")
        fig, axes = st.figures[st.current], None
        fname = text(args[0])
        extra = []
    else:
        raise MatlabError("exportgraphics needs an object and a file name, e.g. exportgraphics(gcf, 'f.png').")
    if extra:
        raise MatlabError("exportgraphics options must be name/value pairs (Resolution, ContentType, "
                          "BackgroundColor, Append, Colorspace).")
    ext = os.path.splitext(fname)[1].lower().lstrip(".")
    if not ext:
        raise MatlabError("exportgraphics: the file name needs an extension such as .png, .jpg, .tif, .gif, "
                          ".pdf, .eps or .svg.")
    fname, kind = _file_kind(fname, None, "exportgraphics")
    if kind in ("fig", "bmp", "ps"):
        raise MatlabError(f"exportgraphics does not write .{kind} files. Use saveas, print or savefig.")
    dpi = _scalar(opts["resolution"]) if "resolution" in opts else 150.0
    if dpi <= 0:
        raise MatlabError("Resolution must be a positive number.")
    content = text(opts["contenttype"]).lower() if "contenttype" in opts else "auto"
    if content not in ("auto", "vector", "image"):
        raise MatlabError("ContentType must be 'auto', 'vector' or 'image'.")
    if content == "vector" and kind in _RASTER:
        raise MatlabError(f"A .{kind} file cannot hold vector content. Use a .pdf, .eps or .svg file.")
    for key in ("width", "height", "units", "preserveaspectratio", "padding"):
        if key in opts:
            raise MatlabError(f"exportgraphics: the option '{key.capitalize()}' is not supported.")
    space = text(opts["colorspace"]).lower() if "colorspace" in opts else "rgb"
    if space not in ("rgb", "gray", "cmyk"):
        raise MatlabError("Colorspace must be 'rgb', 'gray' or 'cmyk'.")
    if space == "cmyk":
        raise MatlabError("exportgraphics: the CMYK colour space is not supported.")
    face, transparent = "white", False
    if "backgroundcolor" in opts:
        b = opts["backgroundcolor"]
        if is_text(b) and text(b).lower() == "none":
            transparent, face = True, None
            if kind in ("jpg", "bmp", "gif", "tif"):
                raise MatlabError(f"A .{kind} file cannot have a transparent background; use .png or .pdf.")
        elif is_text(b) and text(b).lower() == "current":
            face = fig.get_facecolor()
        else:
            face = _color(b)
    append = "append" in opts and _onoff(opts["append"])
    if append and kind not in ("pdf", "gif"):
        raise MatlabError("Append works with .pdf and .gif files.")
    clip = _axes_clip(fig, axes) if axes else None
    raster_in_vector = content == "image" and kind in _VECTOR
    kw = {"tight": clip is None, "facecolor": face, "transparent": transparent, "clip": clip,
          "gray": space == "gray"}
    if kind == "gif":
        from PIL import Image, ImageSequence
        buf = io.BytesIO()
        _write_figure(fig, buf, "png", dpi, **{**kw, "gray": False})
        buf.seek(0)
        new = Image.open(buf).convert("RGB")
        if space == "gray":
            new = new.convert("L").convert("RGB")
        frames = []
        if append and os.path.exists(fname):
            with Image.open(fname) as old:
                frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(old)]
        if frames and frames[0].size != new.size:
            new = new.resize(frames[0].size)
        frames.append(new)
        try:
            frames[0].save(fname, format="GIF", save_all=len(frames) > 1, append_images=frames[1:],
                           duration=500, loop=0)
        except OSError as e:
            raise MatlabError(f"Cannot write '{fname}': {e.strerror or e}") from e
        return None
    if kind == "pdf":
        _export_pdf(interp, fig, fname, dpi, kw, append, raster_in_vector)
        return None
    if raster_in_vector:
        raise MatlabError(f"exportgraphics: image content in a .{kind} file is not supported; use .pdf, or an "
                          "image file such as .png.")
    _write_figure(fig, fname, kind, dpi, **kw)
    return None


def _export_pdf(interp, fig, fname: str, dpi: float, kw: dict, append: bool, as_image: bool) -> None:
    """Write a figure as a one-page PDF file, or add it as a new page to an existing PDF file."""
    plt = _plt()
    buf = io.BytesIO()
    if as_image:
        from PIL import Image
        png = io.BytesIO()
        _write_figure(fig, png, "png", dpi, **{**kw, "gray": False})
        png.seek(0)
        im = Image.open(png).convert("L" if kw.get("gray") else "RGB")
        tmp = plt.figure(figsize=(im.size[0] / dpi, im.size[1] / dpi), dpi=dpi)
        try:
            ax = tmp.add_axes([0, 0, 1, 1])
            ax.imshow(np.asarray(im), cmap="gray" if kw.get("gray") else None, interpolation="nearest",
                      vmin=0, vmax=255)
            ax.set_axis_off()
            tmp.savefig(buf, format="pdf", dpi=dpi)
        finally:
            plt.close(tmp)
    else:
        if kw.get("gray"):
            raise MatlabError("Colorspace 'gray' needs ContentType 'image' for a PDF file.")
        _write_figure(fig, buf, "pdf", dpi, **{k: v for k, v in kw.items() if k != "gray"})
    data = buf.getvalue()
    try:
        if append and os.path.exists(fname) and os.path.getsize(fname) > 0:
            with open(fname, "rb") as f:
                old = f.read()
            try:
                data = merge_pdfs(old, data)
            except ValueError as e:
                data = _merge_with_pypdf(old, data, fname, str(e))
        with open(fname, "wb") as f:
            f.write(data)
    except OSError as e:
        raise MatlabError(f"Cannot write '{fname}': {e.strerror or e}") from e


def _merge_with_pypdf(old: bytes, new: bytes, fname: str, why: str) -> bytes:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise MatlabError(f"exportgraphics cannot append to '{fname}': {why}. Files written by mylab can "
                          "always be appended to; for other PDF files install the Python package "
                          "pypdf.") from None
    writer = PdfWriter()
    for src in (old, new):
        for pg in PdfReader(io.BytesIO(src)).pages:
            writer.add_page(pg)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _pdf_objects(data: bytes) -> tuple[dict, int, int | None]:
    """({number: body bytes}, number of the catalog, number of the info object) of a simple PDF file."""
    m = list(re.finditer(rb"startxref\s+(\d+)", data))
    if not m:
        raise ValueError("it has no cross-reference table")
    pos = int(m[-1].group(1))
    if data[pos:pos + 4] != b"xref":
        raise ValueError("it uses compressed cross-reference streams")
    offsets: dict[int, int] = {}
    lines = data[pos:].split(b"\n")
    k = 1
    while k < len(lines):
        head = lines[k].split()
        if len(head) != 2 or not head[0].isdigit():
            break
        first, count = int(head[0]), int(head[1])
        for i in range(count):
            entry = lines[k + 1 + i].split()
            if len(entry) >= 3 and entry[2] == b"n":
                offsets[first + i] = int(entry[0])
        k += 1 + count
    t = data.find(b"trailer", pos)
    trailer = data[t:data.find(b"startxref", t)] if t >= 0 else b""
    root = re.search(rb"/Root\s+(\d+)\s+0\s+R", trailer)
    if root is None:
        raise ValueError("its trailer has no /Root entry")
    if b"/Encrypt" in trailer:
        raise ValueError("it is encrypted")
    info = re.search(rb"/Info\s+(\d+)\s+0\s+R", trailer)
    order = sorted(offsets.items(), key=lambda kv: kv[1])
    objs: dict[int, bytes] = {}
    for idx, (number, off) in enumerate(order):
        end = order[idx + 1][1] if idx + 1 < len(order) else pos
        chunk = data[off:end]
        head = re.match(rb"\s*(\d+)\s+(\d+)\s+obj", chunk)
        if head is None or int(head.group(1)) != number:
            raise ValueError("its cross-reference table does not match its objects")
        body = chunk[head.end():]
        cut = body.rfind(b"endobj")
        objs[number] = body[:cut] if cut >= 0 else body
    return objs, int(root.group(1)), int(info.group(1)) if info else None


def merge_pdfs(first: bytes, second: bytes) -> bytes:
    """One PDF file with the pages of ``first`` followed by the pages of ``second``.

    Works for files with a plain cross-reference table (what matplotlib and most simple writers produce);
    raises ValueError for files it cannot take apart.
    """
    ref = re.compile(rb"(\d+)(\s+0\s+R)")
    out_objs: list[tuple[int, bytes]] = []
    kids: list[bytes] = []
    count = 0
    next_free = 3  # 1 = catalog, 2 = page tree root
    info_ref = None
    for data in (first, second):
        objs, root, info = _pdf_objects(data)
        shift = next_free - 1
        pages = re.search(rb"/Pages\s+(\d+)\s+0\s+R", objs[root])
        if pages is None:
            raise ValueError("its catalog has no /Pages entry")
        pages_num = int(pages.group(1))
        tree = objs[pages_num]
        kid_list = re.search(rb"/Kids\s*\[(.*?)\]", tree, re.S)
        n_pages = re.search(rb"/Count\s+(\d+)", tree)
        if kid_list is None or n_pages is None:
            raise ValueError("its page tree cannot be read")
        count += int(n_pages.group(1))
        for kid in re.findall(rb"(\d+)\s+0\s+R", kid_list.group(1)):
            kids.append(b"%d 0 R" % (int(kid) + shift))
        for number, body in objs.items():
            if number in (root, pages_num):
                continue
            cut = re.search(rb"stream\r?\n", body)
            head, rest = (body[:cut.start()], body[cut.start():]) if cut else (body, b"")
            head = ref.sub(lambda mm: b"%d" % (int(mm.group(1)) + shift) + mm.group(2), head)
            # objects that pointed at their file's page tree root now hang off the new root
            head = head.replace(b"/Parent %d 0 R" % (pages_num + shift), b"/Parent 2 0 R")
            out_objs.append((number + shift, head + rest))
        if info is not None and info_ref is None:
            info_ref = info + shift
        next_free += max(objs) if objs else 0
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    table: dict[int, int] = {}

    def emit(number: int, body: bytes) -> None:
        table[number] = out.tell()
        out.write(b"%d 0 obj" % number + (b"" if body[:1] in (b"\n", b"\r", b" ") else b"\n") + body
                  + (b"" if body.endswith(b"\n") else b"\n") + b"endobj\n")

    emit(1, b"\n<< /Type /Catalog /Pages 2 0 R >>\n")
    emit(2, b"\n<< /Type /Pages /Kids [ " + b" ".join(kids) + b" ] /Count %d >>\n" % count)
    for number, body in out_objs:
        emit(number, body)
    size = max(table) + 1
    start = out.tell()
    out.write(b"xref\n0 %d\n" % size)
    out.write(b"0000000000 65535 f \n")
    for number in range(1, size):
        if number in table:
            out.write(b"%010d 00000 n \n" % table[number])
        else:
            out.write(b"0000000000 00000 f \n")
    out.write(b"trailer\n<< /Size %d /Root 1 0 R" % size
              + (b" /Info %d 0 R" % info_ref if info_ref is not None else b"") + b" >>\nstartxref\n%d\n%%%%EOF\n"
              % start)
    return out.getvalue()


@builtin("hgexport")
def _hgexport(interp, args, nargout):
    """hgexport(fig, filename[, style], 'Format', fmt): the older way to save a figure to a file."""
    if len(args) == 1 and is_text(args[0]) and text(args[0]).lower() in ("factorystyle", "readstyle"):
        return Struct.scalar({"Version": char("1"), "Format": char("eps"), "Preview": char("none"),
                              "Width": char("auto"), "Height": char("auto"), "Units": char("inches"),
                              "Color": char("rgb"), "Background": char("w"), "Resolution": char("auto"),
                              "Renderer": char("auto"), "Bounds": char("loose")})
    if len(args) < 2:
        raise MatlabError("hgexport needs a figure and a file name.")
    n = _fig_num_arg(interp, args[0])
    fname = text(args[1])
    rest = list(args[2:])
    style = None
    if rest and isinstance(rest[0], Struct):
        style = rest.pop(0)
    fmt, dpi = "eps", 150.0
    if style is not None:
        if "Format" in style.fields:
            fmt = to_str(style.get_field(0, "Format")).lower()
        if "Resolution" in style.fields:
            r = style.get_field(0, "Resolution")
            if not is_text(r):
                dpi = _scalar(r)
            elif to_str(r).isdigit():
                dpi = float(to_str(r))
    if len(rest) % 2:
        raise MatlabError("hgexport options must be name/value pairs.")
    for i in range(0, len(rest), 2):
        name = text(rest[i]).lower()
        if name == "format":
            fmt = text(rest[i + 1]).lower()
        elif name == "resolution":
            dpi = _scalar(rest[i + 1])
        else:
            raise MatlabError(f"hgexport: the option '{text(rest[i])}' is not supported.")
    if fmt not in _FORMATS:
        raise MatlabError(f"hgexport: unknown format '{fmt}'.")
    _save(interp, n, fname, dpi, _FORMATS[fmt], "hgexport")
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


# -- rendering figures to pixels ----------------------------------------------------------------

_NONINTERACTIVE = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


def _noninteractive() -> bool:
    """True when matplotlib draws off-screen (browser, Jupyter, MCP, ``--no-plots``)."""
    import matplotlib
    b = str(matplotlib.get_backend()).lower()
    return b in _NONINTERACTIVE or "inline" in b


def _render_rgb(fig, dpi: float | None = None) -> np.ndarray:
    """The figure as an H x W x 3 uint8 image."""
    dpi = float(dpi or _fig_dpi(fig))
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=dpi)
    raw = buf.getvalue()
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    total = len(raw) // 4
    rgba = None
    for w in (int(w_in * dpi), int(round(w_in * dpi)), int(w_in * dpi) + 1):
        if w > 0 and total % w == 0 and abs(total // w - h_in * dpi) <= 2:
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape(total // w, w, 4)
            break
    if rgba is None:  # unusual canvas size: go through PNG, which records its own shape
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        rgba = np.asarray(Image.open(buf).convert("RGBA"))
    alpha = rgba[..., 3:4]
    if alpha.min() < 255:  # transparent figure background: show it on white
        a = alpha.astype(np.float32) / 255.0
        return (rgba[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def _resize_rgb(a: np.ndarray, shape: tuple) -> np.ndarray:
    if tuple(a.shape[:2]) == tuple(shape):
        return a
    from PIL import Image
    return np.asarray(Image.fromarray(a, "RGB").resize((int(shape[1]), int(shape[0]))))


def _write_gif(target, frames, durations_ms, loop: int = 0) -> None:
    """Write RGB frames (an iterable of equal-sized uint8 arrays) as an animated GIF."""
    from PIL import Image
    fast = getattr(getattr(Image, "Quantize", Image), "FASTOCTREE", 2)
    none = getattr(getattr(Image, "Dither", Image), "NONE", 0)

    def palettised():
        for a in frames:
            yield Image.fromarray(np.ascontiguousarray(a), "RGB").quantize(colors=256, method=fast, dither=none)

    it = palettised()
    first = next(it)
    if isinstance(durations_ms, (list, tuple)):
        durations_ms = [max(20, int(round(d / 10.0)) * 10) for d in durations_ms]
    else:
        durations_ms = max(20, int(round(durations_ms / 10.0)) * 10)
    first.save(target, format="GIF", save_all=True, append_images=it, duration=durations_ms, loop=loop,
               disposal=1, optimize=False)


# -- animation capture for front ends without a live window --------------------------------------
#
# In a non-interactive backend every drawnow / pause / getframe stores a snapshot of the current
# figure in interp.state["animation"]. Session.run turns the snapshots of each figure into an
# animated GIF (RunResult.animations). Set interp.state["animation_capture"] = False to turn it off.

ANIM_MAX_FRAMES = 150
ANIM_MAX_PIXELS = 60_000_000
ANIM_MAX_WIDTH = 720
ANIM_FRAME_SECONDS = 1.0 / 12.0
ANIM_MAX_SECONDS = 20.0
ANIM_RENDER_BUDGET = 5.0  # seconds spent drawing snapshots before they become rapidly sparser
PAUSE_CAP_SECONDS = 0.05
PAUSE_BUDGET_SECONDS = 2.0  # total real waiting per capture; later pauses return at once


class _AnimCapture:
    def __init__(self):
        self.frames: dict[int, list[list]] = {}  # figure number -> [compressed RGB, (h, w), seconds, digest]
        self.count = 0
        self.pixels = 0
        self.stride = 1  # keep one snapshot out of every `stride` requests
        self.calls = 0
        self.slept = 0.0
        self.spent = 0.0

    def thin(self) -> None:
        """Drop every other frame so that a long animation still covers the whole run."""
        self.count = self.pixels = 0
        for n, fr in self.frames.items():
            kept: list[list] = []
            for i, f in enumerate(fr):
                if i % 2 == 0 or i == len(fr) - 1:
                    kept.append(f)
                else:
                    kept[-1][2] += f[2]
            self.frames[n] = kept
            self.count += len(kept)
            self.pixels += sum(f[1][0] * f[1][1] for f in kept)
        self.stride *= 2


def _capture_frame(interp, seconds: float = 0.0) -> bool:
    """Snapshot the current figure; returns True when animation capture is active."""
    st = interp.state.get("plot")
    if st is None or not st.figures or st.current not in st.figures:
        return False
    if interp.state.get("animation_capture") is False or not _noninteractive():
        return False
    cap = interp.state.get("animation")
    if cap is None:
        cap = interp.state["animation"] = _AnimCapture()
    n = st.current
    frames = cap.frames.setdefault(n, [])
    cap.calls += 1
    if frames and cap.calls % cap.stride:
        frames[-1][2] += seconds
        return True
    t0 = time.perf_counter()
    _snapshot(cap, n, st.figures[n], seconds)
    cap.spent += time.perf_counter() - t0
    if cap.count > ANIM_MAX_FRAMES or cap.pixels > ANIM_MAX_PIXELS:
        cap.thin()
    elif cap.spent > ANIM_RENDER_BUDGET:
        cap.stride *= 2  # a long loop: keep following it, but with ever fewer snapshots
    return True


def _snapshot(cap: _AnimCapture, n: int, fig, seconds: float = 0.0) -> None:
    frames = cap.frames.setdefault(n, [])
    try:
        rgb = _render_rgb(fig, min(_fig_dpi(fig), ANIM_MAX_WIDTH / max(float(fig.get_size_inches()[0]), 0.1)))
    except Exception:  # a figure that cannot be drawn must not stop the script here
        return
    raw = rgb.tobytes()
    digest = zlib.adler32(raw)
    if frames and frames[-1][3] == digest and tuple(frames[-1][1]) == rgb.shape[:2]:
        frames[-1][2] += seconds
        return
    frames.append([zlib.compress(raw, 1), rgb.shape[:2], float(seconds), digest])
    cap.count += 1
    cap.pixels += rgb.shape[0] * rgb.shape[1]


def clear_animation(interp) -> None:
    """Forget captured animation frames (Session.run calls this before each run)."""
    interp.state.pop("animation", None)


def animation_frame_counts(interp) -> dict:
    """Number of frames captured so far, by figure number."""
    cap = interp.state.get("animation")
    return {n: len(f) for n, f in cap.frames.items()} if cap else {}


def render_animations(interp, clear: bool = True) -> list:
    """One animated GIF (bytes) per figure that captured at least two different frames."""
    cap = interp.state.get("animation")
    if clear:
        interp.state.pop("animation", None)
    if cap is None:
        return []
    st = interp.state.get("plot")
    out = []
    for n in sorted(cap.frames):
        frames = cap.frames[n]
        if len(frames) < 2:
            continue
        if cap.stride > 1 and st is not None and n in st.figures:
            _snapshot(cap, n, st.figures[n])  # snapshots were skipped: end on the figure as it is now
        shape = frames[0][1]
        secs = [min(f[2], 2.0) if f[2] > 0 else ANIM_FRAME_SECONDS for f in frames]
        total = sum(secs)
        if total > ANIM_MAX_SECONDS:
            secs = [s * ANIM_MAX_SECONDS / total for s in secs]
        secs[-1] = max(secs[-1], 0.8)  # rest on the final picture before looping

        def arrays(frames=frames, shape=shape):
            for z, shp, _, _ in frames:
                a = np.frombuffer(zlib.decompress(z), dtype=np.uint8).reshape(shp[0], shp[1], 3)
                yield _resize_rgb(a, shape)

        buf = io.BytesIO()
        try:
            _write_gif(buf, arrays(), [s * 1000.0 for s in secs])
        except ImportError:  # Pillow is missing: no animations
            return []
        out.append(buf.getvalue())
    return out


@builtin("pause")
def _pause(interp, args, nargout):
    secs = 0.0
    if args and not is_text(args[0]):
        secs = float(num(args[0]).flat[0])
        if not (np.isfinite(secs) and secs > 0):
            secs = 0.0
    st = interp.state.get("plot")
    if st is None or not st.figures:  # nothing on screen: a plain wait
        if secs:
            time.sleep(min(secs, 3600))
        return None
    if args and is_text(args[0]):  # pause on / pause off
        return None
    if _capture_frame(interp, secs):
        cap = interp.state.get("animation")
        if secs and cap is not None and cap.slept < PAUSE_BUDGET_SECONDS and sys.platform != "emscripten":
            nap = min(secs, PAUSE_CAP_SECONDS)  # nobody is watching a live window
            cap.slept += nap
            time.sleep(nap)
        return None
    plt = _plt()
    if secs and not _noninteractive() and plt.isinteractive():
        plt.pause(min(secs, 3600))  # keeps the window responsive
    elif secs:
        time.sleep(min(secs, 3600))
    return None


# -- frames and movies ---------------------------------------------------------------------------

def _image_rgb(v, cmap=None) -> np.ndarray:
    """A MATLAB image value (uint8/uint16/double/logical, gray, RGB or indexed) as H x W x 3 uint8."""
    a = np.asarray(v)
    if a.dtype == object or a.dtype.kind in "USc" or a.ndim not in (2, 3):
        raise MatlabError("Expected an image: an H-by-W or H-by-W-by-3 numeric array.")
    if cmap is not None and np.asarray(cmap).size:
        m = np.real(fnum(cmap))
        if m.ndim != 2 or m.shape[1] != 3:
            raise MatlabError("A colormap must be an N-by-3 matrix.")
        idx = a.astype(np.int64) if a.dtype.kind in "ub" else np.real(a).astype(np.int64) - 1
        a = m[np.clip(idx.reshape(idx.shape[:2]), 0, m.shape[0] - 1)]
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 3 and a.shape[2] != 3:
        raise MatlabError("Expected an image: an H-by-W or H-by-W-by-3 numeric array.")
    if a.dtype == np.uint8:
        out = a
    elif a.dtype == bool:
        out = a.astype(np.uint8) * 255
    elif a.dtype == np.uint16:
        out = (a >> 8).astype(np.uint8)
    elif a.dtype.kind in "iu":
        out = np.clip(a, 0, 255).astype(np.uint8)
    else:
        out = (np.clip(np.nan_to_num(np.real(a)), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    return np.ascontiguousarray(out)


def _frame_struct(cdata: np.ndarray, cmap=None) -> Struct:
    return Struct.scalar({"cdata": cdata, "colormap": np.zeros((0, 0)) if cmap is None else cmap})


def _frames_of(v) -> list:
    """RGB frames from a frame struct (array), an image, or an H x W x C x N stack."""
    if isinstance(v, Struct):
        if "cdata" not in v.fields:
            raise MatlabError("A movie frame must be a struct with the fields cdata and colormap.")
        out = []
        for i in range(v.numel()):
            cd = v.get_field(i, "cdata")
            if isinstance(cd, np.ndarray) and cd.size:
                out.append(_image_rgb(cd, v.get_field(i, "colormap") if "colormap" in v.fields else None))
        return out
    if isinstance(v, np.ndarray) and v.ndim == 4:
        return [_image_rgb(v[:, :, :, k]) for k in range(v.shape[3])]
    return [_image_rgb(v)]


@builtin("getframe")
def _getframe(interp, args, nargout):
    st = _state(interp)
    target = args[0] if args else None
    fig, ax = None, None
    if isinstance(target, GraphicsArray) and target.items:
        target = target.items[0]
    if isinstance(target, GraphicsHandle):
        if not _alive(target):
            raise MatlabError("Invalid or deleted object.")
        if target.kind == "figure":
            fig = target.obj
        elif target.kind in ("axes", "polaraxes"):
            ax = target.obj
        else:
            ax = _obj_axes(target)
            if ax is None:
                raise MatlabError("getframe expects a figure or axes handle.")
    elif target is not None:
        hs = _resolve(interp, target)
        fig = hs[0].obj if hs and hs[0].kind == "figure" else None
        if fig is None:
            raise MatlabError("getframe expects a figure or axes handle.")
    else:
        ax = _axes(interp, fresh=False)
    whole = fig is not None
    if fig is None:
        fig = ax.figure
    n = _fig_number(interp, fig)
    if n is not None and st.current == n:
        _redraw(interp)
        _capture_frame(interp)
    img = _render_rgb(fig)
    H, W = img.shape[:2]
    if not whole:
        b = ax.get_position()
        r0, r1 = int(round(H * (1 - b.y1))), int(round(H * (1 - b.y0)))
        c0, c1 = int(round(W * b.x0)), int(round(W * b.x1))
        r0, c0 = max(r0, 0), max(c0, 0)
        if r1 > r0 and c1 > c0:
            img = img[r0:r1, c0:c1]
    if len(args) > 1 and isinstance(args[1], np.ndarray) and args[1].size == 4:  # [left bottom width height]
        left, bottom, w, h = (int(round(float(x))) for x in np.real(fnum(args[1])).flatten())
        hh = img.shape[0]
        sub = img[max(hh - bottom - h, 0):max(hh - bottom, 0), max(left, 0):max(left + w, 0)]
        if sub.size:
            img = sub
    return _frame_struct(np.ascontiguousarray(img))


@builtin("frame2im")
def _frame2im(interp, args, nargout):
    if not args or not isinstance(args[0], Struct) or "cdata" not in args[0].fields:
        raise MatlabError("frame2im expects a movie frame (a struct with the fields cdata and colormap).")
    f = args[0]
    cdata = f.get_field(0, "cdata")
    cmap = f.get_field(0, "colormap") if "colormap" in f.fields else np.zeros((0, 0))
    return (cdata, cmap) if nargout > 1 else cdata


@builtin("im2frame")
def _im2frame(interp, args, nargout):
    if not args:
        raise MatlabError("im2frame needs an image.")
    x = args[0]
    if not isinstance(x, np.ndarray) or is_char(x):
        raise MatlabError("im2frame expects an image array.")
    if len(args) > 1:
        return _frame_struct(x, args[1])
    if x.ndim == 3 and x.dtype != np.uint8:
        x = _image_rgb(x)
    return _frame_struct(x)


@builtin("movie")
def _movie(interp, args, nargout):
    rest = [a for a in args if not isinstance(a, (GraphicsHandle, GraphicsArray))]
    if not rest or not isinstance(rest[0], Struct):
        raise MatlabError("movie expects an array of frames made by getframe or im2frame.")
    frames = _frames_of(rest[0])
    if not frames or _noninteractive():
        return None  # off-screen: the frames were already recorded while they were drawn
    times = int(abs(_scalar(rest[1]))) if len(rest) > 1 and np.asarray(rest[1]).size == 1 else 1
    fps = _scalar(rest[2]) if len(rest) > 2 else 12.0
    plt = _plt()
    ax = _axes(interp)
    ax.set_axis_off()
    im = ax.imshow(frames[0], aspect="equal")
    for _ in range(max(times, 1)):
        for f in frames:
            im.set_data(f)
            plt.pause(1.0 / max(fps, 0.1))
    return None


def _find_ffmpeg() -> str | None:
    """Path of an ffmpeg program, or None."""
    import shutil
    try:
        import matplotlib
        exe = shutil.which(matplotlib.rcParams.get("animation.ffmpeg_path", "ffmpeg") or "ffmpeg")
    except Exception:
        exe = None
    exe = exe or shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    return exe


_VIDEO_PROFILES = {"motion jpeg avi": ".avi", "mpeg-4": ".mp4", "uncompressed avi": ".avi", "archival": ".mj2",
                   "motion jpeg 2000": ".mj2", "grayscale avi": ".avi", "indexed avi": ".avi"}


class VideoWriterObject(MObject):
    """VideoWriter: collects frames and encodes them when the file is closed."""

    is_handle = True
    _METHODS = ("open", "close", "writeVideo", "delete")
    _SETTABLE = ("FrameRate", "Quality", "Colormap", "CompressionRatio", "LosslessCompression", "MJ2BitDepth")

    def __init__(self, filename: str, profile: str | None):
        prof = (profile or "").lower()
        if profile is not None and prof not in _VIDEO_PROFILES:
            raise MatlabError(f"Unknown VideoWriter profile '{profile}'. Use 'Motion JPEG AVI' or 'MPEG-4'.")
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = _VIDEO_PROFILES.get(prof, ".avi")
            filename = base + ext
        if profile is None:
            profile = {".mp4": "MPEG-4", ".m4v": "MPEG-4", ".mj2": "Motion JPEG 2000"}.get(ext.lower(),
                                                                                           "Motion JPEG AVI")
        self.path = os.path.abspath(filename)
        self.profile = profile
        self.props: dict[str, Any] = {"FrameRate": 30.0, "Quality": 75.0}
        self.count = 0
        self.shape: tuple | None = None
        self.is_open = False
        self.closed = False
        self.store = None
        self.index: list[tuple[int, int]] = []
        self.written: str | None = None  # file produced by close (may be a .gif fallback)

    def class_name(self) -> str:
        return "VideoWriter"

    def isa_class(self, name: str) -> bool:
        return name in ("VideoWriter", "handle")

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    # properties ---------------------------------------------------------------------------
    def get_field(self, interp, name: str) -> Any:
        ext = os.path.splitext(self.path)[1].lower().lstrip(".")
        fixed = {
            "Filename": lambda: char(os.path.basename(self.path)),
            "Path": lambda: char(os.path.dirname(self.path)),
            "FrameCount": lambda: _num_val(self.count),
            "Duration": lambda: _num_val(self.count / self.props["FrameRate"]),
            "Width": lambda: _num_val(self.shape[1]) if self.shape else np.zeros((0, 0)),
            "Height": lambda: _num_val(self.shape[0]) if self.shape else np.zeros((0, 0)),
            "FileFormat": lambda: char(ext),
            "VideoFormat": lambda: char("RGB24"),
            "VideoBitsPerPixel": lambda: _num_val(24),
            "ColorChannels": lambda: _num_val(3),
            "VideoCompressionMethod": lambda: char({"mp4": "H.264", "gif": "GIF"}.get(ext, "Motion JPEG")),
        }
        for k, fn in fixed.items():
            if k.lower() == name.lower():
                return fn()
        for k in self._SETTABLE:
            if k.lower() == name.lower():
                v = self.props.get(k, np.zeros((0, 0)))
                return _num_val(v) if isinstance(v, float) else v
        if self.has_method(name):
            return super().get_field(interp, name)
        raise MatlabError(f"Unrecognized property '{name}' for class 'VideoWriter'.")

    def set_field(self, interp, name: str, value: Any) -> "VideoWriterObject":
        for k in self._SETTABLE:
            if k.lower() == name.lower():
                if k in ("FrameRate", "Quality"):
                    x = _scalar(value)
                    if k == "FrameRate" and not (np.isfinite(x) and x > 0):
                        raise MatlabError("FrameRate must be a positive number.")
                    if k == "Quality" and not 0 <= x <= 100:
                        raise MatlabError("Quality must be between 0 and 100.")
                    self.props[k] = x
                else:
                    self.props[k] = value
                return self
        raise MatlabError(f"Unable to set the '{name}' property of class 'VideoWriter': it is read-only or "
                          "does not exist.")

    def display_lines(self) -> list[str]:
        rows = [("Filename", f"'{os.path.basename(self.path)}'"), ("Path", f"'{os.path.dirname(self.path)}'"),
                ("FrameRate", f"{self.props['FrameRate']:g}"), ("Quality", f"{self.props['Quality']:g}"),
                ("FrameCount", str(self.count)),
                ("Width", str(self.shape[1]) if self.shape else "[]"),
                ("Height", str(self.shape[0]) if self.shape else "[]")]
        return ["  VideoWriter with properties:", ""] + [f"    {k.rjust(10)}: {v}" for k, v in rows]

    # methods ------------------------------------------------------------------------------
    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        rest = [a for a in args if a is not self]
        if name == "open":
            self.open()
        elif name == "writeVideo":
            if not rest:
                raise MatlabError("writeVideo needs a frame, e.g. writeVideo(v, getframe(gcf)).")
            self.write(rest[0])
        elif name in ("close", "delete"):
            self.close(interp)
            if name == "delete":
                self._deleted = True
        else:
            return super().call_method(interp, name, args, nargout)
        return []

    def open(self) -> None:
        if self.is_open:
            return
        folder = os.path.dirname(self.path)
        if not os.path.isdir(folder):
            raise MatlabError(f"Cannot create the video file: the folder '{folder}' does not exist.")
        import tempfile
        self.store = tempfile.TemporaryFile()
        self.index = []
        self.count = 0
        self.shape = None
        self.is_open = True
        self.closed = False

    def write(self, value) -> None:
        if not self.is_open:
            raise MatlabError("The video file was already closed; create a new VideoWriter to write more frames."
                              if self.closed else
                              "The video file is not open. Call open(v) before writeVideo(v, frame).")
        for f in _frames_of(value):
            if self.shape is None:
                self.shape = tuple(f.shape[:2])
            f = _resize_rgb(f, self.shape)
            z = zlib.compress(f.tobytes(), 1)
            self.store.seek(0, os.SEEK_END)
            self.index.append((self.store.tell(), len(z)))
            self.store.write(z)
            self.count += 1

    def _frames(self):
        h, w = self.shape
        for off, size in self.index:
            self.store.seek(off)
            yield np.frombuffer(zlib.decompress(self.store.read(size)), dtype=np.uint8).reshape(h, w, 3)

    def close(self, interp) -> None:
        if not self.is_open:
            return
        self.is_open = False
        self.closed = True
        try:
            if self.count:
                self._encode(interp)
        finally:
            self.store.close()
            self.store = None
            self.index = []

    def _encode(self, interp) -> None:
        ext = os.path.splitext(self.path)[1].lower()
        fps = float(self.props["FrameRate"])
        target = self.path
        if ext == ".avi":  # Motion JPEG and uncompressed AVI files need no external program
            from . import _avi
            mode = {"uncompressed avi": "raw", "grayscale avi": "gray", "indexed avi": "raw"}.get(
                self.profile.lower(), "mjpeg")
            try:
                _avi.write(self.path, self._frames(), self.shape, fps, float(self.props["Quality"]), mode)
            except OSError as e:
                raise MatlabError(f"Cannot write '{target}': {e.strerror or e}") from e
            self.written = target
            return
        if ext != ".gif":
            exe = _find_ffmpeg()
            if exe is not None and self._ffmpeg(exe, ext, fps):
                self.written = target
                return
            target = os.path.splitext(self.path)[0] + ".gif"
            why = "no ffmpeg program was found" if exe is None else "ffmpeg could not encode it"
            interp.warn(f"Cannot write '{os.path.basename(self.path)}' because {why}. The animation was saved "
                        f"as the animated GIF '{os.path.basename(target)}' instead.",
                        "mylab:VideoWriter:gifFallback")
        try:
            _write_gif(target, self._frames(), 1000.0 / fps)
        except OSError as e:
            raise MatlabError(f"Cannot write '{target}': {e.strerror or e}") from e
        self.written = target

    def _ffmpeg(self, exe: str, ext: str, fps: float) -> bool:
        import subprocess
        h, w = self.shape
        q = float(self.props["Quality"])
        even = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        qv = str(max(1, int(round(31 - q * 0.29))))
        if ext in (".mp4", ".m4v", ".mov"):
            attempts = [["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", str(int(round(40 - q * 0.3))), *even],
                        ["-vcodec", "mpeg4", "-q:v", qv, *even]]
        elif ext == ".avi" and self.profile.lower() == "uncompressed avi":
            attempts = [["-vcodec", "rawvideo", "-pix_fmt", "bgr24"]]
        elif ext == ".avi":
            attempts = [["-vcodec", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", qv, *even]]
        else:
            attempts = [[]]
        for codec in attempts:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-r", f"{fps:g}", "-i", "-", "-an", *codec, self.path]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                try:
                    for f in self._frames():
                        proc.stdin.write(f.tobytes())
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                if proc.wait(timeout=600) == 0 and os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
                    return True
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            try:
                if os.path.isfile(self.path):
                    os.remove(self.path)
            except OSError:
                pass
        return False


@builtin("VideoWriter")
def _videowriter(interp, args, nargout):
    if not args or not is_text(args[0]) or not text(args[0]):
        raise MatlabError("VideoWriter needs a file name, e.g. v = VideoWriter('movie.avi').")
    profile = text(args[1]) if len(args) > 1 else None
    return VideoWriterObject(text(args[0]), profile)


@builtin("writeVideo")
def _writevideo(interp, args, nargout):
    raise MatlabError("writeVideo expects a VideoWriter object as its first input.")


@builtin("open")
def _open(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("open expects a VideoWriter object or a file name.")
    name = text(args[0])
    ext = os.path.splitext(name)[1].lower()
    if ext == ".fig":
        from . import figfiles
        return figfiles.openfig_builtin(interp, [char(name)], nargout)
    if not os.path.exists(name):
        raise MatlabError(f"File '{name}' not found.")
    if ext == ".mat":
        from ._registry import BUILTINS
        return BUILTINS["load"](interp, [char(name)], 1)
    interp.warn(f"open: '{name}' was not opened; mylab has no editor or viewer windows.")
    return None


# -- animated lines -------------------------------------------------------------------------------

@builtin("animatedline")
@_with_axes_arg
def _animatedline(interp, args, nargout):
    args = list(args)
    nums = []
    while args and not is_text(args[0]):
        nums.append(np.real(fnum(args.pop(0))).flatten())
    args, opts = _split_name_values(args, _LINE_OPTS | {"maximumnumpoints"})
    if args:
        raise MatlabError(f"Invalid argument to animatedline: '{text(args[0]) if is_text(args[0]) else args[0]}'.")
    maxpts = opts.pop("maximumnumpoints", None)
    ax = _axes(interp, fresh=False)
    if len(nums) >= 3 and not hasattr(ax, "get_zlim"):
        ax = _axes(interp, fresh=False, projection="3d")
    kw = _mpl_kwargs(opts)
    kw.setdefault("color", "k")
    x = nums[0] if nums else np.zeros(0)
    y = nums[1] if len(nums) > 1 else np.zeros(0)
    if hasattr(ax, "get_zlim"):
        line = ax.plot(x, y, nums[2] if len(nums) >= 3 else np.zeros(x.size), **kw)[0]
    else:
        line = ax.plot(x, y, **kw)[0]
    h = _wrap(interp, "animatedline", line)
    _tag_lines(interp, [line], opts, "animatedline")
    if maxpts is not None:
        h.extra["MaximumNumPoints"] = _scalar(maxpts)
    _redraw(interp)
    return h if nargout > 0 else None


def _aline(args, name: str) -> GraphicsHandle:
    if not args or not isinstance(args[0], GraphicsHandle) or args[0].kind != "animatedline":
        raise MatlabError(f"{name} expects an animatedline object as its first input.")
    if not _alive(args[0]):
        raise MatlabError("Invalid or deleted object.")
    return args[0]


def _aline_data(h) -> list:
    line = h.obj
    vals = line.get_data_3d() if hasattr(line, "get_data_3d") else line.get_data()
    return [np.asarray(v, dtype=float).flatten() for v in vals]


def _aline_set(h, data: list, new: list | None = None) -> None:
    """Store the points of an animated line and grow the axes limits (``new``: only these were added)."""
    line = h.obj
    ax = line.axes
    if hasattr(line, "set_data_3d"):
        line.set_data_3d(*data)
        if ax is None or not all(d.size for d in data):
            return
        for d, setlim, auto in ((data[0], ax.set_xlim, ax.get_autoscalex_on()),
                                (data[1], ax.set_ylim, ax.get_autoscaley_on()),
                                (data[2], ax.set_zlim, ax.get_autoscalez_on())):
            d = d[np.isfinite(d)]
            if auto and d.size:
                lo, hi = float(d.min()), float(d.max())
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                setlim(lo, hi, auto=True)
        return
    line.set_data(*data[:2])
    if ax is None:
        return
    if new is None:
        ax.relim()
    else:
        pts = np.column_stack(new[:2])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.size:
            ax.dataLim.update_from_data_xy(pts, ignore=False)
    ax.autoscale_view()


@builtin("addpoints")
def _addpoints(interp, args, nargout):
    h = _aline(args, "addpoints")
    new = [np.real(fnum(a)).flatten() for a in args[1:]]
    old = _aline_data(h)
    if len(new) < 2 or len(new) != len(old):
        need = "x, y and z" if len(old) == 3 else "x and y"
        raise MatlabError(f"addpoints needs {need} values for this line.")
    if len({v.size for v in new}) != 1:
        raise MatlabError("addpoints: the coordinate vectors must have the same length.")
    data = [np.concatenate([o, v]) for o, v in zip(old, new)]
    limit = h.extra.get("MaximumNumPoints", np.inf)
    if np.isfinite(limit) and data[0].size > limit:
        _aline_set(h, [d[-int(limit):] for d in data])
    else:
        _aline_set(h, data, new=new)
    return None


@builtin("clearpoints")
def _clearpoints(interp, args, nargout):
    h = _aline(args, "clearpoints")
    _aline_set(h, [np.zeros(0)] * len(_aline_data(h)))
    return None


@builtin("getpoints")
def _getpoints(interp, args, nargout):
    h = _aline(args, "getpoints")
    data = [_row(d) for d in _aline_data(h)]
    return tuple(data[:nargout]) if nargout > 1 else data[0]


# -- axes, rectangles and finding objects ----------------------------------------------------------

def _name_values(args, fname: str) -> list:
    """[(name, value), ...] from trailing Name,Value arguments (a structure stands for its fields)."""
    flat = []
    for a in args:
        if isinstance(a, Struct) and a.data.size == 1:
            for f in a.fields:
                flat += [char(f), a.get_field(0, f)]
        else:
            flat.append(a)
    args = flat
    if len(args) % 2 or not all(is_text(a) for a in args[::2]):
        raise MatlabError(f"{fname} options must be name/value pairs.")
    return [(text(args[i]), args[i + 1]) for i in range(0, len(args), 2)]


def _apply_props(h: GraphicsHandle, pairs: list) -> None:
    for name, value in pairs:
        _set_prop(h, name, value)  # an unknown name raises "Unrecognized property"


@builtin("axes")
def _axes_b(interp, args, nargout):
    st = _state(interp)
    args = list(args)
    fig = None
    if args and isinstance(args[0], GraphicsHandle):
        h = args.pop(0)
        if not _alive(h):
            raise MatlabError("Invalid or deleted object.")
        if h.kind in ("axes", "polaraxes") and not args:  # axes(ax): make it current
            n = _fig_number(interp, h.obj.figure)
            _figure(interp, n)
            st.axes[n] = h.obj
            return h if nargout > 0 else None
        if h.kind != "figure":
            raise MatlabError("axes expects an axes handle, a figure handle or name/value pairs.")
        fig = h.obj
    pairs = _name_values(args, "axes")
    units, pos, rest = "normalized", None, []
    for name, value in pairs:
        low = name.lower()
        if low == "parent":
            hs = _handles_of(value) or _resolve(interp, value)
            fig = hs[0].obj if hs[0].kind == "figure" else _obj_figure(hs[0])
        elif low == "units":
            units = text(value).lower()
        elif low in ("position", "outerposition", "innerposition"):
            pos = np.real(fnum(value)).flatten()
            if pos.size != 4 or pos[2] <= 0 or pos[3] <= 0:
                raise MatlabError("Position must be [left bottom width height] with a positive width and height.")
        else:
            rest.append((name, value))
    if fig is None:
        fig = _figure(interp)
    n = _fig_number(interp, fig)
    if n is None:
        raise MatlabError("Invalid or deleted object.")
    if pos is None:
        rect = [0.13, 0.11, 0.775, 0.815]
    else:
        rect = [float(x) for x in pos]
        if units == "pixels" or (units == "normalized" and max(rect) > 2):
            w, h_px = fig.get_size_inches() * _fig_dpi(fig)
            rect = [rect[0] / w, rect[1] / h_px, rect[2] / w, rect[3] / h_px]
    st.counter = getattr(st, "counter", 0) + 1
    ax = fig.add_axes(rect, label=f"mylab-axes-{st.counter}")
    _figure(interp, n)
    st.axes[n] = ax
    h = _wrap(interp, "axes", ax)
    h.extra["Units"] = units
    _apply_props(h, rest)
    _redraw(interp)
    return h if nargout > 0 else None


def _rect_path(pos, curv):
    from matplotlib.path import Path
    x, y, w, h = (float(v) for v in pos)
    rx, ry = float(curv[0]) * w / 2.0, float(curv[1]) * h / 2.0
    if rx <= 0 or ry <= 0:
        return Path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], closed=True)
    verts, codes = [], []
    for cx, cy, a0 in ((x + w - rx, y + ry, 270), (x + w - rx, y + h - ry, 0), (x + rx, y + h - ry, 90),
                       (x + rx, y + ry, 180)):
        arc = Path.arc(a0, a0 + 90)
        v = arc.vertices * [rx, ry] + [cx, cy]
        c = arc.codes.copy()
        if verts:
            c[0] = Path.LINETO
        verts.extend(v.tolist())
        codes.extend(c.tolist())
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


def _rect_curvature(v, pos) -> np.ndarray:
    c = np.clip(np.real(fnum(v)).flatten(), 0.0, 1.0)
    if c.size == 1:  # one number rounds the corners with circles sized from the shorter side
        r = float(c[0]) * min(pos[2], pos[3]) / 2.0
        return np.array([2 * r / pos[2] if pos[2] else 0.0, 2 * r / pos[3] if pos[3] else 0.0])
    if c.size != 2:
        raise MatlabError("Curvature must be a number or a two-element vector with values from 0 to 1.")
    return c


def _rect_position(v) -> np.ndarray:
    p = np.real(fnum(v)).flatten()
    if p.size != 4 or p[2] < 0 or p[3] < 0:
        raise MatlabError("Position must be [x y width height] with a non-negative width and height.")
    return p


def _rect_update(h, position=None, curvature=None) -> None:
    if position is not None:
        h.extra["Position"] = _rect_position(position)
    if curvature is not None:
        h.extra["Curvature"] = _rect_curvature(curvature, h.extra["Position"])
    h.obj.set_path(_rect_path(h.extra["Position"], h.extra["Curvature"]))
    ax = h.obj.axes
    if ax is not None:
        ax.relim()
        ax.autoscale_view()


@builtin("rectangle")
@_with_axes_arg
def _rectangle(interp, args, nargout):
    from matplotlib.patches import PathPatch
    pairs = _name_values(list(args), "rectangle")
    pos, curv, rest = np.array([0.0, 0.0, 1.0, 1.0]), [0.0, 0.0], []
    kw: dict = {"facecolor": "none", "edgecolor": "k", "linewidth": 0.5}
    parent = None
    for name, value in pairs:
        low = name.lower()
        if low == "position":
            pos = _rect_position(value)
        elif low == "curvature":
            curv = value
        elif low in ("facecolor", "edgecolor"):
            kw[low] = _color(value)
        elif low == "linewidth":
            kw["linewidth"] = _scalar(value)
        elif low == "linestyle":
            kw["linestyle"] = _mpl_linestyle(text(value))
        elif low == "facealpha":
            kw["alpha"] = _scalar(value)
        elif low == "parent":
            hs = _handles_of(value)
            parent = _obj_axes(hs[0]) if hs else None
        else:
            rest.append((name, value))
    ax = parent if parent is not None else _axes(interp, fresh=False)
    curv = _rect_curvature(curv, pos)
    patch = PathPatch(_rect_path(pos, curv), **kw)
    ax.add_patch(patch)
    ax.autoscale_view()
    h = _wrap(interp, "rectangle", patch)
    h.extra.update({"Position": pos, "Curvature": curv, "axes": ax, "figure": ax.figure})
    _apply_props(h, rest)
    _redraw(interp)
    return h if nargout > 0 else None


def _handle_list(hs: list):
    """A list of handles as a MATLAB value (empty, one handle, or a column of handles)."""
    if not hs:
        return np.zeros((0, 0))
    return hs[0] if len(hs) == 1 else GraphicsArray(hs)


_CONTAINER_KINDS = ("bar", "errorbar", "stem", "histogram", "pie", "patch", "area")


def _child_handles(interp, h: GraphicsHandle, hidden: bool = True) -> list:
    """Children of a handle, including legends and colorbars of a figure, and hidden objects."""
    if h.kind == "figure":
        kids = [_wrap(interp, _axes_kind(a), a) for a in h.obj.axes if not _is_colorbar_axes(a)
                and not getattr(a, "_mylab_internal", False)]
        kids.reverse()
        for ax in h.obj.axes:
            leg = ax.get_legend()
            if leg is not None:
                kids.append(_wrap(interp, "legend", leg))
            cb = getattr(ax, "_mylab_colorbar", None)
            if cb is not None:
                kids.append(_wrap(interp, "colorbar", cb))
    elif h.kind in ("axes", "polaraxes", "geoaxes"):
        kids = _axes_children(interp, h.obj)
    elif h.kind == "root":
        kids = _handles_of(_children_get(h))
    elif h.extra.get("_children") is not None:
        kids = [c for c in h.extra["_children"] if _alive(c)]
    else:
        kids = []
    if not hidden:
        kids = [k for k in kids if str(k.extra.get("HandleVisibility", "on")) != "off"]
    return kids


def _prop_matches(h: GraphicsHandle, name: str, want, regexp: bool) -> bool:
    try:
        have = _get_prop(h, name)
    except (MatlabError, ValueError, TypeError, AttributeError, IndexError, KeyError):
        return False
    if is_text(want):
        if isinstance(have, Cell) and not regexp:
            return False
        if not is_text(have):
            return False
        a, b = text(have), text(want)
        if regexp:
            return re.search(b, a) is not None
        return a.lower() == b.lower() if name.lower() in ("type", "visible", "linestyle", "marker") else a == b
    if isinstance(want, (GraphicsHandle, GraphicsArray)):
        return have is want
    if isinstance(want, np.ndarray) and isinstance(have, np.ndarray) and not is_char(have):
        try:
            return bool(np.array_equal(np.asarray(have, dtype=float).flatten(),
                                       np.asarray(want, dtype=float).flatten()))
        except (TypeError, ValueError):
            return False
    return False


def _find_objects(interp, args, hidden: bool = True) -> list:
    args = list(args)
    st = _state(interp)
    if args and not is_text(args[0]):
        first = args.pop(0)
        if isinstance(first, np.ndarray) and first.size == 0:
            return []
        roots = _resolve(interp, first)
    else:
        roots = [_wrap(interp, "root", _ROOT)]
    if not hidden and str(_wrap(interp, "root", _ROOT).extra.get("ShowHiddenHandles", "off")) == "on":
        hidden = True
    depth = math.inf
    conds: list = []  # (operator, negate, kind, name, value)
    op, negate, regexp = "and", False, False
    k = 0
    while k < len(args):
        if not is_text(args[k]):
            raise MatlabError("findobj expects property names followed by values.")
        s = text(args[k])
        low = s.lower()
        if low == "flat":
            depth = 0
        elif low == "-depth":
            if k + 1 >= len(args):
                raise MatlabError("findobj: '-depth' needs a number.")
            depth = _scalar(args[k + 1])
            k += 1
        elif low in ("-and", "-or", "-xor"):
            op = low[1:]
        elif low == "-not":
            negate = True
        elif low == "-regexp":
            regexp = True
        elif low in ("-property", "-isa", "-function", "-class"):
            if k + 1 >= len(args):
                raise MatlabError(f"findobj: '{s}' needs a value.")
            conds.append((op, negate, low, None, args[k + 1]))
            op, negate = "and", False
            k += 1
        else:
            if k + 1 >= len(args):
                raise MatlabError("findobj expects property names followed by values.")
            conds.append((op, negate, "regexp" if regexp else "value", s, args[k + 1]))
            op, negate = "and", False
            k += 1
        k += 1

    found: list = []
    seen: set = set()

    def visit(h, d):
        if id(h) in seen:
            return
        seen.add(id(h))
        found.append(h)
        if d < depth:
            for c in _child_handles(interp, h, hidden):
                visit(c, d + 1)

    for r in roots:
        visit(r, 0)
    del st

    def test(h, cond) -> bool:
        _, neg, kind, name, value = cond
        if kind == "-property":
            ok = _lookup(h, text(value))[0] is not None
        elif kind in ("-isa", "-class"):
            ok = h.isa_class(text(value))
        elif kind == "-function":
            res = interp.call_function_value(value, [h], 1)
            ok = bool(res and np.all(np.asarray(res[0])))
        else:
            ok = _prop_matches(h, name, value, kind == "regexp")
        return ok != neg

    out = []
    for h in found:
        result = True
        for i, cond in enumerate(conds):
            t = test(h, cond)
            if i == 0:
                result = t
            elif cond[0] == "or":
                result = result or t
            elif cond[0] == "xor":
                result = result != t
            else:
                result = result and t
        if result:
            out.append(h)
    return out


@builtin("findobj")
def _findobj(interp, args, nargout):
    return _handle_list(_find_objects(interp, args, hidden=False))


@builtin("findall")
def _findall(interp, args, nargout):
    return _handle_list(_find_objects(interp, args, hidden=True))


@builtin("allchild")
def _allchild(interp, args, nargout):
    if not args:
        raise MatlabError("allchild needs a graphics handle.")
    out = []
    for h in _resolve(interp, args[0]):
        out.extend(_child_handles(interp, h))
    return _handle_list(out)


@builtin("ancestor")
def _ancestor(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("ancestor needs a handle and a type, e.g. ancestor(h, 'figure').")
    if isinstance(args[0], np.ndarray) and args[0].size == 0:
        return np.zeros((0, 0))
    types = {t.lower() for t in _text_list(args[1])}
    top = len(args) > 2 and is_text(args[2]) and text(args[2]).lower() == "toplevel"
    cur = _resolve(interp, args[0])[0]
    match = None
    while isinstance(cur, GraphicsHandle):
        if _TYPE.get(cur.kind, cur.kind) in types:
            match = cur
            if not top:
                break
        cur = _parent_get(cur)
    return match if match is not None else np.zeros((0, 0))


# -- application data ---------------------------------------------------------------------------

_GUIDATA_KEY = "\0guidata"


def _appdata(interp, v) -> dict:
    hs = _resolve(interp, v)
    if not hs:
        raise MatlabError("Expected a graphics handle.")
    h = hs[0]
    if h.kind == "root":  # survives 'close all'
        return interp.state.setdefault("root_appdata", {})
    if not _alive(h):
        raise MatlabError("Invalid or deleted object.")
    return h.extra.setdefault("_appdata", {})


@builtin("setappdata")
def _setappdata(interp, args, nargout):
    if len(args) != 3:
        raise MatlabError("setappdata needs a handle, a name and a value.")
    _appdata(interp, args[0])[text(args[1], "name")] = args[2]
    return None


@builtin("getappdata")
def _getappdata(interp, args, nargout):
    if not args:
        raise MatlabError("getappdata needs a graphics handle.")
    data = _appdata(interp, args[0])
    if len(args) == 1:
        return Struct.scalar({k: v for k, v in data.items() if k != _GUIDATA_KEY and k.isidentifier()})
    return data.get(text(args[1], "name"), np.zeros((0, 0)))


@builtin("isappdata")
def _isappdata(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("isappdata needs a handle and a name.")
    return np.array([[text(args[1], "name") in _appdata(interp, args[0])]])


@builtin("rmappdata")
def _rmappdata(interp, args, nargout):
    if len(args) != 2:
        raise MatlabError("rmappdata needs a handle and a name.")
    name = text(args[1], "name")
    data = _appdata(interp, args[0])
    if name not in data:
        raise MatlabError(f"Application data '{name}' does not exist.")
    del data[name]
    return None


@builtin("guidata")
def _guidata(interp, args, nargout):
    if not args:
        raise MatlabError("guidata needs a graphics handle.")
    h = _resolve(interp, args[0])[0]
    while isinstance(h, GraphicsHandle) and h.kind not in ("figure", "root"):
        h = _parent_get(h)
    if not isinstance(h, GraphicsHandle):
        raise MatlabError("guidata: the object is not inside a figure.")
    data = _appdata(interp, h)
    if len(args) > 1:
        data[_GUIDATA_KEY] = args[1]
        return None
    return data.get(_GUIDATA_KEY, np.zeros((0, 0)))


# -- colour limits, aspect ratios and accepted-but-ignored commands -----------------------------------

@builtin("caxis", "clim")
@_with_axes_arg
def _caxis(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    h = _wrap(interp, _axes_kind(ax), ax)
    if args:
        if is_text(args[0]):
            mode = text(args[0]).lower()
            if mode == "auto":
                for m in _mappables(ax):
                    m.autoscale()
            elif mode not in ("manual", "tight", "padded", "tickaligned"):
                raise MatlabError("caxis expects [cmin cmax], 'auto' or 'manual'.")
        else:
            lim = np.real(fnum(args[0])).flatten()
            if lim.size != 2 or not lim[0] < lim[1]:
                raise MatlabError("Color limits must be a two-element vector [cmin cmax] with cmin < cmax.")
            _clim_set(h, lim)
        _redraw(interp)
    return _clim_get(h) if nargout > 0 or not args else None


def _aspect(interp, args, nargout, which: str):
    ax = _axes(interp, fresh=False)
    attr = f"_mylab_{which}"
    if args:
        if is_text(args[0]):
            mode = text(args[0]).lower()
            if mode == "mode":
                return char("manual" if getattr(ax, attr, None) is not None else "auto")
            if mode == "auto":
                setattr(ax, attr, None)
                if which == "daspect":
                    ax.set_aspect("auto")
                elif not hasattr(ax, "get_zlim"):
                    ax.set_box_aspect(None)
            elif mode != "manual":
                raise MatlabError(f"{which} expects a vector such as [1 1 1], 'auto' or 'manual'.")
        else:
            v = np.real(fnum(args[0])).flatten()
            if v.size == 2:
                v = np.append(v, 1.0)
            if v.size != 3 or np.any(v <= 0) or not np.all(np.isfinite(v)):
                raise MatlabError(f"{which} expects three positive numbers, e.g. {which}([1 1 1]).")
            setattr(ax, attr, v)
            three_d = hasattr(ax, "get_zlim")
            if which == "daspect":
                if not three_d:
                    ax.set_aspect(float(v[0] / v[1]), adjustable="box")
                elif v[0] == v[1] == v[2]:
                    try:
                        ax.set_aspect("equal")
                    except (NotImplementedError, ValueError):
                        pass
            elif three_d:
                ax.set_box_aspect(tuple(float(x) for x in v))
            else:
                ax.set_box_aspect(float(v[1] / v[0]))
        _redraw(interp)
    if nargout > 0 or not args:
        v = getattr(ax, attr, None)
        return _row(v if v is not None else [1.0, 1.0, 1.0])
    return None


builtin("daspect")(_with_axes_arg(lambda interp, args, nargout: _aspect(interp, args, nargout, "daspect")))
builtin("pbaspect")(_with_axes_arg(lambda interp, args, nargout: _aspect(interp, args, nargout, "pbaspect")))


def _mode_target(interp, args: list):
    """(figure handle, axes or None, remaining args) for zoom, pan, rotate3d and datacursormode."""
    args = list(args)
    ax = None
    if args and isinstance(args[0], GraphicsHandle):
        h = args.pop(0)
        if h.kind in ("axes", "polaraxes"):
            ax = h.obj
            fig = h.obj.figure
        elif h.kind == "figure":
            fig = h.obj
        else:
            raise MatlabError("Expected a figure or an axes.")
    else:
        fig = _figure(interp)
    return _wrap(interp, "figure", fig), ax, args


def _interaction_mode(name: str, words: tuple):
    def f(interp, args, nargout):
        fh, ax, rest = _mode_target(interp, args)
        key = f"_{name}_mode"
        if not rest:
            if nargout > 0:  # the state of the mode; there is no mouse, so it only records on/off
                return Struct.scalar({"Enable": char(fh.extra.get(key, "off")),
                                      "FigureHandle": fh})
            fh.extra[key] = "off" if fh.extra.get(key, "off") != "off" else "on"
            return None
        word = text(rest[0]).lower() if is_text(rest[0]) else None
        if word not in words:
            raise MatlabError(f"{name}: expected one of {', '.join(repr(w) for w in words)}.")
        fh.extra[key] = "off" if word == "off" else ("on" if word in ("on", "toggle") else word)
        return None
    f.__doc__ = f"{name} mode: recorded on the figure; it needs a mouse to have an effect."
    return f


builtin("pan")(_interaction_mode("pan", ("on", "off", "xon", "yon", "toggle")))
builtin("rotate3d")(_interaction_mode("rotate3d", ("on", "off", "toggle")))
builtin("datacursormode")(_interaction_mode("datacursormode", ("on", "off", "toggle")))
builtin("brush")(_interaction_mode("brush", ("on", "off", "toggle")))


@builtin("zoom")
def _zoom(interp, args, nargout):
    """zoom(factor) changes the limits of the current axes; zoom on/off records the mouse mode."""
    fh, ax, rest = _mode_target(interp, args)
    if rest and not is_text(rest[0]):
        factor = _scalar(rest[0])
        if factor <= 0:
            raise MatlabError("zoom: the factor must be positive.")
        st = _state(interp)
        if ax is None:
            n = _fig_number(interp, fh.obj)
            ax = st.axes.get(n)
            if ax is None or ax not in fh.obj.axes:
                raise MatlabError("zoom: the figure has no axes.")
        if not hasattr(ax, "_mylab_zoom_reset"):
            ax._mylab_zoom_reset = (ax.get_xlim(), ax.get_ylim())
        for which in "xy":
            lo, hi = getattr(ax, f"get_{which}lim")()
            if getattr(ax, f"get_{which}scale")() == "log" and lo > 0 and hi > 0:
                c = math.sqrt(lo * hi)
                half = math.sqrt(hi / lo) ** (1 / factor)
                getattr(ax, f"set_{which}lim")(c / half, c * half)
            else:
                c, half = (lo + hi) / 2, (hi - lo) / 2 / factor
                getattr(ax, f"set_{which}lim")(c - half, c + half)
        _redraw(interp)
        return None
    if rest and text(rest[0]).lower() in ("out", "reset"):
        st = _state(interp)
        target = ax or st.axes.get(_fig_number(interp, fh.obj))
        if target is not None and target in fh.obj.axes:
            if text(rest[0]).lower() == "reset":
                target._mylab_zoom_reset = (target.get_xlim(), target.get_ylim())
            else:
                saved = getattr(target, "_mylab_zoom_reset", None)
                if saved is not None:
                    target.set_xlim(saved[0])
                    target.set_ylim(saved[1])
                else:
                    target.autoscale(True)
            _redraw(interp)
        return None
    return _interaction_mode("zoom", ("on", "off", "xon", "yon", "toggle"))(
        interp, [fh] + rest, nargout)


@builtin("hgsave", "savefig")
def _savefig(interp, args, nargout):
    from . import figfiles
    return figfiles.savefig_builtin(interp, list(args), nargout)


@builtin("hgload", "openfig")
def _openfig(interp, args, nargout):
    from . import figfiles
    return figfiles.openfig_builtin(interp, list(args), nargout)


def _link_callback(which: str, group: list):
    def on_change(src):
        if getattr(src, "_mylab_syncing", False):
            return
        lim = getattr(src, f"get_{which}lim")()
        for o in group:
            if o is not src and o.figure is not None:
                o._mylab_syncing = True
                try:
                    getattr(o, f"set_{which}lim")(lim)
                finally:
                    o._mylab_syncing = False
    return on_change


def _relink(ax) -> None:
    """Connect the limit callbacks of a linked axes again (clearing an axes drops its callbacks)."""
    for which, info in getattr(ax, "_mylab_linked", {}).items():
        cid = ax.callbacks.connect(f"{which}lim_changed", _link_callback(which, info[1]))
        ax._mylab_linked[which] = (cid, info[1])


@builtin("linkaxes")
def _linkaxes(interp, args, nargout):
    if not args:
        raise MatlabError("linkaxes needs an array of axes handles.")
    axs = [h.obj for h in _handles_of(args[0]) if h.kind == "axes"]
    if len(axs) != len(_handles_of(args[0])) or not axs:
        raise MatlabError("linkaxes expects axes handles, e.g. linkaxes([ax1 ax2], 'x').")
    mode = text(args[1]).lower() if len(args) > 1 else "xyz"
    if mode not in ("x", "y", "z", "xy", "xz", "yz", "xyz", "off"):
        raise MatlabError("linkaxes: the option must be 'x', 'y', 'z', a combination such as 'xy', or 'off'.")
    for a in axs:  # links set earlier for these axes are replaced
        for cid, _group in list(getattr(a, "_mylab_linked", {}).values()):
            a.callbacks.disconnect(cid)
        a._mylab_linked = {}
    if mode == "off" or len(axs) < 2:
        return None
    for which in mode:
        if not all(hasattr(a, f"get_{which}lim") for a in axs):
            continue
        lims = [getattr(a, f"get_{which}lim")() for a in axs]
        lo, hi = min(min(v) for v in lims), max(max(v) for v in lims)
        for a in axs:
            a._mylab_linked[which] = (a.callbacks.connect(f"{which}lim_changed", _link_callback(which, axs)), axs)
        for a in axs:
            _set_lim(a, which, lo, hi)
    _redraw(interp)
    return None


# -- contour labels, two y-axes, 3-D polygons ---------------------------------------------------------

def _contour_matrix(cs) -> np.ndarray:
    """Contour lines as a 2-row matrix: [level, x...; count, y...] for each line, side by side."""
    cols = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            seg = np.asarray(seg, dtype=float)
            if seg.ndim != 2 or seg.shape[0] == 0:
                continue
            cols.append(np.array([[float(level)], [float(seg.shape[0])]]))
            cols.append(seg.T)
    return np.hstack(cols) if cols else np.zeros((2, 0))


@builtin("clabel")
def _clabel(interp, args, nargout):
    cs = None
    for a in args:
        if isinstance(a, GraphicsHandle) and a.kind == "contour":
            cs = a.obj
    pos, opts = _split_name_values([a for a in args if is_text(a) or not isinstance(a, GraphicsHandle)],
                                   {"fontsize", "color", "labelspacing", "fontweight", "rotation"})
    _reject_text(pos, "text", flags=("manual",))
    if cs is None:
        cs = getattr(_axes(interp, fresh=False), "_mylab_contour", None)
    if cs is None:
        raise MatlabError("clabel: draw a contour plot first, e.g. [C, h] = contour(Z); clabel(C, h).")
    kw: dict = {"inline": True, "fontsize": _scalar(opts["fontsize"]) if "fontsize" in opts else 8}
    if "color" in opts:
        kw["colors"] = [_color(opts["color"])]
    try:
        cs.axes.clabel(cs, **kw)
    except (ValueError, IndexError):
        pass  # nothing to label
    _redraw(interp)
    return np.zeros((0, 0)) if nargout > 0 else None


@builtin("plotyy")
def _plotyy(interp, args, nargout):
    args = list(args)
    if len(args) < 4:
        raise MatlabError("plotyy needs two data sets: plotyy(x1, y1, x2, y2).")
    methods = []
    for f in args[4:6]:
        name = f.name if isinstance(f, FunctionHandle) and not f.is_anonymous else (text(f) if is_text(f) else "")
        if name not in ("plot", "semilogx", "semilogy", "loglog", "stem", "stairs", "bar"):
            raise MatlabError("plotyy: the plotting function must be plot, semilogx, semilogy, loglog, stem, "
                              "stairs or bar.")
        methods.append({"stairs": "step"}.get(name, name))
    methods += ["plot"] * (2 - len(methods)) if not methods else [methods[0]] * (2 - len(methods))
    import matplotlib as mpl
    colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    ax1 = _axes(interp, projection="rectilinear")
    ax2 = ax1.twinx()
    out = []
    for ax, (x, y), method, color in ((ax1, args[0:2], methods[0], colors[0]),
                                      (ax2, args[2:4], methods[1], colors[1 % len(colors)])):
        X, Y = _prep_xy(fnum(x), fnum(y))
        kw = {} if method in ("stem", "bar") else {"color": color}
        res = getattr(ax, method)(np.asarray(X), np.asarray(Y), **kw)
        lines = [a for a in _artists(res) if hasattr(a, "set_color")]
        if method in ("stem", "bar"):
            for a in lines:
                a.set_color(color)
        ax.tick_params(axis="y", colors=color)
        kind = {"stem": "stem", "bar": "bar"}.get(method, "line")
        out.append(_handle_list([_wrap(interp, kind, o) for o in ([res] if kind != "line" else lines)]))
    _redraw(interp)
    hax = GraphicsArray([_wrap(interp, "axes", ax1), _wrap(interp, "axes", ax2)], row=True)
    return (hax, out[0], out[1])[:max(nargout, 1)] if nargout > 1 else (hax if nargout == 1 else None)


def _face_colors(interp, c, nfaces: int):
    """(fixed colour or None, per-face numbers or None) from a colour argument."""
    if c is None:
        return None, None
    if is_text(c):
        return _color(c), None
    v = np.real(fnum(c))
    if v.size == 3 and nfaces != 3:
        return _color(c), None
    if v.ndim == 2 and v.shape[0] > 1 and v.shape[1] == nfaces:
        return None, v.mean(axis=0)
    if v.size == nfaces:
        return None, v.flatten()
    if v.size == 3:
        return _color(c), None
    return None, np.full(nfaces, float(v.mean()))


def _add_polys(interp, ax, polys: list, opts: dict, color, nargout: int):
    """Add 2-D or 3-D polygons (lists of vertices) to ``ax`` as one patch object."""
    three_d = hasattr(ax, "get_zlim")
    fixed, values = _face_colors(interp, color, len(polys))
    kw: dict = {"edgecolor": "k", "linewidth": 0.5}
    fc = opts.get("facecolor")
    if fc is not None and not (is_text(fc) and text(fc).lower() in ("flat", "interp")):
        fixed, values = _color(fc), None
    if "edgecolor" in opts:
        ec = opts["edgecolor"]
        kw["edgecolor"] = "none" if is_text(ec) and text(ec).lower() in ("none", "flat", "interp") else _color(ec)
    if "linewidth" in opts:
        kw["linewidth"] = _scalar(opts["linewidth"])
    if "facealpha" in opts:
        kw["alpha"] = _scalar(opts["facealpha"])
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    if three_d:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as Coll
    else:
        from matplotlib.collections import PolyCollection as Coll
    if values is None:
        kw["facecolor"] = fixed if fixed is not None else "k"
    coll = Coll(polys, **kw)
    if values is not None:
        coll.set_cmap(_current_cmap(interp))
        coll.set_array(np.asarray(values, dtype=float))
    pts = np.vstack(polys)
    if three_d:
        had = ax.has_data()
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2], had)
    else:
        ax.add_collection(coll)
        ax.update_datalim(pts[:, :2])
        ax.autoscale_view()
    _redraw(interp)
    h = _wrap(interp, "patch", coll)
    h.extra.update({"axes": ax, "figure": ax.figure, "_polys": [np.asarray(q, dtype=float) for q in polys]})
    hook = interp.state.get("relight_hook")
    if hook is not None and three_d:
        hook(interp, ax)
    return (h) if nargout > 0 else None


_POLY_OPTS = {"facecolor", "edgecolor", "facealpha", "displayname", "linewidth", "linestyle", "edgealpha"}


def _fill3_core(interp, args, nargout, fresh: bool = True):
    args, opts = _split_name_values(list(args), _POLY_OPTS)
    if len(args) < 3:
        raise MatlabError("fill3 needs x, y and z data and a colour.")
    x, y, z = (np.real(fnum(a)) for a in args[:3])
    if not x.size == y.size == z.size:
        raise MatlabError("The x, y and z data must have the same number of elements.")
    if x.shape[0] > 1 and x.shape[1] > 1:
        polys = [np.column_stack([x[:, j], y[:, j], z[:, j]]) for j in range(x.shape[1])]
    else:
        polys = [np.column_stack([x.flatten(), y.flatten(), z.flatten()])]
    ax = _axes(interp, fresh=fresh, projection="3d")
    return _add_polys(interp, ax, polys, opts, args[3] if len(args) > 3 else None, nargout)


@builtin("fill3")
@_with_axes_arg
def _fill3(interp, args, nargout):
    return _fill3_core(interp, args, nargout)


def _patch_named(interp, args, nargout):
    """patch('Faces', F, 'Vertices', V, ...) and patch('XData', x, 'YData', y, ...)."""
    pairs = {name.lower(): value for name, value in _name_values(list(args), "patch")}
    opts = {k: v for k, v in pairs.items() if k in _POLY_OPTS}
    color = pairs.get("facevertexcdata", pairs.get("cdata"))
    if "xdata" in pairs:
        if "ydata" not in pairs:
            raise MatlabError("patch needs both XData and YData.")
        data = [pairs["xdata"], pairs["ydata"]] + ([pairs["zdata"]] if "zdata" in pairs else [])
        cols = [np.real(fnum(d)) for d in data]
        if cols[0].shape[0] > 1 and cols[0].shape[1] > 1:
            polys = [np.column_stack([c[:, j] for c in cols]) for j in range(cols[0].shape[1])]
        else:
            polys = [np.column_stack([c.flatten() for c in cols])]
    else:
        if "faces" not in pairs or "vertices" not in pairs:
            raise MatlabError("patch needs both Faces and Vertices.")
        V = np.real(fnum(pairs["vertices"]))
        F = np.real(fnum(pairs["faces"]))
        if V.ndim != 2 or V.shape[1] not in (2, 3):
            raise MatlabError("Vertices must be an N-by-2 or N-by-3 matrix.")
        polys = []
        for row in F.reshape(-1, F.shape[-1]) if F.ndim == 2 else [F.flatten()]:
            idx = row[np.isfinite(row)].astype(int) - 1
            if idx.size and (idx.min() < 0 or idx.max() >= V.shape[0]):
                raise MatlabError("Faces contains an index outside the rows of Vertices.")
            if idx.size:
                polys.append(V[idx])
        if color is not None and not is_text(color):
            c = np.real(fnum(color))
            if c.ndim == 2 and c.shape[0] == V.shape[0] and c.shape[0] != len(polys) and c.shape[1] == 1:
                rows = F.reshape(-1, F.shape[-1])  # per-vertex numbers: average them over each face
                color = np.array([np.nanmean(c[r[np.isfinite(r)].astype(int) - 1, 0]) for r in rows])
            elif c.ndim == 2 and c.shape[1] == 3 and c.shape[0] == len(polys) and "facecolor" in opts and \
                    is_text(opts["facecolor"]) and text(opts["facecolor"]).lower() == "flat":
                color = None
                opts["_rgb"] = c
    if not polys:
        return np.zeros((0, 0)) if nargout > 0 else None
    three_d = polys[0].shape[1] == 3
    ax = _axes(interp, fresh=False, projection="3d") if three_d else _axes(interp, fresh=False)
    if not three_d and hasattr(ax, "get_zlim"):
        polys = [np.column_stack([p, np.zeros(len(p))]) for p in polys]
    rgb = opts.pop("_rgb", None)
    h = _add_polys(interp, ax, polys, opts, color, 1)
    if rgb is not None:
        h.obj.set_facecolor(np.clip(rgb, 0, 1))
    return h if nargout > 0 else None


# -- plots of functions -------------------------------------------------------------------------------

_EZ_CONSTANTS = {"pi", "inf", "nan", "eps", "i", "j", "Inf", "NaN"}


def _as_fn(interp, f):
    """(function handle, number of inputs or None, label) from a handle or an expression in text."""
    if isinstance(f, FunctionHandle):
        label = re.sub(r"^@\([^)]*\)\s*", "", f.text) if f.is_anonymous and f.text else str(f)
        return f, (len(f.params) if f.is_anonymous else None), label
    if not is_text(f):
        raise MatlabError("Expected a function handle or an expression in quotes, e.g. @(x) sin(x).")
    expr = text(f).strip()
    m = re.fullmatch(r"([^=<>~]+)=([^=]+)", expr)
    body = f"({m.group(1)}) - ({m.group(2)})" if m else expr
    names = []
    for mm in re.finditer(r"[A-Za-z_]\w*", body):
        name = mm.group(0)
        after = body[mm.end():].lstrip()
        if name in _EZ_CONSTANTS or after.startswith("(") or name in names:
            continue
        if mm.start() > 0 and (body[mm.start() - 1].isdigit() or body[mm.start() - 1] == ".") and name in "eE":
            continue
        names.append(name)
    names = sorted(names) or ["x"]
    vec = re.sub(r"(?<![.])([*/^])", r".\1", body)
    try:
        handle = interp.eval_expression(f"@({','.join(names)}) {vec}")
    except MatlabError as e:
        raise MatlabError(f"Cannot turn '{expr}' into a function: {e.message}") from e
    return handle, len(names), expr


def _eval_fn(interp, f, arrays: list) -> np.ndarray:
    """Evaluate ``f`` on arrays of equal shape, element by element if it is not vectorised."""
    shape = arrays[0].shape
    try:
        res = interp.call_function_value(f, [np.array(a, dtype=float) for a in arrays], 1)
        vals = fnum(res[0]) if res else None
        if vals is not None and vals.shape == shape:
            return np.where(np.abs(np.imag(vals)) > 1e-12, np.nan, np.real(vals)) if np.iscomplexobj(vals) \
                else np.asarray(vals, dtype=float)
        if vals is not None and vals.size == 1:
            return np.full(shape, float(np.real(vals.flat[0])))
    except MatlabError:
        pass
    out = np.full(shape, np.nan)
    for idx in np.ndindex(shape):
        res = interp.call_function_value(f, [np.array([[float(a[idx])]]) for a in arrays], 1)
        if res:
            v = fnum(res[0])
            if v.size and abs(np.imag(v.flat[0])) <= 1e-12:
                out[idx] = float(np.real(v.flat[0]))
    return out


def _fn_args(interp, args):
    """Split (functions..., [limits], options...) for the function plotters."""
    args = list(args)
    funcs = []
    while args and (isinstance(args[0], FunctionHandle) or
                    (is_text(args[0]) and not _is_linespec(text(args[0])) and len(funcs) < 3
                     and text(args[0]).lower() not in ("meshdensity", "linewidth", "color", "linestyle"))):
        funcs.append(_as_fn(interp, args.pop(0)))
    if not funcs:
        raise MatlabError("Expected a function handle, e.g. @(x) sin(x).")
    lims = None
    if args and not is_text(args[0]):
        lims = np.real(fnum(args.pop(0))).flatten()
    return funcs, lims, args


def _range(lims, default: float, n: int = 2) -> list:
    if lims is None:
        return [-default, default] * n
    if lims.size == 2 and lims[0] < lims[1]:
        return [float(lims[0]), float(lims[1])] * n
    if lims.size == 4 and n == 2 and lims[0] < lims[1] and lims[2] < lims[3]:
        return [float(v) for v in lims]
    raise MatlabError("The interval must be [min max]" + (" or [xmin xmax ymin ymax]." if n == 2 else "."))


def _ezplot_core(interp, args, nargout, name: str):
    funcs, lims, rest = _fn_args(interp, args)
    rest, opts = _split_name_values(rest, _LINE_OPTS | {"meshdensity"})
    kw = _mpl_kwargs({k: v for k, v in opts.items() if k != "meshdensity"})
    ez = name == "ezplot"
    span = 2 * math.pi if ez else 5.0
    ax = _axes(interp, projection="rectilinear")
    (f, nin, label) = funcs[0]
    if len(funcs) >= 2 and name != "fimplicit":  # parametric curve x(t), y(t)
        a, b = (0.0, 2 * math.pi) if lims is None else _range(lims, span, 1)
        t = np.linspace(a, b, 400).reshape(1, -1)
        x, y = _eval_fn(interp, funcs[0][0], [t]), _eval_fn(interp, funcs[1][0], [t])
        objs = ax.plot(x.flatten(), y.flatten(), **kw)
        title = f"x = {funcs[0][2]}, y = {funcs[1][2]}"
        kind = "line"
    elif nin == 2 or name == "fimplicit":  # implicit curve f(x, y) = 0
        r = _range(lims, span)
        X, Y = np.meshgrid(np.linspace(r[0], r[1], 250), np.linspace(r[2], r[3], 250))
        Z = _eval_fn(interp, f, [X, Y])
        ckw = {"colors": [kw["color"]]} if "color" in kw else {"colors": ["C0"]}
        cs = ax.contour(X, Y, Z, levels=[0.0], **ckw)
        objs = [cs]
        title = label if "=" in label else f"{label} = 0"
        kind = "contour"
    else:
        a, b = _range(lims, span, 1)
        x = np.linspace(a, b, 400).reshape(1, -1)
        y = _eval_fn(interp, f, [x])
        objs = ax.plot(x.flatten(), np.where(np.isfinite(y), y, np.nan).flatten(), **kw)
        ax.set_xlim(a, b)
        title = label
        kind = "line"
    if ez:
        ax.set_title(title)
        if kind != "line" or len(funcs) >= 2:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.set_xlabel(f.params[0] if isinstance(f, FunctionHandle) and f.params else "x")
    _redraw(interp)
    return _ret(interp, kind, objs, nargout)


builtin("ezplot")(_with_axes_arg(lambda interp, args, nargout: _ezplot_core(interp, args, nargout, "ezplot")))
builtin("fimplicit")(_with_axes_arg(lambda interp, args, nargout: _ezplot_core(interp, args, nargout,
                                                                             "fimplicit")))


def _fsurface(interp, args, nargout, name: str):
    funcs, lims, rest = _fn_args(interp, args)
    rest, opts = _split_name_values(rest, {"meshdensity"})
    ez = name.startswith("ez")
    kind = {"fsurf": "surf", "ezsurf": "surf", "ezsurfc": "surf", "fmesh": "mesh", "ezmesh": "mesh",
            "fcontour": "contour", "ezcontour": "contour", "ezcontourf": "contourf"}[name]
    n = int(_scalar(opts["meshdensity"])) if "meshdensity" in opts else (120 if "contour" in kind else 50)
    r = _range(lims, 2 * math.pi if ez else 5.0)
    X, Y = np.meshgrid(np.linspace(r[0], r[1], max(n, 2)), np.linspace(r[2], r[3], max(n, 2)))
    if len(funcs) >= 3 and "contour" not in kind:  # parametric surface x(u,v), y(u,v), z(u,v)
        data = [_eval_fn(interp, fn[0], [X, Y]) for fn in funcs[:3]]
        label = ", ".join(fn[2] for fn in funcs[:3])
    else:
        data = [X, Y, _eval_fn(interp, funcs[0][0], [X, Y])]
        label = funcs[0][2]
    out = _surface(interp, data + rest, kind, nargout, handle_first=True)
    if ez:
        ax = _axes(interp, fresh=False)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        _redraw(interp)
    return out


for _name in ("fsurf", "fmesh", "fcontour", "ezsurf", "ezsurfc", "ezmesh", "ezcontour", "ezcontourf"):
    builtin(_name)(_with_axes_arg(lambda interp, args, nargout, _n=_name: _fsurface(interp, args, nargout, _n)))


@builtin("comet")
@_with_axes_arg
def _comet(interp, args, nargout):
    nums = [a for a in args if not is_text(a)]
    if not nums:
        raise MatlabError("comet needs data, e.g. comet(x, y).")
    if len(nums) >= 3 and np.asarray(nums[2]).size == 1:
        nums = nums[:2]  # the tail length has no meaning in a still picture
    return _do_plot(interp, nums[:2], "plot", nargout)


@builtin("comet3")
@_with_axes_arg
def _comet3(interp, args, nargout):
    nums = [a for a in args if not is_text(a)]
    if len(nums) < 3:
        raise MatlabError("comet3 needs x, y and z data.")
    return _plot3(interp, nums[:3], nargout)


def _grid_xyz(args, name: str):
    nums = [np.real(fnum(a)) for a in args if not is_text(a)]
    if not nums:
        raise MatlabError(f"{name} needs data.")
    if len(nums) < 3:
        Z = nums[0]
        X, Y = np.meshgrid(np.arange(1, Z.shape[1] + 1), np.arange(1, Z.shape[0] + 1))
    else:
        X, Y, Z = nums[:3]
        if X.shape != Z.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
        if X.shape != Z.shape:
            raise MatlabError(f"{name}: the sizes of X, Y and Z do not match.")
    return X, Y, Z


@builtin("waterfall")
@_with_axes_arg
def _waterfall(interp, args, nargout):
    X, Y, Z = _grid_xyz(args, "waterfall")
    ax = _axes(interp, projection="3d")
    coll = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0, linewidth=0.8)
    _redraw(interp)
    h = _wrap(interp, "surface", coll)
    h.extra.update({"XData": X, "YData": Y, "ZData": Z})
    return (h) if nargout > 0 else None


@builtin("ribbon")
@_with_axes_arg
def _ribbon(interp, args, nargout):
    nums = [np.real(fnum(a)) for a in args if not is_text(a)]
    if not nums:
        raise MatlabError("ribbon needs data, e.g. ribbon(Y).")
    width = 0.75
    if len(nums) >= 2 and nums[1].size == 1 and nums[0].size > 1:
        Y, width, x = nums[0], float(nums[1].flat[0]), None
    elif len(nums) >= 2:
        x, Y = nums[0], nums[1]
        if len(nums) > 2:
            width = float(nums[2].flat[0])
    else:
        x, Y = None, nums[0]
    if Y.shape[0] == 1:
        Y = Y.T
    m, n = Y.shape
    t = np.arange(1, m + 1, dtype=float) if x is None else x.flatten()
    if t.size != m:
        raise MatlabError("ribbon: x must have one element for each row of Y.")
    ax = _axes(interp, projection="3d")
    cmap = _current_cmap(interp)
    if isinstance(cmap, str):
        import matplotlib as mpl
        cmap = mpl.colormaps[cmap]
    hs = []
    for j in range(n):
        xs = np.column_stack([np.full(m, j + 1 - width / 2), np.full(m, j + 1 + width / 2)])
        ys = np.column_stack([t, t])
        zs = np.column_stack([Y[:, j], Y[:, j]])
        coll = ax.plot_surface(xs, ys, zs, color=cmap(j / max(n - 1, 1)), edgecolor="k", linewidth=0.3)
        hs.append(_wrap(interp, "surface", coll))
    _redraw(interp)
    return _handle_list(hs) if nargout > 0 else None


@builtin("copyobj")
def _copyobj(interp, args, nargout):
    from . import figfiles
    return figfiles.copyobj_builtin(interp, list(args), nargout)


@builtin("gtext")
def _gtext(interp, args, nargout):
    if not args:
        raise MatlabError("gtext needs the text to place.")
    ax = _axes(interp, fresh=False)
    label = "\n".join(_text_list(args[0]))
    plt = _plt()
    if not _noninteractive() and plt.isinteractive():
        pts = plt.ginput(1, timeout=0)
        x, y = pts[0] if pts else (np.mean(ax.get_xlim()), np.mean(ax.get_ylim()))
    else:
        interp.warn("gtext places text with a mouse click, which is not possible here; the text was put in "
                    "the middle of the axes. Use text(x, y, str) to choose the position.")
        x, y = float(np.mean(ax.get_xlim())), float(np.mean(ax.get_ylim()))
    t = ax.text(x, y, label)
    _redraw(interp)
    return _wrap(interp, "text", t) if nargout > 0 else None


@builtin("imshow")
@_with_axes_arg
def _imshow(interp, args, nargout):
    """Show an image: integer and logical classes use their own value range, like MATLAB."""
    rest, opts = _split_name_values(args, {"parent", "border", "initialmagnification", "displayrange",
                                           "colormap", "interpolation"})
    if not rest:
        raise MatlabError("imshow: an image or file name is required.")
    for a in rest[1:]:
        if is_text(a):
            raise MatlabError(f"imshow: the option '{text(a)}' is not supported.", "MATLAB:imshow:UnknownOption")
    img = rest[0]
    cmap = None
    limits = None
    if is_text(img):
        out = interp.call_named("imread", [img], 2, interp.frame)
        img = out[0]
        if len(out) > 1 and isinstance(out[1], np.ndarray) and out[1].size:
            cmap = out[1]
    if len(rest) > 1 and isinstance(rest[1], np.ndarray):
        second = rest[1]
        if second.ndim == 2 and second.shape[1] == 3 and second.shape[0] > 1:
            cmap = second                       # imshow(X, map)
        elif second.size == 2:
            limits = [float(v) for v in np.real(fnum(second)).flatten()]   # imshow(I, [lo hi])
        elif second.size == 0:
            limits = "auto"                     # imshow(I, [])
    if "displayrange" in opts and fnum(opts["displayrange"]).size == 2:
        limits = [float(v) for v in fnum(opts["displayrange"]).flatten()]
    if not isinstance(img, np.ndarray) or img.dtype.kind not in "biuf":
        raise MatlabError("imshow: the image must be a numeric or logical array.")
    ax = _axes(interp, projection="rectilinear")
    kw: dict = {"aspect": "equal", "origin": "upper", "interpolation": "nearest"}
    data = img
    if img.ndim == 3 and img.shape[2] == 3:      # truecolor
        if img.dtype.kind == "f":
            data = np.clip(img, 0, 1)
        elif img.dtype != np.uint8:
            data = img.astype(float) / float(np.iinfo(img.dtype).max)
    elif cmap is not None:                       # indexed image
        from matplotlib.colors import ListedColormap
        offset = 0 if img.dtype.kind in "iu" else 1   # integer indices are 0-based, double are 1-based
        data = np.clip(img.astype(int) - offset, 0, cmap.shape[0] - 1)
        kw.update(cmap=ListedColormap(np.clip(fnum(cmap), 0, 1)), vmin=0, vmax=cmap.shape[0] - 1)
    else:                                        # grayscale or logical
        data = img.astype(float)
        if limits == "auto":
            lo, hi = (float(np.nanmin(data)), float(np.nanmax(data))) if data.size else (0.0, 1.0)
        elif limits is not None:
            lo, hi = limits
        elif img.dtype.kind in "iu":
            info = np.iinfo(img.dtype)
            lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        kw.update(cmap="gray", vmin=lo, vmax=hi)
    if "colormap" in opts and not (img.ndim == 3 and img.shape[2] == 3):
        from matplotlib.colors import ListedColormap
        kw["cmap"] = ListedColormap(np.clip(np.real(fnum(opts["colormap"])).reshape(-1, 3), 0, 1))
    if "interpolation" in opts:
        how = text(opts["interpolation"]).lower()
        if how not in ("nearest", "bilinear"):
            raise MatlabError("imshow: 'Interpolation' must be 'nearest' or 'bilinear'.")
        kw["interpolation"] = how
    im = ax.imshow(data, **kw)
    keep_cdata(im, img)
    ax.set_xticks([])
    ax.set_yticks([])
    _redraw(interp)
    return _ret(interp, "image", [im], nargout)
