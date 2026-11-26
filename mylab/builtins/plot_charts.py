"""Chart types beyond the basic plotting functions.

Two kinds of function live here:

* functions that add ordinary objects to an axes (``bar3``, ``swarmchart``, ``bubblechart``, ``pareto``,
  ``compass``, ``voronoi``, ``xregion`` ...), and
* *charts*: objects such as ``heatmap``, ``histogram2``, ``stackedplot``, ``boxchart`` or ``parallelplot``
  that own their picture. A chart is a :class:`Chart` whose ``draw`` method builds the matplotlib artists
  from the chart's data and settings; changing a property through the handle redraws it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, char, is_char
from . import plotmore  # noqa: F401  (boxchart here replaces the alias defined there)
from . import plotting as P
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, text

_EMPTY = np.zeros((0, 0))


# -- shared helpers -------------------------------------------------------------------------------------

def _real(v) -> np.ndarray:
    return np.real(fnum(v)).astype(float)


def _vec(v) -> np.ndarray:
    return _real(v).flatten(order="F")


def _is_table(v) -> bool:
    from .table import TableObject
    return isinstance(v, TableObject)


def _column(tbl, spec, what: str):
    """(name, values) of one table variable given by name or number."""
    if is_text(spec):
        name = text(spec)
        if name not in tbl.names:
            raise MatlabError(f"Unrecognized table variable name '{name}'.")
        return name, tbl.columns[tbl.names.index(name)]
    k = int(_real(spec).flat[0]) - 1
    if not 0 <= k < len(tbl.names):
        raise MatlabError(f"{what}: the table variable number is out of range.")
    return tbl.names[k], tbl.columns[k]


def _labels_of(v) -> list[str]:
    """Text labels for the values of a grouping or category variable, one per element."""
    if hasattr(v, "categories") and hasattr(v, "element_texts"):
        return [str(s) for s in v.element_texts().flatten(order="F")]
    if isinstance(v, (Cell, StringArray, MString)) or is_char(v):
        return P._text_list(v)
    if hasattr(v, "element_texts"):
        return [str(s) for s in v.element_texts().flatten(order="F")]
    return [P._fmt_num(x) for x in _vec(v)]


def _unique_labels(v) -> list[str]:
    """The distinct values of a category variable, in the order a chart lists them."""
    if hasattr(v, "categories") and hasattr(v, "codes"):
        used = set(int(c) for c in v.codes.flatten() if c >= 0)
        return [c for k, c in enumerate(v.categories) if k in used]
    labels = _labels_of(v)
    if isinstance(v, np.ndarray) and not is_char(v):
        vals = sorted(set(float(x) for x in _vec(v) if np.isfinite(x)))
        return [P._fmt_num(x) for x in vals]
    return sorted(set(labels))


def _axes_from(interp, args: list, fresh: bool = True, projection: str | None = "rectilinear"):
    """(axes, remaining args): handles a leading axes handle and a 'Parent' option."""
    args = list(args)
    st = P._state(interp)
    target = None
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in ("axes", "polaraxes", "geoaxes"):
        target = args.pop(0).obj
    elif args and isinstance(args[0], P.GraphicsHandle) and args[0].kind == "figure":
        P._figure(interp, P._fig_number(interp, args.pop(0).obj))
    k = 0
    while k < len(args) - 1:
        if is_text(args[k]) and text(args[k]).lower() == "parent" and isinstance(args[k + 1], P.GraphicsHandle):
            ph = args[k + 1]
            if ph.kind in ("axes", "polaraxes", "geoaxes"):
                target = ph.obj
            elif ph.kind == "figure":
                P._figure(interp, P._fig_number(interp, ph.obj))
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


def _options(args: list, known: tuple, what: str) -> tuple[list, dict]:
    """Split trailing Name,Value options (matched without regard to case); unknown names are errors."""
    low = {k.lower(): k for k in known}
    out, opts = [], {}
    k = 0
    args = list(args)
    while k < len(args):
        a = args[k]
        if is_text(a) and text(a).lower() in low:
            if k + 1 >= len(args):
                raise MatlabError(f"{what}: the option '{text(a)}' needs a value.")
            opts[low[text(a).lower()]] = args[k + 1]
            k += 2
            continue
        out.append(a)
        k += 1
    return out, opts


def _no_text_left(args: list, what: str, allowed: tuple = ()) -> None:
    for a in args:
        if is_text(a) and text(a).lower() not in allowed:
            raise MatlabError(f"{what}: unknown option '{text(a)}'.")


def _cmap_obj(interp, cmap=None):
    import matplotlib as mpl
    c = cmap if cmap is not None else P._current_cmap(interp)
    return mpl.colormaps[c] if isinstance(c, str) else c


def _next_color(ax):
    return ax._get_lines.get_next_color()


def _quantiles(x: np.ndarray, q) -> np.ndarray:
    """Quantiles the way MATLAB's quantile computes them (linear between the (i-0.5)/n points)."""
    x = np.sort(x[np.isfinite(x)])
    n = x.size
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if n == 0:
        return np.full(q.shape, np.nan)
    pos = np.clip(q * n + 0.5, 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n)
    frac = pos - lo
    return x[lo - 1] * (1 - frac) + x[hi - 1] * frac


def _kde(samples: np.ndarray, points: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Gaussian kernel density estimate of ``samples`` at ``points`` (Silverman's bandwidth by default)."""
    x = samples[np.isfinite(samples)]
    n = x.size
    if n == 0:
        return np.zeros_like(points)
    if bandwidth is None:
        sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
        q75, q25 = np.percentile(x, [75, 25]) if n > 1 else (0.0, 0.0)
        sig = min(sd, (q75 - q25) / 1.349) if q75 > q25 else sd
        bandwidth = 0.9 * sig * n ** (-0.2) if sig > 0 else max(abs(float(x[0])) * 0.01, 1e-3)
    z = (points.reshape(-1, 1) - x.reshape(1, -1)) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bandwidth * math.sqrt(2 * math.pi))


# -- chart objects -----------------------------------------------------------------------------------------

class Chart:
    """A chart made of several matplotlib artists, redrawn from its data when a property changes."""

    kind = "chart"

    def __init__(self, interp, ax):
        self.interp = interp
        self.axes = ax
        self.artists: list = []
        self.visible = True

    @property
    def figure(self):
        return self.axes.figure

    def add(self, *artists):
        for a in artists:
            if isinstance(a, (list, tuple)):
                self.add(*a)
            elif a is not None:
                self.artists.extend(P._artists(a))
        return artists[0] if len(artists) == 1 else artists

    def clear(self) -> None:
        for a in self.artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        del self.artists[:]

    def draw(self) -> None:
        raise NotImplementedError

    def redraw(self) -> None:
        self.clear()
        self.draw()
        if not self.visible:
            self.set_visible(False)

    # the artist protocol the handle code uses
    def remove(self) -> None:
        self.clear()

    def get_visible(self) -> bool:
        return self.visible

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        for a in self.artists:
            if hasattr(a, "set_visible"):
                a.set_visible(self.visible)


def _register(interp, chart: Chart):
    """The handle of a chart, listed among the children of its axes."""
    h = P._wrap(interp, chart.kind, chart)
    h.extra["_child_of"] = chart.axes
    h.extra["_artists"] = chart.artists
    h.extra["axes"] = chart.axes
    h.extra["figure"] = chart.axes.figure
    return h


def _attr(name: str, conv=None, redraw: bool = True, out=None) -> tuple:
    """Chart property stored in attribute ``name``."""
    def g(h):
        v = getattr(h.obj, name)
        if out is not None:
            return out(v)
        if isinstance(v, str):
            return char(v)
        if isinstance(v, bool):
            return P._oo(v)
        if isinstance(v, (int, float)):
            return P._num_val(v)
        if isinstance(v, list):
            return Cell.from_list([char(str(x)) for x in v])
        if isinstance(v, tuple):
            return P._row(v)
        return v

    def s(h, v):
        setattr(h.obj, name, conv(v) if conv is not None else v)
        if redraw:
            h.obj.redraw()
    return g, s


def _ro(fn) -> tuple:
    return (lambda h: fn(h.obj)), None


def _choice_conv(name: str, options: tuple):
    def conv(v):
        if isinstance(v, np.ndarray) and not is_char(v) and v.size == 1 and {"on", "off"} <= set(options):
            return "on" if bool(np.real(v).flat[0]) else "off"
        s = text(v).lower()
        if s not in options:
            raise MatlabError(f"{name} must be one of: {', '.join(repr(o) for o in options)}.")
        return s
    return conv


def _color_conv(v):
    from matplotlib.colors import to_rgb
    try:
        return tuple(float(c) for c in to_rgb(P._color(v)))
    except ValueError:
        raise MatlabError("Expected a colour name, a hexadecimal colour code or an RGB triplet.") from None


def _color_out(c):
    return P._rgb(c)


def _num_conv(name: str, lo: float = -math.inf, hi: float = math.inf):
    def conv(v):
        x = _real(v)
        if x.size != 1 or not lo <= float(x.flat[0]) <= hi:
            raise MatlabError(f"{name} must be a number" + (f" between {lo:g} and {hi:g}." if math.isfinite(lo)
                                                           and math.isfinite(hi) else "."))
        return float(x.flat[0])
    return conv


def _label_prop(which: str) -> tuple:
    """Title / XLabel / YLabel of a chart: the text of its axes label."""
    def target(h):
        ax = h.obj.label_axes() if hasattr(h.obj, "label_axes") else h.obj.axes
        return {"Title": ax.title, "XLabel": ax.xaxis.label, "YLabel": ax.yaxis.label}[which]

    def g(h):
        t = target(h)
        raw = getattr(t, "_mylab_raw", None)
        return char(raw if raw is not None else t.get_text())

    def s(h, v):
        t = target(h)
        if which == "Title":
            t.set_fontweight("bold")
        P.set_text(h.interp, t, P._label_text(v))
    return g, s


def _chart_setup(kind: str, cls: str, props: dict, display: list) -> None:
    P._CLASS[kind] = cls
    P._TYPE[kind] = kind
    P._PROPS[kind] = props
    P._DISPLAY[kind] = display


def _apply_opts(h, opts: dict, skip: tuple = ()) -> None:
    for name, value in opts.items():
        if name in skip:
            continue
        P._set_prop(h, name, value)


# -- heatmap ---------------------------------------------------------------------------------------------

_AGG = {"count": None, "mean": np.nanmean, "median": np.nanmedian, "sum": np.nansum, "max": np.nanmax,
        "min": np.nanmin, "none": None}


class Heatmap(Chart):
    kind = "heatmap"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.data = np.zeros((0, 0))       # colour values in the order of ydata (rows) and xdata (columns)
        self.xdata: list[str] = []
        self.ydata: list[str] = []
        self.xdisplay: list[str] | None = None
        self.ydisplay: list[str] | None = None
        self.xlabels: list[str] | None = None
        self.ylabels: list[str] | None = None
        self.cmap = None
        self.climits: tuple | None = None
        self.colorbar = "on"
        self.cell_color: Any = "auto"
        self.cell_format = "%0.4g"
        self.font_size = 10.0
        self.grid = "on"
        self.missing_color = (0.15, 0.15, 0.15)
        self.missing_label = "NaN"
        self.scaling = "scaled"
        self.table = None
        self.xvar = self.yvar = self.cvar = ""
        self.method = "count"
        self.cbar = None

    def from_table(self) -> None:
        tbl = self.table
        _, xc = _column(tbl, char(self.xvar), "heatmap")
        _, yc = _column(tbl, char(self.yvar), "heatmap")
        xl, yl = _labels_of(xc), _labels_of(yc)
        self.xdata, self.ydata = _unique_labels(xc), _unique_labels(yc)
        vals = None
        if self.cvar:
            vals = _vec(_column(tbl, char(self.cvar), "heatmap")[1])
        method = self.method if self.cvar else "count"
        out = np.full((len(self.ydata), len(self.xdata)), 0.0 if method == "count" else np.nan)
        xi = {s: k for k, s in enumerate(self.xdata)}
        yi = {s: k for k, s in enumerate(self.ydata)}
        cells: dict = {}
        for k, (a, b) in enumerate(zip(xl, yl)):
            if a in xi and b in yi:
                cells.setdefault((yi[b], xi[a]), []).append(k)
        for (r, c), idx in cells.items():
            if method == "count":
                out[r, c] = len(idx) if vals is None else int(np.sum(np.isfinite(vals[idx])))
            elif method == "none":
                if len(idx) > 1:
                    raise MatlabError("heatmap: with ColorMethod 'none' each pair of x and y values may occur "
                                      "only once.")
                out[r, c] = vals[idx[0]]
            else:
                good = vals[idx][np.isfinite(vals[idx])]
                out[r, c] = _AGG[method](good) if good.size else np.nan
        self.data = out

    def shown(self):
        xs = self.xdisplay if self.xdisplay is not None else self.xdata
        ys = self.ydisplay if self.ydisplay is not None else self.ydata
        ci = [self.xdata.index(s) for s in xs]
        ri = [self.ydata.index(s) for s in ys]
        return xs, ys, self.data[np.ix_(ri, ci)] if ri and ci else np.zeros((len(ri), len(ci)))

    def colormap(self):
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        if self.cmap is not None:
            return ListedColormap(np.clip(self.cmap, 0, 1))
        return LinearSegmentedColormap.from_list("mylab_heat", [(0.93, 0.96, 1.0), (0.0, 0.447, 0.741)])

    def draw(self) -> None:
        ax = self.axes
        xs, ys, C = self.shown()
        nr, nc = C.shape
        cmap = self.colormap().copy()
        cmap.set_bad(self.missing_color)
        finite = C[np.isfinite(C)]
        if self.climits is not None:
            lo, hi = self.climits
        elif finite.size:
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        img = np.ma.masked_invalid(C)
        if self.scaling == "log":
            from matplotlib.colors import LogNorm
            im = ax.imshow(img, cmap=cmap, norm=LogNorm(vmin=max(lo, 1e-300), vmax=hi), aspect="auto",
                           origin="upper", extent=(0.5, nc + 0.5, nr + 0.5, 0.5), interpolation="nearest")
        else:
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", origin="upper",
                           extent=(0.5, nc + 0.5, nr + 0.5, 0.5), interpolation="nearest")
        self.add(im)
        self.image = im
        ax.set_xlim(0.5, nc + 0.5)
        ax.set_ylim(nr + 0.5, 0.5)
        xl = self.xlabels if self.xlabels is not None and len(self.xlabels) == nc else xs
        yl = self.ylabels if self.ylabels is not None and len(self.ylabels) == nr else ys
        _set_tick_text(ax, "x", range(1, nc + 1), xl, fontsize=self.font_size)
        _set_tick_text(ax, "y", range(1, nr + 1), yl, fontsize=self.font_size)
        ax.tick_params(length=0)
        if self.grid == "on":
            for k in range(nc + 1):
                self.add(ax.plot([k + 0.5, k + 0.5], [0.5, nr + 0.5], color="white", linewidth=0.8, zorder=3))
            for k in range(nr + 1):
                self.add(ax.plot([0.5, nc + 0.5], [k + 0.5, k + 0.5], color="white", linewidth=0.8, zorder=3))
        show_text = not (isinstance(self.cell_color, str) and self.cell_color == "none")
        if show_text and nr * nc <= 2500:
            from .plot_rulers import format_number
            for r in range(nr):
                for c in range(nc):
                    v = C[r, c]
                    if np.isfinite(v):
                        s = format_number(self.cell_format, float(v))
                        if isinstance(self.cell_color, str):
                            frac = (v - lo) / (hi - lo)
                            rgb = cmap(min(max(frac, 0.0), 1.0))
                            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                            color = "black" if lum > 0.5 else "white"
                        else:
                            color = self.cell_color
                    else:
                        s, color = self.missing_label, "white" if not isinstance(self.cell_color, tuple) \
                            else self.cell_color
                    self.add(ax.text(c + 1, r + 1, s, ha="center", va="center", color=color,
                                     fontsize=self.font_size, zorder=4))
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except (ValueError, AttributeError, KeyError):
                pass
            self.cbar = None
            ax._mylab_colorbar = None
        if self.colorbar == "on":
            self.cbar = ax.figure.colorbar(im, ax=ax)
            self.cbar.ax.tick_params(labelsize=self.font_size)
            self.cbar._mylab_parent_ax = ax
            ax._mylab_colorbar = self.cbar

    def remove(self) -> None:
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except (ValueError, AttributeError, KeyError):
                pass
            self.cbar = None
            self.axes._mylab_colorbar = None
        super().remove()

    def sort(self, which: str, args: list) -> list:
        """Reorder the displayed columns (which='x') or rows by label or by the values in a row/column."""
        xs, ys, C = self.shown()
        names = xs if which == "x" else ys
        other = ys if which == "x" else xs
        direction, by = "ascend", None
        missing = "last"
        k = 0
        while k < len(args):
            a = args[k]
            if is_text(a) and text(a).lower() in ("ascend", "descend"):
                direction = text(a).lower()
            elif is_text(a) and text(a).lower() == "missingplacement" and k + 1 < len(args):
                missing = text(args[k + 1]).lower()
                k += 1
            elif is_text(a) or isinstance(a, (Cell, StringArray)):
                by = P._text_list(a)
            else:
                raise MatlabError(f"sort{which}: unknown option.")
            k += 1
        if by is None:
            try:
                keys = [float(s) for s in names]
            except ValueError:
                keys = list(names)
            order = sorted(range(len(names)), key=lambda i: keys[i], reverse=direction == "descend")
        else:
            for b in by:
                if b not in other:
                    raise MatlabError(f"sort{which}: '{b}' is not a {'row' if which == 'x' else 'column'} of "
                                      "the heatmap.")
            rows = [other.index(b) for b in by]

            def key(i):
                vals = [C[r, i] if which == "x" else C[i, r] for r in rows]
                miss = [not np.isfinite(v) for v in vals]
                sign = -1.0 if direction == "descend" else 1.0
                return tuple((m if missing == "last" else not m, 0.0 if m else sign * v)
                             for m, v in zip(miss, vals))
            order = sorted(range(len(names)), key=key)
        new = [names[i] for i in order]
        if which == "x":
            self.xdisplay = new
        else:
            self.ydisplay = new
        self.redraw()
        return new


def _hm_display_set(which: str):
    def s(h, v):
        names = P._text_list(v)
        full = h.obj.xdata if which == "x" else h.obj.ydata
        for n in names:
            if n not in full:
                raise MatlabError(f"'{n}' is not one of the {which.upper()}Data values of the heatmap.")
        setattr(h.obj, f"{which}display", names)
        h.obj.redraw()
    return s


def _hm_labels_set(which: str):
    def s(h, v):
        names = P._text_list(v)
        shown = h.obj.shown()[0 if which == "x" else 1]
        if len(names) != len(shown):
            raise MatlabError(f"{which.upper()}DisplayLabels needs one label for each of the {len(shown)} "
                              f"{'columns' if which == 'x' else 'rows'} shown.")
        setattr(h.obj, f"{which}labels", names)
        h.obj.redraw()
    return s


def _hm_data_set(which: str):
    def s(h, v):
        hm = h.obj
        if hm.table is not None:
            raise MatlabError(f"{which.upper()}Data comes from the table; change the table variable instead.")
        names = P._text_list(v)
        n = hm.data.shape[1 if which == "x" else 0]
        if len(names) != n or len(set(names)) != n:
            raise MatlabError(f"{which.upper()}Data needs {n} different values.")
        setattr(hm, f"{which}data", names)
        setattr(hm, f"{which}display", None)
        setattr(hm, f"{which}labels", None)
        hm.redraw()
    return s


def _hm_color_data_set(h, v):
    hm = h.obj
    if hm.table is not None:
        raise MatlabError("ColorData comes from the table; it cannot be set.")
    C = _real(v)
    if C.ndim != 2:
        raise MatlabError("ColorData must be a matrix.")
    if C.shape != hm.data.shape:
        hm.xdata = [str(k) for k in range(1, C.shape[1] + 1)]
        hm.ydata = [str(k) for k in range(1, C.shape[0] + 1)]
        hm.xdisplay = hm.ydisplay = hm.xlabels = hm.ylabels = None
    hm.data = C
    hm.redraw()


def _hm_table_prop(name: str):
    def g(h):
        return char(getattr(h.obj, name))

    def s(h, v):
        hm = h.obj
        if hm.table is None:
            raise MatlabError("This heatmap was not made from a table.")
        setattr(hm, name, text(v))
        hm.xdisplay = hm.ydisplay = hm.xlabels = hm.ylabels = None
        hm.from_table()
        hm.redraw()
    return g, s


def _hm_method_set(h, v):
    m = text(v).lower()
    if m not in _AGG:
        raise MatlabError("ColorMethod must be 'count', 'mean', 'median', 'sum', 'max', 'min' or 'none'.")
    hm = h.obj
    hm.method = m
    if hm.table is not None:
        hm.from_table()
    hm.redraw()


def _hm_cmap_set(h, v):
    h.obj.cmap = P._cmap_matrix(P._to_cmap(v), 256) if is_text(v) else np.clip(_real(v), 0, 1)
    if h.obj.cmap.ndim != 2 or h.obj.cmap.shape[1] != 3:
        raise MatlabError("Colormap must be an m-by-3 matrix of RGB values.")
    h.obj.redraw()


def _hm_limits_get(h):
    if h.obj.climits is not None:
        return P._row(h.obj.climits)
    C = h.obj.shown()[2]
    f = C[np.isfinite(C)]
    return P._row([f.min(), f.max()] if f.size else [0, 1])


def _hm_limits_set(h, v):
    lim = _vec(v)
    if lim.size != 2 or not lim[0] < lim[1]:
        raise MatlabError("ColorLimits must be [min max] with min < max.")
    h.obj.climits = (float(lim[0]), float(lim[1]))
    h.obj.redraw()


def _hm_cell_color_conv(v):
    if is_text(v) and text(v).lower() in ("auto", "none"):
        return text(v).lower()
    return _color_conv(v)


def _hm_fmt_conv(v):
    from .plot_rulers import _SPEC
    s = text(v)
    if not _SPEC.search(s.replace("%%", "")):
        raise MatlabError("CellLabelFormat must be a number format such as '%0.2f' or '%d'.")
    return s


_chart_setup("heatmap", "matlab.graphics.chart.HeatmapChart", {
    "Title": _label_prop("Title"), "XLabel": _label_prop("XLabel"), "YLabel": _label_prop("YLabel"),
    "ColorData": (lambda h: h.obj.data, _hm_color_data_set),
    "ColorDisplayData": _ro(lambda c: c.shown()[2]),
    "XData": (lambda h: P._cell_column([char(s) for s in h.obj.xdata]), _hm_data_set("x")),
    "YData": (lambda h: P._cell_column([char(s) for s in h.obj.ydata]), _hm_data_set("y")),
    "XDisplayData": (lambda h: P._cell_column([char(s) for s in h.obj.shown()[0]]), _hm_display_set("x")),
    "YDisplayData": (lambda h: P._cell_column([char(s) for s in h.obj.shown()[1]]), _hm_display_set("y")),
    "XDisplayLabels": (lambda h: P._cell_column([char(s) for s in (h.obj.xlabels or h.obj.shown()[0])]),
                       _hm_labels_set("x")),
    "YDisplayLabels": (lambda h: P._cell_column([char(s) for s in (h.obj.ylabels or h.obj.shown()[1])]),
                       _hm_labels_set("y")),
    "Colormap": (lambda h: h.obj.cmap if h.obj.cmap is not None
                 else np.round(np.asarray(h.obj.colormap()(np.linspace(0, 1, 256)))[:, :3], 4), _hm_cmap_set),
    "ColorLimits": (_hm_limits_get, _hm_limits_set),
    "ColorScaling": _attr("scaling", _choice_conv("ColorScaling", ("scaled", "scaledrows", "scaledcolumns", "log"))),
    "ColorbarVisible": _attr("colorbar", _choice_conv("ColorbarVisible", ("on", "off"))),
    "CellLabelColor": _attr("cell_color", _hm_cell_color_conv,
                            out=lambda c: char(c) if isinstance(c, str) else P._rgb(c)),
    "CellLabelFormat": _attr("cell_format", _hm_fmt_conv),
    "FontSize": _attr("font_size", _num_conv("FontSize", 1, 1000)),
    "GridVisible": _attr("grid", _choice_conv("GridVisible", ("on", "off"))),
    "MissingDataColor": _attr("missing_color", _color_conv, out=_color_out),
    "MissingDataLabel": _attr("missing_label", lambda v: text(v)),
    "SourceTable": (lambda h: h.obj.table if h.obj.table is not None else _EMPTY, None),
    "XVariable": _hm_table_prop("xvar"), "YVariable": _hm_table_prop("yvar"),
    "ColorVariable": _hm_table_prop("cvar"),
    "ColorMethod": (lambda h: char(h.obj.method), _hm_method_set),
    "Position": (lambda h: P._row(h.obj.axes.get_position().bounds),
                 lambda h, v: h.obj.axes.set_position([float(x) for x in _vec(v)[:4]])),
}, ["XData", "YData", "ColorData", "Title", "XLabel", "YLabel", "ColorLimits", "ColorbarVisible"])

_HEATMAP_OPTS = ("Title", "XLabel", "YLabel", "Colormap", "ColorLimits", "ColorbarVisible", "CellLabelColor",
                 "CellLabelFormat", "FontSize", "GridVisible", "MissingDataColor", "MissingDataLabel",
                 "ColorVariable", "ColorMethod", "XDisplayLabels", "YDisplayLabels", "XDisplayData",
                 "YDisplayData", "ColorScaling", "Position", "Visible", "Tag", "XData", "YData")


@builtin("heatmap")
def _heatmap(interp, args, nargout):
    """heatmap(C), heatmap(x, y, C) or heatmap(tbl, xvar, yvar): a grid of coloured, labelled cells."""
    ax, args = _axes_from(interp, list(args))
    args, opts = _options(args, _HEATMAP_OPTS, "heatmap")
    hm = Heatmap(interp, ax)
    if not args:
        raise MatlabError("heatmap needs data: heatmap(C), heatmap(x, y, C) or heatmap(tbl, xvar, yvar).")
    if _is_table(args[0]):
        if len(args) != 3:
            raise MatlabError("heatmap(tbl, xvar, yvar) needs the two table variables for the columns and rows.")
        hm.table = args[0]
        hm.xvar, _ = _column(hm.table, args[1], "heatmap")
        hm.yvar, _ = _column(hm.table, args[2], "heatmap")
        if "ColorVariable" in opts:
            hm.cvar, _ = _column(hm.table, opts.pop("ColorVariable"), "heatmap")
            hm.method = "mean"
        if "ColorMethod" in opts:
            m = text(opts.pop("ColorMethod")).lower()
            if m not in _AGG:
                raise MatlabError("ColorMethod must be 'count', 'mean', 'median', 'sum', 'max', 'min' or 'none'.")
            hm.method = m
        hm.from_table()
        ax.set_xlabel(hm.xvar)
        ax.set_ylabel(hm.yvar)
        if hm.cvar:
            title = f"{hm.method.capitalize()} of {hm.cvar}" if hm.method not in ("none",) else hm.cvar
        else:
            title = f"Count of {hm.yvar} vs. {hm.xvar}"
        ax.set_title(title, fontweight="bold")
    else:
        if "ColorVariable" in opts or "ColorMethod" in opts:
            raise MatlabError("heatmap: ColorVariable and ColorMethod apply to table input.")
        if len(args) == 1:
            C = _real(args[0])
            xs = [str(k) for k in range(1, C.shape[1] + 1)]
            ys = [str(k) for k in range(1, C.shape[0] + 1)]
        elif len(args) == 3:
            xs, ys, C = _labels_of(args[0]), _labels_of(args[1]), _real(args[2])
        else:
            raise MatlabError("heatmap expects heatmap(C) or heatmap(xvalues, yvalues, C).")
        if C.ndim != 2:
            raise MatlabError("heatmap: the colour data must be a matrix.")
        if len(xs) != C.shape[1] or len(ys) != C.shape[0]:
            raise MatlabError("heatmap: give one x value for each column and one y value for each row of the "
                              "data.")
        if len(set(xs)) != len(xs) or len(set(ys)) != len(ys):
            raise MatlabError("heatmap: the x values and the y values must not repeat.")
        hm.data, hm.xdata, hm.ydata = C, xs, ys
    hm.draw()
    h = _register(interp, hm)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


def _sort_heatmap(which: str):
    def f(interp, args, nargout):
        if not args or not isinstance(args[0], P.GraphicsHandle) or args[0].kind != "heatmap":
            raise MatlabError(f"sort{which} needs a heatmap chart, e.g. sort{which}(h, 'descend').")
        new = args[0].obj.sort(which, list(args[1:]))
        P._redraw(interp)
        return P._cell_column([char(s) for s in new]) if nargout > 0 else None
    return f


builtin("sortx")(_sort_heatmap("x"))
builtin("sorty")(_sort_heatmap("y"))


# -- histogram2 and histcounts2 --------------------------------------------------------------------------------

def _edges2(x: np.ndarray, opts: dict, which: str, nbins=None) -> np.ndarray:
    key = which.upper()
    if f"{key}BinEdges" in opts:
        e = _vec(opts[f"{key}BinEdges"])
        if e.size < 2 or np.any(np.diff(e) <= 0):
            raise MatlabError(f"{key}BinEdges must be a vector of increasing values.")
        return e
    limits = None
    if f"{key}BinLimits" in opts:
        limits = _vec(opts[f"{key}BinLimits"])
        if limits.size != 2 or not limits[0] < limits[1]:
            raise MatlabError(f"{key}BinLimits must be [min max] with min < max.")
        x = x[(x >= limits[0]) & (x <= limits[1])]
    width = None
    if "BinWidth" in opts:
        w = _vec(opts["BinWidth"])
        width = float(w[0 if which == "x" or w.size == 1 else 1])
    if "NumBins" in opts:
        n = _vec(opts["NumBins"])
        nbins = int(n[0 if which == "x" or n.size == 1 else 1])
    method = text(opts["BinMethod"]).lower() if "BinMethod" in opts else None
    if limits is not None and nbins is None and width is None:
        nbins = max(len(P._hist_edges(x, method=method)) - 1, 1) if x.size else 1
    return P._hist_edges(x, nbins=nbins, width=width, limits=limits, method=method)


def _counts2(x, y, ex, ey):
    good = np.isfinite(x) & np.isfinite(y)
    H, _, _ = np.histogram2d(x[good], y[good], bins=[ex, ey])
    return H


def _normalize2(H, ex, ey, norm: str):
    n = max(H.sum(), 1.0)
    area = np.outer(np.diff(ex), np.diff(ey))
    if norm == "probability":
        return H / n
    if norm == "percentage":
        return 100 * H / n
    if norm == "countdensity":
        return H / area
    if norm == "pdf":
        return H / (n * area)
    if norm == "cumcount":
        return np.cumsum(np.cumsum(H, axis=0), axis=1)
    if norm == "cdf":
        return np.cumsum(np.cumsum(H, axis=0), axis=1) / n
    return H


class Histogram2(Chart):
    kind = "histogram2"

    def __init__(self, interp, ax, x, y, ex, ey):
        super().__init__(interp, ax)
        self.x, self.y, self.ex, self.ey = x, y, ex, ey
        self.norm = "count"
        self.style = "bar3"
        self.face: Any = "auto"
        self.edge: Any = (0.15, 0.15, 0.15)
        self.alpha = 1.0
        self.show_empty = "off"
        self.counts_given = None

    def counts(self):
        return self.counts_given if self.counts_given is not None else _counts2(self.x, self.y, self.ex, self.ey)

    def values(self):
        return _normalize2(self.counts(), self.ex, self.ey, self.norm)

    def draw(self) -> None:
        ax = self.axes
        V = self.values()
        cmap = _cmap_obj(self.interp)
        if self.style == "tile":
            data = V.T if self.show_empty == "on" else np.ma.masked_where(V.T == 0, V.T)
            mesh = ax.pcolormesh(self.ex, self.ey, data, cmap=cmap, shading="flat",
                                 edgecolors=self.edge if not isinstance(self.edge, str) else "none",
                                 linewidth=0.3, alpha=self.alpha)
            self.add(mesh)
            ax.set_xlim(self.ex[0], self.ex[-1])
            ax.set_ylim(self.ey[0], self.ey[-1])
            return
        xs, ys = np.meshgrid(self.ex[:-1], self.ey[:-1], indexing="ij")
        dx = np.repeat(np.diff(self.ex).reshape(-1, 1), self.ey.size - 1, axis=1)
        dy = np.repeat(np.diff(self.ey).reshape(1, -1), self.ex.size - 1, axis=0)
        keep = np.ones(V.shape, dtype=bool) if self.show_empty == "on" else V > 0
        if not keep.any():
            return
        top = float(V.max()) if V.max() > 0 else 1.0
        if isinstance(self.face, str) and self.face == "flat":
            colors = cmap(V[keep] / top)
        elif isinstance(self.face, str):
            from matplotlib.colors import to_rgba
            if not hasattr(self, "auto_color"):
                self.auto_color = to_rgba(_next_color(ax))
            colors = self.auto_color
        else:
            colors = self.face
        bars = ax.bar3d(xs[keep], ys[keep], np.zeros(int(keep.sum())), dx[keep], dy[keep], V[keep], color=colors,
                        edgecolor=None if isinstance(self.edge, str) else self.edge, linewidth=0.3,
                        alpha=self.alpha, shade=True)
        self.add(bars)
        ax.set_xlim(self.ex[0], self.ex[-1])
        ax.set_ylim(self.ey[0], self.ey[-1])


def _h2_edges_set(which: str):
    def s(h, v):
        e = _vec(v)
        if e.size < 2 or np.any(np.diff(e) <= 0):
            raise MatlabError(f"{which.upper()}BinEdges must be a vector of increasing values.")
        setattr(h.obj, f"e{which}", e)
        h.obj.counts_given = None
        h.obj.redraw()
    return s


def _h2_numbins_set(h, v):
    n = _vec(v)
    if n.size not in (1, 2) or np.any(n < 1):
        raise MatlabError("NumBins must be a positive whole number or [nx ny].")
    c = h.obj
    c.ex = P._hist_edges(c.x[np.isfinite(c.x)], nbins=int(n[0]))
    c.ey = P._hist_edges(c.y[np.isfinite(c.y)], nbins=int(n[-1]))
    c.counts_given = None
    c.redraw()


def _h2_width_set(h, v):
    w = _vec(v)
    if w.size not in (1, 2) or np.any(w <= 0):
        raise MatlabError("BinWidth must be a positive number or [wx wy].")
    c = h.obj
    c.ex = P._hist_edges(c.x[np.isfinite(c.x)], width=float(w[0]))
    c.ey = P._hist_edges(c.y[np.isfinite(c.y)], width=float(w[-1]))
    c.counts_given = None
    c.redraw()


def _h2_style_set(h, v):
    s = text(v).lower()
    if s not in ("bar3", "tile"):
        raise MatlabError("DisplayStyle must be 'bar3' or 'tile'.")
    c = h.obj
    if (s == "bar3") != hasattr(c.axes, "get_zlim"):
        raise MatlabError("DisplayStyle cannot be changed after the histogram is drawn; draw it again with "
                          "histogram2(..., 'DisplayStyle', '" + s + "').")
    c.style = s
    c.redraw()


def _h2_face_conv(v):
    if is_text(v) and text(v).lower() in ("auto", "flat", "none"):
        return text(v).lower()
    return _color_conv(v)


_chart_setup("histogram2", "matlab.graphics.chart.primitive.Histogram2", {
    "Data": _ro(lambda c: np.column_stack([c.x, c.y])),
    "Values": _ro(lambda c: c.values()),
    "BinCounts": _ro(lambda c: c.counts()),
    "XBinEdges": (lambda h: P._row(h.obj.ex), _h2_edges_set("x")),
    "YBinEdges": (lambda h: P._row(h.obj.ey), _h2_edges_set("y")),
    "NumBins": (lambda h: P._row([h.obj.ex.size - 1, h.obj.ey.size - 1]), _h2_numbins_set),
    "BinWidth": (lambda h: P._row([np.diff(h.obj.ex)[0], np.diff(h.obj.ey)[0]]), _h2_width_set),
    "XBinLimits": _ro(lambda c: P._row([c.ex[0], c.ex[-1]])),
    "YBinLimits": _ro(lambda c: P._row([c.ey[0], c.ey[-1]])),
    "Normalization": _attr("norm", lambda v: P._check_norm(text(v))),
    "DisplayStyle": (lambda h: char(h.obj.style), _h2_style_set),
    "FaceColor": _attr("face", _h2_face_conv, out=lambda c: char(c) if isinstance(c, str) else P._rgb(c)),
    "EdgeColor": _attr("edge", lambda v: "none" if is_text(v) and text(v).lower() == "none" else _color_conv(v),
                       out=lambda c: char(c) if isinstance(c, str) else P._rgb(c)),
    "FaceAlpha": _attr("alpha", _num_conv("FaceAlpha", 0, 1)),
    "ShowEmptyBins": _attr("show_empty", _choice_conv("ShowEmptyBins", ("on", "off"))),
}, ["Data", "Values", "NumBins", "XBinEdges", "YBinEdges", "BinWidth", "Normalization", "DisplayStyle"])

_H2_OPTS = ("NumBins", "XBinEdges", "YBinEdges", "BinWidth", "XBinLimits", "YBinLimits", "BinMethod",
            "Normalization", "DisplayStyle", "FaceColor", "EdgeColor", "FaceAlpha", "ShowEmptyBins", "BinCounts",
            "Tag", "Visible")


def _hist2_inputs(args: list, opts: dict, what: str):
    if "BinCounts" in opts:
        if "XBinEdges" not in opts or "YBinEdges" not in opts:
            raise MatlabError(f"{what}: 'BinCounts' needs 'XBinEdges' and 'YBinEdges'.")
        ex, ey = _vec(opts["XBinEdges"]), _vec(opts["YBinEdges"])
        counts = _real(opts["BinCounts"])
        if counts.shape != (ex.size - 1, ey.size - 1):
            raise MatlabError(f"{what}: BinCounts must have one row for each x bin and one column for each y "
                              "bin.")
        return np.zeros(0), np.zeros(0), ex, ey, counts
    if len(args) < 2:
        raise MatlabError(f"{what} needs x and y data.")
    x, y = _vec(args[0]), _vec(args[1])
    if x.size != y.size:
        raise MatlabError(f"{what}: x and y must have the same number of elements.")
    nx = ny = None
    rest = args[2:]
    if len(rest) == 1:
        n = _vec(rest[0])
        if n.size > 2:
            raise MatlabError(f"{what}: the number of bins must be a number or [nx ny].")
        nx, ny = int(n[0]), int(n[-1])
    elif len(rest) == 2:
        opts = dict(opts, XBinEdges=rest[0], YBinEdges=rest[1])
    elif rest:
        raise MatlabError(f"{what}: too many inputs.")
    good = np.isfinite(x) & np.isfinite(y)
    ex = _edges2(x[good], opts, "x", nx)
    ey = _edges2(y[good], opts, "y", ny)
    return x, y, ex, ey, None


@builtin("histogram2")
def _histogram2(interp, args, nargout):
    """histogram2(x, y): counts of pairs of values in a grid of rectangular bins, drawn as 3-D bars."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _H2_OPTS, "histogram2")
    _no_text_left(args, "histogram2")
    x, y, ex, ey, counts = _hist2_inputs(args, opts, "histogram2")
    style = text(opts.get("DisplayStyle", char("bar3"))).lower()
    if style not in ("bar3", "tile"):
        raise MatlabError("DisplayStyle must be 'bar3' or 'tile'.")
    ax, _ = _axes_from(interp, lead, projection="3d" if style == "bar3" else "rectilinear")
    c = Histogram2(interp, ax, x, y, ex, ey)
    c.style = style
    c.counts_given = counts
    if "Normalization" in opts:
        c.norm = P._check_norm(text(opts["Normalization"]))
    if "FaceColor" in opts:
        c.face = _h2_face_conv(opts["FaceColor"])
    if "EdgeColor" in opts:
        c.edge = "none" if is_text(opts["EdgeColor"]) and text(opts["EdgeColor"]).lower() == "none" \
            else _color_conv(opts["EdgeColor"])
    if "FaceAlpha" in opts:
        c.alpha = _num_conv("FaceAlpha", 0, 1)(opts["FaceAlpha"])
    if "ShowEmptyBins" in opts:
        c.show_empty = _choice_conv("ShowEmptyBins", ("on", "off"))(opts["ShowEmptyBins"])
    c.draw()
    h = _register(interp, c)
    _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
    P._redraw(interp)
    return h if nargout > 0 else None


# -- 3-D bars and pies ---------------------------------------------------------------------------------------

def _bar3_core(interp, args, nargout, horizontal: bool):
    name = "bar3h" if horizontal else "bar3"
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    style, color = "detached", None
    nums = []
    for a in args:
        if is_text(a):
            low = text(a).lower()
            if low in ("detached", "grouped", "stacked"):
                style = low
            elif P._is_color_text(a):
                color = P._color(a)
            else:
                raise MatlabError(f"{name}: unknown option '{text(a)}'.")
        else:
            nums.append(_real(a))
    if not nums:
        raise MatlabError(f"{name} needs data, e.g. {name}(magic(3)).")
    width = 0.8
    if len(nums) >= 2 and nums[-1].size == 1 and (len(nums) == 3 or nums[0].size > 1):
        width = float(nums.pop().flat[0])
    Z = nums[-1]
    if Z.shape[0] == 1 and len(nums) == 1:
        Z = Z.T
    if len(nums) == 2:
        ypos = nums[0].flatten()
        if ypos.size != Z.shape[0]:
            raise MatlabError(f"{name}: give one position for each row of the data.")
    else:
        ypos = np.arange(1.0, Z.shape[0] + 1)
    m, n = Z.shape
    ax, _ = _axes_from(interp, lead, projection="3d")
    cmap = _cmap_obj(interp)
    hs = []
    spacing = float(np.min(np.diff(np.unique(ypos)))) if np.unique(ypos).size > 1 else 1.0
    base = np.zeros(m)
    for j in range(n):
        vals = np.where(np.isfinite(Z[:, j]), Z[:, j], 0.0)
        c = color if color is not None else cmap(j / max(n - 1, 1) if n > 1 else 0.0)
        if style == "detached":
            along, depth, wa, wd = ypos - width * spacing / 2, np.full(m, j + 1 - width / 2), width * spacing, width
            z0, dz = np.zeros(m), vals
        elif style == "grouped":
            wa = width * spacing / n
            along = ypos - width * spacing / 2 + wa * j
            depth, wd = np.full(m, 1 - width / 2), width
            z0, dz = np.zeros(m), vals
        else:
            along, depth, wa, wd = ypos - width * spacing / 2, np.full(m, 1 - width / 2), width * spacing, width
            z0, dz = base.copy(), vals
            base = base + vals
        if horizontal:  # the bars grow along the y-axis; positions run along z
            coll = ax.bar3d(depth, z0, along, wd, dz, wa, color=c, edgecolor="k", linewidth=0.3, shade=True)
        else:
            coll = ax.bar3d(depth, along, z0, wd, wa, dz, color=c, edgecolor="k", linewidth=0.3, shade=True)
        h = P._wrap(interp, "surface", coll)
        h.extra.update({"ZData": Z[:, j].reshape(-1, 1), "XData": np.full((m, 1), float(j + 1)),
                        "YData": ypos.reshape(-1, 1), "axes": ax, "figure": ax.figure})
        hs.append(h)
    cols = range(1, n + 1) if style == "detached" else [1]
    ax.set_xticks(list(cols))
    if horizontal:
        ax.set_zticks(ypos) if m <= 20 else None
    else:
        ax.set_yticks(ypos) if m <= 20 else None
        if not ax.yaxis_inverted():
            ax.invert_yaxis()  # the first row is drawn at the back, as MATLAB orders it
    P._redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else P.GraphicsArray(hs, row=True)


builtin("bar3")(lambda interp, args, nargout: _bar3_core(interp, args, nargout, False))
builtin("bar3h")(lambda interp, args, nargout: _bar3_core(interp, args, nargout, True))


def _pie_inputs(args: list, name: str):
    x = None
    explode = None
    labels = None
    for a in args:
        if isinstance(a, (Cell, StringArray)) or is_text(a):
            labels = P._text_list(a)
        elif hasattr(a, "categories") and hasattr(a, "codes"):
            names = [c for k, c in enumerate(a.categories)]
            counts = np.array([float(np.sum(a.codes == k)) for k in range(len(names))])
            x, labels = counts[counts > 0], [nm for nm, c in zip(names, counts) if c > 0]
        elif x is None:
            x = _vec(a)
        else:
            explode = _vec(a)
    if x is None or x.size == 0:
        raise MatlabError(f"{name} needs data, e.g. {name}([3 2 1]).")
    if np.any(x < 0) or not np.all(np.isfinite(x)):
        raise MatlabError(f"{name}: the data must be non-negative numbers.")
    if explode is not None and explode.size != x.size:
        raise MatlabError(f"{name}: explode must have one value for each slice.")
    if labels is not None and len(labels) != x.size:
        raise MatlabError(f"{name}: give one label for each slice.")
    total = float(x.sum())
    frac = x / total if total > 1 else x  # values that sum to less than 1 leave part of the pie empty
    if labels is None:
        labels = [f"{round(100 * f):g}%" for f in (x / total if total > 0 else x)]
    return x, frac, explode if explode is not None else np.zeros(x.size), labels


@builtin("pie3")
def _pie3(interp, args, nargout):
    """pie3(x), pie3(x, explode), pie3(x, labels): a pie chart drawn as a 3-D disc."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    x, frac, explode, labels = _pie_inputs(args, "pie3")
    ax, _ = _axes_from(interp, lead, projection="3d")
    cmap = _cmap_obj(interp)
    height = 0.35
    start = math.pi / 2
    polys_all = []
    texts = []
    n = x.size
    for k in range(n):
        a0 = start + 2 * math.pi * float(frac[:k].sum())
        a1 = a0 + 2 * math.pi * float(frac[k])
        t = np.linspace(a0, a1, max(int(abs(a1 - a0) / 0.1), 2))
        mid = (a0 + a1) / 2
        off = 0.15 * (1 if explode[k] else 0)
        cx, cy = off * math.cos(mid), off * math.sin(mid)
        rim = np.column_stack([cx + np.cos(t), cy + np.sin(t)])
        outline = np.vstack([[cx, cy], rim])
        top = np.column_stack([outline, np.full(len(outline), height)])
        bottom = np.column_stack([outline, np.zeros(len(outline))])
        sides = []
        ring = np.vstack([outline, outline[:1]])
        for i in range(len(ring) - 1):
            p, q = ring[i], ring[i + 1]
            sides.append(np.array([[p[0], p[1], 0], [q[0], q[1], 0], [q[0], q[1], height], [p[0], p[1], height]]))
        color = cmap(k / max(n - 1, 1) if n > 1 else 0.0)
        coll = Poly3DCollection([bottom] + sides + [top], facecolor=color, edgecolor="k", linewidth=0.4)
        ax.add_collection3d(coll)
        polys_all.append(coll)
        texts.append(ax.text(cx + 1.25 * math.cos(mid), cy + 1.25 * math.sin(mid), height, labels[k],
                             ha="center", va="center"))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(0, 1.0)
    try:
        ax.set_box_aspect((1, 1, 0.4))
    except (AttributeError, ValueError):
        pass
    ax.set_axis_off()
    P._redraw(interp)
    hs = [P._wrap(interp, "patch", c) for c in polys_all] + [P._wrap(interp, "text", t) for t in texts]
    for h in hs[:len(polys_all)]:
        h.extra.update({"axes": ax, "figure": ax.figure})
    return P.GraphicsArray(hs) if nargout > 0 else None


# -- swarm and bubble charts ---------------------------------------------------------------------------------------

def _swarm_offsets(y: np.ndarray, width: float) -> np.ndarray:
    """Sideways offsets that spread points of similar height into a symmetric swarm of the given width."""
    n = y.size
    out = np.zeros(n)
    good = np.isfinite(y)
    if good.sum() < 2:
        return out
    span = float(np.ptp(y[good]))
    if span == 0:
        k = np.arange(n)
        order = np.where(k % 2 == 0, -(k // 2), (k + 1) // 2).astype(float)
        return order / max(np.abs(order).max(), 1) * width / 2
    bins = max(int(round(math.sqrt(good.sum()) * 2.5)), 5)
    which = np.floor((y - np.nanmin(y)) / span * bins).astype(int)
    which = np.clip(which, 0, bins - 1)
    counts = np.bincount(which[good], minlength=bins)
    step = (width / 2) / max((counts.max() - 1) / 2.0, 0.5)
    step = min(step, width / 8)
    for b in range(bins):
        idx = np.flatnonzero(good & (which == b))
        idx = idx[np.argsort(y[idx], kind="stable")]
        m = idx.size
        for rank, i in enumerate(idx):
            pos = (rank + 1) // 2 * (1 if rank % 2 else -1)
            out[i] = (pos + (0.5 if m % 2 == 0 else 0.0)) * step
    return out


def _marker_inputs(args: list, name: str, ncoord: int):
    """(coordinates, size, colour, filled, marker, remaining options) for scatter-like calls."""
    nums, rest = [], []
    filled = False
    marker = "o"
    k = 0
    args = list(args)
    while k < len(args):
        a = args[k]
        if is_text(a):
            s = text(a)
            if s.lower() == "filled":
                filled = True
            elif len(nums) >= ncoord and P._is_color_text(a) and len(nums) < ncoord + 2:
                while len(nums) < ncoord + 1:
                    nums.append(np.zeros((0, 0)))
                nums.append(a)
            elif len(s) <= 2 and P._is_linespec(s) and not any(ch in s for ch in "-:"):
                marker = P._mpl_marker(s)
            else:
                rest = args[k:]
                break
        else:
            nums.append(a)
        k += 1
    if len(nums) < ncoord:
        raise MatlabError(f"{name} needs {ncoord} coordinate inputs.")
    coords = [_vec(a) for a in nums[:ncoord]]
    n = coords[0].size
    if any(c.size != n for c in coords):
        raise MatlabError(f"{name}: the coordinate inputs must have the same number of elements.")
    size = nums[ncoord] if len(nums) > ncoord else None
    color = nums[ncoord + 1] if len(nums) > ncoord + 1 else None
    if len(nums) > ncoord + 2:
        raise MatlabError(f"{name}: too many inputs.")
    return coords, size, color, filled, marker, rest


def _scatter_colors(interp, ax, color, n: int, filled: bool) -> dict:
    kw: dict = {}
    if color is None or (isinstance(color, np.ndarray) and color.size == 0 and not is_char(color)):
        c = _next_color(ax)
        kw.update(facecolors=c if filled else "none", edgecolors=c)
    elif is_text(color):
        c = P._color(color)
        kw.update(facecolors=c if filled else "none", edgecolors=c)
    else:
        v = _real(color)
        if v.ndim == 2 and v.shape == (n, 3):
            rgb = np.clip(v, 0, 1)
            kw.update(facecolors=rgb if filled else "none", edgecolors=rgb)
        elif v.size == 3 and n != 3:
            c = tuple(float(t) for t in v.flatten())
            kw.update(facecolors=c if filled else "none", edgecolors=c)
        elif v.size == n:
            kw.update(c=v.flatten(), cmap=P._current_cmap(interp))
            if not filled:
                kw["facecolors"] = "none"
        else:
            raise MatlabError("The colour must be a colour name, one RGB triplet, one number per point or one "
                              "RGB row per point.")
    return kw


_SWARM_OPTS = ("XJitter", "XJitterWidth", "YJitter", "YJitterWidth", "MarkerFaceColor", "MarkerEdgeColor",
               "MarkerFaceAlpha", "MarkerEdgeAlpha", "LineWidth", "DisplayName", "Tag", "Visible", "SizeData",
               "Marker")


def _swarm_core(interp, args, nargout, three_d: bool):
    name = "swarmchart3" if three_d else "swarmchart"
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    coords, size, color, filled, marker, rest = _marker_inputs(args, name, 3 if three_d else 2)
    rest, opts = _options(rest, _SWARM_OPTS, name)
    _no_text_left(rest, name)
    if rest:
        raise MatlabError(f"{name}: unexpected input after the data.")
    ax, _ = _axes_from(interp, lead, projection="3d" if three_d else "rectilinear")
    x, y = coords[0], coords[1]
    z = coords[2] if three_d else None
    jitter_axis_data = z if three_d else y
    groups = np.unique(x[np.isfinite(x)])
    spacing = float(np.min(np.diff(groups))) if groups.size > 1 else 1.0
    width = P._scalar(opts["XJitterWidth"]) if "XJitterWidth" in opts else 0.9 * spacing
    mode = text(opts.get("XJitter", char("density"))).lower()
    if mode not in ("density", "rand", "randn", "none"):
        raise MatlabError("XJitter must be 'density', 'rand', 'randn' or 'none'.")
    xs = x.astype(float).copy()
    ys = y.astype(float).copy()
    rng = np.random.default_rng(0)
    for g in groups:
        idx = np.flatnonzero(x == g)
        if mode == "density":
            xs[idx] = g + _swarm_offsets(jitter_axis_data[idx], width)
        elif mode == "rand":
            xs[idx] = g + (rng.random(idx.size) - 0.5) * width
        elif mode == "randn":
            xs[idx] = g + np.clip(rng.standard_normal(idx.size) / 6, -0.5, 0.5) * width
    if three_d:
        ygroups = np.unique(y[np.isfinite(y)])
        yspacing = float(np.min(np.diff(ygroups))) if ygroups.size > 1 else 1.0
        ywidth = P._scalar(opts["YJitterWidth"]) if "YJitterWidth" in opts else 0.9 * yspacing
        ymode = text(opts.get("YJitter", char("density"))).lower()
        for g in ygroups:
            idx = np.flatnonzero(y == g)
            if ymode == "density":
                ys[idx] = g + _swarm_offsets(z[idx][::-1], ywidth)[::-1] * 0.999
            elif ymode in ("rand", "randn"):
                ys[idx] = g + (rng.random(idx.size) - 0.5) * ywidth
    s = 36.0
    if size is not None and isinstance(size, np.ndarray) and size.size and not is_char(size):
        sv = _vec(size)
        s = float(sv[0]) if sv.size == 1 else sv
    if "SizeData" in opts:
        sv = _vec(opts["SizeData"])
        s = float(sv[0]) if sv.size == 1 else sv
    if "Marker" in opts:
        marker = P._mpl_marker(text(opts["Marker"]))
    kw = _scatter_colors(interp, ax, color, x.size, filled)
    if "MarkerFaceColor" in opts:
        fc = opts["MarkerFaceColor"]
        if not (is_text(fc) and text(fc).lower() == "flat"):
            kw["facecolors"] = P._color(fc)
    if "MarkerEdgeColor" in opts:
        ec = opts["MarkerEdgeColor"]
        if not (is_text(ec) and text(ec).lower() == "flat"):
            kw["edgecolors"] = P._color(ec)
    if "LineWidth" in opts:
        kw["linewidths"] = P._scalar(opts["LineWidth"])
    if "MarkerFaceAlpha" in opts:
        kw["alpha"] = P._scalar(opts["MarkerFaceAlpha"])
    if "DisplayName" in opts:
        kw["label"] = text(opts["DisplayName"])
    if three_d:
        coll = ax.scatter(xs, ys, z, s=s, marker=marker, depthshade=False, **kw)
    else:
        coll = ax.scatter(xs, ys, s=s, marker=marker, **kw)
        if groups.size <= 20 and not P._state(interp).hold.get(id(ax), False) and P._ruler_of(ax, "x") is None:
            ax.set_xticks(groups)
    h = P._wrap(interp, "scatter", coll)
    h.extra.update({"Marker": marker, "XJitter": mode, "XJitterWidth": P._num_val(width),
                    "_swarm_x": x.reshape(1, -1)})
    _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
    P._redraw(interp)
    return h if nargout > 0 else None


builtin("swarmchart")(lambda interp, args, nargout: _swarm_core(interp, args, nargout, False))
builtin("swarmchart3")(lambda interp, args, nargout: _swarm_core(interp, args, nargout, True))


def _bubble_state(ax) -> dict:
    return ax.__dict__.setdefault("_mylab_bubbles", {"size": (3.0, 50.0), "lim": None, "charts": []})


def _bubble_areas(ax, sz: np.ndarray) -> np.ndarray:
    """Marker areas (points^2) for bubble values: the area grows linearly with the value between the
    smallest and largest bubble diameters of the axes."""
    st = _bubble_state(ax)
    dmin, dmax = st["size"]
    if st["lim"] is not None:
        lo, hi = st["lim"]
    else:
        vals = np.concatenate([c["sz"] for c in st["charts"]] + [sz]) if st["charts"] else sz
        f = vals[np.isfinite(vals)]
        lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
    t = np.clip((sz - lo) / (hi - lo), 0, 1) if hi > lo else np.ones_like(sz, dtype=float)
    area = dmin ** 2 + t * (dmax ** 2 - dmin ** 2)
    return np.where(np.isfinite(sz), area, 0.0)


def _rescale_bubbles(ax) -> None:
    st = _bubble_state(ax)
    for c in st["charts"]:
        if c["coll"].axes is not None:
            c["coll"].set_sizes(_bubble_areas(ax, c["sz"]))


_BUBBLE_OPTS = ("MarkerFaceColor", "MarkerEdgeColor", "MarkerFaceAlpha", "MarkerEdgeAlpha", "LineWidth",
                "DisplayName", "Tag", "Visible")


def _bubble_core(interp, args, nargout, mode: str):
    name = {"2d": "bubblechart", "3d": "bubblechart3", "polar": "polarbubblechart"}[mode]
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    ncoord = 3 if mode == "3d" else 2
    coords, size, color, _filled, _marker, rest = _marker_inputs(args, name, ncoord)
    rest, opts = _options(rest, _BUBBLE_OPTS, name)
    _no_text_left(rest, name)
    if size is None or not isinstance(size, np.ndarray) or size.size == 0:
        raise MatlabError(f"{name} needs the bubble sizes, e.g. {name}(x, y, sz).")
    sz = _vec(size)
    n = coords[0].size
    if sz.size == 1:
        sz = np.full(n, sz[0])
    if sz.size != n:
        raise MatlabError(f"{name}: give one size for each point.")
    ax, _ = _axes_from(interp, lead, projection={"2d": "rectilinear", "3d": "3d", "polar": "polar"}[mode])
    kw = _scatter_colors(interp, ax, color, n, True)
    kw.setdefault("alpha", 0.6)
    if "MarkerFaceAlpha" in opts:
        kw["alpha"] = P._scalar(opts["MarkerFaceAlpha"])
    if "MarkerFaceColor" in opts and not (is_text(opts["MarkerFaceColor"])
                                          and text(opts["MarkerFaceColor"]).lower() == "flat"):
        kw["facecolors"] = P._color(opts["MarkerFaceColor"])
    if "MarkerEdgeColor" in opts and not (is_text(opts["MarkerEdgeColor"])
                                          and text(opts["MarkerEdgeColor"]).lower() == "flat"):
        kw["edgecolors"] = P._color(opts["MarkerEdgeColor"])
    if "LineWidth" in opts:
        kw["linewidths"] = P._scalar(opts["LineWidth"])
    if "DisplayName" in opts:
        kw["label"] = text(opts["DisplayName"])
    st = _bubble_state(ax)
    if not P._state(interp).hold.get(id(ax), False):
        st["charts"] = [c for c in st["charts"] if c["coll"].axes is ax]
    areas = _bubble_areas(ax, sz)
    if mode == "3d":
        coll = ax.scatter(coords[0], coords[1], coords[2], s=areas, depthshade=False, **kw)
    else:
        coll = ax.scatter(coords[0], coords[1], s=areas, **kw)
    st["charts"].append({"coll": coll, "sz": sz})
    _rescale_bubbles(ax)
    if mode == "2d":
        ax.margins(0.12)
        ax._mylab_padded = True
    h = P._wrap(interp, "scatter", coll)
    h.extra.update({"Marker": "o", "_bubble": True})
    _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
    P._redraw(interp)
    return h if nargout > 0 else None


builtin("bubblechart")(lambda interp, args, nargout: _bubble_core(interp, args, nargout, "2d"))
builtin("bubblechart3")(lambda interp, args, nargout: _bubble_core(interp, args, nargout, "3d"))
builtin("polarbubblechart")(lambda interp, args, nargout: _bubble_core(interp, args, nargout, "polar"))


def _current_or_given(interp, args: list):
    args = list(args)
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind in ("axes", "polaraxes", "geoaxes"):
        return args.pop(0).obj, args
    return P._axes(interp, fresh=False), args


@builtin("bubblesize")
def _bubblesize(interp, args, nargout):
    """bubblesize([dmin dmax]): diameters in points of the smallest and largest bubbles of an axes."""
    ax, args = _current_or_given(interp, args)
    st = _bubble_state(ax)
    if not args:
        return P._row(st["size"])
    v = _vec(args[0])
    if v.size != 2 or not 0 < v[0] <= v[1]:
        raise MatlabError("bubblesize expects [smallest largest] diameters in points, both positive.")
    st["size"] = (float(v[0]), float(v[1]))
    _rescale_bubbles(ax)
    P._redraw(interp)
    return None


@builtin("bubblelim")
def _bubblelim(interp, args, nargout):
    """bubblelim([lo hi]): the data values shown with the smallest and the largest bubble."""
    ax, args = _current_or_given(interp, args)
    st = _bubble_state(ax)
    if not args:
        if st["lim"] is not None:
            return P._row(st["lim"])
        vals = np.concatenate([c["sz"] for c in st["charts"]]) if st["charts"] else np.zeros(0)
        f = vals[np.isfinite(vals)]
        return P._row([f.min(), f.max()] if f.size else [0, 1])
    if is_text(args[0]):
        word = text(args[0]).lower()
        if word == "auto":
            st["lim"] = None
        elif word == "mode":
            return char("manual" if st["lim"] is not None else "auto")
        elif word == "manual":
            st["lim"] = tuple(float(v) for v in _bubblelim(interp, [], 1).flatten())
        else:
            raise MatlabError("bubblelim expects [lo hi], 'auto', 'manual' or 'mode'.")
    else:
        v = _vec(args[0])
        if v.size != 2 or not v[0] < v[1]:
            raise MatlabError("bubblelim expects [lo hi] with lo < hi.")
        st["lim"] = (float(v[0]), float(v[1]))
    _rescale_bubbles(ax)
    P._redraw(interp)
    return None


@builtin("bubblelegend")
def _bubblelegend(interp, args, nargout):
    """bubblelegend(title): a legend that shows what bubble sizes mean."""
    ax, args = _current_or_given(interp, args)
    args, opts = _options(args, ("Location", "NumBubbles", "Style", "Box", "FontSize"), "bubblelegend")
    title = P._label_text(args[0]) if args else ""
    if len(args) > 1:
        if is_text(args[0]) and len(args) == 2:  # bubblelegend('Name', value) with an unknown name
            raise MatlabError(f"Unrecognized property '{text(args[0])}' for class 'BubbleLegend'.",
                              "MATLAB:class:InvalidProperty")
        raise MatlabError("bubblelegend takes a title and name/value options.")
    st = _bubble_state(ax)
    if not st["charts"]:
        raise MatlabError("bubblelegend: draw a bubble chart first.")
    lo, hi = (float(v) for v in _bubblelim(interp, [P._wrap(interp, P._axes_kind(ax), ax)], 1).flatten())
    nb = int(P._scalar(opts["NumBubbles"])) if "NumBubbles" in opts else 3
    if nb not in (2, 3):
        raise MatlabError("NumBubbles must be 2 or 3.")
    values = [hi, (lo + hi) / 2, lo] if nb == 3 else [hi, lo]
    areas = _bubble_areas(ax, np.array(values))
    loc_name = text(opts.get("Location", char("eastoutside"))).lower()
    if loc_name not in P._LOC:
        raise MatlabError(f"'{loc_name}' is not a legend location.")
    loc, anchor = P._LOC[loc_name]
    proxies = [ax.scatter([], [], s=a, facecolors="none", edgecolors="0.3") for a in areas]
    for pr in proxies:
        pr._mylab_internal = True
    from matplotlib.legend import Legend
    kw = {"loc": loc, "title": P._tex(interp, title, None, "legend") if title else None, "labelspacing": 1.6,
          "borderpad": 1.0, "handletextpad": 1.5, "scatterpoints": 1}
    if anchor is not None:
        kw["bbox_to_anchor"] = anchor
    if "FontSize" in opts:
        kw["fontsize"] = P._scalar(opts["FontSize"])
    leg = Legend(ax, proxies, [P._fmt_num(v) for v in values], **kw)
    if "Box" in opts:
        leg.set_frame_on(P._onoff(opts["Box"]))
    old = getattr(ax, "_mylab_bubblelegend", None)
    if old is not None:
        try:
            old.remove()
        except (ValueError, AttributeError):
            pass
    ax.add_artist(leg)
    P._room_for_legend(ax, loc_name)
    ax._mylab_bubblelegend = leg
    leg._mylab_loc = loc_name
    leg._mylab_raw = [P._fmt_num(v) for v in values]
    leg._mylab_title_raw = title
    P._redraw(interp)
    return P._wrap(interp, "legend", leg) if nargout > 0 else None


# -- polar charts ------------------------------------------------------------------------------------------------

@builtin("polarscatter")
def _polarscatter(interp, args, nargout):
    """polarscatter(theta, rho, sz, c): markers in polar coordinates."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    coords, size, color, filled, marker, rest = _marker_inputs(args, "polarscatter", 2)
    rest, opts = _options(rest, ("MarkerFaceColor", "MarkerEdgeColor", "MarkerFaceAlpha", "LineWidth",
                                 "DisplayName", "Tag", "Visible"), "polarscatter")
    _no_text_left(rest, "polarscatter")
    ax, _ = _axes_from(interp, lead, projection="polar")
    s = 36.0
    if size is not None and isinstance(size, np.ndarray) and size.size and not is_char(size):
        sv = _vec(size)
        s = float(sv[0]) if sv.size == 1 else sv
    kw = _scatter_colors(interp, ax, color, coords[0].size, filled)
    if "MarkerFaceColor" in opts:
        kw["facecolors"] = P._color(opts["MarkerFaceColor"])
    if "MarkerEdgeColor" in opts:
        kw["edgecolors"] = P._color(opts["MarkerEdgeColor"])
    if "MarkerFaceAlpha" in opts:
        kw["alpha"] = P._scalar(opts["MarkerFaceAlpha"])
    if "LineWidth" in opts:
        kw["linewidths"] = P._scalar(opts["LineWidth"])
    if "DisplayName" in opts:
        kw["label"] = text(opts["DisplayName"])
    coll = ax.scatter(coords[0], coords[1], s=s, marker=marker, **kw)
    h = P._wrap(interp, "scatter", coll)
    h.extra["Marker"] = marker
    _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
    P._redraw(interp)
    return h if nargout > 0 else None


class PolarHistogram(Chart):
    kind = "polarhistogram"

    def __init__(self, interp, ax, data, edges):
        super().__init__(interp, ax)
        self.data, self.edges = data, edges
        self.norm = "count"
        self.style = "bar"
        self.face: Any = None
        self.edge: Any = (0, 0, 0)
        self.alpha = 0.6
        self.linewidth = 0.5
        self.counts_given = None

    def counts(self) -> np.ndarray:
        if self.counts_given is not None:
            return self.counts_given
        e = self.edges
        d = np.mod(self.data - e[0], 2 * math.pi) + e[0]
        c, _ = np.histogram(d, e)
        return c.astype(float)

    def values(self) -> np.ndarray:
        c = self.counts()
        n = max(self.data.size if self.counts_given is None else c.sum(), 1)
        w = np.diff(self.edges)
        return {"probability": c / n, "percentage": 100 * c / n, "countdensity": c / w, "pdf": c / (n * w),
                "cumcount": np.cumsum(c), "cdf": np.cumsum(c) / n}.get(self.norm, c)

    def draw(self) -> None:
        from matplotlib.colors import to_rgba
        ax = self.axes
        v = self.values()
        w = np.diff(self.edges)
        if self.face is None:
            self.face = to_rgba(_next_color(ax))[:3]
        if self.style == "stairs":
            th, rr = [], []
            for k in range(v.size):
                t = np.linspace(self.edges[k], self.edges[k + 1], 20)
                th.extend(t)
                rr.extend([v[k]] * t.size)
            th.append(th[0])
            rr.append(rr[0])
            self.add(ax.plot(th, rr, color=self.face, linewidth=max(self.linewidth, 1.0)))
        else:
            bars = ax.bar(self.edges[:-1], v, width=w, align="edge", color=(*self.face, self.alpha),
                          edgecolor=None if isinstance(self.edge, str) else self.edge, linewidth=self.linewidth)
            self.add(list(bars.patches))
            if bars in ax.containers:
                ax.containers.remove(bars)


def _ph_edges(theta: np.ndarray, nbins: int | None, edges) -> np.ndarray:
    if edges is not None:
        e = np.asarray(edges, dtype=float)
        if e.size < 2 or np.any(np.diff(e) <= 0) or e[-1] - e[0] > 2 * math.pi + 1e-9:
            raise MatlabError("The bin edges must increase and span at most 2*pi.")
        return e
    if nbins is None:
        n = theta.size
        nbins = max(min(int(math.ceil(2 * n ** (1 / 3))) if n else 10, 100), 4) if n > 30 else \
            max(int(math.ceil(1 + math.log2(n))) if n else 10, 4)
    if nbins < 1:
        raise MatlabError("The number of bins must be a positive whole number.")
    return np.linspace(0, 2 * math.pi, nbins + 1)


def _ph_numbins_set(h, v):
    h.obj.edges = _ph_edges(h.obj.data, int(P._scalar(v)), None)
    h.obj.counts_given = None
    h.obj.redraw()


def _ph_edges_set(h, v):
    h.obj.edges = _ph_edges(h.obj.data, None, _vec(v))
    h.obj.counts_given = None
    h.obj.redraw()


_chart_setup("polarhistogram", "matlab.graphics.chart.primitive.Histogram", {
    "Data": _ro(lambda c: P._row(c.data)),
    "Values": _ro(lambda c: P._row(c.values())),
    "BinCounts": _ro(lambda c: P._row(c.counts())),
    "BinEdges": (lambda h: P._row(h.obj.edges), _ph_edges_set),
    "NumBins": (lambda h: P._num_val(h.obj.edges.size - 1), _ph_numbins_set),
    "BinWidth": (lambda h: P._num_val(np.diff(h.obj.edges)[0]),
                 lambda h, v: _ph_numbins_set(h, np.array([[max(round(2 * math.pi / P._scalar(v)), 1)]]))),
    "BinLimits": _ro(lambda c: P._row([c.edges[0], c.edges[-1]])),
    "Normalization": _attr("norm", lambda v: P._check_norm(text(v))),
    "DisplayStyle": _attr("style", _choice_conv("DisplayStyle", ("bar", "stairs"))),
    "FaceColor": _attr("face", _color_conv, out=_color_out),
    "EdgeColor": _attr("edge", lambda v: "none" if is_text(v) and text(v).lower() == "none" else _color_conv(v),
                       out=lambda c: char(c) if isinstance(c, str) else P._rgb(c)),
    "FaceAlpha": _attr("alpha", _num_conv("FaceAlpha", 0, 1)),
    "LineWidth": _attr("linewidth", _num_conv("LineWidth", 0, 100)),
}, ["Data", "Values", "NumBins", "BinEdges", "BinWidth", "Normalization", "FaceColor"])
P._TYPE["polarhistogram"] = "histogram"


@builtin("polarhistogram")
def _polarhistogram(interp, args, nargout):
    """polarhistogram(theta, nbins): a histogram of angles (in radians) drawn in polar axes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, ("BinEdges", "BinCounts", "NumBins", "BinWidth", "BinLimits", "Normalization",
                                 "DisplayStyle", "FaceColor", "EdgeColor", "FaceAlpha", "LineWidth", "Tag",
                                 "Visible", "DisplayName"), "polarhistogram")
    _no_text_left(args, "polarhistogram")
    counts = None
    if "BinCounts" in opts:
        if "BinEdges" not in opts:
            raise MatlabError("polarhistogram: 'BinCounts' needs 'BinEdges'.")
        edges = _ph_edges(np.zeros(0), None, _vec(opts["BinEdges"]))
        counts = _vec(opts["BinCounts"])
        if counts.size != edges.size - 1:
            raise MatlabError("polarhistogram: BinCounts must have one value fewer than BinEdges.")
        theta = np.zeros(0)
    else:
        if not args:
            raise MatlabError("polarhistogram needs angles in radians, e.g. polarhistogram(theta, 12).")
        theta = _vec(args[0])
        theta = theta[np.isfinite(theta)]
        nbins = edges_in = None
        if len(args) > 1:
            b = _vec(args[1])
            if b.size == 1:
                nbins = int(b[0])
            else:
                edges_in = b
        if "NumBins" in opts:
            nbins = int(P._scalar(opts["NumBins"]))
        if "BinEdges" in opts:
            edges_in = _vec(opts["BinEdges"])
        if "BinWidth" in opts:
            nbins = max(int(round(2 * math.pi / P._scalar(opts["BinWidth"]))), 1)
        if "BinLimits" in opts and edges_in is None:
            lim = _vec(opts["BinLimits"])
            edges_in = np.linspace(lim[0], lim[1], (nbins or 12) + 1)
        edges = _ph_edges(theta, nbins, edges_in)
    ax, _ = _axes_from(interp, lead, projection="polar")
    c = PolarHistogram(interp, ax, theta, edges)
    c.counts_given = counts
    c.draw()
    h = _register(interp, c)
    _apply_opts(h, opts, skip=("BinEdges", "BinCounts", "NumBins", "BinWidth", "BinLimits", "DisplayName"))
    P._redraw(interp)
    return h if nargout > 0 else None


@builtin("rose")
def _rose(interp, args, nargout):
    """rose(theta, n): an angle histogram; with outputs, [tout, rout] trace its outline for polar."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    _no_text_left(args, "rose")
    if not args:
        raise MatlabError("rose needs angles in radians, e.g. rose(theta).")
    theta = np.mod(_vec(args[0]), 2 * math.pi)
    theta = theta[np.isfinite(theta)]
    if len(args) > 1 and _vec(args[1]).size > 1:
        centers = np.sort(np.mod(_vec(args[1]), 2 * math.pi))
        mids = (centers[:-1] + centers[1:]) / 2
        first = (centers[-1] - 2 * math.pi + centers[0]) / 2
        edges = np.concatenate([[first], mids, [first + 2 * math.pi]])
    else:
        n = int(P._scalar(args[1])) if len(args) > 1 else 20
        if n < 1:
            raise MatlabError("rose: the number of bins must be positive.")
        edges = np.linspace(0, 2 * math.pi, n + 1)
    d = np.mod(theta - edges[0], 2 * math.pi) + edges[0]
    counts, _ = np.histogram(d, edges)
    tout = np.zeros(4 * counts.size)
    rout = np.zeros(4 * counts.size)
    tout[0::4], tout[1::4], tout[2::4], tout[3::4] = edges[:-1], edges[:-1], edges[1:], edges[1:]
    rout[1::4] = counts
    rout[2::4] = counts
    if nargout > 0:
        return (P._row(tout), P._row(rout))[:nargout] if nargout > 1 else P._row(tout)
    ax, _ = _axes_from(interp, lead, projection="polar")
    th, rr = [], []
    for k in range(counts.size):
        arc = np.linspace(edges[k], edges[k + 1], 16)
        th.extend([edges[k]] + list(arc) + [edges[k + 1]])
        rr.extend([0.0] + [float(counts[k])] * arc.size + [0.0])
    lines = ax.plot(th, rr, color=_next_color(ax))
    P._wrap(interp, "line", lines[0])
    P._redraw(interp)
    return None


def _uv_inputs(args: list, name: str):
    nums = []
    spec = ""
    for a in args:
        if is_text(a):
            if not P._is_linespec(text(a)):
                raise MatlabError(f"{name}: unknown option '{text(a)}'.")
            spec = text(a)
        else:
            nums.append(a)
    if not nums:
        raise MatlabError(f"{name} needs data: {name}(U, V) or {name}(Z).")
    if len(nums) == 1:
        z = fnum(nums[0]).flatten(order="F")
        u, v = np.real(z).astype(float), np.imag(z).astype(float)
    else:
        u, v = _vec(nums[0]), _vec(nums[1])
        if u.size != v.size:
            raise MatlabError(f"{name}: U and V must have the same number of elements.")
    return u, v, spec


def _arrow_xy(x0, y0, x1, y1, head: float = 0.08):
    """Polyline of an arrow from (x0, y0) to (x1, y1) with a small open head."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L == 0:
        return [x0, x1], [y0, y1]
    ux, uy = dx / L, dy / L
    hl = head * L
    px, py = -uy, ux
    ax_, ay_ = x1 - hl * ux + 0.5 * hl * px, y1 - hl * uy + 0.5 * hl * py
    bx_, by_ = x1 - hl * ux - 0.5 * hl * px, y1 - hl * uy - 0.5 * hl * py
    return [x0, x1, ax_, x1, bx_], [y0, y1, ay_, y1, by_]


def _spec_kw(spec: str) -> dict:
    import re
    kw: dict = {}
    if P.spec_color(spec) is not None:
        kw["color"] = P.spec_color(spec)
    style = re.sub(r"[rgbcmykwo+*.xsd^v<>ph]", "", spec)
    if style in ("-", "--", ":", "-."):
        kw["linestyle"] = style
    return kw


@builtin("compass")
def _compass(interp, args, nargout):
    """compass(U, V) or compass(Z): arrows from the origin, drawn in polar axes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    u, v, spec = _uv_inputs(args, "compass")
    ax, _ = _axes_from(interp, lead, projection="polar")
    kw = _spec_kw(spec)
    kw.setdefault("color", _next_color(ax))
    lines = []
    for a, b in zip(u, v):
        xs, ys = _arrow_xy(0.0, 0.0, float(a), float(b))
        th = [math.atan2(y, x) for x, y in zip(xs, ys)]
        rr = [math.hypot(x, y) for x, y in zip(xs, ys)]
        lines.extend(ax.plot(th, rr, **kw))
    rmax = float(np.max(np.hypot(u, v))) if u.size else 1.0
    ax.set_rlim(0, rmax if rmax > 0 else 1.0)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


@builtin("feather")
def _feather(interp, args, nargout):
    """feather(U, V) or feather(Z): arrows starting at equally spaced points along the x-axis."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    u, v, spec = _uv_inputs(args, "feather")
    ax, _ = _axes_from(interp, lead)
    kw = _spec_kw(spec)
    kw.setdefault("color", _next_color(ax))
    lines = []
    for k, (a, b) in enumerate(zip(u, v)):
        xs, ys = _arrow_xy(k + 1.0, 0.0, k + 1.0 + float(a), float(b))
        lines.extend(ax.plot(xs, ys, **kw))
    base = ax.plot([1, max(u.size, 1)], [0, 0], **kw)
    lines.extend(base)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


@builtin("polar")
def _polar(interp, args, nargout):
    """polar(theta, rho, linespec): a line in polar coordinates (the older form of polarplot)."""
    args = list(args)
    if len([a for a in args if not is_text(a) and not isinstance(a, P.GraphicsHandle)]) < 2:
        raise MatlabError("polar needs theta and rho, e.g. polar(theta, rho).")
    return BUILTINS["polarplot"](interp, args, nargout)


@builtin("ezpolar")
def _ezpolar(interp, args, nargout):
    """ezpolar(f, [a b]): the polar curve rho = f(theta), by default for 0 <= theta <= 2*pi."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    if not args:
        raise MatlabError("ezpolar needs a function, e.g. ezpolar(@(t) 1 + cos(t)).")
    f, _, label = P._as_fn(interp, args[0])
    a, b = 0.0, 2 * math.pi
    if len(args) > 1:
        lim = _vec(args[1])
        if lim.size != 2 or not lim[0] < lim[1]:
            raise MatlabError("ezpolar: the interval must be [a b] with a < b.")
        a, b = float(lim[0]), float(lim[1])
    t = np.linspace(a, b, 400).reshape(1, -1)
    r = P._eval_fn(interp, f, [t])
    ax, _ = _axes_from(interp, lead, projection="polar")
    lines = ax.plot(t.flatten(), r.flatten())
    ax.set_xlabel(f"r = {label}")
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


def _set_tick_text(ax, which: str, positions, labels, **kw) -> None:
    """Put fixed tick marks with the given texts on an axis (and remember the texts for xticklabels)."""
    labels = [str(s) for s in labels]
    getattr(ax, f"set_{which}ticks")(list(positions))
    getattr(ax, f"set_{which}ticklabels")(labels, **kw)
    setattr(ax, f"_mylab_{which}ticklabels", labels)
    setattr(ax, f"_mylab_{which}ticks_fixed", True)


def _colors_conv(v):
    from .plot_objects import to_color_matrix
    return to_color_matrix(v)


def _default_colors():
    from .plot_objects import default_colors
    return default_colors()


# -- reference regions --------------------------------------------------------------------------------------------

_REGION_OPTS = ("FaceColor", "FaceAlpha", "EdgeColor", "LineStyle", "LineWidth", "DisplayName", "Tag", "Visible",
                "HandleVisibility", "Layer")


def _region_core(interp, args, nargout, vertical: bool):
    name = "xregion" if vertical else "yregion"
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _REGION_OPTS, name)
    _no_text_left(args, name)
    if len(args) == 2:
        a, b = _vec(args[0]), _vec(args[1])
        if a.size != b.size:
            raise MatlabError(f"{name}: give the same number of start and end values.")
        spans = np.column_stack([a, b])
    elif len(args) == 1:
        v = _real(args[0])
        if v.size == 2:
            spans = v.reshape(1, 2)
        elif v.ndim == 2 and v.shape[0] == 2:
            spans = v.T
        elif v.ndim == 2 and v.shape[1] == 2:
            spans = v
        else:
            raise MatlabError(f"{name} expects two values, or a matrix with two rows (start; end).")
    else:
        raise MatlabError(f"{name} needs the start and end of the region, e.g. {name}(2, 5).")
    ax, _ = _axes_from(interp, lead, fresh=False)
    color = _color_conv(opts["FaceColor"]) if "FaceColor" in opts else (0.5, 0.5, 0.5)
    alpha = P._scalar(opts["FaceAlpha"]) if "FaceAlpha" in opts else 0.3
    kw: dict = {"facecolor": color, "alpha": alpha, "edgecolor": "none", "zorder": 0.5}
    if "EdgeColor" in opts and not (is_text(opts["EdgeColor"]) and text(opts["EdgeColor"]).lower() == "none"):
        kw["edgecolor"] = _color_conv(opts["EdgeColor"])
        kw["linewidth"] = P._scalar(opts["LineWidth"]) if "LineWidth" in opts else 0.5
        if "LineStyle" in opts:
            kw["linestyle"] = P._mpl_linestyle(text(opts["LineStyle"]))
    if "DisplayName" in opts:
        kw["label"] = text(opts["DisplayName"])
    hs = []
    for lo, hi in spans:
        lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
        patch = ax.axvspan(lo, hi, **kw) if vertical else ax.axhspan(lo, hi, **kw)
        h = P._wrap(interp, "constantregion", patch)
        h.extra.update({"axes": ax, "figure": ax.figure, "Value": np.array([[lo, hi]]),
                        "InterceptAxis": "x" if vertical else "y"})
        _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible", "HandleVisibility")})
        hs.append(h)
    P._redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else P.GraphicsArray(hs)


def _region_value_set(h, v):
    lim = _vec(v)
    if lim.size != 2:
        raise MatlabError("Value must be [start end].")
    lo, hi = float(min(lim)), float(max(lim))
    patch = h.obj
    vertical = h.extra["InterceptAxis"] == "x"
    if hasattr(patch, "set_bounds"):
        patch.set_bounds(lo, 0, hi - lo, 1) if vertical else patch.set_bounds(0, lo, 1, hi - lo)
    else:
        patch.set_xy([[lo, 0], [lo, 1], [hi, 1], [hi, 0]] if vertical else [[0, lo], [0, hi], [1, hi], [1, lo]])
    h.extra["Value"] = np.array([[lo, hi]])


P._CLASS["constantregion"] = "matlab.graphics.chart.decoration.ConstantRegion"
P._TYPE["constantregion"] = "constantregion"
P._PROPS["constantregion"] = {
    **P._PATCH_PROPS,
    "Value": (lambda h: h.extra["Value"], _region_value_set),
    "InterceptAxis": (lambda h: char(h.extra["InterceptAxis"]), None),
}
P._DISPLAY["constantregion"] = ["InterceptAxis", "Value", "FaceColor", "FaceAlpha", "EdgeColor"]

builtin("xregion")(lambda interp, args, nargout: _region_core(interp, args, nargout, True))
builtin("yregion")(lambda interp, args, nargout: _region_core(interp, args, nargout, False))


# -- box charts and violin plots -----------------------------------------------------------------------------------

def _grouped_inputs(interp, args: list, name: str):
    """(positions, labels or None, list of sample arrays, colour-group names or None, colour index per box)."""
    args, extra = _options(list(args), ("GroupByColor",), name)
    if not args:
        raise MatlabError(f"{name} needs data, e.g. {name}(y) or {name}(xgroup, y).")
    if len(args) > 2:
        raise MatlabError(f"{name}: too many data inputs.")
    color_groups = extra.get("GroupByColor")
    if len(args) == 1:
        Y = _real(args[0])
        if Y.shape[0] == 1:
            Y = Y.T
        if color_groups is not None:
            raise MatlabError(f"{name}: GroupByColor needs a grouping variable, {name}(xgroup, y, ...).")
        samples = [Y[:, j] for j in range(Y.shape[1])]
        return np.arange(1.0, Y.shape[1] + 1), None, samples, None, [0] * Y.shape[1]
    g, y = args[0], _vec(args[1])
    labels = None
    if hasattr(g, "categories") or isinstance(g, (Cell, StringArray)):
        from .ndmath import group_index
        idx, names, _ = group_index(g)
        if hasattr(g, "categories"):
            names_all = list(g.categories)
            codes = g.codes.flatten(order="F").astype(float)
            idx = np.where(codes >= 0, codes + 1, np.nan)
            names = names_all
        pos_of = idx
        positions = np.arange(1.0, len(names) + 1)
        labels = names
    else:
        gv = _vec(g)
        pos_of = gv
        positions = np.unique(gv[np.isfinite(gv)])
    if np.size(pos_of) != y.size:
        raise MatlabError(f"{name}: the grouping variable must have one value for each data value.")
    if color_groups is None:
        samples = [y[pos_of == p] for p in positions]
        return positions, labels, samples, None, [0] * len(positions)
    from .ndmath import group_index
    cidx, cnames, _ = group_index(color_groups)
    if cidx.size != y.size:
        raise MatlabError(f"{name}: GroupByColor must have one value for each data value.")
    out_pos, samples, cols = [], [], []
    for p in positions:
        for c in range(len(cnames)):
            out_pos.append(float(p))
            samples.append(y[(pos_of == p) & (cidx == c + 1)])
            cols.append(c)
    return np.array(out_pos), labels, samples, cnames, cols


class BoxChart(Chart):
    kind = "boxchart"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.positions = np.zeros(0)
        self.samples: list = []
        self.offsets = np.zeros(0)
        self.color: Any = None
        self.face_alpha = 0.2
        self.box_width = 0.5
        self.marker = "o"
        self.marker_color: Any = None
        self.marker_size = 6.0
        self.notch = "off"
        self.orientation = "vertical"
        self.whisker_color = (0.0, 0.0, 0.0)
        self.linewidth = 1.0
        self.jitter = "off"
        self.label = ""
        self.xdata = np.zeros(0)
        self.ydata = np.zeros(0)

    def stats(self, s: np.ndarray):
        s = s[np.isfinite(s)]
        if s.size == 0:
            return None
        q1, med, q3 = _quantiles(s, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        inside = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
        lo, hi = (float(inside.min()), float(inside.max())) if inside.size else (q1, q3)
        out = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        return q1, med, q3, lo, hi, out, s.size

    def draw(self) -> None:
        from matplotlib.colors import to_rgba
        ax = self.axes
        if self.color is None:
            self.color = to_rgba(_next_color(ax))[:3]
        mcolor = self.marker_color if self.marker_color is not None else self.color
        horizontal = self.orientation == "horizontal"

        def line(xs, ys, **kw):
            return ax.plot(ys, xs, **kw) if horizontal else ax.plot(xs, ys, **kw)

        w = self.box_width
        first = True
        for p, off, s in zip(self.positions, self.offsets, self.samples):
            st = self.stats(np.asarray(s, dtype=float))
            if st is None:
                continue
            q1, med, q3, lo, hi, out, n = st
            c = p + off
            if self.notch == "on":
                d = 1.57 * (q3 - q1) / math.sqrt(n)
                n1, n2 = med - d, med + d
                xs = [c - w / 2, c + w / 2, c + w / 2, c + w / 4, c + w / 2, c + w / 2, c - w / 2, c - w / 2,
                      c - w / 4, c - w / 2]
                ys = [q1, q1, n1, med, n2, q3, q3, n2, med, n1]
                half = w / 4
            else:
                xs = [c - w / 2, c + w / 2, c + w / 2, c - w / 2]
                ys = [q1, q1, q3, q3]
                half = w / 2
            pts = np.column_stack([ys, xs] if horizontal else [xs, ys])
            from matplotlib.patches import Polygon
            box = Polygon(pts, closed=True, facecolor=(*self.color, self.face_alpha), edgecolor=self.color,
                          linewidth=self.linewidth, label=self.label if first and self.label else None)
            ax.add_patch(box)
            self.add(box)
            first = False
            self.add(line([c - half, c + half], [med, med], color=self.color, linewidth=self.linewidth))
            self.add(line([c, c], [q3, hi], color=self.whisker_color, linewidth=self.linewidth))
            self.add(line([c, c], [q1, lo], color=self.whisker_color, linewidth=self.linewidth))
            if out.size and self.marker != "none":
                ox = np.full(out.size, c)
                if self.jitter == "on":
                    ox = c + (np.random.default_rng(int(abs(c) * 1000) % 2 ** 31).random(out.size) - 0.5) * w / 2
                self.add(line(ox, out, linestyle="None", marker=P._mpl_marker(self.marker), markersize=self.marker_size,
                              markerfacecolor="none", markeredgecolor=mcolor))
        ax.relim()
        ax.autoscale_view()


def _bc_marker_conv(v):
    s = text(v)
    if s.lower() != "none" and P._mpl_marker(s) not in "o+*.xsd^v<>ph_|":
        raise MatlabError(f"'{s}' is not a marker.")
    return s


_chart_setup("boxchart", "matlab.graphics.chart.primitive.BoxChart", {
    "XData": _ro(lambda c: c.xdata), "YData": _ro(lambda c: c.ydata),
    "BoxFaceColor": _attr("color", _color_conv, out=_color_out),
    "BoxEdgeColor": _attr("color", _color_conv, out=_color_out),
    "BoxFaceAlpha": _attr("face_alpha", _num_conv("BoxFaceAlpha", 0, 1)),
    "BoxWidth": _attr("box_width", _num_conv("BoxWidth", 1e-6, 100)),
    "MarkerStyle": _attr("marker", _bc_marker_conv),
    "MarkerColor": _attr("marker_color", _color_conv,
                         out=lambda c: P._rgb(c) if c is not None else char("auto")),
    "MarkerSize": _attr("marker_size", _num_conv("MarkerSize", 0.1, 1000)),
    "Notch": _attr("notch", _choice_conv("Notch", ("on", "off"))),
    "Orientation": _attr("orientation", _choice_conv("Orientation", ("vertical", "horizontal"))),
    "WhiskerLineColor": _attr("whisker_color", _color_conv, out=_color_out),
    "LineWidth": _attr("linewidth", _num_conv("LineWidth", 0.01, 100)),
    "JitterOutliers": _attr("jitter", _choice_conv("JitterOutliers", ("on", "off"))),
    "DisplayName": _attr("label", lambda v: text(v)),
}, ["XData", "YData", "BoxFaceColor", "BoxWidth", "MarkerStyle", "Notch", "Orientation"])

_BOX_OPTS = ("BoxFaceColor", "BoxEdgeColor", "BoxFaceAlpha", "BoxWidth", "MarkerStyle", "MarkerColor", "MarkerSize",
             "Notch", "Orientation", "WhiskerLineColor", "LineWidth", "JitterOutliers", "DisplayName", "Tag",
             "Visible")


def _categorical_axis(ax, which: str, positions, labels) -> None:
    if labels is not None:
        r = P.Ruler("categorical", categories=list(labels))
        P.install_ruler(ax, which, r)
        P.refresh_categorical_ruler(ax, which)
    elif len(positions) <= 20:
        (ax.set_xticks if which == "x" else ax.set_yticks)(np.unique(positions))


@builtin("boxchart")
def _boxchart(interp, args, nargout):
    """boxchart(y) or boxchart(xgroup, y): boxes showing the median, quartiles, whiskers and outliers."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _BOX_OPTS, "boxchart")
    positions, labels, samples, cnames, cols = _grouped_inputs(interp, args, "boxchart")
    ax, _ = _axes_from(interp, lead)
    ngroups = len(cnames) if cnames else 1
    hs = []
    upos = np.unique(positions)
    spacing = float(np.min(np.diff(upos))) if upos.size > 1 else 1.0
    for c in range(ngroups):
        chart = BoxChart(interp, ax)
        pick = [k for k in range(len(samples)) if cols[k] == c]
        chart.positions = np.array([positions[k] for k in pick])
        chart.samples = [samples[k] for k in pick]
        width = 0.5 * spacing / ngroups
        chart.box_width = width
        chart.offsets = np.full(len(pick), (c - (ngroups - 1) / 2) * spacing * 0.8 / ngroups)
        chart.xdata = np.concatenate([np.full(len(s), p) for p, s in zip(chart.positions, chart.samples)]
                                     ).reshape(-1, 1) if pick else np.zeros((0, 1))
        chart.ydata = np.concatenate([np.asarray(s, float) for s in chart.samples]).reshape(-1, 1) if pick \
            else np.zeros((0, 1))
        if cnames:
            chart.label = cnames[c]
        for name, attr, conv in (("BoxFaceColor", "color", _color_conv), ("BoxEdgeColor", "color", _color_conv),
                                 ("BoxFaceAlpha", "face_alpha", _num_conv("BoxFaceAlpha", 0, 1)),
                                 ("BoxWidth", "box_width", _num_conv("BoxWidth", 1e-6, 100)),
                                 ("MarkerStyle", "marker", _bc_marker_conv),
                                 ("MarkerColor", "marker_color", _color_conv),
                                 ("MarkerSize", "marker_size", _num_conv("MarkerSize", 0.1, 1000)),
                                 ("Notch", "notch", _choice_conv("Notch", ("on", "off"))),
                                 ("Orientation", "orientation",
                                  _choice_conv("Orientation", ("vertical", "horizontal"))),
                                 ("WhiskerLineColor", "whisker_color", _color_conv),
                                 ("LineWidth", "linewidth", _num_conv("LineWidth", 0.01, 100)),
                                 ("JitterOutliers", "jitter", _choice_conv("JitterOutliers", ("on", "off"))),
                                 ("DisplayName", "label", lambda v: text(v))):
            if name in opts:
                setattr(chart, attr, conv(opts[name]))
        chart.draw()
        h = _register(interp, chart)
        _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
        hs.append(h)
    horizontal = hs and hs[0].obj.orientation == "horizontal"
    _categorical_axis(ax, "y" if horizontal else "x", positions, labels)
    if labels is None:
        lo, hi = float(upos.min()) - spacing / 2, float(upos.max()) + spacing / 2
        (ax.set_ylim if horizontal else ax.set_xlim)(lo, hi)
    P._redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else P.GraphicsArray(hs, row=True)


class ViolinPlot(Chart):
    kind = "violinplot"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.positions = np.zeros(0)
        self.samples: list = []
        self.offsets = np.zeros(0)
        self.color: Any = None
        self.edge: Any = None
        self.face_alpha = 0.2
        self.width = 0.9
        self.direction = "both"
        self.orientation = "vertical"
        self.scale = "area"
        self.bandwidth = None
        self.linewidth = 0.5
        self.label = ""
        self.xdata = np.zeros(0)
        self.ydata = np.zeros(0)

    def density(self, s: np.ndarray):
        s = s[np.isfinite(s)]
        if s.size < 2 or float(np.ptp(s)) == 0:
            return None
        pts = np.linspace(float(s.min()), float(s.max()), 100)
        return pts, _kde(s, pts, self.bandwidth), s.size

    def draw(self) -> None:
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Polygon
        ax = self.axes
        if self.color is None:
            self.color = to_rgba(_next_color(ax))[:3]
        edge = self.edge if self.edge is not None else self.color
        dens = [self.density(np.asarray(s, dtype=float)) for s in self.samples]
        if self.scale == "count":
            peak = max((d[1].max() * d[2] for d in dens if d is not None), default=1.0)
        elif self.scale == "width":
            peak = None
        else:
            peak = max((d[1].max() for d in dens if d is not None), default=1.0)
        first = True
        for p, off, d in zip(self.positions, self.offsets, dens):
            if d is None:
                continue
            pts, f, n = d
            if peak is None:
                half = f / f.max() * self.width / 2
            else:
                half = (f * (n if self.scale == "count" else 1.0)) / peak * self.width / 2
            c = p + off
            left = c - half if self.direction in ("both", "negative") else np.full(half.shape, c)
            right = c + half if self.direction in ("both", "positive") else np.full(half.shape, c)
            xs = np.concatenate([left, right[::-1]])
            ys = np.concatenate([pts, pts[::-1]])
            verts = np.column_stack([ys, xs] if self.orientation == "horizontal" else [xs, ys])
            poly = Polygon(verts, closed=True, facecolor=(*self.color, self.face_alpha), edgecolor=edge,
                           linewidth=self.linewidth, label=self.label if first and self.label else None)
            ax.add_patch(poly)
            self.add(poly)
            first = False
        ax.relim()
        ax.autoscale_view()


_chart_setup("violinplot", "matlab.graphics.chart.primitive.ViolinPlot", {
    "XData": _ro(lambda c: c.xdata), "YData": _ro(lambda c: c.ydata),
    "FaceColor": _attr("color", _color_conv, out=_color_out),
    "EdgeColor": _attr("edge", _color_conv, out=lambda c: P._rgb(c) if c is not None else char("auto")),
    "FaceAlpha": _attr("face_alpha", _num_conv("FaceAlpha", 0, 1)),
    "DensityWidth": _attr("width", _num_conv("DensityWidth", 1e-6, 100)),
    "DensityDirection": _attr("direction", _choice_conv("DensityDirection", ("both", "positive", "negative"))),
    "DensityScale": _attr("scale", _choice_conv("DensityScale", ("area", "count", "width"))),
    "Orientation": _attr("orientation", _choice_conv("Orientation", ("vertical", "horizontal"))),
    "LineWidth": _attr("linewidth", _num_conv("LineWidth", 0.01, 100)),
    "DisplayName": _attr("label", lambda v: text(v)),
}, ["XData", "YData", "FaceColor", "DensityWidth", "DensityDirection", "Orientation"])

_VIOLIN_OPTS = ("FaceColor", "EdgeColor", "FaceAlpha", "DensityWidth", "DensityDirection", "DensityScale",
                "Orientation", "LineWidth", "DisplayName", "Bandwidth", "Tag", "Visible")


@builtin("violinplot")
def _violinplot(interp, args, nargout):
    """violinplot(y) or violinplot(xgroup, y): the distribution of each group as a density outline."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _VIOLIN_OPTS, "violinplot")
    positions, labels, samples, cnames, cols = _grouped_inputs(interp, args, "violinplot")
    ax, _ = _axes_from(interp, lead)
    ngroups = len(cnames) if cnames else 1
    upos = np.unique(positions)
    spacing = float(np.min(np.diff(upos))) if upos.size > 1 else 1.0
    hs = []
    for c in range(ngroups):
        chart = ViolinPlot(interp, ax)
        pick = [k for k in range(len(samples)) if cols[k] == c]
        chart.positions = np.array([positions[k] for k in pick])
        chart.samples = [samples[k] for k in pick]
        chart.width = 0.9 * spacing / ngroups
        chart.offsets = np.full(len(pick), (c - (ngroups - 1) / 2) * spacing * 0.9 / ngroups)
        chart.xdata = np.concatenate([np.full(len(s), p) for p, s in zip(chart.positions, chart.samples)]
                                     ).reshape(-1, 1) if pick else np.zeros((0, 1))
        chart.ydata = np.concatenate([np.asarray(s, float) for s in chart.samples]).reshape(-1, 1) if pick \
            else np.zeros((0, 1))
        if cnames:
            chart.label = cnames[c]
        for name, attr, conv in (("FaceColor", "color", _color_conv), ("EdgeColor", "edge", _color_conv),
                                 ("FaceAlpha", "face_alpha", _num_conv("FaceAlpha", 0, 1)),
                                 ("DensityWidth", "width", _num_conv("DensityWidth", 1e-6, 100)),
                                 ("DensityDirection", "direction",
                                  _choice_conv("DensityDirection", ("both", "positive", "negative"))),
                                 ("DensityScale", "scale", _choice_conv("DensityScale", ("area", "count", "width"))),
                                 ("Orientation", "orientation",
                                  _choice_conv("Orientation", ("vertical", "horizontal"))),
                                 ("LineWidth", "linewidth", _num_conv("LineWidth", 0.01, 100)),
                                 ("Bandwidth", "bandwidth", _num_conv("Bandwidth", 1e-12, 1e12)),
                                 ("DisplayName", "label", lambda v: text(v))):
            if name in opts:
                setattr(chart, attr, conv(opts[name]))
        chart.draw()
        h = _register(interp, chart)
        _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
        hs.append(h)
    horizontal = hs and hs[0].obj.orientation == "horizontal"
    _categorical_axis(ax, "y" if horizontal else "x", positions, labels)
    if labels is None:
        lo, hi = float(upos.min()) - spacing / 2, float(upos.max()) + spacing / 2
        (ax.set_ylim if horizontal else ax.set_xlim)(lo, hi)
    P._redraw(interp)
    if nargout <= 0:
        return None
    return hs[0] if len(hs) == 1 else P.GraphicsArray(hs, row=True)


# -- charts made of several small axes ---------------------------------------------------------------------------------

class MultiAxesChart(Chart):
    """A chart that lays out several small axes inside the place of one host axes."""

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.subaxes: list = []
        ax.set_axis_off()
        ax._mylab_chart_host = True

    def label_axes(self):
        return self.axes

    def host_box(self):
        return self.axes.get_position().bounds

    def new_axes(self, rect, **kw):
        st = P._state(self.interp)
        st.counter = getattr(st, "counter", 0) + 1
        a = self.axes.figure.add_axes(rect, label=f"mylab-chart-{st.counter}", **kw)
        a._mylab_internal = True
        a._mylab_matrix_of = self.axes  # removed with the host axes when that is cleared
        P._tick_aligned(a)
        self.subaxes.append(a)
        return a

    def clear(self) -> None:
        for a in self.subaxes:
            try:
                a.remove()
            except (ValueError, KeyError, AttributeError):
                pass
        self.subaxes = []
        super().clear()

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        for a in self.subaxes:
            a.set_visible(self.visible)
        self.axes.title.set_visible(self.visible)

    def show_host_labels(self) -> None:
        """The host axes is hidden, but its title and axis labels are drawn around the small axes."""
        ax = self.axes
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            t.set_visible(True)
        ax.set_axis_off()
        ax.title.set_visible(True)
        ax.xaxis.label.set_visible(True)
        ax.yaxis.label.set_visible(True)


def _host_text(chart, which: str):
    """Text object drawn by the chart for a title or axis label of the hidden host axes."""
    key = f"_host_{which}"
    t = getattr(chart, key, None)
    fig = chart.axes.figure
    x0, y0, w, hh = chart.host_box()
    if t is None or t.figure is None:
        if which == "title":
            t = fig.text(x0 + w / 2, y0 + hh + 0.012, "", ha="center", va="bottom", fontweight="bold", fontsize=11)
        elif which == "xlabel":
            t = fig.text(x0 + w / 2, y0 - 0.075, "", ha="center", va="top", fontsize=10)
        else:
            t = fig.text(x0 - 0.085, y0 + hh / 2, "", ha="right", va="center", rotation=90, fontsize=10)
        setattr(chart, key, t)
    return t


def _host_label_prop(which: str) -> tuple:
    def g(h):
        t = _host_text(h.obj, which)
        raw = getattr(t, "_mylab_raw", None)
        return char(raw if raw is not None else t.get_text())

    def s(h, v):
        P.set_text(h.interp, _host_text(h.obj, which), P._label_text(v))
    return g, s


class StackedPlot(MultiAxesChart):
    kind = "stackedplot"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.x = np.zeros(0)
        self.xobj = None            # datetime/duration x values (kept to give the axis its ruler)
        self.ys: list = []          # one array (n-by-k) for each stacked axes
        self.labels: list[str] = []
        self.grid = "off"
        self.linewidth = 0.5
        self.linestyle = "-"
        self.marker = "none"
        self.color: Any = None
        self.xlabel = ""
        self.table = None
        self.lines: list = []

    def draw(self) -> None:
        x0, y0, w, hh = self.host_box()
        n = len(self.ys)
        if n == 0:
            return
        gap = 0.02
        each = (hh - gap * (n - 1)) / n
        self.lines = []
        ruler = None
        xnum = self.x
        if self.xobj is not None:
            ruler = P.new_ruler_for(self.xobj)
            xnum = P.ruler_numbers(ruler, self.xobj).flatten(order="F")
        for k, Y in enumerate(self.ys):
            a = self.new_axes([x0, y0 + hh - (k + 1) * each - k * gap, w, each])
            Y = np.asarray(Y, dtype=float)
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            kw = {"linewidth": self.linewidth, "linestyle": P._mpl_linestyle(self.linestyle),
                  "marker": P._mpl_marker(self.marker)}
            if self.color is not None:
                kw["color"] = self.color
            self.lines.append(a.plot(xnum, Y, **kw))
            a.set_ylabel(self.labels[k] if k < len(self.labels) else "", rotation=0, ha="right", va="center")
            a.grid(self.grid == "on")
            if ruler is not None:
                P.install_ruler(a, "x", ruler)
            if k < n - 1:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel(self.xlabel)
            if xnum.size:
                a.set_xlim(float(np.nanmin(xnum)), float(np.nanmax(xnum)) if np.nanmax(xnum) > np.nanmin(xnum)
                           else float(np.nanmin(xnum)) + 1)


def _sp_labels_set(h, v):
    names = P._text_list(v)
    if len(names) != len(h.obj.ys):
        raise MatlabError(f"DisplayLabels needs {len(h.obj.ys)} labels, one for each stacked plot.")
    h.obj.labels = names
    h.obj.redraw()


def _sp_axes_props(h):
    return P._handle_list([P._wrap(h.interp, "axes", a) for a in h.obj.subaxes])


_chart_setup("stackedplot", "matlab.graphics.chart.StackedLineChart", {
    "Title": _host_label_prop("title"),
    "XLabel": _attr("xlabel", lambda v: P._label_text(v)),
    "DisplayLabels": (lambda h: P._cell_column([char(s) for s in h.obj.labels]), _sp_labels_set),
    "XData": _ro(lambda c: c.xobj if c.xobj is not None else c.x.reshape(-1, 1)),
    "YData": _ro(lambda c: np.column_stack([np.asarray(y, float).reshape(len(c.x), -1) for y in c.ys])
                 if c.ys else _EMPTY),
    "GridVisible": _attr("grid", _choice_conv("GridVisible", ("on", "off"))),
    "LineWidth": _attr("linewidth", _num_conv("LineWidth", 0.01, 100)),
    "LineStyle": _attr("linestyle", lambda v: text(v)),
    "Marker": _attr("marker", lambda v: text(v)),
    "Color": _attr("color", _color_conv, out=lambda c: P._rgb(c) if c is not None else P._rgb("C0")),
    "SourceTable": _ro(lambda c: c.table if c.table is not None else _EMPTY),
    "AxesProperties": (_sp_axes_props, None),
    "Position": (lambda h: P._row(h.obj.host_box()),
                 lambda h, v: (h.obj.axes.set_position([float(x) for x in _vec(v)[:4]]), h.obj.redraw())),
}, ["DisplayLabels", "XData", "YData", "Title", "XLabel", "GridVisible"])

_STACKED_OPTS = ("Title", "XLabel", "DisplayLabels", "GridVisible", "LineWidth", "LineStyle", "Marker", "Color",
                 "XVariable", "Tag", "Visible", "Position", "CombineMatchingNames")


@builtin("stackedplot")
def _stackedplot(interp, args, nargout):
    """stackedplot(tbl) or stackedplot(x, Y): the variables as line plots stacked above a shared x-axis."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _STACKED_OPTS, "stackedplot")
    spec = None
    data = []
    for a in args:
        if is_text(a) and P._is_linespec(text(a)) and data and not _is_table(data[0]):
            spec = text(a)
        else:
            data.append(a)
    if not data:
        raise MatlabError("stackedplot needs a table or numeric data.")
    ax, _ = _axes_from(interp, lead)
    chart = StackedPlot(interp, ax)
    if _is_table(data[0]):
        tbl = data[0]
        chart.table = tbl
        names = list(tbl.names)
        if len(data) > 1:
            picked = []
            v = data[1]
            if isinstance(v, (Cell, StringArray)) or is_text(v):
                for nm in P._text_list(v):
                    if nm not in tbl.names:
                        raise MatlabError(f"Unrecognized table variable name '{nm}'.")
                    picked.append(nm)
            else:
                picked = [tbl.names[int(i) - 1] for i in _vec(v)]
            names = picked
        xname = None
        if "XVariable" in opts:
            xname, xcol = _column(tbl, opts.pop("XVariable"), "stackedplot")
            names = [nm for nm in names if nm != xname]
            if P._is_time_like(xcol):
                chart.xobj = xcol
                chart.x = np.asarray(xcol.data, dtype=float).flatten(order="F")
            else:
                chart.x = _vec(xcol)
            chart.xlabel = xname
        else:
            chart.x = np.arange(1.0, tbl.nrows + 1)
            row_times = getattr(tbl, "row_times", None)
            if row_times is not None and P._is_time_like(row_times):
                chart.xobj = row_times
                chart.x = np.asarray(row_times.data, dtype=float).flatten(order="F")
        for nm in names:
            col = tbl.columns[tbl.names.index(nm)]
            if isinstance(col, np.ndarray) and col.dtype.kind in "biuf":
                chart.ys.append(np.real(np.asarray(col, dtype=float)))
                chart.labels.append(nm)
            elif hasattr(col, "categories") and hasattr(col, "data"):
                chart.ys.append(np.asarray(col.data, dtype=float))
                chart.labels.append(nm)
        if not chart.ys:
            raise MatlabError("stackedplot: the table has no numeric variables to plot.")
    else:
        if len(data) == 1:
            Y = _real(data[0])
            chart.x = np.arange(1.0, Y.shape[0] + 1)
        elif len(data) == 2:
            if P._is_time_like(data[0]):
                chart.xobj = data[0]
                chart.x = np.asarray(data[0].data, dtype=float).flatten(order="F")
            else:
                chart.x = _vec(data[0])
            Y = _real(data[1])
        else:
            raise MatlabError("stackedplot expects stackedplot(Y), stackedplot(x, Y) or stackedplot(tbl).")
        if Y.shape[0] == 1:
            Y = Y.T
        if Y.shape[0] != chart.x.size:
            raise MatlabError("stackedplot: x must have one value for each row of Y.")
        chart.ys = [Y[:, j] for j in range(Y.shape[1])]
        chart.labels = [f"Column {j + 1}" for j in range(Y.shape[1])]
    if spec:
        import re
        m = re.search(r"[rgbcmykw]", spec)
        if m:
            chart.color = _color_conv(char(m.group(0)))
        marker = "".join(ch for ch in spec if ch in "o+*.xsd^v<>ph")
        style = re.sub(r"[rgbcmykwo+*.xsd^v<>ph]", "", spec)
        if marker:
            chart.marker = marker
            chart.linestyle = style or "none"
        elif style:
            chart.linestyle = style
    opts.pop("CombineMatchingNames", None)
    chart.draw()
    h = _register(interp, chart)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


class ScatterHistogram(MultiAxesChart):
    kind = "scatterhistogram"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.group_idx = None
        self.group_names: list[str] = []
        self.nbins = None
        self.hist_style = "stairs"
        self.location = "southwest"
        self.legend = "auto"
        self.legend_title = ""
        self.marker_size = 36.0
        self.marker = "o"
        self.colors = None
        self.xlabel = ""
        self.ylabel = ""
        self.xlim = None
        self.ylim = None

    def draw(self) -> None:
        x0, y0, w, hh = self.host_box()
        frac = 0.22
        loc = self.location
        left = "west" in loc
        bottom = "south" in loc
        sx = x0 if left else x0 + w * frac
        sy = y0 if bottom else y0 + hh * frac
        sw, sh = w * (1 - frac) - 0.01, hh * (1 - frac) - 0.01
        sc = self.new_axes([sx, sy, sw, sh])
        hx = self.new_axes([sx, sy + sh + 0.01 if bottom else y0, sw, hh * frac])
        hy = self.new_axes([sx + sw + 0.01 if left else x0, sy, w * frac, sh])
        ngroups = max(len(self.group_names), 1)
        from .plot_objects import default_colors
        colors = self.colors if self.colors is not None else default_colors()
        good = np.isfinite(self.x) & np.isfinite(self.y)
        nb = self.nbins
        ex = P._hist_edges(self.x[good], nbins=int(nb[0]) if nb is not None else None)
        ey = P._hist_edges(self.y[good], nbins=int(nb[-1]) if nb is not None else None)
        for g in range(ngroups):
            pick = good if self.group_idx is None else good & (self.group_idx == g + 1)
            c = tuple(colors[g % len(colors)])
            label = self.group_names[g] if self.group_names else None
            sc.scatter(self.x[pick], self.y[pick], s=self.marker_size, marker=P._mpl_marker(self.marker),
                       facecolors="none", edgecolors=c, label=label)
            for data, edges, a, orient in ((self.x[pick], ex, hx, "vertical"), (self.y[pick], ey, hy, "horizontal")):
                counts, _ = np.histogram(data, edges)
                dens = counts / max(counts.sum(), 1) / np.diff(edges)
                if self.hist_style == "bar":
                    if orient == "vertical":
                        a.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color=(*c, 0.4), edgecolor=c)
                    else:
                        a.barh(edges[:-1], dens, height=np.diff(edges), align="edge", color=(*c, 0.4), edgecolor=c)
                elif self.hist_style == "smooth":
                    pts = np.linspace(edges[0], edges[-1], 200)
                    f = _kde(data, pts)
                    a.plot(pts, f, color=c) if orient == "vertical" else a.plot(f, pts, color=c)
                else:
                    a.stairs(dens, edges, color=c, orientation=orient)
        if self.xlim is not None:
            sc.set_xlim(*self.xlim)
        if self.ylim is not None:
            sc.set_ylim(*self.ylim)
        hx.set_xlim(sc.get_xlim())
        hy.set_ylim(sc.get_ylim())
        for a in (hx, hy):
            a.set_axis_off()
        sc.set_xlabel(self.xlabel)
        sc.set_ylabel(self.ylabel)
        show = self.legend == "on" or (self.legend == "auto" and ngroups > 1)
        if show and self.group_names:
            sc.legend(title=self.legend_title or None, loc="best", fontsize=8)
        self.scatter_axes = sc


def _sh_limits(which: str):
    def g(h):
        a = getattr(h.obj, "scatter_axes", None)
        return P._row(a.get_xlim() if which == "x" else a.get_ylim()) if a is not None else P._row([0, 1])

    def s(h, v):
        lim = _vec(v)
        if lim.size != 2 or not lim[0] < lim[1]:
            raise MatlabError(f"{which.upper()}Limits must be [min max] with min < max.")
        setattr(h.obj, f"{which}lim", (float(lim[0]), float(lim[1])))
        h.obj.redraw()
    return g, s


def _sh_numbins_conv(v):
    n = _vec(v)
    if n.size not in (1, 2) or np.any(n < 1):
        raise MatlabError("NumBins must be a positive whole number or [nx ny].")
    return n


_chart_setup("scatterhistogram", "matlab.graphics.chart.ScatterHistogramChart", {
    "Title": _host_label_prop("title"),
    "XLabel": _attr("xlabel", lambda v: P._label_text(v)),
    "YLabel": _attr("ylabel", lambda v: P._label_text(v)),
    "XData": _ro(lambda c: c.x.reshape(-1, 1)), "YData": _ro(lambda c: c.y.reshape(-1, 1)),
    "GroupData": _ro(lambda c: P._cell_column([char(c.group_names[int(i) - 1]) if np.isfinite(i) else char("")
                                                for i in c.group_idx]) if c.group_idx is not None else _EMPTY),
    "NumBins": _attr("nbins", _sh_numbins_conv, out=lambda n: P._row(n) if n is not None else _EMPTY),
    "HistogramDisplayStyle": _attr("hist_style", _choice_conv("HistogramDisplayStyle",
                                                              ("stairs", "bar", "smooth"))),
    "ScatterPlotLocation": _attr("location", _choice_conv("ScatterPlotLocation",
                                                          ("southwest", "southeast", "northeast", "northwest"))),
    "LegendVisible": _attr("legend", _choice_conv("LegendVisible", ("on", "off"))),
    "LegendTitle": _attr("legend_title", lambda v: P._label_text(v)),
    "MarkerSize": _attr("marker_size", _num_conv("MarkerSize", 0.1, 1e5)),
    "MarkerStyle": _attr("marker", lambda v: text(v)),
    "Color": _attr("colors", _colors_conv,
                   out=lambda c: c if c is not None else _EMPTY),
    "XLimits": _sh_limits("x"), "YLimits": _sh_limits("y"),
}, ["XData", "YData", "Title", "XLabel", "YLabel", "NumBins", "HistogramDisplayStyle"])

_SH_OPTS = ("Title", "XLabel", "YLabel", "GroupVariable", "GroupData", "NumBins", "HistogramDisplayStyle",
            "ScatterPlotLocation", "LegendVisible", "LegendTitle", "MarkerSize", "MarkerStyle", "Color",
            "XLimits", "YLimits", "Tag", "Visible")


@builtin("scatterhistogram")
def _scatterhistogram(interp, args, nargout):
    """scatterhistogram(tbl, xvar, yvar) or scatterhistogram(x, y): a scatter plot with marginal histograms."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _SH_OPTS, "scatterhistogram")
    _no_text_left(args[1:] if args and _is_table(args[0]) else args, "scatterhistogram",
                  allowed=tuple(n.lower() for n in (args[0].names if args and _is_table(args[0]) else ())))
    ax, _ = _axes_from(interp, lead)
    chart = ScatterHistogram(interp, ax)
    group = None
    if args and _is_table(args[0]):
        tbl = args[0]
        if len(args) == 1:
            if len(tbl.names) < 2:
                raise MatlabError("scatterhistogram needs a table with at least two variables.")
            xn, yn = tbl.names[0], tbl.names[1]
        elif len(args) == 3:
            xn, _ = _column(tbl, args[1], "scatterhistogram")
            yn, _ = _column(tbl, args[2], "scatterhistogram")
        else:
            raise MatlabError("scatterhistogram(tbl, xvar, yvar) needs both variable names.")
        chart.x, chart.y = _vec(tbl.columns[tbl.names.index(xn)]), _vec(tbl.columns[tbl.names.index(yn)])
        chart.xlabel, chart.ylabel = xn, yn
        if "GroupVariable" in opts:
            gname, group = _column(tbl, opts.pop("GroupVariable"), "scatterhistogram")
            chart.legend_title = gname
    else:
        if len(args) != 2:
            raise MatlabError("scatterhistogram expects scatterhistogram(x, y) or scatterhistogram(tbl, xvar, yvar).")
        chart.x, chart.y = _vec(args[0]), _vec(args[1])
        if "GroupVariable" in opts:
            raise MatlabError("scatterhistogram: GroupVariable needs table input; use GroupData with vectors.")
    if chart.x.size != chart.y.size:
        raise MatlabError("scatterhistogram: x and y must have the same number of elements.")
    if "GroupData" in opts:
        group = opts.pop("GroupData")
    if group is not None:
        from .ndmath import group_index
        idx, names, _ = group_index(group)
        if idx.size != chart.x.size:
            raise MatlabError("scatterhistogram: the group data must have one value for each point.")
        chart.group_idx, chart.group_names = idx, names
    chart.draw()
    h = _register(interp, chart)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


class ParallelPlot(Chart):
    kind = "parallelplot"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.columns: list = []        # each: numeric array, or (codes, names) for a categorical coordinate
        self.names: list[str] = []
        self.group_idx = None
        self.group_names: list[str] = []
        self.norm = "range"
        self.linewidth = 1.0
        self.linestyle = "-"
        self.alpha = 0.7
        self.marker = "none"
        self.colors = None
        self.jitter = 0.0
        self.legend = "auto"
        self.table = None

    def normalized(self):
        out, ticks = [], []
        for col in self.columns:
            if isinstance(col, tuple):
                codes, names = col
                n = max(len(names), 1)
                v = (np.asarray(codes, dtype=float) - 1) / max(n - 1, 1)
                out.append(v)
                ticks.append(([k / max(n - 1, 1) for k in range(n)], list(names)))
                continue
            v = np.asarray(col, dtype=float)
            f = v[np.isfinite(v)]
            lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
            if self.norm == "range":
                out.append((v - lo) / (hi - lo) if hi > lo else np.full(v.shape, 0.5))
                ticks.append(([0.0, 1.0], [P._fmt_num(lo), P._fmt_num(hi)]))
            elif self.norm == "zscore":
                sd = float(np.std(f, ddof=1)) if f.size > 1 else 0.0
                out.append((v - float(f.mean())) / sd if sd > 0 else np.zeros(v.shape))
                ticks.append(None)
            elif self.norm == "scale":
                sd = float(np.std(f, ddof=1)) if f.size > 1 else 0.0
                out.append(v / sd if sd > 0 else v)
                ticks.append(None)
            elif self.norm == "center":
                out.append(v - (float(f.mean()) if f.size else 0.0))
                ticks.append(None)
            else:
                out.append(v)
                ticks.append(None)
        return out, ticks

    def draw(self) -> None:
        from .plot_objects import default_colors
        ax = self.axes
        cols, ticks = self.normalized()
        k = len(cols)
        if k == 0:
            return
        M = np.column_stack(cols)
        if self.jitter:
            rng = np.random.default_rng(0)
            cat = [isinstance(c, tuple) for c in self.columns]
            M = M + (rng.random(M.shape) - 0.5) * self.jitter * np.array(cat, dtype=float) / max(
                max((len(c[1]) for c in self.columns if isinstance(c, tuple)), default=1) - 1, 1)
        colors = self.colors if self.colors is not None else default_colors()
        xs = np.arange(1, k + 1)
        ngroups = max(len(self.group_names), 1)
        for g in range(ngroups):
            pick = np.ones(M.shape[0], dtype=bool) if self.group_idx is None else self.group_idx == g + 1
            c = tuple(colors[g % len(colors)])
            lines = ax.plot(xs, M[pick].T, color=c, linewidth=self.linewidth, alpha=self.alpha,
                            linestyle=P._mpl_linestyle(self.linestyle), marker=P._mpl_marker(self.marker))
            if lines and self.group_names:
                lines[0].set_label(self.group_names[g])
            self.add(lines)
        for x in xs:
            self.add(ax.axvline(x, color="0.3", linewidth=0.8, zorder=0.5))
        _set_tick_text(ax, "x", xs, self.names)
        ax.set_xlim(0.9, k + 0.1)
        if self.norm == "range":
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([])
            for x, t in zip(xs, ticks):
                if t is None:
                    continue
                for pos, lab in zip(*t):
                    self.add(ax.text(x + 0.02, pos, lab, fontsize=7, va="center", ha="left", color="0.25"))
        show = self.legend == "on" or (self.legend == "auto" and ngroups > 1)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if show and self.group_names:
            ax.legend(loc="upper right", fontsize=8)


def _pp_ticklabels_set(h, v):
    names = P._text_list(v)
    if len(names) != len(h.obj.columns):
        raise MatlabError(f"CoordinateTickLabels needs {len(h.obj.columns)} labels, one for each coordinate.")
    h.obj.names = names
    h.obj.redraw()


def _pp_data(c):
    cols = [np.asarray(col[0], dtype=float) if isinstance(col, tuple) else np.asarray(col, dtype=float)
            for col in c.columns]
    return np.column_stack(cols) if cols else _EMPTY


_chart_setup("parallelplot", "matlab.graphics.chart.ParallelCoordinatesPlot", {
    "Title": _label_prop("Title"), "XLabel": _label_prop("XLabel"), "YLabel": _label_prop("YLabel"),
    "Data": _ro(_pp_data),
    "CoordinateTickLabels": (lambda h: P._cell_column([char(s) for s in h.obj.names]), _pp_ticklabels_set),
    "CoordinateData": _ro(lambda c: P._row(np.arange(1, len(c.columns) + 1))),
    "GroupData": _ro(lambda c: P._cell_column([char(c.group_names[int(i) - 1]) if np.isfinite(i) else char("")
                                                for i in c.group_idx]) if c.group_idx is not None else _EMPTY),
    "DataNormalization": _attr("norm", _choice_conv("DataNormalization",
                                                    ("range", "none", "zscore", "scale", "center"))),
    "LineWidth": _attr("linewidth", _num_conv("LineWidth", 0.01, 100)),
    "LineStyle": _attr("linestyle", lambda v: text(v)),
    "LineAlpha": _attr("alpha", _num_conv("LineAlpha", 0, 1)),
    "MarkerStyle": _attr("marker", lambda v: text(v)),
    "Jitter": _attr("jitter", _num_conv("Jitter", 0, 1)),
    "LegendVisible": _attr("legend", _choice_conv("LegendVisible", ("on", "off"))),
    "Color": _attr("colors", _colors_conv,
                   out=lambda c: c if c is not None else _EMPTY),
    "SourceTable": _ro(lambda c: c.table if c.table is not None else _EMPTY),
}, ["Data", "CoordinateTickLabels", "DataNormalization", "Title"])

_PP_OPTS = ("Title", "XLabel", "YLabel", "CoordinateVariables", "GroupVariable", "GroupData",
            "CoordinateTickLabels", "CoordinateData", "DataNormalization", "LineWidth", "LineStyle", "LineAlpha",
            "MarkerStyle", "Jitter", "LegendVisible", "Color", "Tag", "Visible")


@builtin("parallelplot")
def _parallelplot(interp, args, nargout):
    """parallelplot(tbl) or parallelplot(matrix): one line per row across parallel coordinate axes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _PP_OPTS, "parallelplot")
    if len(args) != 1:
        raise MatlabError("parallelplot needs a table or a matrix.")
    ax, _ = _axes_from(interp, lead)
    chart = ParallelPlot(interp, ax)
    group = None
    if _is_table(args[0]):
        tbl = args[0]
        chart.table = tbl
        names = list(tbl.names)
        gname = None
        if "GroupVariable" in opts:
            gname, group = _column(tbl, opts.pop("GroupVariable"), "parallelplot")
        if "CoordinateVariables" in opts:
            v = opts.pop("CoordinateVariables")
            if isinstance(v, (Cell, StringArray)) or is_text(v):
                names = P._text_list(v)
                for nm in names:
                    if nm not in tbl.names:
                        raise MatlabError(f"Unrecognized table variable name '{nm}'.")
            else:
                names = [tbl.names[int(i) - 1] for i in _vec(v)]
        elif gname is not None:
            names = [nm for nm in names if nm != gname]
        for nm in names:
            col = tbl.columns[tbl.names.index(nm)]
            if isinstance(col, np.ndarray) and col.dtype.kind in "biuf":
                chart.columns.append(np.real(np.asarray(col, dtype=float)).reshape(tbl.nrows, -1)[:, 0])
            else:
                from .ndmath import group_index
                idx, gnames, _ = group_index(col)
                chart.columns.append((idx, gnames))
            chart.names.append(nm)
    else:
        M = _real(args[0])
        if M.ndim != 2:
            raise MatlabError("parallelplot: the data must be a matrix.")
        if "CoordinateData" in opts:
            pick = [int(i) - 1 for i in _vec(opts.pop("CoordinateData"))]
            M = M[:, pick]
        chart.columns = [M[:, j] for j in range(M.shape[1])]
        chart.names = [str(j + 1) for j in range(M.shape[1])]
        if "GroupVariable" in opts or "CoordinateVariables" in opts:
            raise MatlabError("parallelplot: GroupVariable and CoordinateVariables need table input.")
    if "GroupData" in opts:
        group = opts.pop("GroupData")
    if group is not None:
        from .ndmath import group_index
        idx, gnames, _ = group_index(group)
        n = len(chart.columns[0]) if chart.columns else 0
        if idx.size != n:
            raise MatlabError("parallelplot: the group data must have one value for each row.")
        chart.group_idx, chart.group_names = idx, gnames
    chart.draw()
    h = _register(interp, chart)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


@builtin("parallelcoords")
def _parallelcoords(interp, args, nargout):
    """parallelcoords(X, 'Group', g, 'Labels', names): a parallel coordinates plot; returns the lines."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, ("Group", "Labels", "Standardize", "Quantile", "LineWidth", "Color", "LineStyle",
                                 "Marker"), "parallelcoords")
    _no_text_left(args, "parallelcoords")
    if len(args) != 1:
        raise MatlabError("parallelcoords needs a data matrix.")
    X = _real(args[0])
    std = text(opts["Standardize"]).lower() if "Standardize" in opts else "off"
    if std == "on":
        sd = np.nanstd(X, axis=0, ddof=1)
        X = (X - np.nanmean(X, axis=0)) / np.where(sd > 0, sd, 1.0)
    elif std in ("pca", "pcastd"):
        Xc = X - np.nanmean(X, axis=0)
        u, s, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
        X = Xc @ vt.T
        if std == "pcastd":
            sd = np.std(X, axis=0, ddof=1)
            X = X / np.where(sd > 0, sd, 1.0)
    elif std != "off":
        raise MatlabError("Standardize must be 'on', 'off', 'PCA' or 'PCAStd'.")
    n, k = X.shape
    labels = P._text_list(opts["Labels"]) if "Labels" in opts else [str(j + 1) for j in range(k)]
    if len(labels) != k:
        raise MatlabError("parallelcoords: give one label for each column.")
    idx, names = np.ones(n), [""]
    if "Group" in opts:
        from .ndmath import group_index
        idx, names, _ = group_index(opts["Group"])
        if idx.size != n:
            raise MatlabError("parallelcoords: Group must have one value for each row.")
    ax, _ = _axes_from(interp, lead)
    from .plot_objects import default_colors
    colors = default_colors()
    xs = np.arange(1, k + 1)
    out = []
    kw: dict = {}
    if "LineWidth" in opts:
        kw["linewidth"] = P._scalar(opts["LineWidth"])
    if "LineStyle" in opts:
        kw["linestyle"] = P._mpl_linestyle(text(opts["LineStyle"]))
    if "Marker" in opts:
        kw["marker"] = P._mpl_marker(text(opts["Marker"]))
    for g in range(len(names)):
        rows = X[idx == g + 1]
        c = P._color(opts["Color"]) if "Color" in opts else tuple(colors[g % len(colors)])
        if "Quantile" in opts:
            a = P._scalar(opts["Quantile"])
            if not 0 < a < 0.5:
                raise MatlabError("Quantile must be between 0 and 0.5.")
            med = np.array([_quantiles(rows[:, j], [0.5])[0] for j in range(k)])
            lo = np.array([_quantiles(rows[:, j], [a])[0] for j in range(k)])
            hi = np.array([_quantiles(rows[:, j], [1 - a])[0] for j in range(k)])
            lines = ax.plot(xs, med, color=c, **kw) + ax.plot(xs, lo, color=c, linestyle=":") \
                + ax.plot(xs, hi, color=c, linestyle=":")
        else:
            lines = ax.plot(xs, rows.T, color=c, **kw)
        if lines and names[g]:
            lines[0].set_label(names[g])
        out.extend(lines)
    _set_tick_text(ax, "x", xs, labels)
    ax.set_xlim(1, k)
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Coordinate Value")
    if "Group" in opts:
        ax.legend(loc="best", fontsize=8)
    P._redraw(interp)
    return P._ret(interp, "line", out, nargout)


def _matrix_axes(interp, ax, rows: int, cols: int):
    """A rows-by-cols grid of small axes in the place of ``ax`` (which is hidden and returned as the big axes)."""
    x0, y0, w, hh = ax.get_position().bounds
    ax.set_axis_off()
    gap = 0.012
    cw, ch = (w - gap * (cols - 1)) / cols, (hh - gap * (rows - 1)) / rows
    st = P._state(interp)
    grid = []
    for r in range(rows):
        line = []
        for c in range(cols):
            st.counter = getattr(st, "counter", 0) + 1
            a = ax.figure.add_axes([x0 + c * (cw + gap), y0 + hh - (r + 1) * ch - r * gap, cw, ch],
                                   label=f"mylab-matrix-{st.counter}")
            a._mylab_internal = True
            a._mylab_matrix_of = ax
            P._tick_aligned(a)
            a.tick_params(labelsize=7)
            if c > 0:
                a.tick_params(labelleft=False)
            if r < rows - 1:
                a.tick_params(labelbottom=False)
            line.append(a)
        grid.append(line)
    return grid


def _plotmatrix_core(interp, lead: list, X, Y, group, colors, markers, sizes, show_legend: bool,
                     diag: str, xnames, ynames, spec: str, nargout: int, name: str):
    same = Y is None
    if same:
        Y = X
    if X.shape[0] != Y.shape[0]:
        raise MatlabError(f"{name}: X and Y must have the same number of rows.")
    ax, _ = _axes_from(interp, lead)
    rows, cols = Y.shape[1], X.shape[1]
    grid = _matrix_axes(interp, ax, rows, cols)
    n = X.shape[0]
    if group is not None:
        from .ndmath import group_index
        gidx, gnames, _ = group_index(group)
        if gidx.size != n:
            raise MatlabError(f"{name}: the grouping variable must have one value for each row.")
    else:
        gidx, gnames = np.ones(n), [""]
    from .plot_objects import default_colors
    palette = default_colors()
    spec_color = P.spec_color(spec)
    spec_marker = "".join(ch for ch in spec if ch in "o+*.xsd^v<>ph") or "."
    S, H, HAx = [], [], []
    for r in range(rows):
        for c in range(cols):
            a = grid[r][c]
            if same and r == c and diag != "none":
                for g in range(len(gnames)):
                    col = _group_color(colors, g, palette) if group is not None else (
                        spec_color if spec_color else tuple(palette[0]))
                    data = X[gidx == g + 1, c]
                    data = data[np.isfinite(data)]
                    if diag == "stairs" or (diag == "auto" and len(gnames) > 1):
                        counts, edges = np.histogram(data, P._hist_edges(X[np.isfinite(X[:, c]), c]))
                        H.append(a.stairs(counts, edges, color=col))
                    else:
                        cnt, edges, patches = a.hist(data, bins=P._hist_edges(X[np.isfinite(X[:, c]), c]),
                                                     color=col, edgecolor="k", linewidth=0.3)
                        H.append(patches)
                a.tick_params(labelleft=False)
                HAx.append(a)
                S.append(None)
                continue
            for g in range(len(gnames)):
                pick = gidx == g + 1
                col = _group_color(colors, g, palette) if group is not None else (
                    spec_color if spec_color else tuple(palette[0]))
                mk = P._mpl_marker(markers[g % len(markers)]) if markers else spec_marker
                ms = sizes[g % len(sizes)] if sizes else 6.0
                ln = a.plot(X[pick, c], Y[pick, r], linestyle="None", marker=mk, markersize=ms, color=col,
                            label=gnames[g] or None)
                if g == 0:
                    S.append(ln[0])
        if ynames:
            grid[r][0].set_ylabel(ynames[r], fontsize=8)
    for c in range(cols):
        if xnames:
            grid[rows - 1][c].set_xlabel(xnames[c], fontsize=8)
    if same and diag != "none":  # the histograms use their own vertical scale; the scatter axes share limits
        for c in range(cols):
            lims = [grid[r][c].get_xlim() for r in range(rows) if r != c]
            if lims:
                lo, hi = min(v[0] for v in lims), max(v[1] for v in lims)
                grid[c][c].set_xlim(lo, hi)
    if show_legend and group is not None and len(gnames) > 1:
        grid[0][cols - 1].legend(loc="upper right", fontsize=7)
    P._redraw(interp)
    if nargout <= 0:
        return None
    wrap_line = [P._wrap(interp, "line", s) if s is not None else P.GraphicsHandle(interp, "placeholder", None)
                 for s in S]
    s_arr = P.GraphicsArray([wrap_line[r * cols + c] for c in range(cols) for r in range(rows)],
                            shape=(rows, cols)) if rows * cols > 1 else wrap_line[0]
    ax_handles = [P._wrap(interp, "axes", grid[r][c]) for c in range(cols) for r in range(rows)]
    ax_arr = P.GraphicsArray(ax_handles, shape=(rows, cols)) if rows * cols > 1 else ax_handles[0]
    big = P._wrap(interp, "axes", ax)
    h_arr = P._handle_list([P._wrap(interp, "patch", list(h) if not hasattr(h, "get_path") else h) for h in H])
    hax = P._handle_list([P._wrap(interp, "axes", a) for a in HAx])
    return (s_arr, ax_arr, big, h_arr, hax)[:max(nargout, 1)] if nargout > 1 else s_arr


def _group_color(colors, g: int, palette):
    if colors is None:
        return tuple(palette[g % len(palette)])
    return colors[g % len(colors)]


@builtin("plotmatrix")
def _plotmatrix(interp, args, nargout):
    """plotmatrix(X) or plotmatrix(X, Y): scatter plots of every pair of columns; histograms on the diagonal."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    spec = ""
    nums = []
    for a in args:
        if is_text(a):
            if not P._is_linespec(text(a)):
                raise MatlabError(f"plotmatrix: unknown option '{text(a)}'.")
            spec = text(a)
        else:
            nums.append(_real(a))
    if not nums or len(nums) > 2:
        raise MatlabError("plotmatrix expects plotmatrix(X) or plotmatrix(X, Y).")
    X = nums[0] if nums[0].shape[0] > 1 else nums[0].T
    Y = None
    if len(nums) == 2:
        Y = nums[1] if nums[1].shape[0] > 1 else nums[1].T
    return _plotmatrix_core(interp, lead, X, Y, None, None, None, None, False, "hist", None, None, spec, nargout,
                            "plotmatrix")


def _color_list(v) -> list | None:
    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
        return None
    if is_text(v):
        s = text(v)
        if all(ch in "rgbcmykw" for ch in s):
            return [P.LETTER_RGB[ch] for ch in s]
        return [P._color(v)]
    if isinstance(v, (Cell, StringArray)):
        return [P._color(char(s)) for s in P._text_list(v)]
    m = _real(v)
    if m.ndim == 2 and m.shape[1] == 3:
        return [tuple(float(t) for t in row) for row in m]
    raise MatlabError("Colours must be colour letters such as 'rgb', a cell array of names, or an m-by-3 matrix.")


@builtin("gplotmatrix")
def _gplotmatrix(interp, args, nargout):
    """gplotmatrix(X, Y, group, clr, sym, siz, doleg, dispopt, xnam, ynam): scatter plot matrix by group."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    if not args:
        raise MatlabError("gplotmatrix needs data, e.g. gplotmatrix(X, [], group).")
    args += [None] * (10 - len(args))
    empty = lambda v: v is None or (isinstance(v, np.ndarray) and v.size == 0 and not is_char(v))  # noqa: E731
    X = _real(args[0])
    Y = None if empty(args[1]) else _real(args[1])
    group = None if empty(args[2]) else args[2]
    colors = _color_list(args[3])
    markers = None if empty(args[4]) else (list(text(args[4])) if is_text(args[4]) else P._text_list(args[4]))
    sizes = None if empty(args[5]) else [float(v) for v in _vec(args[5])]
    doleg = True if empty(args[6]) else (text(args[6]).lower() == "on" if is_text(args[6]) else bool(_vec(args[6])[0]))
    diag = "auto" if empty(args[7]) else text(args[7]).lower()
    if diag not in ("auto", "none", "hist", "stairs", "grpbars", "variable"):
        raise MatlabError("gplotmatrix: dispopt must be 'none', 'hist', 'stairs', 'grpbars' or 'variable'.")
    if diag in ("grpbars",):
        diag = "hist"
    xnam = None if empty(args[8]) else P._text_list(args[8])
    ynam = None if empty(args[9]) else P._text_list(args[9])
    if diag == "variable":
        diag = "none"
    return _plotmatrix_core(interp, lead, X, Y, group, colors, markers, sizes, doleg, diag, xnam,
                            ynam if ynam is not None else (xnam if Y is None else None), "", nargout,
                            "gplotmatrix")


@builtin("gscatter")
def _gscatter(interp, args, nargout):
    """gscatter(x, y, group, clr, sym, siz, doleg, xnam, ynam): a scatter plot with one marker style per group."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    if len(args) < 3:
        raise MatlabError("gscatter needs x, y and a grouping variable.")
    if len(args) > 9:
        raise MatlabError("gscatter: too many input arguments.")
    args += [None] * (9 - len(args))
    empty = lambda v: v is None or (isinstance(v, np.ndarray) and v.size == 0 and not is_char(v))  # noqa: E731
    x, y = _vec(args[0]), _vec(args[1])
    from ._stats_util import group_list, multi_group
    # a cell array of grouping variables makes one group per combination, named like "a,1"
    codes, names, _, _ = multi_group(group_list(args[2]), sep=",")
    idx = codes + 1
    if not x.size == y.size == idx.size:
        raise MatlabError("gscatter: x, y and the grouping variable must have the same number of elements.")
    colors = _color_list(args[3])
    markers = None if empty(args[4]) else (list(text(args[4])) if is_text(args[4]) else P._text_list(args[4]))
    sizes = None if empty(args[5]) else [float(v) for v in _vec(args[5])]
    doleg = True if empty(args[6]) else (text(args[6]).lower() == "on" if is_text(args[6]) else bool(_vec(args[6])[0]))
    ax, _ = _axes_from(interp, lead)
    from .plot_objects import default_colors
    palette = default_colors()
    lines = []
    for g, nm in enumerate(names):
        pick = idx == g + 1
        mk = P._mpl_marker(markers[g % len(markers)]) if markers else "."
        ms = sizes[g % len(sizes)] if sizes else (15.0 if mk == "." else 6.0)
        lines.extend(ax.plot(x[pick], y[pick], linestyle="None", marker=mk, markersize=ms,
                             color=_group_color(colors, g, palette), label=nm))
    if not empty(args[7]):
        ax.set_xlabel(P._label_text(args[7]))
    if not empty(args[8]):
        ax.set_ylabel(P._label_text(args[8]))
    if doleg:
        P._make_legend(interp, ax)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


# -- pareto --------------------------------------------------------------------------------------------------------

@builtin("pareto")
def _pareto(interp, args, nargout):
    """pareto(y, names, threshold): bars in decreasing order with a line of the cumulative share."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    if not args:
        raise MatlabError("pareto needs data, e.g. pareto([5 3 2]).")
    y = _vec(args[0])
    names = None
    threshold = 0.95
    for a in args[1:]:
        if isinstance(a, (Cell, StringArray)) or is_text(a):
            names = P._text_list(a)
        elif isinstance(a, np.ndarray) and a.size == 1 and len(args) == 3 and a is args[2]:
            threshold = float(_vec(a)[0])
        elif isinstance(a, np.ndarray) and a.size == y.size and a.size > 1:
            names = [P._fmt_num(v) for v in _vec(a)]
        elif isinstance(a, np.ndarray) and a.size == 1:
            threshold = float(_vec(a)[0])
        else:
            raise MatlabError("pareto expects pareto(y), pareto(y, names) or pareto(y, x), then a threshold.")
    if np.any(y < 0) or not np.all(np.isfinite(y)):
        raise MatlabError("pareto: the data must be non-negative numbers.")
    if not 0 < threshold <= 1:
        raise MatlabError("pareto: the threshold must be between 0 and 1.")
    if names is None:
        names = [str(k + 1) for k in range(y.size)]
    if len(names) != y.size:
        raise MatlabError("pareto: give one name for each value.")
    order = np.argsort(-y, kind="stable")
    ys = y[order]
    total = float(y.sum())
    cum = np.cumsum(ys)
    share = cum / total if total > 0 else cum
    if threshold >= 1:
        k = ys.size
    else:
        k = int(np.searchsorted(share, threshold - 1e-12) + 1)
        k = min(k, 10, ys.size)
    ax, _ = _axes_from(interp, lead)
    x = np.arange(1, k + 1)
    bars = ax.bar(x, ys[:k], width=0.8, edgecolor="black", linewidth=0.5)
    line = ax.plot(x, cum[:k], color=_next_color(ax), marker="o", markersize=3)
    _set_tick_text(ax, "x", x, [names[i] for i in order[:k]])
    top = float(cum[k - 1]) if k else 1.0
    ax.set_ylim(0, top if top > 0 else 1.0)
    ax.set_xlim(0.5, k + 0.5)
    right = ax.twinx()
    right._mylab_matrix_of = ax  # removed again when the bars are replaced by another plot
    right.set_ylim(0, 100 * top / total if total > 0 else 100)
    from matplotlib.ticker import FuncFormatter
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}%"))
    bh = P._wrap(interp, "bar", bars)
    bh.extra.update({"BarWidth": 0.8, "XData": x.astype(float), "YData": ys[:k], "axes": ax, "figure": ax.figure,
                     "Horizontal": False, "BarLayout": "grouped", "Bottom": None})
    lh = P._wrap(interp, "line", line[0])
    P._redraw(interp)
    if nargout <= 0:
        return None
    charts = P.GraphicsArray([bh, lh], row=True)
    axes = P.GraphicsArray([P._wrap(interp, "axes", ax), P._wrap(interp, "axes", right)], row=True)
    return (charts, axes) if nargout > 1 else charts


# -- voronoi ---------------------------------------------------------------------------------------------------------

def _voronoi_edges(x: np.ndarray, y: np.ndarray):
    """Line segments of the Voronoi diagram; unbounded edges are cut off a little outside the points."""
    from scipy.spatial import Voronoi
    pts = np.column_stack([x, y])
    if pts.shape[0] < 3:
        raise MatlabError("voronoi needs at least three points.")
    try:
        vor = Voronoi(pts)
    except Exception as e:  # noqa: BLE001  (Qhull reports degenerate input in several ways)
        raise MatlabError(f"voronoi: the diagram cannot be computed for these points ({str(e).splitlines()[0]}).") \
            from None
    center = pts.mean(axis=0)
    reach = 2.0 * float(np.ptp(pts, axis=0).max() or 1.0)
    segs = []
    for (p, q), (a, b) in zip(vor.ridge_points, vor.ridge_vertices):
        if a >= 0 and b >= 0:
            segs.append((vor.vertices[a], vor.vertices[b]))
            continue
        v = vor.vertices[a if a >= 0 else b]
        t = pts[q] - pts[p]
        t = t / np.linalg.norm(t)
        normal = np.array([-t[1], t[0]])
        mid = (pts[p] + pts[q]) / 2
        direction = np.sign(np.dot(mid - center, normal)) * normal
        if not np.any(direction):
            direction = normal
        segs.append((v, v + direction * reach))
    return segs


@builtin("voronoi")
def _voronoi(interp, args, nargout):
    """voronoi(x, y): the Voronoi diagram of points; [vx, vy] = voronoi(x, y) returns the edges."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    spec = ""
    nums = []
    for a in args:
        if is_text(a):
            if not P._is_linespec(text(a)):
                raise MatlabError(f"voronoi: unknown option '{text(a)}'.")
            spec = text(a)
        else:
            nums.append(a)
    if len(nums) < 2:
        raise MatlabError("voronoi needs x and y coordinates.")
    x, y = _vec(nums[0]), _vec(nums[1])
    if x.size != y.size:
        raise MatlabError("voronoi: x and y must have the same number of elements.")
    segs = _voronoi_edges(x, y)
    vx = np.array([[s[0][0] for s in segs], [s[1][0] for s in segs]])
    vy = np.array([[s[0][1] for s in segs], [s[1][1] for s in segs]])
    if nargout >= 2:
        return vx, vy
    ax, _ = _axes_from(interp, lead)
    kw = _spec_kw(spec)
    color = kw.pop("color", _next_color(ax))
    pts = ax.plot(x, y, linestyle="None", marker=".", color=color)
    xs, ys = [], []
    for k in range(vx.shape[1]):
        xs.extend([vx[0, k], vx[1, k], np.nan])
        ys.extend([vy[0, k], vy[1, k], np.nan])
    edges = ax.plot(xs, ys, color=color, **kw)
    pad = 0.1 * max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + pad)
    P._redraw(interp)
    return P._ret(interp, "line", pts + edges, nargout)


# -- word clouds -----------------------------------------------------------------------------------------------------

class WordCloud(Chart):
    kind = "wordcloud"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.words: list[str] = []
        self.sizes = np.zeros(0)
        self.color = (0.25, 0.25, 0.25)
        self.highlight = (0.85, 0.325, 0.098)
        self.max_words = 100
        self.shape = "oval"
        self.power = 0.5
        self.font = "sans-serif"

    def draw(self) -> None:
        ax = self.axes
        fig = ax.figure
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        order = np.argsort(-self.sizes, kind="stable")[:self.max_words]
        if order.size == 0:
            return
        top = float(self.sizes[order[0]])
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except (ValueError, RuntimeError, AttributeError):
            renderer = None
        box = ax.get_window_extent()
        placed: list = []
        golden = math.pi * (3 - math.sqrt(5))
        for rank, i in enumerate(order):
            rel = (float(self.sizes[i]) / top) ** self.power if top > 0 else 1.0
            fs = 7 + 29 * rel
            color = self.highlight if rank < min(5, order.size) else self.color
            t = ax.text(0.5, 0.5, self.words[i], fontsize=fs, ha="center", va="center", color=color,
                        fontfamily=self.font, transform=ax.transAxes)
            if renderer is None:
                self.add(t)
                continue
            bb = t.get_window_extent(renderer)
            w, hh = bb.width / box.width * 1.06, bb.height / box.height * 1.02
            done = False
            for step in range(0, 2500):
                r = 0.006 * math.sqrt(step)
                ang = step * golden
                cx = 0.5 + r * math.cos(ang) * (1.6 if self.shape == "oval" else 1.3)
                cy = 0.5 + r * math.sin(ang) * (0.9 if self.shape == "oval" else 1.3)
                x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - hh / 2, cy + hh / 2
                if x0 < 0 or x1 > 1 or y0 < 0 or y1 > 1:
                    continue
                if all(x1 <= a or x0 >= b or y1 <= c or y0 >= d for a, b, c, d in placed):
                    t.set_position((cx, cy))
                    placed.append((x0, x1, y0, y1))
                    done = True
                    break
            if done:
                self.add(t)
            else:
                t.remove()  # no room left for this word


def _wc_words_set(h, v):
    words = P._text_list(v)
    if len(words) != h.obj.sizes.size:
        raise MatlabError("WordData must have one word for each size.")
    h.obj.words = words
    h.obj.redraw()


def _wc_sizes_set(h, v):
    sizes = _vec(v)
    if sizes.size != len(h.obj.words) or np.any(sizes < 0):
        raise MatlabError("SizeData must have one non-negative number for each word.")
    h.obj.sizes = sizes
    h.obj.redraw()


_chart_setup("wordcloud", "matlab.graphics.chart.WordCloudChart", {
    "Title": _label_prop("Title"),
    "WordData": (lambda h: P._cell_column([char(s) for s in h.obj.words]), _wc_words_set),
    "SizeData": (lambda h: h.obj.sizes.reshape(-1, 1), _wc_sizes_set),
    "Color": _attr("color", _color_conv, out=_color_out),
    "HighlightColor": _attr("highlight", _color_conv, out=_color_out),
    "MaxDisplayWords": _attr("max_words", lambda v: int(_num_conv("MaxDisplayWords", 1, 1e6)(v))),
    "Shape": _attr("shape", _choice_conv("Shape", ("oval", "rectangle"))),
    "SizePower": _attr("power", _num_conv("SizePower", 0.01, 10)),
    "FontName": _attr("font", lambda v: text(v)),
}, ["WordData", "SizeData", "MaxDisplayWords", "Color", "HighlightColor", "Shape"])

_WC_OPTS = ("Title", "Color", "HighlightColor", "MaxDisplayWords", "Shape", "SizePower", "FontName", "Tag",
            "Visible")


@builtin("wordcloud")
def _wordcloud(interp, args, nargout):
    """wordcloud(words, sizes), wordcloud(tbl, wordVar, sizeVar) or wordcloud(categorical): a word cloud."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _WC_OPTS, "wordcloud")
    if not args:
        raise MatlabError("wordcloud needs words and sizes, e.g. wordcloud(words, counts).")
    if _is_table(args[0]):
        if len(args) != 3:
            raise MatlabError("wordcloud(tbl, wordVar, sizeVar) needs both variable names.")
        words = _labels_of(_column(args[0], args[1], "wordcloud")[1])
        sizes = _vec(_column(args[0], args[2], "wordcloud")[1])
    elif len(args) == 1 and hasattr(args[0], "categories") and hasattr(args[0], "codes"):
        c = args[0]
        counts = np.array([float(np.sum(c.codes == k)) for k in range(len(c.categories))])
        words = [w for w, n in zip(c.categories, counts) if n > 0]
        sizes = counts[counts > 0]
    elif len(args) == 1:
        items = P._text_list(args[0])
        tokens: dict = {}
        for s in items:
            for w in s.replace(",", " ").replace(".", " ").replace(";", " ").split():
                tokens[w] = tokens.get(w, 0) + 1
        words, sizes = list(tokens), np.array([float(v) for v in tokens.values()])
    elif len(args) == 2:
        words, sizes = _labels_of(args[0]), _vec(args[1])
    else:
        raise MatlabError("wordcloud expects wordcloud(words, sizes) or wordcloud(tbl, wordVar, sizeVar).")
    if len(words) != sizes.size:
        raise MatlabError("wordcloud: give one size for each word.")
    if np.any(sizes < 0) or not np.all(np.isfinite(sizes)):
        raise MatlabError("wordcloud: the sizes must be non-negative numbers.")
    ax, _ = _axes_from(interp, lead)
    chart = WordCloud(interp, ax)
    chart.words, chart.sizes = list(words), sizes
    for name, attr, conv in (("Color", "color", _color_conv), ("HighlightColor", "highlight", _color_conv),
                             ("MaxDisplayWords", "max_words", lambda v: int(_num_conv("MaxDisplayWords", 1, 1e6)(v))),
                             ("Shape", "shape", _choice_conv("Shape", ("oval", "rectangle"))),
                             ("SizePower", "power", _num_conv("SizePower", 0.01, 10)),
                             ("FontName", "font", lambda v: text(v))):
        if name in opts:
            setattr(chart, attr, conv(opts.pop(name)))
    chart.draw()
    h = _register(interp, chart)
    _apply_opts(h, opts)
    ax.title.set_visible(True)
    P._redraw(interp)
    return h if nargout > 0 else None


# -- pie and donut charts ----------------------------------------------------------------------------------------------

class PieChart(Chart):
    kind = "piechart"

    def __init__(self, interp, ax, inner: float = 0.0):
        super().__init__(interp, ax)
        self.data = np.zeros(0)
        self.names: list[str] = []
        self.inner = inner
        self.start = 0.0
        self.direction = "clockwise"
        self.exploded = np.zeros(0, dtype=bool)
        self.label_style = "namepercent"
        self.legend = "off"
        self.alpha = 0.6
        self.center_label = ""
        self.font_size = 10.0
        self.colors = None

    def proportions(self) -> np.ndarray:
        total = float(self.data.sum())
        return self.data / total if total > 0 else np.zeros_like(self.data)

    def labels(self) -> list[str]:
        prop = self.proportions()
        out = []
        for nm, v, p in zip(self.names, self.data, prop):
            pct = f"{100 * p:.1f}%"
            out.append({"namepercent": f"{nm} ({pct})", "namedata": f"{nm} ({P._fmt_num(v)})", "name": nm,
                        "percent": pct, "data": P._fmt_num(v), "none": ""}[self.label_style])
        return out

    def draw(self) -> None:
        from matplotlib.patches import Wedge

        from .plot_objects import default_colors
        ax = self.axes
        colors = self.colors if self.colors is not None else default_colors()
        prop = self.proportions()
        labels = self.labels()
        sign = -1.0 if self.direction == "clockwise" else 1.0
        angle = 90.0 - self.start if self.direction == "clockwise" else 90.0 + self.start
        for k, p in enumerate(prop):
            span = 360.0 * float(p)
            a0, a1 = (angle - span, angle) if sign < 0 else (angle, angle + span)
            mid = math.radians((a0 + a1) / 2)
            off = 0.1 if k < self.exploded.size and self.exploded[k] else 0.0
            c = (off * math.cos(mid), off * math.sin(mid))
            w = Wedge(c, 1.0, a0, a1, width=1.0 - self.inner if self.inner > 0 else None,
                      facecolor=(*tuple(colors[k % len(colors)]), self.alpha), edgecolor="white", linewidth=1.0,
                      label=self.names[k] if self.names else None)
            ax.add_patch(w)
            self.add(w)
            if labels[k]:
                r = 1.12 + off
                self.add(ax.text(r * math.cos(mid), r * math.sin(mid), labels[k], fontsize=self.font_size,
                                 ha="left" if math.cos(mid) >= 0 else "right", va="center"))
            angle += sign * span
        if self.center_label and self.inner > 0:
            self.add(ax.text(0, 0, self.center_label, ha="center", va="center", fontsize=self.font_size + 2))
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.title.set_visible(True)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if self.legend == "on":
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=self.font_size)


def _pc_data_set(h, v):
    d = _vec(v)
    if np.any(d < 0) or not np.all(np.isfinite(d)):
        raise MatlabError("Data must be non-negative numbers.")
    c = h.obj
    if d.size != c.data.size:
        c.names = [f"data{k + 1}" for k in range(d.size)]
        c.exploded = np.zeros(d.size, dtype=bool)
    c.data = d
    c.redraw()


def _pc_names_set(h, v):
    names = P._text_list(v)
    if len(names) != h.obj.data.size:
        raise MatlabError("Names must have one name for each slice.")
    h.obj.names = names
    h.obj.redraw()


def _pc_explode_set(h, v):
    c = h.obj
    flags = np.zeros(c.data.size, dtype=bool)
    if isinstance(v, (Cell, StringArray)) or is_text(v):
        for nm in P._text_list(v):
            if nm not in c.names:
                raise MatlabError(f"'{nm}' is not the name of a slice.")
            flags[c.names.index(nm)] = True
    else:
        a = np.asarray(v)
        if a.dtype == bool and a.size == c.data.size:
            flags = a.flatten().astype(bool)
        else:
            for i in _vec(v):
                if not 1 <= int(i) <= c.data.size:
                    raise MatlabError("ExplodedWedges: the slice number is out of range.")
                flags[int(i) - 1] = True
    c.exploded = flags
    c.redraw()


_PIE_PROPS = {
    "Title": _label_prop("Title"),
    "Data": (lambda h: P._row(h.obj.data), _pc_data_set),
    "Names": (lambda h: Cell.from_list([char(s) for s in h.obj.names]), _pc_names_set),
    "Proportions": _ro(lambda c: P._row(c.proportions())),
    "Labels": _ro(lambda c: Cell.from_list([char(s) for s in c.labels()])),
    "LabelStyle": _attr("label_style", _choice_conv("LabelStyle", ("namepercent", "namedata", "name", "percent",
                                                                   "data", "none"))),
    "StartAngle": _attr("start", _num_conv("StartAngle", -1e6, 1e6)),
    "Direction": _attr("direction", _choice_conv("Direction", ("clockwise", "counterclockwise"))),
    "ExplodedWedges": (lambda h: P._row(np.flatnonzero(h.obj.exploded) + 1), _pc_explode_set),
    "FaceAlpha": _attr("alpha", _num_conv("FaceAlpha", 0, 1)),
    "LegendVisible": _attr("legend", _choice_conv("LegendVisible", ("on", "off"))),
    "FontSize": _attr("font_size", _num_conv("FontSize", 1, 1000)),
    "ColorOrder": _attr("colors", _colors_conv,
                        out=lambda c: c if c is not None else _default_colors()),
}
_chart_setup("piechart", "matlab.graphics.chart.PieChart", dict(_PIE_PROPS),
             ["Data", "Names", "Proportions", "LabelStyle", "StartAngle", "Direction"])
_chart_setup("donutchart", "matlab.graphics.chart.DonutChart", {
    **_PIE_PROPS,
    "InnerRadius": _attr("inner", _num_conv("InnerRadius", 0, 0.99)),
    "CenterLabel": _attr("center_label", lambda v: P._label_text(v)),
}, ["Data", "Names", "Proportions", "InnerRadius", "CenterLabel", "LabelStyle"])

_PIE_OPTS = ("Title", "Names", "LabelStyle", "StartAngle", "Direction", "ExplodedWedges", "FaceAlpha",
             "LegendVisible", "FontSize", "ColorOrder", "InnerRadius", "CenterLabel", "Tag", "Visible")


def _pie_chart_core(interp, args, nargout, donut: bool):
    name = "donutchart" if donut else "piechart"
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, _PIE_OPTS, name)
    if not donut and ("InnerRadius" in opts or "CenterLabel" in opts):
        raise MatlabError("piechart has no InnerRadius or CenterLabel; use donutchart.")
    if not args:
        raise MatlabError(f"{name} needs data, e.g. {name}([3 2 1], {{'a', 'b', 'c'}}).")
    names = None
    if _is_table(args[0]):
        tbl = args[0]
        if len(args) == 2:
            nm, col = _column(tbl, args[1], name)
            if isinstance(col, np.ndarray) and col.dtype.kind in "biuf":
                data = _vec(col)
                names = tbl.row_names if tbl.row_names else None
            else:
                labels = _labels_of(col)
                names = _unique_labels(col)
                data = np.array([float(labels.count(n)) for n in names])
        elif len(args) == 3:
            names = _labels_of(_column(tbl, args[1], name)[1])
            data = _vec(_column(tbl, args[2], name)[1])
        else:
            raise MatlabError(f"{name}(tbl, datavar) or {name}(tbl, namesvar, datavar).")
    elif hasattr(args[0], "categories") and hasattr(args[0], "codes"):
        c = args[0]
        counts = np.array([float(np.sum(c.codes == k)) for k in range(len(c.categories))])
        names = [n for n, v in zip(c.categories, counts) if v > 0]
        data = counts[counts > 0]
    else:
        data = _vec(args[0])
        if len(args) > 1:
            names = _labels_of(args[1])
        if len(args) > 2:
            raise MatlabError(f"{name}: too many inputs.")
    if np.any(data < 0) or not np.all(np.isfinite(data)):
        raise MatlabError(f"{name}: the data must be non-negative numbers.")
    if names is None:
        names = [f"data{k + 1}" for k in range(data.size)]
    if len(names) != data.size:
        raise MatlabError(f"{name}: give one name for each value.")
    ax, _ = _axes_from(interp, lead)
    chart = PieChart(interp, ax, inner=0.6 if donut else 0.0)
    chart.kind = name
    chart.data, chart.names = data, list(names)
    chart.exploded = np.zeros(data.size, dtype=bool)
    chart.draw()
    h = _register(interp, chart)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


builtin("piechart")(lambda interp, args, nargout: _pie_chart_core(interp, args, nargout, False))
builtin("donutchart")(lambda interp, args, nargout: _pie_chart_core(interp, args, nargout, True))


# -- geographic axes ---------------------------------------------------------------------------------------------------

def _mercator(lat):
    lat = np.clip(np.asarray(lat, dtype=float), -85.0, 85.0)
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def _inv_mercator(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(np.asarray(y, dtype=float)))) - np.pi / 2)


def _geo_label(v: float, pos: str, neg: str) -> str:
    a = abs(v)
    d = int(math.floor(a + 1e-9))
    m = (a - d) * 60
    if abs(m) < 1e-6 or abs(m - 60) < 1e-6:
        body = f"{int(round(a))}°"
    elif abs(m - round(m)) < 1e-6:
        body = f"{d}°{int(round(m)):02d}'"
    else:
        body = f"{a:.4g}°"
    return body + ("" if v == 0 else pos if v > 0 else neg)


def _is_geo(ax) -> bool:
    return bool(getattr(ax, "_mylab_geo", False))


def make_geoaxes(interp, ax) -> None:
    """Turn a plain axes into geographic axes: longitude across, latitude up on a Mercator scale."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax._mylab_geo = True
    ax._mylab_basemap = getattr(ax, "_mylab_basemap", "streets-light")
    known = P._state(interp).handles.get(id(ax))
    if known is not None:
        known.kind = "geoaxes"
    ax.set_yscale("function", functions=(_mercator, _inv_mercator))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 1.5, 2, 3, 5, 6, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 1.5, 2, 3, 5, 6, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: _geo_label(v, "E", "W")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: _geo_label(v, "N", "S")))
    ax._mylab_tickfmt = {"x": "geo", "y": "geo"}  # tick labels come from the formatters above
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="0.8", linewidth=0.6)
    ax.set_facecolor((0.94, 0.97, 1.0))
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    try:
        ax.set_aspect("equal", adjustable="box")
    except (NotImplementedError, ValueError):
        pass
    if not ax.has_data():
        ax.set_xlim(-180, 180)
        ax.set_ylim(-70, 80)


def _geo_axes(interp, lead: list):
    """The geographic axes to draw in (made from the current axes when that is empty)."""
    st = P._state(interp)
    target = lead[0].obj if lead and lead[0].kind in ("axes", "geoaxes") else None
    prev = st.override_ax
    if target is not None:
        st.override_ax = target
    try:
        cur = P._axes(interp, fresh=False, projection="rectilinear")
        hold = st.hold.get(id(cur), False)
        if not _is_geo(cur) or not hold:
            cur = P._axes(interp, fresh=True, projection="rectilinear")
            make_geoaxes(interp, cur)
    finally:
        st.override_ax = prev
    return cur


def _geo_fit(ax, lat: np.ndarray, lon: np.ndarray) -> None:
    """Limits that show all the data on the axes with a margin."""
    lats = list(getattr(ax, "_mylab_geo_lat", [])) + [float(v) for v in lat[np.isfinite(lat)]]
    lons = list(getattr(ax, "_mylab_geo_lon", [])) + [float(v) for v in lon[np.isfinite(lon)]]
    ax._mylab_geo_lat, ax._mylab_geo_lon = lats, lons
    if getattr(ax, "_mylab_geo_manual", False) or not lats:
        return
    la0, la1, lo0, lo1 = min(lats), max(lats), min(lons), max(lons)
    dla = max((la1 - la0) * 0.15, 0.05 if la1 > la0 else 0.5)
    dlo = max((lo1 - lo0) * 0.15, 0.05 if lo1 > lo0 else 0.5)
    ax.set_xlim(max(lo0 - dlo, -180.0), min(lo1 + dlo, 180.0))
    ax.set_ylim(max(la0 - dla, -85.0), min(la1 + dla, 85.0))


def _check_latlon(lat, lon, name: str):
    lat, lon = _vec(lat), _vec(lon)
    if lat.size != lon.size:
        raise MatlabError(f"{name}: latitude and longitude must have the same number of elements.")
    f = lat[np.isfinite(lat)]
    if f.size and (f.min() < -90 or f.max() > 90):
        raise MatlabError(f"{name}: latitudes must be between -90 and 90 degrees.")
    return lat, lon


@builtin("geoaxes")
def _geoaxes(interp, args, nargout):
    """geoaxes: axes for latitude and longitude data (a Mercator grid; no map imagery is drawn)."""
    args = list(args)
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind == "geoaxes" and len(args) == 1:
        return BUILTINS["axes"](interp, [args[0]], nargout)
    if args and isinstance(args[0], P.GraphicsHandle) and args[0].kind == "figure":
        P._figure(interp, P._fig_number(interp, args.pop(0).obj))
    pairs = P._name_values(args, "geoaxes")
    ax = P._axes(interp, fresh=True, projection="rectilinear")
    make_geoaxes(interp, ax)
    h = P._wrap(interp, "geoaxes", ax)
    for name, value in pairs:
        P._set_prop(h, name, value)
    P._redraw(interp)
    return h if nargout > 0 else None


@builtin("geoplot")
def _geoplot(interp, args, nargout):
    """geoplot(lat, lon, linespec): a line through points given by latitude and longitude."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = P._split_name_values(args, P._LINE_OPTS)
    nums = [a for a in args if not is_text(a)]
    specs = [text(a) for a in args if is_text(a)]
    for sp in specs:
        if not P._is_linespec(sp):
            raise MatlabError(f"geoplot: unknown option '{sp}'.")
    if len(nums) < 2 or len(nums) % 2:
        raise MatlabError("geoplot needs latitude and longitude, e.g. geoplot(lat, lon).")
    ax = _geo_axes(interp, lead)
    kw = P._mpl_kwargs(opts)
    lines = []
    k = 0
    arg_iter = iter(args)
    pending: list = []
    groups = []
    for a in arg_iter:
        if is_text(a):
            if pending:
                groups[-1] = (*groups[-1][:2], text(a))
            continue
        pending.append(a)
        if len(pending) == 2:
            groups.append((pending[0], pending[1], ""))
            pending = []
    for la, lo, sp in groups:
        lat, lon = _check_latlon(la, lo, "geoplot")
        sp, kws = P.split_spec(sp, kw)
        lines.extend(ax.plot(lon, lat, sp, **kws) if sp else ax.plot(lon, lat, **kws))
        _geo_fit(ax, lat, lon)
        k += 1
    P._tag_lines(interp, lines, opts)
    P._redraw(interp)
    return P._ret(interp, "line", lines, nargout)


@builtin("geoscatter")
def _geoscatter(interp, args, nargout):
    """geoscatter(lat, lon, sz, c): markers at points given by latitude and longitude."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    coords, size, color, filled, marker, rest = _marker_inputs(args, "geoscatter", 2)
    rest, opts = _options(rest, ("MarkerFaceColor", "MarkerEdgeColor", "MarkerFaceAlpha", "LineWidth",
                                 "DisplayName", "Tag", "Visible"), "geoscatter")
    _no_text_left(rest, "geoscatter")
    lat, lon = _check_latlon(coords[0], coords[1], "geoscatter")
    ax = _geo_axes(interp, lead)
    s = 36.0
    if size is not None and isinstance(size, np.ndarray) and size.size and not is_char(size):
        sv = _vec(size)
        s = float(sv[0]) if sv.size == 1 else sv
    kw = _scatter_colors(interp, ax, color, lat.size, filled)
    if "MarkerFaceColor" in opts:
        kw["facecolors"] = P._color(opts["MarkerFaceColor"])
    if "MarkerEdgeColor" in opts:
        kw["edgecolors"] = P._color(opts["MarkerEdgeColor"])
    if "MarkerFaceAlpha" in opts:
        kw["alpha"] = P._scalar(opts["MarkerFaceAlpha"])
    if "LineWidth" in opts:
        kw["linewidths"] = P._scalar(opts["LineWidth"])
    if "DisplayName" in opts:
        kw["label"] = text(opts["DisplayName"])
    coll = ax.scatter(lon, lat, s=s, marker=marker, **kw)
    _geo_fit(ax, lat, lon)
    h = P._wrap(interp, "scatter", coll)
    h.extra["Marker"] = marker
    _apply_opts(h, {k: v for k, v in opts.items() if k in ("Tag", "Visible")})
    P._redraw(interp)
    return h if nargout > 0 else None


class GeoBubble(Chart):
    kind = "geobubble"

    def __init__(self, interp, ax):
        super().__init__(interp, ax)
        self.lat = np.zeros(0)
        self.lon = np.zeros(0)
        self.sizes = None
        self.cat_idx = None
        self.cat_names: list[str] = []
        self.width = (5.0, 20.0)
        self.colors = None
        self.size_title = ""
        self.color_title = ""
        self.table = None

    def draw(self) -> None:
        from .plot_objects import default_colors
        ax = self.axes
        colors = self.colors if self.colors is not None else default_colors()
        n = self.lat.size
        if self.sizes is not None and n:
            f = self.sizes[np.isfinite(self.sizes)]
            lo, hi = (float(f.min()), float(f.max())) if f.size else (0.0, 1.0)
            t = (self.sizes - lo) / (hi - lo) if hi > lo else np.ones(n)
            d = self.width[0] + np.clip(t, 0, 1) * (self.width[1] - self.width[0])
        else:
            d = np.full(n, self.width[0] if n else 0.0)
        areas = d ** 2
        groups = range(max(len(self.cat_names), 1))
        for g in groups:
            pick = np.ones(n, dtype=bool) if self.cat_idx is None else self.cat_idx == g + 1
            c = tuple(colors[g % len(colors)])
            coll = ax.scatter(self.lon[pick], self.lat[pick], s=areas[pick], facecolors=(*c, 0.6), edgecolors=c,
                              label=self.cat_names[g] if self.cat_names else None)
            self.add(coll)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if self.cat_names:
            ax.legend(title=self.color_title or None, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def _gb_width_set(h, v):
    w = _vec(v)
    if w.size != 2 or not 0 < w[0] <= w[1]:
        raise MatlabError("BubbleWidthRange must be [smallest largest] in points.")
    h.obj.width = (float(w[0]), float(w[1]))
    h.obj.redraw()


def _geo_limits_prop(which: str) -> tuple:
    def g(h):
        ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
        return P._row(sorted(ax.get_ylim() if which == "lat" else ax.get_xlim()))

    def s(h, v):
        ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
        lim = _vec(v)
        if lim.size != 2 or not lim[0] < lim[1]:
            raise MatlabError("The limits must be [min max] with min < max.")
        if which == "lat":
            if lim[0] < -90 or lim[1] > 90:
                raise MatlabError("Latitude limits must be between -90 and 90 degrees.")
            ax.set_ylim(max(lim[0], -85.0), min(lim[1], 85.0))
        else:
            ax.set_xlim(lim[0], lim[1])
        ax._mylab_geo_manual = True
    return g, s


_BASEMAPS = ("streets-light", "streets-dark", "streets", "satellite", "topographic", "landcover", "colorterrain",
             "grayterrain", "bluegreen", "grayland", "darkwater", "none")


def _basemap_set(h, v):
    name = text(v).lower()
    if name not in _BASEMAPS:
        raise MatlabError(f"'{text(v)}' is not a basemap name. Use one of: {', '.join(_BASEMAPS)}.")
    ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
    ax._mylab_basemap = name
    # No map imagery is available; the basemap chooses the background of the grid.
    dark = name in ("streets-dark", "satellite", "darkwater")
    ax.set_facecolor((0.16, 0.18, 0.22) if dark else (1, 1, 1) if name == "none" else (0.94, 0.97, 1.0))
    ax.grid(True, color="0.35" if dark else "0.8", linewidth=0.6)


def _basemap_get(h):
    ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
    return char(getattr(ax, "_mylab_basemap", "streets-light"))


_GEO_COMMON = {
    "LatitudeLimits": _geo_limits_prop("lat"), "LongitudeLimits": _geo_limits_prop("lon"),
    "Basemap": (_basemap_get, _basemap_set),
    "MapCenter": (lambda h: P._row([float(np.mean((h.obj.axes if isinstance(h.obj, Chart) else h.obj).get_ylim())),
                                    float(np.mean((h.obj.axes if isinstance(h.obj, Chart) else h.obj).get_xlim()))]),
                  None),
}
_chart_setup("geobubble", "matlab.graphics.chart.GeographicBubbleChart", {
    "Title": _label_prop("Title"),
    "LatitudeData": _ro(lambda c: c.lat.reshape(-1, 1)), "LongitudeData": _ro(lambda c: c.lon.reshape(-1, 1)),
    "SizeData": _ro(lambda c: c.sizes.reshape(-1, 1) if c.sizes is not None else _EMPTY),
    "ColorData": _ro(lambda c: P._cell_column([char(c.cat_names[int(i) - 1]) if np.isfinite(i) else char("")
                                                for i in c.cat_idx]) if c.cat_idx is not None else _EMPTY),
    "BubbleWidthRange": (lambda h: P._row(h.obj.width), _gb_width_set),
    "BubbleColorList": _attr("colors", _colors_conv,
                             out=lambda c: c if c is not None else _EMPTY),
    "SizeLegendTitle": _attr("size_title", lambda v: P._label_text(v)),
    "ColorLegendTitle": _attr("color_title", lambda v: P._label_text(v)),
    "SourceTable": _ro(lambda c: c.table if c.table is not None else _EMPTY),
    **_GEO_COMMON,
}, ["LatitudeData", "LongitudeData", "SizeData", "ColorData", "Basemap", "Title"])

P._CLASS["geoaxes"] = "matlab.graphics.axis.GeographicAxes"
P._TYPE["geoaxes"] = "geoaxes"
P._PROPS["geoaxes"] = {
    **{k: v for k, v in P._AXES_PROPS.items() if k not in ("XScale", "YScale", "ZScale", "View")},
    **_GEO_COMMON,
    "LatitudeLabel": (lambda h: P._title_text(h, "YLabel"), P._title_set("YLabel")),
    "LongitudeLabel": (lambda h: P._title_text(h, "XLabel"), P._title_set("XLabel")),
    "Grid": (P._grid_get("xaxis"), lambda h, v: h.obj.grid(P._onoff(v))),
    "ZoomLevel": (lambda h: P._num_val(max(math.log2(360.0 / max(abs(np.diff(h.obj.get_xlim())[0]), 1e-9)), 0.0)),
                  None),
}
P._DISPLAY["geoaxes"] = ["Basemap", "LatitudeLimits", "LongitudeLimits", "MapCenter", "ZoomLevel", "Position"]
P._IGNORED["geoaxes"] = P._AXES_IGNORED


@builtin("geobubble")
def _geobubble(interp, args, nargout):
    """geobubble(lat, lon, sizedata, colordata) or geobubble(tbl, latvar, lonvar): bubbles on geographic axes."""
    args = list(args)
    lead = [args.pop(0)] if args and isinstance(args[0], P.GraphicsHandle) else []
    args, opts = _options(args, ("SizeVariable", "ColorVariable", "Basemap", "Title", "BubbleWidthRange",
                                 "BubbleColorList", "SizeLegendTitle", "ColorLegendTitle", "LatitudeLimits",
                                 "LongitudeLimits", "MapLayout", "Tag", "Visible"), "geobubble")
    ax = _geo_axes(interp, lead)
    chart = GeoBubble(interp, ax)
    cats = None
    if args and _is_table(args[0]):
        tbl = args[0]
        if len(args) != 3:
            raise MatlabError("geobubble(tbl, latvar, lonvar) needs both variable names.")
        chart.table = tbl
        chart.lat, chart.lon = _check_latlon(_column(tbl, args[1], "geobubble")[1],
                                             _column(tbl, args[2], "geobubble")[1], "geobubble")
        if "SizeVariable" in opts:
            nm, col = _column(tbl, opts.pop("SizeVariable"), "geobubble")
            chart.sizes, chart.size_title = _vec(col), nm
        if "ColorVariable" in opts:
            nm, cats = _column(tbl, opts.pop("ColorVariable"), "geobubble")
            chart.color_title = nm
    else:
        if len(args) < 2 or len(args) > 4:
            raise MatlabError("geobubble expects geobubble(lat, lon), with sizes and colour categories optional.")
        chart.lat, chart.lon = _check_latlon(args[0], args[1], "geobubble")
        if len(args) > 2 and isinstance(args[2], np.ndarray) and args[2].size:
            chart.sizes = _vec(args[2])
        if len(args) > 3:
            cats = args[3]
        if "SizeVariable" in opts or "ColorVariable" in opts:
            raise MatlabError("geobubble: SizeVariable and ColorVariable need table input.")
    if chart.sizes is not None and chart.sizes.size != chart.lat.size:
        raise MatlabError("geobubble: give one size for each point.")
    if cats is not None:
        from .ndmath import group_index
        idx, names, _ = group_index(cats)
        if idx.size != chart.lat.size:
            raise MatlabError("geobubble: give one colour category for each point.")
        chart.cat_idx, chart.cat_names = idx, names
    opts.pop("MapLayout", None)
    chart.draw()
    _geo_fit(ax, chart.lat, chart.lon)
    h = _register(interp, chart)
    _apply_opts(h, opts)
    P._redraw(interp)
    return h if nargout > 0 else None


@builtin("geolimits")
def _geolimits(interp, args, nargout):
    """geolimits(latlim, lonlim) sets, and [latlim, lonlim] = geolimits returns, the limits of a map."""
    args = list(args)
    ax = None
    if args and isinstance(args[0], P.GraphicsHandle):
        h = args.pop(0)
        ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
    if ax is None:
        ax = P._axes(interp, fresh=False)
    if not _is_geo(ax):
        raise MatlabError("geolimits works on geographic axes; draw with geoplot, geoscatter or geobubble first.")
    if args:
        if len(args) == 1 and is_text(args[0]):
            mode = text(args[0]).lower()
            if mode == "auto":
                ax._mylab_geo_manual = False
                _geo_fit(ax, np.zeros(0), np.zeros(0))
            elif mode == "manual":
                ax._mylab_geo_manual = True
            else:
                raise MatlabError("geolimits expects latitude and longitude limits, 'auto' or 'manual'.")
        elif len(args) == 2:
            la, lo = _vec(args[0]), _vec(args[1])
            if la.size != 2 or lo.size != 2 or not la[0] < la[1] or not lo[0] < lo[1]:
                raise MatlabError("geolimits expects [latmin latmax] and [lonmin lonmax].")
            if la[0] < -90 or la[1] > 90:
                raise MatlabError("Latitude limits must be between -90 and 90 degrees.")
            ax.set_ylim(max(la[0], -85.0), min(la[1], 85.0))
            ax.set_xlim(lo[0], lo[1])
            ax._mylab_geo_manual = True
        else:
            raise MatlabError("geolimits expects latitude and longitude limits.")
        P._redraw(interp)
    lat, lon = P._row(sorted(ax.get_ylim())), P._row(sorted(ax.get_xlim()))
    if nargout >= 2:
        return lat, lon
    return lat if nargout == 1 else None


@builtin("geobasemap")
def _geobasemap(interp, args, nargout):
    """geobasemap(name) chooses the basemap of geographic axes (it sets the background; no imagery)."""
    args = list(args)
    ax = None
    if args and isinstance(args[0], P.GraphicsHandle):
        h = args.pop(0)
        ax = h.obj.axes if isinstance(h.obj, Chart) else h.obj
    if ax is None:
        ax = P._axes(interp, fresh=False)
        if not _is_geo(ax):
            if ax.has_data():
                raise MatlabError("geobasemap works on geographic axes.")
            make_geoaxes(interp, ax)
    if not args:
        return char(getattr(ax, "_mylab_basemap", "streets-light"))
    _basemap_set(P._wrap(interp, "geoaxes", ax), args[0])
    P._redraw(interp)
    return None
