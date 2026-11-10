"""Plotting, implemented on top of matplotlib.

Figures are numbered like MATLAB (``figure(2)``), ``hold on`` keeps the
current plot, and ``saveas``/``print`` write image files. When no display is
available (e.g. on a server) plots are rendered off-screen and can be saved.
"""

from __future__ import annotations

import os

import numpy as np

from ..errors import MatlabError
from ..values import Cell, is_char, to_int, to_str
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
        if st.current is not None and st.current in st.figures:
            return st.figures[st.current]
        n = 1
        while n in st.figures:
            n += 1
    if n not in st.figures:
        fig = plt.figure(n)
        st.figures[n] = fig
    st.current = n
    plt.figure(st.figures[n].number)
    return st.figures[n]


def _axes(interp, fresh: bool = True):
    """Current axes; cleared first unless ``hold on`` (when ``fresh``)."""
    st = _state(interp)
    fig = _figure(interp)
    ax = st.axes.get(st.current)
    if ax is None or ax.figure is not fig or ax not in fig.axes:
        ax = fig.add_subplot(1, 1, 1) if not fig.axes else fig.axes[-1]
        st.axes[st.current] = ax
    if fresh and not st.hold.get(id(ax), False):
        ax.cla()
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
    st.current = None


def _split_name_values(args, known: set[str]):
    """Separate trailing Name,Value pairs (e.g. 'LineWidth', 2)."""
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
              "markerfacecolor", "markeredgecolor"}


def _mpl_kwargs(opts) -> dict:
    kw = {}
    for k, v in opts.items():
        if k == "linewidth":
            kw["linewidth"] = float(fnum(v).flat[0])
        elif k == "markersize":
            kw["markersize"] = float(fnum(v).flat[0])
        elif k == "color":
            kw["color"] = _color(v)
        elif k == "displayname":
            kw["label"] = text(v)
        elif k == "marker":
            kw["marker"] = text(v)
        elif k == "linestyle":
            kw["linestyle"] = text(v)
        elif k == "markerfacecolor":
            kw["markerfacecolor"] = _color(v)
        elif k == "markeredgecolor":
            kw["markeredgecolor"] = _color(v)
    return kw


def _color(v):
    if is_text(v):
        s = text(v)
        names = {"red": "r", "green": "g", "blue": "b", "cyan": "c", "magenta": "m",
                 "yellow": "y", "black": "k", "white": "w", "none": "none"}
        return names.get(s.lower(), s)
    return tuple(float(x) for x in fnum(v).flatten())


def _is_linespec(s: str) -> bool:
    import re
    style = r"(--|-\.|-|:)?"
    pattern = style + r"[rgbcmykw]?" + style + r"[o+*.xsd^v<>ph]?" + style + r"[rgbcmykw]?"
    return s != "" and bool(re.fullmatch(pattern, s))


def _xy_groups(args):
    """Parse plot(x1,y1,spec1,x2,y2,...) into (x, y, spec) triples."""
    groups = []
    k = 0
    while k < len(args):
        a = args[k]
        if is_text(a):
            raise MatlabError(f"Invalid plot argument '{text(a)}'.")
        if k + 1 < len(args) and not is_text(args[k + 1]):
            x, y = fnum(args[k]), fnum(args[k + 1])
            k += 2
        else:
            y = fnum(a)
            if np.iscomplexobj(y):
                x, y = np.real(y), np.imag(y)
            else:
                x = np.arange(1, (y.shape[0] if y.shape[0] > 1 and y.shape[1] > 1 else y.size) + 1, dtype=float)
            k += 1
        spec = ""
        if k < len(args) and is_text(args[k]) and _is_linespec(text(args[k])):
            spec = text(args[k])
            k += 1
        groups.append((x, y, spec))
    return groups


def _prep_xy(x: np.ndarray, y: np.ndarray):
    x, y = np.real(x), np.real(y)
    if x.size == y.size and (x.shape[0] == 1 or x.shape[1] == 1) and (y.shape[0] == 1 or y.shape[1] == 1):
        return x.flatten(), y.flatten()
    if x.shape[0] == 1 or x.shape[1] == 1:
        xv = x.flatten()
        if y.shape[0] == xv.size:
            return xv, y
        if y.shape[1] == xv.size:
            return xv, y.T
    return x, y


def _do_plot(interp, args, method: str):
    args, opts = _split_name_values(args, _LINE_OPTS)
    ax = _axes(interp)
    kw = _mpl_kwargs(opts)
    lines = []
    for x, y, spec in _xy_groups(args):
        x, y = _prep_xy(x, y)
        fn = getattr(ax, method)
        lines.extend(fn(x, y, spec, **kw) if spec else fn(x, y, **kw))
    _redraw(interp)
    return None


def _redraw(interp):
    plt = _plt()
    if plt.isinteractive():
        plt.pause(0.001)


@builtin("plot")
def _plot(interp, args, nargout):
    return _do_plot(interp, args, "plot")


@builtin("semilogx")
def _semilogx(interp, args, nargout):
    return _do_plot(interp, args, "semilogx")


@builtin("semilogy")
def _semilogy(interp, args, nargout):
    return _do_plot(interp, args, "semilogy")


@builtin("loglog")
def _loglog(interp, args, nargout):
    return _do_plot(interp, args, "loglog")


@builtin("plot3")
def _plot3(interp, args, nargout):
    args, opts = _split_name_values(args, _LINE_OPTS)
    fig = _figure(interp)
    st = _state(interp)
    ax = st.axes.get(st.current)
    if ax is None or getattr(ax, "name", "") != "3d" or not st.hold.get(id(ax), False):
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        st.axes[st.current] = ax
    x, y, z = (np.real(fnum(a)).flatten() for a in args[:3])
    spec = text(args[3]) if len(args) > 3 and is_text(args[3]) else ""
    ax.plot(x, y, z, spec, **_mpl_kwargs(opts)) if spec else ax.plot(x, y, z, **_mpl_kwargs(opts))
    _redraw(interp)
    return None


@builtin("scatter")
def _scatter(interp, args, nargout):
    args, opts = _split_name_values(args, {"markerfacecolor", "markeredgecolor", "displayname", "linewidth"})
    filled = any(is_text(a) and text(a) == "filled" for a in args)
    args = [a for a in args if not (is_text(a) and text(a) == "filled")]
    ax = _axes(interp)
    x, y = np.real(fnum(args[0])).flatten(), np.real(fnum(args[1])).flatten()
    s = float(fnum(args[2]).flat[0]) if len(args) > 2 and isinstance(args[2], np.ndarray) and args[2].size == 1 \
        else (fnum(args[2]).flatten() if len(args) > 2 and isinstance(args[2], np.ndarray) and args[2].size else 36)
    kw = {}
    if len(args) > 3:
        c = args[3]
        kw["c"] = _color(c) if is_text(c) else (fnum(c).flatten() if fnum(c).size == x.size else _color(c))
    marker = next((text(a) for a in args[4:] if is_text(a)), "o")
    if not filled:
        kw.setdefault("facecolors", "none")
        if "c" in kw and not isinstance(kw["c"], np.ndarray):
            kw["edgecolors"] = kw.pop("c")
        elif "c" not in kw:
            kw["edgecolors"] = "C0"
    if "displayname" in opts:
        kw["label"] = text(opts["displayname"])
    ax.scatter(x, y, s=s, marker=marker, **kw)
    _redraw(interp)
    return None


@builtin("bar")
def _bar(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    if len(nums) == 1:
        y = fnum(nums[0])
        x = np.arange(1, (y.size if y.shape[0] == 1 or y.shape[1] == 1 else y.shape[0]) + 1)
    else:
        x, y = fnum(nums[0]).flatten(), fnum(nums[1])
    if y.shape[0] == 1 or y.shape[1] == 1:
        ax.bar(x, y.flatten(), width=0.8)
    else:
        n = y.shape[1]
        w = 0.8 / n
        for k in range(n):
            ax.bar(x - 0.4 + w * (k + 0.5), y[:, k], width=w)
    _redraw(interp)
    return None


@builtin("barh")
def _barh(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    y = fnum(nums[-1]).flatten()
    x = fnum(nums[0]).flatten() if len(nums) > 1 else np.arange(1, y.size + 1)
    ax.barh(x, y, height=0.8)
    _redraw(interp)
    return None


@builtin("histogram", "hist")
def _histogram(interp, args, nargout):
    x = np.real(fnum(args[0])).flatten()
    x = x[np.isfinite(x)]
    bins = "auto"
    if len(args) > 1 and not is_text(args[1]):
        b = fnum(args[1]).flatten()
        bins = int(b[0]) if b.size == 1 else b
    if nargout > 0:
        counts, edges = np.histogram(x, bins=10 if bins == "auto" else bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return (counts.astype(float).reshape(1, -1), centers.reshape(1, -1)) if nargout > 1 else \
            counts.astype(float).reshape(1, -1)
    ax = _axes(interp)
    ax.hist(x, bins=bins, edgecolor="black", linewidth=0.5)
    _redraw(interp)
    return None


@builtin("stem")
def _stem(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    y = np.real(fnum(nums[-1])).flatten()
    x = np.real(fnum(nums[0])).flatten() if len(nums) > 1 else np.arange(1, y.size + 1)
    ax.stem(x, y)
    _redraw(interp)
    return None


@builtin("stairs")
def _stairs(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    y = np.real(fnum(nums[-1])).flatten()
    x = np.real(fnum(nums[0])).flatten() if len(nums) > 1 else np.arange(1, y.size + 1)
    ax.step(x, y, where="post")
    _redraw(interp)
    return None


@builtin("area")
def _area(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    y = np.real(fnum(nums[-1])).flatten()
    x = np.real(fnum(nums[0])).flatten() if len(nums) > 1 else np.arange(1, y.size + 1)
    ax.fill_between(x, y, alpha=0.6)
    ax.plot(x, y)
    _redraw(interp)
    return None


@builtin("fill", "patch")
def _fill(interp, args, nargout):
    ax = _axes(interp)
    x, y = np.real(fnum(args[0])).flatten(), np.real(fnum(args[1])).flatten()
    color = _color(args[2]) if len(args) > 2 else None
    ax.fill(x, y, color=color) if color is not None else ax.fill(x, y)
    _redraw(interp)
    return None


@builtin("errorbar")
def _errorbar(interp, args, nargout):
    ax = _axes(interp)
    nums = [a for a in args if not is_text(a)]
    spec = next((text(a) for a in args if is_text(a)), "")
    if len(nums) == 2:
        y, e = fnum(nums[0]).flatten(), fnum(nums[1]).flatten()
        x = np.arange(1, y.size + 1)
    else:
        x, y, e = (fnum(a).flatten() for a in nums[:3])
    ax.errorbar(x, y, yerr=e, fmt=spec or "-", capsize=3)
    _redraw(interp)
    return None


@builtin("pie")
def _pie(interp, args, nargout):
    ax = _axes(interp)
    x = fnum(args[0]).flatten()
    labels = [to_str(s) for s in args[-1].values_f()] if isinstance(args[-1], Cell) else None
    ax.pie(x, labels=labels, autopct="%1.0f%%" if labels is None else None)
    ax.axis("equal")
    _redraw(interp)
    return None


def _surface(interp, args, kind: str):
    fig = _figure(interp)
    st = _state(interp)
    fig.clf()
    nums = [a for a in args if not is_text(a)]
    if len(nums) == 1:
        Z = np.real(fnum(nums[0]))
        X, Y = np.meshgrid(np.arange(1, Z.shape[1] + 1), np.arange(1, Z.shape[0] + 1))
    else:
        X, Y, Z = (np.real(fnum(a)) for a in nums[:3])
        if X.shape != Z.shape:
            X, Y = np.meshgrid(X.flatten(), Y.flatten())
    if kind in ("contour", "contourf"):
        ax = fig.add_subplot(1, 1, 1)
        levels = int(fnum(nums[3]).flat[0]) if len(nums) > 3 else None
        getattr(ax, kind)(X, Y, Z, levels=levels) if levels else getattr(ax, kind)(X, Y, Z)
    else:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if kind == "surf":
            ax.plot_surface(X, Y, Z, cmap=interp.state.get("colormap", "viridis"), edgecolor="k", linewidth=0.2)
        else:
            ax.plot_wireframe(X, Y, Z, linewidth=0.5)
    st.axes[st.current] = ax
    _redraw(interp)


builtin("surf", "surfc")(lambda interp, args, nargout: _surface(interp, args, "surf"))
builtin("mesh", "meshc")(lambda interp, args, nargout: _surface(interp, args, "mesh"))
builtin("contour")(lambda interp, args, nargout: _surface(interp, args, "contour"))
builtin("contourf")(lambda interp, args, nargout: _surface(interp, args, "contourf"))


@builtin("imagesc", "image", "imshow")
def _imagesc(interp, args, nargout):
    ax = _axes(interp)
    C = np.real(fnum(args[-1] if len(args) in (1, 3) else args[0]))
    ax.imshow(C, aspect="auto", cmap=interp.state.get("colormap", "viridis"), origin="upper")
    _redraw(interp)
    return None


@builtin("colorbar")
def _colorbar(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    mappables = ax.collections + ax.images
    if mappables:
        ax.figure.colorbar(mappables[-1], ax=ax)
    return None


@builtin("colormap")
def _colormap(interp, args, nargout):
    if args and is_text(args[0]):
        interp.state["colormap"] = text(args[0])
    return None


@builtin("view")
def _view(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    if hasattr(ax, "view_init"):
        if len(args) == 2:
            ax.view_init(elev=float(fnum(args[1]).flat[0]), azim=float(fnum(args[0]).flat[0]))
        elif args and fnum(args[0]).size == 1 and int(fnum(args[0]).flat[0]) == 2:
            ax.view_init(elev=90, azim=-90)
    return None


# -- figure / axes management -----------------------------------------------------------

@builtin("figure")
def _figure_b(interp, args, nargout):
    nums = [a for a in args if not is_text(a)]
    if nums:
        fig = _figure(interp, to_int(nums[0], "figure number"))
    else:
        st = _state(interp)
        n = 1
        while n in st.figures:
            n += 1
        fig = _figure(interp, n)
    _, opts = _split_name_values(args, {"name", "position", "color", "visible"})
    if "name" in opts:
        fig.canvas.manager.set_window_title(text(opts["name"])) if fig.canvas.manager else None
    return float(_state(interp).current) if nargout > 0 else None


@builtin("gcf")
def _gcf(interp, args, nargout):
    _figure(interp)
    return float(_state(interp).current)


@builtin("gca")
def _gca(interp, args, nargout):
    _axes(interp, fresh=False)
    return float(_state(interp).current)


@builtin("clf")
def _clf(interp, args, nargout):
    fig = _figure(interp)
    fig.clf()
    _state(interp).axes.pop(_state(interp).current, None)
    return None


@builtin("close")
def _close(interp, args, nargout):
    st = _state(interp)
    plt = _plt()
    if args and is_text(args[0]) and text(args[0]) == "all":
        close_all(interp)
        return None
    n = to_int(args[0], "figure") if args else st.current
    if n is not None and n in st.figures:
        plt.close(st.figures.pop(n))
        st.axes.pop(n, None)
        st.current = max(st.figures) if st.figures else None
    return None


@builtin("hold")
def _hold(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    st = _state(interp)
    if not args:
        st.hold[id(ax)] = not st.hold.get(id(ax), False)
    else:
        st.hold[id(ax)] = text(args[0]).lower() in ("on", "all")
    return None


@builtin("ishold")
def _ishold(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    return bool(_state(interp).hold.get(id(ax), False))


@builtin("subplot")
def _subplot(interp, args, nargout):
    fig = _figure(interp)
    st = _state(interp)
    if len(args) == 1:
        code = to_int(args[0], "subplot")
        m, n, p = code // 100, (code // 10) % 10, code % 10
        pspec = [p]
    else:
        m, n = to_int(args[0], "m"), to_int(args[1], "n")
        pspec = [int(x) for x in num(args[2]).flatten()]
    # MATLAB numbers subplots row-wise, same as matplotlib.
    if len(pspec) == 1:
        ax = fig.add_subplot(m, n, pspec[0])
    else:
        gs = fig.add_gridspec(m, n)
        rows = [(q - 1) // n for q in pspec]
        cols = [(q - 1) % n for q in pspec]
        ax = fig.add_subplot(gs[min(rows):max(rows) + 1, min(cols):max(cols) + 1])
    st.axes[st.current] = ax
    return None


def _label(method):
    def f(interp, args, nargout):
        if not args:
            return None
        ax = _axes(interp, fresh=False)
        s = _label_text(args[0])
        _, opts = _split_name_values(args[1:], {"fontsize", "interpreter", "fontweight", "color"})
        kw = {}
        if "fontsize" in opts:
            kw["fontsize"] = float(fnum(opts["fontsize"]).flat[0])
        if method == "suptitle":
            ax.figure.suptitle(s, **kw)
        elif method == "set_zlabel" and not hasattr(ax, "set_zlabel"):
            pass
        else:
            getattr(ax, method)(s, **kw)
        _redraw(interp)
        return None
    return f


def _label_text(v) -> str:
    if isinstance(v, Cell):
        return "\n".join(to_str(x) for x in v.values_f())
    if is_char(v) and v.shape[0] > 1:
        return "\n".join("".join(r) for r in v)
    return text(v)


builtin("xlabel")(_label("set_xlabel"))
builtin("ylabel")(_label("set_ylabel"))
builtin("zlabel")(_label("set_zlabel"))
builtin("title")(_label("set_title"))
builtin("sgtitle", "suptitle")(_label("suptitle"))


@builtin("legend")
def _legend(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    args, opts = _split_name_values(args, {"location", "fontsize", "orientation", "box"})
    loc_map = {"northeast": "upper right", "northwest": "upper left", "southeast": "lower right",
               "southwest": "lower left", "north": "upper center", "south": "lower center",
               "east": "center right", "west": "center left", "best": "best",
               "northeastoutside": "upper left", "eastoutside": "center left"}
    kw = {}
    if "location" in opts:
        kw["loc"] = loc_map.get(text(opts["location"]).lower(), "best")
    if args and is_text(args[0]) and text(args[0]).lower() in ("off", "hide"):
        leg = ax.get_legend()
        if leg:
            leg.remove()
        return None
    labels = []
    for a in args:
        if isinstance(a, Cell):
            labels.extend(to_str(x) for x in a.values_f())
        elif is_text(a) and text(a).lower() not in ("show", "boxoff", "boxon"):
            labels.append(text(a))
    if labels:
        ax.legend(labels, **kw)
    else:
        ax.legend(**kw)
    _redraw(interp)
    return None


@builtin("grid")
def _grid(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    mode = text(args[0]).lower() if args else "toggle"
    if mode == "minor":
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.3)
    elif mode == "toggle":
        ax.grid(not ax.xaxis._major_tick_kw.get("gridOn", False))
    else:
        ax.grid(mode == "on")
    _redraw(interp)
    return None


@builtin("box")
def _box(interp, args, nargout):
    return None


@builtin("axis")
def _axis(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    for a in args:
        if is_text(a):
            mode = text(a).lower()
            if mode == "equal":
                ax.set_aspect("equal", adjustable="datalim")
            elif mode == "square":
                ax.set_box_aspect(1)
            elif mode == "tight":
                ax.autoscale(enable=True, tight=True)
            elif mode == "off":
                ax.set_axis_off()
            elif mode == "on":
                ax.set_axis_on()
            elif mode in ("auto", "normal"):
                ax.autoscale(True)
                ax.set_aspect("auto")
            elif mode in ("ij",):
                ax.invert_yaxis()
        else:
            lim = np.real(fnum(a)).flatten()
            ax.set_xlim(lim[0], lim[1])
            if lim.size >= 4:
                ax.set_ylim(lim[2], lim[3])
            if lim.size >= 6 and hasattr(ax, "set_zlim"):
                ax.set_zlim(lim[4], lim[5])
    if not args and nargout:
        return np.array([[*ax.get_xlim(), *ax.get_ylim()]])
    _redraw(interp)
    return None


def _lim(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        getter = getattr(ax, f"get_{which}lim", None)
        if not args:
            return np.array([list(getter())]) if getter else None
        if is_text(args[0]):
            ax.autoscale(axis=which if which != "z" else "z")
            return None
        lim = np.real(fnum(args[0])).flatten()
        setter = getattr(ax, f"set_{which}lim", None)
        if setter:
            setter(lim[0], lim[1])
        _redraw(interp)
        return None
    return f


builtin("xlim")(_lim("x"))
builtin("ylim")(_lim("y"))
builtin("zlim")(_lim("z"))


def _ticks(which):
    def f(interp, args, nargout):
        ax = _axes(interp, fresh=False)
        if args:
            getattr(ax, f"set_{which}ticks")(np.real(fnum(args[0])).flatten())
        return None
    return f


builtin("xticks")(_ticks("x"))
builtin("yticks")(_ticks("y"))


@builtin("xticklabels", "yticklabels")
def _ticklabels(interp, args, nargout):
    return None


@builtin("text")
def _text(interp, args, nargout):
    ax = _axes(interp, fresh=False)
    x, y = float(fnum(args[0]).flat[0]), float(fnum(args[1]).flat[0])
    s = _label_text(args[2])
    ax.text(x, y, s)
    _redraw(interp)
    return None


@builtin("drawnow", "shg", "refresh")
def _drawnow(interp, args, nargout):
    _redraw(interp)
    return None


@builtin("saveas")
def _saveas(interp, args, nargout):
    st = _state(interp)
    if len(args) >= 2 and not is_text(args[0]):
        n = to_int(args[0], "figure")
        fname = text(args[1])
    else:
        n = st.current
        fname = text(args[-1])
    if n is None or n not in st.figures:
        raise MatlabError("No figure to save.")
    fmt = text(args[2]) if len(args) > 2 else None
    if fmt and not os.path.splitext(fname)[1]:
        fname += "." + fmt.lstrip("-d")
    st.figures[n].savefig(fname, dpi=150, bbox_inches="tight")
    return None


@builtin("print")
def _print(interp, args, nargout):
    st = _state(interp)
    fmt = "png"
    fname = None
    n = st.current
    for a in args:
        if is_text(a):
            s = text(a)
            if s.startswith("-d"):
                fmt = s[2:].replace("jpeg", "jpg") or "png"
            elif s.startswith("-r") or s.startswith("-"):
                continue
            else:
                fname = s
        elif isinstance(a, np.ndarray) and a.size == 1:
            n = to_int(a, "figure")
    if fname is None:
        raise MatlabError("print: specify a file name, e.g. print('-dpng', 'figure.png').")
    if not os.path.splitext(fname)[1]:
        fname += "." + fmt
    if n is None or n not in st.figures:
        raise MatlabError("No figure to print.")
    st.figures[n].savefig(fname, dpi=150, bbox_inches="tight")
    return None


@builtin("exportgraphics")
def _exportgraphics(interp, args, nargout):
    return _saveas(interp, args, nargout)


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


