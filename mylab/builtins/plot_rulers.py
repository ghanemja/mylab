"""Date, duration and categorical axes, table inputs to plotting functions, and tick label formats.

A plotting function that is given a ``datetime``, ``duration`` or ``categorical`` array draws it on a
*ruler* of that kind (see :class:`mylab.builtins.plotting.Ruler`): the axis then reports its limits and
ticks in those values, labels its ticks with dates, times or category names, and accepts the same kind
of value in ``xlim``, ``xticks``, ``xline`` and friends.

The functions here wrap the plain plotting built-ins: the wrapper converts such arguments to axis
numbers, calls the original function and then installs the rulers on the axes that was drawn into.
The same wrapper understands the table forms ``plot(tbl, xvar, yvar)``.
"""

from __future__ import annotations

import re

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, char, is_char, to_str
from . import plot_charts, plotmore  # noqa: F401  (their functions are wrapped below)
from . import plotting as P
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, text

_X_ONLY = {"xlim", "xticks", "xline", "xregion"}
_Y_ONLY = {"ylim", "yticks", "yline", "yregion"}
_Z_ONLY = {"zlim", "zticks"}
_TRIPLES = {"plot3", "scatter3", "stem3", "fill3", "comet3", "swarmchart3", "bubblechart3"}
_PAIRS = {"plot", "semilogx", "semilogy", "loglog", "scatter", "stairs", "stem", "bar", "barh", "area", "line",
          "fill", "patch", "errorbar", "comet", "swarmchart", "bubblechart", "text", "histogram"}
# Functions that add to the axes without clearing them, whatever the hold state is.
_KEEP = _X_ONLY | _Y_ONLY | _Z_ONLY | {"line", "text", "patch"}
_FLAG_WORDS = {"filled", "stacked", "grouped", "histc", "hist", "horizontal", "vertical", "both"}


def _is_axes(v) -> bool:
    return isinstance(v, P.GraphicsHandle) and v.kind in ("axes", "polaraxes", "geoaxes")


def _peek_axes(interp):
    """The axes a plotting call would draw into, without creating one."""
    st = P._state(interp)
    if st.override_ax is not None:
        return st.override_ax
    if st.current is None or st.current not in st.figures:
        return None
    ax = st.axes.get(st.current)
    return ax if ax is not None and ax in st.figures[st.current].axes else None


def _roles(name: str, args: list) -> dict:
    """Which axis ('x', 'y' or 'z') each datetime/duration/categorical argument belongs to."""
    out: dict = {}
    fixed = "x" if name in _X_ONLY else "y" if name in _Y_ONLY else "z" if name in _Z_ONLY else None
    if fixed:
        return {i: fixed for i, a in enumerate(args) if P._is_time_like(a)}
    groups, cur = [], []
    i = 0
    while i < len(args):
        a = args[i]
        if is_text(a):
            if cur:
                groups.append(cur)
                cur = []
            s = text(a)
            if not P._is_linespec(s) and s.lower() not in _FLAG_WORDS:
                for j in range(i, len(args) - 1):  # option values: the option name says which axis
                    if is_text(args[j]) and P._is_time_like(args[j + 1]):
                        nm = text(args[j]).lower()
                        out[j + 1] = "y" if nm.startswith("y") else "z" if nm.startswith("z") else "x"
                break
        elif isinstance(a, (Cell, StringArray)):
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(i)
        i += 1
    if cur:
        groups.append(cur)
    for g in groups:
        n = len(g)
        for pos, idx in enumerate(g):
            if not P._is_time_like(args[idx]):
                continue
            if name in _TRIPLES or (name == "text" and n >= 3):
                role = "xyz"[pos] if pos < 3 else None
            elif name == "histogram":
                role = "x" if pos == 0 else None
            elif name == "barh":
                role = "x" if n == 1 else ("y", "x")[pos] if pos < 2 else None
            elif name == "errorbar":
                role = "y" if n <= 2 else ("x" if pos == 0 else "y")
            elif n == 1:
                role = "y"
            elif name in ("plot", "semilogx", "semilogy", "loglog", "line", "fill", "patch", "comet"):
                role = "xy"[pos % 2] if name not in ("fill", "patch") or pos < 2 else None
            else:
                role = "xy"[pos] if pos < 2 else None
            if role is None:
                raise MatlabError(f"{name}: a {args[idx].kind} array cannot be used for this input.")
            out[idx] = role
    return out


def _time_call(interp, name: str, fn, args: list, nargout: int):
    args = list(args)
    lead = 1 if args and _is_axes(args[0]) else 0
    target = args[0].obj if lead else _peek_axes(interp)
    keep = target is not None and (name in _KEEP or P._state(interp).hold.get(id(target), False))
    rulers: dict = {}
    for idx, role in _roles(name, args[lead:]).items():
        v = args[lead + idx]
        existing = rulers.get(role) or (P._ruler_of(target, role) if keep else None)
        if existing is not None and existing.kind in ("datetime", "datenum") and v.kind == "duration":
            args[lead + idx] = np.asarray(v.data, dtype=float) / 86400.0  # a length of time on a date axis
            continue
        r = P.new_ruler_for(v, existing)
        rulers[role] = r
        args[lead + idx] = P.ruler_numbers(r, v)
    res = fn(interp, args, nargout)
    ax = args[0].obj if lead else _peek_axes(interp)
    if ax is not None:
        pair = P._yy_pair(ax)
        for role, r in rulers.items():
            for a in (pair if pair is not None and role == "x" else (ax,)):
                if P._ruler_of(a, role) is not r:
                    P.install_ruler(a, role, r)
                if r.kind == "categorical":
                    P.refresh_categorical_ruler(a, role)
        P._redraw(interp)
    return res


# -- table inputs ---------------------------------------------------------------------------------------

_TABLE_COORDS = {"plot": 2, "semilogx": 2, "semilogy": 2, "loglog": 2, "scatter": 2, "stairs": 2, "stem": 2,
                 "bar": 2, "barh": 2, "area": 2, "plot3": 3, "scatter3": 3, "stem3": 3, "swarmchart": 2,
                 "bubblechart": 3, "polarplot": 2, "polarscatter": 2, "geoplot": 2, "geoscatter": 2,
                 "swarmchart3": 3, "bubblechart3": 4}
_TABLE_OPTS = {"colorvariable": "c", "sizevariable": "s"}


def _is_table(v) -> bool:
    from .table import TableObject
    return isinstance(v, TableObject)


def _table_vars(tbl, spec, strict: bool = True):
    """[(name, column), ...] for a variable name, a list of names or variable numbers."""
    if is_text(spec) or isinstance(spec, (Cell, StringArray, MString)):
        names = P._text_list(spec)
        for n in names:
            if n not in tbl.names:
                if strict:
                    raise MatlabError(f"Unrecognized table variable name '{n}'.")
                return None
        return [(n, tbl.columns[tbl.names.index(n)]) for n in names]
    if isinstance(spec, np.ndarray) and spec.dtype.kind in "biuf" and spec.size:
        v = np.asarray(spec).flatten()
        if v.dtype == bool:
            idx = [int(i) for i in np.flatnonzero(v)]
        else:
            idx = [int(round(float(x))) - 1 for x in v]
        if any(i < 0 or i >= len(tbl.names) for i in idx) or (v.dtype != bool and np.any(v != np.round(v))):
            if strict:
                raise MatlabError("Table variable numbers must be between 1 and the number of variables.")
            return None
        return [(tbl.names[i], tbl.columns[i]) for i in idx]
    return None


def _plain_column(interp, col):
    """A table variable as something the plotting functions take (text becomes categorical)."""
    if isinstance(col, (Cell, StringArray)):
        return BUILTINS["categorical"](interp, [col], 1)
    return col


def _table_call(interp, name: str, fn, args: list, nargout: int):
    args = list(args)
    lead = args[:1] if _is_axes(args[0]) else []
    tbl = args[len(lead)]
    rest = args[len(lead) + 1:]
    ncoord = _TABLE_COORDS[name]
    specs: list = []
    while rest and len(specs) < ncoord:
        vs = _table_vars(tbl, rest[0], strict=not specs and not (is_text(rest[0]) and P._is_linespec(text(rest[0]))))
        if vs is None:
            break
        specs.append(vs)
        rest.pop(0)
    need = {"bubblechart": 3, "plot3": 3, "scatter3": 3, "stem3": 3, "swarmchart3": 3, "bubblechart3": 4}.get(name, 1)
    if len(specs) < need:
        raise MatlabError(f"{name}: name the table variables to plot, for example {name}(tbl, "
                          + ", ".join(f"'{v}var'" for v in "xyzs"[:max(need, 2)]) + ").")
    extra: dict = {}
    opts: list = []
    k = 0
    while k < len(rest):
        key = text(rest[k]).lower() if is_text(rest[k]) else None
        if key in _TABLE_OPTS and k + 1 < len(rest):
            extra[_TABLE_OPTS[key]] = _table_vars(tbl, rest[k + 1])[0]
            k += 2
            continue
        opts.append(rest[k])
        k += 1
    if extra and name not in ("scatter", "scatter3", "swarmchart", "swarmchart3", "bubblechart", "bubblechart3",
                              "polarscatter", "geoscatter"):
        raise MatlabError(f"{name} does not take ColorVariable or SizeVariable.")
    if len(specs) == 1:  # only the y variable: plot against the row numbers
        specs = [None] + specs
    counts = {len(sp) for sp in specs if sp is not None and len(sp) > 1}
    if len(counts) > 1:
        raise MatlabError(f"{name}: lists of table variables must have the same number of names.")
    nser = counts.pop() if counts else 1
    st = P._state(interp)
    handles: list = []
    ax = None
    prev_hold = None
    try:
        for i in range(nser):
            cols, names = [], []
            for sp in specs:
                if sp is None:
                    cols.append(np.arange(1.0, tbl.nrows + 1).reshape(-1, 1))
                    names.append("")
                else:
                    nm, col = sp[i if len(sp) > 1 else 0]
                    cols.append(_plain_column(interp, col))
                    names.append(nm)
            call = lead + cols
            if "s" in extra or "c" in extra:
                if name in ("bubblechart", "bubblechart3"):
                    if "s" in extra:
                        raise MatlabError(f"{name}: give the size variable as the last positional variable.")
                else:
                    call.append(_plain_column(interp, extra["s"][1]) if "s" in extra else np.zeros((0, 0)))
                if "c" in extra:
                    c = _plain_column(interp, extra["c"][1])
                    call.append(np.asarray(c.data, float) if P._is_time_like(c) else c)
            res = fn(interp, call + opts, 1)
            hs = P._handles_of(res)
            for h in hs:
                try:
                    P._set_prop(h, "DisplayName", char(names[1 if len(names) > 1 else 0]))
                except MatlabError:
                    pass
            handles.extend(hs)
            if i == 0:
                ax = lead[0].obj if lead else _peek_axes(interp)
                if ax is not None and nser > 1:
                    prev_hold = st.hold.get(id(ax), False)
                    st.hold[id(ax)] = True
    finally:
        if ax is not None and prev_hold is not None:
            st.hold[id(ax)] = prev_hold
    if ax is not None and getattr(ax, "name", "") in ("rectilinear", "3d"):
        labels = [sp[0][0] if sp is not None and len(sp) == 1 else "" for sp in specs]
        if name == "barh":
            labels[:2] = labels[1::-1]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if len(labels) > 2 and hasattr(ax, "set_zlabel") and name != "bubblechart":
            ax.set_zlabel(labels[2])
        if nser > 1:
            P._make_legend(interp, ax)
        if "c" in extra and name.startswith(("scatter", "swarm", "bubble")):
            col = extra["c"][1]
            if isinstance(col, np.ndarray) and col.dtype.kind in "biuf" and (col.ndim < 2 or col.shape[1] == 1):
                BUILTINS["colorbar"](interp, lead, 1).obj.set_label(extra["c"][0])
    P._redraw(interp)
    if nargout <= 0:
        return None
    return P._handle_list(handles)


def _wrap_plot_function(name: str):
    fn = BUILTINS[name]
    table = name in _TABLE_COORDS
    timed = name in _PAIRS or name in _TRIPLES or name in _KEEP

    def wrapper(interp, args, nargout, _fn=fn):
        if table and args:
            first = args[1] if _is_axes(args[0]) and len(args) > 1 else args[0]
            if _is_table(first):
                return _table_call(interp, name, wrapper, list(args), nargout)
        if timed and any(P._is_time_like(a) for a in args):
            return _time_call(interp, name, _fn, list(args), nargout)
        return _fn(interp, args, nargout)

    wrapper.__name__ = getattr(fn, "__name__", name)
    wrapper.__doc__ = fn.__doc__
    BUILTINS[name] = wrapper


def _install() -> None:
    from . import categorical, datetimes
    names = (_PAIRS | _TRIPLES | _KEEP | set(_TABLE_COORDS)) & set(BUILTINS)
    for name in sorted(names):
        _wrap_plot_function(name)
        datetimes._NATIVE.pop(name, None)  # dates and durations are no longer turned into serial numbers
    for name in ("bar", "barh", "plot", "scatter", "stem", "stairs", "area", "line", "text", "swarmchart",
                 "xline", "yline", "xlim", "ylim", "xticks", "yticks"):
        categorical._NATIVE.pop(name, None)


_install()


# -- tick label formats ----------------------------------------------------------------------------------

_NAMED_FORMATS = {"usd": "$%,.2f", "eur": "€%,.2f", "gbp": "£%,.2f", "jpy": "¥%,d",
                  "degrees": "%g°", "percentage": "%g%%"}
_SPEC = re.compile(r"%([-+ 0#,]*)(\d+)?(?:\.(\d+))?([diufeEgGxX%])")


def format_number(pattern: str, x: float) -> str:
    """Format one tick value with a printf-style pattern ('%.2f', '$%,.2f', '%g%%')."""
    def rep(m: re.Match) -> str:
        flags, width, prec, conv = m.group(1), m.group(2), m.group(3), m.group(4)
        if conv == "%":
            return "%"
        comma = "," in flags
        flags = flags.replace(",", "")
        if conv in "diu":
            body = f"{int(round(x)):,d}" if comma else f"{int(round(x)):d}"
        elif conv in "xX":
            body = format(int(round(x)), conv)
        else:
            body = format(x, ("," if comma else "") + (f".{prec}" if prec is not None else "") + conv)
        if "+" in flags and x >= 0:
            body = "+" + body
        elif " " in flags and x >= 0:
            body = " " + body
        if width:
            w = int(width)
            body = body.ljust(w) if "-" in flags else body.rjust(w, "0" if "0" in flags else " ")
        return body
    return _SPEC.sub(rep, pattern)


def tick_format_get(ax, which: str) -> str:
    r = P._ruler_of(ax, which)
    if r is None:
        return getattr(ax, "_mylab_tickfmt", {}).get(which) or "%g"
    if r.kind in ("datetime", "datenum"):
        fmt = getattr(ax, f"{which}axis").get_major_formatter()
        return r.fmt or (fmt._format()[0] if hasattr(fmt, "_format") else "")
    if r.kind == "duration":
        return r.fmt or r.dfmt
    return ""


def tick_format_set(ax, which: str, fmt: str) -> None:
    """Change how the tick labels of one axis are written (numbers, dates or durations)."""
    from matplotlib.ticker import FuncFormatter
    axis = getattr(ax, f"{which}axis", None)
    if axis is None:
        raise MatlabError(f"These axes have no {which}-axis.")
    r = P._ruler_of(ax, which)
    auto = fmt.lower() == "auto"
    setattr(ax, f"_mylab_{which}ticklabels", None)
    if r is not None and r.kind in ("datetime", "datenum"):
        if not auto:
            from .datetimes import _check_datetime_format
            _check_datetime_format(fmt)
        r.fmt, r.classic = (None if auto else fmt), False
        axis.set_major_formatter(P._date_tick_classes()[1](r))
        return
    if r is not None and r.kind == "duration":
        if not auto:
            from .datetimes import _check_duration_format
            fmt = _check_duration_format(fmt)
        r.fmt = None if auto else fmt
        P.install_ruler(ax, which, r)
        return
    if r is not None:
        raise MatlabError("A categorical axis has no tick label format.")
    store = ax.__dict__.setdefault("_mylab_tickfmt", {})
    if auto:
        store.pop(which, None)
        from matplotlib.ticker import ScalarFormatter
        axis.set_major_formatter(ScalarFormatter())
        return
    pattern = _NAMED_FORMATS.get(fmt.lower(), fmt)
    if not _SPEC.search(pattern.replace("%%", "")):
        raise MatlabError(f"Invalid tick label format '{fmt}'. Use a pattern such as '%.2f' or one of 'usd', "
                          "'eur', 'gbp', 'jpy', 'degrees', 'percentage', 'auto'.")
    store[which] = pattern
    axis.set_major_formatter(FuncFormatter(lambda x, pos, p=pattern: format_number(p, float(x))))


def _tickformat(which: str):
    def f(interp, args, nargout):
        args = list(args)
        ax = args.pop(0).obj if args and _is_axes(args[0]) else P._axes(interp, fresh=False)
        if not args:
            return char(tick_format_get(ax, which))
        tick_format_set(ax, which, text(args[0], "tick label format"))
        P._redraw(interp)
        return None
    return f


builtin("xtickformat")(_tickformat("x"))
builtin("ytickformat")(_tickformat("y"))
builtin("ztickformat")(_tickformat("z"))


def _set_secondary(ax, which: str, label: str | None) -> None:
    axis = getattr(ax, f"{which}axis")
    r = P._ruler_of(ax, which)
    if r is not None and r.kind in ("datetime", "datenum"):
        r.secondary = label
        return
    fmt = axis.get_major_formatter()
    if label is None:
        fmt.__dict__.pop("get_offset", None)
    else:
        fmt.get_offset = lambda: label
    ax.__dict__.setdefault("_mylab_secondary", {})[which] = label


def _get_secondary(ax, which: str) -> str:
    axis = getattr(ax, f"{which}axis")
    r = P._ruler_of(ax, which)
    if r is not None and r.secondary is not None:
        return r.secondary
    fmt = axis.get_major_formatter()
    try:
        fmt.format_ticks(list(axis.get_majorticklocs()))
    except (ValueError, AttributeError, TypeError):
        pass
    return str(fmt.get_offset())


def _secondary(which: str):
    def f(interp, args, nargout):
        args = list(args)
        ax = args.pop(0).obj if args and _is_axes(args[0]) else P._axes(interp, fresh=False)
        if not args:
            raise MatlabError(f"{which}secondarylabel needs the label text, or 'Visible' and 'on' or 'off'.")
        if len(args) == 2 and is_text(args[0]) and text(args[0]).lower() == "visible":
            _set_secondary(ax, which, None if P._onoff(args[1]) else "")
        else:
            _set_secondary(ax, which, P._label_text(args[0]))
        P._redraw(interp)
        return None
    return f


builtin("xsecondarylabel")(_secondary("x"))
builtin("ysecondarylabel")(_secondary("y"))
builtin("zsecondarylabel")(_secondary("z"))


@builtin("datetick")
def _datetick(interp, args, nargout):
    """Label an axis that holds serial date numbers with dates."""
    from .datetimes import _CLASSIC_FORMATS
    args = list(args)
    ax = args.pop(0).obj if args and _is_axes(args[0]) else P._axes(interp, fresh=False)
    which, fmt, keep_limits, keep_ticks = "x", None, False, False
    for a in args:
        if is_text(a):
            s = text(a)
            low = s.lower()
            if low in ("x", "y", "z"):
                which = low
            elif low == "keeplimits":
                keep_limits = True
            elif low == "keepticks":
                keep_ticks = True
            else:
                fmt = s
        else:
            code = int(round(float(np.real(fnum(a)).flat[0])))
            if code not in _CLASSIC_FORMATS:
                raise MatlabError(f"datetick: unknown date format number {code} (use 0 to 31).")
            fmt = _CLASSIC_FORMATS[code]
    if not hasattr(ax, f"{which}axis"):
        raise MatlabError(f"datetick: these axes have no {which}-axis.")
    old = P._ruler_of(ax, which)
    if old is not None and old.kind not in ("datenum", "datetime"):
        raise MatlabError("datetick needs an axis with serial date numbers (see datenum).")
    ticks = np.asarray(getattr(ax, f"get_{which}ticks")(), dtype=float)
    lo, hi = sorted(getattr(ax, f"get_{which}lim")())
    r = old if old is not None else P.Ruler("datenum")
    r.fmt, r.classic = fmt, fmt is not None
    P.install_ruler(ax, which, r)
    axis = getattr(ax, f"{which}axis")
    if keep_ticks:
        P._set_tick_values(ax, which, ticks[(ticks >= lo) & (ticks <= hi)])
    elif not keep_limits:  # widen the limits to the nearest ticks on both sides
        loc = axis.get_major_locator()
        t = np.asarray(loc.tick_values(lo, hi), dtype=float)
        if t.size >= 2:
            step = float(np.median(np.diff(t)))
            new_lo = t[0] if t[0] <= lo + 1e-9 * abs(step) else t[0] - step
            new_hi = t[-1] if t[-1] >= hi - 1e-9 * abs(step) else t[-1] + step
            P._set_lim(ax, which, float(new_lo), float(new_hi))
    P._redraw(interp)
    return None


@builtin("ruler2num")
def _ruler2num(interp, args, nargout):
    if len(args) != 2 or not isinstance(args[1], P.GraphicsHandle) or args[1].kind != "ruler":
        raise MatlabError("ruler2num needs a value and a ruler, e.g. ruler2num(t, ax.XAxis).")
    ref = args[1].obj
    return np.asarray(P.axis_numbers(ref.axes, ref.which, args[0]), dtype=float)


@builtin("num2ruler")
def _num2ruler(interp, args, nargout):
    if len(args) != 2 or not isinstance(args[1], P.GraphicsHandle) or args[1].kind != "ruler":
        raise MatlabError("num2ruler needs numbers and a ruler, e.g. num2ruler(x, ax.XAxis).")
    ref = args[1].obj
    nums = np.real(fnum(args[0]))
    out = P.ruler_values(ref.axes, ref.which, nums.flatten(order="F"))
    return out.reshape(nums.shape, order="F") if hasattr(out, "reshape") and nums.ndim == 2 else out


# -- ruler objects (ax.XAxis, ax.YAxis, ax.ZAxis) ----------------------------------------------------------

class AxisRef:
    """One axis (x, y or z) of an axes: the object behind ax.XAxis."""

    def __init__(self, ax, which: str):
        self.axes = ax
        self.which = which

    @property
    def figure(self):
        return self.axes.figure

    @property
    def axis(self):
        return getattr(self.axes, f"{self.which}axis")


_RULER_CLASS = {None: "matlab.graphics.axis.decorator.NumericRuler",
                "datenum": "matlab.graphics.axis.decorator.NumericRuler",
                "datetime": "matlab.graphics.axis.decorator.DatetimeRuler",
                "duration": "matlab.graphics.axis.decorator.DurationRuler",
                "categorical": "matlab.graphics.axis.decorator.CategoricalRuler"}


class RulerHandle(P.GraphicsHandle):
    def class_name(self) -> str:
        r = P._ruler_of(self.obj.axes, self.obj.which)
        return _RULER_CLASS[r.kind if r is not None else None]

    def isa_class(self, name: str) -> bool:
        return name in (self.class_name(), self.class_name().rsplit(".", 1)[-1], "handle",
                        "matlab.graphics.Graphics")

    def display_lines(self) -> list[str]:
        lines = super().display_lines()
        if lines and "with properties" in lines[0]:
            lines[0] = f"  {self.class_name().rsplit('.', 1)[-1]} with properties:"
        return lines


def ruler_handle(interp, ax, which: str):
    if not hasattr(ax, f"{which}axis"):
        raise MatlabError(f"These axes have no {which.upper()}Axis.")
    refs = ax.__dict__.setdefault("_mylab_axisrefs", {})
    ref = refs.get(which)
    if ref is None:
        ref = refs[which] = AxisRef(ax, which)
    st = P._state(interp)
    h = st.handles.get(id(ref))
    if h is None or h.obj is not ref:
        h = RulerHandle(interp, "ruler", ref)
        st.handles[id(ref)] = h
    return h


def _axh(h):
    """The axes handle a ruler handle belongs to."""
    return P._wrap(h.interp, P._axes_kind(h.obj.axes), h.obj.axes)


def _via_axes(prop: str):
    """Ruler property that is the axes property ``X<prop>`` / ``Y<prop>`` / ``Z<prop>``."""
    return (lambda h: P._get_prop(_axh(h), h.obj.which.upper() + prop),
            lambda h, v: P._set_prop(_axh(h), h.obj.which.upper() + prop, v))


def _exponent_get(h):
    from matplotlib.ticker import ScalarFormatter
    fmt = h.obj.axis.get_major_formatter()
    if "Exponent" in h.extra:
        return P._num_val(h.extra["Exponent"])
    if isinstance(fmt, ScalarFormatter):
        try:
            fmt.set_locs(h.obj.axis.get_majorticklocs())
            return P._num_val(fmt.orderOfMagnitude)
        except (ValueError, AttributeError, TypeError):
            pass
    return P._num_val(0)


def _exponent_set(h, v):
    from matplotlib.ticker import ScalarFormatter
    e = float(np.real(fnum(v)).flat[0])
    if e != round(e):
        raise MatlabError("Exponent must be a whole number.")
    if P._ruler_of(h.obj.axes, h.obj.which) is not None:
        raise MatlabError("Only a numeric ruler has an Exponent.")
    fmt = ScalarFormatter(useOffset=False, useMathText=True)
    fmt.set_powerlimits((int(e), int(e)))
    if e == 0:
        fmt.set_scientific(False)
    h.obj.axis.set_major_formatter(fmt)
    h.obj.axes.__dict__.setdefault("_mylab_tickfmt", {}).pop(h.obj.which, None)
    h.extra["Exponent"] = e


def _ruler_fontsize_set(h, v):
    fs = P._scalar(v)
    h.obj.axes.tick_params(axis=h.obj.which, labelsize=fs)
    h.obj.axis.label.set_fontsize(fs * 1.1)
    h.extra["FontSize"] = fs


def _ruler_fontsize_get(h):
    if "FontSize" in h.extra:
        return P._num_val(h.extra["FontSize"])
    labels = h.obj.axis.get_ticklabels()
    return P._num_val(labels[0].get_fontsize() if labels else 10)


def _ruler_visible_set(h, v):
    on = P._onoff(v)
    h.obj.axis.set_visible(on)
    side = {"x": ("bottom", "top"), "y": ("left", "right")}.get(h.obj.which, ())
    for name in side:
        if name in h.obj.axes.spines:
            h.obj.axes.spines[name].set_visible(on)


def _ruler_categories(h):
    r = P._ruler_of(h.obj.axes, h.obj.which)
    if r is None or r.kind != "categorical":
        raise MatlabError("Only a categorical ruler has Categories.")
    return Cell.from_list([char(c) for c in r.categories])


def _ruler_secondary(h):
    ax, which = h.obj.axes, h.obj.which
    t = h.obj.axis.get_offset_text()
    t.set_text(_get_secondary(ax, which))
    return P._wrap(h.interp, "text", t)


def _ruler_secondary_set(h, v):
    if isinstance(v, P.GraphicsHandle):
        _set_secondary(h.obj.axes, h.obj.which, v.obj.get_text())
    else:
        _set_secondary(h.obj.axes, h.obj.which, P._label_text(v))


def _ruler_linewidth_set(h, v):
    lw = P._scalar(v)
    for name in {"x": ("bottom", "top"), "y": ("left", "right")}.get(h.obj.which, ()):
        if name in h.obj.axes.spines:
            h.obj.axes.spines[name].set_linewidth(lw)
    h.obj.axes.tick_params(axis=h.obj.which, width=lw)
    h.extra["LineWidth"] = lw


def _ruler_tickdir_set(h, v):
    d = text(v).lower()
    if d not in ("in", "out", "both", "none"):
        raise MatlabError("TickDirection must be 'in', 'out', 'both' or 'none'.")
    h.obj.axes.tick_params(axis=h.obj.which, direction={"both": "inout", "none": "in"}[d] if d in ("both", "none")
                           else d, length=0 if d == "none" else 3.5)
    h.extra["TickDirection"] = d


def _ruler_ticklength_set(h, v):
    tl = np.real(fnum(v)).flatten()
    bbox = h.obj.axes.get_window_extent()
    longest = max(bbox.width, bbox.height) * 72.0 / h.obj.axes.figure.dpi
    h.obj.axes.tick_params(axis=h.obj.which, length=float(tl[0]) * longest)
    h.extra["TickLength"] = P._row(tl if tl.size == 2 else [tl[0], 0.025])


def _ruler_minor_values(h):
    return P._row(h.obj.axis.get_minorticklocs())


def _ruler_minor_values_set(h, v):
    from matplotlib.ticker import FixedLocator
    h.obj.axis.set_minor_locator(FixedLocator(list(np.real(P.axis_numbers(h.obj.axes, h.obj.which, v)).flatten())))


P._CLASS["ruler"] = _RULER_CLASS[None]
P._TYPE["ruler"] = "numericruler"
P._PROPS["ruler"] = {
    "Limits": _via_axes("Lim"),
    "LimitsMode": _via_axes("LimMode"),
    "Scale": _via_axes("Scale"),
    "Direction": _via_axes("Dir"),
    "TickValues": _via_axes("Tick"),
    "TickValuesMode": _via_axes("TickMode"),
    "TickLabels": _via_axes("TickLabel"),
    "TickLabelsMode": _via_axes("TickLabelMode"),
    "TickLabelRotation": _via_axes("TickLabelRotation"),
    "Color": _via_axes("Color"),
    "Label": _via_axes("Label"),
    "MinorTick": _via_axes("MinorTick"),
    "TickLabelFormat": (lambda h: char(tick_format_get(h.obj.axes, h.obj.which)),
                        lambda h, v: tick_format_set(h.obj.axes, h.obj.which, text(v, "TickLabelFormat"))),
    "Exponent": (_exponent_get, _exponent_set),
    "FontSize": (_ruler_fontsize_get, _ruler_fontsize_set),
    "LineWidth": (lambda h: P._num_val(h.extra.get("LineWidth", 0.5)), _ruler_linewidth_set),
    "TickDirection": (lambda h: char(h.extra.get("TickDirection", "in")), _ruler_tickdir_set),
    "TickLength": (lambda h: h.extra.get("TickLength", P._row([0.01, 0.025])), _ruler_ticklength_set),
    "MinorTickValues": (_ruler_minor_values, _ruler_minor_values_set),
    "SecondaryLabel": (_ruler_secondary, _ruler_secondary_set),
    "Categories": (_ruler_categories, None),
    "Visible": (lambda h: P._oo(h.obj.axis.get_visible()), _ruler_visible_set),
    "Parent": (lambda h: _axh(h), None),
    "Type": (lambda h: char({None: "numericruler", "datenum": "numericruler", "datetime": "datetimeruler",
                             "duration": "durationruler", "categorical": "categoricalruler"}[
                                 getattr(P._ruler_of(h.obj.axes, h.obj.which), "kind", None)]), None),
}
P._DISPLAY["ruler"] = ["Limits", "Scale", "Direction", "TickValues", "TickLabelFormat", "Exponent", "Color",
                       "FontSize", "Visible"]

for _w in "xyz":
    P._AXES_PROPS[f"{_w.upper()}Axis"] = (lambda h, _w=_w: ruler_handle(h.interp, h.obj, _w), None)
P._AXES_PROPS["XTickLabelFormat"] = (lambda h: char(tick_format_get(h.obj, "x")),
                                     lambda h, v: tick_format_set(h.obj, "x", text(v)))
P._AXES_PROPS["YTickLabelFormat"] = (lambda h: char(tick_format_get(h.obj, "y")),
                                     lambda h, v: tick_format_set(h.obj, "y", text(v)))


def text_of(v) -> str:
    """Plain text of a char vector or string (helper shared by the chart modules)."""
    return to_str(v) if isinstance(v, MString) or is_char(v) else text(v)
