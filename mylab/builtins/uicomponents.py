"""App-building functions: uifigure, uibutton, ..., uicontrol, guidata and friends.

The component tree, its properties and the front-end protocol live in
:mod:`mylab.ui`; this module only maps the MATLAB function names onto it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import ui as U
from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char
from ._registry import BUILTINS, builtin

_txt = U._txt
_is_text = U._is_text


def _is_classic_figure(v: Any) -> bool:
    return getattr(v, "kind", None) == "figure" and hasattr(v, "obj") and not isinstance(v, U.UIComponent)


def _is_parent(v: Any) -> bool:
    return isinstance(v, U.UIComponent) or _is_classic_figure(v)


def _pairs(args: list, what: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    k = 0
    while k < len(args):
        a = args[k]
        if isinstance(a, Struct) and a.numel() == 1:
            out.extend((f, a.get_field(0, f)) for f in a.fields)
            k += 1
            continue
        if not _is_text(a) or k + 1 >= len(args):
            raise MatlabError(f"{what}: options must be Name, Value pairs, e.g. {what}(parent, 'Text', 'OK').")
        out.append((_txt(a), args[k + 1]))
        k += 2
    return out


def _classic_parent(interp) -> U.UIComponent:
    """The window that classic functions use when no parent is given: the current figure."""
    st = U.state(interp)
    if st.callbacks and st.callbacks[-1][1] is not None and not st.callbacks[-1][1].deleted:
        fig = st.callbacks[-1][1]
        if fig.mpl is not None:
            return fig
    from .plotting import _figure, _wrap
    return U.classic_node(interp, _wrap(interp, "figure", _figure(interp)))


def _create(interp, kind: str, args: list, what: str, styles: dict | None = None, classic: bool = False,
            first: str | None = None) -> Any:
    """Common constructor: [parent], [style], Name, Value, ..."""
    args = list(args)
    parent = None
    if args and _is_parent(args[0]):
        parent = U.parent_node(interp, args.pop(0))
    elif args and isinstance(args[0], np.ndarray) and args[0].size == 0 and len(args) % 2 == 1 \
            and not is_char(args[0]):
        args.pop(0)
    style = ""
    if styles and args and _is_text(args[0]) and len(args) % 2 == 1:
        s = _txt(args[0]).lower()
        if s not in styles:
            raise MatlabError(f"{what}: '{_txt(args[0])}' is not a style. Use " +
                              " or ".join(f"'{x}'" for x in styles) + ".")
        style = s
        kind = styles[s] or kind
        args.pop(0)
    if first is not None and args and len(args) % 2 == 1:
        args = [char(first)] + args
    pairs = _pairs(args, what)
    rest = []
    for name, value in pairs:
        if name.lower() == "parent":
            parent = U.parent_node(interp, value)
        else:
            rest.append((name, value))
    if parent is None and kind != "figure":
        parent = _classic_parent(interp) if classic else U.UIComponent(interp, "figure")
    node = U.UIComponent(interp, kind, parent)
    node.style = style
    try:
        # Units first, so that a Position given in the same call is read in those units.
        for name, value in sorted(rest, key=lambda nv: nv[0].lower() != "units"):
            node.set_prop(name, value)
    except MatlabError:
        node.destroy(run_callbacks=False)
        raise
    return node


def _ctor(kind: str, what: str, styles: dict | None = None, classic: bool = False, first: str | None = None):
    def make(interp, args, nargout):
        return _create(interp, kind, args, what, styles, classic, first)
    make.__name__ = "_" + what
    return make


for _names, _kind, _styles, _classic in (
    (("uilabel",), "label", None, False),
    (("uibutton",), "button", {"push": "button", "state": "statebutton"}, False),
    (("uieditfield",), "editfield", {"text": "editfield", "numeric": "numericeditfield"}, False),
    (("uitextarea",), "textarea", None, False),
    (("uidropdown",), "dropdown", None, False),
    (("uilistbox",), "listbox", None, False),
    (("uicheckbox",), "checkbox", None, False),
    (("uiradiobutton",), "radiobutton", None, False),
    (("uitogglebutton",), "togglebutton", None, False),
    (("uiswitch",), "switch", {"slider": None, "rocker": None, "toggle": None}, False),
    (("uislider",), "slider", None, False),
    (("uispinner",), "spinner", None, False),
    (("uiknob",), "knob", {"continuous": "knob", "discrete": "discreteknob"}, False),
    (("uigauge",), "gauge", {"circular": None, "linear": None, "ninetydegree": None, "semicircular": None}, False),
    (("uilamp",), "lamp", None, False),
    (("uiimage",), "image", None, False),
    (("uidatepicker",), "datepicker", None, False),
    (("uihyperlink",), "hyperlink", None, False),
    (("uipanel",), "panel", None, True),
    (("uibuttongroup",), "buttongroup", None, True),
    (("uitabgroup",), "tabgroup", None, True),
    (("uitable",), "table", None, True),
    (("uimenu",), "menu", None, True),
):
    builtin(*_names)(_ctor(_kind, _names[0], _styles, _classic))


@builtin("uifigure")
def _uifigure(interp, args, nargout):
    args = list(args)
    if len(args) % 2 == 1 and _is_text(args[0]):  # uifigure('My app')
        args = [char("Name")] + args
    return _create(interp, "figure", args, "uifigure")


@builtin("uitab")
def _uitab(interp, args, nargout):
    args = list(args)
    if not args or not (isinstance(args[0], U.UIComponent) and args[0].kind == "tabgroup") and not any(
            _is_text(a) and _txt(a).lower() == "parent" for a in args[::2]):
        if args and _is_parent(args[0]):
            raise MatlabError("uitab: the parent must be a tab group, e.g. tg = uitabgroup(fig); uitab(tg, "
                              "'Title', 'Data').")
        group = _create(interp, "tabgroup", [], "uitabgroup", classic=True)
        args = [group] + args
    return _create(interp, "tab", args, "uitab")


@builtin("uigridlayout")
def _uigridlayout(interp, args, nargout):
    args = list(args)
    parent = [args.pop(0)] if args and _is_parent(args[0]) else []
    size = None
    if args and isinstance(args[0], np.ndarray) and not is_char(args[0]) and args[0].size == 2:
        size = [int(x) for x in np.asarray(args.pop(0), dtype=float).flatten()]
    node = _create(interp, "gridlayout", parent + args, "uigridlayout")
    if size is not None:
        if min(size) < 1:
            raise MatlabError("uigridlayout: the size must be [rows columns] with positive whole numbers.")
        if "RowHeight" not in node.props:
            node.props["RowHeight"] = Cell.from_list([char("1x")] * size[0])
        if "ColumnWidth" not in node.props:
            node.props["ColumnWidth"] = Cell.from_list([char("1x")] * size[1])
    return node


@builtin("uicontrol")
def _uicontrol(interp, args, nargout):
    args = list(args)
    if len(args) == 1 and isinstance(args[0], U.UIComponent) and args[0].kind == "uicontrol":
        return None  # uicontrol(h) gives h the keyboard focus
    return _create(interp, "uicontrol", args, "uicontrol", classic=True)


@builtin("uiaxes")
def _uiaxes(interp, args, nargout):
    return _make_axes(interp, list(args), normalized=False, classic=False)


def _make_axes(interp, args: list, normalized: bool, classic: bool):
    parent = None
    if args and _is_parent(args[0]):
        parent = U.parent_node(interp, args.pop(0))
    pairs = _pairs(args, "uiaxes")
    rest = []
    for name, value in pairs:
        if name.lower() == "parent":
            parent = U.parent_node(interp, value)
        else:
            rest.append((name, value))
    if parent is None:
        parent = U.UIComponent(interp, "figure")
    ax = U.uiaxes_class()(interp, parent)
    if normalized:
        ax.ui_props["Units"] = char("normalized")
        ax.ui_props["Position"] = np.array([[0.13, 0.11, 0.775, 0.815]])
    try:
        for name, value in sorted(rest, key=lambda nv: nv[0].lower() != "units"):
            ax.set_field(interp, name, value)
    except MatlabError:
        ax.ui_destroy()
        raise
    return ax


# -- dialogs in app windows --------------------

def _dialog_figure(interp, args: list, what: str) -> tuple[U.UIComponent | None, list]:
    args = list(args)
    if args and _is_parent(args[0]):
        return U.figure_of(interp, args[0]), args[1:]
    if args and isinstance(args[0], np.ndarray) and args[0].size == 0 and not is_char(args[0]):
        return None, args[1:]
    raise MatlabError(f"{what}: the first input must be the figure the dialog belongs to, e.g. "
                      f"{what}(fig, ...).")


def _message(v: Any) -> str:
    return "\n".join(U._text_list(v)) if not _is_text(v) else _txt(v)


@builtin("uialert")
def _uialert(interp, args, nargout):
    fig, rest = _dialog_figure(interp, args, "uialert")
    if len(rest) < 2:
        raise MatlabError("uialert needs a message and a title: uialert(fig, 'Not a number', 'Input').")
    opts = {k.lower(): v for k, v in _pairs(rest[2:], "uialert")}
    icon = _txt(opts.get("icon", char("error"))).lower()
    U.add_dialog(interp, "alert", fig, opts.get("closefcn"), message=_message(rest[0]), title=_txt(rest[1]),
                 icon=icon if icon in ("error", "warning", "info", "success", "question") else "none",
                 options=["OK"])
    return None


@builtin("uiconfirm")
def _uiconfirm(interp, args, nargout):
    fig, rest = _dialog_figure(interp, args, "uiconfirm")
    if len(rest) < 2:
        raise MatlabError("uiconfirm needs a message and a title: uiconfirm(fig, 'Delete it?', 'Confirm').")
    message, title = _message(rest[0]), _txt(rest[1])
    opts = {k.lower(): v for k, v in _pairs(rest[2:], "uiconfirm")}
    options = U._text_list(opts["options"]) if "options" in opts else ["OK", "Cancel"]
    if not options:
        raise MatlabError("uiconfirm: Options must hold at least one button text.")

    def pick(key: str, fallback: int) -> str:
        v = opts.get(key)
        if v is None:
            return options[min(fallback, len(options) - 1)]
        if _is_text(v):
            if _txt(v) not in options:
                raise MatlabError(f"uiconfirm: '{_txt(v)}' is not one of the Options.")
            return _txt(v)
        k = int(U._scalar(v, key))
        if not 1 <= k <= len(options):
            raise MatlabError(f"uiconfirm: {key} must be between 1 and {len(options)}.")
        return options[k - 1]

    default = pick("defaultoption", 0)
    pick("canceloption", len(options) - 1)
    icon = _txt(opts.get("icon", char("question"))).lower()
    close_fcn = opts.get("closefcn")
    payload = {"message": message, "title": title, "options": options, "default": default,
               "icon": icon if icon in ("error", "warning", "info", "success", "question") else "none"}
    st = U.state(interp)
    if nargout > 0 or U._empty(close_fcn):
        from .dialogs import _ask_handler, _auto
        answer = _ask_handler(interp, "questdlg", {"question": message, "title": title, "buttons": options,
                                                   "default": default})
        if answer is None and st.wait_hook is not None:
            d = U.add_dialog(interp, "confirm", fig, None, **payload)
            answer = st.wait_hook(interp, None, d)
            U.remove_dialog(interp, d["id"])
        if answer is None or answer not in options:
            if answer is None:
                _auto(interp, "uiconfirm", f"{message} -> {default} (default answer)")
            answer = default
        if not U._empty(close_fcn):
            event = Struct.scalar({"Source": fig if fig is not None else np.zeros((0, 0)),
                                   "EventName": char("Close"), "DialogTitle": char(title),
                                   "SelectedOption": char(answer),
                                   "SelectedOptionIndex": np.array([[float(options.index(answer) + 1)]])})
            U.invoke(interp, fig, close_fcn, event)
        return char(answer)
    U.add_dialog(interp, "confirm", fig, close_fcn, **payload)
    return None


class ProgressDialog(MObject):
    """What uiprogressdlg returns."""

    is_handle = True
    _NAMES = {"value": "Value", "message": "Message", "title": "Title", "indeterminate": "Indeterminate",
              "cancelable": "Cancelable", "canceltext": "CancelText", "showpercentage": "ShowPercentage",
              "cancelrequested": "CancelRequested", "icon": "Icon", "interpreter": "Interpreter"}

    def __init__(self, interp, fig):
        self.interp = interp
        self.cancel_requested = False
        self.closed = False
        self.dialog = U.add_dialog(interp, "progress", fig, self, title="", message="", value=0.0,
                                   indeterminate=False, cancelable=False, cancelText="Cancel", percent=False)

    def class_name(self) -> str:
        return "matlab.ui.dialog.ProgressDialog"

    @property
    def _deleted(self) -> bool:
        return self.closed

    @_deleted.setter
    def _deleted(self, value: bool) -> None:
        if value:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            U.remove_dialog(self.interp, self.dialog["id"])

    def has_method(self, name: str) -> bool:
        return name in ("close", "delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name in ("close", "delete"):
            self.close()
            return []
        return [np.array([[not self.closed]])]

    def get_field(self, interp, name: str) -> Any:
        key = self._NAMES.get(name.lower())
        d = self.dialog
        if key == "Value":
            return np.array([[float(d["value"])]])
        if key in ("Message", "Title", "CancelText"):
            return char(d[{"Message": "message", "Title": "title", "CancelText": "cancelText"}[key]])
        if key in ("Indeterminate", "Cancelable", "ShowPercentage"):
            on = d[{"Indeterminate": "indeterminate", "Cancelable": "cancelable", "ShowPercentage": "percent"}[key]]
            return char("on" if on else "off")
        if key == "CancelRequested":
            return np.array([[bool(self.cancel_requested)]])
        if key is not None:
            return char("")
        raise MatlabError(f"Unrecognized property '{name}' for class 'ProgressDialog'.")

    def set_field(self, interp, name: str, value: Any):
        key = self._NAMES.get(name.lower())
        d = self.dialog
        if key == "Value":
            x = U._scalar(value, "Value")
            if not 0 <= x <= 1:
                raise MatlabError("The Value of a progress dialog must be between 0 and 1.")
            d["value"] = x
        elif key in ("Message", "Title", "CancelText"):
            d[{"Message": "message", "Title": "title", "CancelText": "cancelText"}[key]] = _message(value)
        elif key in ("Indeterminate", "Cancelable", "ShowPercentage"):
            d[{"Indeterminate": "indeterminate", "Cancelable": "cancelable", "ShowPercentage": "percent"}[key]] = \
                U._onoff(value)
        elif key == "CancelRequested":
            self.cancel_requested = U._onoff(value)
        elif key is None:
            raise MatlabError(f"Unrecognized property '{name}' for class 'ProgressDialog'.")
        return self

    def display_lines(self) -> list[str]:
        return ["  ProgressDialog with properties:", "", f"      Value: {self.dialog['value']:g}",
                f"    Message: '{self.dialog['message']}'", f"      Title: '{self.dialog['title']}'"]


@builtin("uiprogressdlg")
def _uiprogressdlg(interp, args, nargout):
    fig, rest = _dialog_figure(interp, args, "uiprogressdlg")
    d = ProgressDialog(interp, fig)
    try:
        for name, value in _pairs(rest, "uiprogressdlg"):
            d.set_field(interp, name, value)
    except MatlabError:
        d.close()
        raise
    return d


# -- data shared by the callbacks of one window --------------------

def _figure_arg(interp, v: Any, what: str) -> U.UIComponent:
    fig = None
    if isinstance(v, U.UIArray) and v.items:
        v = v.items[0]
    if U.is_component(v) or hasattr(v, "obj"):
        fig = U.figure_of(interp, v)
    if fig is None:
        raise MatlabError(f"{what}: the first input must be a figure or one of its components.")
    return fig


@builtin("guihandles")
def _guihandles(interp, args, nargout):
    st = U.state(interp)
    if args:
        fig = _figure_arg(interp, args[0], "guihandles")
    elif st.callbacks and st.callbacks[-1][1] is not None:
        fig = st.callbacks[-1][1]
    else:
        fig = _classic_parent(interp)
    found: dict[str, list] = {}

    def add(tag: str, h: Any) -> None:
        if tag and tag.isidentifier():
            found.setdefault(tag, []).append(h)

    if fig.mpl is not None:
        add(_txt(fig.mpl.extra.get("Tag", "")), fig.mpl)
    else:
        add(_txt(fig.get_prop("Tag")), fig)
    for node in fig.descendants():
        add(_txt(node.get_prop("Tag")), node)
    return Struct.scalar({tag: U.handle_list(hs) for tag, hs in found.items()})


@builtin("gcbo")
def _gcbo(interp, args, nargout):
    st = U.state(interp)
    if not st.callbacks:
        return (np.zeros((0, 0)), np.zeros((0, 0))) if nargout > 1 else np.zeros((0, 0))
    src, fig = st.callbacks[-1]
    figh = _fig_handle(fig)
    return (src, figh) if nargout > 1 else src


def _fig_handle(fig: Any) -> Any:
    if fig is None:
        return np.zeros((0, 0))
    return fig.mpl if fig.mpl is not None else fig


@builtin("gcbf")
def _gcbf(interp, args, nargout):
    st = U.state(interp)
    return _fig_handle(st.callbacks[-1][1]) if st.callbacks else np.zeros((0, 0))


@builtin("closereq")
def _closereq(interp, args, nargout):
    st = U.state(interp)
    fig = st.callbacks[-1][1] if st.callbacks else None
    if fig is None:
        return BUILTINS["close"](interp, [], 0)
    if fig.mpl is not None:
        from .plotting import _close_number, _fig_number
        n = _fig_number(interp, fig.mpl.obj)
        if n is not None:
            _close_number(interp, n)
        U.prune(interp)
    else:
        fig.destroy()
    return None


@builtin("uiwait")
def _uiwait(interp, args, nargout):
    """Wait until uiresume or until the window closes (where a front end can keep it open)."""
    st = U.state(interp)
    fig = None
    if args and (U.is_component(args[0]) or hasattr(args[0], "obj")):
        fig = U.figure_of(interp, args[0])
    elif not args:
        figs = U.visible_figures(interp)
        fig = figs[-1] if figs else None
    if fig is None or st.wait_hook is None:
        return None
    st.waiting.add(fig.uid)
    try:
        st.wait_hook(interp, fig, None)
    finally:
        st.waiting.discard(fig.uid)
    return None


@builtin("uiresume")
def _uiresume(interp, args, nargout):
    st = U.state(interp)
    if args and (U.is_component(args[0]) or hasattr(args[0], "obj")):
        fig = U.figure_of(interp, args[0])
        if fig is not None:
            st.waiting.discard(fig.uid)
    else:
        st.waiting.clear()
    return None


@builtin("waitfor")
def _waitfor(interp, args, nargout):
    if len(args) == 1:
        return _uiwait(interp, args, nargout)
    return None


# -- App Designer class apps --------------------

@builtin("registerApp")
def _register_app(interp, args, nargout):
    if len(args) > 1 and isinstance(args[1], U.UIComponent) and args[1].kind == "figure":
        args[1].props["RunningAppInstance"] = args[0]
    return None


@builtin("runStartupFcn")
def _run_startup(interp, args, nargout):
    if len(args) < 2:
        return None
    app, fn = args[0], args[1]
    if isinstance(fn, FunctionHandle) and not fn.is_anonymous and hasattr(app, "cls") and fn.name in app.cls.methods:
        app.cls.call(interp, fn.name, [app] + list(args[2:]), 0)
    elif isinstance(fn, FunctionHandle):
        interp.call_handle(fn, [app] + list(args[2:]), 0)
    return None


@builtin("createCallbackFcn")
def _create_callback(interp, args, nargout):
    if len(args) < 2 or not isinstance(args[1], FunctionHandle):
        raise MatlabError("createCallbackFcn needs the app and a method handle: createCallbackFcn(app, "
                          "@ButtonPushed, true).")
    app, fn = args[0], args[1]
    with_event = len(args) > 2 and U._onoff(args[2])
    if fn.is_anonymous or not hasattr(app, "cls") or fn.name not in app.cls.methods:
        return fn
    return U.AppCallback(app, fn.name, with_event)


# -- things that are not available --------------------

def _unavailable(name: str, message: str):
    def fn(interp, args, nargout):
        raise MatlabError(message)
    fn.__name__ = "_" + name
    builtin(name)(fn)


_unavailable("guide", "guide (the GUIDE layout editor and its .fig files) is not available in mylab. "
                      "Build the window in code with figure and uicontrol, or with uifigure.")
_unavailable("appdesigner", "App Designer is not available in mylab, and .mlapp files cannot be opened (they are "
                            "binary). Export the app as a .m class file in MATLAB ('Export to .m file'); mylab runs "
                            "that file. The mylab web app has its own simple app builder.")
for _n in ("uitoolbar", "uipushtool", "uitoggletool", "uicontextmenu", "uitree", "uitreenode", "uihtml",
           "uistyle", "uirangeslider", "dialog"):
    _unavailable(_n, f"{_n} is not supported by mylab's app windows.")


# -- generic handle functions also accept components --------------------

def _wrap_builtin(name: str):
    original = BUILTINS.get(name)

    def deco(fn):
        def wrapper(interp, args, nargout):
            res = fn(interp, list(args), nargout, original)
            return res
        wrapper.__name__ = fn.__name__
        BUILTINS[name] = wrapper
        return fn
    return deco


def _ui_target(v: Any) -> bool:
    return U.is_component(v) or (isinstance(v, U.UIArray) and bool(v.items)) or isinstance(v, ProgressDialog)


@_wrap_builtin("set")
def _set(interp, args, nargout, original):
    if not args or not _ui_target(args[0]):
        return original(interp, args, nargout)
    targets = U.components_of(args[0]) or [args[0]]
    rest = args[1:]
    if len(rest) == 2 and isinstance(rest[0], Cell) and isinstance(rest[1], Cell):
        pairs = list(zip([_txt(x) for x in rest[0].values_f()], rest[1].values_f()))
    else:
        pairs = _pairs(rest, "set")
    for h in targets:
        for name, value in pairs:
            h.set_field(interp, name, value)
    return None


def _all_props(interp, h: Any) -> Struct:
    out = {}
    if isinstance(h, U.UIComponent):
        names = list(U.KINDS[h.kind][3]) + ["Position", "Units", "Visible", "Enable", "Tag", "UserData", "Tooltip",
                                            "FontSize", "FontWeight", "FontColor", "BackgroundColor", "Parent",
                                            "Children", "Type"]
    else:
        names = list(U._AXES_UI_PROPS.values())
    for n in names:
        try:
            out[n] = h.get_field(interp, n)
        except MatlabError:
            continue
    return Struct.scalar(out)


@_wrap_builtin("get")
def _get(interp, args, nargout, original):
    if not args or not _ui_target(args[0]):
        return original(interp, args, nargout)
    targets = U.components_of(args[0]) or [args[0]]
    if len(args) == 1:
        if len(targets) != 1:
            raise MatlabError("get with several handles needs a property name.")
        s = _all_props(interp, targets[0])
        if nargout == 0:
            width = max((len(f) for f in s.fields), default=0)
            interp.out("".join(f"    {f.rjust(width)}: {U._show(s.get_field(0, f))}\n" for f in s.fields))
            return None
        return s
    prop = args[1]
    if isinstance(prop, (Cell, StringArray)):
        names = U._text_list(prop)
        vals = np.empty((len(targets), len(names)), dtype=object)
        for i, h in enumerate(targets):
            for j, n in enumerate(names):
                vals[i, j] = h.get_field(interp, n)
        return Cell(vals)
    name = _txt(prop)
    if len(targets) == 1:
        return targets[0].get_field(interp, name)
    return Cell.from_list([h.get_field(interp, name) for h in targets], column=True)


@_wrap_builtin("close")
def _close(interp, args, nargout, original):
    if args and isinstance(args[0], ProgressDialog):
        args[0].close()
        return None
    if args and _ui_target(args[0]):
        for h in U.components_of(args[0]):
            fig = h.figure()
            if fig is not None:
                U.close_figure(interp, fig)
        return None
    if args and _is_classic_figure(args[0]):
        node = U.state(interp).classic.get(id(args[0].obj))
        if node is not None and not node.deleted and node.children:
            U.close_figure(interp, node)
            return None
    res = original(interp, args, nargout)
    U.prune(interp)
    return res


@_wrap_builtin("figure")
def _figure_b(interp, args, nargout, original):
    if args and isinstance(args[0], U.UIComponent):
        if args[0].kind != "figure":
            raise MatlabError("figure(h): h must be a figure.")
        return args[0] if nargout > 0 else None
    return original(interp, args, nargout)


@_wrap_builtin("isprop")
def _isprop(interp, args, nargout, original):
    if len(args) > 1 and isinstance(args[0], U.UIComponent) and _is_text(args[1]):
        return np.array([[args[0].has_prop(_txt(args[1]))]])
    return original(interp, args, nargout)


@_wrap_builtin("ancestor")
def _ancestor(interp, args, nargout, original):
    if len(args) < 2 or not U.is_component(args[0]):
        return original(interp, args, nargout)
    types = {t.lower() for t in U._text_list(args[1])}
    node = args[0]
    while node is not None:
        t = "axes" if getattr(node, "is_ui_axes", False) else U.KINDS[node.kind][1]
        if t in types or (node.kind == "figure" and "figure" in types):
            return _fig_handle(node) if node.kind == "figure" else node
        node = node.parent
    return np.zeros((0, 0))


@_wrap_builtin("axes")
def _axes(interp, args, nargout, original):
    """axes(panel) / axes('Parent', panel, ...): an axes inside a component of an app window."""
    parent = None
    if args and isinstance(args[0], U.UIComponent):
        parent = args[0]
    else:
        for k in range(len(args) - 1):
            if _is_text(args[k]) and _txt(args[k]).lower() == "parent" and isinstance(args[k + 1], U.UIComponent):
                parent = args[k + 1]
    if parent is None:
        return original(interp, args, nargout)
    return _make_axes(interp, args, normalized=True, classic=True)


def _matches(interp, node: Any, pairs: list[tuple[str, Any]]) -> bool:
    for name, want in pairs:
        try:
            got = node.get_field(interp, name)
        except MatlabError:
            return False
        if _is_text(want) or isinstance(want, MString):
            if not _is_text(got) or _txt(got) != _txt(want):
                if not (name.lower() == "type" and _is_text(got) and _txt(got).lstrip("ui") == _txt(want)):
                    return False
        elif isinstance(want, np.ndarray) and isinstance(got, np.ndarray):
            if want.shape != got.shape or not np.array_equal(want, got):
                return False
        elif want is not got:
            return False
    return True


@_wrap_builtin("findobj")
def _findobj(interp, args, nargout, original):
    return _find(interp, args, nargout, original)


@_wrap_builtin("findall")
def _findall(interp, args, nargout, original):
    return _find(interp, args, nargout, original)


def _find(interp, args, nargout, original):
    roots: list | None = None
    rest = list(args)
    if rest and _ui_target(rest[0]):
        roots = U.components_of(rest.pop(0))
    elif rest and _is_classic_figure(rest[0]):
        node = U.state(interp).classic.get(id(rest[0].obj))
        if node is not None and not node.deleted and node.children:
            roots = list(node.children)
            rest.pop(0)
        else:
            return original(interp, args, nargout)
    elif rest and not _is_text(rest[0]):
        if not (isinstance(rest[0], np.ndarray) and rest[0].size == 1 and float(rest[0].flat[0]) == 0):
            return original(interp, args, nargout)
        rest.pop(0)
    if roots is None and not U.visible_figures(interp):
        return original(interp, args, nargout)
    pairs: list[tuple[str, Any]] = []
    k = 0
    flat = False
    while k < len(rest):
        a = rest[k]
        if _is_text(a) and _txt(a).lower() in ("flat",):
            flat = True
            k += 1
        elif _is_text(a) and _txt(a).lower() == "-depth" and k + 1 < len(rest):
            flat = flat or float(np.asarray(rest[k + 1]).flat[0]) == 0
            k += 2
        elif _is_text(a) and _txt(a).startswith("-"):
            k += 1
        elif _is_text(a) and k + 1 < len(rest):
            pairs.append((_txt(a), rest[k + 1]))
            k += 2
        else:
            raise MatlabError("findobj: expected Name, Value pairs such as findobj(fig, 'Tag', 'ok').")
    if roots is None:
        roots = list(U.visible_figures(interp))
    found = []
    for r in roots:
        for node in [r] + ([] if flat else r.descendants()):
            if node not in found and _matches(interp, node, pairs):
                found.append(node)
    if not found and (not args or _is_text(args[0]) or _is_classic_figure(args[0])
                      or isinstance(args[0], np.ndarray)):
        return original(interp, args, nargout)
    found = [(_fig_handle(n) if isinstance(n, U.UIComponent) and n.mpl is not None else n) for n in found]
    if any(not U.is_component(n) for n in found):
        return found[0] if len(found) == 1 else original(interp, args, nargout)
    return U.handle_list(found, column=True)


@_wrap_builtin("guidata")
def _guidata(interp, args, nargout, original):
    if not args or not (_ui_target(args[0])):
        return original(interp, args, nargout)
    fig = _figure_arg(interp, args[0], "guidata")
    if fig.mpl is not None:  # a classic window: the data lives with the figure
        return original(interp, [fig.mpl] + args[1:], nargout)
    st = U.state(interp)
    if len(args) > 1:
        st.guidata[id(fig)] = args[1]
        return None
    return st.guidata.get(id(fig), np.zeros((0, 0)))


def _component_appdata(h: Any) -> dict:
    return h.props.setdefault("_appdata", {})


@_wrap_builtin("setappdata")
def _setappdata(interp, args, nargout, original):
    if len(args) == 3 and isinstance(args[0], U.UIComponent):
        target = args[0].mpl if args[0].mpl is not None else None
        if target is not None:
            return original(interp, [target] + args[1:], nargout)
        _component_appdata(args[0])[_txt(args[1])] = args[2]
        return None
    return original(interp, args, nargout)


@_wrap_builtin("getappdata")
def _getappdata(interp, args, nargout, original):
    if args and isinstance(args[0], U.UIComponent):
        store = _component_appdata(args[0])
        if len(args) == 1:
            return Struct.scalar(dict(store))
        return store.get(_txt(args[1]), np.zeros((0, 0)))
    return original(interp, args, nargout)


@_wrap_builtin("isappdata")
def _isappdata(interp, args, nargout, original):
    if len(args) > 1 and isinstance(args[0], U.UIComponent):
        return np.array([[_txt(args[1]) in _component_appdata(args[0])]])
    return original(interp, args, nargout)


@_wrap_builtin("rmappdata")
def _rmappdata(interp, args, nargout, original):
    if len(args) > 1 and isinstance(args[0], U.UIComponent):
        _component_appdata(args[0]).pop(_txt(args[1]), None)
        return None
    return original(interp, args, nargout)
