"""App windows: a tree of UI components, and the protocol a front end uses to show and drive it.

The functions ``uifigure``, ``uibutton``, ``uicontrol`` ... (see
``mylab/builtins/uicomponents.py``) build a tree of :class:`UIComponent`
objects. Nothing is drawn by the engine. A front end (the browser app, the
local server behind ``mylab app.m``, a test) talks to the tree with two calls:

``snapshot(interp)``
    The whole tree as plain JSON data::

        {"figures": [node, ...], "dialogs": [dialog, ...]}
        node   = {"id": 7, "type": "button", "props": {...}, "children": [node, ...]}
        dialog = {"id": "d3", "kind": "alert" | "confirm" | "progress", ...}

``dispatch(interp, event)``
    Apply one user action and run the callbacks it triggers. ``event`` is
    ``{"id": <node id>, "event": <name>, ...}``:

    ==========  ==================================================================
    ``push``    a button, menu item, image or link was activated
    ``value``   a value was committed: ``value`` is text (edit field, text area),
                a number (slider, spinner, numeric field), true/false (check box,
                radio/toggle/state button), a 0-based item number (drop-down,
                switch) or a list of them (list box), ``"YYYY-MM-DD"`` (date
                picker). With ``"changing": true`` the value is only passed to
                ``ValueChangingFcn`` and not stored.
    ``select``  ``value`` is the 0-based number of the tab that was opened; for a
                table, ``row`` and ``col`` (0-based) of the selected cell
    ``cell``    a table cell was edited: ``row``, ``col`` (0-based), ``value``
    ``close``   the window's close button
    ``resize``  the window changed size: ``width``, ``height`` (pixels)
    ``size``    an axes is shown at ``width`` x ``height`` pixels (no callback)
    ``dialog``  a dialog button was pressed: ``id`` is the dialog id, ``value``
                the button text
    ==========  ==================================================================

    The answer is ``{"error": text or None, "update": update}`` where ``update``
    is either ``{"full": true, "figures": [...], "dialogs": [...]}`` (components
    were added, removed or moved) or ``{"full": false, "patch": {id: {changed
    props}}, "dialogs": [...]}``.

Property names in ``props`` are the same for classic (``uicontrol``) and modern
components: a ``uicontrol('Style', 'popupmenu')`` is sent as a ``dropdown``.
Positions are ``pos`` = [left, bottom, width, height] in pixels from the
bottom-left corner of the parent, ``npos`` (the same as fractions of the
parent), or ``grid`` = [row1, row2, col1, col2] (1-based, inclusive) inside a
grid layout. Axes are sent as PNG images (``image``, a data URL).
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np

from .errors import MatlabError
from .objects import MObject
from .values import Cell, FunctionHandle, MString, StringArray, Struct, char, is_char, to_str

MLAPP_MESSAGE = (".mlapp files cannot be opened: they are binary App Designer files. In MATLAB, open the app in "
                 "App Designer and choose Save > Export to .m file; mylab runs the exported .m class file.")

# -- small value helpers --------------------


def _is_text(v: Any) -> bool:
    return isinstance(v, MString) or (is_char(v) and (v.ndim < 2 or v.shape[0] <= 1))


def _empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, np.ndarray):
        return v.size == 0
    if isinstance(v, Cell):
        return v.data.size == 0
    if isinstance(v, StringArray):
        return v.data.size == 0
    return False


def _txt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, MString):
        return str(v)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        if is_char(v):
            if v.ndim == 2 and v.shape[0] > 1:
                return "\n".join("".join(row).rstrip() for row in v)
            return to_str(v)
        if v.size == 1:
            return _fmt_number(float(np.real(v.flat[0])))
        return " ".join(_fmt_number(float(np.real(x))) for x in v.flatten(order="F"))
    if isinstance(v, (Cell, StringArray)):
        return "\n".join(_text_list(v))
    return str(v)


def _fmt_number(x: float) -> str:
    if x != x:
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.5g}"


def _text_list(v: Any) -> list[str]:
    """Items of a cell array, string array, char matrix or single text."""
    if _empty(v):
        return []
    if isinstance(v, Cell):
        return [_txt(x) for x in v.values_f()]
    if isinstance(v, StringArray):
        return [str(x) for x in v.values_f()]
    if isinstance(v, MString):
        return [str(v)]
    if isinstance(v, np.ndarray) and is_char(v):
        if v.ndim == 2 and v.shape[0] > 1:
            return ["".join(row).rstrip() for row in v]
        return [to_str(v)]
    if isinstance(v, np.ndarray):
        return [_fmt_number(float(np.real(x))) for x in v.flatten(order="F")]
    return [str(v)]


def _onoff(v: Any) -> bool:
    if _is_text(v):
        s = _txt(v).lower()
        if s in ("on", "true", "yes"):
            return True
        if s in ("off", "false", "no", "inactive"):
            return False
        raise MatlabError(f"Expected 'on' or 'off', not '{_txt(v)}'.")
    if isinstance(v, np.ndarray) and v.size == 1:
        return bool(v.flat[0])
    if isinstance(v, bool):
        return v
    raise MatlabError("Expected 'on', 'off', true or false.")


_COLOR_NAMES = {
    "r": (1, 0, 0), "red": (1, 0, 0), "g": (0, 1, 0), "green": (0, 1, 0), "b": (0, 0, 1), "blue": (0, 0, 1),
    "c": (0, 1, 1), "cyan": (0, 1, 1), "m": (1, 0, 1), "magenta": (1, 0, 1), "y": (1, 1, 0),
    "yellow": (1, 1, 0), "k": (0, 0, 0), "black": (0, 0, 0), "w": (1, 1, 1), "white": (1, 1, 1),
}


def _color(v: Any) -> np.ndarray:
    if _is_text(v):
        s = _txt(v).strip().lower()
        if s in _COLOR_NAMES:
            return np.array([list(map(float, _COLOR_NAMES[s]))])
        if s == "none":
            return np.array([[np.nan, np.nan, np.nan]])
        if s.startswith("#") and len(s) in (4, 7):
            h = s[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            try:
                return np.array([[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]])
            except ValueError:
                pass
        raise MatlabError(f"'{_txt(v)}' is not a color. Use a name such as 'red', a hex code such as "
                          "'#0072BD' or three numbers between 0 and 1.")
    if isinstance(v, np.ndarray) and v.size in (3, 4) and v.dtype.kind in "biuf":
        rgb = np.asarray(v, dtype=float).flatten()[:3]
        if v.dtype.kind in "iu" and rgb.max() > 1:
            rgb = rgb / 255
        return np.clip(rgb, 0, 1).reshape(1, 3)
    raise MatlabError("A color must be a name, a hex code or three numbers between 0 and 1.")


def _css_color(rgb: Any) -> str | None:
    a = np.asarray(rgb, dtype=float).flatten()
    if a.size < 3 or np.isnan(a[:3]).any():
        return None
    return "#" + "".join(f"{int(round(float(x) * 255)):02x}" for x in a[:3])


def _num(v: Any, what: str = "value") -> np.ndarray:
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and not is_char(v):
        return np.atleast_2d(np.asarray(v, dtype=float))
    if isinstance(v, (int, float)):
        return np.array([[float(v)]])
    raise MatlabError(f"The {what} must be a number.")


def _scalar(v: Any, what: str = "value") -> float:
    a = _num(v, what)
    if a.size != 1:
        raise MatlabError(f"The {what} must be a single number.")
    return float(a.flat[0])


def _row(vals) -> np.ndarray:
    return np.array([list(map(float, vals))]) if len(vals) else np.zeros((0, 0))


def _bool(b: bool) -> np.ndarray:
    return np.array([[bool(b)]])


def _key(v: Any) -> Any:
    """A hashable stand-in used to compare item values."""
    if _is_text(v):
        return ("t", _txt(v))
    if isinstance(v, np.ndarray) and v.size == 1:
        return ("n", float(np.real(v.flat[0])))
    if isinstance(v, np.ndarray):
        return ("a", v.shape, tuple(np.asarray(v, dtype=float).flatten().tolist()))
    return ("o", id(v))


def _json_num(x: float) -> Any:
    x = float(x)
    if x != x:
        return None
    if x == float("inf"):
        return 1e308
    if x == float("-inf"):
        return -1e308
    return x


# -- property tables --------------------
# kind codes: t text, n numbers, o on/off, b logical, c color, f callback, a anything,
#             l list of text, s grid sizes, v value (handled per component)

_COMMON: dict[str, tuple] = {
    "Tag": ("t", ""), "UserData": ("a", None), "Visible": ("o", True), "Enable": ("o", True),
    "Tooltip": ("a", ""), "Position": ("n", None), "Units": ("t", "pixels"),
    "FontSize": ("n", 12.0), "FontWeight": ("t", "normal"), "FontAngle": ("t", "normal"),
    "FontName": ("t", "Helvetica"), "FontColor": ("c", (0, 0, 0)), "BackgroundColor": ("c", (0.94, 0.94, 0.94)),
    "CreateFcn": ("f", None), "DeleteFcn": ("f", None), "ButtonDownFcn": ("f", None),
    "HandleVisibility": ("t", "on"), "ContextMenu": ("a", None), "BusyAction": ("t", "queue"),
    "Interruptible": ("o", True), "FontUnits": ("t", "pixels"), "HorizontalAlignment": ("t", "left"),
    "VerticalAlignment": ("t", "center"), "WordWrap": ("o", False), "Interpreter": ("t", "none"),
    "Scrollable": ("o", False), "AutoResizeChildren": ("o", True), "SizeChangedFcn": ("f", None),
    "Icon": ("a", ""), "IconAlignment": ("t", "left"), "BorderType": ("t", "line"),
    "BorderWidth": ("n", 1.0), "BorderColor": ("c", (0.49, 0.49, 0.49)), "Placeholder": ("t", ""),
    "Editable": ("o", True), "Orientation": ("t", "horizontal"), "KeyPressFcn": ("f", None),
    "KeyReleaseFcn": ("f", None), "Clipping": ("o", True), "Selected": ("o", False),
    "SelectionHighlight": ("o", True), "Layer": ("t", "bottom"), "Opacity": ("n", 1.0),
    "HighlightColor": ("c", (1, 1, 1)), "ShadowColor": ("c", (0.7, 0.7, 0.7)), "TitlePosition": ("t", "lefttop"),
    "FontSmoothing": ("o", True), "HitTest": ("o", True), "PickableParts": ("t", "visible"),
    "BeingDeleted": ("o", False), "Internal": ("o", False),
}

_ALIASES = {
    "tooltipstring": "Tooltip", "foregroundcolor": "FontColor", "innerposition": "Position",
    "outerposition": "Position", "uicontextmenu": "ContextMenu", "selectionchangefcn": "SelectionChangedFcn",
    "resizefcn": "SizeChangedFcn",
}

_TICKS = {
    "MajorTicks": ("n", None), "MajorTickLabels": ("l", None), "MinorTicks": ("n", None),
    "MajorTicksMode": ("t", "auto"), "MajorTickLabelsMode": ("t", "auto"), "MinorTicksMode": ("t", "auto"),
}
_VALUE_FCNS = {"ValueChangedFcn": ("f", None), "ValueChangingFcn": ("f", None)}
_NUMBER_FIELD = {
    "Limits": ("n", (-np.inf, np.inf)), "RoundFractionalValues": ("o", False),
    "ValueDisplayFormat": ("t", "%11.4g"), "LowerLimitInclusive": ("o", True),
    "UpperLimitInclusive": ("o", True), "AllowEmpty": ("o", False),
}
_FIGURE_FCNS = {name: ("f", None) for name in (
    "CloseRequestFcn", "WindowButtonDownFcn", "WindowButtonMotionFcn", "WindowButtonUpFcn",
    "WindowKeyPressFcn", "WindowKeyReleaseFcn", "WindowScrollWheelFcn")}

# kind -> (class name, Type, default Position, own properties)
KINDS: dict[str, tuple] = {
    "figure": ("matlab.ui.Figure", "figure", (100, 100, 640, 480), {
        "Name": ("t", ""), "Color": ("c", (0.94, 0.94, 0.94)), "Resize": ("o", True),
        "WindowState": ("t", "normal"), "WindowStyle": ("t", "normal"), "NumberTitle": ("o", False),
        "MenuBar": ("t", "none"), "ToolBar": ("t", "none"), "Pointer": ("t", "arrow"),
        "RunningAppInstance": ("a", None), "IntegerHandle": ("o", False), "DockControls": ("o", False),
        "Theme": ("a", None), "ThemeMode": ("t", "auto"), "Colormap": ("a", None), **_FIGURE_FCNS}),
    "gridlayout": ("matlab.ui.container.GridLayout", "uigridlayout", (1, 1, 100, 100), {
        "RowHeight": ("s", ("1x", "1x")), "ColumnWidth": ("s", ("1x", "1x")),
        "Padding": ("n", (10, 10, 10, 10)), "RowSpacing": ("n", 10.0), "ColumnSpacing": ("n", 10.0)}),
    "panel": ("matlab.ui.container.Panel", "uipanel", (20, 20, 260, 221), {"Title": ("t", "")}),
    "buttongroup": ("matlab.ui.container.ButtonGroup", "uibuttongroup", (20, 20, 123, 106), {
        "Title": ("t", ""), "SelectionChangedFcn": ("f", None)}),
    "tabgroup": ("matlab.ui.container.TabGroup", "uitabgroup", (20, 20, 250, 250), {
        "TabLocation": ("t", "top"), "SelectionChangedFcn": ("f", None)}),
    "tab": ("matlab.ui.container.Tab", "uitab", (0, 0, 0, 0), {"Title": ("t", "")}),
    "label": ("matlab.ui.control.Label", "uilabel", (100, 100, 31, 22), {"Text": ("a", "Label")}),
    "button": ("matlab.ui.control.Button", "uibutton", (100, 100, 100, 22), {
        "Text": ("a", "Button"), "ButtonPushedFcn": ("f", None)}),
    "statebutton": ("matlab.ui.control.StateButton", "uistatebutton", (100, 100, 100, 22), {
        "Text": ("a", "State Button"), "Value": ("v", False), "ValueChangedFcn": ("f", None)}),
    "editfield": ("matlab.ui.control.EditField", "uieditfield", (100, 100, 100, 22), {
        "Value": ("v", ""), "CharacterLimits": ("n", (0, np.inf)), "InputType": ("t", "text"), **_VALUE_FCNS}),
    "numericeditfield": ("matlab.ui.control.NumericEditField", "uinumericeditfield", (100, 100, 100, 22), {
        "Value": ("v", 0.0), **_NUMBER_FIELD, **_VALUE_FCNS}),
    "textarea": ("matlab.ui.control.TextArea", "uitextarea", (100, 100, 150, 60), {
        "Value": ("v", ""), **_VALUE_FCNS}),
    "dropdown": ("matlab.ui.control.DropDown", "uidropdown", (100, 100, 100, 22), {
        "Items": ("l", ("Option 1", "Option 2", "Option 3", "Option 4")), "ItemsData": ("a", None),
        "Value": ("v", None), "DropDownOpeningFcn": ("f", None), "ValueChangedFcn": ("f", None)}),
    "listbox": ("matlab.ui.control.ListBox", "uilistbox", (100, 100, 100, 74), {
        "Items": ("l", ("Item 1", "Item 2", "Item 3", "Item 4")), "ItemsData": ("a", None),
        "Value": ("v", None), "Multiselect": ("o", False), "ValueChangedFcn": ("f", None),
        "ClickedFcn": ("f", None), "DoubleClickedFcn": ("f", None)}),
    "checkbox": ("matlab.ui.control.CheckBox", "uicheckbox", (100, 100, 84, 22), {
        "Text": ("a", "Check Box"), "Value": ("v", False), "ValueChangedFcn": ("f", None)}),
    "radiobutton": ("matlab.ui.control.RadioButton", "uiradiobutton", (10, 10, 91, 22), {
        "Text": ("a", "Radio Button"), "Value": ("v", False)}),
    "togglebutton": ("matlab.ui.control.ToggleButton", "uitogglebutton", (10, 10, 100, 22), {
        "Text": ("a", "Toggle Button"), "Value": ("v", False)}),
    "switch": ("matlab.ui.control.Switch", "uiswitch", (100, 100, 45, 20), {
        "Items": ("l", ("Off", "On")), "ItemsData": ("a", None), "Value": ("v", None),
        "ValueChangedFcn": ("f", None)}),
    "slider": ("matlab.ui.control.Slider", "uislider", (100, 100, 150, 3), {
        "Value": ("v", 0.0), "Limits": ("n", (0, 100)), "Step": ("n", None), **_TICKS, **_VALUE_FCNS}),
    "spinner": ("matlab.ui.control.Spinner", "uispinner", (100, 100, 100, 22), {
        "Value": ("v", 0.0), "Step": ("n", 1.0), **_NUMBER_FIELD, **_VALUE_FCNS}),
    "knob": ("matlab.ui.control.Knob", "uiknob", (100, 100, 60, 60), {
        "Value": ("v", 0.0), "Limits": ("n", (0, 100)), **_TICKS, **_VALUE_FCNS}),
    "discreteknob": ("matlab.ui.control.DiscreteKnob", "uidiscreteknob", (100, 100, 60, 60), {
        "Items": ("l", ("Off", "Low", "Medium", "High")), "ItemsData": ("a", None), "Value": ("v", None),
        "ValueChangedFcn": ("f", None)}),
    "gauge": ("matlab.ui.control.Gauge", "uigauge", (100, 100, 120, 120), {
        "Value": ("v", 0.0), "Limits": ("n", (0, 100)), "ScaleColors": ("a", None),
        "ScaleColorLimits": ("a", None), "ScaleDirection": ("t", "clockwise"), **_TICKS}),
    "lamp": ("matlab.ui.control.Lamp", "uilamp", (100, 100, 20, 20), {"Color": ("c", (0, 1, 0))}),
    "table": ("matlab.ui.control.Table", "uitable", (20, 20, 300, 300), {
        "Data": ("a", None), "ColumnName": ("a", "numbered"), "RowName": ("a", "numbered"),
        "ColumnEditable": ("a", None), "ColumnWidth": ("a", "auto"), "ColumnFormat": ("a", None),
        "ColumnSortable": ("a", None), "RowStriping": ("o", True), "Selection": ("a", None),
        "SelectionType": ("t", "cell"), "Multiselect": ("o", True), "CellEditCallback": ("f", None),
        "CellSelectionCallback": ("f", None), "SelectionChangedFcn": ("f", None),
        "DisplayDataChangedFcn": ("f", None), "ClickedFcn": ("f", None), "DoubleClickedFcn": ("f", None),
        "StyleConfigurations": ("a", None)}),
    "image": ("matlab.ui.control.Image", "uiimage", (100, 100, 100, 100), {
        "ImageSource": ("a", ""), "ScaleMethod": ("t", "fit"), "ImageClickedFcn": ("f", None),
        "URL": ("t", ""), "AltText": ("t", "")}),
    "datepicker": ("matlab.ui.control.DatePicker", "uidatepicker", (100, 100, 150, 22), {
        "Value": ("v", None), "DisplayFormat": ("t", "dd-MMM-uuuu"), "Limits": ("a", None),
        "DisabledDates": ("a", None), "DisabledDaysOfWeek": ("a", None), "ValueChangedFcn": ("f", None)}),
    "menu": ("matlab.ui.container.Menu", "uimenu", (0, 0, 0, 0), {
        "Text": ("a", ""), "MenuSelectedFcn": ("f", None), "Separator": ("o", False), "Checked": ("o", False),
        "Accelerator": ("t", "")}),
    "hyperlink": ("matlab.ui.control.Hyperlink", "uihyperlink", (100, 100, 70, 22), {
        "Text": ("a", "Hyperlink"), "URL": ("t", ""), "HyperlinkClickedFcn": ("f", None),
        "VisitedColor": ("c", (0.4, 0.2, 0.6))}),
    "uicontrol": ("matlab.ui.control.UIControl", "uicontrol", (20, 20, 60, 20), {
        "Style": ("t", "pushbutton"), "String": ("a", ""), "Value": ("v", 0.0), "Min": ("n", 0.0),
        "Max": ("n", 1.0), "SliderStep": ("n", (0.01, 0.1)), "Callback": ("f", None),
        "ListboxTop": ("n", 1.0), "CData": ("a", None), "Extent": ("n", (0, 0, 60, 20))}),
}

_MENU_ALIASES = {"label": "Text", "callback": "MenuSelectedFcn"}
_CONTAINERS = {"figure", "gridlayout", "panel", "buttongroup", "tab"}
_CHOICE = {"dropdown", "listbox", "switch", "discreteknob"}
_TOGGLE = {"checkbox", "radiobutton", "togglebutton", "statebutton"}
_CLASSIC_TYPES = {
    "pushbutton": "button", "togglebutton": "statebutton", "checkbox": "checkbox",
    "radiobutton": "radiobutton", "edit": "editfield", "text": "label", "slider": "slider",
    "popupmenu": "dropdown", "popup": "dropdown", "listbox": "listbox", "frame": "panel",
}
_UNIT_PIXELS = {"pixels": 1.0, "points": 4 / 3, "inches": 96.0, "centimeters": 96 / 2.54}
_CHAR_PIXELS = (6.0, 13.0)  # size of one 'characters' unit


def _lookup(kind: str, name: str) -> tuple[str, tuple] | None:
    low = name.lower()
    if kind == "menu" and low in _MENU_ALIASES:
        low = _MENU_ALIASES[low].lower()
    own = KINDS[kind][3]
    for table in (own, _COMMON):
        for key, spec in table.items():
            if key.lower() == low:
                return key, spec
    alias = _ALIASES.get(low)
    if alias:
        return _lookup(kind, alias)
    return None


def _default_value(code: str, default: Any) -> Any:
    if code == "t":
        return char(default or "")
    if code == "o":
        return char("on" if default else "off")
    if code == "c":
        return _row(default)
    if code == "n":
        if default is None:
            return np.zeros((0, 0))
        return _row(default) if isinstance(default, tuple) else np.array([[float(default)]])
    if code == "l":
        return Cell.from_list([char(s) for s in default]) if default else Cell.empty(1, 0)
    if code == "s":
        return Cell.from_list([char(s) if isinstance(s, str) else np.array([[float(s)]]) for s in default])
    if code == "f":
        return char("")
    if default is None:
        return np.zeros((0, 0))
    if isinstance(default, str):
        return char(default)
    return default


def _normalize(code: str, name: str, value: Any) -> Any:
    if code == "t":
        if isinstance(value, MString):
            return char(str(value))
        if _is_text(value) or _empty(value):
            return char(_txt(value))
        raise MatlabError(f"'{name}' must be text.")
    if code == "o":
        return char("on" if _onoff(value) else "off")
    if code == "c":
        return _color(value)
    if code == "n":
        return _num(value, name)
    if code == "l":
        if isinstance(value, (Cell, StringArray)) or _is_text(value) or _empty(value):
            items = _text_list(value)
            return Cell.from_list([char(s) for s in items]) if items else Cell.empty(1, 0)
        raise MatlabError(f"'{name}' must be a cell array of text, e.g. {{'One', 'Two'}}.")
    if code == "s":
        if _is_text(value):
            value = Cell.from_list([value])
        if isinstance(value, StringArray):
            value = Cell.from_list([char(str(s)) for s in value.values_f()])
        if isinstance(value, np.ndarray):
            value = Cell.from_list([np.array([[float(x)]]) for x in value.flatten(order="F")])
        if not isinstance(value, Cell):
            raise MatlabError(f"'{name}' must be a cell array such as {{'1x', 'fit', 100}}.")
        out = []
        for x in value.values_f():
            if _is_text(x):
                s = _txt(x).strip().lower()
                if s != "fit" and not (s.endswith("x") and _is_float(s[:-1])):
                    raise MatlabError(f"'{name}' entries are numbers of pixels, 'fit' or a weight such as '1x'.")
                out.append(char(s))
            else:
                out.append(np.array([[_scalar(x, name)]]))
        return Cell.from_list(out)
    if code == "f":
        if isinstance(value, (FunctionHandle, Cell, AppCallback)) or _is_text(value) or _empty(value):
            return value
        raise MatlabError(f"'{name}' must be a function handle, e.g. @(src, event) disp('pressed').")
    return value


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# -- state --------------------

class UIState:
    def __init__(self) -> None:
        self.figures: list[UIComponent] = []
        self.nodes: dict[int, Any] = {}
        self.next_id = 1
        self.callbacks: list[tuple[Any, Any]] = []   # (object, figure) of the callbacks that are running
        self.dialogs: list[dict] = []
        self.dialog_objects: dict[str, Any] = {}
        self.sent: dict | None = None
        self.classic: dict[int, UIComponent] = {}     # id(matplotlib figure) -> its window node
        self.guidata: dict[int, Any] = {}             # id(figure node) -> data
        self.waiting: set[int] = set()                # figures inside uiwait
        self.wait_hook = None                         # callable(interp, figure node) run by uiwait
        self.answers: dict[str, str] = {}             # dialog id -> button pressed (for a waiting uiconfirm)


def state(interp) -> UIState:
    st = interp.state.get("ui")
    if st is None:
        st = UIState()
        interp.state["ui"] = st
    return st


def reset(interp) -> None:
    st = interp.state.get("ui")
    if st is None:
        return
    for fig in list(st.figures):
        fig.destroy(run_callbacks=False)
    interp.state.pop("ui", None)


class AppCallback(MObject):
    """What createCallbackFcn(app, @method, true) returns: calls a method of an app object."""

    def __init__(self, app: Any, name: str, with_event: bool):
        self.app = app
        self.name = name
        self.with_event = with_event

    def class_name(self) -> str:
        return "function_handle"

    def display_lines(self) -> list[str]:
        return [f"    @(source, event) {self.name}(app, event)"]


# -- components --------------------

class UIComponent(MObject):
    """One UI component (window, container or control). A handle object."""

    is_handle = True

    def __init__(self, interp, kind: str, parent: "UIComponent | None" = None):
        st = state(interp)
        self.interp = interp
        self.kind = kind
        self.uid = st.next_id
        st.next_id += 1
        st.nodes[self.uid] = self
        self.props: dict[str, Any] = {}
        self.parent: UIComponent | None = None
        self.children: list[Any] = []
        self.deleted = False
        self.cell: list[int] | None = None   # [row1, row2, col1, col2] in a grid layout
        self.style = ""                       # variant: 'numeric', 'state', 'rocker', 'linear', ...
        self.mpl = None                       # classic figure windows: the GraphicsHandle of the figure
        if parent is not None:
            self.attach(parent)
        elif kind == "figure":
            st.figures.append(self)

    # -- identity --------------------------------------------------------------------------------
    def class_name(self) -> str:
        return KINDS[self.kind][0]

    def isa_class(self, name: str) -> bool:
        cls = self.class_name()
        return name in (cls, cls.rsplit(".", 1)[-1], "handle", "matlab.graphics.Graphics",
                        "matlab.ui.control.internal.model.ComponentModel")

    @property
    def _deleted(self) -> bool:  # what the generic isvalid/isgraphics built-ins look at
        return self.deleted

    @_deleted.setter
    def _deleted(self, value: bool) -> None:
        if value and not self.deleted:
            self.destroy()

    def summary_text(self, in_cell: bool) -> str:
        return f"[1\u00d71 {self.short()}]" if not self.deleted else f"[deleted {self.short()}]"

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid", "focus", "scroll", "addStyle", "removeStyle", "isprop")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete":
            for a in args:
                for h in components_of(a):
                    h.destroy()
            return []
        if name == "isvalid":
            return [_bool(not self.deleted)]
        if name == "isprop":
            return [_bool(len(args) > 1 and _is_text(args[1]) and self.has_prop(_txt(args[1])))]
        if name in ("focus", "scroll", "addStyle", "removeStyle"):
            return []
        return super().call_method(interp, name, args, nargout)

    def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
        if op in ("==", "~="):
            same = left is right
            return _bool(same if op == "==" else not same)
        return NotImplemented

    # -- tree ------------------------------------------------------------------------------------
    def attach(self, parent: "UIComponent") -> None:
        if parent.deleted:
            raise MatlabError("The parent has been deleted.")
        if self.kind == "tab" and parent.kind != "tabgroup":
            raise MatlabError("A tab must be created inside a tab group: uitab(uitabgroup(fig)).")
        if parent.kind == "tabgroup" and self.kind != "tab":
            raise MatlabError("A tab group can only hold tabs; put components into a uitab.")
        if parent.kind not in _CONTAINERS and parent.kind != "tabgroup" and not (
                self.kind == "menu" and parent.kind == "menu"):
            raise MatlabError(f"A {parent.short()} cannot hold other components. Use a figure, panel, tab "
                              "or grid layout as the parent.")
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
        self.parent = parent
        parent.children.append(self)
        self.cell = None
        if parent.kind == "gridlayout":
            parent.place(self)

    def figure(self) -> "UIComponent | None":
        node: Any = self
        while node is not None and node.kind != "figure":
            node = node.parent
        return node

    def descendants(self) -> list:
        out = []
        for c in self.children:
            out.append(c)
            out.extend(c.descendants())
        return out

    def short(self) -> str:
        return self.class_name().rsplit(".", 1)[-1]

    def destroy(self, run_callbacks: bool = True) -> None:
        if self.deleted:
            return
        st = state(self.interp)
        if run_callbacks:
            cb = self.props.get("DeleteFcn")
            if not _empty(cb):
                try:
                    invoke(self.interp, self, cb, make_event(self, "ObjectBeingDestroyed"))
                except MatlabError:
                    pass
        for c in list(self.children):
            c.destroy(run_callbacks)
        self.deleted = True
        st.nodes.pop(self.uid, None)
        if self.parent is not None and self in self.parent.children:
            self.parent.children.remove(self)
        if self in st.figures:
            st.figures.remove(self)
        if self.kind == "figure":
            st.guidata.pop(id(self), None)
            st.waiting.discard(self.uid)
            if self.mpl is not None:
                st.classic.pop(id(self.mpl.obj), None)
            for d in list(st.dialogs):
                if d.get("figure") == self.uid:
                    st.dialogs.remove(d)

    # -- grid layout -----------------------------------------------------------------------------
    def grid_sizes(self, which: str) -> list:
        return list(self.get_prop(which).values_f())

    def place(self, child: Any) -> None:
        """Give a new child of this grid layout the next free cell."""
        cols = max(len(self.grid_sizes("ColumnWidth")), 1)
        taken = set()
        for c in self.children:
            if c is not child and c.cell:
                for r in range(c.cell[0], c.cell[1] + 1):
                    for k in range(c.cell[2], c.cell[3] + 1):
                        taken.add((r, k))
        n = 0
        while (n // cols + 1, n % cols + 1) in taken:
            n += 1
        child.cell = [n // cols + 1, n // cols + 1, n % cols + 1, n % cols + 1]
        self.grow(child.cell)

    def grow(self, cell: list[int]) -> None:
        for which, need in (("RowHeight", cell[1]), ("ColumnWidth", cell[3])):
            sizes = self.grid_sizes(which)
            if need > len(sizes):
                sizes += [char("1x")] * (need - len(sizes))
                self.props[which] = Cell.from_list(sizes)

    # -- properties ------------------------------------------------------------------------------
    def has_prop(self, name: str) -> bool:
        return _lookup(self.kind, name) is not None or name.lower() in (
            "parent", "children", "type", "layout", "selectedtab", "selectedobject", "buttons",
            "displaydata", "valueindex", "currentobject", "number")

    def get_field(self, interp, name: str) -> Any:
        return self.get_prop(name)

    def set_field(self, interp, name: str, value: Any) -> "UIComponent":
        self.set_prop(name, value)
        return self

    def _check(self) -> None:
        if self.deleted:
            raise MatlabError("Invalid or deleted object.")

    def get_prop(self, name: str) -> Any:
        low = name.lower()
        if low == "type":
            return char(KINDS[self.kind][1])
        self._check()
        if self.mpl is not None and low not in ("children", "type"):
            return self.mpl.get_field(self.interp, name)
        if low == "parent":
            if self.parent is None:
                return np.zeros((0, 0))
            return self.parent.mpl if self.parent.mpl is not None else self.parent
        if low == "children":
            kids = [c for c in self.children if c.kind != "menu" or self.kind == "menu"]
            return handle_list(list(reversed(kids)) if self.kind != "tabgroup" else kids, column=True)
        if low == "layout":
            cell = self.cell or [1, 1, 1, 1]
            span = lambda a, b: np.array([[float(a)]]) if a == b else _row((a, b))
            return Struct.scalar({"Row": span(cell[0], cell[1]), "Column": span(cell[2], cell[3])})
        if low == "selectedtab" and self.kind == "tabgroup":
            tab = self.selected_tab()
            return tab if tab is not None else np.zeros((0, 0))
        if low == "selectedobject" and self.kind == "buttongroup":
            sel = self.selected_button()
            return sel if sel is not None else np.zeros((0, 0))
        if low == "buttons" and self.kind == "buttongroup":
            return handle_list(self.group_buttons(), column=True)
        if low == "displaydata" and self.kind == "table":
            return self.get_prop("Data")
        if low == "valueindex" and self.kind in _CHOICE:
            idx = self.selected()
            return _row([i + 1 for i in idx]) if idx else np.zeros((0, 0))
        if low == "currentobject" and self.kind == "figure":
            return self.props.get("CurrentObject", np.zeros((0, 0)))
        if low == "number" and self.kind == "figure":
            return np.zeros((0, 0))
        found = _lookup(self.kind, name)
        if found is None:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.short()}'.")
        key, (code, default) = found
        if key == "Value":
            return self.get_value()
        if key in self.props:
            return self.props[key]
        if key == "Position":
            return self.default_position()
        if key == "Units" and self.kind in ("panel", "buttongroup", "tabgroup") and self.classic():
            return char("normalized")
        if key == "HorizontalAlignment" and self.kind == "uicontrol":
            return char("center")
        if key == "HorizontalAlignment" and self.kind in ("button", "statebutton"):
            return char("center")
        if key == "HorizontalAlignment" and self.kind in ("numericeditfield", "spinner"):
            return char("right")
        if key == "BackgroundColor" and self.kind in ("editfield", "numericeditfield", "textarea", "dropdown",
                                                      "listbox", "spinner", "table", "datepicker"):
            return _row((1, 1, 1))
        if key == "BackgroundColor" and self.kind in ("button", "statebutton", "togglebutton"):
            return _row((0.96, 0.96, 0.96))
        if key == "Limits" and self.kind == "datepicker":
            return np.zeros((0, 0))
        return _default_value(code, default)

    def default_position(self) -> np.ndarray:
        if self.kind in ("panel", "buttongroup", "tabgroup") and self.classic():
            return _row((0, 0, 1, 1))
        if self.kind == "tab" and self.parent is not None:
            w, h = self.parent.pixel_size()
            return _row((1, 1, w, max(h - 30, 0)))
        return _row(KINDS[self.kind][2])

    def classic(self) -> bool:
        fig = self.figure()
        return fig is not None and fig.mpl is not None

    def set_prop(self, name: str, value: Any) -> None:
        self._check()
        low = name.lower()
        if self.mpl is not None:
            self.mpl.set_field(self.interp, name, value)
            return
        if low == "parent":
            self.attach(parent_node(self.interp, value))
            return
        if low == "layout":
            self.set_layout(value)
            return
        if low in ("children", "type", "buttons", "displaydata", "extent"):
            if low == "children":
                return  # reordering children is accepted and ignored
            raise MatlabError(f"The '{name}' property is read-only.")
        if low == "selectedtab" and self.kind == "tabgroup":
            if not isinstance(value, UIComponent) or value.parent is not self:
                raise MatlabError("SelectedTab must be one of the tabs of this tab group.")
            self.props["SelectedTab"] = value
            return
        if low == "selectedobject" and self.kind == "buttongroup":
            if _empty(value):
                for b in self.group_buttons():
                    b.set_on(False)
                return
            if not isinstance(value, UIComponent) or value.parent is not self:
                raise MatlabError("SelectedObject must be one of the buttons of this button group.")
            self.select_button(value)
            return
        if low == "valueindex" and self.kind in _CHOICE:
            idx = [int(x) - 1 for x in _num(value, name).flatten()] if not _empty(value) else []
            self.select(idx)
            return
        if low == "currentobject":
            self.props["CurrentObject"] = value
            return
        found = _lookup(self.kind, name)
        if found is None:
            raise MatlabError(f"Unrecognized property '{name}' for class '{self.short()}'.")
        key, (code, _default) = found
        if key == "Value":
            self.set_value(value)
            return
        if key == "Units":
            self.set_units(_txt(value).lower())
            return
        if key == "Style" and self.kind == "uicontrol":
            s = _txt(value).lower()
            if s not in _CLASSIC_TYPES:
                raise MatlabError(f"'{_txt(value)}' is not a uicontrol style. Use 'pushbutton', 'togglebutton', "
                                  "'checkbox', 'radiobutton', 'edit', 'text', 'slider', 'popupmenu' or 'listbox'.")
            self.props[key] = char("popupmenu" if s == "popup" else s)
            return
        v = _normalize(code, key, value)
        if isinstance(v, MString) and key in ("Text", "String", "Tooltip", "ImageSource", "Icon"):
            v = char(str(v))  # text properties read back as character vectors
        if key == "Position":
            if v.size != 4:
                raise MatlabError("Position must be [left bottom width height].")
            v = v.reshape(1, 4)
        if key == "Limits" and self.kind != "datepicker":
            if v.size != 2 or not v.flat[0] < v.flat[1]:
                raise MatlabError("Limits must be [low high] with low < high.")
            v = v.reshape(1, 2)
        self.props[key] = v
        if key in ("Items", "ItemsData") and self.kind in _CHOICE:
            self.fix_choice()
        if key == "Limits" and "Value" in self.props and self.kind in ("slider", "knob", "gauge", "spinner",
                                                                       "numericeditfield"):
            lo, hi = v.flat[0], v.flat[1]
            self.props["Value"] = np.array([[min(max(float(self.props["Value"].flat[0]), lo), hi)]])

    def set_layout(self, value: Any) -> None:
        if not isinstance(value, Struct):
            raise MatlabError("Set the grid position with comp.Layout.Row = r and comp.Layout.Column = c.")
        cell = list(self.cell or [1, 1, 1, 1])
        for k, field in ((0, "Row"), (2, "Column")):
            if field in value.fields:
                a = _num(value.get_field(0, field), "Layout." + field).flatten()
                if a.size not in (1, 2) or (a < 1).any() or (a != np.round(a)).any():
                    raise MatlabError(f"Layout.{field} must be a row/column number or [first last].")
                cell[k], cell[k + 1] = int(a[0]), int(a[-1])
        self.cell = cell
        if self.parent is not None and self.parent.kind == "gridlayout":
            self.parent.grow(cell)

    # -- geometry --------------------------------------------------------------------------------
    def pixel_size(self) -> tuple[float, float]:
        """Width and height in pixels (approximate inside grid layouts)."""
        if self.kind == "figure":
            if self.mpl is not None:
                pos = np.asarray(self.mpl.get_field(self.interp, "Position"), dtype=float).flatten()
            else:
                pos = self.get_prop("Position").flatten()
            return float(pos[2]), float(pos[3])
        pos = self.get_prop("Position").flatten()
        units = _txt(self.get_prop("Units")).lower()
        if self.parent is not None and self.parent.kind in ("gridlayout", "tabgroup"):
            return self.parent.pixel_size()
        if units == "normalized":
            pw, ph = self.parent.pixel_size() if self.parent is not None else (640.0, 480.0)
            return float(pos[2]) * pw, float(pos[3]) * ph
        fx, fy = self.unit_factors(units)
        return float(pos[2]) * fx, float(pos[3]) * fy

    @staticmethod
    def unit_factors(units: str) -> tuple[float, float]:
        if units == "characters":
            return _CHAR_PIXELS
        f = _UNIT_PIXELS.get(units, 1.0)
        return f, f

    def set_units(self, units: str) -> None:
        if units not in _UNIT_PIXELS and units not in ("normalized", "characters"):
            raise MatlabError(f"'{units}' is not a unit. Use 'pixels', 'normalized', 'characters', 'points', "
                              "'inches' or 'centimeters'.")
        old = _txt(self.get_prop("Units")).lower()
        if old != units and "Position" in self.props:
            px = self.pixel_position()
            pw, ph = self.parent.pixel_size() if self.parent is not None else (640.0, 480.0)
            if units == "normalized":
                self.props["Position"] = _row((px[0] / pw, px[1] / ph, px[2] / pw, px[3] / ph))
            else:
                fx, fy = self.unit_factors(units)
                off = 1.0 if units == "pixels" else 0.0
                self.props["Position"] = _row((px[0] / fx + off, px[1] / fy + off, px[2] / fx, px[3] / fy))
        self.props["Units"] = char(units)

    def pixel_position(self) -> list[float]:
        """[left, bottom, width, height] in pixels with a 0-based origin."""
        pos = [float(x) for x in self.get_prop("Position").flatten()]
        units = _txt(self.get_prop("Units")).lower()
        if units == "normalized":
            pw, ph = self.parent.pixel_size() if self.parent is not None else (640.0, 480.0)
            return [pos[0] * pw, pos[1] * ph, pos[2] * pw, pos[3] * ph]
        fx, fy = self.unit_factors(units)
        off = 1.0 if units == "pixels" else 0.0
        return [(pos[0] - off) * fx, (pos[1] - off) * fy, pos[2] * fx, pos[3] * fy]

    # -- values ----------------------------------------------------------------------------------
    def style_of(self) -> str:
        return _txt(self.get_prop("Style")).lower() if self.kind == "uicontrol" else ""

    def items(self) -> list[str]:
        if self.kind == "uicontrol":
            v = self.get_prop("String")
            if _is_text(v) and "|" in _txt(v):
                return _txt(v).split("|")
            return _text_list(v)
        return _text_list(self.get_prop("Items"))

    def items_data(self) -> list | None:
        if self.kind == "uicontrol":
            return None
        d = self.props.get("ItemsData")
        if _empty(d):
            return None
        if isinstance(d, Cell):
            return list(d.values_f())
        if isinstance(d, StringArray):
            return [char(str(s)) for s in d.values_f()]
        if isinstance(d, np.ndarray):
            return [np.array([[x]]) for x in d.flatten(order="F")]
        return [d]

    def choices(self) -> list:
        """The values Value can take: ItemsData when it is set, otherwise the item texts."""
        data = self.items_data()
        items = self.items()
        if data is not None:
            return data[:len(items)] if len(data) >= len(items) else data
        return [char(s) for s in items]

    def fix_choice(self) -> None:
        if "Value" not in self.props:
            return
        keys = {_key(c) for c in self.choices()}
        v = self.props["Value"]
        if self.multi():
            vals = list(v.values_f()) if isinstance(v, Cell) else [v]
            if not all(_key(x) in keys for x in vals):
                del self.props["Value"]
        elif _key(v) not in keys and not (self.kind == "dropdown" and _onoff(self.get_prop("Editable"))
                                           and "Editable" in self.props):
            del self.props["Value"]

    def multi(self) -> bool:
        if self.kind == "listbox":
            return _onoff(self.get_prop("Multiselect"))
        if self.kind == "uicontrol" and self.style_of() == "listbox":
            return float(self.get_prop("Max").flat[0]) - float(self.get_prop("Min").flat[0]) > 1
        return False

    def selected(self) -> list[int]:
        """0-based numbers of the selected items."""
        if self.kind == "uicontrol":
            v = self.props.get("Value")
            if v is None:
                return [0] if self.items() else []
            n = len(self.items())
            return [int(x) - 1 for x in np.asarray(v, dtype=float).flatten() if 1 <= int(x) <= n]
        choices = self.choices()
        if "Value" not in self.props:
            return [0] if choices and not (self.kind == "listbox" and self.multi()) else []
        v = self.props["Value"]
        vals = list(v.values_f()) if isinstance(v, Cell) and self.multi() else (
            [np.array([[x]]) for x in v.flatten(order="F")]
            if self.multi() and isinstance(v, np.ndarray) and not is_char(v) else [v])
        keys = [_key(c) for c in choices]
        return [keys.index(_key(x)) for x in vals if _key(x) in keys]

    def select(self, idx: list[int]) -> None:
        if self.kind == "uicontrol":
            self.props["Value"] = _row([i + 1 for i in idx]) if idx else np.zeros((0, 0))
            return
        choices = self.choices()
        idx = [i for i in idx if 0 <= i < len(choices)]
        if self.multi():
            picked = [choices[i] for i in idx]
            if self.items_data() is not None and all(isinstance(p, np.ndarray) and not is_char(p) for p in picked):
                self.props["Value"] = _row([float(p.flat[0]) for p in picked])
            else:
                self.props["Value"] = Cell.from_list(picked) if picked else Cell.empty(1, 0)
        elif idx:
            self.props["Value"] = choices[idx[0]]
        else:
            self.props["Value"] = Cell.empty(0, 0) if self.kind == "listbox" else char("")

    def get_value(self) -> Any:
        k = self.kind
        if k in _CHOICE:
            if "Value" in self.props:
                return self.props["Value"]
            choices = self.choices()
            if self.kind == "listbox" and self.multi():
                return Cell.empty(1, 0)
            return choices[0] if choices else char("")
        if "Value" in self.props:
            return self.props["Value"]
        if k in _TOGGLE:
            return _bool(False)
        if k == "editfield":
            return char("")
        if k == "textarea":
            return Cell.from_list([char("")], column=True)
        if k == "datepicker":
            return self._date_value(None)
        if k in ("slider", "knob", "gauge", "spinner", "numericeditfield"):
            lo, hi = (float(x) for x in self.get_prop("Limits").flatten())
            return np.array([[min(max(0.0, lo), hi)]])
        if k == "uicontrol":
            style = self.style_of()
            if style in ("popupmenu", "listbox"):
                return np.array([[1.0]])
            return np.array([[float(self.get_prop("Min").flat[0])]])
        return np.zeros((0, 0))

    def set_value(self, value: Any) -> None:
        k = self.kind
        if k in _TOGGLE:
            on = _onoff(value)
            if k in ("radiobutton", "togglebutton") and self.parent is not None and \
                    self.parent.kind == "buttongroup":
                if on:
                    self.parent.select_button(self)
                else:
                    self.props["Value"] = _bool(False)
            else:
                self.props["Value"] = _bool(on)
        elif k == "editfield":
            if not (_is_text(value) or _empty(value)):
                raise MatlabError("The Value of a text edit field must be text. Use num2str or a numeric "
                                  "edit field: uieditfield(parent, 'numeric').")
            self.props["Value"] = char(_txt(value))
        elif k == "textarea":
            lines = _text_list(value) if not _is_text(value) else _txt(value).split("\n")
            self.props["Value"] = Cell.from_list([char(s) for s in (lines or [""])], column=True)
        elif k in ("slider", "knob", "gauge", "spinner", "numericeditfield"):
            x = _scalar(value, "Value")
            lo, hi = (float(v) for v in self.get_prop("Limits").flatten())
            if k in ("numericeditfield", "spinner") and not lo <= x <= hi:
                raise MatlabError(f"Value must be between {_fmt_number(lo)} and {_fmt_number(hi)} (the Limits).")
            if k in ("numericeditfield", "spinner") and _onoff(self.get_prop("RoundFractionalValues")):
                x = float(np.round(x))
            self.props["Value"] = np.array([[min(max(x, lo), hi)]])
        elif k in _CHOICE:
            if self.multi():
                vals = list(value.values_f()) if isinstance(value, Cell) else (
                    [char(str(s)) for s in value.values_f()] if isinstance(value, StringArray) else
                    [] if _empty(value) else
                    [np.array([[x]]) for x in value.flatten(order="F")]
                    if isinstance(value, np.ndarray) and not is_char(value) else [value])
                keys = [_key(c) for c in self.choices()]
                bad = [v for v in vals if _key(v) not in keys]
                if bad:
                    raise MatlabError("Value must be made of elements of Items (or of ItemsData).")
                self.select([keys.index(_key(v)) for v in vals])
                return
            if isinstance(value, MString):
                value = char(str(value))
            if _empty(value) and self.kind == "listbox":
                self.select([])
                return
            keys = [_key(c) for c in self.choices()]
            if _key(value) not in keys:
                if self.kind == "dropdown" and _onoff(self.get_prop("Editable")) and "Editable" in self.props \
                        and _is_text(value):
                    self.props["Value"] = char(_txt(value))
                    return
                shown = ", ".join(f"'{s}'" for s in self.items()[:6])
                raise MatlabError(f"Value must be one of the Items ({shown}"
                                  f"{', ...' if len(self.items()) > 6 else ''})"
                                  f"{' or an element of ItemsData' if self.items_data() is not None else ''}.")
            self.props["Value"] = value
        elif k == "datepicker":
            self.props["Value"] = value
        elif k == "uicontrol":
            if _empty(value):
                self.props["Value"] = np.zeros((0, 0))
            else:
                self.props["Value"] = _num(value, "Value")
                if self.parent is not None and self.parent.kind == "buttongroup" and self.is_on() and \
                        self.style_of() in ("radiobutton", "togglebutton"):
                    self.parent.select_button(self)
        else:
            self.props["Value"] = value

    def _date_value(self, iso: str | None) -> Any:
        fn = self.interp.builtins.get("datetime")
        if fn is None:
            return np.zeros((0, 0))
        if iso is None:
            fn = self.interp.builtins.get("NaT", fn)
            try:
                return self.interp.call_builtin("NaT", fn, [], 1)[0]
            except MatlabError:
                return np.zeros((0, 0))
        y, m, d = (float(x) for x in iso.split("-"))
        return self.interp.call_builtin("datetime", fn, [np.array([[y]]), np.array([[m]]), np.array([[d]])], 1)[0]

    def _date_iso(self) -> str:
        v = self.props.get("Value")
        if v is None or _empty(v):
            return ""
        try:
            parts = [self.interp.call_builtin(n, self.interp.builtins[n], [v], 1)[0] for n in ("year", "month", "day")]
            y, m, d = (float(np.asarray(p, dtype=float).flat[0]) for p in parts)
            if y != y:
                return ""
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        except (MatlabError, KeyError, ValueError, TypeError, IndexError):
            return ""

    # toggles (check boxes, radio buttons, classic controls with Min/Max)
    def is_on(self) -> bool:
        if self.kind == "uicontrol":
            v = self.props.get("Value")
            if v is None or v.size == 0:
                return False
            return float(v.flat[0]) == float(self.get_prop("Max").flat[0])
        return bool(self.get_value().flat[0])

    def set_on(self, on: bool) -> None:
        if self.kind == "uicontrol":
            self.props["Value"] = np.array([[float(self.get_prop("Max" if on else "Min").flat[0])]])
        else:
            self.props["Value"] = _bool(on)

    # button groups and tab groups
    def group_buttons(self) -> list:
        return [c for c in self.children if c.kind in ("radiobutton", "togglebutton") or (
            c.kind == "uicontrol" and c.style_of() in ("radiobutton", "togglebutton"))]

    def selected_button(self) -> Any:
        buttons = self.group_buttons()
        for b in buttons:
            if b.is_on():
                return b
        if buttons and not self.props.get("_none"):
            buttons[0].set_on(True)  # the first button of a group starts selected
            return buttons[0]
        return None

    def select_button(self, button: Any) -> None:
        for b in self.group_buttons():
            b.set_on(b is button)

    def selected_tab(self) -> Any:
        tab = self.props.get("SelectedTab")
        tabs = [c for c in self.children if c.kind == "tab"]
        if isinstance(tab, UIComponent) and tab in tabs:
            return tab
        return tabs[0] if tabs else None

    # -- display ---------------------------------------------------------------------------------
    def display_lines(self) -> list[str]:
        if self.deleted:
            return [f"  handle to deleted {self.short()}"]
        lines = [f"  {self.short()} with properties:", ""]
        names = [n for n in ("Name", "Title", "Text", "Style", "String", "Value", "Items", "Limits", "Position",
                             "Units", "Tag") if _lookup(self.kind, n) is not None]
        shown = []
        for n in names:
            if n in ("Units", "Tag") and n not in self.props:
                continue
            try:
                shown.append((n, _show(self.get_prop(n))))
            except MatlabError:
                continue
        width = max((len(n) for n, _ in shown), default=0)
        lines += [f"    {n.rjust(width)}: {v}" for n, v in shown]
        return lines

    # -- serialization ---------------------------------------------------------------------------
    def proto_type(self) -> str:
        if self.kind == "uicontrol":
            style = self.style_of()
            if style == "edit" and float(self.get_prop("Max").flat[0]) - float(self.get_prop("Min").flat[0]) > 1:
                return "textarea"
            return _CLASSIC_TYPES.get(style, "button")
        return {"numericeditfield": "editfield", "togglebutton": "statebutton", "discreteknob": "dropdown",
                "buttongroup": "panel"}.get(self.kind, self.kind)

    def base_props(self) -> dict:
        p: dict[str, Any] = {}
        if self.mpl is not None:  # a classic figure window: its look comes from the plotting figure
            face = self.mpl.obj.get_facecolor()
            if tuple(round(float(x), 3) for x in face[:3]) != (1.0, 1.0, 1.0):
                p["style"] = {"background": _css_color(face[:3])}
            return p
        if not _onoff(self.get_prop("Visible")):
            p["visible"] = False
        en = self.get_prop("Enable")
        if _txt(en).lower() == "inactive":
            p["enabled"] = True
        elif not _onoff(en):
            p["enabled"] = False
        tip = _txt(self.get_prop("Tooltip"))
        if tip:
            p["tooltip"] = tip
        tag = _txt(self.get_prop("Tag"))
        if tag:
            p["tag"] = tag
        if self.kind not in ("figure", "tab", "menu"):
            if self.parent is not None and self.parent.kind == "gridlayout":
                p["grid"] = list(self.cell or [1, 1, 1, 1])
            elif _txt(self.get_prop("Units")).lower() == "normalized":
                p["npos"] = [float(x) for x in self.get_prop("Position").flatten()]
            elif not (self.kind == "gridlayout" and "Position" not in self.props):
                p["pos"] = [round(x, 2) for x in self.pixel_position()]
        style: dict[str, Any] = {}
        if "FontColor" in self.props:
            style["color"] = _css_color(self.props["FontColor"])
        bg_key = "Color" if self.kind in ("figure", "lamp") else "BackgroundColor"
        if bg_key in self.props and self.kind != "lamp":
            style["background"] = _css_color(self.props[bg_key])
        if "FontSize" in self.props:
            size = float(self.props["FontSize"].flat[0])
            if self.kind == "uicontrol" or self.classic():
                size *= 4 / 3  # classic controls measure fonts in points
            style["fontSize"] = round(size, 1)
        if "FontWeight" in self.props and _txt(self.props["FontWeight"]).lower() == "bold":
            style["bold"] = True
        if "FontAngle" in self.props and _txt(self.props["FontAngle"]).lower() == "italic":
            style["italic"] = True
        if "FontName" in self.props:
            style["font"] = _txt(self.props["FontName"])
        if "HorizontalAlignment" in self.props:
            style["align"] = _txt(self.props["HorizontalAlignment"]).lower()
        if "VerticalAlignment" in self.props:
            style["valign"] = _txt(self.props["VerticalAlignment"]).lower()
        style = {k: v for k, v in style.items() if v is not None}
        if style:
            p["style"] = style
        return p

    def serialize_props(self) -> dict:
        p = self.base_props()
        t = self.proto_type()
        k = self.kind
        g = self.get_prop
        if k == "figure":
            w, h = self.pixel_size()
            p.update(name=_txt(g("Name")), width=round(w), height=round(h))
            if self.mpl is not None:
                p["classic"] = True
                img = render_classic_figure(self)
                if img:
                    p["image"] = img
            elif not _onoff(g("Resize")):
                p["resize"] = False
        elif k == "gridlayout":
            sizes = lambda which: [(_txt(x) if _is_text(x) else float(x.flat[0])) for x in self.grid_sizes(which)]
            pad = [float(x) for x in g("Padding").flatten()]
            pad = (pad * 4)[:4] if len(pad) < 4 else pad[:4]
            p.update(rows=sizes("RowHeight"), cols=sizes("ColumnWidth"), padding=pad,
                     rowGap=float(g("RowSpacing").flat[0]), colGap=float(g("ColumnSpacing").flat[0]))
            if _onoff(g("Scrollable")):
                p["scroll"] = True
        elif k in ("panel", "buttongroup", "tab"):
            p["title"] = _txt(g("Title"))
            if k != "tab" and _txt(g("BorderType")).lower() == "none":
                p["border"] = False
            if _onoff(g("Scrollable")):
                p["scroll"] = True
        elif k == "tabgroup":
            tabs = [c for c in self.children if c.kind == "tab"]
            sel = self.selected_tab()
            p["selected"] = tabs.index(sel) if sel in tabs else 0
        elif k == "uicontrol":
            self._classic_props(p, t)
        elif t == "label":
            p["text"] = _txt(g("Text"))
            if _onoff(g("WordWrap")):
                p["wrap"] = True
        elif k in ("button", "hyperlink", "menu"):
            p["text"] = _txt(g("Text"))
            if k == "hyperlink":
                p["url"] = _txt(g("URL"))
            if k == "menu":
                p["checked"] = _onoff(g("Checked"))
                p["separator"] = _onoff(g("Separator"))
            icon = g("Icon") if k == "button" else None
            if icon is not None and not _empty(icon):
                src = image_source(icon)
                if src:
                    p["icon"] = src
        elif k in _TOGGLE:
            p.update(text=_txt(g("Text")), value=self.is_on())
            if k == "togglebutton":
                p["toggle"] = True
        if not _empty(self.props.get("ValueChangingFcn")):
            p["live"] = True  # the front end also reports the value while it is being changed
        if k == "editfield":
            p.update(value=_txt(g("Value")), placeholder=_txt(g("Placeholder")))
            if not _onoff(g("Editable")):
                p["editable"] = False
        elif k == "numericeditfield":
            lo, hi = (float(x) for x in g("Limits").flatten())
            p.update(value=self.format_number(float(g("Value").flat[0])), numeric=True,
                     min=_json_num(lo), max=_json_num(hi))
            if not _onoff(g("Editable")):
                p["editable"] = False
        elif k == "textarea":
            p.update(value="\n".join(_text_list(g("Value"))), placeholder=_txt(g("Placeholder")))
            if not _onoff(g("Editable")):
                p["editable"] = False
        elif k in ("dropdown", "discreteknob"):
            sel = self.selected()
            p.update(items=self.items(), selected=sel[0] if sel else -1)
            if k == "dropdown" and "Editable" in self.props and _onoff(g("Editable")):
                p.update(editable=True, value=_txt(g("Value")))
            if k == "dropdown" and _txt(g("Placeholder")):
                p["placeholder"] = _txt(g("Placeholder"))
        elif k == "listbox":
            p.update(items=self.items(), selected=self.selected(), multi=self.multi())
        elif k == "switch":
            sel = self.selected()
            p.update(items=self.items()[:2], selected=sel[0] if sel else 0)
        elif k in ("slider", "knob", "gauge"):
            lo, hi = (float(x) for x in g("Limits").flatten())
            p.update(value=float(g("Value").flat[0]), min=lo, max=hi)
            if _txt(g("Orientation")).lower() == "vertical" and k == "slider":
                p["vertical"] = True
            if k == "slider" and "Step" in self.props:
                p["step"] = float(self.props["Step"].flat[0])
            if k == "gauge" and self.style:
                p["shape"] = self.style
        elif k == "spinner":
            lo, hi = (float(x) for x in g("Limits").flatten())
            p.update(value=float(g("Value").flat[0]), min=_json_num(lo), max=_json_num(hi),
                     step=float(g("Step").flat[0]))
            if not _onoff(g("Editable")):
                p["editable"] = False
        elif k == "lamp":
            p["color"] = _css_color(g("Color")) or "#00ff00"
        elif k == "table":
            p.update(table_json(self))
            if not _empty(self.props.get("CellSelectionCallback")) or not _empty(self.props.get("SelectionChangedFcn")):
                p["selectable"] = True
        elif k == "image":
            p["src"] = image_source(g("ImageSource")) or ""
            p["scale"] = _txt(g("ScaleMethod")).lower()
            p["alt"] = _txt(g("AltText"))
            if not _empty(self.props.get("ImageClickedFcn")):
                p["clickable"] = True
        elif k == "datepicker":
            p["value"] = self._date_iso()
        return p

    def format_number(self, x: float) -> str:
        fmt = _txt(self.get_prop("ValueDisplayFormat"))
        try:
            return (fmt % x).strip()
        except (TypeError, ValueError):
            return _fmt_number(x)

    def _classic_props(self, p: dict, t: str) -> None:
        g = self.get_prop
        s = g("String")
        lo, hi = float(g("Min").flat[0]), float(g("Max").flat[0])
        p["classic"] = True
        if t in ("button", "label"):
            p["text"] = _txt(s)
            if t == "label":
                p["wrap"] = True
            p.setdefault("style", {}).setdefault("align", _txt(g("HorizontalAlignment")).lower())
        elif t in ("checkbox", "radiobutton", "statebutton"):
            p.update(text=_txt(s), value=self.is_on())
        elif t in ("editfield", "textarea"):
            p["value"] = _txt(s)
            p.setdefault("style", {}).setdefault("align", _txt(g("HorizontalAlignment")).lower())
        elif t == "slider":
            v = self.props.get("Value")
            x = float(v.flat[0]) if v is not None and v.size else lo
            step = float(g("SliderStep").flat[0]) * (hi - lo)
            p.update(value=min(max(x, lo), hi), min=lo, max=hi, step=step if step > 0 else None)
            pos = self.pixel_position()
            if pos[3] > pos[2]:
                p["vertical"] = True
        elif t == "dropdown":
            sel = self.selected()
            p.update(items=self.items(), selected=sel[0] if sel else -1)
        elif t == "listbox":
            p.update(items=self.items(), selected=self.selected(), multi=self.multi())
        elif t == "panel":
            p["title"] = ""

    def serialize(self) -> dict:
        return {"id": self.uid, "type": self.proto_type(), "props": self.serialize_props(),
                "children": [c.serialize() for c in self.children if not c.deleted]}


def _show(v: Any) -> str:
    if _is_text(v):
        return f"'{_txt(v)}'"
    if isinstance(v, Cell):
        items = v.values_f()
        if len(items) <= 6 and all(_is_text(x) for x in items):
            return "{" + ", ".join(f"'{_txt(x)}'" for x in items) + "}"
        return f"{v.data.shape[0]}x{v.data.shape[1]} cell"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "[]"
        if v.size <= 6:
            body = " ".join(_fmt_number(float(x)) for x in np.asarray(v, dtype=float).flatten())
            return body if v.size == 1 else f"[{body}]"
        return "x".join(str(n) for n in v.shape) + " " + ("logical" if v.dtype == bool else "double")
    if isinstance(v, MObject):
        return v.class_name().rsplit(".", 1)[-1]
    return str(v)


# -- arrays of components --------------------

class UIArray(MObject):
    """Several components, e.g. the Children of a container."""

    is_handle = True

    def __init__(self, items: list, row: bool = False):
        self.items = list(items)
        self.row = row

    @property
    def data(self) -> np.ndarray:  # lets for-loops visit one component at a time
        arr = np.empty((1, len(self.items)), dtype=object)
        for i, h in enumerate(self.items):
            arr[0, i] = h
        return arr

    def class_name(self) -> str:
        names = {h.class_name() for h in self.items}
        return names.pop() if len(names) == 1 else "matlab.graphics.Graphics"

    def isa_class(self, name: str) -> bool:
        return name in ("handle", "matlab.graphics.Graphics", self.class_name())

    def matlab_shape(self) -> tuple:
        return (1, len(self.items)) if self.row else (len(self.items), 1)

    def summary_text(self, in_cell: bool) -> str:
        r, c = self.matlab_shape()
        return f"[{r}\u00d7{c} graphics array]"

    def has_method(self, name: str) -> bool:
        return name in ("delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "delete":
            for h in self.items:
                destroy_any(interp, h)
            return []
        if name == "isvalid":
            flags = np.array([alive(h) for h in self.items], dtype=bool)
            return [flags.reshape(self.matlab_shape())]
        return super().call_method(interp, name, args, nargout)

    def field_list(self, interp, name: str) -> list:
        return [h.get_field(interp, name) for h in self.items]

    def get_field(self, interp, name: str) -> Any:
        vals = self.field_list(interp, name)
        return Cell.from_list(vals, column=True)

    def set_field(self, interp, name: str, value: Any) -> "UIArray":
        for h in self.items:
            h.set_field(interp, name, value)
        return self

    def _indices(self, args: list) -> list[int]:
        if len(args) == 2:
            first, second = args
            if is_char(first) and _txt(first) == ":":
                args = [second]
            elif is_char(second) and _txt(second) == ":":
                args = [first]
            else:
                args = [second] if self.row else [first]
        if len(args) != 1:
            raise MatlabError("Use one index into an array of components.")
        a = args[0]
        if is_char(a) and _txt(a) == ":":
            return list(range(len(self.items)))
        v = np.asarray(a)
        if v.dtype == bool:
            return [int(i) for i in np.flatnonzero(v.flatten(order="F"))]
        idx = [int(round(float(x))) - 1 for x in np.asarray(v, dtype=float).flatten(order="F")]
        for i in idx:
            if i < 0 or i >= len(self.items):
                raise MatlabError("Index exceeds the number of array elements. Index must not exceed "
                                  f"{len(self.items)}.")
        return idx

    def index_get(self, interp, args: list) -> Any:
        idx = self._indices(args)
        if len(idx) == 1:
            return self.items[idx[0]]
        return UIArray([self.items[i] for i in idx], self.row)

    def index_set(self, interp, args: list, value: Any) -> Any:
        idx = self._indices(args) if not (len(args) == 1 and isinstance(args[0], np.ndarray)
                                          and args[0].size == 1 and int(args[0].flat[0]) == len(self.items) + 1) \
            else [len(self.items)]
        if len(idx) != 1 or not is_component(value):
            raise MatlabError("Index assignment into an array of components must replace or append one component.")
        if idx[0] == len(self.items):
            self.items.append(value)
        else:
            self.items[idx[0]] = value
        return self

    def display_lines(self) -> list[str]:
        r, c = self.matlab_shape()
        lines = [f"  {r}×{c} graphics array:", ""]
        lines += [f"    {h.class_name().rsplit('.', 1)[-1]}" for h in self.items]
        return lines

    @staticmethod
    def concat(items: list, axis: int) -> Any:
        return concat_handles(items, axis)


def concat_handles(items: list, axis: int) -> Any:
    out: list = []
    for it in items:
        if isinstance(it, UIArray):
            out.extend(it.items)
        elif is_component(it):
            out.append(it)
        elif isinstance(it, np.ndarray) and it.size == 0:
            continue
        else:
            raise MatlabError("Components can only be concatenated with other components.")
    if len(out) == 1:
        return out[0]
    return UIArray(out, row=(axis == 1))


UIComponent.concat = staticmethod(concat_handles)  # type: ignore[attr-defined]


def is_component(v: Any) -> bool:
    return isinstance(v, UIComponent) or getattr(v, "is_ui_axes", False)


def components_of(v: Any) -> list:
    if is_component(v):
        return [v]
    if isinstance(v, UIArray):
        return list(v.items)
    return []


def handle_list(items: list, column: bool = False) -> Any:
    if not items:
        return np.zeros((0, 0))
    if len(items) == 1:
        return items[0]
    return UIArray(items, row=not column)


def alive(h: Any) -> bool:
    if isinstance(h, UIComponent):
        return not h.deleted
    return bool(getattr(h, "ui_alive", lambda: False)())


def destroy_any(interp, h: Any) -> None:
    if isinstance(h, UIComponent):
        h.destroy()
    elif getattr(h, "is_ui_axes", False):
        h.ui_destroy()


# -- axes --------------------

_AXES_UI_PROPS = {"position": "Position", "innerposition": "Position", "outerposition": "Position",
                  "layout": "Layout", "parent": "Parent", "tag": "Tag", "userdata": "UserData",
                  "visible": "Visible", "units": "Units", "backgroundcolor": "BackgroundColor",
                  "buttondownfcn": "ButtonDownFcn", "toolbar": "Toolbar", "interactions": "Interactions",
                  "contextmenu": "ContextMenu", "tooltip": "Tooltip", "enable": "Enable",
                  "handlevisibility": "HandleVisibility", "createfcn": "CreateFcn", "deletefcn": "DeleteFcn",
                  "pickableparts": "PickableParts", "hittest": "HitTest"}
_UIAXES_CLASS = None


def uiaxes_class():
    """The UIAxes class: an axes graphics handle that is also a node of the component tree."""
    global _UIAXES_CLASS
    if _UIAXES_CLASS is not None:
        return _UIAXES_CLASS
    from .builtins.plotting import GraphicsHandle, _state as plot_state

    class UIAxes(GraphicsHandle):
        is_ui_axes = True
        kind = "axes"

        def __init__(self, interp, parent: UIComponent):
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(4.0, 3.0), dpi=100, layout="constrained")
            FigureCanvasAgg(fig)
            fig._mylab_ui = True
            fig.add_subplot(1, 1, 1)
            self.fig = fig
            GraphicsHandle.__init__(self, interp, "axes", None)
            st = state(interp)
            self.uid = st.next_id
            st.next_id += 1
            st.nodes[self.uid] = self
            self.ui_props: dict[str, Any] = {}
            self.parent: UIComponent | None = None
            self.children: list = []
            self.deleted = False
            self.cell: list[int] | None = None
            self.shown: tuple[float, float] | None = None  # size the front end shows the axes at
            self._png: str | None = None
            self._png_key: Any = None
            self.attach(parent)
            plot_state(interp).handles[id(self.obj)] = self

        # the matplotlib axes may be replaced (a 3-D plot swaps it), so look it up each time
        @property
        def obj(self):
            axes = [a for a in self.fig.axes if not hasattr(a, "_colorbar")]
            if not axes:
                return self.fig.add_subplot(1, 1, 1)
            return axes[0]

        @obj.setter
        def obj(self, value) -> None:
            pass

        def ui_alive(self) -> bool:
            return not self.deleted

        def summary_text(self, in_cell: bool) -> str:
            return "[1\u00d71 UIAxes]"

        def class_name(self) -> str:
            return "matlab.ui.control.UIAxes"

        def isa_class(self, name: str) -> bool:
            return name in ("matlab.ui.control.UIAxes", "UIAxes", "matlab.graphics.axis.Axes", "Axes", "handle",
                            "matlab.graphics.Graphics")

        def has_method(self, name: str) -> bool:
            return name in ("delete", "isvalid")

        def call_method(self, interp, name: str, args: list, nargout: int) -> list:
            if name == "delete":
                for a in args:
                    for h in components_of(a):
                        destroy_any(interp, h)
                return []
            if name == "isvalid":
                return [_bool(not self.deleted)]
            return GraphicsHandle.call_method(self, interp, name, args, nargout)

        def binary_op(self, interp, op: str, left: Any, right: Any) -> Any:
            if op in ("==", "~="):
                return _bool((left is right) == (op == "=="))
            return NotImplemented

        attach = UIComponent.attach
        figure = UIComponent.figure
        short = lambda self: "UIAxes"
        descendants = lambda self: []
        classic = UIComponent.classic
        unit_factors = staticmethod(UIComponent.unit_factors)
        concat = staticmethod(concat_handles)

        def ui_destroy(self, run_callbacks: bool = True) -> None:
            if self.deleted:
                return
            self.deleted = True
            self.fig._mylab_ui = False
            st = state(self.interp)
            st.nodes.pop(self.uid, None)
            if self.parent is not None and self in self.parent.children:
                self.parent.children.remove(self)

        destroy = ui_destroy

        def get_field(self, interp, name: str) -> Any:
            key = _AXES_UI_PROPS.get(name.lower())
            if key is None:
                if self.deleted:
                    raise MatlabError("Invalid or deleted object.")
                return GraphicsHandle.get_field(self, interp, name)
            if key == "Parent":
                if self.parent is None:
                    return np.zeros((0, 0))
                return self.parent.mpl if self.parent.mpl is not None else self.parent
            if key == "Layout":
                cell = self.cell or [1, 1, 1, 1]
                span = lambda a, b: np.array([[float(a)]]) if a == b else _row((a, b))
                return Struct.scalar({"Row": span(cell[0], cell[1]), "Column": span(cell[2], cell[3])})
            if key in self.ui_props:
                return self.ui_props[key]
            if key == "Position":
                return _row((10, 10, 400, 300))
            if key in ("Visible", "Enable", "HandleVisibility", "HitTest"):
                return char("on")
            if key == "Units":
                return char("pixels")
            if key == "BackgroundColor":
                return _row((0.94, 0.94, 0.94))
            if key in ("Tag", "Tooltip", "ButtonDownFcn", "CreateFcn", "DeleteFcn"):
                return char("")
            return np.zeros((0, 0))

        get_prop = lambda self, name: self.get_field(self.interp, name)

        def set_field(self, interp, name: str, value: Any):
            key = _AXES_UI_PROPS.get(name.lower())
            if key is None:
                if self.deleted:
                    raise MatlabError("Invalid or deleted object.")
                return GraphicsHandle.set_field(self, interp, name, value)
            if key == "Parent":
                self.attach(parent_node(interp, value))
            elif key == "Layout":
                UIComponent.set_layout(self, value)
            elif key == "Position":
                v = _num(value, "Position")
                if v.size != 4:
                    raise MatlabError("Position must be [left bottom width height].")
                self.ui_props[key] = v.reshape(1, 4)
            elif key in ("Visible", "Enable"):
                self.ui_props[key] = char("on" if _onoff(value) else "off")
            elif key == "Units":
                self.ui_props[key] = char(_txt(value).lower())
            elif key == "BackgroundColor":
                self.ui_props[key] = _color(value)
            elif key in ("Tag", "Tooltip"):
                self.ui_props[key] = char(_txt(value))
            else:
                self.ui_props[key] = value
            return self

        set_prop = lambda self, name, value: self.set_field(self.interp, name, value)

        def pixel_position(self) -> list[float]:
            pos = [float(x) for x in self.get_field(self.interp, "Position").flatten()]
            units = _txt(self.get_field(self.interp, "Units")).lower()
            if units == "normalized":
                pw, ph = self.parent.pixel_size() if self.parent is not None else (640.0, 480.0)
                return [pos[0] * pw, pos[1] * ph, pos[2] * pw, pos[3] * ph]
            return [pos[0] - 1, pos[1] - 1, pos[2], pos[3]]

        def pixel_size(self) -> tuple[float, float]:
            if self.shown:
                return self.shown
            if self.parent is not None and self.parent.kind == "gridlayout":
                return (480.0, 320.0)
            p = self.pixel_position()
            return (p[2], p[3])

        def render(self) -> str:
            w, h = self.pixel_size()
            w, h = max(float(w), 40.0), max(float(h), 40.0)
            key = (round(w), round(h))
            if self._png is None or self._png_key != key or self.fig.stale:
                self.fig.set_size_inches(w / 100.0, h / 100.0, forward=False)
                buf = io.BytesIO()
                try:
                    self.fig.savefig(buf, format="png", dpi=IMAGE_DPI)
                except Exception as e:  # a plot that cannot be drawn must not take the app down
                    raise MatlabError(f"The axes could not be drawn: {e}") from e
                self._png = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                self._png_key = key
                self.fig.stale = False
            return self._png

        def serialize(self) -> dict:
            p: dict[str, Any] = {}
            if not _onoff(self.get_field(self.interp, "Visible")):
                p["visible"] = False
            tag = _txt(self.get_field(self.interp, "Tag"))
            if tag:
                p["tag"] = tag
            if self.parent is not None and self.parent.kind == "gridlayout":
                p["grid"] = list(self.cell or [1, 1, 1, 1])
            elif _txt(self.get_field(self.interp, "Units")).lower() == "normalized":
                p["npos"] = [float(x) for x in self.get_field(self.interp, "Position").flatten()]
            else:
                p["pos"] = [round(x, 2) for x in self.pixel_position()]
            w, h = self.pixel_size()
            p.update(image=self.render(), width=round(w), height=round(h), alt=self.describe())
            return {"id": self.uid, "type": "axes", "props": p, "children": []}

        def describe(self) -> str:
            ax = self.obj
            title = ax.get_title() or "Plot"
            n = len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.images)
            return f"{title} ({n} plotted object{'' if n == 1 else 's'})"

    _UIAXES_CLASS = UIAxes
    return UIAxes


IMAGE_DPI = 150


def render_classic_figure(node: UIComponent) -> str | None:
    """The drawing of a classic figure window (its axes), or None when it has no axes."""
    fig = node.mpl.obj
    if not fig.axes:
        node.props.pop("_png", None)
        return None
    w, h = node.pixel_size()
    key = (round(w), round(h))
    cache = node.props.get("_png")
    if cache is None or cache[0] != key or fig.stale:
        fig.set_size_inches(max(w, 40) / 100.0, max(h, 40) / 100.0, forward=False)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=IMAGE_DPI)
        except Exception as e:
            raise MatlabError(f"The figure could not be drawn: {e}") from e
        cache = (key, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        node.props["_png"] = cache
        fig.stale = False
    return cache[1]


def image_source(v: Any) -> str | None:
    """A data URL for an image file name or an image array."""
    try:
        if _is_text(v):
            path = _txt(v)
            if not path:
                return None
            if path.startswith(("data:", "http://", "https://")):
                return path
            if not os.path.isfile(path):
                return None
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
                    "svg": "image/svg+xml", "bmp": "image/bmp"}.get(ext)
            if mime is None:
                return None
            with open(path, "rb") as fh:
                return f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")
        if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.size:
            from PIL import Image
            a = np.asarray(v)
            if a.dtype.kind == "f":
                a = np.clip(a, 0, 1) * 255
            a = np.clip(a, 0, 255).astype(np.uint8)
            if a.ndim == 3 and a.shape[2] == 1:
                a = a[:, :, 0]
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # an unreadable image is shown as an empty box
        return None
    return None


# -- tables --------------------

def _table_columns(node: UIComponent) -> tuple[list[str] | None, list[list], list[str]]:
    """(names, columns as lists of cells, column types) of the Data of a table component."""
    data = node.props.get("Data")
    if data is None or _empty(data):
        return None, [], []
    if getattr(data, "class_name", lambda: "")() == "table":
        names = list(data.names)
        cols, types = [], []
        for col in data.columns:
            cells, kind = _column_cells(col)
            cols.append(cells)
            types.append(kind)
        return names, cols, types
    if isinstance(data, Cell):
        d = data.data
        cols, types = [], []
        for j in range(d.shape[1]):
            cells = [d[i, j] for i in range(d.shape[0])]
            kinds = {_cell_kind(c) for c in cells}
            cols.append(cells)
            types.append(kinds.pop() if len(kinds) == 1 else "text")
        return None, cols, types
    if isinstance(data, StringArray):
        d = np.atleast_2d(data.data)
        return None, [[char(str(d[i, j])) for i in range(d.shape[0])] for j in range(d.shape[1])], ["text"] * d.shape[1]
    if isinstance(data, np.ndarray):
        if is_char(data):
            return None, [[char("".join(row)) for row in np.atleast_2d(data)]], ["text"]
        d = np.atleast_2d(data)
        kind = "bool" if d.dtype == bool else "num"
        cols = [[np.array([[d[i, j]]]) for i in range(d.shape[0])] for j in range(d.shape[1])]
        return None, cols, [kind] * d.shape[1]
    raise MatlabError("The Data of a table must be a numeric or logical matrix, a cell array, a string array "
                      "or a table.")


def _cell_kind(c: Any) -> str:
    if isinstance(c, np.ndarray) and not is_char(c):
        return "bool" if c.dtype == bool else "num"
    return "text"


def _column_cells(col: Any) -> tuple[list, str]:
    if isinstance(col, Cell):
        cells = list(col.data[:, 0]) if col.data.ndim == 2 and col.data.shape[1] >= 1 else list(col.values_f())
        kinds = {_cell_kind(c) for c in cells}
        return cells, kinds.pop() if len(kinds) == 1 else "text"
    if isinstance(col, StringArray):
        return [char(str(s)) for s in col.values_f()], "text"
    if isinstance(col, np.ndarray):
        if is_char(col):
            return [char("".join(row).rstrip()) for row in np.atleast_2d(col)], "text"
        d = np.atleast_2d(col)
        kind = "bool" if d.dtype == bool else "num"
        return [np.array([[d[i, 0]]]) for i in range(d.shape[0])], kind
    n = col.matlab_shape()[0] if hasattr(col, "matlab_shape") else 0
    cells = []
    for i in range(n):
        try:
            cells.append(char(_object_text(col, i)))
        except Exception:
            cells.append(char(""))
    return cells, "text"


def _object_text(col: Any, i: int) -> str:
    from .interpreter import current_interpreter
    interp = current_interpreter()
    item = col.index_get(interp, [np.array([[float(i + 1)]])])
    fn = interp.builtins.get("string") if interp is not None else None
    if fn is None:
        return str(item)
    return _txt(interp.call_builtin("string", fn, [item], 1)[0])


def _cell_json(c: Any, kind: str) -> Any:
    if kind == "bool":
        return bool(np.asarray(c).flat[0]) if isinstance(c, np.ndarray) and c.size else False
    if isinstance(c, np.ndarray) and not is_char(c):
        if c.size == 0:
            return ""
        if c.size == 1:
            x = c.flat[0]
            if isinstance(x, (bool, np.bool_)):
                return "true" if x else "false"
            return _fmt_number(float(np.real(x)))
        return _txt(c)
    return _txt(c)


def _flags(v: Any, n: int) -> list[bool]:
    if v is None or _empty(v):
        return [False] * n
    a = np.asarray(v).flatten() if isinstance(v, np.ndarray) else np.array([_onoff(x) for x in v.values_f()])
    out = [bool(x) for x in a]
    if len(out) == 1:
        out = out * n
    return (out + [False] * n)[:n]


def table_json(node: UIComponent) -> dict:
    names, cols, types = _table_columns(node)
    ncols = len(cols)
    nrows = len(cols[0]) if cols else 0
    cn = node.props.get("ColumnName")
    if cn is not None and not (_is_text(cn) and _txt(cn) == "numbered") and not _empty(cn):
        head = _text_list(cn)
    elif names is not None:
        head = names
    else:
        head = [str(j + 1) for j in range(ncols)]
    head = (head + [str(j + 1) for j in range(len(head), ncols)])[:max(ncols, len(head))]
    if not ncols and head:
        types = ["text"] * len(head)
    editable = _flags(node.props.get("ColumnEditable"), max(ncols, len(head)))
    columns = [{"name": head[j] if j < len(head) else str(j + 1), "type": types[j] if j < len(types) else "text",
                "editable": editable[j]} for j in range(max(ncols, len(head)))]
    fmt = node.props.get("ColumnFormat")
    if isinstance(fmt, Cell):
        for j, f in enumerate(fmt.values_f()):
            if j < len(columns) and isinstance(f, Cell):
                columns[j]["options"] = _text_list(f)
            elif j < len(columns) and _is_text(f) and _txt(f) == "logical":
                columns[j]["type"] = "bool"
    rows = [[_cell_json(cols[j][i], columns[j]["type"]) for j in range(ncols)] for i in range(nrows)]
    out: dict[str, Any] = {"columns": columns, "rows": rows}
    rn = node.props.get("RowName")
    data = node.props.get("Data")
    if rn is not None and not _empty(rn) and not (_is_text(rn) and _txt(rn) == "numbered"):
        out["rowNames"] = _text_list(rn)
    elif getattr(data, "row_names", None):
        out["rowNames"] = list(data.row_names)
    return out


def table_edit(node: UIComponent, row: int, col: int, value: Any) -> tuple[Any, Any, str] | None:
    """Store an edited cell; returns (previous, new, typed text) or None when nothing changed."""
    data = node.props.get("Data")
    names, cols, types = _table_columns(node)
    if not (0 <= col < len(cols)) or not (0 <= row < len(cols[col])):
        return None
    info = table_json(node)["columns"][col]
    kind = info["type"]
    old = cols[col][row]
    typed = "" if value is None else ("true" if value is True else "false" if value is False else str(value))
    if kind == "bool":
        new: Any = _bool(value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "on"))
    elif kind == "num":
        try:
            x = float(value)
        except (TypeError, ValueError):
            x = _parse_number(typed)
        new = np.array([[x]])
    else:
        new = char(typed)
    if getattr(data, "class_name", lambda: "")() == "table":
        column = data.columns[col]
        if isinstance(column, Cell):
            column.data[row, 0] = new
        elif isinstance(column, StringArray):
            column.data.reshape(-1)[row] = typed
        elif isinstance(column, np.ndarray) and not is_char(column):
            if column.dtype.kind in "iub" and kind == "num":
                column[row, 0] = int(round(float(new.flat[0]))) if new.flat[0] == new.flat[0] else 0
            else:
                column[row, 0] = new.flat[0]
        else:
            return None
    elif isinstance(data, Cell):
        data.data[row, col] = new
    elif isinstance(data, StringArray):
        np.atleast_2d(data.data)[row, col] = typed
    elif isinstance(data, np.ndarray):
        if kind == "text":
            return None
        d = data
        if d.dtype.kind in "iu" and kind == "num":
            d[row, col] = int(round(float(new.flat[0]))) if new.flat[0] == new.flat[0] else 0
        elif d.dtype.kind == "b":
            d[row, col] = bool(new.flat[0])
        else:
            d[row, col] = float(new.flat[0])
    return old, new, typed


def _parse_number(s: str) -> float:
    s = s.strip().lower().replace(",", ".")
    if s in ("inf", "+inf"):
        return float("inf")
    if s == "-inf":
        return float("-inf")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# -- parents and classic figures --------------------

def classic_node(interp, fig_handle) -> UIComponent:
    """The window node of a classic figure (figure(...)), made on first use."""
    st = state(interp)
    node = st.classic.get(id(fig_handle.obj))
    if node is not None and not node.deleted:
        return node
    node = UIComponent(interp, "figure")
    node.mpl = fig_handle
    st.classic[id(fig_handle.obj)] = node
    return node


def parent_node(interp, value: Any) -> UIComponent:
    if isinstance(value, UIComponent):
        return value
    kind = getattr(value, "kind", None)
    if kind == "figure" and hasattr(value, "obj"):
        return classic_node(interp, value)
    if isinstance(value, np.ndarray) and value.size == 1 and not is_char(value):
        from .builtins.plotting import _state as plot_state, _wrap
        n = int(value.flat[0])
        fig = plot_state(interp).figures.get(n)
        if fig is not None:
            return classic_node(interp, _wrap(interp, "figure", fig))
    raise MatlabError("The parent must be a figure, panel, tab, button group or grid layout.")


def prune(interp) -> None:
    """Forget classic figure windows whose figure was closed."""
    st = interp.state.get("ui")
    if st is None or not st.classic:
        return
    plot = interp.state.get("plot")
    live = {id(f) for f in plot.figures.values()} if plot is not None else set()
    for key, node in list(st.classic.items()):
        if key not in live:
            node.mpl = None
            node.destroy(run_callbacks=False)
            st.classic.pop(key, None)


def figure_of(interp, h: Any) -> UIComponent | None:
    """The window node that ``h`` (a component or a classic figure handle) belongs to."""
    if is_component(h):
        return h.figure()
    if getattr(h, "kind", None) == "figure" and hasattr(h, "obj"):
        return classic_node(interp, h)
    if hasattr(h, "obj") and hasattr(h, "kind"):
        from .builtins.plotting import _obj_figure, _wrap
        fig = _obj_figure(h)
        if fig is not None:
            return classic_node(interp, _wrap(interp, "figure", fig))
    return None


def visible_figures(interp) -> list[UIComponent]:
    st = interp.state.get("ui")
    if st is None:
        return []
    prune(interp)
    out = []
    for fig in st.figures:
        if fig.deleted:
            continue
        if fig.mpl is not None and not fig.children:
            continue  # a plain plotting figure
        out.append(fig)
    return out


def classic_figure_numbers(interp) -> set[int]:
    """Numbers of the plotting figures that are app windows (they are not shown as plots)."""
    st = interp.state.get("ui")
    plot = interp.state.get("plot")
    if st is None or plot is None:
        return set()
    return {n for n, f in plot.figures.items()
            if id(f) in st.classic and st.classic[id(f)].children and not st.classic[id(f)].deleted}


def has_ui(interp) -> bool:
    return bool(visible_figures(interp)) or bool(getattr(interp.state.get("ui"), "dialogs", None))


# -- callbacks --------------------

def make_event(source: Any, name: str, **fields: Any) -> Struct:
    data = {"Source": source if not (isinstance(source, UIComponent) and source.mpl is not None) else source.mpl,
            "EventName": char(name)}
    data.update(fields)
    return Struct.scalar(data)


def _arity(interp, fn: FunctionHandle) -> int | None:
    if fn.is_anonymous:
        params = fn.params
        if params and params[-1] == "varargin":
            return None
        return len(params)
    target = fn.closure.get("__functions__", {}).get(fn.name)
    target = getattr(target, "fn", target)
    params = getattr(target, "params", None)
    if params is None:
        return None
    if params and params[-1] == "varargin":
        return None
    return len(params)


def invoke(interp, source: Any, cb: Any, event: Struct, extra: list | None = None) -> None:
    """Run one callback value: a handle, {@fn, args...}, a command text or an app method."""
    if _empty(cb):
        return
    st = state(interp)
    src = source.mpl if isinstance(source, UIComponent) and source.mpl is not None else source
    fig = figure_of(interp, source)
    st.callbacks.append((src, fig))
    try:
        args = [src, event]
        if isinstance(cb, AppCallback):
            app = cb.app
            call_args = [app, event] if cb.with_event else [app]
            app.cls.call(interp, cb.name, call_args, 0)
        elif isinstance(cb, FunctionHandle):
            n = _arity(interp, cb)
            interp.call_handle(cb, args if n is None else args[:max(n, 0)] if n < 2 else args, 0)
        elif isinstance(cb, Cell):
            items = list(cb.values_f())
            fn, rest = items[0], items[1:]
            if isinstance(fn, FunctionHandle):
                interp.call_handle(fn, args + rest, 0)
            else:
                interp.call_function_value(fn, args + rest, 0)
        elif _is_text(cb):
            interp.execute(_txt(cb))
        else:
            raise MatlabError("A callback must be a function handle, a cell array {@fn, args} or text.")
    finally:
        st.callbacks.pop()


def fire(interp, node: Any, prop: str, event: Struct) -> None:
    cb = node.ui_props.get(prop) if getattr(node, "is_ui_axes", False) else node.props.get(prop)
    if not _empty(cb):
        invoke(interp, node, cb, event)


def close_figure(interp, fig: UIComponent) -> None:
    """What the close button (and close(fig)) does: CloseRequestFcn, or delete the window."""
    if fig.deleted:
        return
    if fig.mpl is not None:
        cb = fig.mpl.extra.get("CloseRequestFcn")
        if not _empty(cb) and not (_is_text(cb) and _txt(cb) == "closereq"):
            invoke(interp, fig, cb, make_event(fig, "Close"))
            return
        from .builtins.plotting import _close_number, _fig_number
        n = _fig_number(interp, fig.mpl.obj)
        if n is not None:
            _close_number(interp, n)
        prune(interp)
        return
    cb = fig.props.get("CloseRequestFcn")
    if not _empty(cb) and not (_is_text(cb) and _txt(cb) == "closereq"):
        invoke(interp, fig, cb, make_event(fig, "Close"))
        return
    fig.destroy()


# -- events from the front end --------------------

def _handle_event(interp, ev: dict) -> None:
    st = state(interp)
    name = str(ev.get("event", ""))
    if name == "dialog":
        _dialog_event(interp, ev)
        return
    try:
        node = st.nodes.get(int(ev.get("id")))
    except (TypeError, ValueError):
        node = None
    if node is None or not alive(node):
        return
    if getattr(node, "is_ui_axes", False):
        if name == "size":
            w, h = float(ev.get("width", 0)), float(ev.get("height", 0))
            if w >= 40 and h >= 40:
                node.shown = (min(w, 2000.0), min(h, 2000.0))
        elif name == "push":
            fire(interp, node, "ButtonDownFcn", make_event(node, "Hit"))
        return
    fig = node.figure()
    if fig is not None and fig.mpl is None and node.kind != "figure":
        fig.props["CurrentObject"] = node
    k = node.kind
    value = ev.get("value")
    if k == "figure":
        if name == "close":
            close_figure(interp, node)
        elif name == "resize":
            w, h = float(ev.get("width", 0)), float(ev.get("height", 0))
            if w < 20 or h < 20:
                return
            if node.mpl is not None:
                pos = [float(x) for x in np.asarray(node.mpl.get_field(interp, "Position"), dtype=float).flatten()]
                if (round(pos[2]), round(pos[3])) == (round(w), round(h)):
                    return
                node.mpl.set_field(interp, "Position", _row((pos[0], pos[1], w, h)))
                cb = node.mpl.extra.get("SizeChangedFcn", node.mpl.extra.get("ResizeFcn"))
                if not _empty(cb):
                    invoke(interp, node, cb, make_event(node, "SizeChanged"))
            else:
                pos = [float(x) for x in node.get_prop("Position").flatten()]
                if (round(pos[2]), round(pos[3])) == (round(w), round(h)):
                    return
                node.props["Position"] = _row((pos[0], pos[1], w, h))
                fire(interp, node, "SizeChangedFcn", make_event(node, "SizeChanged"))
        return
    if not _onoff(node.get_prop("Enable")) and _txt(node.get_prop("Enable")).lower() != "inactive":
        return
    if k == "uicontrol":
        _classic_event(interp, node, name, ev)
    elif name == "push":
        prop = {"button": "ButtonPushedFcn", "menu": "MenuSelectedFcn", "image": "ImageClickedFcn",
                "hyperlink": "HyperlinkClickedFcn"}.get(k)
        if prop:
            ename = {"button": "ButtonPushed", "menu": "MenuSelected", "image": "ImageClicked",
                     "hyperlink": "HyperlinkClicked"}[k]
            fire(interp, node, prop, make_event(node, ename))
    elif name == "select" and k == "tabgroup":
        tabs = [c for c in node.children if c.kind == "tab"]
        old = node.selected_tab()
        i = int(value or 0)
        if 0 <= i < len(tabs) and tabs[i] is not old:
            node.props["SelectedTab"] = tabs[i]
            fire(interp, node, "SelectionChangedFcn",
                 make_event(node, "SelectionChanged", OldValue=old, NewValue=tabs[i]))
    elif k == "table":
        _table_event(interp, node, name, ev)
    elif name == "value":
        _value_event(interp, node, value, bool(ev.get("changing")))


def _value_event(interp, node: UIComponent, value: Any, changing: bool) -> None:
    k = node.kind
    old = node.get_value()
    if changing:
        if k in ("slider", "knob", "spinner"):
            new = np.array([[float(value)]])
        elif k in ("editfield", "textarea"):
            new = char(str(value))
        else:
            return
        fire(interp, node, "ValueChangingFcn", make_event(node, "ValueChanging", Value=new))
        return
    if k in _TOGGLE:
        on = bool(value)
        group = node.parent if node.parent is not None and node.parent.kind == "buttongroup" else None
        if group is not None and k in ("radiobutton", "togglebutton"):
            before = group.selected_button()
            if not on or before is node:
                return
            group.select_button(node)
            fire(interp, group, "SelectionChangedFcn",
                 make_event(group, "SelectionChanged", OldValue=before if before is not None else np.zeros((0, 0)),
                            NewValue=node))
            return
        node.props["Value"] = _bool(on)
    elif k == "editfield":
        node.props["Value"] = char(str(value if value is not None else ""))
    elif k == "textarea":
        lines = str(value if value is not None else "").split("\n")
        node.props["Value"] = Cell.from_list([char(s) for s in lines], column=True)
    elif k in ("numericeditfield", "spinner", "slider", "knob"):
        x = _parse_number(str(value)) if not isinstance(value, (int, float)) else float(value)
        lo, hi = (float(v) for v in node.get_prop("Limits").flatten())
        if x != x or (k in ("numericeditfield", "spinner") and not lo <= x <= hi):
            return  # not a number, or outside the limits: the old value stays
        if k in ("numericeditfield", "spinner") and _onoff(node.get_prop("RoundFractionalValues")):
            x = float(np.round(x))
        node.props["Value"] = np.array([[min(max(x, lo), hi)]])
    elif k in _CHOICE:
        if isinstance(value, str):
            if not (k == "dropdown" and "Editable" in node.props and _onoff(node.get_prop("Editable"))):
                return
            node.props["Value"] = char(value)
        else:
            idx = [int(i) for i in value] if isinstance(value, list) else [int(value)]
            if not node.multi():
                idx = idx[:1]
            node.select(idx)
    elif k == "datepicker":
        node.props["Value"] = node._date_value(str(value) if value else None)
    else:
        return
    new = node.get_value()
    if _key(old) == _key(new) and not isinstance(new, Cell) and k != "datepicker":
        return
    fire(interp, node, "ValueChangedFcn", make_event(node, "ValueChanged", Value=new, PreviousValue=old))


def _classic_event(interp, node: UIComponent, name: str, ev: dict) -> None:
    style = node.style_of()
    value = ev.get("value")
    event = make_event(node, "Action")
    if name == "push":
        if style == "pushbutton":
            fire(interp, node, "Callback", event)
        return
    if name != "value" or ev.get("changing"):
        return
    if style in ("checkbox", "togglebutton", "radiobutton"):
        on = bool(value)
        group = node.parent if node.parent is not None and node.parent.kind == "buttongroup" else None
        if group is not None and style != "checkbox":
            before = group.selected_button()
            if not on or before is node:
                return
            group.select_button(node)
            fire(interp, node, "Callback", event)
            fire(interp, group, "SelectionChangedFcn",
                 make_event(group, "SelectionChanged", OldValue=before if before is not None else np.zeros((0, 0)),
                            NewValue=node))
            return
        node.set_on(on)
    elif style == "edit":
        text = str(value if value is not None else "")
        multi = float(node.get_prop("Max").flat[0]) - float(node.get_prop("Min").flat[0]) > 1
        node.props["String"] = Cell.from_list([char(s) for s in text.split("\n")], column=True) if multi and \
            "\n" in text else char(text)
    elif style == "slider":
        lo, hi = float(node.get_prop("Min").flat[0]), float(node.get_prop("Max").flat[0])
        node.props["Value"] = np.array([[min(max(float(value), lo), hi)]])
    elif style in ("popupmenu", "listbox"):
        idx = [int(i) for i in value] if isinstance(value, list) else [int(value)]
        if not node.multi():
            idx = idx[:1]
        node.select(idx)
    else:
        return
    fire(interp, node, "Callback", event)


def _table_event(interp, node: UIComponent, name: str, ev: dict) -> None:
    row, col = int(ev.get("row", -1)), int(ev.get("col", -1))
    if name == "cell":
        info = table_json(node)["columns"]
        if not (0 <= col < len(info)) or not info[col]["editable"]:
            return
        res = table_edit(node, row, col, ev.get("value"))
        if res is None:
            return
        old, new, typed = res
        if _key(old) == _key(new):
            return
        idx = _row((row + 1, col + 1))
        fire(interp, node, "CellEditCallback",
             make_event(node, "CellEdit", Indices=idx, DisplayIndices=idx, PreviousData=old, NewData=new,
                        EditData=char(typed), Error=np.zeros((0, 0))))
        fire(interp, node, "DisplayDataChangedFcn",
             make_event(node, "DisplayDataChanged", DisplaySelection=idx, InteractionColumn=np.array([[col + 1.0]]),
                        Interaction=char("edit")))
    elif name == "select":
        if row < 0 or col < 0:
            return
        idx = _row((row + 1, col + 1))
        old = node.props.get("Selection", np.zeros((0, 0)))
        node.props["Selection"] = idx
        fire(interp, node, "CellSelectionCallback", make_event(node, "CellSelection", Indices=idx))
        fire(interp, node, "SelectionChangedFcn",
             make_event(node, "SelectionChanged", Selection=idx, PreviousSelection=old,
                        SelectionType=char("cell"), DisplaySelection=idx))


# -- dialogs --------------------

def add_dialog(interp, kind: str, fig: UIComponent | None, obj: Any = None, **fields: Any) -> dict:
    st = state(interp)
    d = {"id": f"d{st.next_id}", "kind": kind, "figure": fig.uid if fig is not None else None}
    st.next_id += 1
    d.update(fields)
    st.dialogs.append(d)
    if obj is not None:
        st.dialog_objects[d["id"]] = obj
    return d


def remove_dialog(interp, dialog_id: str) -> None:
    st = state(interp)
    st.dialogs[:] = [d for d in st.dialogs if d["id"] != dialog_id]
    st.dialog_objects.pop(dialog_id, None)


def _dialog_event(interp, ev: dict) -> None:
    st = state(interp)
    did = str(ev.get("id"))
    d = next((x for x in st.dialogs if x["id"] == did), None)
    if d is None:
        return
    obj = st.dialog_objects.get(did)
    choice = str(ev.get("value") or "")
    if d["kind"] == "progress":
        if obj is not None:
            obj.cancel_requested = True
            d["cancelRequested"] = True
        return
    remove_dialog(interp, did)
    st.answers[did] = choice
    while len(st.answers) > 50:
        st.answers.pop(next(iter(st.answers)))
    cb = obj
    if not _empty(cb):
        options = d.get("options") or ["OK"]
        index = options.index(choice) + 1 if choice in options else 0
        fig = st.nodes.get(d.get("figure"))
        event = Struct.scalar({"Source": fig if fig is not None else np.zeros((0, 0)),
                               "EventName": char("Close"), "DialogTitle": char(d.get("title", "")),
                               "SelectedOption": char(choice), "SelectedOptionIndex": np.array([[float(index)]])})
        invoke(interp, fig if fig is not None else np.zeros((0, 0)), cb, event)


# -- the protocol --------------------

def _flatten(nodes: list[dict], out: dict) -> None:
    for n in nodes:
        out[n["id"]] = (n["type"], n["props"], [c["id"] for c in n["children"]])
        _flatten(n["children"], out)


def _tree(interp) -> dict:
    st = state(interp)
    figures = [f.serialize() for f in visible_figures(interp)]
    return {"figures": figures, "dialogs": [dict(d) for d in st.dialogs]}


def snapshot(interp) -> dict | None:
    """The whole component tree as JSON data, or None when there are no app windows."""
    if not has_ui(interp):
        st = interp.state.get("ui")
        if st is not None:
            st.sent = None
        return None
    tree = _tree(interp)
    flat: dict = {}
    _flatten(tree["figures"], flat)
    state(interp).sent = {"flat": flat, "order": [f["id"] for f in tree["figures"]]}
    return tree


def update(interp) -> dict:
    """What changed since the last snapshot/update (see the module docstring)."""
    st = state(interp)
    tree = _tree(interp)
    flat: dict = {}
    _flatten(tree["figures"], flat)
    order = [f["id"] for f in tree["figures"]]
    sent = st.sent
    st.sent = {"flat": flat, "order": order}
    same = sent is not None and sent["order"] == order and sent["flat"].keys() == flat.keys() and all(
        sent["flat"][i][0] == flat[i][0] and sent["flat"][i][2] == flat[i][2] for i in flat)
    if not same:
        return {"full": True, "figures": tree["figures"], "dialogs": tree["dialogs"]}
    patch = {}
    for i, (_t, props, _c) in flat.items():
        old = sent["flat"][i][1]
        if old != props:
            changed = {k: v for k, v in props.items() if old.get(k, _MISSING) != v}
            for k in old:
                if k not in props:
                    changed[k] = None
            patch[str(i)] = changed
    return {"full": False, "patch": patch, "dialogs": tree["dialogs"]}


_MISSING = object()


def dispatch(interp, event: dict) -> dict:
    """Apply one front-end event, run its callbacks and report what changed."""
    error = None
    try:
        _handle_event(interp, event)
    except MatlabError as e:
        line = getattr(e, "line", None)
        file = getattr(e, "file", None)
        error = e.message
        if line and file:
            error += f"\nError in {os.path.basename(file)} (line {line})"
        elif line:
            error += f" (line {line})"
    return {"error": error, "update": update(interp)}


# -- driving an app from Python (tests, scripts) --------------------

def find(interp, tag: str | None = None, type: str | None = None, text: str | None = None) -> Any:  # noqa: A002
    """The first component with this Tag, protocol type and/or text (for tests)."""
    for fig in visible_figures(interp):
        for node in [fig] + fig.descendants():
            if tag is not None and _txt(node.get_prop("Tag")) != tag:
                continue
            if type is not None and (node.proto_type() if isinstance(node, UIComponent) else "axes") != type:
                continue
            if text is not None:
                if not isinstance(node, UIComponent):
                    continue
                label = node.serialize_props().get("text", node.serialize_props().get("title"))
                if label != text:
                    continue
            return node
    return None
