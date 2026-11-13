"""Simple dialogs (waitbar, msgbox, questdlg, inputdlg, listdlg, menu, uigetfile, ...) without a GUI.

Every dialog is answered in one of three ways, tried in this order:

1. **A front-end handler.** If ``interp.state["dialog_handler"]`` is a callable
   ``handler(kind, payload) -> answer``, it is asked first. Returning ``None``
   means "not handled" and mylab continues with step 2. ``payload`` is a plain
   dict of strings, numbers, booleans and lists (JSON-friendly).
2. **A text prompt.** In a terminal, or when a front end replaced
   ``interp.input_func`` (the Jupyter kernel does), the question is printed and
   the answer is read with ``interp.input_func(prompt)``.
3. **The default answer.** When nobody can be asked (standard input is not a
   terminal, ``interp.input_func`` raises, or the front end set
   ``interp.state["dialogs_noninteractive"] = True``), the dialog's default is
   returned and a one-line note is printed, so scripts keep running.

Handler payloads and the answers they expect:

=============  ==========================================================  ================================
kind           payload keys                                                answer
=============  ==========================================================  ================================
``msgbox``     ``message`` (text, lines joined by "\\n"), ``title``,        anything but ``None`` once it is
               ``icon`` ("none", "error", "warn", "help"), ``function``    shown (mylab then prints nothing)
               ("msgbox", "errordlg", "warndlg" or "helpdlg")
``questdlg``   ``question``, ``title``, ``buttons`` (list of text),        the chosen button text; ``""``
               ``default`` (one of the buttons)                            when the dialog was closed
``inputdlg``   ``prompts`` (list), ``title``, ``defaults`` (list, same     list of text, one per prompt;
               length as prompts)                                          ``False`` for Cancel
``listdlg``    ``items`` (list), ``prompt``, ``title``, ``multiple``       list of 1-based item numbers;
               (bool), ``initial`` (list of 1-based numbers), ``ok``,      ``False`` or ``[]`` for Cancel
               ``cancel`` (button texts)
``menu``       ``title``, ``items`` (list)                                 1-based item number; ``0`` when
                                                                           closed
``uigetfile``  ``title``, ``filters`` (list of [pattern, description]),    a path (or a list of paths when
               ``default`` (file name or ""), ``multiselect`` (bool)       multiselect); ``False`` = Cancel
``uiputfile``  ``title``, ``filters``, ``default``                         a path; ``False`` for Cancel
``uigetdir``   ``title``, ``start`` (folder)                               a folder path; ``False`` = Cancel
``waitbar``    ``id`` (number), ``fraction`` (0..1), ``message``,          anything but ``None`` when the
               ``title``, ``state`` ("open", "update" or "close")          front end shows the bar itself
``ginput``     ``n`` (number of points, or ``None`` for "until Enter"),    list of [x, y] or [x, y, button]
               ``figure`` (figure number)                                  in data coordinates
=============  ==========================================================  ================================

Custom windows (``uifigure``, ``uicontrol``, ...) are built by
``mylab/builtins/uicomponents.py`` on top of :mod:`mylab.ui`.
"""

from __future__ import annotations

import builtins
import os
import sys
import time
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, Struct, char, is_char
from ._registry import builtin
from ._util import fnum, is_text, text
from .plotting import _noninteractive, _plt, _split_name_values, _state, _text_list

AUTO_ANSWER_LIMIT = 500  # automatic answers in quick succession before a loop is stopped


# -- asking ------------------------------------------------------------------------------------------

def _ask_handler(interp, kind: str, payload: dict) -> Any:
    fn = interp.state.get("dialog_handler")
    if fn is None:
        return None
    try:
        return fn(kind, payload)
    except MatlabError:
        raise
    except Exception:  # a failing front end must not stop the script: fall back to text or defaults
        return None


def _can_prompt(interp) -> bool:
    """Can a person be asked through interp.input_func?"""
    if interp.state.get("dialogs_noninteractive"):
        return False
    if interp.input_func is builtins.input:
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError):
            return False
    return True


def _prompt(interp, message: str) -> str | None:
    """One line typed by the user, or None when nobody can answer."""
    if not _can_prompt(interp):
        return None
    try:
        reply = interp.input_func(message)
    except KeyboardInterrupt:
        raise
    except Exception:  # EOF, "stdin not supported" in a notebook, ...
        return None
    return "" if reply is None else str(reply)


def _auto(interp, what: str, answer: str) -> None:
    """Note an automatic answer, and stop scripts that would ask forever."""
    now = time.monotonic()
    count, last = interp.state.get("dialog_auto", (0, 0.0))
    count = count + 1 if now - last < 1.0 else 1
    interp.state["dialog_auto"] = (count, now)
    if count > AUTO_ANSWER_LIMIT:
        interp.state["dialog_auto"] = (0, now)
        raise MatlabError(f"{what} was answered automatically {AUTO_ANSWER_LIMIT} times in a row because no "
                          "interactive input is available; stopping so that the script does not loop forever.")
    interp.out(f"[{what}] {answer}\n")


def _lines(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, np.ndarray) and v.size == 0:
        return []
    return _text_list(v)


def _one_line(v) -> str:
    return " ".join(s.strip() for s in _lines(v) if s.strip())


def _opt_text(args: list, k: int, default: str = "") -> str:
    return _one_line(args[k]) if len(args) > k and (is_text(args[k]) or isinstance(args[k], Cell)) else default


def _cell_col(items: list) -> Cell:
    data = np.empty((len(items), 1), dtype=object)
    for i, s in enumerate(items):
        data[i, 0] = char(s)
    return Cell(data)


_EMPTY = np.zeros((0, 0))
_ZERO = np.array([[0.0]])


# -- dialog handles ----------------------------------------------------------------------------------

class DialogHandle(MObject):
    """What waitbar and msgbox return: a figure-like handle that can be closed or deleted."""

    is_handle = True

    def __init__(self, kind: str, title: str = ""):
        self.kind = kind
        self.props: dict[str, Any] = {"Name": char(title), "Tag": char(""), "UserData": _EMPTY,
                                      "Visible": char("on"), "Type": char("figure")}
        self.fraction = 0.0
        self.message = ""
        self.drawn = False     # a progress line is on the terminal
        self.external = False  # a front end shows it
        self.shown = ""

    def class_name(self) -> str:
        return "matlab.ui.Figure"

    def isa_class(self, name: str) -> bool:
        return name in ("matlab.ui.Figure", "Figure", "handle")

    def has_method(self, name: str) -> bool:
        return name in ("close", "delete", "isvalid")

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        if name == "isvalid":
            return [np.array([[not getattr(self, "_deleted", False)]])]
        for a in args:
            if isinstance(a, DialogHandle):
                a.dismiss(interp)
        return []

    def dismiss(self, interp) -> None:
        if getattr(self, "_deleted", False):
            return
        self._deleted = True
        if self.kind == "waitbar":
            if self.external:
                _ask_handler(interp, "waitbar", self.payload("close"))
            if self.drawn:
                interp.out("\n")
                self.drawn = False
            if interp.state.get("waitbar_last") is self:
                interp.state.pop("waitbar_last", None)

    def payload(self, state: str) -> dict:
        return {"id": id(self), "fraction": self.fraction, "message": self.message,
                "title": "".join(self.props["Name"].flatten()) if is_char(self.props["Name"]) else "",
                "state": state}

    def get_field(self, interp, name: str) -> Any:
        for k, v in self.props.items():
            if k.lower() == name.lower():
                return v
        raise MatlabError(f"Unrecognized property '{name}' for this dialog handle.")

    def set_field(self, interp, name: str, value: Any) -> "DialogHandle":
        for k in self.props:
            if k.lower() == name.lower():
                name = k
        self.props[name] = value
        return self

    def display_lines(self) -> list[str]:
        state = "closed" if getattr(self, "_deleted", False) else "open"
        return [f"  Figure ({self.kind}, {state})"]


# -- waitbar -------------------------------------------------------------------------------------------

def _is_terminal(interp) -> bool:
    """True when interp.out writes straight to an interactive terminal."""
    if getattr(interp.out, "__self__", None) is not None:  # a capturing sink such as list.append
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _waitbar_draw(interp, h: DialogHandle, state: str) -> None:
    pct = int(round(h.fraction * 100))
    key = f"{pct}|{h.message}"
    if key == h.shown and state == "update":
        return
    h.shown = key
    if state == "open":
        h.external = _ask_handler(interp, "waitbar", h.payload("open")) is not None
    elif h.external:
        _ask_handler(interp, "waitbar", h.payload("update"))
    if h.external or not _is_terminal(interp):
        return
    filled = int(round(h.fraction * 30))
    line = f"\r[{'#' * filled}{'.' * (30 - filled)}] {pct:3d}% {h.message}"
    interp.out(line.ljust(79)[:200])
    h.drawn = True


@builtin("waitbar")
def _waitbar(interp, args, nargout):
    if not args or isinstance(args[0], DialogHandle) or is_text(args[0]):
        raise MatlabError("waitbar needs a fraction between 0 and 1, e.g. h = waitbar(0, 'Working...').")
    x = float(np.clip(np.nan_to_num(np.real(fnum(args[0])).flat[0]), 0.0, 1.0))
    rest = list(args[1:])
    h = None
    if rest and isinstance(rest[0], DialogHandle):
        h = rest.pop(0)
        if h.kind != "waitbar":
            raise MatlabError("waitbar: the second input must be a handle returned by waitbar.")
    elif not rest:
        h = interp.state.get("waitbar_last")
    new = h is None or getattr(h, "_deleted", False)
    if new:  # also when the bar to update was closed already
        h = DialogHandle("waitbar")
        interp.state["waitbar_last"] = h
    if rest and (is_text(rest[0]) or isinstance(rest[0], Cell)):
        h.message = _one_line(rest.pop(0))
    _, opts = _split_name_values(rest, {"name", "createcancelbtn", "windowstyle", "tag", "visible"})
    if "name" in opts:
        h.props["Name"] = char(_one_line(opts["name"]))
    if "tag" in opts:
        h.props["Tag"] = char(_one_line(opts["tag"]))
    h.fraction = x
    _waitbar_draw(interp, h, "open" if new else "update")
    return h if nargout > 0 else None


# -- message boxes -------------------------------------------------------------------------------------

_BOX = {"msgbox": ("Message", "", "none"), "errordlg": ("Error", "Error Dialog", "error"),
        "warndlg": ("Warning", "Warning Dialog", "warn"), "helpdlg": ("Help", "Help Dialog", "help")}
_ICON_LABEL = {"error": "Error", "warn": "Warning", "help": "Help", "none": "Message", "custom": "Message"}


def _message_box(interp, args, nargout, fname: str):
    label, default_title, icon = _BOX[fname]
    msg = _lines(args[0]) if args else []
    title = default_title
    if len(args) > 1 and (is_text(args[1]) or isinstance(args[1], Cell)) and \
            _one_line(args[1]).lower() not in ("modal", "non-modal", "replace"):
        title = _one_line(args[1])
    if fname == "msgbox" and len(args) > 2 and is_text(args[2]) and text(args[2]).lower() in _ICON_LABEL:
        icon = text(args[2]).lower()
        label = _ICON_LABEL[icon]
    h = DialogHandle("msgbox", title)
    shown = _ask_handler(interp, "msgbox", {"message": "\n".join(msg), "title": title, "icon": icon,
                                            "function": fname})
    if shown is None:
        head = label if not title or title == default_title else f"{label} ({title})"
        body = msg or [""]
        interp.out(f"{head}: {body[0]}\n" + "".join(f"  {s}\n" for s in body[1:]))
    return h if nargout > 0 else None


for _name in _BOX:
    builtin(_name)(lambda interp, args, nargout, _n=_name: _message_box(interp, args, nargout, _n))


# -- questions -----------------------------------------------------------------------------------------

def _pick(reply: str, items: list[str]) -> int | None:
    """Index of the item a typed reply means (a number, the text, or a unique beginning of it)."""
    r = reply.strip()
    if r.isdigit():
        k = int(r)
        return k - 1 if 1 <= k <= len(items) else None
    low = [s.lower() for s in items]
    if r.lower() in low:
        return low.index(r.lower())
    hits = [i for i, s in enumerate(low) if r and s.startswith(r.lower())]
    return hits[0] if len(hits) == 1 else None


def _numbered(items: list[str]) -> str:
    return "".join(f"  [{i}] {s}\n" for i, s in enumerate(items, 1))


@builtin("questdlg")
def _questdlg(interp, args, nargout):
    if not args:
        raise MatlabError("questdlg needs a question, e.g. answer = questdlg('Continue?').")
    args = [a for a in args if not isinstance(a, Struct)]
    question = _lines(args[0])
    title = _opt_text(args, 1)
    strs = [_one_line(a) for a in args[2:]]
    buttons, default = ["Yes", "No", "Cancel"], "Yes"
    if len(strs) == 1:
        default = strs[0]
    elif len(strs) == 2:
        buttons, default = strs, strs[0]
    elif len(strs) in (3, 4):
        buttons, default = strs[:-1], strs[-1]
    elif len(strs) > 4:
        raise MatlabError("questdlg accepts at most three buttons followed by the default button.")
    if default not in buttons:
        default = buttons[0]
    ans = _ask_handler(interp, "questdlg", {"question": "\n".join(question), "title": title,
                                            "buttons": buttons, "default": default})
    if ans is not None:
        return char("" if ans is False else str(ans))
    head = (f"{title}: " if title else "") + "\n".join(question)
    if _can_prompt(interp):
        interp.out(f"{head}\n{_numbered(buttons)}")
        for _ in range(3):
            reply = _prompt(interp, f"Choose [{default}]: ")
            if reply is None:
                break
            if not reply.strip():
                return char(default)
            k = _pick(reply, buttons)
            if k is not None:
                return char(buttons[k])
            interp.out("Please type a number or one of the choices.\n")
        else:
            return char(default)
    _auto(interp, "questdlg", f"{_one_line(args[0])} -> {default} (default answer)")
    return char(default)


@builtin("inputdlg")
def _inputdlg(interp, args, nargout):
    prompts = _lines(args[0]) if args else ["Input:"]
    if not prompts:
        prompts = ["Input:"]
    title = _opt_text(args, 1)
    defaults = _lines(args[3]) if len(args) > 3 and not isinstance(args[3], Struct) else []
    defaults = (defaults + [""] * len(prompts))[:len(prompts)]
    ans = _ask_handler(interp, "inputdlg", {"prompts": prompts, "title": title, "defaults": defaults})
    if ans is not None:
        if ans is False:
            return Cell(np.empty((0, 0), dtype=object))
        vals = [str(a) for a in (ans if isinstance(ans, (list, tuple)) else [ans])]
        return _cell_col((vals + defaults[len(vals):])[:len(prompts)])
    if _can_prompt(interp):
        if title:
            interp.out(f"{title}\n")
        out = []
        for p, d in zip(prompts, defaults):
            label = p.rstrip().rstrip(":").rstrip()
            reply = _prompt(interp, f"{label} [{d}]: " if d else f"{label}: ")
            if reply is None:
                out = None
                break
            out.append(reply if reply.strip() else d)
        if out is not None:
            return _cell_col(out)
    shown = ", ".join(f"'{d}'" for d in defaults)
    _auto(interp, "inputdlg", f"{title or prompts[0]} -> {shown} (default answers)")
    return _cell_col(defaults)


@builtin("listdlg")
def _listdlg(interp, args, nargout):
    _, opts = _split_name_values(list(args), {"liststring", "selectionmode", "initialvalue", "promptstring",
                                              "name", "listsize", "okstring", "cancelstring", "uh", "fus", "ffs"})
    if "liststring" not in opts:
        raise MatlabError("listdlg needs the items: listdlg('ListString', {'a', 'b'}).")
    items = _lines(opts["liststring"])
    multiple = _one_line(opts["selectionmode"]).lower() != "single" if "selectionmode" in opts else True
    initial = [int(v) for v in np.real(fnum(opts["initialvalue"])).flatten()] if "initialvalue" in opts else [1]
    initial = [k for k in initial if 1 <= k <= len(items)] or ([1] if items else [])
    if not multiple:
        initial = initial[:1]
    prompt = _one_line(opts.get("promptstring")) or ("Select one or more items:" if multiple else "Select an item:")
    title = _one_line(opts.get("name"))

    def result(sel):
        idx = np.array([[float(k) for k in sel]]) if sel else _EMPTY
        return (idx, np.array([[1.0 if sel else 0.0]])) if nargout > 1 else idx

    ans = _ask_handler(interp, "listdlg", {"items": items, "prompt": prompt, "title": title,
                                           "multiple": multiple, "initial": initial,
                                           "ok": _one_line(opts.get("okstring")) or "OK",
                                           "cancel": _one_line(opts.get("cancelstring")) or "Cancel"})
    if ans is not None:
        sel = [] if ans is False else [int(k) for k in (ans if isinstance(ans, (list, tuple)) else [ans])]
        sel = [k for k in sel if 1 <= k <= len(items)]
        return result(sel if multiple else sel[:1])
    if _can_prompt(interp):
        interp.out((f"{title}\n" if title else "") + f"{prompt}\n{_numbered(items)}")
        shown = " ".join(str(k) for k in initial)
        for _ in range(3):
            reply = _prompt(interp, f"Number{'s' if multiple else ''} [{shown}] (c = cancel): ")
            if reply is None:
                break
            r = reply.strip().lower()
            if not r:
                return result(initial)
            if r in ("c", "cancel", "0"):
                return result([])
            parts = r.replace(",", " ").split()
            if all(p.isdigit() and 1 <= int(p) <= len(items) for p in parts):
                sel = sorted({int(p) for p in parts})
                return result(sel if multiple else sel[:1])
            interp.out(f"Please type numbers from 1 to {len(items)}.\n")
        else:
            return result([])
    names = ", ".join(items[k - 1] for k in initial)
    _auto(interp, "listdlg", f"{prompt} -> {names} (default selection)")
    return result(initial)


@builtin("menu")
def _menu(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("menu needs a title and at least one choice, e.g. k = menu('Pick', 'A', 'B').")
    title = _one_line(args[0])
    items: list[str] = []
    for a in args[1:]:
        items.extend(_lines(a) if isinstance(a, Cell) or not is_text(a) else [_one_line(a)])
    if not items:
        raise MatlabError("menu needs at least one choice.")
    ans = _ask_handler(interp, "menu", {"title": title, "items": items})
    if ans is not None:
        k = 0 if ans is False else int(ans)
        return np.array([[float(k if 0 <= k <= len(items) else 0)]])
    if _can_prompt(interp):
        interp.out(f"{title}\n{_numbered(items)}")
        for _ in range(3):
            reply = _prompt(interp, "Choose a number: ")
            if reply is None:
                break
            k = _pick(reply, items) if reply.strip() else None
            if k is not None:
                return np.array([[float(k + 1)]])
            interp.out(f"Please type a number from 1 to {len(items)}.\n")
        else:
            return _ZERO.copy()
    _auto(interp, "menu", f"{title} -> {items[0]} (first choice)")
    return np.array([[1.0]])


# -- files and folders ---------------------------------------------------------------------------------

def _filters(v) -> tuple[list, str]:
    """([[pattern, description], ...], default file name) from the first uigetfile/uiputfile input."""
    if v is None or (isinstance(v, np.ndarray) and v.size == 0):
        return [["*.*", "All files"]], ""
    if isinstance(v, Cell):
        rows = v.data if v.data.ndim == 2 else v.data.reshape(-1, 1)
        if rows.shape[0] == 1 and rows.shape[1] > 2:
            rows = rows.reshape(-1, 1)
        out = []
        for r in rows:
            pat = _one_line(r[0])
            out.append([pat, _one_line(r[1]) if len(r) > 1 else pat])
        return out or [["*.*", "All files"]], ""
    s = _one_line(v)
    if "*" in s or "?" in s:
        return [[s, s]], ""
    if s.startswith(".") and os.sep not in s and "/" not in s:
        return [["*" + s, "*" + s]], ""
    ext = os.path.splitext(s)[1]
    return [["*" + ext if ext else "*.*", "*" + ext if ext else "All files"]], s


def _file_result(path, nargout: int):
    """(file, folder, filterindex) for a chosen path, or zeros for Cancel."""
    if not path:
        return (_ZERO, _ZERO, _ZERO)[:nargout] if nargout > 1 else _ZERO
    if isinstance(path, (list, tuple)):
        full = [os.path.abspath(os.path.expanduser(str(p))) for p in path]
        name: Any = Cell.from_list([char(os.path.basename(p)) for p in full])
        folder = os.path.dirname(full[0])
    else:
        full1 = os.path.abspath(os.path.expanduser(str(path)))
        name, folder = char(os.path.basename(full1)), os.path.dirname(full1)
    out = (name, char(folder.rstrip(os.sep) + os.sep), np.array([[1.0]]))
    return out[:nargout] if nargout > 1 else out[0]


def _file_dialog(interp, args, nargout, kind: str):
    args, opts = _split_name_values(list(args), {"multiselect"})
    filters, default = _filters(args[0] if args else None)
    title = _opt_text(args, 1) or ("Select a file to open" if kind == "uigetfile" else "Save as")
    if len(args) > 2 and is_text(args[2]):
        default = _one_line(args[2])
    multi = "multiselect" in opts and _one_line(opts["multiselect"]).lower() == "on"
    payload = {"title": title, "filters": filters, "default": default}
    if kind == "uigetfile":
        payload["multiselect"] = multi
    ans = _ask_handler(interp, kind, payload)
    if ans is not None:
        return _file_result(None if ans is False else ans, nargout)
    must_exist = kind == "uigetfile"
    if _can_prompt(interp):
        pats = "; ".join(f[0] for f in filters)
        for _ in range(3):
            hint = f" [{default}]" if default else " (empty = cancel)"
            reply = _prompt(interp, f"{title} ({pats}){hint}: ")
            if reply is None:
                break
            path = reply.strip().strip("'\"") or default
            if not path:
                return _file_result(None, nargout)
            paths = [p.strip() for p in path.split(";") if p.strip()] if multi else [path]
            missing = [p for p in paths if must_exist and not os.path.isfile(os.path.expanduser(p))]
            if not missing:
                return _file_result(paths if len(paths) > 1 else paths[0], nargout)
            interp.out(f"File '{missing[0]}' not found.\n")
        else:
            return _file_result(None, nargout)
    if default and (not must_exist or os.path.isfile(os.path.expanduser(default))):
        _auto(interp, kind, f"{title} -> {default} (default file)")
        return _file_result(default, nargout)
    _auto(interp, kind, f"{title} -> cancelled (no interactive input and no usable default file)")
    return _file_result(None, nargout)


builtin("uigetfile")(lambda interp, args, nargout: _file_dialog(interp, args, nargout, "uigetfile"))
builtin("uiputfile")(lambda interp, args, nargout: _file_dialog(interp, args, nargout, "uiputfile"))


@builtin("uigetdir")
def _uigetdir(interp, args, nargout):
    start = _one_line(args[0]) if args and is_text(args[0]) else ""
    title = _opt_text(args, 1) or "Select a folder"
    start_dir = os.path.abspath(os.path.expanduser(start)) if start and os.path.isdir(os.path.expanduser(start)) \
        else os.getcwd()
    ans = _ask_handler(interp, "uigetdir", {"title": title, "start": start_dir})
    if ans is not None:
        return _ZERO.copy() if ans is False or not ans else char(os.path.abspath(os.path.expanduser(str(ans))))
    if _can_prompt(interp):
        for _ in range(3):
            reply = _prompt(interp, f"{title} [{start_dir}] (c = cancel): ")
            if reply is None:
                break
            r = reply.strip().strip("'\"")
            if r.lower() in ("c", "cancel"):
                return _ZERO.copy()
            path = os.path.abspath(os.path.expanduser(r)) if r else start_dir
            if os.path.isdir(path):
                return char(path)
            interp.out(f"Folder '{r}' not found.\n")
        else:
            return _ZERO.copy()
    _auto(interp, "uigetdir", f"{title} -> {start_dir} (default folder)")
    return char(start_dir)


# -- waiting and mouse input ---------------------------------------------------------------------------

@builtin("waitforbuttonpress")
def _waitforbuttonpress(interp, args, nargout):
    plt = _plt()
    st = _state(interp)
    if st.figures and not _noninteractive() and plt.isinteractive():
        key = plt.waitforbuttonpress()
        return np.array([[1.0 if key else 0.0]])
    return _ZERO.copy()


@builtin("ginput")
def _ginput(interp, args, nargout):
    n = None
    if args and isinstance(args[0], np.ndarray) and args[0].size == 1 and not is_char(args[0]):
        n = int(np.real(fnum(args[0])).flat[0])
        if n < 1:
            raise MatlabError("ginput: the number of points must be a positive integer.")
    st = _state(interp)
    pts = _ask_handler(interp, "ginput", {"n": n, "figure": st.current})
    if pts is None:
        plt = _plt()
        if _noninteractive() or not plt.isinteractive():
            raise MatlabError("ginput reads mouse clicks from a figure window, which is not available here. "
                              "Run the script with a plot window, or use fixed coordinates instead.")
        from .plotting import _figure
        fig = _figure(interp)
        pts = [(x, y, 1) for x, y in fig.ginput(n if n else -1, timeout=0)]
    rows = [list(p) + [1] * (3 - len(p)) for p in pts]
    cols = [np.array([[float(r[k])] for r in rows]).reshape(-1, 1) for k in range(3)]
    if not rows:
        cols = [np.zeros((0, 1))] * 3
    return tuple(cols[:nargout]) if nargout > 1 else np.hstack(cols[:2])
