"""Debugger commands: keyboard, dbstop, dbclear, dbcont, dbstep, dbquit, dbstack, dbstatus."""

from __future__ import annotations

import sys

import numpy as np

from ..debugger import get_debugger
from ..values import Struct, char
from ._registry import builtin
from ._util import is_text, text


def _interactive(interp) -> bool:
    if interp.state.get("debug_interactive") is not None:
        return bool(interp.state["debug_interactive"])
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _not_interactive(interp, what: str) -> None:
    interp.warn(f"{what} needs an interactive session and was ignored. In a terminal run "
                "'mylab --debug file.m'; in the web app use the Step through button.")


@builtin("keyboard")
def _keyboard(interp, args, nargout):
    if not _interactive(interp):
        return None
    dbg = get_debugger(interp)
    dbg.attach()
    dbg.mode = "step"  # pause at the next statement
    return None


@builtin("dbstop")
def _dbstop(interp, args, nargout):
    words = [text(a) for a in args if is_text(a)]
    if not _interactive(interp):
        _not_interactive(interp, "dbstop")
        return None
    dbg = get_debugger(interp)
    if words[:2] == ["if", "error"] or not words:
        interp.state["dbstop_if_error"] = True
        return None
    if words[0] == "in":
        words = words[1:]
    file = words[0] if words else None
    line = 1
    if "at" in words:
        line = int(words[words.index("at") + 1])
    elif len(words) > 1 and words[1].isdigit():
        line = int(words[1])
    dbg.add_breakpoint(file, line)
    dbg.attach()
    return None


@builtin("dbclear")
def _dbclear(interp, args, nargout):
    dbg = get_debugger(interp, create=False)
    if dbg is not None:
        dbg.breakpoints.clear()
    interp.state.pop("dbstop_if_error", None)
    return None


@builtin("dbstatus")
def _dbstatus(interp, args, nargout):
    dbg = get_debugger(interp, create=False)
    points = sorted(dbg.breakpoints, key=lambda b: (b[0] or "", b[1])) if dbg else []
    if nargout:
        data = np.empty((len(points), 1), dtype=object)
        for k, (f, line) in enumerate(points):
            data[k, 0] = {"name": char(f or ""), "line": np.array([[float(line)]])}
        return Struct(data, ["name", "line"])
    for f, line in points:
        interp.out(f"Breakpoint in {f or 'this file'} at line {line}\n")
    return None


@builtin("dbstack")
def _dbstack(interp, args, nargout):
    frames = list(reversed(interp.frames))
    if nargout:
        data = np.empty((len(frames), 1), dtype=object)
        for k, f in enumerate(frames):
            data[k, 0] = {"file": char(f.file or ""), "name": char(f.name), "line": np.array([[0.0]])}
        return Struct(data, ["file", "name", "line"])
    for f in frames:
        interp.out(f"  In {f.name}\n")
    return None


@builtin("dbcont", "dbstep", "dbquit", "dbup", "dbdown")
def _only_when_paused(interp, args, nargout):
    interp.warn("This command only works at the K>> prompt while the program is paused.")
    return None
