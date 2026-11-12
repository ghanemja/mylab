"""Debugging tools: an execution tracer (for step-through views) and an interactive debugger.

Tracer - used by the browser app's step-through panel::

    from mylab.api import Session
    from mylab.debugger import trace
    t = trace(Session(), "x = 1;\\nfor k = 1:3\\n  x = x * 2;\\nend")
    t.steps[2]            # {"line": 3, "function": "base", "depth": 1, "frame": 0, "set": {...}, ...}
    t.workspace_at(2)     # variables as they were just before step 2 ran

Each step is recorded just *before* a statement runs. ``set`` holds the
variables that changed since the previous step in the same workspace (name ->
{size, class, preview}), ``del`` the ones that were cleared, and ``out`` the
number of characters printed so far, so a viewer can replay the run forwards
and backwards.

Interactive debugger - ``mylab --debug script.m`` or ``dbstop``/``keyboard``
in a terminal session. Commands: n(ext), s(tep), c(ontinue), o(ut), b LINE,
d LINE, w(here), l(ist), q(uit), h(elp); anything else is run as code in the
paused workspace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from . import ast as A
from .display import size_str
from .errors import MatlabError
from .values import class_name, shape_of


class StopExecution(MatlabError):
    """Raised by the debugger's quit command to end the run."""


def _info(value: Any) -> dict:
    from .api import _preview
    try:
        return {"size": size_str(shape_of(value)), "class": class_name(value), "preview": _preview(value)}
    except Exception:  # noqa: BLE001 - a preview must never break a traced run
        return {"size": "?", "class": type(value).__name__, "preview": ""}


def _assigned_names(stmt: A.Node) -> list[str]:
    names = []
    if isinstance(stmt, A.Assign):
        for target in stmt.targets:
            node = target
            while not isinstance(node, A.Ident) and hasattr(node, "base"):
                node = node.base
            if isinstance(node, A.Ident):
                names.append(node.name)
    elif isinstance(stmt, A.For):
        names.append(stmt.var.name)
    elif isinstance(stmt, A.ExprStmt):
        names.append("ans")
    return names


@dataclass
class TraceResult:
    steps: list[dict] = field(default_factory=list)
    output: str = ""
    error: str | None = None
    error_line: int | None = None
    truncated: bool = False
    figures: list[bytes] = field(default_factory=list)

    def workspace_at(self, index: int) -> dict[str, dict]:
        """Variables of the workspace that step ``index`` runs in, as they were before it ran."""
        if not self.steps:
            return {}
        index = max(0, min(index, len(self.steps) - 1))
        frame = self.steps[index]["frame"]
        ws: dict[str, dict] = {}
        for step in self.steps[: index + 1]:
            if step["frame"] == frame:
                ws.update(step["set"])
                for name in step["del"]:
                    ws.pop(name, None)
        return ws

    def to_dict(self) -> dict:
        import base64
        return {"steps": self.steps, "output": self.output, "error": self.error, "error_line": self.error_line,
                "truncated": self.truncated,
                "figures": [base64.b64encode(f).decode("ascii") for f in self.figures]}


def trace(session, code: str, max_steps: int = 5000, max_vars: int = 60) -> TraceResult:
    """Run ``code`` in ``session`` and record every statement and the variable changes it caused."""
    interp = session.interp
    result = TraceResult()
    frame_ids: dict[int, int] = {}
    seen: dict[int, dict[str, int]] = {}      # frame serial -> name -> id(value) at last record
    dirty: dict[int, set[str]] = {}           # names assigned by the previous statement in that frame
    keep_alive: list[Any] = []                # frames, so id() values are not reused during the run

    def hook(stmt: A.Node, frame) -> None:
        if len(result.steps) >= max_steps:
            result.truncated = True
            return
        key = frame_ids.get(id(frame))
        if key is None:
            key = frame_ids[id(frame)] = len(frame_ids)
            keep_alive.append(frame)
            seen[key] = {}
            dirty[key] = set()
        known = seen[key]
        changed: dict[str, dict] = {}
        names = list(frame.vars)[:max_vars] + [n for n in frame.globals if frame.has(n)]
        for name in names:
            value = frame.get(name)
            if known.get(name) != id(value) or name in dirty[key]:
                known[name] = id(value)
                changed[name] = _info(value)
        deleted = [n for n in known if n not in names]
        for n in deleted:
            del known[n]
        dirty[key] = set(_assigned_names(stmt))
        result.steps.append({
            "line": getattr(stmt, "line", 0) or 0,
            "file": os.path.basename(frame.file) if frame.file else None,
            "function": frame.name,
            "depth": len(interp.frames),
            "frame": key,
            "set": changed,
            "del": deleted,
            "out": sum(len(s) for s in session._buffer),
        })

    previous = interp.trace_hook
    interp.trace_hook = hook
    try:
        run = session.run(code)
    finally:
        interp.trace_hook = previous
    result.output = run.output
    result.error = run.error
    result.error_line = run.error_line
    result.figures = run.figures
    # A final step showing the base workspace after the last statement.
    if not result.truncated:
        key = frame_ids.get(id(interp.base))
        if key is not None:
            known = seen[key]
            changed = {}
            for name, value in list(interp.base.vars.items())[:max_vars]:
                if known.get(name) != id(value) or name in dirty[key]:
                    changed[name] = _info(value)
            result.steps.append({"line": 0, "file": None, "function": "base", "depth": 1, "frame": key,
                                 "set": changed, "del": [n for n in known if n not in interp.base.vars],
                                 "out": len(result.output), "end": True})
    return result


# -- interactive debugger ---------------------------------------------------------------------

HELP_TEXT = """Debugger commands:
  n, next        run the current line, stepping over function calls
  s, step        run the current line, stepping into functions
  o, out         run until the current function returns
  c, continue    run until the next breakpoint
  b LINE         set a breakpoint at LINE of the current file (b FILE LINE for another file)
  d LINE         delete a breakpoint         (d alone deletes all)
  w, where       show the call stack
  l, list        show the source around the current line
  q, quit        stop the program
  anything else  is run as code in the paused workspace, e.g.  x  or  disp(size(A))
If a variable has the same name as a shortcut (n, s, c, b ...), typing it shows the variable;
use the long form (next, step, continue, break) for the command.
"""


class Debugger:
    """Pauses a running program at breakpoints and lets the user inspect and step."""

    def __init__(self, interp, read=None, write=None):
        self.interp = interp
        self.read = read or interp.input_func
        self.write = write or interp.out
        self.breakpoints: set[tuple[str | None, int]] = set()
        self.mode = "run"           # run | step | next | out
        self.depth = 0
        self.sources: dict[str | None, list[str]] = {}
        self.active = False

    def attach(self, stop_at_start: bool = False) -> None:
        if stop_at_start:
            self.mode = "step"
        self.interp.trace_hook = self.hook
        self.interp.state["debugger"] = self

    def detach(self) -> None:
        if self.interp.trace_hook == self.hook:
            self.interp.trace_hook = None

    def set_source(self, file: str | None, text: str) -> None:
        self.sources[os.path.basename(file) if file else None] = text.splitlines()

    def add_breakpoint(self, file: str | None, line: int) -> None:
        if file:
            file = os.path.basename(file)
            if not file.endswith(".m"):
                file += ".m"
        self.breakpoints.add((file, int(line)))

    def _lines(self, frame) -> list[str]:
        key = os.path.basename(frame.file) if frame.file else None
        if key not in self.sources and frame.file and os.path.isfile(frame.file):
            with open(frame.file, encoding="utf-8", errors="replace") as fh:
                self.sources[key] = fh.read().splitlines()
        return self.sources.get(key, [])

    def _should_stop(self, stmt: A.Node, frame) -> bool:
        line = getattr(stmt, "line", 0) or 0
        fname = os.path.basename(frame.file) if frame.file else None
        if (fname, line) in self.breakpoints or (None, line) in self.breakpoints and fname is None:
            return True
        depth = len(self.interp.frames)
        if self.mode == "step":
            return True
        if self.mode == "next":
            return depth <= self.depth
        if self.mode == "out":
            return depth < self.depth
        return False

    def hook(self, stmt: A.Node, frame) -> None:
        if self.active or not self._should_stop(stmt, frame):
            return
        self.pause(frame, getattr(stmt, "line", 0) or 0)

    def pause(self, frame, line: int) -> None:
        self.active = True
        try:
            self._prompt(frame, line)
        finally:
            self.active = False

    def _show_line(self, frame, line: int) -> None:
        lines = self._lines(frame)
        where = os.path.basename(frame.file) if frame.file else frame.name
        text = lines[line - 1].rstrip() if 0 < line <= len(lines) else ""
        self.write(f"{where}:{line}  {text}\n")

    def _prompt(self, frame, line: int) -> None:
        self._show_line(frame, line)
        while True:
            try:
                raw = self.read("K>> ")
            except (EOFError, KeyboardInterrupt):
                raw = "q"
            cmd = raw.strip()
            word, _, rest = cmd.partition(" ")
            rest = rest.strip()
            if cmd and frame.has(cmd):
                # A variable with the same name as a shortcut (n, s, c, b ...) wins: show it.
                # The long forms next / step / continue / dbstep / dbcont always work.
                self._run_code(cmd, frame)
                continue
            if word in ("n", "next", "dbstep") and rest in ("", "in", "out"):
                self.mode = {"": "next", "in": "step", "out": "out"}[rest]
                self.depth = len(self.interp.frames)
                return
            if word in ("s", "step"):
                self.mode = "step"
                return
            if word in ("o", "out"):
                self.mode, self.depth = "out", len(self.interp.frames)
                return
            if word in ("c", "continue", "dbcont"):
                self.mode = "run"
                return
            if word in ("q", "quit", "dbquit", "exit"):
                self.mode = "run"
                self.detach()
                raise StopExecution("Program stopped in the debugger.")
            if word in ("h", "help", "?"):
                self.write(HELP_TEXT)
            elif word in ("b", "break"):
                parts = rest.split()
                try:
                    if len(parts) == 1:
                        self.add_breakpoint(frame.file, int(parts[0]))
                    else:
                        self.add_breakpoint(parts[0], int(parts[1]))
                    self.write("Breakpoint set.\n")
                except (ValueError, IndexError):
                    self.write("Usage: b LINE   or   b FILE LINE\n")
            elif word in ("d", "delete"):
                if not rest:
                    self.breakpoints.clear()
                else:
                    self.breakpoints = {b for b in self.breakpoints if str(b[1]) != rest}
                self.write("Breakpoint(s) removed.\n")
            elif word in ("w", "where", "dbstack"):
                for depth, f in enumerate(reversed(self.interp.frames)):
                    marker = ">" if depth == 0 else " "
                    self.write(f"{marker} {f.name}" + (f"  ({os.path.basename(f.file)})" if f.file else "") + "\n")
            elif word in ("l", "list"):
                lines = self._lines(frame)
                for k in range(max(1, line - 4), min(len(lines), line + 4) + 1):
                    self.write(f"{'->' if k == line else '  '} {k:4d}  {lines[k - 1]}\n")
            elif cmd:
                self._run_code(cmd, frame)

    def _run_code(self, code: str, frame) -> None:
        from .parser import parse
        try:
            self.interp.run_program(parse(code), frame)
        except StopExecution:
            raise
        except MatlabError as e:
            self.write(f"Error: {e.message}\n")


def get_debugger(interp, create: bool = True) -> Debugger | None:
    dbg = interp.state.get("debugger")
    if dbg is None and create:
        dbg = Debugger(interp)
        interp.state["debugger"] = dbg
    return dbg
