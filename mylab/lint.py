"""Static checks for editors: syntax errors and calls to unknown functions.

    from mylab.lint import check_source
    check_source("x = foo(1);\\n")   # -> [{"line": 1, "column": 5, "severity": "warning", "message": ...}]

Nothing is executed. Used by ``mylab --check FILE`` (JSON output) and the VS Code extension.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import ast as A
from .errors import MatlabError
from .parser import parse

_POS = re.compile(r"line (\d+), column (\d+): (.*)", re.S)


def _walk(node, visit) -> None:
    visit(node)
    if isinstance(node, A.Node):
        for v in vars(node).values():
            _walk(v, visit)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk(item, visit)
    elif isinstance(node, dict):
        for item in node.values():
            _walk(item, visit)


def check_source(text: str, filename: str | None = None) -> list[dict]:
    """Diagnostics for one file: a syntax error, or warnings for functions that can't be found."""
    try:
        program = parse(text)
    except MatlabError as e:
        m = _POS.search(e.message)
        line, col, msg = (int(m.group(1)), int(m.group(2)), m.group(3)) if m else (1, 1, e.message)
        return [{"line": line, "column": col, "severity": "error", "message": msg}]
    except RecursionError:
        return [{"line": 1, "column": 1, "severity": "error", "message": "The file is nested too deeply."}]

    from .builtins import BUILTINS
    known = set(BUILTINS)
    folder = Path(filename).resolve().parent if filename else Path(os.getcwd())
    try:
        known |= {p.stem for p in folder.glob("*.m")}
    except OSError:
        pass
    assigned: set[str] = set()
    calls: list[tuple[str, int]] = []
    uses_dynamic = False

    def visit(n) -> None:
        nonlocal uses_dynamic
        t = type(n)
        if t is A.Function:
            known.add(n.name)
            assigned.update(n.params)
            assigned.update(n.outputs)
            assigned.update(e.name.split(".")[0] for e in n.arguments)
        elif t is A.ClassDef:
            known.add(n.name)
            known.update(m for m in n.methods if "." not in m)
            assigned.update(p.name for p in n.properties)
        elif t is A.Assign:
            for target in n.targets:
                root = target
                while not isinstance(root, A.Ident) and hasattr(root, "base"):
                    root = root.base
                if isinstance(root, A.Ident):
                    assigned.add(root.name)
        elif t is A.For:
            assigned.add(n.var.name)
        elif t is A.Try and n.ident:
            assigned.add(n.ident)
        elif t is A.Global:
            assigned.update(n.names)
        elif t is A.Lambda:
            assigned.update(n.params)
        elif t is A.Index and isinstance(n.base, A.Ident):
            calls.append((n.base.name, n.base.line or n.line or 0))
            if n.base.name in ("load", "eval", "evalin", "assignin", "run", "import", "syms"):
                uses_dynamic = True
        elif t is A.FuncHandle:
            calls.append((n.name.split(".")[0], getattr(n, "line", 0) or 0))

    _walk(program, visit)
    if uses_dynamic:  # variables may be created at run time; unknown names can't be judged
        return []
    out, reported = [], set()
    for name, line in calls:
        if name in known or name in assigned or name in reported:
            continue
        reported.add(name)
        out.append({"line": max(line, 1), "column": 1, "severity": "warning",
                    "message": f"'{name}' is not a function mylab provides, and no {name}.m was found "
                               "in this folder."})
    return out
