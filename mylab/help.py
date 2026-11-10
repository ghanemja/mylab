"""Help text lookup for built-in and user functions.

Entries live in :mod:`mylab.help_data` as ``HELP[name] = (usage, description)``,
written in the project's own words (see CLEANROOM.md).
"""

from __future__ import annotations

import re
from typing import Any


def _entries() -> dict:
    try:
        from .help_data import HELP
    except ImportError:  # pragma: no cover
        return {}
    return HELP


def _leading_comments(path: str) -> str:
    """The comment block at the top of a user .m file (its help text)."""
    lines = []
    with open(path, encoding="utf-8") as fh:
        started = False
        for raw in fh:
            s = raw.strip()
            if s.startswith("function") and not started:
                continue
            if s.startswith("%"):
                started = True
                lines.append(re.sub(r"^%+ ?", "", s))
            elif started or s:
                break
    return "\n".join(lines)


def help_text(name: str, interp: Any = None) -> str:
    entry = _entries().get(name)
    if entry:
        usage, description = entry
        usages = usage if isinstance(usage, (list, tuple)) else [usage]
        return "\n".join(f"  {u}" for u in usages) + "\n\n" + _wrap(description) + "\n"
    if interp is not None:
        found = interp.find_file_function(name)
        if found:
            doc = _leading_comments(found[1])
            return (doc or f"{name} is defined in {found[1]} (no help comments).") + "\n"
        if name in interp.base.functions:
            return f"{name} is a function defined in the current session.\n"
        if name in interp.builtins:
            return f"{name} is a built-in function. (No help text written yet.)\n"
    return f"{name} not found.\n"


def summary(name: str) -> str | None:
    entry = _entries().get(name)
    if not entry:
        return None
    return entry[1].split(". ")[0].rstrip(".") + "."


def _wrap(text: str, width: int = 76) -> str:
    import textwrap
    paras = text.split("\n")
    return "\n".join(textwrap.fill(p, width, initial_indent="  ", subsequent_indent="  ") if p else ""
                     for p in paras)
