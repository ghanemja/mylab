from __future__ import annotations

from typing import Callable

BUILTINS: dict[str, Callable] = {}

# Every ``@builtin`` registration of each name, in import order: (module, function).
# ``BUILTINS[name]`` keeps the last one; this list lets tests find names that more
# than one module defines (only one implementation per name should exist).
REGISTRATIONS: dict[str, list[tuple[str, Callable]]] = {}


def builtin(*names: str):
    """Register a function under one or more MATLAB names."""

    def deco(fn: Callable) -> Callable:
        module = getattr(fn, "__module__", "") or ""
        for n in names:
            BUILTINS[n] = fn
            REGISTRATIONS.setdefault(n, []).append((module, fn))
        fn.matlab_names = names  # type: ignore[attr-defined]
        return fn

    return deco


def registered_by_several_modules() -> dict[str, list[str]]:
    """Names that ``@builtin`` registrations from more than one module define, with those modules."""
    out: dict[str, list[str]] = {}
    for name, regs in REGISTRATIONS.items():
        modules = list(dict.fromkeys(m for m, _ in regs))
        if len(modules) > 1:
            out[name] = modules
    return out
