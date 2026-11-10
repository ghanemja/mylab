from __future__ import annotations

from typing import Callable

BUILTINS: dict[str, Callable] = {}


def builtin(*names: str):
    """Register a function under one or more MATLAB names."""

    def deco(fn: Callable) -> Callable:
        for n in names:
            BUILTINS[n] = fn
        fn.matlab_names = names  # type: ignore[attr-defined]
        return fn

    return deco
