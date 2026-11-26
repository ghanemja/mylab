"""Deferred imports, so that starting mylab does not pay for libraries a script never uses.

``sp = lazy_module("scipy.special", globals(), "sp")`` gives a stand-in that imports the real
module the first time one of its attributes is used.  When the stand-in is given the importing
module's ``globals()`` and the name it is bound to, it replaces itself there with the real module
on first use, so later uses cost nothing extra.

Nothing here may import anything heavy: this module is loaded on every start.
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stand-in for a module; the import happens on first attribute access."""

    __slots__ = ("_lazy_name", "_lazy_module", "_lazy_namespace", "_lazy_alias")

    def __init__(self, name: str, namespace: dict | None = None, alias: str | None = None):
        object.__setattr__(self, "_lazy_name", name)
        object.__setattr__(self, "_lazy_module", None)
        object.__setattr__(self, "_lazy_namespace", namespace)
        object.__setattr__(self, "_lazy_alias", alias)

    def _lazy_load(self) -> Any:
        module = self._lazy_module
        if module is None:
            module = importlib.import_module(self._lazy_name)
            object.__setattr__(self, "_lazy_module", module)
            namespace, alias = self._lazy_namespace, self._lazy_alias
            if namespace is not None and alias and namespace.get(alias) is self:
                namespace[alias] = module  # later lookups go straight to the module
        return module

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._lazy_load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._lazy_load(), attr, value)

    def __dir__(self) -> list[str]:
        return dir(self._lazy_load())

    def __repr__(self) -> str:
        state = "loaded" if self._lazy_module is not None else "not loaded yet"
        return f"<lazy module {self._lazy_name!r} ({state})>"


def lazy_module(name: str, namespace: dict | None = None, alias: str | None = None) -> Any:
    """A stand-in for module ``name`` that is imported on first use."""
    return LazyModule(name, namespace, alias)


def lazy_function(module: str, attr: str) -> Any:
    """A callable that looks up ``module.attr`` (importing the module) the first time it is called.

    For tables built at import time, e.g. ``{"erf": lazy_function("scipy.special", "erf")}``.
    """
    target = None

    def call(*args: Any, **kwargs: Any) -> Any:
        nonlocal target
        if target is None:
            target = getattr(importlib.import_module(module), attr)
        return target(*args, **kwargs)

    call.__name__ = call.__qualname__ = attr
    return call
