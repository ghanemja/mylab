"""Bus signals: several named signals carried on one line.

A bus is a :class:`BusSignal`, a flat float vector (the elements one after
the other, each in column order) that also knows the names and sizes of its
elements (``spec``). Blocks that do arithmetic see the flat vector, like a
Mux output. Blocks that understand buses (Bus Selector, Bus Assignment, a
MATLAB Function block, To Workspace) read the names. An element can itself
be a bus.

``spec`` is a list of ``(name, shape, inner)`` where ``inner`` is the spec of
a nested bus or None. Arithmetic on a bus gives a bus without a spec, which
counts as a plain vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .library import BlockError, sig


class BusSignal(np.ndarray):
    spec: list | None = None

    def __array_finalize__(self, obj):
        self.spec = None  # slices and results of arithmetic are plain vectors


def spec_of(v: Any) -> list | None:
    return getattr(v, "spec", None)


def size_of(spec: list) -> int:
    return sum(int(np.prod(shape)) if inner is None else size_of(inner) for _n, shape, inner in spec)


def make(names: list[str], values: list) -> BusSignal:
    """A bus from element names and values (values may be buses themselves)."""
    spec = []
    flat = []
    for name, v in zip(names, values):
        inner = spec_of(v)
        a = np.asarray(v, dtype=float)
        spec.append((name, a.shape if inner is None else (a.size,), inner))
        flat.append(np.ravel(a, order="F"))
    data = np.concatenate(flat) if flat else np.zeros(0)
    out = data.view(BusSignal)
    out.spec = spec
    return out


def with_spec(data: np.ndarray, spec: list | None) -> np.ndarray:
    if spec is None:
        return data
    out = np.asarray(data, dtype=float).ravel(order="F").copy().view(BusSignal)
    out.spec = spec
    return out


def copy(v: Any) -> np.ndarray:
    """A copy of a signal that keeps the element names of a bus."""
    return with_spec(np.array(v, dtype=float, copy=True), spec_of(v))


def names(spec: list, prefix: str = "") -> list[str]:
    """All element names, nested ones as 'outer.inner'."""
    out = []
    for n, _shape, inner in spec:
        out.append(prefix + n)
        if inner is not None:
            out += names(inner, prefix + n + ".")
    return out


def _locate(spec: list, path: str) -> tuple[int, tuple, list | None]:
    """(offset, shape, inner spec) of the element at a dotted path."""
    parts = path.split(".")
    off = 0
    cur = spec
    for i, part in enumerate(parts):
        if cur is None:
            raise KeyError(path)
        for n, shape, inner in cur:
            size = int(np.prod(shape)) if inner is None else size_of(inner)
            if n == part:
                if i == len(parts) - 1:
                    return off, shape, inner
                cur = inner
                break
            off += size
        else:
            raise KeyError(path)
    raise KeyError(path)


def select(bus: np.ndarray, path: str) -> np.ndarray:
    spec = spec_of(bus)
    if spec is None:
        raise BlockError("the input is not a bus (it carries no signal names)")
    try:
        off, shape, inner = _locate(spec, path)
    except KeyError:
        raise BlockError(f"the bus has no signal '{path}'; it has: {', '.join(names(spec))}") from None
    flat = np.asarray(bus, dtype=float).ravel()
    if inner is not None:
        return with_spec(flat[off : off + size_of(inner)], inner)
    part = flat[off : off + int(np.prod(shape))]
    return part.reshape(shape, order="F") if len(shape) == 2 and min(shape) > 1 else part.copy()


def assign(bus: np.ndarray, path: str, value: Any) -> np.ndarray:
    spec = spec_of(bus)
    if spec is None:
        raise BlockError("the first input is not a bus (it carries no signal names)")
    try:
        off, shape, inner = _locate(spec, path)
    except KeyError:
        raise BlockError(f"the bus has no signal '{path}'; it has: {', '.join(names(spec))}") from None
    size = int(np.prod(shape)) if inner is None else size_of(inner)
    v = np.ravel(np.asarray(value, dtype=float), order="F")
    if v.size == 1 and size > 1:
        v = np.full(size, v[0])
    if v.size != size:
        raise BlockError(f"the signal '{path}' has {size} element(s), but the value given for it has {v.size}")
    flat = np.asarray(bus, dtype=float).ravel().copy()
    flat[off : off + size] = v
    return with_spec(flat, spec)


def to_struct(bus: np.ndarray) -> Any:
    """A bus as a MATLAB struct (nested buses become nested structs)."""
    from ..values import Struct

    spec = spec_of(bus)
    flat = np.asarray(bus, dtype=float).ravel()
    s = Struct()
    off = 0
    for n, shape, inner in spec or []:
        if inner is not None:
            size = size_of(inner)
            s.set_field(0, n, to_struct(with_spec(flat[off : off + size], inner)))
        else:
            size = int(np.prod(shape))
            part = flat[off : off + size]
            val = part.reshape(shape, order="F") if len(shape) == 2 else part.reshape(-1, 1)
            s.set_field(0, n, np.array(val, dtype=float))
        off += size
    return s


def from_struct(v: Any) -> BusSignal:
    """A scalar MATLAB struct with numeric (or struct) fields as a bus."""
    from ..values import Struct

    if not isinstance(v, Struct) or v.numel() != 1:
        raise BlockError("a bus must come from a scalar struct")
    vals = []
    for f in v.fields:
        x = v.get_field(0, f)
        vals.append(from_struct(x) if isinstance(x, Struct) else sig(x))
    return make(list(v.fields), vals)


def element_names_from_object(obj: Any) -> list[str]:
    """Element names of a bus object (a struct with an Elements array whose entries have a Name)."""
    from ..values import Struct, to_str

    if not isinstance(obj, Struct) or "Elements" not in obj.fields:
        raise BlockError("this is not a bus object (made with Simulink.Bus)")
    els = obj.get_field(0, "Elements")
    if not isinstance(els, Struct):
        return []
    return [to_str(els.get_field(i, "Name")) for i in range(els.numel())]
