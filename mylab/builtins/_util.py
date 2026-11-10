"""Helpers shared by built-in implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import (
    INT_TYPES, MString, is_char, saturate_int, to_float_array, to_int, to_numeric, to_str,
)


def nargs(args: list, lo: int, hi: int | None, name: str) -> None:
    if len(args) < lo:
        raise MatlabError(f"Not enough input arguments for {name}.")
    if hi is not None and len(args) > hi:
        raise MatlabError(f"Too many input arguments for {name}.")


def num(v: Any) -> np.ndarray:
    """Numeric array (chars -> codes, logical -> double)."""
    return to_numeric(v)


def fnum(v: Any) -> np.ndarray:
    """Floating point array."""
    return to_float_array(v)


def is_text(v: Any) -> bool:
    return isinstance(v, MString) or is_char(v)


def text(v: Any, what: str = "argument") -> str:
    return to_str(v, what)


def dims_from_args(args: list, name: str) -> tuple[tuple[int, ...], str | None]:
    """Parse size arguments like zeros(n), zeros(m, n), zeros([m n]), zeros(..., 'int32')."""
    cls = None
    nums = []
    i = 0
    while i < len(args):
        a = args[i]
        if is_text(a):
            s = text(a)
            if s == "like":
                like = args[i + 1]
                cls = like.dtype.name if isinstance(like, np.ndarray) else None
                i += 2
                continue
            cls = s
        else:
            nums.append(a)
        i += 1
    if not nums:
        return (1, 1), cls
    if len(nums) == 1:
        v = num(nums[0]).flatten()
        if v.size == 1:
            n = max(int(v[0]), 0)
            return (n, n), cls
        dims = [max(int(x), 0) for x in v]
    else:
        dims = []
        for x in nums:
            a = num(x)
            if a.size != 1:
                raise MatlabError(f"Size inputs to {name} must be scalar.")
            dims.append(max(int(np.real(a.flat[0])), 0))
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    if len(dims) == 1:
        dims.append(dims[0])
    return tuple(dims), cls


def cast_class(a: np.ndarray, cls: str | None) -> np.ndarray:
    if cls is None or cls == "double":
        return a.astype(float) if a.dtype != float else a
    if cls == "single":
        return a.astype(np.float32)
    if cls == "logical":
        return a.astype(bool)
    if cls in INT_TYPES:
        return saturate_int(a, INT_TYPES[cls])
    raise MatlabError(f"Unsupported class '{cls}'.")


def default_dim(a: np.ndarray) -> int:
    """First non-singleton dimension (0-based axis)."""
    if a.shape[0] != 1:
        return 0
    if a.shape[1] != 1:
        return 1
    return 0


def get_dim(args: list, pos: int, a: np.ndarray) -> int:
    if len(args) > pos and not is_text(args[pos]) and not (isinstance(args[pos], np.ndarray) and args[pos].size == 0):
        d = to_int(args[pos], "dimension")
        if d < 1:
            raise MatlabError("Dimension argument must be a positive integer.")
        return d - 1
    return default_dim(a)


def keep2d(r: np.ndarray, axis: int, ndim: int = 2) -> np.ndarray:
    """Re-insert the reduced ``axis`` (sum of a matrix -> row vector).

    ``ndim`` is the dimension count of the array that was reduced; the reduced
    result has one dimension fewer, so the axis is put back as a singleton.
    """
    r = np.asarray(r)
    if r.ndim < ndim:
        r = np.expand_dims(r, axis)
    from ..values import as_2d
    return as_2d(r)


def flag_present(args: list, *flags: str) -> tuple[list, set[str]]:
    """Split trailing option strings (e.g. 'descend', 'omitnan') from args."""
    found: set[str] = set()
    rest = []
    for a in args:
        if is_text(a) and text(a).lower() in flags:
            found.add(text(a).lower())
        else:
            rest.append(a)
    return rest, found


def as_row_or_col(values, like: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if like.shape[0] == 1 or like.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(-1, 1)


def bool_scalar(x: bool) -> np.ndarray:
    return np.array([[bool(x)]])


def fscalar(x: float) -> np.ndarray:
    return np.array([[float(x)]])
