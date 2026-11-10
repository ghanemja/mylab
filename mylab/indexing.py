"""MATLAB indexing semantics on top of 2-D numpy arrays.

MATLAB indexing is 1-based and column-major, ``A(k)`` uses linear indexing,
assigning past the end grows the array, and ``A(idx) = []`` deletes elements.
These helpers operate on raw numpy arrays (numeric, char, or object arrays used
by cells and struct arrays) so all container types share one implementation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .errors import MatlabError


class _ColonType:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return ":"


COLON = _ColonType()


# -- index conversion -------------------------------------------------------------

def _to_positions(idx: Any, extent: int, *, for_set: bool) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Convert one index value into 0-based integer positions.

    Returns the positions and, for vector-like indices, the shape the index had
    (used to decide the shape of the result of linear indexing).
    """
    if idx is COLON:
        return np.arange(extent), None
    if isinstance(idx, np.ndarray) and idx.dtype.kind == "U":
        if idx.size == 1 and idx.flat[0] == ":":
            return np.arange(extent), None
        raise MatlabError("Character arrays cannot be used as indices.")
    if not isinstance(idx, np.ndarray):
        raise MatlabError("Indices must be numeric or logical arrays.")
    if idx.dtype == bool:
        flat = idx.flatten(order="F")
        pos = np.nonzero(flat)[0]
        if pos.size and pos[-1] >= extent and not for_set:
            raise MatlabError(
                f"The logical indices contain a true value outside of the array bounds "
                f"(index {pos[-1] + 1} exceeds {extent})."
            )
        shape = (1, pos.size) if idx.shape[0] == 1 and idx.ndim == 2 else (pos.size, 1)
        return pos, shape
    vals = np.real(idx).flatten(order="F") if idx.dtype.kind == "c" else idx.flatten(order="F")
    vals = vals.astype(float) if vals.dtype.kind in "iu" else vals
    if vals.size:
        if not np.all(np.isfinite(vals)) or np.any(vals != np.floor(vals)) or np.any(vals < 1):
            bad = vals[(~np.isfinite(vals)) | (vals != np.floor(vals)) | (vals < 1)][0]
            raise MatlabError(
                f"Array indices must be positive integers or logical values. (got {_fmt_index(bad)})"
            )
    pos = vals.astype(np.int64) - 1
    if not for_set and pos.size and pos.max() >= extent:
        raise MatlabError(f"Index exceeds the number of array elements. Index must not exceed {extent}.")
    return pos, idx.shape


def _fmt_index(x: float) -> str:
    if np.isnan(x):
        return "NaN"
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def _linear_result_shape(arr_shape: tuple[int, int], idx_shape: tuple[int, int] | None,
                         count: int) -> tuple[int, int]:
    if idx_shape is None:  # A(:)
        return (count, 1)
    ir, ic = idx_shape
    idx_is_vector = ir == 1 or ic == 1
    ar, ac = arr_shape
    arr_is_vector = ar == 1 or ac == 1
    if not idx_is_vector:
        return idx_shape
    if arr_is_vector and not (ar == 1 and ac == 1):
        return (1, count) if ar == 1 else (count, 1)
    return (1, count) if ir == 1 else (count, 1)


def _normalize_args(args: list[Any]) -> list[Any]:
    """Allow A(i, j, 1, 1) by dropping trailing singleton indices."""
    if len(args) <= 2:
        return args
    extra = args[2:]
    for e in extra:
        if e is COLON:
            continue
        if isinstance(e, np.ndarray) and e.size == 1 and e.dtype != bool and float(np.real(e.flat[0])) == 1:
            continue
        raise MatlabError("Only 2-D arrays are supported (index in dimension 3 or higher must be 1).")
    return args[:2]


# -- get ------------------------------------------------------------------------

def get(arr: np.ndarray, args: list[Any]) -> np.ndarray:
    args = _normalize_args(args)
    if len(args) == 0:
        return arr
    if len(args) == 1:
        n = arr.size
        pos, ishape = _to_positions(args[0], n, for_set=False)
        flat = arr.flatten(order="F") if arr.size else arr.reshape(-1)
        out = flat[pos]
        shape = _linear_result_shape(arr.shape, ishape, pos.size)
        return out.reshape(shape, order="F")
    rows, cols = arr.shape
    rpos, _ = _to_positions(args[0], rows, for_set=False)
    cpos, _ = _to_positions(args[1], cols, for_set=False)
    if rpos.size and rpos.max() >= rows or cpos.size and cpos.max() >= cols:
        bad_dim = 1 if (rpos.size and rpos.max() >= rows) else 2
        limit = rows if bad_dim == 1 else cols
        raise MatlabError(f"Index in position {bad_dim} exceeds array bounds. Index must not exceed {limit}.")
    return arr[np.ix_(rpos, cpos)]


# -- set ------------------------------------------------------------------------

def _grow(arr: np.ndarray, rows: int, cols: int, fill: Callable[[], Any]) -> np.ndarray:
    r0, c0 = arr.shape
    if rows <= r0 and cols <= c0:
        return arr
    rows, cols = max(rows, r0), max(cols, c0)
    if arr.dtype == object:
        out = np.empty((rows, cols), dtype=object)
        for idx in np.ndindex(out.shape):
            out[idx] = fill()
    else:
        out = np.full((rows, cols), fill(), dtype=arr.dtype)
    out[:r0, :c0] = arr
    return out


def _rhs_for(count_shape: tuple[int, int], rhs: np.ndarray) -> np.ndarray:
    n = count_shape[0] * count_shape[1]
    if rhs.size == 1:
        return np.broadcast_to(rhs.reshape(1, 1), count_shape)
    if rhs.size != n:
        raise MatlabError(
            "Unable to perform assignment because the indices on the left side are not compatible "
            f"with the size of the right side ({rhs.shape[0]}x{rhs.shape[1]} into "
            f"{count_shape[0]}x{count_shape[1]})."
        )
    if rhs.shape == count_shape:
        return rhs
    return rhs.flatten(order="F").reshape(count_shape, order="F")


def set_(arr: np.ndarray, args: list[Any], rhs: np.ndarray, fill: Callable[[], Any]) -> np.ndarray:
    """Return ``arr`` with ``arr(args) = rhs`` applied (may grow / reallocate)."""
    args = _normalize_args(args)
    if len(args) == 1:
        idx = args[0]
        n = arr.size
        if idx is COLON:
            if rhs.size != 1 and rhs.size != n:
                raise MatlabError(
                    "Unable to perform assignment because the left and right sides have a different "
                    "number of elements."
                )
            out = arr.copy()
            vals = np.broadcast_to(rhs.reshape(-1, order="F"), (n,)) if rhs.size == 1 else rhs.flatten(order="F")
            rows = out.shape[0] or 1
            for k in range(n):
                out[k % rows, k // rows] = vals[k]
            return out
        pos, _ = _to_positions(idx, n, for_set=True)
        if pos.size == 0:
            return arr
        if rhs.size != 1 and rhs.size != pos.size:
            raise MatlabError(
                "Unable to perform assignment because the left and right sides have a different "
                f"number of elements ({pos.size} on the left, {rhs.size} on the right)."
            )
        need = int(pos.max()) + 1
        r0, c0 = arr.shape
        if need > n:
            if arr.size == 0 and r0 <= 1 and c0 <= 1 or r0 == 0 and c0 == 0:
                arr = _grow(np.empty((1, 0), dtype=arr.dtype) if arr.dtype != object else
                            np.empty((1, 0), dtype=object), 1, need, fill)
            elif r0 == 1:
                arr = _grow(arr, 1, need, fill)
            elif c0 == 1:
                arr = _grow(arr, need, 1, fill)
            elif arr.size == 0:
                arr = _grow(np.empty((1, 0), dtype=arr.dtype), 1, need, fill)
            else:
                raise MatlabError("Attempt to grow array along ambiguous dimension.")
        else:
            arr = arr.copy()
        rows = arr.shape[0]
        vals = rhs.flatten(order="F")
        if rhs.size == 1:
            v = vals[0]
            for p in pos:
                arr[p % rows, p // rows] = v
        else:
            for k, p in enumerate(pos):
                arr[p % rows, p // rows] = vals[k]
        return arr

    r0, c0 = arr.shape
    ri, ci = args
    rpos = None if ri is COLON else _to_positions(ri, r0, for_set=True)[0]
    cpos = None if ci is COLON else _to_positions(ci, c0, for_set=True)[0]
    if rpos is None:
        if r0:
            n = r0
        elif rhs.size == 1:
            n = 1
        elif cpos is not None:
            n = rhs.size // max(cpos.size, 1)
        else:
            n = rhs.shape[0]
        rpos = np.arange(n)
    if cpos is None:
        if c0:
            n = c0
        elif rhs.size == 1:
            n = 1
        else:
            n = rhs.size // max(rpos.size, 1)
        cpos = np.arange(n)
    if rpos.size == 0 or cpos.size == 0:
        return arr
    target_shape = (rpos.size, cpos.size)
    vals = _rhs_for(target_shape, rhs)
    arr = _grow(arr, int(rpos.max()) + 1, int(cpos.max()) + 1, fill)
    arr = arr.copy()
    if arr.dtype == object:
        for a, r in enumerate(rpos):
            for b, c in enumerate(cpos):
                arr[r, c] = vals[a, b]
    else:
        arr[np.ix_(rpos, cpos)] = vals
    return arr


# -- delete ---------------------------------------------------------------------

def delete(arr: np.ndarray, args: list[Any]) -> np.ndarray:
    args = _normalize_args(args)
    if len(args) == 1:
        if args[0] is COLON:
            return arr[:0, :0] if arr.dtype != object else np.empty((0, 0), dtype=object)
        pos, _ = _to_positions(args[0], arr.size, for_set=False)
        keep = np.ones(arr.size, dtype=bool)
        keep[pos] = False
        flat = arr.flatten(order="F")[keep]
        if arr.shape[1] == 1 and arr.shape[0] != 1:
            return flat.reshape(-1, 1)
        return flat.reshape(1, -1)
    ri, ci = args
    r0, c0 = arr.shape
    if ri is COLON or _covers(ri, r0):
        cpos, _ = _to_positions(ci, c0, for_set=False)
        keep = np.ones(c0, dtype=bool)
        keep[cpos] = False
        return arr[:, keep]
    if ci is COLON or _covers(ci, c0):
        rpos, _ = _to_positions(ri, r0, for_set=False)
        keep = np.ones(r0, dtype=bool)
        keep[rpos] = False
        return arr[keep, :]
    raise MatlabError("A null assignment can have only one non-colon index.")


def _covers(idx: Any, extent: int) -> bool:
    try:
        pos, _ = _to_positions(idx, extent, for_set=False)
    except MatlabError:
        return False
    return set(pos.tolist()) == set(range(extent))


def end_value(shape: tuple[int, int], position: int, count: int) -> int:
    """Value of ``end`` in index ``position`` (0-based) of ``count`` indices."""
    rows, cols = shape
    if count == 1:
        return rows * cols
    if position == 0:
        return rows
    if position == 1:
        return cols
    return 1
