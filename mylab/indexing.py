"""MATLAB indexing semantics on top of numpy arrays (2-D and N-D).

MATLAB indexing is 1-based and column-major. ``A(k)`` uses linear indexing;
``A(i, j)`` with fewer indices than dimensions folds the trailing dimensions
into the last index; assigning past the end grows the array; and
``A(idx) = []`` deletes elements. These helpers operate on raw numpy arrays
(numeric, char, or the object arrays used by cells and struct arrays), so all
container types share one implementation.

Arrays always have at least two dimensions, and trailing singleton dimensions
beyond the second are dropped (``canonical``), matching MATLAB's size rules.
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


def canonical(a: np.ndarray) -> np.ndarray:
    """At least 2-D, with trailing singleton dimensions (beyond the 2nd) removed."""
    if a.ndim < 2:
        return a.reshape((1, -1) if a.ndim == 1 else (1, 1))
    if a.ndim > 2 and a.shape[-1] == 1:
        shape = list(a.shape)
        while len(shape) > 2 and shape[-1] == 1:
            shape.pop()
        return a.reshape(shape)
    return a


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# -- index conversion -------------------------------------------------------------

def _to_positions(idx: Any, extent: int, *, for_set: bool) -> tuple[np.ndarray, tuple | None]:
    """Convert one index value into 0-based integer positions.

    Returns the positions and the shape the index had (used to decide the
    result shape of linear indexing); ``None`` for a colon.
    """
    if idx is COLON:
        return np.arange(extent), None
    if isinstance(idx, np.ndarray) and idx.dtype.kind == "U":
        if idx.size == 1 and idx.flat[0] == ":":
            return np.arange(extent), None
        raise MatlabError("Character arrays cannot be used as indices.")
    if hasattr(idx, "to_dense"):  # a sparse (logical) index
        idx = idx.to_dense()
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
        shape = (1, pos.size) if idx.ndim == 2 and idx.shape[0] == 1 else (pos.size, 1)
        return pos, shape
    vals = np.real(idx).flatten(order="F") if idx.dtype.kind == "c" else idx.flatten(order="F")
    vals = vals.astype(float) if vals.dtype.kind in "iu" else vals
    if vals.size:
        bad_mask = (~np.isfinite(vals)) | (vals != np.floor(vals)) | (vals < 1)
        if bad_mask.any():
            raise MatlabError(
                f"Array indices must be positive integers or logical values. (got {_fmt_index(vals[bad_mask][0])})",
                "MATLAB:badsubscript",
            )
    pos = vals.astype(np.int64) - 1
    if not for_set and pos.size and pos.max() >= extent:
        raise MatlabError(f"Index exceeds the number of array elements. Index must not exceed {extent}.",
                          "MATLAB:badsubscript")
    return pos, idx.shape


def _fmt_index(x: float) -> str:
    if np.isnan(x):
        return "NaN"
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def _linear_result_shape(arr_shape: tuple, idx_shape: tuple | None, count: int) -> tuple:
    if idx_shape is None:  # A(:)
        return (count, 1)
    idx_is_vector = len(idx_shape) == 2 and (idx_shape[0] == 1 or idx_shape[1] == 1)
    arr_is_vector = len(arr_shape) == 2 and (arr_shape[0] == 1 or arr_shape[1] == 1)
    if not idx_is_vector:
        return idx_shape
    if arr_is_vector and not (arr_shape[0] == 1 and arr_shape[1] == 1):
        return (1, count) if arr_shape[0] == 1 else (count, 1)
    return (1, count) if idx_shape[0] == 1 else (count, 1)


def _effective_shape(shape: tuple, k: int) -> tuple:
    """Shape seen by ``k`` indices: trailing dims fold into the last; missing dims are 1."""
    if k >= len(shape):
        return tuple(shape) + (1,) * (k - len(shape))
    return tuple(shape[: k - 1]) + (_prod(shape[k - 1:]),)


# -- get ------------------------------------------------------------------------

def get(arr: np.ndarray, args: list[Any]) -> np.ndarray:
    k = len(args)
    if k == 0:
        return arr
    if k == 1:
        n = arr.size
        pos, ishape = _to_positions(args[0], n, for_set=False)
        flat = arr.reshape(-1, order="F")
        out = flat[pos]
        shape = _linear_result_shape(arr.shape, ishape, pos.size)
        return canonical(out.reshape(shape, order="F"))
    eshape = _effective_shape(arr.shape, k)
    view = arr.reshape(eshape, order="F") if eshape != arr.shape else arr
    positions = []
    for d, idx in enumerate(args):
        pos, _ = _to_positions(idx, eshape[d], for_set=True)
        if pos.size and pos.max() >= eshape[d]:
            raise MatlabError(f"Index in position {d + 1} exceeds array bounds. "
                              f"Index must not exceed {eshape[d]}.", "MATLAB:badsubscript")
        positions.append(pos)
    return canonical(view[np.ix_(*positions)])


# -- set ------------------------------------------------------------------------

def _new_array(shape: tuple, dtype, fill: Callable[[], Any]) -> np.ndarray:
    if np.dtype(dtype).kind == "O":
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            out[idx] = fill()
        return out
    return np.full(shape, fill(), dtype=dtype)


def _grow(arr: np.ndarray, new_shape: tuple, fill: Callable[[], Any]) -> np.ndarray:
    if all(n <= o for n, o in zip(new_shape, arr.shape)) and len(new_shape) == arr.ndim:
        return arr
    target = tuple(max(n, o) for n, o in zip(new_shape, arr.shape))
    out = _new_array(target, arr.dtype, fill)
    out[tuple(slice(0, s) for s in arr.shape)] = arr
    return out


def _assign_block(arr: np.ndarray, positions: list[np.ndarray], vals: np.ndarray) -> None:
    if arr.dtype == object:
        for src in np.ndindex(vals.shape):
            arr[tuple(int(positions[d][i]) for d, i in enumerate(src))] = vals[src]
    else:
        arr[np.ix_(*positions)] = vals


def set_(arr: np.ndarray, args: list[Any], rhs: np.ndarray, fill: Callable[[], Any]) -> np.ndarray:
    """Return ``arr`` with ``arr(args) = rhs`` applied (may grow / reallocate)."""
    if hasattr(rhs, "to_dense"):  # a sparse right-hand side stored into a full array
        rhs = rhs.to_dense()
    k = len(args)
    if k == 1:
        return _set_linear(arr, args[0], rhs, fill)

    shape = arr.shape
    eshape = _effective_shape(shape, k)
    folded = k < len(shape) and eshape != shape
    work = arr.reshape(eshape, order="F") if eshape != shape else arr

    positions: list[np.ndarray | None] = []
    for d, idx in enumerate(args):
        positions.append(None if idx is COLON else _to_positions(idx, eshape[d], for_set=True)[0])
    # Size colons over empty dimensions from the right-hand side.
    zero_colons = [d for d, p in enumerate(positions) if p is None and eshape[d] == 0]
    for d, p in enumerate(positions):
        if p is None and eshape[d] > 0:
            positions[d] = np.arange(eshape[d])
    if zero_colons:
        if rhs.size == 1:
            for d in zero_colons:
                positions[d] = np.arange(1)
        elif len(zero_colons) == 1:
            known = _prod(len(p) for p in positions if p is not None)
            positions[zero_colons[0]] = np.arange(rhs.size // max(known, 1))
        else:
            rshape = list(rhs.shape) + [1] * k
            for d in zero_colons:
                positions[d] = np.arange(rshape[d])
    pos_list: list[np.ndarray] = [p for p in positions]  # type: ignore[misc]
    if any(p.size == 0 for p in pos_list):
        return arr
    target = tuple(p.size for p in pos_list)
    n = _prod(target)
    if rhs.size == 1:
        vals = np.broadcast_to(rhs.reshape((1,) * k), target)
    elif rhs.size == n:
        vals = rhs if rhs.shape == target else rhs.reshape(-1, order="F").reshape(target, order="F")
    else:
        tshape = "x".join(str(t) for t in target)
        rshape = "x".join(str(t) for t in rhs.shape)
        raise MatlabError("Unable to perform assignment because the indices on the left side are not "
                          f"compatible with the size of the right side ({rshape} into {tshape}).")
    need = tuple(int(p.max()) + 1 for p in pos_list)
    if folded and need[-1] > eshape[-1]:
        raise MatlabError("Attempt to grow array along ambiguous dimension.")
    work = _grow(work, need, fill)
    work = work.copy() if work is arr or work.base is not None else work
    _assign_block(work, pos_list, vals)
    if folded:
        work = work.reshape(shape, order="F")
    return canonical(work)


def _set_linear(arr: np.ndarray, idx: Any, rhs: np.ndarray, fill: Callable[[], Any]) -> np.ndarray:
    n = arr.size
    if idx is COLON:
        if rhs.size != 1 and rhs.size != n:
            raise MatlabError("Unable to perform assignment because the left and right sides have a "
                              "different number of elements.")
        vals = rhs.reshape(-1, order="F")
        out = arr.copy(order="F")
        flat = out.reshape(-1, order="F")  # a view, since out is Fortran-ordered
        for k in range(n):
            flat[k] = vals[0 if rhs.size == 1 else k]
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
    shape = arr.shape
    if need > n:
        if n == 0 and len(shape) == 2 and (shape[0] <= 1 or shape[1] <= 1) and not (shape[0] > 1):
            arr = _grow(_new_array((1, 0), arr.dtype, fill), (1, need), fill)
        elif len(shape) == 2 and shape[0] == 1:
            arr = _grow(arr, (1, need), fill)
        elif len(shape) == 2 and shape[1] == 1:
            arr = _grow(arr, (need, 1), fill)
        elif n == 0:
            arr = _grow(_new_array((1, 0), arr.dtype, fill), (1, need), fill)
        else:
            raise MatlabError("Attempt to grow array along ambiguous dimension.")
    else:
        arr = arr.copy()
    vals = rhs.reshape(-1, order="F")
    # Assign through a flat view of a Fortran-ordered array (column-major positions).
    fortran = np.asfortranarray(arr)
    flat = fortran.reshape(-1, order="F")
    if arr.dtype == object:
        for k, p in enumerate(pos):
            flat[p] = vals[0 if rhs.size == 1 else k]
    else:
        flat[pos] = vals[0] if rhs.size == 1 else vals
    return fortran


# -- delete ---------------------------------------------------------------------

def delete(arr: np.ndarray, args: list[Any]) -> np.ndarray:
    k = len(args)
    if k == 1:
        if args[0] is COLON:
            return np.empty((0, 0), dtype=arr.dtype)
        pos, _ = _to_positions(args[0], arr.size, for_set=False)
        keep = np.ones(arr.size, dtype=bool)
        keep[pos] = False
        flat = arr.reshape(-1, order="F")[keep]
        if arr.ndim == 2 and arr.shape[1] == 1 and arr.shape[0] != 1:
            return flat.reshape(-1, 1)
        return flat.reshape(1, -1)
    shape = arr.shape
    eshape = _effective_shape(shape, k)
    work = arr.reshape(eshape, order="F") if eshape != shape else arr
    non_full = [d for d, idx in enumerate(args) if not (idx is COLON or _covers(idx, eshape[d]))]
    if not non_full:
        # Everything selected: deletes along the first dimension.
        non_full = [0]
    if len(non_full) > 1:
        raise MatlabError("A null assignment can have only one non-colon index.")
    d = non_full[0]
    pos, _ = _to_positions(args[d], eshape[d], for_set=False)
    keep = np.ones(eshape[d], dtype=bool)
    keep[pos] = False
    out = np.compress(keep, work, axis=d)
    if eshape != shape and d < k - 1:
        new_shape = list(shape)
        new_shape[d] = out.shape[d]
        out = out.reshape(new_shape, order="F")
    return canonical(out)


def _covers(idx: Any, extent: int) -> bool:
    try:
        pos, _ = _to_positions(idx, extent, for_set=False)
    except MatlabError:
        return False
    return pos.size == extent and set(pos.tolist()) == set(range(extent))


def end_value(shape: tuple, position: int, count: int) -> int:
    """Value of ``end`` in index ``position`` (0-based) of ``count`` indices."""
    if count == 1:
        return _prod(shape)
    if position < count - 1:
        return int(shape[position]) if position < len(shape) else 1
    return _prod(shape[position:]) if position < len(shape) else 1
