"""Newer text helpers, the last-error functions and a few recent data-cleaning functions."""

from __future__ import annotations

import math
import sys

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, Struct, char, is_char, string_result, to_str
from ._registry import BUILTINS, builtin
from ._util import bool_scalar, default_dim, is_text, nargs, num, text


# -- string / char conversion ----------------------------------------------------------------

@builtin("isStringScalar")
def _is_string_scalar(interp, args, nargout):
    nargs(args, 1, 1, "isStringScalar")
    v = args[0]
    return bool_scalar(isinstance(v, MString) or (isinstance(v, StringArray) and v.data.size == 1))


def _strings_to_chars(v):
    if isinstance(v, MString):
        return char(str(v))
    if isinstance(v, StringArray):
        if v.data.size == 1:
            x = v.data.flat[0]
            return char("" if x is None else x)
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            x = v.data[idx]
            data[idx] = char("" if x is None else x)
        return Cell(data)
    return v


def _chars_to_strings(v):
    if is_char(v):
        if v.ndim == 2 and v.shape[0] <= 1:
            return MString(to_str(v))
        return MString("".join(v.flatten(order="F")))
    if isinstance(v, Cell) and all(is_char(x) for x in v.data.flat):
        data = np.empty(v.data.shape, dtype=object)
        for idx in np.ndindex(v.data.shape):
            data[idx] = "".join(v.data[idx].flatten(order="F"))
        return StringArray(data) if data.size != 1 else string_result(data)
    return v


def _convert_all(fn, name):
    def f(interp, args, nargout):
        if not args and nargout == 0:
            return ()  # [c{:}] = f(c{:}) with an empty list
        nargs(args, 1, None, name)
        if max(nargout, 1) > len(args):
            raise MatlabError(f"{name}: more outputs than inputs.")
        out = [fn(a) for a in args[:max(nargout, 1)]]
        return tuple(out) if len(out) > 1 else out[0]
    return f


builtin("convertStringsToChars")(_convert_all(_strings_to_chars, "convertStringsToChars"))
builtin("convertCharsToStrings")(_convert_all(_chars_to_strings, "convertCharsToStrings"))


# -- last error -----------------------------------------------------------------------------

def _last_error(interp) -> MatlabError | None:
    """The error being handled by the enclosing catch block, else the one seen by the last query."""
    exc = sys.exc_info()[1]
    while exc is not None and not isinstance(exc, MatlabError):
        exc = exc.__context__
    if exc is not None:
        interp._last_error = exc
    return getattr(interp, "_last_error", None)


@builtin("lasterr")
def _lasterr(interp, args, nargout):
    nargs(args, 0, 2, "lasterr")
    err = _last_error(interp)
    msg = char(err.message if err else "")
    ident = char(err.identifier if err else "")
    if args:
        interp._last_error = MatlabError(text(args[0]), text(args[1]) if len(args) > 1 else "")
    return (msg, ident) if nargout > 1 else msg


@builtin("lasterror")
def _lasterror(interp, args, nargout):
    nargs(args, 0, 1, "lasterror")
    err = _last_error(interp)
    stack = Struct(np.empty((0, 1), dtype=object), ["file", "name", "line"])
    out = Struct.scalar({"message": char(err.message if err else ""),
                         "identifier": char(err.identifier if err else ""), "stack": stack})
    if args:
        a = args[0]
        if is_text(a) and text(a).lower() == "reset":
            interp._last_error = None
        elif isinstance(a, Struct) and a.data.size == 1:
            d = a.data.flat[0]
            interp._last_error = MatlabError(to_str(d.get("message", char(""))), to_str(d.get("identifier", char(""))))
        else:
            raise MatlabError("lasterror: the input must be 'reset' or an error struct.")
    return out


# -- recent numeric helpers ---------------------------------------------------------------------

def _numeric(v, name: str) -> np.ndarray:
    if not isinstance(v, np.ndarray) or v.dtype.kind == "U":
        raise MatlabError(f"{name}: the input must be a numeric array.")
    return v


@builtin("anynan")
def _anynan(interp, args, nargout):
    nargs(args, 1, 1, "anynan")
    a = _numeric(args[0], "anynan")
    return bool_scalar(a.dtype.kind in "fc" and bool(np.isnan(a).any()))


@builtin("allfinite")
def _allfinite(interp, args, nargout):
    nargs(args, 1, 1, "allfinite")
    a = _numeric(args[0], "allfinite")
    return bool_scalar(a.dtype.kind not in "fc" or bool(np.isfinite(a).all()))


def _array_size(dim: int, name: str):
    table_fn = BUILTINS.get(name)

    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        v = args[0]
        if isinstance(v, (Cell, Struct, StringArray)):
            shape = v.data.shape
        elif isinstance(v, np.ndarray):
            shape = v.shape
        elif isinstance(v, MString):
            shape = (1, 1)
        elif table_fn is not None:
            return table_fn(interp, args, nargout)
        else:
            raise MatlabError(f"{name}: unsupported input.")
        return np.array([[float(shape[dim])]])
    return f


builtin("height")(_array_size(0, "height"))
builtin("width")(_array_size(1, "width"))


_MAD_SCALE = 1.4826022185056018  # -1 / (sqrt(2) * erfcinv(3/2))


@builtin("isoutlier")
def _isoutlier(interp, args, nargout):
    nargs(args, 1, None, "isoutlier")
    a = _numeric(args[0], "isoutlier")
    x = np.real(a).astype(float)
    method, factor, dim = "median", None, None
    rest = list(args[1:])
    k = 0
    while k < len(rest):
        r = rest[k]
        if is_text(r):
            key = text(r).lower()
            if key in ("median", "mean", "quartiles"):
                method = key
            elif key == "thresholdfactor" and k + 1 < len(rest):
                factor = float(num(rest[k + 1]).flat[0])
                k += 1
            else:
                raise MatlabError(f"isoutlier: unsupported option '{text(r)}'.")
        else:
            dim = int(num(r).flat[0]) - 1
        k += 1
    if x.size == 0:
        return np.zeros(x.shape, dtype=bool)
    axis = default_dim(x) if dim is None else dim
    if axis >= x.ndim:
        return np.zeros(x.shape, dtype=bool)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "median":
                center = np.nanmedian(x, axis=axis, keepdims=True)
                spread = _MAD_SCALE * np.nanmedian(np.abs(x - center), axis=axis, keepdims=True)
                f = 3.0 if factor is None else factor
                lo, hi = center - f * spread, center + f * spread
            elif method == "mean":
                center = np.nanmean(x, axis=axis, keepdims=True)
                n = np.sum(~np.isnan(x), axis=axis, keepdims=True)
                std0 = np.nanstd(x, axis=axis, keepdims=True, ddof=0)
                spread = np.where(n > 1, std0 * np.sqrt(n / np.maximum(n - 1, 1)), 0.0)
                f = 3.0 if factor is None else factor
                lo, hi = center - f * spread, center + f * spread
            else:
                q1 = np.apply_along_axis(_quantile, axis, x, 0.25)
                q3 = np.apply_along_axis(_quantile, axis, x, 0.75)
                q1, q3 = np.expand_dims(q1, axis), np.expand_dims(q3, axis)
                f = 1.5 if factor is None else factor
                lo, hi = q1 - f * (q3 - q1), q3 + f * (q3 - q1)
        return (x < lo) | (x > hi)


def _quantile(v: np.ndarray, p: float) -> float:
    """Quantile with the plotting positions (k - 0.5) / n, as MATLAB's quantile uses."""
    v = np.sort(v[~np.isnan(v)])
    n = v.size
    if n == 0:
        return math.nan
    pos = min(max(n * p + 0.5, 1.0), float(n))
    k = int(math.floor(pos))
    frac = pos - k
    return float(v[k - 1] + (frac * (v[k] - v[k - 1]) if k < n else 0.0))


@builtin("standardizeMissing")
def _standardize_missing(interp, args, nargout):
    nargs(args, 2, 2, "standardizeMissing")
    a, ind = args[0], args[1]
    if isinstance(a, np.ndarray) and a.dtype.kind in "fc":
        out = a.copy()
        if isinstance(ind, np.ndarray) and ind.dtype.kind != "U":
            for v in num(ind).flatten():
                out[a == v] = np.nan
        elif isinstance(ind, Cell):
            for v in ind.data.flat:
                if isinstance(v, np.ndarray) and v.dtype.kind != "U":
                    for x in num(v).flatten():
                        out[a == x] = np.nan
        return out
    if isinstance(a, np.ndarray):
        return a  # integer, logical and char arrays have no missing value
    words = None
    if is_text(ind):
        words = {text(ind)}
    elif isinstance(ind, Cell):
        words = {to_str(v) for v in ind.data.flat if is_text(v)}
    elif isinstance(ind, StringArray):
        words = {x for x in ind.data.flat if x is not None}
    if words is None:
        return a
    if isinstance(a, Cell):
        data = a.data.copy()
        for idx in np.ndindex(data.shape):
            if is_char(data[idx]) and to_str(data[idx]) in words:
                data[idx] = char("")
        return Cell(data)
    if isinstance(a, (MString, StringArray)):
        src = a.data if isinstance(a, StringArray) else np.array([[str(a)]], dtype=object)
        data = src.copy()
        for idx in np.ndindex(data.shape):
            if data[idx] in words:
                data[idx] = None
        return StringArray(data)
    return a


@builtin("uniquetol")
def _uniquetol(interp, args, nargout):
    nargs(args, 1, 2, "uniquetol")
    a = _numeric(args[0], "uniquetol")
    if a.dtype.kind not in "f":
        raise MatlabError("uniquetol: the input must be a real floating-point array.")
    tol = float(num(args[1]).flat[0]) if len(args) > 1 else (1e-6 if a.dtype == np.float32 else 1e-12)
    flat = a.flatten(order="F")
    order = np.argsort(flat, kind="stable")
    vals = flat[order]
    scale = float(np.max(np.abs(flat[np.isfinite(flat)]))) if np.isfinite(flat).any() else 1.0
    bound = tol * scale
    keep: list[int] = []
    group = np.zeros(flat.size, dtype=float)
    for k, v in enumerate(vals):
        if np.isnan(v) or not keep or not abs(v - vals[keep[-1]]) <= bound:
            keep.append(k)
        group[order[k]] = len(keep)
    c = vals[keep]
    as_row = a.ndim == 2 and a.shape[0] == 1 and a.shape[1] != 1
    c = c.reshape(1, -1) if as_row else c.reshape(-1, 1)
    if nargout <= 1:
        return c
    ia = (order[keep] + 1.0).reshape(-1, 1)
    return c, ia, group.reshape(-1, 1)
