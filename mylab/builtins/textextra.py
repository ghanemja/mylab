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
        interp.last_error = exc
    return interp.last_error


@builtin("lasterr")
def _lasterr(interp, args, nargout):
    nargs(args, 0, 2, "lasterr")
    err = _last_error(interp)
    msg = char(err.message if err else "")
    ident = char(err.identifier if err else "")
    if args:
        interp.last_error = MatlabError(text(args[0]), text(args[1]) if len(args) > 1 else "")
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
            interp.last_error = None
        elif isinstance(a, Struct) and a.data.size == 1:
            d = a.data.flat[0]
            interp.last_error = MatlabError(to_str(d.get("message", char(""))), to_str(d.get("identifier", char(""))))
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


_OUTLIER_METHODS = ("median", "mean", "quartiles", "grubbs", "gesd", "percentiles", "movmedian", "movmean")


def _t_upper(p: np.ndarray | float, df: float) -> float:
    """Upper critical value of Student's t distribution: P(T > t) = p."""
    from scipy.stats import t as tdist
    return float(tdist.isf(p, df))


def _grubbs_mask(v: np.ndarray, alpha: float, max_out: int | None, gesd: bool):
    """Outliers of one vector by repeated Grubbs tests or the generalized ESD test.

    Returns the mask and the (lower, upper, center) of the points that are kept.
    """
    idx = np.nonzero(~np.isnan(v))[0]
    vals = v[idx].astype(float)
    n0 = vals.size
    mask = np.zeros(v.size, dtype=bool)
    removed: list[int] = []
    limit = n0 - 2 if max_out is None else min(max_out, n0 - 2)
    keep = np.ones(n0, dtype=bool)
    last_sig = 0
    lam_used = 0.0
    for step in range(1, max(limit, 0) + 1):
        cur = vals[keep]
        n = cur.size
        if n < 3:
            break
        mu, sd = cur.mean(), cur.std(ddof=1)
        if sd == 0:
            break
        dev = np.abs(vals - mu)
        dev[~keep] = -1.0
        k = int(np.argmax(dev))
        stat = dev[k] / sd
        if gesd:
            pp = 1 - alpha / (2 * (n0 - step + 1))
            t = _t_upper(1 - pp, n0 - step - 1)
            lam = (n0 - step) * t / np.sqrt((n0 - step - 1 + t * t) * (n0 - step + 1))
        else:
            t = _t_upper(alpha / (2 * n), n - 2)
            lam = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
        keep[k] = False
        removed.append(k)
        if stat > lam:
            last_sig = step
            lam_used = lam
        elif not gesd:
            keep[k] = True
            removed.pop()
            break
    if gesd:
        # the generalized test keeps the largest number of removals that was significant
        keep[:] = True
        for k in removed[:last_sig]:
            keep[k] = False
    mask[idx[~keep]] = True
    cur = vals[keep]
    if cur.size == 0:
        return mask, np.nan, np.nan, np.nan
    mu = float(cur.mean())
    sd = float(cur.std(ddof=1)) if cur.size > 1 else 0.0
    if not lam_used:
        n = cur.size
        if n > 2:
            t = _t_upper(alpha / (2 * n), n - 2)
            lam_used = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
    return mask, mu - lam_used * sd, mu + lam_used * sd, mu


def _prctile_vec(v: np.ndarray, p: float) -> float:
    return _quantile(v, p / 100.0)


@builtin("isoutlier")
def _isoutlier(interp, args, nargout):
    nargs(args, 1, None, "isoutlier")
    if nargout > 4:
        raise MatlabError("isoutlier: too many output arguments.")
    a = _numeric(args[0], "isoutlier")
    x = np.real(a).astype(float)
    method, factor, dim = "median", None, None
    pct = None
    window = None
    samples = None
    max_out = None
    rest = list(args[1:])
    k = 0
    while k < len(rest):
        r = rest[k]
        if is_text(r):
            key = text(r).lower()
            if key in ("percentiles", "movmedian", "movmean"):
                if k + 1 >= len(rest) or is_text(rest[k + 1]):
                    raise MatlabError(f"isoutlier: the '{key}' method needs "
                                      + ("a [lower upper] threshold pair." if key == "percentiles"
                                         else "a window length."))
                method = key
                val = np.real(num(rest[k + 1])).astype(float).reshape(-1)
                if key == "percentiles":
                    if val.size != 2 or not 0 <= val[0] <= val[1] <= 100:
                        raise MatlabError("isoutlier: percentile thresholds must be [lower upper] within 0..100.")
                    pct = (float(val[0]), float(val[1]))
                else:
                    window = rest[k + 1]
                k += 1
            elif key in _OUTLIER_METHODS:
                method = key
            elif key == "thresholdfactor" and k + 1 < len(rest):
                factor = float(np.real(num(rest[k + 1])).flat[0])
                if factor < 0:
                    raise MatlabError("isoutlier: ThresholdFactor must be nonnegative.")
                k += 1
            elif key == "samplepoints" and k + 1 < len(rest):
                samples = np.real(num(rest[k + 1])).astype(float).reshape(-1)
                k += 1
            elif key == "maxnumoutliers" and k + 1 < len(rest):
                max_out = int(np.real(num(rest[k + 1])).flat[0])
                k += 1
            else:
                raise MatlabError(f"isoutlier: unsupported option '{text(r)}'.")
        else:
            dim = int(np.real(num(r)).flat[0]) - 1
            if dim < 0:
                raise MatlabError("isoutlier: the dimension must be a positive integer.")
        k += 1
    if max_out is not None and method != "gesd":
        raise MatlabError("isoutlier: MaxNumOutliers applies only to the 'gesd' method.")
    if samples is not None and method not in ("movmedian", "movmean"):
        raise MatlabError("isoutlier: SamplePoints applies only to the moving methods.")
    if factor is not None and method == "percentiles":
        raise MatlabError("isoutlier: ThresholdFactor does not apply to the 'percentiles' method.")
    if method in ("grubbs", "gesd") and factor is not None and not 0 <= factor <= 1:
        raise MatlabError("isoutlier: for 'grubbs' and 'gesd' the ThresholdFactor is a significance level "
                          "between 0 and 1.")

    def result(tf, lo, hi, c):
        outs = (tf, lo, hi, c)
        return outs[:max(nargout, 1)] if nargout > 1 else tf

    if x.size == 0:
        z = np.zeros(x.shape)
        return result(np.zeros(x.shape, dtype=bool), z, z, z)
    axis = default_dim(x) if dim is None else dim
    if axis >= x.ndim:
        return result(np.zeros(x.shape, dtype=bool), x.copy(), x.copy(), x.copy())
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
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
        elif method == "quartiles":
            q1 = np.expand_dims(np.apply_along_axis(_quantile, axis, x, 0.25), axis)
            q3 = np.expand_dims(np.apply_along_axis(_quantile, axis, x, 0.75), axis)
            f = 1.5 if factor is None else factor
            lo, hi = q1 - f * (q3 - q1), q3 + f * (q3 - q1)
            center = (q1 + q3) / 2
        elif method == "percentiles":
            lo = np.expand_dims(np.apply_along_axis(_prctile_vec, axis, x, pct[0]), axis)
            hi = np.expand_dims(np.apply_along_axis(_prctile_vec, axis, x, pct[1]), axis)
            center = (lo + hi) / 2
        elif method in ("grubbs", "gesd"):
            alpha = 0.05 if factor is None else factor
            xm = np.moveaxis(x, axis, 0)
            n = xm.shape[0]
            cols = xm.reshape(n, -1)
            tfc = np.zeros(cols.shape, dtype=bool)
            stats = np.zeros((3, cols.shape[1]))
            limit = max_out if max_out is not None else (int(np.ceil(0.1 * n)) if method == "gesd" else None)
            for c in range(cols.shape[1]):
                tfc[:, c], stats[0, c], stats[1, c], stats[2, c] = _grubbs_mask(cols[:, c], alpha, limit,
                                                                              method == "gesd")
            tf = np.moveaxis(tfc.reshape(xm.shape), 0, axis)
            shape = list(x.shape)
            shape[axis] = 1
            lo, hi, center = (stats[i].reshape(xm.shape[1:]).reshape(shape) for i in range(3))
            return result(tf, lo, hi, center)
        else:
            from .stats import _window_extent
            f = 3.0 if factor is None else factor
            xm = np.moveaxis(x, axis, 0)
            n = xm.shape[0]
            if samples is not None:
                if samples.size != n or np.any(np.diff(samples) <= 0):
                    raise MatlabError("isoutlier: SamplePoints must be increasing, one per element along "
                                      "the operating dimension.")
                wv = np.real(num(window)).astype(float).reshape(-1)
                back, fwd = (wv[0] / 2.0, wv[0] / 2.0) if wv.size == 1 else (wv[0], wv[1])
                half_open = wv.size == 1
            else:
                before, after = _window_extent(window)
            center_m = np.empty(xm.shape)
            spread_m = np.empty(xm.shape)
            for i in range(n):
                if samples is not None:
                    d = samples - samples[i]
                    sel = (d >= -back) & ((d < fwd) if half_open else (d <= fwd))
                    win = xm[sel]
                else:
                    win = xm[max(i - before, 0):min(i + after, n - 1) + 1]
                if method == "movmedian":
                    c = np.nanmedian(win, axis=0)
                    center_m[i] = c
                    spread_m[i] = _MAD_SCALE * np.nanmedian(np.abs(win - c), axis=0)
                else:
                    center_m[i] = np.nanmean(win, axis=0)
                    cnt = np.sum(~np.isnan(win), axis=0)
                    spread_m[i] = np.where(cnt > 1, np.nanstd(win, axis=0, ddof=0)
                                           * np.sqrt(cnt / np.maximum(cnt - 1, 1)), 0.0)
            center = np.moveaxis(center_m, 0, axis)
            spread = np.moveaxis(spread_m, 0, axis)
            lo, hi = center - f * spread, center + f * spread
        return result((x < lo) | (x > hi), lo, hi, center)


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
    nargs(args, 1, None, "uniquetol")
    a = _numeric(args[0], "uniquetol")
    if a.dtype.kind not in "f":
        raise MatlabError("uniquetol: the input must be a real floating-point array.")
    rest = list(args[1:])
    tol = 1e-6 if a.dtype == np.float32 else 1e-12
    if rest and not is_text(rest[0]):
        tol = float(np.real(num(rest.pop(0))).flat[0])
        if tol < 0:
            raise MatlabError("uniquetol: the tolerance must be nonnegative.")
    if len(rest) % 2:
        raise MatlabError("uniquetol: options must be name-value pairs.")
    by_rows = False
    all_idx = False
    scale = None
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower() if is_text(rest[k]) else ""
        val = rest[k + 1]
        if key == "byrows":
            by_rows = bool(np.all(num(val) != 0))
        elif key == "outputallindices":
            all_idx = bool(np.all(num(val) != 0))
        elif key == "datascale":
            scale = np.real(num(val)).astype(float).reshape(-1)
        else:
            raise MatlabError(f"uniquetol: unknown option '{text(rest[k]) if is_text(rest[k]) else '?'}'.")
    if by_rows:
        if a.ndim != 2:
            raise MatlabError("uniquetol: with ByRows the input must be a matrix.")
        rows = a.astype(float)
        fin = np.where(np.isfinite(rows), np.abs(rows), 0.0)
        ds = fin.max(axis=0) if scale is None and rows.size else \
            (np.broadcast_to(scale, (rows.shape[1],)).astype(float) if scale is not None else np.zeros(0))
    else:
        rows = a.reshape(-1, 1, order="F").astype(float)
        fin = np.abs(rows[np.isfinite(rows)])
        ds = np.array([float(fin.max()) if fin.size else 1.0]) if scale is None else scale[:1].astype(float)
    bound = tol * ds
    n = rows.shape[0]
    order = np.lexsort(tuple(rows[:, c] for c in range(rows.shape[1] - 1, -1, -1))) if n else np.zeros(0, int)
    group = np.zeros(n, dtype=float)
    reps: list[int] = []        # index (into the original rows) of the representative of each group
    members: list[list[int]] = []
    done = np.zeros(n, dtype=bool)
    for k in order:
        if done[k]:
            continue
        row = rows[k]
        if np.isnan(row).any():
            close = np.zeros(n, dtype=bool)
            close[k] = True  # NaN is not within tolerance of anything, not even of another NaN
        else:
            with np.errstate(invalid="ignore"):
                close = np.all(np.abs(rows - row) <= bound, axis=1) & ~done
            close[k] = True
        idx = np.nonzero(close)[0]
        done[idx] = True
        reps.append(int(k))
        members.append([int(i) for i in idx])
        group[idx] = len(reps)
    c = a[reps, :] if by_rows else a.reshape(-1, order="F")[reps]
    if not by_rows:
        as_row = a.ndim == 2 and a.shape[0] == 1 and a.shape[1] != 1
        c = c.reshape(1, -1) if as_row else c.reshape(-1, 1)
    if nargout <= 1:
        return c
    if all_idx:
        ia = Cell.from_list([np.array([i + 1.0 for i in mem]).reshape(-1, 1) for mem in members], column=True)
    else:
        ia = np.array([r + 1.0 for r in reps]).reshape(-1, 1)
    return c, ia, group.reshape(-1, 1)
