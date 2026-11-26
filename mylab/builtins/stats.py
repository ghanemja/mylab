"""Statistics and probability: distributions, descriptive statistics, tests, moving windows."""

from __future__ import annotations

import numpy as np

from .._lazy import lazy_module
from ..errors import MatlabError
from ..values import char, to_int
from ._registry import builtin
from ._util import default_dim, fnum, get_dim, is_text, keep2d, nargs, text

st = lazy_module("scipy.stats", globals(), "st")  # imported on first use: keeps start-up fast


# -- helpers -------------------------------------------------------------------------

def _given(args: list, i: int) -> bool:
    return len(args) > i and not (isinstance(args[i], np.ndarray) and args[i].size == 0 and not is_text(args[i]))


def _scalar(v, what="argument") -> float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    return float(np.real(a.flat[0]))


def _bcast(name: str, *vals) -> list[np.ndarray]:
    arrs = [fnum(v).astype(float) for v in vals]
    shapes = {a.shape for a in arrs if a.size != 1}
    if len(shapes) > 1:
        raise MatlabError(f"{name}: non-scalar arguments must have the same size.")
    try:
        return list(np.broadcast_arrays(*arrs))
    except ValueError:
        raise MatlabError(f"{name}: non-scalar arguments must have the same size.") from None


def _rng(interp):
    rng = interp.state.get("rng")
    if rng is None:
        rng = np.random.default_rng()
        interp.state["rng"] = rng
    return rng


def _vec_out(values, like: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    return v.reshape(1, -1) if like.shape[0] == 1 else v.reshape(-1, 1)


# -- probability distributions --------------------------------------------------------

@builtin("randsample")
def _randsample(interp, args, nargout):
    nargs(args, 2, 4, "randsample")
    pop = fnum(args[0])
    k = to_int(args[1], "sample size")
    replace = bool(fnum(args[2]).flat[0]) if _given(args, 2) else False
    w = fnum(args[3]).flatten() if _given(args, 3) else None
    if pop.size == 1 and w is None or (pop.size == 1 and w is not None and w.size != 1):
        n = int(pop.flat[0])
        values = np.arange(1, n + 1, dtype=float)
        is_pop = False
    else:
        values = pop.flatten()
        n = values.size
        is_pop = True
    if w is not None:
        if w.size != n:
            raise MatlabError("randsample: weights must have one entry per population element.")
        replace = True if not _given(args, 2) else replace
        p = w / w.sum()
    else:
        p = None
    if not replace and k > n:
        raise MatlabError("randsample: sample size exceeds population size without replacement.")
    idx = _rng(interp).choice(n, size=k, replace=replace, p=p)
    out = values[idx]
    if is_pop:
        return out.astype(args[0].dtype if isinstance(args[0], np.ndarray) else float).reshape(
            (1, -1) if pop.shape[0] == 1 and pop.size > 1 else (-1, 1))
    return out.reshape(-1, 1)


# -- descriptive statistics -----------------------------------------------------------

def _quantile_core(name, scale):
    def f(interp, args, nargout):
        rest = list(args)
        if len(rest) >= 4 and is_text(rest[-2]) and text(rest[-2]).lower() == "method":
            if not is_text(rest[-1]) or text(rest[-1]).lower() not in ("exact", "approximate"):
                raise MatlabError(f"{name}: Method must be 'exact' or 'approximate'.")
            rest = rest[:-2]  # the approximate method is answered with the exact computation
        nargs(rest, 2, 3, name)
        x = fnum(rest[0]).astype(float)
        p = fnum(rest[1]).flatten().astype(float) / scale
        if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
            raise MatlabError(f"{name}: probabilities must be in range.")
        if x.size == 0:
            return np.full((1, p.size), np.nan)
        if len(rest) > 2 and is_text(rest[2]):
            if text(rest[2]).lower() != "all":
                raise MatlabError(f"{name}: unknown option '{text(rest[2])}'.")
            x = x.reshape(-1, 1, order="F")
            axes = [0]
        elif len(rest) > 2 and isinstance(rest[2], np.ndarray) and rest[2].size > 1:
            axes = sorted({int(d) - 1 for d in fnum(rest[2]).flatten()})
            if axes[0] < 0:
                raise MatlabError("Dimension argument must be a positive integer.")
            while x.ndim <= axes[-1]:
                x = x[..., None]
        else:
            axes = [get_dim(rest, 2, x)]
        xm = np.moveaxis(x, axes, list(range(len(axes))))
        rest_shape = xm.shape[len(axes):]
        xm = xm.reshape(-1, int(np.prod(rest_shape)) if rest_shape else 1)
        r = np.empty((p.size, xm.shape[1]))
        for j in range(xm.shape[1]):
            col = xm[:, j]
            col = np.sort(col[~np.isnan(col)])
            n = col.size
            if n == 0:
                r[:, j] = np.nan
                continue
            # sorted values sit at cumulative probabilities (i - 0.5) / n
            pos = p * n + 0.5
            r[:, j] = np.interp(pos, np.arange(1, n + 1), col)
        r = np.moveaxis(r.reshape((p.size,) + tuple(rest_shape)), 0, axes[0])
        shape = list(x.shape)
        for ax in axes:
            shape[ax] = 1
        shape[axes[0]] = p.size
        from ..values import as_2d
        return as_2d(r.reshape(shape))
    return f


builtin("prctile")(_quantile_core("prctile", 100.0))
builtin("quantile")(_quantile_core("quantile", 1.0))


@builtin("iqr")
def _iqr(interp, args, nargout):
    nargs(args, 1, 2, "iqr")
    q = _quantile_core("iqr", 100.0)
    extra = list(args[1:])
    hi = q(interp, [args[0], np.array([[75.0]])] + extra, 1)
    lo = q(interp, [args[0], np.array([[25.0]])] + extra, 1)
    return hi - lo


def _axis_and(args, pos, x):
    if x.size == 0:
        return 0
    return get_dim(args, pos, x)


def _dim_spec(args, pos):
    """None, 'all' or a list of dimension numbers from an optional dimension argument."""
    if not _given(args, pos):
        return None
    if is_text(args[pos]):
        if text(args[pos]).lower() != "all":
            raise MatlabError(f"Unknown option '{text(args[pos])}'.")
        return "all"
    return [int(d) for d in fnum(args[pos]).flatten()]


@builtin("mad")
def _mad(interp, args, nargout):
    from .stats_desc import reduce_over
    nargs(args, 1, 3, "mad")
    x = fnum(args[0]).astype(float)
    flag = int(_scalar(args[1])) if _given(args, 1) else 0
    if x.size == 0:
        return np.array([[np.nan]])
    dim = _dim_spec(args, 2)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if flag == 0:
            return reduce_over(x, dim, lambda a: np.nanmean(np.abs(a - np.nanmean(a, axis=0)), axis=0))
        return reduce_over(x, dim, lambda a: np.nanmedian(np.abs(a - np.nanmedian(a, axis=0)), axis=0))


@builtin("zscore")
def _zscore(interp, args, nargout):
    from .stats_desc import reduce_over
    nargs(args, 1, 3, "zscore")
    x = fnum(args[0]).astype(float)
    flag = int(_scalar(args[1])) if _given(args, 1) else 0
    if flag not in (0, 1):
        raise MatlabError("zscore: the flag must be 0 or 1.")
    if x.size == 0:
        return x
    dim = _dim_spec(args, 2)
    mu = reduce_over(x, dim, lambda a: a.mean(axis=0))
    sd = reduce_over(x, dim, lambda a: a.std(axis=0, ddof=0 if (flag == 1 or a.shape[0] == 1) else 1))
    sd0 = np.where(sd == 0, 1.0, sd)
    z = (x - mu) / sd0
    return (z, mu, sd)[:max(nargout, 1)] if nargout > 1 else z


def _moment_stat(kind):
    def f(interp, args, nargout):
        nargs(args, 1, 3, kind)
        x = fnum(args[0]).astype(float)
        flag = int(_scalar(args[1])) if _given(args, 1) else 1
        if x.size == 0:
            return np.nan
        axis = get_dim(args, 2, x)
        xm = np.moveaxis(x, axis, 0)
        out = np.empty(xm.shape[1:])
        for j in range(xm.shape[1]):
            v = xm[:, j]
            v = v[~np.isnan(v)]
            n = v.size
            d = v - v.mean() if n else v
            m2 = np.mean(d ** 2) if n else np.nan
            with np.errstate(all="ignore"):
                if kind == "skewness":
                    g = np.mean(d ** 3) / m2 ** 1.5 if n else np.nan
                    if flag == 0:
                        g = g * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else np.nan
                else:
                    g = np.mean(d ** 4) / m2 ** 2 if n else np.nan
                    if flag == 0:
                        g = 3 + (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g - 3 * (n - 1)) if n > 3 else np.nan
            out[j] = g
        return keep2d(out, axis)
    return f


builtin("skewness")(_moment_stat("skewness"))
builtin("kurtosis")(_moment_stat("kurtosis"))


def _as_obs_matrix(v) -> np.ndarray:
    a = fnum(v).astype(float) if not np.iscomplexobj(fnum(v)) else fnum(v)
    if a.shape[0] == 1:
        a = a.reshape(-1, 1)
    return a


@builtin("cov")
def _cov(interp, args, nargout):
    rest = list(args)
    nanflag = None
    if rest and is_text(rest[-1]):
        nanflag = text(rest.pop()).lower()
    nargs(rest, 1, 3, "cov")
    w = 0
    if len(rest) > 1 and fnum(rest[-1]).size == 1 and (len(rest) == 3 or fnum(rest[0]).size != 1):
        if len(rest) == 3 or float(fnum(rest[-1]).flat[0]) in (0.0, 1.0) and fnum(rest[0]).size > 1:
            w = int(_scalar(rest.pop()))
    if len(rest) == 2:
        x, y = fnum(rest[0]).flatten(), fnum(rest[1]).flatten()
        if x.size != y.size:
            raise MatlabError("cov: inputs must have the same number of elements.")
        X = np.column_stack([x, y])
    else:
        X = _as_obs_matrix(rest[0])
    if nanflag == "omitrows":
        X = X[~np.isnan(X).any(axis=1)]
    n = X.shape[0]
    if n == 0:
        return np.full((X.shape[1], X.shape[1]), np.nan)
    ddof = 0 if (w == 1 or n == 1) else 1
    Xc = X - X.mean(axis=0)
    C = (Xc.conj().T @ Xc) / (n - ddof)
    return C


def _pearson(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    with np.errstate(all="ignore"):
        num = Xc.T @ Yc
        den = np.sqrt(np.outer((Xc ** 2).sum(axis=0), (Yc ** 2).sum(axis=0)))
        return np.clip(num / den, -1, 1)


def _corr_pvals(R: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        t = R * np.sqrt((n - 2) / (1 - R ** 2))
        p = 2 * st.t.sf(np.abs(t), n - 2)
    return np.where(np.abs(R) >= 1, np.where(np.isnan(R), np.nan, 0.0), p)


@builtin("corrcoef")
def _corrcoef(interp, args, nargout):
    nargs(args, 1, None, "corrcoef")
    plain = [a for a in args if not is_text(a)]
    if len(plain) >= 2:
        x, y = fnum(plain[0]).flatten(), fnum(plain[1]).flatten()
        if x.size != y.size:
            raise MatlabError("corrcoef: inputs must have the same number of elements.")
        X = np.column_stack([x, y]).astype(float)
    else:
        X = _as_obs_matrix(plain[0])
    R = _pearson(X, X)
    np.fill_diagonal(R, np.where(np.isnan(np.diag(R)), np.nan, 1.0))
    if nargout > 1:
        P = _corr_pvals(R, X.shape[0])
        np.fill_diagonal(P, 1.0)
        return R, P
    return R


@builtin("regress")
def _regress(interp, args, nargout):
    nargs(args, 2, 3, "regress")
    y = fnum(args[0]).astype(float)
    X = fnum(args[1]).astype(float)
    alpha = _scalar(args[2]) if len(args) > 2 else 0.05
    if y.ndim != 2 or (y.shape[1] != 1 and y.shape[0] != 1):
        raise MatlabError("regress: y must be a vector.")
    y = y.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise MatlabError("regress: X and y must have the same number of rows.")
    ok = ~(np.isnan(y).ravel() | np.isnan(X).any(axis=1))
    Xo, yo = X[ok], y[ok]
    n, p = Xo.shape
    b, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    b = b.reshape(-1, 1)
    r_full = np.full((y.shape[0], 1), np.nan)
    r = yo - Xo @ b
    r_full[ok] = r
    dfe = n - p
    with np.errstate(all="ignore"):
        s2 = float((r ** 2).sum() / dfe) if dfe > 0 else np.nan
        XtXi = np.linalg.pinv(Xo.T @ Xo)
        se = np.sqrt(np.diag(XtXi) * s2).reshape(-1, 1)
        tval = st.t.ppf(1 - alpha / 2, dfe) if dfe > 0 else np.nan
        bint = np.hstack([b - tval * se, b + tval * se])
        outs = [b, bint, r_full]
        if nargout > 3:
            h = np.einsum("ij,jk,ik->i", Xo, XtXi, Xo)
            ri = r.ravel()
            si2 = ((dfe * s2) - ri ** 2 / (1 - h)) / (dfe - 1)
            half = tval * np.sqrt(np.clip(si2, 0, None) * (1 - h))
            rint = np.full((y.shape[0], 2), np.nan)
            rint[ok] = np.column_stack([ri - half, ri + half])
            outs.append(rint)
        if nargout > 4:
            yc = yo - yo.mean()
            sst = float((yc ** 2).sum())
            sse = float((r ** 2).sum())
            r2 = 1 - sse / sst
            F = ((sst - sse) / (p - 1)) / (sse / dfe) if p > 1 else np.nan
            pval = float(st.f.sf(F, p - 1, dfe)) if p > 1 else np.nan
            outs.append(np.array([[r2, F, pval, s2]]))
    return tuple(outs[:max(nargout, 1)])


# -- histograms and tabulation ------------------------------------------------------------

@builtin("histc")
def _histc(interp, args, nargout):
    nargs(args, 2, 3, "histc")
    x = fnum(args[0]).astype(float)
    edges = fnum(args[1]).flatten().astype(float)
    if edges.size and np.any(np.diff(edges) < 0):
        raise MatlabError("histc: edges must be monotonically non-decreasing.")
    vec = x.shape[0] == 1 or x.shape[1] == 1
    axis = get_dim(args, 2, x) if len(args) > 2 else (default_dim(x) if vec else 0)

    def bins_of(v):
        b = np.searchsorted(edges, v, side="right").astype(float)
        b[(v < edges[0]) | (v > edges[-1]) | np.isnan(v)] = 0
        if edges.size:
            b[v == edges[-1]] = edges.size
        return b

    if edges.size == 0:
        return np.zeros((0, 0))
    bins = bins_of(x.ravel()).reshape(x.shape)
    xm = np.moveaxis(bins, axis, 0)
    counts = np.zeros((edges.size,) + xm.shape[1:])
    for j in range(xm.shape[1]):
        col = xm[:, j]
        col = col[col > 0].astype(int)
        counts[:, j] = np.bincount(col - 1, minlength=edges.size)[:edges.size]
    counts = np.moveaxis(counts, 0, axis)
    return (counts, bins) if nargout > 1 else counts


# -- NaN-ignoring reductions --------------------------------------------------------------

def _nan_reduce(name, fn, empty):
    def f(interp, args, nargout):
        nargs(args, 1, 2, name)
        x = fnum(args[0]).astype(float) if fnum(args[0]).dtype.kind != "c" else fnum(args[0])
        if x.size == 0 and len(args) < 2:
            return np.array([[empty]])
        axis = get_dim(args, 1, x)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = fn(x, axis=axis)
        return keep2d(np.asarray(r), axis)
    return f


builtin("nanmean")(_nan_reduce("nanmean", np.nanmean, np.nan))
builtin("nansum")(_nan_reduce("nansum", np.nansum, 0.0))


@builtin("nanstd")
def _nanstd(interp, args, nargout):
    nargs(args, 1, 3, "nanstd")
    x = fnum(args[0]).astype(float)
    flag = int(_scalar(args[1])) if _given(args, 1) else 0
    if x.size == 0:
        return np.nan
    axis = get_dim(args, 2, x)
    cnt = np.sum(~np.isnan(x), axis=axis)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=axis, keepdims=True)
        ss_ = np.nansum((x - mu) ** 2, axis=axis)
        denom = np.where((flag == 1) | (cnt == 1), cnt, cnt - 1)
        r = np.sqrt(ss_ / denom)
        r = np.where(cnt == 0, np.nan, r)
    return keep2d(np.asarray(r), axis)


def _nan_extreme(name, pick, pick2):
    def f(interp, args, nargout):
        nargs(args, 1, 3, name)
        if len(args) >= 2 and _given(args, 1):
            a, b = _bcast(name, args[0], args[1])
            return pick2(a, b)
        x = fnum(args[0]).astype(float)
        if x.size == 0:
            return (np.zeros((0, 0)), np.zeros((0, 0))) if nargout > 1 else np.zeros((0, 0))
        axis = get_dim(args, 2, x)
        filled = np.where(np.isnan(x), np.inf if pick is np.argmin else -np.inf, x)
        idx = pick(filled, axis=axis)
        vals = np.take_along_axis(filled, np.expand_dims(idx, axis), axis=axis)
        allnan = np.all(np.isnan(x), axis=axis, keepdims=True)
        vals = np.where(allnan, np.nan, vals)
        idx2 = np.where(allnan, 1.0, np.expand_dims(idx, axis) + 1.0)
        return (vals, idx2) if nargout > 1 else vals
    return f


builtin("nanmax")(_nan_extreme("nanmax", np.argmax, np.fmax))
builtin("nanmin")(_nan_extreme("nanmin", np.argmin, np.fmin))


# -- moving-window statistics -----------------------------------------------------------

def _window_extent(k) -> tuple[int, int]:
    kv = fnum(k).flatten()
    if kv.size == 1:
        w = float(kv[0])
        if w <= 0:
            raise MatlabError("Window length must be positive.")
        if w != int(w):
            # non-integer lengths are centred: floor(w/2) on each side
            h = int(np.floor(w / 2))
            return h, h
        w = int(w)
        if w % 2:
            return (w - 1) // 2, (w - 1) // 2
        return w // 2, w // 2 - 1
    if kv.size == 2:
        return int(kv[0]), int(kv[1])
    raise MatlabError("Window must be a scalar or a two-element vector.")


def _parse_mov(args, name):
    nargs(args, 2, None, name)
    x = fnum(args[0]).astype(float)
    rest = list(args[2:])
    dim = None
    nanflag = "includenan"
    endpoints = "shrink"
    weight = 0
    samples = None
    numeric_pos: list = []
    k = 0
    while k < len(rest):
        a = rest[k]
        if is_text(a):
            s = text(a).lower()
            if s in ("omitnan", "includenan", "omitmissing", "includemissing"):
                nanflag = "omitnan" if s.startswith("omit") else "includenan"
                k += 1
                continue
            if s == "endpoints" and k + 1 < len(rest):
                e = rest[k + 1]
                if is_text(e):
                    endpoints = text(e).lower()
                    if endpoints not in ("shrink", "discard", "fill"):
                        raise MatlabError(f"{name}: Endpoints must be 'shrink', 'discard', 'fill' or a value.")
                else:
                    endpoints = float(_scalar(e))
                k += 2
                continue
            if s == "samplepoints" and k + 1 < len(rest):
                samples = np.real(fnum(rest[k + 1])).astype(float).reshape(-1)
                k += 2
                continue
            raise MatlabError(f"{name}: unknown option '{s}'.")
        numeric_pos.append(a)
        k += 1
    if name in ("movstd", "movvar") and numeric_pos:
        w = numeric_pos.pop(0)
        weight = int(_scalar(w)) if _given([w], 0) else 0
        if weight not in (0, 1):
            raise MatlabError(f"{name}: the weight must be 0 or 1.")
    if numeric_pos:
        dim = to_int(numeric_pos[0], "dimension") - 1
        if dim < 0:
            raise MatlabError("Dimension argument must be a positive integer.")
    if len(numeric_pos) > 1:
        raise MatlabError(f"Too many input arguments for {name}.")
    axis = dim if dim is not None else default_dim(x)
    if samples is None:
        before, after = _window_extent(args[1])
        window = None
    else:
        n = x.shape[axis] if axis < x.ndim else 1
        if samples.size != n or np.any(np.diff(samples) <= 0):
            raise MatlabError(f"{name}: SamplePoints must be increasing, one per element along the "
                              "operating dimension.")
        kv = np.real(fnum(args[1])).astype(float).reshape(-1)
        if kv.size not in (1, 2) or np.any(kv < 0):
            raise MatlabError("Window must be a positive scalar or a two-element vector.")
        before = after = 0
        # a scalar window is centred and half open: [t - k/2, t + k/2)
        window = (kv[0] / 2.0, kv[0] / 2.0, True) if kv.size == 1 else (kv[0], kv[1], False)
        if endpoints == "discard":
            raise MatlabError(f"{name}: Endpoints 'discard' cannot be combined with SamplePoints.")
    return x, before, after, axis, nanflag, endpoints, weight, (samples, window)


def _moving(name, reducer):
    def f(interp, args, nargout):
        x, before, after, axis, nanflag, endpoints, weight, (samples, window) = _parse_mov(args, name)
        if x.size == 0 or axis >= x.ndim:
            return x
        xm = np.moveaxis(x, axis, 0)
        n = xm.shape[0]
        out = np.empty_like(xm)
        omit = nanflag == "omitnan"
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n):
                if samples is not None:
                    d = samples - samples[i]
                    back, fwd, half_open = window
                    sel = (d >= -back) & ((d < fwd) if half_open else (d <= fwd))
                    if not sel.any():
                        sel[i] = True
                    win = xm[sel]
                    partial = (samples[i] - back < samples[0]) or (samples[i] + fwd > samples[-1])
                else:
                    lo, hi = i - before, i + after
                    win = xm[max(lo, 0):min(hi, n - 1) + 1]
                    partial = lo < 0 or hi > n - 1
                out[i] = reducer(win, omit, weight)
                if endpoints != "shrink" and partial:
                    if endpoints == "fill":
                        out[i] = np.nan
                    elif not isinstance(endpoints, str):
                        out[i] = endpoints
        if endpoints == "discard":
            first = before
            last = n - after
            out = out[first:max(last, first)]
        return np.moveaxis(out, 0, axis)
    return f


def _std_red(w, omit, weight):
    cnt = np.sum(~np.isnan(w), axis=0) if omit else np.full(w.shape[1:], w.shape[0])
    mu = (np.nanmean if omit else np.mean)(w, axis=0)
    ss_ = (np.nansum if omit else np.sum)((w - mu) ** 2, axis=0)
    denom = np.where((weight == 1) | (cnt <= 1), cnt, cnt - 1)
    r = np.sqrt(ss_ / denom)
    return np.where(cnt == 0, np.nan, np.where(cnt == 1, 0.0, r))


def _sum_red(w, omit, weight):
    return np.nansum(w, axis=0) if omit else np.sum(w, axis=0)


def _ext_red(fn, nanfn):
    return lambda w, omit, weight: (nanfn if omit else fn)(w, axis=0)


builtin("movmean")(_moving("movmean", _ext_red(np.mean, np.nanmean)))
builtin("movsum")(_moving("movsum", _sum_red))
builtin("movmax")(_moving("movmax", lambda w, omit, wt: np.max(w, axis=0) if not omit
                          else np.nanmax(w, axis=0)))
builtin("movmin")(_moving("movmin", lambda w, omit, wt: np.min(w, axis=0) if not omit
                          else np.nanmin(w, axis=0)))
builtin("movmedian")(_moving("movmedian", _ext_red(np.median, np.nanmedian)))
builtin("movstd")(_moving("movstd", _std_red))
builtin("movvar")(_moving("movvar", lambda w, omit, wt: _std_red(w, omit, wt) ** 2))


def _local_regression(xs: np.ndarray, y: np.ndarray, sel_of, degree: int, robust: bool) -> np.ndarray:
    """Locally weighted polynomial fit (tricube weights), optionally with robust reweighting."""
    n = y.size
    out = np.full(n, np.nan)
    ok = ~np.isnan(y)
    extra = np.ones(n)
    for _ in range(6 if robust else 1):
        for i in range(n):
            sel = sel_of(i) & ok
            if not sel.any():
                continue
            d = np.abs(xs[sel] - xs[i])
            dmax = d.max()
            u = d / (dmax * (1 + 1e-12)) if dmax > 0 else np.zeros(d.size)
            w = (1 - u ** 3) ** 3 * extra[sel]
            if not np.any(w > 0):
                w = np.ones(d.size)
            deg = min(degree, int(np.count_nonzero(w > 0)) - 1)
            t = xs[sel] - xs[i]
            basis = np.vander(t, deg + 1, increasing=True)
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(basis * sw[:, None], y[sel] * sw, rcond=None)[0]
            out[i] = coef[0]
        if not robust:
            break
        resid = y - out
        scale = np.nanmedian(np.abs(resid[ok]))
        # residuals at rounding level mean the fit is already exact: nothing to reweight
        if not scale > 1e-12 * max(float(np.nanmax(np.abs(y[ok]))), 1e-300):
            break
        r = resid / (6.0 * scale)
        extra = np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)
        extra = np.where(ok, extra, 0.0)
    return out


_SMOOTH = ("movmean", "movmedian", "gaussian", "sgolay", "lowess", "loess", "rlowess", "rloess")


@builtin("smoothdata")
def _smoothdata(interp, args, nargout):
    nargs(args, 1, None, "smoothdata")
    if nargout > 2:
        raise MatlabError("smoothdata: too many output arguments.")
    x = fnum(args[0]).astype(float)
    rest = list(args[1:])
    dim = None
    if rest and not is_text(rest[0]):
        dim = to_int(rest.pop(0), "dimension") - 1
    method = "movmean"
    if rest and is_text(rest[0]) and text(rest[0]).lower() in _SMOOTH:
        method = text(rest.pop(0)).lower()
    window = None
    if rest and not is_text(rest[0]):
        window = rest.pop(0)
    nanflag = "omitnan"
    degree = 2
    factor = None
    samples = None
    k = 0
    while k < len(rest):
        if not is_text(rest[k]):
            raise MatlabError("smoothdata: expected an option name.")
        s = text(rest[k]).lower()
        if s in ("omitnan", "includenan", "omitmissing", "includemissing"):
            nanflag = "omitnan" if s.startswith("omit") else "includenan"
            k += 1
        elif s == "degree" and k + 1 < len(rest):
            degree = to_int(rest[k + 1])
            if method != "sgolay":
                raise MatlabError("smoothdata: Degree applies only to the 'sgolay' method.")
            k += 2
        elif s == "smoothingfactor" and k + 1 < len(rest):
            factor = _scalar(rest[k + 1])
            if not 0 <= factor <= 1:
                raise MatlabError("smoothdata: SmoothingFactor must be between 0 and 1.")
            k += 2
        elif s == "samplepoints" and k + 1 < len(rest):
            samples = np.real(fnum(rest[k + 1])).astype(float).reshape(-1)
            k += 2
        else:
            raise MatlabError(f"smoothdata: unknown option '{text(rest[k])}'.")
    if factor is not None and window is not None:
        raise MatlabError("smoothdata: give a window or a SmoothingFactor, not both.")
    axis = dim if dim is not None else default_dim(x)
    if x.size == 0 or axis >= x.ndim:
        return (x, np.zeros((0, 0))) if nargout > 1 else x
    n = x.shape[axis]
    if samples is not None and (samples.size != n or np.any(np.diff(samples) <= 0)):
        raise MatlabError("smoothdata: SamplePoints must be increasing, one per element along the "
                          "operating dimension.")
    span = float(samples[-1] - samples[0]) if samples is not None and n > 1 else float(n - 1)
    if window is None:
        if factor is not None:
            # 0 keeps the data as it is, 1 uses a window as long as the whole signal
            width = factor * span + (0.0 if samples is not None else 1.0)
        else:
            # default: about a tenth of the data, at least three points
            width = float(max(3, int(round(n / 10)))) if samples is None else max(span / 10.0, 0.0)
        if samples is None:
            width = float(max(1, int(np.ceil(width))))
        window = np.array([[width]])
    win_out = fnum(window).astype(float).reshape(1, -1)

    def done(result):
        return (result, win_out) if nargout > 1 else result

    sp_args = [char("SamplePoints"), samples.reshape(1, -1)] if samples is not None else []
    if method in ("movmean", "movmedian"):
        fn = _moving(method, _ext_red(np.mean, np.nanmean) if method == "movmean"
                     else _ext_red(np.median, np.nanmedian))
        return done(fn(interp, [x, window, np.array([[axis + 1.0]]), char(nanflag), *sp_args], 1))
    xs = samples if samples is not None else np.arange(n, dtype=float)
    kv = fnum(window).astype(float).reshape(-1)
    if kv.size not in (1, 2) or np.any(kv < 0):
        raise MatlabError("smoothdata: the window must be a positive scalar or a two-element vector.")
    if samples is None:
        before, after = _window_extent(window)

        def sel_of(i):
            sel = np.zeros(n, dtype=bool)
            sel[max(i - before, 0):min(i + after, n - 1) + 1] = True
            return sel
    else:
        back, fwd = (kv[0] / 2.0, kv[0] / 2.0) if kv.size == 1 else (kv[0], kv[1])

        def sel_of(i):
            d = xs - xs[i]
            sel = (d >= -back) & (d <= fwd)
            sel[i] = True
            return sel
    xm = np.moveaxis(x, axis, 0)
    cols = xm.reshape(n, -1)
    out = np.empty(cols.shape)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        if method == "gaussian":
            # the standard deviation is a fifth of the full window width
            full = float(before + after) if samples is None else float(back + fwd)
            sigma = max(full, 1.0 if samples is None else 1e-300) / 5.0
            for c in range(cols.shape[1]):
                col = cols[:, c]
                for i in range(n):
                    sel = sel_of(i)
                    d = xs[sel] - xs[i]
                    wts = np.exp(-0.5 * (d / sigma) ** 2)
                    seg = col[sel]
                    if nanflag == "omitnan":
                        mask = ~np.isnan(seg)
                        out[i, c] = np.sum(np.where(mask, seg, 0) * wts) / np.sum(wts * mask)
                    else:
                        out[i, c] = np.sum(seg * wts) / np.sum(wts)
        elif method == "sgolay":
            for c in range(cols.shape[1]):
                col = cols[:, c]
                for i in range(n):
                    sel = sel_of(i) & ~np.isnan(col)
                    if nanflag == "includenan" and np.isnan(col[sel_of(i)]).any():
                        out[i, c] = np.nan
                        continue
                    cnt = int(sel.sum())
                    if cnt == 0:
                        out[i, c] = np.nan
                        continue
                    deg = min(degree, cnt - 1)
                    t = xs[sel] - xs[i]
                    out[i, c] = np.linalg.lstsq(np.vander(t, deg + 1, increasing=True), col[sel],
                                                rcond=None)[0][0]
        else:
            deg = 1 if method.endswith("lowess") else 2
            for c in range(cols.shape[1]):
                col = cols[:, c]
                res = _local_regression(xs, col, sel_of, deg, method.startswith("r"))
                if nanflag == "includenan":
                    bad = np.array([np.isnan(col[sel_of(i)]).any() for i in range(n)])
                    res = np.where(bad, np.nan, res)
                out[:, c] = res
    return done(np.moveaxis(out.reshape(xm.shape), 0, axis))


_NORM_METHODS = ("zscore", "norm", "range", "center", "scale", "medianiqr")


@builtin("normalize")
def _normalize(interp, args, nargout):
    nargs(args, 1, 7, "normalize")
    if nargout > 3:
        raise MatlabError("normalize: too many output arguments.")
    x = fnum(args[0]).astype(float)
    rest = list(args[1:])
    dim = None
    if rest and not is_text(rest[0]):
        dim = to_int(rest.pop(0), "dimension") - 1
        if dim < 0:
            raise MatlabError("Dimension argument must be a positive integer.")
    steps: list[tuple[str, object]] = []
    k = 0
    while k < len(rest):
        if not is_text(rest[k]):
            raise MatlabError("normalize: expected a method name.")
        method = text(rest[k]).lower()
        if method not in _NORM_METHODS:
            raise MatlabError(f"normalize: unknown method '{text(rest[k])}'.")
        param = None
        if k + 1 < len(rest) and not (is_text(rest[k + 1]) and text(rest[k + 1]).lower() in _NORM_METHODS):
            param = rest[k + 1]
            k += 1
        steps.append((method, param))
        k += 1
    if not steps:
        steps = [("zscore", None)]
    names = [m for m, _ in steps]
    if len(steps) > 2 or (len(steps) == 2 and sorted(names) != ["center", "scale"]):
        raise MatlabError("normalize: only 'center' and 'scale' can be combined.")
    axis = dim if dim is not None else default_dim(x)
    if x.size == 0 or axis >= x.ndim:
        outs = (x, np.zeros(x.shape), np.ones(x.shape))
        return outs[:max(nargout, 1)] if nargout > 1 else x
    import warnings

    def std_of(v):
        return np.nanstd(v, axis=axis, keepdims=True, ddof=1 if v.shape[axis] > 1 else 0)

    def med_dev(v):
        med = np.nanmedian(v, axis=axis, keepdims=True)
        return np.nanmedian(np.abs(v - med), axis=axis, keepdims=True)

    def iqr_of(v):
        q = np.nanpercentile(v, [25, 75], axis=axis, method="hazen")
        return np.expand_dims(q[1] - q[0], axis)

    def kind_of(param, allowed, what):
        s = text(param).lower()
        if s not in allowed:
            raise MatlabError(f"normalize: unknown {what} type '{text(param)}'.")
        return s

    center = np.zeros([1] * x.ndim)
    scale = np.ones([1] * x.ndim)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for method, param in steps:
            if method == "zscore":
                kind = kind_of(param, ("std", "robust"), "zscore") if param is not None else "std"
                if kind == "std":
                    center, scale = np.nanmean(x, axis=axis, keepdims=True), std_of(x)
                else:
                    center, scale = np.nanmedian(x, axis=axis, keepdims=True), med_dev(x)
            elif method == "medianiqr":
                if param is not None:
                    raise MatlabError("normalize: the 'medianiqr' method takes no parameter.")
                center, scale = np.nanmedian(x, axis=axis, keepdims=True), iqr_of(x)
            elif method == "range":
                lim = fnum(param).flatten() if param is not None else np.array([0.0, 1.0])
                if lim.size != 2 or not lim[0] < lim[1]:
                    raise MatlabError("normalize: the range must be a two-element increasing vector.")
                lo, hi = np.nanmin(x, axis=axis, keepdims=True), np.nanmax(x, axis=axis, keepdims=True)
                scale = (hi - lo) / (lim[1] - lim[0])
                center = lo - lim[0] * scale
            elif method == "norm":
                p = _scalar(param) if param is not None else 2.0
                if not p > 0:
                    raise MatlabError("normalize: the norm type must be a positive number or Inf.")
                if np.isinf(p):
                    scale = np.nanmax(np.abs(x), axis=axis, keepdims=True)
                else:
                    scale = np.nansum(np.abs(x) ** p, axis=axis, keepdims=True) ** (1 / p)
            elif method == "center":
                if param is None or is_text(param):
                    kind = kind_of(param, ("mean", "median"), "center") if param is not None else "mean"
                    center = (np.nanmean if kind == "mean" else np.nanmedian)(x, axis=axis, keepdims=True)
                else:
                    center = fnum(param).astype(float)
            else:  # scale
                if param is None or is_text(param):
                    kind = kind_of(param, ("std", "mad", "first", "iqr"), "scale") if param is not None else "std"
                    if kind == "std":
                        scale = std_of(x)
                    elif kind == "mad":
                        scale = med_dev(x)
                    elif kind == "first":
                        scale = np.take(x, [0], axis=axis)
                    else:
                        scale = iqr_of(x)
                else:
                    scale = fnum(param).astype(float)
        out = (x - center) / scale
    if nargout <= 1:
        return out
    shape = list(x.shape)
    shape[axis] = 1
    c_out = np.broadcast_to(center, shape).copy() if np.asarray(center).size <= np.prod(shape) else center
    s_out = np.broadcast_to(scale, shape).copy() if np.asarray(scale).size <= np.prod(shape) else scale
    return (out, c_out, s_out)[:nargout]


@builtin("rescale")
def _rescale(interp, args, nargout):
    nargs(args, 1, None, "rescale")
    x = fnum(args[0]).astype(float)
    pos = []
    for a in args[1:3]:
        if is_text(a):
            break
        pos.append(a)
    k = 1 + len(pos)
    lo = _scalar(pos[0]) if len(pos) > 0 else 0.0
    hi = _scalar(pos[1]) if len(pos) > 1 else 1.0
    from ._util import parse_options
    opts = {key.lower(): v for key, v in parse_options(list(args[k:]), ("InputMin", "InputMax"), "rescale",
                                                        first_arg=k + 1).items()}
    if x.size == 0:
        return x
    imin = fnum(opts["inputmin"]) if "inputmin" in opts else np.nanmin(x)
    imax = fnum(opts["inputmax"]) if "inputmax" in opts else np.nanmax(x)
    xc = np.clip(x, imin, imax)
    span = imax - imin
    with np.errstate(all="ignore"):
        r = lo + (xc - imin) / span * (hi - lo)
    return np.where(np.broadcast_to(span == 0, r.shape), lo, r)


# -- distances and clustering ------------------------------------------------------------

_METRICS = {"euclidean": "euclidean", "squaredeuclidean": "sqeuclidean", "cityblock": "cityblock",
            "chebychev": "chebyshev", "chebyshev": "chebyshev", "minkowski": "minkowski",
            "cosine": "cosine", "correlation": "correlation", "hamming": "hamming",
            "jaccard": "jaccard", "seuclidean": "seuclidean", "spearman": "spearman"}


@builtin("squareform")
def _squareform(interp, args, nargout):
    from scipy.spatial.distance import squareform
    nargs(args, 1, 2, "squareform")
    a = fnum(args[0]).astype(float)
    if len(args) > 1:
        mode = text(args[1]).lower()
        if mode == "tovector":
            return squareform(a, force="tovector", checks=False).reshape(1, -1)
        return squareform(a.flatten(), force="tomatrix")
    if a.shape[0] == 1 or a.shape[1] == 1:
        if a.size == 1 and a.shape == (1, 1):
            if a.flat[0] != 0:
                return squareform(a.flatten(), force="tomatrix")
            return np.zeros((1, 0))
        return squareform(a.flatten(), force="tomatrix")
    if a.shape[0] != a.shape[1]:
        raise MatlabError("squareform: input must be a vector or a square matrix.")
    return squareform(a, force="tovector", checks=False).reshape(1, -1)
