"""Descriptive statistics, grouped summaries, correlation, resampling and data partitions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, char
from ._registry import builtin
from ._stats_util import (
    StatObject, as_bool, builtin_call, call, cellstr, choice, get_opts, given, group_codes, group_list, intval,
    is_categorical, is_empty_arg, is_table, make_table, multi_group, num_text, outs, rng_of, scalar, text_list,
    tied_rank, vec,
)
from ._util import default_dim, fnum, is_text, text


# -- reductions over dimensions --------------------------------------------------------------------------

def reduce_over(x: np.ndarray, dim: Any, fn: Callable) -> np.ndarray:
    """Apply ``fn`` (reducing axis 0 of an (N, M) array) over ``dim``: None, 'all' or dimension numbers."""
    x = np.asarray(x, dtype=float)
    nd = x.ndim
    if dim == "all":
        axes = list(range(nd))
    elif dim is None:
        axes = [default_dim(x)]
    else:
        dims = sorted({int(d) for d in np.atleast_1d(dim)})
        if any(d < 1 for d in dims):
            raise MatlabError("Dimension arguments must be positive integers.")
        axes = [d - 1 for d in dims if d - 1 < nd]
    if not axes:
        a = x.reshape(1, -1)
        return np.asarray(fn(a), dtype=float).reshape(x.shape)
    moved = np.moveaxis(x, axes, list(range(len(axes))))
    rest = moved.shape[len(axes):]
    a = moved.reshape(-1, int(np.prod(rest)) if rest else 1)
    r = np.asarray(fn(a), dtype=float)
    shape = list(x.shape)
    for ax in axes:
        shape[ax] = 1
    return r.reshape(shape)


def dim_arg(args: list, pos: int) -> Any:
    """Dimension specification at ``args[pos]``: None when absent or empty, 'all', or numbers."""
    if len(args) <= pos or is_empty_arg(args[pos]):
        return None
    a = args[pos]
    if is_text(a):
        if text(a).lower() != "all":
            raise MatlabError(f"Unknown option '{text(a)}'.")
        return "all"
    return [int(d) for d in vec(a)]


def _nanfn(fn: Callable) -> Callable:
    def g(a):
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return fn(a)
    return g


def _pop_nanflag(args: list, default: str) -> tuple[list, bool]:
    """Remove a trailing 'omitnan'/'includenan' flag; returns (args, omit)."""
    rest = list(args)
    omit = default == "omitnan"
    if rest and is_text(rest[-1]) and text(rest[-1]).lower() in ("omitnan", "includenan", "omitmissing",
                                                                   "includemissing"):
        omit = text(rest.pop()).lower().startswith("omit")
    return rest, omit


def _mean_like(name: str, core: Callable) -> None:
    @builtin(name)
    def f(interp, args, nargout):
        rest, omit = _pop_nanflag(args, "includenan")
        if not 1 <= len(rest) <= 2:
            raise MatlabError(f"{name}: expected {name}(X), {name}(X, dim), {name}(X, 'all') or {name}(X, vecdim).")
        x = np.asarray(fnum(rest[0]), dtype=float)
        if x.size == 0:
            return np.array([[np.nan]])
        return reduce_over(x, dim_arg(rest, 1), _nanfn(lambda a: core(a, omit)))


def _geo(a, omit):
    cnt = np.sum(~np.isnan(a), axis=0) if omit else a.shape[0]
    s = (np.nansum if omit else np.sum)(np.log(np.where(a < 0, np.nan, a)), axis=0)
    r = np.exp(s / cnt)
    return np.where(np.any(a < 0, axis=0), np.nan, r)


def _harm(a, omit):
    cnt = np.sum(~np.isnan(a), axis=0) if omit else a.shape[0]
    return cnt / (np.nansum if omit else np.sum)(1.0 / a, axis=0)


_mean_like("geomean", _geo)
_mean_like("harmmean", _harm)


@builtin("moment")
def _moment(interp, args, nargout):
    if not 2 <= len(args) <= 3:
        raise MatlabError("moment: expected moment(X, order), moment(X, order, dim) or moment(X, order, 'all').")
    x = np.asarray(fnum(args[0]), dtype=float)
    order = intval(args[1], "order")
    if order < 0:
        raise MatlabError("moment: the order must be a non-negative integer.")
    if x.size == 0:
        return np.array([[np.nan]])

    def core(a):
        if order == 1:
            return np.zeros(a.shape[1])
        return np.mean((a - a.mean(axis=0)) ** order, axis=0)
    return reduce_over(x, dim_arg(args, 2), _nanfn(core))


@builtin("trimmean")
def _trimmean(interp, args, nargout):
    if not 2 <= len(args) <= 4:
        raise MatlabError("trimmean: expected trimmean(X, percent), trimmean(X, percent, flag) or "
                          "trimmean(X, percent, flag, dim).")
    x = np.asarray(fnum(args[0]), dtype=float)
    pct = scalar(args[1], "percent")
    if not 0 <= pct < 100:
        raise MatlabError("trimmean: percent must be at least 0 and less than 100.")
    rest = list(args[2:])
    flag = "round"
    if rest and is_text(rest[0]) and text(rest[0]).lower() in ("round", "floor", "weighted"):
        flag = text(rest.pop(0)).lower()
    elif rest and is_empty_arg(rest[0]) and len(rest) > 1:
        rest.pop(0)
    if len(rest) > 1:
        raise MatlabError("trimmean: too many input arguments.")
    dim = dim_arg(rest, 0)
    if x.size == 0:
        return np.array([[np.nan]])

    def core(a):
        out = np.empty(a.shape[1])
        for j in range(a.shape[1]):
            v = np.sort(a[:, j][~np.isnan(a[:, j])])
            n = v.size
            if n == 0:
                out[j] = np.nan
                continue
            k = n * pct / 200.0
            if flag == "weighted":
                i, f = int(np.floor(k)), k - np.floor(k)
                w = np.ones(n)
                w[:i] = 0
                w[n - i:] = 0
                if i < n - i:
                    w[i] = 1 - f
                    w[n - i - 1] = 1 - f
                out[j] = np.sum(w * v) / np.sum(w) if np.sum(w) > 0 else np.nan
                continue
            if flag == "round":
                k0 = int(np.ceil(k - 0.5)) if k - np.floor(k) == 0.5 else int(np.round(k))
            else:
                k0 = int(np.floor(k))
            out[j] = v[k0:n - k0].mean() if n - 2 * k0 > 0 else np.nan
        return out
    return reduce_over(x, dim, core)


@builtin("tiedrank")
def _tiedrank(interp, args, nargout):
    if not 1 <= len(args) <= 3:
        raise MatlabError("tiedrank: expected tiedrank(X), tiedrank(X, tieflag) or "
                          "tiedrank(X, tieflag, bidirectional).")
    x = np.asarray(fnum(args[0]), dtype=float)
    tieflag = bool(scalar(args[1], "tieflag")) if given(args, 1) else False
    bidir = bool(scalar(args[2], "bidirectional")) if given(args, 2) else False
    is_row = x.ndim == 2 and x.shape[0] == 1
    X = x.reshape(-1, 1) if is_row else x.reshape(x.shape[0], -1)
    R = np.empty(X.shape)
    adj = []
    for j in range(X.shape[1]):
        r = tied_rank(X[:, j])
        v = X[:, j][~np.isnan(X[:, j])]
        _, counts = np.unique(v, return_counts=True)
        t = counts[counts > 1].astype(float)
        if tieflag:
            adj.append([np.sum(t * (t - 1) / 2), np.sum(t * (t - 1) * (t - 2)), np.sum(t * (t - 1) * (2 * t + 5))])
        else:
            adj.append([np.sum(t * (t ** 2 - 1) / 2)])
        if bidir:
            n = v.size
            r = np.minimum(r, n + 1 - r)
        R[:, j] = r
    R = R.reshape(x.shape)
    A = np.array(adj).T
    if not tieflag:
        A = A.reshape(1, -1)
    return outs(nargout, R, A if A.shape[1] > 1 or tieflag else A.reshape(1, 1))


@builtin("corrcov")
def _corrcov(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("corrcov: expected corrcov(C).")
    C = np.asarray(fnum(args[0]), dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T, atol=1e-10 * max(1.0,
            np.abs(C).max(initial=0))):
        raise MatlabError("corrcov: the input must be a square, symmetric covariance matrix.")
    sd = np.sqrt(np.diag(C))
    with np.errstate(all="ignore"):
        R = C / np.outer(sd, sd)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    return outs(nargout, R, sd.reshape(-1, 1))


@builtin("nancov")
def _nancov(interp, args, nargout):
    rest = list(args)
    mode = "complete"
    if rest and is_text(rest[-1]):
        mode = choice(rest.pop(), ["complete", "pairwise"], "The nancov option")
    if not 1 <= len(rest) <= 3:
        raise MatlabError("nancov: expected nancov(X), nancov(X, Y) or nancov(X, flag, 'pairwise').")
    flag = 0
    if len(rest) > 1 and np.asarray(fnum(rest[-1])).size == 1 and (len(rest) == 3
            or np.asarray(fnum(rest[0])).size > 1) \
            and float(np.asarray(fnum(rest[-1])).flat[0]) in (0.0, 1.0):
        flag = int(scalar(rest.pop()))
    if len(rest) == 2:
        X = np.column_stack([vec(rest[0]), vec(rest[1])])
    else:
        X = np.asarray(fnum(rest[0]), dtype=float)
        if X.shape[0] == 1:
            X = X.reshape(-1, 1)
    return _cov_missing(X, mode, flag)


def _cov_missing(X: np.ndarray, mode: str, flag: int = 0) -> np.ndarray:
    p = X.shape[1]
    if mode == "complete":
        Z = X[~np.isnan(X).any(axis=1)]
        n = Z.shape[0]
        if n == 0:
            return np.full((p, p), np.nan)
        Zc = Z - Z.mean(axis=0)
        return Zc.T @ Zc / (n if (flag == 1 or n == 1) else n - 1)
    C = np.full((p, p), np.nan)
    for i in range(p):
        for j in range(i, p):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            n = int(ok.sum())
            if n:
                a, b = X[ok, i], X[ok, j]
                C[i, j] = C[j, i] = np.sum((a - a.mean()) * (b - b.mean())) / (n if (flag == 1 or n == 1) else n - 1)
    return C


# -- correlation -------------------------------------------------------------------------------------------

def _pearson_cols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    with np.errstate(all="ignore"):
        den = np.sqrt(np.outer((Xc ** 2).sum(axis=0), (Yc ** 2).sum(axis=0)))
        return np.clip(Xc.T @ Yc / den, -1, 1)


def _rank_cols(X: np.ndarray) -> np.ndarray:
    return np.column_stack([tied_rank(X[:, j]) for j in range(X.shape[1])]) if X.size else X


def _t_pvalue(r: np.ndarray, df: float, tail: str) -> np.ndarray:
    import scipy.stats as st
    with np.errstate(all="ignore"):
        t = r * np.sqrt(df / (1 - r ** 2))
        t = np.where(np.abs(r) >= 1, np.sign(r) * np.inf, t)
        if tail == "both":
            p = 2 * st.t.sf(np.abs(t), df)
        elif tail == "right":
            p = st.t.sf(t, df)
        else:
            p = st.t.cdf(t, df)
    return np.where(np.isnan(r), np.nan, p)


def _spearman_exact(x: np.ndarray, y: np.ndarray, r: float, tail: str) -> float:
    """Exact permutation p-value of Spearman's rho for small samples without ties."""
    from itertools import permutations
    n = x.size
    rx = tied_rank(x)
    ry = tied_rank(y)
    d_obs = np.sum((rx - ry) ** 2)
    base = np.arange(1, n + 1, dtype=float)
    perms = np.array(list(permutations(base)))
    d = np.sum((perms - rx[None, :]) ** 2, axis=1)
    tol = 1e-9
    lo = np.mean(d >= d_obs - tol)   # rho <= observed
    hi = np.mean(d <= d_obs + tol)   # rho >= observed
    if tail == "right":
        return float(hi)
    if tail == "left":
        return float(lo)
    return float(min(1.0, 2 * min(lo, hi)))


def _kendall(x: np.ndarray, y: np.ndarray, tail: str) -> tuple[float, float]:
    import scipy.stats as st
    alt = {"both": "two-sided", "right": "greater", "left": "less"}[tail]
    with np.errstate(all="ignore"):
        res = st.kendalltau(x, y, alternative=alt)
    return float(res.statistic), float(res.pvalue)


def corr_matrix(X: np.ndarray, Y: np.ndarray | None, kind: str, rows: str, tail: str, want_p: bool):
    """Correlation between the columns of X and of Y (X itself when Y is None)."""
    same = Y is None
    Y = X if same else Y
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("corr: X and Y must have the same number of rows.")
    if rows == "complete":
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X, Y = X[ok], Y[ok]
    px, py = X.shape[1], Y.shape[1]
    R = np.full((px, py), np.nan)
    P = np.full((px, py), np.nan)

    def pair(a, b):
        n = a.size
        if n < 2:
            return np.nan, np.nan
        if kind == "kendall":
            return _kendall(a, b, tail)
        if kind == "spearman":
            ra, rb = tied_rank(a), tied_rank(b)
        else:
            ra, rb = a, b
        r = float(_pearson_cols(ra.reshape(-1, 1), rb.reshape(-1, 1))[0, 0])
        if not want_p:
            return r, np.nan
        if kind == "spearman" and n < 10 and np.unique(a).size == n and np.unique(b).size == n:
            return r, _spearman_exact(a, b, r, tail)
        return r, float(_t_pvalue(np.array(r), n - 2, tail))

    fast = kind != "kendall" and rows != "pairwise" and not (np.isnan(X).any() or np.isnan(Y).any())
    if fast and not (want_p and kind == "spearman" and X.shape[0] < 10):
        Xr, Yr = (_rank_cols(X), _rank_cols(Y)) if kind == "spearman" else (X, Y)
        R = _pearson_cols(Xr, Yr) if X.shape[0] > 1 else R
        if want_p:
            P = _t_pvalue(R, X.shape[0] - 2, tail)
    else:
        for i in range(px):
            for j in range(py):
                a, b = X[:, i], Y[:, j]
                if rows == "pairwise":
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    a, b = a[ok], b[ok]
                elif np.isnan(a).any() or np.isnan(b).any():
                    continue
                R[i, j], P[i, j] = pair(a, b)
    if same:
        d = np.diag(R).copy()
        np.fill_diagonal(R, np.where(np.isnan(d), np.nan, 1.0))
        if want_p:
            np.fill_diagonal(P, np.where(np.isnan(d), np.nan, 1.0 if tail != "right" else 0.0))
            if tail == "both":
                np.fill_diagonal(P, np.where(np.isnan(d), np.nan, 0.0))
    return R, P


_TAILS = {"both": "both", "ne": "both", "right": "right", "gt": "right", "left": "left", "lt": "left"}


def tail_opt(opts: dict, default: str = "both") -> str:
    if "tail" not in opts:
        return default
    if not is_text(opts["tail"]) or text(opts["tail"]).lower() not in _TAILS:
        raise MatlabError("Tail must be 'both', 'right' or 'left'.")
    return _TAILS[text(opts["tail"]).lower()]


@builtin("corr")
def _corr(interp, args, nargout):
    pos, opts = get_opts(args, 1, ["Type", "Rows", "Tail"], "corr")
    if not 1 <= len(pos) <= 2:
        raise MatlabError("corr: expected corr(X) or corr(X, Y, 'Name', value, ...).")
    kind = choice(opts["type"], ["Pearson", "Kendall", "Spearman"], "Type") if "type" in opts else "pearson"
    rows = choice(opts["rows"], ["all", "complete", "pairwise"], "Rows") if "rows" in opts else "all"
    X = np.asarray(fnum(pos[0]), dtype=float)
    Y = np.asarray(fnum(pos[1]), dtype=float) if len(pos) > 1 else None
    if X.ndim != 2 or (Y is not None and Y.ndim != 2):
        raise MatlabError("corr: the inputs must be matrices with one observation per row.")
    R, P = corr_matrix(X, Y, kind, rows, tail_opt(opts), nargout > 1)
    return outs(nargout, R, P)


def _residualise(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    D = np.column_stack([np.ones(Z.shape[0]), Z])
    coef, *_ = np.linalg.lstsq(D, A, rcond=None)
    return A - D @ coef


@builtin("partialcorr")
def _partialcorr(interp, args, nargout):
    pos, opts = get_opts(args, 1, ["Type", "Rows", "Tail"], "partialcorr")
    if not 1 <= len(pos) <= 3:
        raise MatlabError("partialcorr: expected partialcorr(X), partialcorr(X, Z) or partialcorr(X, Y, Z).")
    kind = choice(opts["type"], ["Pearson", "Spearman"], "Type") if "type" in opts else "pearson"
    rows = choice(opts["rows"], ["all", "complete", "pairwise"], "Rows") if "rows" in opts else "all"
    if rows == "pairwise":
        raise MatlabError("partialcorr: 'Rows', 'pairwise' is not supported; use 'complete'.")
    tail = tail_opt(opts)
    mats = [np.asarray(fnum(a), dtype=float) for a in pos]
    n = mats[0].shape[0]
    if any(m.shape[0] != n for m in mats):
        raise MatlabError("partialcorr: all inputs must have the same number of rows.")
    if rows == "complete":
        ok = ~np.any(np.column_stack([np.isnan(m).any(axis=1) for m in mats]), axis=1)
        mats = [m[ok] for m in mats]
        n = mats[0].shape[0]
    if kind == "spearman":
        mats = [_rank_cols(m) for m in mats]
    if len(mats) == 1:
        X = mats[0]
        p = X.shape[1]
        R = np.eye(p)
        dz = p - 2
        for i in range(p):
            for j in range(i + 1, p):
                others = [k for k in range(p) if k not in (i, j)]
                E = _residualise(X[:, [i, j]], X[:, others])
                R[i, j] = R[j, i] = _pearson_cols(E[:, :1], E[:, 1:])[0, 0]
        same = True
    else:
        X = mats[0]
        Y = mats[1] if len(mats) == 3 else None
        Z = mats[-1]
        dz = np.linalg.matrix_rank(Z - Z.mean(axis=0)) if Z.size else 0
        Ex = _residualise(X, Z)
        Ey = _residualise(Y, Z) if Y is not None else Ex
        R = _pearson_cols(Ex, Ey)
        same = Y is None
        if same:
            np.fill_diagonal(R, 1.0)
    P = _t_pvalue(R, n - dz - 2, tail)
    if same:
        np.fill_diagonal(P, 0.0 if tail == "both" else (0.0 if tail == "right" else 1.0))
    return outs(nargout, R, P)


# -- grouped summaries ---------------------------------------------------------------------------------------

_GRP_STATS = ["mean", "sem", "numel", "gname", "std", "var", "min", "max", "range", "meanci", "predci", "median",
              "mode"]


def _stat_of(interp, spec: Any, D: np.ndarray, alpha: float, gname: list) -> Any:
    """One statistic of a group's data matrix (rows = observations). Returns an array (p,) or (p, 2)."""
    import warnings

    import scipy.stats as st
    if isinstance(spec, FunctionHandle):
        r = np.asarray(fnum(call(interp, spec, [D], 1)[0]), dtype=float)
        if r.size not in (1, D.shape[1]) and r.shape[0] != 1:
            raise MatlabError("grpstats: the function must return one value per column (a row vector).")
        return r.flatten() if r.size == D.shape[1] else np.repeat(r.flatten(), D.shape[1] // max(r.size, 1) or 1)
    n = np.sum(~np.isnan(D), axis=0).astype(float)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mean = np.nanmean(D, axis=0) if D.shape[0] else np.full(D.shape[1], np.nan)
        sd = np.where(n > 1, np.nanstd(D, axis=0, ddof=1) if D.shape[0] > 1 else np.nan, np.where(n == 1, 0.0, np.nan))
        if spec == "mean":
            return mean
        if spec == "numel":
            return n
        if spec == "std":
            return sd
        if spec == "var":
            return sd ** 2
        if spec == "sem":
            return sd / np.sqrt(n)
        if spec == "min":
            return np.nanmin(D, axis=0) if D.shape[0] else np.full(D.shape[1], np.nan)
        if spec == "max":
            return np.nanmax(D, axis=0) if D.shape[0] else np.full(D.shape[1], np.nan)
        if spec == "range":
            return np.nanmax(D, axis=0) - np.nanmin(D, axis=0) if D.shape[0] else np.full(D.shape[1], np.nan)
        if spec == "median":
            return np.nanmedian(D, axis=0) if D.shape[0] else np.full(D.shape[1], np.nan)
        if spec == "mode":
            out = np.full(D.shape[1], np.nan)
            for j in range(D.shape[1]):
                v = D[:, j][~np.isnan(D[:, j])]
                if v.size:
                    u, c = np.unique(v, return_counts=True)
                    out[j] = u[np.argmax(c)]
            return out
        if spec in ("meanci", "predci"):
            t = st.t.ppf(1 - alpha / 2, n - 1)
            half = t * sd / np.sqrt(n) if spec == "meanci" else t * sd * np.sqrt(1 + 1 / n)
            return np.column_stack([mean - half, mean + half])
    raise MatlabError(f"grpstats: unknown statistic '{spec}'.")


def _stat_specs(which: Any) -> list:
    if which is None or is_empty_arg(which):
        return ["mean"]
    items = which.values_f() if isinstance(which, Cell) else [which]
    out = []
    for it in items:
        if isinstance(it, FunctionHandle):
            if not it.is_anonymous and str(it.name) in _GRP_STATS:
                out.append(str(it.name))
            else:
                out.append(it)
        elif is_text(it):
            s = text(it).lower()
            if s not in _GRP_STATS:
                raise MatlabError(f"grpstats: unknown statistic '{text(it)}'.")
            out.append(s)
        else:
            raise MatlabError("grpstats: statistics are given as names or function handles.")
    return out


def _spec_name(spec: Any, k: int) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec.name) if not spec.is_anonymous else f"Fun{k + 1}"


@builtin("grpstats")
def _grpstats(interp, args, nargout):
    if not args:
        raise MatlabError("grpstats: expected grpstats(X, group, whichstats).")
    if is_table(args[0]):
        return _grpstats_table(interp, args, nargout)
    pos, opts = get_opts(args, 1, ["Alpha"], "grpstats")
    if len(pos) > 3:
        raise MatlabError("grpstats: too many input arguments.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    X = np.asarray(fnum(pos[0]), dtype=float)
    if X.ndim != 2:
        raise MatlabError("grpstats: X must be a vector or a matrix.")
    if X.shape[0] == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if len(pos) > 1 and not is_empty_arg(pos[1]):
        codes, names, per_var, _ = multi_group(group_list(pos[1]))
        if codes.size != n:
            raise MatlabError("grpstats: the grouping variable must have one value per row of X.")
    else:
        codes, names, per_var = np.zeros(n, dtype=int), ["1"], [["1"]]
    specs = _stat_specs(pos[2] if len(pos) > 2 else None)
    G = len(names)
    results = []
    for spec in specs:
        if spec == "gname":
            data = np.empty((G, len(per_var[0]) if per_var else 1), dtype=object)
            for g in range(G):
                for j, s in enumerate(per_var[g]):
                    data[g, j] = char(s)
            results.append(Cell(data))
            continue
        vals = [_stat_of(interp, spec, X[codes == g], alpha, per_var[g]) for g in range(G)]
        if vals and np.asarray(vals[0]).ndim == 2:
            arr = np.stack(vals, axis=0)                      # (G, p, 2)
            results.append(arr.reshape(G, 2) if arr.shape[1] == 1 else arr)
        else:
            results.append(np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in vals]) if vals
                           else np.zeros((0, X.shape[1])))
    if nargout > len(results):
        raise MatlabError("grpstats: more outputs requested than statistics.")
    return results[0] if nargout <= 1 else tuple(results[:nargout])


def _grpstats_table(interp, args, nargout):
    tbl = args[0]
    pos, opts = get_opts(args, 1, ["Alpha", "DataVars", "VarNames"], "grpstats")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    if len(pos) > 3:
        raise MatlabError("grpstats: too many input arguments.")

    def var_names(v) -> list[str]:
        items = text_list(v)
        if items is not None:
            for s in items:
                if s not in tbl.names:
                    raise MatlabError(f"grpstats: '{s}' is not a variable of the table.")
            return items
        return [tbl.names[int(i) - 1] for i in vec(v)]

    gvars = var_names(pos[1]) if len(pos) > 1 and not is_empty_arg(pos[1]) else []
    specs = _stat_specs(pos[2] if len(pos) > 2 else None)
    if "gname" in specs:
        raise MatlabError("grpstats: 'gname' is not available for table input (group names are the row names).")
    dvars = var_names(opts["datavars"]) if "datavars" in opts else [n for n in tbl.names if n not in gvars]
    cols = dict(zip(tbl.names, tbl.columns))
    n = tbl.nrows
    if gvars:
        codes, names, _, _ = multi_group([cols[g] for g in gvars])
    else:
        codes, names = np.zeros(n, dtype=int), ["All"]
    G = len(names)
    out_names: list[str] = []
    out_cols: list = []
    for g in gvars:
        first = [int(np.flatnonzero(codes == k)[0]) for k in range(G)]
        from .table import _rows
        out_names.append(g)
        out_cols.append(_rows(cols[g], np.array(first, dtype=int)))
    out_names.append("GroupCount")
    out_cols.append(np.array([float(np.sum(codes == k)) for k in range(G)]).reshape(-1, 1))
    for v in dvars:
        c = cols[v]
        if not isinstance(c, np.ndarray) or c.dtype.kind not in "biuf":
            raise MatlabError(f"grpstats: the data variable '{v}' is not numeric. Choose the variables "
                              "with 'DataVars'.")
        D = np.asarray(c, dtype=float).reshape(n, -1)
        for k, spec in enumerate(specs):
            vals = [_stat_of(interp, spec, D[codes == g], alpha, []) for g in range(G)]
            arr = np.vstack([np.asarray(x, dtype=float).reshape(1, -1) for x in vals])
            out_names.append(f"{_spec_name(spec, k)}_{v}")
            out_cols.append(arr)
    if "varnames" in opts:
        new = text_list(opts["varnames"])
        if new is None or len(new) != len(out_names):
            raise MatlabError(f"grpstats: VarNames must list {len(out_names)} names.")
        out_names = new
    return make_table(out_names, out_cols, row_names=names if gvars else ["All"])


@builtin("crosstab")
def _crosstab(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("crosstab: expected crosstab(x1, x2, ...).")
    parts = [group_codes(a) for a in args]
    n = parts[0][0].size
    if any(p[0].size != n for p in parts):
        raise MatlabError("crosstab: all grouping variables must have the same number of elements.")
    ok = np.all(np.column_stack([p[0] >= 0 for p in parts]), axis=1)
    sizes = [max(len(p[1]), 1) for p in parts]
    shape = tuple(sizes) if len(sizes) > 1 else (sizes[0], 1)
    tbl = np.zeros(shape)
    idx = tuple(p[0][ok] for p in parts) if len(sizes) > 1 else (parts[0][0][ok], np.zeros(int(ok.sum()), dtype=int))
    np.add.at(tbl, idx, 1.0)
    if nargout <= 1:
        return tbl
    total = tbl.sum()
    expected = np.full(tbl.shape, total)
    for ax in range(tbl.ndim):
        marg = tbl.sum(axis=tuple(a for a in range(tbl.ndim) if a != ax))
        sh = [1] * tbl.ndim
        sh[ax] = -1
        expected = expected * (marg / total).reshape(sh)
    with np.errstate(all="ignore"):
        chi2 = float(np.nansum((tbl - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
    df = int(np.prod(sizes)) - 1 - sum(s - 1 for s in sizes)
    p = float(st.chi2.sf(chi2, df)) if df > 0 else np.nan
    labels = np.empty((max(sizes), len(parts)), dtype=object)
    for j, part in enumerate(parts):
        for i in range(max(sizes)):
            labels[i, j] = char(part[1][i]) if i < len(part[1]) else np.zeros((0, 0))
    return outs(nargout, tbl, chi2, p, Cell(labels))


@builtin("dummyvar")
def _dummyvar(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("dummyvar: expected dummyvar(group).")
    g = args[0]
    if is_categorical(g):
        codes = g.codes.flatten(order="F")
        if np.any(codes < 0):
            raise MatlabError("dummyvar: the grouping variable must not contain undefined values.")
        D = np.zeros((codes.size, len(g.categories)))
        D[np.arange(codes.size), codes] = 1
        return D
    gs = list(g.values_f()) if isinstance(g, Cell) else None
    G = np.column_stack([vec(c) for c in gs]) if gs is not None else np.asarray(fnum(g), dtype=float)
    if G.ndim == 2 and G.shape[0] == 1:
        G = G.reshape(-1, 1)
    if np.any(G < 1) or np.any(G != np.round(G)):
        raise MatlabError("dummyvar: the grouping variable must hold positive integers.")
    blocks = []
    for j in range(G.shape[1]):
        k = int(G[:, j].max())
        D = np.zeros((G.shape[0], k))
        D[np.arange(G.shape[0]), G[:, j].astype(int) - 1] = 1
        blocks.append(D)
    return np.hstack(blocks) if blocks else np.zeros((0, 0))


@builtin("gscatter")
def _gscatter(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("gscatter: expected gscatter(x, y, g).")
    if len(args) > 9:
        raise MatlabError("gscatter: too many input arguments.")
    x, y = vec(args[0]), vec(args[1])
    codes, names, _, _ = multi_group(group_list(args[2]), sep=",")
    if x.size != y.size or codes.size != x.size:
        raise MatlabError("gscatter: x, y and g must have the same number of elements.")
    colors = text(args[3]) if given(args, 3) and is_text(args[3]) else "bgrcmyk"
    if given(args, 3) and not is_text(args[3]):
        raise MatlabError("gscatter: colors must be given as letters, e.g. 'rgb'.")
    syms = text(args[4]) if given(args, 4) else "."
    sizes = vec(args[5]) if given(args, 5) else np.array([])
    doleg = as_bool(args[6], "doleg") if given(args, 6) else True
    hs = []
    for g in range(len(names)):
        sel = codes == g
        spec = colors[g % len(colors)] + syms[g % len(syms)]
        extra = [char("MarkerSize"), np.array([[float(sizes[g % sizes.size])]])] if sizes.size else \
            ([char("MarkerSize"), np.array([[15.0]])] if syms[g % len(syms)] == "." else [])
        h = builtin_call(interp, "plot", [x[sel].reshape(1, -1), y[sel].reshape(1, -1), char(spec)] + extra, 1)
        hs.append(h[0])
        builtin_call(interp, "hold", [char("on")], 0)
    builtin_call(interp, "hold", [char("off")], 0)
    if doleg and names:
        builtin_call(interp, "legend", [cellstr(names, column=False)], 0)
    if given(args, 7):
        builtin_call(interp, "xlabel", [args[7]], 0)
    if given(args, 8):
        builtin_call(interp, "ylabel", [args[8]], 0)
    if not nargout:
        return None
    from .plotting import GraphicsArray
    return hs[0] if len(hs) == 1 else GraphicsArray(hs)


# -- resampling ------------------------------------------------------------------------------------------------

def _take(d: Any, idx: np.ndarray) -> Any:
    """Rows ``idx`` of a data argument (elements of a vector); scalars and non-matching data pass through."""
    if isinstance(d, np.ndarray):
        if d.ndim == 2 and d.shape[0] == 1 and d.shape[1] > 1:
            return d[:, idx]
        return d[idx]
    if isinstance(d, Cell):
        return Cell(d.data[:, idx]) if d.data.shape[0] == 1 else Cell(d.data[idx])
    if hasattr(d, "index_get"):
        from .. import indexing as ix
        return d.index_get(None, [(idx + 1.0).reshape(-1, 1), ix.COLON])
    return d


def _nobs(data: list, fname: str) -> int:
    n = None
    for d in data:
        if isinstance(d, np.ndarray):
            k = d.size if (d.ndim == 2 and min(d.shape) == 1) else d.shape[0]
        elif isinstance(d, Cell):
            k = d.data.size if min(d.data.shape) == 1 else d.data.shape[0]
        elif hasattr(d, "matlab_shape"):
            k = d.matlab_shape()[0]
        else:
            continue
        if k == 1:
            continue  # scalars are passed to the function unchanged
        if n is None:
            n = k
        elif k != n:
            raise MatlabError(f"{fname}: the data arguments must have the same number of rows.")
    if n is None:
        raise MatlabError(f"{fname}: at least one data argument must be non-scalar.")
    return n


def _apply(interp, fun, data: list, idx: np.ndarray | None, n: int) -> np.ndarray:
    args = []
    for d in data:
        scalar_like = isinstance(d, np.ndarray) and d.size == 1
        args.append(d if (idx is None or scalar_like) else _take(d, idx))
    r = call(interp, fun, args, 1)[0]
    return np.asarray(fnum(r), dtype=float).flatten(order="F")


def _boot_stats(interp, nboot: int, fun, data: list, weights, rng) -> tuple[np.ndarray, np.ndarray]:
    n = _nobs(data, "bootstrp")
    p = None
    if weights is not None:
        p = vec(weights)
        if p.size != n or np.any(p < 0) or p.sum() <= 0:
            raise MatlabError("bootstrp: Weights must have one non-negative value per observation.")
        p = p / p.sum()
    sam = np.empty((n, nboot))
    rows = []
    for b in range(nboot):
        idx = rng.choice(n, size=n, replace=True, p=p)
        sam[:, b] = idx + 1.0
        rows.append(_apply(interp, fun, data, idx, n))
    stat = np.vstack(rows) if rows else np.zeros((0, 0))
    return stat, sam


@builtin("bootstrp")
def _bootstrp(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("bootstrp: expected bootstrp(nboot, bootfun, d1, ...).")
    pos, opts = get_opts(args, 3, ["Weights", "Options"], "bootstrp")
    nboot = intval(pos[0], "nboot")
    if nboot < 0:
        raise MatlabError("bootstrp: nboot must not be negative.")
    stat, sam = _boot_stats(interp, nboot, pos[1], pos[2:], opts.get("weights"), rng_of(interp))
    return outs(nargout, stat, sam)


@builtin("jackknife")
def _jackknife(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("jackknife: expected jackknife(jackfun, X, ...).")
    pos, _ = get_opts(args, 2, ["Options"], "jackknife")
    data = pos[1:]
    n = _nobs(data, "jackknife")
    rows = [_apply(interp, pos[0], data, np.delete(np.arange(n), i), n) for i in range(n)]
    return np.vstack(rows)


@builtin("bootci")
def _bootci(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("bootci: expected bootci(nboot, bootfun, d1, ...) or bootci(nboot, {bootfun, d1, ...}, ...).")
    nboot = intval(args[0], "nboot")
    known = ["Alpha", "Type", "NBootStd", "StdErr", "Weights", "Options"]
    if isinstance(args[1], Cell):
        items = args[1].values_f()
        fun, data = items[0], items[1:]
        _, opts = get_opts(args, 2, known, "bootci")
    else:
        pos, opts = get_opts(args, 2, known, "bootci")
        fun, data = pos[1], pos[2:]
    if not data:
        raise MatlabError("bootci: no data were given.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    kind = choice(opts["type"], ["norm", "normal", "per", "percentile", "cper", "corrected percentile", "bca",
                                 "stud", "student"], "Type") if "type" in opts else "bca"
    kind = {"normal": "norm", "percentile": "per", "corrected percentile": "cper", "student": "stud"}.get(kind, kind)
    rng = rng_of(interp)
    n = _nobs(data, "bootci")
    stat = _apply(interp, fun, data, None, n)
    w = opts.get("weights")
    if kind == "stud":
        nstd = intval(opts["nbootstd"], "NBootStd") if "nbootstd" in opts else 100
        boot = np.empty((nboot, stat.size))
        tvals = np.empty((nboot, stat.size))
        for b in range(nboot):
            idx = rng.choice(n, size=n, replace=True)
            sub = [d if (isinstance(d, np.ndarray) and d.size == 1) else _take(d, idx) for d in data]
            boot[b] = _apply(interp, fun, sub, None, n)
            if "stderr" in opts:
                se_b = np.asarray(fnum(call(interp, opts["stderr"], sub, 1)[0]), dtype=float).flatten()
            else:
                inner, _ = _boot_stats(interp, nstd, fun, sub, None, rng)
                se_b = inner.std(axis=0, ddof=1)
            with np.errstate(all="ignore"):
                tvals[b] = (boot[b] - stat) / se_b
        se = boot.std(axis=0, ddof=1)
        lo = stat - se * np.nanpercentile(tvals, 100 * (1 - alpha / 2), axis=0, method="hazen")
        hi = stat - se * np.nanpercentile(tvals, 100 * alpha / 2, axis=0, method="hazen")
    else:
        if "nbootstd" in opts or "stderr" in opts:
            raise MatlabError("bootci: NBootStd and StdErr apply to 'Type', 'stud' only.")
        boot, _ = _boot_stats(interp, nboot, fun, data, w, rng)
        za = st.norm.ppf(alpha / 2)
        if kind == "per":
            lo = np.percentile(boot, 100 * alpha / 2, axis=0, method="hazen")
            hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0, method="hazen")
        elif kind == "norm":
            se = boot.std(axis=0, ddof=1)
            bias = boot.mean(axis=0) - stat
            lo, hi = stat - bias + se * za, stat - bias - se * za
        else:
            with np.errstate(all="ignore"):
                z0 = st.norm.ppf(np.mean(boot < stat, axis=0) + np.mean(boot == stat, axis=0) / 2)
                if kind == "cper":
                    p1, p2 = st.norm.cdf(2 * z0 + za), st.norm.cdf(2 * z0 - za)
                else:
                    jack = np.vstack([_apply(interp, fun, data, np.delete(np.arange(n), i), n) for i in range(n)])
                    score = jack.mean(axis=0) - jack
                    acc = np.sum(score ** 3, axis=0) / (6 * np.sum(score ** 2, axis=0) ** 1.5)
                    acc = np.where(np.isfinite(acc), acc, 0.0)
                    p1 = st.norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
                    p2 = st.norm.cdf(z0 + (z0 - za) / (1 - acc * (z0 - za)))
            lo = np.array([np.percentile(boot[:, j], 100 * p1[j], method="hazen") if np.isfinite(p1[j]) else np.nan
                           for j in range(boot.shape[1])])
            hi = np.array([np.percentile(boot[:, j], 100 * p2[j], method="hazen") if np.isfinite(p2[j]) else np.nan
                           for j in range(boot.shape[1])])
    ci = np.vstack([lo, hi])
    return outs(nargout, ci, boot)


# -- cvpartition -------------------------------------------------------------------------------------------------

class CVPartition(StatObject):
    matlab_class = "cvpartition"

    def __init__(self, n: int, kind: str, test_sets: list[np.ndarray], stratified: bool, spec: tuple):
        super().__init__()
        self.n = n
        self.kind = kind
        self.tests = test_sets           # boolean masks of the test observations
        self.stratified = stratified
        self.spec = spec                 # (param, group codes or None) to redo the partition

    def property_names(self) -> list[str]:
        return ["NumObservations", "NumTestSets", "TrainSize", "TestSize", "Type", "IsCustom", "IsGrouped",
                "IsStratified"]

    def get_field(self, interp, name: str) -> Any:
        k = len(self.tests)
        sizes = np.array([float(t.sum()) for t in self.tests])
        if name == "NumObservations":
            return np.array([[float(self.n)]])
        if name == "NumTestSets":
            return np.array([[float(k)]])
        if name == "TestSize":
            return sizes.reshape(1, -1)
        if name == "TrainSize":
            return (self.n - sizes).reshape(1, -1) if self.kind != "resubstitution" else sizes.reshape(1, -1)
        if name == "Type":
            return char(self.kind)
        if name in ("IsCustom", "IsGrouped"):
            return np.array([[False]])
        if name == "IsStratified":
            return np.array([[self.stratified]])
        return super().get_field(interp, name)

    def _index(self, args: list, name: str):
        if len(args) > 1:
            raise MatlabError(f"Too many input arguments for {name}.")
        if not args:
            if len(self.tests) != 1:
                return 0
            return 0
        if is_text(args[0]):
            if text(args[0]).lower() != "all":
                raise MatlabError(f"{name}: the index must be a number or 'all'.")
            return "all"
        i = intval(args[0], "the index")
        if not 1 <= i <= len(self.tests):
            raise MatlabError(f"{name}: the index must be between 1 and {len(self.tests)}.")
        return i - 1

    def m_test(self, interp, args, nargout):
        i = self._index(args, "test")
        if i == "all":
            return np.column_stack(self.tests)
        return self.tests[i].reshape(-1, 1).copy()

    def m_training(self, interp, args, nargout):
        i = self._index(args, "training")
        flip = (lambda t: t.copy()) if self.kind == "resubstitution" else (lambda t: ~t)
        if i == "all":
            return np.column_stack([flip(t) for t in self.tests])
        return flip(self.tests[i]).reshape(-1, 1)

    def m_repartition(self, interp, args, nargout):
        if args:
            raise MatlabError("repartition: a random stream argument is not supported.")
        param, codes = self.spec
        return make_partition(rng_of(interp), self.n, self.kind, param, codes if self.stratified else None)

    def display_lines(self) -> list[str]:
        sizes = [int(t.sum()) for t in self.tests]
        label = {"kfold": "K-fold cross validation partition", "holdout": "Hold-out cross validation partition",
                 "leaveout": "Leave-one-out cross validation partition",
                 "resubstitution": "Resubstitution (no partition of the data)"}[self.kind]
        show = (lambda v: "  ".join(str(x) for x in v[:10]) + ("  ..." if len(v) > 10 else ""))
        train = [self.n - s for s in sizes] if self.kind != "resubstitution" else sizes
        return [label, f"   NumObservations: {self.n}", f"       NumTestSets: {len(sizes)}",
                f"         TrainSize: {show(train)}", f"          TestSize: {show(sizes)}",
                "          IsCustom: 0"]

    def train_test(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.kind == "resubstitution":
            return [(np.flatnonzero(t), np.flatnonzero(t)) for t in self.tests]
        return [(np.flatnonzero(~t), np.flatnonzero(t)) for t in self.tests]


def _fold_ids(rng, n: int, k: int) -> np.ndarray:
    """Fold number 0..k-1 of n shuffled observations, fold sizes differing by at most one."""
    ids = np.arange(n) % k
    return ids[rng.permutation(n)]


def make_partition(rng, n: int, kind: str, param: float, codes: np.ndarray | None) -> CVPartition:
    if kind == "kfold":
        k = int(param)
        if k < 2 or k > n:
            raise MatlabError("cvpartition: the number of folds must be between 2 and the number of observations.")
        fold = np.empty(n, dtype=int)
        if codes is not None:
            # Deal each class out over the folds, continuing where the previous class stopped.
            offset = 0
            for c in np.unique(codes[codes >= 0]):
                members = np.flatnonzero(codes == c)
                members = members[rng.permutation(members.size)]
                fold[members] = (offset + np.arange(members.size)) % k
                offset = (offset + members.size) % k
            fold[codes < 0] = -1
        else:
            fold = _fold_ids(rng, n, k)
        tests = [fold == i for i in range(k)]
    elif kind == "holdout":
        p = float(param)

        def ntest(m):
            if p < 1:
                return int(np.floor(m * p))
            return int(round(p * m / n)) if codes is not None else int(p)
        if not (0 < p < 1 or (p >= 1 and p == int(p) and p < n)):
            raise MatlabError("cvpartition: HoldOut must be a fraction in (0, 1) or an integer below the number of "
                              "observations.")
        mask = np.zeros(n, dtype=bool)
        if codes is not None:
            for c in np.unique(codes[codes >= 0]):
                members = np.flatnonzero(codes == c)
                members = members[rng.permutation(members.size)]
                mask[members[:ntest(members.size)]] = True
            total = int(np.floor(n * p)) if p < 1 else int(p)
            extra = total - int(mask.sum())
            if extra > 0:
                rest = np.flatnonzero(~mask & (codes >= 0))
                mask[rng.permutation(rest)[:extra]] = True
        else:
            mask[rng.permutation(n)[:ntest(n)]] = True
        tests = [mask]
    elif kind == "leaveout":
        tests = [np.arange(n) == i for i in range(n)]
    else:
        tests = [np.ones(n, dtype=bool)]
    return CVPartition(n, kind, tests, codes is not None, (param, codes))


@builtin("cvpartition")
def _cvpartition(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("cvpartition: expected cvpartition(n, 'KFold', k), cvpartition(n, 'HoldOut', p), "
                          "cvpartition(n, 'LeaveOut') or cvpartition(group, ...).")
    first = args[0]
    rest = list(args[1:])
    if not is_text(rest[0]):
        raise MatlabError("cvpartition: the second input must be the partition type.")
    kind = choice(rest[0], ["KFold", "HoldOut", "LeaveOut", "Resubstitution"], "The partition type")
    rest = rest[1:]
    param = {"kfold": 10.0, "holdout": 0.1}.get(kind, 0.0)
    if rest and not is_text(rest[0]) and kind in ("kfold", "holdout"):
        param = scalar(rest.pop(0), "the partition parameter")
    _, opts = get_opts(rest, 0, ["Stratify", "GroupingVariables"], "cvpartition")
    if "groupingvariables" in opts:
        raise MatlabError("cvpartition: the option 'GroupingVariables' is not supported.")
    grouped = not (isinstance(first, np.ndarray) and first.size == 1 and first.dtype.kind in "iuf")
    if grouped:
        codes, _, _ = group_codes(first)
        n = codes.size
        stratify = as_bool(opts["stratify"], "Stratify") if "stratify" in opts else True
        if np.any(codes < 0):
            keep = codes >= 0  # observations with a missing group are in no training or test set
            sub = make_partition(rng_of(interp), int(keep.sum()), kind, param, codes[keep] if stratify else None)
            tests = []
            for t in sub.tests:
                full = np.zeros(n, dtype=bool)
                full[keep] = t
                tests.append(full)
            return CVPartition(n, kind, tests, stratify, (param, codes))
        return make_partition(rng_of(interp), n, kind, param, codes if stratify and kind in ("kfold", "holdout")
                else None)
    if "stratify" in opts and as_bool(opts["stratify"], "Stratify"):
        raise MatlabError("cvpartition: stratification needs a grouping variable as the first input.")
    n = intval(first, "n")
    if n < 1:
        raise MatlabError("cvpartition: n must be a positive integer.")
    return make_partition(rng_of(interp), n, kind, param, None)


def partition_from_opts(interp, opts: dict, n: int, codes: np.ndarray | None, fname: str,
                        default_k: int = 10) -> CVPartition | None:
    """A partition from the usual options CVPartition/KFold/Holdout/Leaveout (None when none is given)."""
    given_opts = [k for k in ("cvpartition", "partition", "kfold", "holdout", "leaveout") if k in opts]
    if len(given_opts) > 1:
        raise MatlabError(f"{fname}: give only one of CVPartition, KFold, Holdout and Leaveout.")
    if not given_opts:
        return None
    key = given_opts[0]
    if key in ("cvpartition", "partition"):
        c = opts[key]
        if not isinstance(c, CVPartition):
            raise MatlabError(f"{fname}: the partition must be a cvpartition object.")
        if c.n != n:
            raise MatlabError(f"{fname}: the partition was made for {c.n} observations but the data have {n}.")
        return c
    rng = rng_of(interp)
    if key == "kfold":
        return make_partition(rng, n, "kfold", intval(opts[key], "KFold"), codes)
    if key == "holdout":
        return make_partition(rng, n, "holdout", scalar(opts[key], "Holdout"), codes)
    v = opts[key]
    on = (text(v).lower() == "on") if is_text(v) else bool(scalar(v, "Leaveout"))
    return make_partition(rng, n, "leaveout", 0.0, None) if on else None


@builtin("crossval")
def _crossval(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("crossval: expected crossval(fun, X, ...) or crossval('mse', X, y, 'Predfun', predfun).")
    known = ["KFold", "Holdout", "Leaveout", "Partition", "Stratify", "MCReps", "Predfun", "Options"]
    pos, opts = get_opts(args, 2, known, "crossval")
    first, data = pos[0], pos[1:]
    criterion = None
    if is_text(first) and text(first).lower() in ("mse", "mcr"):
        criterion = text(first).lower()
        if "predfun" not in opts or len(data) != 2:
            raise MatlabError(f"crossval: the '{criterion}' form is crossval('{criterion}', X, y, 'Predfun', predfun).")
    elif "predfun" in opts:
        raise MatlabError("crossval: 'Predfun' is used with the 'mse' and 'mcr' criteria only.")
    n = _nobs(data, "crossval")
    codes = None
    if "stratify" in opts:
        codes = group_codes(opts["stratify"])[0]
        if codes.size != n:
            raise MatlabError("crossval: the Stratify variable must have one value per observation.")
    reps = intval(opts["mcreps"], "MCReps") if "mcreps" in opts else 1
    rows = []
    total_loss, total_n = 0.0, 0
    for _ in range(reps):
        part = partition_from_opts(interp, opts, n, codes, "crossval")
        if part is None:
            part = make_partition(rng_of(interp), n, "kfold", 10.0 if n >= 10 else float(n), codes)
        for train, test in part.train_test():
            if criterion is not None:
                X, y = data
                yfit = call(interp, opts["predfun"], [_take(X, train), _take(y, train), _take(X, test)], 1)[0]
                ytest = _take(y, test)
                if criterion == "mse":
                    d = vec(yfit) - vec(ytest)
                    total_loss += float(np.sum(d ** 2))
                else:
                    from ._stats_util import Labels
                    lab = Labels(ytest)
                    total_loss += float(np.sum(lab.encode(yfit) != lab.codes))
                total_n += test.size
            else:
                targs = [_take(d, train) for d in data] + [_take(d, test) for d in data]
                r = call(interp, first, targs, 1)[0]
                rows.append(np.asarray(fnum(r), dtype=float).flatten(order="F"))
    if criterion is not None:
        return total_loss / max(total_n, 1)
    return np.vstack(rows)


@builtin("tabulate")
def _tabulate(interp, args, nargout):
    if len(args) != 1:
        raise MatlabError("tabulate: expected tabulate(x).")
    v = args[0]
    numeric = isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.dtype != bool
    if numeric:
        x = vec(v)
        x = x[~np.isnan(x)]
        if x.size and np.all(x == np.round(x)) and np.all(x > 0):
            vals = np.arange(1, int(x.max()) + 1, dtype=float)
        else:
            vals = np.unique(x)
        counts = np.array([np.sum(x == u) for u in vals], dtype=float)
        names = [num_text(u) for u in vals]
    else:
        if is_categorical(v):
            codes = v.codes.flatten(order="F")
            names = list(v.categories)
            counts = np.array([np.sum(codes == k) for k in range(len(names))], dtype=float)
        else:
            codes, names, _ = group_codes(v)
            counts = np.array([np.sum(codes == k) for k in range(len(names))], dtype=float)
        vals = None
    total = counts.sum()
    pct = 100 * counts / total if total else np.zeros_like(counts)
    if nargout == 0:
        w = max([5] + [len(s) for s in names])
        lines = [f"  {'Value'.rjust(w)}    Count   Percent"]
        lines += [f"  {s.rjust(w)}    {int(c):5d}    {p:6.2f}%" for s, c, p in zip(names, counts, pct)]
        interp.out("\n".join(lines) + "\n")
        return None
    if numeric:
        return np.column_stack([vals, counts, pct]) if vals.size else np.zeros((0, 3))
    data = np.empty((len(names), 3), dtype=object)
    for i, s in enumerate(names):
        data[i, 0] = char(s)
        data[i, 1] = np.array([[counts[i]]])
        data[i, 2] = np.array([[pct[i]]])
    return Cell(data)
