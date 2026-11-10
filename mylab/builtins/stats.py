"""Statistics and probability: distributions, descriptive statistics, tests, moving windows."""

from __future__ import annotations

import numpy as np
import scipy.stats as st

from ..errors import MatlabError
from ..values import Cell, MString, Struct, char, is_char, to_int, to_str
from ._registry import builtin
from ._util import default_dim, dims_from_args, fnum, get_dim, is_text, keep2d, nargs, text


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


def _options(args: list, start: int) -> dict[str, object]:
    opts: dict[str, object] = {}
    k = start
    while k < len(args) - 1:
        if is_text(args[k]):
            opts[text(args[k]).lower()] = args[k + 1]
        k += 2
    return opts


def _vec_out(values, like: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    return v.reshape(1, -1) if like.shape[0] == 1 else v.reshape(-1, 1)


# -- probability distributions --------------------------------------------------------

def _dist(name, fn, defaults):
    """fn(x, *params) elementwise with scalar expansion; trailing params optional."""
    def f(interp, args, nargout):
        nparams = len(defaults)
        rest = list(args)
        upper = bool(rest) and is_text(rest[-1]) and text(rest.pop()).lower() == "upper"
        nargs(rest, 1, 1 + nparams, name)
        vals = list(rest) + [None] * (1 + nparams - len(rest))
        vals = [v if v is not None else np.array([[d]]) for v, d in zip(vals, (None,) + tuple(defaults))]
        arrs = _bcast(name, *vals)
        with np.errstate(all="ignore"):
            r = np.asarray(fn(*arrs), dtype=float)
        if upper:
            r = 1.0 - r if not hasattr(fn, "upper") else fn.upper(*arrs)
        return r.reshape(arrs[0].shape) if arrs[0].ndim == 2 else r.reshape(1, 1)
    return f


def _valid_scale(r, s):
    return np.where(s > 0, r, np.nan)


def _norm_cdf(x, mu, s):
    return _valid_scale(st.norm.cdf(x, mu, np.where(s > 0, s, 1)), s)


_norm_cdf.upper = lambda x, mu, s: _valid_scale(st.norm.sf(x, mu, np.where(s > 0, s, 1)), s)  # type: ignore

_DISTS = {
    "normpdf": (lambda x, mu, s: _valid_scale(st.norm.pdf(x, mu, np.where(s > 0, s, 1)), s), (0.0, 1.0)),
    "normcdf": (_norm_cdf, (0.0, 1.0)),
    "norminv": (lambda p, mu, s: _valid_scale(st.norm.ppf(p, mu, np.where(s > 0, s, 1)), s), (0.0, 1.0)),
    "unifpdf": (lambda x, a, b: np.where(a < b, np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0), np.nan),
                (0.0, 1.0)),
    "unifcdf": (lambda x, a, b: np.where(a < b, np.clip((x - a) / (b - a), 0, 1), np.nan), (0.0, 1.0)),
    "exppdf": (lambda x, mu: _valid_scale(st.expon.pdf(x, scale=np.where(mu > 0, mu, 1)), mu), (1.0,)),
    "expcdf": (lambda x, mu: _valid_scale(st.expon.cdf(x, scale=np.where(mu > 0, mu, 1)), mu), (1.0,)),
    "binopdf": (lambda x, n, p: st.binom.pmf(x, n, p), ()),
    "binocdf": (lambda x, n, p: st.binom.cdf(np.floor(x), n, p), ()),
    "poisspdf": (lambda x, lam: st.poisson.pmf(x, lam), ()),
    "poisscdf": (lambda x, lam: st.poisson.cdf(np.floor(x), lam), ()),
    "tpdf": (lambda x, v: st.t.pdf(x, v), ()),
    "tcdf": (lambda x, v: st.t.cdf(x, v), ()),
    "tinv": (lambda p, v: st.t.ppf(p, v), ()),
    "chi2pdf": (lambda x, v: st.chi2.pdf(x, v), ()),
    "chi2cdf": (lambda x, v: st.chi2.cdf(x, v), ()),
    "chi2inv": (lambda p, v: st.chi2.ppf(p, v), ()),
    "fpdf": (lambda x, a, b: st.f.pdf(x, a, b), ()),
    "fcdf": (lambda x, a, b: st.f.cdf(x, a, b), ()),
    "finv": (lambda p, a, b: st.f.ppf(p, a, b), ()),
    "gampdf": (lambda x, a, b: st.gamma.pdf(x, a, scale=b), (1.0,)),
    "gamcdf": (lambda x, a, b: st.gamma.cdf(x, a, scale=b), (1.0,)),
    "betapdf": (lambda x, a, b: st.beta.pdf(x, a, b), ()),
    "betacdf": (lambda x, a, b: st.beta.cdf(x, a, b), ()),
}

# number of required distribution parameters (beyond x)
_REQUIRED = {"binopdf": 2, "binocdf": 2, "poisspdf": 1, "poisscdf": 1, "tpdf": 1, "tcdf": 1, "tinv": 1,
             "chi2pdf": 1, "chi2cdf": 1, "chi2inv": 1, "fpdf": 2, "fcdf": 2, "finv": 2,
             "gampdf": 1, "gamcdf": 1, "betapdf": 2, "betacdf": 2}


def _dist_builtin(name, fn, defaults):
    req = _REQUIRED.get(name, 0)
    total = req + len(defaults)
    # required params get dummy defaults; nargs check enforces presence
    inner = _dist(name, fn, (np.nan,) * req + tuple(defaults))

    def f(interp, args, nargout):
        plain = [a for a in args if not is_text(a)]
        if len(plain) < 1 + req:
            raise MatlabError(f"Not enough input arguments for {name}.")
        if len(plain) > 1 + total:
            raise MatlabError(f"Too many input arguments for {name}.")
        return inner(interp, args, nargout)
    return f


for _n, (_fn, _defs) in _DISTS.items():
    builtin(_n)(_dist_builtin(_n, _fn, _defs))


def _rnd(name, nparams, sampler):
    def f(interp, args, nargout):
        nargs(args, nparams, None, name)
        params = _bcast(name, *args[:nparams]) if nparams else []
        if len(args) > nparams:
            shape, _ = dims_from_args(list(args[nparams:]), name)
        elif params:
            shape = params[0].shape
        else:
            shape = (1, 1)
        params = [np.broadcast_to(p, shape) if p.size == 1 else p for p in params]
        if any(p.shape != tuple(shape) for p in params):
            raise MatlabError(f"{name}: size arguments must match the size of the parameters.")
        return np.asarray(sampler(_rng(interp), shape, *params), dtype=float).reshape(shape)
    return f


def _normrnd(rng, shape, mu, s):
    return np.where(s >= 0, mu + np.where(s >= 0, s, 0) * rng.standard_normal(shape), np.nan)


builtin("normrnd")(_rnd("normrnd", 2, _normrnd))
builtin("unifrnd")(_rnd("unifrnd", 2, lambda rng, shape, a, b: np.where(a <= b, a + (b - a) * rng.random(shape),
                                                                         np.nan)))
builtin("exprnd")(_rnd("exprnd", 1, lambda rng, shape, mu: np.where(mu > 0, rng.exponential(1.0, shape) * mu,
                                                                     np.nan)))
builtin("binornd")(_rnd("binornd", 2, lambda rng, shape, n, p: rng.binomial(n.astype(np.int64), p, shape)))
builtin("poissrnd")(_rnd("poissrnd", 1, lambda rng, shape, lam: rng.poisson(lam, shape)))


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
        nargs(args, 2, 3, name)
        x = fnum(args[0]).astype(float)
        p = fnum(args[1]).flatten().astype(float) / scale
        if np.any((p < 0) | (p > 1)):
            raise MatlabError(f"{name}: probabilities must be in range.")
        if x.size == 0:
            return np.full((1, p.size), np.nan)
        axis = get_dim(args, 2, x)
        xm = np.moveaxis(x, axis, 0)
        r = np.empty((p.size,) + xm.shape[1:])
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
        return np.moveaxis(r, 0, axis)
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


@builtin("mad")
def _mad(interp, args, nargout):
    nargs(args, 1, 3, "mad")
    x = fnum(args[0]).astype(float)
    flag = int(_scalar(args[1])) if _given(args, 1) else 0
    axis = _axis_and(args, 2, x)
    if flag == 0:
        r = np.nanmean(np.abs(x - np.nanmean(x, axis=axis, keepdims=True)), axis=axis)
    else:
        r = np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)
    return keep2d(np.asarray(r), axis)


@builtin("zscore")
def _zscore(interp, args, nargout):
    nargs(args, 1, 3, "zscore")
    x = fnum(args[0]).astype(float)
    flag = int(_scalar(args[1])) if _given(args, 1) else 0
    if x.size == 0:
        return x
    axis = get_dim(args, 2, x)
    n = x.shape[axis]
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True, ddof=0 if (flag == 1 or n == 1) else 1)
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


def _tied_ranks(X: np.ndarray) -> np.ndarray:
    return np.column_stack([st.rankdata(X[:, j]) for j in range(X.shape[1])]) if X.size else X


@builtin("corr")
def _corr(interp, args, nargout):
    nargs(args, 1, None, "corr")
    plain = []
    k = 0
    while k < len(args) and not is_text(args[k]):
        plain.append(args[k])
        k += 1
    opts = _options(args, k)
    kind = text(opts["type"]).lower() if "type" in opts else "pearson"
    X = fnum(plain[0]).astype(float)
    Y = fnum(plain[1]).astype(float) if len(plain) > 1 else X
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("corr: X and Y must have the same number of rows.")
    if kind == "spearman":
        Xr, Yr = _tied_ranks(X), _tied_ranks(Y)
    elif kind == "pearson":
        Xr, Yr = X, Y
    else:
        raise MatlabError(f"corr: unsupported type '{kind}'.")
    R = _pearson(Xr, Yr)
    if len(plain) == 1:
        np.fill_diagonal(R, np.where(np.isnan(np.diag(R)), np.nan, 1.0))
    if nargout > 1:
        return R, _corr_pvals(R, X.shape[0])
    return R


# -- hypothesis tests ------------------------------------------------------------------

def _tail_p(t, df, tail):
    if tail == "both":
        return 2 * st.t.sf(np.abs(t), df)
    if tail == "right":
        return st.t.sf(t, df)
    return st.t.cdf(t, df)


def _t_ci(center, se, df, alpha, tail):
    if tail == "both":
        c = st.t.ppf(1 - alpha / 2, df)
        return center - c * se, center + c * se
    c = st.t.ppf(1 - alpha, df)
    if tail == "right":
        return center - c * se, np.full_like(center, np.inf)
    return np.full_like(center, -np.inf), center + c * se


def _test_opts(args, start):
    opts = _options(args, start)
    alpha = _scalar(opts["alpha"]) if "alpha" in opts else 0.05
    tail = text(opts["tail"]).lower() if "tail" in opts else "both"
    tail = {"two": "both", "ne": "both", "both": "both", "right": "right", "gt": "right",
            "left": "left", "lt": "left"}.get(tail)
    if tail is None:
        raise MatlabError("Tail must be 'both', 'right' or 'left'.")
    dim = to_int(opts["dim"]) - 1 if "dim" in opts else None
    return alpha, tail, dim, opts


def _stack_ci(lo, hi, axis):
    return np.concatenate([lo, hi], axis=axis)


@builtin("ttest")
def _ttest(interp, args, nargout):
    nargs(args, 1, None, "ttest")
    plain = []
    k = 0
    while k < len(args) and not is_text(args[k]):
        plain.append(args[k])
        k += 1
    alpha, tail, dim, _ = _test_opts(args, k)
    x = fnum(plain[0]).astype(float)
    m = 0.0
    if len(plain) > 1:
        y = fnum(plain[1]).astype(float)
        if y.size == 1:
            m = float(y.flat[0])
        else:
            if y.shape != x.shape:
                raise MatlabError("ttest: paired samples must have the same size.")
            x = x - y
    axis = dim if dim is not None else default_dim(x)
    cnt = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    with np.errstate(all="ignore"):
        sd = np.nanstd(x, axis=axis, keepdims=True, ddof=1)
        se = sd / np.sqrt(cnt)
        t = (mean - m) / se
    df = cnt - 1
    p = _tail_p(t, df, tail)
    h = np.where(np.isnan(p), np.nan, (p <= alpha).astype(float))
    lo, hi = _t_ci(mean, se, df, alpha, tail)
    ci = _stack_ci(lo, hi, axis)
    stats = Struct.scalar({"tstat": t, "df": df, "sd": sd})
    return (h, p, ci, stats)[:max(nargout, 1)]


@builtin("ttest2")
def _ttest2(interp, args, nargout):
    nargs(args, 2, None, "ttest2")
    alpha, tail, dim, opts = _test_opts(args, 2)
    vartype = text(opts["vartype"]).lower() if "vartype" in opts else "equal"
    x = fnum(args[0]).astype(float)
    y = fnum(args[1]).astype(float)
    if x.shape[0] == 1 and y.shape[0] == 1 and dim is None:
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        row = True
    else:
        row = False
        if x.shape[1] == 1 and y.shape[0] == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] == 1 and x.shape[0] == 1:
            x = x.reshape(-1, 1)
    axis = dim if dim is not None else 0
    nx = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    ny = np.sum(~np.isnan(y), axis=axis, keepdims=True).astype(float)
    mx, my = np.nanmean(x, axis=axis, keepdims=True), np.nanmean(y, axis=axis, keepdims=True)
    vx, vy = np.nanvar(x, axis=axis, keepdims=True, ddof=1), np.nanvar(y, axis=axis, keepdims=True, ddof=1)
    diff = mx - my
    with np.errstate(all="ignore"):
        if vartype == "equal":
            df = nx + ny - 2
            sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / df)
            se = sp * np.sqrt(1 / nx + 1 / ny)
            sd = sp
        elif vartype == "unequal":
            a, b = vx / nx, vy / ny
            se = np.sqrt(a + b)
            df = (a + b) ** 2 / (a ** 2 / (nx - 1) + b ** 2 / (ny - 1))
            sd = np.concatenate([np.sqrt(vx), np.sqrt(vy)], axis=axis)
        else:
            raise MatlabError("ttest2: Vartype must be 'equal' or 'unequal'.")
        t = diff / se
    p = _tail_p(t, df, tail)
    h = np.where(np.isnan(p), np.nan, (p <= alpha).astype(float))
    lo, hi = _t_ci(diff, se, df, alpha, tail)
    ci = _stack_ci(lo, hi, axis)
    if row:
        ci = ci.reshape(1, -1)
    stats = Struct.scalar({"tstat": t, "df": df, "sd": sd})
    return (h, p, ci, stats)[:max(nargout, 1)]


@builtin("ranksum")
def _ranksum(interp, args, nargout):
    nargs(args, 2, None, "ranksum")
    opts = _options(args, 2)
    alpha = _scalar(opts["alpha"]) if "alpha" in opts else 0.05
    method = text(opts["method"]).lower() if "method" in opts else None
    x = fnum(args[0]).flatten().astype(float)
    y = fnum(args[1]).flatten().astype(float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    nx, ny = x.size, y.size
    ranks = st.rankdata(np.concatenate([x, y]))
    w = float(ranks[:nx].sum())
    ties = np.unique(np.concatenate([x, y])).size != nx + ny
    if method is None:
        method = "exact" if (nx + ny < 20 and min(nx, ny) < 10 and not ties) else "approximate"
    u = w - nx * (nx + 1) / 2
    stats = {"ranksum": np.array([[w]])}
    if method == "exact":
        res = st.mannwhitneyu(x, y, alternative="two-sided", method="exact")
        p = float(res.pvalue)
    else:
        n = nx + ny
        _, counts = np.unique(ranks, return_counts=True)
        tie_term = np.sum(counts ** 3 - counts)
        mu = nx * ny / 2
        sigma = np.sqrt(nx * ny / 12 * ((n + 1) - tie_term / (n * (n - 1))))
        d = u - mu
        z = (d - 0.5 * np.sign(d)) / sigma if sigma > 0 else 0.0
        p = float(min(1.0, 2 * st.norm.sf(abs(z))))
        stats["zval"] = np.array([[float(z)]])
    h = float(p <= alpha)
    return (p, h, Struct.scalar(stats))[:max(nargout, 1)]


def _group_labels(g) -> list:
    if isinstance(g, Cell):
        return [to_str(v) for v in g.values_f()]
    if is_char(g):
        if g.shape[0] > 1:
            return ["".join(r).rstrip() for r in g]
        return list(to_str(g))
    if isinstance(g, MString):
        return [str(g)]
    return list(fnum(g).flatten(order="F"))


def _label_text(v) -> str:
    if isinstance(v, str):
        return v
    from ..display import format_scalar_plain
    return format_scalar_plain(v)


@builtin("anova1")
def _anova1(interp, args, nargout):
    nargs(args, 1, 3, "anova1")
    y = fnum(args[0]).astype(float)
    groups: list[np.ndarray] = []
    names: list[str] = []
    if _given(args, 1):
        labels = _group_labels(args[1])
        yv = y.flatten(order="F")
        if len(labels) != yv.size:
            if y.ndim == 2 and len(labels) == y.shape[1] and y.shape[0] > 1:
                for j, lab in enumerate(labels):
                    groups.append(y[:, j])
                    names.append(_label_text(lab))
            else:
                raise MatlabError("anova1: group must have one label per observation.")
        else:
            order: list = []
            for lab in labels:
                if lab not in order:
                    order.append(lab)
            for lab in order:
                groups.append(np.array([v for v, l2 in zip(yv, labels) if l2 == lab]))
                names.append(_label_text(lab))
    else:
        if y.shape[0] == 1:
            y = y.reshape(-1, 1)
        for j in range(y.shape[1]):
            groups.append(y[:, j])
            names.append(str(j + 1))
    groups = [g[~np.isnan(g)] for g in groups]
    keep = [k for k, g in enumerate(groups) if g.size]
    groups = [groups[k] for k in keep]
    names = [names[k] for k in keep]
    allv = np.concatenate(groups)
    grand = allv.mean()
    k, n = len(groups), allv.size
    ssb = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    sse = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    sst = ssb + sse
    dfb, dfe = k - 1, n - k
    with np.errstate(all="ignore"):
        msb = ssb / dfb if dfb else np.nan
        mse = sse / dfe if dfe else np.nan
        F = msb / mse
    p = float(st.f.sf(F, dfb, dfe)) if np.isfinite(F) else np.nan
    if nargout > 1:
        rows = [["Source", "SS", "df", "MS", "F", "Prob>F"],
                ["Groups", ssb, float(dfb), msb, F, p],
                ["Error", sse, float(dfe), mse, None, None],
                ["Total", sst, float(n - 1), None, None, None]]
        data = np.empty((4, 6), dtype=object)
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                if isinstance(v, str):
                    data[i, j] = char(v)
                elif v is None:
                    data[i, j] = np.zeros((0, 0))
                else:
                    data[i, j] = np.array([[float(v)]])
        tbl = Cell(data)
        stats = Struct.scalar({
            "gnames": Cell.from_list([char(s) for s in names], column=True),
            "n": np.array([[float(g.size) for g in groups]]),
            "source": char("anova1"),
            "means": np.array([[g.mean() for g in groups]]),
            "df": np.array([[float(dfe)]]),
            "s": np.array([[float(np.sqrt(mse))]]),
        })
        return (p, tbl, stats)[:nargout]
    return p


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
            t2 = st.t.ppf(1 - alpha / 2, dfe - 1)
            half = t2 * np.sqrt(si2 * (1 - h))
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


@builtin("tabulate")
def _tabulate(interp, args, nargout):
    nargs(args, 1, 1, "tabulate")
    v = args[0]
    if isinstance(v, Cell) or is_char(v) and v.shape[0] > 1 or isinstance(v, MString):
        labels = _group_labels(v) if not isinstance(v, MString) else [str(v)]
        order: list[str] = []
        for lab in labels:
            if lab not in order:
                order.append(lab)
        counts = [labels.count(lab) for lab in order]
        total = len(labels)
        rows = [[char(lab), float(c), 100.0 * c / total] for lab, c in zip(order, counts)]
        if nargout == 0:
            lines = ["  Value    Count   Percent"]
            lines += [f"  {lab:>5}    {c:5d}    {100.0 * c / total:6.2f}%" for lab, c in zip(order, counts)]
            interp.out("\n".join(lines) + "\n")
            return None
        data = np.empty((len(rows), 3), dtype=object)
        for i, r in enumerate(rows):
            data[i, 0] = r[0]
            data[i, 1] = np.array([[r[1]]])
            data[i, 2] = np.array([[r[2]]])
        return Cell(data)
    x = fnum(v).flatten().astype(float)
    x = x[~np.isnan(x)]
    if x.size and np.all(x == np.round(x)) and np.all(x > 0):
        vals = np.arange(1, int(x.max()) + 1, dtype=float)
    else:
        vals = np.unique(x)
    counts = np.array([np.sum(x == u) for u in vals], dtype=float)
    pct = 100 * counts / max(x.size, 1)
    table = np.column_stack([vals, counts, pct]) if vals.size else np.zeros((0, 3))
    if nargout == 0:
        from ..display import format_scalar_plain
        lines = ["  Value    Count   Percent"]
        lines += [f"  {format_scalar_plain(a):>5}    {int(c):5d}    {p:6.2f}%" for a, c, p in table]
        interp.out("\n".join(lines) + "\n")
        return None
    return table


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
    before, after = _window_extent(args[1])
    rest = list(args[2:])
    dim = None
    nanflag = "includenan"
    endpoints = "shrink"
    weight = 0
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
                endpoints = text(e).lower() if is_text(e) else float(_scalar(e))
                k += 2
                continue
            if s == "samplepoints":
                raise MatlabError(f"{name}: SamplePoints is not supported.")
            raise MatlabError(f"{name}: unknown option '{s}'.")
        numeric_pos.append(a)
        k += 1
    if name in ("movstd", "movvar") and numeric_pos:
        w = numeric_pos.pop(0)
        weight = int(_scalar(w)) if _given([w], 0) else 0
    if numeric_pos:
        dim = to_int(numeric_pos[0], "dimension") - 1
    axis = dim if dim is not None else default_dim(x)
    return x, before, after, axis, nanflag, endpoints, weight


def _moving(name, reducer):
    def f(interp, args, nargout):
        x, before, after, axis, nanflag, endpoints, weight = _parse_mov(args, name)
        if x.size == 0:
            return x
        xm = np.moveaxis(x, axis, 0)
        n = xm.shape[0]
        out = np.empty_like(xm)
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n):
                lo, hi = i - before, i + after
                win = xm[max(lo, 0):min(hi, n - 1) + 1]
                if nanflag == "omitnan":
                    out[i] = reducer(win, True, weight)
                else:
                    out[i] = reducer(win, False, weight)
                if endpoints != "shrink" and (lo < 0 or hi > n - 1):
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


@builtin("smoothdata")
def _smoothdata(interp, args, nargout):
    nargs(args, 1, None, "smoothdata")
    x = fnum(args[0]).astype(float)
    rest = list(args[1:])
    dim = None
    if rest and not is_text(rest[0]):
        dim = to_int(rest.pop(0), "dimension") - 1
    method = "movmean"
    if rest and is_text(rest[0]) and text(rest[0]).lower() in (
            "movmean", "movmedian", "gaussian", "sgolay", "lowess", "loess", "rlowess", "rloess"):
        method = text(rest.pop(0)).lower()
    window = None
    if rest and not is_text(rest[0]):
        window = rest.pop(0)
    nanflag = "omitnan"
    degree = 2
    k = 0
    while k < len(rest):
        s = text(rest[k]).lower()
        if s in ("omitnan", "includenan"):
            nanflag = s
            k += 1
        elif s == "degree":
            degree = to_int(rest[k + 1])
            k += 2
        elif s == "smoothingfactor":
            k += 2
        else:
            raise MatlabError(f"smoothdata: unknown option '{s}'.")
    axis = dim if dim is not None else default_dim(x)
    n = x.shape[axis]
    if window is None:
        # simple heuristic default: about a tenth of the data, at least 3 points
        window = np.array([[float(max(3, int(round(n / 10))))]])
    if method in ("movmean", "movmedian"):
        fn = _moving(method, _ext_red(np.mean, np.nanmean) if method == "movmean"
                     else _ext_red(np.median, np.nanmedian))
        return fn(interp, [x, window, float(axis + 1), nanflag], 1)
    if method == "gaussian":
        before, after = _window_extent(window)
        L = before + after + 1
        sigma = max(L - 1, 1) / 5.0
        offs = np.arange(-before, after + 1)
        wts = np.exp(-0.5 * (offs / sigma) ** 2)
        xm = np.moveaxis(x, axis, 0)
        out = np.empty_like(xm)
        for i in range(n):
            lo, hi = max(i - before, 0), min(i + after, n - 1)
            seg = xm[lo:hi + 1]
            ww = wts[(lo - i + before):(hi - i + before) + 1].reshape((-1,) + (1,) * (seg.ndim - 1))
            ww = np.broadcast_to(ww, seg.shape)
            if nanflag == "omitnan":
                mask = ~np.isnan(seg)
                with np.errstate(all="ignore"):
                    out[i] = np.sum(np.where(mask, seg, 0) * ww, axis=0) / np.sum(ww * mask, axis=0)
            else:
                out[i] = np.sum(seg * ww, axis=0) / np.sum(ww, axis=0)
        return np.moveaxis(out, 0, axis)
    if method == "sgolay":
        import scipy.signal as ss
        L = int(_scalar(window))
        if L % 2 == 0:
            L += 1
        L = min(L, n if n % 2 else n - 1)
        if L <= degree:
            return x
        return ss.savgol_filter(x, L, degree, axis=axis, mode="interp")
    raise MatlabError(f"smoothdata: method '{method}' is not supported.")


@builtin("normalize")
def _normalize(interp, args, nargout):
    nargs(args, 1, 4, "normalize")
    x = fnum(args[0]).astype(float)
    rest = list(args[1:])
    dim = None
    if rest and not is_text(rest[0]):
        dim = to_int(rest.pop(0), "dimension") - 1
    method = text(rest[0]).lower() if rest else "zscore"
    param = rest[1] if len(rest) > 1 else None
    if x.size == 0:
        return x
    axis = dim if dim is not None else default_dim(x)
    import warnings
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == "zscore":
            kind = text(param).lower() if param is not None else "std"
            if kind == "std":
                c = np.nanmean(x, axis=axis, keepdims=True)
                s = np.nanstd(x, axis=axis, keepdims=True, ddof=1 if x.shape[axis] > 1 else 0)
            elif kind == "robust":
                c = np.nanmedian(x, axis=axis, keepdims=True)
                s = np.nanmedian(np.abs(x - c), axis=axis, keepdims=True)
            else:
                raise MatlabError(f"normalize: unknown zscore type '{kind}'.")
            return (x - c) / s
        if method == "range":
            lim = fnum(param).flatten() if param is not None else np.array([0.0, 1.0])
            lo, hi = np.nanmin(x, axis=axis, keepdims=True), np.nanmax(x, axis=axis, keepdims=True)
            return lim[0] + (x - lo) / (hi - lo) * (lim[1] - lim[0])
        if method == "norm":
            p = _scalar(param) if param is not None else 2.0
            if np.isinf(p):
                nrm = np.nanmax(np.abs(x), axis=axis, keepdims=True)
            else:
                nrm = np.nansum(np.abs(x) ** p, axis=axis, keepdims=True) ** (1 / p)
            return x / nrm
        if method == "center":
            kind = param
            if kind is None or is_text(kind) and text(kind).lower() == "mean":
                c = np.nanmean(x, axis=axis, keepdims=True)
            elif is_text(kind) and text(kind).lower() == "median":
                c = np.nanmedian(x, axis=axis, keepdims=True)
            else:
                c = fnum(kind)
            return x - c
        if method == "scale":
            kind = param
            if kind is None or is_text(kind) and text(kind).lower() == "std":
                s = np.nanstd(x, axis=axis, keepdims=True, ddof=1 if x.shape[axis] > 1 else 0)
            elif is_text(kind) and text(kind).lower() == "mad":
                med = np.nanmedian(x, axis=axis, keepdims=True)
                s = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
            elif is_text(kind) and text(kind).lower() == "first":
                s = np.take(x, [0], axis=axis)
            elif is_text(kind) and text(kind).lower() == "iqr":
                s = np.diff(np.nanpercentile(x, [25, 75], axis=axis, method="hazen"), axis=0)[0]
                s = np.expand_dims(s, axis)
            else:
                s = fnum(kind)
            return x / s
    raise MatlabError(f"normalize: unknown method '{method}'.")


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
    opts = _options(args, k)
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


@builtin("pdist")
def _pdist(interp, args, nargout):
    from scipy.spatial.distance import pdist
    nargs(args, 1, 3, "pdist")
    X = fnum(args[0]).astype(float)
    metric = text(args[1]).lower() if len(args) > 1 else "euclidean"
    if metric not in _METRICS:
        raise MatlabError(f"pdist: unsupported distance '{metric}'.")
    if metric == "spearman":
        R = np.apply_along_axis(st.rankdata, 1, X)
        return pdist(R, "correlation").reshape(1, -1)
    kw = {}
    if metric == "minkowski":
        kw["p"] = _scalar(args[2]) if len(args) > 2 else 2.0
    if metric == "seuclidean":
        kw["V"] = np.var(X, axis=0, ddof=1) if len(args) < 3 else fnum(args[2]).flatten()
    d = pdist(X, _METRICS[metric], **kw)
    return d.reshape(1, -1)


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


@builtin("kmeans")
def _kmeans(interp, args, nargout):
    nargs(args, 2, None, "kmeans")
    X = fnum(args[0]).astype(float)
    if X.ndim != 2:
        raise MatlabError("kmeans: X must be a matrix.")
    k = to_int(args[1], "number of clusters")
    opts = _options(args, 2)
    max_iter = int(_scalar(opts["maxiter"])) if "maxiter" in opts else 100
    reps = int(_scalar(opts["replicates"])) if "replicates" in opts else 1
    dist = text(opts["distance"]).lower() if "distance" in opts else "sqeuclidean"
    if dist not in ("sqeuclidean", "cityblock"):
        raise MatlabError(f"kmeans: unsupported distance '{dist}'.")
    keep = ~np.isnan(X).any(axis=1)
    Xk = X[keep]
    n = Xk.shape[0]
    if k < 1 or k > n:
        raise MatlabError("kmeans: number of clusters must be between 1 and the number of points.")
    rng = _rng(interp)
    start = opts.get("start")
    starts: list[np.ndarray] = []
    if start is not None and not is_text(start):
        S = fnum(start).astype(float)
        if S.shape != (k, X.shape[1]):
            raise MatlabError("kmeans: Start matrix must be k-by-p.")
        starts = [S]
    else:
        mode = text(start).lower() if start is not None else "plus"
        for _ in range(reps):
            if mode == "sample":
                starts.append(Xk[rng.choice(n, k, replace=False)].copy())
            elif mode == "uniform":
                lo, hi = Xk.min(axis=0), Xk.max(axis=0)
                starts.append(lo + (hi - lo) * rng.random((k, X.shape[1])))
            else:
                C = [Xk[rng.integers(n)]]
                for _j in range(1, k):
                    d2 = np.min(((Xk[:, None, :] - np.array(C)[None]) ** 2).sum(axis=2), axis=1)
                    tot = d2.sum()
                    C.append(Xk[rng.choice(n, p=d2 / tot)] if tot > 0 else Xk[rng.integers(n)])
                starts.append(np.array(C))

    def distances(C):
        if dist == "cityblock":
            return np.abs(Xk[:, None, :] - C[None]).sum(axis=2)
        return ((Xk[:, None, :] - C[None]) ** 2).sum(axis=2)

    best = None
    for C0 in starts:
        C = C0.copy()
        idx = np.full(n, -1)
        for _it in range(max_iter):
            D = distances(C)
            new = np.argmin(D, axis=1)
            if np.array_equal(new, idx):
                break
            idx = new
            for j in range(k):
                members = Xk[idx == j]
                if members.size:
                    C[j] = np.median(members, axis=0) if dist == "cityblock" else members.mean(axis=0)
                else:
                    # re-seed an empty cluster at the point farthest from its centroid
                    far = np.argmax(D[np.arange(n), idx])
                    C[j] = Xk[far]
        D = distances(C)
        idx = np.argmin(D, axis=1)
        sumd = np.array([D[idx == j, j].sum() for j in range(k)])
        if best is None or sumd.sum() < best[2].sum():
            best = (idx, C, sumd, D)
    idx, C, sumd, D = best
    idx_full = np.full((X.shape[0], 1), np.nan)
    idx_full[keep, 0] = idx + 1.0
    D_full = np.full((X.shape[0], k), np.nan)
    D_full[keep] = D
    return (idx_full, C, sumd.reshape(-1, 1), D_full)[:max(nargout, 1)]

