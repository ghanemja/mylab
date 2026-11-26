"""Hypothesis tests: location, variance and distribution tests, analysis of variance and its
rank-based relatives, and multiple comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct, char
from ._registry import builtin
from ._stats_util import (
    as_bool, builtin_call, call_num, cellstr, choice, get_opts, given, group_codes, group_list, intval,
    is_empty_arg, is_table, outs, scalar, struct, text_list, tied_rank, vec,
)
from ._util import default_dim, fnum, is_text, text

_TAILS = {"both": "both", "ne": "both", "unequal": "both", "right": "right", "gt": "right", "larger": "right",
          "left": "left", "lt": "left", "smaller": "left"}


def _tail(v: Any) -> str:
    if not is_text(v) or text(v).lower() not in _TAILS:
        raise MatlabError("Tail must be 'both', 'right' or 'left'.")
    return _TAILS[text(v).lower()]


def _test_args(args: list, ndata: int, fname: str, extra: tuple = (), legacy: tuple = ("alpha", "tail", "dim")):
    """Data arguments plus alpha/tail/dim given by name or (older style) by position."""
    pos, opts = get_opts(args, 1, ["Alpha", "Tail", "Dim", *extra], fname)
    data = pos[:ndata]
    rest = pos[ndata:]
    if len(rest) > len(legacy):
        raise MatlabError(f"Too many input arguments for {fname}.")
    for name, v in zip(legacy, rest):
        if not is_empty_arg(v):
            if name in opts:
                raise MatlabError(f"{fname}: '{name}' was given twice.")
            opts[name] = v
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    if not 0 < alpha < 1:
        raise MatlabError(f"{fname}: Alpha must be between 0 and 1.")
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    dim = intval(opts["dim"], "Dim") - 1 if "dim" in opts else None
    return data, alpha, tail, dim, opts


def _norm_p(z, tail):
    import scipy.stats as st
    if tail == "both":
        return 2 * st.norm.sf(np.abs(z))
    return st.norm.sf(z) if tail == "right" else st.norm.cdf(z)


def _t_p(t, df, tail):
    import scipy.stats as st
    if tail == "both":
        return 2 * st.t.sf(np.abs(t), df)
    return st.t.sf(t, df) if tail == "right" else st.t.cdf(t, df)


def _interval(center, se, crit_fn, alpha, tail, axis):
    """Confidence limits stacked along ``axis`` (two-sided or one-sided)."""
    if tail == "both":
        c = crit_fn(1 - alpha / 2)
        lo, hi = center - c * se, center + c * se
    else:
        c = crit_fn(1 - alpha)
        if tail == "right":
            lo, hi = center - c * se, np.full_like(center, np.inf)
        else:
            lo, hi = np.full_like(center, -np.inf), center + c * se
    return np.concatenate([lo, hi], axis=axis)


def _decide(p, alpha):
    return np.where(np.isnan(p), np.nan, (p <= alpha).astype(float))


# -- t and z tests -----------------------------------------------------------------------------------------

@builtin("ttest")
def _ttest(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("Not enough input arguments for ttest.")
    data, alpha, tail, dim, _ = _test_args(args, 2, "ttest")
    x = np.asarray(fnum(data[0]), dtype=float)
    m = 0.0
    if len(data) > 1 and not is_empty_arg(data[1]):
        y = np.asarray(fnum(data[1]), dtype=float)
        if y.size == 1:
            m = float(y.flat[0])
        elif y.shape != x.shape:
            raise MatlabError("ttest: paired samples must have the same size.")
        else:
            x = x - y
    axis = dim if dim is not None else default_dim(x)
    cnt = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=axis, keepdims=True)
        sd = np.nanstd(x, axis=axis, keepdims=True, ddof=1)
        se = sd / np.sqrt(cnt)
        t = (mean - m) / se
    df = cnt - 1
    p = _t_p(t, df, tail)
    ci = _interval(mean, se, lambda q: st.t.ppf(q, df), alpha, tail, axis)
    stats = Struct.scalar({"tstat": t, "df": df, "sd": sd})
    return outs(nargout, _decide(p, alpha), p, ci, stats)


@builtin("ttest2")
def _ttest2(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for ttest2.")
    data, alpha, tail, dim, opts = _test_args(args, 2, "ttest2", ("Vartype",), ("alpha", "tail", "vartype", "dim"))
    vartype = choice(opts["vartype"], ["equal", "unequal"], "Vartype") if "vartype" in opts else "equal"
    x = np.asarray(fnum(data[0]), dtype=float)
    y = np.asarray(fnum(data[1]), dtype=float)
    row = False
    if dim is None and x.ndim == 2 and y.ndim == 2 and min(x.shape) == 1 and min(y.shape) == 1:
        row = x.shape[0] == 1 and x.size > 1
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    axis = dim if dim is not None else default_dim(x)
    nx = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    ny = np.sum(~np.isnan(y), axis=axis, keepdims=True).astype(float)
    with np.errstate(all="ignore"):
        mx, my = np.nanmean(x, axis=axis, keepdims=True), np.nanmean(y, axis=axis, keepdims=True)
        vx = np.nanvar(x, axis=axis, keepdims=True, ddof=1)
        vy = np.nanvar(y, axis=axis, keepdims=True, ddof=1)
        diff = mx - my
        if vartype == "equal":
            df = nx + ny - 2
            sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / df)
            se = sd * np.sqrt(1 / nx + 1 / ny)
        else:
            a, b = vx / nx, vy / ny
            se = np.sqrt(a + b)
            df = (a + b) ** 2 / (a ** 2 / (nx - 1) + b ** 2 / (ny - 1))
            sd = np.concatenate([np.sqrt(vx), np.sqrt(vy)], axis=axis)
        t = diff / se
    p = _t_p(t, df, tail)
    ci = _interval(diff, se, lambda q: st.t.ppf(q, df), alpha, tail, axis)
    if row:
        ci = ci.reshape(1, -1)
        sd = sd.reshape(1, -1)
    stats = Struct.scalar({"tstat": t, "df": df, "sd": sd})
    return outs(nargout, _decide(p, alpha), p, ci, stats)


@builtin("ztest")
def _ztest(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 3:
        raise MatlabError("ztest: expected ztest(x, m, sigma).")
    data, alpha, tail, dim, _ = _test_args(args, 3, "ztest")
    x = np.asarray(fnum(data[0]), dtype=float)
    m, sigma = scalar(data[1], "m"), scalar(data[2], "sigma")
    if sigma <= 0:
        raise MatlabError("ztest: sigma must be positive.")
    axis = dim if dim is not None else default_dim(x)
    cnt = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=axis, keepdims=True)
        se = sigma / np.sqrt(cnt)
        z = (mean - m) / se
    p = _norm_p(z, tail)
    ci = _interval(mean, se, st.norm.ppf, alpha, tail, axis)
    return outs(nargout, _decide(p, alpha), p, ci, z)


# -- variance tests --------------------------------------------------------------------------------------------

@builtin("vartest")
def _vartest(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("vartest: expected vartest(x, v).")
    data, alpha, tail, dim, _ = _test_args(args, 2, "vartest")
    x = np.asarray(fnum(data[0]), dtype=float)
    v = scalar(data[1], "v")
    if v <= 0:
        raise MatlabError("vartest: the hypothesised variance must be positive.")
    axis = dim if dim is not None else default_dim(x)
    cnt = np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float)
    df = cnt - 1
    with np.errstate(all="ignore"):
        s2 = np.nanvar(x, axis=axis, keepdims=True, ddof=1)
        chi = df * s2 / v
        if tail == "both":
            p = 2 * np.minimum(st.chi2.cdf(chi, df), st.chi2.sf(chi, df))
            ci = np.concatenate([df * s2 / st.chi2.ppf(1 - alpha / 2, df), df * s2 / st.chi2.ppf(alpha / 2, df)],
                    axis=axis)
        elif tail == "right":
            p = st.chi2.sf(chi, df)
            ci = np.concatenate([df * s2 / st.chi2.ppf(1 - alpha, df), np.full_like(s2, np.inf)], axis=axis)
        else:
            p = st.chi2.cdf(chi, df)
            ci = np.concatenate([np.zeros_like(s2), df * s2 / st.chi2.ppf(alpha, df)], axis=axis)
    return outs(nargout, _decide(p, alpha), p, ci, Struct.scalar({"chisqstat": chi, "df": df}))


@builtin("vartest2")
def _vartest2(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("vartest2: expected vartest2(x, y).")
    data, alpha, tail, dim, _ = _test_args(args, 2, "vartest2")
    x = np.asarray(fnum(data[0]), dtype=float)
    y = np.asarray(fnum(data[1]), dtype=float)
    if dim is None and min(x.shape) == 1 and min(y.shape) == 1:
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    axis = dim if dim is not None else default_dim(x)
    d1 = np.sum(~np.isnan(x), axis=axis, keepdims=True) - 1.0
    d2 = np.sum(~np.isnan(y), axis=axis, keepdims=True) - 1.0
    with np.errstate(all="ignore"):
        F = np.nanvar(x, axis=axis, keepdims=True, ddof=1) / np.nanvar(y, axis=axis, keepdims=True, ddof=1)
        if tail == "both":
            p = 2 * np.minimum(st.f.cdf(F, d1, d2), st.f.sf(F, d1, d2))
            ci = np.concatenate([F / st.f.ppf(1 - alpha / 2, d1, d2), F / st.f.ppf(alpha / 2, d1, d2)], axis=axis)
        elif tail == "right":
            p = st.f.sf(F, d1, d2)
            ci = np.concatenate([F / st.f.ppf(1 - alpha, d1, d2), np.full_like(F, np.inf)], axis=axis)
        else:
            p = st.f.cdf(F, d1, d2)
            ci = np.concatenate([np.zeros_like(F), F / st.f.ppf(alpha, d1, d2)], axis=axis)
    return outs(nargout, _decide(p, alpha), p, ci, Struct.scalar({"fstat": F, "df1": d1, "df2": d2}))


def _groups_from(y_arg: Any, g_arg: Any, fname: str) -> tuple[list[np.ndarray], list[str]]:
    """Samples per group from a matrix (columns) or from a vector with a grouping variable."""
    y = np.asarray(fnum(y_arg), dtype=float)
    if g_arg is None or is_empty_arg(g_arg):
        if y.ndim == 2 and y.shape[0] == 1:
            y = y.reshape(-1, 1)
        groups = [y[:, j] for j in range(y.shape[1])]
        names = [str(j + 1) for j in range(y.shape[1])]
    else:
        gl = group_list(g_arg)
        from ._stats_util import multi_group
        if len(gl) == 1:
            codes, names, _ = group_codes(gl[0])
        else:
            codes, names, _, _ = multi_group(gl, sep=" ")
        yv = y.flatten(order="F")
        if codes.size != yv.size:
            if y.ndim == 2 and codes.size == y.shape[1] and y.shape[0] > 1:
                groups, gn = [], []
                for k, nme in enumerate(names):
                    cols = np.flatnonzero(codes == k)
                    groups.append(y[:, cols].flatten(order="F"))
                    gn.append(nme)
                names = gn
                return _drop_empty(groups, names)
            raise MatlabError(f"{fname}: the grouping variable must have one value per observation.")
        groups = [yv[codes == k] for k in range(len(names))]
    return _drop_empty(groups, names)


def _drop_empty(groups, names):
    groups = [g[~np.isnan(g)] for g in groups]
    keep = [k for k, g in enumerate(groups) if g.size]
    return [groups[k] for k in keep], [names[k] for k in keep]


def _display_opt(v: Any, fname: str) -> bool:
    if v is None or is_empty_arg(v):
        return True
    return as_bool(v, f"{fname} display option")


def _anova_cell(header: list[str], rows: list[list]) -> Cell:
    data = np.empty((len(rows) + 1, len(header)), dtype=object)
    for j, h in enumerate(header):
        data[0, j] = char(h)
    for i, r in enumerate(rows):
        for j in range(len(header)):
            v = r[j] if j < len(r) else None
            data[i + 1, j] = char(v) if isinstance(v, str) else (np.zeros((0, 0)) if v is None
                    else np.array([[float(v)]]))
    return Cell(data)


def _print_table(interp, title: str, tbl: Cell) -> None:
    d = tbl.data
    cells = []
    for i in range(d.shape[0]):
        row = []
        for j in range(d.shape[1]):
            v = d[i, j]
            if isinstance(v, np.ndarray) and v.dtype.kind == "U":
                row.append("".join(v.flatten()))
            elif isinstance(v, np.ndarray) and v.size:
                x = float(v.flat[0])
                row.append(f"{x:.5g}" if x != int(x) or abs(x) >= 1e9 else str(int(x)))
            else:
                row.append("")
        cells.append(row)
    widths = [max(len(r[j]) for r in cells) for j in range(d.shape[1])]
    lines = [title, ""]
    for i, r in enumerate(cells):
        lines.append("  ".join(r[j].ljust(widths[j]) if j == 0 else r[j].rjust(widths[j]) for j in range(len(r))))
        if i == 0:
            lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    interp.out("\n".join(lines) + "\n\n")


@builtin("vartestn")
def _vartestn(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("vartestn: expected vartestn(x) or vartestn(x, group, ...).")
    pos, opts = get_opts(args, 1, ["Display", "TestType"], "vartestn")
    if len(pos) > 4:
        raise MatlabError("Too many input arguments for vartestn.")
    display = _display_opt(opts.get("display", pos[2] if len(pos) > 2 else None), "vartestn")
    kind_arg = opts.get("testtype", pos[3] if len(pos) > 3 else None)
    kinds = ["Bartlett", "LeveneQuadratic", "LeveneAbsolute", "BrownForsythe", "OBrien", "robust", "classical"]
    kind = choice(kind_arg, kinds, "TestType") if kind_arg is not None and not is_empty_arg(kind_arg) else "bartlett"
    kind = {"robust": "leveneabsolute", "classical": "bartlett"}.get(kind, kind)
    groups, names = _groups_from(pos[0], pos[1] if len(pos) > 1 else None, "vartestn")
    k = len(groups)
    n = np.array([g.size for g in groups], dtype=float)
    v = np.array([g.var(ddof=1) if g.size > 1 else np.nan for g in groups])
    N = n.sum()
    use = n > 1
    dfe = float(np.sum(n[use] - 1))
    sp = float(np.sum((n[use] - 1) * v[use]) / dfe) if dfe > 0 else np.nan
    if kind == "bartlett":
        kk = int(use.sum())
        with np.errstate(all="ignore"):
            B = (dfe * np.log(sp) - np.sum((n[use] - 1) * np.log(v[use]))) / \
                (1 + (np.sum(1 / (n[use] - 1)) - 1 / dfe) / (3 * (kk - 1)))
        p = float(st.chi2.sf(B, kk - 1))
        stats = struct(chisqstat=float(B), df=float(kk - 1))
        tbl = _anova_cell(["Group", "Count", "Mean", "Std Dev"],
                          [[names[i], n[i], groups[i].mean(), np.sqrt(v[i])] for i in range(k)]
                          + [["Pooled", N, np.concatenate(groups).mean(), np.sqrt(sp)],
                             ["Bartlett's statistic", float(B)], ["Degrees of freedom", float(kk - 1)], ["p-value", p]])
    else:
        if kind == "levenequadratic":
            z = [(g - g.mean()) ** 2 for g in groups]
        elif kind == "leveneabsolute":
            z = [np.abs(g - g.mean()) for g in groups]
        elif kind == "brownforsythe":
            z = [np.abs(g - np.median(g)) for g in groups]
        else:  # O'Brien's transformation (W = 0.5)
            z = []
            for g in groups:
                ng, mg, vg = g.size, g.mean(), g.var(ddof=1)
                z.append(((ng - 1.5) * ng * (g - mg) ** 2 - 0.5 * vg * (ng - 1)) / ((ng - 1) * (ng - 2)))
        allz = np.concatenate(z)
        grand = allz.mean()
        ssb = sum(zz.size * (zz.mean() - grand) ** 2 for zz in z)
        sse = sum(((zz - zz.mean()) ** 2).sum() for zz in z)
        d1, d2 = k - 1, N - k
        F = (ssb / d1) / (sse / d2)
        p = float(st.f.sf(F, d1, d2))
        stats = struct(fstat=float(F), df=np.array([[float(d1), float(d2)]]))
        tbl = _anova_cell(["Group", "Count", "Mean", "Std Dev"],
                          [[names[i], n[i], groups[i].mean(), np.sqrt(v[i])] for i in range(k)]
                          + [["Pooled", N, np.concatenate(groups).mean(), np.sqrt(sp)],
                             ["F statistic", float(F)], ["Degrees of freedom", f"{d1}, {int(d2)}"], ["p-value", p]])
    if display:
        _print_table(interp, "Group Summary Table", tbl)
    return outs(nargout, p, stats)


@builtin("barttest")
def _barttest(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 2:
        raise MatlabError("barttest: expected barttest(X) or barttest(X, alpha).")
    X = np.asarray(fnum(args[0]), dtype=float)
    alpha = scalar(args[1], "alpha") if given(args, 1) else 0.05
    n, p = X.shape
    if p < 2 or n < 2:
        raise MatlabError("barttest: X needs at least two columns and two rows.")
    lam = np.sort(np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None))[::-1]
    chis, probs = [], []
    with np.errstate(all="ignore"):
        for k in range(p - 1):
            q = p - k
            tail_vals = lam[k:]
            pooled = tail_vals.mean()
            chi = (n - 1) * (q * np.log(pooled) - np.sum(np.log(tail_vals)))
            df = (q - 1) * (q + 2) / 2
            chis.append(float(chi))
            probs.append(float(st.chi2.sf(chi, df)))
    sig = [pr < alpha for pr in probs]
    ndim = p
    for k in range(p - 1):
        if not sig[k]:
            ndim = max(k, 1) if k > 0 else 1
            break
    return outs(nargout, float(ndim), np.array(probs).reshape(-1, 1), np.array(chis).reshape(-1, 1))


# -- sign, signed-rank and rank-sum tests -------------------------------------------------------------------------

def _paired_diff(pos: list, fname: str) -> np.ndarray:
    x = vec(pos[0])
    if len(pos) > 1 and not is_empty_arg(pos[1]):
        y = vec(pos[1])
        if y.size == 1:
            d = x - y[0]
        elif y.size != x.size:
            raise MatlabError(f"{fname}: x and y must have the same number of elements.")
        else:
            d = x - y
    else:
        d = x
    return d[~np.isnan(d)]


def _method_opt(opts: dict, default: str) -> str:
    return choice(opts["method"], ["exact", "approximate"], "Method") if "method" in opts else default


@builtin("signtest")
def _signtest(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("Not enough input arguments for signtest.")
    pos, opts = get_opts(args, 1, ["Alpha", "Tail", "Method"], "signtest")
    if len(pos) > 2:
        raise MatlabError("Too many input arguments for signtest.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    d = _paired_diff(pos, "signtest")
    npos, nneg = int(np.sum(d > 0)), int(np.sum(d < 0))
    n = npos + nneg
    method = _method_opt(opts, "exact" if n < 100 else "approximate")
    stats = {"zval": np.array([[np.nan]]), "sign": np.array([[float(npos)]])}
    if n == 0:
        p = 1.0
    elif method == "exact":
        if tail == "both":
            p = min(1.0, 2 * st.binom.cdf(min(npos, nneg), n, 0.5))
        elif tail == "right":
            p = float(st.binom.sf(npos - 1, n, 0.5))
        else:
            p = float(st.binom.cdf(npos, n, 0.5))
    else:
        diff = npos - nneg
        if tail == "both":
            z = (diff - np.sign(diff)) / np.sqrt(n)
        elif tail == "right":
            z = (diff - 1) / np.sqrt(n)
        else:
            z = (diff + 1) / np.sqrt(n)
        p = float(_norm_p(z, tail))
        stats["zval"] = np.array([[float(z)]])
    return outs(nargout, float(p), float(p <= alpha), Struct.scalar(stats))


def _subset_sum_counts(vals2: np.ndarray) -> np.ndarray:
    """counts[s] = number of subsets of the integer values with sum s."""
    total = int(vals2.sum())
    counts = np.zeros(total + 1)
    counts[0] = 1
    for v in vals2.astype(int):
        if v > 0:
            counts[v:] = counts[v:] + counts[:-v].copy()
        else:
            counts = counts * 2
    return counts


@builtin("signrank")
def _signrank(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for signrank.")
    pos, opts = get_opts(args, 1, ["Alpha", "Tail", "Method"], "signrank")
    if len(pos) > 2:
        raise MatlabError("Too many input arguments for signrank.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    d = _paired_diff(pos, "signrank")
    eps_d = np.finfo(float).eps * 100 * np.max(np.abs(d), initial=0.0)
    d = d[np.abs(d) > eps_d]
    n = d.size
    stats = {"zval": np.array([[np.nan]]), "signedrank": np.array([[0.0]])}
    if n == 0:
        return outs(nargout, 1.0, 0.0, Struct.scalar(stats))
    ranks = tied_rank(np.abs(d))
    w = float(ranks[d > 0].sum())
    stats["signedrank"] = np.array([[w]])
    method = _method_opt(opts, "exact" if n <= 15 else "approximate")
    if method == "exact":
        counts = _subset_sum_counts(np.round(2 * ranks))
        probs = counts / counts.sum()
        k = int(round(2 * w))
        left, right = float(probs[:k + 1].sum()), float(probs[k:].sum())
        p = min(1.0, 2 * min(left, right)) if tail == "both" else (right if tail == "right" else left)
    else:
        _, cnt = np.unique(ranks, return_counts=True)
        tieadj = np.sum(cnt ** 3 - cnt) / 2.0
        sd = np.sqrt((n * (n + 1) * (2 * n + 1) - tieadj) / 24.0)
        z = (w - n * (n + 1) / 4.0) / sd if sd > 0 else 0.0
        p = float(_norm_p(z, tail))
        stats["zval"] = np.array([[float(z)]])
    return outs(nargout, float(p), float(p <= alpha), Struct.scalar(stats))


def _ranksum_exact(ranks: np.ndarray, nx: int, w: float) -> tuple[float, float]:
    """P(W <= w) and P(W >= w) for the rank sum of nx of the pooled ranks (all subsets equally likely)."""
    r2 = np.round(2 * ranks).astype(int)
    total = int(r2.sum())
    dp = np.zeros((nx + 1, total + 1))
    dp[0, 0] = 1
    for v in r2:
        for c in range(nx, 0, -1):
            dp[c, v:] += dp[c - 1, :total + 1 - v]
    probs = dp[nx] / dp[nx].sum()
    k = int(round(2 * w))
    return float(probs[:k + 1].sum()), float(probs[k:].sum())


@builtin("ranksum")
def _ranksum(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for ranksum.")
    pos, opts = get_opts(args, 2, ["Alpha", "Tail", "Method"], "ranksum")
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for ranksum.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else (scalar(pos[2], "alpha") if len(pos) > 2 else 0.05)
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    x, y = vec(pos[0]), vec(pos[1])
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        raise MatlabError("ranksum: each sample needs at least one value that is not NaN.")
    ranks = tied_rank(np.concatenate([x, y]))
    w = float(ranks[:nx].sum())
    method = _method_opt(opts, "exact" if (min(nx, ny) < 10 and nx + ny < 20) else "approximate")
    stats = {"ranksum": np.array([[w]])}
    if method == "exact":
        left, right = _ranksum_exact(ranks, nx, w)
        p = min(1.0, 2 * min(left, right)) if tail == "both" else (right if tail == "right" else left)
    else:
        n = nx + ny
        _, cnt = np.unique(ranks, return_counts=True)
        tie = np.sum(cnt ** 3 - cnt)
        mean = nx * (n + 1) / 2.0
        sd = np.sqrt(nx * ny / 12.0 * ((n + 1) - tie / (n * (n - 1))))
        cc = 0.5 * np.sign(w - mean) if tail == "both" else (0.5 if tail == "right" else -0.5)
        z = (w - mean - cc) / sd if sd > 0 else 0.0
        p = float(min(1.0, _norm_p(z, tail)))
        stats["zval"] = np.array([[float(z)]])
    return outs(nargout, float(p), float(p <= alpha), Struct.scalar(stats))


# -- distribution tests ------------------------------------------------------------------------------------------

def _cdf_values(interp, spec: Any, x: np.ndarray, fname: str) -> np.ndarray:
    """Hypothesised cdf at x: a distribution object, a two-column matrix, a function handle or a cell."""
    from ..values import FunctionHandle
    if hasattr(spec, "cdf") and hasattr(spec, "class_name"):
        return np.asarray(spec.cdf(x), dtype=float)
    if isinstance(spec, FunctionHandle):
        return call_num(interp, spec, x.reshape(-1, 1)).flatten()
    if isinstance(spec, Cell):
        items = spec.values_f()
        return call_num(interp, items[0], x.reshape(-1, 1), *items[1:]).flatten()
    tab = np.asarray(fnum(spec), dtype=float)
    if tab.ndim != 2 or tab.shape[1] != 2:
        raise MatlabError(f"{fname}: CDF must be a distribution object, a function handle or a two-column matrix.")
    order = np.argsort(tab[:, 0], kind="mergesort")
    xs, fs = tab[order, 0], tab[order, 1]
    if np.any(x < xs[0]) or np.any(x > xs[-1]):
        raise MatlabError(f"{fname}: the CDF matrix must cover the range of the data.")
    idx = np.searchsorted(xs, x, side="right") - 1
    exact = xs[np.clip(idx, 0, xs.size - 1)] == x
    return np.where(exact, fs[np.clip(idx, 0, xs.size - 1)], np.interp(x, xs, fs))


def _ks_stats(x: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """One-sided statistics: sup(Fn - F) and sup(F - Fn) for sorted x."""
    n = x.size
    # with ties the empirical cdf jumps once per distinct value
    ux, first, counts = np.unique(x, return_index=True, return_counts=True)
    Fu = F[first]
    hi = np.cumsum(counts) / n
    lo = hi - counts / n
    return float(np.max(hi - Fu)), float(np.max(Fu - lo))


@builtin("kstest")
def _kstest(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("Not enough input arguments for kstest.")
    pos, opts = get_opts(args, 1, ["CDF", "Alpha", "Tail"], "kstest")
    if len(pos) > 4:
        raise MatlabError("Too many input arguments for kstest.")
    legacy = dict(zip(("cdf", "alpha", "tail"), pos[1:]))
    for k, v in legacy.items():
        if not is_empty_arg(v):
            opts.setdefault(k, v)
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    x = np.sort(vec(pos[0]))
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        raise MatlabError("kstest: the sample is empty.")
    F = st.norm.cdf(x) if "cdf" not in opts else _cdf_values(interp, opts["cdf"], x, "kstest")
    dplus, dminus = _ks_stats(x, F)
    if tail == "both":
        d = max(dplus, dminus)
        p = float(st.kstwo.sf(d, n))
        cv = float(st.kstwo.isf(alpha, n))
    else:
        d = dplus if tail == "right" else dminus
        p = float(st.ksone.sf(d, n))
        cv = float(st.ksone.isf(alpha, n))
    return outs(nargout, float(p <= alpha), min(max(p, 0.0), 1.0), d, cv)


@builtin("kstest2")
def _kstest2(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for kstest2.")
    pos, opts = get_opts(args, 2, ["Alpha", "Tail"], "kstest2")
    if len(pos) > 4:
        raise MatlabError("Too many input arguments for kstest2.")
    for k, v in zip(("alpha", "tail"), pos[2:]):
        if not is_empty_arg(v):
            opts.setdefault(k, v)
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    x1, x2 = vec(pos[0]), vec(pos[1])
    x1, x2 = np.sort(x1[~np.isnan(x1)]), np.sort(x2[~np.isnan(x2)])
    n1, n2 = x1.size, x2.size
    if n1 == 0 or n2 == 0:
        raise MatlabError("kstest2: each sample needs at least one value that is not NaN.")
    pts = np.unique(np.concatenate([x1, x2]))
    F1 = np.searchsorted(x1, pts, side="right") / n1
    F2 = np.searchsorted(x2, pts, side="right") / n2
    if tail == "both":
        d = float(np.max(np.abs(F1 - F2)))
    elif tail == "right":
        d = float(np.max(F1 - F2))   # F1 is larger than F2
    else:
        d = float(np.max(F2 - F1))
    n = n1 * n2 / (n1 + n2)
    lam = max((np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d, 0.0)
    if tail == "both":
        j = np.arange(1, 102)
        p = float(2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * lam * lam * j * j)))
        p = min(max(p, 0.0), 1.0)
    else:
        p = float(np.exp(-2 * lam * lam))
    return outs(nargout, float(p <= alpha), p, d)


def _mc_generator():
    return np.random.default_rng(271828)


def _ev_fit_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood extreme-value (minimum type) fits of each row, by Newton iteration on sigma."""
    sg = X.std(axis=1, ddof=1) * np.sqrt(6) / np.pi
    xmax = X.max(axis=1, keepdims=True)
    for _ in range(60):
        w = np.exp((X - xmax) / sg[:, None])
        sw = w.sum(axis=1)
        m1 = (X * w).sum(axis=1) / sw
        m2 = (X * X * w).sum(axis=1) / sw
        f = sg + X.mean(axis=1) - m1
        df = 1 + (m2 - m1 * m1) / sg ** 2
        new = np.maximum(sg - f / df, sg / 10)
        done = np.max(np.abs(new - sg) / sg) < 1e-10
        sg = new
        if done:
            break
    w = np.exp((X - xmax) / sg[:, None])
    mu = xmax[:, 0] + sg * np.log(w.mean(axis=1))
    return mu, sg


def _ks_rows(Z: np.ndarray) -> np.ndarray:
    """Two-sided Kolmogorov statistic of each row of sorted cdf values."""
    n = Z.shape[1]
    i = np.arange(1, n + 1)
    return np.maximum(np.max(i / n - Z, axis=1), np.max(Z - (i - 1) / n, axis=1))


def _ad_rows(Z: np.ndarray) -> np.ndarray:
    n = Z.shape[1]
    i = np.arange(1, n + 1)
    Z = np.clip(Z, 1e-300, 1 - 1e-16)
    return -n - np.sum((2 * i - 1) * (np.log(Z) + np.log1p(-Z[:, ::-1])), axis=1) / n


def _null_stats(dist: str, n: int, stat_rows, reps: int) -> np.ndarray:
    """Simulated null distribution of a goodness-of-fit statistic with estimated parameters."""
    rng = _mc_generator()
    if dist == "exp":
        X = np.sort(rng.standard_exponential((reps, n)), axis=1)
        Z = -np.expm1(-X / X.mean(axis=1, keepdims=True))
    elif dist == "ev":
        X = np.sort(np.log(rng.standard_exponential((reps, n))), axis=1)
        mu, sg = _ev_fit_rows(X)
        Z = -np.expm1(-np.exp((X - mu[:, None]) / sg[:, None]))
    else:
        X = np.sort(rng.standard_normal((reps, n)), axis=1)
        from scipy.special import ndtr
        Z = ndtr((X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True))
    return np.sort(stat_rows(Z))


def _mc_p(null: np.ndarray, stat: float) -> float:
    return float((np.sum(null >= stat) + 1) / (null.size + 1))


def _lillie_p_normal(d: float, n: int) -> float:
    """Dallal-Wilkinson approximation with Stephens' formula for the upper range of p."""
    if n <= 100:
        kd, nd = d, float(n)
    else:
        kd, nd = d * (n / 100.0) ** 0.49, 100.0
    p = np.exp(-7.01256 * kd ** 2 * (nd + 2.78019) + 2.99587 * kd * np.sqrt(nd + 2.78019) - 0.122119
               + 0.974598 / np.sqrt(nd) + 1.67997 / nd)
    if p > 0.1:
        kk = (np.sqrt(n) - 0.01 + 0.85 / np.sqrt(n)) * d
        if kk <= 0.302:
            p = 1.0
        elif kk <= 0.5:
            p = 2.76773 - 19.828315 * kk + 80.709644 * kk ** 2 - 138.55152 * kk ** 3 + 81.218052 * kk ** 4
        elif kk <= 0.9:
            p = -4.901232 + 40.662806 * kk - 97.490286 * kk ** 2 + 94.029866 * kk ** 3 - 32.355711 * kk ** 4
        elif kk <= 1.31:
            p = 6.198765 - 19.558097 * kk + 23.186922 * kk ** 2 - 12.234627 * kk ** 3 + 2.423045 * kk ** 4
        else:
            p = 0.0
    return float(min(max(p, 0.0), 1.0))


_GOF_DIST = {"normal": "norm", "norm": "norm", "exponential": "exp", "exp": "exp", "extreme value": "ev", "ev": "ev",
             "extremevalue": "ev", "lognormal": "logn", "logn": "logn", "weibull": "weibull", "wbl": "weibull"}


def _fitted_cdf(dist: str, x: np.ndarray) -> np.ndarray:
    """cdf of the fitted family at sorted x."""
    from scipy.special import ndtr
    if dist == "norm":
        return ndtr((x - x.mean()) / x.std(ddof=1))
    if dist == "exp":
        return -np.expm1(-x / x.mean())
    mu, sg = _ev_fit_rows(x.reshape(1, -1))
    return -np.expm1(-np.exp((x - mu[0]) / sg[0]))


@builtin("lillietest")
def _lillietest(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for lillietest.")
    pos, opts = get_opts(args, 1, ["Alpha", "Distribution", "MCTol"], "lillietest")
    if len(pos) > 3:
        raise MatlabError("Too many input arguments for lillietest.")
    for k, v in zip(("alpha", "distribution"), pos[1:]):
        if not is_empty_arg(v):
            opts.setdefault(k, v)
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    name = text(opts["distribution"]).lower() if "distribution" in opts else "normal"
    if name not in _GOF_DIST or _GOF_DIST[name] not in ("norm", "exp", "ev"):
        raise MatlabError("lillietest: Distribution must be 'normal', 'exponential' or 'extreme value'.")
    dist = _GOF_DIST[name]
    x = np.sort(vec(pos[0]))
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        raise MatlabError("lillietest: the sample must have at least 4 values that are not NaN.")
    if dist == "exp" and np.any(x < 0):
        raise MatlabError("lillietest: data for the exponential distribution must not be negative.")
    dplus, dminus = _ks_stats(x, _fitted_cdf(dist, x))
    d = max(dplus, dminus)
    if dist == "norm" and "mctol" not in opts:
        p = _lillie_p_normal(d, n)
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _lillie_p_normal(mid, n) > alpha:
                lo = mid
            else:
                hi = mid
        cv = 0.5 * (lo + hi)
    else:
        reps = 20000 if "mctol" in opts else 5000
        null = _null_stats(dist, n, _ks_rows, reps)
        p = _mc_p(null, d)
        cv = float(np.quantile(null, 1 - alpha))
    if "mctol" not in opts:
        p = min(max(p, 0.001), 0.5)
    return outs(nargout, float(p < alpha), p, d, cv)


@builtin("jbtest")
def _jbtest(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("jbtest: expected jbtest(x), jbtest(x, alpha) or jbtest(x, alpha, mctol).")
    x = vec(args[0])
    x = x[~np.isnan(x)]
    n = x.size
    alpha = scalar(args[1], "alpha") if given(args, 1) else 0.05
    if n < 2:
        raise MatlabError("jbtest: the sample must have at least two values that are not NaN.")
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    with np.errstate(all="ignore"):
        s = np.mean(d ** 3) / m2 ** 1.5
        k = np.mean(d ** 4) / m2 ** 2
    jb = float(n * (s ** 2 / 6 + (k - 3) ** 2 / 24))
    if n >= 2000 and not given(args, 2):
        p = float(st.chi2.sf(jb, 2))
        cv = float(st.chi2.isf(alpha, 2))
    else:
        # Small-sample null distribution by simulation (the chi-square limit is reached slowly).
        reps = int(min(40000, max(4000, 4e7 // n)))
        rng = _mc_generator()
        Z = rng.standard_normal((reps, n))
        D = Z - Z.mean(axis=1, keepdims=True)
        v2 = np.mean(D ** 2, axis=1)
        null = n * ((np.mean(D ** 3, axis=1) / v2 ** 1.5) ** 2 / 6 + (np.mean(D ** 4, axis=1) / v2 ** 2 - 3) ** 2 / 24)
        p = _mc_p(null, jb)
        cv = float(np.quantile(null, 1 - alpha))
        if not given(args, 2):
            p = min(max(p, 0.001), 0.5)
    return outs(nargout, float(p < alpha), p, jb, cv)


def _ad_inf(z: float) -> float:
    """Limiting cdf of the Anderson-Darling statistic for a fully specified distribution."""
    if z <= 0:
        return 0.0
    if z < 2:
        return float(z ** -0.5 * np.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (
            0.011672 - 0.00168691 * z) * z) * z) * z) * z))
    return float(np.exp(-np.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z)
            * z)))


def _ad_p_normal(a: float, n: int) -> float:
    s = a * (1 + 0.75 / n + 2.25 / n ** 2)
    if s >= 0.6:
        p = np.exp(1.2937 - 5.709 * s + 0.0186 * s * s)
    elif s >= 0.34:
        p = np.exp(0.9177 - 4.279 * s - 1.38 * s * s)
    elif s > 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * s - 59.938 * s * s)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * s - 223.73 * s * s)
    return float(min(max(p, 0.0), 1.0))


def _ad_p_exp(a: float, n: int) -> float:
    s = a * (1 + 0.6 / n)
    if s > 0.95:
        p = np.exp(0.731 - 3.009 * s + 0.15 * s * s)
    elif s > 0.51:
        p = np.exp(0.9209 - 3.353 * s + 0.3 * s * s)
    elif s > 0.26:
        p = 1 - np.exp(-6.1327 + 20.218 * s - 18.663 * s * s)
    else:
        p = 1 - np.exp(-12.2204 + 67.459 * s - 110.3 * s * s)
    return float(min(max(p, 0.0), 1.0))


def _solve_cv(pfun, alpha: float) -> float:
    lo, hi = 1e-6, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pfun(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@builtin("adtest")
def _adtest(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for adtest.")
    pos, opts = get_opts(args, 1, ["Distribution", "Alpha", "MCTol", "Asymptotic"], "adtest")
    if len(pos) != 1:
        raise MatlabError("adtest: expected adtest(x, 'Name', value, ...).")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    x = np.sort(vec(pos[0]))
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        raise MatlabError("adtest: the sample must have at least 4 values that are not NaN.")
    spec = opts.get("distribution", char("norm"))
    if not is_text(spec):  # a fully specified distribution object
        if not hasattr(spec, "cdf"):
            raise MatlabError("adtest: Distribution must be a name or a probability distribution object.")
        a = float(_ad_rows(np.asarray(spec.cdf(x), dtype=float).reshape(1, -1))[0])
        p = 1 - _ad_inf(a)
        cv = _solve_cv(lambda t: 1 - _ad_inf(t), alpha)
        return outs(nargout, float(p < alpha), p, a, cv)
    name = text(spec).lower()
    if name not in _GOF_DIST:
        raise MatlabError("adtest: Distribution must be 'norm', 'exp', 'ev', 'logn' or 'weibull'.")
    dist = _GOF_DIST[name]
    if dist in ("logn", "weibull"):
        if np.any(x <= 0):
            raise MatlabError("adtest: the data must be positive for this distribution.")
        x = np.log(x)
        dist = "norm" if dist == "logn" else "ev"
    if dist == "exp" and np.any(x < 0):
        raise MatlabError("adtest: data for the exponential distribution must not be negative.")
    a = float(_ad_rows(_fitted_cdf(dist, x).reshape(1, -1))[0])
    asym = as_bool(opts["asymptotic"], "Asymptotic") if "asymptotic" in opts else False
    if asym and "mctol" in opts:
        raise MatlabError("adtest: 'Asymptotic' and 'MCTol' cannot be used together.")
    if dist in ("norm", "exp") and "mctol" not in opts:
        pf = _ad_p_normal if dist == "norm" else _ad_p_exp
        nn = 10 ** 9 if asym else n
        p = min(max(pf(a, nn), 0.0005), 0.99)   # the tabulated range of the small-sample distribution
        cv = _solve_cv(lambda t: pf(t, nn), alpha)
    else:
        null = _null_stats(dist, n, _ad_rows, 20000 if "mctol" in opts else 5000)
        p = _mc_p(null, a)
        cv = float(np.quantile(null, 1 - alpha))
    return outs(nargout, float(p < alpha), p, a, cv)


@builtin("chi2gof")
def _chi2gof(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("Not enough input arguments for chi2gof.")
    pos, opts = get_opts(args, 1, ["NBins", "Ctrs", "Edges", "CDF", "Expected", "NParams", "EMin", "Frequency",
                                   "Alpha"], "chi2gof")
    if len(pos) != 1:
        raise MatlabError("chi2gof: expected chi2gof(x, 'Name', value, ...).")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    x = vec(pos[0])
    freq = vec(opts["frequency"]) if "frequency" in opts else np.ones(x.size)
    if freq.size != x.size:
        raise MatlabError("chi2gof: Frequency must have one value per element of x.")
    keep = ~np.isnan(x) & ~np.isnan(freq)
    x, freq = x[keep], freq[keep]
    if sum(k in opts for k in ("nbins", "ctrs", "edges")) > 1:
        raise MatlabError("chi2gof: give only one of NBins, Ctrs and Edges.")
    if "cdf" in opts and "expected" in opts:
        raise MatlabError("chi2gof: give only one of CDF and Expected.")
    emin = scalar(opts["emin"], "EMin") if "emin" in opts else 5.0
    if x.size == 0:
        raise MatlabError("chi2gof: the sample is empty.")
    lo, hi = x.min(), x.max()
    if "edges" in opts:
        edges = vec(opts["edges"])
    elif "ctrs" in opts:
        c = vec(opts["ctrs"])
        if c.size < 2:
            raise MatlabError("chi2gof: Ctrs needs at least two bin centres.")
        mid = (c[:-1] + c[1:]) / 2
        edges = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
    else:
        nb = intval(opts["nbins"], "NBins") if "nbins" in opts else 10
        edges = np.linspace(lo, hi, nb + 1) if hi > lo else np.array([lo - 0.5, lo + 0.5])
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise MatlabError("chi2gof: the bin edges must be increasing.")
    nb = edges.size - 1
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    inside = (x >= edges[0]) & (x <= edges[-1])
    obs = np.bincount(idx[inside], weights=freq[inside], minlength=nb).astype(float)
    total = float(freq.sum())
    if "expected" in opts:
        exp = vec(opts["expected"])
        if exp.size != nb:
            raise MatlabError(f"chi2gof: Expected must have one value per bin ({nb}).")
        nparams = intval(opts["nparams"], "NParams") if "nparams" in opts else 0
    else:
        if "cdf" in opts:
            spec = opts["cdf"]
            F = _cdf_values(interp, spec, edges, "chi2gof")
            if "nparams" in opts:
                nparams = intval(opts["nparams"], "NParams")
            elif isinstance(spec, Cell):
                nparams = len(spec.values_f()) - 1
            elif getattr(spec, "data", None) is not None and hasattr(spec, "fam"):
                nparams = len([i for i in spec.fam.est if i not in spec.fixed])
            else:
                nparams = 0
        else:
            mu = float(np.sum(freq * x) / total)
            sd = float(np.sqrt(np.sum(freq * (x - mu) ** 2) / (total - 1))) if total > 1 else 0.0
            F = st.norm.cdf(edges, mu, sd) if sd > 0 else (edges >= mu).astype(float)
            nparams = intval(opts["nparams"], "NParams") if "nparams" in opts else 2
        F = np.asarray(F, dtype=float).copy()
        F[0], F[-1] = 0.0, 1.0           # the end bins take the tails of the distribution
        exp = total * np.diff(F)
    # Pool the end bins until both have an expected count of at least emin.
    i, j = 0, nb - 1
    exp, obs, edges = exp.copy(), obs.copy(), edges.copy()
    while i < j - 1 and (exp[i] < emin or exp[j] < emin):
        if exp[i] < exp[j]:
            exp[i + 1] += exp[i]
            obs[i + 1] += obs[i]
            i += 1
        else:
            exp[j - 1] += exp[j]
            obs[j - 1] += obs[j]
            j -= 1
    exp, obs = exp[i:j + 1], obs[i:j + 1]
    edges = np.concatenate([[edges[0]], edges[i + 1:j + 1], [edges[-1]]])
    with np.errstate(all="ignore"):
        chi = float(np.sum((obs - exp) ** 2 / exp))
    df = exp.size - 1 - nparams
    if df > 0:
        p = float(st.chi2.sf(chi, df))
    else:
        df, p = 0, np.nan
    stats = struct(chi2stat=chi, df=float(df), edges=edges.reshape(1, -1), O=obs.reshape(1, -1), E=exp.reshape(1, -1))
    h = float(p <= alpha) if not np.isnan(p) else 0.0
    return outs(nargout, h, p, stats)


@builtin("runstest")
def _runstest(interp, args, nargout):
    from scipy.special import comb
    if not args:
        raise MatlabError("Not enough input arguments for runstest.")
    pos, opts = get_opts(args, 1, ["Alpha", "Method", "Tail"], "runstest")
    if len(pos) > 2:
        raise MatlabError("Too many input arguments for runstest.")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    x = vec(pos[0])
    x = x[~np.isnan(x)]
    updown = len(pos) > 1 and is_text(pos[1])
    if updown:
        if text(pos[1]).lower() != "ud":
            raise MatlabError("runstest: the second input must be a reference value or 'ud'.")
        d = np.diff(x)
        s = np.sign(d[d != 0])
    else:
        v = scalar(pos[1], "v") if len(pos) > 1 and not is_empty_arg(pos[1]) else float(x.mean()) if x.size else 0.0
        s = np.sign(x - v)
        s = s[s != 0]
    n1, n0 = int(np.sum(s > 0)), int(np.sum(s < 0))
    n = n1 + n0
    r = int(1 + np.sum(s[1:] != s[:-1])) if n else 0
    stats = {"nruns": np.array([[float(r)]]), "n1": np.array([[float(n1)]]), "n0": np.array([[float(n0)]]),
             "z": np.array([[np.nan]])}
    method = _method_opt(opts, "exact" if n <= 50 else "approximate")
    if n == 0 or (not updown and (n1 == 0 or n0 == 0)):
        return outs(nargout, 0.0, 1.0, Struct.scalar(stats))
    if updown:
        N = n + 1
        mean, var = (2 * N - 1) / 3.0, (16 * N - 29) / 90.0
        if method == "exact" and N <= 9:
            from itertools import permutations
            base = np.array(list(permutations(range(N))))
            sg = np.sign(np.diff(base, axis=1))
            runs = 1 + np.sum(sg[:, 1:] != sg[:, :-1], axis=1)
            left, right = float(np.mean(runs <= r)), float(np.mean(runs >= r))
            p = min(1.0, 2 * min(left, right)) if tail == "both" else (right if tail == "right" else left)
            return outs(nargout, float(p <= alpha), p, Struct.scalar(stats))
        method = "approximate"
    else:
        mean = 2.0 * n1 * n0 / n + 1
        var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n * n * (n - 1.0)) if n > 1 else 0.0
    if method == "exact":
        rs = np.arange(2, n + 1)
        pr = np.zeros(rs.size)
        denom = comb(n, n1)
        for i, rr in enumerate(rs):
            k = rr // 2
            if rr % 2 == 0:
                pr[i] = 2 * comb(n1 - 1, k - 1) * comb(n0 - 1, k - 1) / denom
            else:
                pr[i] = (comb(n1 - 1, k) * comb(n0 - 1, k - 1) + comb(n1 - 1, k - 1) * comb(n0 - 1, k)) / denom
        left, right = float(pr[rs <= r].sum()), float(pr[rs >= r].sum())
        p = min(1.0, 2 * min(left, right)) if tail == "both" else (right if tail == "right" else left)
    else:
        sd = np.sqrt(var)
        cc = 0.5 * np.sign(mean - r) if tail == "both" else (-0.5 if tail == "right" else 0.5)
        z = (r - mean + cc) / sd if sd > 0 else 0.0
        if tail == "both" and abs(r - mean) <= 0.5:
            z = 0.0
        p = float(_norm_p(z, tail))
        stats["z"] = np.array([[float(z)]])
    return outs(nargout, float(p <= alpha), float(p), Struct.scalar(stats))


@builtin("fishertest")
def _fishertest(interp, args, nargout):
    import scipy.stats as st
    if not args:
        raise MatlabError("Not enough input arguments for fishertest.")
    pos, opts = get_opts(args, 1, ["Tail", "Alpha"], "fishertest")
    if len(pos) != 1:
        raise MatlabError("fishertest: expected fishertest(x, 'Name', value, ...).")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    x = pos[0]
    if is_table(x):
        x = np.column_stack([np.asarray(fnum(c), dtype=float) for c in x.columns])
    x = np.asarray(fnum(x), dtype=float)
    if x.shape != (2, 2) or np.any(x < 0) or np.any(x != np.round(x)):
        raise MatlabError("fishertest: x must be a 2-by-2 table of non-negative integer counts.")
    alt = {"both": "two-sided", "right": "greater", "left": "less"}[tail]
    p = float(st.fisher_exact(x.astype(int), alternative=alt)[1])
    a, b, c, d = x[0, 0], x[0, 1], x[1, 0], x[1, 1]
    with np.errstate(all="ignore"):
        odds = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = st.norm.ppf(1 - alpha / 2)
        ci = np.array([[np.exp(np.log(odds) - z * se), np.exp(np.log(odds) + z * se)]])
        rr = (a / (a + b)) / (c / (c + d))
    stats = struct(OddsRatio=float(odds), ConfidenceInterval=ci, RelativeRisk=float(rr))
    return outs(nargout, bool(p <= alpha), p, stats)


# -- analysis of variance -------------------------------------------------------------------------------------------

@builtin("anova1")
def _anova1(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("anova1: expected anova1(y), anova1(y, group) or anova1(y, group, displayopt).")
    display = _display_opt(args[2] if len(args) > 2 else None, "anova1")
    groups, names = _groups_from(args[0], args[1] if len(args) > 1 else None, "anova1")
    if not groups:
        raise MatlabError("anova1: there are no data.")
    allv = np.concatenate(groups)
    grand = allv.mean()
    k, n = len(groups), allv.size
    ssb = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups))
    sse = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    dfb, dfe = k - 1, n - k
    with np.errstate(all="ignore"):
        msb = ssb / dfb if dfb else np.nan
        mse = sse / dfe if dfe else np.nan
        F = msb / mse
    p = float(st.f.sf(F, dfb, dfe)) if np.isfinite(F) else np.nan
    tbl = _anova_cell(["Source", "SS", "df", "MS", "F", "Prob>F"],
                      [["Groups", ssb, dfb, msb, F, p], ["Error", sse, dfe, mse], ["Total", ssb + sse, n - 1]])
    if display:
        _print_table(interp, "ANOVA Table", tbl)
    stats = Struct.scalar({
        "gnames": cellstr(names), "n": np.array([[float(g.size) for g in groups]]), "source": char("anova1"),
        "means": np.array([[g.mean() for g in groups]]), "df": np.array([[float(dfe)]]),
        "s": np.array([[float(np.sqrt(mse))]]),
    })
    return outs(nargout, p, tbl, stats)


@builtin("kruskalwallis")
def _kruskalwallis(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("kruskalwallis: expected kruskalwallis(x), kruskalwallis(x, group) or "
                          "kruskalwallis(x, group, displayopt).")
    display = _display_opt(args[2] if len(args) > 2 else None, "kruskalwallis")
    groups, names = _groups_from(args[0], args[1] if len(args) > 1 else None, "kruskalwallis")
    allv = np.concatenate(groups)
    N = allv.size
    ranks = tied_rank(allv)
    _, cnt = np.unique(allv, return_counts=True)
    sumt = float(np.sum(cnt ** 3 - cnt))
    bounds = np.cumsum([0] + [g.size for g in groups])
    rg = [ranks[bounds[i]:bounds[i + 1]] for i in range(len(groups))]
    grand = (N + 1) / 2.0
    ssb = float(sum(r.size * (r.mean() - grand) ** 2 for r in rg))
    sst = float(np.sum((ranks - grand) ** 2))
    k = len(groups)
    with np.errstate(all="ignore"):
        chi = (N - 1) * ssb / sst if sst > 0 else np.nan
    p = float(st.chi2.sf(chi, k - 1)) if np.isfinite(chi) else np.nan
    tbl = _anova_cell(["Source", "SS", "df", "MS", "Chi-sq", "Prob>Chi-sq"],
                      [["Groups", ssb, k - 1, ssb / (k - 1) if k > 1 else np.nan, chi, p],
                       ["Error", sst - ssb, N - k, (sst - ssb) / (N - k) if N > k else np.nan],
                       ["Total", sst, N - 1]])
    if display:
        _print_table(interp, "Kruskal-Wallis ANOVA Table", tbl)
    stats = Struct.scalar({"gnames": cellstr(names), "n": np.array([[float(g.size) for g in groups]]),
                           "source": char("kruskalwallis"), "meanranks": np.array([[r.mean() for r in rg]]),
                           "sumt": np.array([[sumt]])})
    return outs(nargout, p, tbl, stats)


@builtin("friedman")
def _friedman(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("friedman: expected friedman(x), friedman(x, reps) or friedman(x, reps, displayopt).")
    X = np.asarray(fnum(args[0]), dtype=float)
    reps = intval(args[1], "reps") if given(args, 1) else 1
    display = _display_opt(args[2] if len(args) > 2 else None, "friedman")
    if X.ndim != 2 or X.shape[1] < 2:
        raise MatlabError("friedman: x must be a matrix with at least two columns.")
    if np.isnan(X).any():
        raise MatlabError("friedman: missing values (NaN) are not supported.")
    r, c = X.shape
    if reps < 1 or r % reps:
        raise MatlabError("friedman: the number of rows must be a multiple of reps.")
    blocks = r // reps
    R = np.empty_like(X)
    sumt = 0.0
    for b in range(blocks):
        blk = X[b * reps:(b + 1) * reps, :]
        R[b * reps:(b + 1) * reps, :] = tied_rank(blk.flatten()).reshape(blk.shape)
        _, cnt = np.unique(blk, return_counts=True)
        sumt += float(np.sum(cnt ** 3 - cnt))
    meanranks = R.mean(axis=0)
    m = c * reps                                  # values ranked together in each block
    sigma2 = m * (m + 1) / 12.0 - sumt / (12.0 * blocks * (m - 1)) if m > 1 else 0.0
    grand = (m + 1) / 2.0
    ss_cols = float(r * np.sum((meanranks - grand) ** 2))
    chi = ss_cols / sigma2 if sigma2 > 0 else np.nan
    p = float(st.chi2.sf(chi, c - 1)) if np.isfinite(chi) else np.nan
    ss_total = float(np.sum((R - grand) ** 2))
    df_total = r * c - 1
    df_int = (blocks - 1) * (c - 1) if reps > 1 else 0
    rows = [["Columns", ss_cols, c - 1, ss_cols / (c - 1), chi, p]]
    if reps > 1:
        cell_means = np.array([[R[b * reps:(b + 1) * reps, j].mean() for j in range(c)] for b in range(blocks)])
        ss_cells = float(reps * np.sum((cell_means - grand) ** 2))
        ss_int = ss_cells - ss_cols
        rows.append(["Interaction", ss_int, df_int, ss_int / df_int if df_int else np.nan])
        ss_err = ss_total - ss_cells
        df_err = df_total - (c - 1) - df_int - (blocks - 1) * 0
        df_err = blocks * c * (reps - 1)
    else:
        ss_err = ss_total - ss_cols
        df_err = (blocks - 1) * (c - 1)
    rows.append(["Error", ss_err, df_err, ss_err / df_err if df_err else np.nan])
    rows.append(["Total", ss_total, df_total])
    tbl = _anova_cell(["Source", "SS", "df", "MS", "Chi-sq", "Prob>Chi-sq"], rows)
    if display:
        _print_table(interp, "Friedman's ANOVA Table", tbl)
    stats = Struct.scalar({"source": char("friedman"), "n": np.array([[float(blocks)]]),
                           "meanranks": meanranks.reshape(1, -1), "sigma": np.array([[float(np.sqrt(sigma2))]])})
    return outs(nargout, p, tbl, stats)


@builtin("anova2")
def _anova2(interp, args, nargout):
    import scipy.stats as st
    if not 1 <= len(args) <= 3:
        raise MatlabError("anova2: expected anova2(y), anova2(y, reps) or anova2(y, reps, displayopt).")
    Y = np.asarray(fnum(args[0]), dtype=float)
    reps = intval(args[1], "reps") if given(args, 1) else 1
    display = _display_opt(args[2] if len(args) > 2 else None, "anova2")
    if Y.ndim != 2 or np.isnan(Y).any():
        raise MatlabError("anova2: y must be a matrix without NaN (use anovan for unbalanced designs).")
    r, c = Y.shape
    if reps < 1 or r % reps:
        raise MatlabError("anova2: the number of rows must be a multiple of reps.")
    m = r // reps
    grand = Y.mean()
    colmeans = Y.mean(axis=0)
    cells = Y.reshape(m, reps, c)
    rowmeans = cells.mean(axis=(1, 2))
    ss_col = float(r * np.sum((colmeans - grand) ** 2))
    ss_row = float(reps * c * np.sum((rowmeans - grand) ** 2))
    ss_tot = float(np.sum((Y - grand) ** 2))
    df_col, df_row = c - 1, m - 1
    if reps > 1:
        cm = cells.mean(axis=1)
        ss_cells = float(reps * np.sum((cm - grand) ** 2))
        ss_int = ss_cells - ss_col - ss_row
        df_int = df_col * df_row
        ss_err = ss_tot - ss_cells
        df_err = m * c * (reps - 1)
    else:
        ss_int, df_int = 0.0, 0
        ss_err = ss_tot - ss_col - ss_row
        df_err = df_col * df_row
    with np.errstate(all="ignore"):
        mse = ss_err / df_err if df_err else np.nan
        f_col, f_row = (ss_col / df_col) / mse, (ss_row / df_row) / mse if df_row else np.nan
        p_col = float(st.f.sf(f_col, df_col, df_err))
        p_row = float(st.f.sf(f_row, df_row, df_err)) if df_row else np.nan
    rows = [["Columns", ss_col, df_col, ss_col / df_col, f_col, p_col],
            ["Rows", ss_row, df_row, ss_row / df_row if df_row else np.nan, f_row, p_row]]
    p = [p_col, p_row]
    if reps > 1:
        f_int = (ss_int / df_int) / mse if df_int else np.nan
        p_int = float(st.f.sf(f_int, df_int, df_err)) if df_int else np.nan
        rows.append(["Interaction", ss_int, df_int, ss_int / df_int if df_int else np.nan, f_int, p_int])
        p.append(p_int)
    rows += [["Error", ss_err, df_err, mse], ["Total", ss_tot, r * c - 1]]
    tbl = _anova_cell(["Source", "SS", "df", "MS", "F", "Prob>F"], rows)
    if display:
        _print_table(interp, "ANOVA Table", tbl)
    stats = Struct.scalar({"source": char("anova2"), "sigmasq": np.array([[float(mse)]]),
                           "colmeans": colmeans.reshape(1, -1), "coln": np.array([[float(r)]]),
                           "rowmeans": rowmeans.reshape(1, -1), "rown": np.array([[float(reps * c)]]),
                           "inter": np.array([[float(reps > 1)]]),
                           "pval": np.array([[p[2] if reps > 1 else np.nan]]), "df": np.array([[float(df_err)]])})
    return outs(nargout, np.array([p]), tbl, stats)


def _effects(codes: np.ndarray, k: int) -> np.ndarray:
    """Sum-to-zero coding of a factor with k levels (k - 1 columns)."""
    D = np.zeros((codes.size, k - 1))
    for j in range(k - 1):
        D[codes == j, j] = 1.0
    D[codes == k - 1, :] = -1.0
    return D


def _term_columns(term: np.ndarray, mains: list[np.ndarray]) -> np.ndarray:
    cols = np.ones((mains[0].shape[0], 1))
    for j in np.flatnonzero(term):
        cols = (cols[:, :, None] * mains[j][:, None, :]).reshape(cols.shape[0], -1)
    return cols


def _model_terms(spec: Any, nf: int) -> np.ndarray:
    eye = np.eye(nf, dtype=int)
    if spec is None or is_empty_arg(spec):
        return eye
    if is_text(spec):
        kind = choice(spec, ["linear", "interaction", "full"], "model")
        order = {"linear": 1, "interaction": 2, "full": nf}[kind]
    else:
        m = np.asarray(fnum(spec))
        if m.size == 1:
            order = int(m.flat[0])
        else:
            m = np.atleast_2d(m).astype(int)
            if m.shape[1] != nf:
                raise MatlabError(f"anovan: a terms matrix must have one column per factor ({nf}).")
            return m
    from itertools import combinations
    rows = []
    for o in range(1, min(order, nf) + 1):
        for c in combinations(range(nf), o):
            t = np.zeros(nf, dtype=int)
            t[list(c)] = 1
            rows.append(t)
    return np.array(rows)


def _rss(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    if X.shape[1] == 0:
        return float(y @ y), 0
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return float(res @ res), int(rank)


@builtin("anovan")
def _anovan(interp, args, nargout):
    import scipy.stats as st
    if len(args) < 2:
        raise MatlabError("anovan: expected anovan(y, group, 'Name', value, ...).")
    pos, opts = get_opts(args, 2, ["model", "sstype", "varnames", "display", "continuous", "random", "nested",
                                   "alpha"], "anovan")
    if len(pos) != 2:
        raise MatlabError("anovan: expected anovan(y, group, 'Name', value, ...).")
    for o in ("random", "nested"):
        if o in opts and not is_empty_arg(opts[o]):
            raise MatlabError(f"anovan: the option '{o}' is not supported.")
    y = vec(pos[0])
    gl = group_list(pos[1])
    if isinstance(pos[1], np.ndarray) and pos[1].ndim == 2 and pos[1].shape[0] == y.size and pos[1].shape[1] > 1 \
            and pos[1].dtype.kind != "U":
        gl = [pos[1][:, j] for j in range(pos[1].shape[1])]
    nf = len(gl)
    cont = {int(i) - 1 for i in vec(opts["continuous"])} if "continuous" in opts else set()
    sstype = intval(opts["sstype"], "sstype") if "sstype" in opts else 3
    if sstype not in (1, 2, 3):
        raise MatlabError("anovan: sstype must be 1, 2 or 3.")
    display = _display_opt(opts.get("display"), "anovan")
    names = text_list(opts["varnames"]) if "varnames" in opts else [f"X{j + 1}" for j in range(nf)]
    if names is None or len(names) != nf:
        raise MatlabError(f"anovan: varnames must list {nf} names.")
    ok = ~np.isnan(y)
    mains, levels, codes_all = [], [], []
    for j, g in enumerate(gl):
        if j in cont:
            v = vec(g)
            if v.size != y.size:
                raise MatlabError("anovan: each grouping variable must have one value per observation.")
            ok &= ~np.isnan(v)
            codes_all.append(v)
            levels.append([])
        else:
            c, nm, _ = group_codes(g)
            if c.size != y.size:
                raise MatlabError("anovan: each grouping variable must have one value per observation.")
            ok &= c >= 0
            codes_all.append(c)
            levels.append(nm)
    y = y[ok]
    n = y.size
    nlev = []
    for j in range(nf):
        if j in cont:
            mains.append(codes_all[j][ok].reshape(-1, 1).astype(float))
            nlev.append(1)
        else:
            c = codes_all[j][ok]
            used = np.unique(c)
            remap = {int(u): i for i, u in enumerate(used)}
            c = np.array([remap[int(v)] for v in c])
            levels[j] = [levels[j][int(u)] for u in used]
            codes_all[j] = c
            nlev.append(len(used))
            mains.append(_effects(c, len(used)))
    terms = _model_terms(opts.get("model"), nf)
    blocks = [_term_columns(t, mains) for t in terms]
    ones = np.ones((n, 1))
    full = np.hstack([ones] + blocks)
    rss_full, rank_full = _rss(full, y)
    dfe = n - rank_full
    mse = rss_full / dfe if dfe > 0 else np.nan
    ss, dfs = [], []
    for i, t in enumerate(terms):
        if sstype == 1:
            small = np.hstack([ones] + blocks[:i])
            big = np.hstack([ones] + blocks[:i + 1])
        elif sstype == 2:
            # compare the models with and without the term, leaving out everything that contains it
            keep = [k for k, u in enumerate(terms) if not (np.all(u >= t) and k != i)]
            big = np.hstack([ones] + [blocks[k] for k in keep])
            small = np.hstack([ones] + [blocks[k] for k in keep if k != i])
        else:
            big = full
            small = np.hstack([ones] + [blocks[k] for k in range(len(terms)) if k != i])
        r_small, k_small = _rss(small, y)
        r_big, k_big = _rss(big, y)
        ss.append(max(r_small - r_big, 0.0))
        dfs.append(k_big - k_small)
    p, rows = [], []
    tnames = []
    for i, t in enumerate(terms):
        nm = "*".join(names[j] for j in np.flatnonzero(t))
        tnames.append(nm)
        with np.errstate(all="ignore"):
            ms = ss[i] / dfs[i] if dfs[i] else np.nan
            F = ms / mse
            pv = float(st.f.sf(F, dfs[i], dfe)) if dfs[i] and dfe > 0 else np.nan
        p.append(pv)
        rows.append([nm, ss[i], dfs[i], ms, F, pv])
    sst = float(np.sum((y - y.mean()) ** 2))
    rows += [["Error", rss_full, dfe, mse], ["Total", sst, n - 1]]
    tbl = _anova_cell(["Source", "Sum Sq.", "d.f.", "Mean Sq.", "F", "Prob>F"], rows)
    if display:
        _print_table(interp, "Analysis of Variance", tbl)
    coef, *_ = np.linalg.lstsq(full, y, rcond=None)
    stats = Struct.scalar({
        "source": char("anovan"), "resid": (y - full @ coef).reshape(-1, 1), "coeffs": coef.reshape(-1, 1),
        "nlevels": np.array(nlev, dtype=float).reshape(-1, 1), "terms": terms.astype(float),
        "varnames": cellstr(names), "grpnames": Cell.from_list([cellstr(lv) for lv in levels], column=True),
        "termnames": cellstr(tnames), "dfe": np.array([[float(dfe)]]), "mse": np.array([[float(mse)]]),
        "continuous": np.array([[float(j in cont) for j in range(nf)]]),
        "y": y.reshape(-1, 1),
        "groups": np.column_stack([np.asarray(codes_all[j] if j not in cont else codes_all[j][ok], dtype=float) + (
            0 if j in cont else 1) for j in range(nf)]),
    })
    return outs(nargout, np.array(p).reshape(-1, 1), tbl, stats, terms.astype(float))


# -- multiple comparisons ----------------------------------------------------------------------------------------------

def _crit_and_p(ctype: str, k: int, df: float, alpha: float, tstat: np.ndarray, ncomp: int):
    """Critical value for |difference| / se and adjusted p-values for the pairwise statistics."""
    import scipy.stats as st
    inf = not np.isfinite(df)
    tdist_ppf = (lambda q: st.norm.ppf(q)) if inf else (lambda q: st.t.ppf(q, df))
    tdist_sf = (lambda t: st.norm.sf(t)) if inf else (lambda t: st.t.sf(t, df))
    if ctype in ("tukey-kramer", "hsd"):
        dfq = np.inf if inf else df
        crit = st.studentized_range.ppf(1 - alpha, k, dfq) / np.sqrt(2)
        pv = st.studentized_range.sf(np.abs(tstat) * np.sqrt(2), k, dfq)
    elif ctype == "bonferroni":
        crit = tdist_ppf(1 - alpha / (2 * ncomp))
        pv = np.minimum(1.0, 2 * ncomp * tdist_sf(np.abs(tstat)))
    elif ctype == "dunn-sidak":
        crit = tdist_ppf(1 - (1 - (1 - alpha) ** (1.0 / ncomp)) / 2)
        pv = 1 - (1 - np.minimum(1.0, 2 * tdist_sf(np.abs(tstat)))) ** ncomp
    elif ctype == "lsd":
        crit = tdist_ppf(1 - alpha / 2)
        pv = 2 * tdist_sf(np.abs(tstat))
    else:  # scheffe
        if inf:
            crit = np.sqrt(st.chi2.ppf(1 - alpha, k - 1))
            pv = st.chi2.sf(tstat ** 2, k - 1)
        else:
            crit = np.sqrt((k - 1) * st.f.ppf(1 - alpha, k - 1, df))
            pv = st.f.sf(tstat ** 2 / (k - 1), k - 1, df)
    return float(crit), np.asarray(pv, dtype=float)


def _sfield(s: Struct, name: str) -> Any:
    if name not in s.fields:
        raise MatlabError(f"multcompare: the stats structure has no field '{name}'.")
    return s.get_field(0, name)


@builtin("multcompare")
def _multcompare(interp, args, nargout):
    if not args or not isinstance(args[0], Struct):
        raise MatlabError("multcompare: the first input must be the stats output of anova1, anova2, anovan, "
                          "kruskalwallis or friedman.")
    s = args[0]
    _, opts = get_opts(args, 1, ["Alpha", "CType", "CriticalValueType", "Display", "Dimension", "Estimate"],
                       "multcompare")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    ct = opts.get("criticalvaluetype", opts.get("ctype"))
    ctype = choice(ct, ["tukey-kramer", "hsd", "bonferroni", "dunn-sidak", "lsd", "scheffe"], "CriticalValueType") \
        if ct is not None else "tukey-kramer"
    display = _display_opt(opts.get("display"), "multcompare")
    source = text(_sfield(s, "source"))
    if source not in ("anovan",) and "dimension" in opts:
        raise MatlabError("multcompare: 'Dimension' applies to anovan results only.")
    if source != "anova2" and "estimate" in opts:
        raise MatlabError("multcompare: 'Estimate' applies to anova2 results only.")
    if source == "anova1":
        means = vec(_sfield(s, "means"))
        n = vec(_sfield(s, "n"))
        df = float(vec(_sfield(s, "df"))[0])
        var = float(vec(_sfield(s, "s"))[0]) ** 2 / n
        names = text_list(_sfield(s, "gnames"))
    elif source == "kruskalwallis":
        means = vec(_sfield(s, "meanranks"))
        n = vec(_sfield(s, "n"))
        N = n.sum()
        sumt = float(vec(_sfield(s, "sumt"))[0])
        var = (N * (N + 1) / 12.0 - sumt / (12.0 * (N - 1))) / n
        df = np.inf
        names = text_list(_sfield(s, "gnames"))
    elif source == "friedman":
        means = vec(_sfield(s, "meanranks"))
        nb = float(vec(_sfield(s, "n"))[0])
        var = np.full(means.size, float(vec(_sfield(s, "sigma"))[0]) ** 2 / nb)
        df = np.inf
        names = [str(i + 1) for i in range(means.size)]
    elif source == "anova2":
        est = choice(opts["estimate"], ["column", "row"], "Estimate") if "estimate" in opts else "column"
        means = vec(_sfield(s, "colmeans" if est == "column" else "rowmeans"))
        cnt = float(vec(_sfield(s, "coln" if est == "column" else "rown"))[0])
        var = np.full(means.size, float(vec(_sfield(s, "sigmasq"))[0]) / cnt)
        df = float(vec(_sfield(s, "df"))[0])
        names = [str(i + 1) for i in range(means.size)]
    elif source == "anovan":
        means, var, names, df = _anovan_margins(s, opts)
    else:
        raise MatlabError(f"multcompare: results from '{source}' are not supported.")
    k = means.size
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if not pairs:
        raise MatlabError("multcompare: at least two groups are needed.")
    diff = np.array([means[i] - means[j] for i, j in pairs])
    if isinstance(var, np.ndarray) and var.ndim == 2:
        se = np.array([np.sqrt(var[i, i] + var[j, j] - 2 * var[i, j]) for i, j in pairs])
        gse = np.sqrt(np.diag(var))
    else:
        se = np.array([np.sqrt(var[i] + var[j]) for i, j in pairs])
        gse = np.sqrt(var)
    with np.errstate(all="ignore"):
        tstat = diff / se
    crit, pv = _crit_and_p(ctype, k, df, alpha, tstat, len(pairs))
    c = np.column_stack([[i + 1.0 for i, _ in pairs], [j + 1.0 for _, j in pairs], diff - crit * se, diff,
                         diff + crit * se, pv])
    m = np.column_stack([means, gse])
    h = None
    if display:
        builtin_call(interp, "figure", [], 0)
        ypos = np.arange(k, 0, -1.0)
        half = crit * gse / np.sqrt(2)
        hh = builtin_call(interp, "errorbar", [means.reshape(1, -1), ypos.reshape(1, -1), half.reshape(1, -1),
                                               char("horizontal"), char("o")], 1)
        gca = builtin_call(interp, "gca", [], 1)[0]
        builtin_call(interp, "set", [gca, char("YTick"), np.arange(1.0, k + 1).reshape(1, -1), char("YTickLabel"),
                                     cellstr(names[::-1], column=False)], 0)
        builtin_call(interp, "title", [char("Group estimates with comparison intervals")], 0)
        h = builtin_call(interp, "gcf", [], 1)[0]
        del hh
    return outs(nargout, c, m, h if h is not None else np.zeros((0, 0)), cellstr(names))


def _anovan_margins(s: Struct, opts: dict):
    """Population marginal means of an anovan fit over the chosen dimension(s)."""
    nlev = [int(v) for v in vec(_sfield(s, "nlevels"))]
    terms = np.asarray(fnum(_sfield(s, "terms"))).astype(int)
    cont = vec(_sfield(s, "continuous")) != 0
    G = np.asarray(fnum(_sfield(s, "groups")), dtype=float)
    y = vec(_sfield(s, "y"))
    dims = [int(d) - 1 for d in vec(opts["dimension"])] if "dimension" in opts else [0]
    if any(cont[d] for d in dims):
        raise MatlabError("multcompare: the dimension must be a categorical factor.")
    nf = len(nlev)
    names_all = [text_list(c) for c in _sfield(s, "grpnames").values_f()]
    vnames = text_list(_sfield(s, "varnames"))
    mains = []
    for j in range(nf):
        mains.append(G[:, j].reshape(-1, 1) if cont[j] else _effects(G[:, j].astype(int) - 1, nlev[j]))
    X = np.hstack([np.ones((y.size, 1))] + [_term_columns(t, mains) for t in terms])
    XtXi = np.linalg.pinv(X.T @ X)
    coef = XtXi @ X.T @ y
    dfe = float(vec(_sfield(s, "dfe"))[0])
    mse = float(vec(_sfield(s, "mse"))[0])
    from itertools import product
    combos = list(product(*[range(nlev[d]) for d in dims]))
    others = [j for j in range(nf) if j not in dims and not cont[j]]
    L = []
    for combo in combos:
        rows = []
        for other in product(*[range(nlev[j]) for j in others]):
            m_rows = []
            for j in range(nf):
                if cont[j]:
                    m_rows.append(np.array([[G[:, j].mean()]]))
                elif j in dims:
                    m_rows.append(_effects(np.array([combo[dims.index(j)]]), nlev[j]))
                else:
                    m_rows.append(_effects(np.array([other[others.index(j)]]), nlev[j]))
            rows.append(np.hstack([np.ones((1, 1))] + [_term_columns(t, m_rows) for t in terms])[0])
        L.append(np.mean(rows, axis=0))
    L = np.array(L)
    means = L @ coef
    cov = mse * (L @ XtXi @ L.T)
    names = [",".join(f"{vnames[d]}={names_all[d][c]}" for d, c in zip(dims, combo)) for combo in combos]
    return means, cov, names, dfe


# -- power and sample size ---------------------------------------------------------------------------------------------

def _power_z(p0, p1, n, alpha, tail):
    import scipy.stats as st
    mu0, sd = p0
    d = (p1 - mu0) / (sd / np.sqrt(n))
    if tail == "both":
        c = st.norm.ppf(1 - alpha / 2)
        return st.norm.sf(c - d) + st.norm.cdf(-c - d)
    c = st.norm.ppf(1 - alpha)
    return st.norm.sf(c - d) if tail == "right" else st.norm.cdf(-c - d)


def _power_t(p0, p1, n, alpha, tail, ratio=None):
    import scipy.stats as st
    mu0, sd = p0
    if ratio is None:
        df, nc = n - 1, (p1 - mu0) / (sd / np.sqrt(n))
    else:
        n2 = ratio * n
        df, nc = n + n2 - 2, (p1 - mu0) / (sd * np.sqrt(1 / n + 1 / n2))
    if df < 1:
        return alpha
    if tail == "both":
        c = st.t.ppf(1 - alpha / 2, df)
        return st.nct.sf(c, df, nc) + st.nct.cdf(-c, df, nc)
    c = st.t.ppf(1 - alpha, df)
    return st.nct.sf(c, df, nc) if tail == "right" else st.nct.cdf(-c, df, nc)


def _power_var(p0, p1, n, alpha, tail):
    import scipy.stats as st
    df = n - 1
    if df < 1:
        return alpha
    r = p0 / p1
    if tail == "both":
        return st.chi2.sf(st.chi2.ppf(1 - alpha / 2, df) * r, df) + st.chi2.cdf(st.chi2.ppf(alpha / 2, df) * r, df)
    if tail == "right":
        return st.chi2.sf(st.chi2.ppf(1 - alpha, df) * r, df)
    return st.chi2.cdf(st.chi2.ppf(alpha, df) * r, df)


def _power_p(p0, p1, n, alpha, tail):
    """Exact power of the binomial test of p = p0 against p1 (rejection region chosen at level alpha)."""
    import scipy.stats as st
    n = int(n)
    k = np.arange(n + 1)
    if tail == "both":
        lo = st.binom.ppf(alpha / 2, n, p0) - 1
        while st.binom.cdf(lo + 1, n, p0) <= alpha / 2:
            lo += 1
        hi = st.binom.isf(alpha / 2, n, p0) + 1
        while st.binom.sf(hi - 2, n, p0) <= alpha / 2:
            hi -= 1
        return float(st.binom.cdf(lo, n, p1) + st.binom.sf(hi - 1, n, p1))
    if tail == "right":
        crit = k[st.binom.sf(k - 1, n, p0) <= alpha]
        return float(st.binom.sf(crit[0] - 1, n, p1)) if crit.size else 0.0
    crit = k[st.binom.cdf(k, n, p0) <= alpha]
    return float(st.binom.cdf(crit[-1], n, p1)) if crit.size else 0.0


def _power_p2(p0, p1, n, alpha, tail, ratio):
    import scipy.stats as st
    n2 = ratio * n
    pbar = (p0 * n + p1 * n2) / (n + n2)
    se0 = np.sqrt(pbar * (1 - pbar) * (1 / n + 1 / n2))
    se1 = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n2)
    d = p1 - p0
    if tail == "both":
        c = st.norm.ppf(1 - alpha / 2)
        return st.norm.sf((c * se0 - d) / se1) + st.norm.cdf((-c * se0 - d) / se1)
    c = st.norm.ppf(1 - alpha)
    return st.norm.sf((c * se0 - d) / se1) if tail == "right" else st.norm.cdf((-c * se0 - d) / se1)


@builtin("sampsizepwr")
def _sampsizepwr(interp, args, nargout):
    if len(args) < 3 or not is_text(args[0]):
        raise MatlabError("sampsizepwr: expected sampsizepwr(testtype, p0, p1, pwr, n, 'Name', value, ...).")
    pos, opts = get_opts(args, 3, ["Alpha", "Tail", "Ratio"], "sampsizepwr")
    if len(pos) > 5:
        raise MatlabError("Too many input arguments for sampsizepwr.")
    kind = choice(pos[0], ["z", "t", "t2", "var", "p", "p2"], "The test type")
    alpha = scalar(opts["alpha"], "Alpha") if "alpha" in opts else 0.05
    tail = _tail(opts["tail"]) if "tail" in opts else "both"
    ratio = scalar(opts["ratio"], "Ratio") if "ratio" in opts else 1.0
    if kind not in ("t2", "p2") and "ratio" in opts:
        raise MatlabError("sampsizepwr: 'Ratio' applies to two-sample tests only.")
    p0v = vec(pos[1])
    if kind in ("z", "t", "t2"):
        if p0v.size != 2:
            raise MatlabError(f"sampsizepwr: for the '{kind}' test p0 must be [mean, standard deviation].")
        p0: Any = (float(p0v[0]), float(p0v[1]))
        base = p0[0]
    else:
        p0 = float(p0v[0])
        base = p0
    p1 = pos[2] if len(pos) > 2 else np.zeros((0, 0))
    pwr = pos[3] if len(pos) > 3 else np.zeros((0, 0))
    nn = pos[4] if len(pos) > 4 else np.zeros((0, 0))
    missing = [is_empty_arg(v) for v in (p1, pwr, nn)]
    if sum(missing) != 1:
        raise MatlabError("sampsizepwr: give exactly two of p1, power and n (use [] for the one to compute).")

    def power(p1v: float, n: float) -> float:
        if n < 1:
            return 0.0
        with np.errstate(all="ignore"):
            if kind == "z":
                return float(_power_z(p0, p1v, n, alpha, tail))
            if kind == "t":
                return float(_power_t(p0, p1v, n, alpha, tail))
            if kind == "t2":
                return float(_power_t(p0, p1v, n, alpha, tail, ratio))
            if kind == "var":
                return float(_power_var(p0, p1v, n, alpha, tail))
            if kind == "p":
                return _power_p(p0, p1v, n, alpha, tail)
            return float(_power_p2(p0, p1v, n, alpha, tail, ratio))

    if missing[1]:
        (a, b), shape = _bc(p1, nn)
        res = np.array([power(float(x), float(m)) for x, m in zip(a, b)])
        return res.reshape(shape)
    if missing[2]:
        (a, b), shape = _bc(p1, pwr)
        res = []
        for x, target in zip(a, b):
            if not 0 < target < 1:
                raise MatlabError("sampsizepwr: power must be between 0 and 1.")
            if x == base:
                res.append(np.nan)
                continue
            lo, hi = 1, 2
            while power(float(x), hi) < target and hi < 10 ** 8:
                lo, hi = hi, hi * 2
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if power(float(x), mid) >= target:
                    hi = mid
                else:
                    lo = mid
            res.append(float(hi if power(float(x), lo) < target else lo))
        res = np.array(res).reshape(shape)
        if kind in ("t2", "p2") and nargout > 1:
            return res, np.ceil(ratio * res)
        return res
    (a, b), shape = _bc(pwr, nn)
    res = []
    for target, m in zip(a, b):
        # search on the side of the alternative given by the tail (upper side for two-sided tests)
        sgn = -1.0 if tail == "left" else 1.0
        scale = p0[1] if isinstance(p0, tuple) else (p0 if kind == "var" else 0.5)
        lo, hi = base, base + sgn * scale * 1e-6
        limit = None if kind in ("z", "t", "t2", "var") else (1.0 if sgn > 0 else 0.0)
        step = scale
        for _ in range(200):
            cand = base + sgn * step
            if limit is not None:
                cand = min(cand, 1 - 1e-12) if sgn > 0 else max(cand, 1e-12)
            if kind == "var" and cand <= 0:
                cand = base * 1e-9
            if power(cand, float(m)) >= target or (limit is not None and abs(cand - limit) < 1e-11):
                hi = cand
                break
            step *= 2
        else:
            hi = np.nan
        if np.isnan(hi):
            res.append(np.nan)
            continue
        lo = base
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if power(mid, float(m)) >= target:
                hi = mid
            else:
                lo = mid
        res.append(hi)
    return np.array(res).reshape(shape)


def _bc(a: Any, b: Any):
    x, y = np.asarray(fnum(a), dtype=float), np.asarray(fnum(b), dtype=float)
    if x.size != 1 and y.size != 1 and x.shape != y.shape:
        raise MatlabError("sampsizepwr: non-scalar inputs must have the same size.")
    shape = x.shape if x.size != 1 else y.shape
    return (np.broadcast_to(x, shape).flatten(), np.broadcast_to(y, shape).flatten()), shape
