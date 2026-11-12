"""N-dimensional transforms and more statistics.

fftn, ifftn, convn, rcond, interpn, interp3, datasample, mvnpdf, mvnrnd, more
random-number samplers, pca, grp2idx, ksdensity, nan-aware means, inline, builtin.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, MString, StringArray, char, is_char, to_str
from ._registry import builtin
from ._util import fnum, fscalar, is_text, nargs, num, text, to_int
from .spatial import _options


def _shape2(shape: tuple) -> tuple:
    shape = tuple(shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape = shape[:-1]
    return shape


# -- N-D FFT and convolution ---------------------------------------------------------------------

def _fftn_common(args: list, inverse: bool, name: str):
    rest = list(args)
    symmetric = False
    if rest and is_text(rest[-1]):
        flag = text(rest.pop()).lower()
        if flag not in ("symmetric", "nonsymmetric"):
            raise MatlabError(f"{name}: unknown option '{flag}'.")
        symmetric = flag == "symmetric"
    nargs(rest, 1, 2, name)
    x = fnum(rest[0])
    s = None
    if len(rest) == 2 and not (isinstance(rest[1], np.ndarray) and rest[1].size == 0):
        s = [int(v) for v in fnum(rest[1]).flatten()]
        if len(s) < x.ndim or any(v < 0 for v in s):
            raise MatlabError(f"{name}: the size vector needs one non-negative entry per dimension of the "
                              f"input ({x.ndim} here).")
        x = x.reshape(x.shape + (1,) * (len(s) - x.ndim))
    single = x.dtype in (np.float32, np.complex64)
    if inverse:
        r = np.fft.ifftn(x, s=s, axes=None if s is None else list(range(len(s))))
        if symmetric:
            r = r.real.copy()
        else:
            from .signal import _real_if_tiny
            r = _real_if_tiny(r)
    else:
        r = np.fft.fftn(x, s=s, axes=None if s is None else list(range(len(s)))).astype(complex)
    if single:
        r = r.astype(np.complex64 if np.iscomplexobj(r) else np.float32)
    return r.reshape(_shape2(r.shape))


@builtin("fftn")
def _fftn(interp, args, nargout):
    return _fftn_common(args, False, "fftn")


@builtin("ifftn")
def _ifftn(interp, args, nargout):
    return _fftn_common(args, True, "ifftn")


@builtin("convn")
def _convn(interp, args, nargout):
    from scipy.signal import convolve
    nargs(args, 2, 3, "convn")
    a, b = num(args[0]), num(args[1])
    shape = text(args[2]).lower() if len(args) > 2 else "full"
    if shape not in ("full", "same", "valid"):
        raise MatlabError("convn: the shape must be 'full', 'same' or 'valid'.")
    single = np.float32 in (a.dtype, b.dtype) or np.complex64 in (a.dtype, b.dtype)
    a = a.astype(complex if np.iscomplexobj(a) else float)
    b = b.astype(complex if np.iscomplexobj(b) else float)
    nd = max(a.ndim, b.ndim)
    a = a.reshape(a.shape + (1,) * (nd - a.ndim))
    b = b.reshape(b.shape + (1,) * (nd - b.ndim))
    if a.size == 0 or b.size == 0:
        full_shape = tuple(max(x + y - 1, 0) for x, y in zip(a.shape, b.shape))
        out = np.zeros(a.shape if shape == "same" else full_shape)
        return out.reshape(_shape2(out.shape))
    if shape == "valid":
        if any(x < y for x, y in zip(a.shape, b.shape)):
            out = np.zeros(tuple(max(x - y + 1, 0) for x, y in zip(a.shape, b.shape)))
        else:
            out = convolve(a, b, mode="valid", method="direct" if a.size * b.size < 1e7 else "auto")
    else:
        out = convolve(a, b, mode="full", method="direct" if a.size * b.size < 1e7 else "auto")
        if shape == "same":  # the central part, the size of A
            idx = tuple(slice(nb // 2, nb // 2 + na) for na, nb in zip(a.shape, b.shape))
            out = out[idx]
    if single:
        out = out.astype(np.complex64 if np.iscomplexobj(out) else np.float32)
    return out.reshape(_shape2(out.shape))


@builtin("rcond")
def _rcond(interp, args, nargout):
    nargs(args, 1, 1, "rcond")
    a = fnum(args[0])
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise MatlabError("rcond: the matrix must be square.")
    if a.size == 0:
        return fscalar(np.inf)
    if not np.all(np.isfinite(a)):
        return fscalar(np.nan)
    from scipy.linalg import lapack
    a = a.astype(complex if np.iscomplexobj(a) else float)
    getrf, gecon = lapack.get_lapack_funcs(("getrf", "gecon"), (a,))
    anorm = float(np.max(np.sum(np.abs(a), axis=0)))
    lu, _, info = getrf(a)
    if info > 0:
        return fscalar(0.0)
    rc, _ = gecon(lu, anorm, norm="1")
    return fscalar(float(rc))


# -- gridded interpolation in N dimensions ---------------------------------------------------------

def _interp_grid(name: str, args: list, meshgrid: bool):
    from scipy.interpolate import RegularGridInterpolator
    rest = list(args)
    method, extrap = "linear", np.nan
    if rest and not is_text(rest[-1]) and len(rest) >= 2 and is_text(rest[-2]):
        extrap = float(fnum(rest.pop()).flat[0])
    if rest and is_text(rest[-1]):
        method = text(rest.pop()).lower().lstrip("*")
    kinds = {"linear": "linear", "nearest": "nearest", "cubic": "cubic", "spline": "cubic",
             "makima": "cubic", "pchip": "pchip"}
    if method not in kinds:
        raise MatlabError(f"{name}: unknown method '{method}'. Use 'linear', 'nearest', 'cubic' or 'spline'.")
    arrays = [fnum(a).astype(float) for a in rest]
    if name == "interp3" and len(arrays) not in (4, 7):
        raise MatlabError("interp3 expects interp3(V, Xq, Yq, Zq) or interp3(X, Y, Z, V, Xq, Yq, Zq).")
    n = len(arrays)
    if n < 2:
        raise MatlabError(f"{name}: give the values and one query array per dimension.")
    gridded = n % 2 == 1 and n >= 5 and not (arrays[0].ndim == n - 1 and n - 1 > 2)
    if gridded:
        nd = (n - 1) // 2
        grid, V, query = arrays[:nd], arrays[nd], arrays[nd + 1:]
    else:
        V, query = arrays[0], arrays[1:]
        nd = len(query)
        grid = None
    V = V.reshape(V.shape + (1,) * (nd - V.ndim))
    if V.ndim != nd:
        raise MatlabError(f"{name}: the values have {V.ndim} dimensions but {nd} coordinates were given.")
    order = list(range(nd))
    if meshgrid and nd >= 2:
        order[0], order[1] = 1, 0  # X runs along columns, Y along rows
    axes: list = [None] * nd
    for k in range(nd):
        dim = order[k]
        if grid is None:
            axes[dim] = np.arange(1.0, V.shape[dim] + 1)
        else:
            g = grid[k]
            if g.size == V.shape[dim] and (g.ndim <= 2 and min(g.shape) == 1):
                axes[dim] = g.flatten()
            elif g.shape == V.shape:
                idx = [0] * nd
                idx[dim] = slice(None)
                axes[dim] = g[tuple(idx)]
            else:
                raise MatlabError(f"{name}: grid coordinate {k + 1} does not match the size of the values.")
    flips = []
    for dim in range(nd):
        d = np.diff(axes[dim])
        if d.size and np.all(d < 0):
            axes[dim] = axes[dim][::-1]
            flips.append(dim)
        elif d.size and not np.all(d > 0):
            raise MatlabError(f"{name}: the grid coordinates must be strictly monotonic.")
    if flips:
        V = np.flip(V, axis=tuple(flips))
    try:
        query = list(np.broadcast_arrays(*query))
    except ValueError:
        raise MatlabError(f"{name}: the query arrays must be the same size.") from None
    pts = np.empty((query[0].size, nd))
    for k in range(nd):
        pts[:, order[k]] = query[k].flatten(order="F")
    kind = kinds[method]
    if kind in ("cubic", "pchip") and min(V.shape) < 4:
        kind = "linear"
    f = RegularGridInterpolator(tuple(axes), V, method=kind, bounds_error=False, fill_value=extrap)
    return f(pts).reshape(query[0].shape, order="F")


@builtin("interpn")
def _interpn(interp, args, nargout):
    return _interp_grid("interpn", args, False)


@builtin("interp3")
def _interp3(interp, args, nargout):
    return _interp_grid("interp3", args, True)


# -- sampling ----------------------------------------------------------------------------------------

@builtin("datasample")
def _datasample(interp, args, nargout):
    from .core import _rng
    rest, opts = _options(args, {"replace", "weights"}, "datasample")
    if rest and getattr(rest[0], "class_name", lambda: "")() == "RandStream":
        gen = rest.pop(0).gen
    else:
        gen = _rng(interp)
    nargs(rest, 2, 3, "datasample")
    data = rest[0]
    k = to_int(rest[1], "number of samples")
    if k < 0:
        raise MatlabError("datasample: the number of samples must not be negative.")
    shape = tuple(getattr(data, "data", data).shape) if not isinstance(data, np.ndarray) else data.shape
    dim = to_int(rest[2], "dimension") - 1 if len(rest) > 2 else next(
        (d for d, n in enumerate(shape) if n != 1), 0)
    n = shape[dim] if dim < len(shape) else 1
    replace = True
    if "replace" in opts:
        replace = bool(np.asarray(opts["replace"]).flat[0])
    p = None
    if "weights" in opts:
        w = fnum(opts["weights"]).astype(float).flatten()
        if w.size != n or np.any(w < 0) or w.sum() <= 0:
            raise MatlabError(f"datasample: Weights must be {n} non-negative numbers that are not all zero.")
        p = w / w.sum()
    if not replace and k > n:
        raise MatlabError(f"datasample: cannot take {k} samples without replacement from {n} items.")
    if n == 0 and k > 0:
        raise MatlabError("datasample: there is no data to sample from.")
    idx = gen.choice(n, size=k, replace=replace, p=p) if k else np.zeros(0, dtype=int)
    from .. import indexing as ix
    sel = [ix.COLON if d != dim else (idx + 1.0).reshape(1, -1) for d in range(max(len(shape), dim + 1))]
    from ..interpreter import index_value
    if hasattr(data, "index_get"):
        picked = data.index_get(interp, sel)
    else:
        picked = index_value(data, sel)
    return picked, (idx + 1.0).reshape((1, -1) if dim == 1 else (-1, 1))


def _mvn_args(args: list, name: str) -> tuple[np.ndarray, np.ndarray]:
    mu = fnum(args[0]).astype(float)
    sigma = fnum(args[1]).astype(float)
    return mu, sigma


@builtin("mvnrnd")
def _mvnrnd(interp, args, nargout):
    from .core import _rng
    nargs(args, 2, 3, "mvnrnd")
    mu, sigma = _mvn_args(args, "mvnrnd")
    if mu.ndim != 2:
        raise MatlabError("mvnrnd: mu must be a vector or a matrix with one mean per row.")
    if mu.shape[1] == 1 and mu.shape[0] > 1 and sigma.shape[0] == mu.shape[0]:
        mu = mu.T
    d = mu.shape[1]
    if sigma.shape[0] == 1 and sigma.shape[1] == d and d > 1:
        sigma = np.diag(sigma.flatten())
    if sigma.shape != (d, d):
        raise MatlabError(f"mvnrnd: Sigma must be a {d}-by-{d} covariance matrix (mu has {d} columns).")
    if not np.allclose(sigma, sigma.T, atol=1e-10 * max(1.0, np.abs(sigma).max())):
        raise MatlabError("mvnrnd: Sigma must be symmetric.")
    n = to_int(args[2], "number of samples") if len(args) > 2 else mu.shape[0]
    if mu.shape[0] not in (1, n):
        raise MatlabError("mvnrnd: mu must have one row, or one row per sample.")
    w, V = np.linalg.eigh((sigma + sigma.T) / 2)
    if w.min() < -1e-8 * max(1.0, w.max()):
        raise MatlabError("mvnrnd: Sigma must be positive semi-definite.")
    root = V * np.sqrt(np.clip(w, 0, None))
    return mu + _rng(interp).standard_normal((n, d)) @ root.T


@builtin("mvnpdf")
def _mvnpdf(interp, args, nargout):
    nargs(args, 1, 3, "mvnpdf")
    X = fnum(args[0]).astype(float)
    n, d = X.shape
    mu = fnum(args[1]).astype(float) if len(args) > 1 and args[1].size else np.zeros((1, d))
    if mu.shape == (d, 1) and d > 1:
        mu = mu.T
    if mu.shape[1] != d or mu.shape[0] not in (1, n):
        raise MatlabError(f"mvnpdf: mu must be a 1-by-{d} row (X has {d} columns), or have one row per "
                          "row of X.")
    sigma = fnum(args[2]).astype(float) if len(args) > 2 and args[2].size else np.eye(d)
    if sigma.shape == (1, d) and d > 1:
        sigma = np.diag(sigma.flatten())
    if sigma.shape != (d, d):
        raise MatlabError(f"mvnpdf: Sigma must be a {d}-by-{d} covariance matrix.")
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        raise MatlabError("mvnpdf: Sigma must be symmetric and positive definite.") from None
    z = np.linalg.solve(L, (X - mu).T)
    logp = -0.5 * np.sum(z ** 2, axis=0) - np.sum(np.log(np.diag(L))) - 0.5 * d * np.log(2 * np.pi)
    return np.exp(logp).reshape(-1, 1)


def _more_samplers() -> None:
    from .stats import _rnd

    def guard(ok, values):
        return np.where(ok, values, np.nan)

    samplers = {
        "gamrnd": (2, lambda g, s, a, b: guard((a > 0) & (b > 0), g.gamma(np.where(a > 0, a, 1), 1.0, s)
                                               * np.where(b > 0, b, 1))),
        "betarnd": (2, lambda g, s, a, b: guard((a > 0) & (b > 0),
                                                g.beta(np.where(a > 0, a, 1), np.where(b > 0, b, 1), s))),
        "chi2rnd": (1, lambda g, s, v: guard(v > 0, g.chisquare(np.where(v > 0, v, 1), s))),
        "trnd": (1, lambda g, s, v: guard(v > 0, g.standard_t(np.where(v > 0, v, 1), s))),
        "frnd": (2, lambda g, s, a, b: guard((a > 0) & (b > 0),
                                             g.f(np.where(a > 0, a, 1), np.where(b > 0, b, 1), s))),
        "lognrnd": (2, lambda g, s, mu, sg: guard(sg >= 0, np.exp(mu + np.where(sg >= 0, sg, 0)
                                                                  * g.standard_normal(s)))),
        "wblrnd": (2, lambda g, s, a, b: guard((a > 0) & (b > 0),
                                               a * g.weibull(np.where(b > 0, b, 1), s))),
        "raylrnd": (1, lambda g, s, b: guard(b > 0, g.rayleigh(np.where(b > 0, b, 1), s))),
        "geornd": (1, lambda g, s, p: guard((p > 0) & (p <= 1),
                                            g.geometric(np.where((p > 0) & (p <= 1), p, 1), s) - 1.0)),
        "nbinrnd": (2, lambda g, s, r, p: guard((r > 0) & (p > 0) & (p <= 1), g.negative_binomial(
            np.where(r > 0, r, 1), np.where((p > 0) & (p <= 1), p, 1), s))),
        "unidrnd": (1, lambda g, s, n: guard(n >= 1, np.floor(g.random(s) * np.where(n >= 1, n, 1)) + 1)),
    }
    from . import BUILTINS
    for name, (count, fn) in samplers.items():
        if name not in BUILTINS:
            builtin(name)(_rnd(name, count, fn))


_more_samplers()


# -- nan-aware and other means -----------------------------------------------------------------------

def _reduce(name: str, fn):
    def f(interp, args, nargout):
        nargs(args, 1, 2, name)
        a = fnum(args[0]).astype(float)
        axis = to_int(args[1], "dimension") - 1 if len(args) > 1 else next(
            (d for d, n in enumerate(a.shape) if n != 1), 0)
        if axis >= a.ndim:
            return a.copy()
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return np.expand_dims(fn(a, axis), axis)
    from . import BUILTINS
    if name not in BUILTINS:
        builtin(name)(f)


_reduce("nanmedian", lambda a, ax: np.nanmedian(a, axis=ax))
_reduce("nanvar", lambda a, ax: np.nanvar(a, axis=ax, ddof=1) if a.shape[ax] > 1 else np.nanvar(a, axis=ax))
_reduce("geomean", lambda a, ax: np.where(np.any(a < 0, axis=ax), np.nan, np.exp(np.mean(np.log(a), axis=ax))))
_reduce("harmmean", lambda a, ax: a.shape[ax] / np.sum(1.0 / a, axis=ax))


# -- pca, grp2idx, ksdensity --------------------------------------------------------------------------

@builtin("pca")
def _pca(interp, args, nargout):
    rest, opts = _options(args, {"numcomponents", "centered", "algorithm", "economy", "rows"}, "pca")
    nargs(rest, 1, 1, "pca")
    X = fnum(rest[0]).astype(float)
    if X.ndim != 2:
        raise MatlabError("pca: the data must be a matrix with one observation per row.")
    rows = text(opts["rows"]).lower() if "rows" in opts else "complete"
    keep = ~np.isnan(X).any(axis=1)
    if rows == "complete":
        Xc = X[keep]
    elif np.isnan(X).any():
        raise MatlabError("pca: the data contains NaN. Use 'Rows', 'complete' (the default) to skip those rows.")
    else:
        Xc = X
    n, p = Xc.shape
    centered = True
    if "centered" in opts:
        c = opts["centered"]
        centered = text(c).lower() not in ("off", "false") if is_text(c) else bool(np.asarray(c).flat[0])
    mu = Xc.mean(axis=0, keepdims=True) if (centered and n) else np.zeros((1, p))
    Z = Xc - mu
    dof = max(n - (1 if centered else 0), 1)
    if n == 0:
        return (np.zeros((p, 0)), np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1)), mu)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    coeff = Vt.T
    ncomp = min(dof if centered else n, p)
    if "numcomponents" in opts:
        ncomp = min(ncomp, to_int(opts["numcomponents"], "NumComponents"))
    # Sign convention: the largest element of each coefficient column is positive.
    big = np.argmax(np.abs(coeff), axis=0)
    signs = np.sign(coeff[big, np.arange(coeff.shape[1])])
    signs[signs == 0] = 1
    coeff = coeff * signs
    latent_all = (s ** 2) / dof
    nvalid = min(dof if centered else n, p)
    latent_all = latent_all[:nvalid]
    coeff = coeff[:, :ncomp]
    score_c = Z @ coeff
    score = np.full((X.shape[0], ncomp), np.nan)
    if rows == "complete":
        score[keep] = score_c
    else:
        score = score_c
    latent = latent_all.reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_scores = (Z @ (Vt.T * signs)[:, :nvalid]) / np.sqrt(latent_all)
    std_scores[:, latent_all <= 1e-14 * max(latent_all.max(initial=0.0), 1e-300)] = 0
    t2 = np.full((X.shape[0], 1), np.nan)
    t2_vals = np.sum(std_scores ** 2, axis=1, keepdims=True)
    if rows == "complete":
        t2[keep] = t2_vals
    else:
        t2 = t2_vals
    total = latent_all.sum()
    explained = (100 * latent_all / total if total > 0 else np.zeros_like(latent_all)).reshape(-1, 1)
    return coeff, score, latent, t2, explained, mu


def group_index(g: Any) -> tuple[np.ndarray, list[str], Any]:
    """Group numbers (NaN for missing), group names and group values for a grouping variable."""
    if hasattr(g, "categories") and hasattr(g, "codes"):  # categorical
        codes = g.codes.flatten(order="F").astype(float)
        used = [k for k in range(len(g.categories)) if np.any(codes == k)]
        remap = {k: i + 1.0 for i, k in enumerate(used)}
        idx = np.array([remap.get(int(c), np.nan) if c >= 0 else np.nan for c in codes])
        names = [g.categories[k] for k in used]
        return idx, names, Cell.from_list([char(n) for n in names], column=True)
    if isinstance(g, (Cell, StringArray, MString)) or is_char(g):
        if isinstance(g, Cell):
            items = [to_str(x) for x in g.data.flatten(order="F")]
        elif isinstance(g, StringArray):
            items = list(g.data.flatten(order="F"))
        elif isinstance(g, MString):
            items = [str(g)]
        else:
            items = ["".join(row).rstrip() for row in g] if g.ndim == 2 else [to_str(g)]
        names: list[str] = []
        idx = np.empty(len(items))
        for k, s in enumerate(items):
            if s is None or s == "":
                idx[k] = np.nan
                continue
            if s not in names:
                names.append(s)
            idx[k] = names.index(s) + 1.0
        if isinstance(g, (StringArray, MString)):
            order = sorted(range(len(names)), key=lambda i: names[i])
            rank = {old + 1.0: new + 1.0 for new, old in enumerate(order)}
            idx = np.array([rank.get(v, np.nan) for v in idx])
            names = [names[i] for i in order]
        return idx, names, Cell.from_list([char(n) for n in names], column=True)
    if isinstance(g, np.ndarray):
        flat = g.flatten(order="F").astype(float)
        values = np.unique(flat[~np.isnan(flat)])
        idx = np.full(flat.size, np.nan)
        ok = ~np.isnan(flat)
        idx[ok] = np.searchsorted(values, flat[ok]) + 1.0
        if g.dtype == bool:
            names = [str(int(v)) for v in values]
            levels = values.astype(bool).reshape(-1, 1)
        else:
            from ..display import format_scalar
            names = [format_scalar(v).strip() for v in values]
            levels = values.astype(g.dtype if g.dtype.kind in "iuf" else float).reshape(-1, 1)
        return idx, names, levels
    raise MatlabError("The grouping variable must be numeric, logical, categorical, text or a cell array "
                      "of text.")


@builtin("grp2idx")
def _grp2idx(interp, args, nargout):
    nargs(args, 1, 1, "grp2idx")
    idx, names, levels = group_index(args[0])
    return idx.reshape(-1, 1), Cell.from_list([char(n) for n in names], column=True), levels


@builtin("ksdensity")
def _ksdensity(interp, args, nargout):
    from scipy.special import ndtr
    rest, opts = _options(args, {"bandwidth", "kernel", "function", "numpoints", "support", "weights",
                                 "width", "npoints"}, "ksdensity")
    nargs(rest, 1, 2, "ksdensity")
    x = fnum(rest[0]).astype(float).flatten()
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise MatlabError("ksdensity: the sample is empty.")
    n = x.size
    w = np.full(n, 1.0 / n)
    if "weights" in opts:
        w = fnum(opts["weights"]).astype(float).flatten()
        if w.size != n:
            raise MatlabError("ksdensity: Weights must have one entry per sample value.")
        w = w / w.sum()
    kernel = text(opts["kernel"]).lower() if "kernel" in opts else "normal"
    func = text(opts["function"]).lower() if "function" in opts else "pdf"
    if func not in ("pdf", "cdf", "survivor"):
        raise MatlabError("ksdensity: Function must be 'pdf', 'cdf' or 'survivor'.")
    bw_arg = opts.get("bandwidth", opts.get("width"))
    if bw_arg is not None and not is_text(bw_arg) and np.asarray(bw_arg).size:
        bw = float(fnum(bw_arg).flat[0])
    else:
        sig = np.median(np.abs(x - np.median(x))) / 0.6745  # robust estimate of the spread
        if sig <= 0:
            sig = float(np.max(x) - np.min(x)) or 1.0
        bw = sig * (4.0 / (3.0 * n)) ** 0.2
    if bw <= 0:
        raise MatlabError("ksdensity: the bandwidth must be positive.")
    positive = "support" in opts and is_text(opts["support"]) and text(opts["support"]).lower() == "positive"
    if positive and np.any(x <= 0):
        raise MatlabError("ksdensity: with 'Support', 'positive' all sample values must be positive.")
    if len(rest) > 1 and rest[1].size:
        xi = fnum(rest[1]).astype(float)
        shape = xi.shape
        xi = xi.flatten(order="F")
    else:
        m = to_int(opts.get("numpoints", opts.get("npoints", np.array([[100.0]]))), "NumPoints")
        if positive:
            lo, hi = np.log(x.min()) - 3 * bw, np.log(x.max()) + 3 * bw
            xi = np.exp(np.linspace(lo, hi, m))
        else:
            xi = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, m)
        shape = (1, m)
    data, pts = (np.log(x), np.log(np.where(xi > 0, xi, np.nan))) if positive else (x, xi)
    z = (pts[:, None] - data[None, :]) / bw
    if func == "pdf":
        if kernel in ("normal", "gaussian"):
            k = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        elif kernel == "box":
            k = (np.abs(z) <= np.sqrt(3)) / (2 * np.sqrt(3))
        elif kernel == "triangle":
            k = np.clip(1 - np.abs(z) / np.sqrt(6), 0, None) / np.sqrt(6)
        elif kernel == "epanechnikov":
            k = np.clip(1 - z ** 2 / 5, 0, None) * 0.75 / np.sqrt(5)
        else:
            raise MatlabError("ksdensity: Kernel must be 'normal', 'box', 'triangle' or 'epanechnikov'.")
        f = (k @ w) / bw
        if positive:
            f = np.where(xi > 0, f / np.where(xi > 0, xi, 1), 0.0)
    else:
        if kernel not in ("normal", "gaussian"):
            raise MatlabError("ksdensity: the cdf is available for the 'normal' kernel only.")
        f = ndtr(z) @ w
        if positive:
            f = np.where(xi > 0, f, 0.0)
        if func == "survivor":
            f = 1 - f
    return f.reshape(shape, order="F"), xi.reshape(shape, order="F"), fscalar(bw)


# -- inline, builtin ---------------------------------------------------------------------------------

_NOT_VARIABLES = {"pi", "inf", "nan", "eps", "i", "j", "e", "end", "true", "false"}


@builtin("inline")
def _inline(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("inline needs the expression as text, e.g. inline('x.^2 + 1').")
    expr = text(args[0])
    if len(args) > 1:
        names = [text(a, "argument name") for a in args[1:]]
    else:
        found = re.findall(r"(?<![\w.'])([A-Za-z]\w*)(?!\w*\s*\()", re.sub(r"'[^']*'", "", expr))
        names = sorted({n for n in found if n.lower() not in _NOT_VARIABLES
                        and not interp.function_exists(n)}) or ["x"]
    for n in names:
        if not n.isidentifier():
            raise MatlabError(f"inline: '{n}' is not a valid argument name.")
    return _eval_handle(interp, f"@({', '.join(names)}) {expr}")


def _eval_handle(interp, code: str):
    from ..interpreter import Frame
    from ..parser import parse
    program = parse(code)
    if len(program.body) != 1:
        raise MatlabError("inline: the text must be a single expression.")
    return interp.eval(program.body[0].expr, Frame(interp, interp.frame.functions))


@builtin("builtin")
def _builtin(interp, args, nargout):
    if not args:
        raise MatlabError("builtin needs a function name, e.g. builtin('disp', x).")
    f = args[0]
    name = f.name if hasattr(f, "name") and getattr(f, "name", None) else text(f, "function name")
    fn = interp.builtins.get(name)
    if fn is None:
        raise MatlabError(f"'{name}' is not a built-in function.")
    res = interp.call_builtin(name, fn, list(args[1:]), nargout)
    return tuple(res) if res else None


# -- random by distribution name ------------------------------------------------------------------------

_RANDOM = {"normal": ("normrnd", 2), "gaussian": ("normrnd", 2), "uniform": ("unifrnd", 2),
           "exponential": ("exprnd", 1), "poisson": ("poissrnd", 1), "binomial": ("binornd", 2),
           "gamma": ("gamrnd", 2), "beta": ("betarnd", 2), "chisquare": ("chi2rnd", 1), "chi2": ("chi2rnd", 1),
           "t": ("trnd", 1), "f": ("frnd", 2), "lognormal": ("lognrnd", 2), "weibull": ("wblrnd", 2),
           "rayleigh": ("raylrnd", 1), "geometric": ("geornd", 1), "negativebinomial": ("nbinrnd", 2),
           "discreteuniform": ("unidrnd", 1)}
_RANDOM.update({v[0][:-3]: v for v in list(_RANDOM.values())})  # short names: 'norm', 'unif', 'exp', ...


@builtin("random")
def _random(interp, args, nargout):
    if not args or not is_text(args[0]):
        raise MatlabError("random needs a distribution name first, e.g. random('Normal', 0, 1, 3, 2).")
    key = text(args[0]).lower().replace(" ", "")
    if key not in _RANDOM:
        raise MatlabError(f"random: unknown distribution '{text(args[0])}'. Known names: "
                          + ", ".join(sorted(k for k in _RANDOM if len(k) > 4)) + ".")
    name, count = _RANDOM[key]
    if len(args) - 1 < count:
        raise MatlabError(f"random: the {text(args[0])} distribution needs {count} parameter(s).")
    return interp.call_builtin(name, interp.builtins[name], list(args[1:]), 1)[0]


# -- spectrogram ---------------------------------------------------------------------------------------

@builtin("spectrogram")
def _spectrogram(interp, args, nargout):
    rest = [a for a in args if not is_text(a)]
    flags = [text(a).lower() for a in args if is_text(a)]
    if not rest:
        raise MatlabError("spectrogram needs a signal: spectrogram(x, window, noverlap, nfft, fs).")
    x = fnum(rest[0])
    if x.ndim != 2 or min(x.shape) != 1:
        raise MatlabError("spectrogram: the signal must be a vector.")
    x = x.flatten()
    n = x.size

    def given(k: int) -> bool:
        return len(rest) > k and rest[k].size > 0

    if given(1):
        w = fnum(rest[1]).astype(float).flatten()
        if w.size == 1:
            w = np.hamming(int(w[0]))
    else:
        w = np.hamming(max(int(n // 4.5), 1))
    L = w.size
    if L > n or L < 1:
        raise MatlabError(f"spectrogram: the window ({L} samples) is longer than the signal ({n} samples).")
    noverlap = to_int(rest[2], "noverlap") if given(2) else L // 2
    if noverlap >= L:
        raise MatlabError("spectrogram: noverlap must be smaller than the window length.")
    nfft = to_int(rest[3], "nfft") if given(3) else max(256, 1 << int(np.ceil(np.log2(L))))
    if nfft < L:
        raise MatlabError("spectrogram: nfft must be at least the window length.")
    fs = float(fnum(rest[4]).flat[0]) if given(4) else None
    hop = L - noverlap
    count = (n - noverlap) // hop
    starts = np.arange(count) * hop
    frames = np.stack([x[s:s + L] * w for s in starts], axis=1) if count else np.zeros((L, 0))
    S = np.fft.fft(frames, n=nfft, axis=0)
    two_sided = np.iscomplexobj(x) or "twosided" in flags or "centered" in flags
    rate = fs if fs is not None else 2 * np.pi
    scale = rate * np.sum(w ** 2)
    if two_sided:
        F = np.arange(nfft) * rate / nfft
        P = np.abs(S) ** 2 / scale
        if "centered" in flags:
            S, P = np.fft.fftshift(S, axes=0), np.fft.fftshift(P, axes=0)
            F = F - rate * (nfft // 2) / nfft
    else:
        bins = nfft // 2 + 1
        S = S[:bins]
        F = np.arange(bins) * rate / nfft
        P = np.abs(S) ** 2 / scale
        last = bins - 1 if nfft % 2 == 0 else bins
        P[1:last] *= 2
    T = (L / 2 + starts) / (fs if fs is not None else 1.0)
    if nargout == 0:
        with np.errstate(divide="ignore"):
            db = 10 * np.log10(np.abs(P))
        on_y = "yaxis" in flags
        b = interp.builtins
        if on_y:
            interp.call_builtin("imagesc", b["imagesc"], [T.reshape(1, -1), F.reshape(1, -1), db], 0)
        else:
            interp.call_builtin("imagesc", b["imagesc"], [F.reshape(1, -1), T.reshape(1, -1), db.T], 0)
        interp.call_builtin("axis", b["axis"], [char("xy")], 0)
        flabel = "Frequency (Hz)" if fs is not None else "Normalized frequency (rad/sample)"
        tlabel = "Time (s)" if fs is not None else "Samples"
        interp.call_builtin("xlabel", b["xlabel"], [char(tlabel if on_y else flabel)], 0)
        interp.call_builtin("ylabel", b["ylabel"], [char(flabel if on_y else tlabel)], 0)
        interp.call_builtin("colorbar", b["colorbar"], [], 0)
        return None
    return S, F.reshape(-1, 1), T.reshape(1, -1), P


# -- small utilities --------------------------------------------------------------------------------------

@builtin("ldl")
def _ldl(interp, args, nargout):
    from scipy.linalg import ldl
    nargs(args, 1, 2, "ldl")
    a = fnum(args[0])
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise MatlabError("ldl: the matrix must be square.")
    lu, d, perm = ldl(a, lower=True, hermitian=True)
    if nargout <= 2:
        return lu, d
    return lu[perm], d, np.eye(a.shape[0])[:, perm]


def _resample_step(name: str, up: bool):
    def f(interp, args, nargout):
        nargs(args, 2, 3, name)
        x = args[0] if isinstance(args[0], np.ndarray) else fnum(args[0])
        n = to_int(args[1], "factor")
        phase = to_int(args[2], "phase") if len(args) > 2 else 0
        if n < 1 or not 0 <= phase < n:
            raise MatlabError(f"{name}: the factor must be a positive integer and the phase between 0 and "
                              "factor-1.")
        row = x.ndim == 2 and x.shape[0] == 1
        a = x.T if row else x
        if up:
            out = np.zeros((a.shape[0] * n,) + a.shape[1:], dtype=a.dtype)
            out[phase::n] = a
        else:
            out = a[phase::n].copy()
        return out.T if row else out
    from . import BUILTINS
    if name not in BUILTINS:
        builtin(name)(f)


_resample_step("upsample", True)
_resample_step("downsample", False)


def _encoding(args: list, pos: int) -> str:
    import codecs
    name = text(args[pos]) if len(args) > pos else "utf-8"
    name = {"": "utf-8", "us-ascii": "ascii", "windows-1252": "cp1252"}.get(name.lower(), name)
    try:
        return codecs.lookup(name).name
    except LookupError:
        raise MatlabError(f"Unknown character encoding '{name}'.") from None


@builtin("native2unicode")
def _native2unicode(interp, args, nargout):
    nargs(args, 1, 2, "native2unicode")
    raw = bytes(int(v) & 0xFF for v in num(args[0]).flatten())
    return char(raw.decode(_encoding(args, 1), errors="replace"))


@builtin("unicode2native")
def _unicode2native(interp, args, nargout):
    nargs(args, 1, 2, "unicode2native")
    raw = text(args[0]).encode(_encoding(args, 1), errors="replace")
    return np.frombuffer(raw, dtype=np.uint8).reshape(1, -1).copy()


@builtin("findprop")
def _findprop(interp, args, nargout):
    from ..values import Struct
    nargs(args, 2, 2, "findprop")
    cls = getattr(args[0], "cls", None)
    name = text(args[1], "property name")
    if cls is None or name not in cls.prop_defs:
        return np.zeros((0, 0))
    prop, owner = cls.prop_defs[name]
    attrs = prop.attrs

    def access(kind: str) -> Any:
        return char(str(attrs.get(kind, attrs.get("Access", "public"))).lower())

    def flag(kind: str) -> Any:
        return np.array([[bool(attrs.get(kind, False))]])

    return Struct.scalar({"Name": char(name), "GetAccess": access("GetAccess"), "SetAccess": access("SetAccess"),
                          "Dependent": flag("Dependent"), "Constant": flag("Constant"),
                          "Abstract": flag("Abstract"), "Transient": flag("Transient"),
                          "Hidden": flag("Hidden"), "SetObservable": flag("SetObservable"),
                          "HasDefault": np.array([[prop.default is not None]]),
                          "DefiningClass": char(owner.name)})
