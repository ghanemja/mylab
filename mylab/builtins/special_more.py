"""More special functions, exact degree trigonometry, and array reduction helpers.

SciPy is imported inside the functions that need it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, FunctionHandle, StringArray, as_2d, char, is_char, simplify, to_int, to_str
from ._registry import BUILTINS, builtin
from ._util import default_dim, fnum, is_text, nargs, num, text
from .mathfns import _direction_list, _row_order


def _sp():
    import scipy.special as sp
    return sp


def _is_single(*vals) -> bool:
    return any(isinstance(v, np.ndarray) and v.dtype in (np.float32, np.complex64) for v in vals)


def _res(r, single: bool = False) -> np.ndarray:
    r = simplify(np.asarray(r))
    if r.ndim < 2:
        r = r.reshape(1, -1) if r.ndim == 1 else r.reshape(1, 1)
    if single and r.dtype.kind in "fc":
        r = r.astype(np.complex64 if r.dtype.kind == "c" else np.float32)
    return r


def _real_arg(v: Any, name: str) -> np.ndarray:
    a = fnum(v)
    if a.dtype.kind == "c":
        raise MatlabError(f"{name}: inputs must be real.")
    return a.astype(float)


def _bcast(name: str, *vals) -> list[np.ndarray]:
    arrs = [np.asarray(v) for v in vals]
    shapes = {a.shape for a in arrs if a.size != 1}
    if len(shapes) > 1:
        raise MatlabError(f"{name}: non-scalar inputs must have the same size.")
    target = next(iter(shapes)) if shapes else (1, 1)
    return [np.broadcast_to(a, target) if a.size == 1 else a for a in arrs]


def _tail(args: list, pos: int, name: str, allowed=("lower", "upper")) -> str:
    if len(args) <= pos:
        return "lower"
    t = text(args[pos]).lower() if is_text(args[pos]) else ""
    if t not in allowed:
        raise MatlabError(f"{name}: the tail must be one of: " + ", ".join(f"'{a}'" for a in allowed) + ".")
    return t


# -- error, gamma and beta functions --------------------------------------------------------------

@builtin("erfcx")
def _erfcx(interp, args, nargout):
    nargs(args, 1, 1, "erfcx")
    x = _real_arg(args[0], "erfcx")
    return _res(_sp().erfcx(x), _is_single(args[0]))


@builtin("betaln")
def _betaln(interp, args, nargout):
    nargs(args, 2, 2, "betaln")
    z, w = _bcast("betaln", _real_arg(args[0], "betaln"), _real_arg(args[1], "betaln"))
    if np.any(z < 0) or np.any(w < 0):
        raise MatlabError("betaln: inputs must be nonnegative.")
    with np.errstate(all="ignore"):
        return _res(_sp().betaln(z, w), _is_single(*args))


@builtin("betaincinv")
def _betaincinv(interp, args, nargout):
    nargs(args, 3, 4, "betaincinv")
    y, z, w = _bcast("betaincinv", *(_real_arg(a, "betaincinv") for a in args[:3]))
    tail = _tail(args, 3, "betaincinv")
    if np.any((y < 0) | (y > 1)):
        raise MatlabError("betaincinv: Y must be in the interval [0, 1].")
    if np.any(z < 0) or np.any(w < 0):
        raise MatlabError("betaincinv: Z and W must be nonnegative.")
    sp = _sp()
    with np.errstate(all="ignore"):
        r = sp.betaincinv(z, w, y) if tail == "lower" else sp.betainccinv(z, w, y)
    return _res(r, _is_single(*args[:3]))


@builtin("gammaincinv")
def _gammaincinv(interp, args, nargout):
    nargs(args, 2, 3, "gammaincinv")
    y, a = _bcast("gammaincinv", _real_arg(args[0], "gammaincinv"), _real_arg(args[1], "gammaincinv"))
    tail = _tail(args, 2, "gammaincinv")
    if np.any((y < 0) | (y > 1)):
        raise MatlabError("gammaincinv: Y must be in the interval [0, 1].")
    if np.any(a < 0):
        raise MatlabError("gammaincinv: A must be nonnegative.")
    sp = _sp()
    with np.errstate(all="ignore"):
        r = sp.gammaincinv(a, y) if tail == "lower" else sp.gammainccinv(a, y)
    return _res(r, _is_single(*args[:2]))


# -- Bessel and elliptic functions ---------------------------------------------------------------

@builtin("besselh")
def _besselh(interp, args, nargout):
    nargs(args, 2, 4, "besselh")
    if len(args) == 2:
        nu_v, kind, z_v, scale = args[0], 1, args[1], 0
    else:
        nu_v, z_v = args[0], args[2]
        kind = to_int(args[1], "K")
        scale = to_int(args[3], "scale") if len(args) > 3 else 0
    if kind not in (1, 2):
        raise MatlabError("besselh: K must be 1 or 2.")
    if scale not in (0, 1):
        raise MatlabError("besselh: scale must be 0 or 1.")
    nu = _real_arg(nu_v, "besselh")
    z = fnum(z_v).astype(complex)
    nu, z = _bcast("besselh", nu, z)
    sp = _sp()
    fn = {(1, 0): sp.hankel1, (1, 1): sp.hankel1e, (2, 0): sp.hankel2, (2, 1): sp.hankel2e}[(kind, scale)]
    with np.errstate(all="ignore"):
        r = fn(nu, z)
    r = np.asarray(r, dtype=complex)
    return r.astype(np.complex64) if _is_single(nu_v, z_v) else as_2d(r)


@builtin("ellipj")
def _ellipj(interp, args, nargout):
    nargs(args, 2, 3, "ellipj")
    if nargout > 3:
        raise MatlabError("ellipj: too many output arguments.")
    u, m = _bcast("ellipj", _real_arg(args[0], "ellipj"), _real_arg(args[1], "ellipj"))
    if len(args) > 2:
        tol = _real_arg(args[2], "ellipj")
        if tol.size != 1 or not tol.flat[0] > 0:
            raise MatlabError("ellipj: the tolerance must be a positive scalar.")
    if np.any((m < 0) | (m > 1)):
        raise MatlabError("ellipj: M must be in the range 0 <= M <= 1.")
    sn, cn, dn, _ = _sp().ellipj(u, m)
    single = _is_single(*args[:2])
    outs = (_res(sn, single), _res(cn, single), _res(dn, single))
    return outs[:max(nargout, 1)] if nargout > 1 else outs[0]


def _legendre_norm(n: int, x: np.ndarray) -> np.ndarray:
    """Fully normalized associated Legendre functions of degree n, orders 0..n (no phase factor).

    Row m holds  sqrt((n + 1/2) (n-m)! / (n+m)!) * (1-x^2)^(m/2) * d^m/dx^m P_n(x).
    """
    s = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    out = np.zeros((n + 1,) + x.shape)
    pmm = np.full(x.shape, math.sqrt(0.5))  # order m, degree m
    for m in range(n + 1):
        if m > 0:
            pmm = math.sqrt((2 * m + 1) / (2.0 * m)) * s * pmm
        if m == n:
            out[m] = pmm
            break
        prev2 = pmm
        prev1 = math.sqrt(2 * m + 3.0) * x * pmm  # degree m + 1
        for k in range(m + 2, n + 1):
            a = math.sqrt((4.0 * k * k - 1.0) / (k * k - m * m))
            b = math.sqrt(((k - 1.0) ** 2 - m * m) / (4.0 * (k - 1.0) ** 2 - 1.0))
            prev2, prev1 = prev1, a * (x * prev1 - b * prev2)
        out[m] = prev1
    return out


@builtin("legendre")
def _legendre(interp, args, nargout):
    nargs(args, 2, 3, "legendre")
    n = to_int(args[0], "n")
    if n < 0:
        raise MatlabError("legendre: N must be a nonnegative integer.")
    x = _real_arg(args[1], "legendre")
    if np.any(np.abs(x) > 1):
        raise MatlabError("legendre: X must be real and in the range [-1, 1].")
    kind = "unnorm"
    if len(args) > 2:
        kind = text(args[2]).lower() if is_text(args[2]) else ""
        if kind not in ("unnorm", "sch", "norm"):
            raise MatlabError("legendre: the normalization must be 'unnorm', 'sch' or 'norm'.")
    # a vector X gives one column per point; an array X adds a leading dimension
    pts = x.reshape(-1) if (x.ndim == 2 and 1 in x.shape) else x
    nrm = _legendre_norm(n, pts)
    m = np.arange(n + 1).reshape((n + 1,) + (1,) * pts.ndim)
    if kind == "norm":
        out = nrm
    elif kind == "sch":
        out = nrm * np.where(m == 0, 1.0 / math.sqrt(n + 0.5), 2.0 / math.sqrt(2 * n + 1.0))
    else:
        # P_n^m = (-1)^m * N_n^m / sqrt((n + 1/2) (n-m)! / (n+m)!)
        lg = np.array([math.lgamma(n + k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)])
        with np.errstate(over="ignore"):
            scale = np.exp(0.5 * (lg - math.log(n + 0.5))) * (-1.0) ** np.arange(n + 1)
        with np.errstate(over="ignore", invalid="ignore"):
            out = nrm * scale.reshape(m.shape)
        if n <= 20:
            out = _round_if_integral(out)
    out = np.asarray(out)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return _res(as_2d(out), _is_single(args[1]))


def _round_if_integral(a: np.ndarray) -> np.ndarray:
    """Remove rounding noise of the size of a few ulps around values that should be exact."""
    r = np.round(a * 1e12) / 1e12
    close = np.abs(a - r) <= 4 * np.spacing(np.abs(a))
    return np.where(close, r, a)


# -- trigonometry in degrees and in multiples of pi --------------------------------------------------

def _deg_fn(name: str, fn, radian_fn):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        a = fnum(args[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            if a.dtype.kind == "c":
                return _res(radian_fn(np.deg2rad(a)))
            return _res(fn(a.astype(float)), _is_single(a))
    return f


def _secd(a):
    from .mathfns import _cosd
    return 1.0 / _cosd(a)


def _cscd(a):
    from .mathfns import _sind
    return 1.0 / _sind(a)


def _cotd(a):
    from .mathfns import _cosd, _sind
    s, c = _sind(a), _cosd(a)
    return np.where(c == 0, 0.0, c / s)


builtin("secd")(_deg_fn("secd", _secd, lambda z: 1 / np.cos(z)))
builtin("cscd")(_deg_fn("cscd", _cscd, lambda z: 1 / np.sin(z)))
builtin("cotd")(_deg_fn("cotd", _cotd, lambda z: 1 / np.tan(z)))


def _inverse_deg(name: str, fn):
    def f(interp, args, nargout):
        nargs(args, 1, 1, name)
        a = fnum(args[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = fn(a.astype(complex) if a.dtype.kind == "c" else a.astype(float))
        return _res(np.asarray(r) * (180.0 / np.pi), _is_single(a))
    return f


builtin("asecd")(_inverse_deg("asecd", lambda x: np.emath.arccos(1.0 / x)))
builtin("acscd")(_inverse_deg("acscd", lambda x: np.emath.arcsin(1.0 / x)))
builtin("acotd")(_inverse_deg("acotd", lambda x: np.arctan(1.0 / x)))


def _pi_fn(name: str, is_sin: bool):
    def f(interp, args, nargout):
        from .mathfns import _cosd, _sind
        nargs(args, 1, 1, name)
        a = fnum(args[0])
        if a.dtype.kind == "c":
            return _res((np.sin if is_sin else np.cos)(np.pi * a))
        x = a.astype(float)
        # sin(pi x) = sin of (180 x) degrees: exact zeros at the integers, exact 1/2 at x = 1/6
        with np.errstate(invalid="ignore"):
            deg = np.fmod(x, 2.0) * 180.0
        return _res((_sind if is_sin else _cosd)(deg), _is_single(a))
    return f


builtin("sinpi")(_pi_fn("sinpi", True))
builtin("cospi")(_pi_fn("cospi", False))


# -- real-only elementary functions ------------------------------------------------------------------

@builtin("reallog")
def _reallog(interp, args, nargout):
    nargs(args, 1, 1, "reallog")
    a = fnum(args[0])
    if a.dtype.kind == "c" or np.any(a < 0):
        raise MatlabError("Reallog produced complex result.", "MATLAB:reallog:complexResult")
    with np.errstate(divide="ignore"):
        return _res(np.log(a))


@builtin("realpow")
def _realpow(interp, args, nargout):
    nargs(args, 2, 2, "realpow")
    x, y = fnum(args[0]), fnum(args[1])
    if x.dtype.kind == "c" or y.dtype.kind == "c":
        raise MatlabError("Realpow produced complex result.", "MATLAB:realpow:complexResult")
    from ..ops import binop
    r = binop(".^", args[0], args[1])
    if isinstance(r, np.ndarray) and r.dtype.kind == "c":
        raise MatlabError("Realpow produced complex result.", "MATLAB:realpow:complexResult")
    return r


# -- reductions and array helpers ---------------------------------------------------------------------

def _empty(v: Any) -> bool:
    return isinstance(v, np.ndarray) and v.size == 0


def _split_nanflag(args: list, name: str) -> tuple[list, bool]:
    """Remove a trailing NaN flag; True means NaNs are left out."""
    rest = list(args)
    omit = False
    while rest and is_text(rest[-1]) and text(rest[-1]).lower() in (
            "omitnan", "includenan", "omitmissing", "includemissing"):
        omit = text(rest.pop()).lower().startswith("omit")
    return rest, omit


def _dims_of(a: np.ndarray, spec: list, name: str) -> list[int]:
    """Operating axes from an optional dim / vecdim / 'all' argument."""
    if not spec:
        if a.shape == (0, 0):
            return [0, 1]
        return [default_dim(a)]
    v = spec[0]
    if is_text(v):
        if text(v).lower() != "all":
            raise MatlabError(f"{name}: unknown option '{text(v)}'.")
        return list(range(a.ndim))
    d = np.real(fnum(v)).reshape(-1)
    if d.size == 0 or np.any(d < 1) or np.any(d != np.round(d)) or np.unique(d).size != d.size:
        raise MatlabError(f"{name}: the dimension must be a positive integer or a vector of unique positive "
                          "integers.")
    return [int(x) - 1 for x in d]


def _reduce(a: np.ndarray, axes: list[int], fn) -> np.ndarray:
    axes = tuple(ax for ax in axes if ax < a.ndim)
    if not axes:
        return fn(a.reshape(a.shape + (1,)), (a.ndim,))
    return as_2d(np.asarray(fn(a, axes)))


@builtin("rms")
def _rms(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for rms.")
    rest, omit = _split_nanflag(args, "rms")
    if len(rest) > 2:
        raise MatlabError("Too many input arguments for rms.")
    a = fnum(rest[0])
    single = _is_single(a)
    work = np.abs(a.astype(complex if a.dtype.kind == "c" else float)) ** 2
    axes = _dims_of(work, rest[1:], "rms")

    def f(x, ax):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nanmean(x, axis=ax, keepdims=True) if omit else np.mean(x, axis=ax, keepdims=True)
        return np.sqrt(m)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = _reduce(work, axes, f)
    return out.astype(np.float32) if single else out


@builtin("topkrows")
def _topkrows(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for topkrows.")
    src = args[0]
    k = to_int(args[1], "k")
    if k < 0:
        raise MatlabError("topkrows: k must be a nonnegative integer.")
    rest = list(args[2:])
    method = "auto"
    if len(rest) >= 2 and is_text(rest[-2]) and text(rest[-2]).lower() == "comparisonmethod":
        method = text(rest[-1]).lower()
        if method not in ("auto", "real", "abs"):
            raise MatlabError("topkrows: ComparisonMethod must be 'auto', 'real' or 'abs'.")
        rest = rest[:-2]
    if hasattr(src, "names") and hasattr(src, "columns"):  # a table: sort it and keep the first rows
        if method != "auto":
            raise MatlabError("topkrows: ComparisonMethod is not supported for tables.")
        direction = [rest[1]] if len(rest) > 1 else [char("descend")]
        key = [rest[0]] if rest and not _empty(rest[0]) else [Cell.from_list([char(n) for n in src.names])]
        res = interp.call_named("sortrows", [src, *key, *direction], 2, interp.frame)
        count = min(k, src.nrows)
        idx = np.arange(1.0, count + 1).reshape(-1, 1)
        top = res[0].index_get(interp, [idx, char(":")])
        order = np.asarray(res[1]).reshape(-1, 1)[:count]
        return (top, order) if nargout > 1 else top
    a = src if isinstance(src, np.ndarray) else num(src)
    if a.ndim != 2:
        raise MatlabError("topkrows: the input must be a matrix or a table.")
    work = num(a)
    ncols = work.shape[1]
    signs = np.ones(ncols)
    cols = list(range(ncols))
    if rest and not _empty(rest[0]):
        signs = np.real(fnum(rest[0])).reshape(-1)
        if np.any(signs == 0) or np.any(np.abs(signs) > ncols) or np.any(signs != np.round(signs)):
            raise MatlabError("topkrows: column indices must be nonzero integers within the number of columns.")
        cols = [int(abs(c)) - 1 for c in signs]
    # the default direction is descending; a negative column index reverses it for that column
    desc = _direction_list(rest[1], len(cols), "topkrows") if len(rest) > 1 else [True] * len(cols)
    if len(rest) > 2:
        raise MatlabError("Too many input arguments for topkrows.")
    desc = [d if sgn > 0 else not d for d, sgn in zip(desc, signs)]
    order = _row_order(work, cols, desc, method)[:k]
    out = a[order, :]
    return (out, (order + 1.0).reshape(-1, 1)) if nargout > 1 else out


@builtin("issortedrows")
def _issortedrows(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for issortedrows.")
    src = args[0]
    rest = list(args[1:])
    method, missing = "auto", "auto"
    while len(rest) >= 2 and is_text(rest[-2]) and text(rest[-2]).lower() in ("comparisonmethod",
                                                                              "missingplacement"):
        key, val = text(rest[-2]).lower(), text(rest[-1]).lower()
        if key == "comparisonmethod":
            if val not in ("auto", "real", "abs"):
                raise MatlabError("issortedrows: ComparisonMethod must be 'auto', 'real' or 'abs'.")
            method = val
        else:
            if val not in ("auto", "first", "last"):
                raise MatlabError("issortedrows: MissingPlacement must be 'auto', 'first' or 'last'.")
            missing = val
        rest = rest[:-2]
    if hasattr(src, "names") and hasattr(src, "columns"):
        res = interp.call_named("sortrows", [src, *rest], 2, interp.frame)
        idx = np.asarray(res[1]).reshape(-1)
        return np.array([[bool(np.array_equal(idx, np.arange(1, idx.size + 1)))]])
    a = num(src)
    if a.ndim != 2:
        raise MatlabError("issortedrows: the input must be a matrix or a table.")
    ncols = a.shape[1]
    cols = list(range(ncols))
    desc = [False] * ncols
    strict = False
    mono = False
    for r in rest:
        if is_text(r) or isinstance(r, (Cell, StringArray)):
            names = [to_str(x) for x in r.values_f()] if isinstance(r, Cell) else \
                ([str(x) for x in r.values_f()] if isinstance(r, StringArray) else [text(r)])
            low = [s.lower() for s in names]
            if len(low) == 1 and low[0] in ("monotonic", "strictmonotonic", "strictascend", "strictdescend"):
                strict = low[0].startswith("strict")
                mono = low[0].endswith("monotonic")
                desc = [low[0].endswith("descend")] * len(cols)
            else:
                desc = _direction_list(r, len(cols), "issortedrows")
        elif not _empty(r):
            spec = np.real(fnum(r)).reshape(-1)
            if np.any(spec == 0) or np.any(np.abs(spec) > ncols) or np.any(spec != np.round(spec)):
                raise MatlabError("issortedrows: column indices must be nonzero integers within the number "
                                  "of columns.")
            cols = [int(abs(c)) - 1 for c in spec]
            desc = [c < 0 for c in spec]

    def check(d):
        order = _row_order(a, cols, d, method, missing)
        ok = bool(np.array_equal(order, np.arange(a.shape[0])))
        if ok and strict and a.shape[0] > 1:
            sub = a[:, cols]
            ok = not bool(np.any(np.all(sub[1:] == sub[:-1], axis=1)))
        return ok

    result = check(desc) or (mono and check([not x for x in desc]))
    return np.array([[bool(result)]])


def _nice_edges(lo: float, hi: float, nb: int) -> np.ndarray:
    """nb + 1 equally spaced edges with a round bin width (1, 2, 2.5 or 5 times a power of ten) that cover [lo, hi]."""
    raw = (hi - lo) / nb
    exp = int(np.floor(np.log10(raw)))
    for e in (exp, exp + 1, exp + 2):
        for mant in (1.0, 2.0, 2.5, 5.0):
            w = mant * 10.0 ** e
            if w < raw * (1 - 1e-12):
                continue
            left = np.floor(lo / w) * w
            if left + nb * w >= hi * (1 - 1e-12 * np.sign(hi)):
                return left + w * np.arange(nb + 1)
    return np.linspace(lo, hi, nb + 1)


def discretize_numeric(interp, args, nargout):
    """discretize for plain numeric X and edges; the discretize built-in (tables_more) calls this."""
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for discretize.")
    x_src = args[0]
    if not isinstance(x_src, np.ndarray) or x_src.dtype.kind not in "biuf":
        raise MatlabError("discretize: the data must be a real numeric array (datetime and duration data "
                          "are not supported).")
    x = np.real(fnum(x_src)).astype(float)
    spec = args[1]
    if not isinstance(spec, np.ndarray) or is_char(spec):
        raise MatlabError("discretize: the second argument must be a vector of bin edges or a number of bins.")
    rest = list(args[2:])
    right = False
    if len(rest) >= 2 and is_text(rest[-2]) and text(rest[-2]).lower() == "includededge":
        side = text(rest[-1]).lower() if is_text(rest[-1]) else ""
        if side not in ("left", "right"):
            raise MatlabError("discretize: IncludedEdge must be 'left' or 'right'.")
        right = side == "right"
        rest = rest[:-2]
    ev = np.real(fnum(spec)).astype(float).reshape(-1)
    if ev.size == 1:
        nb = int(ev[0])
        if nb < 1 or nb != ev[0]:
            raise MatlabError("discretize: the number of bins must be a positive integer.")
        finite = x[np.isfinite(x)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = _nice_edges(lo, hi, nb)
    else:
        edges = ev
        if np.any(np.diff(edges) <= 0) or np.any(np.isnan(edges)):
            raise MatlabError("discretize: bin edges must be strictly increasing.")
        if edges.size < 2:
            raise MatlabError("discretize: at least two bin edges are needed.")
    nbins = edges.size - 1
    if right:
        idx = np.searchsorted(edges, x, side="left").astype(float)
        idx = np.where(x == edges[0], 1.0, idx)
    else:
        idx = np.searchsorted(edges, x, side="right").astype(float)
        idx = np.where(x == edges[-1], float(nbins), idx)
    outside = np.isnan(x) | (x < edges[0]) | (x > edges[-1])
    idx = np.where(outside, np.nan, idx)
    out: Any = idx
    if rest:
        first = rest[0]
        if is_text(first) and text(first).lower() == "categorical":
            if len(rest) > 2:
                raise MatlabError("Too many input arguments for discretize.")
            if len(rest) == 2:
                names_v = rest[1]
                names = [to_str(v) for v in names_v.values_f()] if isinstance(names_v, Cell) else \
                    [str(v) for v in names_v.values_f()] if isinstance(names_v, StringArray) else None
                if names is None or len(names) != nbins:
                    raise MatlabError("discretize: give one category name per bin.")
            else:
                def edge(v):
                    return "%g" % v
                names = []
                for k in range(nbins):
                    last = k == nbins - 1
                    first_bin = k == 0
                    if right:
                        names.append(("[" if first_bin else "(") + f"{edge(edges[k])}, {edge(edges[k + 1])}]")
                    else:
                        names.append(f"[{edge(edges[k])}, {edge(edges[k + 1])}" + ("]" if last else ")"))
            labels = np.empty(idx.shape, dtype=object)
            for i in np.ndindex(idx.shape):
                labels[i] = None if np.isnan(idx[i]) else names[int(idx[i]) - 1]
            cats = Cell.from_list([char(n) for n in names])
            out = interp.call_named("categorical", [StringArray(labels), cats, char("Ordinal"),
                                                    np.array([[True]])], 1, interp.frame)[0]
        else:
            if len(rest) > 1:
                raise MatlabError("Too many input arguments for discretize.")
            vals = first
            if isinstance(vals, Cell) or isinstance(vals, StringArray):
                items = vals.values_f()
                if len(items) != nbins:
                    raise MatlabError("discretize: give one value per bin.")
                data = np.empty(idx.shape, dtype=object)
                missing = None if isinstance(vals, StringArray) else np.zeros((0, 0), dtype="<U1")
                for i in np.ndindex(idx.shape):
                    data[i] = missing if np.isnan(idx[i]) else items[int(idx[i]) - 1]
                out = StringArray(data) if isinstance(vals, StringArray) else Cell(data)
            else:
                v = vals if isinstance(vals, np.ndarray) else num(vals)
                flat = v.reshape(-1, order="F")
                if flat.size != nbins:
                    raise MatlabError("discretize: give one value per bin.")
                safe = np.where(np.isnan(idx), 1, idx).astype(int) - 1
                picked = flat[safe]
                if flat.dtype.kind in "fc":
                    picked = np.where(np.isnan(idx), np.nan, picked)
                elif np.isnan(idx).any():
                    picked = np.where(np.isnan(idx), 0, picked).astype(flat.dtype)
                out = picked
    if nargout > 1:
        return out, edges.reshape(1, -1)
    return out


@builtin("histcounts2")
def _histcounts2(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for histcounts2.")
    if nargout > 5:
        raise MatlabError("histcounts2: too many output arguments.")
    x = np.real(fnum(args[0])).astype(float)
    y = np.real(fnum(args[1])).astype(float)
    if x.shape != y.shape:
        raise MatlabError("histcounts2: X and Y must have the same size.")
    rest = list(args[2:])
    pos = []
    while rest and not is_text(rest[0]):
        pos.append(rest.pop(0))
    if len(rest) % 2:
        raise MatlabError("histcounts2: options must be name-value pairs.")
    opts = {}
    known = {"xbinedges", "ybinedges", "numbins", "binwidth", "xbinlimits", "ybinlimits", "binmethod",
             "normalization"}
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower()
        if key not in known:
            raise MatlabError(f"histcounts2: unknown option '{text(rest[k])}'.")
        opts[key] = rest[k + 1]
    xe = ye = None
    nb = None
    if len(pos) == 1:
        v = np.real(fnum(pos[0])).reshape(-1)
        if v.size not in (1, 2):
            raise MatlabError("histcounts2: the number of bins must be a scalar or a two-element vector.")
        nb = (int(v[0]), int(v[-1]))
    elif len(pos) == 2:
        xe = np.real(fnum(pos[0])).astype(float).reshape(-1)
        ye = np.real(fnum(pos[1])).astype(float).reshape(-1)
    elif len(pos) > 2:
        raise MatlabError("Too many input arguments for histcounts2.")
    if "xbinedges" in opts:
        xe = np.real(fnum(opts["xbinedges"])).astype(float).reshape(-1)
    if "ybinedges" in opts:
        ye = np.real(fnum(opts["ybinedges"])).astype(float).reshape(-1)
    if "numbins" in opts:
        v = np.real(fnum(opts["numbins"])).reshape(-1)
        nb = (int(v[0]), int(v[-1]))
    widths = None
    if "binwidth" in opts:
        v = np.real(fnum(opts["binwidth"])).astype(float).reshape(-1)
        widths = (float(v[0]), float(v[-1]))
    method = text(opts["binmethod"]).lower() if "binmethod" in opts else "auto"
    if method not in ("auto", "scott", "fd", "integers"):
        raise MatlabError("histcounts2: BinMethod must be 'auto', 'scott', 'fd' or 'integers'.")
    hc = BUILTINS["histcounts"]

    def edges_for(data, given, lim_key, k):
        if given is not None:
            if given.size < 2 or np.any(np.diff(given) < 0):
                raise MatlabError("histcounts2: bin edges must be increasing vectors with at least two elements.")
            return given
        a = [data.reshape(1, -1)]
        if nb is not None:
            a.append(np.array([[float(nb[k])]]))
        if widths is not None:
            a += [char("BinWidth"), np.array([[widths[k]]])]
        if lim_key in opts:
            a += [char("BinLimits"), opts[lim_key]]
        if method != "auto" and nb is None and widths is None:
            a += [char("BinMethod"), char(method)]
        return np.asarray(hc(interp, a, 2)[1], dtype=float).reshape(-1)

    xe = edges_for(x, xe, "xbinlimits", 0)
    ye = edges_for(y, ye, "ybinlimits", 1)
    xf, yf = x.reshape(-1, order="F"), y.reshape(-1, order="F")

    def bin_of(v, e):
        b = np.searchsorted(e, v, side="right")
        b = np.where(v == e[-1], e.size - 1, b)
        return np.where(np.isnan(v) | (v < e[0]) | (v > e[-1]), 0, b)

    bx, by = bin_of(xf, xe), bin_of(yf, ye)
    inside = (bx > 0) & (by > 0)
    counts = np.zeros((xe.size - 1, ye.size - 1))
    np.add.at(counts, (bx[inside] - 1, by[inside] - 1), 1.0)
    norm = text(opts["normalization"]).lower() if "normalization" in opts else "count"
    total = float(inside.sum())
    area = np.outer(np.diff(xe), np.diff(ye))
    with np.errstate(divide="ignore", invalid="ignore"):
        if norm == "count":
            out = counts
        elif norm == "probability":
            out = counts / total
        elif norm == "percentage":
            out = 100.0 * counts / total
        elif norm == "countdensity":
            out = counts / area
        elif norm == "pdf":
            out = counts / (total * area)
        elif norm == "cumcount":
            out = np.cumsum(np.cumsum(counts, axis=0), axis=1)
        elif norm == "cdf":
            out = np.cumsum(np.cumsum(counts, axis=0), axis=1) / total
        else:
            raise MatlabError(f"histcounts2: unknown normalization '{norm}'.")
    bx_out = np.where(inside, bx, 0).astype(float).reshape(x.shape, order="F")
    by_out = np.where(inside, by, 0).astype(float).reshape(x.shape, order="F")
    outs = (out, xe.reshape(1, -1), ye.reshape(1, -1), bx_out, by_out)
    return outs[:max(nargout, 1)] if nargout > 1 else out


def _moving_wrapper(name: str, reducer):
    def f(interp, args, nargout):
        from .stats import _moving
        return _moving(name, reducer)(interp, args, nargout)
    return f


def _prod_red(w, omit, weight):
    return np.nanprod(w, axis=0) if omit else np.prod(w, axis=0)


def _mad_red(w, omit, weight):
    med = (np.nanmedian if omit else np.median)(w, axis=0)
    return (np.nanmedian if omit else np.median)(np.abs(w - med), axis=0)


builtin("movprod")(_moving_wrapper("movprod", _prod_red))
builtin("movmad")(_moving_wrapper("movmad", _mad_red))


@builtin("movfun")
def _movfun(interp, args, nargout):
    if len(args) < 3:
        raise MatlabError("movfun expects a function, the data and a window length.")
    fun = args[0]
    if not isinstance(fun, FunctionHandle) and not is_text(fun):
        raise MatlabError("movfun: the first argument must be a function handle.")
    x = fnum(args[1])
    from .stats import _window_extent
    before, after = _window_extent(args[2])
    rest = list(args[3:])
    if len(rest) % 2:
        raise MatlabError("movfun: options must be name-value pairs.")
    endpoints: Any = "shrink"
    dim = None
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower() if is_text(rest[k]) else ""
        val = rest[k + 1]
        if key == "endpoints":
            if is_text(val):
                endpoints = text(val).lower()
                if endpoints not in ("shrink", "fill", "periodic", "same"):
                    raise MatlabError("movfun: Endpoints must be 'shrink', 'fill', 'periodic', 'same' or a value.")
            else:
                endpoints = complex(fnum(val).flat[0]) if fnum(val).dtype.kind == "c" else float(fnum(val).flat[0])
        elif key == "dim":
            dim = to_int(val, "dim") - 1
        else:
            raise MatlabError(f"movfun: unknown option '{text(rest[k]) if is_text(rest[k]) else '?'}'.")
    axis = dim if dim is not None else default_dim(x)
    if x.size == 0 or axis >= x.ndim:
        return x
    xm = np.moveaxis(x, axis, 0)
    n = xm.shape[0]
    lead = xm.shape[1:]
    cols = xm.reshape(n, -1)
    out = np.zeros(cols.shape, dtype=complex if cols.dtype.kind == "c" else float)
    is_complex = False
    for c in range(cols.shape[1]):
        col = cols[:, c]
        for i in range(n):
            lo, hi = i - before, i + after
            if endpoints == "shrink":
                win = col[max(lo, 0):min(hi, n - 1) + 1]
            else:
                idx = np.arange(lo, hi + 1)
                inside = (idx >= 0) & (idx < n)
                if endpoints == "periodic":
                    win = col[np.mod(idx, n)]
                elif endpoints == "same":
                    win = col[np.clip(idx, 0, n - 1)]
                else:
                    fill = np.nan if endpoints == "fill" else endpoints
                    win = np.where(inside, col[np.clip(idx, 0, n - 1)], fill)
            res = interp.call_function_value(fun, [np.asarray(win).reshape(-1, 1)], 1)
            val = fnum(res[0])
            if val.size != 1:
                raise MatlabError("movfun: the function must return a scalar for each window.")
            v = val.flat[0]
            if isinstance(v, complex) or np.iscomplexobj(val):
                is_complex = is_complex or np.imag(v) != 0
            out[i, c] = v
    res = out.reshape((n,) + lead)
    res = np.moveaxis(res, 0, axis)
    return simplify(res) if is_complex or res.dtype.kind == "c" else res


def _outlier_dim(rest: list) -> int | None:
    """The operating dimension (0-based) among isoutlier-style arguments, or None."""
    pair_names = {"thresholdfactor", "samplepoints", "maxnumoutliers", "datavariables", "replacevalues"}
    k = 0
    while k < len(rest):
        r = rest[k]
        if is_text(r):
            key = text(r).lower()
            if key in pair_names or key in ("percentiles", "movmedian", "movmean"):
                k += 2  # the name and its value (a threshold pair or a window length)
                continue
            k += 1
            continue
        if isinstance(r, np.ndarray) and r.size == 1:
            return int(np.real(fnum(r)).flat[0]) - 1
        k += 1
    return None


@builtin("rmoutliers")
def _rmoutliers(interp, args, nargout):
    if not args:
        raise MatlabError("Not enough input arguments for rmoutliers.")
    a = args[0]
    if not isinstance(a, np.ndarray):
        raise MatlabError("rmoutliers: the input must be a numeric array.")
    rest = list(args[1:])
    min_count = None
    k = 0
    passed = []
    while k < len(rest):
        if is_text(rest[k]) and text(rest[k]).lower() == "minnumoutliers" and k + 1 < len(rest):
            min_count = to_int(rest[k + 1], "MinNumOutliers")
            k += 2
            continue
        passed.append(rest[k])
        k += 1
    res = interp.call_named("isoutlier", [a, *passed], 4, interp.frame)
    tf = np.asarray(res[0], dtype=bool)
    dim = _outlier_dim(passed)
    vector = a.ndim == 2 and 1 in a.shape
    if vector and dim is None:
        keep = ~tf.reshape(-1, order="F")
        flat = a.reshape(-1, order="F")[keep]
        out = flat.reshape(1, -1) if a.shape[0] == 1 else flat.reshape(-1, 1)
        removed = (~keep).reshape(a.shape, order="F")
    else:
        if a.ndim != 2:
            raise MatlabError("rmoutliers: the input must be a vector or a matrix.")
        # outliers are found along the operating dimension; whole rows (dim 1) or columns (dim 2) go
        along = 0 if dim is None else dim
        if along not in (0, 1):
            raise MatlabError("rmoutliers: the dimension must be 1 or 2.")
        need = 1 if min_count is None else min_count
        bad = tf.sum(axis=1 - along) >= need
        out = a[~bad, :] if along == 0 else a[:, ~bad]
        removed = bad.reshape(-1, 1) if along == 0 else bad.reshape(1, -1)
    outs = (out, removed, *res[1:4])
    return outs[:max(nargout, 1)] if nargout > 1 else out


@builtin("filloutliers")
def _filloutliers(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("filloutliers expects the data and a fill method.")
    a = args[0]
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "f":
        raise MatlabError("filloutliers: the input must be a single or double array.")
    fill = args[1]
    rest = list(args[2:])
    res = interp.call_named("isoutlier", [a, *rest], 4, interp.frame)
    tf = np.asarray(res[0], dtype=bool)
    lower, upper, center = (np.asarray(r, dtype=float) for r in res[1:4])
    out = a.astype(float).copy()
    dim = _outlier_dim(rest)
    axis = default_dim(a) if dim is None else dim
    if is_text(fill):
        how = text(fill).lower()
        if how == "center":
            out = np.where(tf, np.broadcast_to(center, out.shape), out)
        elif how == "clip":
            out = np.where(tf & (out < lower), np.broadcast_to(lower, out.shape), out)
            out = np.where(tf & (out > upper), np.broadcast_to(upper, out.shape), out)
        elif how in ("previous", "next", "nearest", "linear", "spline", "pchip", "makima"):
            holes = np.where(tf, np.nan, out)
            filled = interp.call_named("fillmissing", [holes, char(how), np.array([[axis + 1.0]])], 1,
                                       interp.frame)[0]
            # values that were NaN to begin with stay NaN
            out = np.where(np.isnan(a) & ~tf, np.nan, np.asarray(filled, dtype=float))
        else:
            raise MatlabError(f"filloutliers: unknown fill method '{text(fill)}'.")
    else:
        v = fnum(fill)
        if v.size != 1:
            raise MatlabError("filloutliers: a fill value must be a scalar.")
        out = np.where(tf, float(np.real(v.flat[0])), out)
    out = out.astype(a.dtype)
    outs = (out, tf, lower, upper, center)
    return outs[:max(nargout, 1)] if nargout > 1 else out


@builtin("ismembertol")
def _ismembertol(interp, args, nargout):
    if len(args) < 2:
        raise MatlabError("Not enough input arguments for ismembertol.")
    a, b = fnum(args[0]), fnum(args[1])
    if a.dtype.kind == "c" or b.dtype.kind == "c":
        raise MatlabError("ismembertol: inputs must be real.")
    rest = list(args[2:])
    single = _is_single(args[0], args[1])
    tol = 1e-6 if single else 1e-12
    if rest and not is_text(rest[0]):
        tol = float(np.real(fnum(rest.pop(0))).flat[0])
    if len(rest) % 2:
        raise MatlabError("ismembertol: options must be name-value pairs.")
    by_rows = False
    all_idx = False
    scale = None
    for k in range(0, len(rest), 2):
        key = text(rest[k]).lower() if is_text(rest[k]) else ""
        val = rest[k + 1]
        if key == "byrows":
            by_rows = bool(np.all(np.asarray(fnum(val)) != 0))
        elif key == "outputallindices":
            all_idx = bool(np.all(np.asarray(fnum(val)) != 0))
        elif key == "datascale":
            scale = np.real(fnum(val)).astype(float).reshape(-1)
        else:
            raise MatlabError(f"ismembertol: unknown option '{text(rest[k]) if is_text(rest[k]) else '?'}'.")
    a, b = a.astype(float), b.astype(float)
    if by_rows:
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise MatlabError("ismembertol: with ByRows the inputs must have the same number of columns.")
        both = np.vstack([a, b])
        fin = np.where(np.isfinite(both), np.abs(both), 0.0)
        ds = fin.max(axis=0) if scale is None else np.broadcast_to(scale, (a.shape[1],)).astype(float)
        bound = tol * ds
        lia = np.zeros((a.shape[0], 1), dtype=bool)
        loc: list = []
        for i in range(a.shape[0]):
            hit = np.nonzero(np.all(np.abs(b - a[i]) <= bound, axis=1))[0]
            lia[i, 0] = hit.size > 0
            loc.append(hit)
        shape = (a.shape[0], 1)
    else:
        both = np.concatenate([a.reshape(-1), b.reshape(-1)])
        fin = np.abs(both[np.isfinite(both)])
        ds = (float(fin.max()) if fin.size else 0.0) if scale is None else float(scale[0])
        bound = tol * ds
        bf = b.reshape(-1, order="F")
        af = a.reshape(-1, order="F")
        lia_flat = np.zeros(af.size, dtype=bool)
        loc = []
        for i, v in enumerate(af):
            hit = np.nonzero(np.abs(bf - v) <= bound)[0]
            lia_flat[i] = hit.size > 0
            loc.append(hit)
        lia = lia_flat.reshape(a.shape, order="F")
        shape = a.shape
    if nargout <= 1:
        return lia
    if all_idx:
        cells = np.empty(shape, dtype=object)
        flat = cells.reshape(-1, order="F")
        for i, hit in enumerate(loc):
            flat[i] = (hit + 1.0).reshape(-1, 1) if hit.size else np.zeros((1, 1))
        return lia, Cell(flat.reshape(shape, order="F"))
    first = np.array([float(h[0] + 1) if h.size else 0.0 for h in loc]).reshape(shape, order="F")
    return lia, first
