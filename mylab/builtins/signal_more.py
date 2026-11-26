"""More signal processing: filter design and analysis, spectra, transforms, windows, waveforms and
signal measurements. The FFT, the basic windows and a few core functions live in signal.py."""

from __future__ import annotations

import numpy as np

from ..errors import MatlabError
from ..objects import MObject
from ..values import Cell, FunctionHandle, Struct, char, simplify, to_int, to_matlab
from ._registry import BUILTINS, builtin
from ._util import fnum, is_text, nargs, text


# -- helpers --------------------------------------------------------------------------------------

def _sig():
    import scipy.signal as ss
    return ss


def _given(args: list, i: int) -> bool:
    """True when argument i exists and is not the empty placeholder []."""
    return len(args) > i and not (isinstance(args[i], np.ndarray) and args[i].size == 0 and not is_text(args[i]))


def _sc(v, what: str = "argument") -> float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    return float(np.real(a.flat[0]))


def _vec(v, what: str = "argument") -> np.ndarray:
    """A numeric input as a 1-D array (column-major order)."""
    if isinstance(v, (Cell, Struct, FunctionHandle)) or isinstance(v, MObject):
        raise MatlabError(f"Expected {what} to be numeric.")
    a = fnum(v)
    return a.flatten(order="F")


def _rvec(v, what: str = "argument") -> np.ndarray:
    a = _vec(v, what)
    return np.real(a).astype(float)


def _isvec(a: np.ndarray) -> bool:
    return a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1)


def _row(v) -> np.ndarray:
    return np.asarray(v).reshape(1, -1)


def _col(v) -> np.ndarray:
    return np.asarray(v).reshape(-1, 1)


def _like(r, x: np.ndarray) -> np.ndarray:
    """Give a 1-D result the orientation of vector x (column unless x is a row)."""
    r = np.asarray(r)
    if r.ndim == 1:
        return r.reshape(1, -1) if (x.ndim == 2 and x.shape[0] == 1 and x.shape[1] != 1) else r.reshape(-1, 1)
    return r


def _tidy(r):
    """Drop an imaginary part that is only rounding noise."""
    r = np.asarray(r)
    if np.iscomplexobj(r) and r.size:
        scale = max(float(np.max(np.abs(r))), 1e-300)
        if float(np.max(np.abs(r.imag))) <= 1e-12 * scale:
            return r.real.copy()
    return r


def _split_flags(args: list) -> tuple[list, list[str]]:
    """Separate text arguments (lower-cased) from the rest."""
    rest, flags = [], []
    for a in args:
        if is_text(a):
            flags.append(text(a).lower())
        else:
            rest.append(a)
    return rest, flags


def _name_values(args: list, start: int, known: tuple[str, ...], name: str) -> dict:
    """Name-value pairs from args[start:], keyed by lower-case name; unknown names are errors."""
    lower = {k.lower() for k in known}
    opts: dict = {}
    k = start
    while k < len(args):
        if not is_text(args[k]):
            raise MatlabError(f"{name}: expected an option name at argument {k + 1}.")
        key = text(args[k]).lower()
        if key not in lower:
            raise MatlabError(f"{name}: unknown option '{text(args[k])}'.")
        if k + 1 >= len(args):
            raise MatlabError(f"{name}: the option '{text(args[k])}' needs a value.")
        opts[key] = args[k + 1]
        k += 2
    return opts


def _columns(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Work on columns: a vector becomes one column. Returns (matrix, was_row_vector)."""
    if x.ndim > 2:
        raise MatlabError("Signals must be vectors or matrices.")
    if x.ndim == 2 and x.shape[0] == 1:
        return x.reshape(-1, 1), True
    return x, False


def _restore(y: np.ndarray, was_row: bool) -> np.ndarray:
    return y.T if was_row and y.ndim == 2 and y.shape[1] == 1 else y


def _poly(roots) -> np.ndarray:
    """Polynomial coefficients from roots, real when the roots come in conjugate pairs."""
    roots = np.atleast_1d(np.asarray(roots))
    if roots.size == 0:
        return np.array([1.0])
    c = np.poly(roots)
    return _tidy(c)


def _plot(interp, name: str, args: list) -> None:
    interp.call_named(name, args, 0, interp.frame)


def _f(x) -> np.ndarray:
    return np.array([[float(x)]])


# -- windows --------------------------------------------------------------------------------------

def _win_n(v, name: str) -> int:
    n = _sc(v, "window length")
    if n < 0 or not np.isfinite(n):
        raise MatlabError(f"{name}: the window length must be a non-negative integer.")
    return int(np.floor(n + 0.5))


def _sflag(args: list, pos: int, name: str) -> bool:
    """True for a periodic window ('periodic' at args[pos]); 'symmetric' is the default."""
    if len(args) <= pos:
        return False
    s = text(args[pos]).lower()
    if s not in ("symmetric", "periodic"):
        raise MatlabError(f"{name}: the sampling flag must be 'symmetric' or 'periodic'.")
    return s == "periodic"


def _edge_cases(n: int):
    if n == 0:
        return np.zeros((0, 1))
    if n == 1:
        return np.ones((1, 1))
    return None


def _cos_sum(n: int, coeffs, periodic: bool) -> np.ndarray:
    span = n if periodic else n - 1
    x = 2 * np.pi * np.arange(n) / span
    w = np.zeros(n)
    for k, c in enumerate(coeffs):
        w += (-1) ** k * c * np.cos(k * x)
    return w


def _w_triang(n: int) -> np.ndarray:
    k = np.arange(1, n + 1)
    k = np.minimum(k, n + 1 - k)
    return 2 * k / (n + 1) if n % 2 else (2 * k - 1) / n


def _w_bartlett(n: int) -> np.ndarray:
    k = np.arange(n)
    return 1 - np.abs(2 * k / (n - 1) - 1)


def _w_barthann(n: int) -> np.ndarray:
    x = np.arange(n) / (n - 1) - 0.5
    return 0.62 - 0.48 * np.abs(x) + 0.38 * np.cos(2 * np.pi * x)


def _w_bohman(n: int) -> np.ndarray:
    x = np.abs(np.linspace(-1, 1, n))
    w = (1 - x) * np.cos(np.pi * x) + np.sin(np.pi * x) / np.pi
    w[0] = w[-1] = 0.0
    return w


def _w_parzen(n: int) -> np.ndarray:
    k = np.abs(np.arange(n) - (n - 1) / 2)
    r = k / (n / 2)
    return np.where(k <= (n - 1) / 4, 1 - 6 * r ** 2 + 6 * r ** 3, 2 * (1 - r) ** 3)


def _w_gauss(n: int, alpha: float) -> np.ndarray:
    k = np.arange(n) - (n - 1) / 2
    return np.exp(-0.5 * (alpha * k / ((n - 1) / 2)) ** 2)


def _w_tukey(n: int, r: float) -> np.ndarray:
    if r <= 0:
        return np.ones(n)
    if r >= 1:
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    lo = x < r / 2
    hi = x >= 1 - r / 2
    w[lo] = 0.5 * (1 + np.cos(2 * np.pi / r * (x[lo] - r / 2)))
    w[hi] = 0.5 * (1 + np.cos(2 * np.pi / r * (x[hi] - 1 + r / 2)))
    return w


def _w_taylor(n: int, nbar: int, sll: float) -> np.ndarray:
    a = np.arccosh(10 ** (abs(sll) / 20)) / np.pi
    s2 = nbar ** 2 / (a ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)
    fm = np.zeros(nbar - 1)
    for i, m in enumerate(ma):
        top = (-1) ** (m + 1) * np.prod(1 - m ** 2 / s2 / (a ** 2 + (ma - 0.5) ** 2))
        others = ma[ma != m]
        fm[i] = top / (2 * np.prod(1 - m ** 2 / others ** 2))
    k = np.arange(n)
    return 1 + 2 * (fm[None, :] * np.cos(2 * np.pi * ma[None, :] * (k[:, None] - n / 2 + 0.5) / n)).sum(axis=1)


def _w_cheb(n: int, r: float) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # SciPy remarks on low attenuations; the window is still right
        return _sig().windows.chebwin(n, at=r, sym=True)


def _w_hamming(n: int, periodic: bool = False) -> np.ndarray:
    if n == 1:
        return np.ones(1)
    return _cos_sum(n, (0.54, 0.46), periodic)


def _w_hann(n: int, periodic: bool = False) -> np.ndarray:
    if n == 1:
        return np.ones(1)
    return _cos_sum(n, (0.5, 0.5), periodic)


def _w_kaiser(n: int, beta: float) -> np.ndarray:
    return np.kaiser(n, beta) if n > 1 else np.ones(n)


def _simple_window(name: str, maker):
    def fn(interp, args, nargout):
        nargs(args, 1, 1, name)
        n = _win_n(args[0], name)
        e = _edge_cases(n)
        return e if e is not None else _col(maker(n))
    builtin(name)(fn)


_simple_window("triang", _w_triang)
_simple_window("bartlett", _w_bartlett)
_simple_window("barthannwin", _w_barthann)
_simple_window("bohmanwin", _w_bohman)
_simple_window("parzenwin", _w_parzen)


def _cos_window(name: str, coeffs):
    def fn(interp, args, nargout):
        nargs(args, 1, 2, name)
        n = _win_n(args[0], name)
        periodic = _sflag(args, 1, name)
        e = _edge_cases(n)
        return e if e is not None else _col(_cos_sum(n, coeffs, periodic))
    builtin(name)(fn)


_cos_window("blackmanharris", (0.35875, 0.48829, 0.14128, 0.01168))
_cos_window("nuttallwin", (0.3635819, 0.4891775, 0.1365995, 0.0106411))
_cos_window("flattopwin", (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368))


@builtin("gausswin")
def _gausswin(interp, args, nargout):
    nargs(args, 1, 2, "gausswin")
    n = _win_n(args[0], "gausswin")
    alpha = _sc(args[1], "alpha") if len(args) > 1 else 2.5
    e = _edge_cases(n)
    return e if e is not None else _col(_w_gauss(n, alpha))


@builtin("tukeywin")
def _tukeywin(interp, args, nargout):
    nargs(args, 1, 2, "tukeywin")
    n = _win_n(args[0], "tukeywin")
    r = _sc(args[1], "cosine fraction") if _given(args, 1) else 0.5
    e = _edge_cases(n)
    return e if e is not None else _col(_w_tukey(n, r))


@builtin("chebwin")
def _chebwin(interp, args, nargout):
    nargs(args, 1, 2, "chebwin")
    n = _win_n(args[0], "chebwin")
    r = _sc(args[1], "sidelobe attenuation") if _given(args, 1) else 100.0
    if r < 0:
        raise MatlabError("chebwin: the sidelobe attenuation must be non-negative.")
    e = _edge_cases(n)
    return e if e is not None else _col(_w_cheb(n, r))


@builtin("taylorwin")
def _taylorwin(interp, args, nargout):
    nargs(args, 1, 3, "taylorwin")
    n = _win_n(args[0], "taylorwin")
    nbar = to_int(args[1], "nbar") if _given(args, 1) else 4
    sll = _sc(args[2], "sidelobe level") if _given(args, 2) else -30.0
    if nbar < 1:
        raise MatlabError("taylorwin: nbar must be a positive integer.")
    if sll > 0:
        raise MatlabError("taylorwin: the sidelobe level must be negative (dB below the main lobe).")
    if n == 0:
        return np.zeros((0, 1))
    return _col(_w_taylor(n, nbar, sll))


@builtin("window")
def _window_fn(interp, args, nargout):
    nargs(args, 2, None, "window")
    f = args[0]
    if not isinstance(f, FunctionHandle) and not is_text(f):
        raise MatlabError("window: the first input must be a window function handle such as @hamming.")
    return interp.call_function_value(f, list(args[1:]), 1)[0]


def _window_from(interp, spec, n: int, name: str) -> np.ndarray:
    """A window of length n from [], a vector, or a window name/handle; the default is Hamming."""
    if spec is None or (isinstance(spec, np.ndarray) and spec.size == 0 and not is_text(spec)):
        return _w_hamming(n)
    if isinstance(spec, FunctionHandle) or is_text(spec):
        w = interp.call_function_value(spec, [_f(n)], 1)[0]
        return _rvec(w, "window")
    if isinstance(spec, Cell):
        items = spec.values_f()
        w = interp.call_function_value(items[0], [_f(n), *items[1:]], 1)[0]
        return _rvec(w, "window")
    w = _rvec(spec, "window")
    if w.size != n:
        raise MatlabError(f"{name}: the window must have {n} elements (the filter order plus one).")
    return w


# -- FIR filter design ------------------------------------------------------------------------------

def _sinc_band(m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return hi * np.sinc(hi * m) - lo * np.sinc(lo * m)


def _fir1_core(n: int, wn: np.ndarray, pass_zero: bool, win: np.ndarray, scale: bool) -> np.ndarray:
    edges = np.concatenate([[0.0], wn, [1.0]])
    m = np.arange(n + 1) - n / 2
    h = np.zeros(n + 1)
    first = None
    for k in range(edges.size - 1):
        if (k % 2 == 0) == pass_zero:
            h += _sinc_band(m, edges[k], edges[k + 1])
            if first is None:
                first = (edges[k], edges[k + 1])
    h = h * win
    if scale and first is not None:
        f0 = 0.0 if first[0] == 0 else (1.0 if first[1] == 1 else 0.5 * (first[0] + first[1]))
        gain = abs(np.sum(h * np.exp(-1j * np.pi * f0 * np.arange(n + 1))))
        h = h / gain
    return h


@builtin("fir1")
def _fir1(interp, args, nargout):
    nargs(args, 2, 5, "fir1")
    n = to_int(args[0], "filter order")
    if n < 1:
        raise MatlabError("fir1: the filter order must be a positive integer.")
    wn = _rvec(args[1], "Wn")
    ftype, win_spec, scale = None, None, True
    for a in args[2:]:
        if is_text(a):
            s = text(a).lower()
            if s in ("scale", "noscale"):
                scale = s == "scale"
            elif s in ("low", "high", "bandpass", "stop", "dc-0", "dc-1"):
                ftype = s
            else:
                raise MatlabError(f"fir1: unknown option '{text(a)}'. Use 'low', 'high', 'bandpass', 'stop', "
                                  "'DC-0', 'DC-1', 'scale' or 'noscale'.")
        else:
            win_spec = a
    if wn.size == 0 or np.any(wn <= 0) or np.any(wn >= 1):
        raise MatlabError("fir1: the frequencies must lie strictly between 0 and 1 (1 is the Nyquist frequency).")
    if np.any(np.diff(wn) <= 0):
        raise MatlabError("fir1: the frequencies must be increasing.")
    if ftype is None:
        ftype = "low" if wn.size == 1 else ("bandpass" if wn.size == 2 else "dc-0")
    if ftype in ("low", "high") and wn.size != 1 and ftype == "low":
        raise MatlabError("fir1: a lowpass filter needs a single cutoff frequency.")
    if ftype == "high" and wn.size != 1:
        raise MatlabError("fir1: a highpass filter needs a single cutoff frequency.")
    if ftype in ("bandpass", "stop") and wn.size != 2:
        raise MatlabError("fir1: bandpass and bandstop filters need two cutoff frequencies.")
    pass_zero = ftype in ("low", "stop", "dc-1")
    last_pass = pass_zero == (wn.size % 2 == 0)
    if last_pass and n % 2 == 1:
        n += 1   # an odd-order symmetric filter is zero at the Nyquist frequency
        if isinstance(win_spec, np.ndarray) and not is_text(win_spec) and win_spec.size == n:
            raise MatlabError("fir1: a filter that passes the Nyquist frequency needs an even order; "
                              "the order was raised by one, so give a window that is one element longer.")
    win = _window_from(interp, win_spec, n + 1, "fir1")
    return _row(_fir1_core(n, wn, pass_zero, win, scale))


def _fir2_response(f: np.ndarray, m: np.ndarray, npt: int, lap: int) -> np.ndarray:
    """The wanted magnitude on a grid of npt+1 points, with jumps turned into ramps of `lap` points."""
    fp, mp = [], []
    half = lap / (2.0 * npt)
    k = 0
    while k < f.size:
        if k + 1 < f.size and f[k + 1] == f[k]:
            lo = max(f[k] - half, fp[-1] if fp else 0.0)
            hi = min(f[k] + half, f[k + 2] if k + 2 < f.size else 1.0)
            fp.extend([lo, hi])
            mp.extend([m[k], m[k + 1]])
            k += 2
        else:
            fp.append(f[k])
            mp.append(m[k])
            k += 1
    grid = np.arange(npt + 1) / npt
    return np.interp(grid, np.array(fp), np.array(mp))


@builtin("fir2")
def _fir2(interp, args, nargout):
    nargs(args, 3, 6, "fir2")
    n = to_int(args[0], "filter order")
    f, m = _rvec(args[1], "frequencies"), _rvec(args[2], "magnitudes")
    if f.size != m.size or f.size < 2:
        raise MatlabError("fir2: the frequency and magnitude vectors must have the same length (at least 2).")
    if f[0] != 0 or f[-1] != 1 or np.any(np.diff(f) < 0):
        raise MatlabError("fir2: frequencies must go from 0 to 1 without decreasing.")
    scalars, win_spec = [], None
    for a in args[3:]:
        if is_text(a) or isinstance(a, (FunctionHandle, Cell)) or (isinstance(a, np.ndarray) and a.size > 1):
            win_spec = a
        elif isinstance(a, np.ndarray) and a.size == 1:
            scalars.append(int(_sc(a)))
    if n % 2 == 1 and m[-1] != 0:
        n += 1
    L = n + 1
    npt = scalars[0] if scalars else (512 if L < 1024 else int(2 ** np.ceil(np.log2(L))))
    lap = scalars[1] if len(scalars) > 1 else int(np.trunc(npt / 25))
    if 2 * npt < L:
        raise MatlabError("fir2: the number of grid points must be at least half the filter length.")
    win = _window_from(interp, win_spec, L, "fir2")
    H = _fir2_response(f, m, npt, lap)
    k = np.arange(npt + 1)
    Hc = H * np.exp(-1j * np.pi * k * (n / 2) / npt)
    full = np.concatenate([Hc, np.conj(Hc[npt - 1:0:-1])])
    ht = np.real(np.fft.ifft(full))
    return _row(ht[:L] * win)


def _band_check(f: np.ndarray, a: np.ndarray, name: str) -> None:
    if f.size % 2 or f.size < 2:
        raise MatlabError(f"{name}: the band edges must come in pairs.")
    if a.size != f.size:
        raise MatlabError(f"{name}: the frequency and amplitude vectors must have the same length.")
    if np.any(f < 0) or np.any(f > 1) or np.any(np.diff(f) < 0):
        raise MatlabError(f"{name}: band edges must lie between 0 and 1 and must not decrease.")


def _fir_type_args(args: list, name: str) -> tuple[np.ndarray | None, str]:
    weights, kind = None, "symmetric"
    for a in args:
        if is_text(a):
            s = text(a).lower()
            if s in ("h", "hilbert"):
                kind = "hilbert"
            elif s in ("d", "differentiator"):
                kind = "differentiator"
            else:
                raise MatlabError(f"{name}: unknown filter type '{text(a)}'. Use 'hilbert' or 'differentiator'.")
        elif isinstance(a, np.ndarray) and a.size:
            weights = _rvec(a, "weights")
    return weights, kind


def _int_cos(q, f1: float, f2: float):
    """Integral of cos(pi*q*f) over [f1, f2] (q may be an array)."""
    q = np.asarray(q, dtype=float)
    safe = np.where(q == 0, 1.0, q)
    val = (np.sin(np.pi * safe * f2) - np.sin(np.pi * safe * f1)) / (np.pi * safe)
    return np.where(q == 0, f2 - f1, val)


def _firls_core(n: int, f: np.ndarray, a: np.ndarray, w: np.ndarray, kind: str) -> np.ndarray:
    L = n + 1
    odd_len = L % 2 == 1
    M = n / 2
    anti = kind != "symmetric"
    if anti:
        mk = np.arange(1, L // 2 + 1, dtype=float) - (0.0 if odd_len else 0.5)
    else:
        mk = np.arange(0, L // 2 + 1, dtype=float) if odd_len else np.arange(1, L // 2 + 1, dtype=float) - 0.5
    K = mk.size
    Q = np.zeros((K, K))
    rhs = np.zeros(K)
    diff = mk[:, None] - mk[None, :]
    summ = mk[:, None] + mk[None, :]
    basis = np.sin if anti else np.cos
    for b in range(f.size // 2):
        f1, f2, a1, a2 = f[2 * b], f[2 * b + 1], a[2 * b], a[2 * b + 1]
        wb = w[b]
        if f2 == f1:
            continue
        if kind == "differentiator" and (a1 != 0 or a2 != 0):
            # the error is weighted by 1/f^2 in bands with a nonzero amplitude: integrate numerically
            nodes, wts = np.polynomial.legendre.leggauss(max(64, 8 * L))
            fx = 0.5 * (f2 - f1) * nodes + 0.5 * (f2 + f1)
            wx = 0.5 * (f2 - f1) * wts * wb / fx ** 2
            B = basis(np.pi * mk[:, None] * fx[None, :])
            D = a1 + (a2 - a1) * (fx - f1) / (f2 - f1)
            Q += (B * wx) @ B.T
            rhs += (B * wx) @ D
            continue
        c1, c2 = _int_cos(diff, f1, f2), _int_cos(summ, f1, f2)
        Q += wb * 0.5 * (c1 - c2 if anti else c1 + c2)
        slope = (a2 - a1) / (f2 - f1)
        c0 = a1 - slope * f1
        pm = np.pi * mk
        with np.errstate(divide="ignore", invalid="ignore"):
            if anti:
                def prim(x):
                    return -(c0 + slope * x) * np.cos(pm * x) / pm + slope * np.sin(pm * x) / pm ** 2
            else:
                def prim(x):
                    return np.where(mk == 0, c0 * x + slope * x * x / 2,
                                    (c0 + slope * x) * np.sin(pm * x) / pm + slope * np.cos(pm * x) / pm ** 2)
            rhs += wb * (prim(f2) - prim(f1))
    coef = np.linalg.lstsq(Q, rhs, rcond=None)[0] if np.linalg.cond(Q) > 1e13 else np.linalg.solve(Q, rhs)
    h = np.zeros(L)
    centre = M
    for c, m in zip(coef, mk):
        if m == 0:
            h[int(centre)] = c
            continue
        lo, hi = int(round(centre - m)), int(round(centre + m))
        if anti:
            # 'hilbert' gives -j*A(w) (a true Hilbert transformer for A = 1); 'differentiator' gives +j*A(w)
            sgn = 1.0 if kind == "differentiator" else -1.0
            h[lo], h[hi] = sgn * c / 2, -sgn * c / 2
        else:
            h[lo] = h[hi] = c / 2
    return h


@builtin("firls")
def _firls(interp, args, nargout):
    nargs(args, 3, 5, "firls")
    n = to_int(args[0], "filter order")
    f, a = _rvec(args[1], "frequencies"), _rvec(args[2], "amplitudes")
    _band_check(f, a, "firls")
    w, kind = _fir_type_args(args[3:], "firls")
    nb = f.size // 2
    if w is None:
        w = np.ones(nb)
    if w.size != nb:
        raise MatlabError("firls: there must be one weight per band.")
    if kind == "symmetric" and n % 2 == 1 and a[-1] != 0 and f[-1] == 1:
        n += 1
    if n < 1:
        raise MatlabError("firls: the filter order must be a positive integer.")
    return _row(_firls_core(n, f, a, w, kind))


def _remez(n: int, f: np.ndarray, a: np.ndarray, w: np.ndarray, kind: str, density: int = 16,
           maxit: int = 80) -> tuple[np.ndarray, float]:
    """Parks-McClellan design by the Remez exchange; returns the taps and the weighted ripple.

    The amplitude response is written A(w) = Q(w) P(w) with P a cosine polynomial of r terms and
    Q = 1, cos(w/2), sin(w) or sin(w/2) for the four linear-phase types, so that every case becomes
    a weighted Chebyshev approximation of D/Q by P with weight W*Q.
    """
    from numpy.polynomial import chebyshev as cheb
    L = n + 1
    odd_len = L % 2 == 1
    anti = kind != "symmetric"
    if not anti:
        r = (L + 1) // 2 if odd_len else L // 2
    else:
        r = (L - 1) // 2 if odd_len else L // 2
    if r < 1:
        raise MatlabError("firpm: the filter order is too small for this kind of filter.")

    def qfun(om):
        if not anti:
            return np.ones_like(om) if odd_len else np.cos(om / 2)
        return np.sin(om) if odd_len else np.sin(om / 2)

    # the frequency grid of the original Parks-McClellan program: steps of 1/(density*r) inside each
    # band, with the last point of a band moved onto the band edge
    nb = f.size // 2
    delf = 1.0 / (density * r)
    fx_l, d_l, w_l, band_l = [], [], [], []
    for b in range(nb):
        f1, f2 = f[2 * b], f[2 * b + 1]
        start = delf if (b == 0 and anti and f1 < delf) else f1
        count = max(int(np.floor((f2 - start) / delf + 1e-9)), 0)
        g = start + delf * np.arange(count + 1)
        g[-1] = f2
        if g.size == 1 and f2 > start:
            g = np.array([start, f2])
        dd = a[2 * b] + (a[2 * b + 1] - a[2 * b]) * ((g - f1) / (f2 - f1) if f2 > f1 else 0.0)
        ww = np.full(g.size, float(w[b]))
        if kind == "differentiator" and (a[2 * b] != 0 or a[2 * b + 1] != 0):
            ww = ww / np.maximum(g, 1e-12)
        keep = np.ones(g.size, dtype=bool)
        if b == nb - 1 and anti == odd_len and g[-1] > 1 - delf:
            keep[-1] = False       # these filter kinds are zero at the Nyquist frequency
        keep &= np.abs(qfun(np.pi * g)) > 1e-12
        fx_l.append(g[keep])
        d_l.append(dd[keep])
        w_l.append(ww[keep])
        band_l.append(np.full(int(keep.sum()), b))
    fx, d, ww, band = (np.concatenate(v) for v in (fx_l, d_l, w_l, band_l))
    om = np.pi * fx
    q = qfun(om)
    dq, wq, x = d / q, ww * np.abs(q), np.cos(om)
    ng = x.size
    if ng < r + 1:
        raise MatlabError("firpm: the bands are too narrow for this filter order.")
    first = np.r_[True, band[1:] != band[:-1]]
    ext = np.round(np.linspace(0, ng - 1, r + 1)).astype(int)
    delta, coef = 0.0, np.zeros(r)
    sgn = (-1.0) ** np.arange(r + 1)
    for _ in range(maxit):
        A = np.column_stack([cheb.chebvander(x[ext], r - 1), sgn / wq[ext]])
        try:
            sol = np.linalg.solve(A, dq[ext])
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, dq[ext], rcond=None)[0]
        coef, delta = sol[:r], float(sol[r])
        err = wq * (dq - cheb.chebval(x, coef))
        ae = np.abs(err)
        # one candidate per lobe: the largest error inside every run of constant sign (within a band)
        pos = err >= 0
        starts = np.flatnonzero(np.r_[True, (pos[1:] != pos[:-1]) | first[1:]])
        ends = np.r_[starts[1:], ng]
        cand = [int(s0 + np.argmax(ae[s0:e0])) for s0, e0 in zip(starts, ends)]
        kept: list[int] = []
        for k in cand:
            if kept and pos[k] == pos[kept[-1]]:
                if ae[k] > ae[kept[-1]]:
                    kept[-1] = k
            else:
                kept.append(k)
        while len(kept) > r + 1:
            excess = len(kept) - (r + 1)
            smallest = min(range(len(kept)), key=lambda i: ae[kept[i]])
            if excess >= 2 and 0 < smallest < len(kept) - 1:
                other = smallest - 1 if ae[kept[smallest - 1]] < ae[kept[smallest + 1]] else smallest + 1
                for i in sorted((smallest, other), reverse=True):
                    kept.pop(i)
            elif ae[kept[0]] < ae[kept[-1]]:
                kept.pop(0)
            else:
                kept.pop()
        while len(kept) < r + 1:
            # too few alternations (a degenerate reference): add the grid point farthest from the others
            dist = np.min(np.abs(np.arange(ng)[:, None] - np.array(kept)[None, :]), axis=1)
            kept = sorted(kept + [int(np.argmax(dist))])
        new = np.array(kept)
        done = np.array_equal(new, ext) or np.max(ae) - abs(delta) <= 1e-8 * max(abs(delta), 1e-14)
        ext = new
        if done:
            break
    alpha = coef
    h = np.zeros(L)
    M = n / 2
    if not anti and odd_len:           # type I: A = sum alpha_k cos(k w)
        h[int(M)] = alpha[0]
        for k in range(1, r):
            h[int(M) - k] = h[int(M) + k] = alpha[k] / 2
        return h, abs(delta)
    c = np.zeros(r + 2)                # c[j]: coefficient of the j-th half/whole-angle term, j = 1..r
    if not anti:                       # type II: cos(w/2) cos(k w) = (cos((k+1/2) w) + cos((k-1/2) w)) / 2
        c[1] += alpha[0]
        for k in range(1, r):
            c[k + 1] += alpha[k] / 2
            c[k] += alpha[k] / 2
        offs = np.arange(r + 2) - 0.5
    elif odd_len:                      # type III: sin(w) cos(k w) = (sin((k+1) w) - sin((k-1) w)) / 2
        c[1] += alpha[0]
        for k in range(1, r):
            c[k + 1] += alpha[k] / 2
            if k >= 2:
                c[k - 1] -= alpha[k] / 2
        offs = np.arange(r + 2, dtype=float)
    else:                              # type IV: sin(w/2) cos(k w) = (sin((k+1/2) w) - sin((k-1/2) w)) / 2
        c[1] += alpha[0]
        for k in range(1, r):
            c[k + 1] += alpha[k] / 2
            c[k] -= alpha[k] / 2
        offs = np.arange(r + 2) - 0.5
    flip = 1.0
    if anti:
        # 'hilbert' gives -j*A(w) (a true Hilbert transformer for A = 1); 'differentiator' gives +j*A(w)
        flip = 1.0 if kind == "differentiator" else -1.0
    for j in range(1, r + 1):
        lo, hi = int(round(M - offs[j])), int(round(M + offs[j]))
        if anti:
            h[lo], h[hi] = flip * c[j] / 2, -flip * c[j] / 2
        else:
            h[lo] = h[hi] = c[j] / 2
    return h, abs(delta)


@builtin("firpm")
def _firpm(interp, args, nargout):
    nargs(args, 3, 6, "firpm")
    n = to_int(args[0], "filter order")
    f, a = _rvec(args[1], "frequencies"), _rvec(args[2], "amplitudes")
    _band_check(f, a, "firpm")
    rest = []
    density = 16
    for x in args[3:]:
        if isinstance(x, Cell):
            density = int(_sc(x.values_f()[0], "grid density"))
        else:
            rest.append(x)
    w, kind = _fir_type_args(rest, "firpm")
    nb = f.size // 2
    if w is None:
        w = np.ones(nb)
    if w.size != nb:
        raise MatlabError("firpm: there must be one weight per band.")
    if np.any(w <= 0):
        raise MatlabError("firpm: weights must be positive.")
    if kind == "symmetric" and n % 2 == 1 and a[-1] != 0 and f[-1] == 1:
        n += 1
    if n < 3:
        raise MatlabError("firpm: the filter order must be at least 3.")
    h, delta = _remez(n, f, a, w, kind, density)
    if nargout > 1:
        return _row(h), delta
    return _row(h)


@builtin("kaiserord")
def _kaiserord(interp, args, nargout):
    nargs(args, 3, 5, "kaiserord")
    rest, flags = _split_flags(list(args))
    f, a, dev = _rvec(rest[0], "band edges"), _rvec(rest[1], "amplitudes"), _rvec(rest[2], "deviations")
    fs = _sc(rest[3], "sample rate") if len(rest) > 3 else 2.0
    if f.size != 2 * a.size - 2:
        raise MatlabError("kaiserord: there must be two band edges for every band after the first.")
    if dev.size == 1:
        dev = np.full(a.size, dev[0])
    if dev.size != a.size:
        raise MatlabError("kaiserord: give one deviation per band.")
    if np.any(dev <= 0):
        raise MatlabError("kaiserord: deviations must be positive.")
    f = f / (fs / 2)
    if np.any(f <= 0) or np.any(f >= 1) or np.any(np.diff(f) <= 0):
        raise MatlabError("kaiserord: band edges must be increasing and lie between 0 and half the sample rate.")
    rel = dev / np.where(a == 0, 1.0, a)        # deviations relative to the band amplitude
    delta = float(np.min(rel))
    att = -20 * np.log10(delta)
    width = float(np.min(f[1::2] - f[0::2]))    # narrowest transition band (in units of Nyquist)
    n = int(np.ceil((att - 7.95) / (2.285 * np.pi * width)))
    n = max(n, 1)
    if att > 50:
        beta = 0.1102 * (att - 8.7)
    elif att >= 21:
        beta = 0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21)
    else:
        beta = 0.0
    wn = 0.5 * (f[0::2] + f[1::2])
    nbands = a.size
    if nbands == 2:
        ftype = "low" if a[0] > a[1] else "high"
    elif nbands == 3:
        ftype = "stop" if a[0] > a[1] else "DC-0"
        if a[0] <= a[1]:
            ftype = "DC-0"
    else:
        ftype = "DC-1" if a[0] > a[1] else "DC-0"
    if n % 2 == 1 and a[-1] != 0:
        n += 1
    if "cell" in flags:
        win = _col(_w_kaiser(n + 1, beta))
        return Cell.from_list([_f(n), _row(wn) if wn.size > 1 else _f(wn[0]), char(ftype), win, char("noscale")])
    out = (_f(n), _row(wn) if wn.size > 1 else _f(wn[0]), _f(beta), char(ftype))
    return out[:max(nargout, 1)] if nargout > 1 else out[0]


# -- IIR filter design ------------------------------------------------------------------------------

_BTYPES = {"low": "lowpass", "lowpass": "lowpass", "high": "highpass", "highpass": "highpass",
           "bandpass": "bandpass", "pass": "bandpass", "stop": "bandstop", "bandstop": "bandstop"}
_IIR_PARAMS = {"butter": 0, "cheby1": 1, "cheby2": 1, "ellip": 2, "besself": 0}


def _zcol(v) -> np.ndarray:
    v = np.atleast_1d(np.asarray(v, dtype=complex))
    return _col(simplify(v)) if v.size else np.zeros((0, 1))


def _zpk2tf(z, p, k) -> tuple[np.ndarray, np.ndarray]:
    """Numerator and denominator of equal length (descending powers)."""
    z, p = np.atleast_1d(z), np.atleast_1d(p)
    b = np.atleast_1d(k * _poly(z))
    a = np.atleast_1d(_poly(p))
    if b.size < a.size:
        b = np.concatenate([np.zeros(a.size - b.size), b])
    return _tidy(b), _tidy(a)


def _tf2ss(b: np.ndarray, a: np.ndarray):
    """Controller canonical form (the layout tf2ss uses)."""
    b, a = np.atleast_1d(b).astype(float), np.atleast_1d(a).astype(float)
    while a.size > 1 and a[0] == 0:
        a = a[1:]
    b = b / a[0]
    a = a / a[0]
    n = a.size - 1
    if b.size > a.size:
        if np.any(b[:b.size - a.size] != 0):
            raise MatlabError("The numerator order must not exceed the denominator order.")
        b = b[b.size - a.size:]
    b = np.concatenate([np.zeros(a.size - b.size), b])
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[b[0]]])
    A = np.zeros((n, n))
    A[0, :] = -a[1:]
    A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    C = (b[1:] - b[0] * a[1:]).reshape(1, n)
    return A, B, C, np.array([[b[0]]])


def _pair_order(p: np.ndarray) -> np.ndarray:
    """Poles or zeros listed as conjugate pairs (positive imaginary part first), real ones last."""
    p = np.asarray(p, dtype=complex)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(p))) if p.size else 1.0)
    out: list[complex] = []
    for c in p[p.imag > tol]:
        out.extend([c, np.conj(c)])
    out.extend(np.sort(p[np.abs(p.imag) <= tol].real).astype(complex))
    return np.array(out, dtype=complex)


def iir_design(name: str, args: list, nargout: int):
    """Shared code of butter, cheby1, cheby2, ellip and besself."""
    ss = _sig()
    rest = list(args)
    analog = name == "besself"
    btype = None
    while rest and is_text(rest[-1]):
        s = text(rest.pop()).lower()
        if s == "s":
            analog = True
        elif s == "z":
            if name == "besself":
                raise MatlabError("besself: only analog Bessel filters are available ('z' is not supported).")
            analog = False
        elif s in _BTYPES:
            btype = _BTYPES[s]
        else:
            raise MatlabError(f"{name}: unknown option '{s}'. Use 'low', 'high', 'bandpass', 'stop' or 's'.")
    nparams = _IIR_PARAMS[name]
    nargs(rest, 2 + nparams, 2 + nparams, name)
    n = to_int(rest[0], "filter order")
    if n < 1:
        raise MatlabError(f"{name}: the filter order must be a positive integer.")
    params = [_sc(v, "ripple") for v in rest[1:1 + nparams]]
    if any(v <= 0 for v in params):
        raise MatlabError(f"{name}: ripple and attenuation values must be positive (in dB).")
    wn = _rvec(rest[1 + nparams], "Wn")
    if wn.size not in (1, 2):
        raise MatlabError(f"{name}: Wn must have one or two elements.")
    if btype is None:
        btype = "bandpass" if wn.size == 2 else "lowpass"
    if btype in ("lowpass", "highpass") and wn.size != 1:
        raise MatlabError(f"{name}: a {btype} filter needs a single cutoff frequency.")
    if btype in ("bandpass", "bandstop") and wn.size != 2:
        raise MatlabError(f"{name}: bandpass and bandstop filters need Wn = [w1 w2].")
    if wn.size == 2 and wn[0] >= wn[1]:
        raise MatlabError(f"{name}: the band edges must be increasing.")
    if np.any(wn <= 0):
        raise MatlabError(f"{name}: cutoff frequencies must be positive.")
    if not analog and np.any(wn >= 1):
        raise MatlabError(f"{name}: cutoff frequencies must be between 0 and 1 (1 is the Nyquist frequency).")
    w = wn if wn.size > 1 else float(wn[0])
    ftype = "bessel" if name == "besself" else name
    rp = params[0] if name in ("cheby1", "ellip") else None
    rs = params[0] if name == "cheby2" else (params[1] if name == "ellip" else None)
    z, p, k = ss.iirfilter(n, w, rp=rp, rs=rs, btype=btype, analog=analog, ftype=ftype, output="zpk")
    k = float(np.real(k))
    if nargout >= 4:
        b, a = _zpk2tf(z, p, k)
        return _tf2ss(np.real(b), np.real(a))
    if nargout == 3:
        return _zcol(z), _zcol(p), k
    b, a = _zpk2tf(z, p, k)
    return _row(np.real(b)), _row(np.real(a))


def _iir_builtin(name: str):
    def fn(interp, args, nargout):
        return iir_design(name, args, nargout)
    builtin(name)(fn)


for _n in ("cheby2", "ellip", "besself"):
    _iir_builtin(_n)


def _proto(name: str, lo: int, hi: int, maker):
    def fn(interp, args, nargout):
        nargs(args, lo, hi, name)
        n = to_int(args[0], "filter order")
        if n < 0:
            raise MatlabError(f"{name}: the filter order must not be negative.")
        z, p, k = maker(n, *[_sc(a) for a in args[1:]])
        z = _pair_order(np.atleast_1d(z)) if np.size(z) else np.zeros(0)
        p = _pair_order(np.atleast_1d(p))
        zz = _zcol(z) if z.size else np.zeros((0, 0))
        return zz, _zcol(p), float(np.real(k))
    builtin(name)(fn)


_proto("buttap", 1, 1, lambda n: _sig().buttap(n))
_proto("cheb1ap", 2, 2, lambda n, rp: _sig().cheb1ap(n, rp))
_proto("cheb2ap", 2, 2, lambda n, rs: _sig().cheb2ap(n, rs))
_proto("ellipap", 3, 3, lambda n, rp, rs: _sig().ellipap(n, rp, rs))
_proto("besselap", 1, 1, lambda n: _sig().besselap(n, norm="phase"))


def _ss_or_tf(args: list, name: str, nextra: int):
    """Split (b, a, extras) from (A, B, C, D, extras)."""
    if len(args) == 4 + nextra:
        return "ss", [fnum(a).astype(float) for a in args[:4]], args[4:]
    if len(args) == 2 + nextra:
        return "tf", [_rvec(args[0]), _rvec(args[1])], args[2:]
    raise MatlabError(f"{name}: expected (b, a, ...) or (A, B, C, D, ...).")


def _trim(b: np.ndarray) -> np.ndarray:
    b = np.atleast_1d(b)
    nz = np.flatnonzero(b != 0)
    return b[nz[0]:] if nz.size else np.zeros(1)


@builtin("lp2lp")
def _lp2lp(interp, args, nargout):
    form, sys_, extra = _ss_or_tf(list(args), "lp2lp", 1)
    wo = _sc(extra[0], "Wo")
    if form == "ss":
        A, B, C, D = sys_
        return wo * A, wo * B, C, D
    b, a = _sig().lp2lp(_trim(sys_[0]), _trim(sys_[1]), wo)
    return _row(b), _row(a)


@builtin("lp2hp")
def _lp2hp(interp, args, nargout):
    form, sys_, extra = _ss_or_tf(list(args), "lp2hp", 1)
    wo = _sc(extra[0], "Wo")
    if form == "ss":
        A, B, C, D = sys_
        Ai = np.linalg.inv(A)
        return wo * Ai, -wo * (Ai @ B), C @ Ai, D - C @ Ai @ B
    b, a = _sig().lp2hp(_trim(sys_[0]), _trim(sys_[1]), wo)
    return _row(b), _row(a)


@builtin("lp2bp")
def _lp2bp(interp, args, nargout):
    form, sys_, extra = _ss_or_tf(list(args), "lp2bp", 2)
    wo, bw = _sc(extra[0], "Wo"), _sc(extra[1], "Bw")
    if form == "ss":
        A, B, C, D = sys_
        n = A.shape[0]
        q = wo / bw
        At = wo * np.block([[A / q, np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
        Bt = wo * np.vstack([B / q, np.zeros((n, B.shape[1]))])
        Ct = np.hstack([C, np.zeros((C.shape[0], n))])
        return At, Bt, Ct, D
    b, a = _sig().lp2bp(_trim(sys_[0]), _trim(sys_[1]), wo, bw)
    return _row(b), _row(a)


@builtin("lp2bs")
def _lp2bs(interp, args, nargout):
    form, sys_, extra = _ss_or_tf(list(args), "lp2bs", 2)
    wo, bw = _sc(extra[0], "Wo"), _sc(extra[1], "Bw")
    if form == "ss":
        A, B, C, D = sys_
        n = A.shape[0]
        q = wo / bw
        Ai = np.linalg.inv(A)
        At = np.block([[wo / q * Ai, wo * np.eye(n)], [-wo * np.eye(n), np.zeros((n, n))]])
        Bt = -np.vstack([wo / q * (Ai @ B), np.zeros((n, B.shape[1]))])
        Ct = np.hstack([C @ Ai, np.zeros((C.shape[0], n))])
        return At, Bt, Ct, D - C @ Ai @ B
    b, a = _sig().lp2bs(_trim(sys_[0]), _trim(sys_[1]), wo, bw)
    return _row(b), _row(a)


def _bilinear_zpk(z: np.ndarray, p: np.ndarray, k: float, fs2: float):
    """Bilinear transform s = fs2*(z-1)/(z+1) of a zero-pole-gain model."""
    z, p = np.asarray(z, dtype=complex), np.asarray(p, dtype=complex)
    zd = (fs2 + z) / (fs2 - z)
    pd = (fs2 + p) / (fs2 - p)
    kd = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    zd = np.concatenate([zd, -np.ones(p.size - z.size)])
    return zd, pd, float(kd)


@builtin("bilinear")
def _bilinear(interp, args, nargout):
    nargs(args, 3, 6, "bilinear")
    a0, a1 = fnum(args[0]), fnum(args[1])
    as_ss = len(args) >= 5
    cols = a1.ndim == 2 and a1.shape[1] == 1 and (a0.size == 0 or a0.shape[1] == 1)
    as_zpk = (not as_ss) and cols and nargout != 4 and not (a0.size == 1 and a1.size == 1 and len(args) == 3)
    if not as_ss and not as_zpk and len(args) == 4 and nargout == 3:
        as_zpk = True
    nsys = 4 if as_ss else (3 if as_zpk else 2)
    extra = args[nsys:]
    if not extra:
        raise MatlabError("bilinear: the sample rate is required.")
    fs = _sc(extra[0], "sample rate")
    if fs <= 0:
        raise MatlabError("bilinear: the sample rate must be positive.")
    if len(extra) > 1:
        fp = 2 * np.pi * _sc(extra[1], "match frequency")
        fs2 = fp / np.tan(fp / fs / 2)
    else:
        fs2 = 2 * fs
    if as_ss:
        A, B, C, D = (fnum(a).astype(float) for a in args[:4])
        t = 2.0 / fs2
        r = np.sqrt(t)
        n = A.shape[0]
        t1 = np.eye(n) + A * t / 2
        t2i = np.linalg.inv(np.eye(n) - A * t / 2)
        return t2i @ t1, t / r * (t2i @ B), r * (C @ t2i), C @ t2i @ B * t / 2 + D
    if as_zpk:
        z, p = _vec(args[0]), _vec(args[1])
        k = _sc(args[2], "gain")
        if z.size > p.size:
            raise MatlabError("bilinear: there must be at least as many poles as zeros.")
        zd, pd, kd = _bilinear_zpk(z, p, k, fs2)
        return _zcol(zd), _zcol(pd), kd
    b, a = _trim(_vec(args[0])), _trim(_vec(args[1]))
    if b.size > a.size:
        raise MatlabError("bilinear: the numerator order must not exceed the denominator order.")
    z, p = np.roots(b), np.roots(a)
    zd, pd, kd = _bilinear_zpk(z, p, float(np.real(b[0] / a[0])), fs2)
    bd, ad = _zpk2tf(zd, pd, kd)
    return _row(np.real(bd)), _row(np.real(ad))


def _power_sum_polys(m: int) -> list[np.ndarray]:
    """N_k with sum_{n>=0} n^k x^n = N_k(x) / (1-x)^(k+1), k = 0..m-1 (ascending powers of x)."""
    P = np.polynomial.polynomial
    out = [np.array([1.0])]
    for k in range(1, m):
        prev = out[-1]
        d = P.polyder(prev) if prev.size > 1 else np.array([0.0])
        # x d/dx [N/(1-x)^k] = [x(1-x)N' + k x N] / (1-x)^(k+1)
        out.append(P.polyadd(P.polymul(np.array([0.0, 1.0, -1.0]), d), k * P.polymul(np.array([0.0, 1.0]), prev)))
    return out


@builtin("impinvar")
def _impinvar(interp, args, nargout):
    nargs(args, 2, 4, "impinvar")
    b, a = _trim(_vec(args[0])), _trim(_vec(args[1]))
    fs = _sc(args[2], "sample rate") if _given(args, 2) else 1.0
    tol = _sc(args[3], "tolerance") if _given(args, 3) else 0.001
    if b.size >= a.size:
        raise MatlabError("impinvar: the numerator order must be less than the denominator order.")
    r, p, _k = _sig().residue(b, a, tol=tol)
    T = 1.0 / fs
    P = np.polynomial.polynomial
    num = np.zeros(1, dtype=complex)
    den = np.ones(1, dtype=complex)
    i = 0
    while i < p.size:
        j = i
        while j + 1 < p.size and abs(p[j + 1] - p[i]) <= tol * max(1.0, abs(p[i])):
            j += 1
        pole, res = p[i], r[i:j + 1]
        i = j + 1
        x = np.exp(pole * T)
        mult = res.size
        polys = _power_sum_polys(mult)
        full = np.array([1.0 + 0j])
        for _ in range(mult):
            full = P.polymul(full, np.array([1.0, -x]))
        tnum = np.zeros(1, dtype=complex)
        for m in range(1, mult + 1):
            # r/(s-p)^m has the impulse response r t^(m-1) e^(pt)/(m-1)!, sampled at t = nT and scaled by T
            fact = float(np.prod(np.arange(1, m))) if m > 1 else 1.0
            c = res[m - 1] * T ** m / fact
            nk = polys[m - 1] * x ** np.arange(polys[m - 1].size)     # in powers of z^-1
            pad = np.array([1.0 + 0j])
            for _ in range(mult - m):
                pad = P.polymul(pad, np.array([1.0, -x]))
            tnum = P.polyadd(tnum, c * P.polymul(nk, pad))
        num = P.polyadd(P.polymul(num, full), P.polymul(tnum, den))
        den = P.polymul(den, full)
    bz, az = np.atleast_1d(_tidy(num)), np.atleast_1d(_tidy(den))
    while bz.size > 1 and abs(bz[-1]) <= 1e-14 * max(1.0, float(np.max(np.abs(bz)))):
        bz = bz[:-1]
    return _row(bz), _row(az)


# -- order estimation -------------------------------------------------------------------------------

def _ord_inputs(args: list, name: str):
    rest = list(args)
    analog = False
    if rest and is_text(rest[-1]):
        if text(rest.pop()).lower() != "s":
            raise MatlabError(f"{name}: the only option is 's' (an analog filter).")
        analog = True
    nargs(rest, 4, 4, name)
    wp, ws = np.sort(_rvec(rest[0], "Wp")), np.sort(_rvec(rest[1], "Ws"))
    rp, rs = _sc(rest[2], "Rp"), _sc(rest[3], "Rs")
    if wp.size != ws.size or wp.size not in (1, 2):
        raise MatlabError(f"{name}: Wp and Ws must both be scalars or both be two-element vectors.")
    if rp <= 0 or rs <= 0 or rs <= rp:
        raise MatlabError(f"{name}: Rp and Rs must be positive, with Rs larger than Rp.")
    if not analog and (np.any(wp <= 0) or np.any(wp >= 1) or np.any(ws <= 0) or np.any(ws >= 1)):
        raise MatlabError(f"{name}: frequencies must lie strictly between 0 and 1.")
    if wp.size == 1:
        kind = "low" if wp[0] < ws[0] else "high"
    else:
        kind = "pass" if ws[0] < wp[0] else "stop"
        if kind == "pass" and not (ws[0] < wp[0] < wp[1] < ws[1]):
            raise MatlabError(f"{name}: for a bandpass filter the passband must lie inside the stopband edges.")
        if kind == "stop" and not (wp[0] < ws[0] < ws[1] < wp[1]):
            raise MatlabError(f"{name}: for a bandstop filter the stopband must lie inside the passband edges.")
    return wp, ws, rp, rs, analog, kind


def _wn_out(wn: np.ndarray):
    return _f(wn[0]) if wn.size == 1 else _row(wn)


@builtin("buttord")
def _buttord(interp, args, nargout):
    wp, ws, rp, rs, analog, kind = _ord_inputs(args, "buttord")
    WP = wp.copy() if analog else np.tan(np.pi * wp / 2)
    WS = ws.copy() if analog else np.tan(np.pi * ws / 2)
    gs, gp = 10 ** (0.1 * rs) - 1, 10 ** (0.1 * rp) - 1

    def order_for(WPx):
        if kind == "low":
            wa = WS / WPx
        elif kind == "high":
            wa = WPx / WS
        elif kind == "pass":
            wa = (WS ** 2 - WPx[0] * WPx[1]) / (WS * (WPx[0] - WPx[1]))
        else:
            wa = (WS * (WPx[0] - WPx[1])) / (WS ** 2 - WPx[0] * WPx[1])
        wa = float(np.min(np.abs(wa)))
        return np.log10(gs / gp) / (2 * np.log10(wa)), wa

    if kind == "stop":
        # the passband edges may move inwards without breaking the specification: pick the best ones
        from scipy.optimize import fminbound
        lo = fminbound(lambda x: order_for(np.array([x, WP[1]]))[0], WP[0], WS[0] - 1e-12, xtol=1e-10)
        WP = np.array([lo, WP[1]])
        hi = fminbound(lambda x: order_for(np.array([WP[0], x]))[0], WS[1] + 1e-12, WP[1], xtol=1e-10)
        WP = np.array([WP[0], hi])
    nreal, wa = order_for(WP)
    n = max(int(np.ceil(nreal - 1e-12)), 1)
    w0 = wa / gs ** (1.0 / (2 * n))        # the stopband requirement is met exactly
    if kind == "low":
        WN = np.array([w0 * WP[0]])
    elif kind == "high":
        WN = np.array([WP[0] / w0])
    elif kind == "pass":
        bw = WP[1] - WP[0]
        w0v = np.array([-w0, w0])
        WN = np.sort(np.abs(-w0v * bw / 2 + np.sqrt(w0v ** 2 / 4 * bw ** 2 + WP[0] * WP[1])))
    else:
        bw = WP[1] - WP[0]
        w0v = np.array([-w0, w0])
        WN = np.sort(np.abs(bw / (2 * w0v) + np.sqrt(bw ** 2 / (4 * w0v ** 2) + WP[0] * WP[1])))
    wn = WN if analog else 2 / np.pi * np.arctan(WN)
    return (_f(n), _wn_out(wn)) if nargout > 1 else _f(n)


def _scipy_ord(name: str):
    def fn(interp, args, nargout):
        wp, ws, rp, rs, analog, _kind = _ord_inputs(args, name)
        f = getattr(_sig(), name)
        wpa = float(wp[0]) if wp.size == 1 else wp
        wsa = float(ws[0]) if ws.size == 1 else ws
        n, wn = f(wpa, wsa, rp, rs, analog=analog)
        wn = np.atleast_1d(np.asarray(wn, dtype=float))
        return (_f(n), _wn_out(wn)) if nargout > 1 else _f(n)
    builtin(name)(fn)


for _n in ("cheb1ord", "cheb2ord", "ellipord"):
    _scipy_ord(_n)


# -- filter representations ---------------------------------------------------------------------------

def _is_sos(v) -> bool:
    return isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 6 and v.shape[0] >= 2 \
        and v.dtype.kind in "biufc"


def _sos_check(v, name: str) -> np.ndarray:
    a = fnum(v)
    if a.ndim != 2 or a.shape[1] != 6:
        raise MatlabError(f"{name}: a second-order-section matrix must have six columns [b0 b1 b2 1 a1 a2].")
    return a


def _sos2tf(sos: np.ndarray, g=1.0) -> tuple[np.ndarray, np.ndarray]:
    b, a = np.array([1.0]), np.array([1.0])
    for row in sos:
        b = np.convolve(b, row[:3])
        a = np.convolve(a, row[3:])
    b = b * g
    # drop trailing zeros that come from first-order sections (keep equal lengths)
    while b.size > 1 and a.size > 1 and b[-1] == 0 and a[-1] == 0:
        b, a = b[:-1], a[:-1]
    return b, a


def _conj_groups(v: np.ndarray) -> tuple[list[complex], list[float]]:
    """Complex roots with positive imaginary part (one per conjugate pair) and the real roots."""
    v = np.asarray(v, dtype=complex)
    tol = 100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(v))) if v.size else 1.0)
    cplx = [c for c in v if c.imag > tol]
    neg = [c for c in v if c.imag < -tol]
    if len(cplx) != len(neg):
        raise MatlabError("Complex zeros and poles must come in conjugate pairs.")
    real = [float(c.real) for c in v if abs(c.imag) <= tol]
    return cplx, real


def _root_sections(v: np.ndarray) -> list[list[complex]]:
    """Group roots into sections of at most two: conjugate pairs, and real roots paired by size."""
    cplx, real = _conj_groups(v)
    groups: list[list[complex]] = [[c, np.conj(c)] for c in cplx]
    real = sorted(real, key=lambda x: -abs(x))
    for i in range(0, len(real) - 1, 2):
        groups.append([complex(real[i]), complex(real[i + 1])])
    if len(real) % 2:
        groups.append([complex(real[-1])])
    return groups


def _zp2sos(z: np.ndarray, p: np.ndarray, k: float, order: str = "up"):
    z, p = np.atleast_1d(z), np.atleast_1d(p)
    if z.size == 0 and p.size == 0:
        return np.array([[k, 0, 0, 1.0, 0, 0]]), 1.0
    pg, zg = _root_sections(p), _root_sections(z)
    nsec = max(len(pg), len(zg))
    # poles closest to the unit circle choose their zeros first
    pg.sort(key=lambda g: abs(1 - max(abs(c) for c in g)))
    pairs = []
    for g in pg:
        if not zg:
            pairs.append((g, []))
            continue
        ref = max(g, key=abs)

        def dist(zs, ref=ref, g=g):
            return (0 if len(zs) == len(g) else 1, min(abs(c - ref) for c in zs))
        best = min(range(len(zg)), key=lambda i: dist(zg[i]))
        pairs.append((g, zg.pop(best)))
    for zs in zg:
        pairs.append(([], zs))
    # 'up': the section with poles closest to the unit circle comes last
    def closeness(pr):
        return abs(1 - max(abs(c) for c in pr[0])) if pr[0] else np.inf
    pairs.sort(key=closeness, reverse=(order == "up"))
    sos = np.zeros((nsec, 6))
    for i, (pp, zz) in enumerate(pairs):
        bq = np.real(np.poly(zz)) if zz else np.array([1.0])
        aq = np.real(np.poly(pp)) if pp else np.array([1.0])
        sos[i, :bq.size] = bq
        sos[i, 3:3 + aq.size] = aq
    g = float(k)
    return sos, g


def _sos_args(args: list, start: int, name: str) -> str:
    order, scale = "up", "none"
    for a in args[start:]:
        s = text(a).lower()
        if s in ("up", "down"):
            order = s
        elif s in ("none", "inf", "two"):
            scale = s
        else:
            raise MatlabError(f"{name}: unknown option '{text(a)}'. Use 'up', 'down', 'none', 'inf' or 'two'.")
    if scale != "none":
        raise MatlabError(f"{name}: the scaling option '{scale}' is not supported; only 'none' is available.")
    return order


def _sos_out(sos: np.ndarray, g: float, nargout: int):
    if nargout > 1:
        return sos, float(g)
    sos = sos.copy()
    sos[0, :3] *= g
    return sos


def _tf2zpk_d(b: np.ndarray, a: np.ndarray):
    """Zeros, poles and gain of a transfer function in powers of z^-1."""
    b, a = np.atleast_1d(b), np.atleast_1d(a)
    n = max(b.size, a.size)
    b = np.concatenate([b, np.zeros(n - b.size)])
    a = np.concatenate([a, np.zeros(n - a.size)])
    return _tf2zp_c(b, a)


def _tf2zp_c(b: np.ndarray, a: np.ndarray):
    """Zeros, poles and gain of a transfer function in descending powers of the variable."""
    b, a = np.atleast_1d(b), np.atleast_1d(a)
    if not np.any(a != 0):
        raise MatlabError("The denominator must not be zero.")
    a = _trim(a)
    nz = np.flatnonzero(b != 0)
    if nz.size == 0:
        return np.zeros(0), np.roots(a), 0.0
    k = b[nz[0]] / a[0]
    return np.roots(b[nz[0]:]), np.roots(a), k


@builtin("tf2zpk")
def _tf2zpk(interp, args, nargout):
    nargs(args, 1, 2, "tf2zpk")
    b = _vec(args[0], "numerator")
    a = _vec(args[1], "denominator") if len(args) > 1 else np.array([1.0])
    z, p, k = _tf2zp_c(*_pad_equal(b, a))
    return _zcol(z), _zcol(p), simplify(np.array([[k]]))


def _pad_equal(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(b.size, a.size)
    return np.concatenate([b, np.zeros(n - b.size)]), np.concatenate([a, np.zeros(n - a.size)])


@builtin("tf2zp")
def _tf2zp(interp, args, nargout):
    nargs(args, 2, 2, "tf2zp")
    bm = fnum(args[0])
    a = _vec(args[1], "denominator")
    if bm.ndim == 2 and bm.shape[0] > 1 and bm.shape[1] > 1:
        zs, ks = [], []
        for row in bm:
            z, p, k = _tf2zp_c(row, a)
            zs.append(z)
            ks.append(k)
        width = max((z.size for z in zs), default=0)
        zmat = np.full((width, len(zs)), np.inf, dtype=complex)
        for j, z in enumerate(zs):
            zmat[:z.size, j] = z
        return simplify(zmat), _zcol(np.roots(_trim(a))), simplify(np.array(ks).reshape(-1, 1))
    z, p, k = _tf2zp_c(bm.flatten(order="F"), a)
    return _zcol(z), _zcol(p), simplify(np.array([[k]]))


@builtin("zp2tf")
def _zp2tf(interp, args, nargout):
    nargs(args, 3, 3, "zp2tf")
    zm = fnum(args[0])
    p = _vec(args[1], "poles")
    k = _vec(args[2], "gain")
    a = np.atleast_1d(_poly(p))
    if zm.ndim == 2 and zm.shape[1] > 1 and zm.shape[0] >= 1 and k.size == zm.shape[1] and k.size > 1:
        rows = []
        for j in range(zm.shape[1]):
            zj = zm[:, j]
            zj = zj[np.isfinite(zj)]
            bj = np.atleast_1d(k[j] * _poly(zj))
            rows.append(np.concatenate([np.zeros(a.size - bj.size), bj]))
        return _tidy(np.array(rows)), _row(a)
    z = zm.flatten(order="F")
    z = z[np.isfinite(z)]
    b = np.atleast_1d(k[0] * _poly(z))
    if b.size < a.size:
        b = np.concatenate([np.zeros(a.size - b.size), b])
    return _row(_tidy(b)), _row(a)


@builtin("zp2sos")
def _zp2sos_b(interp, args, nargout):
    nargs(args, 2, 5, "zp2sos")
    z, p = _vec(args[0], "zeros"), _vec(args[1], "poles")
    k = _sc(args[2], "gain") if _given(args, 2) else 1.0
    order = _sos_args(list(args), 3, "zp2sos")
    sos, g = _zp2sos(z, p, k, order)
    return _sos_out(sos, g, nargout)


@builtin("tf2sos")
def _tf2sos(interp, args, nargout):
    nargs(args, 2, 4, "tf2sos")
    b, a = _rvec(args[0], "numerator"), _rvec(args[1], "denominator")
    order = _sos_args(list(args), 2, "tf2sos")
    z, p, k = _tf2zp_c(*_pad_equal(b, a))
    # zeros and poles at the origin only add delays; drop the ones both sides share
    sos, g = _zp2sos(z, p, float(np.real(k)), order)
    return _sos_out(sos, g, nargout)


@builtin("sos2tf")
def _sos2tf_b(interp, args, nargout):
    nargs(args, 1, 2, "sos2tf")
    sos = _sos_check(args[0], "sos2tf")
    g = float(np.prod(_rvec(args[1], "gain"))) if len(args) > 1 else 1.0
    b, a = _sos2tf(sos, g)
    return _row(b), _row(a)


def _sos2zp(sos: np.ndarray, g: float = 1.0):
    z, p = [], []
    k = g
    for row in sos:
        b, a = row[:3], row[3:]
        nb = np.flatnonzero(b != 0)
        na = np.flatnonzero(a != 0)
        if nb.size == 0:
            return np.zeros(0), np.zeros(0), 0.0
        k *= b[nb[0]] / a[na[0]]
        # second-order sections in z^-1: roots of b0 z^2 + b1 z + b2 (trailing zeros = fewer roots)
        bt = b[:nb[-1] + 1] if nb[-1] < 2 else b
        at = a[:na[-1] + 1] if na[-1] < 2 else a
        z.extend(np.roots(bt[nb[0]:]) if nb[0] == 0 else np.roots(bt))
        p.extend(np.roots(at))
    return np.array(z, dtype=complex), np.array(p, dtype=complex), float(k)


@builtin("sos2zp")
def _sos2zp_b(interp, args, nargout):
    nargs(args, 1, 2, "sos2zp")
    sos = _sos_check(args[0], "sos2zp")
    g = float(np.prod(_rvec(args[1], "gain"))) if len(args) > 1 else 1.0
    z, p, k = _sos2zp(sos, g)
    return _zcol(z), _zcol(p), k


def _ss2zp(A, B, C, D):
    ss = _sig()
    b, a = ss.ss2tf(A, B, C, D)
    b = np.atleast_2d(b)[0]
    tol = 1e-12 * max(1.0, float(np.max(np.abs(b))))
    b = np.where(np.abs(b) < tol, 0.0, b)
    return _tf2zp_c(b, a)


@builtin("ss2sos")
def _ss2sos(interp, args, nargout):
    nargs(args, 4, 7, "ss2sos")
    A, B, C, D = (fnum(a).astype(float) for a in args[:4])
    rest = list(args[4:])
    if rest and not is_text(rest[0]):
        iu = to_int(rest.pop(0), "input index")
        B, D = B[:, iu - 1:iu], D[:, iu - 1:iu]
    order = _sos_args(rest, 0, "ss2sos")
    z, p, k = _ss2zp(A, B, C, D)
    sos, g = _zp2sos(z, p, float(np.real(k)), order)
    return _sos_out(sos, g, nargout)


def _first_dim(x: np.ndarray, args: list, pos: int) -> int:
    if len(args) > pos:
        return to_int(args[pos], "dimension") - 1
    for ax, n in enumerate(x.shape):
        if n != 1:
            return ax
    return 0


@builtin("sosfilt")
def _sosfilt(interp, args, nargout):
    nargs(args, 2, 3, "sosfilt")
    sos = _sos_check(args[0], "sosfilt")
    x = fnum(args[1])
    axis = _first_dim(x, args, 2)
    if x.size == 0:
        return x
    return _sig().sosfilt(_sos_norm(sos), x, axis=axis)


def _sos_norm(sos: np.ndarray) -> np.ndarray:
    """Sections with a leading denominator coefficient of exactly one."""
    sos = np.array(sos, dtype=float)
    a0 = sos[:, 3:4]
    if np.any(a0 == 0):
        raise MatlabError("The first denominator coefficient of every section must be nonzero.")
    return sos / a0


def _maybe_register(name: str, fn) -> None:
    """Register a function only when no other module provides it (shared with the control functions)."""
    if name not in BUILTINS:
        builtin(name)(fn)


def _tf2ss_b(interp, args, nargout):
    nargs(args, 2, 2, "tf2ss")
    return _tf2ss(_rvec(args[0], "numerator"), _rvec(args[1], "denominator"))


def _ss2tf_b(interp, args, nargout):
    nargs(args, 4, 5, "ss2tf")
    A, B, C, D = (fnum(a).astype(float) for a in args[:4])
    iu = to_int(args[4], "input index") if len(args) > 4 else 1
    b, a = _sig().ss2tf(A, B, C, D, input=iu - 1)
    return np.atleast_2d(b), _row(a)


def _zp2ss_b(interp, args, nargout):
    nargs(args, 3, 3, "zp2ss")
    b, a = _zpk2tf(_vec(args[0]), _vec(args[1]), _sc(args[2], "gain"))
    return _tf2ss(np.real(b), np.real(a))


def _ss2zp_b(interp, args, nargout):
    nargs(args, 4, 5, "ss2zp")
    A, B, C, D = (fnum(a).astype(float) for a in args[:4])
    iu = to_int(args[4], "input index") if len(args) > 4 else 1
    z, p, k = _ss2zp(A, B[:, iu - 1:iu], C, D[:, iu - 1:iu])
    return _zcol(z), _zcol(p), float(np.real(k))


_maybe_register("tf2ss", _tf2ss_b)
_maybe_register("ss2tf", _ss2tf_b)
_maybe_register("zp2ss", _zp2ss_b)
_maybe_register("ss2zp", _ss2zp_b)


# -- reflection coefficients, lattices and linear prediction -----------------------------------------

def _poly2rc(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=complex if np.iscomplexobj(a) else float)
    if a.size == 0 or a[0] == 0:
        raise MatlabError("poly2rc: the leading coefficient must be nonzero.")
    a = a / a[0]
    n = a.size - 1
    k = np.zeros(n, dtype=a.dtype)
    cur = a[1:].copy()
    for m in range(n, 0, -1):
        km = cur[m - 1]
        k[m - 1] = km
        den = 1 - abs(km) ** 2
        if den == 0:
            if m > 1:
                k[:m - 1] = np.nan
            break
        if m > 1:
            cur = (cur[:m - 1] - km * np.conj(cur[m - 2::-1])) / den
    return k


def _rc2poly(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k)
    a = np.array([1.0], dtype=complex if np.iscomplexobj(k) else float)
    for km in k:
        a = np.concatenate([a, [0]]) + km * np.conj(np.concatenate([[0], a[::-1]]))
    return a


@builtin("poly2rc")
def _poly2rc_b(interp, args, nargout):
    nargs(args, 1, 2, "poly2rc")
    a = _vec(args[0], "polynomial")
    k = _poly2rc(a)
    if nargout > 1:
        efinal = _sc(args[1], "final prediction error") if len(args) > 1 else 0.0
        r0 = efinal / float(np.prod(1 - np.abs(k) ** 2)) if k.size else efinal
        return _col(k), r0
    return _col(k)


@builtin("rc2poly")
def _rc2poly_b(interp, args, nargout):
    nargs(args, 1, 2, "rc2poly")
    k = _vec(args[0], "reflection coefficients")
    a = _rc2poly(k)
    if nargout > 1:
        r0 = _sc(args[1], "zero-lag autocorrelation") if len(args) > 1 else 0.0
        return _row(a), r0 * float(np.prod(1 - np.abs(k) ** 2))
    return _row(a)


def _levinson(r: np.ndarray, n: int):
    """Levinson-Durbin recursion: prediction polynomial, error and reflection coefficients."""
    cplx = np.iscomplexobj(r)
    a = np.zeros(n + 1, dtype=complex if cplx else float)
    a[0] = 1.0
    e = r[0]
    k = np.zeros(n, dtype=a.dtype)
    for m in range(1, n + 1):
        acc = r[m] + np.dot(a[1:m], r[m - 1:0:-1])
        km = -acc / e if e != 0 else np.nan
        k[m - 1] = km
        prev = a[1:m].copy()
        a[1:m] = prev + km * np.conj(prev[::-1])
        a[m] = km
        e = e * (1 - abs(km) ** 2)
    return a, float(np.real(e)), k


@builtin("levinson")
def _levinson_b(interp, args, nargout):
    nargs(args, 1, 2, "levinson")
    r = fnum(args[0])
    cols, was_row = _columns(r)
    n = to_int(args[1], "order") if _given(args, 1) else cols.shape[0] - 1
    if n < 0 or n > cols.shape[0] - 1:
        raise MatlabError("levinson: the order must be less than the number of autocorrelation lags.")
    A, E, K = [], [], []
    for j in range(cols.shape[1]):
        a, e, k = _levinson(cols[:, j], n)
        A.append(a)
        E.append(e)
        K.append(k)
    out = (simplify(np.array(A)), _col(np.array(E)), simplify(np.array(K).T.reshape(n, -1)))
    return out[:max(nargout, 1)] if nargout > 1 else out[0]


def _autocorr_biased(x: np.ndarray, p: int) -> np.ndarray:
    n = x.size
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1))) if n > 0 else 1
    X = np.fft.fft(x, nfft)
    r = np.fft.ifft(np.abs(X) ** 2)[:p + 1] / n
    return r if np.iscomplexobj(x) else np.real(r)


def _ar_columns(args: list, name: str, default_order: bool = False):
    x = fnum(args[0])
    cols, _ = _columns(x)
    if len(args) > 1 and _given(args, 1):
        p = to_int(args[1], "model order")
    elif default_order:
        p = cols.shape[0] - 1
    else:
        raise MatlabError(f"{name}: the model order is required.")
    if p < 0 or p >= cols.shape[0] + (1 if default_order else 0):
        raise MatlabError(f"{name}: the model order must be smaller than the signal length.")
    return cols, p


def _ar_out(A, E, K, nargout):
    A = simplify(np.array(A))
    out = (A, _col(np.array(E)) if len(E) > 1 else float(E[0]),
           simplify(np.array(K).T) if K else np.zeros((0, 0)))
    return out[:max(nargout, 1)] if nargout > 1 else out[0]


@builtin("aryule")
def _aryule(interp, args, nargout):
    nargs(args, 2, 2, "aryule")
    cols, p = _ar_columns(args, "aryule")
    A, E, K = [], [], []
    for j in range(cols.shape[1]):
        a, e, k = _levinson(_autocorr_biased(cols[:, j], p), p)
        A.append(a)
        E.append(e)
        K.append(k)
    return _ar_out(A, E, K, nargout)


@builtin("lpc")
def _lpc(interp, args, nargout):
    nargs(args, 1, 2, "lpc")
    cols, p = _ar_columns(args, "lpc", default_order=True)
    A, E = [], []
    for j in range(cols.shape[1]):
        a, e, _k = _levinson(_autocorr_biased(cols[:, j], p), p)
        A.append(a)
        E.append(e)
    A = simplify(np.array(A))
    if nargout > 1:
        return A, _col(np.array(E)) if len(E) > 1 else float(E[0])
    return A


def _arburg(x: np.ndarray, p: int):
    n = x.size
    cplx = np.iscomplexobj(x)
    f = x.astype(complex if cplx else float).copy()
    b = f.copy()
    a = np.array([1.0], dtype=f.dtype)
    e = float(np.real(np.vdot(x, x))) / n
    k = np.zeros(p, dtype=f.dtype)
    for m in range(p):
        ef, eb = f[m + 1:], b[m:n - 1]
        den = np.real(np.vdot(ef, ef) + np.vdot(eb, eb))
        km = -2 * np.vdot(eb, ef) / den if den != 0 else 0.0
        k[m] = km
        newf = ef + km * eb
        newb = eb + np.conj(km) * ef
        f[m + 1:], b[m + 1:] = newf, newb
        a = np.concatenate([a, [0]]) + km * np.conj(np.concatenate([[0], a[::-1]]))
        e = e * (1 - abs(km) ** 2)
    return a, e, k


@builtin("arburg")
def _arburg_b(interp, args, nargout):
    nargs(args, 2, 2, "arburg")
    cols, p = _ar_columns(args, "arburg")
    A, E, K = [], [], []
    for j in range(cols.shape[1]):
        a, e, k = _arburg(cols[:, j], p)
        A.append(a)
        E.append(e)
        K.append(k)
    return _ar_out(A, E, K, nargout)


def _arcov(x: np.ndarray, p: int, modified: bool):
    n = x.size
    if n <= 2 * p if not modified else n <= 3 * p / 2:
        raise MatlabError("The signal is too short for this model order.")
    rows = np.array([x[i:i + p + 1][::-1] for i in range(n - p)])      # [x(t) x(t-1) ... x(t-p)]
    if modified:
        back = np.conj(np.array([x[i:i + p + 1] for i in range(n - p)]))
        rows = np.vstack([rows, back])
    target, reg = rows[:, 0], rows[:, 1:]
    coef = np.linalg.lstsq(reg, -target, rcond=None)[0] if p else np.zeros(0)
    a = np.concatenate([[1.0], coef])
    resid = target + reg @ coef if p else target
    e = float(np.real(np.vdot(resid, resid))) / (rows.shape[0] - (0 if not modified else 0))
    if modified:
        e = float(np.real(np.vdot(resid, resid))) / (2 * (n - p))
    else:
        e = float(np.real(np.vdot(resid, resid))) / (n - p)
    return a, e


def _arcov_builtin(name: str, modified: bool):
    def fn(interp, args, nargout):
        nargs(args, 2, 2, name)
        cols, p = _ar_columns(args, name)
        A, E = [], []
        for j in range(cols.shape[1]):
            a, e = _arcov(cols[:, j], p, modified)
            A.append(a)
            E.append(e)
        A = simplify(np.array(A))
        if nargout > 1:
            return A, _col(np.array(E)) if len(E) > 1 else float(E[0])
        return A
    builtin(name)(fn)


_arcov_builtin("arcov", False)
_arcov_builtin("armcov", True)


def _ladder(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lattice (k) and ladder (v) coefficients of the IIR filter b/a."""
    a = a / a[0]
    n = a.size - 1
    if b.size > n + 1:
        raise MatlabError("tf2latc: the numerator order must not exceed the denominator order.")
    b = np.concatenate([b / 1.0, np.zeros(n + 1 - b.size)]).astype(float)
    k = np.real(_poly2rc(a))
    v = np.zeros(n + 1)
    cur = a.copy()
    for m in range(n, -1, -1):
        v[m] = b[m]
        b[:m + 1] = b[:m + 1] - v[m] * cur[::-1]
        if m > 0:
            km = cur[m]
            cur = (cur[:m] - km * cur[m:0:-1]) / (1 - km * km)
    return k, v


@builtin("tf2latc")
def _tf2latc(interp, args, nargout):
    nargs(args, 1, 3, "tf2latc")
    rest, flags = _split_flags(list(args))
    b = _rvec(rest[0], "numerator")
    if len(rest) == 1:
        if b[0] == 0:
            raise MatlabError("tf2latc: the first coefficient must be nonzero.")
        k = np.real(_poly2rc(b / b[0]))
        if "max" in flags:
            k = np.real(_poly2rc(b[::-1] / b[-1]))
        return (_col(k), b[0]) if nargout > 1 else _col(k)
    a = _rvec(rest[1], "denominator")
    if b.size == 1 and nargout <= 2 and np.all(b == 1) or (b.size == 1 and a.size > 1):
        k = np.real(_poly2rc(a))
        v = np.zeros(a.size)
        v[0] = b[0] / a[0]
        return (_col(k), _col(v)) if nargout > 1 else _col(k)
    k, v = _ladder(b / a[0], a)
    return (_col(k), _col(v)) if nargout > 1 else _col(k)


def _latc2tf(k: np.ndarray, v: np.ndarray | None, kind: str):
    a = np.real(_rc2poly(k))
    if v is None:
        if kind in ("allpole", "iir"):
            return np.array([1.0]), a
        if kind == "allpass":
            return a[::-1], a
        if kind == "max":
            return a[::-1], np.array([1.0])
        return a, np.array([1.0])
    n = k.size
    v = np.concatenate([v, np.zeros(max(n + 1 - v.size, 0))])
    num = np.zeros(n + 1)
    polys = [np.array([1.0])]
    cur = np.array([1.0])
    for km in k:
        cur = np.concatenate([cur, [0]]) + km * np.concatenate([[0], cur[::-1]])
        polys.append(cur)
    for m in range(min(n + 1, v.size)):
        num[:m + 1] += v[m] * polys[m][::-1]
    return num, a


@builtin("latc2tf")
def _latc2tf_b(interp, args, nargout):
    nargs(args, 1, 2, "latc2tf")
    k = _rvec(args[0], "lattice coefficients")
    kind, v = "fir", None
    if len(args) > 1:
        if is_text(args[1]):
            kind = text(args[1]).lower()
            if kind not in ("fir", "min", "max", "allpole", "iir", "allpass"):
                raise MatlabError("latc2tf: the option must be 'fir', 'min', 'max', 'allpole' or 'allpass'.")
        else:
            v = _rvec(args[1], "ladder coefficients")
    num, den = _latc2tf(k, v, kind)
    return _row(num), _row(den)


@builtin("latcfilt")
def _latcfilt(interp, args, nargout):
    nargs(args, 2, 4, "latcfilt")
    ss = _sig()
    k = _rvec(args[0], "lattice coefficients")
    if len(args) == 2:
        x = fnum(args[1])
        axis = _first_dim(x, [], 0)
        a = np.real(_rc2poly(k))
        f = ss.lfilter(a, [1.0], x, axis=axis)
        g = ss.lfilter(a[::-1], [1.0], x, axis=axis)
        return (f, g) if nargout > 1 else f
    v = _rvec(args[1], "ladder coefficients")
    x = fnum(args[2])
    axis = _first_dim(x, [], 0)
    a = np.real(_rc2poly(k))
    if v.size == 1:
        num = np.array([v[0]])
    else:
        num, a = _latc2tf(k, v, "fir")
    f = ss.lfilter(num, a, x, axis=axis)
    g = ss.lfilter(a[::-1], a, x, axis=axis)
    return (f, g) if nargout > 1 else f


# -- filter analysis ----------------------------------------------------------------------------------

class _Filt:
    """A filter given as (b, a), as second-order sections or as a digitalFilter object."""

    def __init__(self, b=None, a=None, sos=None):
        self.b, self.a, self.sos = b, a, sos

    def tf(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sos is not None:
            return _sos2tf(self.sos)
        return self.b, self.a

    def response(self, w: np.ndarray) -> np.ndarray:
        x = np.exp(-1j * np.asarray(w, dtype=float))
        if self.sos is not None:
            h = np.ones(x.shape, dtype=complex)
            for row in self.sos:
                h = h * np.polyval(row[2::-1], x) / np.polyval(row[5:2:-1], x)
            return h
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.polyval(self.b[::-1], x) / np.polyval(self.a[::-1], x)

    def impulse(self, n: int) -> np.ndarray:
        x = np.zeros(n)
        if n:
            x[0] = 1.0
        return self.run(x)

    def run(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        ss = _sig()
        if self.sos is not None:
            return ss.sosfilt(_sos_norm(self.sos), x, axis=axis)
        return ss.lfilter(self.b, self.a, x, axis=axis)

    def order(self) -> int:
        b, a = self.tf()
        nb = np.flatnonzero(b != 0)
        na = np.flatnonzero(a != 0)
        return int(max(nb[-1] if nb.size else 0, na[-1] if na.size else 0))

    def is_fir(self) -> bool:
        _b, a = self.tf()
        return not np.any(a[1:] != 0)


def _take_filter(args: list, name: str) -> tuple[_Filt, list]:
    """Read the filter from the front of the argument list; returns it and the remaining arguments."""
    if not args:
        raise MatlabError(f"{name}: a filter is required.")
    first = args[0]
    if isinstance(first, DigitalFilter):
        return first.filt, list(args[1:])
    if _is_sos(first):
        return _Filt(sos=fnum(first).astype(float)), list(args[1:])
    b = _vec(first, "numerator")
    rest = list(args[1:])
    a = np.array([1.0])
    if rest and not is_text(rest[0]):
        cand = rest.pop(0)
        if isinstance(cand, np.ndarray) and cand.size:
            a = _vec(cand, "denominator")
    if b.size == 0:
        b = np.array([0.0])
    if a[0] == 0:
        raise MatlabError(f"{name}: the first denominator coefficient must be nonzero.")
    return _Filt(b=simplify(b), a=simplify(a)), rest


def _freq_grid(rest: list, name: str, default_n: int = 512):
    """Frequencies for freqz-like functions: (w in rad/sample, axis values, fs, template for the shape)."""
    rest, flags = _split_flags(rest)
    for fl in flags:
        if fl not in ("whole", "half"):
            raise MatlabError(f"{name}: unknown option '{fl}'.")
    whole = "whole" in flags
    fs = _sc(rest[1], "sample rate") if _given(rest, 1) else None
    if fs is not None and fs <= 0:
        raise MatlabError(f"{name}: the sample rate must be positive.")
    spec = rest[0] if _given(rest, 0) else None
    if spec is not None and fnum(spec).size > 1:
        pts = fnum(spec)
        f = np.real(pts).astype(float)
        w = f * (2 * np.pi / fs) if fs is not None else f
        return w.flatten(order="F"), f, fs, pts
    n = int(_sc(spec, "number of points")) if spec is not None else default_n
    if n < 1:
        raise MatlabError(f"{name}: the number of points must be positive.")
    w = np.arange(n) * ((2 * np.pi if whole else np.pi) / n)
    axis_vals = w * fs / (2 * np.pi) if fs is not None else w
    return w, axis_vals, fs, None


def _shape_out(v: np.ndarray, template) -> np.ndarray:
    if template is None:
        return _col(v)
    return np.asarray(v).reshape(template.shape, order="F")


def _freq_label(fs) -> str:
    return "Frequency (Hz)" if fs is not None else "Normalized Frequency (\\times\\pi rad/sample)"


def freqz_impl(interp, args, nargout):
    filt, rest = _take_filter(list(args), "freqz")
    w, axis_vals, fs, template = _freq_grid(rest, "freqz")
    h = filt.response(w)
    if nargout == 0:
        x = axis_vals if fs is not None else np.asarray(axis_vals) / np.pi
        with np.errstate(divide="ignore"):
            mag = 20 * np.log10(np.abs(h))
        ph = np.degrees(np.unwrap(np.angle(h)))
        _plot(interp, "subplot", [_f(2), _f(1), _f(1)])
        _plot(interp, "plot", [_row(x), _row(mag)])
        _plot(interp, "ylabel", [char("Magnitude (dB)")])
        _plot(interp, "xlabel", [char(_freq_label(fs))])
        _plot(interp, "grid", [char("on")])
        _plot(interp, "subplot", [_f(2), _f(1), _f(2)])
        _plot(interp, "plot", [_row(x), _row(ph)])
        _plot(interp, "ylabel", [char("Phase (degrees)")])
        _plot(interp, "xlabel", [char(_freq_label(fs))])
        _plot(interp, "grid", [char("on")])
        return None
    hh = _shape_out(h.astype(complex), template)
    return (hh, _shape_out(np.asarray(axis_vals, dtype=float), template)) if nargout > 1 else hh


@builtin("grpdelay")
def _grpdelay(interp, args, nargout):
    filt, rest = _take_filter(list(args), "grpdelay")
    w, axis_vals, fs, template = _freq_grid(rest, "grpdelay")
    b, a = filt.tf()
    c = np.convolve(b, np.conj(a[::-1]))
    x = np.exp(-1j * w)
    num = np.polyval((c * np.arange(c.size))[::-1], x)
    den = np.polyval(c[::-1], x)
    small = np.abs(den) < 10 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(c))))
    with np.errstate(divide="ignore", invalid="ignore"):
        gd = np.real(num / np.where(small, 1.0, den)) - (a.size - 1)
    gd = np.where(small, 0.0, gd)
    if nargout == 0:
        xx = axis_vals if fs is not None else np.asarray(axis_vals) / np.pi
        _plot(interp, "plot", [_row(xx), _row(gd)])
        _plot(interp, "xlabel", [char(_freq_label(fs))])
        _plot(interp, "ylabel", [char("Group delay (samples)")])
        _plot(interp, "grid", [char("on")])
        return None
    g = _shape_out(gd, template)
    return (g, _shape_out(np.asarray(axis_vals, dtype=float), template)) if nargout > 1 else g


@builtin("phasez")
def _phasez(interp, args, nargout):
    filt, rest = _take_filter(list(args), "phasez")
    w, axis_vals, fs, template = _freq_grid(rest, "phasez")
    h = filt.response(w)
    phi = np.unwrap(np.angle(h))
    if nargout == 0:
        xx = axis_vals if fs is not None else np.asarray(axis_vals) / np.pi
        _plot(interp, "plot", [_row(xx), _row(phi)])
        _plot(interp, "xlabel", [char(_freq_label(fs))])
        _plot(interp, "ylabel", [char("Phase (radians)")])
        _plot(interp, "grid", [char("on")])
        return None
    p = _shape_out(phi, template)
    return (p, _shape_out(np.asarray(axis_vals, dtype=float), template)) if nargout > 1 else p


def _impz_length(filt: _Filt) -> int:
    b, a = filt.tf()
    nb = int(np.flatnonzero(b != 0)[-1]) + 1 if np.any(b != 0) else 1
    if not np.any(a[1:] != 0):
        return nb
    p = np.roots(a)
    r = np.abs(p)
    rmax = float(np.max(r))
    if rmax > 1 + 1e-5:                      # unstable: until the response has grown a millionfold
        n = int(np.ceil(6 / np.log10(rmax)))
    elif rmax > 1 - 1e-5:                    # oscillating: five periods of the slowest oscillation
        ang = np.abs(np.angle(p[r > 1 - 1e-5]))
        ang = ang[ang > 1e-8]
        n = int(np.ceil(5 * 2 * np.pi / np.min(ang))) if ang.size else 100
    else:                                    # decaying: until the slowest mode has fallen to 5e-5
        mult = int(np.sum(np.abs(p - p[np.argmax(r)]) < 1e-3))
        n = int(np.ceil(mult * np.log10(5e-5) / np.log10(rmax)))
    return max(n + nb - 1, 2)


def _impz_common(interp, args, nargout, name: str, step: bool):
    filt, rest = _take_filter(list(args), name)
    fs = _sc(rest[1], "sample rate") if _given(rest, 1) else 1.0
    if _given(rest, 0):
        nv = np.real(fnum(rest[0])).flatten()
        if nv.size == 1:
            idx = np.arange(int(nv[0]))
        else:
            idx = nv.astype(int)
            if np.any(idx < 0):
                raise MatlabError(f"{name}: sample numbers must not be negative.")
    else:
        idx = np.arange(_impz_length(filt))
    total = int(idx.max()) + 1 if idx.size else 0
    h = filt.run(np.ones(total)) if step else filt.impulse(total)
    h = h[idx] if idx.size else np.zeros(0)
    t = idx / fs
    if nargout == 0:
        _plot(interp, "stem", [_row(t), _row(np.real(h))])
        _plot(interp, "xlabel", [char("n (samples)" if not _given(rest, 1) else "Time (seconds)")])
        _plot(interp, "ylabel", [char("Amplitude")])
        return None
    return (_col(h), _col(t)) if nargout > 1 else _col(h)


@builtin("impz")
def _impz(interp, args, nargout):
    return _impz_common(interp, args, nargout, "impz", False)


@builtin("stepz")
def _stepz(interp, args, nargout):
    return _impz_common(interp, args, nargout, "stepz", True)


@builtin("freqs")
def _freqs(interp, args, nargout):
    nargs(args, 2, 3, "freqs")
    b, a = _trim(_vec(args[0], "numerator")), _trim(_vec(args[1], "denominator"))
    template = None
    if _given(args, 2) and fnum(args[2]).size > 1:
        template = fnum(args[2])
        w = np.real(template).astype(float).flatten(order="F")
    else:
        n = int(_sc(args[2], "number of points")) if _given(args, 2) else 200
        roots = np.concatenate([np.roots(b), np.roots(a)])
        mags = np.abs(roots[np.abs(roots) > 0])
        lo = np.floor(np.log10(np.min(mags))) - 1 if mags.size else -1.0
        hi = np.ceil(np.log10(np.max(mags))) + 1 if mags.size else 1.0
        w = np.logspace(lo, hi, n)
    s = 1j * w
    h = np.polyval(b, s) / np.polyval(a, s)
    if nargout == 0:
        with np.errstate(divide="ignore"):
            _plot(interp, "subplot", [_f(2), _f(1), _f(1)])
            _plot(interp, "loglog", [_row(w), _row(np.abs(h))])
            _plot(interp, "ylabel", [char("Magnitude")])
            _plot(interp, "grid", [char("on")])
            _plot(interp, "subplot", [_f(2), _f(1), _f(2)])
            _plot(interp, "semilogx", [_row(w), _row(np.degrees(np.unwrap(np.angle(h))))])
            _plot(interp, "ylabel", [char("Phase (degrees)")])
            _plot(interp, "xlabel", [char("Frequency (rad/s)")])
            _plot(interp, "grid", [char("on")])
        return None
    if template is not None:
        return (h.reshape(template.shape, order="F"), w.reshape(template.shape, order="F")) if nargout > 1 \
            else h.reshape(template.shape, order="F")
    return (_row(h), _row(w)) if nargout > 1 else _row(h)


@builtin("zplane")
def _zplane(interp, args, nargout):
    nargs(args, 1, 2, "zplane")
    first = args[0]
    if isinstance(first, DigitalFilter) or _is_sos(first):
        filt, _ = _take_filter(list(args), "zplane")
        b, a = filt.tf()
        z, p, _k = _tf2zpk_d(b, a)
    else:
        v0 = fnum(first)
        v1 = fnum(args[1]) if len(args) > 1 else np.zeros((0, 0))
        columns = (v0.size == 0 or v0.shape[1] == 1) and (v1.size == 0 or v1.shape[1] == 1) \
            and not (v0.size == 1 and v1.size == 1)
        if columns:
            z, p = v0.flatten(), v1.flatten()
        else:
            z, p, _k = _tf2zpk_d(v0.flatten(), v1.flatten() if v1.size else np.array([1.0]))
    th = np.linspace(0, 2 * np.pi, 361)
    _plot(interp, "plot", [_row(np.cos(th)), _row(np.sin(th)), char("k:")])
    _plot(interp, "hold", [char("on")])
    hz = interp.call_named("plot", [_row(np.real(z)), _row(np.imag(z)), char("bo")], 1, interp.frame) \
        if np.size(z) else [np.zeros((0, 0))]
    hp = interp.call_named("plot", [_row(np.real(p)), _row(np.imag(p)), char("bx")], 1, interp.frame) \
        if np.size(p) else [np.zeros((0, 0))]
    _plot(interp, "axis", [char("equal")])
    _plot(interp, "grid", [char("on")])
    _plot(interp, "xlabel", [char("Real Part")])
    _plot(interp, "ylabel", [char("Imaginary Part")])
    _plot(interp, "hold", [char("off")])
    if nargout:
        return (hz[0], hp[0])[:nargout] if nargout > 1 else hz[0]
    return None


@builtin("filtord")
def _filtord(interp, args, nargout):
    nargs(args, 1, 2, "filtord")
    filt, _ = _take_filter(list(args), "filtord")
    return float(filt.order())


@builtin("fftfilt")
def _fftfilt(interp, args, nargout):
    nargs(args, 2, 3, "fftfilt")
    x = fnum(args[1])
    if isinstance(args[0], DigitalFilter):
        if not args[0].filt.is_fir():
            raise MatlabError("fftfilt: the filter must be an FIR filter.")
        b = args[0].filt.tf()[0]
    else:
        b = _vec(args[0], "filter coefficients")
    cols, was_row = _columns(x)
    y = _sig().lfilter(b, [1.0], cols, axis=0)
    return _restore(simplify(y), was_row)


# -- digitalFilter objects and designfilt -------------------------------------------------------------

class DigitalFilter(MObject):
    """The filter object returned by designfilt: coefficients plus the design specification."""

    _METHODS = {"filter", "filtfilt", "freqz", "grpdelay", "phasez", "impz", "stepz", "zplane", "filtord",
                "fftfilt", "tf", "zpk", "ss", "isstable", "islinphase", "isminphase", "ismaxphase",
                "isallpass", "firtype", "double", "info", "isfir", "sos"}

    def __init__(self, filt: _Filt, props: dict):
        self.filt = filt
        self.props = props

    def class_name(self) -> str:
        return "digitalFilter"

    def has_method(self, name: str) -> bool:
        return name in self._METHODS

    def coefficients(self) -> np.ndarray:
        if self.filt.sos is not None:
            return self.filt.sos.copy()
        return _row(self.filt.b)

    def get_field(self, interp, name: str):
        if name == "Coefficients":
            return self.coefficients()
        for key, val in self.props.items():
            if key.lower() == name.lower():
                return val
        if name in self._METHODS:
            return self.call_method(interp, name, [self], 1)[0]
        raise MatlabError(f"Unrecognized property '{name}' for class 'digitalFilter'.")

    def set_field(self, interp, name: str, value):
        raise MatlabError("digitalFilter properties are read-only; design a new filter with designfilt.")

    def _zpk(self):
        if self.filt.sos is not None:
            return _sos2zp(self.filt.sos)
        return _tf2zpk_d(self.filt.b, self.filt.a)

    def call_method(self, interp, name: str, args: list, nargout: int) -> list:
        ss = _sig()
        me = next(i for i, a in enumerate(args) if a is self)
        rest = [a for i, a in enumerate(args) if i != me]
        filt = self.filt
        if name == "filter":
            if not rest:
                raise MatlabError("filter: a signal is required.")
            x = fnum(rest[0])
            axis = _first_dim(x, rest, 1)
            return [filt.run(x, axis=axis)]
        if name == "filtfilt":
            x = fnum(rest[0])
            axis = _first_dim(x, [], 0)
            pad = 3 * filt.order()
            if x.shape[axis] <= pad:
                raise MatlabError(f"filtfilt: the data must have more than {pad} samples (three times the "
                                  "filter order).")
            if filt.sos is not None:
                return [ss.sosfiltfilt(_sos_norm(filt.sos), x, axis=axis, padtype="odd", padlen=pad)]
            return [ss.filtfilt(filt.b, filt.a, x, axis=axis, padtype="odd", padlen=pad)]
        if name in ("freqz", "grpdelay", "phasez", "impz", "stepz", "zplane", "filtord", "fftfilt"):
            out = BUILTINS[name](interp, list(args), nargout)
            if out is None:
                return []
            return [to_matlab(v) for v in out] if isinstance(out, tuple) else [to_matlab(out)]
        if name == "tf":
            b, a = filt.tf()
            return [_row(b), _row(a)]
        if name == "zpk":
            z, p, k = self._zpk()
            return [_zcol(z), _zcol(p), _f(np.real(k))]
        if name == "ss":
            b, a = filt.tf()
            return list(_tf2ss(*_pad_equal(np.real(b), np.real(a))))
        if name == "sos":
            if filt.sos is not None:
                return [filt.sos.copy()]
            z, p, k = self._zpk()
            return [_sos_out(*_zp2sos(z, p, float(np.real(k))), 1)]
        if name == "double":
            return [self.coefficients()]
        if name == "filtord":
            return [_f(filt.order())]
        if name == "isfir":
            return [np.array([[filt.is_fir()]])]
        z, p, _k = self._zpk()
        b, _a = filt.tf()
        if name == "isstable":
            return [np.array([[bool(np.all(np.abs(p) < 1)) if p.size else True]])]
        if name == "isminphase":
            return [np.array([[bool(np.all(np.abs(z) <= 1 + 1e-10) and np.all(np.abs(p) < 1))]])]
        if name == "ismaxphase":
            return [np.array([[bool(np.all(np.abs(z) >= 1 - 1e-10) and np.all(np.abs(p) < 1))]])]
        if name == "isallpass":
            bb, aa = filt.tf()
            n = max(bb.size, aa.size)
            bb, aa = _pad_equal(bb, aa)
            scale = bb[-1] / aa[0] if aa[0] != 0 else 0
            return [np.array([[bool(n > 0 and np.allclose(bb[::-1], scale * aa, atol=1e-9))]])]
        if name in ("islinphase", "firtype"):
            sym = filt.is_fir() and np.allclose(b, b[::-1], atol=1e-10 * max(1.0, float(np.max(np.abs(b)))))
            anti = filt.is_fir() and np.allclose(b, -b[::-1], atol=1e-10 * max(1.0, float(np.max(np.abs(b)))))
            if name == "islinphase":
                return [np.array([[bool(sym or anti)]])]
            if not (sym or anti):
                raise MatlabError("firtype: the filter must be a linear-phase FIR filter.")
            odd_len = b.size % 2 == 1
            return [_f((1 if odd_len else 2) if sym else (3 if odd_len else 4))]
        if name == "info":
            return [char("\n".join(self.display_lines()))]
        return super().call_method(interp, name, args, nargout)

    def display_lines(self) -> list[str]:
        c = self.coefficients()
        rows = [("Coefficients", f"[{c.shape[0]}x{c.shape[1]} double]")]
        for key, val in self.props.items():
            if is_text(val):
                shown = f"'{text(val)}'"
            elif isinstance(val, np.ndarray) and val.size == 1:
                shown = f"{float(np.real(val.flat[0])):g}"
            elif isinstance(val, np.ndarray):
                shown = f"[{val.shape[0]}x{val.shape[1]} double]"
            else:
                shown = str(val)
            rows.append((key, shown))
        width = max(len(k) for k, _ in rows)
        out = [" digitalFilter with properties:", "", f"   {'Coefficients'.rjust(width)}: {rows[0][1]}", "",
               "   Specifications:"]
        out.extend(f"   {k.rjust(width)}: {v}" for k, v in rows[1:])
        return out


def _ripple_dev(rp_db: float) -> float:
    g = 10 ** (rp_db / 20)
    return (g - 1) / (g + 1)


def _minorder_equiripple(edges: np.ndarray, amps: np.ndarray, devs: np.ndarray, even_only: bool) -> np.ndarray:
    """Lowest-order equiripple FIR filter that keeps the error inside `devs` (one per band)."""
    wts = np.max(devs) / devs
    width = float(np.min(edges[2::2] - edges[1:-1:2])) / 2          # cycles per sample
    est = (-20 * np.log10(np.sqrt(np.max(devs) * np.min(devs))) - 13) / (14.6 * width)
    n = max(int(np.ceil(est)), 3)
    a2 = np.repeat(amps, 2)

    def ok(order):
        h, _delta = _remez(order, edges, a2, wts, "symmetric")
        taps = np.arange(h.size)
        for b in range(devs.size):
            # measure the error on a grid much finer than the one the design used
            fq = np.linspace(edges[2 * b], edges[2 * b + 1], 60 * h.size)
            mag = np.abs(np.exp(-1j * np.pi * np.outer(fq, taps)) @ h)
            if np.max(np.abs(mag - amps[b])) > devs[b] * (1 + 1e-9):
                return False, h
        return True, h

    step = 2 if even_only else 1
    if even_only and n % 2:
        n += 1
    good, h = ok(n)
    if good:
        while n - step >= 3:
            g2, h2 = ok(n - step)
            if not g2:
                break
            n, h = n - step, h2
    else:
        for _ in range(400):
            n += step
            good, h = ok(n)
            if good:
                break
        else:
            raise MatlabError("designfilt: could not find a filter order that meets the specification.")
    return h


_DF_RESPONSES = ("lowpassfir", "highpassfir", "bandpassfir", "bandstopfir", "lowpassiir", "highpassiir",
                 "bandpassiir", "bandstopiir", "differentiatorfir", "hilbertfir", "arbmagfir")
_DF_OPTIONS = ("FilterOrder", "CutoffFrequency", "CutoffFrequency1", "CutoffFrequency2", "PassbandFrequency",
               "PassbandFrequency1", "PassbandFrequency2", "StopbandFrequency", "StopbandFrequency1",
               "StopbandFrequency2", "HalfPowerFrequency", "HalfPowerFrequency1", "HalfPowerFrequency2",
               "PassbandRipple", "PassbandRipple1", "PassbandRipple2", "StopbandAttenuation",
               "StopbandAttenuation1", "StopbandAttenuation2", "SampleRate", "DesignMethod", "Window",
               "ScalePassband", "PassbandWeight", "StopbandWeight", "StopbandWeight1", "StopbandWeight2",
               "PassbandWeight1", "PassbandWeight2", "MatchExactly", "TransitionWidth", "Frequencies",
               "Amplitudes", "NumeratorOrder", "DenominatorOrder", "Weights", "BandFrequencies",
               "BandAmplitudes", "NumBands")


def designfilt_impl(interp, args):
    if not args or not is_text(args[0]):
        raise MatlabError("designfilt: the first input must be a response type such as 'lowpassfir'.")
    resp = text(args[0]).lower()
    if resp not in _DF_RESPONSES:
        raise MatlabError(f"designfilt: unknown response '{text(args[0])}'. Use one of: "
                          + ", ".join(_DF_RESPONSES) + ".")
    opts = _name_values(list(args), 1, _DF_OPTIONS, "designfilt")
    fs = _sc(opts["samplerate"], "SampleRate") if "samplerate" in opts else 2.0
    nyq = fs / 2
    used: dict = {}

    def num(key, default=None):
        if key.lower() in opts:
            v = _sc(opts[key.lower()], key)
            used[key] = v
            return v
        if default is not None:
            used[key] = default
        return default

    def freq(key):
        v = num(key)
        if v is None:
            return None
        if not 0 < v < nyq:
            raise MatlabError(f"designfilt: {key} must lie between 0 and half the sample rate.")
        return v / nyq

    def has(*keys):
        return all(k.lower() in opts for k in keys)

    method = text(opts["designmethod"]).lower() if "designmethod" in opts else None
    fir = resp.endswith("fir")
    band = resp.startswith("band")
    kind = {"lowp": "low", "high": "high", "band": "bandpass" if resp.startswith("bandpass") else "stop"}.get(
        resp[:4], resp)
    order = None
    if "filterorder" in opts:
        order = to_int(opts["filterorder"], "FilterOrder")
        if order < 1:
            raise MatlabError("designfilt: FilterOrder must be a positive integer.")
        used["FilterOrder"] = float(order)
    b = None

    def missing():
        raise MatlabError(f"designfilt: this set of options does not describe a '{resp}' filter. Give for "
                          "example 'FilterOrder' with cutoff frequencies, or passband and stopband "
                          "frequencies with 'PassbandRipple' and 'StopbandAttenuation'.")

    if resp in ("lowpassfir", "highpassfir", "bandpassfir", "bandstopfir"):
        if band:
            cut_keys = ("CutoffFrequency1", "CutoffFrequency2")
        else:
            cut_keys = ("CutoffFrequency",)
        if order is not None and has(*cut_keys):
            method = method or "window"
            if method != "window":
                raise MatlabError("designfilt: cutoff frequencies go with 'DesignMethod', 'window'.")
            wn = np.array([freq(k) for k in cut_keys])
            wspec = opts.get("window")
            scale = True
            if "scalepassband" in opts:
                v = opts["scalepassband"]
                scale = text(v).lower() in ("true", "on") if is_text(v) else bool(np.ravel(v)[0])
            n = order
            pass_zero = kind in ("low", "stop")
            if (kind in ("high", "stop")) and n % 2 == 1:
                raise MatlabError("designfilt: highpass and bandstop window designs need an even FilterOrder.")
            win = _window_from(interp, wspec, n + 1, "designfilt")
            b = _fir1_core(n, wn, pass_zero, win, scale)
        else:
            if band and kind == "bandpass":
                keys = ("StopbandFrequency1", "PassbandFrequency1", "PassbandFrequency2", "StopbandFrequency2")
                amps = np.array([0.0, 1.0, 0.0])
            elif band:
                keys = ("PassbandFrequency1", "StopbandFrequency1", "StopbandFrequency2", "PassbandFrequency2")
                amps = np.array([1.0, 0.0, 1.0])
            elif kind == "low":
                keys = ("PassbandFrequency", "StopbandFrequency")
                amps = np.array([1.0, 0.0])
            else:
                keys = ("StopbandFrequency", "PassbandFrequency")
                amps = np.array([0.0, 1.0])
            if not has(*keys):
                missing()
            inner = np.array([freq(k) for k in keys])
            if np.any(np.diff(inner) <= 0):
                raise MatlabError("designfilt: the band edges are not in increasing order for this response.")
            edges = np.concatenate([[0.0], inner, [1.0]])
            if order is not None:
                method = method or "equiripple"
                wts = []
                for i, am in enumerate(amps):
                    side = "" if not band else ("1" if i == 0 else ("2" if i == 2 else ""))
                    key = ("PassbandWeight" if am else "StopbandWeight") + (side if (am != amps[1]) else "")
                    wts.append(num(key) or 1.0)
                wts = np.array(wts)
                if amps[-1] != 0 and order % 2 == 1:
                    raise MatlabError("designfilt: this response needs an even FilterOrder.")
                if method == "equiripple":
                    b, _d = _remez(order, edges, np.repeat(amps, 2), wts, "symmetric")
                elif method == "ls":
                    b = _firls_core(order, edges, np.repeat(amps, 2), wts, "symmetric")
                else:
                    raise MatlabError("designfilt: with band edges and an order the design method must be "
                                      "'equiripple' or 'ls'.")
            else:
                method = method or "equiripple"
                devs = []
                for i, am in enumerate(amps):
                    side = "" if not band else ("1" if i == 0 else ("2" if i == 2 else ""))
                    if am:
                        key = "PassbandRipple" + (side if amps[1] == 0 else "")
                        devs.append(_ripple_dev(num(key, 1.0)))
                    else:
                        key = "StopbandAttenuation" + (side if amps[1] != 0 else "")
                        devs.append(10 ** (-num(key, 60.0) / 20))
                devs = np.array(devs)
                if method == "equiripple":
                    b = _minorder_equiripple(edges, amps, devs, even_only=amps[-1] != 0)
                elif method == "kaiserwin":
                    delta = float(np.min(devs))
                    att = -20 * np.log10(delta)
                    width = float(np.min(inner[1::2] - inner[0::2]))
                    n = max(int(np.ceil((att - 7.95) / (2.285 * np.pi * width))), 1)
                    if amps[-1] != 0 and n % 2:
                        n += 1
                    beta = 0.1102 * (att - 8.7) if att > 50 else (
                        0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21) if att >= 21 else 0.0)
                    wn = 0.5 * (inner[0::2] + inner[1::2])
                    b = _fir1_core(n, wn, amps[0] != 0, _w_kaiser(n + 1, beta), True)
                else:
                    raise MatlabError("designfilt: minimum-order FIR designs use 'equiripple' or 'kaiserwin'.")
        filt = _Filt(b=np.asarray(b, dtype=float), a=np.array([1.0]))
    elif resp in ("lowpassiir", "highpassiir", "bandpassiir", "bandstopiir"):
        ss = _sig()
        btype = {"low": "lowpass", "high": "highpass", "bandpass": "bandpass", "stop": "bandstop"}[kind]
        sfx = ("1", "2") if band else ("",)
        hp_keys = tuple("HalfPowerFrequency" + s for s in sfx)
        pb_keys = tuple("PassbandFrequency" + s for s in sfx)
        sb_keys = tuple("StopbandFrequency" + s for s in sfx)

        def wvec(keys):
            v = np.array([freq(k) for k in keys])
            return v if v.size > 1 else float(v[0])

        if order is not None:
            n = order
            if band:
                if n % 2:
                    raise MatlabError("designfilt: bandpass and bandstop IIR filters need an even FilterOrder.")
                n //= 2
            if has(*hp_keys):
                method = method or "butter"
                if method != "butter":
                    raise MatlabError("designfilt: half-power frequencies go with 'DesignMethod', 'butter'.")
                z, p, k = ss.butter(n, wvec(hp_keys), btype=btype, output="zpk")
            elif has(*pb_keys) and has("StopbandAttenuation") and not has(*sb_keys):
                method = method or "ellip"
                z, p, k = ss.ellip(n, num("PassbandRipple", 1.0), num("StopbandAttenuation", 60.0), wvec(pb_keys),
                                   btype=btype, output="zpk")
            elif has(*pb_keys) and not has(*sb_keys):
                method = method or "cheby1"
                z, p, k = ss.cheby1(n, num("PassbandRipple", 1.0), wvec(pb_keys), btype=btype, output="zpk")
            elif has(*sb_keys) and not has(*pb_keys):
                method = method or "cheby2"
                z, p, k = ss.cheby2(n, num("StopbandAttenuation", 60.0), wvec(sb_keys), btype=btype, output="zpk")
            else:
                missing()
        else:
            if not (has(*pb_keys) and has(*sb_keys)):
                missing()
            method = method or "butter"
            wp, ws = wvec(pb_keys), wvec(sb_keys)
            rp, rs = num("PassbandRipple", 1.0), num("StopbandAttenuation", 60.0)
            exact = text(opts["matchexactly"]).lower() if "matchexactly" in opts else None
            if method == "butter":
                nn, wn = _buttord(interp, [np.atleast_2d(wp), np.atleast_2d(ws), _f(rp), _f(rs)], 2)
                if exact == "passband":
                    nn2, wn2 = ss.buttord(wp, ws, rp, rs)
                    wn = np.atleast_2d(wn2)
                wnv = np.ravel(wn)
                z, p, k = ss.butter(int(np.ravel(nn)[0]), wnv if wnv.size > 1 else float(wnv[0]), btype=btype,
                                    output="zpk")
            elif method == "cheby1":
                nn, wn = ss.cheb1ord(wp, ws, rp, rs)
                z, p, k = ss.cheby1(nn, rp, wn, btype=btype, output="zpk")
            elif method == "cheby2":
                nn, wn = ss.cheb2ord(wp, ws, rp, rs)
                z, p, k = ss.cheby2(nn, rs, wn, btype=btype, output="zpk")
            elif method == "ellip":
                nn, wn = ss.ellipord(wp, ws, rp, rs)
                z, p, k = ss.ellip(nn, rp, rs, wn, btype=btype, output="zpk")
            else:
                raise MatlabError("designfilt: the IIR design method must be 'butter', 'cheby1', 'cheby2' or "
                                  "'ellip'.")
        sos, g = _zp2sos(z, p, float(np.real(k)))
        sos[0, :3] *= g
        filt = _Filt(sos=sos)
    elif resp == "differentiatorfir":
        if order is None:
            raise MatlabError("designfilt: a differentiator needs 'FilterOrder'.")
        method = method or "equiripple"
        fp, fst = freq("PassbandFrequency"), freq("StopbandFrequency")
        if fp is not None and fst is not None:
            f = np.array([0.0, fp, fst, 1.0])
            am = np.array([0.0, np.pi * fp, 0.0, 0.0])
            w = np.array([1.0, 1.0])
        else:
            top = 1.0 if order % 2 == 1 else 0.9
            f = np.array([0.0, top])
            am = np.array([0.0, np.pi * top])
            w = np.array([1.0])
        if method == "equiripple":
            b, _d = _remez(order, f, am, w, "differentiator")
        elif method == "ls":
            b = _firls_core(order, f, am, w, "differentiator")
        else:
            raise MatlabError("designfilt: a differentiator uses 'equiripple' or 'ls'.")
        filt = _Filt(b=b, a=np.array([1.0]))
    elif resp == "hilbertfir":
        if order is None or "transitionwidth" not in opts:
            raise MatlabError("designfilt: a Hilbert transformer needs 'FilterOrder' and 'TransitionWidth'.")
        method = method or "equiripple"
        tw = num("TransitionWidth") / nyq
        f = np.array([tw, 1 - tw]) if order % 2 == 0 else np.array([tw, 1.0])
        am, w = np.array([1.0, 1.0]), np.array([1.0])
        if method == "equiripple":
            b, _d = _remez(order, f, am, w, "hilbert")
        elif method == "ls":
            b = _firls_core(order, f, am, w, "hilbert")
        else:
            raise MatlabError("designfilt: a Hilbert transformer uses 'equiripple' or 'ls'.")
        filt = _Filt(b=b, a=np.array([1.0]))
    else:   # arbmagfir
        if order is None or not has("Frequencies", "Amplitudes"):
            raise MatlabError("designfilt: 'arbmagfir' needs 'FilterOrder', 'Frequencies' and 'Amplitudes'.")
        f = _rvec(opts["frequencies"], "Frequencies") / nyq
        am = _rvec(opts["amplitudes"], "Amplitudes")
        used["Frequencies"] = _row(f * nyq)
        used["Amplitudes"] = _row(am)
        method = method or "freqsamp"
        if method != "freqsamp":
            raise MatlabError("designfilt: 'arbmagfir' supports 'DesignMethod', 'freqsamp'.")
        b = np.ravel(_fir2(interp, [_f(order), _row(f), _row(am)], 1))
        filt = _Filt(b=b, a=np.array([1.0]))
    props: dict = {"FrequencyResponse": char(resp[:-3]), "ImpulseResponse": char("fir" if fir else "iir"),
                   "SampleRate": _f(fs)}
    for key in _DF_OPTIONS:
        if key in used and key not in ("FilterOrder",):
            v = used[key]
            props[key] = v if isinstance(v, np.ndarray) else _f(v)
    props["FilterOrder"] = _f(filt.order())
    props["DesignMethod"] = char(method or "")
    return DigitalFilter(filt, props)


@builtin("designfilt")
def _designfilt(interp, args, nargout):
    return designfilt_impl(interp, list(args))


def _tw_fraction(s: float) -> float:
    return -0.98 * s + 0.99


def _zero_phase_fir(b: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """FIR filtering with the delay of (order/2) samples removed."""
    half = (b.size - 1) // 2
    padded = np.concatenate([cols, np.zeros((half,) + cols.shape[1:])], axis=0)
    return _sig().lfilter(b, [1.0], padded, axis=0)[half:]


def _pass_filter(name: str, kind: str):
    def fn(interp, args, nargout):
        nargs(args, 2, None, name)
        x = fnum(args[0])
        cols, was_row = _columns(x)
        pos = 2
        fs = None
        if len(args) > 2 and not is_text(args[2]):
            fs = _sc(args[2], "sample rate")
            pos = 3
        opts = _name_values(list(args), pos, ("ImpulseResponse", "Steepness", "StopbandAttenuation"), name)
        nyq = fs / 2 if fs is not None else 1.0
        wpass = np.sort(_rvec(args[1], "passband frequency")) / nyq
        want = 2 if kind in ("bandpass", "stop") else 1
        if wpass.size != want or np.any(wpass <= 0) or np.any(wpass >= 1):
            raise MatlabError(f"{name}: the passband frequency must have {want} element(s) between 0 and "
                              "the Nyquist frequency.")
        steep = _rvec(opts["steepness"], "Steepness") if "steepness" in opts else np.array([0.85])
        if np.any(steep < 0.5) or np.any(steep > 1):
            raise MatlabError(f"{name}: 'Steepness' must lie between 0.5 and 1.")
        if steep.size == 1:
            steep = np.repeat(steep, 2)
        atten = _sc(opts["stopbandattenuation"], "StopbandAttenuation") if "stopbandattenuation" in opts else 60.0
        mode = text(opts["impulseresponse"]).lower() if "impulseresponse" in opts else "auto"
        if mode not in ("auto", "fir", "iir"):
            raise MatlabError(f"{name}: 'ImpulseResponse' must be 'auto', 'fir' or 'iir'.")
        t1, t2 = _tw_fraction(steep[0]), _tw_fraction(steep[1])
        if kind == "low":
            edges = [wpass[0], wpass[0] + t1 * (1 - wpass[0])]
            amps = np.array([1.0, 0.0])
        elif kind == "high":
            edges = [wpass[0] - t1 * wpass[0], wpass[0]]
            amps = np.array([0.0, 1.0])
        elif kind == "bandpass":
            edges = [wpass[0] - t1 * wpass[0], wpass[0], wpass[1], wpass[1] + t2 * (1 - wpass[1])]
            amps = np.array([0.0, 1.0, 0.0])
        else:
            half = (wpass[1] - wpass[0]) / 2
            edges = [wpass[0], wpass[0] + t1 * half, wpass[1] - t2 * half, wpass[1]]
            amps = np.array([1.0, 0.0, 1.0])
        inner = np.array(edges)
        n = cols.shape[0]
        # the Kaiser-window FIR filter that meets 0.1 dB ripple and the wanted attenuation
        delta = min(_ripple_dev(0.1), 10 ** (-atten / 20))
        att = -20 * np.log10(delta)
        width = float(np.min(inner[1::2] - inner[0::2]))
        order = max(int(np.ceil((att - 7.95) / (2.285 * np.pi * width))), 2)
        order += order % 2
        resp_name = {"low": "lowpass", "high": "highpass", "bandpass": "bandpass", "stop": "bandstop"}[kind]
        fsv = fs if fs is not None else 2.0
        use_fir = mode == "fir" or (mode == "auto" and n >= 2 * order)
        if use_fir:
            if n <= order // 2:
                raise MatlabError(f"{name}: the signal is too short for the FIR filter this needs (order {order}).")
            beta = 0.1102 * (att - 8.7) if att > 50 else (
                0.5842 * (att - 21) ** 0.4 + 0.07886 * (att - 21) if att >= 21 else 0.0)
            wn = 0.5 * (inner[0::2] + inner[1::2])
            b = _fir1_core(order, wn, amps[0] != 0, _w_kaiser(order + 1, beta), True)
            y = _zero_phase_fir(b, cols)
            filt = _Filt(b=b, a=np.array([1.0]))
            method = "kaiserwin"
        else:
            ss = _sig()
            if kind == "low":
                wp, ws = inner[0], inner[1]
            elif kind == "high":
                wp, ws = inner[1], inner[0]
            elif kind == "bandpass":
                wp, ws = inner[1:3], inner[[0, 3]]
            else:
                wp, ws = inner[[0, 3]], inner[1:3]
            # filtfilt applies the filter twice, so each pass needs half the ripple and attenuation
            nn, wn = ss.ellipord(wp, ws, 0.05, atten / 2)
            z, p, k = ss.ellip(nn, 0.05, atten / 2, wn, btype={"low": "lowpass", "high": "highpass",
                                                              "bandpass": "bandpass", "stop": "bandstop"}[kind],
                               output="zpk")
            sos, g = _zp2sos(z, p, float(np.real(k)))
            sos[0, :3] *= g
            filt = _Filt(sos=sos)
            pad = 3 * filt.order()
            if n <= pad:
                raise MatlabError(f"{name}: the signal is too short to be filtered (it needs more than {pad} "
                                  "samples).")
            y = ss.sosfiltfilt(sos, cols, axis=0, padtype="odd", padlen=pad)
            method = "ellip"
        y = _restore(y, was_row)
        if nargout > 1:
            props = {"FrequencyResponse": char(resp_name), "ImpulseResponse": char("fir" if use_fir else "iir"),
                     "SampleRate": _f(fsv), "PassbandRipple": _f(0.1), "StopbandAttenuation": _f(atten),
                     "FilterOrder": _f(filt.order()), "DesignMethod": char(method)}
            return y, DigitalFilter(filt, props)
        return y
    builtin(name)(fn)


_pass_filter("lowpass", "low")
_pass_filter("highpass", "high")
_pass_filter("bandpass", "bandpass")
_pass_filter("bandstop", "stop")


# -- correlation ------------------------------------------------------------------------------------

def _xcorr_pair(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Raw cross-correlation of two length-n vectors for lags -(n-1)..(n-1)."""
    nfft = int(2 ** np.ceil(np.log2(2 * n - 1))) if n > 1 else 1
    X, Y = np.fft.fft(x, nfft), np.fft.fft(y, nfft)
    c = np.fft.ifft(X * np.conj(Y))
    c = np.concatenate([c[nfft - (n - 1):], c[:n]]) if n > 1 else c[:1]
    if np.isrealobj(x) and np.isrealobj(y):
        c = np.real(c)
        if np.all(x == np.round(x)) and np.all(y == np.round(y)) and np.max(np.abs(c), initial=0) < 2 ** 52:
            c = np.round(c)        # integer data gives exact integer sums
    return c


def xcorr_impl(args: list, nargout: int, name: str = "xcorr", remove_mean: bool = False):
    nargs(args, 1, 4, name)
    rest = list(args)
    scale = "none"
    if rest and is_text(rest[-1]):
        scale = text(rest.pop()).lower()
        if scale == "normalized":
            scale = "coeff"
        if scale not in ("none", "biased", "unbiased", "coeff"):
            raise MatlabError(f"{name}: unknown scaling option '{scale}'. Use 'none', 'biased', 'unbiased' or "
                              "'normalized'.")
    x = fnum(rest[0])
    if x.ndim > 2:
        raise MatlabError(f"{name}: inputs must be vectors or matrices.")
    y = None
    maxlag = None
    if len(rest) > 1:
        r1 = fnum(rest[1])
        if len(rest) == 2 and r1.size == 1 and x.size != 1:
            maxlag = int(np.real(r1.flat[0]))
        else:
            y = r1
            if _given(rest, 2):
                maxlag = to_int(rest[2], "maxlag")
    if maxlag is not None and maxlag < 0:
        raise MatlabError(f"{name}: maxlag must not be negative.")
    matrix = not _isvec(x) and x.size > 1
    if matrix:
        if y is not None:
            raise MatlabError(f"{name}: with a matrix first input there must be no second signal.")
        cols = x
        if remove_mean:
            cols = cols - cols.mean(axis=0, keepdims=True)
        n, nc = cols.shape
        full = np.zeros((2 * n - 1, nc * nc), dtype=complex if np.iscomplexobj(cols) else float)
        for i in range(nc):
            for j in range(nc):
                full[:, i * nc + j] = _xcorr_pair(cols[:, i], cols[:, j], n)
        lags = np.arange(-(n - 1), n, dtype=float)
        if scale == "biased":
            full = full / n
        elif scale == "unbiased":
            full = full / (n - np.abs(lags))[:, None]
        elif scale == "coeff":
            energy = np.real(np.sum(np.abs(cols) ** 2, axis=0))
            den = np.sqrt(np.outer(energy, energy)).reshape(-1)
            full = full / np.where(den == 0, 1.0, den)[None, :]
        out = full
    else:
        xv = x.flatten(order="F")
        yv = xv if y is None else y.flatten(order="F")
        if remove_mean:
            xv = xv - xv.mean() if xv.size else xv
            yv = yv - yv.mean() if yv.size else yv
        n = max(xv.size, yv.size)
        if scale != "none" and xv.size != yv.size:
            raise MatlabError(f"{name}: the scaling option '{scale}' needs signals of equal length.")
        xp = np.concatenate([xv, np.zeros(n - xv.size)])
        yp = np.concatenate([yv, np.zeros(n - yv.size)])
        out = _xcorr_pair(xp, yp, n) if n else np.zeros(0)
        lags = np.arange(-(n - 1), n, dtype=float)
        if scale == "biased":
            out = out / n
        elif scale == "unbiased":
            out = out / (n - np.abs(lags))
        elif scale == "coeff":
            den = np.sqrt(np.sum(np.abs(xp) ** 2) * np.sum(np.abs(yp) ** 2))
            out = out / den if den else out
    if maxlag is not None:
        want = np.arange(-maxlag, maxlag + 1, dtype=float)
        shape = (want.size,) + out.shape[1:]
        res = np.zeros(shape, dtype=out.dtype)
        src = (want >= lags[0]) & (want <= lags[-1]) if lags.size else np.zeros(want.size, dtype=bool)
        res[src] = out[(lags >= -maxlag) & (lags <= maxlag)]
        out, lags = res, want
    out = simplify(np.asarray(out))
    if not matrix:
        out = _like(out, x if x.size > 1 or y is None else y)
    return (out, _row(lags)) if nargout > 1 else out


@builtin("xcov")
def _xcov(interp, args, nargout):
    return xcorr_impl(list(args), nargout, "xcov", remove_mean=True)


@builtin("xcorr2")
def _xcorr2(interp, args, nargout):
    nargs(args, 1, 2, "xcorr2")
    a = fnum(args[0])
    b = fnum(args[1]) if len(args) > 1 else a
    if a.ndim != 2 or b.ndim != 2:
        raise MatlabError("xcorr2: the inputs must be matrices.")
    return simplify(_sig().correlate2d(a, b, mode="full"))


@builtin("cconv")
def _cconv(interp, args, nargout):
    nargs(args, 2, 3, "cconv")
    a, b = fnum(args[0]), fnum(args[1])
    av, bv = a.flatten(order="F"), b.flatten(order="F")
    full = np.convolve(av, bv)
    n = to_int(args[2], "length") if _given(args, 2) else full.size
    if n < 1:
        raise MatlabError("cconv: the length must be a positive integer.")
    out = np.zeros(n, dtype=full.dtype)
    for start in range(0, full.size, n):
        piece = full[start:start + n]
        out[:piece.size] += piece
    row = a.shape[0] == 1 and b.shape[0] == 1
    return simplify(out.reshape(1, -1) if row else out.reshape(-1, 1))


# -- power spectra ------------------------------------------------------------------------------------

_RANGES = ("onesided", "twosided", "centered")
_SPEC_FLAGS = _RANGES + ("psd", "power", "mean", "maxhold", "minhold", "mimo", "half", "whole")


def _spec_split(args: list, name: str) -> tuple[list, set[str], dict]:
    """Positional arguments, option flags and name-value options of a spectrum function."""
    rest, flags, opts = [], set(), {}
    i = 0
    while i < len(args):
        a = args[i]
        if is_text(a):
            s = text(a).lower()
            if s in ("confidencelevel", "estimator", "tolerance", "taper", "droplasttaper"):
                if i + 1 >= len(args):
                    raise MatlabError(f"{name}: the option '{text(a)}' needs a value.")
                opts[s] = args[i + 1]
                i += 2
                continue
            if s not in _SPEC_FLAGS and s not in ("adapt", "unity", "eigen"):
                raise MatlabError(f"{name}: unknown option '{text(a)}'.")
            flags.add("onesided" if s == "half" else ("twosided" if s == "whole" else s))
        else:
            rest.append(a)
        i += 1
    return rest, flags, opts


def _dft_at(seg: np.ndarray, w: np.ndarray) -> np.ndarray:
    """DFT of each column of seg at the angular frequencies w (rad/sample)."""
    n = np.arange(seg.shape[0])
    return np.exp(-1j * np.outer(w, n)) @ seg


def _range_of(flags: set, real_input: bool) -> str:
    chosen = [r for r in _RANGES if r in flags]
    if len(chosen) > 1:
        raise MatlabError("Only one of 'onesided', 'twosided' and 'centered' may be given.")
    if chosen:
        if chosen[0] == "onesided" and not real_input:
            raise MatlabError("A one-sided spectrum is only defined for real signals.")
        return chosen[0]
    return "onesided" if real_input else "twosided"


def _fold(p: np.ndarray, nfft: int, rng: str, fsv: float) -> tuple[np.ndarray, np.ndarray]:
    """Arrange a two-sided spectrum (rows = bins) as asked; returns the spectrum and its frequencies."""
    if rng == "onesided":
        keep = nfft // 2 + 1
        q = p[:keep].copy()
        if nfft % 2 == 0:
            q[1:-1] *= 2
        else:
            q[1:] *= 2
        return q, np.arange(keep) * fsv / nfft
    if rng == "centered":
        # (-fs/2, fs/2]: the Nyquist bin of an even transform sits at the positive end
        k = np.arange(-(nfft // 2 - 1), nfft // 2 + 1) if nfft % 2 == 0 else np.arange(-(nfft - 1) // 2,
                                                                                      (nfft - 1) // 2 + 1)
        return p[k % nfft], k * fsv / nfft
    return p, np.arange(nfft) * fsv / nfft


def _segments(n: int, L: int, noverlap: int, name: str) -> np.ndarray:
    step = L - noverlap
    if step <= 0:
        raise MatlabError(f"{name}: the overlap must be smaller than the window length.")
    if L > n:
        raise MatlabError(f"{name}: the window ({L} samples) is longer than the signal ({n} samples).")
    return np.arange((n - noverlap) // step) * step


def _welch_window(interp, spec, n: int, name: str) -> np.ndarray:
    if spec is None:
        L = int(np.trunc(n / 4.5))
        if L < 1:
            raise MatlabError(f"{name}: the signal is too short.")
        return _w_hamming(L)
    w = _rvec(spec, "window")
    if w.size == 1:
        L = int(w[0])
        if L < 1:
            raise MatlabError(f"{name}: the window length must be positive.")
        return _w_hamming(L)
    return w


def _conf_bounds(p: np.ndarray, dof: float, level: float) -> np.ndarray:
    from scipy.stats import chi2
    alpha = 1 - level
    lo = dof / chi2.ppf(1 - alpha / 2, dof)
    hi = dof / chi2.ppf(alpha / 2, dof)
    return np.concatenate([p * lo, p * hi], axis=1)


def _welch_core(interp, sigs: list[np.ndarray], rest: list, flags: set, name: str, periodogram: bool):
    """Averaged cross-spectra of the columns of sigs[0] (and sigs[1]).

    Returns (Sxx, Syy, Sxy, f, fs_given, K, real_input): two-sided or folded spectra as requested.
    """
    n = sigs[0].shape[0]
    real_input = all(np.isrealobj(s) for s in sigs)
    pos = 0
    if periodogram:
        wspec = rest[pos] if _given(rest, pos) else None
        win = np.ones(n) if wspec is None else _rvec(wspec, "window")
        if win.size != n:
            raise MatlabError(f"{name}: the window must be as long as the signal.")
        noverlap = 0
        pos += 1
    else:
        win = _welch_window(interp, rest[pos] if _given(rest, pos) else None, n, name)
        pos += 1
        noverlap = to_int(rest[pos], "noverlap") if _given(rest, pos) else int(np.trunc(win.size / 2))
        pos += 1
    L = win.size
    fvec = None
    if _given(rest, pos) and fnum(rest[pos]).size > 1:
        fvec = np.real(fnum(rest[pos])).astype(float)
        nfft = None
    elif _given(rest, pos):
        nfft = to_int(rest[pos], "nfft")
        if nfft < 1:
            raise MatlabError(f"{name}: nfft must be positive.")
    else:
        nfft = max(256, int(2 ** np.ceil(np.log2(L))))
    pos += 1
    fs = _sc(rest[pos], "sample rate") if _given(rest, pos) else None
    if len(rest) > pos + 1:
        raise MatlabError(f"{name}: too many input arguments.")
    fsv = fs if fs is not None else 2 * np.pi
    starts = _segments(n, L, noverlap, name)
    K = starts.size
    power = "power" in flags
    U = np.sum(win) ** 2 if power else fsv * np.sum(win ** 2)
    trace = [t for t in ("mean", "maxhold", "minhold") if t in flags]
    trace = trace[0] if trace else "mean"
    acc = [None, None, None]
    for s0 in starts:
        segs = [s[s0:s0 + L] * win[:, None] for s in sigs]
        if fvec is not None:
            wq = fvec.flatten(order="F") * (2 * np.pi / fsv)
            F = [_dft_at(sg, wq) for sg in segs]
        else:
            F = []
            for sg in segs:
                if nfft < L:      # wrap the windowed segment onto nfft samples
                    pad = (-L) % nfft
                    sg = np.concatenate([sg, np.zeros((pad, sg.shape[1]))]).reshape(-1, nfft, sg.shape[1]).sum(0)
                F.append(np.fft.fft(sg, n=nfft, axis=0))
        parts = [np.abs(F[0]) ** 2 / U]
        if len(F) > 1:
            parts.append(np.abs(F[1]) ** 2 / U)
            parts.append(np.conj(F[0]) * F[1] / U)     # cpsd(x, y) is conj(X) .* Y
        for i, part in enumerate(parts):
            if acc[i] is None:
                acc[i] = part.copy()
            elif trace == "mean" or i == 2:
                acc[i] = acc[i] + part
            elif trace == "maxhold":
                acc[i] = np.maximum(acc[i], part)
            else:
                acc[i] = np.minimum(acc[i], part)
    for i in range(3):
        if acc[i] is not None and (trace == "mean" or i == 2):
            acc[i] = acc[i] / K
    if fvec is not None:
        f = fvec
        out = acc
        rng = "given"
    else:
        rng = _range_of(flags, real_input)
        out = []
        f = None
        for a in acc:
            if a is None:
                out.append(None)
                continue
            q, f = _fold(a, nfft, rng, fsv)
            out.append(q)
    return out, f, fs, K, rng


def _spec_plot(interp, f, p, fs, ylabel: str) -> None:
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(np.abs(p))
    x = np.asarray(f).reshape(-1) if fs is not None else np.asarray(f).reshape(-1) / np.pi
    for j in range(db.shape[1]):
        _plot(interp, "plot", [_row(x), _row(db[:, j])])
        if j == 0 and db.shape[1] > 1:
            _plot(interp, "hold", [char("on")])
    if db.shape[1] > 1:
        _plot(interp, "hold", [char("off")])
    _plot(interp, "xlabel", [char(_freq_label(fs))])
    _plot(interp, "ylabel", [char(ylabel)])
    _plot(interp, "grid", [char("on")])


def _shape_f(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return f if f.ndim == 2 and f.shape[0] == 1 and f.shape[1] > 1 else _col(f.reshape(-1, order="F"))


def psd_impl(interp, args: list, nargout: int, name: str):
    rest, flags, opts = _spec_split(list(args), name)
    if not rest:
        raise MatlabError(f"{name}: a signal is required.")
    x = fnum(rest[0])
    cols, _was_row = _columns(x)
    if cols.shape[0] == 0:
        raise MatlabError(f"{name}: the signal must not be empty.")
    (pxx, _a, _b), f, fs, K, rng = _welch_core(interp, [cols], rest[1:], flags, name, name == "periodogram")
    if nargout == 0:
        label = "Power (dB)" if "power" in flags else ("Power/frequency (dB/Hz)" if fs is not None
                                                        else "Power/frequency (dB/(rad/sample))")
        _spec_plot(interp, f, pxx, fs, label)
        return None
    out = [pxx, _shape_f(f)]
    if nargout > 2:
        level = _sc(opts["confidencelevel"], "ConfidenceLevel") if "confidencelevel" in opts else 0.95
        out.append(_conf_bounds(pxx, 2.0 * K, level))
    return tuple(out[:nargout]) if nargout > 1 else out[0]


def _cross_inputs(args: list, name: str):
    rest, flags, opts = _spec_split(list(args), name)
    if len(rest) < 2:
        raise MatlabError(f"{name}: two signals are required.")
    if "mimo" in flags:
        raise MatlabError(f"{name}: the 'mimo' option is not supported.")
    x, y = fnum(rest[0]), fnum(rest[1])
    xc, _ = _columns(x)
    yc, _ = _columns(y)
    if xc.shape[0] != yc.shape[0]:
        raise MatlabError(f"{name}: the two signals must have the same length.")
    if xc.shape[1] != yc.shape[1]:
        if xc.shape[1] == 1:
            xc = np.repeat(xc, yc.shape[1], axis=1)
        elif yc.shape[1] == 1:
            yc = np.repeat(yc, xc.shape[1], axis=1)
        else:
            raise MatlabError(f"{name}: the two signals must have the same number of columns.")
    return xc, yc, rest[2:], flags, opts


@builtin("cpsd")
def _cpsd(interp, args, nargout):
    xc, yc, rest, flags, opts = _cross_inputs(args, "cpsd")
    (_pxx, _pyy, pxy), f, fs, K, rng = _welch_core(interp, [xc, yc], rest, flags, "cpsd", False)
    if nargout == 0:
        _spec_plot(interp, f, pxy, fs, "Cross power spectral density (dB)")
        return None
    pxy = simplify(pxy)
    return (pxy, _shape_f(f)) if nargout > 1 else pxy


@builtin("mscohere")
def _mscohere(interp, args, nargout):
    xc, yc, rest, flags, opts = _cross_inputs(args, "mscohere")
    (pxx, pyy, pxy), f, fs, K, rng = _welch_core(interp, [xc, yc], rest, flags, "mscohere", False)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.abs(pxy) ** 2 / (pxx * pyy)
    if nargout == 0:
        x = np.asarray(f).reshape(-1) if fs is not None else np.asarray(f).reshape(-1) / np.pi
        _plot(interp, "plot", [_row(x), _row(c[:, 0])])
        _plot(interp, "xlabel", [char(_freq_label(fs))])
        _plot(interp, "ylabel", [char("Magnitude-squared coherence")])
        return None
    return (c, _shape_f(f)) if nargout > 1 else c


@builtin("tfestimate")
def _tfestimate(interp, args, nargout):
    xc, yc, rest, flags, opts = _cross_inputs(args, "tfestimate")
    est = text(opts["estimator"]).lower() if "estimator" in opts else "h1"
    if est not in ("h1", "h2"):
        raise MatlabError("tfestimate: the estimator must be 'H1' or 'H2'.")
    (pxx, pyy, pxy), f, fs, K, rng = _welch_core(interp, [xc, yc], rest, flags, "tfestimate", False)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = pxy / pxx if est == "h1" else pyy / np.conj(pxy)
    if nargout == 0:
        _spec_plot(interp, f, np.abs(t) ** 2, fs, "Magnitude (dB)")
        return None
    return (t, _shape_f(f)) if nargout > 1 else t


def _ar_psd(name: str, estimator):
    def fn(interp, args, nargout):
        rest, flags, opts = _spec_split(list(args), name)
        if len(rest) < 2:
            raise MatlabError(f"{name}: a signal and a model order are required.")
        x = fnum(rest[0])
        cols, _ = _columns(x)
        p = to_int(rest[1], "model order")
        fvec = None
        nfft = 256
        if _given(rest, 2):
            if fnum(rest[2]).size > 1:
                fvec = np.real(fnum(rest[2])).astype(float)
            else:
                nfft = to_int(rest[2], "nfft")
        fs = _sc(rest[3], "sample rate") if _given(rest, 3) else None
        fsv = fs if fs is not None else 2 * np.pi
        real_input = np.isrealobj(cols)
        out = []
        f = None
        for j in range(cols.shape[1]):
            a, e = estimator(cols[:, j], p)
            if fvec is not None:
                wq = fvec.flatten(order="F") * (2 * np.pi / fsv)
                den = np.abs(np.polyval(a[::-1], np.exp(-1j * wq))) ** 2
                out.append(e / den / fsv)
                f = fvec
            else:
                den = np.abs(np.fft.fft(a, nfft)) ** 2
                q, f = _fold((e / den / fsv)[:, None], nfft, _range_of(flags, real_input), fsv)
                out.append(q[:, 0])
        pxx = np.column_stack(out)
        if nargout == 0:
            _spec_plot(interp, f, pxx, fs, "Power/frequency (dB)")
            return None
        return (pxx, _shape_f(f)) if nargout > 1 else pxx
    builtin(name)(fn)


_ar_psd("pyulear", lambda x, p: _levinson(_autocorr_biased(x, p), p)[:2])
_ar_psd("pburg", lambda x, p: _arburg(x, p)[:2])
_ar_psd("pcov", lambda x, p: _arcov(x, p, False))
_ar_psd("pmcov", lambda x, p: _arcov(x, p, True))


@builtin("pmtm")
def _pmtm(interp, args, nargout):
    rest, flags, opts = _spec_split(list(args), "pmtm")
    if not rest:
        raise MatlabError("pmtm: a signal is required.")
    x = fnum(rest[0])
    cols, _ = _columns(x)
    n = cols.shape[0]
    nw = _sc(rest[1], "time-halfbandwidth product") if _given(rest, 1) else 4.0
    fvec = None
    nfft = max(256, int(2 ** np.ceil(np.log2(n))))
    if _given(rest, 2):
        if fnum(rest[2]).size > 1:
            fvec = np.real(fnum(rest[2])).astype(float)
        else:
            nfft = to_int(rest[2], "nfft")
    fs = _sc(rest[3], "sample rate") if _given(rest, 3) else None
    fsv = fs if fs is not None else 2 * np.pi
    method = next((m for m in ("adapt", "unity", "eigen") if m in flags), "adapt")
    ntap = max(int(round(2 * nw)) - 1, 1)
    tapers, ratios = _sig().windows.dpss(n, nw, Kmax=ntap, return_ratios=True)
    tapers = np.atleast_2d(tapers)
    ratios = np.atleast_1d(ratios)
    real_input = np.isrealobj(cols)
    out = []
    f = None
    for j in range(cols.shape[1]):
        xj = cols[:, j]
        segs = tapers.T * xj[:, None]
        if fvec is not None:
            Sk = np.abs(_dft_at(segs, fvec.flatten(order="F") * (2 * np.pi / fsv))) ** 2
        else:
            if nfft < n:
                pad = (-n) % nfft
                segs = np.concatenate([segs, np.zeros((pad, ntap))]).reshape(-1, nfft, ntap).sum(0)
            Sk = np.abs(np.fft.fft(segs, n=nfft, axis=0)) ** 2
        if method == "unity" or ntap == 1:
            S = Sk.mean(axis=1)
        elif method == "eigen":
            S = (Sk * ratios[None, :]).sum(axis=1) / ntap
        else:
            var = float(np.real(np.vdot(xj, xj))) / n
            S = (Sk[:, 0] + Sk[:, min(1, ntap - 1)]) / 2
            tol = 0.0005 * var / max(Sk.shape[0], 1)
            for _ in range(200):
                bk = S[:, None] / (S[:, None] * ratios[None, :] + var * (1 - ratios[None, :]))
                wk = bk ** 2 * ratios[None, :]
                S1 = (wk * Sk).sum(axis=1) / wk.sum(axis=1)
                done = np.max(np.abs(S1 - S)) <= tol
                S = S1
                if done:
                    break
        S = S / fsv
        if fvec is not None:
            out.append(S)
            f = fvec
        else:
            q, f = _fold(S[:, None], nfft, _range_of(flags, real_input), fsv)
            out.append(q[:, 0])
    pxx = np.column_stack(out)
    if nargout == 0:
        _spec_plot(interp, f, pxx, fs, "Power/frequency (dB)")
        return None
    return (pxx, _shape_f(f)) if nargout > 1 else pxx


# -- short-time Fourier transform -----------------------------------------------------------------------

_STFT_OPTS = ("Window", "OverlapLength", "FFTLength", "FrequencyRange", "OutputTimeDimension",
              "InputTimeDimension", "Method", "ConjugateSymmetric")


def _stft_setup(interp, args: list, start: int, name: str):
    fs = None
    if len(args) > start and not is_text(args[start]):
        if _given(args, start):
            fs = _sc(args[start], "sample rate")
            if fs <= 0:
                raise MatlabError(f"{name}: the sample rate must be positive.")
        start += 1
    opts = _name_values(args, start, _STFT_OPTS, name)
    win = _rvec(opts["window"], "Window") if "window" in opts else _w_hann(128, periodic=True)
    L = win.size
    noverlap = to_int(opts["overlaplength"], "OverlapLength") if "overlaplength" in opts else int(np.floor(0.75 * L))
    nfft = to_int(opts["fftlength"], "FFTLength") if "fftlength" in opts else max(128, L)
    if nfft < L:
        raise MatlabError(f"{name}: 'FFTLength' must be at least the window length.")
    if not 0 <= noverlap < L:
        raise MatlabError(f"{name}: 'OverlapLength' must be smaller than the window length.")
    rng = text(opts["frequencyrange"]).lower() if "frequencyrange" in opts else "centered"
    if rng not in _RANGES:
        raise MatlabError(f"{name}: 'FrequencyRange' must be 'centered', 'twosided' or 'onesided'.")
    return fs, win, noverlap, nfft, rng, opts


def _stft_bins(nfft: int, rng: str) -> np.ndarray:
    if rng == "onesided":
        return np.arange(nfft // 2 + 1)
    if rng == "centered":
        return (np.arange(-(nfft // 2 - 1), nfft // 2 + 1) if nfft % 2 == 0
                else np.arange(-(nfft - 1) // 2, (nfft - 1) // 2 + 1))
    return np.arange(nfft)


@builtin("stft")
def _stft(interp, args, nargout):
    nargs(args, 1, None, "stft")
    x = fnum(args[0])
    cols, _ = _columns(x)
    fs, win, noverlap, nfft, rng, opts = _stft_setup(interp, list(args), 1, "stft")
    if rng == "onesided" and np.iscomplexobj(cols):
        raise MatlabError("stft: a one-sided transform needs a real signal.")
    n, nch = cols.shape
    L = win.size
    starts = _segments(n, L, noverlap, "stft")
    if starts.size == 0:
        raise MatlabError("stft: the signal is shorter than the window.")
    idx = starts[None, :] + np.arange(L)[:, None]
    frames = cols[idx, :] * win[:, None, None]             # L x frames x channels
    S = np.fft.fft(frames, n=nfft, axis=0)
    bins = _stft_bins(nfft, rng)
    S = S[bins % nfft]
    fsv = fs if fs is not None else 2 * np.pi
    f = bins * fsv / nfft
    t = (L / 2 + starts) / (fs if fs is not None else 1.0)
    if nch == 1:
        S = S[:, :, 0]
    if "outputtimedimension" in opts and text(opts["outputtimedimension"]).lower() == "downrows":
        S = np.swapaxes(S, 0, 1)
    if nargout == 0:
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(np.abs(S if S.ndim == 2 else S[:, :, 0]))
        _plot(interp, "imagesc", [_row(t), _row(f), db])
        _plot(interp, "axis", [char("xy")])
        _plot(interp, "xlabel", [char("Time (s)" if fs is not None else "Samples")])
        _plot(interp, "ylabel", [char("Frequency (Hz)" if fs is not None else "Normalized frequency (rad/sample)")])
        _plot(interp, "colorbar", [])
        return None
    out = (S.astype(complex), _col(f), _col(t))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("istft")
def _istft(interp, args, nargout):
    nargs(args, 1, None, "istft")
    S = fnum(args[0])
    fs, win, noverlap, nfft_opt, rng, opts = _stft_setup(interp, list(args), 1, "istft")
    if "inputtimedimension" in opts and text(opts["inputtimedimension"]).lower() == "downrows":
        S = np.swapaxes(S, 0, 1)
    if S.ndim == 2:
        S = S[:, :, None]
    L = win.size
    nbins, nframes, nch = S.shape
    if "fftlength" in opts:
        nfft = nfft_opt
    elif rng == "onesided":
        nfft = max(2 * (nbins - 1), L)
        if nfft_opt in (2 * (nbins - 1), 2 * nbins - 1):
            nfft = nfft_opt
    else:
        nfft = nbins
    bins = _stft_bins(nfft, rng)
    if bins.size != nbins:
        raise MatlabError("istft: the number of rows does not match 'FFTLength' and 'FrequencyRange'.")
    full = np.zeros((nfft, nframes, nch), dtype=complex)
    full[bins % nfft] = S
    if rng == "onesided":
        k = np.arange(1, (nfft + 1) // 2)
        full[nfft - k] = np.conj(full[k])
    frames = np.fft.ifft(full, axis=0)[:L]
    method = text(opts["method"]).lower() if "method" in opts else "wola"
    if method not in ("wola", "ola"):
        raise MatlabError("istft: 'Method' must be 'wola' or 'ola'.")
    hop = L - noverlap
    n = L + (nframes - 1) * hop
    out = np.zeros((n, nch), dtype=complex)
    norm = np.zeros(n)
    for m in range(nframes):
        sl = slice(m * hop, m * hop + L)
        if method == "wola":
            out[sl] += frames[:, m, :] * win[:, None]
            norm[sl] += win ** 2
        else:
            out[sl] += frames[:, m, :]
            norm[sl] += win
    good = np.abs(norm) > 1e-10 * max(float(np.max(np.abs(norm))), 1e-300)
    out[good] = out[good] / norm[good, None]
    sym = rng == "onesided"
    if "conjugatesymmetric" in opts:
        v = opts["conjugatesymmetric"]
        sym = sym or (text(v).lower() in ("true", "on") if is_text(v) else bool(np.ravel(v)[0]))
    res = np.real(out) if sym else _tidy(out)
    t = np.arange(n) / (fs if fs is not None else 1.0)
    return (res, _col(t)) if nargout > 1 else res


def _czt(x: np.ndarray, m: int, w: complex, a: complex) -> np.ndarray:
    """Chirp z-transform of the columns of x: X(k) = sum_n x(n) (a w^-k)^-n, k = 0..m-1."""
    n = x.shape[0]
    k = np.arange(max(m, n))
    nfft = int(2 ** np.ceil(np.log2(m + n - 1)))
    # w^(k^2/2), with the exponent reduced first when w lies on the unit circle
    if abs(abs(w) - 1) < 1e-14:
        ang = np.angle(w)
        chirp = np.exp(1j * ang * ((k.astype(float) ** 2) / 2))
    else:
        chirp = np.asarray(w, dtype=complex) ** ((k.astype(float) ** 2) / 2)
    an = np.asarray(a, dtype=complex) ** (-np.arange(n, dtype=float))
    y = x * (an * chirp[:n])[:, None]
    kernel = np.zeros(nfft, dtype=complex)
    kernel[:m] = 1 / chirp[:m]
    kernel[nfft - n + 1:] = 1 / chirp[n - 1:0:-1]
    g = np.fft.ifft(np.fft.fft(y, nfft, axis=0) * np.fft.fft(kernel)[:, None], axis=0)
    return g[:m] * chirp[:m, None]


@builtin("czt")
def _czt_b(interp, args, nargout):
    nargs(args, 1, 4, "czt")
    x = fnum(args[0])
    cols, was_row = _columns(x)
    n = cols.shape[0]
    m = to_int(args[1], "m") if _given(args, 1) else n
    w = complex(fnum(args[2]).flat[0]) if _given(args, 2) else np.exp(-2j * np.pi / m)
    a = complex(fnum(args[3]).flat[0]) if _given(args, 3) else 1.0
    y = _czt(cols.astype(complex), m, w, a)
    return _restore(y, was_row)


@builtin("goertzel")
def _goertzel(interp, args, nargout):
    nargs(args, 1, 3, "goertzel")
    x = fnum(args[0])
    axis = _first_dim(x, args, 2)
    n = x.shape[axis]
    idx = np.real(fnum(args[1])).flatten(order="F") if _given(args, 1) else np.arange(1, n + 1, dtype=float)
    xm = np.moveaxis(x, axis, 0)
    kern = np.exp(-2j * np.pi * np.outer(idx - 1, np.arange(n)) / n)
    y = np.tensordot(kern, xm, axes=(1, 0))
    return np.moveaxis(y, 0, axis)


# -- spectral measurements --------------------------------------------------------------------------

def _onesided_psd(cols: np.ndarray, win: np.ndarray, fsv: float, power: bool = False):
    """One-sided periodogram of each column (nfft = signal length)."""
    n = cols.shape[0]
    X = np.fft.fft(cols * win[:, None], axis=0)
    U = np.sum(win) ** 2 if power else fsv * np.sum(win ** 2)
    return _fold(np.abs(X) ** 2 / U, n, "onesided", fsv)


def _psd_input(args: list, name: str, window=None):
    """(pxx columns, f, remaining args) from a signal (x, fs, ...) or from a spectrum (pxx, f, ...)."""
    if not args:
        raise MatlabError(f"{name}: a signal is required.")
    x = fnum(args[0])
    cols, _ = _columns(x)
    rest = [a for a in args[1:]]
    flags = [text(a).lower() for a in rest if is_text(a)]
    rest = [a for a in rest if not is_text(a)]
    if rest and fnum(rest[0]).size == cols.shape[0] and cols.shape[0] > 1 and fnum(rest[0]).size > 1:
        f = np.real(fnum(rest[0])).flatten(order="F").astype(float)
        return np.real(cols).astype(float), f, rest[1:], flags, True
    if "psd" in flags or "power" in flags:
        raise MatlabError(f"{name}: a spectrum needs its frequency vector as the second input.")
    if np.iscomplexobj(cols):
        raise MatlabError(f"{name}: complex signals are not supported.")
    fs = _sc(rest[0], "sample rate") if _given(rest, 0) else None
    fsv = fs if fs is not None else 2 * np.pi
    n = cols.shape[0]
    win = np.ones(n) if window is None else window(n)
    pxx, f = _onesided_psd(cols.astype(float), win, fsv)
    return pxx, f, rest[1:], flags, False


def _bin_edges(f: np.ndarray) -> np.ndarray:
    """Edges of the frequency bins around each spectrum sample (clipped to the covered range)."""
    if f.size == 1:
        return np.array([f[0], f[0]])
    mid = 0.5 * (f[1:] + f[:-1])
    return np.concatenate([[f[0]], mid, [f[-1]]])


def _range_arg(rest: list, f: np.ndarray, name: str) -> tuple[float, float]:
    if _given(rest, 0):
        r = _rvec(rest[0], "frequency range")
        if r.size != 2 or r[0] >= r[1]:
            raise MatlabError(f"{name}: the frequency range must be [low high] with low < high.")
        if r[0] < f[0] - 1e-12 or r[1] > f[-1] + 1e-12:
            raise MatlabError(f"{name}: the frequency range must lie inside the range of the spectrum.")
        return float(r[0]), float(r[1])
    return float(f[0]), float(f[-1])


def _cum_power(p: np.ndarray, f: np.ndarray, lo: float, hi: float):
    """Bin edges within [lo, hi] and the cumulative power at those edges."""
    edges = _bin_edges(f)
    a = np.clip(edges[:-1], lo, hi)
    b = np.clip(edges[1:], lo, hi)
    width = edges[1:] - edges[:-1]
    frac = np.where(width > 0, (b - a) / np.where(width > 0, width, 1.0), 0.0)
    pw = p * width * frac
    keep = (b > a) | (pw > 0)
    e = np.concatenate([[lo], b[keep]])
    c = np.concatenate([[0.0], np.cumsum(pw[keep])])
    return e, c


def _row_or_scalar(vals: list) -> np.ndarray:
    return np.array(vals, dtype=float).reshape(1, -1)


@builtin("bandpower")
def _bandpower(interp, args, nargout):
    nargs(args, 1, 4, "bandpower")
    if len(args) == 1:
        x = fnum(args[0])
        cols, _ = _columns(x)
        return _row_or_scalar(list(np.mean(np.abs(cols) ** 2, axis=0)))
    pxx, f, rest, flags, _is_psd = _psd_input(list(args), "bandpower", _w_hamming)
    lo, hi = _range_arg(rest, f, "bandpower")
    i1 = int(np.flatnonzero(f <= lo + 1e-12 * max(1.0, abs(lo)))[-1])
    i2 = int(np.flatnonzero(f >= hi - 1e-12 * max(1.0, abs(hi)))[0])
    width = np.concatenate([np.diff(f), [0.0]])      # rectangle rule; the last point closes the range
    return _row_or_scalar([float(width[i1:i2 + 1] @ pxx[i1:i2 + 1, j]) for j in range(pxx.shape[1])])


def _crossing(e: np.ndarray, c: np.ndarray, level: float) -> float:
    k = int(np.searchsorted(c, level, side="left"))
    if k <= 0:
        return float(e[0])
    if k >= c.size:
        return float(e[-1])
    if c[k] == c[k - 1]:
        return float(e[k])
    return float(e[k - 1] + (level - c[k - 1]) * (e[k] - e[k - 1]) / (c[k] - c[k - 1]))


@builtin("obw")
def _obw(interp, args, nargout):
    pxx, f, rest, flags, _is_psd = _psd_input(list(args), "obw")
    lo, hi = _range_arg(rest, f, "obw")
    pct = _sc(rest[1], "percentage") if _given(rest, 1) else 99.0
    if not 0 < pct < 100:
        raise MatlabError("obw: the percentage must lie between 0 and 100.")
    bw, flo, fhi, pw = [], [], [], []
    for j in range(pxx.shape[1]):
        e, c = _cum_power(pxx[:, j], f, lo, hi)
        total = c[-1]
        a = _crossing(e, c, total * (100 - pct) / 200)
        b = _crossing(e, c, total * (100 + pct) / 200)
        bw.append(b - a)
        flo.append(a)
        fhi.append(b)
        pw.append(total * pct / 100)
    out = tuple(_row_or_scalar(v) for v in (bw, flo, fhi, pw))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("meanfreq")
def _meanfreq(interp, args, nargout):
    pxx, f, rest, flags, _is_psd = _psd_input(list(args), "meanfreq")
    lo, hi = _range_arg(rest, f, "meanfreq")
    freq, pw = [], []
    edges = _bin_edges(f)
    a, b = np.clip(edges[:-1], lo, hi), np.clip(edges[1:], lo, hi)
    for j in range(pxx.shape[1]):
        w = pxx[:, j] * (b - a)
        centre = 0.5 * (a + b)
        total = w.sum()
        freq.append(float((w * centre).sum() / total) if total else np.nan)
        pw.append(float(total))
    out = (_row_or_scalar(freq), _row_or_scalar(pw))
    return out if nargout > 1 else out[0]


@builtin("medfreq")
def _medfreq(interp, args, nargout):
    pxx, f, rest, flags, _is_psd = _psd_input(list(args), "medfreq")
    lo, hi = _range_arg(rest, f, "medfreq")
    freq, pw = [], []
    for j in range(pxx.shape[1]):
        e, c = _cum_power(pxx[:, j], f, lo, hi)
        freq.append(_crossing(e, c, c[-1] / 2))
        pw.append(float(c[-1]))
    out = (_row_or_scalar(freq), _row_or_scalar(pw))
    return out if nargout > 1 else out[0]


@builtin("powerbw")
def _powerbw(interp, args, nargout):
    pxx, f, rest, flags, _is_psd = _psd_input(list(args), "powerbw")
    lo, hi = _range_arg(rest, f, "powerbw")
    drop = _sc(rest[1], "power drop in dB") if _given(rest, 1) else 10 * np.log10(2)
    bw, flo, fhi, pw = [], [], [], []
    inside = np.flatnonzero((f >= lo - 1e-12) & (f <= hi + 1e-12))
    for j in range(pxx.shape[1]):
        p = pxx[:, j]
        peak = inside[int(np.argmax(p[inside]))]
        level = p[peak] / 10 ** (drop / 10)
        left = peak
        while left > inside[0] and p[left] >= level:
            left -= 1
        right = peak
        while right < inside[-1] and p[right] >= level:
            right += 1

        def interp_at(i0, i1):
            # between a point above the level (i0) and a point below it (i1), linear in dB
            if p[i1] >= level or p[i0] <= 0 or p[i1] <= 0:
                return float(f[i1])
            d0, d1, dl = np.log10(p[i0]), np.log10(p[i1]), np.log10(level)
            return float(f[i0] + (dl - d0) * (f[i1] - f[i0]) / (d1 - d0))

        a = interp_at(left + 1, left) if p[left] < level else float(f[left])
        b = interp_at(right - 1, right) if p[right] < level else float(f[right])
        e, c = _cum_power(p, f, a, b)
        bw.append(b - a)
        flo.append(a)
        fhi.append(b)
        pw.append(float(c[-1]))
    out = tuple(_row_or_scalar(v) for v in (bw, flo, fhi, pw))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("enbw")
def _enbw(interp, args, nargout):
    nargs(args, 1, 2, "enbw")
    w = _rvec(args[0], "window")
    if w.size == 0 or np.sum(w) == 0:
        raise MatlabError("enbw: the window must not be empty or sum to zero.")
    bw = w.size * np.sum(w ** 2) / np.sum(w) ** 2
    if len(args) > 1:
        bw = bw * _sc(args[1], "sample rate") / w.size
    return float(bw)


def _lobe(p: np.ndarray, k: int) -> tuple[int, int]:
    """Indices of the spectral lobe around bin k: out to the nearest local minima."""
    lo = k
    while lo > 0 and p[lo - 1] <= p[lo]:
        lo -= 1
    hi = k
    while hi < p.size - 1 and p[hi + 1] <= p[hi]:
        hi += 1
    return lo, hi


def _tone_analysis(x: np.ndarray, fs: float, nharm: int, aliased: bool = True):
    """Fundamental, harmonics and noise of a real sinusoidal signal from a Kaiser-windowed periodogram."""
    n = x.size
    if n < 8:
        raise MatlabError("The signal is too short for a spectral analysis.")
    win = _w_kaiser(n, 38.0)
    pxx, f = _onesided_psd(x.reshape(-1, 1), win, fs)
    p = pxx[:, 0]
    df = f[1] - f[0]
    used = np.zeros(p.size, dtype=bool)
    # the DC lobe
    dc_hi = 0
    while dc_hi < p.size - 1 and p[dc_hi + 1] <= p[dc_hi]:
        dc_hi += 1
    used[:dc_hi + 1] = True
    dc_power = float(p[:dc_hi + 1].sum() * df)
    rest = np.where(used, 0.0, p)
    k1 = int(np.argmax(rest))
    lo, hi = _lobe(p, k1)
    lo = max(lo, dc_hi + 1)
    p1 = float(p[lo:hi + 1].sum() * df)
    f1 = float((p[lo:hi + 1] * f[lo:hi + 1]).sum() / p[lo:hi + 1].sum())
    used[lo:hi + 1] = True
    half = max(hi - k1, k1 - lo, 1)
    harm_pow, harm_freq = [p1], [f1]
    for h in range(2, nharm + 1):
        fh = h * f1
        if aliased:
            fh = fh % fs
            if fh > fs / 2:
                fh = fs - fh
        elif fh > fs / 2:
            break
        kc = int(round(fh / df))
        a, b = max(kc - half, 0), min(kc + half, p.size - 1)
        cand = np.where(used[a:b + 1], -1.0, p[a:b + 1])
        if cand.size == 0 or np.max(cand) < 0:
            harm_pow.append(0.0)
            harm_freq.append(fh)
            continue
        kh = a + int(np.argmax(cand))
        l2, h2 = _lobe(p, kh)
        l2 = max(l2, kh - half)
        h2 = min(h2, kh + half)
        sel = np.arange(l2, h2 + 1)
        sel = sel[~used[sel]]
        harm_pow.append(float(p[sel].sum() * df))
        harm_freq.append(float((p[sel] * f[sel]).sum() / max(p[sel].sum(), 1e-300)))
        used[sel] = True
    noise_bins = p[~used]
    floor = float(np.median(noise_bins)) if noise_bins.size else 0.0
    noise = float(noise_bins.sum() * df + floor * df * (used.sum() - (dc_hi + 1)))
    return {"p": p, "f": f, "df": df, "used": used, "dc_hi": dc_hi, "dc": dc_power, "harm_pow": harm_pow,
            "harm_freq": harm_freq, "noise": noise, "total": float(p.sum() * df), "fund": (lo, hi)}


def _tone_inputs(args: list, name: str, default_harm: int = 6):
    x = fnum(args[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError(f"{name}: the signal must be a real vector.")
    rest, flags = _split_flags(list(args[1:]))
    fs = _sc(rest[0], "sample rate") if _given(rest, 0) else 1.0
    n = to_int(rest[1], "number of harmonics") if _given(rest, 1) else default_harm
    return np.real(x).flatten(order="F").astype(float), fs, n, "omitaliases" not in flags


def _db(ratio: float) -> float:
    with np.errstate(divide="ignore"):
        return float(10 * np.log10(ratio)) if ratio > 0 else -np.inf


@builtin("snr")
def _snr(interp, args, nargout):
    nargs(args, 1, 4, "snr")
    if len(args) == 2 and fnum(args[1]).size == fnum(args[0]).size and fnum(args[0]).size > 1:
        x, y = fnum(args[0]), fnum(args[1])
        return _db(float(np.sum(np.abs(x) ** 2)) / float(np.sum(np.abs(y) ** 2)))
    x, fs, nh, alias = _tone_inputs(list(args), "snr")
    t = _tone_analysis(x, fs, nh, alias)
    r = _db(t["harm_pow"][0] / t["noise"]) if t["noise"] > 0 else np.inf
    if nargout > 1:
        return r, _db(t["noise"])
    return r


@builtin("thd")
def _thd(interp, args, nargout):
    nargs(args, 1, 4, "thd")
    x, fs, nh, alias = _tone_inputs(list(args), "thd")
    t = _tone_analysis(x, fs, nh, alias)
    hp = np.array(t["harm_pow"])
    r = _db(float(hp[1:].sum()) / hp[0])
    if nargout > 1:
        with np.errstate(divide="ignore"):
            return r, _col(10 * np.log10(hp)), _col(np.array(t["harm_freq"]))[:len(hp)]
    return r


@builtin("sinad")
def _sinad(interp, args, nargout):
    nargs(args, 1, 2, "sinad")
    x, fs, _nh, alias = _tone_inputs(list(args), "sinad", 1)
    t = _tone_analysis(x, fs, 1, alias)
    fund = t["harm_pow"][0]
    rest = t["total"] - t["dc"] - fund
    r = _db(fund / rest) if rest > 0 else np.inf
    if nargout > 1:
        return r, _db(rest)
    return r


@builtin("sfdr")
def _sfdr(interp, args, nargout):
    nargs(args, 1, 3, "sfdr")
    x, fs, _nh, alias = _tone_inputs(list(args), "sfdr", 1)
    t = _tone_analysis(x, fs, 1, alias)
    p, f, df = t["p"], t["f"], t["df"]
    rest = np.where(t["used"], 0.0, p)
    ks = int(np.argmax(rest))
    lo, hi = _lobe(rest, ks)
    spur = float(rest[lo:hi + 1].sum() * df)
    r = _db(t["harm_pow"][0] / spur) if spur > 0 else np.inf
    out = (r, _db(spur), float(f[ks]))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("pspectrum")
def _pspectrum(interp, args, nargout):
    nargs(args, 1, None, "pspectrum")
    x = fnum(args[0])
    cols, _ = _columns(x)
    pos = 1
    fs = None
    if len(args) > 1 and not is_text(args[1]):
        fs = _sc(args[1], "sample rate")
        pos = 2
    kind = "power"
    if len(args) > pos and is_text(args[pos]) and text(args[pos]).lower() in ("power", "spectrogram", "persistence"):
        kind = text(args[pos]).lower()
        pos += 1
    if kind != "power":
        raise MatlabError(f"pspectrum: the '{kind}' type is not supported; use spectrogram or stft.")
    opts = _name_values(list(args), pos, ("FrequencyLimits", "FrequencyResolution", "Leakage", "TwoSided",
                                           "NumPowerBins"), "pspectrum")
    fsv = fs if fs is not None else 2 * np.pi
    leak = _sc(opts["leakage"], "Leakage") if "leakage" in opts else 0.5
    if not 0 <= leak <= 1:
        raise MatlabError("pspectrum: 'Leakage' must lie between 0 and 1.")
    beta = 40 * (1 - leak)
    two = False
    if "twosided" in opts:
        v = opts["twosided"]
        two = text(v).lower() in ("true", "on") if is_text(v) else bool(np.ravel(v)[0])
    two = two or np.iscomplexobj(cols)
    lim = _rvec(opts["frequencylimits"], "FrequencyLimits") if "frequencylimits" in opts else (
        np.array([-fsv / 2, fsv / 2]) if two else np.array([0.0, fsv / 2]))
    n = cols.shape[0]
    L = n
    if "frequencyresolution" in opts:
        rbw = _sc(opts["frequencyresolution"], "FrequencyResolution")
        w0 = _w_kaiser(1000, beta)
        en = 1000 * np.sum(w0 ** 2) / np.sum(w0) ** 2
        L = int(min(n, max(np.ceil(en * fsv / rbw), 2)))
    win = _w_kaiser(L, beta)
    npts = 4096
    fq = np.linspace(lim[0], lim[1], npts)
    wq = fq * (2 * np.pi / fsv)
    en = L * np.sum(win ** 2) / np.sum(win) ** 2
    step = max(int(np.floor(L / (2 * en - 1))), 1) if L < n else L
    starts = np.arange(0, n - L + 1, step)
    acc = np.zeros((npts, cols.shape[1]))
    a = np.exp(1j * wq[0])
    wstep = np.exp(-1j * (wq[1] - wq[0]))
    for s0 in starts:
        seg = cols[s0:s0 + L] * win[:, None]
        acc += np.abs(_czt(seg.astype(complex), npts, wstep, a)) ** 2
    p = acc / starts.size / np.sum(win) ** 2
    if not two:
        inner = (np.abs(fq) > 1e-12) & (np.abs(fq - fsv / 2) > 1e-12)
        p[inner] *= 2
    if nargout == 0:
        _spec_plot(interp, fq, p, fs if fs is not None else None, "Power spectrum (dB)")
        return None
    return (p, _col(fq)) if nargout > 1 else p


# -- transforms ---------------------------------------------------------------------------------------

def _dct_common(args: list, name: str, inverse: bool):
    rest = list(args)
    kind = 2
    for i in range(len(rest) - 1):
        if is_text(rest[i]) and text(rest[i]).lower() == "type":
            kind = to_int(rest[i + 1], "Type")
            del rest[i:i + 2]
            break
    if kind not in (1, 2, 3, 4):
        raise MatlabError(f"{name}: 'Type' must be 1, 2, 3 or 4.")
    nargs(rest, 1, 3, name)
    x = fnum(rest[0])
    axis = to_int(rest[2], "dimension") - 1 if _given(rest, 2) else _first_dim(x, [], 0)
    n = to_int(rest[1], "transform length") if _given(rest, 1) else None
    if x.size == 0:
        return x
    from scipy import fft as sfft
    if np.iscomplexobj(x):
        fn = sfft.idct if inverse else sfft.dct
        return fn(x.real, type=kind, n=n, axis=axis, norm="ortho") + 1j * fn(x.imag, type=kind, n=n, axis=axis,
                                                                         norm="ortho")
    fn = sfft.idct if inverse else sfft.dct
    return fn(x.astype(float), type=kind, n=n, axis=axis, norm="ortho")


@builtin("dct")
def _dct(interp, args, nargout):
    return _dct_common(args, "dct", False)


@builtin("idct")
def _idct(interp, args, nargout):
    return _dct_common(args, "idct", True)


def _dct2_common(args: list, name: str, inverse: bool):
    nargs(args, 1, 3, name)
    a = fnum(args[0]).astype(float)
    if a.ndim != 2:
        raise MatlabError(f"{name}: the input must be a matrix.")
    shape = None
    if len(args) == 2:
        v = _rvec(args[1], "size")
        shape = (int(v[0]), int(v[1])) if v.size > 1 else (int(v[0]), int(v[0]))
    elif len(args) == 3:
        shape = (to_int(args[1], "rows"), to_int(args[2], "columns"))
    from scipy import fft as sfft
    if a.size == 0:
        return a
    fn = sfft.idctn if inverse else sfft.dctn
    return fn(a, type=2, s=shape, norm="ortho")


@builtin("dct2")
def _dct2(interp, args, nargout):
    return _dct2_common(args, "dct2", False)


@builtin("idct2")
def _idct2(interp, args, nargout):
    return _dct2_common(args, "idct2", True)


@builtin("rceps")
def _rceps(interp, args, nargout):
    nargs(args, 1, 1, "rceps")
    x = fnum(args[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError("rceps: the input must be a real vector.")
    v = x.flatten(order="F").astype(float)
    n = v.size
    with np.errstate(divide="ignore"):
        y = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(v)))))
    if nargout > 1:
        w = np.zeros(n)
        w[0] = 1
        half = n // 2
        if n % 2 == 0:
            w[1:half] = 2
            w[half] = 1
        else:
            w[1:half + 1] = 2
        ym = np.real(np.fft.ifft(np.exp(np.fft.fft(w * y))))
        return _like(y, x), _like(ym, x)
    return _like(y, x)


@builtin("cceps")
def _cceps(interp, args, nargout):
    nargs(args, 1, 2, "cceps")
    x = fnum(args[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError("cceps: the input must be a real vector.")
    if nargout > 2:
        raise MatlabError("cceps: the third output (the factorization estimate) is not supported.")
    v = x.flatten(order="F").astype(float)
    n = to_int(args[1], "transform length") if _given(args, 1) else v.size
    h = np.fft.fft(v, n)
    ph = np.unwrap(np.angle(h))
    nh = (n + 1) // 2
    nd = int(np.round(ph[nh] / np.pi)) if nh < n else 0
    ph = ph - np.pi * nd * np.arange(n) / nh
    with np.errstate(divide="ignore"):
        xhat = np.real(np.fft.ifft(np.log(np.abs(h)) + 1j * ph))
    out = _like(xhat, x)
    return (out, float(nd)) if nargout > 1 else out


# -- resampling ---------------------------------------------------------------------------------------

def _upfirdn(cols: np.ndarray, h: np.ndarray, p: int, q: int) -> np.ndarray:
    n = cols.shape[0]
    if n == 0:
        return cols
    up = np.zeros(((n - 1) * p + 1, cols.shape[1]), dtype=np.result_type(cols, h))
    up[::p] = cols
    out = np.stack([np.convolve(up[:, j], h) for j in range(cols.shape[1])], axis=1)
    return out[::q]


@builtin("upfirdn")
def _upfirdn_b(interp, args, nargout):
    nargs(args, 2, 4, "upfirdn")
    x = fnum(args[0])
    h = _vec(args[1], "filter")
    p = to_int(args[2], "p") if _given(args, 2) else 1
    q = to_int(args[3], "q") if _given(args, 3) else 1
    if p < 1 or q < 1:
        raise MatlabError("upfirdn: p and q must be positive integers.")
    cols, was_row = _columns(x)
    return _restore(simplify(_upfirdn(cols, h, p, q)), was_row)


def _resample_filter(p: int, q: int, n: int, beta: float) -> np.ndarray:
    """Kaiser-windowed least-squares lowpass (cutoff at the lower Nyquist rate) with gain p."""
    pq = max(p, q)
    fc = 1.0 / (2 * pq)
    L = 2 * n * pq + 1
    m = np.arange(L) - (L - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * m) * np.kaiser(L, beta)
    return p * h / h.sum()


def resample_impl(interp, args: list, nargout: int):
    nargs(args, 2, None, "resample")
    rest = list(args)
    dim = None
    for i in range(len(rest) - 1):
        if is_text(rest[i]) and text(rest[i]).lower() == "dimension":
            dim = to_int(rest[i + 1], "Dimension") - 1
            del rest[i:i + 2]
            break
    method = None
    if rest and is_text(rest[-1]):
        method = text(rest.pop()).lower()
    x = fnum(rest[0])
    second = fnum(rest[1])
    if second.size > 1 and second.size == (x.shape[dim] if dim is not None else (x.size if _isvec(x) else x.shape[0])):
        return _resample_nonuniform(x, second, rest[2:], method, nargout)
    if method is not None:
        raise MatlabError(f"resample: unknown option '{method}'.")
    if len(rest) < 3:
        raise MatlabError("resample: expected resample(x, p, q).")
    p, q = to_int(rest[1], "p"), to_int(rest[2], "q")
    if p <= 0 or q <= 0:
        raise MatlabError("resample: p and q must be positive integers.")
    g = int(np.gcd(p, q))
    p, q = p // g, q // g
    axis = dim if dim is not None else _first_dim(x, [], 0)
    xm = np.moveaxis(x, axis, 0)
    shape = xm.shape
    cols = xm.reshape(shape[0], -1)
    lx = cols.shape[0]
    lout = int(np.ceil(lx * p / q))
    h = None
    if _given(rest, 3) and fnum(rest[3]).size > 1:
        h = _rvec(rest[3], "filter")
    else:
        n = to_int(rest[3], "n") if _given(rest, 3) else 10
        beta = _sc(rest[4], "beta") if _given(rest, 4) else 5.0
        if n == 0:
            idx = np.minimum(np.floor(np.arange(lout) * q / p + 0.5).astype(int), lx - 1)
            y = cols[idx]
            y = np.moveaxis(y.reshape((lout,) + shape[1:]), 0, axis)
            return (y, np.ones((1, 1))) if nargout > 1 else y
        if p == 1 and q == 1:
            return (x, np.ones((1, 1))) if nargout > 1 else x
        h = _resample_filter(p, q, n, beta)
    L = h.size
    half = (L - 1) / 2
    nz = int(np.floor(q - np.mod(half, q)))
    hp = np.concatenate([np.zeros(nz), h])
    half += nz
    delay = int(np.floor(np.ceil(half) / q))
    extra = 0
    while int(np.ceil(((lx - 1) * p + hp.size + extra) / q)) - delay < lout:
        extra += 1
    hp = np.concatenate([hp, np.zeros(extra)])
    y = _upfirdn(cols, hp, p, q)[delay:delay + lout]
    y = np.moveaxis(y.reshape((lout,) + shape[1:]), 0, axis)
    return (y, _row(h)) if nargout > 1 else y


def _resample_nonuniform(x, tx, rest: list, method, nargout: int):
    from scipy import interpolate as si
    cols, was_row = _columns(x)
    t = np.real(tx).flatten(order="F").astype(float)
    if np.any(np.diff(t) <= 0):
        raise MatlabError("resample: the sample times must be increasing.")
    fs = _sc(rest[0], "sample rate") if _given(rest, 0) else 1.0 / float(np.median(np.diff(t)))
    p = to_int(rest[1], "p") if _given(rest, 1) else 1
    q = to_int(rest[2], "q") if _given(rest, 2) else 1
    method = method or "linear"
    step = (p / q) / fs
    grid = np.arange(t[0], t[-1] + step * 1e-9, step)
    if method == "linear":
        mid = np.stack([np.interp(grid, t, cols[:, j]) for j in range(cols.shape[1])], axis=1)
    elif method == "pchip":
        mid = si.PchipInterpolator(t, cols, axis=0)(grid)
    elif method == "spline":
        mid = si.CubicSpline(t, cols, axis=0)(grid)
    else:
        raise MatlabError("resample: the interpolation method must be 'linear', 'pchip' or 'spline'.")
    if p != 1 or q != 1:
        y = resample_impl(None, [mid, _f(p), _f(q)], 1)
    else:
        y = mid
    ty = t[0] + np.arange(y.shape[0]) / fs
    y = _restore(y, was_row)
    return (y, _like(ty, x)) if nargout > 1 else y


def decimate_impl(args: list, nargout: int):
    nargs(args, 2, 4, "decimate")
    x = fnum(args[0])
    if not _isvec(x):
        raise MatlabError("decimate: the signal must be a vector.")
    r = to_int(args[1], "decimation factor")
    if r < 1:
        raise MatlabError("decimate: the decimation factor must be a positive integer.")
    fir = False
    order = None
    for a in args[2:]:
        if is_text(a):
            s = text(a).lower()
            if s not in ("fir", "iir"):
                raise MatlabError("decimate: the filter type must be 'fir' or 'iir'.")
            fir = s == "fir"
        elif isinstance(a, np.ndarray) and a.size:
            order = to_int(a, "filter order")
    v = x.flatten(order="F").astype(float if np.isrealobj(x) else complex)
    nd = v.size
    if r == 1:
        return x
    nout = int(np.ceil(nd / r))
    ss = _sig()
    if fir:
        nf = order if order is not None else 30
        if nd <= nf:
            raise MatlabError("decimate: the signal must be longer than the FIR filter order.")
        b = _fir1_core(nf, np.array([1.0 / r]), True, _w_hamming(nf + 1), True)
        # mirror the ends to suppress the start-up and ending transients, then remove the filter delay
        pre = 2 * v[0] - v[nf:0:-1]
        post = 2 * v[-1] - v[-2:-nf - 2:-1]
        ext = np.concatenate([pre, v, post])
        yf = ss.lfilter(b, [1.0], ext)
        gd = nf / 2
        start = nf + int(np.round(gd + 0.25))
        y = yf[start:start + (nout - 1) * r + 1:r]
    else:
        n = order if order is not None else 8
        if nd <= 3 * n:
            raise MatlabError("decimate: the signal must be more than three times as long as the filter order.")
        b, a = ss.cheby1(n, 0.05, 0.8 / r)
        yf = ss.filtfilt(b, a, v, padtype="odd", padlen=3 * n)
        nbeg = r - (r * nout - nd)
        y = yf[nbeg - 1::r]
    return _like(y, x)


def _intfilt_band(l: int, p: int, alpha: float) -> np.ndarray:
    """Least-squares interpolator for signals band-limited to alpha times the Nyquist frequency.

    Each of the l-1 in-between positions is estimated from the 2p nearest original samples; the
    weights minimise the mean-square error for a flat spectrum up to alpha (and none above it).
    """
    k = np.arange(-p + 1, p + 1, dtype=float)             # original sample positions around [0, 1)
    R = np.sinc(alpha * (k[:, None] - k[None, :]))
    h = np.zeros(2 * l * p - 1)
    centre = l * p - 1
    h[centre] = 1.0
    for d in range(1, l):
        frac = d / l
        rhs = np.sinc(alpha * (k - frac))
        c = np.linalg.lstsq(R, rhs, rcond=1e-13)[0]
        # output at position frac uses x[k] with weight c: the tap at offset (frac - k) * l
        for kk, ck in zip(k, c):
            h[centre + int(round((frac - kk) * l))] = ck
    return h


def _intfilt_lagrange(l: int, order: int) -> np.ndarray:
    npts = order + 1
    # nodes at 0..order; interpolate inside the middle interval
    nodes = np.arange(npts, dtype=float)
    base = (order - 1) // 2 if order % 2 else order // 2 - 0.5
    taps = {}
    for d in range(l):
        frac = d / l
        xq = base + frac
        for j in range(npts):
            others = nodes[nodes != j]
            wgt = float(np.prod((xq - others) / (j - others)))
            off = int(round((xq - j) * l))
            taps[off] = taps.get(off, 0.0) + wgt
    lo, hi = min(taps), max(taps)
    h = np.zeros(hi - lo + 1)
    for off, wgt in taps.items():
        h[off - lo] = wgt
    while h.size and abs(h[0]) < 1e-14:
        h = h[1:]
    while h.size and abs(h[-1]) < 1e-14:
        h = h[:-1]
    return h


@builtin("intfilt")
def _intfilt(interp, args, nargout):
    nargs(args, 3, 3, "intfilt")
    l = to_int(args[0], "interpolation factor")
    if l < 1:
        raise MatlabError("intfilt: the interpolation factor must be a positive integer.")
    if is_text(args[2]):
        kind = text(args[2]).lower()
        if kind not in ("lagrange", "l"):
            raise MatlabError("intfilt: the third input must be the band limit alpha or 'Lagrange'.")
        n = to_int(args[1], "order")
        b = _intfilt_lagrange(l, n)
    else:
        p = to_int(args[1], "p")
        alpha = _sc(args[2], "alpha")
        if not 0 < alpha <= 1:
            raise MatlabError("intfilt: alpha must lie between 0 and 1.")
        b = _intfilt_band(l, p, alpha)
    return (_row(b), np.ones((1, 1))) if nargout > 1 else _row(b)


@builtin("interp")
def _interp(interp, args, nargout):
    nargs(args, 2, 4, "interp")
    x = fnum(args[0])
    if not _isvec(x):
        raise MatlabError("interp: the signal must be a vector.")
    r = to_int(args[1], "interpolation factor")
    n = to_int(args[2], "n") if _given(args, 2) else 4
    cutoff = _sc(args[3], "cutoff") if _given(args, 3) else 0.5
    if r < 1:
        raise MatlabError("interp: the interpolation factor must be a positive integer.")
    v = x.flatten(order="F")
    if r == 1:
        return (x, np.ones((1, 1))) if nargout > 1 else x
    if v.size < 2 * n + 1:
        raise MatlabError(f"interp: the signal must have at least {2 * n + 1} samples.")
    core = _intfilt_band(r, n, cutoff)
    b = np.concatenate([[0.0], core, [0.0]])
    # extend both ends by odd reflection so the filter has samples to work with
    pre = 2 * v[0] - v[n:0:-1]
    post = 2 * v[-1] - v[-2:-n - 2:-1]
    ext = np.concatenate([pre, v, post])
    up = np.zeros(ext.size * r, dtype=ext.dtype)
    up[::r] = ext
    y = np.convolve(up, b)
    start = n * r + n * r        # the filter delay plus the samples added at the front
    y = y[start:start + v.size * r]
    out = _like(y, x)
    return (out, _row(b)) if nargout > 1 else out


# -- waveform generation --------------------------------------------------------------------------------

@builtin("rectpuls")
def _rectpuls(interp, args, nargout):
    nargs(args, 1, 2, "rectpuls")
    t = np.real(fnum(args[0])).astype(float)
    w = _sc(args[1], "width") if len(args) > 1 else 1.0
    eps_ = np.finfo(float).eps
    y = (np.abs(t) < w / 2 - eps_) | (np.abs(t + w / 2) < eps_)
    return y.astype(float)


@builtin("tripuls")
def _tripuls(interp, args, nargout):
    nargs(args, 1, 3, "tripuls")
    t = np.real(fnum(args[0])).astype(float)
    w = _sc(args[1], "width") if _given(args, 1) else 1.0
    s = _sc(args[2], "skew") if _given(args, 2) else 0.0
    if not -1 <= s <= 1:
        raise MatlabError("tripuls: the skew must lie between -1 and 1.")
    peak = s * w / 2
    y = np.zeros(t.shape)
    left = (t > -w / 2) & (t < peak)
    right = (t >= peak) & (t < w / 2)
    y[left] = (t[left] + w / 2) / (peak + w / 2)
    y[right] = (w / 2 - t[right]) / (w / 2 - peak)
    return y


@builtin("gauspuls")
def _gauspuls(interp, args, nargout):
    nargs(args, 1, 5, "gauspuls")
    cutoff = is_text(args[0])
    if cutoff and text(args[0]).lower() != "cutoff":
        raise MatlabError("gauspuls: the only text option is 'cutoff'.")
    fc = _sc(args[1], "fc") if _given(args, 1) else 1000.0
    bw = _sc(args[2], "bw") if _given(args, 2) else 0.5
    bwr = _sc(args[3], "bwr") if _given(args, 3) else -6.0
    if fc < 0 or bw <= 0 or bwr >= 0:
        raise MatlabError("gauspuls: fc must be non-negative, bw positive and bwr negative.")
    r = 10 ** (bwr / 20)
    fv = -(bw * fc) ** 2 / (8 * np.log(r))
    tv = 1 / (4 * np.pi ** 2 * fv)
    if cutoff:
        tpe = _sc(args[4], "tpe") if _given(args, 4) else -60.0
        if tpe >= 0:
            raise MatlabError("gauspuls: tpe must be negative.")
        return float(np.sqrt(-2 * tv * np.log(10 ** (tpe / 20))))
    t = np.real(fnum(args[0])).astype(float)
    ye = np.exp(-t * t / (2 * tv))
    yi = ye * np.cos(2 * np.pi * fc * t)
    yq = ye * np.sin(2 * np.pi * fc * t)
    out = (yi, yq, ye)
    return out[:nargout] if nargout > 1 else yi


@builtin("gmonopuls")
def _gmonopuls(interp, args, nargout):
    nargs(args, 1, 2, "gmonopuls")
    fc = _sc(args[1], "fc") if _given(args, 1) else 1000.0
    if fc <= 0:
        raise MatlabError("gmonopuls: the centre frequency must be positive.")
    if is_text(args[0]):
        if text(args[0]).lower() != "cutoff":
            raise MatlabError("gmonopuls: the only text option is 'cutoff'.")
        return float(1 / (np.pi * fc))
    t = np.real(fnum(args[0])).astype(float)
    return 2 * np.sqrt(np.e) * np.pi * t * fc * np.exp(-2 * (np.pi * t * fc) ** 2)


@builtin("diric")
def _diric(interp, args, nargout):
    nargs(args, 2, 2, "diric")
    x = np.real(fnum(args[0])).astype(float)
    n = to_int(args[1], "n")
    if n < 1:
        raise MatlabError("diric: n must be a positive integer.")
    den = n * np.sin(x / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sin(n * x / 2) / den
    sing = np.abs(np.sin(x / 2)) < 1e-12
    k = np.round(x / (2 * np.pi))
    return np.where(sing, (-1.0) ** (k * (n - 1)), y)


@builtin("pulstran")
def _pulstran(interp, args, nargout):
    nargs(args, 2, None, "pulstran")
    t = np.real(fnum(args[0])).astype(float)
    d = np.real(fnum(args[1])).astype(float)
    if d.ndim == 2 and d.shape[1] == 2 and d.shape[0] > 1:
        delays, amps = d[:, 0], d[:, 1]
    else:
        delays = d.flatten(order="F")
        amps = np.ones(delays.size)
    y = np.zeros(t.shape)
    if len(args) < 3:
        raise MatlabError("pulstran: a pulse function or a sampled pulse is required.")
    f = args[2]
    if isinstance(f, FunctionHandle) or is_text(f):
        extra = list(args[3:])
        for dk, ak in zip(delays, amps):
            y = y + ak * np.real(fnum(interp.call_function_value(f, [t - dk, *extra], 1)[0]))
        return y
    pulse = np.real(fnum(f)).flatten(order="F").astype(float)
    fs = _sc(args[3], "sample rate") if _given(args, 3) else 1.0
    method = text(args[4]).lower() if len(args) > 4 else "linear"
    tp = np.arange(pulse.size) / fs
    from scipy import interpolate as si
    kinds = {"linear": "linear", "nearest": "nearest", "cubic": "cubic", "spline": "cubic", "pchip": "pchip"}
    if method not in kinds:
        raise MatlabError(f"pulstran: unknown interpolation method '{method}'.")
    if method == "pchip":
        fn = si.PchipInterpolator(tp, pulse, extrapolate=False)
    else:
        fn = si.interp1d(tp, pulse, kind=kinds[method], bounds_error=False, fill_value=0.0)
    for dk, ak in zip(delays, amps):
        v = fn((t - dk).ravel()).reshape(t.shape)
        y = y + ak * np.nan_to_num(v)
    return y


@builtin("vco")
def _vco(interp, args, nargout):
    nargs(args, 1, 3, "vco")
    x = np.real(fnum(args[0])).astype(float)
    fs = _sc(args[2], "sample rate") if _given(args, 2) else 2.0
    if np.max(np.abs(x), initial=0.0) > 1:
        raise MatlabError("vco: the input must lie between -1 and 1.")
    if _given(args, 1):
        fcv = _rvec(args[1], "fc")
        if fcv.size == 2:
            fc = float(np.mean(fcv))
            dev = float(fcv[1] - fcv[0]) / 2
        else:
            fc = float(fcv[0])
            dev = fc
    else:
        fc = fs / 4
        dev = fc
    if fc + abs(dev) > fs / 2 + 1e-12 and fc > fs / 2:
        raise MatlabError("vco: the carrier frequency must not exceed half the sample rate.")
    cols, was_row = _columns(x)
    n = cols.shape[0]
    t = (np.arange(n) / fs)[:, None]
    y = np.cos(2 * np.pi * fc * t + 2 * np.pi * dev * np.cumsum(cols, axis=0) / fs)
    return _restore(y, was_row)


@builtin("buffer")
def _buffer(interp, args, nargout):
    nargs(args, 2, 4, "buffer")
    x = fnum(args[0])
    v = x.flatten(order="F")
    n = to_int(args[1], "frame length")
    p = to_int(args[2], "overlap") if _given(args, 2) else 0
    if n < 1:
        raise MatlabError("buffer: the frame length must be positive.")
    if p >= n:
        raise MatlabError("buffer: the overlap must be smaller than the frame length.")
    opt = args[3] if len(args) > 3 else None
    step = n - p
    if p > 0:
        if opt is not None and is_text(opt):
            if text(opt).lower() != "nodelay":
                raise MatlabError("buffer: the option must be 'nodelay' or a vector of initial samples.")
            data = v
        else:
            init = _vec(opt, "initial condition") if opt is not None else np.zeros(p)
            if init.size != p:
                raise MatlabError(f"buffer: the initial condition must have {p} elements.")
            data = np.concatenate([init, v])
        starts = np.arange(0, max(data.size - p, 1), step) if data.size > p else np.arange(0, 1)
    else:
        skip = int(to_int(opt, "skip")) if opt is not None and not is_text(opt) else 0
        if not 0 <= skip <= -p:
            raise MatlabError("buffer: the number of skipped samples must lie between 0 and -p.")
        data = v[skip:]
        starts = np.arange(0, data.size, step)
    full = [int(s0) for s0 in starts if s0 + n <= data.size]
    if nargout > 1:
        frames = [data[s0:s0 + n] for s0 in full]
        y = np.stack(frames, axis=1) if frames else np.zeros((n, 0))
        nxt = full[-1] + step if full else 0
        if p > 0:
            z = data[full[-1] + n:] if full else v
            state = _col(frames[-1][n - p:]) if frames else _col(data[:p])
        else:
            z = data[nxt:]
            state = float(max(nxt - data.size, 0))
        return y, _like(z, x), state
    frames = []
    for s0 in starts:
        seg = data[s0:s0 + n]
        if seg.size < n:
            seg = np.concatenate([seg, np.zeros(n - seg.size, dtype=v.dtype)])
        frames.append(seg)
    return np.stack(frames, axis=1) if frames else np.zeros((n, 0))


# -- simple measurements --------------------------------------------------------------------------------

def _reduce_dim(x: np.ndarray, args: list, pos: int) -> int:
    return to_int(args[pos], "dimension") - 1 if _given(args, pos) else _first_dim(x, [], 0)


def _keep(r: np.ndarray, axis: int) -> np.ndarray:
    return np.expand_dims(r, axis)


@builtin("peak2peak")
def _peak2peak(interp, args, nargout):
    nargs(args, 1, 2, "peak2peak")
    x = fnum(args[0])
    axis = _reduce_dim(x, args, 1)
    if axis >= x.ndim:
        return np.zeros(x.shape)
    if np.iscomplexobj(x):
        mag = np.abs(x)
        hi = np.take_along_axis(x, np.argmax(mag, axis=axis, keepdims=True), axis)
        lo = np.take_along_axis(x, np.argmin(mag, axis=axis, keepdims=True), axis)
        return hi - lo
    return _keep(np.max(x, axis=axis) - np.min(x, axis=axis), axis)


@builtin("peak2rms")
def _peak2rms(interp, args, nargout):
    nargs(args, 1, 2, "peak2rms")
    x = fnum(args[0])
    axis = _reduce_dim(x, args, 1)
    if axis >= x.ndim:
        return np.ones(x.shape)
    peak = np.max(np.abs(x), axis=axis)
    rms = np.sqrt(np.mean(np.abs(x) ** 2, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _keep(peak / rms, axis)


@builtin("rssq")
def _rssq(interp, args, nargout):
    nargs(args, 1, 2, "rssq")
    x = fnum(args[0])
    axis = _reduce_dim(x, args, 1)
    if axis >= x.ndim:
        return np.abs(x)
    return _keep(np.sqrt(np.sum(np.abs(x) ** 2, axis=axis)), axis)


def _rms(interp, args, nargout):
    nargs(args, 1, 3, "rms")
    rest, flags = _split_flags(list(args))
    x = fnum(rest[0])
    if "all" in flags:
        return float(np.sqrt(np.mean(np.abs(x) ** 2))) if x.size else float("nan")
    axis = _reduce_dim(x, rest, 1)
    if axis >= x.ndim:
        return np.abs(x).astype(float)
    fn = np.nanmean if "omitnan" in flags else np.mean
    return _keep(np.sqrt(fn(np.abs(x) ** 2, axis=axis)), axis)


_maybe_register("rms", _rms)


# -- peaks ------------------------------------------------------------------------------------------------

def _local_maxima(y: np.ndarray) -> np.ndarray:
    """Indices of local maxima; a flat peak is reported at its first sample; end points never count."""
    n = y.size
    locs = []
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1]:
            j = i
            while j < n - 1 and y[j + 1] == y[i]:
                j += 1
            if j < n - 1 and y[j + 1] < y[i]:
                locs.append(i)
            elif j == n - 1 and np.isinf(y[i]) and y[i] > 0:
                pass
            i = j + 1
        else:
            i += 1
    return np.array(locs, dtype=int)


def _prominence(y: np.ndarray, locs: np.ndarray):
    """Prominence of each peak with the positions of its left and right bases."""
    n = y.size
    prom = np.zeros(locs.size)
    lb = np.zeros(locs.size, dtype=int)
    rb = np.zeros(locs.size, dtype=int)
    for k, i in enumerate(locs):
        h = y[i]
        j = i
        lmin, lpos = h, i
        while j > 0:
            j -= 1
            if y[j] > h:
                break
            if y[j] < lmin:
                lmin, lpos = y[j], j
        j = i
        rmin, rpos = h, i
        while j < n - 1:
            j += 1
            if y[j] > h:
                break
            if y[j] < rmin:
                rmin, rpos = y[j], j
        prom[k] = h - max(lmin, rmin)
        lb[k], rb[k] = lpos, rpos
    return prom, lb, rb


def _peak_widths(y: np.ndarray, x: np.ndarray, locs: np.ndarray, prom: np.ndarray, lb: np.ndarray,
                 rb: np.ndarray, halfheight: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = y.size
    left = np.zeros(locs.size)
    right = np.zeros(locs.size)
    for k, i in enumerate(locs):
        ref = y[i] / 2 if halfheight else y[i] - prom[k] / 2
        j = i
        stop = 0 if halfheight else lb[k]
        while j > stop and y[j] > ref:
            j -= 1
        if y[j] > ref or j == i:
            left[k] = x[j]
        else:
            left[k] = x[j] + (ref - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j]) if y[j + 1] != y[j] else x[j]
        j = i
        stop = n - 1 if halfheight else rb[k]
        while j < stop and y[j] > ref:
            j += 1
        if y[j] > ref or j == i:
            right[k] = x[j]
        else:
            right[k] = x[j] - (ref - y[j]) * (x[j] - x[j - 1]) / (y[j - 1] - y[j]) if y[j - 1] != y[j] else x[j]
    return right - left, left, right


_FP_OPTS = ("MinPeakHeight", "MinPeakProminence", "Threshold", "MinPeakDistance", "MinPeakWidth",
            "MaxPeakWidth", "WidthReference", "SortStr", "NPeaks", "Annotate")


def findpeaks_impl(interp, args: list, nargout: int):
    nargs(args, 1, None, "findpeaks")
    y = fnum(args[0])
    if not _isvec(y) or np.iscomplexobj(y):
        raise MatlabError("findpeaks: the data must be a real vector.")
    yv = y.flatten(order="F").astype(float)
    n = yv.size
    xv = np.arange(1, n + 1, dtype=float)
    start = 1
    if len(args) > 1 and not is_text(args[1]):
        xa = np.real(fnum(args[1])).flatten(order="F").astype(float)
        if xa.size == 1 and n > 1:
            xv = np.arange(n) / xa[0]
        elif xa.size == n:
            if np.any(np.diff(xa) <= 0):
                raise MatlabError("findpeaks: the sample locations must be increasing.")
            xv = xa
        else:
            raise MatlabError("findpeaks: the locations must have one value per data point (or be a sample rate).")
        start = 2
    opts = _name_values(list(args), start, _FP_OPTS, "findpeaks")
    locs = _local_maxima(yv)
    if "minpeakheight" in opts and locs.size:
        locs = locs[yv[locs] > _sc(opts["minpeakheight"], "MinPeakHeight")]
    if "threshold" in opts and locs.size:
        th = _sc(opts["threshold"], "Threshold")
        base = np.maximum(yv[locs - 1], yv[np.minimum(locs + 1, n - 1)])
        locs = locs[yv[locs] - base >= th]
    need_shape = nargout > 2 or any(k in opts for k in ("minpeakprominence", "minpeakwidth", "maxpeakwidth")) \
        or nargout == 0
    prom = width = None
    if need_shape:
        ref = text(opts["widthreference"]).lower() if "widthreference" in opts else "halfprom"
        if ref not in ("halfprom", "halfheight"):
            raise MatlabError("findpeaks: 'WidthReference' must be 'halfprom' or 'halfheight'.")
        prom, lb, rb = _prominence(yv, locs)
        if "minpeakprominence" in opts:
            keep = prom >= _sc(opts["minpeakprominence"], "MinPeakProminence")
            locs, prom, lb, rb = locs[keep], prom[keep], lb[keep], rb[keep]
        width, _l, _r = _peak_widths(yv, xv, locs, prom, lb, rb, ref == "halfheight")
        keep = np.ones(locs.size, dtype=bool)
        if "minpeakwidth" in opts:
            keep &= width >= _sc(opts["minpeakwidth"], "MinPeakWidth")
        if "maxpeakwidth" in opts:
            keep &= width <= _sc(opts["maxpeakwidth"], "MaxPeakWidth")
        locs, prom, width = locs[keep], prom[keep], width[keep]
    if "minpeakdistance" in opts and locs.size:
        d = _sc(opts["minpeakdistance"], "MinPeakDistance")
        order = np.argsort(-yv[locs], kind="stable")
        gone = np.zeros(locs.size, dtype=bool)
        px = xv[locs]
        for k in order:
            if gone[k]:
                continue
            near = (px >= px[k] - d) & (px <= px[k] + d)
            near[k] = False
            gone |= near
        keep = ~gone
        locs = locs[keep]
        if prom is not None:
            prom, width = prom[keep], width[keep]
    sort = text(opts["sortstr"]).lower() if "sortstr" in opts else "none"
    if sort not in ("none", "ascend", "descend"):
        raise MatlabError("findpeaks: 'SortStr' must be 'ascend', 'descend' or 'none'.")
    if sort != "none":
        order = np.argsort(-yv[locs] if sort == "descend" else yv[locs], kind="stable")
        locs = locs[order]
        if prom is not None:
            prom, width = prom[order], width[order]
    if "npeaks" in opts:
        k = to_int(opts["npeaks"], "NPeaks")
        locs = locs[:k]
        if prom is not None:
            prom, width = prom[:k], width[:k]
    shape = (1, -1) if y.shape[0] == 1 else (-1, 1)
    if nargout == 0:
        _plot(interp, "plot", [_row(xv), _row(yv)])
        _plot(interp, "hold", [char("on")])
        _plot(interp, "plot", [_row(xv[locs]), _row(yv[locs]), char("v")])
        _plot(interp, "hold", [char("off")])
        return None
    out = [yv[locs].reshape(shape), xv[locs].reshape(shape)]
    if nargout > 2:
        out += [width.reshape(shape), prom.reshape(shape)]
    return tuple(out[:nargout]) if nargout > 1 else out[0]


# -- envelopes --------------------------------------------------------------------------------------------

def _hilbert_fir(L: int) -> np.ndarray:
    """Kaiser-windowed (beta = 8) FIR Hilbert transformer of length L."""
    m = np.arange(L) - (L - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(m == 0, 0.0, (1 - np.cos(np.pi * m)) / (np.pi * m))
    return h * np.kaiser(L, 8.0)


def _moving_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average along axis 0 whose window shrinks at the ends."""
    n = x.shape[0]
    before = w // 2
    after = w - 1 - before
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    i = np.arange(n)
    lo = np.maximum(i - before, 0)
    hi = np.minimum(i + after, n - 1) + 1
    cnt = (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (c[hi] - c[lo]) / cnt


def _peak_envelope(v: np.ndarray, sep: int, upper: bool) -> np.ndarray:
    from scipy import interpolate as si
    n = v.size
    s = v if upper else -v
    locs = _local_maxima(s)
    if locs.size and sep > 1:
        order = np.argsort(-s[locs], kind="stable")
        gone = np.zeros(locs.size, dtype=bool)
        for k in order:
            if gone[k]:
                continue
            near = np.abs(locs - locs[k]) < sep
            near[k] = False
            gone |= near
        locs = locs[~gone]
    if locs.size == 0:
        return np.full(n, v.max() if upper else v.min())
    if locs.size == 1:
        return np.full(n, v[locs[0]])
    xs = np.arange(n, dtype=float)
    if locs.size < 4:
        return si.interp1d(locs.astype(float), v[locs], kind="linear", fill_value="extrapolate")(xs)
    return si.CubicSpline(locs.astype(float), v[locs], bc_type="not-a-knot")(xs)


def envelope_impl(interp, args: list, nargout: int):
    nargs(args, 1, 3, "envelope")
    x = fnum(args[0])
    if np.iscomplexobj(x):
        raise MatlabError("envelope: the signal must be real.")
    cols, was_row = _columns(x.astype(float))
    method = text(args[2]).lower() if len(args) > 2 else "analytic"
    if method not in ("analytic", "rms", "peak"):
        raise MatlabError("envelope: the method must be 'analytic', 'rms' or 'peak'.")
    param = to_int(args[1], "window length") if len(args) > 1 else None
    if len(args) == 2 and param is None:
        raise MatlabError("envelope: a filter or window length is required.")
    if param is not None and param < 1:
        raise MatlabError("envelope: the length must be a positive integer.")
    n = cols.shape[0]
    if method == "peak":
        up = np.stack([_peak_envelope(cols[:, j], param, True) for j in range(cols.shape[1])], axis=1)
        lo = np.stack([_peak_envelope(cols[:, j], param, False) for j in range(cols.shape[1])], axis=1)
    else:
        mu = cols.mean(axis=0, keepdims=True)
        ac = cols - mu
        if method == "rms":
            amp = np.sqrt(_moving_mean(ac ** 2, min(param, n)))
        elif param is None:
            amp = np.abs(_sig().hilbert(ac, axis=0))
        else:
            h = _hilbert_fir(param)
            half = (param - 1) // 2
            q = np.stack([np.convolve(ac[:, j], h)[half:half + n] for j in range(ac.shape[1])], axis=1)
            amp = np.abs(ac + 1j * q)
        up, lo = mu + amp, mu - amp
    up, lo = _restore(up, was_row), _restore(lo, was_row)
    if nargout == 0:
        t = _row(np.arange(1, n + 1))
        _plot(interp, "plot", [t, _row(cols[:, 0]), t, _row(np.ravel(up) if up.ndim == 2 and min(up.shape) == 1
                                                           else up[:, 0]),
                               t, _row(np.ravel(lo) if lo.ndim == 2 and min(lo.shape) == 1 else lo[:, 0])])
        return None
    return (up, lo) if nargout > 1 else up


# -- smoothing, outliers and delays -----------------------------------------------------------------------

@builtin("sgolay")
def _sgolay(interp, args, nargout):
    nargs(args, 2, 3, "sgolay")
    order = to_int(args[0], "order")
    f = to_int(args[1], "frame length")
    if f % 2 == 0 or f < 1:
        raise MatlabError("sgolay: the frame length must be odd.")
    if order >= f or order < 0:
        raise MatlabError("sgolay: the order must be less than the frame length.")
    w = _rvec(args[2], "weights") if _given(args, 2) else np.ones(f)
    if w.size != f or np.any(w <= 0):
        raise MatlabError("sgolay: there must be one positive weight per frame sample.")
    half = (f - 1) // 2
    S = np.vander(np.arange(-half, half + 1, dtype=float), order + 1, increasing=True)
    W = np.diag(w)
    inv = np.linalg.inv(S.T @ W @ S)
    G = W @ S @ inv
    B = G @ S.T
    return (B.T, G) if nargout > 1 else B.T


@builtin("hampel")
def _hampel(interp, args, nargout):
    nargs(args, 1, 3, "hampel")
    x = fnum(args[0])
    if np.iscomplexobj(x):
        raise MatlabError("hampel: the data must be real.")
    k = to_int(args[1], "window half-width") if _given(args, 1) else 3
    nsig = _sc(args[2], "number of deviations") if _given(args, 2) else 3.0
    if k < 1 or nsig < 0:
        raise MatlabError("hampel: the half-width must be positive and the threshold non-negative.")
    cols, was_row = _columns(x.astype(float))
    n = cols.shape[0]
    med = np.empty_like(cols)
    sig = np.empty_like(cols)
    for i in range(n):
        seg = cols[max(i - k, 0):min(i + k, n - 1) + 1]
        m = np.nanmedian(seg, axis=0)
        med[i] = m
        sig[i] = 1.4826 * np.nanmedian(np.abs(seg - m), axis=0)
    bad = np.abs(cols - med) > nsig * sig
    y = np.where(bad, med, cols)
    out = tuple(_restore(v, was_row) for v in (y, bad, med, sig))
    return out[:nargout] if nargout > 1 else out[0]


def _finddelay(x: np.ndarray, y: np.ndarray, maxlag: int | None) -> int:
    n = max(x.size, y.size)
    if maxlag is None:
        maxlag = n - 1
    xp = np.concatenate([x, np.zeros(n - x.size)])
    yp = np.concatenate([y, np.zeros(n - y.size)])
    c = _xcorr_pair(yp, xp, n)
    lags = np.arange(-(n - 1), n)
    sel = np.abs(lags) <= maxlag
    c, lags = np.abs(c[sel]), lags[sel]
    den = np.sqrt(np.sum(np.abs(xp) ** 2) * np.sum(np.abs(yp) ** 2))
    if den == 0 or c.size == 0:
        return 0
    best = np.max(c)
    cand = lags[c >= best * (1 - 1e-12)]
    # among equal maxima take the smallest shift, preferring a positive one
    cand = sorted(cand, key=lambda v: (abs(v), -v))
    return int(cand[0])


@builtin("finddelay")
def _finddelay_b(interp, args, nargout):
    nargs(args, 2, 3, "finddelay")
    x, y = fnum(args[0]), fnum(args[1])
    maxlag = to_int(args[2], "maxlag") if _given(args, 2) else None
    xc, _ = _columns(x)
    yc, _ = _columns(y)
    if xc.shape[1] == 1 and yc.shape[1] == 1:
        return float(_finddelay(xc[:, 0], yc[:, 0], maxlag))
    if xc.shape[1] == 1:
        xc = np.repeat(xc, yc.shape[1], axis=1)
    if yc.shape[1] == 1:
        yc = np.repeat(yc, xc.shape[1], axis=1)
    if xc.shape[1] != yc.shape[1]:
        raise MatlabError("finddelay: the inputs must have the same number of columns.")
    return _row([float(_finddelay(xc[:, j], yc[:, j], maxlag)) for j in range(xc.shape[1])])


@builtin("alignsignals")
def _alignsignals(interp, args, nargout):
    nargs(args, 2, 4, "alignsignals")
    rest, flags = _split_flags(list(args))
    x, y = fnum(rest[0]), fnum(rest[1])
    if not _isvec(x) or not _isvec(y):
        raise MatlabError("alignsignals: the signals must be vectors.")
    maxlag = to_int(rest[2], "maxlag") if _given(rest, 2) else None
    xv, yv = x.flatten(order="F"), y.flatten(order="F")
    d = _finddelay(xv, yv, maxlag)
    if d > 0:
        xv = np.concatenate([np.zeros(d, dtype=xv.dtype), xv])
    elif d < 0:
        yv = np.concatenate([np.zeros(-d, dtype=yv.dtype), yv])
    if "truncate" in flags:
        if d > 0:
            xv = xv[:x.size]
        elif d < 0:
            yv = yv[:y.size]
    out = (_like(xv, x), _like(yv, y), float(d))
    return out[:nargout] if nargout > 1 else out[0]


@builtin("dtw")
def _dtw(interp, args, nargout):
    nargs(args, 2, 4, "dtw")
    rest, flags = _split_flags(list(args))
    x, y = fnum(rest[0]), fnum(rest[1])
    X = x.reshape(1, -1) if _isvec(x) else x
    Y = y.reshape(1, -1) if _isvec(y) else y
    if X.shape[0] != Y.shape[0]:
        raise MatlabError("dtw: the signals must have the same number of rows.")
    maxsamp = to_int(rest[2], "maxsamp") if _given(rest, 2) else None
    metric = flags[0] if flags else "euclidean"
    diff = X[:, :, None] - Y[:, None, :]
    if metric == "euclidean":
        cost = np.sqrt(np.sum(np.abs(diff) ** 2, axis=0))
    elif metric == "absolute":
        cost = np.sum(np.abs(diff), axis=0)
    elif metric == "squared":
        cost = np.sum(np.abs(diff) ** 2, axis=0)
    else:
        raise MatlabError("dtw: the metric must be 'euclidean', 'absolute' or 'squared'.")
    m, n = cost.shape
    D = np.full((m, n), np.inf)
    for i in range(m):
        for j in range(n):
            if maxsamp is not None and abs(i - j) > maxsamp:
                continue
            if i == 0 and j == 0:
                D[i, j] = cost[i, j]
                continue
            best = min(D[i - 1, j] if i else np.inf, D[i, j - 1] if j else np.inf,
                       D[i - 1, j - 1] if i and j else np.inf)
            D[i, j] = cost[i, j] + best
    if not np.isfinite(D[-1, -1]):
        raise MatlabError("dtw: no warping path exists inside the given band.")
    ix, iy = [m - 1], [n - 1]
    i, j = m - 1, n - 1
    while i or j:
        choices = []
        if i and j:
            choices.append((D[i - 1, j - 1], i - 1, j - 1))
        if i:
            choices.append((D[i - 1, j], i - 1, j))
        if j:
            choices.append((D[i, j - 1], i, j - 1))
        _c, i, j = min(choices, key=lambda t: t[0])
        ix.append(i)
        iy.append(j)
    out = (float(D[-1, -1]), _col(np.array(ix[::-1], dtype=float) + 1), _col(np.array(iy[::-1], dtype=float) + 1))
    return out[:nargout] if nargout > 1 else out[0]


def _seg_costs(v: np.ndarray, stat: str):
    """A function cost(i, j) for the segment v[i:j] (j exclusive) under the chosen statistic."""
    n = v.size
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    t = np.arange(n, dtype=float)
    ct = np.concatenate([[0.0], np.cumsum(t)])
    ctt = np.concatenate([[0.0], np.cumsum(t * t)])
    ctv = np.concatenate([[0.0], np.cumsum(t * v)])
    tiny = np.finfo(float).tiny

    def sse(i, j):
        m = j - i
        return c2[j] - c2[i] - (c1[j] - c1[i]) ** 2 / m

    def cost(i, j):
        m = j - i
        if stat == "mean":
            return sse(i, j)
        if stat == "rms":
            return m * np.log(max((c2[j] - c2[i]) / m, tiny))
        if stat == "std":
            return m * np.log(max(sse(i, j) / m, tiny))
        st, stt, sv, stv = ct[j] - ct[i], ctt[j] - ctt[i], c1[j] - c1[i], ctv[j] - ctv[i]
        den = stt - st * st / m
        if den <= 0:
            return sse(i, j)
        slope = (stv - st * sv / m) / den
        return max(sse(i, j) - slope * slope * den, 0.0)

    return cost


@builtin("findchangepts")
def _findchangepts(interp, args, nargout):
    nargs(args, 1, None, "findchangepts")
    x = fnum(args[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError("findchangepts: the signal must be a real vector.")
    opts = _name_values(list(args), 1, ("Statistic", "MaxNumChanges", "MinThreshold", "MinDistance"),
                        "findchangepts")
    stat = text(opts["statistic"]).lower() if "statistic" in opts else "mean"
    if stat not in ("mean", "rms", "std", "linear"):
        raise MatlabError("findchangepts: 'Statistic' must be 'mean', 'rms', 'std' or 'linear'.")
    if "minthreshold" in opts and "maxnumchanges" in opts:
        raise MatlabError("findchangepts: give 'MaxNumChanges' or 'MinThreshold', not both.")
    v = x.flatten(order="F").astype(float)
    n = v.size
    lmin = to_int(opts["mindistance"], "MinDistance") if "mindistance" in opts else (2 if stat in ("std", "linear")
                                                                                     else 1)
    lmin = max(lmin, 1)
    cost = _seg_costs(v, stat)
    kmax = to_int(opts["maxnumchanges"], "MaxNumChanges") if "maxnumchanges" in opts else 1
    penalty = _sc(opts["minthreshold"], "MinThreshold") if "minthreshold" in opts else None
    if n < 2 * lmin:
        return (np.zeros((0, 1)), float(cost(0, n)) if n else 0.0) if nargout > 1 else np.zeros((0, 1))
    if penalty is not None:
        # optimal partitioning: every extra change must lower the cost by more than the threshold
        best = np.full(n + 1, np.inf)
        prev = np.zeros(n + 1, dtype=int)
        best[0] = -penalty
        for j in range(lmin, n + 1):
            for i in range(0, j - lmin + 1):
                if i != 0 and i < lmin:
                    continue
                c = best[i] + cost(i, j) + penalty
                if c < best[j]:
                    best[j], prev[j] = c, i
        pts = []
        j = n
        while j > 0:
            j = prev[j]
            if j > 0:
                pts.append(j)
        pts = sorted(pts)
    else:
        # dynamic programme over the number of changes; keep extra changes only while they help
        INF = np.inf
        tab = np.full((kmax + 1, n + 1), INF)
        back = np.zeros((kmax + 1, n + 1), dtype=int)
        for j in range(lmin, n + 1):
            tab[0, j] = cost(0, j)
        for k in range(1, kmax + 1):
            for j in range((k + 1) * lmin, n + 1):
                bestc, besti = INF, 0
                for i in range(k * lmin, j - lmin + 1):
                    c = tab[k - 1, i] + cost(i, j)
                    if c < bestc:
                        bestc, besti = c, i
                tab[k, j], back[k, j] = bestc, besti
        kbest = int(np.argmin(tab[:, n]))
        if "maxnumchanges" in opts and kmax > 1:
            # prefer fewer changes unless more reduce the residual noticeably
            kbest = 0
            for k in range(1, kmax + 1):
                if tab[k, n] < tab[kbest, n] - 1e-12 * max(abs(tab[0, n]), 1.0):
                    kbest = k
        pts = []
        j = n
        for k in range(kbest, 0, -1):
            j = back[k, j]
            pts.append(j)
        pts = sorted(pts)
    bounds = [0] + pts + [n]
    resid = float(sum(cost(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)))
    ipt = _like(np.array(pts, dtype=float) + 1, x) if pts else np.zeros((0, 1) if x.shape[0] != 1 else (1, 0))
    return (ipt, resid) if nargout > 1 else ipt


# -- pulse and transition metrics -------------------------------------------------------------------------

def _state_levels(v: np.ndarray, nbins: int = 100) -> tuple[float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi == lo:
        return lo, hi
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    nz = np.flatnonzero(counts)
    first, last = nz[0], nz[-1]
    span = last - first + 1
    lo_part = np.arange(first, first + max(span // 2, 1))
    hi_part = np.arange(first + span // 2, last + 1)
    lower = centres[lo_part[np.argmax(counts[lo_part])]]
    upper = centres[hi_part[np.argmax(counts[hi_part])]]
    return float(lower), float(upper)


@builtin("statelevels")
def _statelevels(interp, args, nargout):
    nargs(args, 1, 4, "statelevels")
    x = fnum(args[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError("statelevels: the signal must be a real vector.")
    nbins = to_int(args[1], "number of bins") if _given(args, 1) else 100
    lo, hi = _state_levels(x.flatten(order="F").astype(float), nbins)
    return np.array([[lo, hi]])


def _time_axis(v: np.ndarray, rest: list, name: str) -> np.ndarray:
    if _given(rest, 0):
        t = np.real(fnum(rest[0])).flatten(order="F").astype(float)
        if t.size == 1:
            return np.arange(v.size) / t[0]
        if t.size != v.size:
            raise MatlabError(f"{name}: the time vector must have the length of the signal.")
        return t
    return np.arange(1, v.size + 1, dtype=float)


def _crossings(v: np.ndarray, t: np.ndarray, level: float, rising: bool) -> np.ndarray:
    a, b = v[:-1], v[1:]
    hit = (a < level) & (b >= level) if rising else (a > level) & (b <= level)
    idx = np.flatnonzero(hit)
    frac = (level - a[idx]) / (b[idx] - a[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])


def _transition_opts(args: list, name: str):
    rest, opts = [], {}
    i = 0
    while i < len(args):
        if is_text(args[i]):
            key = text(args[i]).lower()
            if key not in ("percentreferencelevels", "statelevels", "tolerance", "region", "midpercentreferencelevel"):
                raise MatlabError(f"{name}: unknown option '{text(args[i])}'.")
            opts[key] = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return rest, opts


def _transitions(args: list, name: str, rising: bool):
    rest, opts = _transition_opts(list(args), name)
    x = fnum(rest[0])
    if not _isvec(x) or np.iscomplexobj(x):
        raise MatlabError(f"{name}: the signal must be a real vector.")
    v = x.flatten(order="F").astype(float)
    t = _time_axis(v, rest[1:], name)
    lo, hi = (_rvec(opts["statelevels"]) if "statelevels" in opts else _state_levels(v))
    pct = _rvec(opts["percentreferencelevels"]) if "percentreferencelevels" in opts else np.array([10.0, 90.0])
    lref = lo + (hi - lo) * pct[0] / 100
    uref = lo + (hi - lo) * pct[1] / 100
    mid = 0.5 * (lo + hi)
    mids = _crossings(v, t, mid, rising)
    lower = _crossings(v, t, lref, rising)
    upper = _crossings(v, t, uref, rising)
    dur, lt, ut = [], [], []
    for m in mids:
        if rising:
            before = lower[lower <= m]
            after = upper[upper >= m]
            if before.size and after.size:
                lt.append(before[-1])
                ut.append(after[0])
                dur.append(after[0] - before[-1])
        else:
            before = upper[upper <= m]
            after = lower[lower >= m]
            if before.size and after.size:
                ut.append(before[-1])
                lt.append(after[0])
                dur.append(after[0] - before[-1])
    return dur, lt, ut, lref, uref, mids, (lo, hi), v, t


def _transition_time(name: str, rising: bool):
    def fn(interp, args, nargout):
        nargs(args, 1, None, name)
        dur, lt, ut, lref, uref, _m, _lv, _v, _t = _transitions(args, name, rising)
        out = (_col(np.array(dur)), _col(np.array(lt)), _col(np.array(ut)), float(lref), float(uref))
        return out[:nargout] if nargout > 1 else out[0]
    builtin(name)(fn)


_transition_time("risetime", True)
_transition_time("falltime", False)


@builtin("midcross")
def _midcross(interp, args, nargout):
    nargs(args, 1, None, "midcross")
    r = _transitions(args, "midcross", True)
    f = _transitions(args, "midcross", False)
    lo, hi = r[6]
    c = np.sort(np.concatenate([r[5], f[5]]))
    return (_col(c), float(0.5 * (lo + hi))) if nargout > 1 else _col(c)


def _shoot(name: str, over: bool):
    def fn(interp, args, nargout):
        nargs(args, 1, None, name)
        rows = []
        for rising in (True, False):
            dur, lt, ut, _lref, _uref, _mids, (lo, hi), v, t = _transitions(args, name, rising)
            if hi == lo:
                continue
            for d, a, b in zip(dur, lt, ut):
                done = b if rising else a
                seg = (t > done) & (t <= done + 3 * d)      # the stretch just after the transition
                if not np.any(seg):
                    continue
                ref = hi if rising else lo
                if rising == over:
                    i = int(np.argmax(np.where(seg, v, -np.inf)))
                    amount = 100 * (v[i] - ref) / (hi - lo)
                else:
                    i = int(np.argmin(np.where(seg, v, np.inf)))
                    amount = 100 * (ref - v[i]) / (hi - lo)
                rows.append((done, max(amount, 0.0), v[i], t[i]))
        rows.sort(key=lambda r: r[0])
        out = tuple(_col(np.array([r[k] for r in rows])) for k in (1, 2, 3))
        return out[:nargout] if nargout > 1 else out[0]
    builtin(name)(fn)


_shoot("overshoot", True)
_shoot("undershoot", False)
