"""Signal processing: FFTs, filter design, windows, spectra, peaks and resampling."""

from __future__ import annotations

import numpy as np

from .. import fastnum
from .._lazy import lazy_module
from ..errors import MatlabError
from ..values import simplify, to_int
from ._registry import builtin
from ._util import default_dim, fnum, is_text, nargs, text

ss = lazy_module("scipy.signal", globals(), "ss")  # imported on first use: keeps start-up fast


# -- helpers --------------------------------------------------------------------------

def _given(args: list, i: int) -> bool:
    """True when argument i exists and is not the empty placeholder []."""
    return len(args) > i and not (isinstance(args[i], np.ndarray) and args[i].size == 0 and not is_text(args[i]))


def _scalar(v, what="argument") -> float:
    a = fnum(v)
    if a.size != 1:
        raise MatlabError(f"Expected {what} to be a scalar.")
    return float(np.real(a.flat[0]))


def _is_vec(a: np.ndarray) -> bool:
    return a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1)


def _axis_of(a: np.ndarray) -> int:
    return default_dim(a)


def _like(r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Give a 1-D result the orientation of vector x (column for matrices)."""
    r = np.asarray(r)
    if r.ndim == 1:
        return r.reshape(1, -1) if x.shape[0] == 1 and x.ndim == 2 else r.reshape(-1, 1)
    return r


def _col(v) -> np.ndarray:
    return np.asarray(v).reshape(-1, 1)


def _row(v) -> np.ndarray:
    return np.asarray(v).reshape(1, -1)


def _real_if_tiny(r: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(r) and r.size:
        scale = max(np.max(np.abs(r)), 1e-300)
        if np.max(np.abs(r.imag)) <= 64 * np.finfo(float).eps * scale:
            return r.real.copy()
    return r


def _options(args: list, start: int) -> dict[str, object]:
    opts: dict[str, object] = {}
    k = start
    while k < len(args) - 1:
        if not is_text(args[k]):
            raise MatlabError("Expected a name-value option name.")
        opts[text(args[k]).lower()] = args[k + 1]
        k += 2
    return opts


# -- FFT ----------------------------------------------------------------------------------

def _fft_common(args, nargout, inverse: bool, name: str):
    nargs(args, 1, 4, name)
    rest = list(args)
    symmetric = False
    if rest and is_text(rest[-1]):
        flag = text(rest.pop()).lower()
        if flag == "symmetric":
            symmetric = True
        elif flag != "nonsymmetric":
            raise MatlabError(f"{name}: unknown option '{flag}'.")
    x = fnum(rest[0])
    if x.size == 0 and not _given(rest, 1):
        return x.astype(complex)
    axis = to_int(rest[2], "dimension") - 1 if _given(rest, 2) else _axis_of(x)
    if axis < 0:
        raise MatlabError("Dimension argument must be a positive integer.")
    n = to_int(rest[1], "transform length") if _given(rest, 1) else None
    if n is not None and n < 0:
        raise MatlabError(f"{name}: the transform length must be nonnegative.")
    single = x.dtype in (np.float32, np.complex64)
    if axis >= x.ndim:
        if n is None or n == 1:
            return x  # a dimension of length one transforms to itself
        x = x.reshape(x.shape + (1,) * (axis + 1 - x.ndim))
    if inverse:
        r = fastnum.fft(x, n, axis, inverse=True)
        r = r.real.copy() if symmetric else _real_if_tiny(r)
    else:
        r = simplify(fastnum.fft(x, n, axis))
    if single:
        r = r.astype(np.complex64 if np.iscomplexobj(r) else np.float32)
    return r


@builtin("fft")
def _fft(interp, args, nargout):
    return _fft_common(args, nargout, False, "fft")


@builtin("ifft")
def _ifft(interp, args, nargout):
    return _fft_common(args, nargout, True, "ifft")


def _fft2_common(args, inverse: bool, name: str):
    rest = list(args)
    symmetric = bool(rest) and is_text(rest[-1]) and text(rest.pop()).lower() == "symmetric"
    nargs(rest, 1, 3, name)
    x = fnum(rest[0])
    s = None
    if len(rest) == 3:
        s = (to_int(rest[1]), to_int(rest[2]))
    elif len(rest) == 2:
        v = fnum(rest[1]).flatten()
        s = (int(v[0]), int(v[1]))
    if inverse:
        r = np.fft.ifft2(x, s=s)
        return r.real.copy() if symmetric else _real_if_tiny(r)
    return simplify(np.fft.fft2(x, s=s))


@builtin("fft2")
def _fft2(interp, args, nargout):
    return _fft2_common(args, False, "fft2")


@builtin("ifft2")
def _ifft2(interp, args, nargout):
    return _fft2_common(args, True, "ifft2")


def _shift(fn):
    def f(interp, args, nargout):
        nargs(args, 1, 2, "fftshift")
        x = args[0]
        a = x if isinstance(x, np.ndarray) else fnum(x)
        if len(args) > 1:
            d = to_int(args[1], "dimension") - 1
            return fn(a, axes=d) if d < 2 else a
        return fn(a)
    return f


builtin("fftshift")(_shift(np.fft.fftshift))
builtin("ifftshift")(_shift(np.fft.ifftshift))


@builtin("nextpow2")
def _nextpow2(interp, args, nargout):
    nargs(args, 1, 1, "nextpow2")
    a = np.abs(fnum(args[0])).astype(float)
    mant, ex = np.frexp(a)
    p = np.where(mant == 0.5, ex - 1, ex).astype(float)
    p = np.where(a == 0, 0.0, p)
    p = np.where(np.isinf(a), np.inf, p)
    return p


# -- filtering -----------------------------------------------------------------------------

@builtin("filtfilt")
def _filtfilt(interp, args, nargout):
    nargs(args, 3, 3, "filtfilt")
    b, a, x = fnum(args[0]).flatten(), fnum(args[1]).flatten(), fnum(args[2])
    axis = 1 if x.shape[0] == 1 else 0
    nfilt = max(b.size, a.size)
    padlen = 3 * (nfilt - 1)
    if x.shape[axis] <= padlen:
        raise MatlabError(f"filtfilt: data must have length greater than {padlen}.")
    return ss.filtfilt(b, a, x, axis=axis, padtype="odd", padlen=padlen)


@builtin("butter")
def _butter(interp, args, nargout):
    from .signal_more import iir_design
    return iir_design("butter", args, nargout)


@builtin("cheby1")
def _cheby1(interp, args, nargout):
    from .signal_more import iir_design
    return iir_design("cheby1", args, nargout)


@builtin("freqz")
def _freqz(interp, args, nargout):
    from .signal_more import freqz_impl
    return freqz_impl(interp, args, nargout)


# -- windows ----------------------------------------------------------------------------

def _win_len(args, name) -> tuple[int, bool]:
    nargs(args, 1, 3, name)
    n = to_int(args[0], "window length")
    if n < 0:
        raise MatlabError(f"{name}: window length must be non-negative.")
    periodic = any(is_text(a) and text(a).lower() == "periodic" for a in args[1:])
    return n, periodic


def _window(maker):
    def f(interp, args, nargout):
        n, periodic = _win_len(args, "window")
        if n == 0:
            return np.zeros((0, 1))
        if n == 1:
            return np.ones((1, 1))
        return _col(maker(n, not periodic))
    return f


builtin("hamming")(_window(lambda n, sym: ss.windows.hamming(n, sym=sym)))
builtin("hann")(_window(lambda n, sym: ss.windows.hann(n, sym=sym)))
builtin("blackman")(_window(lambda n, sym: np.clip(ss.windows.blackman(n, sym=sym), 0, None)))


@builtin("hanning")
def _hanning(interp, args, nargout):
    n, periodic = _win_len(args, "hanning")
    if n == 0:
        return np.zeros((0, 1))
    if periodic:
        return _col(ss.windows.hann(n + 1, sym=True)[1:])
    return _col(ss.windows.hann(n + 2, sym=True)[1:-1])


@builtin("rectwin")
def _rectwin(interp, args, nargout):
    n, _ = _win_len(args, "rectwin")
    return np.ones((n, 1))


@builtin("kaiser")
def _kaiser(interp, args, nargout):
    nargs(args, 1, 2, "kaiser")
    n = to_int(args[0], "window length")
    beta = _scalar(args[1]) if len(args) > 1 else 0.5
    if n == 0:
        return np.zeros((0, 1))
    return _col(np.kaiser(n, beta))


# -- correlation and spectra -------------------------------------------------------------

@builtin("xcorr")
def _xcorr(interp, args, nargout):
    from .signal_more import xcorr_impl
    return xcorr_impl(list(args), nargout)


@builtin("periodogram")
def _periodogram(interp, args, nargout):
    from .signal_more import psd_impl
    return psd_impl(interp, list(args), nargout, "periodogram")


@builtin("pwelch")
def _pwelch(interp, args, nargout):
    from .signal_more import psd_impl
    return psd_impl(interp, list(args), nargout, "pwelch")


# -- peaks -----------------------------------------------------------------------------

@builtin("findpeaks")
def _findpeaks(interp, args, nargout):
    from .signal_more import findpeaks_impl
    return findpeaks_impl(interp, list(args), nargout)


# -- resampling -----------------------------------------------------------------------

def _along(x: np.ndarray):
    return 1 if x.shape[0] == 1 else 0


@builtin("resample")
def _resample(interp, args, nargout):
    from .signal_more import resample_impl
    return resample_impl(interp, list(args), nargout)


@builtin("decimate")
def _decimate(interp, args, nargout):
    from .signal_more import decimate_impl
    return decimate_impl(list(args), nargout)


@builtin("interpft")
def _interpft(interp, args, nargout):
    nargs(args, 2, 3, "interpft")
    x = fnum(args[0])
    ny = to_int(args[1], "length")
    axis = to_int(args[2]) - 1 if len(args) > 2 else _axis_of(x)
    if ny <= 0:
        shape = list(x.shape)
        shape[axis] = 0
        return np.zeros(shape)
    m = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    xm = np.moveaxis(X, axis, 0)
    if ny >= m:
        nyq = (m + 1) // 2
        Y = np.zeros((ny,) + xm.shape[1:], dtype=complex)
        Y[:nyq] = xm[:nyq]
        Y[ny - (m - nyq):] = xm[nyq:]
        if m % 2 == 0:
            # split the Nyquist bin between the positive and negative halves
            Y[nyq] = xm[nyq] / 2
            Y[ny - (m - nyq)] = xm[nyq] / 2
    else:
        keep = (ny + 1) // 2
        Y = np.concatenate([xm[:keep], xm[m - (ny - keep):]], axis=0)
    y = np.fft.ifft(Y, axis=0) * (ny / m)
    y = np.moveaxis(y, 0, axis)
    return y.real.copy() if np.isrealobj(x) else simplify(y)


# -- waveforms -----------------------------------------------------------------------

@builtin("sinc")
def _sinc(interp, args, nargout):
    nargs(args, 1, 1, "sinc")
    return np.sinc(fnum(args[0]))


@builtin("sawtooth")
def _sawtooth(interp, args, nargout):
    nargs(args, 1, 2, "sawtooth")
    w = _scalar(args[1]) if len(args) > 1 else 1.0
    if not 0 <= w <= 1:
        raise MatlabError("sawtooth: width must be between 0 and 1.")
    return ss.sawtooth(fnum(args[0]), w)


@builtin("square")
def _square(interp, args, nargout):
    nargs(args, 1, 2, "square")
    duty = _scalar(args[1]) if len(args) > 1 else 50.0
    if not 0 <= duty <= 100:
        raise MatlabError("square: duty cycle must be between 0 and 100.")
    t = fnum(args[0])
    # +1 while the phase (mod 2*pi) is below duty% of the period, -1 otherwise
    ph = np.mod(t, 2 * np.pi)
    return np.where(ph < duty / 100 * 2 * np.pi, 1.0, -1.0)


@builtin("chirp")
def _chirp(interp, args, nargout):
    nargs(args, 1, 6, "chirp")
    t = fnum(args[0])
    f0 = _scalar(args[1]) if _given(args, 1) else 0.0
    t1 = _scalar(args[2]) if _given(args, 2) else 1.0
    f1 = _scalar(args[3]) if _given(args, 3) else 100.0
    method = text(args[4]).lower() if len(args) > 4 else "linear"
    phi = _scalar(args[5]) if len(args) > 5 else 0.0
    if method not in ("linear", "quadratic", "logarithmic"):
        raise MatlabError(f"chirp: unsupported method '{method}'.")
    if method == "logarithmic" and f0 <= 0:
        raise MatlabError("chirp: logarithmic sweeps need a positive start frequency.")
    return ss.chirp(t, f0, t1, f1, method=method, phi=phi)


# -- smoothing and cleanup ---------------------------------------------------------------

@builtin("medfilt1")
def _medfilt1(interp, args, nargout):
    nargs(args, 1, 5, "medfilt1")
    rest = list(args)
    mode = "zeropad"
    nanflag = "includenan"
    while rest and is_text(rest[-1]):
        s = text(rest.pop()).lower()
        if s in ("zeropad", "truncate"):
            mode = s
        elif s in ("omitnan", "includenan"):
            nanflag = s
    x = fnum(rest[0])
    n = to_int(rest[1], "order") if _given(rest, 1) else 3
    axis = to_int(rest[3]) - 1 if _given(rest, 3) else _along(x)
    xm = np.moveaxis(x.astype(float), axis, 0)
    m = xm.shape[0]
    before = (n - 1) // 2 if n % 2 else n // 2
    after = n - 1 - before
    out = np.empty_like(xm)
    med = np.nanmedian if nanflag == "omitnan" else np.median
    for k in range(m):
        lo, hi = k - before, k + after
        w = xm[max(lo, 0):min(hi, m - 1) + 1]
        if mode == "zeropad":
            pad = max(0, -lo) + max(0, hi - (m - 1))
            if pad:
                w = np.concatenate([w, np.zeros((pad,) + w.shape[1:])], axis=0)
        out[k] = med(w, axis=0)
    return np.moveaxis(out, 0, axis)


@builtin("sgolayfilt")
def _sgolayfilt(interp, args, nargout):
    nargs(args, 3, 5, "sgolayfilt")
    x = fnum(args[0])
    order = to_int(args[1], "order")
    framelen = to_int(args[2], "frame length")
    if framelen % 2 == 0:
        raise MatlabError("sgolayfilt: frame length must be odd.")
    if order >= framelen:
        raise MatlabError("sgolayfilt: order must be less than the frame length.")
    axis = to_int(args[4]) - 1 if len(args) > 4 else _along(x)
    if x.shape[axis] < framelen:
        raise MatlabError("sgolayfilt: data must be at least as long as the frame.")
    return ss.savgol_filter(x, framelen, order, axis=axis, mode="interp")


@builtin("detrend")
def _detrend(interp, args, nargout):
    nargs(args, 1, 2, "detrend")
    x = fnum(args[0])
    deg = 1
    if len(args) > 1:
        if is_text(args[1]):
            s = text(args[1]).lower()
            deg = {"constant": 0, "linear": 1}.get(s)
            if deg is None:
                raise MatlabError(f"detrend: unknown option '{s}'.")
        else:
            deg = to_int(args[1], "polynomial degree")
    xm = x.reshape(-1, 1) if _is_vec(x) else x
    m = xm.shape[0]
    if m == 0:
        return x
    t = np.linspace(-1, 1, m) if m > 1 else np.zeros(1)
    V = np.vander(t, deg + 1)
    coef, *_ = np.linalg.lstsq(V, xm, rcond=None)
    r = xm - V @ coef
    return r.reshape(x.shape)


@builtin("unwrap")
def _unwrap(interp, args, nargout):
    nargs(args, 1, 3, "unwrap")
    p = fnum(args[0])
    tol = _scalar(args[1]) if _given(args, 1) else np.pi
    axis = to_int(args[2]) - 1 if _given(args, 2) else _axis_of(p)
    if p.size == 0:
        return p
    return np.unwrap(p, discont=tol, axis=axis)


builtin("db2mag")(lambda interp, args, nargout: 10.0 ** (fnum(args[0]) / 20))
builtin("db2pow")(lambda interp, args, nargout: 10.0 ** (fnum(args[0]) / 10))


@builtin("mag2db")
def _mag2db(interp, args, nargout):
    a = fnum(args[0]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 20 * np.log10(a)
    return np.where(a < 0, np.nan, r)


@builtin("pow2db")
def _pow2db(interp, args, nargout):
    a = fnum(args[0]).astype(float)
    if np.any(a < 0):
        raise MatlabError("pow2db: power values must be non-negative.")
    with np.errstate(divide="ignore"):
        return 10 * np.log10(a)


@builtin("hilbert")
def _hilbert(interp, args, nargout):
    nargs(args, 1, 2, "hilbert")
    x = fnum(args[0])
    if np.iscomplexobj(x):
        x = x.real
    n = to_int(args[1], "length") if len(args) > 1 else None
    axis = _along(x)
    return ss.hilbert(x, N=n, axis=axis).astype(complex)


@builtin("envelope")
def _envelope(interp, args, nargout):
    from .signal_more import envelope_impl
    return envelope_impl(interp, list(args), nargout)
