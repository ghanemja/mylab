"""Signal processing: FFTs, filter design, windows, spectra, peaks and resampling."""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

from ..errors import MatlabError
from ..values import simplify, to_int
from ._registry import builtin
from ._util import default_dim, fnum, is_text, nargs, text


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
    if axis > 1:
        return x.astype(complex) if not inverse else x
    n = to_int(rest[1], "transform length") if _given(rest, 1) else None
    if inverse:
        r = np.fft.ifft(x, n=n, axis=axis)
        return r.real.copy() if symmetric else _real_if_tiny(r)
    return np.fft.fft(x, n=n, axis=axis).astype(complex)


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
    return np.fft.fft2(x, s=s).astype(complex)


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


_BTYPES = {"low": "lowpass", "high": "highpass", "bandpass": "bandpass", "stop": "bandstop",
           "bandstop": "bandstop", "pass": "bandpass"}


def _design_out(result, nargout):
    if nargout >= 3:
        z, p, k = result
        return _col(simplify(np.asarray(z, dtype=complex))), _col(simplify(np.asarray(p, dtype=complex))), float(k)
    b, a = result
    return _row(np.real_if_close(b)), _row(np.real_if_close(a))


def _iir(args, nargout, name, nparams):
    # name(n, [params...], Wn, [ftype], ['s'])
    rest = list(args)
    analog = False
    btype = None
    while rest and is_text(rest[-1]):
        s = text(rest.pop()).lower()
        if s == "s":
            analog = True
        elif s in _BTYPES:
            btype = _BTYPES[s]
        else:
            raise MatlabError(f"{name}: unknown filter type '{s}'.")
    nargs(rest, 2 + nparams, 2 + nparams, name)
    n = to_int(rest[0], "filter order")
    params = [_scalar(v) for v in rest[1:1 + nparams]]
    wn = fnum(rest[1 + nparams]).flatten().astype(float)
    if btype is None:
        btype = "bandpass" if wn.size == 2 else "lowpass"
    if not analog and (np.any(wn <= 0) or np.any(wn >= 1)):
        raise MatlabError(f"{name}: cutoff frequencies must be between 0 and 1.")
    output = "zpk" if nargout >= 3 else "ba"
    w = wn if wn.size > 1 else float(wn[0])
    if name == "butter":
        res = ss.butter(n, w, btype=btype, analog=analog, output=output)
    else:
        res = ss.cheby1(n, params[0], w, btype=btype, analog=analog, output=output)
    return _design_out(res, nargout)


@builtin("butter")
def _butter(interp, args, nargout):
    return _iir(args, nargout, "butter", 0)


@builtin("cheby1")
def _cheby1(interp, args, nargout):
    return _iir(args, nargout, "cheby1", 1)


@builtin("freqz")
def _freqz(interp, args, nargout):
    nargs(args, 1, 5, "freqz")
    rest = list(args)
    whole = False
    if rest and is_text(rest[-1]):
        whole = text(rest.pop()).lower() == "whole"
    b = fnum(rest[0]).flatten()
    a = fnum(rest[1]).flatten() if _given(rest, 1) else np.array([1.0])
    fs = None
    worN = 512
    if _given(rest, 2):
        nv = fnum(rest[2]).flatten()
        if nv.size == 1:
            worN = int(nv[0])
        else:
            worN = nv.astype(float)
    if _given(rest, 3):
        fs = _scalar(rest[3], "sample rate")
    if isinstance(worN, np.ndarray):
        w = worN if fs is None else worN
        wr, h = ss.freqz(b, a, worN=w, fs=fs if fs is not None else 2 * np.pi)
    else:
        wr, h = ss.freqz(b, a, worN=worN, whole=whole, fs=fs if fs is not None else 2 * np.pi)
    return (_col(h.astype(complex)), _col(wr))[:max(nargout, 1)] if nargout > 1 else _col(h.astype(complex))


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
    nargs(args, 1, 4, "xcorr")
    rest = list(args)
    scale = "none"
    if rest and is_text(rest[-1]):
        scale = text(rest.pop()).lower()
    x = fnum(rest[0])
    if not _is_vec(x):
        raise MatlabError("xcorr: only vector inputs are supported.")
    y = x
    maxlag = None
    if len(rest) > 1:
        r1 = fnum(rest[1])
        if len(rest) == 2 and r1.size == 1 and x.size != 1:
            maxlag = int(r1.flat[0])
        else:
            y = r1
            if len(rest) > 2 and _given(rest, 2):
                maxlag = to_int(rest[2], "maxlag")
    xv, yv = x.flatten(), y.flatten()
    n = max(xv.size, yv.size)
    if scale in ("biased", "unbiased", "coeff") and xv.size != yv.size:
        raise MatlabError("xcorr: scaling options require vectors of equal length.")
    xp = np.concatenate([xv, np.zeros(n - xv.size)])
    yp = np.concatenate([yv, np.zeros(n - yv.size)])
    r = np.correlate(xp, yp, mode="full")
    lags = np.arange(-(n - 1), n, dtype=float)
    if scale == "biased":
        r = r / n
    elif scale == "unbiased":
        r = r / (n - np.abs(lags))
    elif scale == "coeff":
        denom = np.sqrt(np.sum(np.abs(xp) ** 2) * np.sum(np.abs(yp) ** 2))
        r = r / denom if denom else r
    elif scale != "none":
        raise MatlabError(f"xcorr: unknown scaling option '{scale}'.")
    if maxlag is not None:
        if maxlag < 0:
            raise MatlabError("xcorr: maxlag must be non-negative.")
        keep = np.abs(lags) <= maxlag
        r_full = np.zeros(2 * maxlag + 1, dtype=r.dtype)
        lag_full = np.arange(-maxlag, maxlag + 1, dtype=float)
        r_full[np.isin(lag_full, lags[keep])] = r[keep]
        r, lags = r_full, lag_full
    r = simplify(np.asarray(r))
    orient = x if x.size > 1 else y
    out = _like(r, orient)
    return (out, _like(lags, orient)) if nargout > 1 else out


def _onesided(p: np.ndarray, nfft: int, axis_len: int):
    if nfft % 2 == 0:
        keep = nfft // 2 + 1
        p = p[:keep].copy()
        p[1:-1] *= 2
    else:
        keep = (nfft + 1) // 2
        p = p[:keep].copy()
        p[1:] *= 2
    return p


@builtin("periodogram")
def _periodogram(interp, args, nargout):
    nargs(args, 1, 5, "periodogram")
    rest = list(args)
    range_opt = None
    if rest and is_text(rest[-1]):
        range_opt = text(rest.pop()).lower()
    x = fnum(rest[0])
    if not _is_vec(x):
        raise MatlabError("periodogram: only vector inputs are supported.")
    xv = x.flatten()
    n = xv.size
    w = fnum(rest[1]).flatten() if _given(rest, 1) else np.ones(n)
    if w.size != n:
        raise MatlabError("periodogram: window length must match the signal length.")
    nfft = to_int(rest[2], "nfft") if _given(rest, 2) else max(256, int(2 ** np.ceil(np.log2(n))))
    fs = _scalar(rest[3], "sample rate") if _given(rest, 3) else None
    fsv = fs if fs is not None else 2 * np.pi
    seg = xv * w
    if nfft < n:
        # wrap the data (sum segments of length nfft)
        pad = (-n) % nfft
        seg = np.concatenate([seg, np.zeros(pad)]).reshape(-1, nfft).sum(axis=0)
    X = np.fft.fft(seg, n=nfft)
    p = np.abs(X) ** 2 / (fsv * np.sum(w ** 2))
    f = np.arange(nfft) * fsv / nfft
    onesided = np.isrealobj(xv) and range_opt not in ("twosided", "centered")
    if onesided:
        p = _onesided(p, nfft, n)
        f = f[:p.size]
    elif range_opt == "centered":
        p = np.fft.fftshift(p)
        f = (np.arange(nfft) - nfft // 2) * fsv / nfft
    return (_col(p), _col(f)) if nargout > 1 else _col(p)


@builtin("pwelch")
def _pwelch(interp, args, nargout):
    nargs(args, 1, 6, "pwelch")
    rest = list(args)
    range_opt = None
    if rest and is_text(rest[-1]):
        range_opt = text(rest.pop()).lower()
    x = fnum(rest[0])
    if not _is_vec(x):
        raise MatlabError("pwelch: only vector inputs are supported.")
    xv = x.flatten()
    n = xv.size
    if _given(rest, 1):
        wv = fnum(rest[1]).flatten()
        win = ss.windows.hamming(int(wv[0]), sym=True) if wv.size == 1 else wv
    else:
        seg = int(np.trunc(n / 4.5))
        win = ss.windows.hamming(max(seg, 1), sym=True)
    L = win.size
    noverlap = to_int(rest[2], "noverlap") if _given(rest, 2) else int(np.trunc(L / 2))
    nfft = to_int(rest[3], "nfft") if _given(rest, 3) else max(256, int(2 ** np.ceil(np.log2(L))))
    fs = _scalar(rest[4], "sample rate") if _given(rest, 4) else None
    fsv = fs if fs is not None else 2 * np.pi
    step = L - noverlap
    if step <= 0:
        raise MatlabError("pwelch: noverlap must be smaller than the window length.")
    k = (n - noverlap) // step
    if k < 1:
        raise MatlabError("pwelch: signal is shorter than the window.")
    acc = np.zeros(nfft)
    U = np.sum(win ** 2)
    for i in range(k):
        seg = xv[i * step:i * step + L] * win
        acc += np.abs(np.fft.fft(seg, n=nfft)) ** 2
    p = acc / (k * fsv * U)
    f = np.arange(nfft) * fsv / nfft
    if np.isrealobj(xv) and range_opt not in ("twosided", "centered"):
        p = _onesided(p, nfft, L)
        f = f[:p.size]
    return (_col(p), _col(f)) if nargout > 1 else _col(p)


# -- peaks -----------------------------------------------------------------------------

@builtin("findpeaks")
def _findpeaks(interp, args, nargout):
    nargs(args, 1, None, "findpeaks")
    y = fnum(args[0])
    if not _is_vec(y):
        raise MatlabError("findpeaks: input must be a vector.")
    yv = np.real(y.flatten()).astype(float)
    xv = np.arange(1, yv.size + 1, dtype=float)
    start = 1
    if len(args) > 1 and not is_text(args[1]):
        xv = fnum(args[1]).flatten().astype(float)
        if xv.size == 1 and yv.size > 1:
            xv = (np.arange(yv.size)) / xv[0]
        start = 2
    opts = _options(args, start)
    # local maxima: a rise followed by a fall, plateaus reported at their left edge
    locs = []
    i = 1
    n = yv.size
    while i < n - 1:
        if yv[i] > yv[i - 1]:
            j = i
            while j < n - 1 and yv[j + 1] == yv[i]:
                j += 1
            if j < n - 1 and yv[j + 1] < yv[i]:
                locs.append(i)
            i = j + 1
        else:
            i += 1
    locs_a = np.array(locs, dtype=int)
    if "minpeakheight" in opts and locs_a.size:
        h = _scalar(opts["minpeakheight"])
        locs_a = locs_a[yv[locs_a] > h]
    if "threshold" in opts and locs_a.size:
        t = _scalar(opts["threshold"])
        keep = [(yv[k] - yv[k - 1] >= t) and (yv[k] - yv[k + 1] >= t) for k in locs_a]
        locs_a = locs_a[np.array(keep, dtype=bool)]
    if "minpeakprominence" in opts and locs_a.size:
        prom = ss.peak_prominences(yv, locs_a)[0]
        locs_a = locs_a[prom >= _scalar(opts["minpeakprominence"])]
    if "minpeakdistance" in opts and locs_a.size:
        d = _scalar(opts["minpeakdistance"])
        order = sorted(range(locs_a.size), key=lambda k: (-yv[locs_a[k]], locs_a[k]))
        alive = np.ones(locs_a.size, dtype=bool)
        for k in order:
            if not alive[k]:
                continue
            close = np.abs(xv[locs_a] - xv[locs_a[k]]) < d
            close[k] = False
            alive &= ~close
        locs_a = locs_a[alive]
    sort = text(opts["sortstr"]).lower() if "sortstr" in opts else "none"
    if sort == "descend":
        locs_a = locs_a[np.argsort(-yv[locs_a], kind="stable")]
    elif sort == "ascend":
        locs_a = locs_a[np.argsort(yv[locs_a], kind="stable")]
    if "npeaks" in opts:
        locs_a = locs_a[:to_int(opts["npeaks"], "NPeaks")]
    pks = yv[locs_a]
    loc_vals = xv[locs_a]
    shape = (1, -1) if y.shape[0] == 1 else (-1, 1)
    pk = pks.reshape(shape)
    if nargout > 1:
        return pk, loc_vals.reshape(shape)
    return pk


# -- resampling -----------------------------------------------------------------------

def _along(x: np.ndarray):
    return 1 if x.shape[0] == 1 else 0


@builtin("resample")
def _resample(interp, args, nargout):
    nargs(args, 3, 5, "resample")
    x = fnum(args[0])
    p, q = to_int(args[1], "p"), to_int(args[2], "q")
    if p <= 0 or q <= 0:
        raise MatlabError("resample: p and q must be positive integers.")
    g = np.gcd(p, q)
    p, q = p // g, q // g
    n = to_int(args[3]) if len(args) > 3 else 10
    beta = _scalar(args[4]) if len(args) > 4 else 5.0
    axis = _along(x)
    if n == 0:
        return ss.resample_poly(x, p, q, axis=axis, window=np.ones(1))
    half = n * max(p, q)
    fc = 1.0 / max(p, q)
    h = ss.firwin(2 * half + 1, fc, window=("kaiser", beta))
    return ss.resample_poly(x, p, q, axis=axis, window=h)


@builtin("decimate")
def _decimate(interp, args, nargout):
    nargs(args, 2, 4, "decimate")
    x = fnum(args[0])
    r = to_int(args[1], "decimation factor")
    ftype = "iir"
    order = None
    for a in args[2:]:
        if is_text(a):
            ftype = text(a).lower()
        else:
            order = to_int(a, "filter order")
    axis = _along(x)
    if r == 1:
        return x
    if ftype == "fir":
        return ss.decimate(x, r, n=order if order is not None else 30, ftype="fir", axis=axis, zero_phase=True)
    # Chebyshev type I low-pass (0.05 dB ripple, cutoff 0.8/r), zero-phase, then keep every r-th sample
    n = order if order is not None else 8
    b, a = ss.cheby1(n, 0.05, 0.8 / r)
    padlen = min(3 * n, x.shape[axis] - 1)
    y = ss.filtfilt(b, a, x, axis=axis, padtype="odd", padlen=padlen)
    return y[:, ::r] if axis == 1 else y[::r, :]


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
    nargs(args, 1, 3, "envelope")
    if len(args) > 2 and text(args[2]).lower() != "analytic":
        raise MatlabError("envelope: only the 'analytic' method is supported.")
    x = fnum(args[0]).astype(float)
    axis = _along(x)
    mu = np.mean(x, axis=axis, keepdims=True)
    amp = np.abs(ss.hilbert(x - mu, axis=axis))
    up, lo = mu + amp, mu - amp
    return (up, lo) if nargout > 1 else up
