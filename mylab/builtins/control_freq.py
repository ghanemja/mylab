"""Frequency-domain analysis of control-system models.

bode, bodemag, nyquist, nichols, sigma, freqresp, evalfr; margin, allmargin, bandwidth,
getPeakGain, getGainCrossover and the H2/H-infinity norms; pzmap and rlocus.

For rational SISO models the crossover frequencies, bandwidth and peak gain are roots of
polynomials in the squared frequency, so they are accurate to rounding error and do not depend
on a frequency grid. The Bode phase is built from the phase of each zero and pole, which keeps
it continuous for any choice of frequencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import MatlabError
from ..values import Cell, Struct
from . import _ctrl_num as cn
from . import _ctrl_plot as cp
from .control import (
    FRD, LTI, PID, _no_delay_copy, absorb_delay, as_scalar, as_vector, col, frd_interp, freq_unit_factor,
    is_empty_num, is_num, lti_evalfr, lti_freqresp, lti_poles, ltifn, need_lti, pid_to_tf, row, to_ss, to_zpk,
)
from .control_analysis import lti_zeros, need_siso, one_system, siso_numden, split_systems
from ._util import bool_scalar, is_text, text


# -- frequency grids --------------------------------------------------------------------------

def equivalent_roots(sys: LTI) -> np.ndarray:
    """Poles and zeros of all channels as continuous-time roots (log(z)/Ts for discrete models)."""
    if isinstance(sys, FRD):
        return np.zeros(0)
    g = to_zpk(_no_delay_copy(sys))
    parts = []
    for i in range(g.ny):
        for j in range(g.nu):
            if g.k[i, j] != 0:
                parts += [np.asarray(g.z[i][j], dtype=complex), np.asarray(g.p[i][j], dtype=complex)]
    r = np.concatenate(parts) if parts else np.zeros(0, dtype=complex)
    if sys.is_dt:
        r = r[np.abs(r) > 1e-10]
        r = np.log(r) / sys.tsv()
    return r


def auto_freqs(sys: LTI, wmin: float | None = None, wmax: float | None = None, dense: int = 30) -> np.ndarray:
    if isinstance(sys, FRD):
        w = np.sort(sys.rad_freq())
        if wmin is not None:
            w = w[(w >= wmin) & (w <= wmax)]
        return w
    roots = equivalent_roots(sys)
    mags = np.abs(roots)
    mags = mags[mags > 1e-12 * max(1.0, float(mags.max()) if mags.size else 1.0)]
    nyq = np.pi / sys.tsv() if sys.is_dt else None
    if sys.has_delay() and not sys.is_dt:
        tau = float(np.max(sys.total_delay()))
        if tau > 0:
            mags = np.concatenate([mags, [1.0 / tau]])
    if mags.size:
        lo = 10.0 ** np.floor(np.log10(mags.min()) - 1 - 1e-9)
        hi = 10.0 ** np.ceil(np.log10(mags.max()) + 1 + 1e-9)
    else:
        lo, hi = (1.0, 10.0) if nyq is None else (nyq / 100.0, nyq)
    if nyq is not None:
        hi = nyq
        if lo >= nyq:
            lo = nyq / 100.0
        lo = min(lo, nyq / 100.0)
    if wmin is not None:
        lo = wmin
    if wmax is not None:
        hi = wmax
    if not (hi > lo > 0):
        raise MatlabError("The frequency range must satisfy 0 < wmin < wmax.")
    n = int(max(50, np.ceil(dense * np.log10(hi / lo)))) + 1
    w = list(np.logspace(np.log10(lo), np.log10(hi), n))
    for r in roots:
        w0 = abs(r)
        if w0 <= 0:
            continue
        zeta = abs(r.real) / w0
        if zeta < 0.5 and lo <= w0 <= hi:
            z = max(zeta, 1e-5)
            for f in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
                w += [w0 * (1 - f * z), w0 * (1 + f * z)]
            w.append(w0)
            if zeta < 0.7:
                w.append(w0 * np.sqrt(max(1 - 2 * zeta * zeta, 0.0)))
    w = np.unique(np.array([x for x in w if lo <= x <= hi]))
    return w


def parse_freq(arg: Any, sys: LTI, name: str) -> np.ndarray:
    if isinstance(arg, Cell):
        vals = arg.values_f()
        if len(vals) != 2:
            raise MatlabError(f"{name}: a frequency range is given as {{wmin, wmax}}.")
        lo, hi = as_scalar(vals[0], "wmin"), as_scalar(vals[1], "wmax")
        return auto_freqs(sys, lo, hi)
    w = as_vector(arg, "frequency vector").real.astype(float)
    if w.size == 0:
        return auto_freqs(sys)
    if np.any(w < 0) or np.any(np.isnan(w)):
        raise MatlabError(f"{name}: frequencies must be real and not negative.")
    return w


# -- phase ------------------------------------------------------------------------------------

def analytic_phase(z, p, k, w: np.ndarray, ts: float) -> np.ndarray:
    """Continuous phase in degrees of k*prod(s - z)/prod(s - p) along the frequency axis."""
    z, p = np.asarray(z, dtype=complex), np.asarray(p, dtype=complex)
    w = np.asarray(w, dtype=float)
    if k == 0:
        return np.zeros(w.shape)
    disc = ts != 0
    tsv = ts if ts > 0 else 1.0
    origin = 1.0 if disc else 0.0
    scale = max([1.0] + [abs(v) for v in np.concatenate([z, p])])
    tol = 1e-9 * scale

    def at_origin(r):
        return np.abs(r - origin) <= tol

    z0, p0 = at_origin(z), at_origin(p)
    m = int(np.sum(z0)) - int(np.sum(p0))
    k0 = complex(k)
    for v in z[~z0]:
        k0 *= (origin - v)
    for v in p[~p0]:
        k0 /= (origin - v)
    real_sys = abs(k0.imag) <= 1e-8 * max(abs(k0), 1e-300)
    phi0 = (0.0 if k0.real >= 0 else 180.0) if real_sys else float(np.degrees(np.angle(k0)))
    total = np.full(w.shape, phi0 + 90.0 * m)
    if disc:
        theta = w * tsv

        def fac(r, th):
            rho, alpha = abs(r), float(np.angle(r))
            if rho == 0:
                return th
            beta = (th - alpha) / 2
            if rho <= 1:
                return (th + alpha) / 2 + np.arctan2((1 + rho) * np.sin(beta), (1 - rho) * np.cos(beta) + 0.0)
            return (th + alpha) / 2 + np.pi + np.arctan2(-(1 + rho) * np.sin(beta), (rho - 1) * np.cos(beta))

        total = total + np.degrees(theta / 2) * m
        for roots, sign in ((z[~z0], 1.0), (p[~p0], -1.0)):
            for r in roots:
                total = total + sign * np.degrees(fac(r, theta) - fac(r, 0.0))
        return total
    for roots, sign in ((z[~z0], 1.0), (p[~p0], -1.0)):
        for r in roots:
            x = -r.real + 0.0
            total = total + sign * np.degrees(np.arctan2(w - r.imag, x) - np.arctan2(0.0 - r.imag, x))
    return total


def _unwrap_deg(h: np.ndarray) -> np.ndarray:
    ang = np.angle(h)
    ok = np.isfinite(ang)
    out = np.full(h.shape, np.nan)
    if np.any(ok):
        out[ok] = np.degrees(np.unwrap(ang[ok]))
    return out


def bode_data(sys: LTI, w: np.ndarray):
    """Gain and phase (degrees), each (ny, nu, nw), at the frequencies w."""
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    if isinstance(sys, FRD):
        h = frd_interp(sys, w)
        mag = np.abs(h)
        phase = np.empty(h.shape)
        order = np.argsort(w)
        for i in range(sys.ny):
            for j in range(sys.nu):
                ph = np.empty(w.size)
                ph[order] = _unwrap_deg(h[i, j][order])
                phase[i, j] = ph
        return mag, phase
    base = _no_delay_copy(sys)
    with np.errstate(all="ignore"):
        h = lti_freqresp(base, w)
    mag = np.abs(h)
    g = to_zpk(base)
    phase = np.empty(h.shape)
    total = sys.total_delay()
    for i in range(sys.ny):
        for j in range(sys.nu):
            ph = analytic_phase(g.z[i][j], g.p[i][j], g.k[i, j], w, sys.ts)
            if total[i, j]:
                ph = ph - np.degrees(w * total[i, j] * (sys.tsv() if sys.is_dt else 1.0))
            phase[i, j] = ph
    return mag, phase


# -- response functions -----------------------------------------------------------------------

def _freq_args(rest: list, sys: LTI, name: str) -> np.ndarray:
    if len(rest) > 1:
        raise MatlabError(f"Too many arguments for {name}; use {name}(sys) or {name}(sys, w).")
    if rest:
        if not (is_num(rest[0]) or isinstance(rest[0], Cell)):
            raise MatlabError(f"{name}: the second argument must be a frequency vector or a range {{wmin, wmax}}.")
        return parse_freq(rest[0], sys, name)
    return auto_freqs(sys)


def _bode_like(interp, args, nargout, name: str):
    systems, styles, rest = split_systems(args, name)
    one_system(systems, nargout, name)
    items = []
    for sys, style in zip(systems, styles):
        w = _freq_args(rest, sys, name)
        mag, phase = bode_data(sys, w)
        if nargout > 0:
            return (mag, phase, col(w), np.zeros((0, 0)), np.zeros((0, 0)))[:max(nargout, 1)]
        items.append({"w": w, "mag": mag, "phase": phase, "style": style})
    return systems, items


@ltifn("bode", "bodeplot")
def _bode(interp, args, nargout):
    res = _bode_like(interp, args, nargout, "bode")
    if nargout > 0:
        return res
    systems, items = res
    first = systems[0]
    _same_size(systems, "bode")
    cp.bode_plot(interp, items, first.in_name, first.out_name, first.time_unit)
    return None


@ltifn("bodemag")
def _bodemag(interp, args, nargout):
    if nargout > 0:
        raise MatlabError("bodemag only draws the magnitude plot; use [mag, phase, w] = bode(sys) for data.")
    systems, items = _bode_like(interp, args, 0, "bodemag")
    first = systems[0]
    _same_size(systems, "bodemag")
    cp.bode_plot(interp, items, first.in_name, first.out_name, first.time_unit, mag_only=True)
    return None


def _same_size(systems: list, name: str) -> None:
    first = systems[0]
    if any((s.ny, s.nu) != (first.ny, first.nu) for s in systems):
        raise MatlabError(f"{name}: all models plotted together must have the same numbers of inputs and outputs.")


@ltifn("nichols", "nicholsplot")
def _nichols(interp, args, nargout):
    res = _bode_like(interp, args, nargout, "nichols")
    if nargout > 0:
        return res[:3]
    systems, items = res
    for s in systems:
        need_siso(s, "nichols")
    cp.nichols_plot(interp, items)
    return None


@ltifn("nyquist", "nyquistplot")
def _nyquist(interp, args, nargout):
    systems, styles, rest = split_systems(args, "nyquist")
    one_system(systems, nargout, "nyquist")
    items = []
    for sys, style in zip(systems, styles):
        if rest:
            w = _freq_args(rest, sys, "nyquist")
        else:
            w = auto_freqs(sys, dense=80)
            if not isinstance(sys, FRD):
                lo, hi = w[0], w[-1]
                extra_lo = np.logspace(np.log10(lo) - 2, np.log10(lo), 40)[:-1]
                extra_hi = np.logspace(np.log10(hi), np.log10(hi) + 2, 40)[1:] if not sys.is_dt else np.zeros(0)
                w = np.concatenate([extra_lo, w, extra_hi])
        with np.errstate(all="ignore"):
            h = lti_freqresp(sys, w)
        if nargout > 0:
            return (h.real.copy(), h.imag.copy(), col(w))[:max(nargout, 1)]
        items.append({"re": h.real, "im": h.imag, "style": style})
    _same_size(systems, "nyquist")
    cp.nyquist_plot(interp, items, systems[0].in_name, systems[0].out_name)
    return None


@ltifn("sigma", "sigmaplot")
def _sigma(interp, args, nargout):
    systems, styles, rest = split_systems(args, "sigma")
    one_system(systems, nargout, "sigma")
    kind = 0
    if len(rest) == 2:
        kind = int(as_scalar(rest[1], "type"))
        if kind not in (0, 1, 2, 3):
            raise MatlabError("sigma: the type must be 0 (sys), 1 (inverse), 2 (I + sys) or 3 (I + inverse).")
        rest = rest[:1]
    items = []
    for sys, style in zip(systems, styles):
        w = _freq_args([r for r in rest if not is_empty_num(r)], sys, "sigma")
        with np.errstate(all="ignore"):
            h = lti_freqresp(sys, w)
        if kind and sys.ny != sys.nu:
            raise MatlabError("sigma: types 1 to 3 need a square model.")
        nsv = min(sys.ny, sys.nu)
        sv = np.zeros((nsv, w.size))
        eye = np.eye(sys.ny)
        for k in range(w.size):
            m = h[:, :, k]
            if not np.all(np.isfinite(m)):
                sv[:, k] = np.inf
                continue
            if kind == 1:
                m = np.linalg.inv(m)
            elif kind == 2:
                m = eye + m
            elif kind == 3:
                m = eye + np.linalg.inv(m)
            sv[:, k] = np.linalg.svd(m, compute_uv=False)
        if nargout > 0:
            return (sv, col(w))[:max(nargout, 1)]
        items.append({"w": w, "sv": sv, "style": style})
    cp.sigma_plot(interp, items, systems[0].time_unit)
    return None


@ltifn("freqresp")
def _freqresp(interp, args, nargout):
    sys = need_lti(args, "freqresp")
    factor = 1.0
    rest = list(args[1:])
    if rest and is_text(rest[-1]):
        factor = freq_unit_factor(text(rest[-1]))
        rest = rest[:-1]
    if len(rest) > 1:
        raise MatlabError("Use freqresp(sys, w) or freqresp(sys, w, units).")
    if rest and not is_empty_num(rest[0]):
        w = as_vector(rest[0], "frequency vector").real.astype(float)
        wr = w * factor
    else:
        wr = auto_freqs(sys)
        w = wr / factor
    with np.errstate(all="ignore"):
        h = lti_freqresp(sys, wr)
    if nargout > 1:
        return h, col(w)
    return h


@ltifn("evalfr")
def _evalfr(interp, args, nargout):
    sys = need_lti(args, "evalfr")
    if len(args) != 2:
        raise MatlabError("Use evalfr(sys, x) with one complex number x.")
    x = np.asarray(args[1])
    if x.size != 1:
        raise MatlabError("evalfr evaluates the model at a single complex number; use freqresp for a vector "
                          "of frequencies.")
    with np.errstate(all="ignore"):
        h = lti_evalfr(sys, [complex(x.flat[0])])[:, :, 0]
    return h.real.copy() if not np.any(h.imag) else h


# -- loop margins -----------------------------------------------------------------------------

class Loop:
    """A SISO open loop prepared for exact crossover computations."""

    def __init__(self, sys: LTI, name: str):
        need_siso(sys, name)
        if isinstance(sys, PID):
            sys = pid_to_tf(sys)
        self.disc = sys.is_dt
        self.ts = sys.tsv()
        self.tau = 0.0
        if sys.has_delay():
            if sys.is_dt:
                sys = absorb_delay(sys)
            else:
                self.tau = float(sys.total_delay()[0, 0])
                sys = _no_delay_copy(sys)
        self.sys = sys
        num, den = siso_numden(sys)
        if np.iscomplexobj(num) or np.iscomplexobj(den):
            raise MatlabError(f"{name} needs a model with real coefficients.")
        self.num, self.den = num, den
        self.wn, self.wd = cn.wplane(num, den) if self.disc else (num, den)

    def freq(self, v: float) -> float:
        """Frequency in rad/TimeUnit of a point of the transformed axis."""
        return 2 * np.arctan(v) / self.ts if self.disc else v

    def value(self, w: float) -> complex:
        s = np.exp(1j * w * self.ts) if self.disc else 1j * w
        with np.errstate(all="ignore"):
            h = complex(cn.poly_freqresp(self.num, self.den, np.array([s]))[0])
        if self.tau:
            h *= np.exp(-1j * w * self.tau)
        return h

    def gain_crossings(self, level: float = 1.0) -> np.ndarray:
        """Frequencies where the gain equals ``level`` (including 0 when the DC gain equals it)."""
        px = np.polysub(cn.abs2_x(self.wn), level ** 2 * cn.abs2_x(self.wd))
        px = np.atleast_1d(px)
        out = []
        zero_root = px.size > 0 and px[-1] == 0 and np.any(px != 0)
        if zero_root:
            out.append(0.0)
            px = np.trim_zeros(px, "b")
        for x in cn.positive_roots(px):
            out.append(self.freq(float(np.sqrt(x))))
        if self.disc:   # the Nyquist frequency is the point at infinity of the transformed axis
            with np.errstate(all="ignore"):
                g = abs(self.value(np.pi / self.ts))
            if np.isfinite(g) and abs(g - level) <= 1e-12 * max(level, 1e-300):
                out.append(np.pi / self.ts)
        return np.array(sorted(out))

    def phase_crossings(self):
        """Frequencies where the response is real and negative, with the gain there."""
        freqs, gains = [], []
        nr, ni = cn.jw_parts(self.wn)
        dr, di = cn.jw_parts(self.wd)
        if self.tau == 0:
            q = np.polysub(np.convolve(ni, dr), np.convolve(nr, di))       # Im(n * conj(d)), odd in w
            re = np.polyadd(np.convolve(nr, dr), np.convolve(ni, di))      # Re(n * conj(d))
            d0 = np.polyval(self.wd, 0.0)
            if d0 != 0:
                g0 = np.polyval(self.wn, 0.0) / d0
                if g0 < 0:
                    freqs.append(0.0)
                    gains.append(abs(g0))
            for x in cn.positive_roots(cn.odd_part(q)):
                v = float(np.sqrt(x))
                if np.polyval(re, v) < 0:
                    w = self.freq(v)
                    freqs.append(w)
                    gains.append(abs(self.value(w)))
            # the high-frequency end
            if self.disc:
                with np.errstate(all="ignore"):
                    h = self.value(np.pi / self.ts)
                if np.isfinite(h) and abs(h.imag) <= 1e-9 * max(abs(h), 1e-300) and h.real < 0:
                    freqs.append(np.pi / self.ts)
                    gains.append(abs(h))
            else:
                n, d = cn.ptrim(self.num), cn.ptrim(self.den)
                rel = (d.size - 1) - (n.size - 1)
                lead = n[0] / d[0]
                if rel == 0 and lead < 0:
                    freqs.append(np.inf)
                    gains.append(abs(lead))
                elif rel > 0 and ((lead > 0 and rel % 4 == 2) or (lead < 0 and rel % 4 == 0)):
                    freqs.append(np.inf)
                    gains.append(0.0)
        else:
            freqs, gains = self._delay_phase_crossings()
        order = np.argsort(freqs)
        return np.array(freqs)[order], np.array(gains)[order]

    def _delay_phase_crossings(self):
        g = to_zpk(self.sys)
        z, p, k = np.asarray(g.z[0][0]), np.asarray(g.p[0][0]), g.k[0, 0]
        roots = np.abs(np.concatenate([z, p]).astype(complex))
        roots = roots[roots > 0]
        wtop = max(100.0 * (roots.max() if roots.size else 1.0), 60 * np.pi / self.tau)
        wlow = min(1e-3 * (roots.min() if roots.size else 1.0), 1e-3 / self.tau)
        grid = np.unique(np.concatenate([np.logspace(np.log10(wlow), np.log10(wtop), 3000),
                                         np.arange(0.0, wtop, (2 * np.pi / self.tau) / 60)[1:]]))

        def f(w):
            return analytic_phase(z, p, k, np.atleast_1d(w), 0.0) - np.degrees(np.atleast_1d(w) * self.tau)

        ph = f(grid)
        freqs, gains = [], []
        kmin, kmax = int(np.floor((ph.min() + 180) / 360)) - 1, int(np.ceil((ph.max() + 180) / 360)) + 1
        from scipy.optimize import brentq
        for n in range(kmin, kmax + 1):
            level = -180.0 + 360.0 * n
            d = ph - level
            idx = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)
            for i in idx:
                if abs(d[i] - d[i + 1]) > 170:      # a jump at an undamped root, not a crossing
                    continue
                w = brentq(lambda x: float(f(x)[0]) - level, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-13)
                freqs.append(float(w))
                gains.append(abs(self.value(w)))
                if len(freqs) >= 400:
                    return freqs, gains
        return freqs, gains

    def closed_loop_stable(self):
        num, den = self.num, self.den
        if self.tau:
            pn, pd = cn.pade_poly(self.tau, 12)
            num, den = cn.pmul(num, pn), cn.pmul(den, pd)
        roots = cn.proots(cn.padd(num, den))
        return cn.is_stable(roots, self.disc)


def _wrap180(x: float) -> float:
    return float((x + 180.0) % 360.0 - 180.0)


def allmargin_data(sys: LTI, name: str = "allmargin") -> dict:
    if isinstance(sys, FRD):
        return _frd_margins(sys, name)
    loop = Loop(sys, name)
    wg, gains = loop.phase_crossings()
    with np.errstate(divide="ignore"):
        gm = np.where(gains > 0, 1.0 / np.where(gains > 0, gains, 1.0), np.inf)
    wp = loop.gain_crossings(1.0)
    pm, dm = [], []
    for w in wp:
        h = loop.value(w)
        ph = _wrap180(float(np.degrees(np.angle(h))) + 180.0)
        pm.append(ph)
        if w > 0:
            pos = ph if ph >= 0 else ph + 360.0
            d = np.radians(pos) / w
            dm.append(d / loop.ts if loop.disc else d)
        else:
            dm.append(np.inf)
    return {"GainMargin": np.asarray(gm, dtype=float), "GMFrequency": np.asarray(wg, dtype=float),
            "PhaseMargin": np.array(pm), "PMFrequency": np.asarray(wp, dtype=float),
            "DelayMargin": np.array(dm), "DMFrequency": np.asarray(wp, dtype=float),
            "Stable": bool(loop.closed_loop_stable())}


def _interp_cross(w: np.ndarray, v: np.ndarray, level: float):
    """Frequencies where the sampled curve v(w) crosses ``level`` (linear in log frequency)."""
    out = []
    d = v - level
    for i in range(len(w) - 1):
        if d[i] == 0:
            out.append((w[i], i, 0.0))
        elif d[i] * d[i + 1] < 0 and np.isfinite(d[i]) and np.isfinite(d[i + 1]):
            f = d[i] / (d[i] - d[i + 1])
            if w[i] > 0:
                wc = float(np.exp(np.log(w[i]) + f * (np.log(w[i + 1]) - np.log(w[i]))))
            else:
                wc = float(w[i] + f * (w[i + 1] - w[i]))
            out.append((wc, i, f))
    if len(w) and d[-1] == 0:
        out.append((w[-1], len(w) - 1, 0.0))
    return out


def margins_from_data(mag: np.ndarray, phase: np.ndarray, w: np.ndarray) -> dict:
    """Margins by interpolation of sampled gain and phase (degrees)."""
    order = np.argsort(w)
    w, mag, phase = w[order], mag[order], phase[order]
    with np.errstate(divide="ignore"):
        logm = np.log(mag)
    gm, wg = [], []
    lo, hi = np.nanmin(phase), np.nanmax(phase)
    for n in range(int(np.floor((lo + 180) / 360)), int(np.ceil((hi + 180) / 360)) + 1):
        for wc, i, f in _interp_cross(w, phase, -180.0 + 360.0 * n):
            j = min(i + 1, len(w) - 1)
            g = np.exp(logm[i] + f * (logm[j] - logm[i]))
            gm.append(1.0 / g if g > 0 else np.inf)
            wg.append(wc)
    pm, wp, dm = [], [], []
    for wc, i, f in _interp_cross(w, logm, 0.0):
        j = min(i + 1, len(w) - 1)
        ph = _wrap180(phase[i] + f * (phase[j] - phase[i]) + 180.0)
        pm.append(ph)
        wp.append(wc)
        dm.append(np.radians(ph if ph >= 0 else ph + 360.0) / wc if wc > 0 else np.inf)
    og, op = np.argsort(wg), np.argsort(wp)
    return {"GainMargin": np.array(gm)[og], "GMFrequency": np.array(wg)[og], "PhaseMargin": np.array(pm)[op],
            "PMFrequency": np.array(wp)[op], "DelayMargin": np.array(dm)[op], "DMFrequency": np.array(wp)[op],
            "Stable": float("nan")}


def _frd_margins(sys: FRD, name: str) -> dict:
    need_siso(sys, name)
    w = sys.rad_freq()
    mag, phase = bode_data(sys, w)
    out = margins_from_data(mag[0, 0], phase[0, 0], w)
    if sys.is_dt:
        out["DelayMargin"] = out["DelayMargin"] / sys.tsv()
    return out


def pick_margins(m: dict):
    """The smallest gain margin (closest to 0 dB) and the smallest phase margin, with frequencies."""
    gm, wg = m["GainMargin"], m["GMFrequency"]
    if gm.size:
        with np.errstate(divide="ignore"):
            k = int(np.argmin(np.abs(np.log(np.where(gm > 0, gm, 1e-300)))))
        g, wcg = float(gm[k]), float(wg[k])
    else:
        g, wcg = np.inf, np.nan
    pm, wp = m["PhaseMargin"], m["PMFrequency"]
    if pm.size:
        k = int(np.argmin(np.abs(pm)))
        p, wcp = float(pm[k]), float(wp[k])
    else:
        p, wcp = np.inf, np.nan
    return g, p, wcg, wcp


@ltifn("allmargin")
def _allmargin(interp, args, nargout):
    if args and isinstance(args[0], LTI):
        if len(args) > 1:
            raise MatlabError("allmargin(sys) takes one model; for data use allmargin(mag, phase, w, ts).")
        m = allmargin_data(args[0])
    else:
        if len(args) < 3:
            raise MatlabError("Use allmargin(sys) or allmargin(mag, phase, w, ts).")
        mag = np.asarray(args[0], dtype=float).ravel()
        ph = np.asarray(args[1], dtype=float).ravel()
        w = as_vector(args[2], "frequency vector").real
        if not (mag.size == ph.size == w.size):
            raise MatlabError("allmargin: mag, phase and w must have the same number of points.")
        m = margins_from_data(mag, ph, w)
        if len(args) > 3:
            ts = as_scalar(args[3], "ts")
            if ts > 0:
                m["DelayMargin"] = m["DelayMargin"] / ts
    out = {k: (row(v) if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    out["Stable"] = np.array([[float(m["Stable"])]]) if not isinstance(m["Stable"], bool) else \
        bool_scalar(m["Stable"])
    return Struct.scalar(out)


def _g3(x: float) -> str:
    if np.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.3g}"


def _margin_title(gm: float, pm: float, wcg: float, wcp: float, unit: str) -> str:
    fu = cp.freq_unit_label(unit)
    if np.isnan(wcg):
        gs = "Gm = Inf"
    else:
        with np.errstate(divide="ignore"):
            gdb = 20 * np.log10(gm) if gm > 0 else -np.inf
        gs = f"Gm = {_g3(gdb)} dB (at {_g3(wcg)} {fu})"
    ps = "Pm = Inf" if np.isnan(wcp) else f"Pm = {_g3(pm)} deg (at {_g3(wcp)} {fu})"
    return f"Bode Diagram\n{gs} ,  {ps}"


@ltifn("margin")
def _margin(interp, args, nargout):
    if not args:
        raise MatlabError("margin needs a model: margin(sys), or data: margin(mag, phase, w).")
    if isinstance(args[0], LTI):
        sys = args[0]
        if len(args) > 1 and not (is_num(args[1]) or isinstance(args[1], Cell)):
            raise MatlabError("margin(sys) takes one model (and optionally frequencies for the plot).")
        m = allmargin_data(sys, "margin")
        gm, pm, wcg, wcp = pick_margins(m)
        if nargout == 0:
            w = parse_freq(args[1], sys, "margin") if len(args) > 1 else auto_freqs(sys)
            mag, phase = bode_data(sys, w)
            with np.errstate(divide="ignore"):
                gdb = 20 * np.log10(gm) if gm > 0 else -np.inf
            marks = {"wcg": wcg, "wcp": wcp, "gm_db": gdb, "pm": pm}
            if np.isfinite(wcp) and wcp > 0:
                marks["phase_at_wcp"] = float(bode_data(sys, np.array([wcp]))[1][0, 0, 0])
            cp.bode_plot(interp, [{"w": w, "mag": mag, "phase": phase, "style": None}], sys.in_name, sys.out_name,
                         sys.time_unit, title=_margin_title(gm, pm, wcg, wcp, sys.time_unit), marks=marks)
            return None
    else:
        if len(args) != 3:
            raise MatlabError("Use margin(sys) or margin(mag, phase, w).")
        mag = np.asarray(args[0], dtype=float).ravel()
        ph = np.asarray(args[1], dtype=float).ravel()
        w = as_vector(args[2], "frequency vector").real
        if not (mag.size == ph.size == w.size):
            raise MatlabError("margin: mag, phase and w must have the same number of points.")
        gm, pm, wcg, wcp = pick_margins(margins_from_data(mag, ph, w))
        if nargout == 0:
            cp.bode_plot(interp, [{"w": w, "mag": mag.reshape(1, 1, -1), "phase": ph.reshape(1, 1, -1),
                                   "style": None}], [""], [""], "seconds",
                         title=_margin_title(gm, pm, wcg, wcp, "seconds"))
            return None
    return (gm, pm, wcg, wcp)[:max(nargout, 1)]


@ltifn("bandwidth", builtin=False)
def _bandwidth(interp, args, nargout):
    sys = args[0]
    if not isinstance(sys, LTI):
        raise MatlabError("bandwidth: a model is expected.")
    drop = -3.0
    if len(args) > 1:
        drop = as_scalar(args[1], "dbdrop")
        if drop >= 0:
            raise MatlabError("bandwidth: the gain drop must be negative, e.g. -3 (dB).")
    if isinstance(sys, FRD):
        need_siso(sys, "bandwidth")
        w = np.sort(sys.rad_freq())
        mag = np.abs(frd_interp(sys, w))[0, 0]
        target = mag[0] * 10 ** (drop / 20)
        below = np.flatnonzero(mag < target)
        if below.size == 0:
            return np.inf
        k = int(below[0])
        if k == 0:
            return float(w[0])
        f = (np.log(mag[k - 1]) - np.log(target)) / (np.log(mag[k - 1]) - np.log(mag[k]))
        return float(np.exp(np.log(w[k - 1]) + f * (np.log(w[k]) - np.log(w[k - 1]))))
    loop = Loop(sys, "bandwidth")
    from .control_analysis import lti_dcgain
    g0 = abs(float(np.real(lti_dcgain(sys)[0, 0])))
    if not np.isfinite(g0) or g0 == 0:
        return np.nan
    cross = loop.gain_crossings(g0 * 10 ** (drop / 20))
    cross = cross[cross > 0]
    return float(cross[0]) if cross.size else np.inf


@ltifn("getGainCrossover")
def _get_gain_crossover(interp, args, nargout):
    sys = need_lti(args, "getGainCrossover")
    if len(args) != 2:
        raise MatlabError("Use getGainCrossover(sys, gain).")
    gain = as_scalar(args[1], "gain")
    if gain <= 0:
        raise MatlabError("getGainCrossover: the gain must be positive.")
    if isinstance(sys, FRD):
        need_siso(sys, "getGainCrossover")
        w = np.sort(sys.rad_freq())
        mag = np.abs(frd_interp(sys, w))[0, 0]
        with np.errstate(divide="ignore"):
            return col(np.array([c[0] for c in _interp_cross(w, np.log(mag), np.log(gain))]))
    return col(Loop(sys, "getGainCrossover").gain_crossings(gain))


def siso_peak_gain(sys: LTI, band: tuple | None = None):
    """Exact peak of |G| over frequency for a rational SISO model: (peak, frequency)."""
    loop = Loop(sys, "getPeakGain")
    poles = lti_poles(loop.sys)
    lo, hi = band if band is not None else (0.0, np.inf)
    if poles.size:
        if loop.disc:
            on = np.abs(np.abs(poles) - 1) <= 1e-12
            wpole = np.abs(np.angle(poles[on])) / loop.ts
        else:
            on = np.abs(poles.real) <= 1e-12 * np.maximum(np.abs(poles), 1.0)
            wpole = np.abs(poles[on].imag)
        wpole = wpole[(wpole >= lo) & (wpole <= hi)]
        if wpole.size:
            return np.inf, float(np.min(wpole))
    n2, d2 = cn.abs2_x(loop.wn), cn.abs2_x(loop.wd)
    stat = np.polysub(np.convolve(cn.pder(n2), d2), np.convolve(n2, cn.pder(d2)))
    wtop = np.pi / loop.ts if loop.disc else np.inf
    cands = [0.0, wtop] + [loop.freq(float(np.sqrt(x))) for x in cn.positive_roots(stat)]
    cands = [min(max(c, lo), min(hi, wtop)) for c in cands] + [c for c in (lo, min(hi, wtop))]
    best, wbest = -1.0, 0.0
    for wc in sorted(set(cands)):
        if np.isinf(wc):
            n, d = cn.ptrim(loop.num), cn.ptrim(loop.den)
            g = 0.0 if n.size < d.size else (abs(n[0] / d[0]) if n.size == d.size else np.inf)
        else:
            g = abs(loop.value(wc))
        if g > best * (1 + 1e-13):
            best, wbest = g, wc
    return float(best), float(wbest)


def peak_gain(sys: LTI, tol: float = 1e-9, band: tuple | None = None):
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    if isinstance(sys, FRD):
        w = sys.rad_freq()
        sv = np.array([np.linalg.norm(sys.resp[:, :, k], 2) for k in range(w.size)])
        if band is not None:
            ok = (w >= band[0]) & (w <= band[1])
            if not np.any(ok):
                return 0.0, float("nan")
            sv, w = sv[ok], w[ok]
        k = int(np.argmax(sv))
        return float(sv[k]), float(w[k])
    base = _no_delay_copy(sys)
    if base.is_siso and not any(np.iscomplexobj(v) for v in siso_numden(base)):
        return siso_peak_gain(base, band)
    from .control import lti_isproper
    if not lti_isproper(base):
        return np.inf, np.inf
    s = to_ss(base)
    if band is not None:
        w = auto_freqs(base, max(band[0], 1e-12), band[1] if np.isfinite(band[1]) else None, dense=200)
        sv = [cn.sigma_max(s.a, s.b, s.c, s.d, np.exp(1j * x * s.tsv()) if s.is_dt else 1j * x) for x in w]
        k = int(np.argmax(sv))
        return float(sv[k]), float(w[k])
    return cn.hinf_norm(s.a, s.b, s.c, s.d, s.ts, tol)


@ltifn("getPeakGain")
def _get_peak_gain(interp, args, nargout):
    sys = need_lti(args, "getPeakGain")
    tol = as_scalar(args[1], "tolerance") if len(args) > 1 and not is_empty_num(args[1]) else 0.01
    band = None
    if len(args) > 2:
        b = as_vector(args[2], "frequency band").real
        if b.size != 2 or not b[0] < b[1]:
            raise MatlabError("getPeakGain: the frequency band must be [fmin, fmax] with fmin < fmax.")
        band = (float(b[0]), float(b[1]))
    g, f = peak_gain(sys, min(tol, 1e-9), band)
    return (g, f)[:max(nargout, 1)]


@ltifn("norm", builtin=False)
def _norm(interp, args, nargout):
    sys = args[0]
    if not isinstance(sys, LTI):
        raise MatlabError("norm: when a model is given it must be the first argument.")
    kind = 2.0
    if len(args) > 1:
        if is_text(args[1]):
            t = text(args[1]).lower()
            if t not in ("inf", "fro"):
                raise MatlabError("norm of a model: the type must be 2 or Inf.")
            kind = np.inf if t == "inf" else 2.0
        else:
            kind = as_scalar(args[1], "norm type")
    if kind == 2:
        if len(args) > 2:
            raise MatlabError("norm(sys, 2) takes no further argument.")
        if isinstance(sys, FRD):
            raise MatlabError("The H2 norm of frequency-response data is not defined; use norm(sys, Inf).")
        from .control import lti_isproper
        if not lti_isproper(sys):
            return np.inf
        s = to_ss(_no_delay_copy(sys))
        return cn.h2_norm(s.a, s.b, s.c, s.d, s.is_dt)
    if not np.isinf(kind):
        raise MatlabError("norm of a model: the type must be 2 (H2 norm) or Inf (peak gain).")
    g, f = peak_gain(sys)
    return (g, f)[:max(nargout, 1)]


# -- pole-zero maps and root locus ------------------------------------------------------------

@ltifn("pzmap", "pzplot", "iopzmap")
def _pzmap(interp, args, nargout):
    systems, styles, rest = split_systems(args, "pzmap")
    if rest:
        raise MatlabError("pzmap takes only models (and line styles).")
    one_system(systems, nargout, "pzmap")
    items = []
    for sys, style in zip(systems, styles):
        p, z = lti_poles(sys), lti_zeros(sys)
        if nargout > 0:
            return (col(_tidy(p)), col(_tidy(z)))[:max(nargout, 1)]
        items.append({"p": p, "z": z, "style": style})
    cp.pz_plot(interp, items, systems[0].time_unit, systems[0].is_dt)
    return None


def _tidy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return v.real.copy() if np.iscomplexobj(v) and not np.any(v.imag) else v


def _match_columns(cols_: list) -> np.ndarray:
    """Order the roots of each gain so that each row follows one branch."""
    n = max(len(c) for c in cols_)
    out = np.full((n, len(cols_)), np.inf + 0j, dtype=complex)
    prev = None
    for k, c in enumerate(cols_):
        c = np.asarray(c, dtype=complex)
        if c.size < n:
            c = np.concatenate([c, np.full(n - c.size, np.inf + 0j)])
        if prev is None:
            order = np.argsort(-c.imag + 0 * c.real, kind="stable")
            out[:, k] = c[order]
        else:
            fin_p, fin_c = np.isfinite(prev), np.isfinite(c)
            cost = np.abs(np.where(fin_p, prev, 0)[:, None] - np.where(fin_c, c, 0)[None, :])
            big = 1e6 * (1.0 + (cost[np.isfinite(cost)].max() if cost.size else 1.0))
            cost = np.where(fin_p[:, None] ^ fin_c[None, :], big, cost)
            cost = np.where(~fin_p[:, None] & ~fin_c[None, :], 0.0, cost)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            col_ = np.empty(n, dtype=complex)
            col_[ri] = c[ci]
            out[:, k] = col_
        prev = out[:, k]
    return out


def rlocus_data(num: np.ndarray, den: np.ndarray, gains: np.ndarray | None = None):
    num, den = cn.ptrim(num), cn.ptrim(den)
    if cn.piszero(num):
        raise MatlabError("rlocus: the model is identically zero, so the gain has no effect on the poles.")
    nroots = max(num.size, den.size) - 1

    def roots_at(k: float) -> np.ndarray:
        if np.isinf(k):
            r = cn.proots(num)
        else:
            r = cn.proots(cn.padd(den, k * num))
        return np.concatenate([r, np.full(nroots - r.size, np.inf + 0j)]) if r.size < nroots else r

    if gains is not None:
        return _match_columns([roots_at(float(k)) for k in gains]), np.asarray(gains, dtype=float)
    poles, zeros = cn.proots(den), cn.proots(num)
    allr = np.concatenate([poles, zeros])
    scale = float(np.max(np.abs(allr))) if allr.size and np.max(np.abs(allr)) > 0 else 1.0
    center = float(np.mean(allr.real)) if allr.size else 0.0
    ang = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))

    def kval(s):
        with np.errstate(all="ignore"):
            v = np.abs(np.polyval(den, s) / np.polyval(num, s))
        return v[np.isfinite(v) & (v > 0)]

    lows = np.concatenate([kval(p + 1e-3 * scale * ang) for p in poles] or [np.zeros(0)])
    highs = np.concatenate([kval(center + 8 * scale * ang)] + [kval(z + 1e-2 * scale * ang) for z in zeros])
    klo = float(lows.min()) if lows.size else 1e-3
    khi = float(highs.max()) if highs.size else 1e3
    if not (khi > klo > 0):
        klo, khi = min(klo, khi, 1e-3), max(klo, khi, 1e3)
    ks = list(np.logspace(np.log10(klo), np.log10(khi), 80))
    # gains at which branches meet (repeated roots)
    brk = cn.proots(np.polysub(np.convolve(num, cn.pder(den)), np.convolve(cn.pder(num), den)))
    for s in brk:
        with np.errstate(all="ignore"):
            kb = -np.polyval(den, s) / np.polyval(num, s)
        if np.isfinite(kb) and abs(kb.imag) <= 1e-6 * max(abs(kb), 1e-300) and kb.real > 0:
            ks += [kb.real * f for f in (0.98, 0.995, 1.0, 1.005, 1.02)]
    ks = sorted(set(float(k) for k in ks if k > 0))
    span = 16 * scale
    for _ in range(12):
        r = _match_columns([roots_at(k) for k in ks])
        step = np.where(np.isfinite(r[:, 1:]) & np.isfinite(r[:, :-1]), np.abs(r[:, 1:] - r[:, :-1]), 0.0)
        inside = (np.abs(np.where(np.isfinite(r[:, 1:]), r[:, 1:], np.inf) - center) < span) | \
                 (np.abs(np.where(np.isfinite(r[:, :-1]), r[:, :-1], np.inf) - center) < span)
        jump = np.max(np.where(inside, step, 0.0), axis=0) if r.size else np.zeros(0)
        bad = np.flatnonzero(jump > 0.02 * span)
        if bad.size == 0 or len(ks) > 1500:
            break
        ks = sorted(set(ks + [float(np.sqrt(ks[i] * ks[i + 1])) for i in bad]))
    gains_all = np.array([0.0] + ks + [np.inf])
    return _match_columns([roots_at(k) for k in gains_all]), gains_all


@ltifn("rlocus", "rlocusplot")
def _rlocus(interp, args, nargout):
    systems, styles, rest = split_systems(args, "rlocus")
    one_system(systems, nargout, "rlocus")
    gains = None
    if rest:
        if len(rest) > 1 or not is_num(rest[0]):
            raise MatlabError("Use rlocus(sys) or rlocus(sys, k) with a vector of gains k.")
        gains = as_vector(rest[0], "gain vector").real.astype(float)
    for sys, style in zip(systems, styles):
        need_siso(sys, "rlocus")
        if isinstance(sys, FRD):
            raise MatlabError("rlocus needs a model with poles and zeros, not frequency-response data.")
        if sys.has_delay():
            if not sys.is_dt:
                raise MatlabError("rlocus does not support continuous-time delays; approximate them first "
                                  "with pade(sys, N).")
            sys = absorb_delay(sys)
        num, den = siso_numden(sys)
        if np.iscomplexobj(num) or np.iscomplexobj(den):
            raise MatlabError("rlocus needs a model with real coefficients.")
        r, k = rlocus_data(num, den, gains)
        if nargout > 0:
            return (r, row(k))[:max(nargout, 1)]
        cp.rlocus_plot(interp, r, cn.proots(den), cn.proots(num), sys.time_unit, sys.is_dt, style)
        if len(systems) > 1:
            cp.call(interp, "hold", "on")
    if len(systems) > 1:
        cp.call(interp, "hold", "off")
    return None


def _axis_limits(interp):
    res = cp.call(interp, "axis", nargout=1)
    lim = np.asarray(res[0], dtype=float).ravel()
    return lim[0], lim[1], lim[2], lim[3]


@ltifn("sgrid")
def _sgrid(interp, args, nargout):
    """Lines of constant damping ratio and natural frequency in the s-plane."""
    zetas = as_vector(args[0], "damping ratios").real if len(args) > 0 else np.arange(0.1, 1.0, 0.1)
    x0, x1, y0, y1 = _axis_limits(interp)
    rmax = float(np.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1))))
    if len(args) > 1:
        wns = as_vector(args[1], "natural frequencies").real
    else:
        step = 10.0 ** np.floor(np.log10(max(rmax, 1e-12) / 4))
        wns = np.arange(step, rmax + step / 2, step * (2 if rmax / step > 12 else 1))
    cp.call(interp, "hold", "on")
    for z in zetas:
        th = np.arccos(np.clip(z, 0, 1))
        for sgn in (1.0, -1.0):
            cp.call(interp, "plot", np.array([[0.0], [-rmax * np.cos(th)]]),
                    np.array([[0.0], [sgn * rmax * np.sin(th)]]), "k:")
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 91)
    for wn in wns:
        cp.call(interp, "plot", (wn * np.cos(th)).reshape(-1, 1), (wn * np.sin(th)).reshape(-1, 1), "k:")
    cp.call(interp, "axis", np.array([[x0, x1, y0, y1]]))
    cp.call(interp, "hold", "off")
    return None


@ltifn("zgrid")
def _zgrid(interp, args, nargout):
    """Lines of constant damping ratio and natural frequency in the z-plane."""
    zetas = as_vector(args[0], "damping ratios").real if len(args) > 0 else np.arange(0.0, 1.0, 0.1)
    wns = as_vector(args[1], "natural frequencies").real if len(args) > 1 else np.arange(0.1, 1.01, 0.1) * np.pi
    cp.call(interp, "hold", "on")
    th = np.linspace(0, 2 * np.pi, 361)
    cp.call(interp, "plot", np.cos(th).reshape(-1, 1), np.sin(th).reshape(-1, 1), "k:")
    for z in zetas:
        w = np.linspace(0, np.pi, 200)
        wn = w / np.sqrt(max(1 - z * z, 1e-12))
        s = -z * wn + 1j * w
        zz = np.exp(s)
        for sgn in (1.0, -1.0):
            cp.call(interp, "plot", zz.real.reshape(-1, 1), (sgn * zz.imag).reshape(-1, 1), "k:")
    for wn in wns:
        z = np.linspace(0, 1, 100)
        s = -z * wn + 1j * wn * np.sqrt(1 - z * z)
        zz = np.exp(s)
        for sgn in (1.0, -1.0):
            cp.call(interp, "plot", zz.real.reshape(-1, 1), (sgn * zz.imag).reshape(-1, 1), "k:")
    cp.call(interp, "axis", "equal")
    cp.call(interp, "hold", "off")
    return None
