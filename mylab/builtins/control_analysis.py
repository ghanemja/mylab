"""Poles, zeros and time responses of control-system models.

pole, zero, tzero, dcgain, damp; step, impulse, initial, lsim with their plots; stepinfo and
lsiminfo. Continuous-time responses are exact at the sample instants: the state is advanced
with the matrix exponential instead of a numerical integrator.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..errors import MatlabError
from ..values import Struct
from . import _ctrl_num as cn
from . import _ctrl_plot as cp
from .control import (
    FRD, LTI, PID, SS, TF, ZPK, _no_delay_copy, as_matrix, as_scalar, as_vector, col, is_empty_num, is_num,
    lti_poles, ltifn, need_lti, pid_to_tf, split_options, to_ss, to_tf, to_zpk,
)
from ._util import is_text, text


# -- shared argument handling -----------------------------------------------------------------

_SPEC_CHARS = set("rgbcmykw-:.ox+*sd^v<>ph")


def is_linespec(s: str) -> bool:
    return 0 < len(s) <= 4 and set(s) <= _SPEC_CHARS


def split_systems(args: list, name: str):
    """Leading models, each optionally followed by a line style, and the remaining arguments."""
    systems, styles = [], []
    k = 0
    while k < len(args) and isinstance(args[k], LTI):
        systems.append(args[k])
        k += 1
        if k < len(args) and is_text(args[k]) and is_linespec(text(args[k])):
            styles.append(text(args[k]))
            k += 1
        else:
            styles.append(None)
    if not systems:
        raise MatlabError(f"{name} expects a control-system model (tf, zpk, ss, ...) as its first argument.")
    return systems, styles, list(args[k:])


def one_system(systems: list, nargout: int, name: str) -> None:
    if nargout > 0 and len(systems) > 1:
        raise MatlabError(f"{name} returns data for one model at a time; call it once per model, or call it "
                          "without output arguments to plot several models together.")


def siso_zpk(sys: LTI):
    """Zeros, poles and gain of a SISO model (delays ignored)."""
    g = to_zpk(sys)
    return np.asarray(g.z[0][0]), np.asarray(g.p[0][0]), g.k[0, 0]


def siso_numden(sys: LTI):
    g = to_tf(sys)
    return g.num[0][0], g.den[0][0]


def need_siso(sys: LTI, name: str) -> None:
    if not sys.is_siso:
        raise MatlabError(f"{name} needs a SISO model (one input and one output); pick a channel first, e.g. "
                          f"{name}(sys(1, 1)).")


# -- poles, zeros, gain -----------------------------------------------------------------------

def _colc(v) -> np.ndarray:
    v = np.asarray(v)
    return col(v.real.copy() if np.iscomplexobj(v) and not np.any(v.imag) else v)


@ltifn("pole")
def _pole(interp, args, nargout):
    return _colc(lti_poles(need_lti(args, "pole")))


@ltifn("eig", builtin=False)
def _eig(interp, args, nargout):
    return _colc(lti_poles(args[0]))


def lti_zeros(sys: LTI) -> np.ndarray:
    if isinstance(sys, FRD):
        raise MatlabError("Zeros are not defined for frequency-response data.")
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    if sys.is_siso:
        if isinstance(sys, TF):
            return cn.proots(sys.num[0][0])
        if isinstance(sys, ZPK):
            return np.asarray(sys.z[0][0]) if sys.k[0, 0] != 0 else np.zeros(0)
        return cn.siso_zeros_gain(sys.a, sys.b, sys.c, sys.d[0, 0])[0]
    s = to_ss(_no_delay_copy(sys))
    return cn.ss_tzero(s.a, s.b, s.c, s.d)


@ltifn("zero")
def _zero(interp, args, nargout):
    sys = need_lti(args, "zero")
    z = lti_zeros(sys)
    if nargout > 1:
        if not sys.is_siso:
            raise MatlabError("zero returns the gain as second output for SISO models only.")
        return _colc(z), siso_zpk(sys)[2]
    return _colc(z)


@ltifn("tzero")
def _tzero(interp, args, nargout):
    if args and isinstance(args[0], LTI):
        sys = args[0]
        if isinstance(sys, PID):
            sys = pid_to_tf(sys)
        if sys.is_siso and not isinstance(sys, SS):
            z = lti_zeros(sys)
            rank = 1.0 if np.any(np.abs(siso_zpk(sys)[2]) > 0) else 0.0
        else:
            s = to_ss(_no_delay_copy(sys))
            z = cn.ss_tzero(s.a, s.b, s.c, s.d)
            rank = _normal_rank(s.a, s.b, s.c, s.d)
    else:
        if len(args) < 4:
            raise MatlabError("Use tzero(sys) or tzero(A, B, C, D).")
        a, b, c, d = (as_matrix(m, nm) for m, nm in zip(args[:4], "ABCD"))
        if len(args) > 4 and not is_empty_num(args[4]):
            e = as_matrix(args[4], "E")
            a, b = np.linalg.solve(e, a), np.linalg.solve(e, b)
        s = SS(a, b, c, d)
        z = cn.ss_tzero(s.a, s.b, s.c, s.d)
        rank = _normal_rank(s.a, s.b, s.c, s.d)
    if nargout > 1:
        return _colc(z), rank
    return _colc(z)


def _normal_rank(a, b, c, d) -> float:
    pts = [0.7310585 + 0.4142135j, -1.618034 + 2.236068j]
    h = cn.ss_freqresp(a, b, c, d, pts)
    return float(max(np.linalg.matrix_rank(h[:, :, k]) for k in range(len(pts)))) if h.size else 0.0


def lti_dcgain(sys: LTI) -> np.ndarray:
    if isinstance(sys, FRD):
        raise MatlabError("dcgain is not defined for frequency-response data (it has no value at zero "
                          "frequency unless one was measured).")
    if isinstance(sys, PID):
        sys = pid_to_tf(sys)
    s0 = 1.0 if sys.is_dt else 0.0
    out = np.zeros((sys.ny, sys.nu))
    if isinstance(sys, SS):
        m = (np.eye(sys.nx) - sys.a) if sys.is_dt else -sys.a
        if sys.nx == 0:
            return sys.d.copy()
        if np.linalg.cond(m) < 1e12:
            g = sys.d + sys.c @ np.linalg.solve(m, sys.b)
            return g.real.copy() if np.iscomplexobj(g) and not np.any(g.imag) else g
        sys = to_tf(sys)
    cplx = False
    vals = np.zeros((sys.ny, sys.nu), dtype=complex)
    for i in range(sys.ny):
        for j in range(sys.nu):
            if isinstance(sys, TF):
                n, d = np.polyval(sys.num[i][j], s0), np.polyval(sys.den[i][j], s0)
                if d == 0 and n != 0:
                    # the sign of an infinite gain follows the lowest-order terms
                    nd = _strip_roots(sys.den[i][j], s0)
                    vals[i, j] = np.inf * np.sign(np.real(n / nd)) if np.real(n / nd) != 0 else np.inf
                elif d == 0:
                    num_r, den_r = _cancel_at(sys.num[i][j], sys.den[i][j], s0)
                    dv = np.polyval(den_r, s0)
                    vals[i, j] = np.polyval(num_r, s0) / dv if dv != 0 else np.inf
                else:
                    vals[i, j] = n / d
            else:
                z, p, k = np.asarray(sys.z[i][j]), np.asarray(sys.p[i][j]), sys.k[i, j]
                nz, npz = int(np.sum(z == s0)), int(np.sum(p == s0))
                core = k * np.prod(s0 - z[z != s0]) / np.prod(s0 - p[p != s0]) if k != 0 else 0.0
                if npz > nz and core != 0:
                    vals[i, j] = np.inf * (np.sign(np.real(core)) or 1.0)
                elif nz > npz:
                    vals[i, j] = 0.0
                else:
                    vals[i, j] = core
            cplx = cplx or abs(np.imag(vals[i, j])) > 0
    out = vals if cplx else vals.real.copy()
    return out


def _strip_roots(p: np.ndarray, s0: float) -> complex:
    """Value at s0 of p with all its roots at s0 divided out."""
    q = np.asarray(p, dtype=complex)
    for _ in range(q.size):
        if np.polyval(q, s0) != 0:
            break
        q, _r = np.polydiv(q, np.array([1.0, -s0]))
    return complex(np.polyval(q, s0))


def _cancel_at(n: np.ndarray, d: np.ndarray, s0: float):
    n, d = np.asarray(n, dtype=complex), np.asarray(d, dtype=complex)
    while n.size > 1 and d.size > 1 and np.polyval(n, s0) == 0 and np.polyval(d, s0) == 0:
        n, _a = np.polydiv(n, np.array([1.0, -s0]))
        d, _b = np.polydiv(d, np.array([1.0, -s0]))
    return n, d


@ltifn("dcgain")
def _dcgain(interp, args, nargout):
    return lti_dcgain(need_lti(args, "dcgain"))


def damp_data(sys: LTI):
    p = lti_poles(sys)
    if sys.is_dt:
        with np.errstate(divide="ignore"):
            s = np.log(p.astype(complex)) / sys.tsv()
    else:
        s = p.astype(complex)
    wn = np.abs(s)
    with np.errstate(invalid="ignore", divide="ignore"):
        zeta = np.where(wn > 0, -s.real / np.where(wn > 0, wn, 1.0), -1.0)
        zeta = np.where(np.isinf(wn), 1.0, zeta)
    order = np.argsort(wn, kind="stable")
    return wn[order], zeta[order], p[order]


@ltifn("damp")
def _damp(interp, args, nargout):
    sys = need_lti(args, "damp")
    wn, zeta, p = damp_data(sys)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 1.0 / (wn * zeta)
    if nargout == 0:
        unit = sys.time_unit
        lines = ["",
                 "         Pole              Damping       Frequency      Time Constant  ",
                 f"                                       (rad/{unit})      ({unit})    ",
                 "                                                                       "]
        for k in range(p.size):
            pk = complex(p[k])
            if pk.imag == 0:
                ps = f"{pk.real: .2e}".ljust(23)
            else:
                ps = f"{pk.real: .2e} {'+' if pk.imag >= 0 else '-'} {abs(pk.imag):.2e}i"
            lines.append(f" {ps}    {zeta[k]: .2e}      {wn[k]: .2e}        {tau[k]: .2e}    ")
        if sys.is_dt:
            mags = np.abs(p)
            lines = ["",
                     "         Pole             Magnitude     Damping       Frequency      Time Constant  ",
                     f"                                                      (rad/{unit})      ({unit})    ",
                     "                                                                                      "]
            for k in range(p.size):
                pk = complex(p[k])
                if pk.imag == 0:
                    ps = f"{pk.real: .2e}".ljust(23)
                else:
                    ps = f"{pk.real: .2e} {'+' if pk.imag >= 0 else '-'} {abs(pk.imag):.2e}i"
                lines.append(f" {ps}   {mags[k]: .2e}    {zeta[k]: .2e}      {wn[k]: .2e}        {tau[k]: .2e}    ")
        interp.out("\n".join(lines) + "\n")
        return None
    return col(wn), col(zeta), _colc(p), col(tau)


# -- time responses ---------------------------------------------------------------------------

def _improper_guard(sys: LTI, name: str) -> SS:
    if isinstance(sys, FRD):
        raise MatlabError(f"{name} needs a model with dynamics (tf, zpk, ss); frequency-response data cannot "
                          "be simulated in time.")
    from .control import lti_isproper
    if not lti_isproper(sys):
        raise MatlabError(f"{name}: the model is improper (more zeros than poles), so its time response is "
                          "not defined. Add a fast pole, e.g. a derivative filter.")
    return to_ss(_no_delay_copy(sys))


class TimeResponse:
    """Undelayed response of a state-space model to a step, an impulse or an initial state."""

    def __init__(self, s: SS, kind: str, x0: np.ndarray | None = None):
        self.s, self.kind, self.x0 = s, kind, x0
        self.disc = s.is_dt
        self.ts = s.tsv()
        self._cache: dict = {}

    @property
    def n_inputs(self) -> int:
        return 1 if self.kind == "initial" else self.s.nu

    def _setup(self, j: int):
        s = self.s
        n = s.nx
        if self.kind == "step":
            m = np.zeros((n + 1, n + 1), dtype=np.result_type(s.a, s.b, float))
            m[:n, :n] = s.a
            m[:n, n] = s.b[:, j]
            if self.disc:
                m[n, n] = 1.0
            z0 = np.zeros(n + 1)
            z0[n] = 1.0
            cy = np.hstack([s.c, s.d[:, [j]]])
            return m, z0, cy
        if self.kind == "impulse":
            return s.a, s.b[:, j].copy(), s.c
        return s.a, np.asarray(self.x0, dtype=float).ravel(), s.c

    def raw(self, times: np.ndarray, j: int):
        """Outputs (nt, ny) and states (nt, nx) at increasing times >= 0 (sample numbers if discrete)."""
        s = self.s
        n = s.nx
        m, z0, cy = self._setup(j)
        nt = len(times)
        zs = np.empty((nt, z0.size), dtype=np.result_type(m, z0, float))
        if self.disc:
            idx = np.round(np.asarray(times)).astype(int)
            z = z0.astype(zs.dtype)
            if self.kind == "impulse":
                # u[0] = 1/Ts: the state is zero at sample 0 and B/Ts at sample 1
                z = z / self.ts
                last = int(idx.max()) if nt else 0
                seq = np.zeros((last + 1, z0.size), dtype=zs.dtype)
                cur = z
                for k in range(1, last + 1):
                    seq[k] = cur
                    cur = m @ cur
                zs = seq[idx]
                y = zs @ cy.T
                y[idx == 0] = s.d[:, j] / self.ts
                return y, zs[:, :n]
            last = int(idx.max()) if nt else 0
            seq = np.empty((last + 1, z0.size), dtype=zs.dtype)
            for k in range(last + 1):
                seq[k] = z
                z = m @ z
            zs = seq[idx]
            return zs @ cy.T, zs[:, :n]
        z = z0.astype(zs.dtype)
        tprev = 0.0
        for k, tk in enumerate(times):
            dt = tk - tprev
            if dt > 0:
                key = float(f"{dt:.12g}")
                e = self._cache.get((j, key))
                if e is None:
                    e = cn.expm(m * dt)
                    self._cache[(j, key)] = e
                z = e @ z
                tprev = tk
            zs[k] = z
        return zs @ cy.T, zs[:, :n]

    def at(self, t: float, j: int) -> np.ndarray:
        """Outputs at one time, computed directly from the matrix exponential."""
        m, z0, cy = self._setup(j)
        if self.disc:
            return self.raw(np.array([t]), j)[0][0]
        return cy @ (cn.expm(m * t) @ z0)

    def final_value(self) -> np.ndarray | None:
        """Steady-state outputs (ny, n_inputs) when the model is stable."""
        s = self.s
        if not cn.is_stable(cn.eigvals(s.a), self.disc):
            return None
        if self.kind != "step":
            return np.zeros((s.ny, self.n_inputs))
        m = (np.eye(s.nx) - s.a) if self.disc else -s.a
        return np.real(s.d + (s.c @ np.linalg.solve(m, s.b) if s.nx else 0.0))


def _nice(dt: float) -> float:
    if dt <= 0 or not np.isfinite(dt):
        return dt
    d = 10.0 ** np.floor(np.log10(dt))
    return float(np.floor(dt / d * 10 + 1e-9) / 10 * d)


def auto_time(tr: TimeResponse):
    """A final time and a time step that show the whole transient."""
    s = tr.s
    p = cn.eigvals(s.a)
    disc, ts = tr.disc, tr.ts
    if disc:
        nz = p[np.abs(p) > 1e-12]
        ps = np.log(nz.astype(complex)) / ts
    else:
        ps = p.astype(complex)
    if ps.size == 0:
        tf = (max(s.nx + 2, 10)) * ts if disc else 1.0
        return (tf, ts) if disc else (tf, 0.01)
    scale = max(float(np.max(np.abs(ps))), 1e-12)
    tol = 1e-7 * scale
    st, un = ps.real < -tol, ps.real > tol
    mg = ~st & ~un
    if np.any(un):
        tf = 5.0 / float(np.max(ps.real[un]))
    elif np.any(mg):
        base = 5.0 / float(np.min(-ps.real[st])) if np.any(st) else 0.0
        osc = mg & (np.abs(ps.imag) > tol)
        tf = 2.5 * base
        if np.any(osc):
            tf = max(tf, 6 * 2 * np.pi / float(np.min(np.abs(ps.imag[osc]))))
        if tf == 0:
            tf = 10.0 * (ts if disc else 1.0)
    else:
        tf = 5.0 / float(np.min(-ps.real[st]))
        yss = tr.final_value()
        for _ in range(40):
            npts = 400
            if disc:
                nsamp = max(int(np.ceil(tf / ts)), 2)
                grid = np.unique(np.round(np.linspace(0, nsamp, min(npts, nsamp + 1)))).astype(float)
                times = grid
                tsec = grid * ts
            else:
                tsec = np.linspace(0, tf, npts)
                times = tsec
            dev = np.concatenate([np.abs(tr.raw(times, j)[0] - yss[:, j][None, :]) for j in range(tr.n_inputs)],
                                 axis=1)
            peak = dev.max(axis=0)
            top = float(peak.max()) if peak.size else 0.0
            if top <= 0:
                break
            thr = 0.005 * np.maximum(peak, 1e-6 * top)
            late = np.flatnonzero(np.any(dev > thr[None, :], axis=1))
            last = int(late[-1]) if late.size else 0
            frac = last / (len(tsec) - 1)
            if frac > 0.75:
                tf *= 2.0
                continue
            t_settle = tsec[min(last + 1, len(tsec) - 1)]
            new_tf = 1.4 * t_settle
            if frac < 0.3 and new_tf < tf and new_tf > 0:
                tf = new_tf
                if disc and tf / ts <= 20:
                    break
                continue
            tf = new_tf if new_tf > 0 else tf
            break
    if disc:
        n = max(int(np.ceil(tf / ts - 1e-9)), 5)
        return n * ts, ts
    dt = tf / 200.0
    visible = (np.abs(ps.imag) > tol) & (np.abs(ps.imag) > np.abs(ps.real) / 40)
    if np.any(visible):
        dt = min(dt, 2 * np.pi / float(np.max(np.abs(ps.imag[visible]))) / 30.0)
    if tf / dt > 5000:
        dt = tf / 5000.0
    dt = _nice(dt)
    tf = float(np.ceil(tf / dt - 1e-9) * dt)
    return tf, dt


def parse_time(arg: Any, sys: LTI, name: str):
    """A time argument: (time vector, None) or (None, final time)."""
    a = as_vector(arg, "time argument").real.astype(float)
    if a.size == 0:
        return None, None
    if a.size == 1:
        if not (a[0] > 0) or not np.isfinite(a[0]):
            raise MatlabError(f"{name}: the final time must be positive and finite.")
        return None, float(a[0])
    d = np.diff(a)
    if np.any(d <= 0):
        raise MatlabError(f"{name}: the time vector must be increasing.")
    if np.any(np.abs(d - d[0]) > 1e-6 * abs(d[0])):
        raise MatlabError(f"{name}: the time samples must be evenly spaced, e.g. 0:0.01:5.")
    if a[0] < 0:
        raise MatlabError(f"{name}: time starts at 0; negative times are not allowed.")
    if sys.is_dt and sys.ts > 0 and abs(d[0] - sys.ts) > 1e-6 * sys.ts:
        raise MatlabError(f"{name}: for a discrete-time model the time step must equal the sample time "
                          f"Ts = {sys.ts:g} (use t = 0:Ts:Tfinal).")
    return a, None


def response_data(sys: LTI, kind: str, t=None, tfinal=None, x0=None, name: str = "step"):
    """Time vector, outputs (nt, ny, nu) and states (nt, nx, nu) of a step/impulse/initial response."""
    s = _improper_guard(sys, name)
    if kind == "initial":
        if x0 is None or np.asarray(x0).size != s.nx:
            raise MatlabError(f"initial: the initial state must have {s.nx} entries, one per state.")
    tr = TimeResponse(s, kind, x0)
    disc, ts = tr.disc, tr.ts
    total = sys.total_delay() if kind != "initial" else np.repeat(sys.out_delay[:, None], 1, axis=1)
    in_d = sys.in_delay if kind != "initial" else np.zeros(1)
    if disc:
        total, in_d = total * ts, in_d * ts       # delays of discrete models are counted in samples
    if t is None:
        if tfinal is None:
            tfinal, dt = auto_time(tr)
            tfinal += float(np.max(total)) if total.size else 0.0
        else:
            _tf, dt = auto_time(tr)
            if not disc:
                dt = min(dt, tfinal / 100.0) if tfinal / dt < 100 else dt
                if tfinal / dt > 20000:
                    dt = tfinal / 20000.0
        if disc:
            t = np.arange(int(np.floor(tfinal / ts + 1e-9)) + 1) * ts
        else:
            n = int(np.floor(tfinal / dt + 1e-9))
            t = np.arange(n + 1) * dt
    t = np.asarray(t, dtype=float)
    nin = tr.n_inputs
    y = np.zeros((t.size, s.ny, nin), dtype=float)
    x = np.zeros((t.size, s.nx, nin), dtype=float)
    cplx = any(np.iscomplexobj(m) for m in s.abcd)
    if cplx:
        y, x = y.astype(complex), x.astype(complex)
    for j in range(nin):
        delays = sorted(set(float(v) for v in total[:, j])) if s.ny else []
        for tau in delays:
            mask = t >= tau - 1e-12 * max(1.0, tau)
            if not np.any(mask):
                continue
            shifted = np.maximum(t[mask] - tau, 0.0)
            yy, _xx = tr.raw(shifted / ts if disc else shifted, j)
            rows = np.flatnonzero(np.abs(total[:, j] - tau) <= 1e-12 * max(1.0, tau))
            for i in rows:
                y[mask, i, j] = yy[:, i] if cplx else np.real(yy[:, i])
        tau = float(in_d[j]) if in_d.size > j else 0.0
        mask = t >= tau - 1e-12 * max(1.0, tau)
        if np.any(mask) and s.nx:
            shifted = np.maximum(t[mask] - tau, 0.0)
            _yy, xx = tr.raw(shifted / ts if disc else shifted, j)
            x[mask, :, j] = xx if cplx else np.real(xx)
    return t, y, x


def _step_options(v: Any):
    """Amplitude and input offset from stepDataOptions / RespConfig structures."""
    if not isinstance(v, Struct):
        return None
    get = {f.lower(): v.get_field(0, f) for f in v.fields}
    amp = get.get("stepamplitude", get.get("amplitude"))
    off = get.get("inputoffset", get.get("bias"))
    amp = 1.0 if amp is None or is_empty_num(amp) else as_vector(amp, "StepAmplitude").real
    off = 0.0 if off is None or is_empty_num(off) else as_vector(off, "InputOffset").real
    return amp, off


def _time_and_options(rest: list, sys: LTI, name: str):
    t = tfinal = None
    opts = None
    for a in rest:
        if isinstance(a, Struct):
            opts = _step_options(a)
            if opts is None:
                raise MatlabError(f"{name}: unknown options structure.")
        elif is_num(a):
            if t is not None or tfinal is not None:
                raise MatlabError(f"Too many numeric arguments for {name}.")
            t, tfinal = parse_time(a, sys, name)
        else:
            raise MatlabError(f"{name}: unexpected argument; use {name}(sys), {name}(sys, tFinal) or "
                              f"{name}(sys, t).")
    return t, tfinal, opts


def _apply_step_options(sys: LTI, y: np.ndarray, x: np.ndarray, opts):
    if opts is None:
        return y, x
    amp, off = opts
    nu = y.shape[2]
    amp = np.broadcast_to(np.atleast_1d(amp).astype(float), (nu,)) if np.size(amp) in (1, nu) else None
    off = np.broadcast_to(np.atleast_1d(off).astype(float), (nu,)) if np.size(off) in (1, nu) else None
    if amp is None or off is None:
        raise MatlabError(f"StepAmplitude and InputOffset need one value per input ({nu}) or a scalar.")
    y = y * amp[None, None, :]
    x = x * amp[None, None, :]
    if np.any(off != 0):
        g = lti_dcgain(sys)
        if not np.all(np.isfinite(g)):
            raise MatlabError("InputOffset needs a model with a finite DC gain: the response starts from the "
                              "steady state that belongs to the offset.")
        y = y + (g * off[None, :])[None, :, :]
    return y, x


def _time_outputs(sys: LTI, t, y, x, nargout: int):
    xo = x if isinstance(sys, SS) else np.zeros((0, 0))
    if isinstance(sys, SS) and x.shape[2] == 1:
        xo = x[:, :, 0]
    yo = y[:, :, 0] if y.shape[2] == 1 else y
    return (yo, col(t), xo, np.zeros((0, 0)))[:max(nargout, 1)]


def _time_response(interp, args, nargout, kind: str, name: str, title: str):
    systems, styles, rest = split_systems(args, name)
    one_system(systems, nargout, name)
    items = []
    for sys, style in zip(systems, styles):
        t, tfinal, opts = _time_and_options(rest, sys, name)
        tt, y, x = response_data(sys, kind, t, tfinal, name=name)
        if kind == "step":
            y, x = _apply_step_options(sys, y, x, opts)
        elif opts is not None:
            amp = np.atleast_1d(opts[0]).astype(float)
            if amp.size not in (1, y.shape[2]):
                raise MatlabError(f"{name}: the amplitude needs one value per input or a scalar.")
            y, x = y * np.broadcast_to(amp, (y.shape[2],))[None, None, :], x
        if nargout > 0:
            return _time_outputs(sys, tt, y, x, nargout)
        items.append({"t": tt, "y": np.real(y), "discrete": sys.is_dt, "style": style})
    first = systems[0]
    if any((s.ny, s.nu) != (first.ny, first.nu) for s in systems):
        raise MatlabError(f"{name}: all models plotted together must have the same numbers of inputs and outputs.")
    cp.time_plot(interp, title, items, first.in_name, first.out_name, first.time_unit)
    return None


@ltifn("step", "stepplot", builtin=False)
def _step(interp, args, nargout):
    return _time_response(interp, args, nargout, "step", "step", "Step Response")


@ltifn("impulse", "impulseplot")
def _impulse(interp, args, nargout):
    return _time_response(interp, args, nargout, "impulse", "impulse", "Impulse Response")


@ltifn("initial", "initialplot")
def _initial(interp, args, nargout):
    systems, styles, rest = split_systems(args, "initial")
    one_system(systems, nargout, "initial")
    if not rest:
        raise MatlabError("initial needs the initial state: initial(sys, x0).")
    x0 = as_vector(rest[0], "initial state x0")
    items = []
    for sys, style in zip(systems, styles):
        if not isinstance(sys, SS):
            raise MatlabError("initial works on state-space models, because the initial condition is a state "
                              "vector; convert with ss(sys) first.")
        t, tfinal, _ = _time_and_options(rest[1:], sys, "initial")
        tt, y, x = response_data(sys, "initial", t, tfinal, x0, "initial")
        if nargout > 0:
            return (y[:, :, 0], col(tt), x[:, :, 0])[:max(nargout, 1)]
        items.append({"t": tt, "y": np.real(y), "discrete": sys.is_dt, "style": style})
    first = systems[0]
    cp.time_plot(interp, "Response to Initial Conditions", items, [""], first.out_name, first.time_unit,
                 single_column=True)
    return None


# -- lsim -------------------------------------------------------------------------------------

def _shift_signal(v: np.ndarray, t: np.ndarray, tau: float, hold: bool) -> np.ndarray:
    """v(t - tau) on the same time grid, zero before the signal starts."""
    if tau == 0:
        return v
    dt = t[1] - t[0] if t.size > 1 else 1.0
    k = tau / dt
    if abs(k - round(k)) < 1e-9:
        k = int(round(k))
        out = np.zeros_like(v)
        if k < v.shape[0]:
            out[k:] = v[:v.shape[0] - k]
        return out
    ts = t - tau
    out = np.zeros_like(v)
    for c in range(v.shape[1]):
        if hold:
            idx = np.floor((ts - t[0]) / dt + 1e-12).astype(int)
            ok = idx >= 0
            out[ok, c] = v[np.clip(idx[ok], 0, v.shape[0] - 1), c]
        else:
            out[:, c] = np.interp(ts, t, v[:, c], left=0.0)
    return out


def lsim_data(sys: LTI, u: np.ndarray, t: np.ndarray | None, x0: np.ndarray | None, method: str | None):
    s = _improper_guard(sys, "lsim")
    disc = s.is_dt
    nt = u.shape[0]
    if u.shape[1] != s.nu:
        raise MatlabError(f"lsim: the input signal must have one column per input ({s.nu}); it has "
                          f"{u.shape[1]} for {nt} time samples.")
    if t is None:
        if not disc:
            raise MatlabError("lsim: a continuous-time model needs a time vector, lsim(sys, u, t).")
        t = np.arange(nt) * s.tsv()
    if t.size != nt:
        raise MatlabError(f"lsim: the input has {nt} samples but the time vector has {t.size}; they must "
                          "be equal.")
    if nt > 1:
        d = np.diff(t)
        if np.any(d <= 0) or np.any(np.abs(d - d[0]) > 1e-6 * abs(d[0])):
            raise MatlabError("lsim: the time samples must be increasing and evenly spaced.")
        dt = float(d[0])
        if disc and s.ts > 0 and abs(dt - s.ts) > 1e-6 * s.ts:
            raise MatlabError(f"lsim: the time step ({dt:g}) must equal the sample time of the model "
                              f"(Ts = {s.ts:g}).")
    else:
        dt = s.tsv()
    x = np.zeros(s.nx) if x0 is None else np.asarray(x0, dtype=float).ravel()
    if x.size != s.nx:
        raise MatlabError(f"lsim: the initial state must have {s.nx} entries, one per state.")
    if x0 is not None and not isinstance(sys, SS):
        raise MatlabError("lsim: an initial state can only be given for state-space models.")
    if method is None:
        if disc:
            method = "zoh"
        else:
            steps = np.diff(u, axis=0)
            flat = np.sum(np.all(steps == 0, axis=1)) if steps.size else 0
            method = "zoh" if steps.shape[0] and flat >= 0.75 * steps.shape[0] and flat < steps.shape[0] \
                else "foh"
    in_d = sys.in_delay * (dt if disc else 1.0)
    out_d = sys.out_delay * (dt if disc else 1.0)
    if np.any(sys.io_delay):
        from .control import nonzero_mask, split_delays
        ind, outd, io = split_delays(sys.total_delay(), nonzero_mask(sys))
        if np.any(io):
            raise MatlabError("lsim: this model has a different delay for each input/output pair; simulate "
                              "each channel separately or approximate the delays with pade.")
        in_d, out_d = ind * (dt if disc else 1.0), outd * (dt if disc else 1.0)
    ud = u.astype(float)
    if np.any(in_d):
        ud = np.column_stack([_shift_signal(ud[:, [j]], t, float(in_d[j]), method == "zoh" or disc)[:, 0]
                              for j in range(s.nu)])
    xs = np.zeros((nt, s.nx))
    if disc:
        for k in range(nt):
            xs[k] = x
            x = s.a @ x + s.b @ ud[k]
    elif method == "zoh":
        phi, gam = cn.zoh_matrices(s.a, s.b, dt)
        for k in range(nt):
            xs[k] = x
            x = phi @ x + gam @ ud[k]
    else:
        phi, g1, g2 = cn.foh_matrices(s.a, s.b, dt)
        for k in range(nt):
            xs[k] = x
            if k + 1 < nt:
                x = phi @ x + g1 @ ud[k] + g2 @ (ud[k + 1] - ud[k])
    y = xs @ s.c.T + ud @ s.d.T
    if np.any(out_d):
        y = np.column_stack([_shift_signal(y[:, [i]], t, float(out_d[i]), disc)[:, 0] for i in range(s.ny)])
    return t, np.real(y), np.real(xs)


@ltifn("lsim", "lsimplot")
def _lsim(interp, args, nargout):
    systems, styles, rest = split_systems(args, "lsim")
    one_system(systems, nargout, "lsim")
    if not rest:
        raise MatlabError("lsim needs an input signal: lsim(sys, u, t).")
    method = None
    if rest and is_text(rest[-1]):
        method = text(rest[-1]).lower()
        if method not in ("zoh", "foh"):
            raise MatlabError("lsim: the interpolation method must be 'zoh' or 'foh'.")
        rest = rest[:-1]
    if len(rest) > 3:
        raise MatlabError("Too many arguments for lsim; use lsim(sys, u, t, x0, method).")
    first = systems[0]
    u = as_matrix(rest[0], "input signal u").real
    t = None
    if len(rest) > 1 and not is_empty_num(rest[1]):
        t = as_vector(rest[1], "time vector t").real.astype(float)
    x0 = as_vector(rest[2], "initial state x0").real if len(rest) > 2 and not is_empty_num(rest[2]) else None
    nt = t.size if t is not None else None
    if u.shape[0] != nt and (nt is not None and u.shape[1] == nt or nt is None and u.shape[0] == 1
                             and first.nu == 1):
        u = u.T
    if first.nu == 1 and u.shape[1] != 1 and u.shape[0] == 1:
        u = u.T
    items = []
    for sys, style in zip(systems, styles):
        tt, y, x = lsim_data(sys, u, t, x0, method)
        if nargout > 0:
            xo = x if isinstance(sys, SS) else np.zeros((0, 0))
            return (y, col(tt), xo)[:max(nargout, 1)]
        items.append({"t": tt, "y": y[:, :, None], "discrete": sys.is_dt, "style": style})
    cp.lsim_plot(interp, items, u, items[0]["t"], first.out_name, first.time_unit, first.is_dt)
    return None


@ltifn("gensig")
def _gensig(interp, args, nargout):
    if len(args) < 2 or not is_text(args[0]):
        raise MatlabError("Use [u, t] = gensig(type, tau) with type 'sine', 'square' or 'pulse' and period tau.")
    kind = text(args[0]).lower()
    tau = as_scalar(args[1], "period tau")
    if tau <= 0:
        raise MatlabError("gensig: the period must be positive.")
    tf = as_scalar(args[2], "duration Tf") if len(args) > 2 else 5 * tau
    ts = as_scalar(args[3], "sample time Ts") if len(args) > 3 else tau / 64
    if ts <= 0 or tf <= 0:
        raise MatlabError("gensig: the duration and the sample time must be positive.")
    t = np.arange(int(np.floor(tf / ts + 1e-9)) + 1) * ts
    phase = np.mod(t, tau)
    phase = np.where(np.abs(phase - tau) < 1e-9 * tau, 0.0, phase)
    if kind in ("sin", "sine"):
        u = np.sin(2 * np.pi * t / tau)
    elif kind == "square":
        u = (phase >= tau / 2 - 1e-9 * tau).astype(float)
    elif kind == "pulse":
        u = (phase < ts * (1 - 1e-9)).astype(float)
    else:
        raise MatlabError("gensig: the signal type must be 'sine', 'square' or 'pulse'.")
    return col(u), col(t)


@ltifn("stepDataOptions")
def _step_data_options(interp, args, nargout):
    vals = {"InputOffset": np.zeros((1, 1)), "StepAmplitude": np.ones((1, 1))}
    _fill_options(vals, args, "stepDataOptions")
    return Struct.scalar(vals)


@ltifn("RespConfig")
def _resp_config(interp, args, nargout):
    vals = {"Bias": np.zeros((1, 1)), "Amplitude": np.ones((1, 1)), "Delay": np.zeros((1, 1)),
            "InitialState": np.zeros((0, 0))}
    _fill_options(vals, args, "RespConfig")
    if np.any(as_matrix(vals["Delay"]) != 0) or not is_empty_num(vals["InitialState"]):
        raise MatlabError("RespConfig: only Amplitude and Bias are supported; Delay and InitialState are not.")
    return Struct.scalar(vals)


def _fill_options(vals: dict, args: list, name: str) -> None:
    _pos, pairs = split_options(args)
    if _pos:
        raise MatlabError(f"{name} takes name/value pairs only.")
    lower = {k.lower(): k for k in vals}
    for nm, v in pairs:
        key = lower.get(nm.lower())
        if key is None:
            raise MatlabError(f"{name}: unknown option '{nm}'. The options are: " + ", ".join(vals) + ".")
        vals[key] = v


# -- stepinfo ---------------------------------------------------------------------------------

def _cross_time(t: np.ndarray, y: np.ndarray, level: float, rising: bool, exact: Callable | None = None):
    """First time y reaches ``level`` (from below when rising)."""
    d = (y - level) if rising else (level - y)
    idx = np.flatnonzero(d >= 0)
    if idx.size == 0:
        return np.nan
    k = int(idx[0])
    if k == 0:
        return float(t[0])
    return _refine(t, y, k - 1, level, exact)


def _refine(t: np.ndarray, y: np.ndarray, k: int, level: float, exact: Callable | None) -> float:
    """The time between samples k and k+1 where y crosses ``level``."""
    y0, y1 = y[k] - level, y[k + 1] - level
    if y1 == y0:
        return float(t[k])
    guess = float(t[k] + (t[k + 1] - t[k]) * (-y0) / (y1 - y0))
    if exact is None:
        return guess
    f0, f1 = exact(t[k]) - level, exact(t[k + 1]) - level
    if f0 == 0:
        return float(t[k])
    if f0 * f1 > 0:
        return guess
    from scipy.optimize import brentq
    return float(brentq(lambda x: exact(x) - level, t[k], t[k + 1], xtol=1e-14 * max(1.0, abs(t[k + 1])),
                        rtol=1e-13))


def _last_outside(t: np.ndarray, y: np.ndarray, center: float, band: float, exact: Callable | None) -> float:
    """First time after which |y - center| stays within ``band``."""
    err = np.abs(y - center)
    out = np.flatnonzero(err > band)
    if out.size == 0:
        return float(t[0])
    k = int(out[-1])
    if k == len(t) - 1:
        return np.nan
    level = center + band if y[k] > center else center - band
    return _refine(t, y, k, level, exact)


def step_metrics(t: np.ndarray, y: np.ndarray, yfinal: float, yinit: float, st: float, rt: tuple,
                 exact: Callable | None = None) -> dict:
    nan = float("nan")
    if not np.isfinite(yfinal) or np.any(~np.isfinite(y)):
        return {"RiseTime": nan, "TransientTime": nan, "SettlingTime": nan, "SettlingMin": nan,
                "SettlingMax": nan, "Overshoot": nan, "Undershoot": nan, "Peak": float("inf"),
                "PeakTime": float("inf")}
    span = yfinal - yinit
    sgn = 1.0 if span >= 0 else -1.0
    yn = (y - yinit) * sgn             # response normalized to rise upwards from 0 to |span|
    mag = abs(span)

    def ex(x):
        return (exact(x) - yinit) * sgn

    exn = ex if exact is not None else None
    if mag > 0:
        tlo = _cross_time(t, yn, rt[0] * mag, True, exn)
        thi = _cross_time(t, yn, rt[1] * mag, True, exn)
        rise = thi - tlo
    else:
        tlo = thi = float(t[0])
        rise = 0.0
    settle = _last_outside(t, yn, mag, st * mag, exn) if mag > 0 else nan
    emax = float(np.max(np.abs(yn - mag)))
    transient = _last_outside(t, yn, mag, st * emax, exn) if emax > 0 else float(t[0])
    if mag == 0:
        settle = transient
    after = t >= thi if np.isfinite(thi) else np.zeros(t.size, dtype=bool)
    if np.any(after):
        seg = y[after]
        smin, smax = (float(seg.min()), float(seg.max()))
        if span < 0:
            smin, smax = smax, smin
    else:
        smin = smax = nan
    kmax = int(np.argmax(np.abs(y)))
    peak, tpeak = float(np.abs(y[kmax])), float(t[kmax])
    ymax, ymin = float(yn.max()), float(yn.min())
    if exact is not None and 0 < kmax < len(t) - 1:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda x: -abs(exact(x)), bounds=(t[kmax - 1], t[kmax + 1]), method="bounded",
                              options={"xatol": 1e-12 * max(1.0, abs(t[kmax]))})
        if -res.fun >= peak:
            peak, tpeak = float(-res.fun), float(res.x)
            val = (exact(tpeak) - yinit) * sgn
            ymax, ymin = max(ymax, val), min(ymin, val)
    if mag > 0:
        over = 100.0 * max(0.0, ymax - mag) / mag
        under = 100.0 * max(0.0, -ymin) / mag
    else:
        over = float("inf") if ymax > 0 else 0.0
        under = float("inf") if ymin < 0 else 0.0
    if over < 1e-9:
        over = 0.0
    if under < 1e-9:
        under = 0.0
    return {"RiseTime": float(rise), "TransientTime": float(transient), "SettlingTime": float(settle),
            "SettlingMin": smin, "SettlingMax": smax, "Overshoot": float(over), "Undershoot": float(under),
            "Peak": peak, "PeakTime": tpeak}


def _info_options(pairs: list, name: str):
    st, rt = 0.02, (0.1, 0.9)
    for nm, v in pairs:
        key = nm.lower()
        if key == "settlingtimethreshold":
            st = as_scalar(v, "SettlingTimeThreshold")
            if not 0 < st < 1:
                raise MatlabError("SettlingTimeThreshold must be between 0 and 1 (0.02 means 2 %).")
        elif key in ("risetimethreshold", "risetimelimits"):
            r = as_vector(v, nm).real
            if r.size != 2 or not 0 <= r[0] < r[1] <= 1:
                raise MatlabError(f"{nm} must be two increasing fractions, e.g. [0.1 0.9].")
            rt = (float(r[0]), float(r[1]))
        else:
            raise MatlabError(f"{name}: unknown option '{nm}'. Use 'SettlingTimeThreshold' or "
                              "'RiseTimeThreshold'.")
    return st, rt


def _struct_grid(items: list, ny: int, nu: int) -> Struct:
    if ny == 1 and nu == 1:
        return Struct.scalar(items[0])
    data = np.empty((ny, nu), dtype=object)
    for i in range(ny):
        for j in range(nu):
            data[i, j] = items[i * nu + j]
    return Struct(data, list(items[0].keys()))


@ltifn("stepinfo")
def _stepinfo(interp, args, nargout):
    if not args:
        raise MatlabError("stepinfo needs a model or response data: stepinfo(sys) or stepinfo(y, t).")
    if isinstance(args[0], LTI):
        sys = args[0]
        _pos, pairs = split_options(args, 1)
        if len(_pos) > 1:
            raise MatlabError("stepinfo(sys) takes only name/value options after the model.")
        st, rt = _info_options(pairs, "stepinfo")
        return _stepinfo_sys(sys, st, rt)
    pos, pairs = split_options(args, 1)
    st, rt = _info_options(pairs, "stepinfo")
    y = np.asarray(as_matrix(pos[0], "response y") if np.asarray(pos[0]).ndim <= 2 else pos[0]).real
    if y.ndim == 2 and y.shape[0] == 1:
        y = y.T
    nt = y.shape[0]
    t = as_vector(pos[1], "time vector t").real if len(pos) > 1 and not is_empty_num(pos[1]) \
        else np.arange(nt, dtype=float)
    if t.size != nt:
        raise MatlabError("stepinfo: the time vector must have one entry per row of the response.")
    y3 = y.reshape(nt, y.shape[1] if y.ndim > 1 else 1, -1)
    ny, nu = y3.shape[1], y3.shape[2]
    yfin = np.broadcast_to(as_matrix(pos[2], "yfinal").real, (ny, nu)) if len(pos) > 2 and \
        not is_empty_num(pos[2]) else y3[-1]
    yini = np.broadcast_to(as_matrix(pos[3], "yinit").real, (ny, nu)) if len(pos) > 3 and \
        not is_empty_num(pos[3]) else np.zeros((ny, nu))
    items = [step_metrics(t, y3[:, i, j], float(yfin[i, j]), float(yini[i, j]), st, rt)
             for i in range(ny) for j in range(nu)]
    return _struct_grid(items, ny, nu)


def _stepinfo_sys(sys: LTI, st: float, rt: tuple) -> Struct:
    s = _improper_guard(sys, "stepinfo")
    tr = TimeResponse(s, "step")
    yss = tr.final_value()
    disc, ts = tr.disc, tr.ts
    total = sys.total_delay() * (ts if disc else 1.0)
    items = []
    if yss is None:
        for _ in range(s.ny * s.nu):
            items.append(step_metrics(np.zeros(1), np.zeros(1), float("inf"), 0.0, st, rt))
        return _struct_grid(items, s.ny, s.nu)
    tf, _dt = auto_time(tr)
    for _ in range(30):   # make sure the tail is well inside the settling band
        probe = np.linspace(0.8 * tf, tf, 40)
        probe = np.unique(np.round(probe / ts)) if disc else probe
        ok = True
        for j in range(s.nu):
            dev = np.abs(tr.raw(probe, j)[0] - yss[:, j][None, :])
            ref = np.maximum(np.abs(yss[:, j]), 1e-300)
            full = np.abs(tr.raw(np.unique(np.round(np.linspace(0, tf, 200) / ts)) if disc
                                 else np.linspace(0, tf, 200), j)[0] - yss[:, j][None, :]).max(axis=0)
            lim = 0.1 * st * np.where(np.abs(yss[:, j]) > 0, ref, np.maximum(full, 1e-300))
            if np.any(dev > lim[None, :]) and np.any(full > 0):
                ok = False
        if ok:
            break
        tf *= 1.5
    if disc:
        t = np.arange(int(np.ceil(tf / ts)) + 1) * ts
        times = t / ts
    else:
        t = np.linspace(0.0, tf, 4001)
        times = t
    for i in range(s.ny):
        for j in range(s.nu):
            y = np.real(tr.raw(times, j)[0][:, i])
            tau = float(total[i, j])
            exact = None if disc else (lambda x, i=i, j=j: float(np.real(tr.at(float(x), j)[i])))
            m = step_metrics(t, y, float(yss[i, j]), 0.0, st, rt, exact)
            if tau:
                for key in ("TransientTime", "SettlingTime", "PeakTime"):
                    m[key] = m[key] + tau
            items.append(m)
    return _struct_grid(items, s.ny, s.nu)


@ltifn("lsiminfo")
def _lsiminfo(interp, args, nargout):
    pos, pairs = split_options(args, 1)
    if not pos:
        raise MatlabError("Use lsiminfo(y, t) or lsiminfo(y, t, yfinal).")
    st = 0.02
    for nm, v in pairs:
        if nm.lower() != "settlingtimethreshold":
            raise MatlabError(f"lsiminfo: unknown option '{nm}'. Use 'SettlingTimeThreshold'.")
        st = as_scalar(v, "SettlingTimeThreshold")
    y = as_matrix(pos[0], "response y").real
    if y.shape[0] == 1:
        y = y.T
    nt, ny = y.shape
    t = as_vector(pos[1], "time vector t").real if len(pos) > 1 and not is_empty_num(pos[1]) \
        else np.arange(nt, dtype=float)
    if t.size != nt:
        raise MatlabError("lsiminfo: the time vector must have one entry per row of the response.")
    yfin = np.broadcast_to(as_vector(pos[2], "yfinal").real, (ny,)) if len(pos) > 2 and \
        not is_empty_num(pos[2]) else y[-1]
    yini = np.broadcast_to(as_vector(pos[3], "yinit").real, (ny,)) if len(pos) > 3 and \
        not is_empty_num(pos[3]) else np.zeros(ny)
    items = []
    for i in range(ny):
        yi = y[:, i]
        emax = float(np.max(np.abs(yi - yfin[i])))
        transient = _last_outside(t, yi, float(yfin[i]), st * emax, None) if emax > 0 else float(t[0])
        span = abs(float(yfin[i] - yini[i]))
        settle = _last_outside(t, yi, float(yfin[i]), st * span, None) if span > 0 else transient
        kmin, kmax = int(np.argmin(yi)), int(np.argmax(yi))
        items.append({"TransientTime": transient, "SettlingTime": settle, "Min": float(yi[kmin]),
                      "MinTime": float(t[kmin]), "Max": float(yi[kmax]), "MaxTime": float(t[kmax])})
    if ny == 1:
        return Struct.scalar(items[0])
    data = np.empty((ny, 1), dtype=object)
    for i in range(ny):
        data[i, 0] = items[i]
    return Struct(data, list(items[0].keys()))
